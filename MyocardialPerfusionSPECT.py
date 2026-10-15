"""
Myocardial Perfusion SPECT for 3D Slicer - step 1: reorientation to the cardiac axes

Workflow
    1. Select a reconstructed SPECT volume (e.g. from the DICOM SPECT Recon plugin) and press
       "Start processing". The view switches to a 3x3 grid plus a tall MIP view on the right:
           row 1  transaxial / coronal / sagittal of the whole volume  -> place the spherical heart ROI
                  (drag the yellow point to move it, the orange points to resize it)
           row 2  cropped views around the ROI with one axis line each  -> adjust the axes
                    transaxial       : line sets the long axis direction in the transaxial plane
                    vertical LA      : line sets the long axis elevation (apex up/down)
                    short axis       : line sets the rotation around the long axis (points anterior)
           row 3  live resliced preview: short axis, horizontal long axis, vertical long axis
                  (optionally masked with an ellipsoid aligned to the cardiac axes; drag the
                  cyan points in these views to resize it)
           right  rotating parallel-projection MIP of the whole volume, anterior view upright
       Drag a line end point to rotate the axis, drag its middle point to move the LV center.
    2. Press "Create SA / HLA / VLA volumes". Three reoriented volumes are resampled and shown
       in a slice panel with three rows, each a grid (default 2 x 12) of consecutive slices:
           SA  apex -> base         (anterior up, septum left)
           HLA inferior -> anterior (apex up, septum left)
           VLA septum -> lateral    (anterior up, apex right)
       The panel is a Qt widget embedded in the view layout (Slicer 5.8+ removed lightbox mode).

Display follows the ASNC/SNMMI standard orientation.

Install: put this file in a folder and add that folder to
Edit > Application Settings > Modules > Additional module paths, then restart.
The module appears under "Nuclear Medicine".
"""

import os
import math
import logging
import tempfile
import traceback

import numpy as np
import qt
import ctk
import vtk
import vtk.util.numpy_support as numpy_support
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)
from slicer.util import VTKObservationMixin


# ---------------------------------------------------------------------------
# View / layout identifiers
# ---------------------------------------------------------------------------
REORIENT_LAYOUT_ID = 5710
RESULT_LAYOUT_ID = 5711

ROW1_VIEWS = ("MPIAxial", "MPICoronal", "MPISagittal")
ROW1_ORIENTATIONS = ("Axial", "Coronal", "Sagittal")
EDIT_ORDER = ("TA", "VLA", "SA")
EDIT_VIEWS = {"TA": "MPIEditTA", "VLA": "MPIEditVLA", "SA": "MPIEditSA"}
ORIENTATIONS = ("SA", "HLA", "VLA")
PREVIEW_VIEWS = {"SA": "MPIPreviewSA", "HLA": "MPIPreviewHLA", "VLA": "MPIPreviewVLA"}
MIP_VIEW = "MPIMIP"
MOSAIC_TAG = "mpislicemosaic"   # XML tag of the embedded slice panel

DEFAULT_GRID_ROWS = 2
DEFAULT_GRID_COLUMNS = 12
DEFAULT_PIXEL_SIZE_MM = 2.0       # in-plane
DEFAULT_SLICE_THICKNESS_MM = 5.0
MOSAIC_PANEL_VERSION = 3   # bump when SliceMosaicPanel changes, so a reload replaces the old panel

MIP_SPIN_DEGREES = 2.0 / 3.0    # per animation frame (Slicer default is 2)

ORIENTATION_TITLES = {
    "SA": "Short axis (apex \u2192 base)",
    "HLA": "Horizontal long axis (inferior \u2192 anterior)",
    "VLA": "Vertical long axis (septum \u2192 lateral)",
}

# (combo label, color node name or ID)
COLORMAPS = [
    ("PET-DICOM", "PET-DICOM"),
    ("PET-Heat", "PET-Heat"),
    ("PET-Rainbow2", "PET-Rainbow2"),
    ("Grey", "vtkMRMLColorTableNodeGrey"),
    ("Inverted grey", "vtkMRMLColorTableNodeInvertedGrey"),
]

SUPERIOR = np.array([0.0, 0.0, 1.0])
PATIENT_LEFT = np.array([-1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _sliceViewXml(tag, label, color, orientation="Axial"):
    return (
        f'<view class="vtkMRMLSliceNode" singletontag="{tag}">'
        f'<property name="orientation" action="default">{orientation}</property>'
        f'<property name="viewlabel" action="default">{label}</property>'
        f'<property name="viewcolor" action="default">{color}</property>'
        "</view>"
    )


def _horizontalRow(views):
    return ('<item><layout type="horizontal" split="true">'
            + "".join(f"<item>{v}</item>" for v in views)
            + "</layout></item>")


def reorientLayoutXml():
    row1 = [_sliceViewXml(tag, o, c, o) for tag, o, c in
            zip(ROW1_VIEWS, ROW1_ORIENTATIONS, ("#F34A33", "#6EB04B", "#EDD54C"))]
    row2 = [_sliceViewXml(EDIT_VIEWS["TA"], "Adjust TA", "#D81B60"),
            _sliceViewXml(EDIT_VIEWS["VLA"], "Adjust VLA", "#43A047"),
            _sliceViewXml(EDIT_VIEWS["SA"], "Adjust SA", "#FDD835")]
    row3 = [_sliceViewXml(PREVIEW_VIEWS[n], n, "#1E88E5") for n in ORIENTATIONS]
    grid = ('<layout type="vertical" split="true">'
            + _horizontalRow(row1) + _horizontalRow(row2) + _horizontalRow(row3)
            + "</layout>")
    mip = (f'<view class="vtkMRMLViewNode" singletontag="{MIP_VIEW}">'
           '<property name="viewlabel" action="default">MIP</property></view>')
    return ('<layout type="horizontal" split="true">'
            f'<item splitSize="780">{grid}</item>'
            f'<item splitSize="260">{mip}</item>'
            "</layout>")


def resultLayoutXml():
    return f'<layout type="vertical"><item><{MOSAIC_TAG}></{MOSAIC_TAG}></item></layout>'


def registerLayout(layoutId, xml):
    layoutNode = slicer.app.layoutManager().layoutLogic().GetLayoutNode()
    if layoutNode.IsLayoutDescription(layoutId):
        layoutNode.SetLayoutDescription(layoutId, xml)
    else:
        layoutNode.AddLayoutDescription(layoutId, xml)


def sliceWidgetByTag(tag):
    return slicer.app.layoutManager().sliceWidget(tag)


def sliceNodeByTag(tag):
    widget = sliceWidgetByTag(tag)
    return widget.mrmlSliceNode() if widget else None


def threeDWidgetByTag(tag):
    layoutManager = slicer.app.layoutManager()
    for index in range(layoutManager.threeDViewCount):
        widget = layoutManager.threeDWidget(index)
        if widget and widget.mrmlViewNode().GetLayoutName() == tag:
            return widget
    return None


def showVolumeInView(tag, volumeNode):
    widget = sliceWidgetByTag(tag)
    if not widget:
        return
    composite = widget.sliceLogic().GetSliceCompositeNode()
    composite.SetLinkedControl(False)
    composite.SetHotLinkedControl(False)
    composite.SetBackgroundVolumeID(volumeNode.GetID() if volumeNode else None)
    composite.SetForegroundVolumeID(None)
    composite.SetLabelVolumeID(None)


def setSliceGeometry(sliceNode, right, up, normal, origin, fieldOfView):
    """Point a slice view at an arbitrary plane.
    right/up are the screen directions in RAS, normal is the slice stepping direction."""
    if sliceNode is None:
        return
    matrix = vtk.vtkMatrix4x4()
    for r in range(3):
        matrix.SetElement(r, 0, float(right[r]))
        matrix.SetElement(r, 1, float(up[r]))
        matrix.SetElement(r, 2, float(normal[r]))
        matrix.SetElement(r, 3, float(origin[r]))
    dims = sliceNode.GetDimensions()
    aspect = dims[0] / float(dims[1]) if dims[0] > 0 and dims[1] > 0 else 1.0
    fovX, fovY = (fieldOfView * aspect, fieldOfView) if aspect >= 1.0 else (fieldOfView, fieldOfView / aspect)
    wasModifying = sliceNode.StartModify()
    sliceNode.GetSliceToRAS().DeepCopy(matrix)
    sliceNode.SetXYZOrigin(0, 0, 0)
    sliceNode.SetFieldOfView(fovX, fovY, sliceNode.GetFieldOfView()[2])
    sliceNode.UpdateMatrices()
    sliceNode.EndModify(wasModifying)


def worldToLocalMatrix(node):
    """World RAS -> the node's own RAS (identity unless the node is under a transform)."""
    matrix = vtk.vtkMatrix4x4()
    parent = node.GetParentTransformNode()
    if parent:
        if not parent.IsTransformToWorldLinear():
            raise ValueError(f"'{node.GetName()}' is under a non-linear transform. Harden the transform first.")
        parent.GetMatrixTransformFromWorld(matrix)
    return matrix


def ellipsoidInsideGrid(shape, ijkToRas, centerWorld, axesWorld, semiAxes):
    """Boolean [k, j, i] mask of a grid (given its 4x4 IJK->RAS as numpy) inside an ellipsoid."""
    nk, nj, ni = shape
    m = np.asarray(ijkToRas, dtype=float)
    i = np.arange(ni)[None, None, :]
    j = np.arange(nj)[None, :, None]
    k = np.arange(nk)[:, None, None]
    offset = [m[a, 0] * i + m[a, 1] * j + m[a, 2] * k + (m[a, 3] - centerWorld[a]) for a in range(3)]
    quadratic = 0.0
    for direction, semi in zip(axesWorld, np.maximum(np.asarray(semiAxes, dtype=float), 1e-3)):
        projection = direction[0] * offset[0] + direction[1] * offset[1] + direction[2] * offset[2]
        quadratic = quadratic + (projection / semi) ** 2
    return quadratic <= 1.0


def _ijkToRas(iColumn, jColumn, kColumn, origin):
    matrix = vtk.vtkMatrix4x4()
    for r in range(3):
        matrix.SetElement(r, 0, float(iColumn[r]))
        matrix.SetElement(r, 1, float(jColumn[r]))
        matrix.SetElement(r, 2, float(kColumn[r]))
        matrix.SetElement(r, 3, float(origin[r]))
    return matrix


def colorNodeForLabel(label):
    for name, key in COLORMAPS:
        if name == label:
            node = slicer.mrmlScene.GetNodeByID(key) or slicer.mrmlScene.GetFirstNodeByName(key)
            if node:
                return node
    return slicer.mrmlScene.GetNodeByID("vtkMRMLColorTableNodeGrey")


def colorTableArray(label):
    """256 x 3 uint8 RGB table sampled from a Slicer color node."""
    table = np.repeat(np.linspace(0.0, 1.0, 256)[:, None], 3, axis=1)
    node = colorNodeForLabel(label)
    scalarsToColors = node.GetScalarsToColors() if node else None
    if scalarsToColors:
        low, high = scalarsToColors.GetRange()
        rgb = [0.0, 0.0, 0.0]
        for i in range(256):
            scalarsToColors.GetColor(low + (high - low) * i / 255.0, rgb)
            table[i] = rgb
    return np.clip(np.round(table * 255.0), 0, 255).astype(np.uint8)


def rgbToQImage(rgb):
    """uint8 [h, w, 3] array -> QImage (via an in-memory PNG, as in the recon plugin)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    image = vtk.vtkImageData()
    image.SetDimensions(w, h, 1)
    flipped = np.ascontiguousarray(rgb[::-1])  # VTK images are bottom-up
    scalars = numpy_support.numpy_to_vtk(flipped.reshape(-1, 3), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    image.GetPointData().SetScalars(scalars)
    writer = vtk.vtkPNGWriter()
    writer.SetInputData(image)
    writer.SetWriteToMemory(True)
    writer.Write()
    qimage = qt.QImage()
    try:
        data = numpy_support.vtk_to_numpy(writer.GetResult()).tobytes()
        qimage.loadFromData(qt.QByteArray(data), "PNG")
    except Exception:
        qimage = qt.QImage()
    if qimage.isNull():
        # fall back to a temporary file
        path = os.path.join(tempfile.gettempdir(), "mpi_slice_tile.png")
        fileWriter = vtk.vtkPNGWriter()
        fileWriter.SetInputData(image)
        fileWriter.SetFileName(path)
        fileWriter.Write()
        qimage = qt.QImage(path)
    return qimage


def setVolumeDisplay(volumeNode, colorLabel, windowMin=None, windowMax=None):
    if not volumeNode:
        return
    volumeNode.CreateDefaultDisplayNodes()
    displayNode = volumeNode.GetDisplayNode()
    if not displayNode:
        return
    displayNode.SetInterpolate(True)
    colorNode = colorNodeForLabel(colorLabel)
    if colorNode:
        displayNode.SetAndObserveColorNodeID(colorNode.GetID())
    if windowMin is not None and windowMax is not None and windowMax > windowMin:
        displayNode.AutoWindowLevelOff()
        displayNode.SetWindowLevelMinMax(windowMin, windowMax)


def placeNextTo(node, referenceNode):
    """Put a new node in the same subject hierarchy folder/study as the reference."""
    try:
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        referenceItem = shNode.GetItemByDataNode(referenceNode)
        item = shNode.GetItemByDataNode(node)
        if referenceItem and item:
            shNode.SetItemParent(item, shNode.GetItemParent(referenceItem))
    except Exception:
        logging.warning("Could not place node in subject hierarchy:\n" + traceback.format_exc())


# ---------------------------------------------------------------------------
# Slice panel embedded in the layout (replaces the removed lightbox mode)
# ---------------------------------------------------------------------------
class _EventForwarder(qt.QObject):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def eventFilter(self, obj, event):
        try:
            return bool(self.callback(obj, event))
        except Exception:
            logging.error(traceback.format_exc())
            return False


class SliceMosaicPanel:
    """Three rows (SA, HLA, VLA), each a 2 x 10 grid of slice tiles with slice numbers.
    Mouse wheel over a row scrolls that row."""

    GAP = 3

    def __init__(self):
        self.onScroll = None
        self.gridRows = DEFAULT_GRID_ROWS
        self.gridColumns = DEFAULT_GRID_COLUMNS
        self.widget = qt.QWidget()
        self.widget.setObjectName("MPISliceMosaic")
        self.widget.setStyleSheet("QWidget#MPISliceMosaic { background-color: black; }"
                                  "QLabel { color: #e0e0e0; background-color: black; }")
        layout = qt.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        self.filter = _EventForwarder(self._onEvent)
        self.rows = {}
        for name in ORIENTATIONS:
            title = qt.QLabel(ORIENTATION_TITLES[name])
            title.setStyleSheet("font-weight: bold;")
            image = qt.QLabel()
            image.setObjectName(f"MPIMosaic_{name}")
            image.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Ignored)  # pixmap must not drive the size
            image.setMinimumSize(50, 20)
            image.setAlignment(qt.Qt.AlignCenter)
            image.installEventFilter(self.filter)
            layout.addWidget(title)
            layout.addWidget(image, 1)
            self.rows[name] = {"title": title, "image": image, "tiles": []}

    def setRow(self, name, tiles, titleText):
        """tiles: list of (QImage or None, slice number or None), row-major."""
        row = self.rows[name]
        row["tiles"] = tiles
        row["title"].setText(titleText)
        self.paintRow(name)

    def setGrid(self, rows, columns):
        self.gridRows, self.gridColumns = int(rows), int(columns)
        for name in ORIENTATIONS:
            self.paintRow(name)

    def clear(self):
        for name in ORIENTATIONS:
            self.setRow(name, [], ORIENTATION_TITLES[name])

    def paintRow(self, name):
        row = self.rows[name]
        label = row["image"]
        rect = label.contentsRect()
        width, height = rect.width(), rect.height()
        if width < 10 or height < 10:
            return
        gap = self.GAP
        rows, columns = self.gridRows, self.gridColumns
        tile = min((width - gap * (columns - 1)) // columns,
                   (height - gap * (rows - 1)) // rows)
        tile = max(tile, 4)
        pixmap = qt.QPixmap(tile * columns + gap * (columns - 1),
                            tile * rows + gap * (rows - 1))
        pixmap.fill(qt.QColor(0, 0, 0))
        painter = qt.QPainter(pixmap)
        painter.setRenderHint(qt.QPainter.SmoothPixmapTransform, True)
        font = painter.font()
        font.setPixelSize(max(9, tile // 9))
        painter.setFont(font)
        for index, (qimage, number) in enumerate(row["tiles"][:rows * columns]):
            r, c = divmod(index, columns)
            x, y = c * (tile + gap), r * (tile + gap)
            if qimage is not None and not qimage.isNull():
                painter.drawImage(qt.QRect(x, y, tile, tile), qimage)
            if number is not None:
                painter.setPen(qt.QColor(0, 0, 0))
                painter.drawText(x + 4, y + 3 + font.pixelSize(), str(number))
                painter.setPen(qt.QColor(255, 255, 255))
                painter.drawText(x + 3, y + 2 + font.pixelSize(), str(number))
        painter.end()
        label.setPixmap(pixmap)

    def _onEvent(self, obj, event):
        objectName = obj.objectName if isinstance(obj.objectName, str) else obj.objectName()
        if not objectName.startswith("MPIMosaic_"):
            return False
        name = objectName.split("_", 1)[1]
        if event.type() == qt.QEvent.Resize:
            self.paintRow(name)
            return False
        if event.type() == qt.QEvent.Wheel:
            try:
                delta = event.angleDelta().y()
            except AttributeError:
                delta = event.delta()
            if delta != 0 and self.onScroll:
                self.onScroll(name, -1 if delta > 0 else 1)
            return True
        return False


def mosaicPanel():
    """The panel and its view factory live for the whole Slicer session (also across module reloads),
    because the layout manager keeps using the factory registered first. After a module update the
    factory is pointed at a new panel; restarting Slicer is the most reliable way to pick it up."""
    state = getattr(slicer, "_mpiSliceMosaicState", None)
    if state is None:
        factory = slicer.qSlicerSingletonViewFactory()
        factory.setTagName(MOSAIC_TAG)
        slicer.app.layoutManager().registerViewFactory(factory)
        state = {"panel": None, "factory": factory, "version": None}
        slicer._mpiSliceMosaicState = state
    if state.get("version") != MOSAIC_PANEL_VERSION or state.get("panel") is None:
        state["panel"] = SliceMosaicPanel()
        state["factory"].setWidget(state["panel"].widget)
        state["version"] = MOSAIC_PANEL_VERSION
    return state["panel"]


# ---------------------------------------------------------------------------
# Module registration
# ---------------------------------------------------------------------------
class MyocardialPerfusionSPECT(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Myocardial Perfusion SPECT"
        self.parent.categories = ["Nuclear Medicine"]
        self.parent.dependencies = []
        self.parent.contributors = [""]
        self.parent.helpText = (
            "Myocardial perfusion SPECT processing. Step 1: reorient the reconstructed volume "
            "to short axis, horizontal long axis and vertical long axis.<br><br>"
            "Select a volume and press <b>Start processing</b>. Click the left ventricle in the top row "
            "to place the spherical ROI (drag the orange points to resize it). In the second row drag the end points of each line to rotate "
            "the axis and the middle point to move the center. The third row shows the resliced result. "
            "Optionally mask the bottom row and the created volumes with an ellipsoid (drag its cyan points to resize). "
            "Press <b>Create SA / HLA / VLA volumes</b> to resample and review the slices."
        )
        self.parent.acknowledgementText = ""


# ---------------------------------------------------------------------------
# Logic: cardiac frame, resampling
# ---------------------------------------------------------------------------
class MyocardialPerfusionSPECTLogic(ScriptedLoadableModuleLogic):
    """Holds the cardiac coordinate frame.

    longAxis  unit vector from base to apex (RAS)
    roll      rotation around the long axis (radians); 0 means the VLA plane is vertical
    center    LV center (world RAS, mm)
    radius    ROI radius (mm); the output volumes are cubes of side 2 * radius

    Derived frame (see frame()):
        u  base -> apex
        w  toward the anterior wall (perpendicular to u)
        s  toward the septum = u x w
    """

    MAX_ELEVATION = math.radians(80.0)

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.center = np.zeros(3)
        self.radius = 55.0
        self.resetAxes()
        self.resetMaskEllipsoid()

    # --- axis state --------------------------------------------------------
    def resetAxes(self):
        """Typical LV orientation: apex toward patient left, anterior and slightly inferior."""
        horizontal = _unit([-1.0, 1.0, 0.0])  # RAS: -x = left, +y = anterior
        elevation = math.radians(-25.0)
        self.longAxis = _unit(math.cos(elevation) * horizontal + math.sin(elevation) * SUPERIOR)
        self.roll = 0.0

    def horizontalDirection(self):
        d = self.longAxis - np.dot(self.longAxis, SUPERIOR) * SUPERIOR
        return _unit(d) if np.linalg.norm(d) > 1e-6 else _unit([-1.0, 1.0, 0.0])

    def elevation(self):
        return math.asin(float(np.clip(np.dot(self.longAxis, SUPERIOR), -1.0, 1.0)))

    def azimuthDegrees(self):
        """Angle of the long axis in the transaxial plane, from patient left toward anterior."""
        d = self.horizontalDirection()
        return math.degrees(math.atan2(d[1], -d[0]))

    def frame(self):
        u = _unit(self.longAxis)
        d = self.horizontalDirection()
        w0 = _unit(SUPERIOR - np.dot(SUPERIOR, u) * u)  # in the vertical plane through the long axis
        s0 = np.cross(u, w0)
        w = math.cos(self.roll) * w0 + math.sin(self.roll) * s0
        s = np.cross(u, w)
        return {"u": u, "d": d, "w0": w0, "s0": s0, "w": _unit(w), "s": _unit(s)}

    def editViewGeometry(self, which):
        """Plane and line direction of the three adjustment views (row 2).
        Each view's plane does not depend on the angle its own line controls,
        so the image stays still while the user drags that line."""
        f = self.frame()
        if which == "TA":
            return {"right": np.array([-1.0, 0.0, 0.0]), "up": np.array([0.0, 1.0, 0.0]),
                    "normal": SUPERIOR.copy(), "line": f["d"]}
        if which == "VLA":
            return {"right": f["d"], "up": SUPERIOR.copy(),
                    "normal": _unit(np.cross(f["d"], SUPERIOR)), "line": f["u"]}
        if which == "SA":
            return {"right": -f["s0"], "up": f["w0"], "normal": f["u"], "line": f["w"]}
        raise ValueError(which)

    def setLineDirection(self, which, direction):
        """Update the axes from a line direction drawn in one of the adjustment views."""
        geometry = self.editViewGeometry(which)
        normal = _unit(geometry["normal"])
        v = np.asarray(direction, dtype=float)
        v = v - np.dot(v, normal) * normal
        if np.linalg.norm(v) < 1e-6:
            return False
        v = _unit(v)
        if which == "TA":
            elevation = self.elevation()
            self.longAxis = _unit(math.cos(elevation) * v + math.sin(elevation) * SUPERIOR)
        elif which == "VLA":
            d = self.horizontalDirection()
            elevation = math.atan2(np.dot(v, SUPERIOR), np.dot(v, d))
            elevation = float(np.clip(elevation, -self.MAX_ELEVATION, self.MAX_ELEVATION))
            self.longAxis = _unit(math.cos(elevation) * d + math.sin(elevation) * SUPERIOR)
        elif which == "SA":
            f = self.frame()
            self.roll = math.atan2(np.dot(v, f["s0"]), np.dot(v, f["w0"]))
        return True

    def orientationAxes(self):
        """(screen right, screen up, slice stacking direction) for each standard view."""
        f = self.frame()
        return {
            "SA": (-f["s"], f["w"], -f["u"]),   # septum left, anterior up, apex -> base
            "HLA": (-f["s"], f["u"], f["w"]),   # septum left, apex up, inferior -> anterior
            "VLA": (f["u"], f["w"], -f["s"]),   # apex right, anterior up, septum -> lateral
        }

    # --- image operations --------------------------------------------------
    # --- bottom-row mask ---------------------------------------------------
    def resetMaskEllipsoid(self):
        """Semi-axes along (long axis, anterior-inferior, septal-lateral), roughly LV shaped."""
        r = self.radius
        self.maskSemiAxes = np.array([0.85 * r, 0.65 * r, 0.65 * r])

    def maskAxes(self):
        f = self.frame()
        return np.array([f["u"], f["w"], f["s"]])

    # --- image operations --------------------------------------------------
    @staticmethod
    def ellipsoidBlock(volumeNode, centerWorld, axesWorld, semiAxes):
        """Index box around an ellipsoid and the inside mask for that box.
        axesWorld: 3 unit vectors (rows), semiAxes: the 3 semi-axis lengths in mm.
        Returns (lo, hi, inside) with lo/hi as IJK and inside shaped [k, j, i], or None."""
        semiAxes = np.maximum(np.asarray(semiAxes, dtype=float), 1e-3)
        dims = np.array(volumeNode.GetImageData().GetDimensions())
        worldToLocal = slicer.util.arrayFromVTKMatrix(worldToLocalMatrix(volumeNode))
        localCenter = (worldToLocal @ np.append(centerWorld, 1.0))[:3]
        localAxes = [_unit(worldToLocal[:3, :3] @ np.asarray(a, dtype=float)) for a in axesWorld]
        rasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(rasToIjk)
        ijkToRas = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRas)
        centerIjk = (slicer.util.arrayFromVTKMatrix(rasToIjk) @ np.append(localCenter, 1.0))[:3]
        spacing = np.array(volumeNode.GetSpacing())
        bound = float(semiAxes.max())
        lo = np.clip(np.floor(centerIjk - bound / spacing).astype(int), 0, dims)
        hi = np.clip(np.ceil(centerIjk + bound / spacing).astype(int) + 1, 0, dims)
        if np.any(hi <= lo):
            return None
        m = slicer.util.arrayFromVTKMatrix(ijkToRas)
        i = np.arange(lo[0], hi[0])[None, None, :]
        j = np.arange(lo[1], hi[1])[None, :, None]
        k = np.arange(lo[2], hi[2])[:, None, None]
        offset = [m[axis, 0] * i + m[axis, 1] * j + m[axis, 2] * k + (m[axis, 3] - localCenter[axis])
                  for axis in range(3)]
        quadratic = 0.0
        for direction, semi in zip(localAxes, semiAxes):
            projection = direction[0] * offset[0] + direction[1] * offset[1] + direction[2] * offset[2]
            quadratic = quadratic + (projection / semi) ** 2
        return lo, hi, quadratic <= 1.0

    @classmethod
    def sphereBlock(cls, volumeNode, centerWorld, radius):
        return cls.ellipsoidBlock(volumeNode, centerWorld, np.eye(3), [radius, radius, radius])

    @classmethod
    def maxInSphere(cls, volumeNode, centerWorld, radius):
        array = slicer.util.arrayFromVolume(volumeNode)
        block = cls.sphereBlock(volumeNode, centerWorld, radius)
        if block is None:
            return float(array.max())
        lo, hi, inside = block
        values = array[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]][inside]
        return float(values.max()) if values.size else float(array.max())

    @classmethod
    def maskedArray(cls, volumeNode, centerWorld, axesWorld, semiAxes):
        """Copy of the voxel array with everything outside the ellipsoid set to zero."""
        array = slicer.util.arrayFromVolume(volumeNode)
        masked = np.zeros_like(array)
        block = cls.ellipsoidBlock(volumeNode, centerWorld, axesWorld, semiAxes)
        if block is not None:
            lo, hi, inside = block
            source = array[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
            masked[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = np.where(inside, source, 0)
        return masked

    @classmethod
    def applyEllipsoidMask(cls, volumeNode, centerWorld, axesWorld, semiAxes):
        """Zero the voxels of a volume outside the ellipsoid, in place.
        Applied on the reoriented grid, so the edge follows the ellipsoid at the output voxel size."""
        array = slicer.util.arrayFromVolume(volumeNode)
        array[:] = cls.maskedArray(volumeNode, centerWorld, axesWorld, semiAxes)
        slicer.util.arrayFromVolumeModified(volumeNode)

    def resampleOrientation(self, inputNode, axes, spacing, name, sliceThickness=None, mask=None):
        """Resample the input onto a block around the LV center whose i/j/k axes are
        screen right / screen up / stacking direction of the requested orientation.

        spacing         in-plane pixel size (mm)
        sliceThickness  slice thickness = slice spacing (mm); default: same as spacing
        mask            optional (axesWorld, semiAxes) ellipsoid; voxels outside are zeroed

        Each output slice is the mean of thin sections (at most 'spacing' apart) covering its whole
        thickness, like summing thin slices on a clinical workstation: thick slices average the counts
        in the slab instead of sampling one plane and skipping the rest. The mask is applied to the thin
        sections before averaging, so its edge is partial-volumed rather than stair-stepped."""
        iDir, jDir, kDir = (_unit(a) for a in axes)
        spacing = float(spacing)
        thickness = float(sliceThickness) if sliceThickness else spacing
        center = np.asarray(self.center, dtype=float)

        nPlane = max(2, int(math.ceil(2.0 * self.radius / spacing)))
        nSlices = max(1, int(math.ceil(2.0 * self.radius / thickness)))
        sectionsPerSlice = max(1, int(math.ceil(thickness / spacing - 1e-6)))
        sectionSpacing = thickness / sectionsPerSlice
        nSections = nSlices * sectionsPerSlice

        halfPlane = (nPlane - 1) / 2.0 * spacing
        sectionOrigin = center - halfPlane * (iDir + jDir) - (nSections - 1) / 2.0 * sectionSpacing * kDir
        sectionIjkToRas = _ijkToRas(iDir * spacing, jDir * spacing, kDir * sectionSpacing, sectionOrigin)

        # thin sections: output IJK -> world RAS -> input RAS -> input IJK
        rasToIjk = vtk.vtkMatrix4x4()
        inputNode.GetRASToIJKMatrix(rasToIjk)
        sectionToLocal = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Multiply4x4(worldToLocalMatrix(inputNode), sectionIjkToRas, sectionToLocal)
        resliceAxes = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Multiply4x4(rasToIjk, sectionToLocal, resliceAxes)

        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(inputNode.GetImageData())
        reslice.SetResliceAxes(resliceAxes)
        reslice.SetInterpolationModeToLinear()  # no overshoot, counts stay non-negative
        reslice.SetOutputScalarType(vtk.VTK_FLOAT)
        reslice.SetOutputOrigin(0.0, 0.0, 0.0)
        reslice.SetOutputSpacing(1.0, 1.0, 1.0)
        reslice.SetOutputExtent(0, nPlane - 1, 0, nPlane - 1, 0, nSections - 1)
        reslice.SetBackgroundLevel(0.0)
        reslice.Update()
        sections = numpy_support.vtk_to_numpy(reslice.GetOutput().GetPointData().GetScalars())
        sections = np.array(sections, dtype=np.float32).reshape(nSections, nPlane, nPlane)

        if mask is not None:
            maskAxes, semiAxes = mask
            sections *= ellipsoidInsideGrid(sections.shape, slicer.util.arrayFromVTKMatrix(sectionIjkToRas),
                                            center, maskAxes, semiAxes)

        slices = sections.reshape(nSlices, sectionsPerSlice, nPlane, nPlane).mean(axis=1).astype(np.float32)

        # slice n is centered on the middle of its thin sections
        sliceOrigin = center - halfPlane * (iDir + jDir) - (nSlices - 1) / 2.0 * thickness * kDir
        ijkToRas = _ijkToRas(iDir * spacing, jDir * spacing, kDir * thickness, sliceOrigin)

        outputNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", slicer.mrmlScene.GenerateUniqueName(name))
        outputNode.SetIJKToRASMatrix(ijkToRas)
        slicer.util.updateVolumeFromArray(outputNode, slices)
        outputNode.CreateDefaultDisplayNodes()
        geometry = {"i": iDir, "j": jDir, "k": kDir, "center": center.copy(),
                    "n": nSlices, "nPlane": nPlane, "spacing": spacing, "thickness": thickness,
                    "sectionsPerSlice": sectionsPerSlice}
        return outputNode, geometry


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class MyocardialPerfusionSPECTWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    RADIUS_MIN = 10.0
    RADIUS_MAX = 150.0

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self.inputNode = None
        self.centerNode = None
        self.radiusNode = None       # two resize handles on the ROI sphere
        self.sphereNode = None
        self.maskedNode = None       # masked copy of the input for the bottom row
        self.maskHandleNode = None   # three handles on the mask ellipsoid (bottom-row views)
        self.maskModelNode = None    # mask ellipsoid, drawn as outlines in the bottom row
        self.maskHandlePositions = None
        self.draggedMaskHandle = None
        self.curves = {}             # "TA" / "VLA" / "SA" -> markups curve node
        self.curveKey = {}           # curve node ID -> "TA" / "VLA" / "SA"
        self.curvePositions = {}     # last positions we wrote, to find which point the user dragged
        self.radiusHandlePositions = None
        self.draggedRadiusHandle = None
        self.halfLength = {}
        self.helperObservations = []
        self.roiPlaced = False
        self.viewCenter = np.zeros(3)
        self.windowMax = None
        self.outputWindowMax = None   # display max of the created volumes (masked: myocardial max)
        self.mipDisplayNode = None
        self.outputNodes = {}
        self.outputGeometry = {}
        self._updating = False

    # --- UI ------------------------------------------------------------------
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = MyocardialPerfusionSPECTLogic()

        # Input
        inputSection = ctk.ctkCollapsibleButton()
        inputSection.text = "Input"
        self.layout.addWidget(inputSection)
        form = qt.QFormLayout(inputSection)

        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = False
        self.inputSelector.showHidden = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.setToolTip("Reconstructed (static or summed-gate) SPECT volume")
        form.addRow("SPECT volume:", self.inputSelector)

        self.colormapCombo = qt.QComboBox()
        self.colormapCombo.addItems([label for label, _ in COLORMAPS])
        form.addRow("Color map:", self.colormapCombo)

        self.startButton = qt.QPushButton("Start processing")
        self.startButton.toolTip = "Switch to the reorientation layout and place the heart ROI"
        form.addRow(self.startButton)

        # 1. Heart ROI
        self.roiSection = ctk.ctkCollapsibleButton()
        self.roiSection.text = "1. Heart ROI"
        self.layout.addWidget(self.roiSection)
        roiForm = qt.QFormLayout(self.roiSection)

        self.roiHint = qt.QLabel("Press Start processing first.")
        self.roiHint.wordWrap = True
        roiForm.addRow(self.roiHint)

        self.placeButton = qt.QPushButton("Place heart ROI")
        self.placeButton.toolTip = "Click the center of the left ventricle in any of the top-row views"
        roiForm.addRow(self.placeButton)

        self.radiusSlider = ctk.ctkSliderWidget()
        self.radiusSlider.minimum = self.RADIUS_MIN
        self.radiusSlider.maximum = self.RADIUS_MAX
        self.radiusSlider.singleStep = 1.0
        self.radiusSlider.decimals = 0
        self.radiusSlider.value = 55.0
        self.radiusSlider.suffix = " mm"
        self.radiusSlider.toolTip = ("Radius of the spherical ROI (also: drag the orange points in the top row). "
                                     "It is also the half size of the output volumes.")
        roiForm.addRow("ROI radius:", self.radiusSlider)

        # 2. Axes
        self.axesSection = ctk.ctkCollapsibleButton()
        self.axesSection.text = "2. Cardiac axes"
        self.layout.addWidget(self.axesSection)
        axesForm = qt.QFormLayout(self.axesSection)

        axesHint = qt.QLabel(
            "Second row: drag an end point to rotate a line, drag the middle point to move the center.<br>"
            "<b>Adjust TA</b> (transaxial): align the line with the LV long axis, base \u2192 apex.<br>"
            "<b>Adjust VLA</b> (vertical plane through that line): align the long axis elevation.<br>"
            "<b>Adjust SA</b> (short axis): rotate so the line points to the anterior wall.<br>"
            "The third row shows the resliced SA / HLA / VLA through the center.")
        axesHint.wordWrap = True
        axesForm.addRow(axesHint)

        self.angleLabel = qt.QLabel("-")
        axesForm.addRow("Angles:", self.angleLabel)

        self.resetAxesButton = qt.QPushButton("Reset axes")
        axesForm.addRow(self.resetAxesButton)

        # Bottom-row mask
        self.maskSection = ctk.ctkCollapsibleButton()
        self.maskSection.text = "Bottom-row mask"
        self.layout.addWidget(self.maskSection)
        maskForm = qt.QFormLayout(self.maskSection)

        self.maskCheckBox = qt.QCheckBox("Mask the bottom row outside the ellipsoid")
        self.maskCheckBox.checked = False
        self.maskCheckBox.toolTip = "Hide extracardiac activity (liver, gut) in the SA / HLA / VLA preview"
        maskForm.addRow(self.maskCheckBox)

        maskHint = qt.QLabel("The ellipsoid is centered on the LV and turns with the cardiac axes. "
                             "Drag the cyan points in the bottom-row views to resize it.")
        maskHint.wordWrap = True
        maskForm.addRow(maskHint)

        self.maskSemiAxisSpins = []
        for label, tip in (("Long axis:", "Half length along the long axis (apex/base)"),
                           ("Anterior-inferior:", "Half width toward the anterior/inferior walls"),
                           ("Septal-lateral:", "Half width toward the septal/lateral walls")):
            spin = qt.QDoubleSpinBox()
            spin.setRange(5.0, 150.0)
            spin.setDecimals(0)
            spin.setSingleStep(1.0)
            spin.setSuffix(" mm")
            spin.toolTip = tip
            self.maskSemiAxisSpins.append(spin)
            maskForm.addRow(label, spin)

        self.maskOutputsCheckBox = qt.QCheckBox("Also mask the created SA / HLA / VLA volumes")
        self.maskOutputsCheckBox.checked = True
        self.maskOutputsCheckBox.enabled = False
        self.maskOutputsCheckBox.toolTip = ("Zero everything outside the ellipsoid in the reoriented volumes. "
                                            "The ellipsoid is evaluated on the output grid, so its edge is not "
                                            "limited by the input voxel size.")
        maskForm.addRow(self.maskOutputsCheckBox)

        self.resetMaskButton = qt.QPushButton("Reset ellipsoid")
        maskForm.addRow(self.resetMaskButton)

        # MIP
        self.mipSection = ctk.ctkCollapsibleButton()
        self.mipSection.text = "MIP view"
        self.mipSection.collapsed = True
        self.layout.addWidget(self.mipSection)
        mipForm = qt.QFormLayout(self.mipSection)

        self.spinCheckBox = qt.QCheckBox("Rotate")
        self.spinCheckBox.checked = True
        mipForm.addRow(self.spinCheckBox)

        self.mipLevelSlider = ctk.ctkSliderWidget()
        self.mipLevelSlider.minimum = 5.0
        self.mipLevelSlider.maximum = 100.0
        self.mipLevelSlider.singleStep = 1.0
        self.mipLevelSlider.decimals = 0
        self.mipLevelSlider.value = 60.0
        self.mipLevelSlider.suffix = " %"
        self.mipLevelSlider.toolTip = "Intensity shown as full black, as a percentage of the volume maximum"
        mipForm.addRow("Upper level:", self.mipLevelSlider)

        self.mipResetButton = qt.QPushButton("Reset MIP view (anterior)")
        mipForm.addRow(self.mipResetButton)

        # 3. Output
        self.outputSection = ctk.ctkCollapsibleButton()
        self.outputSection.text = "3. Reoriented volumes"
        self.layout.addWidget(self.outputSection)
        outForm = qt.QFormLayout(self.outputSection)

        self.spacingSpin = qt.QDoubleSpinBox()
        self.spacingSpin.setRange(0.5, 20.0)
        self.spacingSpin.setDecimals(2)
        self.spacingSpin.setSingleStep(0.25)
        self.spacingSpin.setSuffix(" mm")
        self.spacingSpin.setValue(DEFAULT_PIXEL_SIZE_MM)
        self.spacingSpin.toolTip = ("In-plane pixel size of the reoriented slices. Small values (2-3 mm) "
                                    "avoid resampling artefacts.")
        outForm.addRow("In-plane pixel size:", self.spacingSpin)

        self.thicknessSpin = qt.QDoubleSpinBox()
        self.thicknessSpin.setRange(0.5, 30.0)
        self.thicknessSpin.setDecimals(2)
        self.thicknessSpin.setSingleStep(0.5)
        self.thicknessSpin.setSuffix(" mm")
        self.thicknessSpin.setValue(DEFAULT_SLICE_THICKNESS_MM)
        self.thicknessSpin.toolTip = ("Thickness and spacing of the reoriented slices. Each slice averages "
                                      "the counts across its whole thickness (thin sections are summed), "
                                      "so thicker slices are smoother and fewer.")
        outForm.addRow("Slice thickness:", self.thicknessSpin)

        self.createButton = qt.QPushButton("Create SA / HLA / VLA volumes")
        self.createButton.toolTip = "Resample the volume along the confirmed axes and show the slices"
        outForm.addRow(self.createButton)

        self.displayMaxSlider = ctk.ctkSliderWidget()
        self.displayMaxSlider.minimum = 20.0
        self.displayMaxSlider.maximum = 150.0
        self.displayMaxSlider.singleStep = 5.0
        self.displayMaxSlider.decimals = 0
        self.displayMaxSlider.value = 100.0
        self.displayMaxSlider.suffix = " %"
        self.displayMaxSlider.toolTip = "Top of the color scale in the slice panel, as a percentage of the ROI maximum"
        outForm.addRow("Display max:", self.displayMaxSlider)

        gridRow = qt.QHBoxLayout()
        self.gridRowsSpin = qt.QSpinBox()
        self.gridRowsSpin.setRange(1, 8)
        self.gridRowsSpin.setValue(DEFAULT_GRID_ROWS)
        self.gridRowsSpin.setSuffix(" rows")
        self.gridColumnsSpin = qt.QSpinBox()
        self.gridColumnsSpin.setRange(2, 30)
        self.gridColumnsSpin.setValue(DEFAULT_GRID_COLUMNS)
        self.gridColumnsSpin.setSuffix(" columns")
        gridRow.addWidget(self.gridRowsSpin)
        gridRow.addWidget(qt.QLabel("\u00d7"))
        gridRow.addWidget(self.gridColumnsSpin)
        outForm.addRow("Grid per row:", gridRow)

        self.stepSpin = qt.QSpinBox()
        self.stepSpin.setRange(1, 10)
        self.stepSpin.setValue(1)
        self.stepSpin.toolTip = "Slices advanced from one tile to the next (1 = consecutive slices)"
        outForm.addRow("Slice step:", self.stepSpin)

        self.firstSliceSpins = {}
        for name in ORIENTATIONS:
            spin = qt.QSpinBox()
            spin.setRange(1, 1)
            spin.toolTip = f"First slice of the {ORIENTATION_TITLES[name]} row (mouse wheel over the row also scrolls)"
            self.firstSliceSpins[name] = spin
            outForm.addRow(f"First {name} slice:", spin)

        buttons = qt.QHBoxLayout()
        self.showSlicesButton = qt.QPushButton("Show slices")
        self.backButton = qt.QPushButton("Back to reorientation")
        buttons.addWidget(self.showSlicesButton)
        buttons.addWidget(self.backButton)
        outForm.addRow(buttons)

        self.layout.addStretch(1)

        # Masked preview is recomputed at most every 100 ms while dragging
        self.maskTimer = qt.QTimer()
        self.maskTimer.setSingleShot(True)
        self.maskTimer.setInterval(100)

        # Connections
        self.startButton.connect("clicked(bool)", self.onStart)
        self.placeButton.connect("clicked(bool)", self.startPlacingRoi)
        self.radiusSlider.connect("valueChanged(double)", self.onRadiusChanged)
        self.resetAxesButton.connect("clicked(bool)", self.onResetAxes)
        self.maskCheckBox.connect("toggled(bool)", self.onMaskToggled)
        for index, spin in enumerate(self.maskSemiAxisSpins):
            spin.connect("valueChanged(double)", lambda value, i=index: self.onMaskSemiAxisChanged(i, value))
        self.resetMaskButton.connect("clicked(bool)", self.onResetMask)
        self.gridRowsSpin.connect("valueChanged(int)", lambda _: self.onGridChanged())
        self.gridColumnsSpin.connect("valueChanged(int)", lambda _: self.onGridChanged())
        self.maskTimer.connect("timeout()", self.updateMaskedPreview)
        self.colormapCombo.connect("currentIndexChanged(int)", self.onColormapChanged)
        self.spinCheckBox.connect("toggled(bool)", self.onSpinToggled)
        self.mipLevelSlider.connect("valueChanged(double)", lambda _: self.updateMipTransfer())
        self.mipResetButton.connect("clicked(bool)", self.resetMipCamera)
        self.createButton.connect("clicked(bool)", self.onCreateVolumes)
        self.displayMaxSlider.connect("valueChanged(double)", lambda _: self.refreshMosaic())
        self.stepSpin.connect("valueChanged(int)", lambda _: self.refreshMosaic())
        for spin in self.firstSliceSpins.values():
            spin.connect("valueChanged(int)", lambda _: self.refreshMosaic())
        self.showSlicesButton.connect("clicked(bool)", self.showResults)
        self.backButton.connect("clicked(bool)", self.onBack)

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneClosed)
        self.syncMaskSpins()
        self.updateButtonStates()

    def cleanup(self):
        self.maskTimer.stop()
        self.removeObservers()

    def updateButtonStates(self):
        started = self.inputNode is not None
        self.roiSection.enabled = started
        self.axesSection.enabled = started and self.roiPlaced
        self.maskSection.enabled = started and self.roiPlaced
        self.mipSection.enabled = started
        self.outputSection.enabled = started
        self.createButton.enabled = started and self.roiPlaced
        hasOutputs = len(self.outputNodes) == len(ORIENTATIONS)
        for widget in [self.displayMaxSlider, self.gridRowsSpin, self.gridColumnsSpin, self.stepSpin,
                       self.showSlicesButton, self.backButton] \
                + list(self.firstSliceSpins.values()):
            widget.enabled = hasOutputs

    # --- session -------------------------------------------------------------
    def onStart(self):
        node = self.inputSelector.currentNode()
        if not node or not node.GetImageData():
            slicer.util.errorDisplay("Select a SPECT volume first.")
            return
        try:
            self.removeHelperNodes()
            self.inputNode = node
            self.roiPlaced = False
            self.outputNodes = {}
            self.outputGeometry = {}
            self.logic = MyocardialPerfusionSPECTLogic()
            self.logic.radius = self.radiusSlider.value

            bounds = [0.0] * 6
            node.GetRASBounds(bounds)
            self.logic.center = np.array([(bounds[0] + bounds[1]) / 2.0,
                                          (bounds[2] + bounds[3]) / 2.0,
                                          (bounds[4] + bounds[5]) / 2.0])
            self.viewCenter = self.logic.center.copy()
            self.halfLength = {"TA": 0.75 * self.logic.radius, "VLA": 0.75 * self.logic.radius,
                               "SA": 0.6 * self.logic.radius}
            self.logic.resetMaskEllipsoid()
            self.syncMaskSpins()

            scalarRange = node.GetImageData().GetScalarRange()
            self.windowMax = None
            setVolumeDisplay(node, self.colormapCombo.currentText, 0.0, scalarRange[1])

            self.createHelperNodes()
            self.activateReorientLayout()
            self.updateButtonStates()
            self.startPlacingRoi()
        except Exception as e:
            slicer.util.errorDisplay(f"Could not start processing: {e}", detailedText=traceback.format_exc())

    def onSceneClosed(self, caller=None, event=None):
        self.maskTimer.stop()
        for node, eventId, callback in self.helperObservations:
            self.removeObserver(node, eventId, callback)
        self.helperObservations = []
        self.inputNode = None
        self.centerNode = None
        self.radiusNode = None
        self.sphereNode = None
        self.maskedNode = None
        self.maskHandleNode = None
        self.maskModelNode = None
        self.curves = {}
        self.curveKey = {}
        self.curvePositions = {}
        self.roiPlaced = False
        self.mipDisplayNode = None
        self.outputNodes = {}
        self.outputGeometry = {}
        self.outputWindowMax = None
        self.roiHint.text = "Press Start processing first."
        self.angleLabel.text = "-"
        if getattr(slicer, "_mpiSliceMosaicState", None):
            mosaicPanel().clear()
        self.updateButtonStates()

    def _observe(self, node, eventId, callback):
        self.addObserver(node, eventId, callback)
        self.helperObservations.append((node, eventId, callback))

    def createHelperNodes(self):
        scene = slicer.mrmlScene
        yellow, orange = (1.0, 0.85, 0.1), (1.0, 0.5, 0.0)

        # ROI center
        self.centerNode = scene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode",
                                                  scene.GenerateUniqueName("MPI heart center"))
        try:
            self.centerNode.SetMaximumNumberOfControlPoints(1)
        except AttributeError:
            pass
        self.centerNode.CreateDefaultDisplayNodes()
        display = self.centerNode.GetDisplayNode()
        display.SetSelectedColor(*yellow)
        display.SetColor(*yellow)
        display.SetPointLabelsVisibility(False)
        self._observe(self.centerNode, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.onCenterModified)
        self._observe(self.centerNode, slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.onCenterDefined)
        self._observe(self.centerNode, slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent, self.onInteractionEnded)

        # ROI resize handles: one toward patient left (axial + coronal views), one superior (coronal + sagittal)
        self.radiusNode = scene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode",
                                                  scene.GenerateUniqueName("MPI heart ROI radius"))
        for label in ("R1", "R2"):
            self.radiusNode.AddControlPoint(vtk.vtkVector3d(0.0, 0.0, 0.0), label)
        try:
            self.radiusNode.SetFixedNumberOfControlPoints(True)
        except AttributeError:
            pass
        self.radiusNode.CreateDefaultDisplayNodes()
        radiusDisplay = self.radiusNode.GetDisplayNode()
        radiusDisplay.SetSelectedColor(*orange)
        radiusDisplay.SetColor(*orange)
        radiusDisplay.SetPointLabelsVisibility(False)
        radiusDisplay.SetVisibility(False)
        self._observe(self.radiusNode, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.onRadiusHandleModified)
        self._observe(self.radiusNode, slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent, self.onInteractionEnded)

        # Spherical ROI, drawn as its intersection circle in the top row
        self.sphereNode = scene.AddNewNodeByClass("vtkMRMLModelNode", scene.GenerateUniqueName("MPI heart ROI"))
        self.sphereNode.CreateDefaultDisplayNodes()
        sphereDisplay = self.sphereNode.GetDisplayNode()
        sphereDisplay.SetColor(*yellow)
        sphereDisplay.SetVisibility2D(True)
        sphereDisplay.SetSliceIntersectionThickness(2)
        sphereDisplay.SetVisibility(False)

        # Mask ellipsoid (bottom row): outline model + one handle per semi-axis.
        # Handle 0 (long axis) lies in the HLA and VLA planes, handle 1 (anterior) in SA and VLA,
        # handle 2 (lateral) in SA and HLA, so every bottom view shows two of them.
        cyan = (0.1, 0.85, 1.0)
        self.maskModelNode = scene.AddNewNodeByClass("vtkMRMLModelNode", scene.GenerateUniqueName("MPI mask ellipsoid"))
        self.maskModelNode.CreateDefaultDisplayNodes()
        maskDisplay = self.maskModelNode.GetDisplayNode()
        maskDisplay.SetColor(*cyan)
        maskDisplay.SetVisibility2D(True)
        maskDisplay.SetSliceIntersectionThickness(2)
        maskDisplay.SetVisibility(False)
        self.maskHandleNode = scene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode",
                                                      scene.GenerateUniqueName("MPI mask handles"))
        for label in ("Long", "Ant", "Lat"):
            self.maskHandleNode.AddControlPoint(vtk.vtkVector3d(0.0, 0.0, 0.0), label)
        try:
            self.maskHandleNode.SetFixedNumberOfControlPoints(True)
        except AttributeError:
            pass
        self.maskHandleNode.CreateDefaultDisplayNodes()
        handleDisplay = self.maskHandleNode.GetDisplayNode()
        handleDisplay.SetSelectedColor(*cyan)
        handleDisplay.SetColor(*cyan)
        handleDisplay.SetPointLabelsVisibility(False)
        handleDisplay.SetVisibility(False)
        self._observe(self.maskHandleNode, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.onMaskHandleModified)
        self._observe(self.maskHandleNode, slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent, self.onInteractionEnded)

        # One axis line per adjustment view: base end, center, apex (or inferior, center, anterior)
        colors = {"TA": (0.85, 0.11, 0.38), "VLA": (0.26, 0.63, 0.28), "SA": (0.99, 0.85, 0.21)}
        labels = {"TA": ("Base", "", "Apex"), "VLA": ("Base", "", "Apex"), "SA": ("Inf", "", "Ant")}
        self.curves, self.curveKey, self.curvePositions = {}, {}, {}
        for which in EDIT_ORDER:
            curve = scene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", scene.GenerateUniqueName(f"MPI axis {which}"))
            curve.SetCurveTypeToLinear()
            for label in labels[which]:
                curve.AddControlPoint(vtk.vtkVector3d(0.0, 0.0, 0.0), label)
            try:
                curve.SetFixedNumberOfControlPoints(True)
            except AttributeError:
                pass
            try:
                measurement = curve.GetMeasurement("length")
                if measurement:
                    measurement.SetEnabled(False)
            except AttributeError:
                pass
            curve.CreateDefaultDisplayNodes()
            curveDisplay = curve.GetDisplayNode()
            curveDisplay.SetSelectedColor(*colors[which])
            curveDisplay.SetColor(*colors[which])
            curveDisplay.SetPropertiesLabelVisibility(False)
            curveDisplay.SetPointLabelsVisibility(True)
            curveDisplay.SetVisibility(False)
            self.curves[which] = curve
            self.curveKey[curve.GetID()] = which
            self._observe(curve, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.onAxisCurveModified)
            self._observe(curve, slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent, self.onInteractionEnded)

    def assignHelperViews(self):
        """Restrict helper nodes to their views (view nodes exist only once the layout is shown)."""
        row1 = [sliceNodeByTag(t).GetID() for t in ROW1_VIEWS if sliceNodeByTag(t)]
        for node in (self.centerNode, self.radiusNode, self.sphereNode):
            if node and node.GetDisplayNode() and row1:
                node.GetDisplayNode().SetViewNodeIDs(row1)
        bottom = [sliceNodeByTag(t).GetID() for t in PREVIEW_VIEWS.values() if sliceNodeByTag(t)]
        for node in (self.maskHandleNode, self.maskModelNode):
            if node and node.GetDisplayNode() and bottom:
                node.GetDisplayNode().SetViewNodeIDs(bottom)
        for which, curve in self.curves.items():
            sliceNode = sliceNodeByTag(EDIT_VIEWS[which])
            if sliceNode and curve.GetDisplayNode():
                curve.GetDisplayNode().SetViewNodeIDs([sliceNode.GetID()])

    def removeHelperNodes(self):
        self.maskTimer.stop()
        for node, eventId, callback in self.helperObservations:
            self.removeObserver(node, eventId, callback)
        self.helperObservations = []
        for node in [self.centerNode, self.radiusNode, self.sphereNode, self.maskedNode,
                     self.maskHandleNode, self.maskModelNode] + list(self.curves.values()):
            if node and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        self.centerNode = None
        self.radiusNode = None
        self.sphereNode = None
        self.maskedNode = None
        self.maskHandleNode = None
        self.maskModelNode = None
        self.curves, self.curveKey, self.curvePositions = {}, {}, {}
        self.radiusHandlePositions = None
        self.maskHandlePositions = None

    # --- layouts -------------------------------------------------------------
    def activateReorientLayout(self):
        registerLayout(REORIENT_LAYOUT_ID, reorientLayoutXml())
        slicer.app.layoutManager().setLayout(REORIENT_LAYOUT_ID)
        slicer.app.processEvents()

        for tag in ROW1_VIEWS + tuple(EDIT_VIEWS.values()):
            showVolumeInView(tag, self.inputNode)
        for tag, orientation in zip(ROW1_VIEWS, ROW1_ORIENTATIONS):
            sliceNode = sliceNodeByTag(tag)
            if sliceNode:
                sliceNode.SetOrientation(orientation)
        self.updateMaskedPreview()  # sets the bottom-row background (masked or not)

        self.assignHelperViews()
        self.setupMip()
        # Views get their real pixel size only after Qt has laid them out
        qt.QTimer.singleShot(150, self.refreshReorientViews)

    def refreshReorientViews(self):
        if not self.inputNode:
            return
        layoutManager = slicer.app.layoutManager()
        tags = ROW1_VIEWS
        if not self.roiPlaced:
            tags = tags + tuple(EDIT_VIEWS.values()) + tuple(PREVIEW_VIEWS.values())
        for tag in tags:
            widget = layoutManager.sliceWidget(tag)
            if widget:
                widget.sliceLogic().FitSliceToAll()
        if self.roiPlaced:
            self.viewCenter = self.logic.center.copy()
            self.updateGeometry()
        self.resetMipCamera()

    # --- MIP -----------------------------------------------------------------
    def setupMip(self):
        widget = threeDWidgetByTag(MIP_VIEW)
        if not widget or not self.inputNode:
            return
        viewNode = widget.mrmlViewNode()
        viewNode.SetBackgroundColor(1.0, 1.0, 1.0)
        viewNode.SetBackgroundColor2(1.0, 1.0, 1.0)
        viewNode.SetBoxVisible(False)
        viewNode.SetAxisLabelsVisible(False)
        viewNode.SetRenderMode(slicer.vtkMRMLViewNode.Orthographic)  # parallel projection like a planar MIP
        viewNode.SetRaycastTechnique(slicer.vtkMRMLViewNode.MaximumIntensityProjection)

        vrLogic = slicer.modules.volumerendering.logic()
        displayNode = vrLogic.GetFirstVolumeRenderingDisplayNode(self.inputNode)
        if not displayNode:
            displayNode = vrLogic.CreateDefaultVolumeRenderingNodes(self.inputNode)
        displayNode.SetViewNodeIDs([viewNode.GetID()])
        displayNode.SetVisibility(True)
        self.mipDisplayNode = displayNode
        self.updateMipTransfer()
        self.onSpinToggled(self.spinCheckBox.checked)

    def resetMipCamera(self):
        """Anterior view, head up, whole volume in view for every rotation angle."""
        widget = threeDWidgetByTag(MIP_VIEW)
        if not widget or not self.inputNode:
            return
        try:
            viewNode = widget.mrmlViewNode()
            cameraNode = slicer.modules.cameras.logic().GetViewActiveCameraNode(viewNode)
            if not cameraNode:
                return
            bounds = [0.0] * 6
            self.inputNode.GetRASBounds(bounds)
            center = [(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, (bounds[4] + bounds[5]) / 2.0]
            extentX, extentY, extentZ = bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]
            horizontal = math.hypot(extentX, extentY)  # widest footprint while rotating about the vertical axis
            distance = 2.0 * max(extentX, extentY, extentZ) + 500.0

            rect = widget.threeDView().contentsRect()
            aspect = rect.width() / float(rect.height()) if rect.height() > 0 else 0.4

            cameraNode.SetFocalPoint(center[0], center[1], center[2])
            cameraNode.SetPosition(center[0], center[1] + distance, center[2])  # camera anterior, looking posterior
            cameraNode.SetViewUp(0.0, 0.0, 1.0)                                  # superior up
            cameraNode.SetParallelScale(0.5 * max(extentZ, horizontal / aspect) * 1.05)
            cameraNode.ResetClippingRange()
        except Exception:
            logging.warning("Could not reset the MIP camera:\n" + traceback.format_exc())

    def updateMipTransfer(self):
        if not (self.mipDisplayNode and self.inputNode and self.inputNode.GetImageData()):
            return
        maxValue = self.inputNode.GetImageData().GetScalarRange()[1]
        upper = max(maxValue * self.mipLevelSlider.value / 100.0, 1e-6)
        opacity = vtk.vtkPiecewiseFunction()
        opacity.AddPoint(0.0, 0.0)
        opacity.AddPoint(upper, 1.0)
        color = vtk.vtkColorTransferFunction()  # inverted grey: black activity on white, NM convention
        color.AddRGBPoint(0.0, 1.0, 1.0, 1.0)
        color.AddRGBPoint(upper, 0.0, 0.0, 0.0)
        propertyNode = self.mipDisplayNode.GetVolumePropertyNode()
        propertyNode.SetScalarOpacity(opacity)
        propertyNode.SetColor(color)
        propertyNode.GetVolumeProperty().ShadeOff()

    def onSpinToggled(self, on):
        widget = threeDWidgetByTag(MIP_VIEW)
        if not widget:
            return
        viewNode = widget.mrmlViewNode()
        try:
            viewNode.SetSpinDirection(slicer.vtkMRMLViewNode.YawLeft)  # about the vertical (superior) axis
            viewNode.SetSpinDegrees(MIP_SPIN_DEGREES)
            viewNode.SetAnimationMode(slicer.vtkMRMLViewNode.Spin if on else slicer.vtkMRMLViewNode.Off)
        except AttributeError:
            logging.warning("This Slicer version does not support spinning the 3D view.")
        if not on:
            self.resetMipCamera()  # stopped: back to the upright anterior view

    # --- ROI -----------------------------------------------------------------
    def startPlacingRoi(self):
        if not self.centerNode:
            return
        self._updating = True
        try:
            self.centerNode.RemoveAllControlPoints()
        finally:
            self._updating = False
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        selectionNode.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selectionNode.SetActivePlaceNodeID(self.centerNode.GetID())
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetPlaceModePersistence(0)
        interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        self.roiHint.text = ("<b>Click the center of the left ventricle</b> in one of the top-row views. "
                             "Afterwards drag the yellow point to move the ROI and the orange points to resize it.")
        slicer.util.showStatusMessage("Click the center of the left ventricle in the top row.", 5000)

    def centerFromMarkup(self):
        if not self.centerNode or self.centerNode.GetNumberOfControlPoints() < 1:
            return None
        if self.centerNode.GetNthControlPointPositionStatus(0) != slicer.vtkMRMLMarkupsNode.PositionDefined:
            return None  # placement preview following the mouse
        position = [0.0, 0.0, 0.0]
        self.centerNode.GetNthControlPointPositionWorld(0, position)
        return np.array(position)

    def onCenterModified(self, caller, event):
        if self._updating:
            return
        center = self.centerFromMarkup()
        if center is None:
            return
        self.logic.center = center
        self.viewCenter = center.copy()  # top-row drags never pan these views, so re-centering is safe
        if not self.roiPlaced:
            self.onRoiPlaced()
        self.updateGeometry()

    def onCenterDefined(self, caller, event):
        self.onCenterModified(caller, event)
        self.updateWindowLevel()

    def onRoiPlaced(self):
        self.roiPlaced = True
        for node in [self.sphereNode, self.radiusNode] + list(self.curves.values()):
            node.GetDisplayNode().SetVisibility(True)
        self.roiHint.text = ("ROI placed. Drag the yellow point to move it and the orange points to resize it. "
                             "Then adjust the axes in the second row.")
        self.updateMaskVisibility()
        self.updateButtonStates()

    def radiusHandleTargets(self):
        c, r = self.logic.center, self.logic.radius
        return np.array([c + r * PATIENT_LEFT, c + r * SUPERIOR])

    def onRadiusHandleModified(self, caller, event):
        if self._updating or not self.roiPlaced or self.radiusHandlePositions is None:
            return
        for index in range(2):
            position = self.pointWorld(caller, index)
            if np.linalg.norm(position - self.radiusHandlePositions[index]) > 1e-3:
                self.draggedRadiusHandle = index
                self.radiusHandlePositions[index] = position
                radius = float(np.clip(np.linalg.norm(position - self.logic.center),
                                       self.RADIUS_MIN, self.RADIUS_MAX))
                self.logic.radius = radius
                wasBlocked = self.radiusSlider.blockSignals(True)
                self.radiusSlider.value = radius
                self.radiusSlider.blockSignals(wasBlocked)
                self.updateGeometry()
                return

    def onRadiusChanged(self, value):
        self.logic.radius = float(value)
        if self.roiPlaced:
            self.updateGeometry()
            self.updateWindowLevel()

    def updateSphere(self):
        if not self.sphereNode:
            return
        source = vtk.vtkSphereSource()
        source.SetCenter(*[float(x) for x in self.logic.center])
        source.SetRadius(self.logic.radius)
        source.SetThetaResolution(64)
        source.SetPhiResolution(32)
        source.Update()
        self.sphereNode.SetAndObservePolyData(source.GetOutput())

    def updateRadiusHandles(self):
        if not self.radiusNode or self.radiusNode.GetNumberOfControlPoints() != 2:
            return
        targets = self.radiusHandleTargets()
        wasModifying = self.radiusNode.StartModify()
        for index in range(2):
            if index == self.draggedRadiusHandle:
                continue  # leave the handle under the mouse alone until the drag ends
            p = targets[index]
            self.radiusNode.SetNthControlPointPositionWorld(index, float(p[0]), float(p[1]), float(p[2]))
        self.radiusNode.EndModify(wasModifying)
        self.radiusHandlePositions = np.array([self.pointWorld(self.radiusNode, i) for i in range(2)])

    def updateWindowLevel(self):
        """Scale the display to the hottest voxel in the ROI (the myocardium) rather than
        the whole volume, where gut or liver activity would wash the heart out."""
        if not (self.inputNode and self.roiPlaced):
            return
        try:
            self.windowMax = self.logic.maxInSphere(self.inputNode, self.logic.center, self.logic.radius)
        except Exception:
            logging.warning("Could not compute ROI maximum:\n" + traceback.format_exc())
            return
        colormap = self.colormapCombo.currentText
        setVolumeDisplay(self.inputNode, colormap, 0.0, self.windowMax)
        if self.maskedNode and slicer.mrmlScene.IsNodePresent(self.maskedNode):
            setVolumeDisplay(self.maskedNode, colormap, 0.0, self.windowMax)

    def updateMaskedPreview(self):
        """Bottom row shows either the input or a copy with everything outside the ROI zeroed."""
        if not self.inputNode:
            return
        background = self.inputNode
        if self.maskCheckBox.checked and self.roiPlaced:
            try:
                if not (self.maskedNode and slicer.mrmlScene.IsNodePresent(self.maskedNode)):
                    self.maskedNode = slicer.mrmlScene.AddNewNodeByClass(
                        "vtkMRMLScalarVolumeNode", slicer.mrmlScene.GenerateUniqueName("MPI masked preview"))
                    self.maskedNode.SetHideFromEditors(True)
                    self.maskedNode.SetSaveWithScene(False)
                ijkToRas = vtk.vtkMatrix4x4()
                self.inputNode.GetIJKToRASMatrix(ijkToRas)
                self.maskedNode.SetIJKToRASMatrix(ijkToRas)
                self.maskedNode.SetAndObserveTransformNodeID(self.inputNode.GetTransformNodeID())
                masked = self.logic.maskedArray(self.inputNode, self.logic.center,
                                                self.logic.maskAxes(), self.logic.maskSemiAxes)
                slicer.util.updateVolumeFromArray(self.maskedNode, masked)
                windowMax = self.windowMax if self.windowMax else float(masked.max())
                setVolumeDisplay(self.maskedNode, self.colormapCombo.currentText, 0.0, windowMax)
                background = self.maskedNode
            except Exception:
                logging.error("Could not mask the preview:\n" + traceback.format_exc())
        for tag in PREVIEW_VIEWS.values():
            showVolumeInView(tag, background)

    # --- mask ellipsoid --------------------------------------------------------
    def maskHandleDirections(self):
        f = self.logic.frame()
        return np.array([f["u"], f["w"], -f["s"]])  # apex, anterior, lateral

    def syncMaskSpins(self):
        for spin, value in zip(self.maskSemiAxisSpins, self.logic.maskSemiAxes):
            wasBlocked = spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(wasBlocked)

    def updateMaskVisibility(self):
        visible = bool(self.maskCheckBox.checked and self.roiPlaced)
        for node in (self.maskHandleNode, self.maskModelNode):
            if node and node.GetDisplayNode():
                node.GetDisplayNode().SetVisibility(visible)

    def onMaskToggled(self, checked):
        self.maskOutputsCheckBox.enabled = checked
        self.updateMaskVisibility()
        self.updateGeometry()
        self.updateMaskedPreview()

    def onMaskSemiAxisChanged(self, index, value):
        self.logic.maskSemiAxes[index] = float(value)
        self.updateGeometry()

    def onResetMask(self):
        self.logic.resetMaskEllipsoid()
        self.syncMaskSpins()
        self.updateGeometry()

    def onMaskHandleModified(self, caller, event):
        if self._updating or not self.roiPlaced or self.maskHandlePositions is None:
            return
        directions = self.maskHandleDirections()
        for index in range(3):
            position = self.pointWorld(caller, index)
            if np.linalg.norm(position - self.maskHandlePositions[index]) > 1e-3:
                self.draggedMaskHandle = index
                self.maskHandlePositions[index] = position
                semi = abs(float(np.dot(position - self.logic.center, directions[index])))
                self.logic.maskSemiAxes[index] = float(np.clip(semi, 5.0, 150.0))
                self.syncMaskSpins()
                self.updateGeometry()
                return

    def updateMaskShape(self):
        """Ellipsoid outline and handles; only drawn while masking is on."""
        if not (self.maskModelNode and self.maskHandleNode and self.maskCheckBox.checked):
            return
        c = np.asarray(self.logic.center, dtype=float)
        semi = self.logic.maskSemiAxes
        directions = self.maskHandleDirections()

        transform = vtk.vtkTransform()
        matrix = vtk.vtkMatrix4x4()
        for r in range(3):
            for col in range(3):
                matrix.SetElement(r, col, directions[col][r] * semi[col])
            matrix.SetElement(r, 3, c[r])
        transform.SetMatrix(matrix)
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(1.0)
        sphere.SetThetaResolution(64)
        sphere.SetPhiResolution(48)
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputConnection(sphere.GetOutputPort())
        transformFilter.SetTransform(transform)
        transformFilter.Update()
        self.maskModelNode.SetAndObservePolyData(transformFilter.GetOutput())

        if self.maskHandleNode.GetNumberOfControlPoints() == 3:
            wasModifying = self.maskHandleNode.StartModify()
            for index in range(3):
                if index == self.draggedMaskHandle:
                    continue  # leave the handle under the mouse alone until the drag ends
                p = c + semi[index] * directions[index]
                self.maskHandleNode.SetNthControlPointPositionWorld(index, float(p[0]), float(p[1]), float(p[2]))
            self.maskHandleNode.EndModify(wasModifying)
            self.maskHandlePositions = np.array([self.pointWorld(self.maskHandleNode, i) for i in range(3)])

    # --- axes ----------------------------------------------------------------
    def cropFieldOfView(self):
        return 2.2 * self.logic.radius

    @staticmethod
    def pointWorld(markupsNode, index):
        position = [0.0, 0.0, 0.0]
        markupsNode.GetNthControlPointPositionWorld(index, position)
        return np.array(position)

    def onAxisCurveModified(self, caller, event):
        if self._updating or not self.roiPlaced:
            return
        which = self.curveKey.get(caller.GetID())
        old = self.curvePositions.get(which)
        if which is None or old is None or caller.GetNumberOfControlPoints() != 3:
            return
        new = np.array([self.pointWorld(caller, i) for i in range(3)])
        moved = np.linalg.norm(new - old, axis=1) > 1e-3
        if not moved.any():
            return
        if moved[1]:
            # middle point (or whole line) moved: translate the center
            self.logic.center = new[1].copy()
        else:
            index = 0 if moved[0] else 2
            vector = new[index] - self.logic.center
            if index == 0:
                vector = -vector
            length = float(np.linalg.norm(vector))
            if length > 3.0:
                self.logic.setLineDirection(which, vector)
                self.halfLength[which] = float(np.clip(length, 8.0, 1.5 * self.logic.radius))
        # The view being dragged in is not re-centered until the drag ends; panning under the
        # mouse would feed back into the drag.
        self.updateGeometry()

    def onInteractionEnded(self, caller, event):
        if self._updating or not self.roiPlaced:
            return
        self.draggedRadiusHandle = None
        self.draggedMaskHandle = None
        self.viewCenter = self.logic.center.copy()
        self.updateGeometry()
        self.updateWindowLevel()

    def onResetAxes(self):
        self.logic.resetAxes()
        r = self.logic.radius
        self.halfLength = {"TA": 0.75 * r, "VLA": 0.75 * r, "SA": 0.6 * r}
        self.updateGeometry()

    def setCurve(self, which, center, direction):
        curve = self.curves.get(which)
        if not curve:
            return
        half = float(np.clip(self.halfLength.get(which, 0.7 * self.logic.radius), 8.0, 1.5 * self.logic.radius))
        points = [center - half * direction, center, center + half * direction]
        wasModifying = curve.StartModify()
        for index, p in enumerate(points):
            curve.SetNthControlPointPositionWorld(index, float(p[0]), float(p[1]), float(p[2]))
        curve.EndModify(wasModifying)
        self.curvePositions[which] = np.array(points)

    def updateGeometry(self):
        if not self.roiPlaced or not self.inputNode:
            return
        self._updating = True
        try:
            c = np.asarray(self.logic.center, dtype=float)

            # top row: ROI marker, resize handles, sphere, slices through the center
            markupCenter = self.centerFromMarkup()
            if markupCenter is not None and np.linalg.norm(markupCenter - c) > 1e-3:
                self.centerNode.SetNthControlPointPositionWorld(0, float(c[0]), float(c[1]), float(c[2]))
            self.updateRadiusHandles()
            self.updateSphere()
            self.updateMaskShape()
            for tag in ROW1_VIEWS:
                sliceNode = sliceNodeByTag(tag)
                if sliceNode:
                    sliceNode.JumpSliceByOffsetting(float(c[0]), float(c[1]), float(c[2]))

            fov = self.cropFieldOfView()

            # second row: adjustment planes pass through the center, panned to the view center
            for which in EDIT_ORDER:
                geometry = self.logic.editViewGeometry(which)
                normal = _unit(geometry["normal"])
                origin = self.viewCenter - np.dot(self.viewCenter - c, normal) * normal
                setSliceGeometry(sliceNodeByTag(EDIT_VIEWS[which]),
                                 geometry["right"], geometry["up"], normal, origin, fov)
                self.setCurve(which, c, _unit(geometry["line"]))

            # third row: live SA / HLA / VLA preview
            axes = self.logic.orientationAxes()
            for name in ORIENTATIONS:
                right, up, stack = axes[name]
                setSliceGeometry(sliceNodeByTag(PREVIEW_VIEWS[name]), right, up, stack, c, fov)

            self.angleLabel.text = (f"azimuth {self.logic.azimuthDegrees():.0f}\u00b0   "
                                    f"elevation {math.degrees(self.logic.elevation()):.0f}\u00b0   "
                                    f"roll {math.degrees(self.logic.roll):.0f}\u00b0")
        except Exception:
            logging.error("Failed to update reorientation views:\n" + traceback.format_exc())
        finally:
            self._updating = False
        if self.maskCheckBox.checked and not self.maskTimer.isActive():
            self.maskTimer.start()

    # --- output --------------------------------------------------------------
    def onCreateVolumes(self):
        if not (self.inputNode and self.roiPlaced):
            slicer.util.errorDisplay("Place the heart ROI first.")
            return
        with slicer.util.tryWithErrorDisplay("Failed to create the reoriented volumes.", waitCursor=True):
            # Re-running replaces the volumes from the previous confirmation
            for node in self.outputNodes.values():
                if node and slicer.mrmlScene.IsNodePresent(node):
                    slicer.mrmlScene.RemoveNode(node)
            self.outputNodes, self.outputGeometry = {}, {}

            if self.windowMax is None:
                self.updateWindowLevel()
            spacing = float(self.spacingSpin.value)
            thickness = float(self.thicknessSpin.value)
            axes = self.logic.orientationAxes()
            baseName = self.inputNode.GetName()
            f = self.logic.frame()
            maskOutputs = bool(self.maskCheckBox.checked and self.maskOutputsCheckBox.checked)
            maskAxes = self.logic.maskAxes()
            maskSemiAxes = np.array(self.logic.maskSemiAxes, dtype=float)
            maskText = ("ellipsoid semi-axes (long, anterior-inferior, septal-lateral) "
                        + ", ".join(f"{a:.0f}" for a in maskSemiAxes) + " mm") if maskOutputs else "none"
            suffix = " masked" if maskOutputs else ""
            for name in ORIENTATIONS:
                node, geometry = self.logic.resampleOrientation(
                    self.inputNode, axes[name], spacing, f"{baseName} {name}{suffix}",
                    sliceThickness=thickness, mask=(maskAxes, maskSemiAxes) if maskOutputs else None)
                for attribute in self.inputNode.GetAttributeNames() or ():
                    node.SetAttribute(attribute, self.inputNode.GetAttribute(attribute))
                node.SetAttribute("MPI.Orientation", name)
                node.SetAttribute("MPI.SourceVolume", self.inputNode.GetID())
                node.SetAttribute("MPI.Center", " ".join(f"{x:.2f}" for x in self.logic.center))
                node.SetAttribute("MPI.LongAxis", " ".join(f"{x:.4f}" for x in f["u"]))
                node.SetAttribute("MPI.AnteriorAxis", " ".join(f"{x:.4f}" for x in f["w"]))
                node.SetAttribute("MPI.RollDeg", f"{math.degrees(self.logic.roll):.2f}")
                node.SetAttribute("MPI.RadiusMm", f"{self.logic.radius:.1f}")
                node.SetAttribute("MPI.Mask", maskText)
                node.SetAttribute("MPI.PixelSizeMm", f"{spacing:.2f}")
                node.SetAttribute("MPI.SliceThicknessMm", f"{thickness:.2f}")
                placeNextTo(node, self.inputNode)
                self.outputNodes[name] = node
                self.outputGeometry[name] = geometry

            # Scale to the hottest voxel of the reoriented data itself (thick slices average counts, so
            # their peak is lower than the input's): the whole volume when masked, the ROI otherwise.
            if maskOutputs:
                peak = max(float(slicer.util.arrayFromVolume(n).max()) for n in self.outputNodes.values())
            else:
                peak = max(self.logic.maxInSphere(n, self.logic.center, self.logic.radius)
                           for n in self.outputNodes.values())
            self.outputWindowMax = peak if peak > 0 else None
            for node in self.outputNodes.values():
                setVolumeDisplay(node, self.colormapCombo.currentText, 0.0, self.outputWindowMax or self.windowMax)
            self.resetFirstSlices()

        self.updateButtonStates()
        if len(self.outputNodes) == len(ORIENTATIONS):
            self.showResults()

    def cellCount(self):
        return self.gridRowsSpin.value * self.gridColumnsSpin.value

    def cellsSpan(self):
        return (self.cellCount() - 1) * self.stepSpin.value + 1

    def onGridChanged(self):
        state = getattr(slicer, "_mpiSliceMosaicState", None)
        if state and state.get("panel"):
            state["panel"].setGrid(self.gridRowsSpin.value, self.gridColumnsSpin.value)
        self.refreshMosaic()

    def resetFirstSlices(self):
        for name in ORIENTATIONS:
            n = self.outputGeometry[name]["n"]
            spin = self.firstSliceSpins[name]
            wasBlocked = spin.blockSignals(True)
            spin.setRange(1, n)
            spin.setValue(max(0, (n - self.cellsSpan()) // 2) + 1)
            spin.blockSignals(wasBlocked)

    def showResults(self):
        if len(self.outputNodes) != len(ORIENTATIONS):
            return
        # Leave place mode in case the user was about to re-place the ROI
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

        panel = mosaicPanel()
        panel.onScroll = self.onMosaicScroll
        panel.gridRows, panel.gridColumns = self.gridRowsSpin.value, self.gridColumnsSpin.value
        registerLayout(RESULT_LAYOUT_ID, resultLayoutXml())
        slicer.app.layoutManager().setLayout(RESULT_LAYOUT_ID)
        slicer.app.processEvents()
        self.refreshMosaic()

    def onMosaicScroll(self, name, direction):
        spin = self.firstSliceSpins.get(name)
        if spin:
            spin.setValue(spin.value + direction * self.stepSpin.value)

    def refreshMosaic(self):
        """Render each row: tile t shows slice first + t * step of that orientation."""
        if len(self.outputNodes) != len(ORIENTATIONS):
            return
        state = getattr(slicer, "_mpiSliceMosaicState", None)
        if state is None:
            return
        panel = state["panel"]
        if panel is None:
            return
        panel.gridRows, panel.gridColumns = self.gridRowsSpin.value, self.gridColumnsSpin.value
        colorTable = colorTableArray(self.colormapCombo.currentText)
        reference = self.outputWindowMax or self.windowMax or max(float(slicer.util.arrayFromVolume(n).max())
                                          for n in self.outputNodes.values())
        top = max(reference * self.displayMaxSlider.value / 100.0, 1e-9)
        step = self.stepSpin.value
        for name in ORIENTATIONS:
            node = self.outputNodes.get(name)
            if not node or not slicer.mrmlScene.IsNodePresent(node):
                continue
            array = slicer.util.arrayFromVolume(node)  # [k, j, i]; i = screen right, j = screen up
            first = self.firstSliceSpins[name].value - 1
            tiles = []
            for cell in range(self.cellCount()):
                k = first + cell * step
                if k >= array.shape[0]:
                    tiles.append((None, None))
                    continue
                sliceImage = array[k, ::-1, :]  # top image row = highest j
                indices = np.clip(sliceImage / top * 255.0, 0, 255).astype(np.uint8)
                tiles.append((rgbToQImage(colorTable[indices]), k + 1))
            last = min(first + (self.cellCount() - 1) * step, array.shape[0] - 1)
            panel.setRow(name, tiles,
                         f"{ORIENTATION_TITLES[name]}   \u2014   slices {first + 1}\u2013{last + 1} of {array.shape[0]}"
                         + (f", every {step}" if step > 1 else ""))

    def onBack(self):
        if not self.inputNode:
            return
        self.activateReorientLayout()

    def onColormapChanged(self, index):
        label = self.colormapCombo.currentText
        for node in [self.inputNode, self.maskedNode] + list(self.outputNodes.values()):
            if node and slicer.mrmlScene.IsNodePresent(node):
                setVolumeDisplay(node, label)
        self.refreshMosaic()


# ---------------------------------------------------------------------------
# Tests (Reload and Test in developer mode)
# ---------------------------------------------------------------------------
class MyocardialPerfusionSPECTTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_FrameIsOrthonormal()
        self.setUp()
        self.test_ResamplingKeepsAnatomy()

    def _obliqueLogic(self):
        logic = MyocardialPerfusionSPECTLogic()
        logic.setLineDirection("TA", [-0.3, 1.0, 0.0])
        logic.setLineDirection("VLA", logic.horizontalDirection() - 0.6 * SUPERIOR)
        logic.roll = math.radians(15.0)
        return logic

    def test_FrameIsOrthonormal(self):
        logic = self._obliqueLogic()
        f = logic.frame()
        for name, axes in logic.orientationAxes().items():
            m = np.column_stack(axes)
            self.assertTrue(np.allclose(m.T @ m, np.eye(3), atol=1e-6), name)
        self.assertAlmostEqual(float(np.dot(logic.orientationAxes()["SA"][2], f["u"])), -1.0, places=6)
        # septum lies to the patient's right of the LV for a normal heart
        self.assertGreater(float(f["s"][0]), 0.0)
        self.delayDisplay("Frame test passed")

    def test_ResamplingKeepsAnatomy(self):
        array = np.zeros((30, 40, 50), dtype=np.float32)   # k, j, i
        array[14:17, 19:22, 29:32] = 100.0                 # hot spot centered at IJK (30, 20, 15)
        volume = slicer.util.addVolumeFromArray(array, name="MPI test phantom")
        volume.SetSpacing(4.0, 4.0, 4.0)
        volume.SetOrigin(-100.0, -80.0, -60.0)
        hotRas = np.array([-100.0 + 30 * 4.0, -80.0 + 20 * 4.0, -60.0 + 15 * 4.0])

        logic = self._obliqueLogic()
        logic.center = hotRas + np.array([6.0, -4.0, 5.0])
        logic.radius = 50.0
        self.assertAlmostEqual(logic.maxInSphere(volume, logic.center, logic.radius), 100.0)
        axes = logic.maskAxes()
        masked = logic.maskedArray(volume, logic.center, axes, [30.0, 15.0, 15.0])
        self.assertEqual(float(masked.max()), 100.0)
        self.assertEqual(float(logic.maskedArray(volume, hotRas + 60.0, axes, [20.0, 20.0, 20.0]).max()), 0.0)
        sa = logic.orientationAxes()["SA"]
        output, _ = logic.resampleOrientation(volume, sa, 4.0, "test masked SA", mask=(axes, [30.0, 15.0, 15.0]))
        self.assertGreater(float(slicer.util.arrayFromVolume(output).max()), 0.0)
        output, _ = logic.resampleOrientation(volume, sa, 4.0, "test masked SA",
                                              mask=(axes, [10.0, 10.0, 10.0]))
        logic.applyEllipsoidMask(output, hotRas + 60.0, axes, [10.0, 10.0, 10.0])
        self.assertEqual(float(slicer.util.arrayFromVolume(output).max()), 0.0)

        # thick slices: fewer slices, anisotropic spacing, counts averaged across the slab
        thick, geometry = logic.resampleOrientation(volume, sa, 2.0, "test thick SA", sliceThickness=5.0)
        self.assertEqual(slicer.util.arrayFromVolume(thick).shape, (20, 50, 50))
        self.assertEqual(geometry["sectionsPerSlice"], 3)
        self.assertTrue(np.allclose(thick.GetSpacing(), (2.0, 2.0, 5.0)))
        thin, _ = logic.resampleOrientation(volume, sa, 2.0, "test thin SA")
        totalThick = float(slicer.util.arrayFromVolume(thick).sum()) * 2.0 * 2.0 * 5.0
        totalThin = float(slicer.util.arrayFromVolume(thin).sum()) * 2.0 * 2.0 * 2.0
        self.assertLess(abs(totalThick - totalThin) / totalThin, 0.05)

        for name, axes in logic.orientationAxes().items():
            output, _ = logic.resampleOrientation(volume, axes, 4.0, f"test {name}")
            values = slicer.util.arrayFromVolume(output).astype(float)
            kji = np.indices(values.shape).reshape(3, -1)
            weights = values.reshape(-1)
            ijk = np.vstack([kji[2], kji[1], kji[0], np.ones(kji.shape[1])])
            m = vtk.vtkMatrix4x4()
            output.GetIJKToRASMatrix(m)
            ras = (slicer.util.arrayFromVTKMatrix(m) @ ijk)[:3]
            centroid = (ras * weights).sum(axis=1) / weights.sum()
            self.assertLess(float(np.linalg.norm(centroid - hotRas)), 2.0, name)
        self.delayDisplay("Resampling test passed")
