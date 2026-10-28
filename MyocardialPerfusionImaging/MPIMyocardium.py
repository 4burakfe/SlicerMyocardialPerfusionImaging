"""
Myocardial Perfusion Imaging - 3. Myocardium (prototype)

Automatic detection of the left ventricular wall of the active study, in the cardiac frame set in
Processing: mid-wall, endocardial and epicardial surfaces and the valve plane, by count-profile
sampling (see MyocardialPerfusionImagingLib/myocardium.py). The counts are sampled from the SA volume
made in Processing, so the ellipsoid mask set there (when on) keeps the liver and gut out. The result is shown as contours on
SA / HLA / VLA and as a 3D wall, and kept as surface models and a segmentation (myocardium and LV
cavity) with the cavity and myocardial volumes. Perfusion polar maps and the gated analysis will
sample the same surfaces.
"""

import os
import sys
import json
import logging
import traceback

import numpy as np
import qt
import ctk
import vtk
import slicer
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleWidget,
                                           ScriptedLoadableModuleTest)
from slicer.util import VTKObservationMixin

_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from MyocardialPerfusionImagingLib import myocardium as wallLib  # noqa: E402
from MyocardialPerfusionImagingLib import session  # noqa: E402
from MyocardialPerfusionImagingLib import parameters as parametersLib  # noqa: E402
from MyocardialPerfusionImagingLib import studies as studiesLib  # noqa: E402
from MyocardialPerfusionImagingLib.geometry import unit, worldToLocalMatrix, sampleTrilinear  # noqa: E402
from MyocardialPerfusionImagingLib.views import (  # noqa: E402
    MYOCARDIUM_LAYOUT_ID, DUAL_MYOCARDIUM_LAYOUT_ID, LANE_SUFFIXES, ORIENTATIONS, myocardiumLayoutXml,
    dualMyocardiumLayoutXml, myocardiumViews, registerLayout, sliceNodeByTag, threeDWidgetByTag, showVolumeInView, setSliceGeometry,
    setViewTitle)
from MyocardialPerfusionImagingLib.widgets import breakable, narrowLabel, styleButton, StudyPairSelector  # noqa: E402

LABEL_SPACING_MM = 1.5      # grid of the myocardium / cavity segmentation
# 3D view, styled as Taranis' segmentation layout: segments as flat wireframes, low opacity (dense meshes)
SEGMENTS = (  # (label, name, color)
    (wallLib.LABEL_MYOCARDIUM, "Myocardium", (0.85, 0.35, 0.25)),
    (wallLib.LABEL_CAVITY, "LV cavity", (0.3, 0.5, 0.9)),
)
# In 3D the LV is one model: epicardium, endocardium and the rim between them in the valve plane
SHELL_SPACING_MM = 1.25
SHELL_COLOR = (0.85, 0.35, 0.25)
SHELL_OPACITY = 0.25
LABEL_DISTANCE_MM = 45.0    # floating wall labels in 3D, outside the epicardium
LABEL_COLOR = (0.85, 0.88, 0.9)
HIDDEN_VIEW_ID = "MPINoView"   # view list of the result nodes of a study not on screen: shown in no view
SURFACES = {                # key: (name, color, shown by default)
    "endo": ("endocardium", (0.95, 0.25, 0.25), True),
    "epi": ("epicardium", (0.25, 0.85, 0.35), True),
    "mid": ("mid-wall", (1.0, 0.85, 0.1), False),
}


def _installToolbar():
    try:
        from MyocardialPerfusionImagingLib import toolbar
        toolbar.install()
    except Exception:
        logging.exception("MPI: could not install the workflow toolbar")


class MPIMyocardium(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MPI 3 · Myocardium"
        self.parent.categories = ["Nuclear Medicine.Myocardial Perfusion"]
        self.parent.hidden = True   # opened from the Myocardial Perfusion Imaging hub and the workflow toolbar
        self.parent.dependencies = []
        self.parent.contributors = ["Burak Demir"]
        self.parent.helpText = (
            "Automatic detection of the left ventricular wall (prototype): mid-wall, endocardium, "
            "epicardium and valve plane of the active study, in the cardiac axes set in Processing. "
            "The counts are taken from the SA volume made there, masked when the Processing mask is on. "
            "The result is kept as surface models and a segmentation with the LV cavity and myocardial volumes.")
        self.parent.acknowledgementText = ""
        iconPath = os.path.join(_MODULE_DIR, "Resources", "Icons", "MyocardialPerfusionImaging.png")
        if os.path.exists(iconPath):
            self.parent.icon = qt.QIcon(iconPath)
        if not slicer.app.commandOptions().noMainWindow:
            slicer.app.connect("startupCompleted()", _installToolbar)


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
def cardiacFrame(studyNode):
    """Cardiac frame saved on the reoriented volumes of a study: center, u (base -> apex),
    w (anterior), s (septum), radius. None if the study was not processed."""
    outputs = session.outputsForStudy(studyNode)
    node = outputs.get("SA") or next(iter(outputs.values()), None)
    if node is None:
        return None
    try:
        center = np.array([float(x) for x in node.GetAttribute("MPI.Center").split()])
        u = unit([float(x) for x in node.GetAttribute("MPI.LongAxis").split()])
        w = unit([float(x) for x in node.GetAttribute("MPI.AnteriorAxis").split()])
        radius = float(node.GetAttribute("MPI.RadiusMm") or 55.0)
    except (AttributeError, ValueError):
        return None
    w = unit(w - np.dot(w, u) * u)
    return {"center": center, "u": u, "w": w, "s": np.cross(u, w), "radius": radius}


def detectionVolume(studyNode):
    """Volume the wall is detected on: the SA volume made in Processing (masked when the mask was
    on), or the study itself when there is none."""
    return session.outputsForStudy(studyNode).get("SA") or studyNode


def orientationAxes(frame):
    """(screen right, screen up, stacking direction) as in MPIProcessingLogic.orientationAxes."""
    u, w, s = frame["u"], frame["w"], frame["s"]
    return {"SA": (-s, w, -u), "HLA": (-s, u, w), "VLA": (u, w, -s)}


def volumeSampler(volumeNode):
    """sample(world points [N, 3]) -> trilinear counts of a volume (0 outside)."""
    array = slicer.util.arrayFromVolume(volumeNode).astype(np.float32)
    rasToIjk = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(rasToIjk)
    worldToIjk = slicer.util.arrayFromVTKMatrix(rasToIjk) @ slicer.util.arrayFromVTKMatrix(worldToLocalMatrix(volumeNode))

    def sample(points):
        ijk = np.asarray(points, dtype=float) @ worldToIjk[:3, :3].T + worldToIjk[:3, 3]
        return sampleTrilinear(array, ijk)
    return sample


def surfacePolyData(result, which, closeBase=False, rings=None, meridians=None):
    """Surface of the wall from the valve plane to the apex, as quads on the ray grid (rings and
    meridians in a wireframe). With closeBase it is closed by a fan in the valve plane. rings /
    meridians thin the grid out for a calmer wireframe (the valve-plane ring and the apex are kept)."""
    gridPoints = wallLib.surfaceGrid(result, which)
    if rings:
        rowIndices = np.unique(np.round(np.linspace(0, gridPoints.shape[0] - 1, int(rings))).astype(int))
        gridPoints = gridPoints[rowIndices]
    if meridians:
        gridPoints = gridPoints[:, ::max(1, gridPoints.shape[1] // int(meridians))]
    rows, angles = gridPoints.shape[:2]
    points = gridPoints.reshape(-1, 3)
    apexIndex = len(points)
    points = np.vstack([points, gridPoints[-1].mean(axis=0)])
    if closeBase:
        points = np.vstack([points, gridPoints[0].mean(axis=0)])
    cellList = wallLib.surfaceCells(rows, angles, closeBase=closeBase,
                                    baseCenterIndex=apexIndex + 1 if closeBase else None,
                                    apexCenterIndex=apexIndex)
    vtkPoints = vtk.vtkPoints()
    vtkPoints.SetNumberOfPoints(len(points))
    for index, p in enumerate(points):
        vtkPoints.SetPoint(index, float(p[0]), float(p[1]), float(p[2]))
    cells = vtk.vtkCellArray()
    for cell in cellList:
        cells.InsertNextCell(len(cell))
        for index in cell:
            cells.InsertCellPoint(int(index))
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(vtkPoints)
    polyData.SetPolys(cells)
    clean = vtk.vtkCleanPolyData()   # the apex row collapses to (nearly) one point
    clean.SetInputData(polyData)
    clean.ConvertPolysToLinesOff()   # drop collapsed triangles instead of keeping them as lines
    clean.ConvertLinesToPointsOff()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(clean.GetOutputPort())
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


def wallShellPolyData(result, frame, spacing=None):
    """The left ventricle as one closed surface in world RAS: epicardium outside, endocardium inside,
    joined by a rim in the valve plane (the zero level of myocardium.wallField on a grid aligned
    with the cardiac frame, so the surface is smooth without a voxel staircase)."""
    from vtk.util import numpy_support
    spacing = SHELL_SPACING_MM if spacing is None else spacing
    reach = wallLib.wallReach(result) + 3.0 * spacing
    n = int(np.ceil(2.0 * reach / spacing)) + 1
    axes = (frame["w"], frame["s"], frame["u"])   # right-handed: w x s = u
    origin = result["grid"].center - reach * (axes[0] + axes[1] + axes[2])
    k, j, i = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    points = (origin[None, :] + spacing * (i.reshape(-1, 1) * axes[0] + j.reshape(-1, 1) * axes[1]
                                           + k.reshape(-1, 1) * axes[2]))
    field = wallLib.wallField(result, points).astype(np.float32)
    image = vtk.vtkImageData()
    image.SetDimensions(n, n, n)
    image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(field, deep=1))
    surface = vtk.vtkFlyingEdges3D()
    surface.SetInputData(image)
    surface.SetValue(0, 0.0)
    surface.ComputeNormalsOff()
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputConnection(surface.GetOutputPort())
    smooth.SetNumberOfIterations(20)
    smooth.SetPassBand(0.05)
    smooth.FeatureEdgeSmoothingOff()
    smooth.BoundarySmoothingOff()
    smooth.NormalizeCoordinatesOn()
    matrix = vtk.vtkMatrix4x4()
    for r in range(3):
        for col in range(3):
            matrix.SetElement(r, col, float(axes[col][r] * spacing))
        matrix.SetElement(r, 3, float(origin[r]))
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    toWorld = vtk.vtkTransformPolyDataFilter()
    toWorld.SetTransform(transform)
    toWorld.SetInputConnection(smooth.GetOutputPort())
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(toWorld.GetOutputPort())
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    return normals.GetOutput()


def labelLeaderPolyData(positions):
    """Thin lines from the epicardium to each floating label."""
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for _, label, wall in positions:
        start = points.InsertNextPoint(*[float(x) for x in wall])
        end = points.InsertNextPoint(*[float(x) for x in label])
        lines.InsertNextCell(2)
        lines.InsertCellPoint(start)
        lines.InsertCellPoint(end)
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(lines)
    return polyData


def addWallLabelNodes(result, study, viewIds, distanceMm=None):
    """Floating labels (anterior, septal, inferior, lateral, apex, base) distanceMm outside the
    epicardium with thin leader lines, shown in the given 3D views only. Returns the two new nodes."""
    scene = slicer.mrmlScene
    distanceMm = LABEL_DISTANCE_MM if distanceMm is None else distanceMm
    positions = wallLib.wallLabelPositions(result, distanceMm)

    labels = scene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"{study.GetName()} wall labels")
    labels.SetMarkupLabelFormat("%N")
    for text, point, _ in positions:
        index = labels.AddControlPoint(float(point[0]), float(point[1]), float(point[2]), text)
        labels.SetNthControlPointLocked(index, True)
    labels.SetLocked(True)
    labels.CreateDefaultDisplayNodes()
    display = labels.GetDisplayNode()
    display.SetColor(*LABEL_COLOR)
    display.SetSelectedColor(*LABEL_COLOR)
    display.SetActiveColor(*LABEL_COLOR)
    display.SetGlyphType(slicer.vtkMRMLMarkupsDisplayNode.Sphere3D)
    display.SetGlyphScale(1.2)
    display.SetTextScale(4.0)
    display.SetPropertiesLabelVisibility(False)
    display.SetPointLabelsVisibility(True)
    display.SetVisibility2D(False)
    try:
        display.SetOccludedVisibility(True)
        display.SetOccludedOpacity(0.6)
        textProperty = display.GetTextProperty()
        textProperty.SetBold(True)
        textProperty.SetShadow(True)
        textProperty.SetBackgroundOpacity(0.0)
    except AttributeError:
        pass
    if viewIds:
        display.SetViewNodeIDs(viewIds)
    _placeNextTo(labels, study)

    leaders = scene.AddNewNodeByClass("vtkMRMLModelNode", f"{study.GetName()} wall label lines")
    leaders.SetAndObservePolyData(labelLeaderPolyData(positions))
    leaders.CreateDefaultDisplayNodes()
    display = leaders.GetDisplayNode()
    display.SetColor(*LABEL_COLOR)
    display.SetOpacity(0.45)
    display.SetLineWidth(1.0)
    display.SetVisibility2D(False)
    if viewIds:
        display.SetViewNodeIDs(viewIds)
    _placeNextTo(leaders, study)
    return [labels, leaders]


def style3DView(viewNode):
    """Black background, no box, axis labels or orientation marker (as in Taranis' segmentation
    layout): the wall stands on its own."""
    viewNode.SetBackgroundColor(0.0, 0.0, 0.0)
    viewNode.SetBackgroundColor2(0.0, 0.0, 0.0)
    viewNode.SetBoxVisible(False)
    viewNode.SetAxisLabelsVisible(False)
    viewNode.SetRenderMode(slicer.vtkMRMLViewNode.Orthographic)   # no perspective distortion of the wall
    try:
        viewNode.SetOrientationMarkerType(slicer.vtkMRMLAbstractViewNode.OrientationMarkerTypeNone)
    except AttributeError:
        pass


def cameraAnchor(frame, result=None):
    """(center, axes) the 3D camera is placed around: the middle of the LV (of the wall when detected) and
    the cardiac frame axes as columns (u, w, s). Linked 3D views copy the camera in these coordinates."""
    center = np.asarray(frame["center"], dtype=float)
    if result is not None:
        center = np.asarray(result["grid"].center, dtype=float)
        apex = float(np.mean(result["epi"][-1]))
        center = center + 0.5 * (apex + wallLib.valvePlaneZ(result)) * frame["u"]
    return center, np.column_stack([frame["u"], frame["w"], frame["s"]])


def defaultParallelScale(frame, result=None):
    """Half the height of the orthographic 3D view, mm: the LV and its floating labels fit in."""
    if result is not None:
        return 0.75 * (wallLib.wallReach(result) + LABEL_DISTANCE_MM) + 10.0
    return 1.6 * float(frame.get("radius", 55.0))


def placeCamera(viewNode, frame, result=None, distance=330.0, parallelScale=None, direction=None, up=None):
    """Look at the LV from superior-anterior: the camera sits in front of the heart (anterior wall
    of the cardiac frame) and above it (patient superior), aimed at the middle of the LV. direction (from
    the LV toward the camera) and up (the screen's up) can be given instead."""
    center, _ = cameraAnchor(frame, result)
    if parallelScale is None:
        parallelScale = defaultParallelScale(frame, result)
    superior = np.array([0.0, 0.0, 1.0])
    direction = unit(unit(frame["w"]) + superior) if direction is None else unit(direction)
    up = superior if up is None else np.asarray(up, dtype=float)
    up = unit(up - np.dot(up, direction) * direction)
    position = center + distance * direction
    camera = _cameraNode(viewNode)
    if camera is not None:
        camera.SetFocalPoint(*center)
        camera.SetPosition(*position)
        camera.SetViewUp(*up)
        if parallelScale:
            camera.SetParallelScale(float(parallelScale))
        camera.ResetClippingRange()
    # and on the renderer's camera itself, in case the view has not picked up its camera node yet
    widget = threeDWidgetByTag(viewNode.GetLayoutName())
    if widget is not None:
        renderer = widget.threeDView().renderWindow().GetRenderers().GetFirstRenderer()
        if renderer is not None:
            vtkCamera = renderer.GetActiveCamera()
            vtkCamera.SetFocalPoint(*center)
            vtkCamera.SetPosition(*position)
            vtkCamera.SetViewUp(*up)
            if parallelScale:
                vtkCamera.SetParallelScale(float(parallelScale))
            renderer.ResetCameraClippingRange()
        widget.threeDView().forceRender()


def _cameraNode(viewNode):
    """Camera node of a 3D view that is in the scene now. After a scene close the view widget can
    still hold its old, removed camera node while a new one drives the view, so the scene's own
    camera of the view is asked first (the API differs between Slicer versions)."""
    scene = slicer.mrmlScene
    candidates = []
    try:
        candidates.append(slicer.modules.cameras.logic().GetViewActiveCameraNode(viewNode))
    except AttributeError:
        pass
    candidates += [camera for camera in slicer.util.getNodesByClass("vtkMRMLCameraNode")
                   if camera.GetLayoutName() == viewNode.GetLayoutName()]
    widget = threeDWidgetByTag(viewNode.GetLayoutName())
    if widget is not None:
        try:
            candidates.append(widget.threeDView().cameraNode())
        except AttributeError:
            pass
    for camera in candidates:
        if camera is not None and scene.IsNodePresent(camera):
            return camera
    return None


def labelGrid(result, frame, spacing=None):
    """Labels (0 / myocardium / cavity) on a grid aligned with the cardiac frame around the LV.
    Returns (labels [k, j, i] uint8, IJK->RAS vtkMatrix4x4)."""
    spacing = LABEL_SPACING_MM if spacing is None else spacing
    reach = wallLib.wallReach(result) + 2.0 * spacing
    n = int(np.ceil(2.0 * reach / spacing)) + 1
    axes = (frame["w"], frame["s"], frame["u"])   # right-handed: w x s = u
    origin = result["grid"].center - reach * (axes[0] + axes[1] + axes[2])
    k, j, i = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    points = (origin[None, :] + spacing * (i.reshape(-1, 1) * axes[0] + j.reshape(-1, 1) * axes[1]
                                           + k.reshape(-1, 1) * axes[2]))
    labels = wallLib.classify(result, points).reshape(n, n, n)
    matrix = vtk.vtkMatrix4x4()
    for r in range(3):
        for col in range(3):
            matrix.SetElement(r, col, float(axes[col][r] * spacing))
        matrix.SetElement(r, 3, float(origin[r]))
    return labels, matrix


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class MPIMyocardiumWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.studyNode = None
        self.frame = None
        self.results = {}     # study node ID -> fit result (valve offset is applied without refitting)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.studySelector = StudyPairSelector(self.layout)
        self.studyLabel = narrowLabel(qt.QLabel())
        self.layout.addWidget(self.studyLabel)

        self.detectButton = styleButton(qt.QPushButton("Detect myocardium"), "create")
        self.detectButton.toolTip = "Find the LV wall automatically in the cardiac axes set in Processing"
        self.layout.addWidget(self.detectButton)

        # Parameters
        parameters = ctk.ctkCollapsibleButton()
        parameters.text = "Detection parameters"
        parameters.collapsed = True
        self.layout.addWidget(parameters)
        form = qt.QFormLayout(parameters)
        defaults = wallLib.DEFAULTS

        self.anglesSpin = qt.QSpinBox()
        self.anglesSpin.setRange(24, 240)
        self.anglesSpin.setSingleStep(12)
        self.anglesSpin.setValue(defaults["angles"])
        self.anglesSpin.toolTip = "Rays around the long axis"
        form.addRow("Rays around the axis:", self.anglesSpin)

        self.fitCombo = qt.QComboBox()
        for text, value in wallLib.WALL_FITS:
            self.fitCombo.addItem(text, value)
        self.fitCombo.setCurrentIndex(max(0, self.fitCombo.findData(defaults["fit"])))
        self.fitCombo.toolTip = (
            "Surface fit through the rays whose wall stands out of the blood pool, which draws the others (a "
            "defect) and smooths them, as in Gated QC: B-spline (default, a bending penalty: a defect continues the "
            "curve of the walls around it), deformable mesh (every ray tied to its neighbours, bridges straighter) "
            "or legacy (limited jump between neighbouring rays). With Shape model through defects (Advanced "
            "parameters, on by default) the fit is of the walls' deviation from an ellipsoid fitted to the whole "
            "LV, so a defect keeps its curve (a prolate apex behind an apical defect). Ellipsoid (previous): the "
            "shape model with Gaussian smoothing, set by Surface smoothing")
        self.fitCombo.connect("currentIndexChanged(int)", lambda _: self.updateFitControls())
        form.addRow("Fit:", self.fitCombo)

        self.fitPullSlider = ctk.ctkSliderWidget()
        self.fitPullSlider.minimum, self.fitPullSlider.maximum = 0.0, 1.0
        self.fitPullSlider.singleStep = 0.05
        self.fitPullSlider.decimals = 2
        self.fitPullSlider.value = defaults["fitPull"]
        self.fitPullSlider.toolTip = ("How far a traced ray is pulled from its own count peak to the fit: 1 the fit, "
                                      "0 its own peak (the rays without a wall always take the fit)")
        form.addRow("Smoothing (pull to the fit):", self.fitPullSlider)

        self.smoothingSlider = ctk.ctkSliderWidget()
        self.smoothingSlider.minimum, self.smoothingSlider.maximum = 0.0, 4.0
        self.smoothingSlider.singleStep = 0.25
        self.smoothingSlider.decimals = 2
        self.smoothingSlider.value = defaults["smoothing"]
        self.smoothingSlider.toolTip = ("Fit: Ellipsoid only. Smoothing of the wall surface (rays), 1.25 by default. Higher values "
                                        "bridge perfusion defects more, lower values follow the counts more closely. "
                                        "The apex is kept consistent around the axis at any value.")
        form.addRow("Surface smoothing:", self.smoothingSlider)

        self.baseMethodCombo = qt.QComboBox()
        for text, value in wallLib.BASE_METHODS:
            self.baseMethodCombo.addItem(text, value)
        self.baseMethodCombo.setCurrentIndex(max(0, self.baseMethodCombo.findData(defaults["baseMethod"])))
        self.baseMethodCombo.toolTip = (
            "Where the wall ends at the base, along each angle. Adaptive, global (default): a preliminary segment "
            "(50 % of each angle's wall counts) gives the blood pool and the mean counts of the whole myocardium, "
            "and every angle's wall ends half way between the two: one level for all angles, steadier in the noisy "
            "gate frames; angles whose basal wall is dim (a scar) count less in the plane fit. Adaptive: the same "
            "with each angle's own basal wall counts instead of the whole myocardium's, so a basal-only defect "
            "(inferobasal) is followed to the valve plane, but the level is noisier in a gate frame. Fixed share: "
            "the wall ends at Base threshold of each angle's wall counts")
        self.baseMethodCombo.connect("currentIndexChanged(int)", lambda _: self.updateFitControls())
        form.addRow("Base detection:", self.baseMethodCombo)

        self.baseSlider = ctk.ctkSliderWidget()
        self.baseSlider.minimum, self.baseSlider.maximum = 20.0, 80.0
        self.baseSlider.singleStep = 5.0
        self.baseSlider.decimals = 0
        self.baseSlider.suffix = " %"
        self.baseSlider.value = 100.0 * defaults["baseFraction"]
        self.baseSlider.toolTip = ("Base detection: Fixed share only. Along each angle the wall ends where its counts fall below this share of that "
                                   "angle's wall counts, or where the cavity fills up (the rays reach the atria); "
                                   "the valve plane is fitted through these ends, with low-uptake (infarcted) "
                                   "angles counting less")
        form.addRow("Base threshold:", self.baseSlider)

        self.resolutionSpin = qt.QDoubleSpinBox()
        self.resolutionSpin.setRange(0.0, 25.0)
        self.resolutionSpin.setSuffix(" mm")
        self.resolutionSpin.setValue(defaults["resolution"])
        self.resolutionSpin.toolTip = ("System resolution (FWHM) at the heart, removed from the width of the count "
                                       "profile to estimate the wall thickness (set to 12 mm for SPECT and 7 mm for "
                                       "PET when a study is chosen)")
        form.addRow("System resolution:", self.resolutionSpin)

        thicknessRow = qt.QHBoxLayout()
        self.minThicknessSpin = qt.QDoubleSpinBox()
        self.minThicknessSpin.setRange(2.0, 20.0)
        self.minThicknessSpin.setSuffix(" mm")
        self.minThicknessSpin.setValue(defaults["minThickness"])
        self.maxThicknessSpin = qt.QDoubleSpinBox()
        self.maxThicknessSpin.setRange(5.0, 30.0)
        self.maxThicknessSpin.setSuffix(" mm")
        self.maxThicknessSpin.setValue(defaults["maxThickness"])
        thicknessRow.addWidget(self.minThicknessSpin)
        thicknessRow.addWidget(qt.QLabel("to"))
        thicknessRow.addWidget(self.maxThicknessSpin)
        form.addRow("Wall thickness:", thicknessRow)

        # Result
        resultSection = ctk.ctkCollapsibleButton()
        resultSection.text = "Result"
        self.layout.addWidget(resultSection)
        resultForm = qt.QFormLayout(resultSection)

        self.valveSlider = ctk.ctkSliderWidget()
        self.valveSlider.minimum, self.valveSlider.maximum = -20.0, 20.0
        self.valveSlider.singleStep = 1.0
        self.valveSlider.decimals = 0
        self.valveSlider.suffix = " mm"
        self.valveSlider.value = 0.0
        self.valveSlider.toolTip = "Move the detected valve plane toward the apex (+) or the atria (-)"
        resultForm.addRow("Valve plane offset:", self.valveSlider)

        self.tiltSliders = {}
        for key, label, tip in (("valveTiltAnterior", "Valve tilt, anterior:",
                                 "Tilt the valve plane: + moves its anterior edge toward the apex, - its inferior edge"),
                                ("valveTiltSeptal", "Valve tilt, septal:",
                                 "Tilt the valve plane: + moves its septal edge toward the apex, - its lateral edge")):
            slider = ctk.ctkSliderWidget()
            slider.minimum, slider.maximum = -30.0, 30.0
            slider.singleStep = 1.0
            slider.decimals = 0
            slider.suffix = " deg"
            slider.value = 0.0
            slider.toolTip = tip + " (added to the detected tilt)"
            resultForm.addRow(label, slider)
            self.tiltSliders[key] = slider

        self.statsLabel = qt.QLabel("-")
        self.statsLabel.wordWrap = True
        self.statsLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        resultForm.addRow(self.statsLabel)

        showRow = qt.QHBoxLayout()
        self.showChecks = {}
        for key, (name, _, shown) in SURFACES.items():
            check = qt.QCheckBox(name.capitalize())
            check.checked = shown
            check.connect("toggled(bool)", lambda on, k=key: self.setSurfaceVisible(k, on))
            self.showChecks[key] = check
            showRow.addWidget(check)
        resultForm.addRow("Show:", showRow)

        layoutBox = qt.QGroupBox("Layout")
        layoutRow = qt.QHBoxLayout(layoutBox)
        self.layoutGroup = qt.QButtonGroup()
        self.layoutGroup.exclusive = True
        self.singleButton = qt.QPushButton("ACTIVE STUDY")
        self.singleButton.toolTip = "SA, HLA, VLA and the 3D wall of the active study"
        self.dualButton = qt.QPushButton("STRESS-REST")
        self.dualButton.toolTip = ("Study 1 (e.g. stress) on the top row, Study 2 (e.g. rest) below (roles from "
                                   "Data roles); Detect and the valve sliders act on the active study")
        for button in (self.singleButton, self.dualButton):
            button.checkable = True
            self.layoutGroup.addButton(button)
            layoutRow.addWidget(button)
        self.dualButton.checked = True
        self.layout.addWidget(layoutBox)

        self.perfusionButton = styleButton(qt.QPushButton("Perfusion ›"), "primary")
        self.perfusionButton.toolTip = "Open the reoriented slices of this study"
        self.layout.addWidget(self.perfusionButton)
        self.parameterPanel = parametersLib.ParameterPanel(
            "myocardium", self.layout, onChanged=None,
            hint=("The wall fit parameters apply the next time the myocardium is detected, the display ones the next time it is shown."))
        self.parameterPanel.bind("angles", self.anglesSpin)
        self.parameterPanel.bind("smoothing", self.smoothingSlider)
        self.parameterPanel.bind("fit", self.fitCombo)
        self.parameterPanel.bind("fitPull", self.fitPullSlider)
        self.updateFitControls()
        self.parameterPanel.bind("baseMethod", self.baseMethodCombo)
        self.parameterPanel.bind("baseThreshold", self.baseSlider)
        self.parameterPanel.bind("minThickness", self.minThicknessSpin)
        self.parameterPanel.bind("maxThickness", self.maxThicknessSpin)
        self.parameterPanel.bind("showEndo", self.showChecks["endo"])
        self.parameterPanel.bind("showEpi", self.showChecks["epi"])
        self.parameterPanel.bind("showMid", self.showChecks["mid"])
        self.layout.addStretch(1)

        self.valveTimer = qt.QTimer()
        self.valveTimer.setSingleShot(True)
        self.valveTimer.setInterval(150)

        self.detectButton.connect("clicked(bool)", self.onDetect)
        self.valveSlider.connect("valueChanged(double)", lambda _: self.valveTimer.start())
        for slider in self.tiltSliders.values():
            slider.connect("valueChanged(double)", lambda _: self.valveTimer.start())
        self.valveTimer.connect("timeout()", self.onValveMoved)
        self.perfusionButton.connect("clicked(bool)", lambda: session.selectStep("perfusion"))
        self.parameterPanel.loadSaved()   # saved settings of this step, as if set by hand
        for button in (self.singleButton, self.dualButton):
            button.connect("clicked(bool)", lambda _: self.onLayoutChosen())

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneClosed)
        session.addListener(self.onSessionChanged)
        _installToolbar()
        self.updateState()

    # --- workflow toolbar (second row) -----------------------------------------------------------------
    def buildToolbar(self, bar):
        self.toolbarBar = bar
        bar.mirror(self.detectButton, "Detect", kind="create")
        self.tbMask = bar.toggle("Mask", "Make the SA / HLA / VLA volumes again masked outside the Processing "
                                         "ellipsoid (liver and gut out), or unmasked, and detect the wall again",
                                 self.onToolbarMask)
        bar.separator()
        bar.label("Show")
        for key, check in self.showChecks.items():
            bar.mirror(check, SURFACES[key][0].capitalize(), f"Show the {SURFACES[key][0]} in the views")
        bar.separator()
        bar.mirror(self.singleButton, "ACTIVE STUDY")
        self.tbDualButton = bar.mirror(self.dualButton, self.dualButton.text)
        bar.separator()
        bar.mirror(self.perfusionButton, "Perfusion ›", kind="primary")

    def syncToolbar(self):
        bar = getattr(self, "toolbarBar", None)
        if bar is None:
            return
        bar.syncMirrors()
        try:
            states = [self.processingWidget().outputsMasked(study) for study, _ in self.lanes() if study is not None]
            states = [state for state in states if state is not None]
            bar.setChecked(self.tbMask, bool(states) and all(states))
            self.tbMask.enabled = bool(states)
        except Exception:
            logging.debug("MPI: could not update the toolbar:\n" + traceback.format_exc())

    @staticmethod
    def processingWidget():
        return slicer.modules.mpiprocessing.widgetRepresentation().self()

    def onToolbarMask(self, masked):
        """Make the volumes of the studies on screen again with / without the mask, then detect their walls."""
        processing = self.processingWidget()
        with slicer.util.tryWithErrorDisplay("Could not make the volumes again.", waitCursor=True):
            for study, _ in self.lanes():
                if study is not None and processing.outputsMasked(study) not in (None, bool(masked)):
                    processing.remakeVolumes(study, masked)
        session.notify()
        if self.isShown():
            self.enter()   # shows the new volumes and detects on them
        self.syncToolbar()

    def enter(self):
        self.studySelector.refresh()
        self.updateState()
        if self.frame is None:
            return
        self.showLayout()
        qt.QTimer.singleShot(200, self.detectMissing)

    def detectMissing(self):
        """Detect the wall of each study on screen that has none yet, or whose SA volume was made
        again in Processing (new axes or mask)."""
        for study, _ in self.lanes():
            segmentation = session.myocardiumForStudy(study)
            if study is None or cardiacFrame(study) is None:
                continue
            if segmentation is None or segmentation.GetAttribute("MPI.DetectedOn") != detectionVolume(study).GetID():
                if study is self.studyNode:
                    self.onDetect()
                else:
                    with slicer.util.tryWithErrorDisplay("Could not detect the myocardium.", waitCursor=True):
                        self.detectStudy(study)
                    self.placeCamera()
                    session.notify()

    def onLayoutChosen(self):
        if self.isShown() and self.frame is not None:
            self.enter()

    def cleanup(self):
        self.valveTimer.stop()
        session.removeListener(self.onSessionChanged)
        self.studySelector.cleanup()
        self.removeObservers()

    def onSceneClosed(self, caller=None, event=None):
        self.results = {}
        self.studyNode = None
        self.frame = None
        self.updateState()

    @staticmethod
    def isShown():
        return slicer.util.selectedModule() == session.HUB_MODULE and session.currentStep() == "myocardium"

    def onSessionChanged(self):
        previous = self.studyNode
        self.updateState()
        if self.isShown() and self.frame is not None and (self.studyNode is not previous
                                                          or self.layoutId() != slicer.app.layoutManager().layout):
            self.enter()

    # --- state ---------------------------------------------------------------
    def updateState(self):
        self.studyNode = session.activeStudy()
        self.frame = cardiacFrame(self.studyNode) if self.studyNode is not None else None
        if self.studyNode is not None and self.studyNode.GetID() != getattr(self, "_resolutionStudyId", None):
            # PET resolves the wall better than SPECT: start from the modality's system resolution
            self._resolutionStudyId = self.studyNode.GetID()
            self.resolutionSpin.value = studiesLib.SYSTEM_RESOLUTION_MM.get(
                studiesLib.modality(self.studyNode), wallLib.DEFAULTS["resolution"])
        if self.studyNode is None:
            self.studyLabel.text = "No active study. Choose one in Data roles or in the toolbar."
        elif self.frame is None:
            self.studyLabel.text = (f"<b>{breakable(self.studyNode.GetName())}</b><br>Not processed yet: create the "
                                    "SA / HLA / VLA volumes in Processing first.")
        else:
            source = detectionVolume(self.studyNode)
            mask = source.GetAttribute("MPI.Mask")
            maskText = "no mask" if not mask or mask == "none" else "masked"
            self.studyLabel.text = (f"<b>{breakable(self.studyNode.GetName())}</b><br>"
                                    f"<span style='color:#7d868d'>Detected on {breakable(source.GetName())} ({maskText})</span>")
        self.detectButton.enabled = self.frame is not None
        result = self.results.get(self.studyNode.GetID()) if self.studyNode is not None else None
        self.valveSlider.enabled = result is not None
        for slider in self.tiltSliders.values():
            slider.enabled = result is not None
        segmentation = session.myocardiumForStudy(self.studyNode)
        self.statsLabel.text = self.statsText(segmentation)
        self.perfusionButton.enabled = len(session.outputsForStudy(self.studyNode)) == len(ORIENTATIONS)
        self.dualButton.enabled = self.dualAvailable()
        titles = session.slotTitles()
        self.dualButton.text = f"{titles['stress']}-{titles['rest']}".upper()
        if getattr(self, "tbDualButton", None) is not None:
            try:
                self.tbDualButton.setText(self.dualButton.text)
            except Exception:
                pass   # the toolbar was rebuilt
        self.syncToolbar()

    @staticmethod
    def statsText(segmentation):
        if segmentation is None:
            return "-"
        get = lambda key: segmentation.GetAttribute(key) or "-"
        return (f"LV cavity: <b>{get('MPI.LVCavityMl')} ml</b><br>"
                f"Myocardium: {get('MPI.MyocardiumMl')} ml ({get('MPI.MyocardiumG')} g)<br>"
                f"Mean wall thickness: {get('MPI.MeanThicknessMm')} mm<br>"
                f"Valve plane: {get('MPI.ValvePlaneMm')} mm from the LV center on the axis, "
                f"tilted {get('MPI.ValveTiltDeg')} deg")

    # --- layout --------------------------------------------------------------
    @staticmethod
    def dualAvailable():
        found = session.stressRestStudies()
        return (found["stress"] is not None and found["rest"] is not None and found["stress"] is not found["rest"]
                and cardiacFrame(found["stress"]) is not None and cardiacFrame(found["rest"]) is not None)

    def dualShown(self):
        return bool(self.dualButton.checked) and self.dualAvailable()

    def layoutId(self):
        return DUAL_MYOCARDIUM_LAYOUT_ID if self.dualShown() else MYOCARDIUM_LAYOUT_ID

    def lanes(self):
        """[(study, view tag suffix)] of the studies on screen: the active one, or Study 1 and Study 2."""
        if self.dualShown():
            found = session.stressRestStudies()
            return [(found["stress"], LANE_SUFFIXES[0]), (found["rest"], LANE_SUFFIXES[1])]
        return [(self.studyNode, LANE_SUFFIXES[0])]

    def laneSuffix(self, study):
        """View tag suffix of a study on screen, None when it is not shown."""
        return next((suffix for node, suffix in self.lanes() if node is not None and node is study), None)

    def showLayout(self):
        dual = self.dualShown()
        layoutId = self.layoutId()
        titles = session.slotTitles()
        registerLayout(layoutId, dualMyocardiumLayoutXml((titles["stress"], titles["rest"])) if dual
                       else myocardiumLayoutXml())
        slicer.app.layoutManager().setLayout(layoutId)
        slicer.app.processEvents()
        for study, suffix in self.lanes():
            frame = cardiacFrame(study) if study is not None else None
            if frame is None:
                continue
            axes = orientationAxes(frame)
            outputs = session.outputsForStudy(study)
            fov = 2.0 * frame["radius"]
            slices, threeDTag = myocardiumViews(suffix)
            role = titles["stress" if suffix == LANE_SUFFIXES[0] else "rest"] if dual else session.studyRoleTitle(study)
            for name, tag in slices.items():
                shown = outputs.get(name) or study
                showVolumeInView(tag, shown)
                right, up, stack = axes[name]
                setSliceGeometry(sliceNodeByTag(tag), right, up, stack, frame["center"], fov)
                setViewTitle(tag, f"{role}: {shown.GetName()}" if role else shown.GetName())
            setViewTitle(threeDTag, f"{role}: {study.GetName()}" if role else study.GetName(), (0.9, 0.9, 0.9))
        self.assignViews()
        try:
            self.placeCamera()
        except Exception:
            logging.warning("MPI: could not set up the 3D view:\n" + traceback.format_exc())

    def viewNodeIds(self, suffix, only3D=False):
        """View node IDs of one row of the layout (slice views and the 3D view, or only the 3D view)."""
        if suffix is None:
            return []
        slices, threeDTag = myocardiumViews(suffix)
        ids = [] if only3D else [sliceNodeByTag(tag).GetID() for tag in slices.values() if sliceNodeByTag(tag)]
        widget = threeDWidgetByTag(threeDTag)
        if widget:
            ids.append(widget.mrmlViewNode().GetID())
        return ids

    def assignViews(self):
        """Each study's result nodes in its own row of views; other studies' nodes in no view (an
        empty view list would show them everywhere)."""
        shown = {study.GetID(): suffix for study, suffix in self.lanes() if study is not None}
        for className in ("vtkMRMLModelNode", "vtkMRMLSegmentationNode", "vtkMRMLMarkupsFiducialNode"):
            for node in slicer.util.getNodesByClass(className):
                studyId = node.GetAttribute("MPI.MyocardiumOf")
                display = node.GetDisplayNode()
                if not studyId or display is None:
                    continue
                ids = self.viewNodeIds(shown.get(studyId), only3D=node.GetAttribute("MPI.Only3D") == "1")
                display.SetViewNodeIDs(ids or [HIDDEN_VIEW_ID])

    def view3D(self, suffix=LANE_SUFFIXES[0]):
        widget = threeDWidgetByTag(myocardiumViews(suffix)[1])
        return widget.mrmlViewNode() if widget else None

    def placeCamera(self):
        for study, suffix in self.lanes():
            viewNode = self.view3D(suffix)
            frame = cardiacFrame(study) if study is not None else None
            if viewNode is None or frame is None:
                continue
            style3DView(viewNode)
            placeCamera(viewNode, frame, self.results.get(study.GetID()))
        # again after the layout and the new models have settled (a view can reset its camera then,
        # e.g. when a new camera node is made for it after a scene close)
        qt.QTimer.singleShot(400, self._placeCameraAgain)
        qt.QTimer.singleShot(1200, self._placeCameraAgain)

    def _placeCameraAgain(self):
        try:
            for study, suffix in self.lanes():
                viewNode = self.view3D(suffix)
                frame = cardiacFrame(study) if study is not None else None
                if viewNode is not None and frame is not None:
                    placeCamera(viewNode, frame, self.results.get(study.GetID()))
        except Exception:
            logging.warning("MPI: could not place the 3D camera:\n" + traceback.format_exc())

    # --- detection -------------------------------------------------------------
    def updateFitControls(self):
        """Surface smoothing belongs to the ellipsoid fit, the pull to the surface fits, the base threshold to the
        fixed-share base detection."""
        ellipsoid = (self.fitCombo.itemData(self.fitCombo.currentIndex) or wallLib.DEFAULTS["fit"]) == "ellipsoid"
        self.smoothingSlider.enabled = ellipsoid
        self.fitPullSlider.enabled = not ellipsoid
        self.baseSlider.enabled = self.baseMethod() == "fraction"

    def baseMethod(self):
        return self.baseMethodCombo.itemData(self.baseMethodCombo.currentIndex) or wallLib.DEFAULTS["baseMethod"]

    def parameters(self):
        return {"angles": int(self.anglesSpin.value), "smoothing": float(self.smoothingSlider.value),
                "fit": self.fitCombo.itemData(self.fitCombo.currentIndex) or wallLib.DEFAULTS["fit"],
                "fitPull": float(self.fitPullSlider.value),
                "baseMethod": self.baseMethod(),
                "baseFraction": float(self.baseSlider.value) / 100.0,
                "resolution": float(self.resolutionSpin.value),
                "minThickness": float(self.minThicknessSpin.value),
                "maxThickness": max(float(self.maxThicknessSpin.value), float(self.minThicknessSpin.value)),
                "valveOffset": float(self.valveSlider.value),
                **{key: float(slider.value) for key, slider in self.tiltSliders.items()}}

    def onDetect(self):
        self.updateState()
        if self.frame is None:
            return
        with slicer.util.tryWithErrorDisplay("Could not detect the myocardium.", waitCursor=True):
            for slider in [self.valveSlider] + list(self.tiltSliders.values()):
                wasBlocked = slider.blockSignals(True)
                slider.value = 0.0
                slider.blockSignals(wasBlocked)
            self.detectStudy(self.studyNode)
        if slicer.app.layoutManager().layout != self.layoutId() and self.isShown():
            self.showLayout()
        else:
            try:
                self.placeCamera()
            except Exception:
                logging.warning("MPI: could not set up the 3D view:\n" + traceback.format_exc())
        self.updateState()
        session.notify()

    def detectStudy(self, study):
        """Fit the wall of a study with the current parameters (another study than the active one:
        its own modality's resolution and no valve adjustment) and store the result."""
        frame = cardiacFrame(study)
        params = self.parameters()
        if study is not self.studyNode:
            params["resolution"] = studiesLib.SYSTEM_RESOLUTION_MM.get(studiesLib.modality(study), params["resolution"])
            params["valveOffset"] = 0.0
            params.update({key: 0.0 for key in self.tiltSliders})
        result = wallLib.fitWall(volumeSampler(detectionVolume(study)), frame["center"], frame["u"], frame["w"],
                                 frame["radius"], params)
        self.results[study.GetID()] = result
        self.storeResult(result, study)
        return result

    def onValveMoved(self):
        result = self.results.get(self.studyNode.GetID()) if self.studyNode is not None else None
        if result is None:
            return
        result["grid"].params["valveOffset"] = float(self.valveSlider.value)
        for key, slider in self.tiltSliders.items():
            result["grid"].params[key] = float(slider.value)
        with slicer.util.tryWithErrorDisplay("Could not move the valve plane.", waitCursor=True):
            self.storeResult(result)
        self.updateState()

    def pruneResults(self):
        """Forget the fits of studies no longer in the scene (removed without a scene close)."""
        scene = slicer.mrmlScene
        self.results = {nodeId: result for nodeId, result in self.results.items() if scene.GetNodeByID(nodeId)}

    def resultForStudy(self, studyNode):
        """Fit result of a study (for Perfusion): the one in memory, else fitted again with the
        parameters saved on its segmentation. None when the myocardium was not detected."""
        if studyNode is None:
            return None
        result = self.results.get(studyNode.GetID())
        if result is not None:
            return result
        self.pruneResults()
        segmentation = session.myocardiumForStudy(studyNode)
        frame = cardiacFrame(studyNode)
        if segmentation is None or frame is None:
            return None
        try:
            params = json.loads(segmentation.GetAttribute("MPI.MyocardiumParams") or "{}")
        except ValueError:
            params = {}
        params.setdefault("fit", "ellipsoid")       # detected before the surface fits: the same wall again
        result = wallLib.fitWall(volumeSampler(detectionVolume(studyNode)), frame["center"], frame["u"],
                                 frame["w"], frame["radius"], params)
        self.results[studyNode.GetID()] = result
        return result

    # --- result nodes ------------------------------------------------------------
    @staticmethod
    def resultNodes(studyNode):
        if studyNode is None:
            return []
        nodes = []
        for className in ("vtkMRMLModelNode", "vtkMRMLSegmentationNode", "vtkMRMLMarkupsFiducialNode"):
            for node in slicer.util.getNodesByClass(className):
                if node.GetAttribute("MPI.MyocardiumOf") == studyNode.GetID():
                    nodes.append(node)
        return nodes

    def storeResult(self, result, study=None):
        """Replace the surface models and the segmentation of a study (the active one by default)
        with a fit result."""
        study = study or self.studyNode
        frame = cardiacFrame(study)
        for node in self.resultNodes(study):
            slicer.mrmlScene.RemoveNode(node)
        suffix = self.laneSuffix(study)
        ids = self.viewNodeIds(suffix) or [HIDDEN_VIEW_ID]
        threeDIds = self.viewNodeIds(suffix, only3D=True) or [HIDDEN_VIEW_ID]
        scene = slicer.mrmlScene

        for key, (name, color, shown) in SURFACES.items():
            # full-resolution surface for the contours on the slices
            model = scene.AddNewNodeByClass("vtkMRMLModelNode", f"{study.GetName()} {name}")
            model.SetAndObservePolyData(surfacePolyData(result, key))
            model.CreateDefaultDisplayNodes()
            display = model.GetDisplayNode()
            display.SetColor(*color)
            display.SetVisibility2D(True)
            display.SetVisibility3D(False)
            display.SetSliceIntersectionThickness(2)
            display.SetVisibility(bool(self.showChecks[key].checked))
            display.SetViewNodeIDs(ids)
            model.SetAttribute("MPI.MyocardiumOf", study.GetID())
            model.SetAttribute("MPI.Surface", key)
            _placeNextTo(model, study)

        self.storeWallLabels(result, study, threeDIds)

        labels, ijkToRas = labelGrid(result, frame)
        reference = scene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "MPI myocardium grid")
        reference.SetIJKToRASMatrix(ijkToRas)
        slicer.util.updateVolumeFromArray(reference, labels)
        segmentation = scene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{study.GetName()} myocardium")
        segmentation.CreateDefaultDisplayNodes()
        segmentation.SetReferenceImageGeometryParameterFromVolumeNode(reference)
        for label, segmentName, color in SEGMENTS:
            segmentId = segmentation.GetSegmentation().AddEmptySegment(segmentName, segmentName, color)
            slicer.util.updateSegmentBinaryLabelmapFromArray((labels == label).astype(np.uint8), segmentation,
                                                             segmentId, reference)
        scene.RemoveNode(reference)
        segmentation.GetDisplayNode().SetVisibility(False)   # slices: the surface contours carry the display
        self.storeWireframes(result, study, frame, threeDIds)
        stats = wallLib.wallStatistics(result, labels, LABEL_SPACING_MM ** 3)
        segmentation.SetAttribute("MPI.MyocardiumOf", study.GetID())
        segmentation.SetAttribute("MPI.DetectedOn", detectionVolume(study).GetID())
        segmentation.SetAttribute("MPI.MyocardiumParams", json.dumps(result["grid"].params))
        segmentation.SetAttribute("MPI.LVCavityMl", f"{stats['cavityMl']:.0f}")
        segmentation.SetAttribute("MPI.LVCavityMlExact", f"{stats['cavityMl']:.3f}")   # for ratios (TID)
        segmentation.SetAttribute("MPI.MyocardiumMl", f"{stats['myocardiumMl']:.0f}")
        segmentation.SetAttribute("MPI.MyocardiumG", f"{stats['massG']:.0f}")
        segmentation.SetAttribute("MPI.MeanThicknessMm", f"{stats['meanThicknessMm']:.1f}")
        segmentation.SetAttribute("MPI.ValvePlaneMm", f"{stats['baseZ']:.0f}")
        segmentation.SetAttribute("MPI.ValveOffsetMm", f"{result['grid'].params['valveOffset']:.0f}")
        segmentation.SetAttribute("MPI.ValveTiltDeg", f"{stats['valveTiltDeg']:.0f}")
        _placeNextTo(segmentation, study)

    @staticmethod
    def storeWallLabels(result, study, viewIds):
        for node in addWallLabelNodes(result, study, viewIds):
            node.SetAttribute("MPI.MyocardiumOf", study.GetID())
            node.SetAttribute("MPI.Only3D", "1")

    def storeWireframes(self, result, study, frame, viewIds):
        """The LV in 3D as one model (epicardium, endocardium and the rim in the valve plane),
        styled as Taranis' segmentation layout: flat wireframe, no backface culling, low opacity."""
        model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{study.GetName()} left ventricle 3D")
        model.SetAndObservePolyData(wallShellPolyData(result, frame))
        model.CreateDefaultDisplayNodes()
        display = model.GetDisplayNode()
        display.SetColor(*SHELL_COLOR)
        display.SetVisibility2D(False)
        display.SetRepresentation(slicer.vtkMRMLDisplayNode.WireframeRepresentation)
        display.SetAmbient(1.0)
        display.SetDiffuse(0.0)
        display.SetSpecular(0.0)
        display.SetBackfaceCulling(False)
        display.SetOpacity(SHELL_OPACITY)
        display.SetScalarVisibility(False)
        display.SetVisibility(bool(self.showChecks["epi"].checked or self.showChecks["endo"].checked))
        display.SetViewNodeIDs(viewIds)
        model.SetAttribute("MPI.MyocardiumOf", study.GetID())
        model.SetAttribute("MPI.Surface", "shell")
        model.SetAttribute("MPI.Only3D", "1")
        _placeNextTo(model, study)

    def setSurfaceVisible(self, key, on):
        shellShown = bool(self.showChecks["epi"].checked or self.showChecks["endo"].checked)
        for node in [n for study, _ in self.lanes() for n in self.resultNodes(study)]:
            surface = node.GetAttribute("MPI.Surface")
            if node.GetDisplayNode() and surface in (key, "shell"):
                node.GetDisplayNode().SetVisibility(bool(on) if surface == key else shellShown)


def _placeNextTo(node, referenceNode):
    try:
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        referenceItem = shNode.GetItemByDataNode(referenceNode)
        item = shNode.GetItemByDataNode(node)
        if referenceItem and item:
            shNode.SetItemParent(item, shNode.GetItemParent(referenceItem))
    except Exception:
        logging.warning("Could not place node in subject hierarchy:\n" + traceback.format_exc())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class MPIMyocardiumTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_PhantomWall()
        self.test_ApicalDefect()
        self.test_TiltedApicalDefect()

    def test_PhantomWall(self):
        """A blurred LV-shaped shell with an oblique base (20 deg) and dim atria behind it: the
        detected mid-wall, valve plane, cavity volume and labels match it."""
        spacing, n = 2.0, 80
        center = np.array([80.0, 80.0, 80.0])
        u = unit([-0.6, 0.6, -0.4])
        w = unit(np.cross(u, [1.0, 0.0, 0.0]))
        k, j, i = np.mgrid[0:n, 0:n, 0:n]
        points = np.stack([i, j, k], axis=-1) * spacing - center
        z = points @ u
        rho = np.linalg.norm(points - z[..., None] * u, axis=-1)
        midRadius, apex, half, base = 25.0, 40.0, 5.0, -30.0
        capDistance = np.sqrt((rho / midRadius) ** 2 + (np.maximum(z, 0) / apex) ** 2)
        wall = np.where(z <= 0, np.abs(rho - midRadius) < half, np.abs(capDistance - 1.0) * midRadius < half)
        slope = np.tan(np.radians(20.0))                  # anterior edge of the base nearer the apex
        plane = base + slope * (points @ w)
        wall &= z >= plane
        array = wall.astype(np.float32) * 100.0
        array[(z < plane - 3.0) & (z > plane - 40.0) & (rho < midRadius + 8.0)] += 25.0   # atria
        # blur ~12 mm FWHM with three box passes per axis (no scipy needed)
        for axis in range(3):
            for _ in range(3):
                array = (np.roll(array, -1, axis) + array + np.roll(array, 1, axis)) / 3.0
        volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "phantom")
        volume.SetSpacing(spacing, spacing, spacing)
        slicer.util.updateVolumeFromArray(volume, array)

        result = wallLib.fitWall(volumeSampler(volume), center, u, w, 55.0, {"resolution": 8.0})
        grid = result["grid"]
        middle = result["mid"][np.searchsorted(grid.z, -20.0):grid.nCyl]
        self.assertAlmostEqual(float(np.median(middle)), midRadius, delta=2.0)
        self.assertAlmostEqual(result["baseZ"] + result["centerShift"], base, delta=6.0)
        self.assertAlmostEqual(result["valvePlane"][1], slope, delta=0.12)
        self.assertAlmostEqual(result["valvePlane"][2], 0.0, delta=0.12)
        frame = {"center": center, "u": u, "w": w, "s": np.cross(u, w), "radius": 55.0}
        labels, _ = labelGrid(result, frame, spacing=2.0)
        stats = wallLib.wallStatistics(result, labels, 8.0)
        truth = (np.pi * (midRadius - half) ** 2 * -base + 2.0 / 3.0 * np.pi * (midRadius - half) ** 2 * (apex - half)) / 1000.0
        self.assertAlmostEqual(stats["cavityMl"], truth, delta=0.25 * truth)
        polyData = surfacePolyData(result, "endo")
        self.assertGreater(polyData.GetNumberOfPolys(), 1000)
        polar = wallLib.polarMap(result, volumeSampler(volume))
        segments = 100.0 * wallLib.segmentValues(polar) / np.nanmax(polar["values"])
        self.assertTrue(np.all(np.isfinite(segments)))
        self.assertGreater(float(segments.min()), 50.0)    # uniform wall: no segment looks like a defect
        self.delayDisplay(f"Phantom wall OK: cavity {stats['cavityMl']:.0f} ml (truth {truth:.0f} ml), "
                          f"valve tilt {stats['valveTiltDeg']:.0f} deg (truth 20)")

    def test_ApicalDefect(self):
        """An LV-shaped shell (truncated ellipsoid, 10 mm wall) with no uptake over the apex (35 deg around
        the long axis): the mid-wall follows the curve of the walls around the defect across the apex
        instead of bridging it (the rays there have no peak of their own)."""
        spacing, n = 2.0, 80
        center = np.array([80.0, 80.0, 80.0])
        u = unit([-0.6, 0.6, -0.4])
        w = unit(np.cross(u, [1.0, 0.0, 0.0]))
        s = np.cross(u, w)
        k, j, i = np.mgrid[0:n, 0:n, 0:n]
        points = np.stack([i, j, k], axis=-1) * spacing - center
        z = points @ u
        rho = np.linalg.norm(points - z[..., None] * u, axis=-1)
        a, c, z0, half = 28.0, 70.0, -30.0, 5.0         # mid-wall ellipsoid, center at the base, apex at z = 40

        def normalDistance(rho, z):
            g = np.sqrt((rho / a) ** 2 + ((z - z0) / c) ** 2)
            gradient = np.sqrt((rho / a ** 2) ** 2 + ((z - z0) / c ** 2) ** 2) / np.maximum(g, 1e-9)
            return (g - 1.0) / np.maximum(gradient, 1e-9)
        wall = (np.abs(normalDistance(rho, z)) < half) & (z >= z0)
        q = np.stack([points @ w / a, points @ s / a, (z - z0) / c], axis=-1)
        apical = q[..., 2] / np.maximum(np.linalg.norm(q, axis=-1), 1e-9) > np.cos(np.radians(35.0))
        array = (wall & ~apical).astype(np.float32) * 100.0
        array[(z < z0 - 3.0) & (z > z0 - 40.0) & (rho < a + 8.0)] += 25.0   # atria
        for axis in range(3):
            for _ in range(3):
                array = (np.roll(array, -1, axis) + array + np.roll(array, 1, axis)) / 3.0

        def sample(world):
            return sampleTrilinear(array, np.asarray(world, dtype=float) / spacing)
        for fit in ("ellipsoid", "mesh", "legacy", "bspline"):      # the default last, for the message
            result = wallLib.fitWall(sample, center, u, w, 55.0, {"resolution": 8.0, "fit": fit})
            apexMid = result["centerShift"] + float(np.median(result["mid"][-1]))
            self.assertAlmostEqual(apexMid, z0 + c, delta=4.0, msg=fit)
            for surface in ("endo", "mid", "epi"):       # the last row's rays coincide on the axis: one point
                self.assertLess(float(np.ptp(result[surface][-1])), 1e-6, msg=(fit, surface))
        bridged = wallLib.fitWall(sample, center, u, w, 55.0, {"resolution": 8.0, "shapeModel": False})
        self.assertLess(bridged["centerShift"] + float(np.median(bridged["mid"][-1])), z0 + c - 8.0)
        self.delayDisplay(f"Apical defect OK: apex mid-wall {apexMid:.1f} mm (truth {z0 + c:.0f}), "
                          f"{bridged['centerShift'] + float(np.median(bridged['mid'][-1])):.1f} mm without the shape model")

    def test_TiltedApicalDefect(self):
        """The LV of test_ApicalDefect with its own axis tilted 15 deg against the fitted one and the
        defect around that axis: the tilted shape model carries the mid-wall across the defect on the
        curve of the LV (the untilted ellipsoid misses it by several mm on one side)."""
        spacing, n = 2.0, 80
        center = np.array([80.0, 80.0, 80.0])
        u = unit([-0.6, 0.6, -0.4])
        w = unit(np.cross(u, [1.0, 0.0, 0.0]))
        tilt = np.radians(15.0)
        axis = np.cos(tilt) * u + np.sin(tilt) * np.cross(u, w)     # the LV's own long axis
        side = np.cross(axis, w)
        k, j, i = np.mgrid[0:n, 0:n, 0:n]
        points = np.stack([i, j, k], axis=-1) * spacing - center
        a, c, z0, half = 28.0, 70.0, -30.0, 5.0

        def normalDistance(points):
            z = points @ axis
            rho = np.linalg.norm(points - z[..., None] * axis, axis=-1)
            g = np.sqrt((rho / a) ** 2 + ((z - z0) / c) ** 2)
            gradient = np.sqrt((rho / a ** 2) ** 2 + ((z - z0) / c ** 2) ** 2) / np.maximum(g, 1e-9)
            return (g - 1.0) / np.maximum(gradient, 1e-9)
        z = points @ axis
        wall = (np.abs(normalDistance(points)) < half) & (z >= z0)
        q = np.stack([points @ w / a, points @ side / a, (z - z0) / c], axis=-1)
        apical = q[..., 2] / np.maximum(np.linalg.norm(q, axis=-1), 1e-9) > np.cos(np.radians(35.0))
        array = (wall & ~apical).astype(np.float32) * 100.0
        array[(z < z0 - 3.0) & (z > z0 - 40.0)
              & (np.linalg.norm(points - z[..., None] * axis, axis=-1) < a + 8.0)] += 25.0   # atria
        for k in range(3):
            for _ in range(3):
                array = (np.roll(array, -1, k) + array + np.roll(array, 1, k)) / 3.0

        def sample(world):
            return sampleTrilinear(array, np.asarray(world, dtype=float) / spacing)
        result = wallLib.fitWall(sample, center, u, w, 55.0, {"resolution": 8.0})
        grid = result["grid"]
        origins, directions = grid.rays()
        steps = np.linspace(0.0, 60.0, 601)
        along = origins[..., None, :] + steps[:, None] * directions[..., None, :] - center
        truth = steps[np.argmax(normalDistance(along) > 0, axis=-1)]   # true mid-wall along every ray
        error = np.abs(result["mid"] - truth)[grid.nCyl:]
        self.assertLess(float(np.percentile(error, 90)), 3.0)
        self.delayDisplay(f"Tilted apical defect OK: apical cap mid-wall error p90 {np.percentile(error, 90):.1f} mm, "
                          f"max {error.max():.1f} mm")

# saved Advanced parameters of every step, this module's constants included (see parameters.py)
parametersLib.applySaved()
