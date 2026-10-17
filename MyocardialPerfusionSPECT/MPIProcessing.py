"""
Myocardial Perfusion SPECT - 2. Processing: reorientation to the cardiac axes

Hidden step module of the Myocardial Perfusion SPECT hub (MyocardialPerfusionSPECT.py):
1. Data roles (MPIDataRoles.py) sets the study roles and the active study, 2. Processing (this file)
reorients it, 3. Myocardium (MPIMyocardium.py) detects the LV wall, 4. Perfusion (MPIReview.py) shows
the slices, 5. Gated QC (MPIGatedQC.py) checks the gate frames and 6. Gated Review (MPIGated.py) the gated study. The steps are opened from the hub and from
the workflow toolbar, which is shown only while the hub or a step is open.

Workflow
    1. Opening Processing loads the active study (set in Data roles or the workflow toolbar) into a
       layout of two fixed parallel-projection MIPs on the left and a 2x3 grid of slices:
           left   anterior MIP (top) and left lateral MIP (bottom) of the whole volume, upright.
                  Click either to place the spherical heart ROI: the point goes to the count-weighted
                  center along the projection line. Drag the yellow point to move it (parallel to
                  the screen) and the orange point to resize it.
           row 1  cropped views around the ROI with one axis line each  -> adjust the axes
                    transaxial       : line sets the long axis direction in the transaxial plane
                    vertical LA      : line sets the long axis elevation (apex up/down)
                    short axis       : line sets the rotation around the long axis (points anterior)
           row 2  live resliced preview: short axis, horizontal long axis, vertical long axis
                  (optionally masked with an ellipsoid aligned to the cardiac axes; drag the
                  cyan points in these views to resize it)
       Drag a line end point to rotate the axis, drag its middle point to move the LV center.
       Stress + rest: when Data roles has a Stress and a Rest study, two such screens are stacked,
       stress above rest. The panel edits one of them at a time (Editing: Stress / Rest, or simply
       drag something in its views), and Create makes the volumes of both.
    2. Press "Create SA / HLA / VLA volumes". Three reoriented volumes are resampled and Perfusion
       opens with a slice panel with three rows, each a grid (default 2 x 12) of consecutive slices:
           SA  apex -> base         (anterior up, septum left)
           HLA inferior -> anterior (apex up, septum left)
           VLA septum -> lateral    (anterior up, apex right)
       The panel is a Qt widget embedded in the view layout (Slicer 5.10 removed lightbox mode).

Display follows the ASNC/SNMMI standard orientation.

Resampling uses Lanczos windowed sinc interpolation by default (cubic B-spline and linear are offered):
on coarse SPECT voxels linear interpolation leaves faceted, blocky slices. See
MyocardialPerfusionSPECTLib/resampling.py for the reasoning.

The helpers shared by the hub and the steps are in the MyocardialPerfusionSPECTLib package
(see its __init__.py).
"""

import os
import sys
import math
import re
import logging
import traceback

import numpy as np
import qt
import ctk
import vtk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)
from slicer.util import VTKObservationMixin

# The helper package lives next to this file; make sure it is importable when the
# folder is used through "Additional module paths" as well as in an extension build
_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

import MyocardialPerfusionSPECTLib  # noqa: E402
MyocardialPerfusionSPECTLib.reloadInDeveloperMode()

from MyocardialPerfusionSPECTLib.geometry import (  # noqa: E402
    SUPERIOR, unit, worldToLocalMatrix, ellipsoidInsideGrid, ijkToRasMatrix, sampleTrilinear)
from MyocardialPerfusionSPECTLib.resampling import (  # noqa: E402
    INTERPOLATION_MODES, INTERPOLATION_LABELS, DEFAULT_INTERPOLATION, reslice, zoomInPlane)
from MyocardialPerfusionSPECTLib.views import (  # noqa: E402
    REORIENT_LAYOUT_ID, DUAL_REORIENT_LAYOUT_ID, EDIT_ORDER, ORIENTATIONS, MIP_KINDS, LANE_SUFFIXES,
    laneViews, reorientLayoutXml, dualReorientLayoutXml,
    registerLayout, notSaved, sliceNodeByTag, threeDWidgetByTag, showVolumeInView, setSliceGeometry)
from MyocardialPerfusionSPECTLib.display import (  # noqa: E402
    COLORMAPS, setVolumeDisplay, mosaicPanel)
from MyocardialPerfusionSPECTLib import studies as studiesLib  # noqa: E402
from MyocardialPerfusionSPECTLib import session  # noqa: E402
from MyocardialPerfusionSPECTLib import parameters as parametersLib  # noqa: E402
from MyocardialPerfusionSPECTLib.widgets import breakable, narrowLabel, styleButton, StudyPairSelector  # noqa: E402

DEFAULT_PIXEL_SIZE_MM = 2.0       # in-plane
DEFAULT_SLICE_THICKNESS_MM = 5.0

# Fixed MIP cameras: direction of projection (camera -> patient) and screen up, in RAS
MIP_CAMERAS = {
    "ANT": (np.array([0.0, -1.0, 0.0]), SUPERIOR),   # anterior view, looking posterior
    "LAT": (np.array([1.0, 0.0, 0.0]), SUPERIOR),    # left lateral view, looking to the right
}
MIP_TITLES = {"ANT": "Anterior MIP", "LAT": "Left lateral MIP"}
SNAP_THRESHOLD = 0.5   # samples below this fraction of the line maximum do not pull the snapped point
CLICK_TOLERANCE_PX = 5
RING_POINTS = 48   # control points of the ROI outline circle in each MIP
DEFAULT_ROI_RADIUS_MM = 55.0      # heart ROI of a new study (also the half size of the output volumes)
DEFAULT_AXIS_AZIMUTH_DEG = 45.0   # starting long axis: from patient left toward anterior ...
DEFAULT_AXIS_ELEVATION_DEG = -25.0   # ... and this far toward inferior
MAX_ELEVATION_DEG = 80.0          # the long axis stays within this of the transaxial plane
MASK_LONG_FRACTION = 1.0          # semi-axes of a new mask ellipsoid in ROI radii: long axis ...
MASK_SHORT_FRACTION = 0.85        # ... anterior-inferior and septal-lateral
NO_STUDY_HINT = "No active study. Choose one in Data roles or in the study box of the toolbar."


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
# Module registration
# ---------------------------------------------------------------------------
class MPIProcessing(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MPI 2 \u00b7 Processing"
        self.parent.categories = ["Nuclear Medicine.Myocardial Perfusion"]
        self.parent.hidden = True   # opened from the Myocardial Perfusion SPECT hub and the workflow toolbar
        self.parent.dependencies = []
        self.parent.contributors = ["Burak Demir"]
        self.parent.helpText = (
            "Myocardial perfusion SPECT processing: reorient the active study "
            "to short axis, horizontal long axis and vertical long axis.<br><br>"
            "The active study loads when the step opens. Click the left ventricle in one of the MIPs on the "
            "left to place the spherical ROI (drag the yellow point to move it and the orange point to "
            "resize it). In the top row drag the end points of each line to rotate "
            "the axis and the middle point to move the center. The bottom row shows the resliced result. "
            "Optionally mask the bottom row and the created volumes with an ellipsoid (drag its cyan points to resize). "
            "With a Stress and a Rest study, <b>Stress + rest</b> stacks both screens (stress on top). "
            "Press <b>Create SA / HLA / VLA volumes</b> to resample; the slices open in Perfusion."
        )
        self.parent.acknowledgementText = ""
        iconPath = os.path.join(_MODULE_DIR, "Resources", "Icons", "MyocardialPerfusionSPECT.png")
        if os.path.exists(iconPath):
            self.parent.icon = qt.QIcon(iconPath)
        if not slicer.app.commandOptions().noMainWindow:
            slicer.app.connect("startupCompleted()", _installToolbar)


def _installToolbar():
    try:
        from MyocardialPerfusionSPECTLib import toolbar
        toolbar.install()
    except Exception:
        logging.exception("MPI: could not install the workflow toolbar")


# ---------------------------------------------------------------------------
# Logic: cardiac frame, resampling
# ---------------------------------------------------------------------------
class MPIProcessingLogic(ScriptedLoadableModuleLogic):
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

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.center = np.zeros(3)
        self.radius = DEFAULT_ROI_RADIUS_MM
        self.resetAxes()
        self.resetMaskEllipsoid()

    # --- axis state --------------------------------------------------------
    def resetAxes(self):
        """Typical LV orientation: apex toward patient left, anterior and slightly inferior."""
        azimuth = math.radians(DEFAULT_AXIS_AZIMUTH_DEG)
        horizontal = unit([-math.cos(azimuth), math.sin(azimuth), 0.0])  # RAS: -x = left, +y = anterior
        elevation = math.radians(DEFAULT_AXIS_ELEVATION_DEG)
        self.longAxis = unit(math.cos(elevation) * horizontal + math.sin(elevation) * SUPERIOR)
        self.roll = 0.0

    def horizontalDirection(self):
        d = self.longAxis - np.dot(self.longAxis, SUPERIOR) * SUPERIOR
        return unit(d) if np.linalg.norm(d) > 1e-6 else unit([-1.0, 1.0, 0.0])

    def elevation(self):
        return math.asin(float(np.clip(np.dot(self.longAxis, SUPERIOR), -1.0, 1.0)))

    def azimuthDegrees(self):
        """Angle of the long axis in the transaxial plane, from patient left toward anterior."""
        d = self.horizontalDirection()
        return math.degrees(math.atan2(d[1], -d[0]))

    def frame(self):
        u = unit(self.longAxis)
        d = self.horizontalDirection()
        w0 = unit(SUPERIOR - np.dot(SUPERIOR, u) * u)  # in the vertical plane through the long axis
        s0 = np.cross(u, w0)
        w = math.cos(self.roll) * w0 + math.sin(self.roll) * s0
        s = np.cross(u, w)
        return {"u": u, "d": d, "w0": w0, "s0": s0, "w": unit(w), "s": unit(s)}

    def editViewGeometry(self, which):
        """Plane and line direction of the three adjustment views (top row).
        Each view's plane does not depend on the angle its own line controls,
        so the image stays still while the user drags that line."""
        f = self.frame()
        if which == "TA":
            return {"right": np.array([-1.0, 0.0, 0.0]), "up": np.array([0.0, 1.0, 0.0]),
                    "normal": SUPERIOR.copy(), "line": f["d"]}
        if which == "VLA":
            return {"right": f["d"], "up": SUPERIOR.copy(),
                    "normal": unit(np.cross(f["d"], SUPERIOR)), "line": f["u"]}
        if which == "SA":
            return {"right": -f["s0"], "up": f["w0"], "normal": f["u"], "line": f["w"]}
        raise ValueError(which)

    def setLineDirection(self, which, direction):
        """Update the axes from a line direction drawn in one of the adjustment views."""
        geometry = self.editViewGeometry(which)
        normal = unit(geometry["normal"])
        v = np.asarray(direction, dtype=float)
        v = v - np.dot(v, normal) * normal
        if np.linalg.norm(v) < 1e-6:
            return False
        v = unit(v)
        if which == "TA":
            elevation = self.elevation()
            self.longAxis = unit(math.cos(elevation) * v + math.sin(elevation) * SUPERIOR)
        elif which == "VLA":
            d = self.horizontalDirection()
            elevation = math.atan2(np.dot(v, SUPERIOR), np.dot(v, d))
            limit = math.radians(MAX_ELEVATION_DEG)
            elevation = float(np.clip(elevation, -limit, limit))
            self.longAxis = unit(math.cos(elevation) * d + math.sin(elevation) * SUPERIOR)
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
        """Semi-axes along (long axis, anterior-inferior, septal-lateral): generous around the LV,
        so the wall and the base stay well inside (the Myocardium step detects on the masked volume)."""
        r = self.radius
        self.maskSemiAxes = np.array([MASK_LONG_FRACTION * r, MASK_SHORT_FRACTION * r, MASK_SHORT_FRACTION * r])

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
        localAxes = [unit(worldToLocal[:3, :3] @ np.asarray(a, dtype=float)) for a in axesWorld]
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

    @staticmethod
    def weightedCenterAlongLine(volumeNode, point, direction, threshold=None):
        """Count-weighted center of the volume on the line through point (world RAS) along direction.
        Only counts above threshold x the line maximum weigh in, so the hottest structure on the line
        (the LV walls) sets the depth, not background or a fainter organ behind it. None if empty.
        threshold defaults to SNAP_THRESHOLD."""
        threshold = SNAP_THRESHOLD if threshold is None else float(threshold)
        direction = unit(direction)
        point = np.asarray(point, dtype=float)
        array = slicer.util.arrayFromVolume(volumeNode)
        rasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(rasToIjk)
        worldToIjk = slicer.util.arrayFromVTKMatrix(rasToIjk) @ slicer.util.arrayFromVTKMatrix(
            worldToLocalMatrix(volumeNode))
        bounds = [0.0] * 6
        volumeNode.GetRASBounds(bounds)
        corners = np.array([[x, y, z] for x in bounds[0:2] for y in bounds[2:4] for z in bounds[4:6]])
        reach = (corners - point) @ direction
        step = 0.5 * min(volumeNode.GetSpacing())
        t = np.arange(reach.min(), reach.max() + step, step)
        world = point[None, :] + t[:, None] * direction[None, :]
        ijk = world @ worldToIjk[:3, :3].T + worldToIjk[:3, 3]
        values = sampleTrilinear(array, ijk)
        peak = float(values.max()) if values.size else 0.0
        if peak <= 0.0:
            return None
        weights = np.maximum(values - threshold * peak, 0.0)
        return point + float(np.sum(weights * t) / np.sum(weights)) * direction

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

    def resampleOrientation(self, inputNode, axes, spacing, name, sliceThickness=None, mask=None,
                            interpolation=DEFAULT_INTERPOLATION):
        """Resample the input onto a block around the LV center whose i/j/k axes are
        screen right / screen up / stacking direction of the requested orientation.

        spacing         in-plane pixel size (mm)
        sliceThickness  slice thickness = slice spacing (mm); default: same as spacing
        mask            optional (axesWorld, semiAxes) ellipsoid; voxels outside are zeroed
        interpolation   "sinc" (default), "bspline" or "linear", see resampling.py

        Each output slice is the mean of thin sections (at most 'spacing' apart) covering its whole
        thickness, like summing thin slices on a clinical workstation: thick slices average the counts
        in the slab instead of sampling one plane and skipping the rest. The mask is applied to the thin
        sections before averaging, so its edge is partial-volumed rather than stair-stepped."""
        iDir, jDir, kDir = (unit(a) for a in axes)
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
        sectionIjkToRas = ijkToRasMatrix(iDir * spacing, jDir * spacing, kDir * sectionSpacing, sectionOrigin)

        # thin sections: output IJK -> world RAS -> input RAS -> input IJK
        rasToIjk = vtk.vtkMatrix4x4()
        inputNode.GetRASToIJKMatrix(rasToIjk)
        sectionToLocal = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Multiply4x4(worldToLocalMatrix(inputNode), sectionIjkToRas, sectionToLocal)
        resliceAxes = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Multiply4x4(rasToIjk, sectionToLocal, resliceAxes)

        sections = reslice(inputNode.GetImageData(), resliceAxes, (nPlane, nPlane, nSections), interpolation)

        if mask is not None:
            maskAxes, semiAxes = mask
            sections *= ellipsoidInsideGrid(sections.shape, slicer.util.arrayFromVTKMatrix(sectionIjkToRas),
                                            center, maskAxes, semiAxes)

        slices = sections.reshape(nSlices, sectionsPerSlice, nPlane, nPlane).mean(axis=1).astype(np.float32)

        # slice n is centered on the middle of its thin sections
        sliceOrigin = center - halfPlane * (iDir + jDir) - (nSlices - 1) / 2.0 * thickness * kDir
        ijkToRas = ijkToRasMatrix(iDir * spacing, jDir * spacing, kDir * thickness, sliceOrigin)

        outputNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", slicer.mrmlScene.GenerateUniqueName(name))
        outputNode.SetIJKToRASMatrix(ijkToRas)
        slicer.util.updateVolumeFromArray(outputNode, slices)
        outputNode.CreateDefaultDisplayNodes()
        geometry = {"i": iDir, "j": jDir, "k": kDir, "center": center.copy(),
                    "n": nSlices, "nPlane": nPlane, "spacing": spacing, "thickness": thickness,
                    "sectionsPerSlice": sectionsPerSlice, "interpolation": interpolation}
        return outputNode, geometry

    def createOrientations(self, inputNode, spacing, thickness, interpolation, masked):
        """SA / HLA / VLA of a study in the current frame, masked with the current ellipsoid when asked, with
        the frame saved on them (MPI.* attributes, also read by the later steps). Returns ({orientation: volume},
        {orientation: geometry}, hottest voxel: of the whole volumes when masked, else of the ROI)."""
        axes = self.orientationAxes()
        baseName = inputNode.GetName()
        maskAxes = self.maskAxes()
        maskSemiAxes = np.array(self.maskSemiAxes, dtype=float)
        semiAxesText = ", ".join(f"{a:.0f}" for a in maskSemiAxes)
        maskText = f"ellipsoid semi-axes (long, anterior-inferior, septal-lateral) {semiAxesText} mm" if masked else "none"
        suffix = " masked" if masked else ""
        nodes, geometries = {}, {}
        for name in ORIENTATIONS:
            node, geometry = self.resampleOrientation(
                inputNode, axes[name], spacing, f"{baseName} {name}{suffix}",
                sliceThickness=thickness, mask=(maskAxes, maskSemiAxes) if masked else None,
                interpolation=interpolation)
            for attribute in inputNode.GetAttributeNames() or ():
                node.SetAttribute(attribute, inputNode.GetAttribute(attribute))
            node.SetAttribute("MPI.Orientation", name)
            node.SetAttribute("MPI.SourceVolume", inputNode.GetID())
            for attribute, value in self.frameAttributes().items():
                node.SetAttribute(attribute, value)
            node.SetAttribute("MPI.Mask", maskText)
            node.SetAttribute("MPI.PixelSizeMm", f"{spacing:.2f}")
            node.SetAttribute("MPI.SliceThicknessMm", f"{thickness:.2f}")
            node.SetAttribute("MPI.Interpolation", interpolation)
            placeNextTo(node, inputNode)
            nodes[name], geometries[name] = node, geometry
        # Scale to the hottest voxel of the reoriented data itself (thick slices average counts, so
        # their peak is lower than the input's): the whole volume when masked, the ROI otherwise.
        if masked:
            peak = max(float(slicer.util.arrayFromVolume(n).max()) for n in nodes.values())
        else:
            peak = max(self.maxInSphere(n, self.center, self.radius) for n in nodes.values())
        if peak > 0:
            for node in nodes.values():
                node.SetAttribute("MPI.DisplayMax", f"{peak:.6g}")
        return nodes, geometries, peak

    def frameAttributes(self):
        """MPI.* attributes of the reoriented volumes that keep this frame and the mask ellipsoid (fromOutputs)."""
        f = self.frame()
        return {"MPI.Center": " ".join(f"{x:.2f}" for x in self.center),
                "MPI.LongAxis": " ".join(f"{x:.4f}" for x in f["u"]),
                "MPI.AnteriorAxis": " ".join(f"{x:.4f}" for x in f["w"]),
                "MPI.RollDeg": f"{math.degrees(self.roll):.2f}",
                "MPI.RadiusMm": f"{self.radius:.1f}",
                "MPI.MaskSemiAxesMm": " ".join(f"{a:.2f}" for a in self.maskSemiAxes)}   # also when not masked

    @classmethod
    def fromOutputs(cls, outputNode):
        """Logic in the frame a study's reoriented volume was made in (its MPI.* attributes), with the mask
        ellipsoid it was made with (a default one for volumes made before the semi-axes were kept)."""
        logic = cls()
        logic.center = np.array([float(x) for x in outputNode.GetAttribute("MPI.Center").split()])
        logic.radius = float(outputNode.GetAttribute("MPI.RadiusMm") or DEFAULT_ROI_RADIUS_MM)
        logic.longAxis = unit([float(x) for x in outputNode.GetAttribute("MPI.LongAxis").split()])
        logic.roll = 0.0
        f = logic.frame()
        w = np.array([float(x) for x in outputNode.GetAttribute("MPI.AnteriorAxis").split()])
        logic.roll = math.atan2(float(np.dot(w, f["s0"])), float(np.dot(w, f["w0"])))
        semiAxes = outputNode.GetAttribute("MPI.MaskSemiAxesMm")
        found = re.findall(r"[-+]?\d*\.?\d+", outputNode.GetAttribute("MPI.Mask") or "") if not semiAxes else []
        if semiAxes:
            logic.maskSemiAxes = np.array([float(x) for x in semiAxes.split()])
        elif len(found) == 3:
            logic.maskSemiAxes = np.array([float(x) for x in found])
        else:
            logic.resetMaskEllipsoid()
        return logic


# ---------------------------------------------------------------------------
# One study on the processing screen
# ---------------------------------------------------------------------------
def studyTitle(node):
    role, guessed = studiesLib.studyRole(node)
    return f"{role}{'?' if guessed else ''} · {node.GetName()}"


def keepPointsMovableInMip(display):
    """Drag points parallel to the screen instead of onto the rendered volume, and keep them
    drawn even where the volume rendering reports a surface in front of them."""
    try:
        display.SetSnapMode(slicer.vtkMRMLMarkupsDisplayNode.SnapModeUnconstrained)
    except AttributeError:
        pass
    try:
        display.SetOccludedVisibility(True)
        display.SetOccludedOpacity(1.0)
    except AttributeError:
        pass


class ProcessingLane(VTKObservationMixin):
    """One study on the processing screen: its two MIPs, axis lines, preview row, mask and outputs.
    The single-study screen uses one lane; the stress-rest screen stacks two (stress above rest).
    Panel controls act on the lane the widget calls current; the lane reports back through
    widget.refreshPanel() and widget.setCurrentLane()."""

    def __init__(self, widget, index):
        VTKObservationMixin.__init__(self)
        self.widget = widget
        self.index = index
        self.views = laneViews(LANE_SUFFIXES[index])
        self.mipKind = {tag: kind for kind, tag in self.views["mip"].items()}
        self.mipTags = tuple(self.views["mip"][kind] for kind in MIP_KINDS)
        # Masked preview is recomputed at most every 100 ms while dragging
        self.maskTimer = qt.QTimer()
        self.maskTimer.setSingleShot(True)
        self.maskTimer.setInterval(100)
        self.maskTimer.connect("timeout()", self.updateMaskedPreview)
        self.helperObservations = []
        self.viewObservations = []   # MIP cameras and interactors: (object, event, callback)
        self.annotationActors = []   # (renderer, study caption actor)
        self._lockingCamera = False
        self._updating = False
        self.clearState()

    def clearState(self):
        self.logic = MPIProcessingLogic()
        self.inputNode = None
        self.centerNode = None
        self.mipRadiusNode = None    # resize handle above the center (MIP views)
        self.previewNode = None      # where a click would place the ROI, while placing
        self.mipRadiusPosition = None
        self.mipRingNodes = {}       # MIP view tag -> ROI outline circle facing that camera
        self.maskedNode = None       # masked copy of the input for the bottom row
        self.maskHandleNode = None   # three handles on the mask ellipsoid (bottom-row views)
        self.maskModelNode = None    # mask ellipsoid, drawn as outlines in the bottom row
        self.maskHandlePositions = None
        self.draggedMaskHandle = None
        self.curves = {}             # "TA" / "VLA" / "SA" -> markups curve node
        self.curveKey = {}           # curve node ID -> "TA" / "VLA" / "SA"
        self.curvePositions = {}     # last positions we wrote, to find which point the user dragged
        self.draggedRadiusHandle = None
        self.halfLength = {}
        self.roiPlaced = False
        self.viewCenter = np.zeros(3)
        self.windowMax = None
        self.outputWindowMax = None   # display max of the created volumes (masked: myocardial max)
        self.mipDisplayNode = None
        self.mipPress = None         # (view tag, display position) of a left press in a MIP view
        self.placingRoi = False
        self.outputNodes = {}
        self.outputGeometry = {}
        self.hint = NO_STUDY_HINT
        self.angleText = "-"

    def title(self):
        return studyTitle(self.inputNode) if self.inputNode is not None else ""

    # --- life cycle ------------------------------------------------------------
    def start(self, node):
        """Load a study and wait for the heart ROI (the widget shows the layout afterwards)."""
        self.stop()
        self.inputNode = node
        self.logic.radius = self.widget.radiusSlider.value
        bounds = [0.0] * 6
        node.GetRASBounds(bounds)
        self.logic.center = np.array([(bounds[0] + bounds[1]) / 2.0,
                                      (bounds[2] + bounds[3]) / 2.0,
                                      (bounds[4] + bounds[5]) / 2.0])
        self.viewCenter = self.logic.center.copy()
        self.halfLength = {"TA": 0.75 * self.logic.radius, "VLA": 0.75 * self.logic.radius,
                           "SA": 0.6 * self.logic.radius}
        self.logic.resetMaskEllipsoid()
        setVolumeDisplay(node, self.widget.colormapCombo.currentText, 0.0, node.GetImageData().GetScalarRange()[1])
        self.createHelperNodes()

    def stop(self):
        """Remove this study from the screen (helper nodes, MIP rendering, captions)."""
        self.removeHelperNodes()
        self.removeViewObservers()
        self.clearViewAnnotations()
        if self.mipDisplayNode and slicer.mrmlScene.IsNodePresent(self.mipDisplayNode):
            self.mipDisplayNode.SetVisibility(False)
        self.clearState()

    def sceneClosed(self):
        self.maskTimer.stop()
        for node, eventId, callback in self.helperObservations:
            self.removeObserver(node, eventId, callback)
        self.helperObservations = []
        self.removeViewObservers()
        self.clearViewAnnotations()
        self.clearState()

    def cleanup(self):
        self.maskTimer.stop()
        self.removeViewObservers()
        self.clearViewAnnotations()
        self.removeObservers()

    def attachViews(self):
        """Show this study in its views (the layout must be set already)."""
        for tag in self.views["edit"].values():
            showVolumeInView(tag, self.inputNode)
        self.updateMaskedPreview()  # sets the bottom-row background (masked or not)
        self.assignHelperViews()
        self.setupMip()
        self.updateViewAnnotations()

    def refreshViews(self):
        if not self.inputNode:
            return
        layoutManager = slicer.app.layoutManager()
        tags = () if self.roiPlaced else tuple(self.views["edit"].values()) + tuple(self.views["preview"].values())
        for tag in tags:
            widget = layoutManager.sliceWidget(tag)
            if widget:
                widget.sliceLogic().FitSliceToAll()
        if self.roiPlaced:
            self.viewCenter = self.logic.center.copy()
            self.updateGeometry()
        self.resetMipCamera()

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
        keepPointsMovableInMip(display)
        self._observe(self.centerNode, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.onCenterModified)
        self._observe(self.centerNode, slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, self.onCenterDefined)
        self._observe(self.centerNode, slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent, self.onInteractionEnded)

        # Placement preview: follows the mouse over the MIPs while placing, already snapped in depth
        self.previewNode = scene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode",
                                                   scene.GenerateUniqueName("MPI heart ROI preview"))
        self.previewNode.AddControlPoint(vtk.vtkVector3d(0.0, 0.0, 0.0), "")
        self.previewNode.SetLocked(True)   # never picked, so it cannot catch the placing click
        self.previewNode.CreateDefaultDisplayNodes()
        previewDisplay = self.previewNode.GetDisplayNode()
        previewDisplay.SetColor(*yellow)
        previewDisplay.SetSelectedColor(*yellow)
        previewDisplay.SetOpacity(0.7)
        previewDisplay.SetPointLabelsVisibility(False)
        keepPointsMovableInMip(previewDisplay)
        previewDisplay.SetVisibility(False)

        # Resize handle of the MIP views: above the center, so it is beside it in both projections
        self.mipRadiusNode = scene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode",
                                                     scene.GenerateUniqueName("MPI heart ROI radius (MIP)"))
        self.mipRadiusNode.AddControlPoint(vtk.vtkVector3d(0.0, 0.0, 0.0), "R")
        try:
            self.mipRadiusNode.SetFixedNumberOfControlPoints(True)
        except AttributeError:
            pass
        self.mipRadiusNode.CreateDefaultDisplayNodes()
        mipRadiusDisplay = self.mipRadiusNode.GetDisplayNode()
        mipRadiusDisplay.SetSelectedColor(*orange)
        mipRadiusDisplay.SetColor(*orange)
        mipRadiusDisplay.SetPointLabelsVisibility(False)
        mipRadiusDisplay.SetVisibility(False)
        keepPointsMovableInMip(mipRadiusDisplay)
        self._observe(self.mipRadiusNode, slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self.onMipRadiusHandleModified)
        self._observe(self.mipRadiusNode, slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent, self.onInteractionEnded)

        # ROI outline in each MIP: a circle facing that view's camera. A locked markups curve rather
        # than a model, because a model inside the fully opaque MIP volume would be hidden by it.
        self.mipRingNodes = {}
        for tag in self.mipTags:
            ring = scene.AddNewNodeByClass("vtkMRMLMarkupsClosedCurveNode",
                                           scene.GenerateUniqueName(f"MPI heart ROI {MIP_TITLES[self.mipKind[tag]]}"))
            ring.SetCurveTypeToLinear()
            for _ in range(RING_POINTS):
                ring.AddControlPoint(vtk.vtkVector3d(0.0, 0.0, 0.0))
            ring.SetLocked(True)
            try:
                ring.SetFixedNumberOfControlPoints(True)
                measurement = ring.GetMeasurement("length")
                if measurement:
                    measurement.SetEnabled(False)
            except AttributeError:
                pass
            ring.CreateDefaultDisplayNodes()
            ringDisplay = ring.GetDisplayNode()
            ringDisplay.SetColor(*orange)
            ringDisplay.SetSelectedColor(*orange)
            ringDisplay.SetPropertiesLabelVisibility(False)
            ringDisplay.SetPointLabelsVisibility(False)
            ringDisplay.SetUseGlyphScale(False)
            ringDisplay.SetGlyphSize(0.01)        # line only, no point glyphs
            ringDisplay.SetCurveLineSizeMode(ringDisplay.UseLineDiameter)
            ringDisplay.SetLineDiameter(1.0)
            keepPointsMovableInMip(ringDisplay)
            ringDisplay.SetVisibility(False)
            self.mipRingNodes[tag] = ring

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

        # Interaction helpers, made again for every study shown: not saved with the scene (a saved copy
        # would come back as a stray node next to the new ones)
        for node in ([self.centerNode, self.previewNode, self.mipRadiusNode, self.maskModelNode, self.maskHandleNode]
                     + list(self.mipRingNodes.values()) + list(self.curves.values())):
            notSaved(node)

    def assignHelperViews(self):
        """Restrict helper nodes to their views (view nodes exist only once the layout is shown)."""
        mipIds = {tag: threeDWidgetByTag(tag).mrmlViewNode().GetID() for tag in self.mipTags if threeDWidgetByTag(tag)}
        for node in (self.centerNode, self.mipRadiusNode, self.previewNode):
            if node and node.GetDisplayNode() and mipIds:
                node.GetDisplayNode().SetViewNodeIDs(list(mipIds.values()))
        for tag, ring in self.mipRingNodes.items():
            if tag in mipIds and ring.GetDisplayNode():
                ring.GetDisplayNode().SetViewNodeIDs([mipIds[tag]])
        bottom = [sliceNodeByTag(t).GetID() for t in self.views["preview"].values() if sliceNodeByTag(t)]
        for node in (self.maskHandleNode, self.maskModelNode):
            if node and node.GetDisplayNode() and bottom:
                node.GetDisplayNode().SetViewNodeIDs(bottom)
        for which, curve in self.curves.items():
            sliceNode = sliceNodeByTag(self.views["edit"][which])
            if sliceNode and curve.GetDisplayNode():
                curve.GetDisplayNode().SetViewNodeIDs([sliceNode.GetID()])

    def removeHelperNodes(self):
        self.maskTimer.stop()
        for node, eventId, callback in self.helperObservations:
            self.removeObserver(node, eventId, callback)
        self.helperObservations = []
        for node in ([self.centerNode, self.mipRadiusNode, self.previewNode, self.maskedNode,
                      self.maskHandleNode, self.maskModelNode] + list(self.curves.values())
                     + list(self.mipRingNodes.values())):
            if node and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        self.centerNode = None
        self.mipRadiusNode = None
        self.mipRadiusPosition = None
        self.previewNode = None
        self.mipRingNodes = {}
        self.maskedNode = None
        self.maskHandleNode = None
        self.maskModelNode = None
        self.curves, self.curveKey, self.curvePositions = {}, {}, {}
        self.setPlacingRoi(False)
        self.maskHandlePositions = None

    def setupMip(self):
        """Anterior and left lateral MIPs of the input with fixed, upright cameras."""
        if not self.inputNode:
            return
        viewIds = []
        for tag in self.mipTags:
            widget = threeDWidgetByTag(tag)
            if not widget:
                continue
            viewNode = widget.mrmlViewNode()
            viewNode.SetBackgroundColor(1.0, 1.0, 1.0)
            viewNode.SetBackgroundColor2(1.0, 1.0, 1.0)
            viewNode.SetBoxVisible(False)
            viewNode.SetAxisLabelsVisible(False)
            viewNode.SetRenderMode(slicer.vtkMRMLViewNode.Orthographic)  # parallel projection like a planar MIP
            viewNode.SetRaycastTechnique(slicer.vtkMRMLViewNode.MaximumIntensityProjection)
            try:
                viewNode.SetAnimationMode(slicer.vtkMRMLViewNode.Off)   # no spinning (older versions rotated)
            except AttributeError:
                pass
            viewIds.append(viewNode.GetID())
        if not viewIds:
            return

        vrLogic = slicer.modules.volumerendering.logic()
        displayNode = vrLogic.GetFirstVolumeRenderingDisplayNode(self.inputNode)
        if not displayNode:
            displayNode = vrLogic.CreateDefaultVolumeRenderingNodes(self.inputNode)
        displayNode.SetViewNodeIDs(viewIds)
        displayNode.SetVisibility(True)
        self.mipDisplayNode = displayNode
        self.updateMipTransfer()
        self.installViewObservers()

    def installViewObservers(self):
        """Lock the MIP cameras to their direction (pan and zoom stay free) and take ROI clicks."""
        self.removeViewObservers()
        for tag in self.mipTags:
            widget = threeDWidgetByTag(tag)
            if not widget:
                continue
            cameraNode = slicer.modules.cameras.logic().GetViewActiveCameraNode(widget.mrmlViewNode())
            interactor = widget.threeDView().interactor()
            observations = []
            if cameraNode and cameraNode.GetCamera():
                observations.append((cameraNode.GetCamera(), vtk.vtkCommand.ModifiedEvent,
                                     lambda caller, event, t=tag: self.onMipCameraModified(t, caller)))
            if interactor:
                # Higher priority than the view's own interactor style, so the press is seen first
                observations.append((interactor, vtk.vtkCommand.LeftButtonPressEvent,
                                     lambda caller, event, t=tag: self.onMipPress(t, caller)))
                observations.append((interactor, vtk.vtkCommand.LeftButtonReleaseEvent,
                                     lambda caller, event, t=tag: self.onMipRelease(t, caller)))
                observations.append((interactor, vtk.vtkCommand.MouseMoveEvent,
                                     lambda caller, event, t=tag: self.onMipMouseMove(t, caller)))
                observations.append((interactor, vtk.vtkCommand.LeaveEvent,
                                     lambda caller, event: self.hidePlacementPreview()))
            for obj, eventId, callback in observations:
                self.addObserver(obj, eventId, callback, priority=10.0)
                self.viewObservations.append((obj, eventId, callback))

    def removeViewObservers(self):
        for obj, eventId, callback in self.viewObservations:
            self.removeObserver(obj, eventId, callback)
        self.viewObservations = []

    def onMipCameraModified(self, tag, camera):
        """Undo any rotation: keep the focal point, distance and zoom, restore direction and up."""
        if self._lockingCamera:
            return
        direction, up = MIP_CAMERAS[self.mipKind[tag]]
        if (np.dot(unit(camera.GetDirectionOfProjection()), direction) > 0.99999
                and np.dot(unit(camera.GetViewUp()), up) > 0.99999):
            return
        self._lockingCamera = True
        try:
            focal = np.array(camera.GetFocalPoint())
            position = focal - direction * camera.GetDistance()
            camera.SetPosition(*[float(x) for x in position])
            camera.SetViewUp(*[float(x) for x in up])
            camera.OrthogonalizeViewUp()
        finally:
            self._lockingCamera = False

    def resetMipCamera(self):
        """Both MIPs upright, whole volume in view."""
        if not self.inputNode:
            return
        bounds = [0.0] * 6
        self.inputNode.GetRASBounds(bounds)
        center = np.array([(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, (bounds[4] + bounds[5]) / 2.0])
        extent = np.array([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
        distance = 2.0 * float(extent.max()) + 500.0
        for tag in self.mipTags:
            widget = threeDWidgetByTag(tag)
            if not widget:
                continue
            try:
                cameraNode = slicer.modules.cameras.logic().GetViewActiveCameraNode(widget.mrmlViewNode())
                if not cameraNode:
                    continue
                direction, up = MIP_CAMERAS[self.mipKind[tag]]
                horizontal = float(np.linalg.norm(np.cross(direction, up) * extent))   # screen-width extent
                rect = widget.threeDView().contentsRect()
                aspect = rect.width() / float(rect.height()) if rect.height() > 0 else 1.0
                position = center - direction * distance
                self._lockingCamera = True
                try:
                    cameraNode.SetFocalPoint(*[float(x) for x in center])
                    cameraNode.SetPosition(*[float(x) for x in position])
                    cameraNode.SetViewUp(*[float(x) for x in up])
                    cameraNode.SetParallelScale(0.5 * max(extent[2], horizontal / aspect) * 1.05)
                    cameraNode.ResetClippingRange()
                finally:
                    self._lockingCamera = False
            except Exception:
                logging.warning("Could not reset the MIP camera:\n" + traceback.format_exc())

    def updateMipTransfer(self):
        if not (self.mipDisplayNode and self.inputNode and self.inputNode.GetImageData()):
            return
        maxValue = self.inputNode.GetImageData().GetScalarRange()[1]
        upper = max(maxValue * self.widget.mipLevelSlider.value / 100.0, 1e-6)
        low = min(0.0, self.inputNode.GetImageData().GetScalarRange()[0])
        opacity = vtk.vtkPiecewiseFunction()   # MIP: every voxel fully opaque, the color map does the work
        opacity.AddPoint(low, 1.0)
        opacity.AddPoint(max(maxValue, upper), 1.0)
        color = vtk.vtkColorTransferFunction()  # inverted grey: black activity on white, NM convention
        color.AddRGBPoint(0.0, 1.0, 1.0, 1.0)
        color.AddRGBPoint(upper, 0.0, 0.0, 0.0)
        propertyNode = self.mipDisplayNode.GetVolumePropertyNode()
        propertyNode.SetScalarOpacity(opacity)
        propertyNode.SetColor(color)
        propertyNode.GetVolumeProperty().ShadeOff()

    def onMipPress(self, tag, interactor):
        self.mipPress = (tag, np.array(interactor.GetEventPosition(), dtype=float))

    def onMipRelease(self, tag, interactor):
        press, self.mipPress = self.mipPress, None
        if not (press and press[0] == tag and self.placingRoi and self.inputNode and self.centerNode):
            return
        position = np.array(interactor.GetEventPosition(), dtype=float)
        if np.linalg.norm(position - press[1]) > CLICK_TOLERANCE_PX:
            return   # a pan or zoom, not a click
        point = self.mipDisplayToWorld(tag, position)
        if point is not None:
            self.placeCenterFromMip(tag, point)

    @staticmethod
    def mipDisplayToWorld(tag, position):
        widget = threeDWidgetByTag(tag)
        renderer = widget.threeDView().renderWindow().GetRenderers().GetFirstRenderer() if widget else None
        if renderer is None:
            return None
        renderer.SetDisplayPoint(float(position[0]), float(position[1]), 0.5)
        renderer.DisplayToWorld()
        world = renderer.GetWorldPoint()
        return np.array(world[:3]) / world[3] if world[3] else np.array(world[:3])

    def snapAlongMip(self, tag, point):
        """Depth of a point picked in a MIP: the count-weighted center on its projection line."""
        direction = MIP_CAMERAS[self.mipKind[tag]][0]
        try:
            snapped = self.logic.weightedCenterAlongLine(self.inputNode, point, direction)
        except Exception:
            logging.warning("Could not snap the ROI center:\n" + traceback.format_exc())
            snapped = None
        if snapped is None:   # nothing on the line: keep the current depth
            snapped = point + np.dot(self.logic.center - point, direction) * direction
        return snapped

    def placeCenterFromMip(self, tag, point):
        center = self.snapAlongMip(tag, point)
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)   # the click is ours
        self._updating = True
        try:
            self.centerNode.RemoveAllControlPoints()
            self.centerNode.AddControlPoint(vtk.vtkVector3d(*[float(x) for x in center]))
        finally:
            self._updating = False
        self.setPlacingRoi(False)
        self.widget.setCurrentLane(self)
        self.logic.center = center
        self.viewCenter = center.copy()
        if not self.roiPlaced:
            self.onRoiPlaced()
        self.updateGeometry()
        self.updateWindowLevel()

    def setPlacingRoi(self, on):
        """Placing mode: cross cursor over the MIPs and a preview point under the mouse."""
        self.placingRoi = bool(on)
        for tag in self.mipTags:
            widget = threeDWidgetByTag(tag)
            if widget:
                if on:
                    widget.threeDView().setCursor(qt.QCursor(qt.Qt.CrossCursor))
                else:
                    widget.threeDView().unsetCursor()
        if not on:
            self.hidePlacementPreview()

    def onMipMouseMove(self, tag, interactor):
        if not (self.placingRoi and self.inputNode and self.previewNode):
            return
        if self.mipPress is not None:
            self.hidePlacementPreview()   # panning or zooming
            return
        point = self.mipDisplayToWorld(tag, interactor.GetEventPosition())
        if point is None:
            return
        p = self.snapAlongMip(tag, point)
        self.previewNode.SetNthControlPointPositionWorld(0, float(p[0]), float(p[1]), float(p[2]))
        display = self.previewNode.GetDisplayNode()
        if display and not display.GetVisibility():
            display.SetVisibility(True)

    def hidePlacementPreview(self):
        display = self.previewNode.GetDisplayNode() if self.previewNode else None
        if display and display.GetVisibility():
            display.SetVisibility(False)

    def clearViewAnnotations(self):
        for renderer, actor in self.annotationActors:
            renderer.RemoveViewProp(actor)
        self.annotationActors = []

    def updateViewAnnotations(self):
        """Role and volume name in the top left corner of every processing view."""
        self.clearViewAnnotations()
        if not self.inputNode:
            return
        title = self.title()
        layoutManager = slicer.app.layoutManager()
        targets = []
        for tag in self.mipTags:
            widget = threeDWidgetByTag(tag)
            if widget:
                targets.append((widget.threeDView(), f"{title}\n{MIP_TITLES[self.mipKind[tag]]}", (0.0, 0.0, 0.0)))
        for tag in tuple(self.views["edit"].values()) + tuple(self.views["preview"].values()):
            widget = layoutManager.sliceWidget(tag)
            if widget:
                targets.append((widget.sliceView(), title, (1.0, 0.95, 0.6)))
        for view, text, color in targets:
            renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
            if renderer is None:
                continue
            actor = vtk.vtkCornerAnnotation()
            actor.SetText(vtk.vtkCornerAnnotation.UpperLeft, text)
            actor.SetMinimumFontSize(9)
            actor.SetMaximumFontSize(14)
            actor.GetTextProperty().SetColor(*color)
            actor.GetTextProperty().SetBold(True)
            actor.PickableOff()
            renderer.AddViewProp(actor)
            self.annotationActors.append((renderer, actor))
            view.scheduleRender()

    def startPlacingRoi(self):
        if not self.centerNode:
            return
        self._updating = True
        try:
            self.centerNode.RemoveAllControlPoints()
        finally:
            self._updating = False
        self.setPlacingRoi(True)   # the next click in a MIP places it (handled in onMipRelease)
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
        self.hint = ("<b>Click the center of the left ventricle</b> in one of the MIPs. "
                     "The depth is set to the count-weighted center along the projection line. "
                     "Afterwards drag the yellow point to move the ROI and the orange point to resize it.")
        self.widget.refreshPanel(self)

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
        self.widget.setCurrentLane(self)
        self.logic.center = center
        self.viewCenter = center.copy()  # MIP drags never pan the slice views, so re-centering is safe
        if not self.roiPlaced:
            self.onRoiPlaced()
        self.updateGeometry()

    def onCenterDefined(self, caller, event):
        if self._updating:
            return
        self.setPlacingRoi(False)
        self.onCenterModified(caller, event)
        self.updateWindowLevel()

    def onRoiPlaced(self):
        self.roiPlaced = True
        for node in ([self.mipRadiusNode] + list(self.curves.values())
                     + list(self.mipRingNodes.values())):
            node.GetDisplayNode().SetVisibility(True)
        self.hint = ("ROI placed. In the MIPs drag the yellow point to move it and the orange point to "
                     "resize it. Then adjust the axes in the top row of slices.")
        self.widget.refreshPanel(self)
        self.updateMaskVisibility()
        self.widget.updateButtonStates()

    def onMipRadiusHandleModified(self, caller, event):
        if self._updating or not self.roiPlaced or self.mipRadiusPosition is None:
            return
        position = self.pointWorld(caller, 0)
        if np.linalg.norm(position - self.mipRadiusPosition) <= 1e-3:
            return
        self.widget.setCurrentLane(self)
        self.draggedRadiusHandle = "mip"
        self.mipRadiusPosition = position
        self.setRadiusFromHandle(position)

    def setRadiusFromHandle(self, position):
        radius = float(np.clip(np.linalg.norm(position - self.logic.center), self.widget.RADIUS_MIN, self.widget.RADIUS_MAX))
        self.logic.radius = radius
        self.updateGeometry()
        self.widget.refreshPanel(self)

    def setRadius(self, value):
        self.logic.radius = float(value)
        if self.roiPlaced:
            self.updateGeometry()
            self.updateWindowLevel()

    def updateRoiOutlines(self):
        angles = np.linspace(0.0, 2.0 * math.pi, RING_POINTS, endpoint=False)
        for tag, ring in self.mipRingNodes.items():
            if ring.GetNumberOfControlPoints() != RING_POINTS:
                continue
            direction, up = MIP_CAMERAS[self.mipKind[tag]]
            across = np.cross(direction, up)
            points = (self.logic.center[None, :] + self.logic.radius
                      * (np.cos(angles)[:, None] * across[None, :] + np.sin(angles)[:, None] * up[None, :]))
            wasModifying = ring.StartModify()
            for index, p in enumerate(points):
                ring.SetNthControlPointPositionWorld(index, float(p[0]), float(p[1]), float(p[2]))
            ring.EndModify(wasModifying)

    def updateRadiusHandles(self):
        if not self.mipRadiusNode or self.mipRadiusNode.GetNumberOfControlPoints() != 1:
            return
        if self.draggedRadiusHandle != "mip":   # leave the handle under the mouse alone until the drag ends
            p = self.logic.center + self.logic.radius * SUPERIOR
            self.mipRadiusNode.SetNthControlPointPositionWorld(0, float(p[0]), float(p[1]), float(p[2]))
        self.mipRadiusPosition = self.pointWorld(self.mipRadiusNode, 0)

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
        colormap = self.widget.colormapCombo.currentText
        setVolumeDisplay(self.inputNode, colormap, 0.0, self.windowMax)
        if self.maskedNode and slicer.mrmlScene.IsNodePresent(self.maskedNode):
            setVolumeDisplay(self.maskedNode, colormap, 0.0, self.windowMax)

    def updateMaskedPreview(self):
        """Bottom row shows either the input or a copy with everything outside the ROI zeroed."""
        if not self.inputNode:
            return
        background = self.inputNode
        if self.widget.maskCheckBox.checked and self.roiPlaced:
            try:
                if not (self.maskedNode and slicer.mrmlScene.IsNodePresent(self.maskedNode)):
                    self.maskedNode = slicer.mrmlScene.AddNewNodeByClass(
                        "vtkMRMLScalarVolumeNode", slicer.mrmlScene.GenerateUniqueName("MPI masked preview"))
                    self.maskedNode.SetHideFromEditors(True)
                    notSaved(self.maskedNode)
                ijkToRas = vtk.vtkMatrix4x4()
                self.inputNode.GetIJKToRASMatrix(ijkToRas)
                self.maskedNode.SetIJKToRASMatrix(ijkToRas)
                self.maskedNode.SetAndObserveTransformNodeID(self.inputNode.GetTransformNodeID())
                masked = self.logic.maskedArray(self.inputNode, self.logic.center,
                                                self.logic.maskAxes(), self.logic.maskSemiAxes)
                slicer.util.updateVolumeFromArray(self.maskedNode, masked)
                windowMax = self.windowMax if self.windowMax else float(masked.max())
                setVolumeDisplay(self.maskedNode, self.widget.colormapCombo.currentText, 0.0, windowMax)
                background = self.maskedNode
            except Exception:
                logging.error("Could not mask the preview:\n" + traceback.format_exc())
        for tag in self.views["preview"].values():
            showVolumeInView(tag, background)

    def maskHandleDirections(self):
        f = self.logic.frame()
        return np.array([f["u"], f["w"], -f["s"]])  # apex, anterior, lateral

    def updateMaskVisibility(self):
        visible = bool(self.widget.maskCheckBox.checked and self.roiPlaced)
        for node in (self.maskHandleNode, self.maskModelNode):
            if node and node.GetDisplayNode():
                node.GetDisplayNode().SetVisibility(visible)

    def onMaskSemiAxisChanged(self, index, value):
        self.logic.maskSemiAxes[index] = float(value)
        self.updateGeometry()

    def onResetMask(self):
        self.logic.resetMaskEllipsoid()
        self.widget.refreshPanel(self)
        self.updateGeometry()

    def onMaskHandleModified(self, caller, event):
        if self._updating or not self.roiPlaced or self.maskHandlePositions is None:
            return
        directions = self.maskHandleDirections()
        for index in range(3):
            position = self.pointWorld(caller, index)
            if np.linalg.norm(position - self.maskHandlePositions[index]) > 1e-3:
                self.widget.setCurrentLane(self)
                self.draggedMaskHandle = index
                self.maskHandlePositions[index] = position
                semi = abs(float(np.dot(position - self.logic.center, directions[index])))
                self.logic.maskSemiAxes[index] = float(np.clip(semi, 5.0, 150.0))
                self.widget.refreshPanel(self)
                self.updateGeometry()
                return

    def updateMaskShape(self):
        """Ellipsoid outline and handles; only drawn while masking is on."""
        if not (self.maskModelNode and self.maskHandleNode and self.widget.maskCheckBox.checked):
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
        self.widget.setCurrentLane(self)
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

            # MIPs: ROI marker, resize handle, outlines
            markupCenter = self.centerFromMarkup()
            if markupCenter is not None and np.linalg.norm(markupCenter - c) > 1e-3:
                self.centerNode.SetNthControlPointPositionWorld(0, float(c[0]), float(c[1]), float(c[2]))
            self.updateRadiusHandles()
            self.updateRoiOutlines()
            self.updateMaskShape()

            fov = self.cropFieldOfView()

            # top row: adjustment planes pass through the center, panned to the view center
            for which in EDIT_ORDER:
                geometry = self.logic.editViewGeometry(which)
                normal = unit(geometry["normal"])
                origin = self.viewCenter - np.dot(self.viewCenter - c, normal) * normal
                setSliceGeometry(sliceNodeByTag(self.views["edit"][which]),
                                 geometry["right"], geometry["up"], normal, origin, fov)
                self.setCurve(which, c, unit(geometry["line"]))

            # bottom row: live SA / HLA / VLA preview
            axes = self.logic.orientationAxes()
            for name in ORIENTATIONS:
                right, up, stack = axes[name]
                setSliceGeometry(sliceNodeByTag(self.views["preview"][name]), right, up, stack, c, fov)

            self.angleText = (f"azimuth {self.logic.azimuthDegrees():.0f}\u00b0   "
                              f"elevation {math.degrees(self.logic.elevation()):.0f}\u00b0   "
                              f"roll {math.degrees(self.logic.roll):.0f}\u00b0")
        except Exception:
            logging.error("Failed to update reorientation views:\n" + traceback.format_exc())
        finally:
            self._updating = False
        self.widget.refreshPanel(self, angleOnly=True)
        if self.widget.maskCheckBox.checked and not self.maskTimer.isActive():
            self.maskTimer.start()

    def createVolumes(self):
        """Resample SA / HLA / VLA of this study. True when all three were made."""
        if not (self.inputNode and self.roiPlaced):
            return False
        with slicer.util.tryWithErrorDisplay(f"Failed to create the reoriented volumes of {self.title()}.",
                                             waitCursor=True):
            # Re-running replaces the volumes made before from this study
            previous = list(self.outputNodes.values()) + list(session.outputsForStudy(self.inputNode).values())
            for node in previous:
                if node and slicer.mrmlScene.IsNodePresent(node):
                    slicer.mrmlScene.RemoveNode(node)
            self.outputNodes, self.outputGeometry = {}, {}

            if self.windowMax is None:
                self.updateWindowLevel()
            spacing = float(self.widget.spacingSpin.value)
            thickness = float(self.widget.thicknessSpin.value)
            interpolation = self.widget.interpolationCombo.itemData(self.widget.interpolationCombo.currentIndex) \
                or DEFAULT_INTERPOLATION
            maskOutputs = bool(self.widget.maskCheckBox.checked and self.widget.maskOutputsCheckBox.checked)
            self.outputNodes, self.outputGeometry, peak = self.logic.createOrientations(
                self.inputNode, spacing, thickness, interpolation, maskOutputs)
            self.outputWindowMax = peak if peak > 0 else None
            for node in self.outputNodes.values():
                setVolumeDisplay(node, self.widget.colormapCombo.currentText, 0.0, self.outputWindowMax or self.windowMax)

        return len(self.outputNodes) == len(ORIENTATIONS)

    def applyColormap(self):
        label = self.widget.colormapCombo.currentText
        for node in [self.inputNode, self.maskedNode] + list(self.outputNodes.values()):
            if node and slicer.mrmlScene.IsNodePresent(node):
                setVolumeDisplay(node, label)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
MODE_SINGLE, MODE_DUAL = "single", "dual"
DUAL_ROLES = ("Stress", "Rest")   # top and bottom study of the stress-rest screen


class MPIProcessingWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    RADIUS_MIN = 10.0
    RADIUS_MAX = 150.0

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self.lanes = []
        self.currentLane = None
        self.mode = MODE_SINGLE       # screen in use
        self.preferredMode = None     # chosen by the user; None: stress-rest whenever both studies exist

    # --- UI ------------------------------------------------------------------
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = MPIProcessingLogic()

        # Studies: the active study of the toolbar, or the stress and rest studies of Data roles
        self.studySelector = StudyPairSelector(self.layout)
        header = qt.QFrame()
        self.layout.addWidget(header)
        form = qt.QFormLayout(header)

        modeRow = qt.QHBoxLayout()
        self.singleRadio = qt.QRadioButton("Active study")
        self.singleRadio.toolTip = "One processing screen for the active study of the toolbar"
        self.dualRadio = qt.QRadioButton("Stress + rest")
        self.dualRadio.toolTip = ("Two processing screens stacked: the stress study above the rest study "
                                  "(roles set in Data roles)")
        modeGroup = qt.QButtonGroup(header)
        modeGroup.addButton(self.singleRadio)
        modeGroup.addButton(self.dualRadio)
        modeRow.addWidget(self.singleRadio)
        modeRow.addWidget(self.dualRadio)
        modeRow.addStretch(1)
        form.addRow("Screen:", modeRow)

        self.studyLabel = narrowLabel(qt.QLabel())
        self.studyLabel.toolTip = "Change the active study in the toolbar and the roles in Data roles"
        form.addRow("Study:", self.studyLabel)

        self.laneRow = qt.QWidget()
        laneLayout = qt.QHBoxLayout(self.laneRow)
        laneLayout.setContentsMargins(0, 0, 0, 0)
        self.laneButtons = []
        laneGroup = qt.QButtonGroup(self.laneRow)
        laneGroup.setExclusive(True)
        for index, role in enumerate(DUAL_ROLES):
            button = qt.QPushButton(role)
            button.checkable = True
            button.toolTip = (f"The panel below edits the {role.lower()} study. Dragging a point in its views "
                              "switches here too.")
            laneGroup.addButton(button)
            laneLayout.addWidget(button)
            self.laneButtons.append(button)
        self.laneLabel = qt.QLabel("Editing:")
        form.addRow(self.laneLabel, self.laneRow)

        self.colormapCombo = qt.QComboBox()
        self.colormapCombo.addItems([label for label, _ in COLORMAPS])
        form.addRow("Color map:", self.colormapCombo)

        # 1. Heart ROI
        self.roiSection = ctk.ctkCollapsibleButton()
        self.roiSection.text = "1. Heart ROI"
        self.layout.addWidget(self.roiSection)
        roiForm = qt.QFormLayout(self.roiSection)

        self.roiHint = qt.QLabel(NO_STUDY_HINT)
        self.roiHint.wordWrap = True
        roiForm.addRow(self.roiHint)

        self.placeButton = styleButton(qt.QPushButton("Place heart ROI"), "action")
        self.placeButton.toolTip = "Click the center of the left ventricle in one of the MIPs"
        roiForm.addRow(self.placeButton)

        self.radiusSlider = ctk.ctkSliderWidget()
        self.radiusSlider.minimum = self.RADIUS_MIN
        self.radiusSlider.maximum = self.RADIUS_MAX
        self.radiusSlider.singleStep = 1.0
        self.radiusSlider.decimals = 0
        self.radiusSlider.value = DEFAULT_ROI_RADIUS_MM
        self.radiusSlider.suffix = " mm"
        self.radiusSlider.toolTip = ("Radius of the spherical ROI (also: drag the orange point in the MIPs). "
                                     "It is also the half size of the output volumes.")
        roiForm.addRow("ROI radius:", self.radiusSlider)

        # 2. Axes
        self.axesSection = ctk.ctkCollapsibleButton()
        self.axesSection.text = "2. Cardiac axes"
        self.axesSection.collapsed = True
        self.layout.addWidget(self.axesSection)
        axesForm = qt.QFormLayout(self.axesSection)

        axesHint = qt.QLabel(
            "Top row: drag an end point to rotate a line, drag the middle point to move the center.<br>"
            "<b>Adjust TA</b> (transaxial): align the line with the LV long axis, base → apex.<br>"
            "<b>Adjust VLA</b> (vertical plane through that line): align the long axis elevation.<br>"
            "<b>Adjust SA</b> (short axis): rotate so the line points to the anterior wall.<br>"
            "The bottom row shows the resliced SA / HLA / VLA through the center.")
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
        self.mipSection.text = "MIP views"
        self.mipSection.collapsed = True
        self.layout.addWidget(self.mipSection)
        mipForm = qt.QFormLayout(self.mipSection)

        self.mipLevelSlider = ctk.ctkSliderWidget()
        self.mipLevelSlider.minimum = 5.0
        self.mipLevelSlider.maximum = 100.0
        self.mipLevelSlider.singleStep = 1.0
        self.mipLevelSlider.decimals = 0
        self.mipLevelSlider.value = 60.0
        self.mipLevelSlider.suffix = " %"
        self.mipLevelSlider.toolTip = "Intensity shown as full black, as a percentage of the volume maximum"
        mipForm.addRow("Upper level:", self.mipLevelSlider)

        self.mipResetButton = qt.QPushButton("Reset MIP views")
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

        self.interpolationCombo = qt.QComboBox()
        for mode in INTERPOLATION_MODES:
            self.interpolationCombo.addItem(INTERPOLATION_LABELS[mode], mode)
        self.interpolationCombo.setCurrentIndex(INTERPOLATION_MODES.index(DEFAULT_INTERPOLATION))
        self.interpolationCombo.toolTip = (
            "How the volume is sampled along the cardiac axes. Windowed sinc (Lanczos, default) is the "
            "sharpest but can ring slightly next to hot edges; cubic B-spline is smooth and passes through "
            "the original voxel values; linear leaves faceted, blocky slices on coarse SPECT voxels. "
            "The slice panel is resized to the screen with the same method.")
        outForm.addRow("Interpolation:", self.interpolationCombo)

        self.createButton = styleButton(qt.QPushButton("Create SA / HLA / VLA volumes"), "create")
        self.createButton.toolTip = "Resample along the confirmed axes and review the slices"
        outForm.addRow(self.createButton)

        self.reviewButton = styleButton(qt.QPushButton("Review slices ›"), "primary")
        self.reviewButton.toolTip = "Open Perfusion with the reoriented volumes of the active study"
        self.layout.addWidget(self.reviewButton)   # outside the sections: usable before the ROI is placed

        self.parameterPanel = parametersLib.ParameterPanel(
            "processing", self.layout, onChanged=None,
            hint=("Used for the next heart ROI, axes, ellipsoid or click; the cardiac axes and masks of the studies already processed are kept."))
        self.parameterPanel.bind("colormap", self.colormapCombo)
        self.parameterPanel.bind("maskPreview", self.maskCheckBox)
        self.parameterPanel.bind("maskOutputs", self.maskOutputsCheckBox)
        self.parameterPanel.bind("mipLevel", self.mipLevelSlider)
        self.parameterPanel.bind("pixelSize", self.spacingSpin)
        self.parameterPanel.bind("sliceThickness", self.thicknessSpin)
        self.parameterPanel.bind("interpolation", self.interpolationCombo)
        self.layout.addStretch(1)

        self.lanes = [ProcessingLane(self, index) for index in range(len(LANE_SUFFIXES))]
        self.currentLane = self.lanes[0]

        # Connections: the ROI, axes and mask controls edit the current study, the rest apply to all
        self.singleRadio.connect("clicked(bool)", lambda checked: self.onModeChosen(MODE_SINGLE))
        self.dualRadio.connect("clicked(bool)", lambda checked: self.onModeChosen(MODE_DUAL))
        for index, button in enumerate(self.laneButtons):
            button.connect("clicked(bool)", lambda checked, i=index: self.setCurrentLane(self.lanes[i]))
        self.placeButton.connect("clicked(bool)", lambda: self.currentLane.startPlacingRoi())
        self.radiusSlider.connect("valueChanged(double)", lambda value: self.currentLane.setRadius(value))
        self.resetAxesButton.connect("clicked(bool)", lambda: self.currentLane.onResetAxes())
        self.maskCheckBox.connect("toggled(bool)", self.onMaskToggled)
        for index, spin in enumerate(self.maskSemiAxisSpins):
            spin.connect("valueChanged(double)",
                         lambda value, i=index: self.currentLane.onMaskSemiAxisChanged(i, value))
        self.resetMaskButton.connect("clicked(bool)", lambda: self.currentLane.onResetMask())
        self.colormapCombo.connect("currentIndexChanged(int)", self.onColormapChanged)
        self.mipLevelSlider.connect("valueChanged(double)", lambda _: self.forLanes("updateMipTransfer"))
        self.mipResetButton.connect("clicked(bool)", lambda: self.forLanes("resetMipCamera"))
        self.createButton.connect("clicked(bool)", self.onCreateVolumes)
        self.reviewButton.connect("clicked(bool)", lambda: session.selectStep("perfusion"))
        self.parameterPanel.loadSaved()   # saved settings of this step, as if set by hand

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneClosed)
        session.addListener(self.onSessionChanged)
        _installToolbar()
        self.onSessionChanged()
        self.refreshPanel(self.currentLane)
        self.updateButtonStates()

    # --- workflow toolbar (second row) -----------------------------------------------------------------
    def buildToolbar(self, bar):
        self.toolbarBar = bar
        bar.mirror(self.placeButton, "Place ROI", kind="action")
        bar.mirror(self.resetAxesButton, "Reset axes", "Back to the axes found automatically")
        bar.separator()
        bar.mirror(self.maskCheckBox, "Mask", "Mask the bottom row (and the created volumes, when that is on) "
                                              "outside the ellipsoid")
        bar.separator()
        bar.mirror(self.createButton, "Create SA / HLA / VLA", kind="create")

    def syncToolbar(self):
        bar = getattr(self, "toolbarBar", None)
        if bar is not None:
            bar.syncMirrors()

    def enter(self):
        self.studySelector.refresh()
        self.loadStudies(showLayout=True)

    def cleanup(self):
        session.removeListener(self.onSessionChanged)
        self.studySelector.cleanup()
        for lane in self.lanes:
            lane.cleanup()
        self.removeObservers()

    # --- studies ---------------------------------------------------------------
    @staticmethod
    def isShown():
        return slicer.util.selectedModule() == session.HUB_MODULE and session.currentStep() == "processing"

    @staticmethod
    def stressRestPair():
        """(Study 1, Study 2) static volumes of the stress-rest screen (picked in the Study 1 / Study 2
        boxes, else from the roles), or None."""
        pair = session.stressRestStudies()
        stress, rest = pair["stress"], pair["rest"]
        if stress is None or rest is None or stress is rest:
            return None
        if stress.GetImageData() is None or rest.GetImageData() is None:
            return None
        return stress, rest

    def wantedStudies(self):
        """(mode, [input volume per lane]) the screen should show now."""
        pair = self.stressRestPair()
        mode = self.preferredMode or (MODE_DUAL if pair else MODE_SINGLE)
        if mode == MODE_DUAL and pair:
            return MODE_DUAL, list(pair), pair
        active = session.activeStudy()
        inputs = [active] if active is not None and active.GetImageData() is not None else []
        return MODE_SINGLE, inputs, pair

    def activeLanes(self):
        return self.lanes[:2] if self.mode == MODE_DUAL else self.lanes[:1]

    def loadStudies(self, showLayout=False):
        """Bring the screen in line with the wanted studies; keep lanes whose study did not change."""
        mode, inputs, pair = self.wantedStudies()
        self.updateModeButtons(mode, pair)
        if not inputs:
            self.updateStudyLabel()
            self.updateButtonStates()
            return
        lanes = self.lanes[:len(inputs)]
        changed = mode != self.mode or any(not (lane.inputNode is node and lane.curves)
                                           for lane, node in zip(lanes, inputs))
        if changed:
            try:
                for lane in self.lanes[len(inputs):]:
                    lane.stop()
                started = []
                for lane, node in zip(lanes, inputs):
                    if not (lane.inputNode is node and lane.curves):
                        lane.start(node)
                        started.append(lane)
                self.mode = mode
                self.activateLayout()
                for lane in started:
                    lane.startPlacingRoi()
            except Exception as e:
                slicer.util.errorDisplay(f"Could not start processing: {e}", detailedText=traceback.format_exc())
        elif showLayout:
            self.activateLayout()   # coming back from another step: keep the running session

        active = session.activeStudy()
        current = next((lane for lane in self.activeLanes() if lane.inputNode is active), None)
        if self.currentLane not in self.activeLanes() or (changed and current):
            self.currentLane = None
        self.setCurrentLane(current or self.currentLane or self.lanes[0])
        self.updateStudyLabel()
        self.updateButtonStates()

    def activateLayout(self):
        if self.mode == MODE_DUAL:
            layoutId = DUAL_REORIENT_LAYOUT_ID
            registerLayout(layoutId, dualReorientLayoutXml(tuple(role[0] for role in DUAL_ROLES)))
        else:
            layoutId = REORIENT_LAYOUT_ID
            registerLayout(layoutId, reorientLayoutXml())
        slicer.app.layoutManager().setLayout(layoutId)
        slicer.app.processEvents()
        lanes = [lane for lane in self.activeLanes() if lane.inputNode]
        for lane in lanes:
            lane.attachViews()
        # Views get their real pixel size only after Qt has laid them out
        qt.QTimer.singleShot(150, lambda: [lane.refreshViews() for lane in lanes if lane.inputNode])

    def layoutShown(self):
        return slicer.app.layoutManager().layout in (REORIENT_LAYOUT_ID, DUAL_REORIENT_LAYOUT_ID)

    def onModeChosen(self, mode):
        self.preferredMode = mode
        self.loadStudies(showLayout=True)

    def onSessionChanged(self):
        if self.isShown():
            self.loadStudies()   # another study or role picked while this step is open
        else:
            self.updateModeButtons(*self.wantedStudies()[::2])
            self.updateStudyLabel()
            self.updateButtonStates()

    def updateModeButtons(self, mode, pair):
        self.dualRadio.enabled = bool(pair)
        self.dualRadio.toolTip = ("Two processing screens stacked: the stress study above the rest study"
                                  if pair else "Needs a Stress and a Rest study (set the roles in Data roles)")
        for radio, value in ((self.singleRadio, MODE_SINGLE), (self.dualRadio, MODE_DUAL)):
            wasBlocked = radio.blockSignals(True)
            radio.checked = mode == value
            radio.blockSignals(wasBlocked)

    def updateStudyLabel(self):
        lanes = [lane for lane in self.activeLanes() if lane.inputNode is not None]
        if self.mode == MODE_DUAL and lanes:
            self.studyLabel.text = "<br>".join(f"<b>{breakable(lane.title())}</b>" for lane in lanes)
        else:
            active = session.activeStudy()
            self.studyLabel.text = f"<b>{breakable(studyTitle(active))}</b>" if active is not None else "<i>none</i>"
        if self.layoutShown():
            for lane in lanes:
                lane.updateViewAnnotations()

    # --- panel <-> current study -------------------------------------------------
    def setCurrentLane(self, lane):
        if lane is self.currentLane:
            return
        self.currentLane = lane
        for index, button in enumerate(self.laneButtons):
            wasBlocked = button.blockSignals(True)
            button.checked = self.lanes[index] is lane
            button.blockSignals(wasBlocked)
        self.refreshPanel(lane)

    def refreshPanel(self, lane, angleOnly=False):
        """Show the state of the current study in the panel (other studies are ignored)."""
        if lane is not self.currentLane:
            return
        self.angleLabel.text = lane.angleText
        if angleOnly:
            return
        self.roiHint.text = lane.hint
        wasBlocked = self.radiusSlider.blockSignals(True)
        self.radiusSlider.value = lane.logic.radius
        self.radiusSlider.blockSignals(wasBlocked)
        for spin, value in zip(self.maskSemiAxisSpins, lane.logic.maskSemiAxes):
            wasBlocked = spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(wasBlocked)
        self.updateButtonStates()

    def updateButtonStates(self):
        lanes = [lane for lane in self.activeLanes() if lane.inputNode is not None]
        lane = self.currentLane
        started = bool(lanes)
        dual = self.mode == MODE_DUAL and started
        self.laneLabel.visible = dual
        self.laneRow.visible = dual
        self.roiSection.enabled = started
        self.axesSection.enabled = bool(lane and lane.inputNode and lane.roiPlaced)
        self.maskSection.enabled = bool(lane and lane.inputNode and lane.roiPlaced)
        self.mipSection.enabled = started
        self.outputSection.enabled = started
        self.createButton.enabled = started and all(l.roiPlaced for l in lanes)
        self.createButton.text = ("Create SA / HLA / VLA volumes (stress and rest)" if dual
                                  else "Create SA / HLA / VLA volumes")
        self.reviewButton.enabled = len(session.outputsForStudy(session.activeStudy())) == len(ORIENTATIONS)
        self.syncToolbar()

    def forLanes(self, method):
        for lane in self.activeLanes():
            if lane.inputNode is not None:
                getattr(lane, method)()

    # --- all studies --------------------------------------------------------------
    def onSceneClosed(self, caller=None, event=None):
        for lane in self.lanes:
            lane.sceneClosed()
        self.roiHint.text = NO_STUDY_HINT
        self.angleLabel.text = "-"
        if getattr(slicer, "_mpiSliceMosaicState", None):
            mosaicPanel().clear()
        self.updateStudyLabel()
        self.updateButtonStates()

    def onMaskToggled(self, checked):
        self.maskOutputsCheckBox.enabled = checked
        for lane in self.activeLanes():
            if lane.inputNode is not None:
                lane.updateMaskVisibility()
                lane.updateGeometry()
                lane.updateMaskedPreview()

    # --- mask from the later steps (Myocardium toolbar) ------------------------------
    @staticmethod
    def outputsMasked(study):
        """True / False: the SA / HLA / VLA of a study were made with / without the mask; None: not made."""
        outputs = session.outputsForStudy(study)
        reference = outputs.get("SA") or next(iter(outputs.values()), None)
        if reference is None:
            return None
        return (reference.GetAttribute("MPI.Mask") or "none") != "none"

    def remakeVolumes(self, study, masked):
        """Make the SA / HLA / VLA of a study again with or without the mask ellipsoid, in the frame, pixel size,
        slice thickness and interpolation they were made with (the ellipsoid as drawn here when the study is on
        this screen, else the one kept on the volumes). The mask check boxes follow, so Create does the same.
        True when all three were made."""
        outputs = session.outputsForStudy(study)
        reference = outputs.get("SA") or next(iter(outputs.values()), None)
        if reference is None:
            return False
        logic = MPIProcessingLogic.fromOutputs(reference)
        lane = next((l for l in self.lanes if l.inputNode is study and l.roiPlaced), None)
        if lane is not None:
            logic.maskSemiAxes = np.array(lane.logic.maskSemiAxes, dtype=float)
        spacing = float(reference.GetAttribute("MPI.PixelSizeMm") or self.spacingSpin.value)
        thickness = float(reference.GetAttribute("MPI.SliceThicknessMm") or self.thicknessSpin.value)
        interpolation = reference.GetAttribute("MPI.Interpolation") or DEFAULT_INTERPOLATION
        if masked:
            self.maskCheckBox.checked = True
        self.maskOutputsCheckBox.checked = bool(masked)
        nodes, geometries, peak = logic.createOrientations(study, spacing, thickness, interpolation, bool(masked))
        for node in outputs.values():   # the new ones are made: the old ones go
            if node and slicer.mrmlScene.IsNodePresent(node):
                slicer.mrmlScene.RemoveNode(node)
        for node in nodes.values():
            setVolumeDisplay(node, self.colormapCombo.currentText, 0.0, peak if peak > 0 else None)
        if lane is not None:
            lane.outputNodes, lane.outputGeometry, lane.outputWindowMax = nodes, geometries, (peak if peak > 0 else None)
        self.updateButtonStates()
        return len(nodes) == len(ORIENTATIONS)

    def onColormapChanged(self, index):
        self.forLanes("applyColormap")

    def onCreateVolumes(self):
        lanes = [lane for lane in self.activeLanes() if lane.inputNode is not None]
        missing = [lane.title() for lane in lanes if not lane.roiPlaced]
        if not lanes or missing:
            slicer.util.errorDisplay("Place the heart ROI first" + (": " + ", ".join(missing) if missing else "."))
            return
        done = [lane.createVolumes() for lane in lanes]
        self.updateButtonStates()
        if all(done):
            # Leave place mode in case the user was about to re-place the ROI
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
            session.setActiveStudy(lanes[0].inputNode)
            session.notify()
            session.selectStep("myocardium")   # detects the LV wall on entry


# ---------------------------------------------------------------------------
# Tests (Reload and Test in developer mode)
# ---------------------------------------------------------------------------
class MPIProcessingTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_FrameIsOrthonormal()
        self.test_FrameFromOutputs()
        self.setUp()
        self.test_ResamplingKeepsAnatomy()
        self.setUp()
        self.test_InterpolationModes()
        self.test_ZoomInPlane()
        self.setUp()
        self.test_Studies()
        self.setUp()
        self.test_SnapAlongLine()
        self.test_DualLayout()

    def _obliqueLogic(self):
        logic = MPIProcessingLogic()
        logic.setLineDirection("TA", [-0.3, 1.0, 0.0])
        logic.setLineDirection("VLA", logic.horizontalDirection() - 0.6 * SUPERIOR)
        logic.roll = math.radians(15.0)
        return logic

    def test_FrameFromOutputs(self):
        """The frame and mask ellipsoid kept on the volumes give the same axes back (Mask in the Myocardium
        toolbar makes the volumes again from them)."""
        logic = self._obliqueLogic()
        logic.center = np.array([12.0, -30.0, 41.0])
        logic.radius = 62.0
        logic.maskSemiAxes = np.array([58.0, 44.0, 47.0])

        class Volume:
            def __init__(self, attributes):
                self.attributes = attributes

            def GetAttribute(self, name):
                return self.attributes.get(name)
        again = MPIProcessingLogic.fromOutputs(Volume(logic.frameAttributes()))
        for name, axes in logic.orientationAxes().items():
            for a, b in zip(axes, again.orientationAxes()[name]):
                self.assertTrue(np.allclose(a, b, atol=1e-3), name)
        self.assertTrue(np.allclose(again.center, logic.center, atol=0.01))
        self.assertAlmostEqual(again.radius, 62.0, places=3)
        self.assertTrue(np.allclose(again.maskSemiAxes, logic.maskSemiAxes))
        # volumes made before the semi-axes were kept: read from the mask text, else the default ellipsoid
        old = {k: v for k, v in logic.frameAttributes().items() if k != "MPI.MaskSemiAxesMm"}
        text = dict(old, **{"MPI.Mask": "ellipsoid semi-axes (long, anterior-inferior, septal-lateral) 50, 40, 41 mm"})
        self.assertTrue(np.allclose(MPIProcessingLogic.fromOutputs(Volume(text)).maskSemiAxes, [50.0, 40.0, 41.0]))
        self.assertTrue(np.allclose(MPIProcessingLogic.fromOutputs(Volume(dict(old, **{"MPI.Mask": "none"}))).maskSemiAxes,
                                    [MASK_LONG_FRACTION * 62.0, MASK_SHORT_FRACTION * 62.0, MASK_SHORT_FRACTION * 62.0]))

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

    def test_InterpolationModes(self):
        """Every mode keeps the counts and the position of a smooth blob; the higher-order modes
        follow it more closely than linear on a coarse (6.6 mm) grid."""
        voxel = 6.6
        k, j, i = np.indices((24, 24, 24)).astype(float)
        blobCenter = np.array([11.3, 12.6, 10.8])   # i, j, k, off the voxel grid
        r2 = (i - blobCenter[0]) ** 2 + (j - blobCenter[1]) ** 2 + (k - blobCenter[2]) ** 2
        sigmaVoxels = 12.0 / 2.355 / voxel
        array = (100.0 * np.exp(-r2 / (2 * sigmaVoxels ** 2))).astype(np.float32)
        volume = slicer.util.addVolumeFromArray(array, name="MPI test blob")
        volume.SetSpacing(voxel, voxel, voxel)
        blobRas = blobCenter * voxel

        logic = self._obliqueLogic()
        logic.center = blobRas + np.array([3.0, -2.0, 4.0])
        logic.radius = 40.0
        sa = logic.orientationAxes()["SA"]
        errors = {}
        for mode in INTERPOLATION_MODES:
            output, geometry = logic.resampleOrientation(volume, sa, 2.0, f"test {mode}", interpolation=mode)
            self.assertEqual(geometry["interpolation"], mode)
            values = slicer.util.arrayFromVolume(output).astype(float)
            self.assertGreaterEqual(float(values.min()), 0.0, mode)
            kji = np.indices(values.shape).reshape(3, -1)
            ijk = np.vstack([kji[2], kji[1], kji[0], np.ones(kji.shape[1])])
            m = vtk.vtkMatrix4x4()
            output.GetIJKToRASMatrix(m)
            ras = (slicer.util.arrayFromVTKMatrix(m) @ ijk)[:3]
            truth = 100.0 * np.exp(-((ras - blobRas[:, None]) ** 2).sum(axis=0) / (2 * (sigmaVoxels * voxel) ** 2))
            errors[mode] = float(np.sqrt(((values.reshape(-1) - truth) ** 2).mean()))
            total = values.sum() * 8.0
            self.assertLess(abs(total - array.sum() * voxel ** 3) / (array.sum() * voxel ** 3), 0.03, mode)
        self.assertLess(errors["bspline"], errors["linear"])
        self.assertLess(errors["sinc"], errors["linear"])
        self.delayDisplay("Interpolation test passed: RMS error " +
                          ", ".join(f"{m} {e:.3f}" for m, e in errors.items()))

    def test_ZoomInPlane(self):
        stack = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        for mode in INTERPOLATION_MODES:
            zoomed = zoomInPlane(stack, 6, 8, mode)
            self.assertEqual(zoomed.shape, (1, 6, 8))
            self.assertAlmostEqual(float(zoomed.mean()), float(stack.mean()), delta=0.5)
        same = zoomInPlane(stack, 3, 4)
        self.assertTrue(np.allclose(same, stack))
        self.delayDisplay("Zoom test passed")

    def test_Studies(self):
        """Roles are guessed from names, recon-style gated sequences are paired with their static
        volume, confirmed roles are stored on the nodes, and gates can be summed."""
        self.assertEqual(studiesLib.guessRole("STRESS GATED SPECT OSEM AC SC"), "Stress")
        self.assertEqual(studiesLib.guessRole("Istirahat MPI"), "Rest")
        self.assertEqual(studiesLib.guessRole("Tl-201 redistribution 4h"), "Redistribution")
        self.assertEqual(studiesLib.guessRole("SPECT OSEM"), studiesLib.UNKNOWN_ROLE)
        for name, role, gated in (("SGATE", "Stress", True), ("STomo", "Stress", False),
                                  ("RGATE", "Rest", True), ("RTomo", "Rest", False),
                                  ("MPI_SGATED_OSEM", "Stress", True), ("rtomo ac", "Rest", False)):
            self.assertEqual(studiesLib.guessRole(name), role, name)
            self.assertEqual(studiesLib.looksGated(name), gated, name)

        gate = np.ones((16, 16, 16), dtype=np.float32)
        stress = slicer.util.addVolumeFromArray(gate * 8, name="Stress OSEM AC SC")
        rest = slicer.util.addVolumeFromArray(gate * 8, name="Rest OSEM AC SC")
        stress.SetAttribute("SPECTRecon.AcquisitionStart", "2026-01-01T10:00:00")
        rest.SetAttribute("SPECTRecon.AcquisitionStart", "2026-01-01T08:00:00")
        sequences = {}
        for name, start in (("Rest OSEM AC SC gated", "2026-01-01T08:00:00"),
                            ("Stress OSEM AC SC gated", "2026-01-01T10:00:00")):
            sequence = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", name)
            sequence.SetIndexName("Gate")
            for g in range(1, 9):
                gateNode = slicer.util.addVolumeFromArray(gate * g, name=f"{name} G{g}")
                gateNode.SetAttribute("SPECTRecon.AcquisitionStart", start)
                gateNode.SetAttribute("SPECTRecon.Gate", str(g))
                sequence.SetDataNodeAtValue(gateNode, str(g))
                slicer.mrmlScene.RemoveNode(gateNode)
            sequences[name] = sequence

        found = {study.name: study for study in studiesLib.scanStudies()}
        self.assertEqual(set(found), {"Stress OSEM AC SC", "Rest OSEM AC SC"})
        self.assertEqual(found["Stress OSEM AC SC"].role, "Stress")
        self.assertTrue(found["Stress OSEM AC SC"].guessed)
        self.assertIs(found["Stress OSEM AC SC"].gatedNode, sequences["Stress OSEM AC SC gated"])
        self.assertIs(found["Rest OSEM AC SC"].gatedNode, sequences["Rest OSEM AC SC gated"])
        self.assertEqual(found["Rest OSEM AC SC"].gates, 8)

        study = found["Rest OSEM AC SC"]
        study.role = "Viability"
        studiesLib.saveStudy(study)
        again = {s.name: s for s in studiesLib.scanStudies()}["Rest OSEM AC SC"]
        self.assertEqual(again.role, "Viability")
        self.assertFalse(again.guessed)
        self.assertEqual(rest.GetAttribute(studiesLib.GATED_ATTRIBUTE), sequences["Rest OSEM AC SC gated"].GetID())

        summed = studiesLib.sumGates(sequences["Stress OSEM AC SC gated"])
        self.assertTrue(np.allclose(slicer.util.arrayFromVolume(summed), 36.0))
        self.assertIsNone(summed.GetAttribute("SPECTRecon.Gate"))
        self.delayDisplay("Studies test passed")

    def test_DualLayout(self):
        """The stress-rest layout holds two complete processing screens with distinct view tags."""
        import xml.dom.minidom
        document = xml.dom.minidom.parseString(dualReorientLayoutXml())
        tags = [v.getAttribute("singletontag") for v in document.getElementsByTagName("view")]
        self.assertEqual(len(tags), 2 * (2 + 3 + 3))
        self.assertEqual(len(set(tags)), len(tags))
        for suffix in LANE_SUFFIXES:
            views = laneViews(suffix)
            for group in views.values():
                for tag in group.values():
                    self.assertIn(tag, tags)
        self.delayDisplay("Stress-rest layout OK")

    def test_SnapAlongLine(self):
        """A MIP click snaps to the center of the hot LV shell, not to a fainter organ on the same line."""
        k, j, i = np.mgrid[0:40, 0:50, 0:60]
        shell = np.sqrt((i - 30) ** 2 + (j - 20) ** 2 + (k - 20) ** 2)
        array = 100.0 * np.exp(-((shell - 5.0) / 1.5) ** 2)
        array += 35.0 * ((abs(i - 30) < 6) & (abs(j - 40) < 5) & (abs(k - 20) < 6))
        volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "snap test")
        volume.SetSpacing(2.0, 2.0, 2.0)
        slicer.util.updateVolumeFromArray(volume, array.astype(np.float32))
        ijkToRas = vtk.vtkMatrix4x4()
        volume.GetIJKToRASMatrix(ijkToRas)
        lvCenter = np.array(ijkToRas.MultiplyPoint([30, 20, 20, 1])[:3])
        # anterior view: the line runs posterior through x/z of the LV, starting off the LV depth
        snapped = MPIProcessingLogic.weightedCenterAlongLine(
            volume, lvCenter + np.array([0.0, 60.0, 0.0]), MIP_CAMERAS["ANT"][0])
        self.assertIsNotNone(snapped)
        self.assertLess(np.linalg.norm(snapped - lvCenter), 1.0, snapped)
        # left lateral view
        snapped = MPIProcessingLogic.weightedCenterAlongLine(
            volume, lvCenter + np.array([-40.0, 0.0, 0.0]), MIP_CAMERAS["LAT"][0])
        self.assertLess(np.linalg.norm(snapped - lvCenter), 1.0, snapped)
        self.delayDisplay("Snapping along the projection line OK")

# saved Advanced parameters of every step, this module's constants included (see parameters.py)
parametersLib.applySaved()
