"""
Myocardial Perfusion SPECT - 5. Gated QC

Quality control of the gated study of Study 1 (stress) or Study 2 (rest) before the gated review:

  - the myocardium is segmented in every gate frame on the rays of the static wall detected in
    Myocardium (MyocardialPerfusionSPECTLib/myocardium.py fitGate: tracked from the static wall, or every
    frame from scratch; a surface fit, B-spline, deformable mesh or the legacy jump-limited path, runs
    through the rays whose wall the frame shows and draws the others; every frame's thickness from its
    own profile widths, a blurred slab leaning to a prior where the blur hides it, fitted the same way);
  - every frame is resliced to the SA / HLA / VLA grids made in Processing and shown as three rows with
    one tile per frame, all frames at the same slice (mouse wheel over a row scrolls it), with the
    frame's endo- and epicardium;
  - top right: total counts of every frame (projections when the reconstruction stored them, the image
    and the segmented myocardium) in % of their mean;
  - bottom right: the ECG gating information the DICOM SPECT Recon extension stored on the gates
    (R-R window, accepted / rejected beats, heart rate, R-R per projection view when available) with
    the per-frame counts and LV cavity volumes and the QC findings.
The per-frame results are also saved on the gated sequence (MPI.Gate* attributes) for Gated Review.
"""

import os
import re
import sys
import json
import math
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

from MyocardialPerfusionSPECTLib import session  # noqa: E402
from MyocardialPerfusionSPECTLib import parameters as parametersLib  # noqa: E402
from MyocardialPerfusionSPECTLib import studies as studiesLib  # noqa: E402
from MyocardialPerfusionSPECTLib import myocardium as wallLib  # noqa: E402
from MyocardialPerfusionSPECTLib import gating as gatingLib  # noqa: E402
from MyocardialPerfusionSPECTLib.geometry import ellipsoidInsideGrid, sampleTrilinear, unit, worldToLocalMatrix  # noqa: E402
from MyocardialPerfusionSPECTLib.resampling import DEFAULT_INTERPOLATION, reslice, zoomInPlane  # noqa: E402
from MyocardialPerfusionSPECTLib.views import (  # noqa: E402
    GATED_QC_LAYOUT_ID, ORIENTATIONS, ORIENTATION_TITLES, gatedQcLayoutXml, registerLayout)
from MyocardialPerfusionSPECTLib.display import (  # noqa: E402
    COLORMAPS, colorTableArray, colorize, rgbToQImage, gatedMosaicPanel, countCurvePanel, gatingInfoPanel,
    existingPanel, tilePixelsToIjk, BoundedCache, packMask, unpackMask, FRAME_SCALES)
from MyocardialPerfusionSPECTLib.widgets import breakable, narrowLabel, styleButton, StudyPairSelector  # noqa: E402

STRESS, REST = "stress", "rest"
SLOT_TITLES = {STRESS: "STUDY 1 (STRESS) GATED", REST: "STUDY 2 (REST) GATED"}
CONTOUR_COLORS = {"epi": (64, 220, 90), "endo": (60, 190, 255)}
CONTOUR_MAX_SIZE = 160
TILE_CACHE_BYTES = 96 * 1024 * 1024      # frame tiles kept for scrolling back (oldest dropped first)
CONTOUR_CACHE_BYTES = 16 * 1024 * 1024   # contour masks, bit-packed (also used by Gated Review)
LABEL_GRID_MM = 2.0          # grid of the per-frame myocardium / cavity labels (counts and volumes)
SERIES_COLORS = {"projections": (120, 180, 255), "image": (210, 210, 210), "myocardium": (255, 140, 60)}
MOSAIC_STATE, CURVE_STATE, INFO_STATE = "_mpiGatedMosaicState", "_mpiCountCurveState", "_mpiGatingInfoState"


def _installToolbar():
    try:
        from MyocardialPerfusionSPECTLib import toolbar
        toolbar.install()
    except Exception:
        logging.exception("MPI: could not install the workflow toolbar")


class MPIGatedQC(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MPI 5 · Gated QC"
        self.parent.categories = ["Nuclear Medicine.Myocardial Perfusion"]
        self.parent.hidden = True   # opened from the Myocardial Perfusion SPECT hub and the workflow toolbar
        self.parent.dependencies = []
        self.parent.contributors = ["Burak Demir"]
        self.parent.helpText = (
            "Quality control of the gated study: the myocardium segmented in every gate frame, the frames in "
            "SA / HLA / VLA rows (one tile per frame, all at the same slice; mouse wheel over a row scrolls), "
            "the counts of every frame and the ECG gating information (R-R window, accepted and rejected beats).")
        self.parent.acknowledgementText = ""
        iconPath = os.path.join(_MODULE_DIR, "Resources", "Icons", "MyocardialPerfusionSPECT.png")
        if os.path.exists(iconPath):
            self.parent.icon = qt.QIcon(iconPath)
        if not slicer.app.commandOptions().noMainWindow:
            slicer.app.connect("startupCompleted()", _installToolbar)


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
def gatedSequenceOf(volumeNode):
    """The gated sequence paired with a static volume in Data roles, or None."""
    if volumeNode is None:
        return None
    for study in studiesLib.scanStudies():
        if study.staticNode is volumeNode:
            return study.gatedNode
    return None


def gateNodes(sequence):
    """[(index value, volume node)] of a gated sequence."""
    out = []
    for i in range(sequence.GetNumberOfDataNodes()):
        node = sequence.GetNthDataNode(i)
        if node is not None and node.GetImageData() is not None:
            out.append((sequence.GetNthIndexValue(i) or str(i + 1), node))
    return out


def worldToIjk(volumeNode, spaceNode):
    """World RAS -> IJK of a volume that lives in the space of spaceNode (the gates are not in the
    scene, so they take the transform of their static study)."""
    rasToIjk = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(rasToIjk)
    return slicer.util.arrayFromVTKMatrix(rasToIjk) @ slicer.util.arrayFromVTKMatrix(worldToLocalMatrix(spaceNode))


def sampler(array, matrix, mask=None, center=None):
    """Counts of a gate at world points; with mask (maskOf) and center, 0 outside the Processing ellipsoid,
    as in the masked SA / HLA / VLA volumes the static wall was detected on."""
    def sample(points):
        points = np.asarray(points, dtype=float)
        ijk = points @ matrix[:3, :3].T + matrix[:3, 3]
        values = sampleTrilinear(array, ijk)
        if mask is not None and center is not None:
            values = np.where(insideEllipsoid(points, center, *mask), values, 0.0)
        return values
    return sample


def insideEllipsoid(points, center, axes, semiAxes):
    """[...] True for the world points inside the ellipsoid (axes: its unit axes, semiAxes: mm)."""
    offset = np.asarray(points, dtype=float) - np.asarray(center, dtype=float)
    quadratic = np.zeros(offset.shape[:-1])
    for direction, semi in zip(np.asarray(axes, dtype=float), np.maximum(np.asarray(semiAxes, dtype=float), 1e-3)):
        quadratic = quadratic + (offset @ direction / semi) ** 2
    return quadratic <= 1.0


def maskOf(referenceNode, frame):
    """(axes, semi-axes) of the Processing ellipsoid when the reoriented volumes were masked, else None."""
    text = referenceNode.GetAttribute("MPI.Mask") or ""
    match = re.search(r"\)\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", text)
    if not match:
        return None
    return np.array([frame["u"], frame["w"], frame["s"]]), np.array([float(v) for v in match.groups()])


def resliceLike(array, arrayToIjk, referenceNode, mask=None, center=None):
    """A gate resampled on the grid of one reoriented volume of Processing (same pixel size, slice
    thickness made of thin sections as there, same interpolation, same mask). Returns [k, j, i]."""
    dims = referenceNode.GetImageData().GetDimensions()
    ijkToRas = vtk.vtkMatrix4x4()
    referenceNode.GetIJKToRASMatrix(ijkToRas)
    m = slicer.util.arrayFromVTKMatrix(ijkToRas)
    localToWorld = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(worldToLocalMatrix(referenceNode), localToWorld)
    m = slicer.util.arrayFromVTKMatrix(localToWorld) @ m                     # reference IJK -> world
    spacing = float(np.linalg.norm(m[:3, 0]))
    kColumn = m[:3, 2].copy()
    sections = max(1, int(math.ceil(float(np.linalg.norm(kColumn)) / max(spacing, 1e-6) - 1e-6)))
    section = m.copy()
    section[:3, 2] = kColumn / sections
    section[:3, 3] = m[:3, 3] - (sections - 1) / (2.0 * sections) * kColumn
    axes = vtk.vtkMatrix4x4()
    axes.DeepCopy((arrayToIjk @ section).ravel().tolist())
    from MyocardialPerfusionSPECTLib.resampling import arrayToImageData
    interpolation = referenceNode.GetAttribute("MPI.Interpolation") or DEFAULT_INTERPOLATION
    values = reslice(arrayToImageData(array), axes, (dims[0], dims[1], dims[2] * sections), interpolation)
    if mask is not None and center is not None:
        values *= ellipsoidInsideGrid(values.shape, section, center, mask[0], mask[1])
    return values.reshape(dims[2], sections, dims[1], dims[0]).mean(axis=1).astype(np.float32)


def frameStatistics(result, frame, array, matrix, voxelVolumeMm3, mask=None):
    """Myocardium counts (the gate's counts inside the frame's myocardium, inside the Processing ellipsoid when
    the volumes were masked), cavity and myocardium ml."""
    labels, labelMatrix = _labelGrid(result, frame, LABEL_GRID_MM)
    m = slicer.util.arrayFromVTKMatrix(labelMatrix)
    index = np.argwhere(labels == wallLib.LABEL_MYOCARDIUM)[:, ::-1].astype(float)   # (i, j, k)
    points = index @ m[:3, :3].T + m[:3, 3]
    voxel = LABEL_GRID_MM ** 3
    sample = sampler(array, matrix, mask, np.asarray(frame["center"], dtype=float))
    counts = float(sample(points).sum()) * voxel / max(voxelVolumeMm3, 1e-9) if len(points) else 0.0
    return {"myocardiumCounts": counts,
            "cavityMl": float(np.count_nonzero(labels == wallLib.LABEL_CAVITY)) * voxel / 1000.0,
            "myocardiumMl": float(len(points)) * voxel / 1000.0}


def _labelGrid(result, frame, spacing):
    import MPIMyocardium
    return MPIMyocardium.labelGrid(result, frame, spacing)


class GatedStudy:
    """Everything Gated QC shows for one study: the gate frames resliced to SA / HLA / VLA, the
    per-frame wall and the counts. Made by build()."""

    def __init__(self, study, sequence):
        self.study = study
        self.sequence = sequence
        self.labels = []          # frame index values
        self.slices = {}          # orientation -> [frame arrays [k, j, i]]
        self.results = []         # per-frame wall (fitGate)
        self.stats = []           # per-frame frameStatistics
        self.imageCounts = []
        self.peaks = []           # per frame: hottest voxel near the LV (window reference)
        self.reference = {}       # orientation -> reoriented volume of Processing (geometry)
        self.attributes = {}      # SPECTRecon.* of the first gate
        self.errors = []

    def build(self, staticResult, frame, outputs, params=None):
        gates = gateNodes(self.sequence)
        if not gates:
            raise ValueError("The gated sequence has no frames.")
        self.labels = [label for label, _ in gates]
        self.reference = dict(outputs)
        self.attributes = gatingLib.attributes(gates[0][1])
        mask = maskOf(outputs["SA"], frame)
        center = np.asarray(frame["center"], dtype=float)
        reach = wallLib.wallReach(staticResult, center) + 5.0
        self.slices = {name: [] for name in ORIENTATIONS}
        frames = []
        for label, node in gates:
            array = slicer.util.arrayFromVolume(node).astype(np.float32)
            matrix = worldToIjk(node, self.study)
            spacing = node.GetSpacing()
            try:
                # on the gate masked like the static volume the wall was detected on (liver and gut out)
                result = wallLib.fitGate(sampler(array, matrix, mask, center), staticResult, params)
            except Exception as e:
                logging.warning(f"MPI: gate {label} could not be segmented:\n" + traceback.format_exc())
                self.errors.append(f"frame {label}: {e}")
                result = None
            self.results.append(result)
            frames.append((label, node, matrix, spacing[0] * spacing[1] * spacing[2]))   # not the array: memory
            self.imageCounts.append(float(array.sum(dtype=np.float64)))
            for name in ORIENTATIONS:
                self.slices[name].append(resliceLike(array, matrix, outputs[name], mask, center))
            self.peaks.append(self._peakNear(self.slices["SA"][-1], outputs["SA"], center, reach))
        for index, (label, node, matrix, voxel) in enumerate(frames):
            array = slicer.util.arrayFromVolume(node).astype(np.float32)   # again, one frame at a time
            stats = {"myocardiumCounts": None, "cavityMl": None, "myocardiumMl": None}
            if self.results[index] is not None:
                try:
                    stats = frameStatistics(self.results[index], frame, array, matrix, voxel, mask)
                except Exception as e:
                    logging.warning(f"MPI: gate {label} statistics failed:\n" + traceback.format_exc())
                    self.errors.append(f"frame {label}: {e}")
            self.stats.append(stats)
        return self

    @staticmethod
    def _peakNear(values, node, center, reach):
        ijkToRas = vtk.vtkMatrix4x4()
        node.GetIJKToRASMatrix(ijkToRas)
        m = slicer.util.arrayFromVTKMatrix(ijkToRas)
        nk, nj, ni = values.shape
        inside = ellipsoidInsideGrid(values.shape, m, center, np.eye(3), [reach] * 3)
        selected = values[inside]
        return float(selected.max()) if selected.size else float(values.max())

    def myocardiumCounts(self):
        return [s["myocardiumCounts"] for s in self.stats]

    def cavityMl(self):
        return [s["cavityMl"] for s in self.stats]

    def projectionCounts(self):
        values = gatingLib.jsonList(self.attributes, "GatePhotopeakCounts")
        return values if values and len(values) == len(self.labels) else None

    def save(self):
        """Per-frame results on the gated sequence, for Gated Review and for the record."""
        sequence = self.sequence
        sequence.SetAttribute("MPI.GateLabels", json.dumps(self.labels))
        sequence.SetAttribute("MPI.GateImageCounts", json.dumps([round(v) for v in self.imageCounts]))
        sequence.SetAttribute("MPI.GateMyocardiumCounts",
                              json.dumps([None if v is None else round(v) for v in self.myocardiumCounts()]))
        sequence.SetAttribute("MPI.GateCavityMl", json.dumps([None if v is None else round(v, 1) for v in self.cavityMl()]))
        sequence.SetAttribute("MPI.GateMyocardiumMl",
                              json.dumps([None if s["myocardiumMl"] is None else round(s["myocardiumMl"], 1) for s in self.stats]))
        sequence.SetAttribute("MPI.GatedQCOf", self.study.GetID())


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class MPIGatedQCWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.slot = STRESS
        self.data = {}            # slot -> (key, GatedStudy)
        self.tileCache = BoundedCache(TILE_CACHE_BYTES)        # (slot, orientation, frame, slice, ...) -> QImage
        self.contourCache = BoundedCache(CONTOUR_CACHE_BYTES)  # (frame, slice, size, ...) -> packed masks

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.studySelector = StudyPairSelector(self.layout)
        self.studyLabel = narrowLabel(qt.QLabel())
        self.layout.addWidget(self.studyLabel)

        section = ctk.ctkCollapsibleButton()
        section.text = "Frames"
        self.layout.addWidget(section)
        form = qt.QFormLayout(section)
        self.colormapCombo = qt.QComboBox()
        self.colormapCombo.addItems([label for label, _ in COLORMAPS])
        form.addRow("Color map:", self.colormapCombo)
        self.windowRange = ctk.ctkRangeWidget()
        self.windowRange.minimum, self.windowRange.maximum = 0.0, 150.0
        self.windowRange.decimals = 0
        self.windowRange.singleStep = 1.0
        self.windowRange.suffix = " %"
        self.windowRange.minimumValue, self.windowRange.maximumValue = 0.0, 100.0
        self.windowRange.toolTip = "Color window in % of the hottest voxel near the LV"
        form.addRow("Window:", self.windowRange)
        self.normalizationCombo = qt.QComboBox()
        for text, value, _, _ in FRAME_SCALES:
            self.normalizationCombo.addItem(text, value)
        self.normalizationCombo.toolTip = ("All frames together: one color scale for every frame, so count changes "
                                           "between frames show. Each frame: every frame scaled to its own maximum")
        form.addRow("Scale:", self.normalizationCombo)
        self.contoursButton = qt.QPushButton("Myocardium contours")
        self.contoursButton.checkable = True
        self.contoursButton.checked = True
        self.contoursButton.toolTip = "Epicardium (green) and endocardium (blue) segmented in every frame"
        form.addRow("Show:", self.contoursButton)
        self.sliceSpins = {}
        for name in ORIENTATIONS:
            spin = qt.QSpinBox()
            spin.setRange(1, 1)
            spin.toolTip = f"{ORIENTATION_TITLES[name]} slice shown for every frame (mouse wheel over the row too)"
            self.sliceSpins[name] = spin
            form.addRow(f"{name} slice:", spin)

        advanced = ctk.ctkCollapsibleButton()
        advanced.text = "Frame segmentation"
        advanced.collapsed = True
        self.layout.addWidget(advanced)
        advancedForm = qt.QFormLayout(advanced)
        self.segmentationCombo = qt.QComboBox()
        for text, value in wallLib.GATE_SEGMENTATIONS:
            self.segmentationCombo.addItem(text, value)
        self.segmentationCombo.setCurrentIndex(
            max(0, self.segmentationCombo.findData(wallLib.GATE_DEFAULTS["segmentation"])))
        self.segmentationCombo.toolTip = (
            "How the wall of every frame is found. Track the static wall (default): the mid-wall of a frame is "
            "searched near the static wall of Myocardium and the fit is of its displacement from it, so the frame "
            "keeps the static wall's shape where it does not show its own. Each frame from scratch: every frame "
            "is detected on its own on the same rays and the fit is of its wall itself; the motion and valve "
            "plane limits do not apply")
        self.segmentationCombo.connect("currentIndexChanged(int)", lambda _: self.updateSegmentationControls())
        advancedForm.addRow("Frames:", self.segmentationCombo)
        self.wallShiftSpin = qt.QDoubleSpinBox()
        self.wallShiftSpin.setRange(2.0, 40.0)
        self.wallShiftSpin.setValue(wallLib.GATE_DEFAULTS["maxWallShift"])
        self.wallShiftSpin.setSuffix(" mm")
        self.wallShiftSpin.toolTip = "How far from the static mid-wall the mid-wall of a frame is searched"
        advancedForm.addRow("Wall motion limit:", self.wallShiftSpin)
        self.valveShiftSpin = qt.QDoubleSpinBox()
        self.valveShiftSpin.setRange(0.0, 40.0)
        self.valveShiftSpin.setValue(wallLib.GATE_DEFAULTS["maxValveShift"])
        self.valveShiftSpin.setSuffix(" mm")
        self.valveShiftSpin.toolTip = "How far the valve plane of a frame may move from the static one (base descent)"
        advancedForm.addRow("Valve plane limit:", self.valveShiftSpin)
        self.fitCombo = qt.QComboBox()
        for text, value in wallLib.GATE_FITS:
            self.fitCombo.addItem(text, value)
        self.fitCombo.setCurrentIndex(max(0, self.fitCombo.findData(wallLib.GATE_DEFAULTS["fit"])))
        self.fitCombo.toolTip = (
            "Surface fit through the rays whose wall a frame shows (above its blood pool), which draws the rays "
            "it does not show (a defect) and smooths the others; the wall thickness is fitted the same way. "
            "B-spline (default): a smooth spline surface with a bending penalty, a defect continues the curve of "
            "the walls around it. Deformable mesh: every ray tied to its neighbours around and along the axis, a "
            "defect is bridged straighter. Legacy: the change between neighbouring rays is limited (Brigger "
            "1999), ring by ring and along the axis. Their stiffness is in Advanced parameters")
        advancedForm.addRow("Fit:", self.fitCombo)
        self.fitPullSlider = ctk.ctkSliderWidget()
        self.fitPullSlider.minimum, self.fitPullSlider.maximum = 0.0, 1.0
        self.fitPullSlider.singleStep = 0.05
        self.fitPullSlider.decimals = 2
        self.fitPullSlider.value = wallLib.GATE_DEFAULTS["fitPull"]
        self.fitPullSlider.toolTip = ("How far a traced ray is pulled from its own count peak to the fit: 1 the fit, "
                                      "0 its own peak (the rays without a wall always take the fit)")
        advancedForm.addRow("Smoothing (pull to the fit):", self.fitPullSlider)
        self.segmentButton = styleButton(qt.QPushButton("Segment frames again"), "create")
        advancedForm.addRow(self.segmentButton)

        self.layoutButtons = {}
        self.layoutGroup = qt.QButtonGroup()
        self.layoutGroup.exclusive = True
        layoutBox = qt.QGroupBox("Layout")
        grid = qt.QVBoxLayout(layoutBox)
        for slot in (STRESS, REST):
            button = qt.QPushButton(SLOT_TITLES[slot])
            button.checkable = True
            self.layoutGroup.addButton(button)
            grid.addWidget(button)
            self.layoutButtons[slot] = button
        self.layout.addWidget(layoutBox)
        self.nextButton = styleButton(qt.QPushButton("Gated Review ›"), "primary")
        self.layout.addWidget(self.nextButton)
        self.parameterPanel = parametersLib.ParameterPanel(
            "gatedqc", self.layout, onChanged=self.onParameterChanged,
            hint=("Frame segmentation and fit parameters apply when the frames are segmented again (Segment frames again)."))
        self.parameterPanel.bind("colormap", self.colormapCombo)
        self.parameterPanel.bind("window", self.windowRange)
        self.parameterPanel.bind("scale", self.normalizationCombo)
        self.parameterPanel.bind("contours", self.contoursButton)
        self.parameterPanel.bind("wallShift", self.wallShiftSpin)
        self.parameterPanel.bind("valveShift", self.valveShiftSpin)
        self.parameterPanel.bind("gateFit", self.fitCombo)
        self.parameterPanel.bind("gateFitPull", self.fitPullSlider)
        self.parameterPanel.bind("gateSegmentation", self.segmentationCombo)
        self.updateSegmentationControls()
        self.layout.addStretch(1)

        self.resizeTimer = qt.QTimer()
        self.resizeTimer.setSingleShot(True)
        self.resizeTimer.setInterval(150)
        self.resizeTimer.connect("timeout()", self.refresh)
        self.colormapCombo.connect("currentIndexChanged(int)", lambda _: self.refresh())
        self.windowRange.connect("valuesChanged(double,double)", lambda a, b: self.refresh())
        self.normalizationCombo.connect("currentIndexChanged(int)", lambda _: self.refresh())
        self.contoursButton.connect("toggled(bool)", lambda _: self.refresh())
        for spin in self.sliceSpins.values():
            spin.connect("valueChanged(int)", lambda _: self.refresh())
        self.segmentButton.connect("clicked(bool)", lambda: self.show(self.slot, rebuild=True))
        for slot, button in self.layoutButtons.items():
            button.connect("clicked(bool)", lambda _, s=slot: self.show(s))
        self.nextButton.connect("clicked(bool)", lambda: session.selectStep("gated"))
        self.parameterPanel.loadSaved()   # saved settings of this step, as if set by hand
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneClosed)
        session.addListener(self.onSessionChanged)
        _installToolbar()
        self.updateState()

    # --- workflow toolbar (second row) -----------------------------------------------------------------
    def buildToolbar(self, bar):
        self.toolbarBar = bar
        bar.mirror(self.segmentButton, "Segment again", "Segment the frames again with the current settings", "create")
        bar.separator()
        self.tbMax = bar.spin("Max", 10, 150, 5, " %", "Top of the color window (% of the hottest voxel near the LV)",
                              self.onToolbarMax)
        self.tbScale = bar.choice([(text, value, tip) for _, value, text, tip in FRAME_SCALES], self.onToolbarScale)
        bar.mirrorCombo(self.colormapCombo, toolTip="Color map of the frames")
        bar.mirror(self.contoursButton, "Contours")
        bar.separator()
        for slot, button in self.layoutButtons.items():
            bar.mirror(button, SLOT_TITLES[slot], f"Gated frames of the {SLOT_TITLES[slot].lower()} study")
        bar.separator()
        bar.mirror(self.nextButton, "Gated Review ›", kind="primary")
        if not getattr(self, "_toolbarFollows", False):   # the toolbar follows this panel's controls
            self._toolbarFollows = True
            self.windowRange.connect("valuesChanged(double,double)", lambda a, b: self.syncToolbar())
            self.normalizationCombo.connect("currentIndexChanged(int)", lambda _: self.syncToolbar())

    def onToolbarMax(self, value):
        self.windowRange.maximumValue = max(float(value), self.windowRange.minimumValue + 1.0)
        self.syncToolbar()

    def onToolbarScale(self, value):
        index = self.normalizationCombo.findData(value)
        if index >= 0:
            self.normalizationCombo.setCurrentIndex(index)
        self.syncToolbar()   # a click on the checked button changed nothing: check it again

    def syncToolbar(self):
        bar = getattr(self, "toolbarBar", None)
        if bar is None:
            return
        try:
            bar.syncMirrors()
            bar.setValue(self.tbMax, self.windowRange.maximumValue)
            bar.setChoice(self.tbScale, self.normalizationCombo.itemData(self.normalizationCombo.currentIndex))
        except Exception:
            logging.debug("MPI: could not update the toolbar:\n" + traceback.format_exc())

    def enter(self):
        self.studySelector.refresh()
        self.updateState()
        if not self.available(self.slot)[0]:
            self.slot = next((s for s in (STRESS, REST) if self.available(s)[0]), self.slot)
        if self.available(self.slot)[0]:
            self.show(self.slot)

    def cleanup(self):
        self.resizeTimer.stop()
        session.removeListener(self.onSessionChanged)
        self.studySelector.cleanup()
        panel = existingPanel(MOSAIC_STATE)
        if panel is not None:
            panel.onScroll = None
            panel.onResized = None
        self.removeObservers()

    def onSceneClosed(self, caller=None, event=None):
        self.data = {}
        self.tileCache.clear()
        self.contourCache.clear()
        for attribute in (MOSAIC_STATE, CURVE_STATE, INFO_STATE):
            panel = existingPanel(attribute)
            if panel is not None:
                panel.clear()
        self.updateState()

    def onSessionChanged(self):
        self.updateState()
        if self.parent.isVisible() and slicer.app.layoutManager().layout == GATED_QC_LAYOUT_ID \
                and self.available(self.slot)[0]:
            self.show(self.slot)

    # --- studies ---------------------------------------------------------------
    @staticmethod
    def studyOf(slot):
        return session.stressRestStudies()[slot]

    def available(self, slot):
        """(ready, reason) for showing the gated study of a slot."""
        study = self.studyOf(slot)
        if study is None:
            return False, "no study selected"
        if gatedSequenceOf(study) is None:
            return False, "no gated sequence paired (see Data roles)"
        if len(session.outputsForStudy(study)) < len(ORIENTATIONS):
            return False, "not reoriented yet (Processing)"
        if session.myocardiumForStudy(study) is None:
            return False, "myocardium not detected yet (Myocardium)"
        return True, ""

    def updateState(self):
        lines = []
        for slot, title in ((STRESS, "Study 1"), (REST, "Study 2")):
            study = self.studyOf(slot)
            ready, reason = self.available(slot)
            if study is None:
                continue
            sequence = gatedSequenceOf(study)
            text = f"{title}: <b>{breakable(study.GetName())}</b>"
            if sequence is not None:
                text += (f", gated <b>{breakable(sequence.GetName())}</b> "
                         f"({sequence.GetNumberOfDataNodes()} frames)")
            if not ready:
                text += f" <span style='color:#d08030'>({reason})</span>"
            lines.append(text)
            self.layoutButtons[slot].enabled = ready
        if not lines:
            lines = ["No study selected. Pick Study 1 / Study 2 above or set the roles in Data roles."]
        self.studyLabel.text = "<br>".join(lines)
        if self.slot in self.layoutButtons:
            self.layoutButtons[self.slot].checked = True
        self.syncToolbar()

    def segmentation(self):
        return (self.segmentationCombo.itemData(self.segmentationCombo.currentIndex)
                or wallLib.GATE_DEFAULTS["segmentation"])

    def updateSegmentationControls(self):
        """The motion and valve limits belong to the tracking of the static wall."""
        static = self.segmentation() == "static"
        for widget in (self.wallShiftSpin, self.valveShiftSpin):
            widget.enabled = static

    def params(self):
        return {"maxWallShift": float(self.wallShiftSpin.value), "maxValveShift": float(self.valveShiftSpin.value),
                "fit": self.fitCombo.itemData(self.fitCombo.currentIndex) or wallLib.GATE_DEFAULTS["fit"],
                "fitPull": float(self.fitPullSlider.value), "segmentation": self.segmentation()}

    def gatedStudy(self, slot, rebuild=False):
        """GatedStudy of a slot, built once per study, gated sequence, static wall and settings."""
        import MPIMyocardium
        study = self.studyOf(slot)
        sequence = gatedSequenceOf(study)
        outputs = session.outputsForStudy(study)
        myocardiumWidget = slicer.modules.mpimyocardium.widgetRepresentation().self()
        staticResult = myocardiumWidget.resultForStudy(study)
        frame = MPIMyocardium.cardiacFrame(study)
        if staticResult is None or frame is None:
            raise ValueError("Detect the myocardium of this study first (Myocardium step).")
        key = (study.GetID(), sequence.GetID(), sequence.GetNumberOfDataNodes(), id(staticResult),
               tuple(sorted((k, n.GetID(), n.GetAttribute("MPI.Mask") or "") for k, n in outputs.items())),
               tuple(sorted(self.params().items())))
        cached = self.data.get(slot)
        if cached and cached[0] == key and not rebuild:
            return cached[1]
        data = GatedStudy(study, sequence).build(staticResult, frame, outputs, self.params())
        data.save()
        self.data[slot] = (key, data)
        self.tileCache.clear()
        self.contourCache.clear()
        self.resetSlices(data, frame)
        return data

    def resetSlices(self, data, frame):
        """Every row starts at the slice through the LV center."""
        center = np.append(np.asarray(frame["center"], dtype=float), 1.0)
        for name in ORIENTATIONS:
            node = data.reference[name]
            n = data.slices[name][0].shape[0]
            k = int(round((worldToIjk(node, node) @ center)[2]))
            spin = self.sliceSpins[name]
            wasBlocked = spin.blockSignals(True)
            spin.setRange(1, n)
            spin.setValue(min(max(k, 0), n - 1) + 1)
            spin.blockSignals(wasBlocked)

    # --- display ----------------------------------------------------------------
    def show(self, slot, rebuild=False):
        ready, reason = self.available(slot)
        if not ready:
            slicer.util.warningDisplay(f"The gated study cannot be shown: {reason}.", windowTitle="Gated QC")
            return
        self.slot = slot
        self.layoutButtons[slot].checked = True
        with slicer.util.tryWithErrorDisplay("Could not prepare the gated frames.", waitCursor=True):
            data = self.gatedStudy(slot, rebuild)
            panel = gatedMosaicPanel()
            panel.onScroll = self.onScroll
            panel.onResized = self.resizeTimer.start
            panel.gridRows, panel.gridColumns = 1, max(1, len(data.labels))
            panel.setRowKeys([(name, ORIENTATION_TITLES[name]) for name in ORIENTATIONS])
            countCurvePanel()
            gatingInfoPanel()
            registerLayout(GATED_QC_LAYOUT_ID, gatedQcLayoutXml())
            slicer.app.layoutManager().setLayout(GATED_QC_LAYOUT_ID)
            slicer.app.processEvents()
            self.refresh()
            self.refreshCurves(data)

    def onScroll(self, key, direction):
        spin = self.sliceSpins.get(key)
        if spin:
            spin.setValue(spin.value + direction)

    def current(self):
        cached = self.data.get(self.slot)
        return cached[1] if cached else None

    def windows(self, data):
        """(low, top) counts of every frame's color window."""
        if self.normalizationCombo.itemData(self.normalizationCombo.currentIndex) == "frame":
            references = list(data.peaks)
        else:
            references = [max(data.peaks)] * len(data.peaks)
        low, high = self.windowRange.minimumValue / 100.0, self.windowRange.maximumValue / 100.0
        return [(r * low, max(r * high, r * low + 1e-9)) for r in references]

    def onParameterChanged(self, key):
        """An Advanced parameter changed (None: all were restored): new cache budgets, redraw."""
        self.tileCache.budget = int(TILE_CACHE_BYTES)
        self.contourCache.budget = int(CONTOUR_CACHE_BYTES)
        self.tileCache.clear()      # tiles and contours drawn with the old colors / sizes
        self.contourCache.clear()
        self.refresh()

    def refresh(self):
        data = self.current()
        panel = existingPanel(MOSAIC_STATE)
        if data is None or panel is None or slicer.app.layoutManager().layout != GATED_QC_LAYOUT_ID:
            return
        colorTable = colorTableArray(self.colormapCombo.currentText)
        windows = self.windows(data)
        panel.gridRows, panel.gridColumns = 1, max(1, len(data.labels))
        for name in ORIENTATIONS:
            size = panel.tileSize(name)
            k = self.sliceSpins[name].value - 1
            count = data.slices[name][0].shape[0]
            tiles = []
            for index, label in enumerate(data.labels):
                image = self.tile(data, name, index, k, size, windows[index], colorTable, panel.pixelRatio()) if size else None
                tiles.append((image, label))
            panel.setRow(name, tiles, f"{ORIENTATION_TITLES[name]}   —   slice {k + 1} of {count}, all frames")
        panel.setLegend(colorTable, self.windowRange.minimumValue, self.windowRange.maximumValue)

    def tile(self, data, name, index, k, size, window, colorTable, ratio):
        contours = bool(self.contoursButton.checked) and data.results[index] is not None
        key = (self.slot, id(data), name, index, k, int(size), self.colormapCombo.currentText,
               round(window[0], 6), round(window[1], 6), contours)
        image = self.tileCache.get(key)
        if image is not None:
            return image
        values = data.slices[name][index][k]
        interpolation = data.reference[name].GetAttribute("MPI.Interpolation") or DEFAULT_INTERPOLATION
        zoomed = zoomInPlane(np.asarray(values[::-1, :]), size, size, interpolation)[0]   # top row = highest j
        rgb = colorize(zoomed, window[1], colorTable, window[0])
        if contours:
            epi, endo = self.contourMasks(data, name, index, k, size, ratio)
            rgb[epi] = CONTOUR_COLORS["epi"]
            rgb[endo] = CONTOUR_COLORS["endo"]
        image = rgbToQImage(np.ascontiguousarray(rgb))
        self.tileCache[key] = image
        return image

    def contourMasks(self, data, name, index, k, size, ratio, crop=None):
        """(epicardium, endocardium) contour masks [size, size] of one frame on one slice, classified
        at the tile's own pixels (as the Perfusion contours); crop: the window of the slices shown
        (display.fovCrop, None: the whole slice)."""
        key = (id(data), name, index, k, int(size), round(float(ratio), 2), None if crop is None else tuple(crop))
        cached = self.contourCache.get(key)
        if cached is not None:
            return tuple(unpackMask(packed) for packed in cached)
        result = data.results[index]
        node = data.reference[name]
        n = int(min(size, CONTOUR_MAX_SIZE))
        ii, jj = tilePixelsToIjk(n, crop, data.slices[name][index].shape)
        ijkToRas = vtk.vtkMatrix4x4()
        node.GetIJKToRASMatrix(ijkToRas)
        localToWorld = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(worldToLocalMatrix(node), localToWorld)
        matrix = slicer.util.arrayFromVTKMatrix(localToWorld) @ slicer.util.arrayFromVTKMatrix(ijkToRas)
        points = np.column_stack([ii.ravel(), jj.ravel(), np.full(ii.size, float(k))]) @ matrix[:3, :3].T + matrix[:3, 3]
        grid = result["grid"]
        reach = wallLib.wallReach(result) + 3.0
        near = np.einsum("ij,ij->i", points - grid.center, points - grid.center) <= reach * reach
        labels = np.zeros(ii.size, dtype=np.uint8)
        if np.any(near):
            labels[near] = wallLib.classify(result, points[near])
        labels = labels.reshape(n, n)
        if n != size:
            pick = np.minimum((np.arange(size) * n) // size, n - 1)
            labels = labels[pick][:, pick]
        wall = labels == wallLib.LABEL_MYOCARDIUM
        masks = []
        for other in (0, wallLib.LABEL_CAVITY):
            target = labels == other
            neighbour = np.zeros_like(wall)
            neighbour[1:, :] |= target[:-1, :]
            neighbour[:-1, :] |= target[1:, :]
            neighbour[:, 1:] |= target[:, :-1]
            neighbour[:, :-1] |= target[:, 1:]
            edge = wall & neighbour
            if ratio >= 1.5:
                edge[1:, :] |= edge[:-1, :].copy()
                edge[:, 1:] |= edge[:, :-1].copy()
            masks.append(edge)
        self.contourCache[key] = tuple(packMask(mask) for mask in masks)
        return tuple(masks)

    # --- counts and gating information ------------------------------------------------
    def refreshCurves(self, data):
        series = []
        projections = data.projectionCounts()
        sources = [("Projections", projections, "projections"), ("Image", data.imageCounts, "image"),
                   ("Myocardium", data.myocardiumCounts(), "myocardium")]
        for name, values, colorKey in sources:
            if values is None or not any(v is not None for v in values):
                continue
            known = [v for v in values if v is not None]
            series.append((name, values, SERIES_COLORS[colorKey], f"mean {gatingLib.formatCounts(sum(known) / len(known))} counts"))
        countCurvePanel().setCurves(data.labels, series, "Counts per frame (% of the mean)",
                                    "No counts.", mode="percent", band=gatingLib.COUNT_BAND_PERCENT)
        chart = None
        perView = gatingLib.rrPerView(data.attributes)
        if perView is not None:
            views, nominal, low, high = perView
            chart = {"labels": views, "titleText": "R-R interval per projection view",
                     "series": [("R-R", nominal, (255, 200, 80), "mean of the accepted beats"),
                                ("Accepted window", low, (150, 150, 150), "", True),
                                ("", high, (150, 150, 150), "", True)],
                     "mode": "absolute", "unit": "ms", "xTitle": "projection view"}
        gatingInfoPanel().setInfo(self.infoHtml(data, projections), chart)

    def infoHtml(self, data, projections):
        values = data.attributes
        css = "td { padding: 1px 6px 1px 0px; } th { text-align: left; padding: 1px 6px 1px 0px; color: #a0a8b0; }"
        parts = [f"<style>{css}</style>",
                 f"<p><b>{data.study.GetName()}</b><br>{data.sequence.GetName()}, {len(data.labels)} frames</p>"]
        findings = gatingLib.qcFindings(data.imageCounts, data.myocardiumCounts(), projections, values)
        if findings:
            parts.append("<p>" + "<br>".join(
                f"<span style='color:{'#f0a040' if level == 'warning' else '#7cc47c'}'>"
                f"{'⚠' if level == 'warning' else '✓'} {text}</span>" for level, text in findings) + "</p>")
        rows = gatingLib.gatingRows(values)
        if rows:
            parts.append("<table>" + "".join(f"<tr><th>{label}</th><td>{text}</td></tr>" for label, text in rows) + "</table>")
        elif not values.get("GatingInfoSource"):
            parts.append("<p style='color:#a0a8b0'>No ECG gating information stored on the frames (reconstruct with "
                         "the DICOM SPECT Recon extension to keep the R-R window and beat counts).</p>")
        cavity = data.cavityMl()
        header = "<tr><th nowrap>Frame</th>" + ("<th nowrap>Projections</th>" if projections else "") + \
                 "<th nowrap>Image</th><th nowrap>Myocardium</th><th nowrap>Cavity ml</th></tr>"
        deviations = {name: gatingLib.deviations(series) for name, series in
                      (("p", projections or []), ("i", data.imageCounts), ("m", data.myocardiumCounts()))}

        def cell(series, key, index):
            if not series or series[index] is None:
                return "<td>-</td>"
            d = deviations[key][index]
            text = gatingLib.formatCounts(series[index]) + ("" if d is None else f" ({d:+.1f}%)")
            flagged = d is not None and abs(d) > gatingLib.COUNT_BAND_PERCENT
            return f"<td nowrap style='color:#f0a040'>{text}</td>" if flagged else f"<td nowrap>{text}</td>"
        body = ""
        for index, label in enumerate(data.labels):
            body += (f"<tr><td>{label}</td>" + (cell(projections, "p", index) if projections else "")
                     + cell(data.imageCounts, "i", index) + cell(data.myocardiumCounts(), "m", index)
                     + f"<td>{'-' if cavity[index] is None else f'{cavity[index]:.0f}'}</td></tr>")
        parts.append(f"<p><table>{header}{body}</table></p>")
        known = [(v, i) for i, v in enumerate(cavity) if v is not None]
        if len(known) >= 2:
            (edv, ed), (esv, es) = max(known), min(known)
            ef = 100.0 * (edv - esv) / edv if edv > 0 else float("nan")
            parts.append(f"<p style='color:#a0a8b0'>LV cavity largest {edv:.0f} ml (frame {data.labels[ed]}), smallest "
                         f"{esv:.0f} ml (frame {data.labels[es]}): preliminary EF {ef:.0f} %. Volumes and EF are "
                         "for checking the segmentation; the gated analysis is in Gated Review.</p>")
        if data.errors:
            parts.append("<p style='color:#f06060'>Not segmented: " + "; ".join(data.errors) + "</p>")
        return "".join(parts)


class MPIGatedQCTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_GatingText()
        self.test_GateFit()
        self.test_ScarFitted()
        self.test_FitSurface()
        self.test_PoolFloor()
        self.test_SlabThickness()
        self.test_MaskedSampler()

    def test_GatingText(self):
        values = {"GatingTimeSlots": "8", "GatingIntervalsAcquired": "1332", "GatingIntervalsRejected": "0",
                  "GatingRejectedPercent": "0.0", "GatingNominalIntervalMs": "983", "GatingLowRRMs": "774",
                  "GatingHighRRMs": "1210", "GatingHeartRateBpm": "61", "GatingItemsAreViews": "1",
                  "GatingDataItems": "64", "GatingPerItemNominalIntervalMs": json.dumps([900] * 32 + [1000] * 32)}
        labels = [label for label, _ in gatingLib.gatingRows(values)]
        self.assertIn("Accepted R-R window", labels)
        findings = gatingLib.qcFindings([100, 101, 99, 100, 100, 101, 99, 80], values=values)
        self.assertTrue(any(level == "warning" and "frame 8" in text for level, text in findings))
        self.delayDisplay("Gating text test passed")

    def test_MaskedSampler(self):
        """A gate sampled with the Processing mask reads 0 outside the ellipsoid and its counts inside, as the
        masked static volume does; the mask text of the reoriented volumes gives its semi-axes."""
        array = np.ones((20, 20, 20), dtype=np.float32)
        matrix = np.eye(4)
        center = np.array([10.0, 10.0, 10.0])
        frame = {"u": np.array([0.0, 0.0, 1.0]), "w": np.array([1.0, 0.0, 0.0]), "s": np.array([0.0, 1.0, 0.0]),
                 "center": center}

        class Reference:
            def GetAttribute(self, name):
                return "ellipsoid semi-axes (long, anterior-inferior, septal-lateral) 8, 4, 4 mm"
        mask = maskOf(Reference(), frame)
        sample = sampler(array, matrix, mask, center)
        points = center + np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 6.0], [6.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        self.assertTrue(np.allclose(sample(points), [1.0, 1.0, 0.0, 1.0]))
        self.assertTrue(np.allclose(sampler(array, matrix)(points), 1.0))
        self.delayDisplay("Masked gate sampler test passed")

    def test_GateFit(self):
        """A frame of a thinner, smaller LV is found inside the static wall."""
        import MPIMyocardium
        n, spacing = 64, 3.0
        center = np.array([n * spacing / 2.0] * 3)
        u, w = unit([-0.6, 0.6, -0.4]), unit([0.0, 0.5, 0.75])

        def ventricle(radius):
            ii = np.arange(n) * spacing
            p = np.stack(np.meshgrid(ii, ii, ii, indexing="ij"), -1) - center
            z = p @ u
            rho = np.linalg.norm(p - z[..., None] * u, axis=-1)
            r = np.where(z > 0, np.linalg.norm(p, axis=-1), rho)                # cylinder, then an apical cap
            return (100.0 * ((np.abs(r - radius) < 5) & (z > -30))).astype(np.float32)

        def sampleOf(array):
            return lambda pts: sampleTrilinear(array, np.asarray(pts) / spacing)
        static = wallLib.fitWall(sampleOf(ventricle(25.0)), center, u, w, 55.0)
        middle = static["grid"].nCyl - 4
        for mode in ("static", "own"):
            for fit in ("bspline", "mesh", "legacy"):
                params = {"segmentation": mode, "fit": fit}
                ed = wallLib.fitGate(sampleOf(ventricle(25.0)), static, params)
                gate = wallLib.fitGate(sampleOf(ventricle(21.0)), static, params)
                self.assertEqual(gate["grid"], static["grid"])
                self.assertAlmostEqual(float(np.median(ed["mid"][middle])), 25.0, delta=1.5, msg=params)
                self.assertAlmostEqual(float(np.median(gate["mid"][middle])), 21.0, delta=1.5, msg=params)
                # each frame from its own profiles: both walls are 10 mm slabs, so both read about the same
                walls = [ed, None, gate]
                self.assertAlmostEqual(float(np.median(walls[2]["thickness"][middle])),
                                       float(np.median(walls[0]["thickness"][middle])), delta=2.0, msg=params)
                self.assertLess(float(np.median(walls[2]["endo"][middle])), float(np.median(walls[0]["endo"][middle])) - 2.0)
        # from scratch the motion limit does not apply
        scratch = wallLib.fitGate(sampleOf(ventricle(21.0)), static, {"segmentation": "own", "maxWallShift": 2.0})
        self.assertAlmostEqual(float(np.median(scratch["mid"][middle])), 21.0, delta=1.5)
        self.delayDisplay("Gate fit test passed")

    def test_ScarFitted(self):
        """A wall that does not stand out of the blood pool (scar) is drawn by the fit from the walls around it
        in every segmentation mode and fit: it moves with them, with no step at its edge, and a dim wall that
        can be traced moves on its own."""
        n, spacing = 64, 3.0
        center = np.array([n * spacing / 2.0] * 3)
        u, w = unit([-0.6, 0.6, -0.4]), unit([0.0, 0.5, 0.75])
        s = np.cross(u, w)

        def ventricle(radius, scar=22.0, dim=45.0):
            ii = np.arange(n) * spacing
            p = np.stack(np.meshgrid(ii, ii, ii, indexing="ij"), -1)[..., ::-1] - center   # array [k, j, i]
            z = p @ u
            rho = np.linalg.norm(p - z[..., None] * u, axis=-1)
            r = np.where(z > 0, np.linalg.norm(p, axis=-1), rho)                # cylinder, then an apical cap
            theta = np.degrees(np.arctan2(p @ s, p @ w))
            inside = z > -30
            wall = np.where(np.abs(theta) < 40, scar, np.where(np.abs(np.abs(theta) - 180) < 40, dim, 100.0))
            image = np.where(inside & (np.abs(r - radius) < 5), wall, 0.0)
            image = np.where(inside & (r < radius - 5), 20.0, image)            # blood pool
            return image.astype(np.float32)

        def sampleOf(array):
            return lambda pts: sampleTrilinear(array, np.asarray(pts) / spacing)
        static = wallLib.fitWall(sampleOf(ventricle(25.0)), center, u, w, 55.0)
        grid = static["grid"]
        theta = np.degrees(np.angle(np.exp(1j * grid.theta)))
        scarCols, dimCols = np.abs(theta) < 25, np.abs(np.abs(theta) - 180) < 25
        normalCols = (np.abs(np.abs(theta) - 90) < 25)
        rows = slice(grid.nCyl - 6, grid.nCyl - 2)
        reliability = wallLib._staticReliability(static)
        self.assertLess(float(np.median(reliability[rows][:, scarCols])), 0.1)
        self.assertGreater(float(np.median(reliability[rows][:, dimCols])), 0.9)
        scarEdges = np.abs(theta) < 60
        for mode in ("static", "own"):
            for fit in ("bspline", "mesh", "legacy"):
                params = {"segmentation": mode, "fit": fit}
                gate = wallLib.fitGate(sampleOf(ventricle(21.0)), static, params)
                self.assertLess(float(np.median(gate["gateWeight"][rows][:, scarCols])), 0.01, params)
                moved = static["mid"][rows] - gate["mid"][rows]
                self.assertGreater(float(np.median(moved[:, normalCols])), 2.5, params)
                self.assertGreater(float(np.median(moved[:, dimCols])), 2.5, params)
                self.assertGreater(float(np.median(moved[:, scarCols])), 2.5, params)
                mid = gate["mid"][rows]
                step = np.abs(mid - np.roll(mid, 1, axis=1))[:, scarEdges].max()   # between neighbouring rays
                self.assertLess(float(step), 2.5, params)
        self.delayDisplay("Scar fitted test passed")

    def test_PoolFloor(self):
        """The width of a wall profile is measured with each side against its own floor: a blood pool on the
        cavity side does not widen it (one floor for both sides puts the inner edge out in the pool, more so
        the dimmer the wall against it)."""
        t = np.arange(0.0, 60.0, 0.5)
        sigma = 10.0 / 2.3548
        edge = lambda x: 0.5 * (1.0 + np.vectorize(math.erf)(x / (sigma * math.sqrt(2.0))))
        widths = {}
        for pool in (0.0, 15.0, 40.0, 60.0):
            # a 10 mm wall at 25 mm (counts 100) and the blood pool inside it, both blurred to 10 mm FWHM
            profile = 100.0 * (edge(t - 20.0) - edge(t - 30.0)) + pool * (1.0 - edge(t - 20.0))
            profiles = profile[None, None, :]
            peak = np.array([[profile.max()]])
            position = np.array([[float(t[np.argmax(profile)])]])
            widths[pool] = float(wallLib._profileWidth(profiles, t, position, peak, 14.0)[0, 0])
            single = float(wallLib._fwhm(profiles, t, position, peak, 14.0, np.array([[profile[-1]]]))[0, 0])
            if pool >= 40.0:
                self.assertGreater(single, widths[0.0] + 2.0)     # one floor: the pool widens the wall
        for pool, width in widths.items():
            self.assertAlmostEqual(width, widths[0.0], delta=1.0, msg=pool)
        self.delayDisplay("Pool floor test passed")

    def test_SlabThickness(self):
        """A wall's thickness from its profile width, measured as the walls are (_profileWidth, within reach of
        the peak), at 12 mm blur: a 10 mm wall (the prior) reads 10, a wall thicker than the blur is read
        from its width, a thin one leans to the prior (its width hardly differs from a 10 mm wall's), and a
        width that is not finite gives the prior."""
        t = np.arange(0.0, 60.0, 0.5)
        sigma = 12.0 / 2.3548
        edge = lambda x: 0.5 * (1.0 + np.vectorize(math.erf)(x / (sigma * math.sqrt(2.0))))
        read = {}
        for thickness in (6.0, 10.0, 20.0):
            profile = 100.0 * (edge(t - 27.0 + thickness / 2.0) - edge(t - 27.0 - thickness / 2.0))
            width = wallLib._profileWidth(profile[None, None, :], t, np.array([[27.0]]),
                                          np.array([[profile.max()]]), 15.0)
            read[thickness] = float(wallLib.slabThickness(width, 12.0, 15.0)[0, 0])
        self.assertAlmostEqual(read[10.0], 10.0, delta=0.5)
        self.assertGreater(read[20.0], 17.0)
        self.assertTrue(6.0 < read[6.0] < read[10.0] - 0.5, read)
        self.assertAlmostEqual(float(wallLib.slabThickness(np.array([np.nan]), 12.0, 15.0)[0]),
                               wallLib.THICKNESS_PRIOR_MM, delta=0.01)
        self.delayDisplay("Slab thickness test passed")

    def test_FitSurface(self):
        """The fits keep what the traced rays show and bridge a gap of untraced rays without a step: the
        B-spline continues the curve of the walls around it, the mesh runs straighter, the legacy path keeps
        the jump between neighbouring rays within its limit; pull 0 keeps the traced rays' own values. At a
        bending length of 8 mm: the default 15 mm also bends a wall 3 mm off the axis about 1.3 mm toward it."""
        grid = wallLib.WallGrid(np.zeros(3), [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 50.0, length=30.0)
        theta = grid.theta[None, :].repeat(grid.nRows, axis=0)
        amplitude = np.concatenate([np.ones(grid.nCyl), np.sin(grid.phi)])[:, None]
        values = 20.0 + 3.0 * amplitude * np.cos(theta)           # an LV 3 mm off the axis
        weights = np.ones_like(values)
        gap = np.abs(np.angle(np.exp(1j * theta))) < math.radians(35)      # untraced around 0 deg (the 23 mm side)
        weights[gap] = 0.0
        distance = values
        geometry = wallLib._FitGeometry(grid, distance)
        middle = grid.nCyl // 2
        length = wallLib.SPLINE_LENGTH_MM
        wallLib.SPLINE_LENGTH_MM = 8.0
        try:
            self._checkFitSurface(grid, values, weights, gap, distance, geometry, middle)
        finally:
            wallLib.SPLINE_LENGTH_MM = length
        self.delayDisplay("Fit surface test passed")

    def _checkFitSurface(self, grid, values, weights, gap, distance, geometry, middle):
        for fit in ("bspline", "mesh", "legacy"):
            out = wallLib.fitSurface(grid, np.where(gap, 0.0, values), weights, fit, 1.0, distance)
            self.assertLess(float(np.abs(out - values)[~gap].max()), 0.6, fit)
            jump = np.abs(out[:grid.nRows - 1] - np.roll(out[:grid.nRows - 1], 1, axis=1)).max(axis=1)
            self.assertTrue(np.all(jump <= wallLib.LEGACY_MAX_SLOPE * geometry.hTheta[:grid.nRows - 1] + 0.6), fit)
            self.assertTrue(np.allclose(out[-1], out[-1].mean()), fit)        # the apex is one point
            if fit == "bspline":
                self.assertGreater(float(out[middle][gap[middle]].max()), 22.0)   # the curve goes on into the gap
            if fit == "mesh":
                self.assertLess(float(out[middle][gap[middle]].max()), 22.6)
        raw = np.where(gap, 0.0, values + np.where(np.arange(grid.nTheta) % 2, 0.5, -0.5)[None])
        own = wallLib.fitSurface(grid, raw, weights, "bspline", 0.0, distance)
        self.assertTrue(np.allclose(own[middle][~gap[middle]], raw[middle][~gap[middle]], atol=1e-6))

# saved Advanced parameters of every step, this module's constants included (see parameters.py)
parametersLib.applySaved()
