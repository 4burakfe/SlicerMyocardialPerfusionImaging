"""
Myocardial Perfusion SPECT - 5. Gated QC

Quality control of the gated study of Study 1 (stress) or Study 2 (rest) before the gated review:

  - the myocardium is segmented in every gate frame, starting from the static wall detected in
    Myocardium (MyocardialPerfusionSPECTLib/myocardium.py fitGate: mid-wall refitted per ray near the
    static one, valve plane refitted within a limit; by default midWallGates keeps every ray at its
    end-diastolic wall thickness, so only the mid-wall moves; selectable: constantVolumeGates, every frame with
    the myocardial volume of the ES frame (or ED), whose thickness comes from its own profile widths at
    the blur estimated from the frames, rays through a defect at their static thickness, or the per-ray
    model of the static wall);
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
from MyocardialPerfusionSPECTLib import studies as studiesLib  # noqa: E402
from MyocardialPerfusionSPECTLib import myocardium as wallLib  # noqa: E402
from MyocardialPerfusionSPECTLib import gating as gatingLib  # noqa: E402
from MyocardialPerfusionSPECTLib.geometry import ellipsoidInsideGrid, sampleTrilinear, unit, worldToLocalMatrix  # noqa: E402
from MyocardialPerfusionSPECTLib.resampling import DEFAULT_INTERPOLATION, reslice, zoomInPlane  # noqa: E402
from MyocardialPerfusionSPECTLib.views import (  # noqa: E402
    GATED_QC_LAYOUT_ID, ORIENTATIONS, ORIENTATION_TITLES, gatedQcLayoutXml, registerLayout)
from MyocardialPerfusionSPECTLib.display import (  # noqa: E402
    COLORMAPS, colorTableArray, colorize, rgbToQImage, gatedMosaicPanel, countCurvePanel, gatingInfoPanel,
    existingPanel, tilePixelsToIjk, BoundedCache, packMask, unpackMask)
from MyocardialPerfusionSPECTLib.widgets import styleButton, StudyPairSelector  # noqa: E402

STRESS, REST = "stress", "rest"
SLOT_TITLES = {STRESS: "STUDY 1 (STRESS) GATED", REST: "STUDY 2 (REST) GATED"}
CONTOUR_COLORS = {"epi": (64, 220, 90), "endo": (60, 190, 255)}
CONTOUR_MAX_SIZE = 160
TILE_CACHE_BYTES = 96 * 1024 * 1024      # frame tiles kept for scrolling back (oldest dropped first)
CONTOUR_CACHE_BYTES = 16 * 1024 * 1024   # contour masks, bit-packed (also used by Gated Review)
LABEL_GRID_MM = 2.0          # grid of the per-frame myocardium / cavity labels (counts and volumes)
NORMALIZATIONS = (("All frames together", "common"), ("Each frame on its own", "frame"))
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


def sampler(array, matrix):
    def sample(points):
        ijk = np.asarray(points, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]
        return sampleTrilinear(array, ijk)
    return sample


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


def frameStatistics(result, frame, array, matrix, voxelVolumeMm3):
    """Myocardium counts (the gate's counts inside the frame's myocardium), cavity and myocardium ml."""
    labels, labelMatrix = _labelGrid(result, frame, LABEL_GRID_MM)
    m = slicer.util.arrayFromVTKMatrix(labelMatrix)
    index = np.argwhere(labels == wallLib.LABEL_MYOCARDIUM)[:, ::-1].astype(float)   # (i, j, k)
    points = index @ m[:3, :3].T + m[:3, 3]
    voxel = LABEL_GRID_MM ** 3
    counts = float(sampler(array, matrix)(points).sum()) * voxel / max(voxelVolumeMm3, 1e-9) if len(points) else 0.0
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
                result = wallLib.fitGate(sampler(array, matrix), staticResult, params)
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
        model = dict(wallLib.GATE_DEFAULTS, **(params or {}))["thicknessModel"]
        if model in ("volume", "es"):
            # the wall thickness of all frames together: each keeps the myocardial volume of ED (or ES)
            self.results = wallLib.constantVolumeGates(self.results, params, "es" if model == "es" else "ed")
        elif model == "mid":
            # only the mid-wall moves: every ray keeps its end-diastolic wall thickness in all frames
            self.results = wallLib.midWallGates(self.results, params)
        for index, (label, node, matrix, voxel) in enumerate(frames):
            array = slicer.util.arrayFromVolume(node).astype(np.float32)   # again, one frame at a time
            stats = {"myocardiumCounts": None, "cavityMl": None, "myocardiumMl": None}
            if self.results[index] is not None:
                try:
                    stats = frameStatistics(self.results[index], frame, array, matrix, voxel)
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
        self.studyLabel = qt.QLabel()
        self.studyLabel.wordWrap = True
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
        for text, value in NORMALIZATIONS:
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
        self.gateSmoothingSlider = ctk.ctkSliderWidget()
        self.gateSmoothingSlider.minimum, self.gateSmoothingSlider.maximum = 0.0, 4.0
        self.gateSmoothingSlider.singleStep = 0.25
        self.gateSmoothingSlider.decimals = 2
        self.gateSmoothingSlider.value = wallLib.GATE_DEFAULTS["gateSmoothing"]
        self.gateSmoothingSlider.toolTip = ("Smoothing of the wall surface of every frame (rays), 2 by default. Higher "
                                            "values bridge defects and noise more, lower values follow the counts")
        advancedForm.addRow("Surface smoothing:", self.gateSmoothingSlider)
        self.thicknessModelCombo = qt.QComboBox()
        for text, value in wallLib.THICKNESS_MODELS:
            self.thicknessModelCombo.addItem(text, value)
        self.thicknessModelCombo.setCurrentIndex(
            max(0, self.thicknessModelCombo.findData(wallLib.GATE_DEFAULTS["thicknessModel"])))
        self.thicknessModelCombo.toolTip = (
            "Wall thickness of the frames. Mid-wall only (default): only the mid-wall (the count peak, the surface "
            "the gates show best) is taken from each frame and every ray keeps its end-diastolic wall thickness, "
            "so no thickening model enters the volumes; it reads lower than the others in hearts that thicken well and "
            "does not overestimate when the gates are too blurred to show the thickening. Constant myocardial "
            "volume, ES reference: every frame keeps the myocardial volume of end-systole, the thickest and "
            "brightest wall, whose thickness comes from its own count profiles (the blur estimated from the "
            "frames). ED reference: the same from end-diastole. Per ray: each ray keeps its wall volume of the "
            "static fit (the summed image blurs the wall over the cycle, so the walls come out thicker and the EF "
            "higher, most in small hearts)")
        advancedForm.addRow("Wall thickness:", self.thicknessModelCombo)
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
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneClosed)
        session.addListener(self.onSessionChanged)
        _installToolbar()
        self.updateState()

    # --- workflow toolbar (second row) -----------------------------------------------------------------
    def buildToolbar(self, bar):
        self.toolbarBar = bar
        bar.mirror(self.segmentButton, "Segment again", "Segment the frames again with the current settings")
        bar.mirror(self.contoursButton, "Contours")
        bar.separator()
        for slot, button in self.layoutButtons.items():
            bar.mirror(button, SLOT_TITLES[slot], f"Gated frames of the {SLOT_TITLES[slot].lower()} study")

    def syncToolbar(self):
        bar = getattr(self, "toolbarBar", None)
        if bar is not None:
            bar.syncMirrors()

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
            text = f"{title}: <b>{study.GetName()}</b>"
            if sequence is not None:
                text += f", gated <b>{sequence.GetName()}</b> ({sequence.GetNumberOfDataNodes()} frames)"
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

    def params(self):
        return {"maxWallShift": float(self.wallShiftSpin.value), "maxValveShift": float(self.valveShiftSpin.value),
                "gateSmoothing": float(self.gateSmoothingSlider.value),
                "thicknessModel": (self.thicknessModelCombo.itemData(self.thicknessModelCombo.currentIndex)
                                   or wallLib.GATE_DEFAULTS["thicknessModel"])}

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
               tuple(sorted((k, n.GetID()) for k, n in outputs.items())), tuple(sorted(self.params().items())))
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
            return (100.0 * ((np.abs(rho - radius) < 5) & (z > -30) & (z < 0))).astype(np.float32)

        def sampleOf(array):
            return lambda pts: sampleTrilinear(array, np.asarray(pts) / spacing)
        static = wallLib.fitWall(sampleOf(ventricle(25.0)), center, u, w, 55.0)
        gate = wallLib.fitGate(sampleOf(ventricle(21.0)), static)
        middle = static["grid"].nCyl - 4
        self.assertLess(float(np.median(gate["mid"][middle])), float(np.median(static["mid"][middle])) - 2.0)
        # constant volume: the smaller frame gets a thicker wall, with the myocardial volume of ED
        ed = wallLib.fitGate(sampleOf(ventricle(25.0)), static)
        walls = wallLib.constantVolumeGates([ed, gate])
        self.assertEqual(walls[1]["edFrame"], 0)
        self.assertAlmostEqual(wallLib.wallVolumeMl(walls[1]), wallLib.wallVolumeMl(walls[0]),
                               delta=0.03 * wallLib.wallVolumeMl(walls[0]))
        self.assertGreater(float(np.median(walls[1]["thickness"][middle])), float(np.median(walls[0]["thickness"][middle])))
        self.delayDisplay("Gate fit test passed")
