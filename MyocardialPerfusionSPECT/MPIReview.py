"""
Myocardial Perfusion SPECT - 4. Perfusion

Layouts (buttons at the bottom of the panel):
  STRESS PERF / REST PERF          slices (SA / HLA / VLA rows), polar map and 3D uptake of one study
  STRESS SLICES / REST SLICES      only the slice rows
  STRESS-REST PERF                 stress and rest rows interleaved; on the right stress polar map and
                                   3D uptake above the rest ones
  STRESS-REST SLICES               only the interleaved rows
  STRESS REST POLAR                stress, rest and difference (rest - stress, ischemia) polar maps
Slices are made from the reoriented volumes of Processing, rendered once at screen resolution and
kept in memory (scrolling only swaps tiles), optionally with the myocardium contours. With "Sync"
the rest rows follow the stress rows: the same position relative to the center of each study's
segmented myocardium, slice and in-plane. Polar maps and 3D uptake use the LV wall detected in
Myocardium (MPIMyocardium.py). Gated review is MPIGated.py.
"""

import os
import sys
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

from MyocardialPerfusionSPECTLib.resampling import DEFAULT_INTERPOLATION, zoomInPlane  # noqa: E402
from MyocardialPerfusionSPECTLib.views import (  # noqa: E402
    RESULT_LAYOUT_ID, SLICES_LAYOUT_ID, DUAL_RESULT_LAYOUT_ID, POLAR_LAYOUT_ID, ORIENTATIONS, ORIENTATION_TITLES,
    POLAR_TAG, POLAR_REST_TAG, POLAR_DIFF_TAG, PERFUSION_3D_VIEW, PERFUSION_3D_VIEW_REST,
    resultLayoutXml, slicesLayoutXml, dualResultLayoutXml, polarLayoutXml, registerLayout, threeDWidgetByTag)
from MyocardialPerfusionSPECTLib.display import (  # noqa: E402
    DEFAULT_GRID_ROWS, DEFAULT_GRID_COLUMNS, COLORMAPS, colorTableArray, colorize, colorNodeForLabel,
    setVolumeDisplay, rgbToQImage, mosaicPanel, polarPanel, polarPanelState, existingPanel)
from MyocardialPerfusionSPECTLib.geometry import worldToLocalMatrix  # noqa: E402
from MyocardialPerfusionSPECTLib import session  # noqa: E402
from MyocardialPerfusionSPECTLib import myocardium as wallLib  # noqa: E402


STRESS, REST, DIFF = "stress", "rest", "diff"
SLOT_TITLES = {STRESS: "STRESS", REST: "REST"}
LAYOUT_XML = {RESULT_LAYOUT_ID: resultLayoutXml, SLICES_LAYOUT_ID: slicesLayoutXml,
              DUAL_RESULT_LAYOUT_ID: dualResultLayoutXml, POLAR_LAYOUT_ID: polarLayoutXml}
# key: (button text, layout ID, studies of the slice rows, {polar map tag: study or DIFF}, {3D view tag: study}, tooltip)
MODES = {
    "stressPerf": ("STRESS PERF", RESULT_LAYOUT_ID, (STRESS,), {POLAR_TAG: STRESS}, {PERFUSION_3D_VIEW: STRESS},
                   "Stress study: SA / HLA / VLA slices, polar map and 3D uptake"),
    "stressSlices": ("STRESS SLICES", SLICES_LAYOUT_ID, (STRESS,), {}, {}, "Stress study: only the slices"),
    "restPerf": ("REST PERF", RESULT_LAYOUT_ID, (REST,), {POLAR_TAG: REST}, {PERFUSION_3D_VIEW: REST},
                 "Rest study: SA / HLA / VLA slices, polar map and 3D uptake"),
    "restSlices": ("REST SLICES", SLICES_LAYOUT_ID, (REST,), {}, {}, "Rest study: only the slices"),
    "stressRestPerf": ("STRESS-REST PERF", DUAL_RESULT_LAYOUT_ID, (STRESS, REST),
                       {POLAR_TAG: STRESS, POLAR_REST_TAG: REST},
                       {PERFUSION_3D_VIEW: STRESS, PERFUSION_3D_VIEW_REST: REST},
                       "Stress and rest slices row by row, with both polar maps and 3D uptake views"),
    "stressRestSlices": ("STRESS-REST SLICES", SLICES_LAYOUT_ID, (STRESS, REST), {}, {},
                         "Stress and rest slices row by row"),
    "stressRestPolar": ("STRESS REST POLAR", POLAR_LAYOUT_ID, (), {POLAR_TAG: STRESS, POLAR_REST_TAG: REST,
                                                                   POLAR_DIFF_TAG: DIFF}, {},
                        "Stress, rest and difference (rest - stress, ischemia) polar maps"),
}
MODE_GRID = (("stressPerf", "stressSlices"), ("restPerf", "restSlices"),
             ("stressRestPerf", "stressRestSlices"), ("stressRestPolar",))   # rows of layout buttons
CONTOUR_COLORS = {"epi": (64, 220, 90), "endo": (60, 190, 255)}   # myocardium contours on the slices
CONTOUR_MAX_SIZE = 160    # contours are computed at most at this tile size and enlarged above it
CENTER_GRID_MM = 3.0      # grid of the myocardium mass center (slice / pan matching of stress and rest)
DIFF_WINDOW = (-50.0, 100.0, 0.0, 40.0)   # range and default window of the difference map, % points


def shiftImage(rgb, dx, dy):
    """RGB image moved by whole pixels (right / down positive), black where it comes from outside."""
    dx, dy = int(dx), int(dy)
    if dx == 0 and dy == 0:
        return rgb
    out = np.zeros_like(rgb)
    h, w = rgb.shape[:2]
    if abs(dx) >= w or abs(dy) >= h:
        return out
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = rgb[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def volumeText(segmentation):
    """LV cavity and myocardium volumes saved by Myocardium on the segmentation, as one line."""
    if segmentation is None:
        return ""
    get = lambda key: segmentation.GetAttribute(key) or "-"
    return f"LV cavity {get('MPI.LVCavityMl')} ml  ·  myocardium {get('MPI.MyocardiumMl')} ml ({get('MPI.MyocardiumG')} g)"


def _installToolbar():
    try:
        from MyocardialPerfusionSPECTLib import toolbar
        toolbar.install()
    except Exception:
        logging.exception("MPI: could not install the workflow toolbar")


class MPIReview(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MPI 4 · Perfusion"
        self.parent.categories = ["Nuclear Medicine.Myocardial Perfusion"]
        self.parent.hidden = True   # opened from the Myocardial Perfusion SPECT hub and the workflow toolbar
        self.parent.dependencies = []
        self.parent.contributors = ["Burak Demir"]
        self.parent.helpText = (
            "Review the reoriented slices of the stress and rest studies (roles set in Data roles), "
            "alone or row by row. Mouse wheel over a row scrolls it. Polar maps and 3D views of the LV "
            "with the wall counts on its surface use the wall detected in Myocardium; STRESS REST POLAR "
            "adds the rest - stress difference.")
        self.parent.acknowledgementText = ""
        iconPath = os.path.join(_MODULE_DIR, "Resources", "Icons", "MyocardialPerfusionSPECT.png")
        if os.path.exists(iconPath):
            self.parent.icon = qt.QIcon(iconPath)
        if not slicer.app.commandOptions().noMainWindow:
            slicer.app.connect("startupCompleted()", _installToolbar)


class MPIReviewWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.studies = {STRESS: None, REST: None}   # static volume of each study shown
        self.outputs = {STRESS: {}, REST: {}}       # its reoriented volumes {orientation: node}
        self.mode = "stressPerf"
        self.grids = {"single": [DEFAULT_GRID_ROWS, DEFAULT_GRID_COLUMNS], "dual": [1, DEFAULT_GRID_COLUMNS]}
        self.polarCache = {}         # study ID -> (wall key, polar map)
        self.tileCache = {}          # row key -> (key, [QImage per slice]): rendered once, scrolling reuses them
        self.contourCache = {}       # row key -> (key, (epi, endo) bool masks [slices, size, size])
        self.polarImageCache = {}    # polar map tag -> (key, (QImage, segment values))
        self.centerCache = {}        # study ID -> (wall key, myocardium mass center in world coordinates)
        self.surfaceKeys = {}        # 3D view tag -> wall its uptake model and camera were made for

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.studyLabel = qt.QLabel("No study selected.")
        self.studyLabel.wordWrap = True
        self.layout.addWidget(self.studyLabel)

        # Slices
        section = ctk.ctkCollapsibleButton()
        section.text = "Slices"
        self.layout.addWidget(section)
        form = qt.QFormLayout(section)

        self.colormapCombo = qt.QComboBox()
        self.colormapCombo.addItems([label for label, _ in COLORMAPS])
        form.addRow("Color map:", self.colormapCombo)

        self.windowRanges = {}
        for slot in (STRESS, REST):
            self.windowRanges[slot] = self._windowWidget(
                f"Color window (min - max) of the {slot} slices as a percentage of its myocardial (ROI) "
                "maximum: counts below the minimum get the lowest color, counts above the maximum the highest")
            form.addRow(f"{slot.capitalize()} window:", self.windowRanges[slot])

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
        gridRow.addWidget(qt.QLabel("×"))
        gridRow.addWidget(self.gridColumnsSpin)
        form.addRow("Grid per row:", gridRow)

        showRow = qt.QHBoxLayout()
        self.contoursButton = self._toggleButton("Myocardium contours", False,
                                                 "Show the epicardium (green) and endocardium (blue) detected in "
                                                 "Myocardium on the slices")
        self.syncButton = self._toggleButton("Sync stress and rest", True,
                                             "Stress and rest rows show the same place: slices and panning matched "
                                             "on the center of each study's segmented myocardium, scrolling moves "
                                             "both. Off: each row scrolls on its own and is not moved")
        showRow.addWidget(self.contoursButton)
        showRow.addWidget(self.syncButton)
        form.addRow("Show:", showRow)

        # Polar map
        polarSection = ctk.ctkCollapsibleButton()
        polarSection.text = "Polar map"
        self.layout.addWidget(polarSection)
        polarForm = qt.QFormLayout(polarSection)
        self.polarLinkCheck = qt.QCheckBox("Same as the slices")
        self.polarLinkCheck.checked = True
        self.polarLinkCheck.toolTip = ("Use each study's slice window for its polar map and 3D uptake view (in % of "
                                       "the polar map maximum); untick to window them on their own")
        polarForm.addRow("Window:", self.polarLinkCheck)
        self.polarWindowRanges = {}
        for slot in (STRESS, REST):
            self.polarWindowRanges[slot] = self._windowWidget(
                f"Color window (min - max) of the {slot} polar map and 3D uptake view, as a percentage of the "
                "polar map maximum")
            polarForm.addRow(f"{slot.capitalize()}:", self.polarWindowRanges[slot])
        self.diffWindowRange = self._windowWidget(
            "Color window of the difference map (rest - stress, each in % of its own maximum): positive "
            "values are lower uptake at stress (ischemia)", *DIFF_WINDOW)
        polarForm.addRow("Difference:", self.diffWindowRange)
        overlayRow = qt.QHBoxLayout()
        self.segmentsButton = self._toggleButton("Segments", True,
                                                 "Show the borders of the 17 segments on the polar maps and in 3D")
        self.scoresButton = self._toggleButton("Scores", True,
                                               "Show the segment values (mean counts, % of the map maximum)")
        overlayRow.addWidget(self.segmentsButton)
        overlayRow.addWidget(self.scoresButton)
        polarForm.addRow("Show:", overlayRow)

        # Slice navigation (hidden by default)
        navigation = ctk.ctkCollapsibleButton()
        navigation.text = "Slice navigation"
        navigation.collapsed = True
        self.layout.addWidget(navigation)
        navigationForm = qt.QFormLayout(navigation)

        self.stepSpin = qt.QSpinBox()
        self.stepSpin.setRange(1, 10)
        self.stepSpin.setValue(1)
        self.stepSpin.toolTip = "Slices advanced from one tile to the next (1 = consecutive slices)"
        navigationForm.addRow("Slice step:", self.stepSpin)

        self.firstSliceSpins = {}
        for slot in (STRESS, REST):
            for name in ORIENTATIONS:
                spin = qt.QSpinBox()
                spin.setRange(1, 1)
                spin.toolTip = (f"First slice of the {slot} {ORIENTATION_TITLES[name]} row (mouse wheel over the "
                                "row also scrolls)")
                self.firstSliceSpins[(slot, name)] = spin
                navigationForm.addRow(f"First {slot} {name} slice:", spin)

        self.backButton = qt.QPushButton("Back to reorientation")
        self.layout.addWidget(self.backButton)

        # Layouts
        self.layoutButtons = {}
        self.layoutGroup = qt.QButtonGroup()
        self.layoutGroup.exclusive = True
        layoutBox = qt.QGroupBox("Layout")
        grid = qt.QGridLayout(layoutBox)
        for row, keys in enumerate(MODE_GRID):
            for column, key in enumerate(keys):
                button = qt.QPushButton(MODES[key][0])
                button.checkable = True
                button.toolTip = MODES[key][5]
                grid.addWidget(button, row, column, 1, 2 // len(keys))
                self.layoutGroup.addButton(button)
                self.layoutButtons[key] = button
        self.layout.addWidget(layoutBox)
        self.layout.addStretch(1)

        self.mosaicResizeTimer = qt.QTimer()
        self.mosaicResizeTimer.setSingleShot(True)
        self.mosaicResizeTimer.setInterval(150)

        self.mosaicResizeTimer.connect("timeout()", self.refreshMosaic)
        self.colormapCombo.connect("currentIndexChanged(int)", self.onColormapChanged)
        for slot in (STRESS, REST):
            self.windowRanges[slot].connect("valuesChanged(double,double)", lambda a, b: self.onSliceWindowChanged())
            self.polarWindowRanges[slot].connect("valuesChanged(double,double)", lambda a, b: self.refreshPolar())
        self.diffWindowRange.connect("valuesChanged(double,double)", lambda a, b: self.refreshPolar())
        self.polarLinkCheck.connect("toggled(bool)", lambda _: self.onPolarLinkToggled())
        self.gridRowsSpin.connect("valueChanged(int)", lambda _: self.onGridChanged())
        self.gridColumnsSpin.connect("valueChanged(int)", lambda _: self.onGridChanged())
        self.contoursButton.connect("toggled(bool)", lambda _: self.refreshMosaic(polar=False))
        self.syncButton.connect("toggled(bool)", lambda _: self.onSyncToggled())
        self.segmentsButton.connect("toggled(bool)", lambda _: self.onPolarOverlaysChanged())
        self.scoresButton.connect("toggled(bool)", lambda _: self.onPolarOverlaysChanged())
        self.stepSpin.connect("valueChanged(int)", lambda _: self.refreshMosaic(polar=False))
        for spin in self.firstSliceSpins.values():
            spin.connect("valueChanged(int)", lambda _: self.refreshMosaic(polar=False))
        for key, button in self.layoutButtons.items():
            button.connect("clicked(bool)", lambda _, k=key: self.showResults(k))
        self.backButton.connect("clicked(bool)", lambda: session.selectStep("processing"))

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneClosed)
        session.addListener(self.onSessionChanged)
        _installToolbar()
        self.loadOutputs()

    @staticmethod
    def _windowWidget(toolTip, minimum=0.0, maximum=150.0, low=0.0, high=100.0):
        widget = ctk.ctkRangeWidget()
        widget.minimum, widget.maximum = minimum, maximum
        widget.singleStep = 1.0
        widget.decimals = 0
        widget.suffix = " %"
        widget.minimumValue, widget.maximumValue = low, high
        widget.toolTip = toolTip
        return widget

    @staticmethod
    def _toggleButton(text, checked, toolTip):
        button = qt.QPushButton(text)
        button.checkable = True
        button.checked = checked
        button.toolTip = toolTip
        return button

    def enter(self):
        self.loadOutputs()
        self.showResults()

    def cleanup(self):
        self.mosaicResizeTimer.stop()
        session.removeListener(self.onSessionChanged)
        for tag in (POLAR_TAG, POLAR_REST_TAG, POLAR_DIFF_TAG):
            panel = existingPanel(polarPanelState(tag))
            if panel is not None:
                panel.onResized = None
        panel = existingPanel("_mpiSliceMosaicState")
        if panel is not None:
            panel.onResized = None
            panel.onScroll = None
        self.removeObservers()

    # --- studies ---------------------------------------------------------------
    @staticmethod
    def findStudies():
        """{STRESS: node, REST: node} of the studies to show (see session.stressRestStudies)."""
        found = session.stressRestStudies()
        return {STRESS: found["stress"], REST: found["rest"]}

    def ready(self, slot):
        return len(self.outputs.get(slot) or {}) == len(ORIENTATIONS)

    def modeAvailable(self, mode):
        _, _, slots, polars, views, _ = MODES[mode]
        needed = set(slots) | {s for s in list(polars.values()) + list(views.values()) if s != DIFF}
        if DIFF in polars.values():
            needed |= {STRESS, REST}
        return bool(needed) and all(self.ready(slot) for slot in needed)

    def onSessionChanged(self):
        before = dict(self.studies)
        self.loadOutputs()
        if any(self.studies[s] is not before[s] for s in self.studies) or \
                any(self.studies[s] is None for s in self.studies):
            active = session.activeStudy()
            singles = {STRESS: ("stressPerf", "stressSlices"), REST: ("restPerf", "restSlices")}
            for slot, other in ((STRESS, REST), (REST, STRESS)):   # follow the study chosen in the toolbar
                if active is not None and self.studies[slot] is active and self.mode in singles[other]:
                    self.mode = singles[slot][singles[other].index(self.mode)]
            if self.parent.isVisible():
                self.showResults()

    def onSceneClosed(self, caller=None, event=None):
        self.studies = {STRESS: None, REST: None}
        self.outputs = {STRESS: {}, REST: {}}
        self.tileCache = {}
        self.contourCache = {}
        self.polarCache = {}
        self.polarImageCache = {}
        self.centerCache = {}
        self.surfaceKeys = {}
        panel = existingPanel("_mpiSliceMosaicState")
        if panel is not None:
            panel.clear()
        for tag in (POLAR_TAG, POLAR_REST_TAG, POLAR_DIFF_TAG):
            panel = existingPanel(polarPanelState(tag))
            if panel is not None:
                panel.clear()
        self.updateState()

    def loadOutputs(self):
        found = self.findStudies()
        for slot in (STRESS, REST):
            study = found[slot]
            outputs = session.outputsForStudy(study)
            changed = study is not self.studies[slot] or \
                {k: n.GetID() for k, n in outputs.items()} != {k: n.GetID() for k, n in self.outputs[slot].items()}
            self.studies[slot], self.outputs[slot] = study, outputs
            if changed and self.ready(slot):
                self.resetFirstSlices(slot)
        self.updateState()

    def updateState(self):
        lines = []
        for slot in (STRESS, REST):
            study = self.studies[slot]
            if study is None:
                lines.append(f"{SLOT_TITLES[slot]}: none")
            elif not self.ready(slot):
                lines.append(f"{SLOT_TITLES[slot]}: <b>{study.GetName()}</b> has no reoriented volumes yet "
                             "(create them in Processing)")
            else:
                lines.append(f"{SLOT_TITLES[slot]}: <b>{study.GetName()}</b>")
        if self.studies[STRESS] is None and self.studies[REST] is None:
            lines = ["No study selected. Choose one in the workflow toolbar or set the roles in Data roles."]
        self.studyLabel.text = "<br>".join(lines)
        anyReady = self.ready(STRESS) or self.ready(REST)
        for widget in [self.gridRowsSpin, self.gridColumnsSpin, self.stepSpin, self.contoursButton,
                       self.polarLinkCheck, self.segmentsButton, self.scoresButton]:
            widget.enabled = anyReady
        self.syncButton.enabled = self.ready(STRESS) and self.ready(REST)
        self.diffWindowRange.enabled = self.ready(STRESS) and self.ready(REST)
        for slot in (STRESS, REST):
            self.windowRanges[slot].enabled = self.ready(slot)
            self.polarWindowRanges[slot].enabled = self.ready(slot) and not self.polarLinkCheck.checked
            for name in ORIENTATIONS:
                self.firstSliceSpins[(slot, name)].enabled = self.ready(slot) and not (slot == REST and self.synced())
        for key, button in self.layoutButtons.items():
            button.enabled = self.modeAvailable(key)

    # --- slice panel ----------------------------------------------------------
    def rowSlots(self, mode=None):
        return MODES[mode or self.mode][2]

    def gridKind(self, mode=None):
        return "dual" if len(self.rowSlots(mode)) > 1 else "single"

    def synced(self):
        """Rest rows follow the stress rows (both shown, Sync on)."""
        return bool(self.syncButton.checked) and len(self.rowSlots()) > 1 and self.ready(STRESS) and self.ready(REST)

    def cellCount(self):
        return self.gridRowsSpin.value * self.gridColumnsSpin.value

    def cellsSpan(self):
        return (self.cellCount() - 1) * self.stepSpin.value + 1

    def onGridChanged(self):
        self.grids[self.gridKind()] = [self.gridRowsSpin.value, self.gridColumnsSpin.value]
        panel = existingPanel("_mpiSliceMosaicState")
        if panel is not None:
            panel.setGrid(self.gridRowsSpin.value, self.gridColumnsSpin.value)
        self.refreshMosaic(polar=False)

    def onSyncToggled(self):
        self.updateState()
        self.refreshMosaic(polar=False)

    def resetFirstSlices(self, slot):
        for name in ORIENTATIONS:
            node = self.outputs[slot].get(name)
            if node is None:
                continue
            n = slicer.util.arrayFromVolume(node).shape[0]
            spin = self.firstSliceSpins[(slot, name)]
            wasBlocked = spin.blockSignals(True)
            spin.setRange(1, n)
            spin.setValue(max(0, (n - self.cellsSpan()) // 2) + 1)
            spin.blockSignals(wasBlocked)

    def displayReference(self, slot):
        """Count level shown at 100 %: the myocardial maximum stored by Processing, else the volume maximum."""
        values = []
        for node in self.outputs[slot].values():
            try:
                values.append(float(node.GetAttribute("MPI.DisplayMax") or "nan"))
            except ValueError:
                pass
        values = [v for v in values if v == v and v > 0]
        if values:
            return max(values)
        return max(float(slicer.util.arrayFromVolume(n).max()) for n in self.outputs[slot].values()) or 1.0

    def rowKeys(self, mode=None):
        slots = self.rowSlots(mode)
        return [(f"{slot}:{name}", f"{SLOT_TITLES[slot]}  ·  {ORIENTATION_TITLES[name]}")
                for name in ORIENTATIONS for slot in slots]

    def setMode(self, mode):
        self.mode = mode
        rows, columns = self.grids[self.gridKind(mode)]
        for spin, value in ((self.gridRowsSpin, rows), (self.gridColumnsSpin, columns)):
            wasBlocked = spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(wasBlocked)
        button = self.layoutButtons.get(mode)
        if button is not None:
            button.checked = True
        self.updateState()

    def showResults(self, mode=None):
        mode = mode or self.mode
        if not self.modeAvailable(mode):
            mode = next((key for key in ("stressPerf", "restPerf") if self.modeAvailable(key)), None)
            if mode is None:
                return
        self.setMode(mode)
        # Leave place mode in case the user was placing the ROI
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

        _, layoutId, slots, polars, _, _ = MODES[mode]
        if slots:
            panel = mosaicPanel()
            panel.onScroll = self.onMosaicScroll
            panel.onResized = self.mosaicResizeTimer.start
            panel.gridRows, panel.gridColumns = self.gridRowsSpin.value, self.gridColumnsSpin.value
            panel.setRowKeys(self.rowKeys(mode))
        for tag in polars:
            polar = polarPanel(tag)
            polar.onResized = self.mosaicResizeTimer.start
            polar.showSegments, polar.showScores = self.segmentsButton.checked, self.scoresButton.checked
        registerLayout(layoutId, LAYOUT_XML[layoutId]())
        slicer.app.layoutManager().setLayout(layoutId)
        slicer.app.processEvents()
        self.surfaceKeys = {}   # place the 3D cameras again
        self.refreshMosaic()

    def onMosaicScroll(self, key, direction):
        if ":" not in key:
            return
        slot, name = key.split(":", 1)
        if self.synced():
            slot = STRESS   # the rest rows follow
        spin = self.firstSliceSpins.get((slot, name))
        if spin:
            spin.setValue(spin.value + direction * self.stepSpin.value)

    def sliceWindow(self, slot):
        """(low, top) counts of the slice color window of a study."""
        reference = self.displayReference(slot)
        window = self.windowRanges[slot]
        low = reference * window.minimumValue / 100.0
        top = max(reference * window.maximumValue / 100.0, low + 1e-9)
        return low, top

    # --- stress / rest matching ---------------------------------------------------
    def lvCenter(self, slot):
        """Mass center (world) of the segmented myocardium of a study; the center of the cardiac
        frame set in Processing when the myocardium was not detected."""
        study = self.studies.get(slot)
        if study is None:
            return None
        import MPIMyocardium
        state = self.wallState(study)
        key = state[0] if state is not None else ("frame",) + tuple(n.GetID() for n in self.outputs[slot].values())
        cached = self.centerCache.get(study.GetID())
        if cached and cached[0] == key:
            return cached[1]
        frame = MPIMyocardium.cardiacFrame(study)
        center = None
        if state is not None and frame is not None:
            try:
                labels, matrix = MPIMyocardium.labelGrid(state[1], frame, CENTER_GRID_MM)
                index = np.argwhere(labels == wallLib.LABEL_MYOCARDIUM)
                if len(index):
                    m = slicer.util.arrayFromVTKMatrix(matrix)
                    center = m[:3, :3] @ index[:, ::-1].mean(axis=0) + m[:3, 3]
            except Exception:
                logging.warning("MPI: could not find the myocardium center:\n" + traceback.format_exc())
        if center is None and frame is not None:
            center = np.asarray(frame["center"], dtype=float)
        self.centerCache[study.GetID()] = (key, center)
        return center

    def centerIjk(self, slot, name):
        """The myocardium center as (i, j, k) of one reoriented volume of a study."""
        node = self.outputs[slot].get(name)
        center = self.lvCenter(slot)
        if node is None or center is None:
            return None
        rasToIjk = vtk.vtkMatrix4x4()
        node.GetRASToIJKMatrix(rasToIjk)
        m = slicer.util.arrayFromVTKMatrix(rasToIjk) @ slicer.util.arrayFromVTKMatrix(worldToLocalMatrix(node))
        return m[:3, :3] @ center + m[:3, 3]

    def syncMapping(self, name):
        """(stress center ijk, rest center ijk, stress / rest slice spacing) of one orientation, or None."""
        stress, rest = self.centerIjk(STRESS, name), self.centerIjk(REST, name)
        if stress is None or rest is None:
            return None
        spacing = self.outputs[STRESS][name].GetSpacing()[2] / max(self.outputs[REST][name].GetSpacing()[2], 1e-6)
        return stress, rest, spacing

    @staticmethod
    def restSlice(mapping, k):
        """Rest slice at the same distance from the myocardium center as stress slice k."""
        if mapping is None:
            return k
        stress, rest, spacing = mapping
        return int(round(rest[2] + (k - stress[2]) * spacing))

    def panShift(self, mapping, name, size):
        """Tile pixels (right, down) to move the rest tiles so its myocardium center sits where the
        stress one does."""
        if mapping is None or not size:
            return 0, 0

        def pixel(node, ijk):   # the mapping of zoomInPlane, rows flipped (top = highest j)
            w, h = node.GetImageData().GetDimensions()[:2]
            return (ijk[0] + 0.5) * size / w - 0.5, (h - 1 - ijk[1] + 0.5) * size / h - 0.5
        xs, ys = pixel(self.outputs[STRESS][name], mapping[0])
        xr, yr = pixel(self.outputs[REST][name], mapping[1])
        return int(round(xs - xr)), int(round(ys - yr))

    # --- rendering --------------------------------------------------------------------
    def refreshMosaic(self, polar=True):
        """Render each row: tile t shows slice first + t * step of that row (rest rows follow the
        stress rows when synced)."""
        slots = self.rowSlots()
        if any(not slicer.mrmlScene.IsNodePresent(n) for slot in (STRESS, REST) for n in self.outputs[slot].values()):
            self.loadOutputs()
        panel = existingPanel("_mpiSliceMosaicState")
        if slots and panel is not None and all(self.ready(slot) for slot in slots):
            self.renderRows(panel, slots)
        if polar:
            self.refreshPolar()

    def renderRows(self, panel, slots):
        panel.gridRows, panel.gridColumns = self.gridRowsSpin.value, self.gridColumnsSpin.value
        panel.setRowKeys(self.rowKeys())
        colorTable = colorTableArray(self.colormapCombo.currentText)
        step, cells = self.stepSpin.value, self.cellCount()
        synced = self.synced()
        for name in ORIENTATIONS:
            mapping = self.syncMapping(name) if synced else None
            stressFirst = self.firstSliceSpins[(STRESS, name)].value - 1
            for slot in slots:
                key = f"{slot}:{name}"
                node = self.outputs[slot][name]
                count = slicer.util.arrayFromVolume(node).shape[0]
                if synced and slot == REST:
                    shown = [self.restSlice(mapping, stressFirst + cell * step) for cell in range(cells)]
                    spin = self.firstSliceSpins[(REST, name)]
                    wasBlocked = spin.blockSignals(True)
                    spin.setValue(min(max(shown[0], 0), count - 1) + 1)
                    spin.blockSignals(wasBlocked)
                else:
                    first = self.firstSliceSpins[(slot, name)].value - 1
                    shown = [first + cell * step for cell in range(cells)]
                size = panel.tileSize(key)
                images = []
                if size:
                    shift = self.panShift(mapping, name, size) if (synced and slot == REST) else (0, 0)
                    images = self.sliceImages(key, slot, node, size, self.sliceWindow(slot), colorTable,
                                              panel.pixelRatio(), shift)
                tiles = [(images[k], k + 1) if images and 0 <= k < count else (None, None) for k in shown]
                valid = [k for k in shown if 0 <= k < count]
                text = f"{SLOT_TITLES[slot]}  ·  {ORIENTATION_TITLES[name]}"
                if valid:
                    text += f"   —   slices {valid[0] + 1}–{valid[-1] + 1} of {count}" + (f", every {step}" if step > 1 else "")
                panel.setRow(key, tiles, text)
        dual = len(slots) > 1
        panel.setLegends([(colorTable, self.windowRanges[slot].minimumValue, self.windowRanges[slot].maximumValue,
                           SLOT_TITLES[slot] if dual else "") for slot in slots])

    def sliceImages(self, rowKey, slot, node, size, window, colorTable, ratio=1.0, shift=(0, 0)):
        """Every slice of one row rendered at the tile size (resized in the data domain, then colour
        mapped, so Qt draws them 1:1), with the myocardium contours when they are on, moved by
        shift tile pixels. Made once per volume, tile size, colour map, window, contours and shift;
        scrolling only picks tiles from this list."""
        imageData = node.GetImageData()
        contours = self.contourMasks(rowKey, slot, node, size, ratio) if self.contoursButton.checked else None
        key = (node.GetID(), imageData.GetMTime() if imageData else 0, int(size), self.colormapCombo.currentText,
               tuple(round(float(v), 6) for v in window),
               node.GetAttribute("MPI.Interpolation") or DEFAULT_INTERPOLATION,
               self.contourCache.get(rowKey, (None,))[0] if contours is not None else None, tuple(shift))
        cached = self.tileCache.get(rowKey)
        if cached and cached[0] == key:
            return cached[1]
        interpolation = key[5]
        array = slicer.util.arrayFromVolume(node)  # [k, j, i]; i = screen right, j = screen up
        stack = zoomInPlane(np.asarray(array[:, ::-1, :]), size, size, interpolation)  # top row = highest j
        images = []
        for k, values in enumerate(stack):
            rgb = colorize(values, window[1], colorTable, window[0])
            if contours is not None:
                for which, mask in zip(("epi", "endo"), contours):
                    rgb[mask[k]] = CONTOUR_COLORS[which]
            images.append(rgbToQImage(np.ascontiguousarray(shiftImage(rgb, *shift))))
        self.tileCache[rowKey] = (key, images)
        return images

    # --- myocardium contours on the slices ----------------------------------------
    def contourMasks(self, rowKey, slot, node, size, ratio=1.0):
        """(epicardium, endocardium) contour masks [slices, size, size] of one row, from the wall
        detected in Myocardium, classified at the tiles' own pixels (smooth at any zoom).
        None when the myocardium was not detected."""
        state = self.wallState(self.studies.get(slot))
        if state is None:
            return None
        wallKey, result = state[0], state[1]
        key = (wallKey, node.GetID(), int(size), round(float(ratio), 2))
        cached = self.contourCache.get(rowKey)
        if cached and cached[0] == key:
            return cached[1]
        array = slicer.util.arrayFromVolume(node)
        slices, h, w = array.shape
        n = int(min(size, CONTOUR_MAX_SIZE))
        # tile pixel centers -> volume (i, j): the same mapping as zoomInPlane, rows flipped (top = highest j)
        i = (np.arange(n) + 0.5) * (w / float(n)) - 0.5
        j = (h - 1) - ((np.arange(n) + 0.5) * (h / float(n)) - 0.5)
        ii, jj = np.meshgrid(i, j)
        ijkToRas = vtk.vtkMatrix4x4()
        node.GetIJKToRASMatrix(ijkToRas)
        localToWorld = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(worldToLocalMatrix(node), localToWorld)
        matrix = slicer.util.arrayFromVTKMatrix(localToWorld) @ slicer.util.arrayFromVTKMatrix(ijkToRas)
        grid = result["grid"]
        reach = float(np.max(result["epi"])) + 3.0     # only pixels this close to the LV center can be wall
        labels = np.zeros((slices, n, n), dtype=np.uint8)
        for k in range(slices):
            ijk = np.column_stack([ii.ravel(), jj.ravel(), np.full(ii.size, float(k))])
            points = ijk @ matrix[:3, :3].T + matrix[:3, 3]
            near = np.einsum("ij,ij->i", points - grid.center, points - grid.center) <= reach * reach
            if np.any(near):
                flat = np.zeros(ii.size, dtype=np.uint8)
                flat[near] = wallLib.classify(result, points[near])
                labels[k] = flat.reshape(n, n)
        if n != size:   # enlarge (nearest) to the tile size
            index = np.minimum((np.arange(size) * n) // size, n - 1)
            labels = labels[:, index][:, :, index]
        wall = labels == wallLib.LABEL_MYOCARDIUM
        masks = []
        for other in (0, wallLib.LABEL_CAVITY):   # wall pixels next to the outside / to the cavity
            neighbour = np.zeros_like(wall)
            target = labels == other
            neighbour[:, 1:, :] |= target[:, :-1, :]
            neighbour[:, :-1, :] |= target[:, 1:, :]
            neighbour[:, :, 1:] |= target[:, :, :-1]
            neighbour[:, :, :-1] |= target[:, :, 1:]
            edge = wall & neighbour
            if ratio >= 1.5:   # about one logical pixel wide on scaled displays
                edge[:, 1:, :] |= edge[:, :-1, :].copy()
                edge[:, :, 1:] |= edge[:, :, :-1].copy()
            masks.append(edge)
        self.contourCache[rowKey] = (key, tuple(masks))
        return self.contourCache[rowKey][1]

    # --- polar maps ------------------------------------------------------------------
    def wallState(self, study):
        """(key, fit result, segmentation) of the wall detected in Myocardium for a study, or None."""
        if study is None:
            return None
        try:
            import MPIMyocardium
            segmentation = session.myocardiumForStudy(study)
            if segmentation is None:
                return None
            myocardiumWidget = slicer.modules.mpimyocardium.widgetRepresentation().self()
            result = myocardiumWidget.resultForStudy(study)
            if result is None:
                return None
            source = MPIMyocardium.detectionVolume(study)
            return (study.GetID(), id(result), segmentation.GetID(), source.GetID()), result, segmentation
        except Exception:
            logging.error("MPI: could not get the myocardium of this study:\n" + traceback.format_exc())
            return None

    def polarMap(self, study):
        """(polar map, message): the counts across the wall detected in Myocardium, sampled on the
        volume the wall was detected on. Cached until the detection changes."""
        if study is None:
            return None, "No study."
        try:
            import MPIMyocardium
            if session.myocardiumForStudy(study) is None:
                return None, "Detect the myocardium (step 3) to see the polar map."
            state = self.wallState(study)
            if state is None:
                return None, "No myocardium result for this study."
            key, result, _ = state
            cached = self.polarCache.get(study.GetID())
            if cached is None or cached[0] != key:
                source = MPIMyocardium.detectionVolume(study)
                cached = (key, wallLib.polarMap(result, MPIMyocardium.volumeSampler(source)))
                self.polarCache[study.GetID()] = cached
            return cached[1], ""
        except Exception:
            logging.error("MPI: could not make the polar map:\n" + traceback.format_exc())
            return None, "Could not make the polar map (see the Python console)."

    def polarWindow(self, slot):
        """(low, top) of a polar map and its 3D colors, % of the polar map maximum (DIFF: % points):
        the study's slice window unless the polar maps have their own."""
        if slot == DIFF:
            source = self.diffWindowRange
        else:
            source = self.windowRanges[slot] if self.polarLinkCheck.checked else self.polarWindowRanges[slot]
        low = float(source.minimumValue)
        return low, max(float(source.maximumValue), low + 1e-6)

    def onSliceWindowChanged(self):
        self.refreshMosaic(polar=self.polarLinkCheck.checked)

    def onPolarLinkToggled(self):
        self.updateState()
        self.refreshPolar()

    def onPolarOverlaysChanged(self):
        for tag in (POLAR_TAG, POLAR_REST_TAG, POLAR_DIFF_TAG):
            panel = existingPanel(polarPanelState(tag))
            if panel is not None:
                panel.setOverlays(self.segmentsButton.checked, self.scoresButton.checked)
        self.updateSegmentBorders()

    @staticmethod
    def polarPercent(polar, size):
        """(bullseye image, 17 segment values), both in % of the map maximum."""
        maximum = float(np.nanmax(polar["values"])) or 1.0
        image, _ = wallLib.bullseye(polar, size)
        return 100.0 * image / maximum, 100.0 * wallLib.segmentValues(polar) / maximum

    def refreshPolar(self):
        _, _, _, polars, views, _ = MODES[self.mode]
        maps = {}
        for slot in set(polars.values()) | set(views.values()):
            if slot != DIFF:
                maps[slot] = self.polarMap(self.studies.get(slot))
        for tag, slot in polars.items():
            panel = existingPanel(polarPanelState(tag))
            if panel is None:
                continue
            if slot == DIFF:
                self.refreshDiffPanel(panel, tag, maps)
            else:
                self.refreshPolarPanel(panel, tag, slot, *maps[slot])
        for tag, slot in views.items():
            self.updateSurfaceView(tag, slot, maps[slot][0])

    def refreshPolarPanel(self, panel, tag, slot, polar, message):
        title = f"{SLOT_TITLES[slot]} polar map"
        study = self.studies.get(slot)
        if study is None:
            panel.setLegend(None)
            panel.setPolar(None, None, title, message=f"No {slot} study: set the study roles in Data roles.")
            return
        if polar is None:
            panel.setLegend(None)
            panel.setPolar(None, None, title, message=message)
            return
        low, top = self.polarWindow(slot)
        table = colorTableArray(self.colormapCombo.currentText)
        own = self.mode == "stressRestPolar" or not self.polarLinkCheck.checked   # not the slice legend's window
        panel.setLegend(table if own else None, low, top)
        size = panel.polarSize()
        if not size:
            return
        key = (self.polarCache[study.GetID()][0], int(size), self.colormapCombo.currentText, low, top)
        cached = self.polarImageCache.get(tag)
        if cached is None or cached[0] != key:
            image, segments = self.polarPercent(polar, size)
            rgb = colorize(np.nan_to_num(image), top, table, low)
            rgb[np.isnan(image)] = 0
            cached = (key, (rgbToQImage(rgb), [float(v) for v in segments]))
            self.polarImageCache[tag] = cached
        qimage, segments = cached[1]
        panel.setPolar(qimage, segments, title, note=volumeText(session.myocardiumForStudy(study)))

    def refreshDiffPanel(self, panel, tag, maps):
        title = "Difference (rest − stress)"
        stress, rest = maps.get(STRESS, (None, "")), maps.get(REST, (None, ""))
        low, top = self.polarWindow(DIFF)
        table = colorTableArray(self.colormapCombo.currentText)
        if stress[0] is None or rest[0] is None:
            panel.setLegend(None)
            panel.setPolar(None, None, title, message="Needs the stress and the rest polar maps "
                                                      "(detect the myocardium of both studies).")
            return
        panel.setLegend(table, low, top)
        size = panel.polarSize()
        if not size:
            return
        key = (self.polarCache[self.studies[STRESS].GetID()][0], self.polarCache[self.studies[REST].GetID()][0],
               int(size), self.colormapCombo.currentText, low, top)
        cached = self.polarImageCache.get(tag)
        if cached is None or cached[0] != key:
            stressImage, stressSegments = self.polarPercent(stress[0], size)
            restImage, restSegments = self.polarPercent(rest[0], size)
            image = restImage - stressImage
            rgb = colorize(np.nan_to_num(image, nan=low), top, table, low)
            rgb[np.isnan(image)] = 0
            cached = (key, (rgbToQImage(rgb), [float(v) for v in restSegments - stressSegments]))
            self.polarImageCache[tag] = cached
        qimage, segments = cached[1]
        panel.setPolar(qimage, segments, title,
                       note="Rest minus stress, each in % of its own maximum: positive is lower uptake at stress "
                            "(ischemia).")

    # --- 3D uptake views ------------------------------------------------------------
    @staticmethod
    def surfaceNodes(viewTag):
        return [node for className in ("vtkMRMLModelNode", "vtkMRMLMarkupsFiducialNode")
                for node in slicer.util.getNodesByClass(className)
                if node.GetAttribute("MPI.PerfusionOf") and node.GetAttribute("MPI.PerfusionView") == viewTag]

    @staticmethod
    def setCornerText(widget, text):
        """Text in the lower left corner of a 3D view (study and volumes)."""
        try:
            view = widget.threeDView()
            corner = view.cornerAnnotation()
            corner.SetText(vtk.vtkCornerAnnotation.LowerLeft, text)
            corner.SetMaximumFontSize(14)
            corner.SetMinimumFontSize(9)
            corner.GetTextProperty().SetColor(0.9, 0.9, 0.9)
            corner.SetVisibility(True)
            view.forceRender()
        except Exception:
            logging.debug("MPI: no corner annotation in this 3D view:\n" + traceback.format_exc())

    def updateSurfaceView(self, viewTag, slot, polar):
        """The LV of one study as a solid model in a 3D view, colored by the wall counts projected
        on its surface (SSP style, same window and colors as its polar map), with the floating wall
        labels and the segment borders, seen from anterior-superior as in Myocardium."""
        widget = threeDWidgetByTag(viewTag)
        if widget is None:
            return
        try:
            import MPIMyocardium
            viewNode = widget.mrmlViewNode()
            study = self.studies.get(slot)
            state = self.wallState(study) if polar is not None else None
            frame = MPIMyocardium.cardiacFrame(study) if state is not None else None
            if state is None or frame is None:
                for node in self.surfaceNodes(viewTag):
                    slicer.mrmlScene.RemoveNode(node)
                self.surfaceKeys.pop(viewTag, None)
                self.setCornerText(widget, "")
                return
            key, result, segmentation = state
            nodes = self.surfaceNodes(viewTag)
            model = next((n for n in nodes if n.GetAttribute("MPI.Surface") == "uptake"), None)
            placeCamera = viewTag not in self.surfaceKeys
            if model is None or model.GetAttribute("MPI.WallKey") != repr(key):
                for node in nodes:
                    slicer.mrmlScene.RemoveNode(node)
                model = self.makeSurfaceModel(study, result, frame, polar, viewNode.GetID())
                model.SetAttribute("MPI.WallKey", repr(key))
                borders = self.makeSegmentBorderModel(study, result, viewNode.GetID())
                for node in [model, borders] + MPIMyocardium.addWallLabelNodes(result, study, [viewNode.GetID()]):
                    node.SetAttribute("MPI.PerfusionOf", study.GetID())
                    node.SetAttribute("MPI.PerfusionView", viewTag)
                placeCamera = True
            display = model.GetDisplayNode()
            colorNode = colorNodeForLabel(self.colormapCombo.currentText)
            if colorNode:
                display.SetAndObserveColorNodeID(colorNode.GetID())
            display.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
            display.SetScalarRange(*self.polarWindow(slot))
            self.updateSegmentBorders()
            self.setCornerText(widget, f"{SLOT_TITLES[slot]}\n{volumeText(segmentation)}")
            if placeCamera or self.surfaceKeys.get(viewTag) != key:
                self.surfaceKeys[viewTag] = key
                self.placeSurfaceCamera(viewTag, slot, key)
                # again after the layout and the new models have settled (a view can reset its camera
                # then, e.g. when a new camera node is made for it after a scene close)
                qt.QTimer.singleShot(400, lambda: self.placeSurfaceCamera(viewTag, slot, key))
                qt.QTimer.singleShot(1200, lambda: self.placeSurfaceCamera(viewTag, slot, key))
        except Exception:
            logging.error("MPI: could not make the 3D uptake view:\n" + traceback.format_exc())

    def placeSurfaceCamera(self, viewTag, slot, key):
        widget = threeDWidgetByTag(viewTag)
        study = self.studies.get(slot)
        state = self.wallState(study) if widget is not None else None
        if state is None or state[0] != key or self.surfaceKeys.get(viewTag) != key:
            return
        try:
            import MPIMyocardium
            viewNode = widget.mrmlViewNode()
            MPIMyocardium.style3DView(viewNode)   # no orientation marker: the wall labels show the orientation
            viewNode.SetRenderMode(slicer.vtkMRMLViewNode.Orthographic)
            MPIMyocardium.placeCamera(viewNode, MPIMyocardium.cardiacFrame(study), state[1],
                                      parallelScale=self.parallelScale(state[1]))
        except Exception:
            logging.warning("MPI: could not place the 3D camera:\n" + traceback.format_exc())

    @staticmethod
    def parallelScale(result):
        """Half the height of the orthographic 3D view, mm: the LV and its floating labels fit in."""
        import MPIMyocardium
        return 0.75 * (float(np.max(result["epi"])) + MPIMyocardium.LABEL_DISTANCE_MM) + 10.0

    @staticmethod
    def makeSegmentBorderModel(study, result, viewId):
        """The 17-segment borders of the polar map drawn on the epicardium of the 3D uptake view."""
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        for line in wallLib.segmentBorderLines(result):
            lines.InsertNextCell(len(line))
            for point in line:
                lines.InsertCellPoint(points.InsertNextPoint(*[float(x) for x in point]))
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetLines(lines)
        model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{study.GetName()} LV segments 3D")
        model.SetAndObservePolyData(polyData)
        model.CreateDefaultDisplayNodes()
        display = model.GetDisplayNode()
        display.SetVisibility2D(False)
        display.SetColor(1.0, 1.0, 1.0)
        display.SetOpacity(0.7)
        display.SetLineWidth(2.0)
        display.SetAmbient(1.0)
        display.SetDiffuse(0.0)
        display.SetSpecular(0.0)
        display.SetScalarVisibility(False)
        display.SetViewNodeIDs([viewId])
        model.SetAttribute("MPI.Surface", "segments")
        return model

    def updateSegmentBorders(self):
        for tag in (PERFUSION_3D_VIEW, PERFUSION_3D_VIEW_REST):
            for node in self.surfaceNodes(tag):
                if node.GetAttribute("MPI.Surface") == "segments" and node.GetDisplayNode():
                    node.GetDisplayNode().SetVisibility(bool(self.segmentsButton.checked))

    @staticmethod
    def makeSurfaceModel(study, result, frame, polar, viewId):
        import MPIMyocardium
        from vtk.util import numpy_support
        polyData = MPIMyocardium.wallShellPolyData(result, frame)
        points = numpy_support.vtk_to_numpy(polyData.GetPoints().GetData())
        sample = MPIMyocardium.volumeSampler(MPIMyocardium.detectionVolume(study))
        maximum = float(np.nanmax(polar["values"])) or 1.0
        uptake = 100.0 * wallLib.uptakeAtPoints(result, sample, points) / maximum
        scalars = numpy_support.numpy_to_vtk(uptake.astype(np.float32), deep=1)
        scalars.SetName("Uptake")
        polyData.GetPointData().AddArray(scalars)
        polyData.GetPointData().SetActiveScalars("Uptake")
        model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{study.GetName()} LV uptake 3D")
        model.SetAndObservePolyData(polyData)
        model.CreateDefaultDisplayNodes()
        display = model.GetDisplayNode()
        display.SetVisibility2D(False)
        try:
            display.SetActiveScalar("Uptake", vtk.vtkAssignAttribute.POINT_DATA)
        except AttributeError:
            display.SetActiveScalarName("Uptake")
        display.SetScalarVisibility(True)
        display.SetAmbient(1.0)    # flat colors: the counts, not the shading
        display.SetDiffuse(0.0)
        display.SetSpecular(0.0)
        display.SetViewNodeIDs([viewId])
        model.SetAttribute("MPI.Surface", "uptake")
        return model

    def onColormapChanged(self, index):
        for slot in (STRESS, REST):
            for node in self.outputs[slot].values():
                if node and slicer.mrmlScene.IsNodePresent(node):
                    setVolumeDisplay(node, self.colormapCombo.currentText)
        self.refreshMosaic()


class MPIReviewTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_ReviewFindsOutputs()

    def test_ReviewFindsOutputs(self):
        study = slicer.util.addVolumeFromArray(np.ones((8, 8, 8), dtype=np.float32), name="Rest MPI")
        self.assertEqual(session.outputsForStudy(study), {})
        for name in ORIENTATIONS:
            node = slicer.util.addVolumeFromArray(np.ones((6, 8, 8), dtype=np.float32), name=f"Rest MPI {name}")
            node.SetAttribute("MPI.Orientation", name)
            node.SetAttribute("MPI.SourceVolume", study.GetID())
            node.SetAttribute("MPI.DisplayMax", "1")
        session.setActiveStudy(study)
        widget = slicer.modules.mpireview.widgetRepresentation().self()
        widget.loadOutputs()
        self.assertIs(widget.studies[REST], study)
        self.assertEqual(set(widget.outputs[REST]), set(ORIENTATIONS))
        self.assertAlmostEqual(widget.displayReference(REST), 1.0)
        self.assertEqual(widget.firstSliceSpins[(REST, "SA")].maximum, 6)
        self.assertTrue(widget.modeAvailable("restPerf"))
        self.assertFalse(widget.modeAvailable("stressRestPerf"))
        self.assertEqual(shiftImage(np.arange(9).reshape(3, 3), 1, 0).tolist(), [[0, 0, 1], [0, 3, 4], [0, 6, 7]])
        self.delayDisplay("Review test passed")
