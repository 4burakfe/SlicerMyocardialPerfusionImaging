"""
Myocardial Perfusion Imaging - 4. Perfusion

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
Myocardium (MPIMyocardium.py). Gated QC is MPIGatedQC.py, Gated Review MPIGated.py.
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

from MyocardialPerfusionImagingLib.resampling import DEFAULT_INTERPOLATION, zoomInPlane  # noqa: E402
from MyocardialPerfusionImagingLib.views import (  # noqa: E402
    RESULT_LAYOUT_ID, SLICES_LAYOUT_ID, DUAL_RESULT_LAYOUT_ID, POLAR_LAYOUT_ID, PERF_DUAL_LAYOUT_ID,
    DUAL_PERF_DUAL_LAYOUT_ID, PERFUSION_MONITOR2_TITLE, perfusionDualMonitorLayoutXml, placeMonitor2Window,
    ORIENTATIONS, ORIENTATION_TITLES, defaultDualScreen,
    POLAR_TAG, POLAR_REST_TAG, POLAR_DIFF_TAG, PERFUSION_3D_VIEW, PERFUSION_3D_VIEW_REST,
    resultLayoutXml, slicesLayoutXml, dualResultLayoutXml, polarLayoutXml, registerLayout, threeDWidgetByTag,
    CameraLink)
from MyocardialPerfusionImagingLib.display import (  # noqa: E402
    DEFAULT_GRID_ROWS, DEFAULT_GRID_COLUMNS, COLORMAPS, colorTableArray, colorize, colorNodeForLabel,
    setVolumeDisplay, rgbToQImage, mosaicPanel, polarPanel, polarPanelState, existingPanel, DEFAULT_FOV_CM, fovCrop,
    REFIT_DELAY_MS, lastFirstSlice, cropStack, tilePixelsToIjk, bestGrid)
from MyocardialPerfusionImagingLib.geometry import worldToLocalMatrix  # noqa: E402
from MyocardialPerfusionImagingLib import session  # noqa: E402
from MyocardialPerfusionImagingLib import parameters as parametersLib  # noqa: E402
from MyocardialPerfusionImagingLib.widgets import breakable, narrowLabel, StudyPairSelector, styleButton  # noqa: E402
from MyocardialPerfusionImagingLib import myocardium as wallLib  # noqa: E402


STRESS, REST, DIFF = "stress", "rest", "diff"
# layout ID -> XML, given the (Study 1, Study 2) labels of the 3D views
LAYOUT_XML = {RESULT_LAYOUT_ID: lambda titles: resultLayoutXml(), SLICES_LAYOUT_ID: lambda titles: slicesLayoutXml(),
              DUAL_RESULT_LAYOUT_ID: dualResultLayoutXml, POLAR_LAYOUT_ID: lambda titles: polarLayoutXml(),
              PERF_DUAL_LAYOUT_ID: lambda titles: perfusionDualMonitorLayoutXml(False),
              DUAL_PERF_DUAL_LAYOUT_ID: lambda titles: perfusionDualMonitorLayoutXml(True, titles=titles)}
DUAL_MONITOR_LAYOUT_IDS = (PERF_DUAL_LAYOUT_ID, DUAL_PERF_DUAL_LAYOUT_ID)
MONITOR2_PLACE_DELAY_MS = 300   # the second window is placed after the layout manager made its views
# key: (button text, layout ID, studies of the slice rows, {polar map tag: study or DIFF}, {3D view tag: study}, tooltip)
MODES = {
    "stressPerf": ("STRESS PERF", RESULT_LAYOUT_ID, (STRESS,), {POLAR_TAG: STRESS}, {PERFUSION_3D_VIEW: STRESS},
                   "Stress study: SA / HLA / VLA slices, polar map and 3D uptake"),
    "stressSlices": ("STRESS SLICES", SLICES_LAYOUT_ID, (STRESS,), {}, {}, "Stress study: only the slices"),
    "stressPerfDual": ("STRESS DUAL", PERF_DUAL_LAYOUT_ID, (STRESS,), {POLAR_TAG: STRESS}, {PERFUSION_3D_VIEW: STRESS},
                       "Stress study on two monitors: the slices on this window, the polar map and 3D uptake "
                       "in a second window on the second monitor"),
    "restPerf": ("REST PERF", RESULT_LAYOUT_ID, (REST,), {POLAR_TAG: REST}, {PERFUSION_3D_VIEW: REST},
                 "Rest study: SA / HLA / VLA slices, polar map and 3D uptake"),
    "restSlices": ("REST SLICES", SLICES_LAYOUT_ID, (REST,), {}, {}, "Rest study: only the slices"),
    "restPerfDual": ("REST DUAL", PERF_DUAL_LAYOUT_ID, (REST,), {POLAR_TAG: REST}, {PERFUSION_3D_VIEW: REST},
                     "Rest study on two monitors: the slices on this window, the polar map and 3D uptake in a "
                     "second window on the second monitor"),
    "stressRestPerf": ("STRESS-REST PERF", DUAL_RESULT_LAYOUT_ID, (STRESS, REST),
                       {POLAR_TAG: STRESS, POLAR_REST_TAG: REST},
                       {PERFUSION_3D_VIEW: STRESS, PERFUSION_3D_VIEW_REST: REST},
                       "Stress and rest slices row by row, with both polar maps and 3D uptake views"),
    "stressRestSlices": ("STRESS-REST SLICES", SLICES_LAYOUT_ID, (STRESS, REST), {}, {},
                         "Stress and rest slices row by row"),
    "stressRestPerfDual": ("STRESS-REST DUAL", DUAL_PERF_DUAL_LAYOUT_ID, (STRESS, REST),
                           {POLAR_TAG: STRESS, POLAR_REST_TAG: REST, POLAR_DIFF_TAG: DIFF},
                           {PERFUSION_3D_VIEW: STRESS, PERFUSION_3D_VIEW_REST: REST},
                           "Stress and rest on two monitors: the slices row by row on this window; the stress, "
                           "rest and difference polar maps and both 3D uptake views on the second monitor"),
    "stressRestPolar": ("STRESS REST POLAR", POLAR_LAYOUT_ID, (), {POLAR_TAG: STRESS, POLAR_REST_TAG: REST,
                                                                   POLAR_DIFF_TAG: DIFF}, {},
                        "Stress, rest and difference (rest - stress, ischemia) polar maps"),
}
MODE_GRID = (("stressPerf", "stressSlices", "stressPerfDual"), ("restPerf", "restSlices", "restPerfDual"),
             ("stressRestPerf", "stressRestSlices", "stressRestPerfDual"), ("stressRestPolar",))   # rows of layout buttons
OVERLAY_CHOICES = (("Vessels (LAD / RCA / LCX)", "vessels"), ("17 segments", "segments"), ("None", ""))
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
        from MyocardialPerfusionImagingLib import toolbar
        toolbar.install()
    except Exception:
        logging.exception("MPI: could not install the workflow toolbar")


class MPIReview(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MPI 4 · Perfusion"
        self.parent.categories = ["Nuclear Medicine.Myocardial Perfusion"]
        self.parent.hidden = True   # opened from the Myocardial Perfusion Imaging hub and the workflow toolbar
        self.parent.dependencies = []
        self.parent.contributors = ["Burak Demir"]
        self.parent.helpText = (
            "Review the reoriented slices of the stress and rest studies (roles set in Data roles), "
            "alone or row by row. Mouse wheel over a row scrolls it. Polar maps and 3D views of the LV "
            "with the wall counts on its surface use the wall detected in Myocardium; STRESS REST POLAR "
            "adds the rest - stress difference.")
        self.parent.acknowledgementText = ""
        iconPath = os.path.join(_MODULE_DIR, "Resources", "Icons", "MyocardialPerfusionImaging.png")
        if os.path.exists(iconPath):
            self.parent.icon = qt.QIcon(iconPath)
        if not slicer.app.commandOptions().noMainWindow:
            slicer.app.connect("startupCompleted()", _installToolbar)


class MPIReviewWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.studies = {STRESS: None, REST: None}   # static volume of each study shown
        self.slotLabels = []   # (QLabel, slot, template) of the panel labels naming a study by its role
        self.outputs = {STRESS: {}, REST: {}}       # its reoriented volumes {orientation: node}
        self.mode = "stressPerf"
        self.chosenMode = None      # (layout the user picked, studies ready then): kept while those do not change
        self.grids = {"single": [DEFAULT_GRID_ROWS, DEFAULT_GRID_COLUMNS], "dual": [1, DEFAULT_GRID_COLUMNS]}
        self.polarCache = {}         # study ID -> (wall key, polar map)
        self.tileCache = {}          # row key -> (key, [QImage per slice]): rendered once, scrolling reuses them
        self.contourCache = {}       # row key -> (key, (epi, endo) bool masks [slices, size, size])
        self.polarImageCache = {}    # polar map tag -> (key, (QImage, segment values))
        self.centerCache = {}        # study ID -> (wall key, myocardium mass center in world coordinates)
        self.surfaceKeys = {}        # 3D view tag -> wall its uptake model and camera were made for
        self.extentCache = {}        # (slot, orientation) -> (key, (first, last) slice with myocardium or None)
        self.cameraLink = CameraLink()   # stress and rest 3D views moved together
        self.scrollOffsets = {}      # (slot, orientation) -> slices the mouse wheel moved a fitted row

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.studySelector = StudyPairSelector(self.layout)
        self.studyLabel = narrowLabel(qt.QLabel("No study selected."))
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
            form.addRow(self._slotLabel(slot, "{title} window:"), self.windowRanges[slot])

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
        self.autoFitCheck = qt.QCheckBox("Fit to the myocardium")
        self.autoFitCheck.checked = True
        self.autoFitCheck.toolTip = ("Show every slice with myocardium on the grid giving the largest tiles for the "
                                     "screen (fewest rows), each row filled with the neighbouring slices as far as "
                                     "the tiles fit (set again when the view is resized). "
                                     "Changing the grid or the slices by hand turns it off")
        form.addRow("", self.autoFitCheck)
        self.fovSpin = qt.QDoubleSpinBox()
        self.fovSpin.setRange(4.0, 30.0)
        self.fovSpin.setSingleStep(0.5)
        self.fovSpin.setDecimals(1)
        self.fovSpin.setSuffix(" cm")
        self.fovSpin.setValue(DEFAULT_FOV_CM)
        self.fovSpin.toolTip = ("Field of view of every tile, centered on the LV: the same for stress and rest, so "
                                "both show at the same zoom")
        form.addRow("Field of view:", self.fovSpin)

        showRow = qt.QHBoxLayout()
        self.contoursButton = self._toggleButton("Myocardium contours", False,
                                                 "Show the epicardium (green) and endocardium (blue) detected in "
                                                 "Myocardium on the slices")
        self.syncButton = self._toggleButton("Sync stress and rest", True,
                                             "Stress and rest rows show the same place: slices and panning matched "
                                             "on the center of each study's segmented myocardium, scrolling moves "
                                             "both. Off: each row scrolls on its own and is not moved")
        self.link3DButton = self._toggleButton("Link 3D views", True,
                                               "Rotating, zooming or panning the stress or rest 3D view moves the "
                                               "other the same way, around its own heart")
        showRow.addWidget(self.contoursButton)
        showRow.addWidget(self.syncButton)
        showRow.addWidget(self.link3DButton)
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
            polarForm.addRow(self._slotLabel(slot, "{title}:"), self.polarWindowRanges[slot])
        self.diffWindowRange = self._windowWidget(
            "Color window of the difference map (rest - stress, each in % of its own maximum): positive "
            "values are lower uptake at stress (ischemia)", *DIFF_WINDOW)
        polarForm.addRow("Difference:", self.diffWindowRange)
        overlayRow = qt.QHBoxLayout()
        self.overlayCombo = qt.QComboBox()
        for text, value in OVERLAY_CHOICES:
            self.overlayCombo.addItem(text, value)
        self.overlayCombo.toolTip = ("Regions drawn on the polar maps and the 3D uptake views: the coronary "
                                     "territories (LAD, RCA, LCX, from the 17 segments), the 17 segments, or none")
        self.scoresButton = self._toggleButton("Scores", True,
                                               "Show the region values (mean counts, % of the map maximum)")
        overlayRow.addWidget(self.overlayCombo, 1)
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
                navigationForm.addRow(self._slotLabel(slot, "First {lower} " + name + " slice:"), spin)

        self.backButton = qt.QPushButton("Back to reorientation")
        self.layout.addWidget(self.backButton)
        self.gatedQCButton = styleButton(qt.QPushButton("Gated QC ›"), "primary")
        self.gatedQCButton.toolTip = "Next step: segment and check the gate frames of the gated studies"
        self.layout.addWidget(self.gatedQCButton)

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
                grid.addWidget(button, row, column * (6 // len(keys)), 1, 6 // len(keys))
                self.layoutGroup.addButton(button)
                self.layoutButtons[key] = button
        self.layout.addWidget(layoutBox)
        self.parameterPanel = parametersLib.ParameterPanel(
            "perfusion", self.layout, onChanged=self.onParameterChanged)
        self.parameterPanel.bind("colormap", self.colormapCombo)
        self.parameterPanel.bind("stressWindow", self.windowRanges[STRESS])
        self.parameterPanel.bind("restWindow", self.windowRanges[REST])
        self.parameterPanel.bind("gridRows", self.gridRowsSpin)
        self.parameterPanel.bind("gridColumns", self.gridColumnsSpin)
        self.parameterPanel.bind("autoFit", self.autoFitCheck)
        self.parameterPanel.bind("fov", self.fovSpin)
        self.parameterPanel.bind("contours", self.contoursButton)
        self.parameterPanel.bind("sync", self.syncButton)
        self.parameterPanel.bind("link3D", self.link3DButton)
        self.parameterPanel.bind("polarLink", self.polarLinkCheck)
        self.parameterPanel.bind("stressPolarWindow", self.polarWindowRanges[STRESS])
        self.parameterPanel.bind("restPolarWindow", self.polarWindowRanges[REST])
        self.parameterPanel.bind("diffWindow", self.diffWindowRange)
        self.parameterPanel.bind("overlay", self.overlayCombo)
        self.parameterPanel.bind("scores", self.scoresButton)
        self.parameterPanel.bind("sliceStep", self.stepSpin)
        self.layout.addStretch(1)

        self.mosaicResizeTimer = qt.QTimer()
        self.mosaicResizeTimer.setSingleShot(True)
        self.mosaicResizeTimer.setInterval(REFIT_DELAY_MS)

        self.mosaicResizeTimer.connect("timeout()", self.refreshMosaic)
        self.colormapCombo.connect("currentIndexChanged(int)", self.onColormapChanged)
        for slot in (STRESS, REST):
            self.windowRanges[slot].connect("valuesChanged(double,double)", lambda a, b: self.onSliceWindowChanged())
            self.polarWindowRanges[slot].connect("valuesChanged(double,double)", lambda a, b: self.refreshPolar())
        self.diffWindowRange.connect("valuesChanged(double,double)", lambda a, b: self.refreshPolar())
        self.polarLinkCheck.connect("toggled(bool)", lambda _: self.onPolarLinkToggled())
        self.gridRowsSpin.connect("valueChanged(int)", lambda _: self.onGridChanged(manual=True))
        self.gridColumnsSpin.connect("valueChanged(int)", lambda _: self.onGridChanged(manual=True))
        self.autoFitCheck.connect("toggled(bool)", lambda _: self.onAutoFitToggled())
        self.fovSpin.connect("valueChanged(double)", lambda _: self.refreshMosaic(polar=False))
        self.contoursButton.connect("toggled(bool)", lambda _: self.refreshMosaic(polar=False))
        self.syncButton.connect("toggled(bool)", lambda _: self.onSyncToggled())
        self.link3DButton.connect("toggled(bool)", lambda _: self.updateCameraLink())
        self.overlayCombo.connect("currentIndexChanged(int)", lambda _: self.onPolarOverlaysChanged())
        self.scoresButton.connect("toggled(bool)", lambda _: self.onPolarOverlaysChanged())
        self.stepSpin.connect("valueChanged(int)", lambda _: self.onSlicesChangedByHand())
        for spin in self.firstSliceSpins.values():
            spin.connect("valueChanged(int)", lambda _: self.onSlicesChangedByHand())
        for key, button in self.layoutButtons.items():
            button.connect("clicked(bool)", lambda _, k=key: self.chooseMode(k))
        self.backButton.connect("clicked(bool)", lambda: session.selectStep("processing"))
        self.gatedQCButton.connect("clicked(bool)", lambda: session.selectStep("gatedqc"))
        self.parameterPanel.loadSaved()   # saved settings of this step, as if set by hand

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneClosed)
        session.addListener(self.onSessionChanged)
        _installToolbar()
        self.loadOutputs()

    def _slotLabel(self, slot, template):
        """Form label naming Study 1 / Study 2 by its role ({title} / {lower} in the template), kept up to date
        by updateState."""
        label = qt.QLabel()
        self.slotLabels.append((label, slot, template))
        return label

    def slotTitle(self, slot):
        """Label of Study 1 / Study 2 in the views and plots: its role in Data roles ("STRESS", "VIABILITY")."""
        return session.slotTitles(self.studies)[slot].upper()

    def modeText(self, key):
        """Layout button text with the roles of the studies ("STRESS-REST PERF", "REST-VIABILITY PERF")."""
        text = MODES[key][0]
        first, second = self.slotTitle(STRESS), self.slotTitle(REST)
        for prefix, label in (("STRESS-REST ", f"{first}-{second} "), ("STRESS REST ", f"{first} {second} "),
                              ("STRESS ", f"{first} "), ("REST ", f"{second} ")):
            if text.startswith(prefix):
                return label + text[len(prefix):]
        return text

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

    # --- workflow toolbar (second row) -----------------------------------------------------------------
    def buildToolbar(self, bar):
        """Most used controls in the toolbar (toolbar.StepBar); they drive the controls of this panel."""
        self.toolbarBar = bar
        self.tbMax = bar.spin("Max", 10, 150, 5, " %", "Top of the slice color window (% of the myocardial maximum) "
                              "of the studies shown", self.onToolbarMax)
        bar.mirrorCombo(self.colormapCombo, toolTip="Color map of the slices and polar maps")
        self.tbFov = bar.spin("FOV", 4.0, 30.0, 0.5, " cm", self.fovSpin.toolTip,
                              lambda v: self.fromToolbar(lambda: self.fovSpin.setValue(v)), decimals=1)
        bar.separator()
        self.tbContours = bar.toggle("Contours", self.contoursButton.toolTip,
                                     lambda c: self.fromToolbar(lambda: setattr(self.contoursButton, "checked", c)))
        self.tbSync = bar.toggle("Sync", self.syncButton.toolTip,
                                 lambda c: self.fromToolbar(lambda: setattr(self.syncButton, "checked", c)))
        bar.separator()
        self.tbOverlay = bar.choice([("Vessels", "vessels", "Coronary territories (LAD / RCA / LCX) on the polar maps"),
                                     ("17 seg", "segments", "The 17 segments on the polar maps"),
                                     ("None", "", "No segmentation on the polar maps")],
                                    self.onToolbarOverlay)
        self.tbScores = bar.toggle("Scores", self.scoresButton.toolTip,
                                   lambda c: self.fromToolbar(lambda: setattr(self.scoresButton, "checked", c)))
        bar.separator()
        self.tbLayouts = bar.buttons([(key, self.shortLayoutName(key), MODES[key][5]) for row in MODE_GRID
                                      for key in row], lambda key: self.fromToolbar(lambda: self.chooseMode(key)))
        bar.separator()
        bar.mirror(self.gatedQCButton, "Gated QC ›", kind="primary")
        if not getattr(self, "_toolbarFollows", False):   # the toolbar follows this panel's controls
            self._toolbarFollows = True
            for slot in (STRESS, REST):
                self.windowRanges[slot].connect("valuesChanged(double,double)", lambda a, b: self.syncToolbar())
            self.fovSpin.connect("valueChanged(double)", lambda _: self.syncToolbar())
            for button in (self.contoursButton, self.syncButton, self.scoresButton):
                button.connect("toggled(bool)", lambda _: self.syncToolbar())
            self.overlayCombo.connect("currentIndexChanged(int)", lambda _: self.syncToolbar())

    @staticmethod
    def shortLayoutName(key):
        text = MODES[key][0]
        for prefix in ("STRESS-REST ", "STRESS REST ", "STRESS ", "REST "):
            if text.startswith(prefix):
                return text[len(prefix):]
        return text

    def toolbarLayouts(self):
        """Layouts offered in the toolbar: the stress-rest ones when both studies are ready, else those of the
        one study that is."""
        if self.ready(STRESS) and self.ready(REST):
            return [key for row in MODE_GRID[2:] for key in row]
        if self.ready(STRESS):
            return list(MODE_GRID[0])
        if self.ready(REST):
            return list(MODE_GRID[1])
        return []

    def toolbarSlots(self):
        _, _, slots, polars, views, _ = MODES[self.mode]
        shown = set(slots) | {s for s in list(polars.values()) + list(views.values()) if s != DIFF}
        if DIFF in polars.values():
            shown |= {STRESS, REST}
        return [slot for slot in (STRESS, REST) if slot in shown and self.ready(slot)] or \
            [slot for slot in (STRESS, REST) if self.ready(slot)]

    def fromToolbar(self, action):
        action()
        self.syncToolbar()   # e.g. a click on the checked overlay button changed nothing: check it again

    def onToolbarMax(self, value):
        for slot in self.toolbarSlots():
            window = self.windowRanges[slot]
            window.maximumValue = max(float(value), window.minimumValue + 1.0)
        self.syncToolbar()

    def onToolbarOverlay(self, value):
        index = self.overlayCombo.findData(value)
        if index >= 0:
            self.overlayCombo.setCurrentIndex(index)
        self.syncToolbar()

    def syncToolbar(self):
        bar = getattr(self, "toolbarBar", None)
        if bar is None:
            return
        try:
            bar.syncMirrors()
            slots = self.toolbarSlots()
            anyReady = bool(slots)
            if slots:
                bar.setValue(self.tbMax, self.windowRanges[slots[0]].maximumValue)
            bar.setValue(self.tbFov, self.fovSpin.value)
            bar.setChecked(self.tbContours, self.contoursButton.checked)
            bar.setChecked(self.tbSync, self.syncButton.checked)
            bar.setChoice(self.tbOverlay, self.overlay() or "")
            bar.setChecked(self.tbScores, self.scoresButton.checked)
            for widget in [self.tbMax, self.tbFov, self.tbContours, self.tbScores] + list(self.tbOverlay.values()):
                widget.enabled = anyReady
            self.tbSync.enabled = self.syncButton.enabled
            offered = self.toolbarLayouts()
            for key, button in self.tbLayouts.items():
                button.setVisible(key in offered)
                button.enabled = self.modeAvailable(key)
                bar.setChecked(button, key == self.mode)
        except Exception:
            logging.debug("MPI: could not update the toolbar:\n" + traceback.format_exc())

    def enter(self):
        self.studySelector.refresh()
        self.loadOutputs()
        self.showResults(self.preferredMode())

    # --- default layout ---------------------------------------------------------------
    def readyStudies(self):
        return tuple(slot for slot in (STRESS, REST) if self.ready(slot))

    def defaultMode(self):
        """Slices and plots of the studies there: stress-rest when both are ready, on two monitors when there
        is a second screen and the hub setting allows it."""
        dual = defaultDualScreen()
        ready = self.readyStudies()
        if len(ready) == 2:
            return "stressRestPerfDual" if dual else "stressRestPerf"
        if ready:
            return f"{ready[0]}PerfDual" if dual else f"{ready[0]}Perf"
        return self.mode

    def preferredMode(self):
        """The layout the user picked while the same studies are ready, else the default one."""
        if self.chosenMode is not None:
            mode, ready = self.chosenMode
            if ready == self.readyStudies() and self.modeAvailable(mode):
                return mode
        return self.defaultMode()

    def chooseMode(self, mode):
        self.chosenMode = (mode, self.readyStudies())
        self.showResults(mode)

    def cleanup(self):
        self.mosaicResizeTimer.stop()
        self.cameraLink.unlink()
        session.removeListener(self.onSessionChanged)
        self.studySelector.cleanup()
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
            if self.parent.isVisible():
                self.showResults(self.preferredMode())

    def onSceneClosed(self, caller=None, event=None):
        self.cameraLink.unlink()   # its anchors and cameras belong to the closed studies
        self.studies = {STRESS: None, REST: None}
        self.outputs = {STRESS: {}, REST: {}}
        self.tileCache = {}
        self.contourCache = {}
        self.polarCache = {}
        self.polarImageCache = {}
        self.centerCache = {}
        self.extentCache = {}
        self.scrollOffsets = {}
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
            if study is not None and not self.ready(slot):
                lines.append(f"<b>{breakable(study.GetName())}</b> has no reoriented volumes yet (create them in Processing)")
        if self.studies[STRESS] is None and self.studies[REST] is None:
            lines = ["No study selected. Pick Study 1 / Study 2 above or set the roles in Data roles."]
        self.studyLabel.text = "<br>".join(lines)
        anyReady = self.ready(STRESS) or self.ready(REST)
        for widget in [self.gridRowsSpin, self.gridColumnsSpin, self.stepSpin, self.contoursButton,
                       self.polarLinkCheck, self.overlayCombo, self.scoresButton]:
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
            button.text = self.modeText(key)
        titles = session.slotTitles(self.studies)
        for label, slot, template in self.slotLabels:
            label.text = template.format(title=titles[slot], lower=titles[slot].lower())
        self.syncButton.text = f"Sync {titles[STRESS].lower()} and {titles[REST].lower()}"
        self.syncToolbar()

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

    def onAutoFitToggled(self):
        self.scrollOffsets = {}
        self.refreshMosaic(polar=False)

    def onSlicesChangedByHand(self):
        wasBlocked = self.autoFitCheck.blockSignals(True)
        self.autoFitCheck.checked = False
        self.autoFitCheck.blockSignals(wasBlocked)
        self.refreshMosaic(polar=False)

    def onGridChanged(self, manual=False):
        if manual:
            wasBlocked = self.autoFitCheck.blockSignals(True)
            self.autoFitCheck.checked = False
            self.autoFitCheck.blockSignals(wasBlocked)
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
            spin.setRange(1, lastFirstSlice(n))   # past the end, until a few slices are left
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
        return [(f"{slot}:{name}", self.rowTitle(slot, name))
                for name in ORIENTATIONS for slot in slots]

    def rowTitle(self, slot, name):
        """Role, volume name and orientation of a slice row: "STRESS: Stress OSEM AC  ·  Short axis (...)"."""
        study = self.studies.get(slot)
        return self.slotTitle(slot) + (f": {study.GetName()}" if study is not None else "") + \
            f"  ·  {ORIENTATION_TITLES[name]}"

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
            mode = self.defaultMode()
            if not self.modeAvailable(mode):
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
            panel.pairRows = len(slots) > 1     # each grid row of stress followed by that of rest
            panel.gridRows, panel.gridColumns = self.gridRowsSpin.value, self.gridColumnsSpin.value
            panel.setRowKeys(self.rowKeys(mode))
        for tag in polars:
            polar = polarPanel(tag)
            polar.onResized = self.mosaicResizeTimer.start
            polar.overlay, polar.showScores = self.overlay(), self.scoresButton.checked
        titles = session.slotTitles(self.studies)
        registerLayout(layoutId, LAYOUT_XML[layoutId]((titles[STRESS], titles[REST])))
        slicer.app.layoutManager().setLayout(layoutId)
        slicer.app.processEvents()
        if layoutId in DUAL_MONITOR_LAYOUT_IDS:
            qt.QTimer.singleShot(MONITOR2_PLACE_DELAY_MS,
                                 lambda: placeMonitor2Window(PERFUSION_3D_VIEW, PERFUSION_MONITOR2_TITLE))
        self.surfaceKeys = {}   # place the 3D cameras again
        self.refreshMosaic()

    def onMosaicScroll(self, key, direction):
        if ":" not in key:
            return
        slot, name = key.split(":", 1)
        if self.synced():
            slot = STRESS   # the rest rows follow
        if self.autoFitCheck.checked:   # keep fitting (to the viewport size) with the row moved
            self.scrollOffsets[(slot, name)] = self.scrollOffsets.get((slot, name), 0) + direction
            self.refreshMosaic(polar=False)
            return
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

    def fovMm(self):
        return 10.0 * float(self.fovSpin.value)

    def tileCrop(self, slot, name, node):
        """Window of a row's slices shown in the tiles (fovCrop): the field of view around the LV center
        of the study, so stress and rest show the same size at the same place."""
        center = self.centerIjk(slot, name)
        array = slicer.util.arrayFromVolume(node)
        return fovCrop(array.shape, node.GetSpacing()[:2], None if center is None else center[:2], self.fovMm())

    def myocardiumSlices(self, slot, name, node=None):
        """(first, last) slice index (0-based) of a reoriented volume holding the myocardium detected in
        Myocardium, or None. Made once per wall and volume."""
        node = node or self.outputs[slot].get(name)
        state = self.wallState(self.studies.get(slot))
        if node is None or state is None:
            return None
        key = (state[0], node.GetID())
        cached = self.extentCache.get((slot, name))
        if cached and cached[0] == key:
            return cached[1]
        result = state[1]
        array = slicer.util.arrayFromVolume(node)
        slices, h, w = array.shape
        n = 48
        ii, jj = tilePixelsToIjk(n, None, array.shape)
        ijkToRas = vtk.vtkMatrix4x4()
        node.GetIJKToRASMatrix(ijkToRas)
        localToWorld = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(worldToLocalMatrix(node), localToWorld)
        matrix = slicer.util.arrayFromVTKMatrix(localToWorld) @ slicer.util.arrayFromVTKMatrix(ijkToRas)
        grid = result["grid"]
        reach = wallLib.wallReach(result) + 3.0
        found = []
        for k in range(slices):
            points = np.column_stack([ii.ravel(), jj.ravel(), np.full(ii.size, float(k))]) @ matrix[:3, :3].T + matrix[:3, 3]
            near = np.einsum("ij,ij->i", points - grid.center, points - grid.center) <= reach * reach
            if np.any(near) and np.any(wallLib.classify(result, points[near]) == wallLib.LABEL_MYOCARDIUM):
                found.append(k)
        extent = (found[0], found[-1]) if found else None
        self.extentCache[(slot, name)] = (key, extent)
        return extent

    def autoFit(self, panel, slots):
        """Fit to the myocardium: the grid and first slices showing every slice with myocardium at the
        largest tile size the rows' area allows, the rows filled with neighbouring slices (bestGrid)."""
        rows = {}
        for slot in slots:
            if self.synced() and slot == REST:
                continue    # the rest rows follow the stress rows
            for name in ORIENTATIONS:
                node = self.outputs[slot].get(name)
                extent = self.myocardiumSlices(slot, name)
                if node is None or extent is None:
                    continue
                count = slicer.util.arrayFromVolume(node).shape[0]
                rows[(slot, name)] = (extent[0], extent[1], count)
        width, height = panel.rowArea()
        if not rows or width < 20 or height < 20:
            return
        needed = max(last - first + 1 for first, last, _ in rows.values())
        gridRows, gridColumns = bestGrid(needed, width, height, rowOverhead=panel.stripOverhead())
        cells = gridRows * gridColumns
        widgets = [self.gridRowsSpin, self.gridColumnsSpin, self.stepSpin] + list(self.firstSliceSpins.values())
        blocked = [w.blockSignals(True) for w in widgets]
        try:
            self.gridRowsSpin.setValue(gridRows)
            self.gridColumnsSpin.setValue(gridColumns)
            self.stepSpin.setValue(1)
            for (slot, name), (first, last, count) in rows.items():
                centered = first - (cells - (last - first + 1)) // 2    # the myocardium in the middle of the grid
                start = centered + self.scrollOffsets.get((slot, name), 0)
                start = int(min(max(start, 0), lastFirstSlice(count) - 1))   # past the end: black tiles
                self.scrollOffsets[(slot, name)] = start - centered     # no scrolling past the ends
                self.firstSliceSpins[(slot, name)].setValue(start + 1)
        finally:
            for w, b in zip(widgets, blocked):
                w.blockSignals(b)
        self.grids[self.gridKind()] = [gridRows, gridColumns]

    # --- rendering --------------------------------------------------------------------
    def onParameterChanged(self, key):
        """An Advanced parameter changed (None: all were restored): tiles and contours drawn again."""
        self.tileCache = {}
        self.contourCache = {}
        self.refreshMosaic()

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
        panel.pairRows = len(slots) > 1
        panel.setRowKeys(self.rowKeys())
        if self.autoFitCheck.checked:
            self.autoFit(panel, slots)
        panel.gridRows, panel.gridColumns = self.gridRowsSpin.value, self.gridColumnsSpin.value
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
                    # each study cropped around its own LV center: stress and rest line up without panning
                    images = self.sliceImages(key, slot, node, size, self.sliceWindow(slot), colorTable,
                                              panel.pixelRatio(), crop=self.tileCrop(slot, name, node))
                tiles = [(images[k], k + 1) if images and 0 <= k < count else (None, None) for k in shown]
                valid = [k for k in shown if 0 <= k < count]
                text = self.rowTitle(slot, name)
                if valid:
                    text += f"   —   slices {valid[0] + 1}–{valid[-1] + 1} of {count}" + (f", every {step}" if step > 1 else "")
                panel.setRow(key, tiles, text)
        dual = len(slots) > 1
        panel.setLegends([(colorTable, self.windowRanges[slot].minimumValue, self.windowRanges[slot].maximumValue,
                           self.slotTitle(slot) if dual else "") for slot in slots])

    def sliceImages(self, rowKey, slot, node, size, window, colorTable, ratio=1.0, shift=(0, 0), crop=None):
        """Every slice of one row rendered at the tile size (resized in the data domain, then colour
        mapped, so Qt draws them 1:1), cut to crop (fovCrop; None: the whole slice), with the myocardium
        contours when they are on, moved by shift tile pixels. Made once per volume, tile size, colour
        map, window, contours, crop and shift; scrolling only picks tiles from this list."""
        imageData = node.GetImageData()
        contours = self.contourMasks(rowKey, slot, node, size, ratio, crop) if self.contoursButton.checked else None
        key = (node.GetID(), imageData.GetMTime() if imageData else 0, int(size), self.colormapCombo.currentText,
               tuple(round(float(v), 6) for v in window),
               node.GetAttribute("MPI.Interpolation") or DEFAULT_INTERPOLATION,
               self.contourCache.get(rowKey, (None,))[0] if contours is not None else None, tuple(shift),
               None if crop is None else tuple(crop))
        cached = self.tileCache.get(rowKey)
        if cached and cached[0] == key:
            return cached[1]
        interpolation = key[5]
        array = slicer.util.arrayFromVolume(node)  # [k, j, i]; i = screen right, j = screen up
        if crop is not None:
            array = cropStack(array, crop)
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
    def contourMasks(self, rowKey, slot, node, size, ratio=1.0, crop=None):
        """(epicardium, endocardium) contour masks [slices, size, size] of one row, from the wall
        detected in Myocardium, classified at the tiles' own pixels (smooth at any zoom).
        None when the myocardium was not detected."""
        state = self.wallState(self.studies.get(slot))
        if state is None:
            return None
        wallKey, result = state[0], state[1]
        key = (wallKey, node.GetID(), int(size), round(float(ratio), 2), None if crop is None else tuple(crop))
        cached = self.contourCache.get(rowKey)
        if cached and cached[0] == key:
            return cached[1]
        array = slicer.util.arrayFromVolume(node)
        slices, h, w = array.shape
        n = int(min(size, CONTOUR_MAX_SIZE))
        # tile pixel centers -> volume (i, j): the same mapping as cropStack + zoomInPlane, rows flipped
        ii, jj = tilePixelsToIjk(n, crop, array.shape)
        ijkToRas = vtk.vtkMatrix4x4()
        node.GetIJKToRASMatrix(ijkToRas)
        localToWorld = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(worldToLocalMatrix(node), localToWorld)
        matrix = slicer.util.arrayFromVTKMatrix(localToWorld) @ slicer.util.arrayFromVTKMatrix(ijkToRas)
        grid = result["grid"]
        reach = wallLib.wallReach(result) + 3.0     # only pixels this close to the LV center can be wall
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
                panel.setOverlays(self.overlay(), self.scoresButton.checked)
        self.updateSegmentBorders()

    def overlay(self):
        """Region overlay chosen: "vessels", "segments" or None."""
        return self.overlayCombo.itemData(self.overlayCombo.currentIndex) or None

    @staticmethod
    def polarPercent(polar, size):
        """(bullseye image, {overlay: region values}), all in % of the map maximum."""
        maximum = float(np.nanmax(polar["values"])) or 1.0
        image, _ = wallLib.bullseye(polar, size)
        return 100.0 * image / maximum, {overlay: 100.0 * wallLib.regionValues(polar, overlay) / maximum
                                         for overlay in wallLib.OVERLAYS}

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
        self.updateCameraLink()

    def refreshPolarPanel(self, panel, tag, slot, polar, message):
        study = self.studies.get(slot)
        title = f"{self.slotTitle(slot)} polar map" + (f": {study.GetName()}" if study is not None else "")
        if study is None:
            panel.setLegend(None)
            panel.setPolar(None, None, title, message=f"No {session.SLOT_FALLBACK_TITLES[slot]}: pick it above or "
                                                      "set the study roles in Data roles.")
            return
        if polar is None:
            panel.setLegend(None)
            panel.setPolar(None, None, title, message=message)
            return
        low, top = self.polarWindow(slot)
        table = colorTableArray(self.colormapCombo.currentText)
        # its own color bar when the slice legend is not on the same screen or shows another window
        own = (self.mode == "stressRestPolar" or MODES[self.mode][1] in DUAL_MONITOR_LAYOUT_IDS
               or not self.polarLinkCheck.checked)
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
            cached = (key, (rgbToQImage(rgb), {k: [float(v) for v in values] for k, values in segments.items()}))
            self.polarImageCache[tag] = cached
        qimage, segments = cached[1]
        panel.setPolar(qimage, segments, title, note=volumeText(session.myocardiumForStudy(study)),
                       corner=self.tidText() if slot == STRESS else "")

    def tidText(self):
        """'TID 1.08' (transient ischemic dilation: stress / rest LV cavity volume of Myocardium) when the
        layout shows both studies and both have their cavity volume, else ''."""
        if REST not in MODES[self.mode][3].values() and REST not in MODES[self.mode][4].values():
            return ""
        if not session.isStressRest(self.studies):
            return ""   # e.g. rest and FDG viability: no stress dilation
        volumes = []
        for slot in (STRESS, REST):
            study = self.studies.get(slot)
            segmentation = session.myocardiumForStudy(study) if study is not None else None
            try:
                volumes.append(float(segmentation.GetAttribute("MPI.LVCavityMlExact")
                                     or segmentation.GetAttribute("MPI.LVCavityMl")))
            except (AttributeError, TypeError, ValueError):
                return ""
        if volumes[1] <= 0:
            return ""
        return f"TID {volumes[0] / volumes[1]:.2f}"

    def refreshDiffPanel(self, panel, tag, maps):
        first, second = self.slotTitle(STRESS).lower(), self.slotTitle(REST).lower()
        title = f"Difference ({second} − {first})"
        stress, rest = maps.get(STRESS, (None, "")), maps.get(REST, (None, ""))
        low, top = self.polarWindow(DIFF)
        table = colorTableArray(self.colormapCombo.currentText)
        if stress[0] is None or rest[0] is None:
            panel.setLegend(None)
            panel.setPolar(None, None, title, message=f"Needs the {first} and the {second} polar maps "
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
            cached = (key, (rgbToQImage(rgb), {k: [float(v) for v in restSegments[k] - stressSegments[k]]
                                               for k in wallLib.OVERLAYS}))
            self.polarImageCache[tag] = cached
        qimage, segments = cached[1]
        note = (f"{second.capitalize()} minus {first}, each in % of its own maximum: positive is lower uptake in "
                f"the {first} study")
        panel.setPolar(qimage, segments, title,
                       note=note + (" (ischemia)." if session.isStressRest(self.studies) else "."))

    # --- 3D uptake views ------------------------------------------------------------
    @staticmethod
    def surfaceNodes(viewTag):
        return [node for className in ("vtkMRMLModelNode", "vtkMRMLMarkupsFiducialNode")
                for node in slicer.util.getNodesByClass(className)
                if node.GetAttribute("MPI.PerfusionOf") and node.GetAttribute("MPI.PerfusionView") == viewTag]

    @staticmethod
    def setCornerText(widget, text, upperLeft=""):
        """Text in the lower left corner of a 3D view (study and volumes), and in the upper left one."""
        try:
            view = widget.threeDView()
            corner = view.cornerAnnotation()
            corner.SetText(vtk.vtkCornerAnnotation.LowerLeft, text)
            corner.SetText(vtk.vtkCornerAnnotation.UpperLeft, upperLeft)
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
                borders = [self.makeSegmentBorderModel(study, result, viewNode.GetID(), overlay)
                           for overlay in wallLib.OVERLAYS]
                for node in [model] + borders + MPIMyocardium.addWallLabelNodes(result, study, [viewNode.GetID()]):
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
            self.setCornerText(widget, f"{self.slotTitle(slot)}: {study.GetName()}\n{volumeText(segmentation)}",
                               self.tidText() if slot == STRESS else "")
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
        self.updateCameraLink()

    def updateCameraLink(self):
        """Link the stress and rest 3D views of the layout (Link 3D views), each around its own heart."""
        anchors = {}
        try:
            import MPIMyocardium
            if self.link3DButton.checked:
                for tag, slot in MODES[self.mode][4].items():
                    study = self.studies.get(slot)
                    state = self.wallState(study) if study is not None else None
                    frame = MPIMyocardium.cardiacFrame(study) if state is not None else None
                    if frame is not None and threeDWidgetByTag(tag) is not None:
                        anchors[tag] = MPIMyocardium.cameraAnchor(frame, state[1])
        except Exception:
            logging.warning("MPI: could not link the 3D views:\n" + traceback.format_exc())
        self.cameraLink.link(anchors)

    @staticmethod
    def parallelScale(result):
        """Half the height of the orthographic 3D view, mm: the LV and its floating labels fit in."""
        import MPIMyocardium
        return 0.75 * (wallLib.wallReach(result) + MPIMyocardium.LABEL_DISTANCE_MM) + 10.0

    @staticmethod
    def makeSegmentBorderModel(study, result, viewId, overlay="segments"):
        """The region borders of the polar map (17 segments or the vessel territories) drawn on the
        epicardium of the 3D uptake view."""
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        for line in wallLib.segmentBorderLines(result, overlay=overlay):
            lines.InsertNextCell(len(line))
            for point in line:
                lines.InsertCellPoint(points.InsertNextPoint(*[float(x) for x in point]))
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        polyData.SetLines(lines)
        name = "territories" if overlay == "vessels" else "segments"
        model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{study.GetName()} LV {name} 3D")
        model.SetAndObservePolyData(polyData)
        model.CreateDefaultDisplayNodes()
        display = model.GetDisplayNode()
        display.SetVisibility2D(False)
        display.SetColor(1.0, 1.0, 1.0)
        display.SetOpacity(0.85 if overlay == "vessels" else 0.7)
        display.SetLineWidth(3.0 if overlay == "vessels" else 2.0)
        display.SetAmbient(1.0)
        display.SetDiffuse(0.0)
        display.SetSpecular(0.0)
        display.SetScalarVisibility(False)
        display.SetViewNodeIDs([viewId])
        model.SetAttribute("MPI.Surface", "segments")
        model.SetAttribute("MPI.Overlay", overlay)
        return model

    def updateSegmentBorders(self):
        for tag in (PERFUSION_3D_VIEW, PERFUSION_3D_VIEW_REST):
            for node in self.surfaceNodes(tag):
                if node.GetAttribute("MPI.Surface") == "segments" and node.GetDisplayNode():
                    node.GetDisplayNode().SetVisibility(node.GetAttribute("MPI.Overlay") == self.overlay())

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

# saved Advanced parameters of every step, this module's constants included (see parameters.py)
parametersLib.applySaved()
