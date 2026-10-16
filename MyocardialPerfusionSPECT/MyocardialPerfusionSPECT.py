"""
Myocardial Perfusion SPECT for 3D Slicer - hub module

The only module of the extension in Slicer's module list. It shows the active study and the
workflow steps, which are hidden modules opened from here and from the workflow toolbar:

    1. Data roles   (MPIDataRoles.py)   studies in the scene, their roles and gated sequences
    2. Processing   (MPIProcessing.py)  reorientation to SA / HLA / VLA
    3. Myocardium   (MPIMyocardium.py)  automatic LV wall detection (endo/epi surfaces, valve plane)
    4. Perfusion    (MPIReview.py)      stress / rest slices, polar maps and 3D uptake
    5. Gated QC     (MPIGatedQC.py)     every gate frame segmented, counts per frame, ECG gating checks
    6. Gated Review (MPIGated.py)       cine slices, perfusion / thickening polar maps, volumes and EF, cine 3D

The workflow toolbar at the top of the window is shown while the hub or a step is open.
The helpers shared by all modules are in the MyocardialPerfusionSPECTLib package next to this file.

Install: from the Extensions Manager, or without building: add this folder (the inner
MyocardialPerfusionSPECT folder that holds the module files and the MyocardialPerfusionSPECTLib
package) to Edit > Application Settings > Modules > Additional module paths, then restart.
The module appears under "Nuclear Medicine".
"""

import os
import sys
import logging

import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleWidget,
                                           ScriptedLoadableModuleTest)
from slicer.util import VTKObservationMixin

_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

import MyocardialPerfusionSPECTLib  # noqa: E402
MyocardialPerfusionSPECTLib.reloadInDeveloperMode()
from MyocardialPerfusionSPECTLib import session  # noqa: E402
from MyocardialPerfusionSPECTLib import studies as studiesLib  # noqa: E402
from MyocardialPerfusionSPECTLib.widgets import breakable, narrowLabel, styleButton  # noqa: E402

ICON_PATH = os.path.join(_MODULE_DIR, "Resources", "Icons", "MyocardialPerfusionSPECT.png")
BANNER_PATH = os.path.join(_MODULE_DIR, "Resources", "Images", "Banner.jpg")
STATE_TEXT = {session.STATE_DONE: "Done", session.STATE_AVAILABLE: "Ready", session.STATE_LOCKED: "Waiting"}


def _installToolbar():
    try:
        from MyocardialPerfusionSPECTLib import toolbar
        return toolbar.install()
    except Exception:
        logging.exception("MPI: could not install the workflow toolbar")
        return None


PANEL_MAX_WIDTH = 900   # px, module panel width while the hub is open


class MyocardialPerfusionSPECT(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Tarvos - Myocardial Perfusion Imaging"
        self.parent.categories = ["Nuclear Medicine"]
        self.parent.dependencies = []
        self.parent.contributors = ["Burak Demir"]
        self.parent.helpText = (
            "Myocardial perfusion SPECT workflow: <b>1. Data roles</b> (studies, stress/rest/viability roles, "
            "gated sequences), <b>2. Processing</b> (reorientation to the cardiac axes), <b>3. Myocardium</b> "
            "(LV wall detection), <b>4. Perfusion</b> (stress / rest slices and polar maps), <b>5. Gated QC</b> (gate frames, counts and ECG gating checks) and <b>6. Gated Review</b> (cine slices, perfusion and thickening polar maps, LV volumes and EF, cine 3D wall). Open a step here or from the workflow toolbar.")
        self.parent.acknowledgementText = ""
        if os.path.exists(ICON_PATH):
            self.parent.icon = qt.QIcon(ICON_PATH)
        if not slicer.app.commandOptions().noMainWindow:
            slicer.app.connect("startupCompleted()", _installToolbar)


class _BannerScaler(qt.QObject):
    """Keeps the intro banner as wide as the panel (aspect kept, sharp on scaled displays)."""

    def __init__(self, label, pixmap):
        super().__init__()
        self.label = label
        self.pixmap = pixmap
        self.width = None
        self.label.setFixedHeight(120)

    def eventFilter(self, obj, event):
        if event.type() == qt.QEvent.Resize:
            self.rescale()
        return False

    def rescale(self):
        width = self.label.contentsRect().width()
        if width < 20 or width == self.width or self.pixmap.isNull():
            return
        self.width = width
        try:
            ratio = float(self.label.devicePixelRatioF())
        except Exception:
            ratio = 1.0
        scaled = self.pixmap.scaledToWidth(int(round(width * ratio)), qt.Qt.SmoothTransformation)
        try:
            scaled.setDevicePixelRatio(ratio)
        except Exception:
            pass
        self.label.setFixedHeight(int(round(scaled.height() / ratio)))
        self.label.setPixmap(scaled)


class MyocardialPerfusionSPECTWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Overview page plus the panels of the step modules in one stack. The step modules are hidden
    (Slicer's module selector cannot open them), so their widget representations are shown here."""

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.stepWidgets = {}
        self.stepPanels = {}     # step key -> widget representation shown in the stack

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Step navigation lives in the workflow toolbar and on the overview page
        self.stack = qt.QStackedWidget()
        self.layout.addWidget(self.stack)
        self.overview = qt.QWidget()
        self.stack.addWidget(self.overview)
        self._buildOverview(qt.QVBoxLayout(self.overview))

        self.refreshTimer = qt.QTimer()
        self.refreshTimer.setSingleShot(True)
        self.refreshTimer.setInterval(300)
        self.refreshTimer.connect("timeout()", self.refresh)
        for event in (slicer.mrmlScene.NodeAddedEvent, slicer.mrmlScene.NodeRemovedEvent,
                      slicer.mrmlScene.EndCloseEvent, slicer.mrmlScene.EndImportEvent):
            self.addObserver(slicer.mrmlScene, event, lambda caller, event: self.scheduleRefresh())
        session.addListener(self.scheduleRefresh)
        _installToolbar()
        self.showStep(session.currentStep(), notifyStep=False)
        self.refresh()

    def _buildOverview(self, layout):
        layout.setContentsMargins(0, 0, 0, 0)
        if os.path.exists(BANNER_PATH):
            self.banner = qt.QLabel()
            self.banner.setObjectName("MPIBanner")
            self.banner.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
            self.banner.setAlignment(qt.Qt.AlignCenter)
            self.bannerScaler = _BannerScaler(self.banner, qt.QPixmap(BANNER_PATH))
            self.banner.installEventFilter(self.bannerScaler)
            layout.addWidget(self.banner)
            title = qt.QLabel("Reorientation and review of stress, rest and viability studies")
        else:
            title = qt.QLabel("<b style='font-size: 15px'>Tarvos - Myocardial Perfusion Imaging</b><br>"
                              "Reorientation and review of stress, rest and viability studies")
        title.wordWrap = True
        layout.addWidget(title)

        studyBox = ctk.ctkCollapsibleButton()
        studyBox.text = "Active study"
        layout.addWidget(studyBox)
        studyLayout = qt.QVBoxLayout(studyBox)
        self.studyLabel = narrowLabel(qt.QLabel("-"))
        studyLayout.addWidget(self.studyLabel)

        stepsBox = ctk.ctkCollapsibleButton()
        stepsBox.text = "Workflow"
        layout.addWidget(stepsBox)
        stepsLayout = qt.QVBoxLayout(stepsBox)
        for number, (key, moduleName, stepTitle, explanation) in enumerate(session.STEPS, start=1):
            frame = qt.QFrame()
            frame.setFrameShape(qt.QFrame.StyledPanel)
            row = qt.QHBoxLayout(frame)
            badge = qt.QLabel()
            row.addWidget(badge)
            text = qt.QLabel(f"<b>{number}. {stepTitle}</b><br><span style='color:#7d868d'>{explanation}</span>")
            text.wordWrap = True
            row.addWidget(text, 1)
            state = qt.QLabel("")
            row.addWidget(state)
            button = styleButton(qt.QPushButton("Open"), "primary")
            button.setMinimumWidth(70)
            button.connect("clicked(bool)", lambda checked, k=key: self.showStep(k))
            row.addWidget(button)
            stepsLayout.addWidget(frame)
            self.stepWidgets[key] = {"badge": badge, "state": state, "button": button, "number": number}

        self.startButton = styleButton(qt.QPushButton("Start with Data roles \u203a"), "primary")
        self.startButton.connect("clicked(bool)", lambda: self.showStep("roles"))
        layout.addWidget(self.startButton)
        layout.addStretch(1)

    # --- steps ----------------------------------------------------------------
    def stepPanel(self, key):
        """The step module's own widget representation, put into the stack. After a developer reload
        of a step module its representation is a new widget, which then replaces the old one."""
        moduleName = session.stepModuleName(key)
        module = getattr(slicer.modules, moduleName.lower(), None) if moduleName else None
        if module is None:
            raise RuntimeError(f"Module {moduleName} is not loaded")
        panel = module.widgetRepresentation()
        old = self.stepPanels.get(key)
        if old is not None and old is not panel:
            self.stack.removeWidget(old)
        if self.stack.indexOf(panel) < 0:
            self.stack.addWidget(panel)
        self.stepPanels[key] = panel
        return panel

    @staticmethod
    def _pythonWidget(panel):
        try:
            return panel.self()
        except Exception:
            return None

    def _callStep(self, key, method):
        """Call enter() / exit() of a step widget if it defines one (they are optional in Slicer)."""
        stepWidget = self._pythonWidget(self.stepPanels[key]) if key in self.stepPanels else None
        function = getattr(stepWidget, method, None)
        if callable(function):
            try:
                function()
            except Exception:
                logging.warning(f"MPI hub: {method}() of step {key} failed", exc_info=True)

    def showStep(self, key, notifyStep=True):
        previous = session.currentStep()
        try:
            if key == session.HUB_STEP:
                widget = self.overview
            else:
                widget = self.stepPanel(key)
        except Exception as e:
            logging.warning(f"MPI hub: cannot show step {key}: {e}")
            key, widget = session.HUB_STEP, self.overview
        if previous != key:
            self._callStep(previous, "exit")
        # Only the shown page may set the height of the stack (QStackedWidget uses the tallest page)
        for index in range(self.stack.count):
            page = self.stack.widget(index)
            policy = qt.QSizePolicy.Preferred if page is widget else qt.QSizePolicy.Ignored
            page.setSizePolicy(policy, policy)
        self.stack.setCurrentWidget(widget)
        self.stack.adjustSize()
        if key in self.stepPanels:
            if slicer.util.selectedModule() == session.HUB_MODULE:
                self._callStep(key, "enter")
        else:
            self.refresh()
        if notifyStep:
            session.setCurrentStep(key)

    def enter(self):
        self.limitPanelWidth(True)
        self.showStep(session.currentStep(), notifyStep=False)
        self.refresh()

    def exit(self):
        self._callStep(session.currentStep(), "exit")
        self.limitPanelWidth(False)

    @staticmethod
    def limitPanelWidth(on):
        """Keep Slicer's module panel at most PANEL_MAX_WIDTH wide while the hub is open, so the
        views get the room; the previous limit comes back when another module is selected."""
        try:
            dock = slicer.util.mainWindow().findChild(qt.QDockWidget, "PanelDockWidget")
        except Exception:
            dock = None
        if dock is None:
            return
        if on:
            if getattr(slicer, "_mpiPanelMaxWidth", None) is None:
                slicer._mpiPanelMaxWidth = dock.maximumWidth
            dock.setMaximumWidth(PANEL_MAX_WIDTH)
        elif getattr(slicer, "_mpiPanelMaxWidth", None) is not None:
            dock.setMaximumWidth(slicer._mpiPanelMaxWidth)
            slicer._mpiPanelMaxWidth = None

    def cleanup(self):
        self.limitPanelWidth(False)
        self.refreshTimer.stop()
        session.removeListener(self.scheduleRefresh)
        self.removeObservers()
        # The step panels belong to their modules: give them back before this widget goes away
        for panel in self.stepPanels.values():
            try:
                self.stack.removeWidget(panel)
                panel.setParent(None)
            except Exception:
                pass
        self.stepPanels = {}

    def scheduleRefresh(self):
        self.refreshTimer.start()

    def refresh(self):
        from MyocardialPerfusionSPECTLib.toolbar import badgeIcon
        try:
            studies = [s for s in studiesLib.scanStudies() if s.staticNode is not None]
        except Exception:
            logging.warning("MPI hub: could not list the studies", exc_info=True)
            studies = []
        active = session.activeStudy()
        if active is None:
            self.studyLabel.text = (f"{len(studies)} SPECT volume(s) in the scene. "
                                    "Choose the study to work on in Data roles or in the toolbar.")
        else:
            role = active.GetAttribute(studiesLib.ROLE_ATTRIBUTE) or "role not set"
            outputs = session.outputsForStudy(active)
            self.studyLabel.text = (f"<b>{role}</b>: {breakable(active.GetName())}<br>"
                                    + ("Reoriented volumes: " + ", ".join(sorted(outputs)) if outputs
                                       else "Not reoriented yet."))
        states = session.stepStates(studies, active)
        for key, widgets in self.stepWidgets.items():
            state = states[key]
            widgets["badge"].setPixmap(badgeIcon(state, widgets["number"]).pixmap(24, 24))
            widgets["state"].text = STATE_TEXT[state]
            widgets["button"].enabled = state != session.STATE_LOCKED


class MyocardialPerfusionSPECTTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_StepModulesAreHidden()

    def test_StepModulesAreHidden(self):
        """The hub is listed in the module menu; the steps exist but are hidden."""
        manager = slicer.app.moduleManager()
        hub = manager.module(session.HUB_MODULE)
        self.assertIsNotNone(hub)
        self.assertFalse(hub.isHidden())
        for _, moduleName, _, _ in session.STEPS:
            module = manager.module(moduleName)
            self.assertIsNotNone(module, moduleName)
            self.assertTrue(module.isHidden(), moduleName)
        self.assertEqual(set(session.stepStates([], None).values()) - {session.STATE_AVAILABLE, session.STATE_LOCKED},
                         set())
        # Steps open inside the hub
        hub = slicer.util.getModuleWidget(session.HUB_MODULE)
        for key, moduleName, _, _ in session.STEPS:
            session.selectStep(key)
            self.assertEqual(session.currentStep(), key)
            self.assertIs(hub.stack.currentWidget(), getattr(slicer.modules, moduleName.lower()).widgetRepresentation())
        session.selectStep(session.HUB_STEP)
        self.assertIs(hub.stack.currentWidget(), hub.overview)
        self.delayDisplay("Hub test passed")
