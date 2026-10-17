"""Workflow toolbar of the MPI modules, at the top of the Slicer main window.

    [icon] Tarvos | Name · ID | (1) Data roles  >  (2) Processing  >  ...  >  (5) Gated QC  >  (6) Gated Review
    [controls of the step shown: e.g. Max % | color map | FOV | Contours | Sync | Vessels / 17 seg | Scores |
     layouts | next step]

Shown only while the Myocardial Perfusion SPECT hub is the current module. The step buttons show
that step in the hub; their badge shows the step's state (done, available, locked). The case (patient
name and ID, case.py) is shown next to the logo, in orange when the studies belong to more than one
patient; a click opens Data roles. The second row
holds the most used controls of the step shown, made by its widget (buildToolbar(bar) with a
StepBar, kept up to date by its syncToolbar()); they drive the step panel's own controls.

One toolbar lives for the whole Slicer session. A developer reload never deletes the old one
(deleting Qt widgets that Python still wraps can crash Slicer): it is hidden and replaced.
"""

import html
import logging
import os
import traceback

import qt
import slicer

from MyocardialPerfusionSPECTLib import session
from MyocardialPerfusionSPECTLib import studies as studiesLib
from MyocardialPerfusionSPECTLib import case as caseLib
from MyocardialPerfusionSPECTLib.widgets import BUTTON_COLORS

TOOLBAR_OBJECT_NAME = "MPIWorkflowToolbar"
CONTROLS_OBJECT_NAME = "MPIStepControlsToolbar"
TOOLBAR_VERSION = 11      # bump when the class changes, so a reload replaces the toolbar
BADGE_SIZE = 22
CASE_MAX_CHARACTERS = 40   # longer case names are cut in the toolbar (the tooltip has all of it)
ICON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Resources", "Icons",
                         "MyocardialPerfusionSPECT.png")

STATE_DONE, STATE_AVAILABLE, STATE_LOCKED = session.STATE_DONE, session.STATE_AVAILABLE, session.STATE_LOCKED
STATE_COLORS = {STATE_DONE: "#2e9e5b", STATE_AVAILABLE: "#2f7ed8", STATE_LOCKED: "#8f969c"}


def badgeIcon(state, number):
    scale = 2   # drawn at twice the size for high-DPI screens
    size = BADGE_SIZE * scale
    pixmap = qt.QPixmap(size, size)
    pixmap.fill(qt.QColor(0, 0, 0, 0))
    painter = qt.QPainter(pixmap)
    try:
        painter.setRenderHint(qt.QPainter.Antialiasing)
        color = qt.QColor(STATE_COLORS[state])
        margin = 2 * scale
        if state == STATE_DONE:
            painter.setPen(qt.Qt.NoPen)
            painter.setBrush(qt.QBrush(color))
        else:
            pen = qt.QPen(color)
            pen.setWidth(2 * scale)
            painter.setPen(pen)
            painter.setBrush(qt.QBrush(qt.QColor(0, 0, 0, 0)))
        painter.drawEllipse(margin, margin, size - 2 * margin, size - 2 * margin)
        font = qt.QFont()
        font.setBold(True)
        font.setPixelSize(int(size * 0.55))
        painter.setFont(font)
        painter.setPen(qt.QColor("white") if state == STATE_DONE else color)
        painter.drawText(qt.QRect(0, 0, size, size), qt.Qt.AlignCenter, "✓" if state == STATE_DONE else str(number))
    finally:
        painter.end()
    return qt.QIcon(pixmap)


class StepBar:
    """Controls of one workflow step for the second toolbar row. The step widget makes them with these
    helpers in buildToolbar(bar) and sets their values with setValue / setChecked / setChoice (signals
    blocked, so the step's controls are not driven back)."""

    BUTTON_STYLE = ("QToolButton { border-radius: 4px; padding: 2px 7px; font-weight: bold; }"
                    "QToolButton:checked { background-color: rgba(47, 126, 216, 0.30); border: 1px solid #2f7ed8; }")

    @staticmethod
    def colorStyle(kind):
        """Style of a main action button in the colors of its panel button (widgets.styleButton)."""
        normal, hover, pressed = BUTTON_COLORS[kind]
        return (f"QToolButton {{ background-color: {normal}; color: white; font-weight: bold; border: none;"
                f" border-radius: 4px; padding: 2px 9px; }}"
                f"QToolButton:hover {{ background-color: {hover}; }}"
                f"QToolButton:pressed, QToolButton:checked {{ background-color: {pressed}; }}"
                "QToolButton:disabled { background-color: #9aa1a7; color: #eceff1; }")

    def __init__(self):
        self.widget = qt.QWidget()
        self.layout = qt.QHBoxLayout(self.widget)
        self.layout.setContentsMargins(4, 0, 4, 0)
        self.layout.setSpacing(4)
        self.mirrors = []   # (toolbar button, panel button)
        self.comboMirrors = []   # (toolbar combo box, panel combo box)

    def mirror(self, target, text, toolTip=None, kind=None):
        """A toolbar copy of a panel button: a click clicks the panel button (same effect, also in an exclusive
        group); enabled and checked follow it (syncMirrors). kind: colored as a main action (widgets.BUTTON_COLORS),
        like its panel button."""
        button = qt.QToolButton()
        button.setText(text)
        button.setCheckable(bool(target.checkable))
        button.setAutoRaise(kind is None)
        button.setFocusPolicy(qt.Qt.NoFocus)
        button.setStyleSheet(self.colorStyle(kind) if kind else self.BUTTON_STYLE)
        button.toolTip = toolTip or target.toolTip or target.text

        def onClick(checked=False):
            target.click()
            self.syncMirrors()
        button.connect("clicked(bool)", onClick)
        if target.checkable:
            target.connect("toggled(bool)", lambda _: self.syncMirrors())
        self.layout.addWidget(button)
        self.mirrors.append((button, target))
        return button

    def mirrorCombo(self, target, label=None, toolTip=None):
        """A toolbar copy of a panel combo box (same items): a choice here sets the panel one, and it follows
        the panel one."""
        if label:
            self.label(label)
        combo = qt.QComboBox()
        for index in range(target.count):
            combo.addItem(target.itemText(index), target.itemData(index))
        combo.setFocusPolicy(qt.Qt.NoFocus)
        combo.toolTip = toolTip or target.toolTip
        combo.setCurrentIndex(target.currentIndex)

        def onChoose(index):
            if target.currentIndex != index:
                target.setCurrentIndex(index)
        combo.connect("activated(int)", onChoose)
        target.connect("currentIndexChanged(int)", lambda _: self.syncMirrors())
        self.layout.addWidget(combo)
        self.comboMirrors.append((combo, target))
        return combo

    def syncMirrors(self):
        for combo, target in self.comboMirrors:
            try:
                if combo.currentIndex != target.currentIndex:
                    wasBlocked = combo.blockSignals(True)
                    combo.setCurrentIndex(target.currentIndex)
                    combo.blockSignals(wasBlocked)
                combo.setEnabled(bool(target.isEnabled()))
            except Exception:   # the panel combo box is gone (module reloaded)
                logging.debug("MPI toolbar: a mirrored combo box is gone", exc_info=True)
        for button, target in self.mirrors:
            try:
                button.setEnabled(bool(target.isEnabled()))
                if target.checkable:
                    self.setChecked(button, target.checked)
            except Exception:   # the panel button is gone (module reloaded)
                logging.debug("MPI toolbar: a mirrored button is gone", exc_info=True)

    def label(self, text):
        label = qt.QLabel(text)
        label.setStyleSheet("font-weight: bold; margin-left: 2px;")
        self.layout.addWidget(label)
        return label

    def separator(self):
        line = qt.QFrame()
        line.setFrameShape(qt.QFrame.VLine)
        line.setFrameShadow(qt.QFrame.Sunken)
        self.layout.addWidget(line)
        return line

    def spin(self, label, minimum, maximum, step, suffix, toolTip, onChange, decimals=0):
        """Spin box (committed on Enter, the arrows or leaving it); onChange(value)."""
        if label:
            self.label(label)
        spin = qt.QDoubleSpinBox() if decimals else qt.QSpinBox()
        if decimals:
            spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        spin.keyboardTracking = False
        spin.toolTip = toolTip
        spin.setMaximumWidth(90)
        spin.connect("valueChanged(double)" if decimals else "valueChanged(int)", onChange)
        self.layout.addWidget(spin)
        return spin

    def toggle(self, text, toolTip, onToggle, kind=None):
        """Checkable button; onToggle(checked) when the user clicks it. kind: colored as in mirror."""
        button = qt.QToolButton()
        button.setText(text)
        button.setCheckable(True)
        button.setAutoRaise(kind is None)
        button.setFocusPolicy(qt.Qt.NoFocus)
        button.setStyleSheet(self.colorStyle(kind) if kind else self.BUTTON_STYLE)
        button.toolTip = toolTip
        button.connect("clicked(bool)", onToggle)
        self.layout.addWidget(button)
        return button

    def action(self, text, toolTip, onClick, kind=None):
        """Push button; onClick() when the user clicks it. kind: colored as in mirror."""
        button = qt.QToolButton()
        button.setText(text)
        button.setAutoRaise(kind is None)
        button.setFocusPolicy(qt.Qt.NoFocus)
        button.setStyleSheet(self.colorStyle(kind) if kind else self.BUTTON_STYLE)
        button.toolTip = toolTip
        button.connect("clicked(bool)", lambda checked=False: onClick())
        self.layout.addWidget(button)
        return button

    def choice(self, options, onChoose):
        """Buttons of which one is checked: options [(text, value, tooltip)]; onChoose(value). Returns
        {value: button}."""
        buttons = {}
        for text, value, toolTip in options:
            button = self.toggle(text, toolTip, lambda checked, v=value: onChoose(v))
            buttons[value] = button
        return buttons

    def buttons(self, entries, onClick):
        """Checkable buttons, e.g. layouts: entries [(key, text, tooltip)]; onClick(key). Returns {key: button}."""
        return self.choice([(text, key, toolTip) for key, text, toolTip in entries], onClick)

    @staticmethod
    def setValue(spin, value):
        wasBlocked = spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(wasBlocked)

    @staticmethod
    def setChecked(button, checked):
        wasBlocked = button.blockSignals(True)
        button.setChecked(bool(checked))
        button.blockSignals(wasBlocked)

    def setChoice(self, buttons, value):
        for key, button in buttons.items():
            self.setChecked(button, key == value)


class WorkflowToolbar:

    def __init__(self):
        mainWindow = slicer.util.mainWindow()
        for old in [w for w in mainWindow.children()
                    if isinstance(w, qt.QToolBar) and w.objectName in (TOOLBAR_OBJECT_NAME, CONTROLS_OBJECT_NAME)]:
            old.setVisible(False)
            old.objectName = old.objectName + "Retired"
            try:
                old.toggleViewAction().setVisible(False)
            except Exception:
                pass
        self.toolbar = qt.QToolBar("Myocardial perfusion workflow", mainWindow)
        self.toolbar.objectName = TOOLBAR_OBJECT_NAME
        mainWindow.addToolBarBreak()
        mainWindow.addToolBar(self.toolbar)
        self.toolbar.setMovable(True)
        self.toolbar.setVisible(False)
        # second row: controls of the step shown
        self.controls = qt.QToolBar("Myocardial perfusion step controls", mainWindow)
        self.controls.objectName = CONTROLS_OBJECT_NAME
        mainWindow.addToolBarBreak()
        mainWindow.addToolBar(self.controls)
        self.controls.setMovable(True)
        self.controls.setVisible(False)
        self.sections = {}    # step key -> (step widget, StepBar, QAction)
        self.retiredSections = []   # bars of replaced step widgets (kept: Python still wraps them)
        self._build()

        self.refreshTimer = qt.QTimer()
        self.refreshTimer.setSingleShot(True)
        self.refreshTimer.setInterval(300)
        self.refreshTimer.connect("timeout()", self.refresh)

        self.moduleSelector = None
        try:
            self.moduleSelector = mainWindow.moduleSelector()
            self.moduleSelector.connect("moduleSelected(QString)", self.onModuleSelected)
        except Exception:
            logging.warning("MPI toolbar: could not follow the module selector:\n" + traceback.format_exc())
        self.sceneObservations = [slicer.mrmlScene.AddObserver(event, self.onSceneChanged) for event in
                                  (slicer.mrmlScene.NodeAddedEvent, slicer.mrmlScene.NodeRemovedEvent,
                                   slicer.mrmlScene.EndCloseEvent, slicer.mrmlScene.EndImportEvent)]
        # a closed or loaded scene starts again on the overview, in Four-Up (the toolbar exists from startup,
        # also before the hub was opened)
        self.sceneObservations += [slicer.mrmlScene.AddObserver(event, self.onSceneReset) for event in
                                   (slicer.mrmlScene.EndCloseEvent, slicer.mrmlScene.EndImportEvent)]
        session.addListener(self.scheduleRefresh)
        self.onModuleSelected(self.currentModuleName())

    # -- Construction --
    def _build(self):
        tb = self.toolbar
        # Hub button: logo and name, opens the Myocardial Perfusion SPECT overview
        self.hubButton = qt.QToolButton()
        self.hubButton.setText("Tarvos")
        if os.path.exists(ICON_PATH):
            self.hubButton.setIcon(qt.QIcon(ICON_PATH))
            self.hubButton.setIconSize(qt.QSize(28, 28))
        self.hubButton.setToolButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
        self.hubButton.setAutoRaise(True)
        self.hubButton.checkable = True
        self.hubButton.toolTip = "Tarvos - Myocardial Perfusion Imaging: overview of the workflow"
        self.hubButton.setStyleSheet(
            "QToolButton { border-radius: 5px; padding: 2px 8px; font-weight: bold; }"
            "QToolButton:checked { background-color: rgba(47, 126, 216, 0.28); border: 1px solid #2f7ed8; }")
        self.hubButton.connect("clicked(bool)", lambda checked: session.selectStep(session.HUB_STEP))
        tb.addWidget(self.hubButton)
        tb.addSeparator()

        # Case: patient name and ID (Data roles)
        self.caseButton = qt.QToolButton()
        self.caseButton.setAutoRaise(True)
        self.caseButton.setFocusPolicy(qt.Qt.NoFocus)
        self.caseButton.connect("clicked(bool)", lambda checked: session.selectStep("roles"))
        tb.addWidget(self.caseButton)
        tb.addSeparator()
        self.showCase([])

        self.stepButtons = {}
        group = qt.QButtonGroup(tb)
        group.setExclusive(True)
        group.addButton(self.hubButton)
        self.stepGroup = group
        for index, (key, moduleName, stepTitle, explanation) in enumerate(session.STEPS):
            if index:
                arrow = qt.QLabel("›")
                arrow.setStyleSheet("color: #9aa3ab; font-size: 18px; padding: 0px 2px;")
                tb.addWidget(arrow)
            button = qt.QToolButton()
            button.setText(f"{stepTitle}")
            button.setToolButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
            button.setIconSize(qt.QSize(BADGE_SIZE, BADGE_SIZE))
            button.checkable = True
            button.setAutoRaise(True)
            button.toolTip = f"{index + 1}. {stepTitle}: {explanation}"
            button.setStyleSheet(
                "QToolButton { border-radius: 5px; padding: 3px 8px; font-weight: bold; }"
                "QToolButton:checked { background-color: rgba(47, 126, 216, 0.28); border: 1px solid #2f7ed8; }")
            button.connect("clicked(bool)", lambda checked, k=key: session.selectStep(k))
            group.addButton(button)
            tb.addWidget(button)
            self.stepButtons[key] = (button, index + 1)

        # hide / show Slicer's module panel (it comes back when another module is selected)
        tb.addSeparator()
        self.panelButton = qt.QToolButton()
        self.panelButton.setAutoRaise(True)
        self.panelButton.setFocusPolicy(qt.Qt.NoFocus)
        self.panelButton.setStyleSheet("QToolButton { border-radius: 5px; padding: 3px 8px; font-weight: bold; }")
        self.panelButton.connect("clicked(bool)", lambda checked: self.togglePanel())
        tb.addWidget(self.panelButton)
        self._panelHidden = False
        self.syncPanelButton()

    # -- Module panel --
    @staticmethod
    def panelDock():
        try:
            return slicer.util.findChild(slicer.util.mainWindow(), "PanelDockWidget")
        except Exception:
            return None

    def togglePanel(self):
        dock = self.panelDock()
        if dock is None:
            return
        hide = dock.isVisible()
        dock.setVisible(not hide)
        self._panelHidden = hide
        self.syncPanelButton()

    def restorePanel(self):
        """The panel hidden here comes back (another module selected, toolbar removed)."""
        if self._panelHidden:
            dock = self.panelDock()
            if dock is not None:
                dock.setVisible(True)
            self._panelHidden = False
        self.syncPanelButton()

    def syncPanelButton(self):
        dock = self.panelDock()
        hidden = dock is not None and not dock.isVisible()
        self.panelButton.setText("▶ Show panel" if hidden else "◀ Hide panel")
        self.panelButton.toolTip = ("Show the module panel" if hidden else
                                    "Hide the module panel to give the views the room (it comes back when you "
                                    "leave the module)")

    # -- Events --
    @staticmethod
    def currentModuleName():
        try:
            selected = slicer.util.moduleSelector().selectedModule
            return (selected() if callable(selected) else selected) or ""
        except Exception:
            return ""

    def onModuleSelected(self, moduleName):
        visible = moduleName in session.MODULE_NAMES
        self.toolbar.setVisible(visible)
        if visible:
            self.refresh()
        else:
            self.controls.setVisible(False)
            self.restorePanel()

    def onSceneChanged(self, caller=None, event=None):
        self.scheduleRefresh()

    def onSceneReset(self, caller=None, event=None):
        qt.QTimer.singleShot(0, self.resetToStart)   # after the steps' own scene close handlers

    @staticmethod
    def resetToStart():
        from MyocardialPerfusionSPECTLib.views import resetOwnLayout
        session.resetToHub()
        resetOwnLayout()

    def scheduleRefresh(self):
        if self.toolbar.visible:
            self.refreshTimer.start()

    # -- Case --
    def showCase(self, patients):
        name, patientId, source = caseLib.currentCase()
        text = caseLib.caseText(name, patientId)
        warning = caseLib.mixedWarning(patients)
        color = "#d08030" if warning else ("" if text else "#8f969c")
        shown = text if len(text) <= CASE_MAX_CHARACTERS else text[:CASE_MAX_CHARACTERS - 1] + "\u2026"
        self.caseButton.setText(("\u26a0 " if warning else "") + (shown or "No case"))
        self.caseButton.setStyleSheet("QToolButton { border-radius: 5px; padding: 3px 8px; font-weight: bold; "
                                      + (f"color: {color}; " if color else "") + "}")
        tip = [f"Case: {text}" + (" (typed by hand)" if source == caseLib.SOURCE_MANUAL else
                                  " (from the DICOM headers)") if text else "No patient name or ID"]
        if warning:
            tip.append(warning)
        tip.append("Click to open Data roles")
        self.caseButton.toolTip = "<br>".join(html.escape(t).replace("\n", "<br>") for t in tip)

    # -- State --
    def refresh(self):
        try:
            allStudies = studiesLib.scanStudies()
            studies = [s for s in allStudies if s.staticNode is not None and s.role != studiesLib.EXCLUDED_ROLE]
        except Exception:
            logging.warning("MPI toolbar: could not list the studies:\n" + traceback.format_exc())
            allStudies, studies = [], []
        patients = caseLib.updateFromDicom(allStudies)   # a change comes back here through the session listener
        caseLib.warnIfMixed(patients)
        self.showCase(patients)
        active = session.activeStudy()
        states = session.stepStates(studies, active)
        for key, (button, number) in self.stepButtons.items():
            button.setIcon(badgeIcon(states[key], number))
        self.states = states
        current = session.currentStep()
        self.hubButton.checked = current == session.HUB_STEP
        for key, (button, _) in self.stepButtons.items():
            button.checked = key == current
        self.refreshControls(current)
        self.syncPanelButton()

    # -- Step controls (second row) --
    @staticmethod
    def stepWidget(key):
        """Python widget of a step module, if its panel was made (the hub makes it when showing the step)."""
        moduleName = session.stepModuleName(key) if key != session.HUB_STEP else None
        module = getattr(slicer.modules, moduleName.lower(), None) if moduleName else None
        if module is None:
            return None
        try:
            return module.widgetRepresentation().self()
        except Exception:
            return None

    def refreshControls(self, key=None):
        key = key or session.currentStep()
        widget = self.stepWidget(key) if self.toolbar.visible else None
        shown = None
        if widget is not None and callable(getattr(widget, "buildToolbar", None)):
            try:
                section = self.sections.get(key)
                if section is None or section[0] is not widget:
                    if section is not None:
                        section[2].setVisible(False)
                        self.retiredSections.append(section)
                    bar = StepBar()
                    widget.buildToolbar(bar)
                    section = (widget, bar, self.controls.addWidget(bar.widget))
                    self.sections[key] = section
                shown = section
                widget.syncToolbar()
            except Exception:
                logging.warning(f"MPI toolbar: could not show the controls of step {key}:\n" + traceback.format_exc())
                shown = None
        for section in self.sections.values():
            section[2].setVisible(section is shown)
        self.controls.setVisible(shown is not None)

    def destroy(self):
        self.restorePanel()
        session.removeListener(self.scheduleRefresh)
        self.refreshTimer.stop()
        for tag in self.sceneObservations:
            slicer.mrmlScene.RemoveObserver(tag)
        self.sceneObservations = []
        if self.moduleSelector is not None:
            try:
                self.moduleSelector.disconnect("moduleSelected(QString)", self.onModuleSelected)
            except Exception:
                pass
        self.toolbar.setVisible(False)
        self.controls.setVisible(False)


def install():
    """Create the toolbar once Slicer has a main window; replace it after a developer reload."""
    if slicer.app.commandOptions().noMainWindow or slicer.util.mainWindow() is None:
        return None
    state = getattr(slicer, "_mpiWorkflowToolbarState", None)
    if state is not None and state.get("version") == TOOLBAR_VERSION and state.get("module") is WorkflowToolbar:
        return state["toolbar"]
    if state is not None and state.get("toolbar") is not None:
        try:
            state["toolbar"].destroy()
        except Exception:
            logging.warning("MPI toolbar: could not retire the previous toolbar:\n" + traceback.format_exc())
        state.setdefault("retired", []).append(state["toolbar"])   # keep the Qt objects alive
    try:
        toolbar = WorkflowToolbar()
    except Exception:
        logging.error("MPI toolbar: could not be created:\n" + traceback.format_exc())
        return None
    retired = state.get("retired", []) if state else []
    slicer._mpiWorkflowToolbarState = {"toolbar": toolbar, "version": TOOLBAR_VERSION,
                                       "module": WorkflowToolbar, "retired": retired}
    return toolbar
