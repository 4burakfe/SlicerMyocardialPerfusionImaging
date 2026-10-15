"""Workflow toolbar of the MPI modules, at the top of the Slicer main window.

    [icon] Myocardial Perfusion | Study: [role · volume v] | (1) Data roles  >  (2) Processing  >  (3) Perfusion  >  (4) Gated

Shown only while the Myocardial Perfusion SPECT hub is the current module. The step buttons show
that step in the hub; their badge shows the step's state (done, available, locked). The study box
sets the active study for all steps.

One toolbar lives for the whole Slicer session. A developer reload never deletes the old one
(deleting Qt widgets that Python still wraps can crash Slicer): it is hidden and replaced.
"""

import logging
import os
import traceback

import qt
import slicer

from MyocardialPerfusionSPECTLib import session
from MyocardialPerfusionSPECTLib import studies as studiesLib

TOOLBAR_OBJECT_NAME = "MPIWorkflowToolbar"
TOOLBAR_VERSION = 5       # bump when the class changes, so a reload replaces the toolbar
BADGE_SIZE = 22
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


class WorkflowToolbar:

    def __init__(self):
        mainWindow = slicer.util.mainWindow()
        for old in [w for w in mainWindow.children()
                    if isinstance(w, qt.QToolBar) and w.objectName == TOOLBAR_OBJECT_NAME]:
            old.setVisible(False)
            old.objectName = TOOLBAR_OBJECT_NAME + "Retired"
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
        self._filling = False
        self._studyIds = []
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
        session.addListener(self.scheduleRefresh)
        self.onModuleSelected(self.currentModuleName())

    # -- Construction --
    def _build(self):
        tb = self.toolbar
        # Hub button: logo and name, opens the Myocardial Perfusion SPECT overview
        self.hubButton = qt.QToolButton()
        self.hubButton.setText("Myocardial Perfusion")
        if os.path.exists(ICON_PATH):
            self.hubButton.setIcon(qt.QIcon(ICON_PATH))
            self.hubButton.setIconSize(qt.QSize(28, 28))
        self.hubButton.setToolButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
        self.hubButton.setAutoRaise(True)
        self.hubButton.checkable = True
        self.hubButton.toolTip = "Overview of the Myocardial Perfusion SPECT workflow"
        self.hubButton.setStyleSheet(
            "QToolButton { border-radius: 5px; padding: 2px 8px; font-weight: bold; }"
            "QToolButton:checked { background-color: rgba(47, 126, 216, 0.28); border: 1px solid #2f7ed8; }")
        self.hubButton.connect("clicked(bool)", lambda checked: session.selectStep(session.HUB_STEP))
        tb.addWidget(self.hubButton)
        tb.addSeparator()

        tb.addWidget(qt.QLabel(" Study: "))
        self.studyCombo = qt.QComboBox()
        self.studyCombo.setMinimumContentsLength(22)
        self.studyCombo.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.studyCombo.toolTip = "Active study for all Myocardial Perfusion modules (set roles in Data roles)"
        self.studyCombo.connect("activated(int)", self.onStudyActivated)
        tb.addWidget(self.studyCombo)
        tb.addSeparator()

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

    def onSceneChanged(self, caller=None, event=None):
        self.scheduleRefresh()

    def scheduleRefresh(self):
        if self.toolbar.visible:
            self.refreshTimer.start()

    def onStudyActivated(self, index):
        if self._filling or not (0 <= index < len(self._studyIds)):
            return
        node = slicer.mrmlScene.GetNodeByID(self._studyIds[index]) if self._studyIds[index] else None
        session.setActiveStudy(node)

    # -- State --
    def refresh(self):
        try:
            studies = [s for s in studiesLib.scanStudies() if s.staticNode is not None]
        except Exception:
            logging.warning("MPI toolbar: could not list the studies:\n" + traceback.format_exc())
            studies = []
        active = session.activeStudy()
        self._filling = True
        try:
            self.studyCombo.clear()
            self._studyIds = []
            if not studies:
                self.studyCombo.addItem("(no SPECT volume)")
                self._studyIds.append("")
            for study in studies:
                label = f"{study.role}{'?' if study.guessed else ''} · {study.name}"
                if study.gatedNode is not None:
                    label += f" · gated {study.gates}"
                self.studyCombo.addItem(label)
                self._studyIds.append(study.staticNode.GetID())
            if active is not None and active.GetID() in self._studyIds:
                self.studyCombo.setCurrentIndex(self._studyIds.index(active.GetID()))
            else:
                self.studyCombo.setCurrentIndex(-1 if studies else 0)
        finally:
            self._filling = False

        states = session.stepStates(studies, active)
        for key, (button, number) in self.stepButtons.items():
            button.setIcon(badgeIcon(states[key], number))
        self.states = states
        current = session.currentStep()
        self.hubButton.checked = current == session.HUB_STEP
        for key, (button, _) in self.stepButtons.items():
            button.checked = key == current

    def destroy(self):
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
