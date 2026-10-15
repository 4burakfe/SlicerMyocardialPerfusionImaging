"""
Myocardial Perfusion SPECT - 5. Gated

Placeholder step for the gated review of the active study: wall motion and thickening, EF and
volumes, cine of the reoriented slices and an animated orthographic 3D wall. The gated sequence of a
study is found in Data roles (MyocardialPerfusionSPECTLib/studies.py).
"""

import os
import sys
import logging

import qt
import slicer
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleWidget,
                                           ScriptedLoadableModuleTest)

_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from MyocardialPerfusionSPECTLib import session  # noqa: E402
from MyocardialPerfusionSPECTLib import studies as studiesLib  # noqa: E402


def _installToolbar():
    try:
        from MyocardialPerfusionSPECTLib import toolbar
        toolbar.install()
    except Exception:
        logging.exception("MPI: could not install the workflow toolbar")


class MPIGated(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MPI 5 · Gated"
        self.parent.categories = ["Nuclear Medicine.Myocardial Perfusion"]
        self.parent.hidden = True   # opened from the Myocardial Perfusion SPECT hub and the workflow toolbar
        self.parent.dependencies = []
        self.parent.contributors = ["Burak Demir"]
        self.parent.helpText = "Gated review of the active study (wall motion, thickening, EF). Coming soon."
        self.parent.acknowledgementText = ""
        iconPath = os.path.join(_MODULE_DIR, "Resources", "Icons", "MyocardialPerfusionSPECT.png")
        if os.path.exists(iconPath):
            self.parent.icon = qt.QIcon(iconPath)
        if not slicer.app.commandOptions().noMainWindow:
            slicer.app.connect("startupCompleted()", _installToolbar)


class MPIGatedWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.studyLabel = qt.QLabel()
        self.studyLabel.wordWrap = True
        self.layout.addWidget(self.studyLabel)
        placeholder = qt.QLabel("Gated review will come here: wall motion and thickening polar maps, "
                                "EF and volumes, cine of the reoriented slices and an animated 3D wall.")
        placeholder.wordWrap = True
        placeholder.setStyleSheet("color: #8f969c; padding: 8px 0px;")
        self.layout.addWidget(placeholder)
        self.layout.addStretch(1)
        session.addListener(self.updateStudyLabel)
        _installToolbar()
        self.updateStudyLabel()

    def enter(self):
        self.updateStudyLabel()

    def cleanup(self):
        session.removeListener(self.updateStudyLabel)

    def updateStudyLabel(self):
        active = session.activeStudy()
        if active is None:
            self.studyLabel.text = "No active study."
            return
        gated = gatedSequenceOf(active)
        self.studyLabel.text = (f"<b>{active.GetName()}</b><br>"
                                + (f"Gated sequence: {gated.GetName()}" if gated is not None
                                   else "No gated sequence paired with this study (see Data roles)."))


def gatedSequenceOf(volumeNode):
    """The gated sequence paired with a static volume in Data roles, or None."""
    for study in studiesLib.scanStudies():
        if study.staticNode is volumeNode:
            return study.gatedNode
    return None


class MPIGatedTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_GatedPlaceholder()

    def test_GatedPlaceholder(self):
        widget = slicer.modules.mpigated.widgetRepresentation().self()
        session.setActiveStudy(None)
        widget.updateStudyLabel()
        self.assertIn("No active study", widget.studyLabel.text)
        self.delayDisplay("Gated placeholder test passed")
