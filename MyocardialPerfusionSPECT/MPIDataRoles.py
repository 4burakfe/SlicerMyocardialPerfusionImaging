"""
Myocardial Perfusion SPECT - 1. Data roles

Lists the SPECT studies in the scene: each static (or summed-gate) volume with its role (stress,
rest, redistribution, viability, other) and the gated sequence that goes with it. Roles are guessed
from the names and confirmed here; they are stored on the nodes (MPI.StudyRole, MPI.GatedSequence)
and saved with the scene. The selected row is the active study of all MPI modules.
The case (patient name and ID) is read from the DICOM headers of the studies, can be typed instead,
and is shown in the workflow toolbar; studies of more than one patient are warned about
(MyocardialPerfusionSPECTLib/case.py).
See MyocardialPerfusionSPECTLib/studies.py for how gated sequences are recognised and paired.
"""

import html
import os
import sys
import logging
import traceback

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
from MyocardialPerfusionSPECTLib import studies as studiesLib  # noqa: E402
from MyocardialPerfusionSPECTLib import session  # noqa: E402
from MyocardialPerfusionSPECTLib import parameters as parametersLib  # noqa: E402
from MyocardialPerfusionSPECTLib import case as caseLib  # noqa: E402
from MyocardialPerfusionSPECTLib.widgets import narrowLabel, styleButton  # noqa: E402

STUDIES_TABLE_HEIGHT = 500   # px, fixed height of the studies table


def _installToolbar():
    try:
        from MyocardialPerfusionSPECTLib import toolbar
        toolbar.install()
    except Exception:
        logging.exception("MPI: could not install the workflow toolbar")


class MPIDataRoles(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MPI 1 \u00b7 Data roles"
        self.parent.categories = ["Nuclear Medicine.Myocardial Perfusion"]
        self.parent.hidden = True   # opened from the Myocardial Perfusion SPECT hub and the workflow toolbar
        self.parent.dependencies = []
        self.parent.contributors = ["Burak Demir"]
        self.parent.helpText = (
            "Myocardial perfusion SPECT studies in the scene, their roles (stress, rest, redistribution, "
            "viability) and gated sequences. Roles in italics are guesses: pick the right one to confirm it; "
            "None leaves a volume out of the analysis. Study 1 / Study 2 are picked from the roles and can be "
            "changed in every step.")
        self.parent.acknowledgementText = ""
        iconPath = os.path.join(_MODULE_DIR, "Resources", "Icons", "MyocardialPerfusionSPECT.png")
        if os.path.exists(iconPath):
            self.parent.icon = qt.QIcon(iconPath)
        if not slicer.app.commandOptions().noMainWindow:
            slicer.app.connect("startupCompleted()", _installToolbar)


class MPIDataRolesWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.studies = []            # studiesLib.Study rows of the table
        self._fillingStudies = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        caseSection = ctk.ctkCollapsibleButton()
        caseSection.text = "Case"
        self.layout.addWidget(caseSection)
        caseForm = qt.QFormLayout(caseSection)
        self.caseNameEdit = qt.QLineEdit()
        self.caseNameEdit.placeholderText = "Patient name"
        self.caseNameEdit.toolTip = ("Patient name, read from the DICOM headers of the studies; type to change it "
                                     "(kept until Read from DICOM)")
        caseForm.addRow("Name:", self.caseNameEdit)
        self.caseIdEdit = qt.QLineEdit()
        self.caseIdEdit.placeholderText = "Patient ID"
        self.caseIdEdit.toolTip = ("Patient ID, read from the DICOM headers of the studies; type to change it "
                                   "(kept until Read from DICOM)")
        caseForm.addRow("ID:", self.caseIdEdit)
        caseRow = qt.QHBoxLayout()
        self.caseSourceLabel = narrowLabel(qt.QLabel())
        self.caseSourceLabel.setStyleSheet("color: #7d868d;")
        self.caseDicomButton = qt.QPushButton("Read from DICOM")
        self.caseDicomButton.toolTip = ("Take the name and ID from the DICOM headers of the studies again "
                                        "(with more than one patient: the one of the active study)")
        caseRow.addWidget(self.caseSourceLabel, 1)
        caseRow.addWidget(self.caseDicomButton)
        caseForm.addRow(caseRow)
        self.caseWarningLabel = narrowLabel(qt.QLabel())
        self.caseWarningLabel.setStyleSheet("color: #d08030; font-weight: bold;")
        self.caseWarningLabel.visible = False
        caseForm.addRow(self.caseWarningLabel)
        self.patients = []   # caseLib.scanPatients groups of the studies shown

        studiesSection = ctk.ctkCollapsibleButton()
        studiesSection.text = "Studies"
        self.layout.addWidget(studiesSection)
        studiesLayout = qt.QVBoxLayout(studiesSection)

        self.studiesTable = qt.QTableWidget()
        self.studiesTable.setColumnCount(4)
        self.studiesTable.setHorizontalHeaderLabels(["Volume", "Role", "Gated", "Gates"])
        self.studiesTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.studiesTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.studiesTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.studiesTable.verticalHeader().setVisible(False)
        header = self.studiesTable.horizontalHeader()
        header.setSectionResizeMode(0, qt.QHeaderView.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, qt.QHeaderView.ResizeToContents)
        self.studiesTable.setFixedHeight(STUDIES_TABLE_HEIGHT)
        self.studiesTable.toolTip = (
            "Static (or summed-gate) volumes found in the scene, their study role and the gated sequence "
            "that goes with them. Roles in italics are guesses from the names: pick the right one to "
            "confirm it. The selected row is the active study.")
        studiesLayout.addWidget(self.studiesTable)

        studyButtons = qt.QHBoxLayout()
        self.detectStudiesButton = qt.QPushButton("Detect studies")
        self.detectStudiesButton.toolTip = "Look for SPECT volumes and gated sequences in the scene again"
        self.sumGatesButton = styleButton(qt.QPushButton("Create static from gates"), "create")
        self.sumGatesButton.toolTip = "Sum the gates of the selected gated study into a static volume"
        self.sumGatesButton.enabled = False
        studyButtons.addWidget(self.detectStudiesButton)
        studyButtons.addWidget(self.sumGatesButton)
        studiesLayout.addLayout(studyButtons)

        self.nextButton = styleButton(qt.QPushButton("Process \u203a"), "primary")
        self.nextButton.toolTip = ("Open Processing (reorientation). Study 1 / Study 2 are picked from the roles "
                                   "and can be changed in every step; set a volume to None to leave it out")
        self.nextButton.enabled = False
        self.layout.addWidget(self.nextButton)

        self.parameterPanel = parametersLib.ParameterPanel(
            "roles", self.layout, onChanged=lambda key: self.refreshStudies())
        self.layout.addStretch(1)

        # Studies are detected again shortly after volumes or sequences are added (e.g. a DICOM load)
        self.studiesTimer = qt.QTimer()
        self.studiesTimer.setSingleShot(True)
        self.studiesTimer.setInterval(500)

        self.studiesTimer.connect("timeout()", self.refreshStudies)
        self.detectStudiesButton.connect("clicked(bool)", self.refreshStudies)
        self.sumGatesButton.connect("clicked(bool)", self.onSumGates)
        self.nextButton.connect("clicked(bool)", self.onProcess)
        self.studiesTable.connect("itemSelectionChanged()", self.onStudySelected)
        self.caseNameEdit.connect("editingFinished()", self.onCaseEdited)
        self.caseIdEdit.connect("editingFinished()", self.onCaseEdited)
        self.caseDicomButton.connect("clicked(bool)", self.onCaseFromDicom)
        self.parameterPanel.loadSaved()   # saved settings of this step, as if set by hand

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAddedEvent, self.onNodeAdded)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeRemovedEvent, self.onNodeAdded)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onNodeAdded)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onNodeAdded)
        session.addListener(self.onSessionChanged)
        _installToolbar()
        self.refreshStudies()

    # --- workflow toolbar (second row) -----------------------------------------------------------------
    def buildToolbar(self, bar):
        """Most used controls in the toolbar (toolbar.StepBar); they drive the controls of this panel."""
        self.toolbarBar = bar
        bar.action("Add data…", "Open Slicer's Add data dialog to load studies (files, or a DICOM folder)",
                   slicer.util.openAddDataDialog)
        bar.mirror(self.detectStudiesButton, "Detect studies")
        bar.separator()
        self.tbStudy = bar.combo("Study", "Study of the selected row: the active study of all steps",
                                 self.onToolbarStudy, 140)
        self.tbRole = bar.combo("Role", "Role of that study (italic: guessed from the names, pick to confirm; None "
                                        "leaves it out)", self.onToolbarRole)
        self.tbGated = bar.combo("Gated", "Gated sequence of that study", self.onToolbarGated, 120)
        bar.mirror(self.sumGatesButton, "Create static from gates", kind="create")
        bar.separator()
        bar.mirror(self.nextButton, "Process ›", kind="primary")

    def selectedRow(self):
        rows = self.studiesTable.selectionModel().selectedRows() if self.studiesTable.selectionModel() else []
        return rows[0].row() if rows else -1

    def onToolbarStudy(self, index):
        if 0 <= index < len(self.studies):
            self.studiesTable.selectRow(index)   # makes it the active study (onStudySelected)
        self.syncToolbar()

    def onToolbarRole(self, index):
        row = self.selectedRow()
        combo = self.studiesTable.cellWidget(row, 1) if row >= 0 else None
        if combo is not None and 0 <= index < combo.count:
            combo.setCurrentIndex(index)
            self.onStudyRoleChanged(row, index)
        self.syncToolbar()

    def onToolbarGated(self, index):
        row = self.selectedRow()
        combo = self.studiesTable.cellWidget(row, 2) if row >= 0 else None
        if combo is not None and combo.enabled and 0 <= index < combo.count:
            combo.setCurrentIndex(index)
            self.onStudyGatedChanged(row, index)
        self.syncToolbar()

    def syncToolbar(self):
        bar = getattr(self, "toolbarBar", None)
        if bar is None:
            return
        try:
            bar.syncMirrors()
            row = self.selectedRow()
            names = [(s.name if s.staticNode else f"{s.name} (gated only)", None) for s in self.studies]
            bar.setItems(self.tbStudy, names, row)
            roleCombo = self.studiesTable.cellWidget(row, 1) if row >= 0 else None
            gatedCombo = self.studiesTable.cellWidget(row, 2) if row >= 0 else None
            for toolbarCombo, panelCombo in ((self.tbRole, roleCombo), (self.tbGated, gatedCombo)):
                items = ([(panelCombo.itemText(i), panelCombo.itemData(i)) for i in range(panelCombo.count)]
                         if panelCombo is not None else [])
                bar.setItems(toolbarCombo, items, panelCombo.currentIndex if panelCombo is not None else -1)
                toolbarCombo.enabled = bool(panelCombo is not None and panelCombo.enabled)
                if panelCombo is not None:
                    font = toolbarCombo.font
                    font.setItalic(bool(panelCombo.font.italic()))
                    toolbarCombo.font = font
            self.tbStudy.enabled = bool(self.studies)
        except Exception:
            logging.debug("MPI: could not update the toolbar:\n" + traceback.format_exc())

    def enter(self):
        self.refreshStudies()

    def cleanup(self):
        self.studiesTimer.stop()
        session.removeListener(self.onSessionChanged)
        self.removeObservers()

    def onSessionChanged(self):
        """Follow the active study chosen elsewhere (e.g. in the toolbar)."""
        self.showCase()
        if self._fillingStudies:
            return
        active = session.activeStudy()
        selected = self.selectedStudy()
        if selected is not None and selected.staticNode is active:
            return
        for row, study in enumerate(self.studies):
            if active is not None and study.staticNode is active:
                self._fillingStudies = True
                try:
                    self.studiesTable.selectRow(row)
                finally:
                    self._fillingStudies = False
                self.updateStudyButtons()
                return

    def onNodeAdded(self, caller=None, event=None, *args):
        if slicer.mrmlScene.IsImporting() or slicer.mrmlScene.IsClosing():
            return
        self.studiesTimer.start()

    def refreshStudies(self):
        with slicer.util.tryWithErrorDisplay("Could not detect the studies.", waitCursor=False):
            selected = self.selectedStudy()
            selectedKey = None
            if selected is not None:
                selectedKey = (selected.staticNode or selected.gatedNode).GetID()
            self.studies = studiesLib.scanStudies()
            self.fillStudiesTable(selectedKey)
            self.patients = caseLib.updateFromDicom(self.studies)
            caseLib.warnIfMixed(self.patients)
            self.showCase()

    # --- case -------------------------------------------------------------------------------
    def showCase(self):
        name, patientId, source = caseLib.currentCase()
        for edit, value in ((self.caseNameEdit, name), (self.caseIdEdit, patientId)):
            if not edit.hasFocus() and edit.text != value:
                edit.text = value
        self.caseSourceLabel.text = {caseLib.SOURCE_DICOM: "From the DICOM headers",
                                     caseLib.SOURCE_MANUAL: "Typed by hand"}.get(source, "No DICOM patient data")
        warning = caseLib.mixedWarning(self.patients)
        self.caseWarningLabel.text = ("\u26a0 " + html.escape(warning).replace("\n", "<br>") + "<br>Set the volumes of another "
                                      "patient to None below.") if warning else ""
        self.caseWarningLabel.visible = bool(warning)

    def onCaseEdited(self):
        name, patientId, _ = caseLib.currentCase()
        if (self.caseNameEdit.text.strip(), self.caseIdEdit.text.strip()) != (name, patientId):
            caseLib.setCase(self.caseNameEdit.text, self.caseIdEdit.text, caseLib.SOURCE_MANUAL)
        self.showCase()

    def onCaseFromDicom(self):
        self.patients = caseLib.updateFromDicom(self.studies, force=True)
        self.showCase()

    def fillStudiesTable(self, selectedKey=None):
        self._fillingStudies = True
        try:
            table = self.studiesTable
            table.clearContents()
            table.setRowCount(len(self.studies))
            sequences = [seq for seq, _, _ in studiesLib.gatedSequences()] if self.studies else []
            selectRow = -1
            for row, study in enumerate(self.studies):
                node = study.staticNode or study.gatedNode
                nameItem = qt.QTableWidgetItem(study.name if study.staticNode else f"{study.name} (gated only)")
                nameItem.setToolTip(node.GetName())
                table.setItem(row, 0, nameItem)

                roleCombo = qt.QComboBox()
                roleCombo.addItems(list(studiesLib.ROLES))
                roleCombo.setCurrentIndex(studiesLib.ROLES.index(study.role))
                font = roleCombo.font
                font.setItalic(study.guessed)
                roleCombo.font = font
                roleCombo.toolTip = "Guessed from the names, pick to confirm" if study.guessed else "Study role"
                roleCombo.connect("activated(int)", lambda index, r=row: self.onStudyRoleChanged(r, index))
                table.setCellWidget(row, 1, roleCombo)

                gatedCombo = qt.QComboBox()
                gatedCombo.addItem("None", "")
                for sequence in sequences:
                    gatedCombo.addItem(sequence.GetName(), sequence.GetID())
                current = study.gatedNode.GetID() if study.gatedNode else ""
                gatedCombo.setCurrentIndex(max(0, gatedCombo.findData(current)))
                gatedCombo.enabled = study.staticNode is not None
                gatedCombo.connect("activated(int)", lambda index, r=row: self.onStudyGatedChanged(r, index))
                table.setCellWidget(row, 2, gatedCombo)

                table.setItem(row, 3, qt.QTableWidgetItem(str(study.gates) if study.gates else "-"))
                if selectedKey and node.GetID() == selectedKey:
                    selectRow = row
            if selectRow < 0:
                current = session.activeStudy()
                for row, study in enumerate(self.studies):
                    if current is not None and study.staticNode is current:
                        selectRow = row
            if selectRow >= 0:
                table.selectRow(selectRow)
        finally:
            self._fillingStudies = False
        self.updateStudyButtons()

    def selectedStudy(self):
        rows = self.studiesTable.selectionModel().selectedRows() if self.studiesTable.selectionModel() else []
        if not rows:
            return None
        row = rows[0].row()
        return self.studies[row] if 0 <= row < len(self.studies) else None

    def updateStudyButtons(self):
        study = self.selectedStudy()
        self.sumGatesButton.enabled = bool(study and study.staticNode is None and study.gatedNode)
        self.nextButton.enabled = any(s.staticNode is not None and s.role != studiesLib.EXCLUDED_ROLE
                                      for s in self.studies)
        self.syncToolbar()

    def onProcess(self):
        """Processing needs an active study: keep the one chosen, else take Study 1 (or any study)."""
        active = session.activeStudy()
        if active is None or studiesLib.isExcluded(active) or not slicer.mrmlScene.IsNodePresent(active):
            pair = session.stressRestStudies()
            candidates = session.candidateStudies()
            study = pair["stress"] or pair["rest"] or (candidates[0] if candidates else None)
            if study is not None:
                session.setActiveStudy(study)
        session.selectStep("processing")

    def onStudySelected(self):
        self.updateStudyButtons()
        if self._fillingStudies:
            return
        study = self.selectedStudy()
        if study and study.staticNode is not None:
            session.setActiveStudy(study.staticNode)

    def onStudyRoleChanged(self, row, index):
        if not (0 <= row < len(self.studies)):
            return
        study = self.studies[row]
        study.role = studiesLib.ROLES[index]
        studiesLib.saveStudy(study)
        self.patients = caseLib.updateFromDicom(self.studies)   # a study set to None no longer counts
        session.notify()
        self.showCase()
        combo = self.studiesTable.cellWidget(row, 1)
        if combo:
            font = combo.font
            font.setItalic(False)
            combo.font = font
            combo.toolTip = "Study role"
        self.syncToolbar()

    def onStudyGatedChanged(self, row, index):
        if not (0 <= row < len(self.studies)):
            return
        study = self.studies[row]
        combo = self.studiesTable.cellWidget(row, 2)
        sequenceId = combo.itemData(index) if combo else ""
        study.gatedNode = slicer.mrmlScene.GetNodeByID(sequenceId) if sequenceId else None
        studiesLib.saveStudy(study)
        # A sequence can belong to one study only
        self.refreshStudies()

    def onSumGates(self):
        study = self.selectedStudy()
        if not (study and study.gatedNode):
            return
        with slicer.util.tryWithErrorDisplay("Could not sum the gates.", waitCursor=True):
            node = studiesLib.sumGates(study.gatedNode)
            node.SetAttribute(studiesLib.ROLE_ATTRIBUTE, study.role)
            session.setActiveStudy(node)
            self.refreshStudies()


class MPIDataRolesTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_ActiveStudy()

    def test_ActiveStudy(self):
        import numpy as np
        volume = slicer.util.addVolumeFromArray(np.ones((8, 8, 8), dtype=np.float32), name="Stress MPI")
        session.setActiveStudy(volume)
        self.assertIs(session.activeStudy(), volume)
        self.assertEqual(session.outputsForStudy(volume), {})
        output = slicer.util.addVolumeFromArray(np.ones((8, 8, 8), dtype=np.float32), name="Stress MPI SA")
        output.SetAttribute("MPI.Orientation", "SA")
        output.SetAttribute("MPI.SourceVolume", volume.GetID())
        self.assertIs(session.outputsForStudy(volume)["SA"], output)
        names = [s.name for s in studiesLib.scanStudies()]
        self.assertIn("Stress MPI", names)
        self.assertNotIn("Stress MPI SA", names)
        session.setActiveStudy(None)
        self.assertIsNone(session.activeStudy())
        caseLib.setCase("Doe, John", "123", caseLib.SOURCE_MANUAL)
        self.assertEqual(caseLib.currentCase(), ("Doe, John", "123", caseLib.SOURCE_MANUAL))
        caseLib.updateFromDicom()   # typed by hand: kept
        self.assertEqual(caseLib.currentCase()[0], "Doe, John")
        self.assertEqual(caseLib.formatPersonName("DOE^JOHN^Q"), "DOE, JOHN Q")
        self.assertEqual(len(caseLib.patientGroups([("a", "DOE^JOHN", "1"), ("b", "Doe, John", ""),
                                                     ("c", "ROE^JANE", "2")])), 2)
        self.delayDisplay("Data roles test passed")

# saved Advanced parameters of every step, this module's constants included (see parameters.py)
parametersLib.applySaved()
