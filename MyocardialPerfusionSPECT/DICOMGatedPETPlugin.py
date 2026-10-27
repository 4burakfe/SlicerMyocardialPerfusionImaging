"""
DICOM Gated PET plugin for 3D Slicer

Loads ECG gated PET series (Modality PT, Series Type GATED\\IMAGE or Number of Time Slots > 1, e.g. GE
Discovery cardiac gated PET) from the DICOM browser as a sequence of gate volumes, in SUVbw or Bq/ml.
Slicer's PET SUV and MultiVolume plugins stack all the images of such a series into one volume or fail
to split it; this plugin orders them by Image Index (0054,1330), else Trigger Time, else repeated slice
positions, and applies every image's own rescale slope. The work is done in
MyocardialPerfusionSPECTLib/gatedpet.py (no Slicer dependency).

Loadables (one series gives both, the DICOM browser selects the first):
    "<series> gated SUVbw"    when weight, dose, half life and times allow SUVbw (confidence 1.05)
    "<series> gated Bq/ml"    activity concentration (confidence 1.04)

Result: a sequence (index "Gate") with a browser and a "(current gate)" proxy volume, placed in the
series' subject hierarchy study. Every gate carries DICOM.Modality = PT, DICOM.instanceUIDs and GatedPET.*
attributes (Gate, Units, SUV inputs, trigger time, frame duration and the ECG gating fields read by
Gated QC), so Data roles finds it as a gated study.
"""

import os
import sys
import logging
import traceback

import numpy as np
import qt
import vtk
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleWidget, ScriptedLoadableModuleTest
from DICOMLib import DICOMPlugin, DICOMLoadable

_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from MyocardialPerfusionSPECTLib import gatedpet  # noqa: E402

PREFIX = gatedpet.GATE_ATTRIBUTE_PREFIX
UNIT_CODES = {gatedpet.UNITS_ACTIVITY: ("Bq/ml", "UCUM", "Becquerels/milliliter"),
              gatedpet.UNITS_SUV: ("{SUVbw}g/ml", "UCUM", "Standardized Uptake Value body weight")}


# ---------------------------------------------------------------------------
# Module registration (hidden module, only used by the DICOM browser)
# ---------------------------------------------------------------------------
class DICOMGatedPETPlugin(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "DICOM Gated PET Plugin"
        parent.categories = ["Developer Tools.DICOM Plugins"]
        parent.contributors = ["Burak Demir"]
        parent.helpText = ("Loads ECG gated PET series from the DICOM browser as a sequence of gates "
                           "in SUVbw or Bq/ml.")
        parent.acknowledgementText = "SUV calculation follows the QIBA vendor-neutral pseudo-code."
        parent.hidden = True

        try:
            slicer.modules.dicomPlugins
        except AttributeError:
            slicer.modules.dicomPlugins = {}
        slicer.modules.dicomPlugins["DICOMGatedPETPlugin"] = DICOMGatedPETPluginClass


# ---------------------------------------------------------------------------
# DICOM plugin: examine + load
# ---------------------------------------------------------------------------
class DICOMGatedPETPluginClass(DICOMPlugin):
    def __init__(self):
        super().__init__()
        self.loadType = "Gated PET"
        self.tags["modality"] = "0008,0060"
        self.tags["seriesDescription"] = "0008,103e"
        self.tags["seriesType"] = "0054,1000"
        self.tags["timeSlots"] = "0054,0071"

    def examineForImport(self, fileLists):
        return self.examine(fileLists)

    def examine(self, fileLists):
        loadables = []
        for files in fileLists:
            if not files:
                continue
            try:
                loadables += self.examineFiles(list(files))
            except Exception:
                logging.warning("Gated PET plugin could not examine a series:\n" + traceback.format_exc())
        return loadables

    def examineFiles(self, files):
        db = slicer.dicomDatabase
        first = files[0]
        if not gatedpet.isGated(db.fileValue(first, self.tags["modality"]), db.fileValue(first, self.tags["seriesType"]),
                                db.fileValue(first, self.tags["timeSlots"])):
            return []
        import pydicom
        header = pydicom.dcmread(first, stop_before_pixels=True)
        slots = int(gatedpet._float(getattr(header, "NumberOfTimeSlots", None), 0) or 0)
        description = (db.fileValue(first, self.tags["seriesDescription"]) or "PET").strip() or "PET"
        gatesText = f"{slots} gates, " if slots > 1 else ""
        suv = gatedpet.suvInfo([header])

        loadables = []
        if suv["suvScale"] is not None:
            loadables.append(self._loadable(files, f"{description} gated SUVbw", gatedpet.UNITS_SUV, 1.05,
                                            f"Gated PET ({gatesText}{len(files)} images) in SUVbw"))
        if suv["activityScale"] is not None:
            tooltip = f"Gated PET ({gatesText}{len(files)} images) in Bq/ml"
            if suv["suvScale"] is None and suv["problems"]:
                tooltip += "\nSUV not available: " + ", ".join(suv["problems"])
            loadables.append(self._loadable(files, f"{description} gated Bq/ml", gatedpet.UNITS_ACTIVITY, 1.04, tooltip))
        if loadables:
            for loadable in loadables[1:]:
                loadable.selected = False
            if suv["problems"] and suv["suvScale"] is None:
                loadables[0].warning = "SUV not available: " + ", ".join(suv["problems"])
        return loadables

    def _loadable(self, files, name, units, confidence, tooltip):
        loadable = DICOMLoadable()
        loadable.files = files
        loadable.name = name
        loadable.tooltip = tooltip
        loadable.selected = True
        # Above 1.0 (MultiVolumeImporter's) so the browser keeps this one for the series
        loadable.confidence = confidence
        _UNITS_FOR_LOADABLE[name + "|" + files[0]] = units
        return loadable

    def load(self, loadable):
        units = _UNITS_FOR_LOADABLE.get(loadable.name + "|" + loadable.files[0])
        if units is None:
            units = gatedpet.UNITS_SUV if "SUVbw" in loadable.name else gatedpet.UNITS_ACTIVITY
        try:
            proxyNode = self.loadGated(list(loadable.files), loadable.name, units, loadable)
        except Exception as error:
            logging.error("Gated PET loading failed:\n" + traceback.format_exc())
            slicer.util.errorDisplay(f"Could not load the gated PET series:\n{error}")
            return None
        return proxyNode

    def loadGated(self, files, name, units, loadable=None):
        import pydicom
        progress = slicer.util.createProgressDialog(labelText="Reading gated PET images...", maximum=len(files) + 1)
        datasets = []
        try:
            for i, path in enumerate(files):
                datasets.append(pydicom.dcmread(path))
                if i % 20 == 0:
                    progress.setValue(i)
                    slicer.app.processEvents()
        finally:
            progress.close()

        gates, method = gatedpet.groupGates(datasets)
        suv = gatedpet.suvInfo(datasets)
        scale = suv["suvScale"] if units == gatedpet.UNITS_SUV else suv["activityScale"]
        if scale is None:
            raise ValueError(f"{units} not available: " + ", ".join(suv["problems"] or ["unknown units"]))

        origin, spacing, direction = gatedpet.geometry(gates[0])
        ijkToRas = vtk.vtkMatrix4x4()
        for r, row in enumerate(gatedpet.lpsToRasMatrix(origin, spacing, direction)):
            for c, value in enumerate(row):
                ijkToRas.SetElement(r, c, value)

        common = {"Units": units, "Scale": f"{scale:.8g}", "GateOrdering": method, "Gates": str(len(gates))}
        for key in ("WeightKg", "InjectedDoseBq", "DecayedDoseBq", "HalfLifeS", "InjectionTime", "ScanStart",
                    "DecayCorrection", "DecayReference", "SUVSource"):
            if suv.get(key) not in (None, ""):
                common[key] = suv[key] if isinstance(suv[key], str) else f"{suv[key]:.6g}"
        if suv["problems"]:
            common["SUVProblems"] = "; ".join(suv["problems"])
        common.update({k: str(v) for k, v in gatedpet.gatingInfo(gates).items()})

        baseName = slicer.mrmlScene.GenerateUniqueName(name)
        sequenceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", baseName)
        sequenceNode.SetIndexName("Gate")
        sequenceNode.SetIndexUnit("")
        sequenceNode.SetAttribute("DICOM.Modality", "PT")
        upper = 0.0
        for g, gate in enumerate(gates, start=1):
            array = gatedpet.gateArray(gate, scale)
            upper = max(upper, float(np.percentile(array, 99.9)))
            gateNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{baseName} G{g}")
            slicer.util.updateVolumeFromArray(gateNode, array)
            gateNode.SetIJKToRASMatrix(ijkToRas)
            gateNode.SetAttribute("DICOM.Modality", "PT")
            gateNode.SetAttribute("DICOM.instanceUIDs", " ".join(str(ds.SOPInstanceUID) for ds in gate))
            for key, value in common.items():
                gateNode.SetAttribute(PREFIX + key, value)
            gateNode.SetAttribute(PREFIX + "Gate", str(g))
            trigger, duration = gatedpet.gateTiming(gate)
            if trigger is not None:
                gateNode.SetAttribute(PREFIX + "TriggerTimeMs", str(trigger))
            if duration is not None:
                gateNode.SetAttribute(PREFIX + "FrameDurationMs", str(duration))
            setVoxelUnits(gateNode, units)
            sequenceNode.SetDataNodeAtValue(gateNode, str(g))
            slicer.mrmlScene.RemoveNode(gateNode)

        browserNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", f"{baseName} browser")
        slicer.modules.sequences.logic().AddSynchronizedNode(sequenceNode, None, browserNode)
        if hasattr(browserNode, "SetAndObservePrimarySequenceNodeID"):
            browserNode.SetAndObservePrimarySequenceNodeID(sequenceNode.GetID())
        else:
            browserNode.SetAndObserveMasterSequenceNodeID(sequenceNode.GetID())
        browserNode.SetPlaybackRateFps(len(gates))
        browserNode.SetPlaybackLooped(True)

        proxyNode = browserNode.GetProxyNode(sequenceNode)
        proxyNode.SetName(f"{baseName} (current gate)")
        proxyNode.CreateDefaultDisplayNodes()
        displayNode = proxyNode.GetDisplayNode()
        if displayNode:
            colorNode = slicer.mrmlScene.GetFirstNodeByName("PET-DICOM")
            if colorNode:
                displayNode.SetAndObserveColorNodeID(colorNode.GetID())
            # One window for all gates, so the cine does not flicker
            displayNode.AutoWindowLevelOff()
            displayNode.SetWindowLevelMinMax(0.0, upper if upper > 0 else 1.0)
            displayNode.SetInterpolate(True)

        if loadable is not None:
            self._placeInSubjectHierarchy(loadable, sequenceNode, (browserNode, proxyNode))
        logging.info(f"Gated PET: {len(gates)} gates x {len(gates[0])} slices loaded as '{baseName}' in {units} "
                     f"(ordered by {method}, scale {scale:.6g})")

        proxyId = proxyNode.GetID()
        qt.QTimer.singleShot(0, lambda: _show(proxyId))
        return proxyNode

    def _placeInSubjectHierarchy(self, loadable, sequenceNode, others):
        try:
            self.addSeriesInSubjectHierarchy(loadable, sequenceNode)
            shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            parent = shNode.GetItemParent(shNode.GetItemByDataNode(sequenceNode))
            if parent:
                for node in others:
                    item = shNode.GetItemByDataNode(node) or shNode.CreateItem(parent, node)
                    shNode.SetItemParent(item, parent)
        except Exception:
            logging.warning("Could not place the gated PET in the subject hierarchy:\n" + traceback.format_exc())


class DICOMGatedPETPluginWidget(ScriptedLoadableModuleWidget):
    """No interface: the plugin works inside the DICOM browser."""


# Loadable name + first file -> units (Qt-wrapped loadables can lose Python attributes)
_UNITS_FOR_LOADABLE = {}


def setVoxelUnits(volumeNode, units):
    try:
        entry = slicer.vtkCodedEntry()
        entry.SetValueSchemeMeaning(*UNIT_CODES[units])
        volumeNode.SetVoxelValueUnits(entry)
    except Exception:
        logging.warning("Could not set voxel value units:\n" + traceback.format_exc())


def _show(volumeNodeID):
    volumeNode = slicer.mrmlScene.GetNodeByID(volumeNodeID)
    if volumeNode is None:
        return
    try:
        slicer.util.setSliceViewerLayers(background=volumeNode, fit=True)
    except Exception:
        logging.warning("Could not show the gated PET:\n" + traceback.format_exc())


# ---------------------------------------------------------------------------
# Test: a synthetic GE-like gated PET written to a temporary folder and loaded through the plugin
# ---------------------------------------------------------------------------
class DICOMGatedPETPluginTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.test_LoadSyntheticGatedPet()

    def test_LoadSyntheticGatedPet(self):
        import tempfile
        import pydicom
        folder = tempfile.mkdtemp(prefix="gatedpet")
        files, expected = gatedpet.writeSyntheticSeries(folder)
        gates, method = gatedpet.groupGates([pydicom.dcmread(f) for f in files])
        self.assertEqual((len(gates), len(gates[0])), (4, 6))
        proxy = DICOMGatedPETPluginClass().loadGated(files, "Synthetic gated SUVbw", gatedpet.UNITS_SUV)
        browser = slicer.modules.sequences.logic().GetFirstBrowserNodeForProxyNode(proxy)
        sequence = browser.GetMasterSequenceNode() if not hasattr(browser, "GetPrimarySequenceNode") \
            else browser.GetPrimarySequenceNode()
        self.assertEqual(sequence.GetNumberOfDataNodes(), 4)
        for g in range(4):
            node = sequence.GetNthDataNode(g)
            self.assertEqual(node.GetAttribute(PREFIX + "Gate"), str(g + 1))
            np.testing.assert_allclose(slicer.util.arrayFromVolume(node), expected[g], rtol=1e-4)
        self.assertAlmostEqual(sequence.GetNthDataNode(0).GetSpacing()[2], 3.27, places=3)
        from MyocardialPerfusionSPECTLib import studies
        self.assertIn(sequence.GetID(), [seq.GetID() for seq, _, _ in studies.gatedSequences()])
        self.delayDisplay("Gated PET plugin test passed")
