"""State shared by the MPI modules: the active study and the reoriented volumes of each study.

The active study is a node reference on a singleton scripted module node, so it is saved with the
scene. Modules and the workflow toolbar register a listener to follow changes.
"""

import logging
import traceback

import slicer

from MyocardialPerfusionSPECTLib import studies as studiesLib

SESSION_TAG = "MPISession"
ACTIVE_STUDY_REFERENCE = "MPI.ActiveStudy"

HUB_MODULE = "MyocardialPerfusionSPECT"
HUB_STEP = "hub"

# Workflow steps: hidden modules whose panels the hub shows. (key, module name, title, explanation)
STEPS = [
    ("roles", "MPIDataRoles", "Data roles", "Studies, roles and gated sequences"),
    ("processing", "MPIProcessing", "Processing", "Reorientation to the cardiac axes"),
    ("myocardium", "MPIMyocardium", "Myocardium", "LV wall surfaces and volumes (automatic)"),
    ("perfusion", "MPIReview", "Perfusion", "Stress / rest slices, polar maps and 3D uptake"),
    ("gatedqc", "MPIGatedQC", "Gated QC", "Frames segmented, counts per frame and ECG gating checks"),
    ("gated", "MPIGated", "Gated Review", "Cine slices, perfusion and thickening polar maps, LV volumes and EF, cine 3D wall"),
]
MODULE_NAMES = (HUB_MODULE,)   # the toolbar is shown while one of these is the current module

_listeners = []
_currentStep = HUB_STEP


def currentStep():
    """Step shown in the hub: HUB_STEP (overview) or a key of STEPS."""
    return _currentStep


def setCurrentStep(key):
    global _currentStep
    if key != _currentStep:
        _currentStep = key
        notify()


def addListener(callback):
    if callback not in _listeners:
        _listeners.append(callback)


def removeListener(callback):
    if callback in _listeners:
        _listeners.remove(callback)


def notify():
    for callback in list(_listeners):
        try:
            callback()
        except Exception:
            logging.warning("MPI session listener failed:\n" + traceback.format_exc())


def sessionNode(create=True):
    scene = slicer.mrmlScene
    node = scene.GetSingletonNode(SESSION_TAG, "vtkMRMLScriptedModuleNode")
    if node is None and create:
        node = slicer.vtkMRMLScriptedModuleNode()
        node.SetSingletonTag(SESSION_TAG)
        node.SetName("MPI session")
        node.SetHideFromEditors(True)
        node = scene.AddNode(node)
    return node


def activeStudy():
    """The static volume of the active study, or None."""
    node = sessionNode(create=False)
    return node.GetNodeReference(ACTIVE_STUDY_REFERENCE) if node else None


def setActiveStudy(volumeNode):
    if volumeNode is activeStudy():
        return
    sessionNode().SetNodeReferenceID(ACTIVE_STUDY_REFERENCE, volumeNode.GetID() if volumeNode else None)
    notify()


def outputsForStudy(volumeNode):
    """{orientation: volume} of the reoriented volumes made from this study (the newest of each)."""
    outputs = {}
    if volumeNode is None:
        return outputs
    sourceId = volumeNode.GetID()
    for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
        orientation = node.GetAttribute("MPI.Orientation")
        if orientation and node.GetAttribute("MPI.SourceVolume") == sourceId:
            outputs[orientation] = node   # later nodes in the scene are newer
    return outputs


STUDY_SLOT_ROLES = {"stress": ("Stress",), "rest": ("Rest", "Redistribution")}   # roles filling each place
STUDY_SLOTS = ("stress", "rest")   # Study 1 and Study 2 of the two-study screens
STUDY_REFERENCES = {"stress": "MPI.Study1", "rest": "MPI.Study2"}
STUDY_PAIR_ATTRIBUTE = "MPI.StudyPairChosen"   # "1": Study 1 / 2 were picked by the user, not by role


def candidateStudies():
    """Static volumes that can be Study 1 / Study 2: the SPECT / PET studies not set to None."""
    return [n for n in studiesLib.staticVolumes() if not studiesLib.isExcluded(n)]


def _defaultStudyPair():
    """Study 1 / Study 2 from the roles (Data roles), preferring the active study, else the newest
    processed one; a study without a stress or rest role counts as Study 1."""
    active = activeStudy()
    if studiesLib.isExcluded(active):
        active = None
    candidates = candidateStudies()
    processed = [n for n in candidates if len(outputsForStudy(n)) == 3]
    if active is not None and not any(n is active for n in processed):
        processed.append(active)
    roles = {n.GetID(): studiesLib.studyRole(n)[0] for n in processed}
    found = {}
    for slot, slotRoles in STUDY_SLOT_ROLES.items():
        withRole = [n for n in processed if roles[n.GetID()] in slotRoles]
        if not withRole:   # not processed yet: any study of that role
            withRole = [n for n in candidates if studiesLib.studyRole(n)[0] in slotRoles]
        if active is not None and any(n is active for n in withRole):
            found[slot] = active
        else:
            found[slot] = withRole[-1] if withRole else None
    if found["stress"] is None and active is not None and active is not found["rest"]:
        found["stress"] = active
    return found


def stressRestStudies():
    """{"stress": node, "rest": node}: Study 1 and Study 2 of the two-study screens, as picked in
    the Study 1 / Study 2 boxes of the steps, else from the roles. Either can be None."""
    node = sessionNode(create=False)
    if node is not None and node.GetAttribute(STUDY_PAIR_ATTRIBUTE) == "1":
        scene = slicer.mrmlScene
        found = {}
        for slot in STUDY_SLOTS:
            study = node.GetNodeReference(STUDY_REFERENCES[slot])
            ok = study is not None and scene.IsNodePresent(study) and not studiesLib.isExcluded(study)
            found[slot] = study if ok else None
        if found["stress"] is not None or found["rest"] is not None:
            return found
    return _defaultStudyPair()


def setStudyPair(stress, rest):
    """Pick Study 1 (stress) and Study 2 (rest) by hand; None for none. (None, None) and a later
    resetStudyPair() go back to the roles."""
    node = sessionNode()
    for slot, study in (("stress", stress), ("rest", rest)):
        node.SetNodeReferenceID(STUDY_REFERENCES[slot], study.GetID() if study is not None else None)
    node.SetAttribute(STUDY_PAIR_ATTRIBUTE, "1" if (stress is not None or rest is not None) else "0")
    notify()


def setStudySlot(slot, study):
    """Pick one of Study 1 / Study 2; picking the study already in the other place swaps them."""
    pair = stressRestStudies()
    other = "rest" if slot == "stress" else "stress"
    if study is not None and pair[other] is study:
        pair[other] = pair[slot]
    pair[slot] = study
    setStudyPair(pair["stress"], pair["rest"])


def resetStudyPair():
    node = sessionNode(create=False)
    if node is not None:
        node.SetAttribute(STUDY_PAIR_ATTRIBUTE, "0")
        notify()


def myocardiumForStudy(volumeNode):
    """Segmentation node of the LV wall detected for this study in Myocardium (the newest), or None."""
    if volumeNode is None:
        return None
    found = None
    for node in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
        if node.GetAttribute("MPI.MyocardiumOf") == volumeNode.GetID():
            found = node
    return found


def stepModuleName(key):
    for stepKey, moduleName, _, _ in STEPS:
        if stepKey == key:
            return moduleName
    return None


def selectStep(key):
    """Open the hub on a step (HUB_STEP for the overview). Hidden modules cannot be selected in
    Slicer's module selector, so the hub shows their panels itself."""
    slicer.util.selectModule(HUB_MODULE)
    try:
        slicer.util.getModuleWidget(HUB_MODULE).showStep(key)
    except Exception:
        logging.warning(f"MPI: could not open step {key}:\n" + traceback.format_exc())


STATE_DONE, STATE_AVAILABLE, STATE_LOCKED = "done", "available", "locked"


def stepStates(studyList, active):
    """{step key: done / available / locked} for the studies with a static volume and the active study."""
    rolesDone = any(not study.guessed for study in studyList)
    processed = len(outputsForStudy(active)) >= 3
    return {
        "roles": STATE_DONE if rolesDone else STATE_AVAILABLE,
        "processing": STATE_DONE if processed else (STATE_AVAILABLE if active is not None else STATE_LOCKED),
        "myocardium": (STATE_DONE if myocardiumForStudy(active) is not None
                       else (STATE_AVAILABLE if processed else STATE_LOCKED)),
        "perfusion": STATE_AVAILABLE if processed else STATE_LOCKED,
        "gatedqc": STATE_AVAILABLE if myocardiumForStudy(active) is not None else STATE_LOCKED,
        "gated": STATE_AVAILABLE if myocardiumForStudy(active) is not None else STATE_LOCKED,
    }

