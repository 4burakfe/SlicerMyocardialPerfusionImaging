"""State shared by the MPI modules: the active study and the reoriented volumes of each study.

The active study is a node reference on a singleton scripted module node, so it is saved with the
scene. Modules and the workflow toolbar register a listener to follow changes.
"""

import logging
import traceback

import numpy as np
import slicer

from MyocardialPerfusionImagingLib import studies as studiesLib

# ---------------------------------------------------------------------------
# Settings of this file: edit the values here. A value shown in Advanced parameters takes the one here as its
# built-in default; a value saved with Save parameters overrides it until Restore defaults.
# ---------------------------------------------------------------------------
SESSION_TAG = "MPISession"
ACTIVE_STUDY_REFERENCE = "MPI.ActiveStudy"

HUB_MODULE = "MyocardialPerfusionImaging"
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
STUDY_SLOT_ROLES = {"stress": ("Stress",), "rest": ("Rest", "Redistribution")}   # roles filling each place
STUDY_SLOTS = ("stress", "rest")   # Study 1 and Study 2 of the two-study screens
STUDY_REFERENCES = {"stress": "MPI.Study1", "rest": "MPI.Study2"}
STUDY_PAIR_ATTRIBUTE = "MPI.StudyPairChosen"   # "1": Study 1 / 2 were picked by the user, not by role
SLOT_FALLBACK_TITLES = {"stress": "Study 1", "rest": "Study 2"}
AXIS_ALIGNED_MIN_DOT = 0.9         # a view counts as along an axis from this cosine


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
            restoreAxisOrder(node)
            outputs[orientation] = node   # later nodes in the scene are newer
    return outputs


def expectedAxes(node):
    """(screen right, screen up, stacking direction) a reoriented volume was made with, from the frame on its
    MPI.* attributes (MPIProcessing orientationAxes), or None."""
    try:
        u = np.array([float(x) for x in node.GetAttribute("MPI.LongAxis").split()])
        w = np.array([float(x) for x in node.GetAttribute("MPI.AnteriorAxis").split()])
    except (AttributeError, ValueError):
        return None
    u = u / np.linalg.norm(u)
    w = w - np.dot(w, u) * u
    w = w / np.linalg.norm(w)
    s = np.cross(u, w)
    return {"SA": (-s, w, -u), "HLA": (-s, u, w), "VLA": (u, w, -s)}.get(node.GetAttribute("MPI.Orientation"))


def axisOrder(ijkToRas, expected):
    """(source axis, sign) for each expected axis: which IJK column of ijkToRas (4x4) runs along it, or None when
    the columns do not line up with the expected axes."""
    columns = np.asarray(ijkToRas, dtype=float)[:3, :3]
    columns = columns / np.linalg.norm(columns, axis=0)
    order = []
    for axis in expected:
        dots = columns.T @ np.asarray(axis, dtype=float)
        source = int(np.argmax(np.abs(dots)))
        if abs(dots[source]) < AXIS_ALIGNED_MIN_DOT or source in [o[0] for o in order]:
            return None
        order.append((source, 1 if dots[source] > 0 else -1))
    return order


def reorderAxes(array, ijkToRas, order):
    """array [k, j, i] and ijkToRas with the IJK axes put in the given order (axisOrder): the same voxels at the
    same places, only indexed differently. Returns (array, ijkToRas)."""
    matrix = np.asarray(ijkToRas, dtype=float)
    dims = array.shape[::-1]   # (i, j, k)
    # numpy axis of IJK axis a is 2 - a
    out = np.transpose(array, [2 - order[axis][0] for axis in (2, 1, 0)])
    start = np.zeros(4)
    start[3] = 1.0
    result = np.eye(4)
    for axis, (source, sign) in enumerate(order):
        result[:3, axis] = sign * matrix[:3, source]
        if sign < 0:
            out = np.flip(out, axis=2 - axis)
            start[source] = dims[source] - 1
    result[:3, 3] = (matrix @ start)[:3]
    return np.ascontiguousarray(out), result


def restoreAxisOrder(node):
    """Put the slices of a reoriented volume back in the order the steps show them (SA apex -> base, HLA
    inferior -> anterior, VLA septum -> lateral). The volumes are made left-handed (screen right, screen up,
    stacking direction); Slicer reads such a volume back from a saved scene (.mrb, DICOM scene) with its slice
    axis reversed: the same voxels at the same places, but slice 1 at the base. True when it was reordered."""
    expected = expectedAxes(node)
    if expected is None or node.GetImageData() is None:
        return False
    import vtk
    ijkToRas = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(ijkToRas)
    matrix = slicer.util.arrayFromVTKMatrix(ijkToRas)
    order = axisOrder(matrix, expected)
    if order is None or order == [(0, 1), (1, 1), (2, 1)]:
        return False
    array, matrix = reorderAxes(slicer.util.arrayFromVolume(node), matrix, order)
    slicer.util.updateVolumeFromArray(node, array)
    node.SetIJKToRASMatrix(slicer.util.vtkMatrixFromArray(matrix))
    logging.info(f"MPI: slice order of {node.GetName()} restored (axes {order})")
    return True


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


def roleTitles(roles):
    """{slot: label} of Study 1 / Study 2 from their roles ({slot: role or None}): the role itself
    ("Stress", "Rest", "Viability", ...), "Study 1" / "Study 2" without a study, and numbered when
    both studies have the same role ("Other 1", "Other 2")."""
    titles = {slot: roles.get(slot) or SLOT_FALLBACK_TITLES[slot] for slot in STUDY_SLOTS}
    if all(roles.get(slot) for slot in STUDY_SLOTS) and titles["stress"] == titles["rest"]:
        titles = {slot: f"{titles[slot]} {index + 1}" for index, slot in enumerate(STUDY_SLOTS)}
    return titles


def studyRoleTitle(node):
    """Label of a study from its role in Data roles ("" without a study); see roleTitles."""
    if node is None:
        return ""
    role = studiesLib.studyRole(node)[0]
    return "" if role == studiesLib.EXCLUDED_ROLE else role


def slotTitles(pair=None):
    """{"stress": label, "rest": label} of Study 1 / Study 2 (pair: {slot: node}, default
    stressRestStudies()) for views, plots and reports, e.g. Rest / Viability for a PET rest + FDG exam."""
    if pair is None:
        pair = stressRestStudies()
    return roleTitles({slot: studyRoleTitle(pair.get(slot)) for slot in STUDY_SLOTS})


def isStressRest(pair=None):
    """True when Study 1 is a Stress and Study 2 a Rest (or Redistribution) study: the stress-rest
    readouts (ischemia difference, TID) apply."""
    if pair is None:
        pair = stressRestStudies()
    roles = {slot: studyRoleTitle(pair.get(slot)) for slot in STUDY_SLOTS}
    return roles["stress"] in STUDY_SLOT_ROLES["stress"] and roles["rest"] in STUDY_SLOT_ROLES["rest"]


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


def resetToHub():
    """Show the overview page again (after a scene close or load): at once when the hub is open, else the
    next time it is opened."""
    if _currentStep == HUB_STEP:
        return
    try:
        if slicer.util.selectedModule() == HUB_MODULE:
            slicer.util.getModuleWidget(HUB_MODULE).showStep(HUB_STEP)   # leaves the step (exit) first
            return
    except Exception:
        logging.warning("MPI: could not show the overview:\n" + traceback.format_exc())
    setCurrentStep(HUB_STEP)


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

