"""Case (patient) name and ID of the exam, shown in Data roles and in the workflow toolbar.

Stored on the MPI session node, so it is saved with the scene:

    MPI.CaseName     patient name as shown ("Family, Given Middle" when read from DICOM)
    MPI.CaseID       patient ID
    MPI.CaseSource   "dicom": read from the DICOM headers of the studies (read again when they change),
                     "manual": typed in Data roles (kept until Read from DICOM is pressed)

The DICOM values come from the subject hierarchy (the patient item of a volume loaded from DICOM, or
placed under it as the DICOM SPECT Recon extension does: DICOM.PatientName / DICOM.PatientID), else
from the DICOM database for volumes with DICOM.instanceUIDs. Studies set to None in Data roles are
left out, so a volume of another patient can be excluded there. The pure functions at the top have
no Slicer dependency.
"""

import logging
import re
import traceback
import unicodedata

# ---------------------------------------------------------------------------
# Settings of this file: edit the values here. A value shown in Advanced parameters takes the one here as its
# built-in default; a value saved with Save parameters overrides it until Restore defaults.
# ---------------------------------------------------------------------------
NAME_ATTRIBUTE = "MPI.CaseName"
ID_ATTRIBUTE = "MPI.CaseID"
SOURCE_ATTRIBUTE = "MPI.CaseSource"
SOURCE_DICOM = "dicom"
SOURCE_MANUAL = "manual"

SH_NAME_ATTRIBUTE = "DICOM.PatientName"   # set by Slicer's DICOM plugins on the patient item
SH_ID_ATTRIBUTE = "DICOM.PatientID"
NAME_TAG = "0010,0010"
ID_TAG = "0010,0020"


def formatPersonName(value):
    """DICOM person name (Family^Given^Middle^Prefix^Suffix; only the first of the '=' groups) as
    "Family, Given Middle"; text without '^' is returned as it is (trimmed)."""
    text = str(value or "").split("=")[0].strip()
    if "^" not in text:
        return " ".join(text.split())
    parts = [" ".join(p.split()) for p in text.split("^")] + [""] * 5
    family, given, middle, prefix, suffix = parts[:5]
    first = " ".join(p for p in (prefix, given, middle) if p)
    name = ", ".join(p for p in (family, first) if p)
    return f"{name} {suffix}".strip() if suffix else name


def _words(text):
    """Comparable words of a name or ID: case, accents, punctuation and word order ignored."""
    text = unicodedata.normalize("NFKD", str(text or "").casefold())
    text = "".join(c for c in text if not unicodedata.combining(c)).replace("ı", "i")
    return tuple(sorted(re.findall(r"[^\W_]+", text)))


def sameName(a, b):
    return _words(formatPersonName(a)) == _words(formatPersonName(b))


def patientGroups(entries):
    """Distinct patients of [(label, name, ID)]: [{"name", "id", "labels"}] in order of first appearance.
    Entries with the same name and ID are one patient (an empty name or ID matches any); entries with
    neither are left out."""
    groups = []
    for label, name, patientId in entries:
        name, patientId = formatPersonName(name), " ".join(str(patientId or "").split())
        if not name and not patientId:
            continue
        for group in groups:
            nameMatches = not name or not group["name"] or _words(name) == _words(group["name"])
            idMatches = not patientId or not group["id"] or _words(patientId) == _words(group["id"])
            if nameMatches and idMatches:
                group["name"] = group["name"] or name
                group["id"] = group["id"] or patientId
                if label not in group["labels"]:
                    group["labels"].append(label)
                break
        else:
            groups.append({"name": name, "id": patientId, "labels": [label]})
    return groups


def caseText(name, patientId):
    """One line for the toolbar and the overview: "Name · ID"."""
    return " · ".join(p for p in (name, patientId) if p)


def mixedWarning(groups):
    """Warning text when the studies belong to more than one patient, else ""."""
    if len(groups) < 2:
        return ""
    lines = [f"{caseText(g['name'], g['id']) or '?'}: {', '.join(g['labels'])}" for g in groups]
    return ("The studies belong to more than one patient (different names or IDs in the DICOM headers):\n"
            + "\n".join(lines))


# --- Slicer ------------------------------------------------------------------------------------------
def _slicer():
    import slicer
    return slicer


def _fromSubjectHierarchy(node):
    """(name, ID) of the nearest subject hierarchy ancestor carrying the DICOM patient attributes."""
    slicer = _slicer()
    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    item = shNode.GetItemByDataNode(node) if shNode else 0
    sceneItem = shNode.GetSceneItemID() if shNode else 0
    while item and item != sceneItem:
        name = shNode.GetItemAttribute(item, SH_NAME_ATTRIBUTE) or ""
        patientId = shNode.GetItemAttribute(item, SH_ID_ATTRIBUTE) or ""
        if name or patientId:
            return name, patientId
        item = shNode.GetItemParent(item)
    return "", ""


def _fromDatabase(node):
    slicer = _slicer()
    uids = (node.GetAttribute("DICOM.instanceUIDs") or "").split()
    if not uids or not getattr(slicer, "dicomDatabase", None):
        return "", ""
    fileName = slicer.dicomDatabase.fileForInstance(uids[0])
    if not fileName:
        return "", ""
    return (slicer.dicomDatabase.fileValue(fileName, NAME_TAG) or "",
            slicer.dicomDatabase.fileValue(fileName, ID_TAG) or "")


def dicomPatient(node):
    """(name, ID) of a volume or gated sequence from DICOM, ("", "") when unknown."""
    if node is None:
        return "", ""
    candidates = [node]
    if node.IsA("vtkMRMLSequenceNode"):
        try:
            browser = _slicer().modules.sequences.logic().GetFirstBrowserNodeForSequenceNode(node)
            proxy = browser.GetProxyNode(node) if browser else None
            if proxy is not None:
                candidates.append(proxy)
        except Exception:
            pass
        if node.GetNumberOfDataNodes():
            candidates.append(node.GetNthDataNode(0))
    for candidate in candidates:
        for reader in (_fromSubjectHierarchy, _fromDatabase):
            try:
                name, patientId = reader(candidate)
            except Exception:
                logging.debug("MPI case: could not read the DICOM patient", exc_info=True)
                continue
            if name or patientId:
                return formatPersonName(name), " ".join(str(patientId).split())
    return "", ""


def scanPatients(studies=None):
    """Distinct patients of the studies in the scene (Data roles None left out), see patientGroups.
    Each group also gets "nodes": the IDs of its volumes and sequences."""
    from MyocardialPerfusionImagingLib import studies as studiesLib
    if studies is None:
        studies = studiesLib.scanStudies()
    entries, nodeIds = [], {}
    for study in studies:
        if study.role == studiesLib.EXCLUDED_ROLE:
            continue
        for node in (study.staticNode, study.gatedNode):
            if node is None:
                continue
            name, patientId = dicomPatient(node)
            entries.append((node.GetName(), name, patientId))
            nodeIds.setdefault(node.GetName(), []).append(node.GetID())
    groups = patientGroups(entries)
    for group in groups:
        group["nodes"] = [i for label in group["labels"] for i in nodeIds.get(label, [])]
    return groups


def currentCase():
    """(name, ID, source) stored for the scene; source "" when nothing is stored yet."""
    from MyocardialPerfusionImagingLib import session
    node = session.sessionNode(create=False)
    if node is None:
        return "", "", ""
    return (node.GetAttribute(NAME_ATTRIBUTE) or "", node.GetAttribute(ID_ATTRIBUTE) or "",
            node.GetAttribute(SOURCE_ATTRIBUTE) or "")


def setCase(name, patientId, source=SOURCE_MANUAL):
    """Store the case; listeners are told only when something changed."""
    from MyocardialPerfusionImagingLib import session
    name, patientId = " ".join(str(name or "").split()), " ".join(str(patientId or "").split())
    if (name, patientId, source) == currentCase():
        return False
    node = session.sessionNode()
    node.SetAttribute(NAME_ATTRIBUTE, name)
    node.SetAttribute(ID_ATTRIBUTE, patientId)
    node.SetAttribute(SOURCE_ATTRIBUTE, source)
    session.notify()
    return True


def updateFromDicom(studies=None, force=False):
    """Fill the case from the DICOM headers unless it was typed by hand (force: also then). With more
    than one patient, the one of the active study is taken (else the first). Returns the patients."""
    try:
        groups = scanPatients(studies)
    except Exception:
        logging.warning("MPI case: could not read the patients of the studies:\n" + traceback.format_exc())
        return []
    _, _, source = currentCase()
    if source == SOURCE_MANUAL and not force:
        return groups
    if not groups:
        if source == SOURCE_DICOM or force:
            setCase("", "", SOURCE_DICOM)
        return groups
    from MyocardialPerfusionImagingLib import session
    active = session.activeStudy()
    chosen = next((g for g in groups if active is not None and active.GetID() in g["nodes"]), groups[0])
    setCase(chosen["name"], chosen["id"], SOURCE_DICOM)
    return groups


_warned = set()   # mixed-patient sets already shown in a message box this Slicer session


def warnIfMixed(groups):
    """Message box (once per set of patients) when the studies belong to more than one patient."""
    text = mixedWarning(groups)
    if not text:
        return
    key = tuple(sorted((tuple(_words(g["name"])), tuple(_words(g["id"]))) for g in groups))
    if key in _warned:
        return
    _warned.add(key)
    slicer = _slicer()
    if slicer.app.commandOptions().noMainWindow:
        logging.warning("MPI: " + text)
        return
    import qt
    message = (text + "\n\nCheck that the right studies are used: set the volumes of another patient to None "
               "in Data roles.")
    qt.QTimer.singleShot(0, lambda: slicer.util.warningDisplay(message, windowTitle="Myocardial Perfusion Imaging"))
