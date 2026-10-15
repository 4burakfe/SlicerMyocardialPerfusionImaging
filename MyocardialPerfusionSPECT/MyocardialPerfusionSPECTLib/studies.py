"""Studies of a perfusion exam: which volume is stress, rest, viability..., and its gated images.

A study is one static (or summed-gate) SPECT or PET volume, optionally paired with a gated sequence.
Everything is stored as node attributes, so it is saved with the scene and survives a reload:

    on the static volume   MPI.StudyRole        Stress / Rest / Redistribution / Viability / Other
                           MPI.GatedSequence    ID of the paired vtkMRMLSequenceNode ("" = none)
    on the sequence        MPI.StudyRole        same role (so it can be found from either side)

Recognition
    gated     a sequence of scalar volumes with 4 to 32 items of identical geometry. Sequences from
              the DICOM SPECT Recon extension are recognised by SPECTRecon.Gate on their items.
    pairing   a gated sequence goes with the static volume that has the same reconstruction
              (SPECTRecon.AcquisitionStart), else the same subject hierarchy study folder and the
              longest common name, else the same grid.
    role      guessed from the volume / series / study names (English and Turkish keywords); the
              user confirms or changes it in the Studies table.
The pure functions at the top have no Slicer dependency and are unit tested outside Slicer.
"""

import logging
import re
import traceback

ROLES = ("Stress", "Rest", "Redistribution", "Viability", "Other", "None")
EXCLUDED_ROLE = "None"   # set in Data roles to leave a volume out of the analysis (never guessed)
UNKNOWN_ROLE = "Other"

ROLE_ATTRIBUTE = "MPI.StudyRole"
GATED_ATTRIBUTE = "MPI.GatedSequence"

MIN_GATES = 4
MAX_GATES = 32

# Order matters: the first role whose pattern matches wins ("stress redistribution" -> Redistribution)
_ROLE_PATTERNS = [
    ("Viability", r"viab|fdg|nitrat|reinj|re-inj|canl[ıi]l[ıi]k"),
    ("Redistribution", r"redist|delay|\blate\b|\bge[cç]\b|\b(?:3|4|24)\s*h\b|\b(?:3|4|24)\s*sa(?:at)?\b"),
    ("Stress", r"\bsgate|\bstomo|stres|\bstr\b|\bst\b|exercise|egzersiz|efor|pharm|dipyrid|adenos|regadenos|dobutam"),
    ("Rest", r"\brgate|\brtomo|\brest|\brst\b|\bre\b|istirahat|dinlenme|baseline|bazal"),
]
_GATED_PATTERN = r"gated|\bgate|\b[sr]gate|gspect|\bg-?spect|ekg|ecg"


def guessRole(*texts):
    """Role from free-text names; UNKNOWN_ROLE if none matches."""
    text = " ".join(t for t in texts if t).lower()
    text = re.sub(r"[_\-./]+", " ", text)
    for role, pattern in _ROLE_PATTERNS:
        if re.search(pattern, text):
            return role
    return UNKNOWN_ROLE


def studyRole(node):
    """(role, guessed) of a static volume: the stored role, else a guess from its names."""
    stored = node.GetAttribute(ROLE_ATTRIBUTE) if node is not None else None
    if stored in ROLES:
        return stored, False
    if node is None:
        return UNKNOWN_ROLE, True
    return guessRole(node.GetName(), _seriesDescription(node), _shParentAndStudyName(node)[1]), True


def isExcluded(node):
    """True for a volume given the role None in Data roles."""
    return node is not None and node.GetAttribute(ROLE_ATTRIBUTE) == EXCLUDED_ROLE


def looksGated(*texts):
    text = " ".join(t for t in texts if t).lower()
    return bool(re.search(_GATED_PATTERN, text))


def commonPrefixLength(a, b):
    a, b = (a or "").lower(), (b or "").lower()
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def pairGated(statics, gateds):
    """Match gated sequences to static volumes.

    statics, gateds: lists of dicts with keys
        id, name, parent (SH folder item or None), acquisition (SPECTRecon.AcquisitionStart or ""),
        geometry (hashable grid description or None), and for statics optionally 'gated' (an ID
        already chosen by the user, kept as is).
    Returns {static id: gated id} with each gated sequence used at most once."""
    pairs = {}
    used = set()
    for static in statics:
        chosen = static.get("gated")
        if chosen and any(g["id"] == chosen for g in gateds):
            pairs[static["id"]] = chosen
            used.add(chosen)

    def score(static, gated):
        if static["acquisition"] and static["acquisition"] == gated["acquisition"]:
            return (3, commonPrefixLength(static["name"], gated["name"]))
        sameFolder = static["parent"] is not None and static["parent"] == gated["parent"]
        prefix = commonPrefixLength(static["name"], gated["name"])
        if sameFolder and prefix >= 3:
            return (2, prefix)
        if static["acquisition"] and gated["acquisition"]:
            return None   # both know their acquisition and it differs: not the same study
        if sameFolder:
            return (1, prefix)
        if static["geometry"] is not None and static["geometry"] == gated["geometry"] and prefix >= 3:
            return (0, prefix)
        return None

    candidates = []
    for static in statics:
        if static["id"] in pairs:
            continue
        for gated in gateds:
            if gated["id"] in used:
                continue
            s = score(static, gated)
            if s is not None:
                candidates.append((s, static["id"], gated["id"]))
    for _, staticId, gatedId in sorted(candidates, reverse=True):
        if staticId in pairs or gatedId in used:
            continue
        pairs[staticId] = gatedId
        used.add(gatedId)
    return pairs


# ---------------------------------------------------------------------------
# Scene access
# ---------------------------------------------------------------------------
def _slicer():
    import slicer
    return slicer


def _shParentAndStudyName(node):
    slicer = _slicer()
    try:
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        item = shNode.GetItemByDataNode(node)
        if not item:
            return None, ""
        parent = shNode.GetItemParent(item)
        if not parent or parent == shNode.GetSceneItemID():
            return None, ""
        return parent, shNode.GetItemName(parent) or ""
    except Exception:
        return None, ""


def _seriesDescription(node):
    """DICOM series description of a volume loaded from the DICOM database, if known."""
    slicer = _slicer()
    try:
        uids = (node.GetAttribute("DICOM.instanceUIDs") or "").split()
        if uids and slicer.dicomDatabase:
            fileName = slicer.dicomDatabase.fileForInstance(uids[0])
            if fileName:
                return slicer.dicomDatabase.fileValue(fileName, "0008,103E") or ""
    except Exception:
        pass
    return ""


def _geometry(volumeNode):
    if volumeNode is None or volumeNode.GetImageData() is None:
        return None
    dims = tuple(volumeNode.GetImageData().GetDimensions())
    spacing = tuple(round(s, 3) for s in volumeNode.GetSpacing())
    origin = tuple(round(o, 1) for o in volumeNode.GetOrigin())
    return dims, spacing, origin


EMISSION_MODALITIES = ("NM", "PT")
SYSTEM_RESOLUTION_MM = {"NM": 12.0, "PT": 7.0}   # default FWHM at the heart, by modality


def modality(node):
    """DICOM modality of a volume ("NM", "PT", "CT", ...): its DICOM.Modality attribute, else read
    from the DICOM database for volumes loaded from it; "" when unknown."""
    if node is None:
        return ""
    value = (node.GetAttribute("DICOM.Modality") or "").strip().upper()
    if value:
        return value
    slicer = _slicer()
    try:
        uids = (node.GetAttribute("DICOM.instanceUIDs") or "").split()
        if uids and slicer.dicomDatabase:
            fileName = slicer.dicomDatabase.fileForInstance(uids[0])
            if fileName:
                return (slicer.dicomDatabase.fileValue(fileName, "0008,0060") or "").strip().upper()
    except Exception:
        pass
    return ""


def looksLikeEmission(imageData, modalityValue=""):
    """A SPECT or PET reconstruction: NM / PT whatever the grid (PET matrices reach 400 or more),
    never CT / MR, and with an unknown modality any grid finer than 512 pixels in-plane (CT size)."""
    if imageData is None:
        return False
    if modalityValue in EMISSION_MODALITIES:
        return True
    if modalityValue in ("CT", "MR"):
        return False
    dims = imageData.GetDimensions()
    return max(dims[0], dims[1]) < 512


def isProxyNode(node):
    """True for the 'current item' node a sequence browser shows (e.g. the recon '(current gate)')."""
    slicer = _slicer()
    try:
        return slicer.modules.sequences.logic().GetFirstBrowserNodeForProxyNode(node) is not None
    except Exception:
        return False


def gatedSequences():
    """Sequences in the scene that hold gated SPECT or PET: [(sequence node, number of gates, first item)]."""
    slicer = _slicer()
    found = []
    for sequence in slicer.util.getNodesByClass("vtkMRMLSequenceNode"):
        try:
            n = sequence.GetNumberOfDataNodes()
            if n < 2:
                continue
            first = sequence.GetNthDataNode(0)
            if first is None or not first.IsA("vtkMRMLScalarVolumeNode") or first.IsA("vtkMRMLLabelMapVolumeNode"):
                continue
            fromRecon = first.GetAttribute("SPECTRecon.Gate") is not None
            if not looksLikeEmission(first.GetImageData(), modality(first)):
                continue
            if not fromRecon:
                if not (MIN_GATES <= n <= MAX_GATES):
                    continue
                geometry = _geometry(first)
                if any(_geometry(sequence.GetNthDataNode(i)) != geometry for i in range(1, n)):
                    continue
                indexName = (sequence.GetIndexName() or "").lower()
                # A time series is gated if its name says so, or its index is not plain time
                if not (looksGated(sequence.GetName(), indexName) or indexName not in ("time", "")):
                    continue
            found.append((sequence, n, first))
        except Exception:
            logging.warning("Could not inspect sequence:\n" + traceback.format_exc())
    return found


def staticVolumes():
    """SPECT or PET scalar volumes that are not sequence proxies or outputs of this module."""
    slicer = _slicer()
    volumes = []
    for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
        if node.IsA("vtkMRMLLabelMapVolumeNode") or node.GetImageData() is None:
            continue
        if node.GetHideFromEditors() or node.GetAttribute("MPI.Orientation") or node.GetAttribute("MPI.Helper"):
            continue
        if isProxyNode(node):
            continue
        if not looksLikeEmission(node.GetImageData(), modality(node)):
            continue
        volumes.append(node)
    return volumes


def _describeStatic(node):
    parent, studyName = _shParentAndStudyName(node)
    return {"id": node.GetID(), "name": node.GetName(), "parent": parent, "study": studyName,
            "series": _seriesDescription(node),
            "acquisition": node.GetAttribute("SPECTRecon.AcquisitionStart") or "",
            "geometry": _geometry(node), "gated": node.GetAttribute(GATED_ATTRIBUTE) or None}


def _describeGated(sequence, first):
    slicer = _slicer()
    parent, studyName = _shParentAndStudyName(sequence)
    if parent is None:
        try:
            browser = slicer.modules.sequences.logic().GetFirstBrowserNodeForSequenceNode(sequence)
            proxy = browser.GetProxyNode(sequence) if browser else None
            if proxy is not None:
                parent, studyName = _shParentAndStudyName(proxy)
        except Exception:
            pass
    return {"id": sequence.GetID(), "name": sequence.GetName(), "parent": parent, "study": studyName,
            "acquisition": first.GetAttribute("SPECTRecon.AcquisitionStart") or "",
            "geometry": _geometry(first)}


class Study:
    """One row of the Studies table."""

    def __init__(self, staticNode, gatedNode=None, role=UNKNOWN_ROLE, guessed=True, gates=0):
        self.staticNode = staticNode
        self.gatedNode = gatedNode
        self.role = role
        self.guessed = guessed   # role not confirmed by the user yet
        self.gates = gates

    @property
    def name(self):
        if self.staticNode is not None:
            return self.staticNode.GetName()
        return self.gatedNode.GetName() if self.gatedNode is not None else ""


def scanStudies():
    """All studies in the scene. Stored roles and pairings are kept; new ones are guessed."""
    gatedList = gatedSequences()
    gatesById = {seq.GetID(): (seq, n, first) for seq, n, first in gatedList}
    statics = staticVolumes()
    descriptions = [_describeStatic(node) for node in statics]
    pairs = pairGated(descriptions, [_describeGated(seq, first) for seq, n, first in gatedList])

    studies = []
    used = set()
    for node, description in zip(statics, descriptions):
        gatedId = pairs.get(node.GetID())
        sequence, nGates = (gatesById[gatedId][0], gatesById[gatedId][1]) if gatedId else (None, 0)
        if gatedId:
            used.add(gatedId)
        stored = node.GetAttribute(ROLE_ATTRIBUTE)
        if stored in ROLES:
            role, guessed = stored, False
        else:
            role = guessRole(node.GetName(), description["series"], description["study"],
                             sequence.GetName() if sequence else "")
            guessed = True
        studies.append(Study(node, sequence, role, guessed, nGates))

    # Gated sequences without a static volume (summed image can be made from the gates)
    for seq, n, first in gatedList:
        if seq.GetID() in used:
            continue
        stored = seq.GetAttribute(ROLE_ATTRIBUTE)
        role = stored if stored in ROLES else guessRole(seq.GetName(), _seriesDescription(first))
        studies.append(Study(None, seq, role, stored not in ROLES, n))
    return studies


def saveStudy(study):
    """Write the role and pairing to the nodes (marks the role as confirmed)."""
    if study.staticNode is not None:
        study.staticNode.SetAttribute(ROLE_ATTRIBUTE, study.role)
        study.staticNode.SetAttribute(GATED_ATTRIBUTE, study.gatedNode.GetID() if study.gatedNode else "")
    if study.gatedNode is not None:
        study.gatedNode.SetAttribute(ROLE_ATTRIBUTE, study.role)
    study.guessed = False


def sumGates(sequence, name=None):
    """New scalar volume holding the sum of all gates (the static image of a gated acquisition)."""
    slicer = _slicer()
    import numpy as np
    n = sequence.GetNumberOfDataNodes()
    first = sequence.GetNthDataNode(0)
    total = None
    for i in range(n):
        array = slicer.util.arrayFromVolume(sequence.GetNthDataNode(i)).astype(np.float64)
        total = array if total is None else total + array
    node = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLScalarVolumeNode", slicer.mrmlScene.GenerateUniqueName(name or f"{sequence.GetName()} summed"))
    node.CopyOrientation(first)
    slicer.util.updateVolumeFromArray(node, total.astype(np.float32))
    for attribute in first.GetAttributeNames() or ():
        if attribute != "SPECTRecon.Gate":
            node.SetAttribute(attribute, first.GetAttribute(attribute))
    node.SetAttribute("MPI.SummedFromGates", str(n))
    node.SetAttribute(GATED_ATTRIBUTE, sequence.GetID())
    node.CreateDefaultDisplayNodes()
    try:
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        parent = shNode.GetItemParent(shNode.GetItemByDataNode(sequence))
        if parent and parent != shNode.GetSceneItemID():
            shNode.SetItemParent(shNode.GetItemByDataNode(node), parent)
    except Exception:
        pass
    return node
