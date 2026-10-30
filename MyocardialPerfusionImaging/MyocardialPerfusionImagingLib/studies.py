"""Studies of a perfusion exam: which volume is stress, rest, viability..., and its gated images.

A study is one static (or summed-gate) SPECT or PET volume, optionally paired with a gated sequence.
Everything is stored as node attributes, so it is saved with the scene and survives a reload:

    on the static volume   MPI.StudyRole        Stress / Rest / Redistribution / Viability / Other
                           MPI.GatedSequence    ID of the paired vtkMRMLSequenceNode ("" = none)
    on the sequence        MPI.StudyRole        same role (so it can be found from either side)

Recognition
    gated     a sequence of scalar volumes with 4 to 32 items of identical geometry. Sequences from
              the DICOM SPECT Recon extension (SPECTRecon.Gate on their items) and from the DICOM Gated /
              Dynamic PET extension (GatedPET.Gate) are always recognised.
              A MultiVolume (e.g. gated PET loaded by the MultiVolume importer: one volume with a frame
              per gate) is copied once into such a sequence (convertMultiVolumes), placed next to it.
    pairing   a gated sequence goes with the static volume that has the same reconstruction
              (SPECTRecon.AcquisitionStart), else the same subject hierarchy study folder and the
              longest common name, else the same grid.
    role      guessed from the volume / series / study names (English and Turkish keywords); the
              user confirms or changes it in the Studies table. When the PET tracer is known
              (Radiopharmaceutical / Radionuclide attributes of the DICOM Gated / Dynamic PET extension,
              else the DICOM header) it
              comes first: FDG is Viability; a perfusion tracer (O-15 water, N-13 ammonia, Rb-82,
              F-18 flurpiridaz, Cu-62) is Stress or Rest from the names, never Viability.
The pure functions at the top have no Slicer dependency and are unit tested outside Slicer.
"""

import logging
import re
import traceback

# ---------------------------------------------------------------------------
# Settings of this file: edit the values here. A value shown in Advanced parameters takes the one here as its
# built-in default; a value saved with Save parameters overrides it until Restore defaults.
# ---------------------------------------------------------------------------
ROLES = ("Stress", "Rest", "Redistribution", "Viability", "Other", "None")
EXCLUDED_ROLE = "None"   # set in Data roles to leave a volume out of the analysis (never guessed)
UNKNOWN_ROLE = "Other"

ROLE_ATTRIBUTE = "MPI.StudyRole"
GATED_ATTRIBUTE = "MPI.GatedSequence"
FROM_MULTIVOLUME_ATTRIBUTE = "MPI.FromMultiVolume"   # on a sequence made from a MultiVolume: its ID

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

# PET tracers, matched on the lower-case text with spaces, ^, -, _ and parentheses removed
# (DICOM code meanings look like "Fluorodeoxyglucose F^18^" or "^82^Rubidium")
TRACER_METABOLIC = "metabolic"
TRACER_PERFUSION = "perfusion"
_TRACER_PATTERNS = [
    (TRACER_METABOLIC, r"fdg|fluorodeoxyglucose|deoxyglucose"),
    (TRACER_PERFUSION, r"rubidium|rb82|82rb|ammonia|nh3|n13|13n|nitrogen13|13nitrogen|o15|15o|oxygen15|15oxygen"
                       r"|h2o|water|flurpiridaz|cu62|62cu|copper62|62copper|ptsm"),
]
# Written by the DICOM Gated / Dynamic PET extension (one prefix per kind of series)
TRACER_ATTRIBUTES = tuple(prefix + name for prefix in ("GatedPET.", "StaticPET.", "DynamicPET.")
                          for name in ("Radiopharmaceutical", "Radionuclide"))
EMISSION_MODALITIES = ("NM", "PT")
SYSTEM_RESOLUTION_MM = {"NM": 15.0, "PT": 7.0}   # default FWHM at the heart, by modality
WALL_FIT = {"NM": "bspline", "PT": "mesh"}       # surface fit of the wall with fit "auto", by modality
AUTO_FIT = ("Automatic: mesh for PET, B-spline for SPECT", "auto")


def tracerKind(*texts):
    """TRACER_METABOLIC (FDG), TRACER_PERFUSION (PET perfusion tracers) or "" from tracer names."""
    text = re.sub(r"[\s^_\-()]+", "", " ".join(t for t in texts if t).lower())
    for kind, pattern in _TRACER_PATTERNS:
        if text and re.search(pattern, text):
            return kind
    return ""


def guessRole(*texts, tracer=""):
    """Role from free-text names, and from the tracer when known; UNKNOWN_ROLE if none matches."""
    kind = tracerKind(tracer)
    if kind == TRACER_METABOLIC:
        return "Viability"
    text = " ".join(t for t in texts if t).lower()
    text = re.sub(r"[_\-./]+", " ", text)
    for role, pattern in _ROLE_PATTERNS:
        if kind == TRACER_PERFUSION and role not in ("Stress", "Rest"):
            continue   # e.g. "fdg" or "delay" in the names of a rubidium study
        if re.search(pattern, text):
            return role
    return UNKNOWN_ROLE


def tracer(node):
    """Tracer text of a volume (radiopharmaceutical and radionuclide): its PET attributes, else the
    Radiopharmaceutical Information Sequence of its first DICOM file; "" when unknown."""
    if node is None:
        return ""
    text = " ".join(node.GetAttribute(name) or "" for name in TRACER_ATTRIBUTES).strip()
    if text:
        return text
    uids = (node.GetAttribute("DICOM.instanceUIDs") or "").split()
    if not uids:
        return ""
    if uids[0] not in _TRACER_CACHE:
        _TRACER_CACHE[uids[0]] = _tracerFromDicom(uids[0])
    return _TRACER_CACHE[uids[0]]


_TRACER_CACHE = {}   # first instance UID -> tracer text (the header is read once)


def _tracerFromDicom(uid):
    slicer = _slicer()
    try:
        fileName = slicer.dicomDatabase.fileForInstance(uid) if slicer.dicomDatabase else ""
        if not fileName:
            return ""
        import pydicom
        radio = getattr(pydicom.dcmread(fileName, stop_before_pixels=True),
                        "RadiopharmaceuticalInformationSequence", None)
        return radiopharmaceuticalText(radio[0]) if radio else ""
    except Exception:
        return ""


def radiopharmaceuticalText(item):
    """Radiopharmaceutical (its text, else code meaning) and radionuclide code meaning of one item of the
    Radiopharmaceutical Information Sequence (0054,0016), as one text."""
    def codeMeaning(keyword):
        sequence = getattr(item, keyword, None)
        if not sequence:
            return ""
        return str(getattr(sequence[0], "CodeMeaning", "") or getattr(sequence[0], "CodeValue", "") or "").strip()
    try:
        name = str(getattr(item, "Radiopharmaceutical", "") or "").strip() or \
            codeMeaning("RadiopharmaceuticalCodeSequence")
        return " ".join(v for v in (name, codeMeaning("RadionuclideCodeSequence")) if v)
    except Exception:
        return ""


def studyRole(node):
    """(role, guessed) of a static volume: the stored role, else a guess from its names and tracer."""
    stored = node.GetAttribute(ROLE_ATTRIBUTE) if node is not None else None
    if stored in ROLES:
        return stored, False
    if node is None:
        return UNKNOWN_ROLE, True
    return guessRole(node.GetName(), _seriesDescription(node), _shParentAndStudyName(node)[1],
                     tracer=tracer(node)), True


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


def wallFit(node, fit="auto"):
    """Surface fit for a study's wall (myocardium.WALL_FITS / GATE_FITS): fit itself, or with "auto" the one
    of the study's modality (WALL_FIT: the mesh follows PET's sharper walls closely, the B-spline smooths
    SPECT's), B-spline when the modality is not known."""
    return WALL_FIT.get(modality(node), "bspline") if fit == "auto" else fit


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


def isMultiVolume(node):
    """A MultiVolume node (or a vector volume holding MultiVolume frames): one volume, one frame per component."""
    if node is None or node.GetImageData() is None:
        return False
    return node.IsA("vtkMRMLMultiVolumeNode") or (
        node.IsA("vtkMRMLVectorVolumeNode") and bool(node.GetAttribute("MultiVolume.NumberOfFrames")))


def multiVolumeFrameLabels(node, count):
    """Frame labels of a MultiVolume (MultiVolume.FrameLabels), else 1..count."""
    labels = [t.strip() for t in (node.GetAttribute("MultiVolume.FrameLabels") or "").split(",") if t.strip()]
    return labels if len(labels) == count else [str(i + 1) for i in range(count)]


def splitFrames(array):
    """Frames of a MultiVolume voxel array [k, j, i, frames] as [k, j, i] arrays."""
    import numpy as np
    array = np.asarray(array)
    if array.ndim != 4:
        raise ValueError("A MultiVolume has a frame axis after k, j, i.")
    return [np.ascontiguousarray(array[..., n]) for n in range(array.shape[3])]


def _dicomInstanceUids(node):
    uids = node.GetAttribute("DICOM.instanceUIDs") or ""
    if uids:
        return uids
    slicer = _slicer()
    try:
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        item = shNode.GetItemByDataNode(node)
        if item:
            return shNode.GetItemUID(item, slicer.vtkMRMLSubjectHierarchyConstants.GetDICOMInstanceUIDName()) or ""
    except Exception:
        pass
    return ""


def multiVolumeSequence(node):
    """The sequence made from a MultiVolume, if it is still in the scene (also after a scene load that renamed
    the node IDs: the one the MultiVolume points to, else any that names it)."""
    slicer = _slicer()
    sequence = slicer.mrmlScene.GetNodeByID(node.GetAttribute(GATED_ATTRIBUTE) or "")
    if sequence is not None and sequence.IsA("vtkMRMLSequenceNode"):
        return sequence
    return next((s for s in slicer.util.getNodesByClass("vtkMRMLSequenceNode")
                 if s.GetAttribute(FROM_MULTIVOLUME_ATTRIBUTE) == node.GetID()), None)


def sequenceFromMultiVolume(node):
    """A sequence of scalar volumes (one per frame, index "Gate") with the frames of a MultiVolume, in its
    subject hierarchy folder, with a browser so it can be played. The DICOM attributes are copied to the frames
    (patient, modality, series description are read from them)."""
    slicer = _slicer()
    frames = splitFrames(slicer.util.arrayFromVolume(node))
    labels = multiVolumeFrameLabels(node, len(frames))
    scene = slicer.mrmlScene
    sequence = scene.AddNewNodeByClass("vtkMRMLSequenceNode", scene.GenerateUniqueName(f"{node.GetName()} gated"))
    sequence.SetIndexName("Gate")
    sequence.SetIndexUnit("")
    sequence.SetIndexType(sequence.NumericIndex)
    uids = _dicomInstanceUids(node)
    attributes = {name: node.GetAttribute(name) for name in (node.GetAttributeNames() or ())
                  if not name.startswith("MultiVolume.") and not name.startswith("MPI.")}
    if uids:
        attributes["DICOM.instanceUIDs"] = uids
    if not attributes.get("DICOM.Modality"):
        found = modality(node)
        if found:
            attributes["DICOM.Modality"] = found
    for index, (label, frame) in enumerate(zip(labels, frames)):
        volume = slicer.vtkMRMLScalarVolumeNode()
        volume.SetName(f"{node.GetName()} gate {label}")
        volume.CopyOrientation(node)
        slicer.util.updateVolumeFromArray(volume, frame)
        for name, value in attributes.items():
            volume.SetAttribute(name, value)
        sequence.SetDataNodeAtValue(volume, str(index + 1))
    sequence.SetAttribute(FROM_MULTIVOLUME_ATTRIBUTE, node.GetID())
    stored = node.GetAttribute(ROLE_ATTRIBUTE)
    if stored in ROLES:
        sequence.SetAttribute(ROLE_ATTRIBUTE, stored)
    node.SetAttribute(GATED_ATTRIBUTE, sequence.GetID())
    try:
        browser = scene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", f"{sequence.GetName()} browser")
        browser.SetAndObserveMasterSequenceNodeID(sequence.GetID())
        proxy = browser.GetProxyNode(sequence)
        if proxy is not None:
            proxy.SetName(f"{sequence.GetName()} (current gate)")
    except Exception:
        logging.warning("MPI: could not make a browser for the gated sequence", exc_info=True)
    try:
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(scene)
        parent = shNode.GetItemParent(shNode.GetItemByDataNode(node))
        if parent and parent != shNode.GetSceneItemID():
            shNode.SetItemParent(shNode.GetItemByDataNode(sequence), parent)
    except Exception:
        pass
    return sequence


def convertMultiVolumes():
    """Make a gated sequence of every emission MultiVolume of 4 to 32 frames not converted yet. Returns the new
    sequences."""
    slicer = _slicer()
    made = []
    for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
        try:
            if not isMultiVolume(node) or multiVolumeSequence(node) is not None:
                continue
            frames = node.GetImageData().GetNumberOfScalarComponents()
            if not (MIN_GATES <= frames <= MAX_GATES):
                continue
            if not looksLikeEmission(node.GetImageData(), modality(node)):
                continue
            made.append(sequenceFromMultiVolume(node))
            logging.info(f"MPI: gated sequence made from the MultiVolume {node.GetName()} ({frames} frames)")
        except Exception:
            logging.warning("MPI: could not convert a MultiVolume:\n" + traceback.format_exc())
    return made


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
            fromRecon = first.GetAttribute("SPECTRecon.Gate") is not None or first.GetAttribute("GatedPET.Gate") is not None
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
        if isMultiVolume(node) or node.GetImageData().GetNumberOfScalarComponents() != 1:
            continue   # gated frames in one volume: used through its sequence (convertMultiVolumes)
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
    """All studies in the scene. Stored roles and pairings are kept; new ones are guessed. MultiVolumes are
    made into gated sequences first."""
    convertMultiVolumes()
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
                             sequence.GetName() if sequence else "",
                             tracer=tracer(node) or (tracer(sequence.GetNthDataNode(0)) if sequence else ""))
            guessed = True
        studies.append(Study(node, sequence, role, guessed, nGates))

    # Gated sequences without a static volume (summed image can be made from the gates)
    for seq, n, first in gatedList:
        if seq.GetID() in used:
            continue
        stored = seq.GetAttribute(ROLE_ATTRIBUTE)
        role = stored if stored in ROLES else guessRole(seq.GetName(), _seriesDescription(first),
                                                        tracer=tracer(first))
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
    """New scalar volume holding the sum of all gates (the static image of a gated acquisition).
    Gated PET gates (GatedPET.Gate) are each a full concentration / SUV image, so they are averaged."""
    slicer = _slicer()
    import numpy as np
    n = sequence.GetNumberOfDataNodes()
    first = sequence.GetNthDataNode(0)
    average = first.GetAttribute("GatedPET.Gate") is not None
    total = None
    for i in range(n):
        array = slicer.util.arrayFromVolume(sequence.GetNthDataNode(i)).astype(np.float64)
        total = array if total is None else total + array
    if average:
        total /= n
    node = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLScalarVolumeNode",
        slicer.mrmlScene.GenerateUniqueName(name or f"{sequence.GetName()} {'mean' if average else 'summed'}"))
    node.CopyOrientation(first)
    slicer.util.updateVolumeFromArray(node, total.astype(np.float32))
    perGate = ("SPECTRecon.Gate", "GatedPET.Gate", "GatedPET.TriggerTimeMs", "GatedPET.FrameDurationMs")
    for attribute in first.GetAttributeNames() or ():
        if attribute not in perGate:
            node.SetAttribute(attribute, first.GetAttribute(attribute))
    if first.GetVoxelValueUnits() is not None:
        node.SetVoxelValueUnits(first.GetVoxelValueUnits())
    node.SetAttribute("MPI.SummedFromGates", str(n))
    if average:
        node.SetAttribute("MPI.AveragedGates", "1")
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
