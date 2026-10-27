"""Gated PET series from DICOM: recognition, gate/slice ordering, geometry, SUV and ECG gating (pydicom and
numpy only, no Slicer dependency; the DICOM plugin is DICOMGatedPETPlugin.py).

A gated PET series (Series Type (0054,1000) GATED\\IMAGE, Number of Time Slots (0054,0071) > 1) is a stack
of single slice images, one per slice and gate. Image Index (0054,1330) orders them:
    ImageIndex = (time slot - 1) x Number of Slices (0054,0081) + slice
(the PET image IOD; more than one R-R interval would add a third level, which is folded into the gates).
Without a usable Image Index the images are grouped by Trigger Time (0018,1060), then by repeated slice
positions in instance order. Slices are sorted along the normal of the image plane.

Pixel values are the stored value x Rescale Slope + Intercept of each image (slopes differ per image).
SUVbw follows the QIBA vendor-neutral pseudo-code (also what Slicer's PET SUV plugin computes):
    SUVbw = activity concentration (Bq/ml) x body weight (g) / injected dose decayed to the scan start (Bq)
    Units BQML   decay reference: Decay Correction START -> series date/time if not later than the first
                 acquisition, else the GE scan start (0009,100d), else the first acquisition date/time;
                 ADMIN -> no decay; NONE -> the first acquisition date/time.
    Units CNTS   Philips: SUV scale factor (7053,1000), Bq/ml from the activity concentration scale
                 factor (7053,1009).
    Units GML    already SUVbw.
The dose, its time and the half life come from the Radiopharmaceutical Information Sequence (0054,0016);
when the dose is missing GE's own tracer activity record (GEMS_PETD_01 private group) is used.
Each gate of a GE gated PET is a full activity concentration image (normalised to its own frame
duration), so the gates are on the scale of the static image, and their mean (not sum) is the static.
"""

import datetime
import math
import os

import numpy as np

GATE_ATTRIBUTE_PREFIX = "GatedPET."
UNITS_SUV = "SUVbw"
UNITS_ACTIVITY = "Bq/ml"

GE_PET_CREATOR = "GEMS_PETD_01"
PHILIPS_CREATOR = "Philips PET Private Group"


# ---------------------------------------------------------------------------
# Small DICOM helpers
# ---------------------------------------------------------------------------
def _value(ds, keyword, default=None):
    value = getattr(ds, keyword, None)
    if value is None or value == "":
        return default
    return value


def _float(value, default=None):
    try:
        if value is None or value == "":
            return default
        if isinstance(value, (list, tuple)) or (hasattr(value, "__len__") and not isinstance(value, (str, bytes))):
            value = value[0]
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError, IndexError):
        return default


def _textList(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip().upper() for v in value.split("\\")]
    try:
        return [str(v).strip().upper() for v in value]
    except TypeError:
        return [str(value).strip().upper()]


def privateValue(ds, group, creator, element):
    """Value of a private element (group, xx element) of a creator, or None. Finds the creator's block
    itself, so it works whichever block (0x10, 0x11, ...) the scanner put it in."""
    try:
        for block in range(0x10, 0x100):
            tag = (group << 16) | block
            if tag not in ds:
                continue
            if str(ds[tag].value).strip() == creator:
                full = (group << 16) | (block << 8) | element
                return ds[full].value if full in ds else None
    except Exception:
        pass
    return None


def parseDateTime(dateText, timeText=None):
    """datetime from DICOM DA + TM, or a DT when timeText is None; None when unreadable.
    UTC offsets are dropped (all times of one study are in the same zone)."""
    text = str(dateText or "").strip()
    if timeText is not None:
        if not text or not str(timeText).strip():
            return None
        text = text[:8] + str(timeText).strip()
    if not text:
        return None
    for sign in ("+", "-"):
        cut = text.find(sign, 8)
        if cut > 0:
            text = text[:cut]
    date, _, rest = text.partition(".")
    fraction = rest[:6].ljust(6, "0") if rest else "000000"
    date = date.ljust(14, "0")[:14]
    try:
        stamp = datetime.datetime.strptime(date, "%Y%m%d%H%M%S")
        return stamp + datetime.timedelta(microseconds=int(fraction or 0))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------
def isGated(modality, seriesType, timeSlots):
    """A gated PET series from its Modality, Series Type (text 'GATED\\IMAGE' or list) and Number of Time
    Slots values as stored in the header (strings are fine)."""
    if str(modality or "").strip().upper() != "PT":
        return False
    types = _textList(seriesType)
    slots = _float(timeSlots, 0) or 0
    return (bool(types) and types[0] == "GATED") or slots > 1


def isGatedDataset(ds):
    return isGated(_value(ds, "Modality"), _value(ds, "SeriesType"), _value(ds, "NumberOfTimeSlots"))


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------
def sliceNormal(ds):
    orientation = [float(v) for v in ds.ImageOrientationPatient]
    row, column = np.array(orientation[:3]), np.array(orientation[3:])
    normal = np.cross(row, column)
    return normal / (np.linalg.norm(normal) or 1.0)


def _position(ds):
    return np.array([float(v) for v in ds.ImagePositionPatient])


def groupGates(datasets):
    """Split the images of a gated series into gates of slices: (gates, method), where gates is a list
    (one per gate, in gate order) of lists of datasets sorted along the slice normal.
    Raises ValueError when the images cannot be split into gates of the same slices."""
    if not datasets:
        raise ValueError("No images")
    normal = sliceNormal(datasets[0])
    slots = int(_float(_value(datasets[0], "NumberOfTimeSlots"), 0) or 0)

    def bySlots(keyOf, method):
        groups = {}
        for ds in datasets:
            groups.setdefault(keyOf(ds), []).append(ds)
        if len(groups) < 2:
            return None
        return [groups[k] for k in sorted(groups)], method

    attempts = []
    nSlices = int(_float(_value(datasets[0], "NumberOfSlices"), 0) or 0)
    if nSlices and all(_float(_value(ds, "ImageIndex")) for ds in datasets):
        attempts.append(lambda: bySlots(lambda ds: (int(_float(ds.ImageIndex)) - 1) // nSlices,
                                        "Image Index (0054,1330)"))
    if all(_float(_value(ds, "TriggerTime")) is not None for ds in datasets):
        attempts.append(lambda: bySlots(lambda ds: round(_float(ds.TriggerTime), 1), "Trigger Time (0018,1060)"))
    attempts.append(lambda: _byRepeatedPositions(datasets, normal))

    problems = []
    for attempt in attempts:
        result = attempt()
        if result is None:
            continue
        gates, method = result
        problem = _checkGates(gates, normal, slots)
        if problem is None:
            gates = [sorted(gate, key=lambda ds: float(np.dot(_position(ds), normal))) for gate in gates]
            return gates, method
        problems.append(f"{method}: {problem}")
    raise ValueError("Cannot split the images into gates. " + "; ".join(problems or ["only one gate found"]))


def _byRepeatedPositions(datasets, normal):
    """Images at the same slice position, taken in instance (then image index) order, are the gates."""
    order = sorted(datasets, key=lambda ds: (_float(_value(ds, "InstanceNumber"), 0), _float(_value(ds, "ImageIndex"), 0)))
    positions = {}
    for ds in order:
        positions.setdefault(round(float(np.dot(_position(ds), normal)), 2), []).append(ds)
    counts = {len(v) for v in positions.values()}
    if len(counts) != 1 or counts.pop() < 2:
        return None
    nGates = len(next(iter(positions.values())))
    return [[positions[p][g] for p in positions] for g in range(nGates)], "repeated slice positions"


def _checkGates(gates, normal, slots):
    sizes = {len(g) for g in gates}
    if len(sizes) != 1:
        return f"gates have different numbers of slices ({sorted(sizes)})"
    if slots > 1 and len(gates) != slots:
        return f"{len(gates)} gates found, the header says {slots} time slots"
    reference = sorted(round(float(np.dot(_position(ds), normal)), 2) for ds in gates[0])
    if len(set(reference)) != len(reference):
        return "a gate has two images at the same slice position"
    for gate in gates[1:]:
        positions = sorted(round(float(np.dot(_position(ds), normal)), 2) for ds in gate)
        if np.max(np.abs(np.array(positions) - np.array(reference))) > 0.05:
            return "gates do not cover the same slice positions"
    return None


# ---------------------------------------------------------------------------
# Geometry and pixels
# ---------------------------------------------------------------------------
def geometry(sortedSlices):
    """(origin LPS, spacing (column, row, slice) mm, direction 3x3 with column/row/normal unit vectors as
    columns) of a gate. Raises ValueError for uneven slice spacing (more than 1 % or 0.1 mm)."""
    first = sortedSlices[0]
    orientation = [float(v) for v in first.ImageOrientationPatient]
    row, column = np.array(orientation[:3]), np.array(orientation[3:])
    normal = sliceNormal(first)
    rowSpacing, columnSpacing = (float(v) for v in first.PixelSpacing)   # between rows, between columns
    depths = np.array([float(np.dot(_position(ds), normal)) for ds in sortedSlices])
    if len(depths) > 1:
        steps = np.diff(depths)
        sliceSpacing = float(np.median(steps))
        if sliceSpacing <= 0 or np.max(np.abs(steps - sliceSpacing)) > max(0.1, 0.01 * sliceSpacing):
            raise ValueError(f"Uneven slice spacing ({steps.min():.3f} to {steps.max():.3f} mm)")
    else:
        sliceSpacing = _float(_value(first, "SliceThickness"), 1.0)
    direction = np.column_stack([row, column, normal])
    return _position(first), (columnSpacing, rowSpacing, sliceSpacing), direction


def lpsToRasMatrix(origin, spacing, direction):
    """4x4 IJK to RAS matrix (list of lists) for a volume whose voxel (i, j, k) is column i, row j, slice k."""
    flip = np.diag([-1.0, -1.0, 1.0])
    matrix = np.eye(4)
    matrix[:3, :3] = flip @ direction @ np.diag(spacing)
    matrix[:3, 3] = flip @ np.asarray(origin, dtype=float)
    return matrix.tolist()


def rescaled(ds):
    """Pixel values of one image after its own Rescale Slope / Intercept (float64, rows x columns)."""
    pixels = ds.pixel_array.astype(np.float64)
    slope = _float(_value(ds, "RescaleSlope"), 1.0)
    intercept = _float(_value(ds, "RescaleIntercept"), 0.0)
    return pixels * slope + intercept


def gateArray(sortedSlices, scale=1.0):
    """Gate volume as a float32 array (slices, rows, columns) times scale."""
    return (np.stack([rescaled(ds) for ds in sortedSlices]) * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# SUV
# ---------------------------------------------------------------------------
def _radiopharmaceutical(ds):
    sequence = _value(ds, "RadiopharmaceuticalInformationSequence")
    return sequence[0] if sequence else None


def scanStart(datasets):
    """Decay reference of a decay-corrected (START) series: (datetime, source text)."""
    first = datasets[0]
    series = parseDateTime(_value(first, "SeriesDate"), _value(first, "SeriesTime"))
    acquisitions = [parseDateTime(_value(ds, "AcquisitionDate") or _value(ds, "SeriesDate"), _value(ds, "AcquisitionTime"))
                    for ds in datasets]
    acquisitions = [a for a in acquisitions if a is not None]
    firstAcquisition = min(acquisitions) if acquisitions else None
    if series is not None and (firstAcquisition is None or series <= firstAcquisition):
        return series, "series date/time"
    geStart = privateValue(first, 0x0009, GE_PET_CREATOR, 0x0D)
    if geStart:
        stamp = parseDateTime(geStart)
        if stamp is not None:
            return stamp, "GE scan start (0009,100d)"
    if firstAcquisition is not None:
        return firstAcquisition, "first acquisition date/time"
    return series, "series date/time"


def suvInfo(datasets):
    """SUVbw conversion of a PET series: a dict with
        activityScale   factor from rescaled pixel values to Bq/ml (None when not possible)
        suvScale        factor from rescaled pixel values to SUVbw (None when not possible)
        units           DICOM Units (0054,1001)
        and the inputs (WeightKg, InjectedDoseBq, DecayedDoseBq, InjectionTime, ScanStart, HalfLifeS,
        DecayCorrection, SUVSource) as text for the node attributes,
        problems        why SUV is not available (list of text)."""
    first = datasets[0]
    units = str(_value(first, "Units", "") or "").strip().upper()
    info = {"units": units, "activityScale": None, "suvScale": None, "problems": []}

    if units == "GML":
        info.update(suvScale=1.0, SUVSource="image already in SUVbw (Units GML)")
        info["problems"].append("activity concentration not available (image stored in SUVbw)")
        return info

    if units == "CNTS":
        suvFactor = _float(privateValue(first, 0x7053, PHILIPS_CREATOR, 0x00))
        activityFactor = _float(privateValue(first, 0x7053, PHILIPS_CREATOR, 0x09))
        if activityFactor:
            info["activityScale"] = activityFactor
        if suvFactor:
            info.update(suvScale=suvFactor, SUVSource="Philips SUV scale factor (7053,1000)")
        else:
            info["problems"].append("counts image without the Philips SUV scale factor")
        return info

    if units not in ("BQML", ""):
        info["problems"].append(f"units {units} not supported for SUV")
        return info
    info["activityScale"] = 1.0
    if not units:
        info["problems"].append("Units (0054,1001) missing, assumed Bq/ml")

    weight = _float(_value(first, "PatientWeight"))
    if not weight or weight <= 0:
        info["problems"].append("patient weight missing")
    radio = _radiopharmaceutical(first)
    dose = _float(_value(radio, "RadionuclideTotalDose")) if radio is not None else None
    halfLife = _float(_value(radio, "RadionuclideHalfLife")) if radio is not None else None
    injection = None
    if radio is not None:
        injection = parseDateTime(_value(radio, "RadiopharmaceuticalStartDateTime"))
        if injection is None:
            injection = parseDateTime(_value(first, "SeriesDate") or _value(first, "StudyDate"),
                                      _value(radio, "RadiopharmaceuticalStartTime"))
    source = "Radiopharmaceutical Information Sequence"
    if not dose:
        # GE's own record of the measured tracer activity, MBq, and when it was measured
        geDose = _float(privateValue(first, 0x0009, GE_PET_CREATOR, 0x38))
        geTime = parseDateTime(privateValue(first, 0x0009, GE_PET_CREATOR, 0x39))
        if geDose and geTime:
            dose, injection, source = geDose * 1e6, geTime, "GE tracer activity (0009,1038)"
    if not dose or dose <= 0:
        info["problems"].append("injected dose missing")
    if not halfLife or halfLife <= 0:
        info["problems"].append("half life missing")

    decayCorrection = str(_value(first, "DecayCorrection", "") or "").strip().upper()
    decayed = None
    if dose and halfLife:
        if decayCorrection == "ADMIN":
            decayed, reference = dose, "ADMIN (none)"
            info["ScanStart"] = ""
        else:
            if decayCorrection == "NONE":
                stamps = [parseDateTime(_value(ds, "AcquisitionDate") or _value(ds, "SeriesDate"), _value(ds, "AcquisitionTime"))
                          for ds in datasets]
                stamps = [s for s in stamps if s is not None]
                start, reference = (min(stamps) if stamps else None), "first acquisition date/time"
            else:
                start, reference = scanStart(datasets)
            if injection is None:
                info["problems"].append("injection time missing")
            elif start is None:
                info["problems"].append("scan time missing")
            else:
                elapsed = (start - injection).total_seconds()
                if elapsed < 0:
                    info["problems"].append("injection time after the scan start")
                else:
                    decayed = dose * math.pow(2.0, -elapsed / halfLife)
                info["ScanStart"] = start.isoformat()
        info["DecayReference"] = reference
    info.update({"WeightKg": weight, "InjectedDoseBq": dose, "HalfLifeS": halfLife,
                 "InjectionTime": injection.isoformat() if injection else "",
                 "DecayCorrection": decayCorrection, "DecayedDoseBq": decayed, "SUVSource": source})
    if weight and decayed and not info["problems"]:
        info["suvScale"] = weight * 1000.0 / decayed
    return info


# ---------------------------------------------------------------------------
# ECG gating
# ---------------------------------------------------------------------------
def _round(value, digits=1):
    number = _float(value)
    if number is None:
        return None
    number = round(number, digits)
    return int(number) if float(number).is_integer() else number


def gatingInfo(gates):
    """ECG gating fields of a gated PET as {key: value}, keyed like the gating attributes of the DICOM SPECT
    Recon extension (MyocardialPerfusionSPECTLib.gating), so Gated QC shows them. 0 = not recorded."""
    first = gates[0][0]
    info = {"GatingTimeSlots": len(gates), "GatingInfoSource": "DICOM PET gating attributes"}
    for key, keyword in (("GatingLowRRMs", "LowRRValue"), ("GatingHighRRMs", "HighRRValue"),
                         ("GatingIntervalsAcquired", "IntervalsAcquired"),
                         ("GatingIntervalsRejected", "IntervalsRejected"), ("GatingSkipBeats", "SkipBeats")):
        value = _round(_value(first, keyword))
        if value is not None and (value or key in ("GatingIntervalsRejected", "GatingSkipBeats")):
            info[key] = value
    flag = str(_value(first, "BeatRejectionFlag", "") or "").strip()
    if flag:
        info["GatingBeatRejection"] = flag
    framing = str(_value(first, "FramingType", "") or "").strip()
    if framing:
        info["GatingFramingType"] = framing

    nominal = _round(_value(first, "NominalInterval"))
    rate = _round(_value(first, "HeartRate"))
    if nominal:
        info["GatingNominalIntervalMs"], info["GatingNominalIntervalSource"] = nominal, "DICOM nominal interval"
    elif rate:
        info["GatingNominalIntervalMs"], info["GatingNominalIntervalSource"] = round(60000.0 / rate, 1), "DICOM heart rate"
    elif info.get("GatingLowRRMs") and info.get("GatingHighRRMs"):
        info["GatingNominalIntervalMs"] = round(0.5 * (info["GatingLowRRMs"] + info["GatingHighRRMs"]), 1)
        info["GatingNominalIntervalSource"] = "middle of the accepted R-R window"
    if rate:
        info["GatingHeartRateBpm"], info["GatingHeartRateSource"] = rate, "DICOM heart rate"
    elif info.get("GatingNominalIntervalMs"):
        info["GatingHeartRateBpm"] = round(60000.0 / info["GatingNominalIntervalMs"], 1)
        info["GatingHeartRateSource"] = "from the R-R interval"

    acquired, rejected = info.get("GatingIntervalsAcquired"), info.get("GatingIntervalsRejected")
    if acquired is not None and rejected is not None and acquired + rejected > 0:
        info["GatingRejectedPercent"] = round(100.0 * rejected / (acquired + rejected), 1)
    mean = info.get("GatingNominalIntervalMs")
    if mean:
        info["GatingSlotWidthMs"] = round(mean / len(gates), 1)
        if info.get("GatingLowRRMs") and info.get("GatingHighRRMs"):
            info["GatingRRWindowPercent"] = round(50.0 * (info["GatingHighRRMs"] - info["GatingLowRRMs"]) / mean, 1)
    info["GatingBeatsScope"] = "acquisition"
    return info


def gateTiming(gate):
    """(trigger time ms, frame duration ms) of one gate (None when not recorded)."""
    first = gate[0]
    return _round(_value(first, "TriggerTime")), _round(_value(first, "ActualFrameDuration"))


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
def writeSyntheticSeries(folder, gates=4, slices=6, rows=16, columns=16, shuffle=True):
    """Write a small gated PET series (Bq/ml, per image slope) and return (files, expected SUVbw arrays).
    Gate g has the value 1000 x g Bq/ml at every voxel; 70 kg, 370 MBq F-18 injected 60 min before the scan."""
    import random
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    os.makedirs(folder, exist_ok=True)
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    files, order = [], [(g, s) for g in range(gates) for s in range(slices)]
    if shuffle:
        random.Random(1).shuffle(order)
    for n, (g, s) in enumerate(order):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.128"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.Modality, ds.SeriesDescription = "PT", "Synthetic gated PET"
        ds.PatientName, ds.PatientID, ds.PatientWeight = "Phantom^Gated", "PHANTOM", "70"
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, series, frame
        ds.StudyDate = ds.SeriesDate = ds.AcquisitionDate = "20260101"
        ds.StudyTime, ds.SeriesTime, ds.AcquisitionTime = "095500", "100000", "100000"
        ds.SeriesType, ds.Units, ds.DecayCorrection = ["GATED", "IMAGE"], "BQML", "START"
        ds.NumberOfTimeSlots, ds.NumberOfSlices = gates, slices
        ds.ImageIndex = g * slices + s + 1
        ds.InstanceNumber = n + 1
        ds.TriggerTime = g * 100
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [-40.0, -40.0, -100.0 + 3.27 * s]
        ds.PixelSpacing, ds.SliceThickness = [5.0, 5.0], 3.27
        radio = Dataset()
        radio.RadionuclideTotalDose, radio.RadionuclideHalfLife = 370e6, 6586.2
        radio.RadiopharmaceuticalStartDateTime = "20260101090000"
        ds.RadiopharmaceuticalInformationSequence = [radio]
        ds.Rows, ds.Columns = rows, columns
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        slope = 1000.0 * (g + 1) / 20000.0
        ds.RescaleSlope, ds.RescaleIntercept = f"{slope:.8g}", "0"
        ds.PixelData = np.full((rows, columns), 20000, dtype=np.int16).tobytes()
        path = os.path.join(folder, f"img{n:04d}.dcm")
        try:
            ds.save_as(path, enforce_file_format=True)          # pydicom 3
        except TypeError:
            ds.is_little_endian, ds.is_implicit_VR = True, False  # pydicom 2
            ds.save_as(path, write_like_original=False)
        files.append(path)
    decayed = 370e6 * 2.0 ** (-3600.0 / 6586.2)
    expected = [np.full((slices, rows, columns), 1000.0 * (g + 1) * 70000.0 / decayed) for g in range(gates)]
    return files, expected
