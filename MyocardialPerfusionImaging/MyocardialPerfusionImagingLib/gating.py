"""Gated QC helpers without a Slicer dependency: the ECG gating information stored on the gate volumes
by the DICOM SPECT Recon extension, per-frame count checks and their text.

Attributes read (all optional, "SPECTRecon." prefix, written by DICOMSpectRecon nmdata.readGatingInfo, or
"GatedPET." prefix, written by the DICOM Gated / Dynamic PET extension):
    GatingTimeSlots, GatingBeatRejection, GatingLowRRMs, GatingHighRRMs, GatingRRWindowPercent,
    GatingNominalIntervalMs (+ Source), GatingHeartRateBpm (+ Source), GatingIntervalsAcquired,
    GatingIntervalsRejected, GatingRejectedPercent, GatingSlotWidthMs, GatingFrameTimeMs,
    GatingFramingType, GatingSkipBeats, GatingDataItems, GatingItemsAreViews, GatingViews,
    GatingBeatsScope (view / acquisition), GatingStartHeartRateBpm (GE),
    GatingPerItem<Name> (JSON lists, one value per Data Information item, per projection view when
    GatingItemsAreViews is 1), GatePhotopeakCounts (JSON, projection counts of every gate).
"""

import json
import math

import numpy as np

# ---------------------------------------------------------------------------
# Settings of this file: edit the values here. A value shown in Advanced parameters takes the one here as its
# built-in default; a value saved with Save parameters overrides it until Restore defaults.
# ---------------------------------------------------------------------------
PREFIX = "SPECTRecon."
PREFIXES = (PREFIX, "GatedPET.")   # gated PET: same keys, written by the DICOMGatedDynamicPET extension
COUNT_BAND_PERCENT = 10.0      # a frame whose counts differ more from the mean is flagged
REJECTED_WARNING_PERCENT = 20.0
RR_DRIFT_WARNING_PERCENT = 10.0


def attributes(node):
    """{name without prefix: text} of the SPECTRecon (or GatedPET) attributes of a volume node."""
    out = {}
    if node is None:
        return out
    for name in node.GetAttributeNames() or ():
        for prefix in PREFIXES:
            if name.startswith(prefix):
                out[name[len(prefix):]] = node.GetAttribute(name)
    return out


def number(values, key):
    try:
        value = float(values.get(key))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def jsonList(values, key):
    try:
        data = json.loads(values.get(key) or "")
    except (TypeError, ValueError):
        return None
    if not isinstance(data, list):
        return None
    out = []
    for v in data:
        try:
            out.append(None if v is None else float(v))
        except (TypeError, ValueError):
            out.append(None)
    return out


def formatCounts(value):
    if value is None or not math.isfinite(value):
        return "-"
    for limit, suffix in ((1e9, " G"), (1e6, " M"), (1e3, " k")):
        if abs(value) >= limit:
            return f"{value / limit:.3g}{suffix}"
    return f"{value:.0f}"


def deviations(values):
    """% difference of every value from the mean of the values (None when not computable)."""
    values = np.asarray([np.nan if v is None else v for v in values], dtype=float)
    mean = float(np.nanmean(values)) if np.any(np.isfinite(values)) else 0.0
    if mean <= 0:
        return [None] * len(values)
    return [None if not np.isfinite(v) else 100.0 * (v / mean - 1.0) for v in values]


def gatingRows(values):
    """[(label, text)] of the gating information present, in reading order."""
    rows = []
    get = lambda key: number(values, key)

    def add(label, text):
        if text:
            rows.append((label, text))

    slots = get("GatingTimeSlots") or get("Gates")
    add("Frames (time slots)", f"{slots:.0f}" if slots else "")
    framing = (values.get("GatingFramingType") or "").strip()
    add("Framing", {"FORW": "forward from the R wave", "BACK": "backward from the R wave",
                    "PCNT": "% of the R-R interval"}.get(framing.upper(), framing))
    rate, rateSource = get("GatingHeartRateBpm"), values.get("GatingHeartRateSource") or ""
    add("Heart rate", f"{rate:.0f} bpm" + (f" ({rateSource})" if rateSource and "DICOM" not in rateSource else "")
        if rate else "")
    start = get("GatingStartHeartRateBpm")
    add("Heart rate at the start", f"{start:.0f} bpm" if start else "")
    rr, rrSource = get("GatingNominalIntervalMs"), values.get("GatingNominalIntervalSource") or ""
    rrNote = {"DICOM heart rate": " (from the heart rate)"}.get(rrSource, " (estimated from the frame time)"
                                                                if rrSource.startswith("estimated") else "")
    add("Mean R-R interval", f"{rr:.0f} ms{rrNote}" if rr else "")
    width = get("GatingSlotWidthMs")
    add("Frame width", f"{width:.0f} ms" if width else "")
    low, high, window = get("GatingLowRRMs"), get("GatingHighRRMs"), get("GatingRRWindowPercent")
    if low and high:
        add("Accepted R-R window", f"{low:.0f} - {high:.0f} ms" + (f" (± {window:.0f} %)" if window else ""))
    flag = (values.get("GatingBeatRejection") or "").strip().upper()
    add("Beat rejection", {"Y": "on", "N": "off"}.get(flag, flag))
    skip = get("GatingSkipBeats")
    add("Beats skipped after a rejection", f"{skip:.0f}" if skip is not None else "")
    acquired, rejected = get("GatingIntervalsAcquired"), get("GatingIntervalsRejected")
    items, perView = get("GatingDataItems"), (values.get("GatingItemsAreViews") or "") == "1"
    scopeValue = values.get("GatingBeatsScope") or ""
    scope = ""
    if perView:
        scope = f" (summed over {items:.0f} views)"
    elif scopeValue == "view":
        scope = " (one projection view: the only beat counts in the file)"
    elif scopeValue == "acquisition":
        scope = " (whole acquisition)"
    if acquired is not None:
        add("Accepted beats", f"{acquired:.0f}{scope}")
    if rejected is not None:
        percent = get("GatingRejectedPercent")
        add("Rejected beats", f"{rejected:.0f}" + (f" ({percent:.1f} %)" if percent is not None else ""))
    return rows


def rrPerView(values):
    """(views, nominal R-R per view, low, high) when the scanner stores the gating per projection
    view (Mediso), else None."""
    if (values.get("GatingItemsAreViews") or "") != "1":
        return None
    nominal = jsonList(values, "GatingPerItemNominalIntervalMs")
    if not nominal or not any(v is not None for v in nominal):
        return None
    n = len(nominal)
    low = jsonList(values, "GatingPerItemLowRRMs") or [None] * n
    high = jsonList(values, "GatingPerItemHighRRMs") or [None] * n
    return list(range(1, n + 1)), nominal, low[:n], high[:n]


def qcFindings(imageCounts, myocardiumCounts=None, projectionCounts=None, values=None):
    """[(level, text)] with level "warning" or "ok": frame count deviations, rejected beats and heart
    rate drift during the acquisition."""
    findings = []
    values = values or {}
    for name, series in (("Projection", projectionCounts), ("Image", imageCounts), ("Myocardium", myocardiumCounts)):
        if not series:
            continue
        dev = deviations(series)
        bad = [(i + 1, d) for i, d in enumerate(dev) if d is not None and abs(d) > COUNT_BAND_PERCENT]
        if bad:
            frames = ", ".join(f"{i} ({d:+.0f} %)" for i, d in bad)
            findings.append(("warning", f"{name} counts of frame {frames} differ more than "
                                        f"{COUNT_BAND_PERCENT:.0f} % from the mean: arrhythmia or gating error?"))
        elif any(d is not None for d in dev):
            spread = max(abs(d) for d in dev if d is not None)
            findings.append(("ok", f"{name} counts within ± {spread:.1f} % of the mean."))
    percent = number(values, "GatingRejectedPercent")
    if percent is not None and percent > REJECTED_WARNING_PERCENT:
        findings.append(("warning", f"{percent:.0f} % of the beats were rejected (irregular rhythm)."))
    perView = rrPerView(values)
    if perView is not None:
        rr = np.array([np.nan if v is None else v for v in perView[1]], dtype=float)
        quarter = max(1, len(rr) // 4)
        first, last = np.nanmean(rr[:quarter]), np.nanmean(rr[-quarter:])
        if np.isfinite(first) and np.isfinite(last) and first > 0:
            drift = 100.0 * (last / first - 1.0)
            level = "warning" if abs(drift) > RR_DRIFT_WARNING_PERCENT else "ok"
            findings.append((level, f"R-R interval {first:.0f} ms at the start, {last:.0f} ms at the end of the "
                                    f"acquisition ({drift:+.0f} %)."))
    return findings
