"""Adjustable parameters of the workflow steps.

Every constant that changes a result or a display is listed here per step (STEP_PARAMETERS), with its
built-in default, range and the module constant(s) it sets (targets). A step panel shows them under
"Advanced parameters" (ParameterPanel) next to its own controls, which it binds to the same panel, and
offers "Save parameters" (kept in the Slicer settings, used from then on) and "Restore defaults".
Saved values are applied to the module constants when a step module is imported (applySaved), so the
analysis uses them even before its panel is opened.

A target is "module:NAME", optionally followed by ".key" for a dict entry (repeatable) and "#i" for one
element of a tuple, e.g. "MyocardialPerfusionSPECTLib.myocardium:DEFAULTS.rowSpacing" or
"MPIMyocardium:SURFACES.endo#1". Only modules already imported are set; each step module calls
applySaved() once it is imported, so its own constants get their saved values too.
"""

import importlib
import json
import logging
import sys

SETTINGS_GROUP = "MyocardialPerfusionSPECT/Parameters"
LIB = "MyocardialPerfusionSPECTLib"


class Parameter:
    """One adjustable constant. default: bool, int, float, a tuple of numbers, or a color (kind="color",
    a tuple of 3 numbers 0..colorScale). factor: target value = shown value x factor (e.g. MB -> bytes).
    choices: ((label, value), ...) for a choice. restart: the change takes effect after restarting Slicer."""

    def __init__(self, key, label, default, targets, minimum=None, maximum=None, step=None, decimals=None,
                 suffix="", tooltip="", group="", kind=None, colorScale=1.0, factor=1, choices=None, restart=False):
        self.key = key
        self.label = label
        self.default = default
        self.targets = (targets,) if isinstance(targets, str) else tuple(targets)
        self.minimum, self.maximum, self.step, self.decimals = minimum, maximum, step, decimals
        self.suffix = suffix
        self.tooltip = tooltip
        self.group = group
        self.kind = kind or _kindOf(default, choices)
        self.colorScale = colorScale
        self.factor = factor
        self.choices = choices
        self.restart = restart

    def normalize(self, value):
        """A saved or entered value in the type of the default (tuples from JSON lists)."""
        if self.kind in ("tuple", "color"):
            items = list(value)
            if len(items) != len(self.default):
                raise ValueError(f"{self.key}: {len(self.default)} values expected")
            kinds = [type(x) for x in self.default]
            return tuple(int(round(float(v))) if k is int else float(v) for v, k in zip(items, kinds))
        if self.kind == "bool":
            return bool(value)
        if self.kind == "int":
            return int(round(float(value)))
        if self.kind == "float":
            return float(value)
        return value

    def targetValue(self, value):
        if self.factor != 1 and self.kind in ("int", "float"):
            return type(self.default)(value * self.factor) if self.kind == "int" else value * self.factor
        return value


def _kindOf(default, choices):
    if choices is not None:
        return "choice"
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, tuple):
        return "tuple"
    raise TypeError(f"unsupported parameter default {default!r}")


def P(*args, **kwargs):
    return Parameter(*args, **kwargs)


def _color(key, label, default, targets, scale, group, tooltip=""):
    return Parameter(key, label, default, targets, kind="color", colorScale=scale, group=group, tooltip=tooltip)


MYO = f"{LIB}.myocardium"
DISPLAY = f"{LIB}.display"
GATING = f"{LIB}.gating"
STUDIES = f"{LIB}.studies"

STEP_PARAMETERS = {
    "roles": [
        P("minGates", "Fewest gates", 4, f"{STUDIES}:MIN_GATES", 2, 64, 1, group="Gated sequences",
          tooltip="A sequence with fewer frames is not taken as a gated study"),
        P("maxGates", "Most gates", 32, f"{STUDIES}:MAX_GATES", 2, 128, 1, group="Gated sequences",
          tooltip="A sequence with more frames is not taken as a gated study"),
    ],
    "processing": [
        P("roiRadius", "Heart ROI radius", 55.0, "MPIProcessing:DEFAULT_ROI_RADIUS_MM", 10.0, 150.0, 1.0, 0, " mm",
          "Radius of the heart ROI of a new study (also the half size of the output volumes)", "Heart ROI"),
        P("snapThreshold", "Click snapping threshold", 0.5, "MPIProcessing:SNAP_THRESHOLD", 0.0, 1.0, 0.05, 2,
          tooltip="A MIP click snaps to the count-weighted center along the projection line; samples below this "
                  "fraction of the line maximum do not pull the point", group="Heart ROI"),
        P("clickTolerance", "Click tolerance", 5, "MPIProcessing:CLICK_TOLERANCE_PX", 1, 30, 1, suffix=" px",
          tooltip="Mouse movement up to this many pixels still counts as a click in the MIPs", group="Heart ROI"),
        P("ringPoints", "ROI outline points", 48, "MPIProcessing:RING_POINTS", 12, 180, 4,
          tooltip="Control points of the ROI circle drawn in each MIP", group="Heart ROI"),
        P("axisAzimuth", "Initial long axis azimuth", 45.0, "MPIProcessing:DEFAULT_AXIS_AZIMUTH_DEG", -180.0, 180.0,
          1.0, 0, " deg", "Starting long axis in the transaxial plane, from patient left toward anterior",
          "Cardiac axes"),
        P("axisElevation", "Initial long axis elevation", -25.0, "MPIProcessing:DEFAULT_AXIS_ELEVATION_DEG",
          -80.0, 80.0, 1.0, 0, " deg", "Starting long axis elevation (negative: the apex lies inferior)",
          "Cardiac axes"),
        P("maxElevation", "Largest long axis elevation", 80.0, "MPIProcessing:MAX_ELEVATION_DEG", 10.0, 89.0, 1.0, 0,
          " deg", "The long axis cannot be tilted further out of the transaxial plane", "Cardiac axes"),
        P("maskLong", "Ellipsoid long semi-axis", 1.0, "MPIProcessing:MASK_LONG_FRACTION", 0.3, 3.0, 0.05, 2,
          " x ROI", "Long axis half length of a new mask ellipsoid, in ROI radii", "Bottom-row mask"),
        P("maskShort", "Ellipsoid short semi-axes", 0.85, "MPIProcessing:MASK_SHORT_FRACTION", 0.3, 3.0, 0.05, 2,
          " x ROI", "Anterior-inferior and septal-lateral half widths of a new mask ellipsoid, in ROI radii",
          "Bottom-row mask"),
    ],
    "myocardium": [
        P("rowSpacing", "Ray row spacing", 2.0, f"{MYO}:DEFAULTS.rowSpacing", 0.5, 6.0, 0.25, 2, " mm",
          "Distance between the rings of rays along the long axis", "Wall fit"),
        P("capRows", "Apical cap rows", 12, f"{MYO}:DEFAULTS.capRows", 4, 30, 1,
          tooltip="Rows of rays over the apical cap (90 deg / rows apart)", group="Wall fit"),
        P("sampleStep", "Sample step along a ray", 1.0, f"{MYO}:DEFAULTS.step", 0.25, 4.0, 0.25, 2, " mm",
          group="Wall fit"),
        P("iterations", "Fit iterations", 3, f"{MYO}:DEFAULTS.iterations", 1, 10, 1, group="Wall fit",
          tooltip="Rounds of mid-wall search and smoothing"),
        P("thicknessSmoothing", "Thickness map smoothing", 3.0, f"{MYO}:DEFAULTS.thicknessSmoothing", 0.0, 8.0, 0.25,
          2, " samples", "Gaussian sigma of the wall thickness map", "Wall fit"),
        P("poleSpacing", "Smoothing width at the axis", 3.0, f"{MYO}:POLE_SPACING_MM", 0.5, 10.0, 0.25, 2, " mm",
          "Smoothing near the long axis (apex) per unit of surface smoothing", "Wall fit"),
        P("shapeModel", "Shape model through defects", True, f"{MYO}:DEFAULTS.shapeModel",
          tooltip="Rays without a wall peak (a defect, also at the apex) follow an ellipsoid fitted to the whole "
                  "LV instead of bridging between their neighbours", group="Defects"),
        P("shapePriorWeight", "Shape model weight", 0.03, f"{MYO}:SHAPE_PRIOR_WEIGHT", 0.0, 1.0, 0.01, 3,
          tooltip="Weight of the shape model against the rays' own count peaks (weights 0..1); higher: dim walls "
                  "follow the model more", group="Defects"),
        P("visibilityReference", "Typical wall (percentile)", 50.0, f"{MYO}:VISIBILITY_REFERENCE", 10.0, 95.0, 5.0,
          0, "", "A wall's visibility is its counts above the cavity's blood pool against those of this percentile "
                 "of all walls (a hot spot above it does not make the other walls look dim)", "Defects"),
        P("shapeDim", "Defect for the model below", 0.45, f"{MYO}:SHAPE_DIM", 0.1, 1.0, 0.05, 2, " x peak",
          "Walls dimmer than this share of the bright walls, and the walls next to them, are left out of the shape "
          "model fit and of the wall thickness", "Defects"),
        P("shapeBorder", "Defect border left out", 2.0, f"{MYO}:SHAPE_BORDER_SIGMA", 0.0, 6.0, 0.25, 2,
          " samples", "Width of the border around a defect left out of the shape model fit", "Defects"),
        P("shapeBasalWeight", "Shape model basal weight", 0.25, f"{MYO}:SHAPE_BASAL_WEIGHT", 0.0, 1.0, 0.05, 2,
          tooltip="Weight of the walls basal of the LV center in the shape model fit (the apical half sets the "
                  "curve across the apex)", group="Defects"),
        P("shapeCenterSigma", "Shape model center spread", 15.0, f"{MYO}:SHAPE_CENTER_SIGMA", 1.0, 100.0, 1.0, 0,
          " mm", "How far the ellipsoid center may stray from the valve plane before the prior pulls it back",
          "Defects"),
        P("shapeMaxTilt", "Shape model largest tilt", 25.0, f"{MYO}:SHAPE_MAX_TILT", 0.0, 45.0, 1.0, 0, " deg",
          "How far the ellipsoid's long axis may turn away from the LV axis", "Defects"),
        P("shapeTiltSigma", "Shape model tilt spread", 15.0, f"{MYO}:SHAPE_TILT_SIGMA", 1.0, 90.0, 1.0, 0, " deg",
          "How far the ellipsoid's axis may tilt before the prior pulls it back", "Defects"),
        P("maxValveTilt", "Largest valve plane tilt", 40.0, f"{MYO}:DEFAULTS.maxValveTilt", 0.0, 80.0, 1.0, 0, " deg",
          "Largest tilt of the valve plane against the short-axis plane", "Valve plane"),
        P("maxBaseNarrowing", "Steepest basal narrowing", 35.0, f"{MYO}:DEFAULTS.maxBaseNarrowing", 0.0, 80.0, 1.0, 0,
          " deg", "Steepest the wall may turn toward the axis on its way to the base", "Valve plane"),
        P("resolutionNM", "System resolution, SPECT", 12.0, f"{STUDIES}:SYSTEM_RESOLUTION_MM.NM", 3.0, 30.0, 0.5, 1,
          " mm", "Resolution (FWHM) set when a SPECT study is chosen", "System resolution"),
        P("resolutionPT", "System resolution, PET", 7.0, f"{STUDIES}:SYSTEM_RESOLUTION_MM.PT", 3.0, 30.0, 0.5, 1,
          " mm", "Resolution (FWHM) set when a PET study is chosen", "System resolution"),
        P("labelSpacing", "Segmentation grid", 1.5, "MPIMyocardium:LABEL_SPACING_MM", 0.5, 4.0, 0.25, 2, " mm",
          "Voxel size of the myocardium / cavity segmentation (volumes and mass)", "Results"),
        P("shellSpacing", "3D wall mesh spacing", 1.25, "MPIMyocardium:SHELL_SPACING_MM", 0.5, 5.0, 0.25, 2, " mm",
          group="3D view"),
        P("shellOpacity", "3D wall opacity", 0.25, "MPIMyocardium:SHELL_OPACITY", 0.0, 1.0, 0.05, 2, group="3D view"),
        _color("shellColor", "3D wall color", (0.85, 0.35, 0.25), "MPIMyocardium:SHELL_COLOR", 1.0, "3D view"),
        P("labelDistance", "Wall label distance", 45.0, "MPIMyocardium:LABEL_DISTANCE_MM", 0.0, 120.0, 1.0, 0, " mm",
          "How far outside the epicardium the wall labels float in 3D", "3D view"),
        _color("labelColor", "Wall label color", (0.85, 0.88, 0.9), "MPIMyocardium:LABEL_COLOR", 1.0, "3D view"),
        _color("endoColor", "Endocardium color", (0.95, 0.25, 0.25), "MPIMyocardium:SURFACES.endo#1", 1.0, "Contours"),
        _color("epiColor", "Epicardium color", (0.25, 0.85, 0.35), "MPIMyocardium:SURFACES.epi#1", 1.0, "Contours"),
        _color("midColor", "Mid-wall color", (1.0, 0.85, 0.1), "MPIMyocardium:SURFACES.mid#1", 1.0, "Contours"),
    ],
    "perfusion": [
        P("contourMaxSize", "Contour resolution", 160, "MPIReview:CONTOUR_MAX_SIZE", 40, 600, 10, suffix=" px",
          tooltip="Contours are computed at most at this tile size and enlarged above it", group="Slices"),
        _color("epiColor", "Epicardium contour", (64, 220, 90), "MPIReview:CONTOUR_COLORS.epi", 255, "Slices"),
        _color("endoColor", "Endocardium contour", (60, 190, 255), "MPIReview:CONTOUR_COLORS.endo", 255, "Slices"),
        P("minVisibleSlices", "Slices left when scrolling", 8, f"{DISPLAY}:MIN_VISIBLE_SLICES", 1, 40, 1,
          tooltip="A row scrolls past its last slice until this many slices are left", group="Slices"),
        P("centerGrid", "Stress-rest matching grid", 3.0, "MPIReview:CENTER_GRID_MM", 1.0, 8.0, 0.5, 1, " mm",
          "Grid of the myocardium mass center that matches the stress and rest slices", "Slices"),
        P("refitDelay", "Refit delay after resizing", 1000, [f"{DISPLAY}:REFIT_DELAY_MS", "MPIReview:REFIT_DELAY_MS",
                                                             "MPIGated:REFIT_DELAY_MS"], 100, 5000, 100, suffix=" ms",
          tooltip="The slice tiles are fitted again this long after the last resize (Perfusion and Gated Review)",
          group="Screen", restart=True),
        P("monitor2Delay", "Second monitor placement delay", 300, ["MPIReview:MONITOR2_PLACE_DELAY_MS",
                                                                   "MPIGated:MONITOR2_PLACE_DELAY_MS"],
          0, 3000, 50, suffix=" ms", tooltip="Wait before the second-monitor window is placed (Perfusion and "
                                             "Gated Review)", group="Screen"),
    ],
    "gatedqc": [
        P("maxThicknessChange", "Thickness change limit", 0.5, f"{MYO}:GATE_DEFAULTS.maxThicknessChange", 0.0, 3.0,
          0.05, 2, tooltip="Constant-volume models: a ray's thickness stays within reference / (1 + x) .. "
                           "reference x (1 + x)", group="Frame segmentation"),
        P("traceLow", "Defect: untraceable up to", 1.25, f"{MYO}:TRACE_RATIO_LOW", 1.0, 5.0, 0.05, 2,
          " x blood pool", "Rays whose wall counts in the static image are at most this multiple of the cavity's "
                           "blood pool cannot be traced (scar): their wall stays where the static wall is in "
                           "every frame (akinetic, no thickening)", "Defects"),
        P("traceHigh", "Defect: traced from", 1.75, f"{MYO}:TRACE_RATIO_HIGH", 1.0, 5.0, 0.05, 2,
          " x blood pool", "Rays whose wall reaches this multiple of the blood pool are traced in every frame "
                           "like normal wall, however dim; in between they are partly held", "Defects"),
        P("minDefectShift", "Defect search window", 0.3, f"{MYO}:MIN_DEFECT_SHIFT", 0.0, 1.0, 0.05, 2,
          " x limit", "Share of the wall motion limit searched on rays through a deep defect", "Defects"),
        P("defectMotionSigma", "Defect motion field width", 3.0, f"{MYO}:DEFECT_MOTION_SIGMA", 0.5, 10.0, 0.25, 2,
          " x smoothing", "Width of the motion field of the walls around a defect that its rays follow", "Defects"),
        P("defectPriorWeight", "Defect motion prior weight", 0.03, f"{MYO}:DEFECT_PRIOR_WEIGHT", 0.0, 1.0, 0.01, 3,
          tooltip="Weight of that motion field against the rays' own count peaks (weights 0..1)", group="Defects"),
        P("labelGrid", "Frame label grid", 2.0, "MPIGatedQC:LABEL_GRID_MM", 0.5, 5.0, 0.25, 2, " mm",
          "Voxel size of the per-frame myocardium / cavity labels (counts, cavity volumes, EF)", "Results"),
        P("countBand", "Count deviation flagged", 10.0, f"{GATING}:COUNT_BAND_PERCENT", 1.0, 50.0, 1.0, 0, " %",
          "A frame whose counts differ more from the mean is flagged", "QC findings"),
        P("rejectedWarning", "Rejected beats warning", 20.0, f"{GATING}:REJECTED_WARNING_PERCENT", 1.0, 90.0, 1.0, 0,
          " %", group="QC findings"),
        P("rrDriftWarning", "R-R drift warning", 10.0, f"{GATING}:RR_DRIFT_WARNING_PERCENT", 1.0, 50.0, 1.0, 0, " %",
          group="QC findings"),
        P("contourMaxSize", "Contour resolution", 160, "MPIGatedQC:CONTOUR_MAX_SIZE", 40, 600, 10, suffix=" px",
          tooltip="Contours are computed at most at this tile size and enlarged above it", group="Display"),
        _color("epiColor", "Epicardium contour", (64, 220, 90), ["MPIGatedQC:CONTOUR_COLORS.epi"], 255, "Display"),
        _color("endoColor", "Endocardium contour", (60, 190, 255), ["MPIGatedQC:CONTOUR_COLORS.endo"], 255, "Display"),
        _color("projectionsColor", "Curve: projections", (120, 180, 255), "MPIGatedQC:SERIES_COLORS.projections",
               255, "Display"),
        _color("imageColor", "Curve: image", (210, 210, 210), "MPIGatedQC:SERIES_COLORS.image", 255, "Display"),
        _color("myocardiumColor", "Curve: myocardium", (255, 140, 60), "MPIGatedQC:SERIES_COLORS.myocardium", 255,
               "Display"),
        P("tileCache", "Frame tile cache", 96, "MPIGatedQC:TILE_CACHE_BYTES", 8, 2048, 8, suffix=" MB",
          factor=1024 * 1024, tooltip="Memory for frame tiles kept for scrolling back", group="Memory", restart=True),
        P("contourCache", "Contour cache", 16, "MPIGatedQC:CONTOUR_CACHE_BYTES", 2, 512, 2, suffix=" MB",
          factor=1024 * 1024, tooltip="Memory for contour masks (Gated QC and Gated Review)", group="Memory",
          restart=True),
    ],
    "gated": [
        P("representativeFractions", "Three-slice positions", (0.25, 0.5, 0.75), "MPIGated:REPRESENTATIVE_FRACTIONS",
          0.0, 1.0, 0.05, 2, tooltip="Where the apical / mid / basal (or inferior / mid / anterior, septal / mid / "
                                     "lateral) slices lie along the myocardium of each axis, 0..1",
          group="Three slices per axis"),
        P("staticElevation", "Static 3D view elevation", 30.0, "MPIGated:STATIC_VIEW_ELEVATION_DEG", -90.0, 90.0, 1.0,
          0, " deg", "The static wall is seen from anteroseptal, raised this much toward the base",
          "Three slices per axis"),
        P("slicePlaneOpacity", "Slice plane opacity", 0.14, "MPIGated:SLICE_PLANE_OPACITY", 0.0, 1.0, 0.01, 2,
          group="Three slices per axis"),
        P("slicePlaneMargin", "Slice plane margin", 8.0, "MPIGated:SLICE_PLANE_MARGIN_MM", 0.0, 50.0, 1.0, 0, " mm",
          "The planes reach this far beyond the LV", "Three slices per axis"),
        _color("planeSA", "SA plane color", (0.99, 0.85, 0.21), "MPIGated:SLICE_PLANE_COLORS.SA", 1.0,
               "Three slices per axis"),
        _color("planeHLA", "HLA plane color", (0.85, 0.11, 0.38), "MPIGated:SLICE_PLANE_COLORS.HLA", 1.0,
               "Three slices per axis"),
        _color("planeVLA", "VLA plane color", (0.26, 0.63, 0.28), "MPIGated:SLICE_PLANE_COLORS.VLA", 1.0,
               "Three slices per axis"),
        P("thickeningSmoothing", "Thickening map smoothing", 2.5, f"{MYO}:THICKENING_SMOOTHING", 0.0, 8.0, 0.25, 2,
          " cells", "Smoothing of the % thickening map", "Polar maps"),
        P("thickeningCountSmoothing", "Thickening count smoothing", 3.0, f"{MYO}:THICKENING_COUNT_SMOOTHING", 0.0, 8.0,
          0.25, 2, " cells", "Smoothing of the ED and ES count maps before their ratio", "Polar maps"),
        P("phaseCoherenceSmoothing", "Phase coherence neighbourhood", 3.0, f"{MYO}:PHASE_COHERENCE_SMOOTHING", 0.5,
          8.0, 0.25, 2, " cells", "Neighbourhood (Gaussian sigma) over which the first-harmonic phases of the wall "
          "counts are pooled for the phase coherence (overlay and coherence-weighted hybrid thickening)",
          "Polar maps"),
        P("phaseCoherenceLow", "Phase coherence hatched below", 50.0, f"{MYO}:PHASE_COHERENCE_LOW", 0.0, 100.0, 5.0,
          0, " %", "The phase coherence overlay hatches the thickening map where the coherence is below this "
          "(0: random phases, 100: all agree)", "Polar maps"),
        P("motionSmoothing", "Wall motion map smoothing", 1.5, f"{MYO}:MOTION_SMOOTHING", 0.0, 8.0, 0.25, 2, " cells",
          group="Polar maps"),
        P("outwardMotion", "Outward motion at full color", 4.0, f"{DISPLAY}:OUTWARD_MOTION_MM", 0.5, 20.0, 0.5, 1,
          " mm", "Outward (dyskinetic) motion reaches the full turquoise at this many mm", "Polar maps"),
        P("efFontSize", "EF text size", 26, "MPIGated:EF_FONT_SIZE", 8, 80, 1, suffix=" px", group="3D wall"),
        _color("ghostColor", "ED reference color", (1.0, 1.0, 1.0), "MPIGated:GHOST_COLOR", 1.0, "3D wall"),
        P("labelDistance", "Wall label distance", 45.0, "MPIGated:LABEL_DISTANCE_MM", 0.0, 120.0, 1.0, 0, " mm",
          "How far outside the epicardium the wall labels float in 3D", "3D wall"),
        _color("volumeColor", "LV volume curve color", (255, 200, 80), "MPIGated:VOLUME_COLOR", 255, "Curves"),
        _color("epiColor", "Epicardium contour", (64, 220, 90), "MPIGated:CONTOUR_COLORS.epi", 255, "Slices"),
        _color("endoColor", "Endocardium contour", (60, 190, 255), "MPIGated:CONTOUR_COLORS.endo", 255, "Slices"),
    ],
}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def _settings():
    import slicer
    return slicer.app.userSettings()


def _settingsKey(step, key):
    return f"{SETTINGS_GROUP}/{step}/{key}"


def savedValue(step, key):
    """The saved value of a parameter or bound control (decoded JSON), or None."""
    try:
        text = _settings().value(_settingsKey(step, key))
    except Exception:
        return None
    if text is None or text == "":
        return None
    try:
        return json.loads(str(text))
    except ValueError:
        return None


def saveValue(step, key, value):
    _settings().setValue(_settingsKey(step, key), json.dumps(value))


def clearSaved(step):
    settings = _settings()
    settings.beginGroup(f"{SETTINGS_GROUP}/{step}")
    settings.remove("")
    settings.endGroup()


def parameter(step, key):
    for p in STEP_PARAMETERS.get(step, ()):
        if p.key == key:
            return p
    raise KeyError(f"{step}/{key}")


def currentValue(step, key):
    """Saved value of a parameter (normalized), else its default."""
    p = parameter(step, key)
    saved = savedValue(step, key)
    if saved is not None:
        try:
            return p.normalize(saved)
        except (TypeError, ValueError):
            logging.warning(f"MPI: saved parameter {step}/{key} ignored: {saved!r}")
    return p.default


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def _parseTarget(target):
    moduleName, path = target.split(":", 1)
    index = None
    if "#" in path:
        path, indexText = path.split("#", 1)
        index = int(indexText)
    parts = path.split(".")
    return moduleName, parts, index


def assign(target, value, importMissing=False):
    """Set one target to value. False when its module is not imported (and not importable)."""
    moduleName, parts, index = _parseTarget(target)
    module = sys.modules.get(moduleName)
    if module is None and importMissing:
        try:
            module = importlib.import_module(moduleName)
        except ImportError:
            module = None
    if module is None:
        return False
    if len(parts) == 1:
        container, name = module, parts[0]
        current = getattr(module, name)
    else:
        container = getattr(module, parts[0])
        for part in parts[1:-1]:
            container = container[part]
        name = parts[-1]
        current = container[name]
    if index is not None:
        items = list(current)
        items[index] = value
        value = type(current)(items)
    if container is module:
        setattr(module, name, value)
    else:
        container[name] = value
    return True


def apply(p, value):
    for target in p.targets:
        try:
            assign(target, p.targetValue(value))
        except Exception:
            logging.warning(f"MPI: could not set {target}:\n" + __import__("traceback").format_exc())


def applySaved(steps=None):
    """Set the module constants of every parameter (of the given steps, all by default) to its saved
    value or its default; modules not imported yet are skipped (they apply theirs on import)."""
    for step in steps or STEP_PARAMETERS:
        for p in STEP_PARAMETERS[step]:
            try:
                value = currentValue(step, p.key)
            except Exception:
                value = p.default
            apply(p, value)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
class ParameterPanel:
    """"Advanced parameters" of one step plus the step's own controls (bind), with Save parameters and
    Restore defaults. Create it where the section should go, bind the step's controls after they exist,
    then call loadSaved() at the end of setup (after the signal connections, so the step reacts as to a
    change by hand). onChanged(key) is called after an advanced parameter changed."""

    def __init__(self, step, parentLayout, onChanged=None, title="Advanced parameters", hint=None):
        import ctk
        import qt
        self.step = step
        self.onChanged = onChanged
        self.bindings = []          # (key, widget, kind, default)
        self.editors = {}           # parameter key -> (getter, setter)
        self._loading = False

        self.section = ctk.ctkCollapsibleButton()
        self.section.text = title
        self.section.collapsed = True
        parentLayout.addWidget(self.section)
        form = qt.QFormLayout(self.section)
        if hint:
            hintLabel = qt.QLabel(hint)
            hintLabel.wordWrap = True
            form.addRow(hintLabel)
        group = None
        for p in STEP_PARAMETERS.get(step, ()):
            if p.group and p.group != group:
                group = p.group
                header = qt.QLabel(f"<b>{group}</b>")
                form.addRow(header)
            widget, getter, setter = self._editor(p)
            tip = p.tooltip
            tip = (tip + " " if tip else "") + f"Default: {self._text(p, p.default)}."
            if p.restart:
                tip += " Takes effect after restarting Slicer."
            widget.toolTip = tip
            self.editors[p.key] = (getter, setter)
            self._loading = True      # the module constants already have these values (applySaved)
            try:
                setter(currentValue(step, p.key))
            finally:
                self._loading = False
            label = qt.QLabel(p.label + ":")
            label.toolTip = tip
            form.addRow(label, widget)

        buttons = qt.QHBoxLayout()
        self.saveButton = qt.QPushButton("Save parameters")
        self.saveButton.toolTip = ("Keep the current settings of this step (its controls and advanced parameters) "
                                   "as the starting values from now on, also after restarting Slicer")
        self.restoreButton = qt.QPushButton("Restore defaults")
        self.restoreButton.toolTip = ("Set the controls and advanced parameters of this step back to the built-in "
                                      "values and forget the saved ones")
        buttons.addWidget(self.saveButton)
        buttons.addWidget(self.restoreButton)
        parentLayout.addLayout(buttons)
        self.saveButton.connect("clicked(bool)", lambda _: self.save())
        self.restoreButton.connect("clicked(bool)", lambda _: self.restoreDefaults())

    # --- editors of the advanced parameters ---
    def _editor(self, p):
        import ctk
        import qt
        if p.kind == "bool":
            w = qt.QCheckBox()
            w.connect("toggled(bool)", lambda _: self._edited(p))
            return w, (lambda: bool(w.checked)), (lambda v: setattr(w, "checked", bool(v)))
        if p.kind == "choice":
            w = qt.QComboBox()
            for text, value in p.choices:
                w.addItem(text, value)
            w.connect("currentIndexChanged(int)", lambda _: self._edited(p))

            def setChoice(v):
                w.setCurrentIndex(max(0, w.findData(v)))
            return w, (lambda: w.itemData(w.currentIndex)), setChoice
        if p.kind == "color":
            w = ctk.ctkColorPickerButton()
            w.displayColorName = False
            w.connect("colorChanged(QColor)", lambda _: self._edited(p))

            def getColor():
                c = w.color
                values = (c.red(), c.green(), c.blue())
                return tuple(round(v / 255.0 * p.colorScale, 3) if p.colorScale != 255 else int(v) for v in values)

            def setColor(v):
                w.color = qt.QColor(*[int(round(float(x) / p.colorScale * 255.0)) for x in v])
            return w, getColor, setColor
        if p.kind == "tuple":
            frame = qt.QFrame()
            row = qt.QHBoxLayout(frame)
            row.setContentsMargins(0, 0, 0, 0)
            spins = []
            for value in p.default:
                spin = self._spin(p, isinstance(value, int))
                spin.connect("valueChanged(double)" if not isinstance(value, int) else "valueChanged(int)",
                             lambda _: self._edited(p))
                row.addWidget(spin)
                spins.append(spin)

            def setTuple(v):
                for spin, x in zip(spins, v):
                    spin.setValue(x)
            return frame, (lambda: p.normalize([s.value for s in spins])), setTuple
        spin = self._spin(p, p.kind == "int")
        spin.connect("valueChanged(int)" if p.kind == "int" else "valueChanged(double)", lambda _: self._edited(p))
        return spin, (lambda: p.normalize(spin.value)), (lambda v: spin.setValue(v))

    @staticmethod
    def _spin(p, integer):
        import qt
        spin = qt.QSpinBox() if integer else qt.QDoubleSpinBox()
        low = p.minimum if p.minimum is not None else (-1e6 if not integer else -1000000)
        high = p.maximum if p.maximum is not None else (1e6 if not integer else 1000000)
        if not integer:
            spin.setDecimals(p.decimals if p.decimals is not None else 2)
        spin.setRange(int(low) if integer else float(low), int(high) if integer else float(high))
        if p.step is not None:
            spin.setSingleStep(int(p.step) if integer else float(p.step))
        if p.suffix:
            spin.setSuffix(p.suffix)
        return spin

    @staticmethod
    def _text(p, value):
        if p.kind == "color":
            return "(" + ", ".join(f"{x:g}" for x in value) + ")"
        if p.kind == "tuple":
            return ", ".join(f"{x:g}" for x in value)
        if p.kind == "choice":
            return next((t for t, v in p.choices if v == value), str(value))
        if p.kind == "bool":
            return "on" if value else "off"
        return f"{value:g}{p.suffix}"

    def _edited(self, p):
        if self._loading:
            return
        try:
            apply(p, self.editors[p.key][0]())
        except Exception:
            logging.warning("MPI: parameter not applied:\n" + __import__("traceback").format_exc())
            return
        if self.onChanged is not None:
            try:
                self.onChanged(p.key)
            except Exception:
                logging.warning("MPI: refresh after a parameter change failed:\n" + __import__("traceback").format_exc())

    # --- the step's own controls ---
    def bind(self, key, widget, kind=None):
        """Include one of the step's controls in Save / Restore. kind: "value" (spin box, slider), "checked"
        (check box, checkable button), "range" (ctkRangeWidget), "combo" (item data, else text), "color"
        (ctkColorPickerButton, 0-255 RGB); guessed
        from the widget when not given. Its current value is taken as its default, so bind it right after
        it got its built-in value."""
        kind = kind or _controlKind(widget)
        self.bindings.append((key, widget, kind, _controlValue(widget, kind)))

    def loadSaved(self):
        """Put the saved values into the bound controls (their signals fire, as for a change by hand)."""
        for key, widget, kind, default in self.bindings:
            saved = savedValue(self.step, "control." + key)
            if saved is None:
                continue
            try:
                _setControlValue(widget, kind, saved)
            except Exception:
                logging.warning(f"MPI: saved setting {self.step}/{key} ignored")

    def save(self):
        import slicer
        for p in STEP_PARAMETERS.get(self.step, ()):
            saveValue(self.step, p.key, _jsonValue(self.editors[p.key][0]()))
        for key, widget, kind, _ in self.bindings:
            saveValue(self.step, "control." + key, _jsonValue(_controlValue(widget, kind)))
        slicer.util.showStatusMessage("Parameters of this step saved", 3000)

    def restoreDefaults(self):
        import slicer
        if not slicer.util.confirmOkCancelDisplay(
                "Set every control and advanced parameter of this step back to its built-in value and forget the "
                "saved values?", windowTitle="Restore defaults"):
            return
        clearSaved(self.step)
        for p in STEP_PARAMETERS.get(self.step, ()):
            self._loading = True
            try:
                self.editors[p.key][1](p.default)
            finally:
                self._loading = False
            apply(p, p.default)
        for key, widget, kind, default in self.bindings:
            try:
                _setControlValue(widget, kind, default)
            except Exception:
                logging.warning(f"MPI: default of {self.step}/{key} not restored")
        if self.onChanged is not None:
            try:
                self.onChanged(None)
            except Exception:
                logging.warning("MPI: refresh after restoring the defaults failed:\n"
                                + __import__("traceback").format_exc())
        slicer.util.showStatusMessage("Defaults of this step restored", 3000)


def _jsonValue(value):
    return list(value) if isinstance(value, tuple) else value


def _controlKind(widget):
    name = widget.className()
    if name == "ctkRangeWidget":
        return "range"
    if name == "ctkColorPickerButton":
        return "color"
    if name == "QComboBox":
        return "combo"
    if name in ("QCheckBox", "QPushButton", "QRadioButton", "QToolButton"):
        return "checked"
    return "value"


def _controlValue(widget, kind):
    if kind == "range":
        return [float(widget.minimumValue), float(widget.maximumValue)]
    if kind == "checked":
        return bool(widget.checked)
    if kind == "combo":
        data = widget.itemData(widget.currentIndex)
        return {"data": data} if data is not None else {"text": widget.currentText}
    if kind == "color":
        color = widget.color
        return [color.red(), color.green(), color.blue()]
    return float(widget.value) if not isinstance(widget.value, int) else int(widget.value)


def _setControlValue(widget, kind, value):
    if kind == "range":
        low, high = float(value[0]), float(value[1])
        widget.setValues(low, high)
    elif kind == "checked":
        widget.checked = bool(value)
    elif kind == "combo":
        index = widget.findData(value["data"]) if "data" in value else widget.findText(value.get("text", ""))
        if index >= 0:
            widget.setCurrentIndex(index)
    elif kind == "color":
        import qt
        widget.color = qt.QColor(*[int(v) for v in value[:3]])
    elif widget.className() in ("QSpinBox", "QDoubleSpinBox"):
        widget.setValue(value)
    else:
        widget.value = value
