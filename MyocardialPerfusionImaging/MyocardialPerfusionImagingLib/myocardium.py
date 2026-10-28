"""Left ventricular wall detection by count-profile sampling in the cardiac frame (numpy only).

The approach follows the classic quantitative SPECT programs (Emory, QPS/QGS) rather than voxel
labelling:

1. Hybrid sampling. Rays leave the LV long axis: perpendicular to it from the base up to the LV
   center (cylindrical part), and from the LV center toward the apex (spherical cap, polar angle
   90 deg at the center down to 0 deg on the axis). Every ray is a row (base -> apex) and column
   (angle around the axis) of one grid, so all results are 2D maps over the wall.
2. Mid-wall. The count peak along each ray, refined to sub-sample precision. Iterated a few times:
   the peaks are smoothed over the grid, weighted by how clearly each wall stands out above the cavity's
   blood pool (_wallVisibility, against the typical wall, so a hot spot does not make the other walls
   look dim), so rays through a perfusion defect follow their neighbours instead of jumping to noise,
   and the next search is limited to a window around the smoothed surface, which keeps the liver and gut
   out. Rays without a peak of their own (a deep defect) follow a shape model of the whole LV instead
   (_shapeModel: a truncated ellipsoid, its axis free to tilt, fitted to the walls that are seen, away
   from the defect's border, plus their smooth misfit; the peaks' deviations from it fade out into a
   defect), so a defect takes the curve of the walls around it; at the apex the neighbours alone bridge
   it flat.
3. Base. Along every column of rays (one angle around the axis), going from the LV center toward
   the base, the wall ends where its counts fall below a threshold (so the dim atria and great vessels
   behind the valves are not taken as wall). By default (baseMethod "global") a preliminary segment at
   a fraction of the column's own wall counts gives the cavity's blood pool and the mean counts of the whole
   myocardium, and every column's wall ends half way between the two (one level: steady in noisy gate
   frames); "adaptive" takes each column's own basal wall counts instead of the whole myocardium's, so a
   defect at the base only (inferobasal) does not end its column early; "fraction" keeps the fixed share of the column's wall counts (which already
   copes with a defect along the whole column, not with one at the base only). A base rim is fitted through
   these end points (robust least squares, tilt limited): a plane oblique to the long axis that, with
   valveShape "saddle" (default), may also rise and fall twice around the LV, as the mitral annulus does.
   Every column is cut at one height, where its mid-wall meets the rim, so the endo- and epicardium end
   together. The rim can be moved along the axis and tilted afterwards (valve offset and tilt).
4. Endo- and epicardium. With the surface fits (thicknessModel "width") where the count profile falls
   half way from the wall peak to the blood pool (_poolLevel: the cavity sampled CAVITY_GAP_MM inside
   the endocardium) and to the lowest counts outside, each half width fitted over the wall, with no
   model of the system resolution; with "slab" the mid-wall -/+ half a thickness read from the width as
   a slab of uniform uptake blurred at the system resolution, leaning to a prior thickness where the
   blur hides it (slabThickness). The ellipsoid fit removes the resolution from the width in quadrature.

Gate frames (fitGate): every frame's wall is tracked from the static wall or detected from scratch on
the same rays, and a surface fit (fitSurface: a B-spline, a deformable mesh or the legacy jump-limited
path) runs through the rays whose wall the frame shows and draws the others. Every frame's borders
come from its own profiles, fitted the same way, the cavity side against the static image's blood pool
(_gatePool: a frame's own cavity is filled by the walls' blur at end-systole).

Coordinates: center c, u base -> apex, w anterior, s = u x w septum (as in MPIProcessing). The angle
around the axis is measured from w toward s.
"""

import math

import numpy as np

DEFAULTS = {
    "angles": 60,              # rays around the long axis
    "rowSpacing": 2.0,         # mm between cylinder rows
    "capRows": 12,             # rows of the apical cap (90 deg / capRows apart)
    "step": 1.0,               # mm between samples along a ray
    "iterations": 3,
    "fit": "bspline",          # surface fit through the traced rays that also draws the untraced ones (WALL_FITS):
                               # "bspline", "mesh", "legacy" as the gate frames, or "ellipsoid": the shape model
                               # and Gaussian smoothing (the settings below marked "ellipsoid")
    "fitPull": 0.8,            # 0..1: how far a traced ray is pulled from its own peak to the fit (1: the fit)
    "smoothing": 1.25,         # "ellipsoid": Gaussian sigma of the surface smoothing, in grid samples (0 = none)
    "thicknessSmoothing": 3.0, # "ellipsoid": Gaussian sigma of the wall thickness map, in grid samples
    "shapeModel": True,        # "ellipsoid": rays without a wall peak (defects) follow an ellipsoid model of the whole LV
    "baseMethod": "global",    # BASE_METHODS: "global" ends the wall half way between the whole myocardium's counts and
                               # the blood pool, "adaptive" half way between its column's basal counts and the blood
                               # pool, "fraction" at baseFraction of its column's wall counts (_fitValvePlane)
    "baseFraction": 0.55,      # "fraction": wall ends where its counts fall below this fraction of its column's wall counts
    "basePrelimFraction": 0.50,  # "adaptive": the same for the preliminary segment the blood pool and basal wall come from
    "baseEdgeFraction": 0.50,  # "adaptive": the wall ends this far from its basal counts down to the blood pool
    "maxValveTilt": 40.0,      # deg, largest tilt of the valve plane against the short-axis plane
    "valveShape": "saddle",    # VALVE_SHAPES: "saddle" lets the base rim rise and fall twice around the LV on top of
                               # the plane's tilt (up to SADDLE_MAX_MM), "plane" keeps it flat
    "maxBaseNarrowing": 35.0,  # deg, steepest the wall may turn toward the axis on its way to the base
    "resolution": 12.0,        # system resolution (FWHM, mm) removed from the profile width
    "thicknessModel": "width", # surface fits and gate frames (THICKNESS_MODELS): "width" puts the endo- and epicardium
                               # where the profile falls half way to the blood pool / the counts outside, "slab"
                               # reads the width as a slab blurred at the resolution, leaning to a prior thickness
    "minThickness": 6.0,       # mm
    "maxThickness": 18.0,      # mm
    "valveOffset": 0.0,        # mm, + moves the valve plane toward the apex
    "valveTiltAnterior": 0.0,  # deg added to the detected tilt, + moves the anterior edge toward the apex
    "valveTiltSeptal": 0.0,    # deg added to the detected tilt, + moves the septal edge toward the apex
}


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
class WallGrid:
    """Ray grid in the cardiac frame. rows: cylinder rows (base -> center), then cap rows (-> apex)."""

    def __init__(self, center, u, w, radius, params=None, length=None):
        p = dict(DEFAULTS, **(params or {}))
        self.params = p
        self.center = np.asarray(center, dtype=float)
        self.u = _unit(u)
        self.w = _unit(np.asarray(w, dtype=float) - np.dot(w, self.u) * self.u)
        self.s = np.cross(self.u, self.w)
        self.radius = float(radius)
        self.nTheta = int(p["angles"])
        self.theta = np.arange(self.nTheta) * (2.0 * math.pi / self.nTheta)
        self.dz = float(p["rowSpacing"])
        self.length = float(radius if length is None else length)   # cylinder reach toward the base
        nCyl = int(math.ceil(self.length / self.dz)) + 1
        self.z = -self.dz * np.arange(nCyl)[::-1]           # -zMax ... 0 (0 = LV center)
        self.nCyl = nCyl
        self.nCap = int(p["capRows"])
        self.dphi = 0.5 * math.pi / self.nCap
        self.phi = 0.5 * math.pi - self.dphi * np.arange(1, self.nCap + 1)   # 90-d ... 0 deg
        self.nRows = nCyl + self.nCap
        self.t = np.arange(0.0, self.radius + 1e-6, float(p["step"]))

    def radial(self, theta):
        return np.cos(theta)[..., None] * self.w + np.sin(theta)[..., None] * self.s

    def rays(self):
        """(origins, directions), each [rows, angles, 3] in world RAS."""
        n = self.radial(self.theta)                                      # [A, 3]
        origins = np.empty((self.nRows, self.nTheta, 3))
        directions = np.empty((self.nRows, self.nTheta, 3))
        origins[:self.nCyl] = self.center + self.z[:, None, None] * self.u
        directions[:self.nCyl] = n[None]
        origins[self.nCyl:] = self.center
        directions[self.nCyl:] = (np.cos(self.phi)[:, None, None] * self.u
                                  + np.sin(self.phi)[:, None, None] * n[None])
        return origins, directions

    def points(self, distance):
        """World points at a distance along every ray ([rows, angles] map)."""
        origins, directions = self.rays()
        return origins + np.asarray(distance)[..., None] * directions

    def rowPosition(self):
        """Signed position of each row along the axis (z for cylinder rows, 0 for cap rows) and
        a continuous row coordinate, used to map points back onto the grid."""
        return np.concatenate([self.z, np.zeros(self.nCap)])

    def toGrid(self, points):
        """Continuous (row, angle) grid coordinates and distance along the ray of world points."""
        d = np.asarray(points, dtype=float) - self.center
        z = d @ self.u
        q = d - z[:, None] * self.u
        rho = np.linalg.norm(q, axis=1)
        theta = np.mod(np.arctan2(q @ self.s, q @ self.w), 2.0 * math.pi)
        col = theta / (2.0 * math.pi / self.nTheta)
        row = np.empty(len(d))
        dist = np.empty(len(d))
        cyl = z <= 0.0
        row[cyl] = (z[cyl] - self.z[0]) / self.dz
        dist[cyl] = rho[cyl]
        cap = ~cyl
        phi = np.arctan2(rho[cap], z[cap])                                  # 0 on the axis, 90 deg at z = 0
        row[cap] = (self.nCyl - 1) + (0.5 * math.pi - phi) / self.dphi
        dist[cap] = np.linalg.norm(d[cap], axis=1)
        return row, col, dist, z


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def fitWall(sample, center, u, w, radius, params=None):
    """Detect the LV wall. sample(points [N, 3] world) -> counts [N].

    Returns a dict with the WallGrid ("grid"), maps over [rows, angles]: "mid", "endo", "epi"
    (distance along each ray, mm), "thickness" (mm), "peak" (counts at the mid-wall, the polar map
    data), and "baseZ" (valve plane position along the axis, mm from the LV center, negative).

    Only the axis and its direction are taken from the user's frame. The grid's own center (where
    the cylinder rows end and the apical cap starts) is moved along the axis to the middle between
    the valve plane and the apical endocardium found by a first fit, and the wall is fitted again
    there: a center placed close to the base (or the apex) would otherwise leave a steep valve plane
    nowhere to cut the wall on one side. "centerShift" is that move (mm toward the apex); the
    distances of the result ("baseZ", "valvePlane", the maps) are from the moved center."""
    center = np.asarray(center, dtype=float)
    u = _unit(u)
    result = _fitOnGrid(sample, WallGrid(center, u, w, radius, params))
    grid = result["grid"]
    apex = float(np.median(result["endo"][-1]))                    # apical endocardium on the axis
    a, slope, saddle = _rimColumns(grid.theta, result["valvePlane"])
    radiusAtBase = np.array([np.interp(a, grid.z, result["epi"][:grid.nCyl, c]) for c in range(grid.nTheta)])
    rise = float(np.max(slope * radiusAtBase + saddle))
    # the middle of the LV, or further toward the apex when the steep side of the valve plane needs
    # it, but the cap keeps at least 8 mm of cavity in front of the apical endocardium
    shift = max(0.5 * (a + apex), a + max(rise, 0.0) + 3.0 * grid.dz)
    shift = float(np.clip(min(shift, apex - 8.0), -0.3 * radius, 0.3 * radius))
    if abs(shift) >= 2.0:
        try:
            result = _fitOnGrid(sample, WallGrid(center + shift * u, u, w, radius, params,
                                                 length=radius + max(shift, 0.0)))
        except ValueError:
            shift = 0.0
    else:
        shift = 0.0
    result["centerShift"] = shift
    return result


def _rayProfiles(sample, grid):
    """Counts along every ray of a grid [rows, angles, t]."""
    origins, directions = grid.rays()
    points = origins[:, :, None, :] + grid.t[None, None, :, None] * directions[:, :, None, :]
    profiles = np.asarray(sample(points.reshape(-1, 3)), dtype=float).reshape(grid.nRows, grid.nTheta, len(grid.t))
    return np.nan_to_num(profiles)


def _fitOnGrid(sample, grid, profiles=None):
    """fitWall on one grid (fixed center). profiles: the grid's ray profiles when already sampled."""
    p = grid.params
    profiles = _rayProfiles(sample, grid) if profiles is None else profiles
    if profiles.max() <= 0:
        raise ValueError("No counts around the LV center.")
    if p.get("fit", "ellipsoid") != "ellipsoid":
        return _fitOnGridSurface(grid, profiles)

    # 1. initial radius from the mean profile (the ring of the wall dominates any single organ)
    mean = profiles.mean(axis=(0, 1))
    usable = grid.t >= 3.0
    t0 = float(grid.t[usable][np.argmax(mean[usable])])
    t0 = max(t0, 8.0)
    lo = np.full((grid.nRows, grid.nTheta), 0.45 * t0)
    hi = np.full((grid.nRows, grid.nTheta), min(1.7 * t0, grid.t[-1]))

    half = max(6.0, 0.35 * t0)
    window = _angleWindow(grid, 9)
    for _ in range(int(p["iterations"])):
        peakT, peakValue = _peaks(profiles, grid.t, lo, hi)
        cavity = _cavityLevel(profiles, grid.t, peakT)
        weight, reference = _peakWeights(grid, peakT, peakValue, lo, hi, window, 0.3 * t0, cavity)
        surface, weight = _wallSurface(grid, profiles, peakT, weight, reference, p, p["smoothing"], 0.3 * t0,
                                       counts=peakValue)
        lo = np.maximum(surface - half, 1.0)
        hi = np.minimum(surface + half, grid.t[-1])
    peakT, peakValue = _peaks(profiles, grid.t, lo, hi)
    cavity = _cavityLevel(profiles, grid.t, peakT)
    weight, reference = _peakWeights(grid, peakT, peakValue, lo, hi, window, 0.3 * t0, cavity)
    mid, weight = _wallSurface(grid, profiles, peakT, weight, reference, p, 0.75 * p["smoothing"], 0.3 * t0,
                               p["smoothing"], peakValue)
    peak = _interp(profiles, grid.t, mid)

    # 3. base: where each column's wall ends, then a plane through those points
    plane, basePoints = _fitValvePlane(grid, mid, peak, p, _cavityFill(profiles, grid.t, mid, peak))
    rowCounts = np.median(peak[:grid.nCyl], axis=1)
    baseZ = plane[0]
    beyond = _beyondBase(grid, mid, plane)
    mid = _limitBasalNarrowing(grid, _holdBeyondBase(mid, beyond), p["maxBaseNarrowing"])
    peak = _interp(profiles, grid.t, mid)

    # 4. thickness from the profile width
    fwhm = _fwhm(profiles, grid.t, mid, peak, 1.6 * half)
    resolution = float(p["resolution"])
    physical = np.sqrt(np.maximum(fwhm ** 2 - resolution ** 2, 0.0))
    if p.get("shapeModel", True):
        thickness = _smoothFilled(physical, weight, p["thicknessSmoothing"], peakValue)
    else:
        thickness = _smoothMap(physical, np.maximum(weight, 1e-6), p["thicknessSmoothing"])
    thickness = np.clip(_poleSmooth(grid, thickness, np.maximum(weight, 1e-6), p["thicknessSmoothing"], mid),
                        p["minThickness"], p["maxThickness"])
    thickness = _holdBeyondBase(thickness, beyond)
    endo = _limitBasalNarrowing(grid, np.maximum(mid - 0.5 * thickness, 1.0), p["maxBaseNarrowing"])
    epi = _limitBasalNarrowing(grid, mid + 0.5 * thickness, p["maxBaseNarrowing"])

    return {"grid": grid, "mid": mid, "endo": endo, "epi": epi, "thickness": thickness, "peak": peak,
            "baseZ": baseZ, "detectedBaseZ": baseZ, "valvePlane": plane, "basePoints": basePoints,
            "rowCounts": rowCounts, "cavityLevel": cavity}


def _fitOnGridSurface(grid, profiles):
    """_fitOnGrid with a surface fit (fit "bspline", "mesh" or "legacy", as the gate frames detected from
    scratch, _fittedWall): the rays whose wall stands out of the blood pool set the fit, which draws the
    others; with shapeModel (the default) the fit is of their deviation from the ellipsoid shape model, so a
    defect keeps the model's curve, without it the fit alone bridges it (flatter). The thickness is the
    slab whose blurred profile has the profile's width, leaning to a prior where the blur hides it
    (slabThickness, as the gate frames), fitted the same way on the walls clear of a defect."""
    p = grid.params
    mid, weight, plane, basePoints = _fittedWall(grid, profiles, p["fit"], p["fitPull"], p.get("shapeModel", True))
    beyond = _beyondBase(grid, mid, plane)
    mid = _limitBasalNarrowing(grid, _holdBeyondBase(mid, beyond), p["maxBaseNarrowing"])
    peak = _interp(profiles, grid.t, mid)
    reach = 1.6 * max(6.0, 0.35 * float(np.median(mid[:grid.nCyl])))
    inner, outer = _wallHalfWidths(profiles, grid.t, mid, peak, reach)       # each side against its own floor
    pool = _poolLevel(profiles, grid.t, mid - inner)
    inner, outer = _wallHalfWidths(profiles, grid.t, mid, peak, reach, pool)  # the cavity side against the pool
    result = {"grid": grid, "mid": mid, "peak": peak, "gateWeight": weight, "valvePlane": plane,
              "profileInner": inner, "profileOuter": outer, "profileReach": reach,
              "fit": p["fit"], "fitPull": p["fitPull"]}
    thickness, endo, epi = _wallFromWidths(result, p["resolution"])
    return {"grid": grid, "mid": mid, "endo": endo, "epi": epi, "thickness": thickness, "peak": peak,
            "baseZ": plane[0], "detectedBaseZ": plane[0], "valvePlane": plane, "basePoints": basePoints,
            "rowCounts": np.median(peak[:grid.nCyl], axis=1), "cavityLevel": _cavityLevel(profiles, grid.t, mid),
            "poolLevel": pool, "profileSum": _profileCounts(grid, profiles)}


def _cavityLevel(profiles, t, peakT):
    """Counts of the blood pool in the LV cavity: the median over all rays of each profile's minimum
    between the axis and its peak."""
    inside = t[None, None, :] <= peakT[..., None]
    return float(np.median(np.min(np.where(inside, profiles, np.inf), axis=2)))


def _poolLevel(profiles, t, endo, gap=None):
    """Counts of the blood pool in the LV cavity for the inner edge of the wall: the median, over the rays,
    of each ray's median counts from the axis up to gap mm (CAVITY_GAP_MM) short of the endocardium, so the
    wall's blurred edge does not raise it. Where the cavity is too small for the gap (fewer than
    POOL_MIN_RAYS of the rays with 3 mm of cavity in front of it) it is taken up to the endocardium, and
    failing that it is the lowest counts inside (_cavityLevel)."""
    gap = CAVITY_GAP_MM if gap is None else float(gap)
    endo = np.asarray(endo, dtype=float)
    for g in (gap, 0.0):
        inside = t[None, None, :] <= (endo - g)[..., None]
        rays = inside.sum(axis=2) >= 1 + int(round(3.0 / max(float(t[1] - t[0]), 1e-6)))
        if rays.mean() >= POOL_MIN_RAYS:
            values = np.where(inside, profiles, np.nan)[rays]
            return float(np.median(np.nanmedian(values, axis=1)))
    return _cavityLevel(profiles, t, endo)


def _profileCounts(grid, profiles):
    """Counts on a grid's rays weighted by the volume each sample stands for (rays fan out from the axis
    and from the LV center), so a frame whose counts sit closer to the center does not count more."""
    t = grid.t
    cyl = (profiles[:grid.nCyl] * t).sum() * grid.dz
    cap = (profiles[grid.nCyl:] * t ** 2).sum() * grid.dphi
    return float(cyl + cap)


def _gatePool(reference, profiles, peakValue=None):
    """The static image's blood pool (_poolLevel, "poolLevel") in a frame's counts: the blood pool of a frame
    changes with its cavity (at end-systole the walls' blur fills a small cavity), so the frames take the
    static one, scaled by the frame's share of the static counts on the same rays (or of its wall counts).
    None for a static fit without it (older scenes): the frame then measures its own."""
    if "poolLevel" not in reference:
        return None
    if reference.get("profileSum", 0.0) > 0:
        scale = _profileCounts(reference["grid"], profiles) / float(reference["profileSum"])
    else:
        seen = _staticReliability(reference) > 0.99
        refPeak = np.asarray(reference["peak"], dtype=float)
        peakValue = _interp(profiles, reference["grid"].t, reference["mid"]) if peakValue is None else peakValue
        scale = float(np.median(peakValue[seen])) / max(float(np.median(refPeak[seen])), 1e-9) if np.any(seen) else 1.0
    return float(reference["poolLevel"]) * scale


def _wallVisibility(peakValue, cavity, seen=None):
    """0..1 per ray: how clearly the wall stands out, its counts above the cavity's blood pool against
    those of the typical wall (VISIBILITY_REFERENCE percentile of the seen rays, by default of those above
    the blood pool), 1 from the typical wall up. A hot spot (a bright lateral wall, a papillary muscle)
    raises only its own rays, so a dimmer but plainly seen wall (a septum at half the lateral wall's
    counts) still counts as wall; a defect, whose counts sink toward the blood pool's, does not."""
    above = np.asarray(peakValue, dtype=float) - float(cavity)
    seen = above > 0 if seen is None else seen
    pool = above[seen] if np.any(seen) else above
    typical = float(np.percentile(pool, VISIBILITY_REFERENCE)) if pool.size else 0.0
    if typical <= 0:
        return np.ones_like(above)
    return np.clip(above / typical, 0.0, 1.0)


def _peakWeights(grid, peakT, peakValue, lo, hi, window, tolerance, cavity=0.0):
    """Weight of every ray's peak (its visibility above the cavity, _wallVisibility, squared) and the
    neighbour reference the peaks are checked against. No weight: rays far from their neighbours (another
    organ, liver or gut) and rays whose maximum lies on the edge of their search window (no peak there:
    a defect, the counts only rise toward the bright walls around it)."""
    step = grid.t[1] - grid.t[0]
    edge = (peakT - lo < step) | (hi - peakT < step)
    weight = _wallVisibility(peakValue, cavity, ~edge) ** 2
    weight[edge] = 0.0
    reference = _circularMedian(peakT, window)
    nearPole = _nearPoleRows(grid, peakT)
    reference[nearPole] = np.median(peakT[nearPole], axis=1)[:, None]   # rays there nearly coincide
    weight[np.abs(peakT - reference) > tolerance] = 0.0
    reference = _ringReference(grid, peakT, weight, reference, tolerance)
    weight[np.abs(peakT - reference) > tolerance] = 0.0
    return weight, reference


def _wallSurface(grid, profiles, peakT, weight, reference, p, sigma, tolerance, poleSigma=None, counts=None):
    """Smooth mid-wall surface through the weighted peaks. Rays without a peak of their own take the shape
    model (_shapeModel, fitted to the walls on the LV side of a first valve plane) or, without it, their
    neighbours' reference; where the wall is dim over a wider area the surface leans on the model too.
    With the model the surface is the model plus the weighted peaks' smoothed deviations from it, which
    fade out where the local weight drops (SHAPE_PRIOR_WEIGHT): rays next to a defect keep the model's
    curve instead of their neighbours' level. Returns the surface and the weights (peaks far off the model
    lose theirs)."""
    prior = reference
    if p.get("shapeModel", True):
        rough = np.where(weight > 0, peakT, reference)
        peak = _interp(profiles, grid.t, rough)
        plane, _ = _fitValvePlane(grid, rough, peak, p, _cavityFill(profiles, grid.t, rough, peak))
        model, off = _shapeModel(grid, peakT, weight, ~_beyondBase(grid, rough, plane), tolerance, plane[0], counts)
        if model is not None:
            prior = model
            weight = np.where(off, 0.0, weight)
    if prior is reference:
        surface = _smoothSurface(np.where(weight > 0, peakT, prior), np.maximum(weight, 1e-6), sigma)
    else:
        # the model plus the peaks' smoothed deviations from it, which fade out where the local mean weight
        # drops (a defect): rays next to a defect keep the model's curve instead of their neighbours' level
        w = np.maximum(weight, 0.0)
        deviation = np.where(w > 0, peakT - prior, 0.0) * w
        if sigma > 0:
            kernel = _gaussianKernel(sigma)
            deviation, w = _convolve2d(deviation, kernel), _convolve2d(w, kernel)
        surface = prior + deviation / (w + SHAPE_PRIOR_WEIGHT)
    return _poleSmooth(grid, surface, np.maximum(weight, 1e-6), sigma if poleSigma is None else poleSigma), weight


# ---------------------------------------------------------------------------
# Gated frames
# ---------------------------------------------------------------------------
GATE_DEFAULTS = {
    "maxWallShift": 25.0,      # mm, "static": the mid-wall of a frame is searched this far from the static mid-wall
    "maxValveShift": 25.0,     # mm, "static": the valve plane of a frame stays this close to the static one
    "segmentation": "static",  # "static": every frame's wall tracked from the static wall (its displacement is fitted),
                               # "own": every frame detected from scratch on the static rays (its wall is fitted)
    "fit": "bspline",          # surface fit of the traced rays that also draws the untraced ones (GATE_FITS)
    "fitPull": 0.8,            # 0..1: how far a traced ray is pulled from its own peak to the fit (1: the fit)
    "myocardialVolume": "own", # GATE_VOLUMES: "own" every frame's wall thickness from its own profile widths,
                               # "ed" / "es" every frame's thickness scaled to the myocardial volume of that frame
    "valveCycle": "smooth",    # GATE_VALVE_CYCLES, "static": "smooth" the frames' base rim heights follow a smooth
                               # curve over the cycle (smoothValveHeights), "own" every frame's own height
}
GATE_SEGMENTATIONS = (("Track the static wall", "static"),
                      ("Each frame from scratch", "own"))
GATE_FITS = (("B-spline surface, bending penalty", "bspline"),
             ("Deformable mesh, every ray tied to its neighbours", "mesh"),
             ("Legacy: limited jump between neighbouring rays", "legacy"))
GATE_VOLUMES = (("Each frame's own (from its profile widths)", "own"),
                ("Constant, at the end-diastolic volume (QGS-like)", "ed"),
                ("Constant, at the end-systolic volume", "es"))
GATE_VALVE_CYCLES = (("Smooth over the cycle", "smooth"),
                     ("Each frame's own", "own"))
VALVE_CYCLE_HARMONICS = 2   # smoothValveHeights: harmonics of the cardiac cycle the base rim height may follow
WALL_FITS = GATE_FITS + (("Ellipsoid shape model, Gaussian smoothing (previous)", "ellipsoid"),)
SPLINE_KNOT_MM = 8.0       # B-spline fit: knot spacing over the wall (mm)
SPLINE_LENGTH_MM = 15.0    # B-spline fit: bending length, wall features shorter than this are smoothed out (mm)
MESH_LENGTH_MM = 4.0       # mesh fit: tension length, wall features shorter than this are smoothed out (mm)
LEGACY_MAX_SLOPE = 0.6     # legacy fit: largest change between neighbouring rays, mm per mm of wall
THICKNESS_PRIOR_MM = 10.0    # slabThickness: wall thickness a profile leans to where the blur hides its own ...
THICKNESS_PRIOR_SIGMA = 3.0  # ... with this spread (mm)
THICKNESS_MODELS = (("Half way to the blood pool and to the counts outside", "width"),
                    ("Blurred slab at the system resolution, with a thickness prior", "slab"))
CAVITY_GAP_MM = 5.0          # _poolLevel: the blood pool is sampled this far inside the endocardium ...
POOL_MIN_RAYS = 0.5          # ... where at least this share of the rays has 3 mm of cavity there (else without the gap)
WIDTH_SIGMA_MM = 0.75        # slabThickness: uncertainty of a measured profile width (mm)
BASE_METHODS = (("Adaptive, global: half way from the whole myocardium's counts to the blood pool", "global"),
                ("Adaptive: half way from the basal wall to the blood pool", "adaptive"),
                ("Fixed share of each angle's wall counts", "fraction"))
VALVE_SHAPES = (("Saddle: a tilted plane that may rise and fall twice around the LV", "saddle"),
                ("Plane", "plane"))
SADDLE_MAX_MM = 8.0        # "saddle" base rim: largest rise / fall of the rim against the tilted plane (mm)
SADDLE_RIDGE = 0.1         # "saddle" base rim: cost of the saddle in the fit, as a share of the wall ends' weights
BASE_BAND_MM = 12.0        # "adaptive" base: the basal wall counts are taken this far apical of the preliminary plane
BASE_SEARCH_MM = 16.0      # "adaptive" base: each column is searched toward the base from this far apical of it
BASE_MIN_CONTRAST = 0.25   # "adaptive" base: basal wall counts taken at least this share of the way from blood pool to wall
FIT_OUTLIER = 3.0          # rays further than this many robust misfits from the first fit lose their weight


SHAPE_PRIOR_WEIGHT = 0.03  # weight of the shape model against the rays' own peaks (weights 0..1) in the wall fit
SHAPE_RESIDUAL_SIGMA = 3.0 # grid samples: width of the field that carries the model's misfit into a defect
SHAPE_BASAL_WEIGHT = 0.25 # weight of the walls on the basal side of the grid center in the model fit
SHAPE_CENTER_SIGMA = 15.0  # mm: how far the shape model's ellipsoid center may stray from the valve plane ...
SHAPE_CENTER_PRIOR = 0.05  # ... and the weight of that prior (share of all wall weights)
SHAPE_MAX_TILT = 25.0      # deg: how far the shape model's long axis may turn away from the LV axis ...
SHAPE_TILT_SIGMA = 15.0    # ... and the width of the prior that keeps it close (same weight as the center's)
SHAPE_TILT_CAP = 0.25      # share of the apical cap fitted from which any better-fitting tilt is kept ...
SHAPE_TILT_GAIN = 0.25     # ... below it, only one that lowers the model's misfit by this share
VISIBILITY_REFERENCE = 50.0  # percentile of the walls' counts above the cavity that counts as the typical wall
SHAPE_DIM = 0.45          # walls dimmer than this share of the bright walls (counts) count as defect for the model ...
SHAPE_BORDER_SIGMA = 2.0  # ... and the walls within about this many grid samples of them are left out of its fit
SHAPE_FADE = 0.1           # how fast that misfit fades inside a defect (share of the typical local weight)
TRACE_RATIO_LOW = 1.25      # wall / blood pool counts (static image) up to which a ray's wall is untraceable (scar) ...
TRACE_RATIO_HIGH = 1.75     # ... and from which it is traced as wall; in between it is partly trusted
TRACE_POOL_FLOOR = 0.05     # share of the bright walls' counts below which the blood pool is not taken lower
THICKENING_SMOOTHING = 2.5        # thickeningMap: smoothing of the % map (map cells)
THICKENING_COUNT_SMOOTHING = 3.0  # thickeningMap: smoothing of the ED / ES count maps
MOTION_SMOOTHING = 1.5            # motionMap: smoothing of the mm map
PHASE_COHERENCE_SMOOTHING = 3.0   # phaseCoherenceMap: neighbourhood (Gaussian sigma, map cells) the phases are pooled over
PHASE_COHERENCE_LOW = 50.0        # %: coherence below which the overlay hatches the thickening map
MOTION_REFERENCES = (("Fixed LV center and axis", "fixed"),
                     ("Floating: each frame's LV center and epicardial apex", "floating"),
                     ("Floating: each frame's myocardium mass center", "mass"),
                     ("Floating: each frame's base center", "base"))
MOTION_APEX_ROWS = 2              # motionMap "floating": cap rows nearest the axis that place the epicardial apex
REGIONAL_EF_SMOOTHING = 2.0       # regionalEFMap: smoothing of the % map (map cells)
REGIONAL_EF_CUTS = (("Wedges from the long axis, cones from the LV center at the apex", "axis"),
                    ("Cones from each frame's base center", "base"))   # regionalEFMap: how the cavity is cut


def _traceability(peakValue, cavity):
    """0..1 per ray: whether its wall can be traced, from the wall's counts against the cavity's blood pool
    (ratio): 0 up to TRACE_RATIO_LOW (the wall does not stand out of the blood pool: a transmural defect),
    1 from TRACE_RATIO_HIGH. Unlike _wallVisibility it does not compare the wall with the other walls, so
    a hypoperfused wall that is still plainly above the blood pool counts as traceable."""
    peak = np.nan_to_num(np.asarray(peakValue, dtype=float))
    # a blood pool below TRACE_POOL_FLOOR of the bright walls (a phantom, no counts in the cavity) is taken
    # at that level, so rays without counts still count as untraceable
    pool = max(float(cavity or 0.0), TRACE_POOL_FLOOR * float(np.percentile(peak, 90)), 1e-9)
    low, high = float(TRACE_RATIO_LOW), float(TRACE_RATIO_HIGH)
    return np.clip((peak / pool - low) / max(high - low, 1e-3), 0.0, 1.0)


def _staticReliability(reference):
    """0..1 per ray: whether the wall can be traced in the static image (_traceability, the static image has
    all the counts), 0 where it cannot (scar), 1 where it can. Tracking the static wall, a frame's peaks
    count in its fit only as far as this allows. A static fit without the cavity level (older scenes) measures the peaks
    against the bright walls."""
    peak = np.nan_to_num(np.asarray(reference["peak"], dtype=float))
    if "cavityLevel" in reference:
        return _traceability(peak, reference["cavityLevel"])
    top = float(np.percentile(peak, 90))
    if top <= 0:
        return np.ones_like(peak)
    return np.clip((peak / top - 0.25) / 0.35, 0.0, 1.0)


def fitGate(sample, reference, params=None):
    """LV wall of one gate of a gated study, on the rays of the static fit (reference, from fitWall).

    The count peak of every ray is taken where its wall can be traced (_gateWeights: above the frame's blood
    pool, not at the edge of its search window, close to its neighbours), and the surface fit (fitSurface,
    GATE_FITS) runs through those peaks and draws the rays that cannot be traced: nothing holds a defect
    where the static wall is. With segmentation "static" the peaks are searched within maxWallShift of
    the static mid-wall and the fit is of their displacement from it, the valve plane stays within
    maxValveShift of the static one; with "own" every frame is detected from scratch as fitWall detects the
    static image and the fit is of the wall itself (of its deviation from the ellipsoid shape model with the
    static fit's shapeModel, as there). The wall thickness is the frame's own, from its count
    profile widths (gateThickness, at the static study's resolution). Returns a result dict like fitWall,
    so classify / labelGrid / wallStatistics apply."""
    grid = reference["grid"]
    p = grid.params
    g = dict(GATE_DEFAULTS, **(params or {}))
    profiles = _rayProfiles(sample, grid)
    if profiles.max() <= 0:
        raise ValueError("No counts around the LV center in this gate.")
    if g["segmentation"] == "own":
        mid, weight, plane, basePoints = _fittedWall(grid, profiles, g["fit"], g["fitPull"], p.get("shapeModel", True))
    else:
        mid, weight, plane, basePoints = _trackedGateWall(grid, profiles, reference, g)
    beyond = _beyondBase(grid, mid, plane)
    mid = _limitBasalNarrowing(grid, _holdBeyondBase(mid, beyond), p["maxBaseNarrowing"])
    peak = _interp(profiles, grid.t, mid)
    reach = 1.6 * max(6.0, 0.35 * float(np.median(mid[:grid.nCyl])))
    pool = _gatePool(reference, profiles, peak)
    inner, outer = _wallHalfWidths(profiles, grid.t, mid, peak, reach, pool)
    result = {"grid": grid, "mid": mid, "peak": peak, "baseZ": plane[0], "detectedBaseZ": plane[0],
              "valvePlane": plane, "basePoints": basePoints, "rowCounts": np.median(peak[:grid.nCyl], axis=1),
              "gateWeight": weight, "profileFwhm": inner + outer, "profileInner": inner, "profileOuter": outer,
              "profileReach": reach, "poolLevel": pool,
              "segmentation": g["segmentation"], "fit": g["fit"], "fitPull": float(g["fitPull"])}
    return _withThickness(result, p["resolution"])


def _gateWeights(grid, peakT, peakValue, lo, hi, cavity, tolerance):
    """0..1 per ray: how far its peak in a gate is trusted as its wall. The wall's counts against the
    frame's blood pool (_traceability, squared), 0 where the maximum lies on the edge of the search window
    (no peak: the counts only rise toward the walls around a defect) and where the peak is further than
    tolerance from its neighbours (noise, a neighbouring organ inside the window)."""
    step = grid.t[1] - grid.t[0]
    weight = _traceability(peakValue, cavity) ** 2
    weight[(peakT - lo < step) | (hi - peakT < step)] = 0.0
    neighbours = _circularMedian(peakT, _angleWindow(grid, 9))
    nearPole = _nearPoleRows(grid, peakT)
    neighbours[nearPole] = np.median(peakT[nearPole], axis=1)[:, None]   # rays there nearly coincide
    weight[np.abs(peakT - neighbours) > tolerance] = 0.0
    neighbours = _ringReference(grid, peakT, weight, neighbours, tolerance)
    weight[np.abs(peakT - neighbours) > tolerance] = 0.0
    return weight


def _trackedGateWall(grid, profiles, reference, g):
    """segmentation "static": (mid, weights, valve plane, base points) of a frame. The peaks are searched
    within maxWallShift of the static mid-wall and trusted where both the frame and the static image (which
    has all the counts) trace the wall; the fit is of their displacement from the static mid-wall, so the
    frame keeps the static wall's shape where it does not show its own. Rays past the static valve plane
    do not count. The base rim keeps the static rim's shape (tilt and saddle, from all the counts), moved to
    the frame's own height: the median height of the frame's wall ends over that shape."""
    p = grid.params
    refMid = reference["mid"]
    shift = float(g["maxWallShift"])
    lo = np.maximum(refMid - shift, 1.0)
    hi = np.minimum(refMid + shift, grid.t[-1])
    peakT, peakValue = _peaks(profiles, grid.t, lo, hi)
    tolerance = min(0.5 * shift, max(0.25 * float(np.median(refMid[:grid.nCyl])), 5.0))
    weight = _gateWeights(grid, peakT, peakValue, lo, hi, _cavityLevel(profiles, grid.t, peakT), tolerance)
    weight = weight * _staticReliability(reference) ** 2
    weight[_beyondBase(grid, refMid, reference["valvePlane"])] = 0.0
    mid = refMid + fitSurface(grid, peakT - refMid, weight, g["fit"], g["fitPull"], refMid)
    mid = np.clip(mid, lo, hi)
    peak = _interp(profiles, grid.t, mid)
    static = _rim(reference["valvePlane"])
    plane, basePoints = _fitValvePlane(grid, mid, peak, p, _cavityFill(profiles, grid.t, mid, peak))
    a = plane[0]
    if len(basePoints) >= max(6, grid.nTheta // 4):
        _, _, saddle = _rimColumns(np.arctan2(basePoints[:, 1], basePoints[:, 0]), static)
        a = float(np.median(basePoints[:, 2] - static[1] * basePoints[:, 0] - static[2] * basePoints[:, 1] - saddle))
    maxValve = float(g["maxValveShift"])
    return mid, weight, (float(np.clip(a, static[0] - maxValve, static[0] + maxValve)),) + static[1:], basePoints


def _fittedWall(grid, profiles, fit, pull, shapePrior=False):
    """(mid, weights, valve plane, base points) of a wall detected from scratch with a surface fit
    (fitSurface): the static image (_fitOnGridSurface) and the gate frames with segmentation "own". The
    search window starts from the mean profile and narrows around the fitted wall, the valve plane is
    fitted on the wall, the rays past it do not count in the next fit. With shapePrior the fit is of the
    peaks' deviation from the ellipsoid shape model (_shapeModel) fitted to the traced walls, so a defect
    keeps the model's curve, the deviations of the walls around it fading into it (a prolate apex behind an
    apical defect, which a fit alone bridges flatter)."""
    p = grid.params
    mean = profiles.mean(axis=(0, 1))
    usable = grid.t >= 3.0
    t0 = max(float(grid.t[usable][np.argmax(mean[usable])]), 8.0)
    lo = np.full((grid.nRows, grid.nTheta), 0.45 * t0)
    hi = np.full((grid.nRows, grid.nTheta), min(1.7 * t0, grid.t[-1]))
    half = max(6.0, 0.35 * t0)
    beyond = np.zeros((grid.nRows, grid.nTheta), dtype=bool)
    plane = None
    for _ in range(int(p["iterations"]) + 1):
        peakT, peakValue = _peaks(profiles, grid.t, lo, hi)
        weight = _gateWeights(grid, peakT, peakValue, lo, hi, _cavityLevel(profiles, grid.t, peakT), 0.3 * t0)
        weight[beyond] = 0.0
        model = None
        if shapePrior:
            if plane is None:
                rough = np.where(weight > 0, peakT, np.median(peakT[weight > 0]) if np.any(weight > 0) else t0)
                roughPeak = _interp(profiles, grid.t, rough)
                first, _ = _fitValvePlane(grid, rough, roughPeak, p, _cavityFill(profiles, grid.t, rough, roughPeak))
                inside = ~_beyondBase(grid, rough, first)
                baseZ = first[0]
            else:
                inside, baseZ = ~beyond, plane[0]
            model, off = _shapeModel(grid, peakT, weight, inside, 0.3 * t0, baseZ, peakValue)
        if model is not None:
            weight = np.where(off, 0.0, weight)
            mid = model + fitSurface(grid, peakT - model, weight, fit, pull, model)
        else:
            mid = fitSurface(grid, peakT, weight, fit, pull, np.where(weight > 0, peakT, t0))
        mid = np.clip(mid, 1.0, grid.t[-1])
        peak = _interp(profiles, grid.t, mid)
        plane, basePoints = _fitValvePlane(grid, mid, peak, p, _cavityFill(profiles, grid.t, mid, peak))
        beyond = _beyondBase(grid, mid, plane)
        lo = np.maximum(mid - half, 1.0)
        hi = np.minimum(mid + half, grid.t[-1])
    return mid, weight, plane, basePoints


def gateThickness(result, resolution):
    """Wall thickness of a gate frame from its own count profiles (fitGate), for a system resolution (FWHM
    mm), as _wallFromWidths."""
    return _wallFromWidths(result, resolution)[0]


def _wallFromWidths(result, resolution):
    """(thickness, endo, epi) of a wall from its profiles' half widths (_wallHalfWidths: "profileInner",
    "profileOuter"), fitted over the wall with the wall's surface fit (fitSurface) on the walls clear of a
    defect (a profile cut short by a defect's edge reads thin), which draws the others, limited like the
    static fit. thicknessModel "width": each half width is fitted on its own and the endo- and epicardium
    are where the profile falls half way to the blood pool and to the counts outside. "slab": the width is
    read as a blurred slab with a prior (slabThickness, at resolution) and the surfaces are the mid-wall
    -/+ half of it."""
    grid, p = result["grid"], result["grid"].params
    weight = np.where(_clearWalls(result["gateWeight"], result["peak"]), result["gateWeight"], 0.0)
    fit = result.get("fit", GATE_DEFAULTS["fit"])
    pull = result.get("fitPull", GATE_DEFAULTS["fitPull"])
    mid = result["mid"]
    beyond = _beyondBase(grid, mid, result["valvePlane"])
    inner = result.get("profileInner")
    outer = result.get("profileOuter")
    if inner is None:   # older results: the whole width, split evenly
        inner = outer = 0.5 * result["profileFwhm"]
    if p.get("thicknessModel", "width") == "slab":
        widths = slabThickness(inner + outer, resolution, result["profileReach"])
        widths = fitSurface(grid, widths, weight, fit, pull, mid)
        thickness = _holdBeyondBase(np.clip(widths, p["minThickness"], p["maxThickness"]), beyond)
        endo, epi = _wallSurfaces(grid, mid, thickness, p)
        return thickness, endo, epi
    inner = np.maximum(fitSurface(grid, inner, weight, fit, pull, mid), 0.5)
    outer = np.maximum(fitSurface(grid, outer, weight, fit, pull, mid), 0.5)
    thickness = inner + outer
    scale = np.clip(thickness, p["minThickness"], p["maxThickness"]) / thickness
    inner = _holdBeyondBase(inner * scale, beyond)
    outer = _holdBeyondBase(outer * scale, beyond)
    endo = _limitBasalNarrowing(grid, np.maximum(mid - inner, 1.0), p["maxBaseNarrowing"])
    epi = _limitBasalNarrowing(grid, mid + outer, p["maxBaseNarrowing"])
    return inner + outer, endo, epi


def _withThickness(result, resolution):
    """The result with the frame's own wall thickness at a resolution and its endo- and epicardium."""
    thickness, endo, epi = _wallFromWidths(result, resolution)
    return dict(result, thickness=thickness, endo=endo, epi=epi, resolutionMm=float(resolution))


def _wallSurfaces(grid, mid, thickness, p):
    """Endo- and epicardium: mid-wall -/+ half the thickness, the base kept from folding in."""
    endo = _limitBasalNarrowing(grid, np.maximum(mid - 0.5 * thickness, 1.0), p["maxBaseNarrowing"])
    epi = _limitBasalNarrowing(grid, mid + 0.5 * thickness, p["maxBaseNarrowing"])
    return endo, epi


def smoothValveHeights(results, harmonics=None):
    """The gate frames (fitGate results, None for a frame that failed) with their base rim heights on a smooth
    curve over the cardiac cycle: a robust fit (outlier frames out) of the mean and the first harmonics of the
    cycle to the frames' own heights, frame i of n at the phase 2 pi i / n. The base descends toward the apex
    in systole and returns in diastole; a frame whose noisy wall ends put its rim a few mm off no longer
    jumps. The rim's shape (tilt, saddle) is kept; "detectedBaseZ" keeps the frame's own height. Too few
    frames for the fit: the frames as they are."""
    harmonics = VALVE_CYCLE_HARMONICS if harmonics is None else int(harmonics)
    known = [i for i, r in enumerate(results) if r is not None]
    if len(known) < 2 * harmonics + 3:
        return list(results)
    phase = 2.0 * math.pi * np.array(known, dtype=float) / len(results)
    design = np.column_stack([np.ones(len(known))] + [f(k * phase) for k in range(1, harmonics + 1)
                                                      for f in (np.cos, np.sin)])
    heights = np.array([_rim(results[i]["valvePlane"])[0] for i in known])
    weights = np.ones(len(known))
    for _ in range(5):
        sw = np.sqrt(weights)
        coef = np.linalg.lstsq(design * sw[:, None], heights * sw, rcond=None)[0]
        residual = heights - design @ coef
        scale = max(1.4826 * float(np.median(np.abs(residual))), 1.0)
        weights = np.clip(1.0 - (residual / (3.0 * scale)) ** 2, 0.0, 1.0) ** 2 + 1e-6
    out = list(results)
    for i, a in zip(known, design @ coef):
        rim = _rim(results[i]["valvePlane"])
        out[i] = dict(results[i], valvePlane=(float(a),) + rim[1:], baseZ=float(a), valveCycle="smooth")
    return out


def _rayVolumes(result, inner, outer, power):
    """Volume (ml) between two distances along every ray of the wall below the valve plane, summed:
    cylinder rows as rings (r^2 / 2 dtheta dz), apical cap rows as spherical shells
    (r^3 / 3 sin(phi) dphi dtheta). power 1: the shell between inner and outer, power 0: inside outer."""
    grid = result["grid"]
    nCyl = grid.nCyl
    dTheta = 2.0 * math.pi / grid.nTheta
    inside = ~_beyondBase(grid, result["mid"], result["valvePlane"])
    inner = inner if power else np.zeros_like(outer)
    ring = (outer[:nCyl] ** 2 - inner[:nCyl] ** 2) / 2.0 * dTheta * grid.dz
    shell = (outer[nCyl:] ** 3 - inner[nCyl:] ** 3) / 3.0 * np.sin(grid.phi)[:, None] * grid.dphi * dTheta
    return (float(ring[inside[:nCyl]].sum()) + float(shell.sum())) / 1000.0


def myocardialVolumeMl(result):
    """Myocardial volume (ml) of a wall fit: between endo- and epicardium, below the valve plane."""
    return _rayVolumes(result, np.asarray(result["endo"], dtype=float), np.asarray(result["epi"], dtype=float), 1)


def cavityVolumeMl(result):
    """LV cavity volume (ml) of a wall fit: inside the endocardium, below the valve plane."""
    return _rayVolumes(result, None, np.asarray(result["endo"], dtype=float), 0)


def constantVolumeGates(results, reference="ed"):
    """The gate frames (fitGate results, None for a frame that failed) with the myocardial volume of one frame,
    as QGS constrains its surfaces (Germano et al., J Nucl Med 1995: "constant myocardial volume throughout the
    cardiac cycle"). reference "ed": the frame with the largest cavity, "es": the smallest. Every other frame
    keeps its mid-wall and the shape of its thickness map (from its own profile widths); its endo- and
    epicardial half widths are scaled by one factor, found by bisection, until its myocardial volume is the
    reference's. A wall that moves inward then thickens, so the thickness, the cavity and the EF follow from
    the motion of the mid-wall. Any other reference returns the frames as they are."""
    known = [(cavityVolumeMl(r), i) for i, r in enumerate(results) if r is not None]
    if reference not in ("ed", "es") or len(known) < 2:
        return list(results)
    index = (max(known) if reference == "ed" else min(known))[1]
    target = myocardialVolumeMl(results[index])
    adjusted = []
    for i, result in enumerate(results):
        if result is None:
            adjusted.append(None)
            continue
        if i == index:
            adjusted.append(dict(result, myocardialVolume=reference, thicknessScale=1.0))
            continue
        grid, p = result["grid"], result["grid"].params
        mid = np.asarray(result["mid"], dtype=float)
        inner = np.maximum(mid - np.asarray(result["endo"], dtype=float), 0.0)
        outer = np.maximum(np.asarray(result["epi"], dtype=float) - mid, 0.0)

        def surfaces(scale):
            endo = _limitBasalNarrowing(grid, np.maximum(mid - scale * inner, 1.0), p["maxBaseNarrowing"])
            epi = _limitBasalNarrowing(grid, mid + scale * outer, p["maxBaseNarrowing"])
            return endo, epi

        def volume(scale):
            endo, epi = surfaces(scale)
            return _rayVolumes(result, endo, epi, 1)
        low, high = 0.2, 5.0
        if volume(low) >= target:
            scale = low
        elif volume(high) <= target:
            scale = high
        else:
            for _ in range(40):
                scale = 0.5 * (low + high)
                if volume(scale) < target:
                    low = scale
                else:
                    high = scale
            scale = 0.5 * (low + high)
        endo, epi = surfaces(scale)
        adjusted.append(dict(result, endo=endo, epi=epi, thickness=epi - endo,
                             myocardialVolume=reference, thicknessScale=float(scale)))
    return adjusted


# ---------------------------------------------------------------------------
# Surface fits of the gate frames: a smooth surface through the traced rays that draws the others
# ---------------------------------------------------------------------------
class _FitGeometry:
    """Spacing of the ray grid over the wall (mm) for a distance map: per row the spacing of its rays around
    the axis (hTheta) and the wall area per ray (area), between neighbouring rows their spacing along the
    wall (hRow), every row's arc position from the base (s), the distance of its rays from the axis (rho)
    and the typical radius of the cylinder rows (radius). The last row is the apex: one point."""

    def __init__(self, grid, distance):
        d = np.nan_to_num(np.asarray(distance, dtype=float), nan=1.0)
        radius = np.maximum(np.median(d, axis=1), 1.0)
        self.rho = radius.copy()
        self.rho[grid.nCyl:] = radius[grid.nCyl:] * np.sin(grid.phi)
        self.hTheta = self.rho * (2.0 * math.pi / grid.nTheta)
        self.hRow = np.empty(grid.nRows - 1)
        self.hRow[:grid.nCyl - 1] = grid.dz
        self.hRow[grid.nCyl - 1:] = 0.5 * (radius[grid.nCyl - 1:-1] + radius[grid.nCyl:]) * grid.dphi
        self.s = np.concatenate([[0.0], np.cumsum(self.hRow)])
        along = 0.5 * (np.concatenate([[0.0], self.hRow]) + np.concatenate([self.hRow, [0.0]]))
        self.area = self.hTheta * along
        self.area[-1] = math.pi * (0.5 * self.hRow[-1]) ** 2 / grid.nTheta     # the apex: a disc around the axis
        self.radius = float(np.median(radius[:grid.nCyl]))


def fitSurface(grid, values, weights, method="bspline", pull=1.0, distance=None):
    """A [rows, angles] map (a wall distance, its displacement, a wall thickness) fitted over the wall:
    the rays with weight (0..1, how far their value is trusted) set the fit, which draws the rays without
    (a defect) from them. method (GATE_FITS): "bspline" (_splineFit, a bending-penalized spline surface: a
    defect continues the curve of the walls around it), "mesh" (_meshFit, every ray tied to its neighbours:
    a defect is bridged straighter) or "legacy" (_legacyFit, the change between neighbouring rays limited).
    Rays that sit far off the first fit (FIT_OUTLIER robust misfits) lose their weight and the fit is made
    again. pull (0..1) is how far a traced ray is moved from its own value to the fit (1: the fit,
    0: its own value). distance: the wall's distance map that sets the spacing of the rays (values when
    None). The apex rows are made to agree (_poleSmooth)."""
    values = np.asarray(values, dtype=float)
    weights = np.where(np.isfinite(values), np.clip(np.nan_to_num(np.asarray(weights, dtype=float)), 0.0, 1.0), 0.0)
    values = np.nan_to_num(values)
    distance = values if distance is None else np.asarray(distance, dtype=float)
    if not np.any(weights > 1e-9):
        return np.full(values.shape, float(np.median(values)))
    geometry = _FitGeometry(grid, distance)
    solve = {"mesh": _meshFit, "legacy": _legacyFit}.get(method, _splineFit)
    fit = solve(grid, geometry, values, weights)
    misfit = values - fit
    seen = weights > 1e-9
    scale = max(1.4826 * float(np.median(np.abs(misfit[seen]))), 1.0)
    robust = weights * np.clip(1.0 - (misfit / (FIT_OUTLIER * scale)) ** 2, 0.0, 1.0) ** 2
    if np.any(robust > 1e-9):
        fit = solve(grid, geometry, values, robust)
        weights = np.where(robust > 1e-9, weights, 0.0)          # an outlier takes the fit
    pull = float(np.clip(pull, 0.0, 1.0))
    out = fit + (1.0 - pull) * weights * (values - fit)
    return _poleSmooth(grid, out, np.ones_like(out), 0.5, distance)


def _cubicBSpline(x):
    """Uniform cubic B-spline (support -2..2)."""
    x = np.abs(x)
    return np.where(x < 1.0, 2.0 / 3.0 - x ** 2 + 0.5 * x ** 3, np.where(x < 2.0, (2.0 - x) ** 3 / 6.0, 0.0))


def _splineFit(grid, geometry, values, weights):
    """Cubic B-spline surface over the wall (periodic around the axis, knots SPLINE_KNOT_MM apart along and
    around it) minimizing sum(area w (f - value)^2) + L^4 * integral(f_ss^2 + 2 f_st^2 + f_tt^2) dA (thin
    plate bending over the wall, s along it, t around the axis, both in mm; L = SPLINE_LENGTH_MM): wall
    features shorter than L are smoothed out, and where no ray is traced the surface keeps bending as the
    walls around it do. Near the apex the rays around the axis converge, so the same bending in mm ties
    them together. The penalty acts on the spline coefficients (Eilers and Marx's P-splines)."""
    knot = max(float(SPLINE_KNOT_MM), 1.0)
    span = float(geometry.s[-1])
    nr = max(int(math.ceil(span / knot)), 1)
    hr = span / nr
    centers = (np.arange(nr + 3) - 1.0) * hr                             # coefficient rows along the wall
    br = _cubicBSpline((geometry.s[:, None] - centers[None]) / hr)       # [rows, Kr]
    kt = max(6, int(round(2.0 * math.pi * geometry.radius / knot)))
    u = np.arange(grid.nTheta) * (kt / grid.nTheta)
    bt = _cubicBSpline(np.mod(u[:, None] - np.arange(kt)[None] + 0.5 * kt, kt) - 0.5 * kt)   # [angles, Kt]
    kr = len(centers)
    data = weights * geometry.area[:, None]
    normal = np.einsum("ik,il,imn->kmln", br, br, np.einsum("ij,jm,jn->imn", data, bt, bt)).reshape(kr * kt, kr * kt)
    rhs = (br.T @ (data * values) @ bt).ravel()
    # bending energy of the coefficients on the wall as a surface of revolution (rho: distance from the
    # axis, s along the wall, t = rho theta around it): f_ss, f_tt = f_thetatheta / rho^2 + rho' f_s / rho and
    # the twist f_st = f_stheta / rho - rho' f_theta / rho^2, in mm, each over its share of the wall area.
    # The rho' terms keep a wall that only moves off the axis free of bending at the apex too.
    dTheta = 2.0 * math.pi / kt
    rho = np.maximum(np.interp(centers, geometry.s, geometry.rho), 0.05 * geometry.radius)
    slope = np.interp(centers, geometry.s, np.gradient(geometry.rho, geometry.s))
    eye, rows = np.eye(kt), np.eye(kr)
    d1t = (np.roll(eye, -1, axis=1) - eye) / dTheta
    d2t = (np.roll(eye, -1, axis=1) - 2.0 * eye + np.roll(eye, 1, axis=1)) / dTheta ** 2
    d1r = np.diff(rows, axis=0) / hr
    d2r = np.diff(rows, n=2, axis=0) / hr ** 2
    dcr = np.gradient(rows, axis=0) / hr                                 # central difference along s
    mean = 0.5 * (rows[:-1] + rows[1:])
    rhoMid, slopeMid = mean @ rho, mean @ slope
    area = rho * dTheta * hr
    ss = np.kron(d2r, eye)
    tt = np.kron(np.diag(1.0 / rho ** 2), d2t) + np.kron(np.diag(slope / rho) @ dcr, eye)
    st = np.kron(np.diag(1.0 / rhoMid) @ d1r, d1t) - np.kron(np.diag(slopeMid / rhoMid ** 2) @ mean, d1t)
    penalty = (ss.T @ np.kron(np.diag(area[1:-1]), eye) @ ss + tt.T @ np.kron(np.diag(area), eye) @ tt
               + 2.0 * st.T @ np.kron(np.diag(mean @ area), eye) @ st)
    system = normal + float(SPLINE_LENGTH_MM) ** 4 * penalty
    system += 1e-9 * max(float(np.trace(normal)) / len(rhs), 1e-12) * np.eye(len(rhs))
    coefficients = np.linalg.solve(system, rhs).reshape(kr, kt)
    return br @ coefficients @ bt.T


def _meshFit(grid, geometry, values, weights, iterations=3000):
    """Deformable mesh over the rays: minimizes sum(area w (f - value)^2) + L^2 * integral(|grad f|^2) dA
    (a membrane, L = MESH_LENGTH_MM; every ray is tied to its neighbours around the axis and along it,
    base to apex, with the weights of their spacing in mm), solved by conjugate gradients. Wall features
    shorter than L are smoothed out; where no ray is traced the surface runs straight between the walls
    around it (a membrane does not keep their curve). The apex row is one node tied to the row before it."""
    n = grid.nRows - 1
    alpha = float(MESH_LENGTH_MM) ** 2
    ht = np.maximum(geometry.hTheta[:n], 1e-3)
    tieTheta = geometry.area[:n] / ht ** 2                               # [n]: ray to its neighbour around the axis
    tieRow = 0.5 * (geometry.hTheta[:-1] + geometry.hTheta[1:]) / geometry.hRow   # [n]: row i to row i + 1 (last: apex)
    data = weights * geometry.area[:, None]
    ridge = 1e-9 * float(np.mean(geometry.area))
    dataRays, dataApex = data[:n] + ridge, float(data[n].sum()) + ridge
    b = np.concatenate([(data[:n] * values[:n]).ravel(), [float((data[n] * values[n]).sum())]])

    def apply(x):
        f, apex = x[:-1].reshape(n, grid.nTheta), x[-1]
        out = dataRays * f + alpha * tieTheta[:, None] * (2.0 * f - np.roll(f, 1, axis=1) - np.roll(f, -1, axis=1))
        step = tieRow[:n - 1, None] * (f[1:] - f[:-1])
        out[:-1] -= alpha * step
        out[1:] += alpha * step
        edge = tieRow[n - 1] * (f[-1] - apex)
        out[-1] += alpha * edge
        return np.concatenate([out.ravel(), [dataApex * apex - alpha * float(edge.sum())]])

    diagonal = dataRays + alpha * (2.0 * tieTheta[:, None] + np.concatenate([[0.0], tieRow[:n - 1]])[:, None]
                                   + tieRow[:n, None])
    diagonal = np.concatenate([diagonal.ravel(), [dataApex + alpha * tieRow[n - 1] * grid.nTheta]])
    x = np.full(len(b), float((data * values).sum() / max(float(data.sum()), 1e-12)))
    r = b - apply(x)
    z = r / diagonal
    d = z.copy()
    rz = float(r @ z)
    tolerance = 1e-10 * max(float(b @ b), 1e-12)
    for _ in range(iterations):                                          # preconditioned conjugate gradients
        if float(r @ r) <= tolerance:
            break
        q = apply(d)
        step = rz / max(float(d @ q), 1e-300)
        x += step * d
        r -= step * q
        z = r / diagonal
        rzNew = float(r @ z)
        d = z + (rzNew / max(rz, 1e-300)) * d
        rz = rzNew
    return np.concatenate([x[:-1].reshape(n, grid.nTheta), np.full((1, grid.nTheta), x[-1])], axis=0)


def _legacyFit(grid, geometry, values, weights):
    """Brigger et al. (J Nucl Cardiol 1999;6:286) on the 3D ray grid: the change between neighbouring rays
    is limited to LEGACY_MAX_SLOPE mm per mm of wall, and within that limit the path closest to the traced
    rays' values is found by dynamic programming (_jumpLimitedPaths), ring by ring around the axis, then
    column by column along it from the base to the apex, then ring by ring again, so both limits hold.
    Where no ray is traced the path runs nearly straight. The apex row takes the mean of the row before it."""
    n = grid.nRows - 1
    slope = max(float(LEGACY_MAX_SLOPE), 1e-3)
    seen = weights > 1e-9
    lo, hi = np.percentile(values[seen], [1.0, 99.0])
    lo, hi = float(lo) - 2.0, float(hi) + 2.0
    step = max(0.25, (hi - lo) / 160.0)
    bins = lo + step * np.arange(int(math.ceil((hi - lo) / step)) + 1)
    aroundLimit = slope * geometry.hTheta[:n]
    rings = _jumpLimitedPaths(np.clip(values[:n], lo, hi), weights[:n], bins, aroundLimit, True)
    known = np.repeat(seen[:n].any(axis=1)[None].astype(float), grid.nTheta, axis=0)
    columns = _jumpLimitedPaths(rings.T, known, bins, slope * geometry.hRow[:n - 1], False).T
    out = _jumpLimitedPaths(columns, np.ones_like(columns), bins, aroundLimit, True)
    return np.concatenate([out, np.full((1, grid.nTheta), float(out[-1].mean()))], axis=0)


def _jumpLimitedPaths(values, weights, bins, limit, periodic, smooth=1e-2):
    """Dynamic programming over paths [paths, steps]: at every step a value from bins, at most limit apart
    from the step before (periodic: limit per path, the path closes on itself; else: limit per step),
    minimizing sum(w (value - x)^2) + smooth * sum(dx^2) (the small second term draws a straight path where
    w is 0). A closed path is found on three copies of it, the middle one kept."""
    paths, length = values.shape
    if periodic:
        values, weights = np.tile(values, (1, 3)), np.tile(weights, (1, 3))
    steps = values.shape[1]
    if periodic:
        limits = np.repeat(np.asarray(limit, dtype=float)[:, None], steps - 1, axis=1)
    else:
        limits = np.repeat(np.asarray(limit, dtype=float)[None, :], paths, axis=0)
    m = len(bins)
    binStep = float(bins[1] - bins[0]) if m > 1 else 1.0
    reach = int(min(np.floor(float(limits.max()) / binStep + 1e-9), m - 1)) if limits.size else 0
    unary = weights[..., None] * (bins[None, None, :] - values[..., None]) ** 2
    cost = unary[:, 0].copy()
    back = np.zeros((paths, steps, m), dtype=np.int16)
    for j in range(1, steps):
        best = np.full((paths, m), np.inf)
        arg = np.zeros((paths, m), dtype=np.int16)
        for shift in range(-reach, reach + 1):
            moved = np.full((paths, m), np.inf)
            if shift >= 0:
                moved[:, shift:] = cost[:, :m - shift]
            else:
                moved[:, :m + shift] = cost[:, -shift:]
            moved += smooth * (shift * binStep) ** 2
            moved[abs(shift) * binStep > limits[:, j - 1] + 1e-9] = np.inf
            better = moved < best
            best = np.where(better, moved, best)
            arg = np.where(better, shift, arg)
        back[:, j] = arg
        cost = best + unary[:, j]
    index = np.argmin(cost, axis=1)
    path = np.empty((paths, steps), dtype=int)
    path[:, -1] = index
    rows = np.arange(paths)
    for j in range(steps - 1, 0, -1):
        index = index - back[rows, j, index]
        path[:, j - 1] = index
    out = bins[path]
    return out[:, length:2 * length] if periodic else out


def endSystoleDiastole(volumes):
    """(ED index, ES index) of a cardiac cycle: the frames with the largest and the smallest LV cavity
    (frames without a volume are skipped); None when fewer than two frames have one."""
    known = [(float(v), i) for i, v in enumerate(volumes) if v is not None and np.isfinite(v)]
    if len(known) < 2:
        return None
    return max(known)[1], min(known)[1]


def thickeningMap(gatePolars, ed, es, smoothing=None, countSmoothing=None):
    """Count-based wall thickening polar map (partial volume effect: a thicker wall looks brighter),
    in % = (ES counts / ED counts - 1) x 100. The wall counts of every ray (polarMap of each frame)
    are fitted with the first harmonic over the cardiac cycle, so all frames count and the ED / ES
    values are less noisy than two single frames. The ED and ES count maps are smoothed first
    (countSmoothing, in map cells), since the ratio of two noisy maps is far noisier than either,
    and the % map is smoothed again (smoothing); both are count weighted, so a defect does not
    pull its neighbours down. This gives a coarser map than the wall motion, on purpose.
    Returns a polar map dict like polarMap (on the rows of the ED frame) with "values" in %.
    smoothing / countSmoothing default to THICKENING_SMOOTHING / THICKENING_COUNT_SMOOTHING."""
    smoothing = THICKENING_SMOOTHING if smoothing is None else smoothing
    countSmoothing = THICKENING_COUNT_SMOOTHING if countSmoothing is None else countSmoothing
    stack = np.array([np.asarray(p["values"], dtype=float) for p in gatePolars])     # [frames, rows, angles]
    mean, edCounts, esCounts = _firstHarmonic(stack, ed, es)
    weights = np.clip(mean / max(float(np.nanmax(mean)), 1e-9), 0.0, 1.0) ** 2
    edCounts = _smoothMap(np.nan_to_num(edCounts), weights + 1e-6, countSmoothing)
    esCounts = _smoothMap(np.nan_to_num(esCounts), weights + 1e-6, countSmoothing)
    values = 100.0 * (esCounts / np.maximum(edCounts, 1e-9) - 1.0)
    values = np.clip(_smoothMap(np.nan_to_num(values), weights + 1e-6, smoothing), -100.0, 200.0)
    reference = gatePolars[ed]
    return {"values": values, "rho": np.asarray(reference["rho"]).copy(), "theta": np.asarray(reference["theta"]).copy()}


def geometricThickeningMap(gateResults, ed, es, weights=None, smoothing=None):
    """Wall thickening polar map from the wall thickness itself, in % = (ES / ED thickness - 1) x 100: the
    distance from the endo- to the epicardium along every polar map ray (polarRays) of each frame, fitted with
    the first harmonic over the cycle (as thickeningMap). It shows what the frames' surfaces do: with every
    frame's own thickness (from its profile widths, which a blur near the wall's own width hardly changes) it
    reads low; with a constant myocardial volume (constantVolumeGates, as QGS) a wall that moves inward
    thickens by construction. weights (e.g. the mean wall counts of each ray) let rays through a defect follow
    their neighbours when the map is smoothed. Returns a polar map dict like polarMap (rho of the ED frame)
    with "values" in %. smoothing defaults to MOTION_SMOOTHING."""
    smoothing = MOTION_SMOOTHING if smoothing is None else smoothing
    thickness, rho = [], None
    for index, result in enumerate(gateResults):
        _, _, endo, epi, frameRho = polarRays(result)
        thickness.append(np.maximum(epi - endo, 0.0))
        if index == ed:
            rho = frameRho
    stack = np.array(thickness)                                       # [frames, rows, angles]
    _, edThickness, esThickness = _firstHarmonic(stack, ed, es)
    values = 100.0 * (esThickness / np.maximum(edThickness, 1e-6) - 1.0)
    if weights is None:
        weights = np.ones(values.shape)
    weights = np.clip(np.asarray(weights, dtype=float) / max(float(np.nanmax(weights)), 1e-9), 0.0, 1.0) ** 2
    values = np.clip(_smoothMap(np.nan_to_num(values), weights + 1e-6, smoothing), -100.0, 200.0)
    return {"values": values, "rho": np.asarray(rho).copy(), "theta": gateResults[ed]["grid"].theta.copy()}


THICKENING_SOURCES = (("Wall counts (partial volume)", "counts"),
                      ("Wall thickness, endo- to epicardium", "geometry"))


def phaseCoherenceMap(gatePolars, ed, es, smoothing=None):
    """How consistently the wall counts around every ray change over the cycle, in % (after Cooke et al.,
    J Nucl Med 1994: the first-harmonic phase of a pixel whose wall thickens falls in systole; in noise it is
    random). A ray is positive when the peak of its counts' first harmonic lies within a quarter cycle of
    ES. The share f of positive rays is pooled over a neighbourhood (Gaussian, smoothing map cells,
    PHASE_COHERENCE_SMOOTHING by default) and the coherence is |2 f - 1|: 0 where the phases are random
    (f about 50 %), 100 % where they all agree, thickening (f near 100 %) or thinning (f near 0). It shows
    where the thickening is more than noise; a background that changes the same way in every
    cycle (the walls around a scar brightening in systole) is coherent too.
    Returns a polar map dict like thickeningMap with "values" in % and "positive" (f, %)."""
    smoothing = PHASE_COHERENCE_SMOOTHING if smoothing is None else smoothing
    stack = np.array([np.asarray(p["values"], dtype=float) for p in gatePolars])     # [frames, rows, angles]
    n = stack.shape[0]
    phase = 2.0 * np.pi * np.arange(n) / n
    cosine = np.tensordot(np.cos(phase), stack, axes=1)
    sine = np.tensordot(np.sin(phase), stack, axes=1)
    peak = np.arctan2(sine, cosine)                       # phase of the first harmonic's maximum
    positive = (np.cos(peak - phase[es]) > 0.0).astype(float)
    share = _smoothMap(positive, np.ones_like(positive), smoothing)
    reference = gatePolars[ed]
    return {"values": 100.0 * np.abs(2.0 * share - 1.0), "positive": 100.0 * share,
            "rho": np.asarray(reference["rho"]).copy(), "theta": np.asarray(reference["theta"]).copy()}


def _firstHarmonic(stack, ed, es):
    """(mean, value at ED, value at ES) of [frames, ...] after a first-harmonic fit over the cycle
    (the frames as is when there are fewer than 3)."""
    n = stack.shape[0]
    mean = stack.mean(axis=0)
    if n < 3:
        return mean, stack[ed], stack[es]
    phase = 2.0 * np.pi * np.arange(n) / n
    cosine = 2.0 / n * np.tensordot(np.cos(phase), stack, axes=1)
    sine = 2.0 / n * np.tensordot(np.sin(phase), stack, axes=1)

    def fitted(index):
        return mean + cosine * np.cos(phase[index]) + sine * np.sin(phase[index])
    return mean, fitted(ed), fitted(es)


def myocardiumCenter(result):
    """Center of the myocardium of a wall fit (its volume between endo- and epicardium, uniform density),
    from the mid-wall points weighted by the wall volume each ray stands for."""
    grid = result["grid"]
    nCyl = grid.nCyl
    mid, thickness = np.asarray(result["mid"]), np.asarray(result["thickness"])
    points = grid.points(mid)
    area = np.empty(mid.shape)
    area[:nCyl] = mid[:nCyl] * grid.dz                                   # cylinder rows: r dz dtheta
    area[nCyl:] = mid[nCyl:] ** 2 * np.sin(grid.phi)[:, None] * grid.dphi   # cap rows: r^2 sin(phi) dphi dtheta
    keep = ~_beyondBase(grid, mid, result["valvePlane"])
    weight = np.where(keep, area * thickness, 0.0)
    return (points * weight[..., None]).reshape(-1, 3).sum(axis=0) / max(float(weight.sum()), 1e-9)


def frameShifts(gateResults, ed, reference="floating"):
    """[frames, 3] translation of the whole LV in every frame against ED, for a floating motion reference.
    "floating": across the long axis, the shift of the center of the endocardial rings (radial contraction
    is symmetric, so it does not move it); along the axis, the shift of the epicardial apex, which stays put
    over the cycle while the base descends toward it, so the valve descent is kept and only a heart that
    moves as a whole is taken out. "mass": the shift of the myocardium's center (myocardiumCenter); the base
    descent moves it toward the apex too, so part of the longitudinal motion is taken out with it. "base":
    across the axis as "floating", along it the shift of the valve plane on the axis (each frame's base
    center), so the wall motion is measured from the base: the valve descent is taken out and the
    longitudinal shortening shows as the apex moving toward the base."""
    if reference == "mass":
        centers = np.array([myocardiumCenter(result) for result in gateResults])
        return centers - centers[ed]
    centers, apexes = [], []
    for result in gateResults:
        grid = result["grid"]
        nCyl = grid.nCyl
        if reference == "base":
            apexes.append(valvePlane(result)[0])
        else:
            rows = slice(grid.nRows - min(MOTION_APEX_ROWS, grid.nCap), grid.nRows)
            along = np.asarray(result["epi"])[rows] * np.cos(grid.phi[-(rows.stop - rows.start):])[:, None]
            apexes.append(float(np.median(along)))
        endo = np.asarray(result["endo"])[:nCyl]
        rings = ~_beyondBase(grid, result["mid"], result["valvePlane"])[:nCyl].any(axis=1)
        rings = rings if rings.any() else np.ones(nCyl, dtype=bool)
        # a ring r(theta) = R + c . n(theta) around the axis has its center c at twice the mean of r n
        offAxis = endo[rings][..., None] * grid.radial(grid.theta)[None]
        centers.append(2.0 * offAxis.mean(axis=1).mean(axis=0))
    u = gateResults[ed]["grid"].u
    centers = np.array(centers)
    apexes = np.array(apexes)
    return (centers - centers[ed]) + (apexes - apexes[ed])[:, None] * u


def shiftedResult(result, shift):
    """The wall of a frame seen from its LV center moved by shift (world mm): the same endo-, mid- and
    epicardium cut by the rays of a grid centered there (a few fixed-point steps from the distances along the
    original rays) and the same valve plane, so its points are compared with other frames' at the same place
    relative to the moved center."""
    grid = result["grid"]
    shift = np.asarray(shift, dtype=float)
    moved = WallGrid(grid.center + shift, grid.u, grid.w, grid.radius, grid.params, grid.length)
    origins, directions = moved.rays()
    out = dict(result, grid=moved)
    for key in ("endo", "mid", "epi"):
        surface = np.asarray(result[key], dtype=float)
        t = surface.copy()
        for _ in range(8):
            row, col, dist, _ = grid.toGrid((origins + t[..., None] * directions).reshape(-1, 3))
            t = np.maximum(t + (_bilinear(surface, row, col) - dist).reshape(t.shape), 0.5)
        out[key] = t
    out["thickness"] = out["epi"] - out["endo"]
    a, gw, gs, b2, c2 = _rim(result["valvePlane"])
    out["valvePlane"] = (a - float(shift @ grid.u) + gw * float(shift @ grid.w) + gs * float(shift @ grid.s),
                         gw, gs, b2, c2)
    return out


def motionMap(gateResults, ed, es, weights=None, smoothing=None, reference="fixed"):
    """Wall motion polar map (QGS-like endocardial excursion), in mm: how far the endocardium of every
    polar map ray moves from ED to ES. The rays of each frame run from its own valve plane to the apex
    (polarRays), so a basal point follows the valve descent and the longitudinal shortening counts too.
    The distance is signed: positive when the ES point lies inside the ED endocardium (toward the cavity),
    negative for outward (dyskinetic) motion. The endocardial points are fitted with the first harmonic
    over the cycle, like the thickening; weights (e.g. the mean wall counts of each ray) let rays through
    a defect follow their neighbours when the map is smoothed. The rays are those of the static fit
    (fixed LV center and axis), so with reference "fixed" a translation of the whole heart is not corrected;
    with "floating" each frame's translation against ED (frameShifts: its LV center across the axis, its
    epicardial apex along it), with "mass" that of its myocardium's center, with "base" that of its LV center
    and base center (so the valve descent goes too and the apex moves toward the base), is taken out first.
    Returns a polar map dict like polarMap (rho of the ED frame) with "values" in mm.
    smoothing defaults to MOTION_SMOOTHING."""
    smoothing = MOTION_SMOOTHING if smoothing is None else smoothing
    gateResults, shifts = _referencedResults(gateResults, ed, reference)
    points, rhos, directions = [], [], []
    for result, shift in zip(gateResults, shifts):
        origins, rays, endo, _, rho = polarRays(result)
        points.append(origins + endo[..., None] * rays - shift)      # [rows, angles, 3]
        rhos.append(rho)
        directions.append(rays)
    stack = np.array(points)                                          # [frames, rows, angles, 3]
    _, edPoints, esPoints = _firstHarmonic(stack, ed, es)
    displacement = esPoints - edPoints
    distance = np.linalg.norm(displacement, axis=-1)
    inward = -np.sum(displacement * directions[ed], axis=-1)
    values = np.where(inward >= 0.0, distance, -distance)
    if weights is None:
        weights = np.ones(values.shape)
    weights = np.clip(np.asarray(weights, dtype=float) / max(float(np.nanmax(weights)), 1e-9), 0.0, 1.0) ** 2
    values = np.clip(_smoothMap(np.nan_to_num(values), weights + 1e-6, smoothing), -30.0, 30.0)
    return {"values": values, "rho": np.asarray(rhos[ed]).copy(), "theta": gateResults[ed]["grid"].theta.copy()}


def _referencedResults(gateResults, ed, reference):
    """(frames, [frames, 3] shifts) for a motion reference (MOTION_REFERENCES): the frames as they are with
    "fixed", else every frame seen from its own moved center (frameShifts, shiftedResult)."""
    if reference not in ("floating", "mass", "base"):
        return list(gateResults), np.zeros((len(gateResults), 3))
    shifts = frameShifts(gateResults, ed, reference)
    return [shiftedResult(result, shift) for result, shift in zip(gateResults, shifts)], shifts


def _wedgeVolumes(result):
    """[rows, angles] volume of the LV cavity each polar map ray stands for, up to a factor fixed per row
    (polarRays): on the cylinder a wedge from the long axis out to the endocardium, between the valve plane
    and the LV center (the rows run from that angle's valve plane, so a wedge also shrinks as the base
    descends), endo^2 x that length; on the apical cap a cone from the LV center to the endocardium, endo^3."""
    grid = result["grid"]
    nCyl = grid.nCyl
    _, _, endo, _, _ = polarRays(result)
    length = -_polarBaseZ(result)                                      # [angles] valve plane to LV center, mm
    volume = np.empty(endo.shape)
    volume[:nCyl] = endo[:nCyl] ** 2 * length[None, :]
    volume[nCyl:] = endo[nCyl:] ** 3
    return volume


def _baseConeVolumes(result):
    """[rows, angles] volume of the LV cavity each polar map ray stands for, as cones from the frame's base
    center (where the long axis meets its valve plane) to its patch of endocardium: |(p - b) . dA| / 3, the
    patch's vector area dA from the neighbouring endocardial points (polarRays), up to one constant factor.
    The base center lies in the valve plane, so the valve opening adds nothing and the cones add up to the
    cavity; as it moves down with the valve, the descent of the base shortens the apical cones."""
    grid = result["grid"]
    origins, directions, endo, _, _ = polarRays(result)
    points = origins + endo[..., None] * directions                       # [rows, angles, 3]
    base = grid.center + valvePlane(result)[0] * grid.u
    alongRows = np.gradient(points, axis=0)
    aroundAxis = 0.5 * (np.roll(points, -1, axis=1) - np.roll(points, 1, axis=1))
    area = np.cross(alongRows, aroundAxis)
    return np.abs(np.sum((points - base) * area, axis=-1)) / 3.0


def regionalEFMap(gateResults, ed, es, weights=None, smoothing=None, reference="fixed", cut="axis"):
    """Regional ejection fraction polar map, in % = (1 - ES / ED volume) x 100 of the cavity under every polar
    map ray (_wedgeVolumes: wedges from the long axis on the cylinder, cones from the LV center on the apical
    cap; the cavity is cut into them like the bullseye, so they add up to the LV cavity). A percentage, so a
    narrow apex that empties well reads high even though its wall moves only a few mm. The wedge volumes are
    fitted with the first harmonic over the cycle; with reference "floating" or "mass" every frame is seen
    from its own moved center first (as motionMap), which matters here: a wedge grows or shrinks with a
    heart that moves as a whole. weights (e.g. the mean wall counts of each ray) let rays through a defect
    follow their neighbours when the map is smoothed. Returns a polar map dict like polarMap (rho of the ED
    frame) with "values" in %. cut "base" (REGIONAL_EF_CUTS) cuts the cavity into cones from every frame's
    own base center instead (_baseConeVolumes): the descent of the base then counts toward the apical and mid
    cones rather than along the rings. smoothing defaults to REGIONAL_EF_SMOOTHING."""
    smoothing = REGIONAL_EF_SMOOTHING if smoothing is None else smoothing
    gateResults, _ = _referencedResults(gateResults, ed, reference)
    volumes = _baseConeVolumes if cut == "base" else _wedgeVolumes
    stack = np.array([volumes(result) for result in gateResults])            # [frames, rows, angles]
    _, edVolume, esVolume = _firstHarmonic(stack, ed, es)
    values = 100.0 * (1.0 - esVolume / np.maximum(edVolume, 1e-9))
    if weights is None:
        weights = np.ones(values.shape)
    weights = np.clip(np.asarray(weights, dtype=float) / max(float(np.nanmax(weights)), 1e-9), 0.0, 1.0) ** 2
    values = np.clip(_smoothMap(np.nan_to_num(values), weights + 1e-6, smoothing), -100.0, 100.0)
    _, _, _, _, rho = polarRays(gateResults[ed])
    return {"values": values, "rho": np.asarray(rho).copy(), "theta": gateResults[ed]["grid"].theta.copy()}


def _cavityFill(profiles, t, mid, peak):
    """Lowest counts between the axis and the mid-wall along each ray, as a fraction of the wall peak:
    low inside the LV (dark cavity), near 1 where the ray runs through the atria or the valve ring."""
    inside = t[None, None, :] <= mid[..., None]
    lowest = np.where(inside, profiles, np.inf).min(axis=2)
    return np.clip(lowest / np.maximum(peak, 1e-9), 0.0, 1.0)


def _fitValvePlane(grid, mid, peak, p, fill=None):
    """Base rim (a, gw, gs, b2, c2) in the cardiac frame (x along w, y along s, angle theta from w toward s):
    z = a + gw * x + gs * y + b2 * cos 2 theta + c2 * sin 2 theta, a tilted plane that may rise and fall twice
    around the LV (valveShape "saddle", _planeThroughEnds), fitted through the end of the wall in every column. Along a column the wall ends where its counts fall
    below a threshold, or where the cavity fills up (the rays leave the LV for the atria, whose counts
    can pass the count threshold in small hearts).

    baseMethod "fraction": the threshold is baseFraction of the column's wall counts; columns with little
    uptake (an infarct) have unreliable ends and count less in the fit. A defect at the base only (an
    inferobasal one) leaves the column's wall counts high, so its wall ends early, at the defect.
    baseMethod "adaptive" (default): that is only the preliminary segment (basePrelimFraction). From it
    the blood pool (the cavity's lowest counts at the mid-ventricle) and every column's basal wall counts
    (the wall within BASE_BAND_MM of the preliminary plane) are taken, and the wall ends where its counts
    fall baseEdgeFraction of the way from those basal wall counts down to the blood pool: the half way
    point of the step at the valve plane, whatever the wall's uptake, so a basal defect is followed to the
    base. Each column is searched from BASE_SEARCH_MM apical of the preliminary plane toward the base
    (a defect at the mid-ventricle does not end it), and counts in the fit by the contrast of its basal
    wall against the blood pool (a scar as dim as the blood pool has no edge to find).
    Returns ((a, gw, gs, b2, c2), end points [K, 3] frame)."""
    nCyl = grid.nCyl
    cyl = peak[:nCyl]
    if fill is not None:
        # the column's own cavity contrast at the mid-ventricle, and how far it may fill before the wall ends
        midRows = (grid.z >= -12.0) & (grid.z <= -2.0)
        if not np.any(midRows):
            midRows = np.ones(nCyl, dtype=bool)
        midFill = np.median(fill[:nCyl][midRows], axis=0)
        fillLimit = midFill + 0.5 * (1.0 - midFill)
        filled = fill[:nCyl] > fillLimit[None, :]
    else:
        filled = np.zeros(cyl.shape, dtype=bool)
    method = p.get("baseMethod", "adaptive")
    adaptive = method in ("adaptive", "global")

    reference = np.percentile(peak, 85, axis=0)                  # each column's wall counts
    reference = _circularMedian(reference[None, :], _angleWindow(grid, 5))[0]
    lateral = np.percentile(cyl, 85, axis=0)                     # the same on the side wall only
    lateral = _circularMedian(lateral[None, :], _angleWindow(grid, 5))[0]
    columnWeight = np.clip(lateral / max(float(np.percentile(lateral, 90)), 1e-9), 0.0, 1.0) ** 2
    reference = np.maximum(reference, 0.4 * float(np.percentile(peak, 85)))
    fraction = p.get("basePrelimFraction", 0.5) if adaptive else p["baseFraction"]
    ends, endWeights = _wallEnds(grid, mid, cyl, fraction * reference, filled, columnWeight)
    plane = _planeThroughEnds(grid, ends, endWeights, p)
    if not adaptive or len(ends) < max(6, grid.nTheta // 4):
        return plane, ends

    # blood pool: the lowest counts in the cavity at the mid-ventricle (fill is relative to the wall peak)
    pool = 0.0
    if fill is not None:
        pool = float(np.median((fill[:nCyl] * cyl)[midRows]))
    inside = ~_beyondBase(grid, mid, plane)
    wall = float(np.mean(peak[inside]))                           # mean counts of the preliminary segment
    pool = min(pool, 0.5 * wall)
    contrast = max(wall - pool, 1e-9)
    # every column's basal wall: within BASE_BAND_MM of the preliminary plane, on the LV side
    a, slope, saddle = _rimColumns(grid.theta, plane)
    above = grid.z[:, None] - (a + slope[None, :] * mid[:nCyl] + saddle[None, :])   # mm on the apical side of the rim
    band = (above >= 0.0) & (above <= BASE_BAND_MM)
    basal = np.array([np.median(cyl[band[:, c], c]) if np.any(band[:, c]) else reference[c]
                      for c in range(grid.nTheta)])
    basal = _circularMedian(basal[None, :], _angleWindow(grid, 3))[0]
    basal = np.clip(basal, pool + BASE_MIN_CONTRAST * contrast, None)
    # "global": one level for every column, the mean wall counts of the whole preliminary myocardium (steadier in
    # noisy frames, but a basal-only defect ends its column early, as "fraction" does); "adaptive": each column's
    # own basal wall
    level = np.full(grid.nTheta, wall) if method == "global" else basal
    threshold = pool + p.get("baseEdgeFraction", 0.5) * (level - pool)
    weight = np.clip((basal - pool) / (0.5 * contrast), 0.0, 1.0) ** 2
    start = np.array([np.searchsorted(grid.z, a + slope[c] * np.interp(a, grid.z, mid[:nCyl, c]) + saddle[c]
                                      + BASE_SEARCH_MM)
                      for c in range(grid.nTheta)])
    ends, endWeights = _wallEnds(grid, mid, cyl, threshold, filled, weight, start)
    if len(ends) < max(6, grid.nTheta // 4):
        return plane, ends
    return _planeThroughEnds(grid, ends, endWeights, p), ends


def _wallEnds(grid, mid, cyl, threshold, filled, columnWeight, start=None):
    """End of the wall in every column of the cylinder: going from the start row (default the LV center)
    toward the base, the first of two consecutive rows below the column's threshold or with the cavity
    filled, after a row on the wall. Returns (end points [K, 3] frame, their weights [K])."""
    nCyl = grid.nCyl
    ends = []
    endWeights = []
    for col in range(grid.nTheta):
        values = cyl[:, col]
        below = (values < threshold[col]) | filled[:, col]
        first = nCyl - 2 if start is None else int(np.clip(start[col], 1, nCyl - 2))
        onWall = start is None
        for row in range(first, 0, -1):
            if not below[row]:
                onWall = True
            elif onWall and below[row - 1]:
                v0, v1 = values[row], values[row + 1]
                f = np.clip((threshold[col] - v0) / (v1 - v0), 0.0, 1.0) if v1 > v0 and not filled[row, col] else 0.5
                z = grid.z[row] + f * grid.dz
                r = mid[row, col] + f * (mid[row + 1, col] - mid[row, col])
                ends.append((r * math.cos(grid.theta[col]), r * math.sin(grid.theta[col]), z))
                endWeights.append(columnWeight[col])
                break
    return np.array(ends).reshape(-1, 3), np.maximum(np.array(endWeights), 1e-3)


def _planeThroughEnds(grid, ends, endWeights, p):
    """Base rim (a, gw, gs, b2, c2) through the wall's end points, z = a + gw * x + gs * y + b2 * cos 2 theta +
    c2 * sin 2 theta: robust least squares (outlier columns out), tilt limited to maxValveTilt. valveShape
    "saddle" fits the saddle (b2, c2) too, held toward flat (SADDLE_RIDGE) and at most SADDLE_MAX_MM, when the
    ends go around at least half the LV; "plane" keeps it flat. Too few ends: the wall runs to the edge of the
    sampled region."""
    zMin, zMax = float(grid.z[0]), -2.0 * grid.dz
    if len(ends) < max(6, grid.nTheta // 4):
        a = zMin if len(ends) == 0 else float(np.median(ends[:, 2]))
        return (float(np.clip(a, zMin, zMax)), 0.0, 0.0, 0.0, 0.0)
    theta = np.arctan2(ends[:, 1], ends[:, 0])
    harmonics = np.column_stack([np.cos(2.0 * theta), np.sin(2.0 * theta)])
    saddle = p.get("valveShape", "saddle") == "saddle" and len(ends) >= grid.nTheta // 2
    plane = np.column_stack([np.ones(len(ends)), ends[:, 0], ends[:, 1]])
    # the plane first, its outlier columns out; then the saddle on the columns the plane kept (a run of columns
    # ending early at a basal defect would otherwise be taken up by tilt and saddle together, a one-sided bump)
    weights = endWeights.copy()
    for design, ridged, rounds in ((plane, False, 5), (np.column_stack([plane, harmonics]), True, 3))[:1 + saddle]:
        for _ in range(rounds):                                  # iteratively reweighted (outlier columns out)
            sw = np.sqrt(weights)
            lhs, rhs = design * sw[:, None], ends[:, 2] * sw
            if ridged:   # the saddle costs as much as SADDLE_RIDGE of the ends missing the rim by as many mm
                ridge = math.sqrt(SADDLE_RIDGE * float(weights.sum()))
                lhs = np.vstack([lhs, [[0.0, 0.0, 0.0, ridge, 0.0], [0.0, 0.0, 0.0, 0.0, ridge]]])
                rhs = np.concatenate([rhs, [0.0, 0.0]])
            coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            residual = ends[:, 2] - design @ coef
            scale = max(1.4826 * float(np.median(np.abs(residual))), 2.0)
            weights = endWeights * np.clip(1.0 - (residual / (3.0 * scale)) ** 2, 0.0, 1.0) ** 2
    a, gw, gs, b2, c2 = (float(v) for v in np.append(coef, [0.0, 0.0])[:5])
    amplitude = math.hypot(b2, c2)
    slope = math.hypot(gw, gs)
    maxSlope = math.tan(math.radians(p["maxValveTilt"]))
    if amplitude > SADDLE_MAX_MM or slope > maxSlope:
        b2, c2 = (v * min(1.0, SADDLE_MAX_MM / max(amplitude, 1e-9)) for v in (b2, c2))
        gw, gs = (v * min(1.0, maxSlope / max(slope, 1e-9)) for v in (gw, gs))
        rest = ends[:, 2] - gw * ends[:, 0] - gs * ends[:, 1] - harmonics @ np.array([b2, c2])
        a = float(np.average(rest, weights=weights + 1e-9))
    return (float(np.clip(a, zMin, zMax)), gw, gs, b2, c2)


def _rim(plane):
    """(a, gw, gs, b2, c2) of a base rim; the valve plane (a, gw, gs) of older results has no saddle."""
    plane = tuple(float(v) for v in plane)
    return plane + (0.0,) * (5 - len(plane))


def _rimColumns(theta, plane):
    """The base rim at angles theta around the axis: z = a + slope * r + saddle at a distance r from the axis.
    Returns (a, slope, saddle)."""
    a, gw, gs, b2, c2 = _rim(plane)
    theta = np.asarray(theta, dtype=float)
    return a, gw * np.cos(theta) + gs * np.sin(theta), b2 * np.cos(2.0 * theta) + c2 * np.sin(2.0 * theta)


def _beyondBase(grid, mid, plane):
    """Cylinder rows of each column on the atrial side of the base rim, [rows, angles] bool."""
    a, slope, saddle = _rimColumns(grid.theta, plane)
    beyond = np.zeros(mid.shape, dtype=bool)
    beyond[:grid.nCyl] = grid.z[:, None] < a + slope[None, :] * mid[:grid.nCyl] + saddle[None, :]
    return beyond


def _holdBeyondBase(values, beyond):
    """Past the valve plane there is no wall to follow: each column keeps the value of its last row
    inside, so the surfaces run on straight (the valve plane can still be moved toward the atria)."""
    values = values.copy()
    for col in range(values.shape[1]):
        inside = np.nonzero(~beyond[:, col])[0]
        if len(inside) and inside[0] > 0:
            values[:inside[0], col] = values[inside[0], col]
    return values


def _limitBasalNarrowing(grid, distance, maxDegrees):
    """From the LV center toward the base, a cylinder row may come at most tan(maxDegrees) * row
    spacing closer to the axis than the row above it. A base narrower than the mid-ventricle (or an
    aneurysm above it) is kept, but a wall that folds in toward the axis (the counts of the atria
    or the valve ring pulling the rays in) can no longer close the endo- or epicardium into a cap."""
    distance = distance.copy()
    step = math.tan(math.radians(float(maxDegrees))) * grid.dz
    for row in range(grid.nCyl - 2, -1, -1):
        distance[row] = np.maximum(distance[row], distance[row + 1] - step)
    return distance


def wallReach(result, point=None):
    """Largest distance (mm) of the epicardium from point (default: the grid center), base rim included."""
    grid = result["grid"]
    point = grid.center if point is None else np.asarray(point, dtype=float)
    return float(np.max(np.linalg.norm(surfaceGrid(result, "epi") - point, axis=-1)))


def valvePlane(result):
    """(a, gw, gs, b2, c2) of the base rim z = a + gw * x + gs * y + b2 * cos 2 theta + c2 * sin 2 theta
    (_planeThroughEnds) after the user's offset and tilt."""
    grid = result["grid"]
    p = grid.params
    a, gw, gs, b2, c2 = _rim(result["valvePlane"])
    gw = math.tan(np.clip(math.atan(gw) + math.radians(p["valveTiltAnterior"]), -1.2, 1.2))
    gs = math.tan(np.clip(math.atan(gs) + math.radians(p["valveTiltSeptal"]), -1.2, 1.2))
    a = float(np.clip(a + p["valveOffset"], grid.z[0], -grid.dz))
    scale = _tiltScale(grid, result.get("epi"), a, gw, gs, b2, c2)
    return (a, gw * scale, gs * scale, b2 * scale, c2 * scale)


def _tiltScale(grid, epi, a, gw, gs, b2=0.0, c2=0.0):
    """Factor (0..1) on the valve tilt and saddle so the rim meets the epicardium of every column inside the
    sampled cylinder: below the LV center and above the base end. A steep plane in a small or basally
    centered LV would otherwise reach past the LV center on one side and past the sampled region on
    the other; both are clipped, the wall is cut flat there and the endo- and epicardium end at
    different places (a notch or hole at the base)."""
    if epi is None or not any((gw, gs, b2, c2)):
        return 1.0
    nCyl = grid.nCyl
    radius = np.array([np.interp(a, grid.z, epi[:nCyl, c]) for c in range(grid.nTheta)])
    _, slope, saddle = _rimColumns(grid.theta, (a, gw, gs, b2, c2))
    rise = slope * radius + saddle                                            # rim z - a at the epicardium
    top, bottom = -2.0 * grid.dz - a, float(grid.z[0]) + grid.dz - a
    scale = 1.0
    if rise.max() > 1e-9:
        scale = min(scale, max(top, 0.0) / float(rise.max()))
    if rise.min() < -1e-9:
        scale = min(scale, min(bottom, 0.0) / float(rise.min()))
    return float(np.clip(scale, 0.0, 1.0))


def valvePlaneZ(result):
    """Valve plane position on the long axis after the user's offset (mm from the LV center)."""
    return valvePlane(result)[0]


def valveTilt(result):
    """Angle between the valve plane and the short-axis plane, degrees."""
    gw, gs = valvePlane(result)[1:3]
    return math.degrees(math.atan(math.hypot(gw, gs)))


def planeZAt(result, x, y):
    """Base rim z at frame coordinates (x along w, y along s), kept between the base end of the
    sampled region and just below the LV center. The wall itself is cut at rimZAt."""
    grid = result["grid"]
    plane = valvePlane(result)
    _, _, saddle = _rimColumns(np.arctan2(y, x), plane)
    return np.clip(plane[0] + plane[1] * x + plane[2] * y + saddle, grid.z[0], -grid.dz)


def rimZAt(result, theta, zBase=None):
    """Height (z, mm) at which the wall is cut at angles theta around the axis: where the mid-wall of each
    column meets the base rim (_polarBaseZ), interpolated between the columns. The whole column is cut at that
    one height, endo- and epicardium alike: the rim crossing the epicardium elsewhere (a basal wall flaring
    out under a tilted rim) leaves no notch or loose piece."""
    grid = result["grid"]
    zBase = _polarBaseZ(result) if zBase is None else zBase
    return np.interp(np.mod(theta, 2.0 * math.pi), np.append(grid.theta, 2.0 * math.pi), np.append(zBase, zBase[0]))


# ---------------------------------------------------------------------------
# Labels and surfaces
# ---------------------------------------------------------------------------
LABEL_MYOCARDIUM = 1
LABEL_CAVITY = 2


def classify(result, points):
    """Label of world points: 0 outside, LABEL_MYOCARDIUM, LABEL_CAVITY (closed at the base rim, rimZAt)."""
    grid = result["grid"]
    row, col, dist, z = grid.toGrid(points)
    endo = _bilinear(result["endo"], row, col)
    epi = _bilinear(result["epi"], row, col)
    labels = np.zeros(len(dist), dtype=np.uint8)
    inside = (z >= rimZAt(result, col * (2.0 * math.pi / grid.nTheta))) & (row <= grid.nRows - 1)
    labels[inside & (dist <= epi)] = LABEL_MYOCARDIUM
    labels[inside & (dist < endo)] = LABEL_CAVITY
    return labels


def surfaceGrid(result, which):
    """World points [rows, angles, 3] of one surface ("endo", "mid", "epi") from the base rim to
    the apex. In every column the cylinder rows are spread evenly from the height where the mid-wall
    meets the base rim (rimZAt, the same for every surface: the endo- and epicardium end together) up to
    the LV center, so the first row lies on the rim."""
    grid = result["grid"]
    distance = result[which]
    nCyl = grid.nCyl
    cylinder = distance[:nCyl]
    zBase = _polarBaseZ(result)
    fractions = np.linspace(0.0, 1.0, nCyl)[:, None]
    zRows = zBase[None, :] * (1.0 - fractions)                         # [nCyl, angles], 0 at the center
    rRows = np.column_stack([np.interp(zRows[:, c], grid.z, cylinder[:, c]) for c in range(grid.nTheta)])
    radial = grid.radial(grid.theta)
    cylinderPoints = grid.center + zRows[..., None] * grid.u + rRows[..., None] * radial[None]
    origins, directions = grid.rays()
    capPoints = origins[nCyl:] + distance[nCyl:, :, None] * directions[nCyl:]
    return np.concatenate([cylinderPoints, capPoints], axis=0)


def surfaceTriangles(rows, angles, closeBase=True, baseCenterIndex=None, apexCenterIndex=None):
    """Triangle indices of a [rows, angles] grid surface, wrapped around the axis. The last row
    (apex, all points nearly equal) is closed with a fan to apexCenterIndex, and with closeBase the
    first row is closed with a fan to baseCenterIndex (the valve plane)."""
    triangles = []
    for r in range(rows - 1):
        for a in range(angles):
            b = (a + 1) % angles
            p00, p01 = r * angles + a, r * angles + b
            p10, p11 = (r + 1) * angles + a, (r + 1) * angles + b
            triangles.append((p00, p10, p11))
            triangles.append((p00, p11, p01))
    if closeBase and baseCenterIndex is not None:
        for a in range(angles):
            triangles.append((baseCenterIndex, (a + 1) % angles, a))
    if apexCenterIndex is not None:
        last = (rows - 1) * angles
        for a in range(angles):
            triangles.append((apexCenterIndex, last + a, last + (a + 1) % angles))
    return np.array(triangles, dtype=np.int64)


# ---------------------------------------------------------------------------
# Polar map (bullseye)
# ---------------------------------------------------------------------------
# Ring edges of the 17-segment model as a fraction of the bullseye radius (apex at 0, valve plane
# at 1): apex cap (17), apical (13-16), mid (7-12), basal (1-6)
POLAR_RING_EDGES = (0.0, 0.2, 0.47, 0.73, 1.0)
SEGMENT_NAMES = (
    "basal anterior", "basal anteroseptal", "basal inferoseptal", "basal inferior", "basal inferolateral",
    "basal anterolateral", "mid anterior", "mid anteroseptal", "mid inferoseptal", "mid inferior",
    "mid inferolateral", "mid anterolateral", "apical anterior", "apical septal", "apical inferior",
    "apical lateral", "apex")


def polarRays(result):
    """Rays through the wall from the valve plane to the apex, [rows, angles]: origins, directions,
    endo and epi distances along them, and the bullseye radius of each row (arc length along the
    mid-wall from the apex, 0 at the apex, 1 at the valve plane). Row 0 is in the valve plane."""
    grid = result["grid"]
    nCyl = grid.nCyl
    zBase = _polarBaseZ(result)
    zRows = zBase[None, :] * (1.0 - np.linspace(0.0, 1.0, nCyl)[:, None])     # [nCyl, angles]

    def alongRows(values):
        cylinder = np.column_stack([np.interp(zRows[:, c], grid.z, values[:nCyl, c]) for c in range(grid.nTheta)])
        return np.concatenate([cylinder, values[nCyl:]], axis=0)

    radial = grid.radial(grid.theta)
    origins, directions = grid.rays()
    origins = origins.copy()
    directions = directions.copy()
    origins[:nCyl] = grid.center + zRows[..., None] * grid.u
    directions[:nCyl] = radial[None]
    endo, epi, midRows = alongRows(result["endo"]), alongRows(result["epi"]), alongRows(result["mid"])
    points = origins + midRows[..., None] * directions
    steps = np.linalg.norm(np.diff(points, axis=0), axis=2).mean(axis=1)    # [rows - 1]
    fromApex = np.concatenate([np.cumsum(steps[::-1])[::-1], [0.0]])
    rho = fromApex / max(fromApex[0], 1e-9)
    return origins, directions, endo, epi, rho


DIAMETER_LEVEL = 0.8       # lvDiameterMm: bullseye radius of the level (0 apex .. 1 valve plane), basal-mid like echo
DIAMETER_ANGLES = (60.0, 240.0)   # lvDiameterMm: anteroseptal and inferolateral (the parasternal long-axis line)


def lvDiameterMm(result, level=None, angles=None):
    """LV internal diameter (mm) of a wall fit, as echo measures LVEDD / LVESD in the parasternal long axis:
    from the endocardium of the anteroseptal to that of the inferolateral wall (DIAMETER_ANGLES, deg from
    anterior toward the septum, as the bullseye), across the long axis, at a level (DIAMETER_LEVEL) on the
    polar map rays (polarRays: rho 1 at the valve plane, 0 at the apex), where echo takes it near the mitral
    leaflet tips. The distance along the long axis between the two points (a tilted valve plane) is left
    out. None when the fit has no endocardium."""
    level = DIAMETER_LEVEL if level is None else float(level)
    angles = DIAMETER_ANGLES if angles is None else angles
    if result is None or result.get("endo") is None:
        return None
    grid = result["grid"]
    origins, directions, endo, _, rho = polarRays(result)
    points = origins + endo[..., None] * directions                     # [rows, angles, 3]
    order = np.argsort(rho)
    row = float(np.interp(level, np.asarray(rho)[order], np.arange(len(rho), dtype=float)[order]))
    r0 = int(np.clip(math.floor(row), 0, len(rho) - 1))
    r1 = min(r0 + 1, len(rho) - 1)
    fr = row - r0
    ends = []
    for degrees in angles:
        col = np.mod(math.radians(degrees), 2.0 * math.pi) / (2.0 * math.pi / grid.nTheta)
        c0 = int(math.floor(col)) % grid.nTheta
        c1 = (c0 + 1) % grid.nTheta
        fc = col - math.floor(col)
        rowPoints = (1.0 - fr) * points[r0] + fr * points[r1]            # [angles, 3]
        ends.append((1.0 - fc) * rowPoints[c0] + fc * rowPoints[c1])
    d = ends[0] - ends[1]
    d = d - np.dot(d, grid.u) * grid.u
    return float(np.linalg.norm(d))


def _polarBaseZ(result):
    """Where the mid-wall of every column (angle) meets the base rim, along the axis (mm)."""
    grid = result["grid"]
    a, slope, saddle = _rimColumns(grid.theta, valvePlane(result))
    mid = result["mid"][:grid.nCyl]
    zBase = np.full(grid.nTheta, a)
    for _ in range(6):
        radius = np.array([np.interp(zBase[c], grid.z, mid[:, c]) for c in range(grid.nTheta)])
        zBase = np.clip(a + slope * radius + saddle, grid.z[0], -grid.dz)
    return zBase


def polarAtPoints(polar, result, points):
    """Values of a polar map (polarMap, motionMap, ...) at world points of the wall of result: each point
    at the bullseye radius and angle of the polar map ray through it (polarRays: on the cylinder the
    rows run from the valve plane at that angle to the LV center, on the apical cap they are the fit's)."""
    grid = result["grid"]
    row, col, _, z = grid.toGrid(points)
    angle = col * (2.0 * math.pi / grid.nTheta)
    _, _, _, _, rho = polarRays(result)
    zBase = _polarBaseZ(result)
    base = np.interp(angle, np.append(grid.theta, 2.0 * math.pi), np.append(zBase, zBase[0]))
    cyl = z <= 0.0
    polarRow = np.where(cyl, (grid.nCyl - 1) * (1.0 - np.clip(z / np.minimum(base, -1e-6), 0.0, 1.0)), row)
    radius = np.interp(np.clip(polarRow, 0.0, grid.nRows - 1.0), np.arange(grid.nRows, dtype=float), rho)
    return _polarLookup(polar, radius, angle)


def polarMap(result, sample, samplesAcrossWall=9):
    """Counts of the wall for the polar map: the maximum along each ray between endo- and
    epicardium. Returns {"values" [rows, angles], "rho" [rows] (0 apex .. 1 base), "theta" [angles]
    (from anterior toward the septum)}."""
    origins, directions, endo, epi, rho = polarRays(result)
    fractions = np.linspace(0.0, 1.0, int(samplesAcrossWall))
    distance = endo[..., None] + fractions * (epi - endo)[..., None]        # [rows, angles, n]
    points = origins[:, :, None, :] + distance[..., None] * directions[:, :, None, :]
    counts = np.asarray(sample(points.reshape(-1, 3)), dtype=float).reshape(distance.shape)
    return {"values": counts.max(axis=2), "rho": rho, "theta": result["grid"].theta.copy()}


def _wallRays(result, points):
    """Ray through each world point as in the fit (radial on the cylinder, from the LV center on the
    apical cap): origins, directions, endo and epi distances along it, distance of the point, z."""
    grid = result["grid"]
    points = np.asarray(points, dtype=float)
    row, col, dist, z = grid.toGrid(points)
    d = points - grid.center
    q = d - z[:, None] * grid.u
    cyl = z <= 0.0
    origins = np.where(cyl[:, None], grid.center + z[:, None] * grid.u, grid.center)
    lengths = np.where(cyl, np.linalg.norm(q, axis=1), np.linalg.norm(d, axis=1))
    directions = np.where(cyl[:, None], q, d) / np.maximum(lengths, 1e-9)[:, None]
    return origins, directions, _bilinear(result["endo"], row, col), _bilinear(result["epi"], row, col), dist, z


def wallField(result, points):
    """Signed distance-like value of world points, positive inside the myocardium: the smaller of
    (epi - distance), (distance - endo) and the height above the base rim (rimZAt), in mm. Its zero level
    is the whole wall as one closed surface: epicardium, endocardium and the rim at the base."""
    grid = result["grid"]
    _, _, endo, epi, dist, z = _wallRays(result, points)
    d = np.asarray(points, dtype=float) - grid.center
    rim = z - rimZAt(result, np.arctan2(d @ grid.s, d @ grid.w))
    return np.minimum(np.minimum(epi - dist, dist - endo), rim)


def uptakeAtPoints(result, sample, points, samplesAcrossWall=9):
    """Counts of the wall projected onto surface points (SSP style): the maximum between the endo-
    and epicardium along the ray through each point, as in polarMap."""
    origins, directions, endo, epi, _, _ = _wallRays(result, points)
    fractions = np.linspace(0.0, 1.0, int(samplesAcrossWall))
    distance = endo[:, None] + fractions[None, :] * (epi - endo)[:, None]          # [N, n]
    along = origins[:, None, :] + distance[..., None] * directions[:, None, :]
    return np.asarray(sample(along.reshape(-1, 3)), dtype=float).reshape(distance.shape).max(axis=1)


def _polarLookup(polar, radius, angle):
    """Values of the polar map at bullseye radius (0..1) and angle (radians, from anterior)."""
    rho, values = polar["rho"], polar["values"]
    rows, angles = values.shape
    order = np.argsort(rho)
    row = np.interp(radius, rho[order], np.arange(rows, dtype=float)[order])
    col = np.mod(angle, 2.0 * math.pi) / (2.0 * math.pi / angles)
    return _bilinear(values, row, col)


def segmentIndex(radius, angle):
    """17-segment number (1..17) at bullseye radius (0..1) and angle (from anterior toward the
    septum); 0 outside the bullseye."""
    radius = np.asarray(radius, dtype=float)
    degrees = np.mod(np.degrees(angle), 360.0)
    sixths = np.mod(np.floor((degrees + 30.0) / 60.0), 6).astype(int)
    quarters = np.mod(np.floor((degrees + 45.0) / 90.0), 4).astype(int)
    edges = POLAR_RING_EDGES
    segment = np.zeros(radius.shape, dtype=int)
    segment[radius < edges[1]] = 17
    apical = (radius >= edges[1]) & (radius < edges[2])
    segment[apical] = 13 + quarters[apical]
    middle = (radius >= edges[2]) & (radius < edges[3])
    segment[middle] = 7 + sixths[middle]
    basal = (radius >= edges[3]) & (radius <= edges[4])
    segment[basal] = 1 + sixths[basal]
    return segment


VESSELS = ("LAD", "RCA", "LCX")
VESSEL_SEGMENTS = {   # coronary territories of the 17 segments (AHA 2002 standard assignment)
    "LAD": (1, 2, 7, 8, 13, 14, 17),
    "RCA": (3, 4, 9, 10, 15),
    "LCX": (5, 6, 11, 12, 16),
}
OVERLAYS = ("vessels", "segments")   # region overlays of the polar maps and 3D views
# where the values are written on the bullseye: (radius, angle from anterior toward the septum, deg)
SEGMENT_LABEL_PLACES = tuple(
    [(0.5 * (POLAR_RING_EDGES[3] + POLAR_RING_EDGES[4]), 60.0 * k) for k in range(6)]
    + [(0.5 * (POLAR_RING_EDGES[2] + POLAR_RING_EDGES[3]), 60.0 * k) for k in range(6)]
    + [(0.5 * (POLAR_RING_EDGES[1] + POLAR_RING_EDGES[2]), 90.0 * k) for k in range(4)] + [(0.0, 0.0)])
VESSEL_LABEL_PLACES = ((0.6, 35.0), (0.6, 160.0), (0.6, 270.0))   # LAD anteroseptal, RCA inferior, LCX lateral


def regionIndex(radius, angle, overlay="segments"):
    """Region at bullseye radius / angle: the segment number (1..17), for "vessels" the index of its
    territory in VESSELS plus one, for "walls" the index in WALLS plus one (wallIndex); 0 outside the
    bullseye."""
    if overlay == "walls":
        return wallIndex(radius, angle)
    segment = segmentIndex(radius, angle)
    if overlay != "vessels":
        return segment
    lookup = np.zeros(18, dtype=int)
    for index, vessel in enumerate(VESSELS):
        lookup[list(VESSEL_SEGMENTS[vessel])] = index + 1
    return lookup[segment]


def regionBorders(overlay="segments", step=0.5):
    """Borders between the regions of an overlay on the bullseye, as polylines of (radius,
    angle in degrees) [n, 2], with the outer rim. Arcs run along the ring edges where the region
    changes across them, spokes along the segment edges where it changes around; for "vessels"
    this gives the stepped territory outline that follows the segment borders."""
    if overlay == "walls":
        return wallBorders()
    edges = POLAR_RING_EDGES
    eps = 1e-4
    lines = []
    degrees = np.arange(0.0, 360.0, step) + 0.5 * step

    def runs(mask):
        """Contiguous (start, stop) index runs of a circular boolean array."""
        if mask.all():
            return [(0, len(mask))]
        found, start = [], None
        rolled = int(np.argmin(mask))   # start the walk where the mask is off
        order = [(rolled + i) % len(mask) for i in range(len(mask))]
        for position, i in enumerate(order):
            if mask[i] and start is None:
                start = position
            if not mask[i] and start is not None:
                found.append((order[start], order[start] + position - start))
                start = None
        if start is not None:
            found.append((order[start], order[start] + len(mask) - start))
        return found

    for edge in edges[1:4]:
        inside = regionIndex(np.full(degrees.shape, edge - eps), np.radians(degrees), overlay)
        outside = regionIndex(np.full(degrees.shape, edge + eps), np.radians(degrees), overlay)
        for start, stop in runs(inside != outside):
            angles = np.concatenate([[start * step], (np.arange(start, stop) + 0.5) * step, [stop * step]])
            lines.append(np.column_stack([np.full(angles.shape, edge), angles]))
    rim = np.linspace(0.0, 360.0, 721)
    lines.append(np.column_stack([np.ones(rim.shape), rim]))
    rings = ((edges[1], edges[2], [45.0 + 90.0 * k for k in range(4)]),
             (edges[2], edges[3], [30.0 + 60.0 * k for k in range(6)]),
             (edges[3], edges[4], [30.0 + 60.0 * k for k in range(6)]))
    for low, high, spokes in rings:
        middle = 0.5 * (low + high)
        for spoke in spokes:
            before = regionIndex(middle, math.radians(spoke - 1.0), overlay)
            after = regionIndex(middle, math.radians(spoke + 1.0), overlay)
            if before != after:
                lines.append(np.array([[low, spoke], [high, spoke]]))
    return lines


def regionValues(polar, overlay="segments", resolution=181):
    """Mean polar map value of each region of an overlay (17 segments, the VESSELS territories or the
    five WALLS)."""
    if overlay == "walls":
        return wallValues(polar, resolution)
    if overlay != "vessels":
        return segmentValues(polar, resolution)
    image, segments = bullseye(polar, resolution)
    values = []
    for vessel in VESSELS:
        inside = np.isin(segments, VESSEL_SEGMENTS[vessel])
        values.append(float(np.nanmean(image[inside])) if np.any(inside) else float("nan"))
    return np.array(values)


def regionLabelPlaces(overlay="segments"):
    if overlay == "walls":
        return WALL_LABEL_PLACES
    return VESSEL_LABEL_PLACES if overlay == "vessels" else SEGMENT_LABEL_PLACES


def bullseye(polar, size):
    """Polar map as a [size, size] image (row 0 at the top; NaN outside the circle). Anterior up,
    septum left, lateral right, inferior down; apex in the center, valve plane at the rim.
    Also returns the segment number of every pixel."""
    radius, angle = _bullseyeCoordinates(size)
    image = _polarLookup(polar, np.clip(radius, 0.0, 1.0).ravel(), angle.ravel()).reshape(size, size)
    image[radius > 1.0] = np.nan
    return image, segmentIndex(radius, angle)


def _bullseyeCoordinates(size):
    """Bullseye radius (0 apex .. 1 rim) and angle (radians, anterior up 0, septum left +90 deg) of every
    pixel of a [size, size] image."""
    coordinate = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    x = coordinate[None, :]                     # right
    y = -coordinate[:, None]                    # up
    radius = np.sqrt(x ** 2 + y ** 2) * np.ones((size, size))
    angle = np.arctan2(-x, y) * np.ones((size, size))
    return radius, angle


def segmentValues(polar, resolution=181):
    """Mean polar map value of each of the 17 segments (index 0 = segment 1), by bullseye area."""
    image, segments = bullseye(polar, resolution)
    return np.array([float(np.nanmean(image[segments == n])) if np.any(segments == n) else float("nan")
                     for n in range(1, 18)])


WALLS = ("Apex", "Anterior", "Septal", "Inferior", "Lateral")
# The five walls are drawn on the bullseye itself, not grouped from the 17 segments: the apex is the disk
# inside APEX_RADIUS (the size of segment 17), and the rest is cut into four quarters by an X, each
# centred on its wall (anterior up, septal left, inferior down, lateral right), borders at 45, 135, 225
# and 315 deg.
APEX_RADIUS = POLAR_RING_EDGES[1]
WALL_CENTERS_DEG = (0.0, 90.0, 180.0, 270.0)        # anterior, septal, inferior, lateral
WALL_LABEL_PLACES = ((0.0, 0.0),) + tuple((0.5 * (APEX_RADIUS + 1.0), c) for c in WALL_CENTERS_DEG)
# how a polar map is painted: per pixel, or the mean of each of the 17 segments, the five walls or the territories
REGION_VIEWS = ("pixels", "segments", "walls", "vessels")
REGION_OVERLAYS = ("vessels", "segments", "walls")   # the segmentations a polar map can show (borders and values)


def wallIndex(radius, angle):
    """Wall at bullseye radius (0..1) and angle (radians, from anterior toward the septum): its index
    in WALLS plus one (1 apex, 2 anterior, 3 septal, 4 inferior, 5 lateral); 0 outside the bullseye."""
    radius = np.asarray(radius, dtype=float)
    degrees = np.mod(np.degrees(np.asarray(angle, dtype=float)), 360.0)
    quarter = np.mod(np.floor((degrees + 45.0) / 90.0), 4).astype(int)
    index = np.where(radius < APEX_RADIUS, 1, 2 + quarter)
    return np.where(radius > 1.0, 0, index).astype(int)


def wallBorders(samples=721):
    """Borders of the five walls on the bullseye as polylines of (radius, angle in degrees): the apex
    circle, the four arms of the X from it to the rim, and the rim."""
    circle = np.linspace(0.0, 360.0, samples)
    lines = [np.column_stack([np.full(circle.shape, APEX_RADIUS), circle]),
             np.column_stack([np.ones(circle.shape), circle])]
    for center in WALL_CENTERS_DEG:
        lines.append(np.array([[APEX_RADIUS, center + 45.0], [1.0, center + 45.0]]))
    return lines


def wallValues(polar, resolution=181):
    """Mean polar map value of each of the five WALLS, by bullseye area (pixels of the map)."""
    image, _ = bullseye(polar, resolution)
    walls = wallIndex(*_bullseyeCoordinates(resolution))
    values = []
    for index in range(1, len(WALLS) + 1):
        inside = (walls == index) & np.isfinite(image)
        values.append(float(image[inside].mean()) if np.any(inside) else float("nan"))
    return np.array(values)


def wallBullseye(polar, size, resolution=181):
    """Bullseye of the five wall means (apex, anterior, septal, inferior, lateral): every pixel of a
    wall shows its mean, NaN outside the circle. Also returns the five means (order of WALLS)."""
    values = wallValues(polar, resolution)
    walls = wallIndex(*_bullseyeCoordinates(size))
    image = np.full(walls.shape, np.nan)
    for index, value in enumerate(values, start=1):
        image[walls == index] = value
    return image, values


def regionBullseye(polar, size, view="pixels"):
    """Bullseye painted per REGION_VIEWS: "walls" (five wall means), "segments" (17 segment means), "vessels"
    (territory means) or "pixels" (the polar map itself)."""
    if view == "walls":
        return wallBullseye(polar, size)[0]
    if view == "segments":
        return segmentBullseye(polar, size)[0]
    if view == "vessels":
        return vesselBullseye(polar, size)[0]
    return bullseye(polar, size)[0]


def vesselBullseye(polar, size, resolution=181):
    """Bullseye of the territory means (VESSELS, from the 17 segments): every pixel of a territory shows its
    mean, NaN outside the circle. Also returns the means (order of VESSELS)."""
    values = regionValues(polar, "vessels", resolution)
    _, segments = bullseye(polar, size)
    territories = regionIndex(*_bullseyeCoordinates(size), overlay="vessels")
    image = np.full(segments.shape, np.nan)
    for index, value in enumerate(values, start=1):
        image[territories == index] = value
    return image, values


def segmentBullseye(polar, size, resolution=181):
    """Bullseye of the 17 segment means: every pixel of a segment shows its mean (segmentValues),
    NaN outside the circle and in a segment without values. Also returns the 17 means."""
    values = segmentValues(polar, resolution)
    _, segments = bullseye(polar, size)
    image = np.full(segments.shape, np.nan)
    inside = segments > 0
    image[inside] = values[segments[inside] - 1]
    return image, values


def polarToWorld(result, radius, angle, outsideMm=1.5):
    """World points on the epicardium (outsideMm beyond it) at bullseye radius (0 apex .. 1 valve
    plane) and angle (radians, from anterior toward the septum): the inverse of the polar map."""
    origins, directions, _, epi, rho = polarRays(result)
    points = origins + (epi + float(outsideMm))[..., None] * directions          # [rows, angles, 3]
    rows, angles = epi.shape
    order = np.argsort(rho)
    row = np.clip(np.interp(np.asarray(radius, dtype=float), rho[order], np.arange(rows, dtype=float)[order]),
                  0.0, rows - 1.0)
    col = np.mod(np.asarray(angle, dtype=float), 2.0 * math.pi) / (2.0 * math.pi / angles)
    r0 = np.minimum(np.floor(row).astype(int), rows - 2)
    c0 = np.floor(col).astype(int) % angles
    c1 = (c0 + 1) % angles
    fr, fc = (row - r0)[..., None], (col - np.floor(col))[..., None]
    return ((1 - fr) * ((1 - fc) * points[r0, c0] + fc * points[r0, c1])
            + fr * ((1 - fc) * points[r0 + 1, c0] + fc * points[r0 + 1, c1]))


def segmentBorderLines(result, samples=180, overlay="segments"):
    """Borders of the regions of an overlay (17 segments or the vessel territories) of the polar
    map, drawn on the epicardium: a list of polylines (world points [n, 3])."""
    lines = []
    for line in regionBorders(overlay):
        if len(line) == 2:   # spoke: sample along the radius
            radius = np.linspace(line[0, 0], line[1, 0], 24)
            angles = np.full(radius.shape, line[0, 1])
        else:
            angles = np.linspace(line[0, 1], line[-1, 1], max(2, int(samples * (line[-1, 1] - line[0, 1]) / 360.0) + 1))
            radius = np.full(angles.shape, line[0, 0])
        if np.all(radius >= 1.0):
            continue   # the rim is the valve plane edge of the model already
        lines.append(polarToWorld(result, radius, np.radians(angles)))
    return lines


def surfaceCells(rows, angles, closeBase=False, baseCenterIndex=None, apexCenterIndex=None):
    """Cells of a [rows, angles] grid surface: quads between the rows (so a wireframe shows clean
    rings and meridians), a triangle fan to apexCenterIndex at the apex and, with closeBase, a fan
    to baseCenterIndex in the valve plane."""
    cells = []
    for r in range(rows - 1):
        for a in range(angles):
            b = (a + 1) % angles
            cells.append((r * angles + a, (r + 1) * angles + a, (r + 1) * angles + b, r * angles + b))
    if closeBase and baseCenterIndex is not None:
        for a in range(angles):
            cells.append((baseCenterIndex, (a + 1) % angles, a))
    if apexCenterIndex is not None:
        last = (rows - 1) * angles
        for a in range(angles):
            cells.append((apexCenterIndex, last + a, last + (a + 1) % angles))
    return cells


WALL_LABELS = (   # (text, angle around the axis from anterior toward the septum, deg)
    ("Anterior", 0.0), ("Septal", 90.0), ("Inferior", 180.0), ("Lateral", 270.0))


def wallLabelPositions(result, distanceMm=45.0, levelMm=-10.0):
    """Floating label positions [(text, label point, wall point)] distanceMm outside the epicardium:
    anterior, septal, inferior and lateral at the mid-ventricle (levelMm on the axis), apex beyond
    the apex and base beyond the valve plane."""
    grid = result["grid"]
    epi = result["epi"]
    row = int(np.clip(np.searchsorted(grid.z, levelMm), 0, grid.nCyl - 1))
    z = float(grid.z[row])
    out = []
    for text, degrees in WALL_LABELS:
        theta = math.radians(degrees)
        col = int(round(theta / (2.0 * math.pi / grid.nTheta))) % grid.nTheta
        radial = grid.radial(np.array(theta))
        wall = grid.center + z * grid.u + float(epi[row, col]) * radial
        out.append((text, wall + distanceMm * radial, wall))
    apexWall = grid.center + float(np.mean(epi[-1])) * grid.u
    out.append(("Apex", apexWall + distanceMm * grid.u, apexWall))
    baseWall = grid.center + valvePlaneZ(result) * grid.u
    out.append(("Base", baseWall - distanceMm * grid.u, baseWall))
    return out


def wallStatistics(result, labels, voxelVolumeMm3):
    cavity = float(np.count_nonzero(labels == LABEL_CAVITY)) * voxelVolumeMm3 / 1000.0
    myocardium = float(np.count_nonzero(labels == LABEL_MYOCARDIUM)) * voxelVolumeMm3 / 1000.0
    grid = result["grid"]
    a = valvePlaneZ(result)
    inWall = np.ones(result["mid"].shape, dtype=bool)
    inWall[:grid.nCyl] = grid.z[:, None] >= _polarBaseZ(result)[None, :]
    return {"cavityMl": cavity, "myocardiumMl": myocardium, "massG": 1.05 * myocardium,
            "meanThicknessMm": float(result["thickness"][inWall].mean()), "baseZ": a,
            "valveTiltDeg": valveTilt(result)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _peaks(profiles, t, lo, hi):
    """Sub-sample position and value of the maximum of each profile inside [lo, hi]."""
    inside = (t[None, None, :] >= lo[..., None]) & (t[None, None, :] <= hi[..., None])
    masked = np.where(inside, profiles, -np.inf)
    index = np.argmax(masked, axis=2)
    index = np.clip(index, 1, len(t) - 2)
    y0 = np.take_along_axis(profiles, (index - 1)[..., None], 2)[..., 0]
    y1 = np.take_along_axis(profiles, index[..., None], 2)[..., 0]
    y2 = np.take_along_axis(profiles, (index + 1)[..., None], 2)[..., 0]
    denominator = y0 - 2.0 * y1 + y2
    shift = np.where(np.abs(denominator) > 1e-12, 0.5 * (y0 - y2) / np.where(denominator == 0, 1, denominator), 0.0)
    shift = np.clip(shift, -0.5, 0.5)
    step = t[1] - t[0]
    position = t[index] + shift * step
    value = y1 - 0.25 * (y0 - y2) * shift
    empty = ~inside.any(axis=2)
    value[empty] = 0.0
    position[empty] = 0.5 * (lo[empty] + hi[empty])
    return position, np.maximum(value, 0.0)


def _interp(profiles, t, position):
    step = t[1] - t[0]
    f = np.clip(position / step, 0, len(t) - 1.001)
    i = np.floor(f).astype(int)
    frac = f - i
    a = np.take_along_axis(profiles, i[..., None], 2)[..., 0]
    b = np.take_along_axis(profiles, (i + 1)[..., None], 2)[..., 0]
    return (1.0 - frac) * a + frac * b


def _profileWidth(profiles, t, position, peak, reach):
    """Width of the wall profile of every ray with each side measured against its own floor: the lowest
    count within reach on the cavity side (the blood pool) and on the outer side. Each edge is where the
    profile is half way between the peak and that side's floor, where a blurred step between the wall
    and the level beyond it sits; one floor for both sides (the outer, lower one) would put the inner edge
    out in the blood pool, the more so the dimmer the wall against it, and thicken a dim end-diastolic wall
    more than the brighter end-systolic one. slabThickness turns the width into a wall thickness."""
    d = t[None, None, :] - position[..., None]
    window = np.abs(d) <= reach

    def floor(side):
        level = np.where(window & side, profiles, np.inf).min(axis=2)
        return np.where(np.isfinite(level), np.minimum(level, peak), 0.0)
    inner, outer = floor(d < 0), floor(d > 0)
    toInner, toOuter = _halfWidths(profiles, t, position, peak, reach, inner, outer)
    return toInner + toOuter


def _wallHalfWidths(profiles, t, position, peak, reach, pool=None):
    """(toward the cavity, away from it): distances from the mid-wall to where the profile falls half way
    from the peak to each side's floor (_profileWidth, within reach). With pool (the blood pool's counts,
    _poolLevel) the cavity side is measured against it instead, searched up to the axis; a ray whose
    profile does not fall that far (a cavity filled by the walls' blur) keeps its own floor."""
    d = t[None, None, :] - position[..., None]
    window = np.abs(d) <= reach

    def floor(side):
        level = np.where(window & side, profiles, np.inf).min(axis=2)
        return np.where(np.isfinite(level), np.minimum(level, peak), 0.0)
    toInner, toOuter = _halfWidths(profiles, t, position, peak, reach, floor(d < 0), floor(d > 0))
    if pool is not None:
        far = float(t[-1])
        toPool, _ = _halfWidths(profiles, t, position, peak, far, np.minimum(pool, peak), peak, sides=(-1.0,))
        found = (toPool < far - 1e-6) & (toPool <= position)
        toInner = np.where(found, toPool, toInner)
    return toInner, toOuter


_SLAB_WIDTHS = {}
SLAB_T = np.arange(0.0, 40.01, 0.5)   # slab widths of the slabThickness table (mm)


def _slabWidths(resolution, reach):
    """Profile width (mm) of a slab of every width in SLAB_T, of uniform uptake, blurred by a Gaussian of
    FWHM resolution, measured as _profileWidth measures a wall: each half against the profile's floor within
    reach of the peak (the blur's tails beyond reach raise it, so the width reads narrower than the full
    width at half maximum, by up to 2 mm at 15 mm blur). Cached by resolution and reach, rounded to 0.5 mm."""
    resolution = max(round(2.0 * float(resolution)) / 2.0, 1.0)
    reach = max(round(2.0 * float(reach)) / 2.0, 1.0)
    key = (resolution, reach)
    if key not in _SLAB_WIDTHS:
        from math import erf
        erfv = np.vectorize(erf)
        s = math.sqrt(2.0) * resolution / 2.3548
        x = np.arange(0.0, reach + 1e-6, 0.02)
        y = 0.5 * (erfv((x[None, :] + SLAB_T[:, None] / 2.0) / s) - erfv((x[None, :] - SLAB_T[:, None] / 2.0) / s))
        level = 0.5 * (y[:, 0] + y[:, -1])
        below = y <= level[:, None]
        first = np.where(below.any(axis=1), np.argmax(below, axis=1), len(x) - 1)
        _SLAB_WIDTHS[key] = 2.0 * x[np.maximum(first, 1)]
    return _SLAB_WIDTHS[key]


def slabThickness(width, resolution, reach):
    """Wall thickness (mm) from profile widths (_profileWidth, measured within reach mm of the peak) at a
    system resolution (FWHM, mm): the slab whose blurred profile has this width (_slabWidths), weighed
    against a prior wall of THICKNESS_PRIOR_MM +- THICKNESS_PRIOR_SIGMA by how well the width tells slabs
    apart at this blur (each width known to WIDTH_SIGMA_MM). A slab thinner than the blur keeps about the
    blur's width (at 12 mm a 10 mm slab reads 14.1 mm, a 6 mm one 12.8 mm), so there a small error of the
    width would move the thickness a lot and it leans to the prior; a wall thicker than the blur is read
    from its width. The mean over SLAB_T; a width that is not finite gives the prior."""
    table = _slabWidths(resolution, reach)
    width = np.asarray(width, dtype=float)
    known = np.isfinite(width)[..., None]
    misfit = np.where(known, (table - np.nan_to_num(width)[..., None]) / WIDTH_SIGMA_MM, 0.0)
    log = -0.5 * misfit ** 2 - 0.5 * ((SLAB_T - THICKNESS_PRIOR_MM) / THICKNESS_PRIOR_SIGMA) ** 2
    weight = np.exp(log - log.max(axis=-1, keepdims=True))
    return (weight * SLAB_T).sum(axis=-1) / weight.sum(axis=-1)


def _fwhm(profiles, t, position, peak, reach, base=None):
    """Full width at half maximum around the peak (half way between base and peak when given),
    searched up to reach mm on each side."""
    base = np.zeros_like(peak) if base is None else base
    return sum(_halfWidths(profiles, t, position, peak, reach, base, base))


def _halfWidths(profiles, t, position, peak, reach, innerBase, outerBase, sides=(-1.0, 1.0)):
    """Distances from the mid-wall to where the profile falls half way from the peak to innerBase (toward
    the axis) and to outerBase (away from it), searched up to reach mm on each side."""
    step = 0.5 * (t[1] - t[0])
    offsets = np.arange(step, reach + 1e-6, step)
    widths = []
    for sign, base in ((-1.0, innerBase), (1.0, outerBase)):
        if sign not in sides:
            widths.append(None)
            continue
        halfValue = base + 0.5 * (peak - base)
        previousValue, previousOffset = peak.copy(), 0.0
        found = np.full(position.shape, reach)
        done = np.zeros(position.shape, dtype=bool)
        for offset in offsets:
            value = _interp(profiles, t, position + sign * offset)
            crossing = (~done) & (value <= halfValue)
            if crossing.any():
                drop = previousValue - value
                fraction = np.clip(np.where(drop > 1e-12, (previousValue - halfValue) / np.where(drop > 1e-12, drop, 1.0), 1.0), 0.0, 1.0)
                found[crossing] = (previousOffset + fraction * (offset - previousOffset))[crossing]
                done |= crossing
            previousValue, previousOffset = value, offset
        widths.append(found)
    return widths[0], widths[1]


POLE_SPACING_MM = 3.0   # smoothing width around the axis is set by the spacing of the rays at the equator


def _nearPoleRows(grid, distance, spacingMm=1.5):
    """Cap rows whose neighbouring rays (one angle step apart) are closer than spacingMm: there all
    rays of the row look at nearly the same piece of wall."""
    rows = np.zeros(grid.nRows, dtype=bool)
    arc = np.median(distance[grid.nCyl:], axis=1) * np.sin(grid.phi) * (2.0 * math.pi / grid.nTheta)
    rows[grid.nCyl:] = arc < spacingMm
    return rows


def _poleSmooth(grid, values, weights, sigma, distance=None):
    """Smooth the apical cap rows around the axis with the same width in mm as at the equator: near
    the apex the rays of one row converge, so their values must agree (otherwise the surface folds
    at the apex). sigma is in rays at the equator, as in _smoothMap; the row closest to the axis
    becomes one weighted mean."""
    distance = values if distance is None else distance
    out = values.copy()
    n = grid.nTheta
    step = 2.0 * math.pi / n
    sigmaMm = max(float(sigma), 0.5) * POLE_SPACING_MM
    for index, phi in enumerate(grid.phi):
        row = grid.nCyl + index
        arc = max(float(np.median(distance[row])) * math.sin(phi) * step, 1e-6)
        sigmaRays = sigmaMm / arc
        w = weights[row]
        if sigmaRays > n / 4.0:
            out[row] = float((values[row] * w).sum() / max(w.sum(), 1e-12))
            continue
        if sigmaRays <= float(sigma):
            continue          # _smoothMap already smoothed this much
        kernel = _gaussianKernel(sigmaRays)
        k = min(len(kernel) // 2, n - 1)
        kernel = kernel[len(kernel) // 2 - k:len(kernel) // 2 + k + 1]
        wv = np.concatenate([(values[row] * w)[-k:], values[row] * w, (values[row] * w)[:k]]) if k else values[row] * w
        ww = np.concatenate([w[-k:], w, w[:k]]) if k else w
        out[row] = np.convolve(wv, kernel, mode="valid") / np.maximum(np.convolve(ww, kernel, mode="valid"), 1e-12)
    return out


def _angleWindow(grid, raysAt48):
    """Odd number of rays covering the same angle around the axis as raysAt48 rays do with 48."""
    size = int(round(raysAt48 * grid.nTheta / 48.0))
    return max(3, size | 1)


def _ringReference(grid, peakT, weight, reference, tolerance):
    """The neighbour reference of the cylinder rows replaced by their robust ring (_ringFit) where the two
    disagree: there the neighbours share a wrong peak (a wide defect next to an organ)."""
    ring = _ringFit(grid, peakT, weight, tolerance)
    out = reference.copy()
    off = np.abs(reference - ring) > tolerance
    off[grid.nCyl:] = False
    out[off] = ring[off]
    return out


def _ringFit(grid, values, weights, tolerance, iterations=4):
    """Robust ring of every cylinder row: distance(theta) = r0 + a cos(theta) + b sin(theta) (a circle,
    also off the axis), fitted with count weights while the rays further than tolerance from it lose their
    weight. A defect wide enough that its neighbours share its wrong peak (an organ next to it) is still
    a bump on that ring. Cap rows are returned as they are (their rays converge on the apex)."""
    out = np.array(values, dtype=float, copy=True)
    theta = np.asarray(grid.theta, dtype=float)
    basis = np.stack([np.ones_like(theta), np.cos(theta), np.sin(theta)], axis=1)
    for row in range(grid.nCyl):
        y = np.asarray(values[row], dtype=float)
        w = np.maximum(np.asarray(weights[row], dtype=float), 0.0)
        if np.count_nonzero(w > 1e-9) < 6:
            continue
        fit = None
        for _ in range(iterations):
            sw = np.sqrt(w)
            try:
                coefficients = np.linalg.lstsq(basis * sw[:, None], y * sw, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            fit = basis @ coefficients
            keep = np.abs(y - fit) <= tolerance
            if np.count_nonzero(keep & (w > 1e-9)) < 6:
                break
            w = np.where(keep, w, 0.0)
        if fit is not None:
            out[row] = fit
    return out


# ---------------------------------------------------------------------------
# Shape model: the whole LV as one smooth surface, for the rays the counts do not show
# ---------------------------------------------------------------------------
def _frameRays(grid):
    """(origins, directions) of every ray in the cardiac frame (x along w, y along s, z along u, from
    the grid center), each [rows, angles, 3]."""
    radial = np.stack([np.cos(grid.theta), np.sin(grid.theta), np.zeros(grid.nTheta)], axis=1)
    origins = np.zeros((grid.nRows, grid.nTheta, 3))
    origins[:grid.nCyl, :, 2] = grid.z[:, None]
    directions = np.empty((grid.nRows, grid.nTheta, 3))
    directions[:grid.nCyl] = radial[None]
    directions[grid.nCyl:] = (np.sin(grid.phi)[:, None, None] * radial[None]
                              + np.cos(grid.phi)[:, None, None] * np.array([0.0, 0.0, 1.0]))
    return origins, directions


def _ellipsoidDistance(shape, origins, directions):
    """Distance along every ray to the ellipsoid ((x' - x0) / A)^2 + ((y' - y0) / B)^2 + ((z' - z0) / C)^2 = 1,
    shape = (x0, y0, z0, A, B, C[, a, b]) in the cardiac frame, its axes turned by a about x and then by b
    about y (radians; x', y', z' are the ellipsoid's own axes). NaN where the ray does not leave it."""
    x0, y0, z0, sa, sb, sc = shape[:6]
    m = 1.0 / np.array([sa, sb, sc]) ** 2
    o = origins - np.array([x0, y0, z0])
    if len(shape) > 6:
        ca, sa_, cb, sb_ = math.cos(shape[6]), math.sin(shape[6]), math.cos(shape[7]), math.sin(shape[7])
        turn = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa_], [0.0, sa_, ca]]) @ \
            np.array([[cb, 0.0, sb_], [0.0, 1.0, 0.0], [-sb_, 0.0, cb]])     # ellipsoid axes in the frame
        o = o @ turn
        directions = directions @ turn
    q2 = (directions ** 2) @ m
    q1 = 2.0 * (o * directions) @ m
    q0 = (o ** 2) @ m - 1.0
    disc = q1 ** 2 - 4.0 * q2 * q0
    with np.errstate(invalid="ignore"):
        t = (-q1 + np.sqrt(disc)) / (2.0 * q2)
    return np.where((disc >= 0) & (t > 0), t, np.nan)


def _clearWalls(weight, counts=None):
    """Rays of seen wall away from any defect: weight > 0, counts (the peak counts, or else the sqrt of the
    weight) at least SHAPE_DIM of the bright walls' (90th percentile), and no dimmer ray within about
    SHAPE_BORDER_SIGMA grid samples. Stricter than the weights: the walls kept here set the shape model and
    the wall thickness, which need the walls well clear of a defect's blurred edge."""
    seen = weight > 1e-9
    if not np.any(seen):
        return seen
    counts = np.sqrt(np.maximum(weight, 0.0)) if counts is None else np.where(seen, counts, 0.0)
    dim = (counts < SHAPE_DIM * max(float(np.percentile(counts[seen], 90)), 1e-9)).astype(float)
    return seen & (_convolve2d(dim, _gaussianKernel(SHAPE_BORDER_SIGMA)) <= 0.1)


def _shapeModel(grid, distance, weight, inside, tolerance, baseZ, counts=None):
    """Mid-wall distance of every ray from a smooth model of the whole LV, fitted to the rays the counts
    show (weight > 0, inside: on the LV side of the valve plane): a truncated ellipsoid, its center free
    (off the axis, along it), its long axis tilted against the LV axis where that fits better (at most
    SHAPE_MAX_TILT, a weak prior toward none), fitted to the distances along the rays
    (Levenberg-Marquardt, rays far from it given less weight). Where only straight side walls are seen
    (the apex is a defect) the center and the long semi-axis trade off; a weak prior keeps the center
    near the valve plane (baseZ, SHAPE_CENTER_SIGMA), where the classic truncated-ellipsoid model of the
    LV has it, so the taper of the side walls sets the apex. The model's misfit on the walls that are seen
    (a straight base, a flat septum) is added back as a broad smooth field that fades out inside a defect.
    A defect then takes the curve of the walls around it (also at the apex, where neighbouring rays alone
    bridge it flat) instead of the mean of its own noisy peaks. Peaks further than tolerance (and than
    three times the typical misfit) from the ellipsoid are another organ next to a defect (liver, gut)
    and are left out. Returns (model map, off-model mask), or (None, None) if too little wall is seen."""
    origins, directions = _frameRays(grid)
    w = np.where(inside & np.isfinite(distance), np.maximum(weight, 0.0), 0.0)
    if np.count_nonzero(w > 1e-9) < max(12, grid.nTheta // 2):
        return None, None
    seen = w > 1e-9
    # the walls next to a defect are left out of the fit: blurred together with the dark defect, their
    # peaks lie off the wall (the side walls bulge out at the edge of an apical defect, and an ellipsoid
    # through them would run far past the apex)
    clear = _clearWalls(w, counts)
    if np.count_nonzero(clear) >= max(12, grid.nTheta // 2):
        seen &= clear
    o, d, y = origins[seen], directions[seen], distance[seen]
    # the apical half counts more: there the model carries a defect across the apex; a base that runs
    # on nearly straight (a cylinder) would otherwise stretch the ellipsoid past it
    base = w[seen] * np.where((o + y[:, None] * d)[:, 2] > 0.0, 1.0, SHAPE_BASAL_WEIGHT)
    side = seen.copy()
    side[grid.nCyl:] = False
    radius = float(np.median(distance[side])) if np.any(side) else float(np.median(y))
    reach = float(grid.t[-1])
    tilt = math.radians(SHAPE_MAX_TILT)
    lower = np.array([-reach, -reach, float(grid.z[0]) - reach, 5.0, 5.0, 5.0, -tilt, -tilt])
    upper = np.array([reach, reach, 0.0, 1.5 * reach, 1.5 * reach, 4.0 * reach, tilt, tilt])
    shape = np.array([0.0, 0.0, min(float(baseZ), -1.0), radius, radius, max(1.2 * radius - float(baseZ), 10.0),
                      0.0, 0.0])
    shape = np.clip(shape, lower, upper)
    priorWeight = SHAPE_CENTER_PRIOR * float(base.sum())

    def residuals(values, fitWeight):
        t = _ellipsoidDistance(values, o, d)
        r = np.where(np.isfinite(t), t - y, tolerance)       # a ray that misses: one tolerance off
        prior = [(values[2] - baseZ) / SHAPE_CENTER_SIGMA, values[6] / math.radians(SHAPE_TILT_SIGMA),
                 values[7] / math.radians(SHAPE_TILT_SIGMA)]
        return np.concatenate([np.sqrt(fitWeight) * r, math.sqrt(priorWeight) * np.array(prior)]), r

    def solve(shape, lower, upper):
        fitWeight = base.copy()
        scale = 2.0
        for _ in range(4):                                     # robust: reweighted from the misfit
            damping = 1e-2
            current, r = residuals(shape, fitWeight)
            cost = float(current @ current)
            for _ in range(20):                                # Levenberg-Marquardt, numeric Jacobian
                jacobian = np.empty((len(current), len(shape)))
                for k in range(len(shape)):
                    step = np.zeros(len(shape))
                    step[k] = 1e-3 * max(abs(shape[k]), 1.0) if k < 6 else 1e-3
                    jacobian[:, k] = (residuals(shape + step, fitWeight)[0] - current) / step[k]
                normal = jacobian.T @ jacobian
                gradient = jacobian.T @ current
                improved = False
                for _ in range(8):
                    try:
                        delta = np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-9), -gradient)
                    except np.linalg.LinAlgError:
                        break
                    trial = np.clip(shape + delta, lower, upper)
                    trialResiduals, trialR = residuals(trial, fitWeight)
                    trialCost = float(trialResiduals @ trialResiduals)
                    if trialCost < cost:
                        shape, current, r, damping = trial, trialResiduals, trialR, damping / 3.0
                        improved = cost - trialCost > 1e-6 * cost
                        cost = trialCost
                        break
                    damping *= 4.0
                if not improved:
                    break
            scale = max(1.4826 * float(np.median(np.abs(r))), 2.0)
            fitWeight = base * np.clip(1.0 - (r / (3.0 * scale)) ** 2, 0.0, 1.0) ** 2
        return shape, r, fitWeight, scale

    # first with the axes along the cardiac frame, then tilted. With part of the apical cap seen the tilt
    # is kept whenever it fits better; with (nearly) none of it seen only where it fits the walls clearly
    # better: there a tilt trades off against the long axis and can carry the apex far off (a straight tube)
    capSeen = float(np.mean(seen[grid.nCyl:])) if grid.nRows > grid.nCyl else 0.0
    gain = 0.0 if capSeen >= SHAPE_TILT_CAP else SHAPE_TILT_GAIN
    level = lower.copy(), upper.copy()
    level[0][6:], level[1][6:] = 0.0, 0.0
    shape, r, fitWeight, scale = solve(shape, *level)
    if SHAPE_MAX_TILT > 0:
        tilted, tiltedR, tiltedWeight, tiltedScale = solve(shape, lower, upper)
        straightCost = float(fitWeight @ np.minimum(r ** 2, (3.0 * scale) ** 2))
        tiltedCost = float(fitWeight @ np.minimum(tiltedR ** 2, (3.0 * scale) ** 2))
        if tiltedCost < (1.0 - gain) * straightCost:
            shape, r, fitWeight, scale = tilted, tiltedR, tiltedWeight, tiltedScale
    model = _ellipsoidDistance(shape, origins, directions)
    if not np.all(np.isfinite(model[grid.nCyl:])):
        return None, None
    model = np.minimum(np.where(np.isfinite(model), model, distance), reach)
    off = (np.abs(distance - model) > max(float(tolerance), 3.0 * scale)) & (w > 0)
    # what the ellipsoid misses on the walls it was fitted to, spread smoothly and fading out inside a
    # defect (not on the walls at its edge: their bulging peaks would carry the bulge into it)
    w = np.where(off | ~seen, 0.0, w)
    residual = np.where(w > 0, distance - model, 0.0)
    kernel = _gaussianKernel(SHAPE_RESIDUAL_SIGMA)
    spread = _convolve2d(residual * w, kernel)
    norm = _convolve2d(w, kernel)
    level = max(float(np.percentile(norm[norm > 0], 50)), 1e-9) if np.any(norm > 0) else 1.0
    correction = spread / (norm + SHAPE_FADE * level)
    # the grid smoothing narrows to nothing in mm toward the apex, where the rays of a row converge (and
    # those of the last row coincide): there the field is smoothed over the same mm as at the equator
    correction = _poleSmooth(grid, correction, np.ones_like(correction), SHAPE_RESIDUAL_SIGMA, model)
    return np.maximum(model + correction, 1.0), off


def _circularMedian(values, size):
    """Median over a window of angles (wrapped), row by row, then over three rows."""
    half = size // 2
    padded = np.concatenate([values[:, -half:], values, values[:, :half]], axis=1)
    stack = np.stack([padded[:, i:i + values.shape[1]] for i in range(size)], axis=0)
    byAngle = np.median(stack, axis=0)
    padded = np.concatenate([byAngle[:1], byAngle, byAngle[-1:]], axis=0)
    return np.median(np.stack([padded[i:i + values.shape[0]] for i in range(3)], axis=0), axis=0)


def _smoothMap(values, weights, sigma):
    """Count-weighted Gaussian smoothing over the [rows, angles] grid (wrapped around the axis,
    mirrored at the base and the apex)."""
    if sigma <= 0:
        return values.copy()
    kernel = _gaussianKernel(sigma)
    weighted = _convolve2d(values * weights, kernel)
    norm = _convolve2d(weights, kernel)
    out = np.where(norm > 1e-12, weighted / np.maximum(norm, 1e-12), values)
    return out


def _smoothFilled(values, weights, sigma, counts=None):
    """_smoothMap where the walls are seen; a ray far from any seen wall (inside a wide defect, where the
    profile width is that of noise) takes the weighted median of all seen walls instead of its own value.
    Walls in or next to a defect (_clearWalls) do not count as seen: the profile of a wall at the edge of a
    defect is cut short by it and looks thin."""
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    w = np.where(_clearWalls(w, counts), w, 0.0)
    if not np.any(w > 0):
        return _smoothMap(values, np.maximum(w, 1e-6), sigma)
    order = np.argsort(values, axis=None)
    cumulative = np.cumsum(w.ravel()[order])
    overall = float(values.ravel()[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])
    kernel = _gaussianKernel(max(float(sigma), 0.5))
    norm = _convolve2d(w, kernel)
    level = max(float(np.percentile(norm[norm > 0], 50)), 1e-9)
    local = _convolve2d(values * w, kernel) / np.maximum(norm, 1e-12)
    share = norm / (norm + SHAPE_FADE * 0.2 * level)
    return np.where(norm > 1e-12, share * local + (1.0 - share) * overall, overall)


def _smoothSurface(values, weights, sigma):
    """_smoothMap for a surface distance map, without flattening its shape: the weighted mean of
    every row (the LV profile from base to apex, a maximum at the apex) is kept nearly as it is and
    only the deviations from it are smoothed. Plain smoothing of the distances pulls the apex in
    (a maximum is lowered by any blur), more so the stronger the smoothing."""
    trend = (values * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-12)
    rows = len(trend)
    kernel = _gaussianKernel(0.75)
    k = len(kernel) // 2
    padded = np.concatenate([trend[k:0:-1], trend, trend[-2:-k - 2:-1]]) if rows > k + 1 else trend
    if rows > k + 1:
        trend = np.convolve(padded, kernel, mode="valid")
    return trend[:, None] + _smoothMap(values - trend[:, None], weights, sigma)


def _gaussianKernel(sigma):
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _convolve2d(values, kernel):
    radius = len(kernel) // 2
    # angles wrap around
    padded = np.concatenate([values[:, -radius:], values, values[:, :radius]], axis=1)
    out = sum(kernel[i] * padded[:, i:i + values.shape[1]] for i in range(len(kernel)))
    # rows mirror at both ends
    padded = np.concatenate([out[radius:0:-1], out, out[-2:-radius - 2:-1]], axis=0)
    return sum(kernel[i] * padded[i:i + values.shape[0]] for i in range(len(kernel)))


def _bilinear(grid, row, col):
    rows, cols = grid.shape
    row = np.clip(row, 0.0, rows - 1.0)
    r0 = np.clip(np.floor(row).astype(int), 0, rows - 2)
    fr = row - r0
    c0 = np.floor(col).astype(int) % cols
    c1 = (c0 + 1) % cols
    fc = col - np.floor(col)
    top = (1 - fc) * grid[r0, c0] + fc * grid[r0, c1]
    bottom = (1 - fc) * grid[r0 + 1, c0] + fc * grid[r0 + 1, c1]
    return (1 - fr) * top + fr * bottom
