"""Left ventricular wall detection by count-profile sampling in the cardiac frame (numpy only).

The approach follows the classic quantitative SPECT programs (Emory, QPS/QGS) rather than voxel
labelling:

1. Hybrid sampling. Rays leave the LV long axis: perpendicular to it from the base up to the LV
   center (cylindrical part), and from the LV center toward the apex (spherical cap, polar angle
   90 deg at the center down to 0 deg on the axis). Every ray is a row (base -> apex) and column
   (angle around the axis) of one grid, so all results are 2D maps over the wall.
2. Mid-wall. The count peak along each ray, refined to sub-sample precision. Iterated a few times:
   the peaks are smoothed over the grid with count weights, so rays through a perfusion defect (few
   counts) follow their neighbours instead of jumping to noise, and the next search is limited to a
   window around the smoothed surface, which keeps the liver and gut out.
3. Base. Along every column of rays (one angle around the axis), going from the LV center toward
   the base, the wall ends where its counts fall below a fraction of that column's own wall counts
   (so the dim atria and great vessels behind the valves are not taken as wall, and a perfusion
   defect does not end its column early relative to the others). A plane is fitted through these
   end points (robust least squares, tilt limited), so the valve plane may be oblique to the long
   axis. It can be moved along the axis and tilted afterwards (valve offset and tilt).
4. Endo- and epicardium. Mid-wall -/+ half the wall thickness, the thickness taken from the width
   (FWHM) of the count profile with the system resolution removed in quadrature, and limited to a
   plausible range.

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
    "smoothing": 2.0,          # Gaussian sigma of the surface smoothing, in grid samples (0 = none)
    "thicknessSmoothing": 3.0, # Gaussian sigma of the wall thickness map, in grid samples
    "baseFraction": 0.55,      # wall ends where its counts fall below this fraction of its column's wall counts
    "maxValveTilt": 40.0,      # deg, largest tilt of the valve plane against the short-axis plane
    "maxBaseNarrowing": 35.0,  # deg, steepest the wall may turn toward the axis on its way to the base
    "resolution": 12.0,        # system resolution (FWHM, mm) removed from the profile width
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

    def __init__(self, center, u, w, radius, params=None):
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
        nCyl = int(math.ceil(self.radius / self.dz)) + 1
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
    data), and "baseZ" (valve plane position along the axis, mm from the LV center, negative)."""
    grid = WallGrid(center, u, w, radius, params)
    p = grid.params
    origins, directions = grid.rays()
    points = origins[:, :, None, :] + grid.t[None, None, :, None] * directions[:, :, None, :]
    profiles = np.asarray(sample(points.reshape(-1, 3)), dtype=float).reshape(grid.nRows, grid.nTheta, len(grid.t))
    profiles = np.nan_to_num(profiles)
    if profiles.max() <= 0:
        raise ValueError("No counts around the LV center.")

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
        weight = (peakValue / max(peakValue.max(), 1e-9)) ** 2
        # A ray that caught another organ (liver, gut) lies far from its neighbours: no weight
        reference = _circularMedian(peakT, window)
        nearPole = _nearPoleRows(grid, peakT)
        reference[nearPole] = np.median(peakT[nearPole], axis=1)[:, None]   # rays there nearly coincide
        weight[np.abs(peakT - reference) > 0.3 * t0] = 0.0
        surface = _smoothSurface(np.where(weight > 0, peakT, reference), np.maximum(weight, 1e-6),
                                 p["smoothing"])
        surface = _poleSmooth(grid, surface, np.maximum(weight, 1e-6), p["smoothing"])
        lo = np.maximum(surface - half, 1.0)
        hi = np.minimum(surface + half, grid.t[-1])
    peakT, peakValue = _peaks(profiles, grid.t, lo, hi)
    weight = (peakValue / max(peakValue.max(), 1e-9)) ** 2
    reference = _circularMedian(peakT, window)
    nearPole = _nearPoleRows(grid, peakT)
    reference[nearPole] = np.median(peakT[nearPole], axis=1)[:, None]
    weight[np.abs(peakT - reference) > 0.3 * t0] = 0.0
    mid = _smoothSurface(np.where(weight > 0, peakT, reference), np.maximum(weight, 1e-6), 0.75 * p["smoothing"])
    mid = _poleSmooth(grid, mid, np.maximum(weight, 1e-6), p["smoothing"])
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
    physical = np.sqrt(np.maximum(fwhm ** 2 - float(p["resolution"]) ** 2, 0.0))
    thickness = _smoothMap(physical, np.maximum(weight, 1e-6), p["thicknessSmoothing"])
    thickness = np.clip(_poleSmooth(grid, thickness, np.maximum(weight, 1e-6), p["thicknessSmoothing"], mid),
                        p["minThickness"], p["maxThickness"])
    thickness = _holdBeyondBase(thickness, beyond)
    endo = _limitBasalNarrowing(grid, np.maximum(mid - 0.5 * thickness, 1.0), p["maxBaseNarrowing"])
    epi = _limitBasalNarrowing(grid, mid + 0.5 * thickness, p["maxBaseNarrowing"])

    return {"grid": grid, "mid": mid, "endo": endo, "epi": epi, "thickness": thickness, "peak": peak,
            "baseZ": baseZ, "detectedBaseZ": baseZ, "valvePlane": plane, "basePoints": basePoints,
            "rowCounts": rowCounts}


# ---------------------------------------------------------------------------
# Gated frames
# ---------------------------------------------------------------------------
GATE_DEFAULTS = {
    "maxWallShift": 25.0,      # mm, the mid-wall of a gate is searched this far from the static mid-wall
    "maxValveShift": 25.0,     # mm, the valve plane of a gate stays this close to the static one (base descent ~1 cm)
    "gateSmoothing": 2.0,      # Gaussian sigma of the gate surface smoothing (grid samples)
}


def fitGate(sample, reference, params=None):
    """LV wall of one gate of a gated study, with the static fit (reference, from fitWall) as prior.

    The rays are those of the static fit. Along every ray the mid-wall is the count peak within
    maxWallShift of the static mid-wall (smoothed with count weights, so rays through a defect follow
    their neighbours). The valve plane is fitted again as in fitWall, keeps the static tilt and moves
    at most maxValveShift along the axis. Endo- and epicardium keep the myocardial volume of the static
    fit locally (QGS-like): the wall thickness of a ray scales with (static mid / gate mid), squared on
    the apical cap. Returns a result dict like fitWall, so classify / labelGrid / wallStatistics apply."""
    grid = reference["grid"]
    p = grid.params
    g = dict(GATE_DEFAULTS, **(params or {}))
    origins, directions = grid.rays()
    points = origins[:, :, None, :] + grid.t[None, None, :, None] * directions[:, :, None, :]
    profiles = np.asarray(sample(points.reshape(-1, 3)), dtype=float).reshape(grid.nRows, grid.nTheta, len(grid.t))
    profiles = np.nan_to_num(profiles)
    if profiles.max() <= 0:
        raise ValueError("No counts around the LV center in this gate.")

    refMid = reference["mid"]
    shift = float(g["maxWallShift"])
    lo = np.maximum(refMid - shift, 1.0)
    hi = np.minimum(refMid + shift, grid.t[-1])
    sigma = float(g["gateSmoothing"])
    peakT, peakValue = _peaks(profiles, grid.t, lo, hi)
    weight = (peakValue / max(peakValue.max(), 1e-9)) ** 2
    # rays far from their neighbours (noise, a neighbouring organ inside the window) do not count
    reference2 = _circularMedian(peakT, _angleWindow(grid, 9))
    nearPole = _nearPoleRows(grid, peakT)
    reference2[nearPole] = np.median(peakT[nearPole], axis=1)[:, None]
    weight[np.abs(peakT - reference2) > 0.5 * shift] = 0.0
    mid = _smoothSurface(np.where(weight > 0, peakT, reference2), np.maximum(weight, 1e-6), sigma)
    mid = _poleSmooth(grid, mid, np.maximum(weight, 1e-6), sigma)
    mid = np.clip(mid, lo, hi)
    peak = _interp(profiles, grid.t, mid)

    a0, gw, gs = reference["valvePlane"]
    plane, basePoints = _fitValvePlane(grid, mid, peak, p, _cavityFill(profiles, grid.t, mid, peak))
    maxValve = float(g["maxValveShift"])
    a = float(np.clip(plane[0], a0 - maxValve, a0 + maxValve))
    plane = (a, gw, gs)
    beyond = _beyondBase(grid, mid, plane)
    mid = _limitBasalNarrowing(grid, _holdBeyondBase(mid, beyond), p["maxBaseNarrowing"])
    peak = _interp(profiles, grid.t, mid)

    ratio = np.clip(refMid / np.maximum(mid, 1.0), 0.5, 2.0)
    ratio[grid.nCyl:] = ratio[grid.nCyl:] ** 2
    thickness = np.clip(reference["thickness"] * ratio, p["minThickness"], 1.5 * p["maxThickness"])
    thickness = _holdBeyondBase(thickness, beyond)
    endo = _limitBasalNarrowing(grid, np.maximum(mid - 0.5 * thickness, 1.0), p["maxBaseNarrowing"])
    epi = _limitBasalNarrowing(grid, mid + 0.5 * thickness, p["maxBaseNarrowing"])
    return {"grid": grid, "mid": mid, "endo": endo, "epi": epi, "thickness": thickness, "peak": peak,
            "baseZ": a, "detectedBaseZ": a, "valvePlane": plane, "basePoints": basePoints,
            "rowCounts": np.median(peak[:grid.nCyl], axis=1), "gateWeight": weight}


def endSystoleDiastole(volumes):
    """(ED index, ES index) of a cardiac cycle: the frames with the largest and the smallest LV cavity
    (frames without a volume are skipped); None when fewer than two frames have one."""
    known = [(float(v), i) for i, v in enumerate(volumes) if v is not None and np.isfinite(v)]
    if len(known) < 2:
        return None
    return max(known)[1], min(known)[1]


def thickeningMap(gatePolars, ed, es, smoothing=1.5):
    """Count-based wall thickening polar map (partial volume effect: a thicker wall looks brighter),
    in % = (ES counts / ED counts - 1) x 100. The wall counts of every ray (polarMap of each frame)
    are fitted with the first harmonic over the cardiac cycle, so all frames count and the ED / ES
    values are less noisy than two single frames; then smoothed over the map (count weighted).
    Returns a polar map dict like polarMap (on the rows of the ED frame) with "values" in %."""
    stack = np.array([np.asarray(p["values"], dtype=float) for p in gatePolars])     # [frames, rows, angles]
    mean, edCounts, esCounts = _firstHarmonic(stack, ed, es)
    values = 100.0 * (esCounts / np.maximum(edCounts, 1e-9) - 1.0)
    weights = np.clip(mean / max(float(np.nanmax(mean)), 1e-9), 0.0, 1.0) ** 2
    values = np.clip(_smoothMap(np.nan_to_num(values), weights + 1e-6, smoothing), -100.0, 200.0)
    reference = gatePolars[ed]
    return {"values": values, "rho": np.asarray(reference["rho"]).copy(), "theta": np.asarray(reference["theta"]).copy()}


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


def motionMap(gateResults, ed, es, weights=None, smoothing=1.5):
    """Wall motion polar map (QGS-like endocardial excursion), in mm: how far the endocardium of every
    polar map ray moves from ED to ES. The rays of each frame run from its own valve plane to the apex
    (polarRays), so a basal point follows the valve descent and the longitudinal shortening counts too.
    The distance is signed: positive when the ES point lies inside the ED endocardium (toward the cavity),
    negative for outward (dyskinetic) motion. The endocardial points are fitted with the first harmonic
    over the cycle, like the thickening; weights (e.g. the mean wall counts of each ray) let rays through
    a defect follow their neighbours when the map is smoothed. The rays are those of the static fit
    (fixed LV center and axis), so a translation of the whole heart is not corrected.
    Returns a polar map dict like polarMap (rho of the ED frame) with "values" in mm."""
    points, rhos, directions = [], [], []
    for result in gateResults:
        origins, rays, endo, _, rho = polarRays(result)
        points.append(origins + endo[..., None] * rays)              # [rows, angles, 3]
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


def _cavityFill(profiles, t, mid, peak):
    """Lowest counts between the axis and the mid-wall along each ray, as a fraction of the wall peak:
    low inside the LV (dark cavity), near 1 where the ray runs through the atria or the valve ring."""
    inside = t[None, None, :] <= mid[..., None]
    lowest = np.where(inside, profiles, np.inf).min(axis=2)
    return np.clip(lowest / np.maximum(peak, 1e-9), 0.0, 1.0)


def _fitValvePlane(grid, mid, peak, p, fill=None):
    """Valve plane z = a + gw * x + gs * y in the cardiac frame (x along w, y along s), fitted
    through the end of the wall in every column. Along a column the wall ends where its counts fall
    below baseFraction of the column's wall counts, or where the cavity fills up (the rays leave the
    LV for the atria, whose counts can pass the count threshold in small hearts). Columns with little
    uptake (an infarct) have unreliable ends and count less in the fit.
    Returns ((a, gw, gs), end points [K, 3] frame)."""
    nCyl = grid.nCyl
    reference = np.percentile(peak, 85, axis=0)                  # each column's wall counts
    reference = _circularMedian(reference[None, :], _angleWindow(grid, 5))[0]
    lateral = np.percentile(peak[:nCyl], 85, axis=0)             # the same on the side wall only
    lateral = _circularMedian(lateral[None, :], _angleWindow(grid, 5))[0]
    columnWeight = np.clip(lateral / max(float(np.percentile(lateral, 90)), 1e-9), 0.0, 1.0) ** 2
    reference = np.maximum(reference, 0.4 * float(np.percentile(peak, 85)))
    threshold = p["baseFraction"] * reference
    cyl = peak[:nCyl]
    if fill is not None:
        # the column's own cavity contrast at the mid-ventricle, and how far it may fill before the wall ends
        midRows = (grid.z >= -12.0) & (grid.z <= -2.0)
        midFill = np.median(fill[:nCyl][midRows], axis=0) if np.any(midRows) else np.median(fill[:nCyl], axis=0)
        fillLimit = midFill + 0.5 * (1.0 - midFill)
        filled = fill[:nCyl] > fillLimit[None, :]
    else:
        filled = np.zeros(cyl.shape, dtype=bool)
    ends = []
    endWeights = []
    for col in range(grid.nTheta):
        values = cyl[:, col]
        below = (values < threshold[col]) | filled[:, col]
        # from the center row toward the base: first of two consecutive rows off the wall
        for row in range(nCyl - 2, 0, -1):
            if below[row] and below[row - 1]:
                v0, v1 = values[row], values[row + 1]
                f = np.clip((threshold[col] - v0) / (v1 - v0), 0.0, 1.0) if v1 > v0 and not filled[row, col] else 0.5
                z = grid.z[row] + f * grid.dz
                r = mid[row, col] + f * (mid[row + 1, col] - mid[row, col])
                ends.append((r * math.cos(grid.theta[col]), r * math.sin(grid.theta[col]), z))
                endWeights.append(columnWeight[col])
                break
    ends = np.array(ends).reshape(-1, 3)
    endWeights = np.maximum(np.array(endWeights), 1e-3)
    zMin, zMax = float(grid.z[0]), -2.0 * grid.dz
    if len(ends) < max(6, grid.nTheta // 4):
        # the counts never fall: the wall runs to the edge of the sampled region
        a = zMin if len(ends) == 0 else float(np.median(ends[:, 2]))
        return (float(np.clip(a, zMin, zMax)), 0.0, 0.0), ends
    design = np.column_stack([np.ones(len(ends)), ends[:, 0], ends[:, 1]])
    weights = endWeights.copy()
    for _ in range(5):                                           # iteratively reweighted (outlier columns out)
        sw = np.sqrt(weights)
        coef = np.linalg.lstsq(design * sw[:, None], ends[:, 2] * sw, rcond=None)[0]
        residual = ends[:, 2] - design @ coef
        scale = max(1.4826 * float(np.median(np.abs(residual))), 2.0)
        weights = endWeights * np.clip(1.0 - (residual / (3.0 * scale)) ** 2, 0.0, 1.0) ** 2
    a, gw, gs = (float(v) for v in coef)
    slope = math.hypot(gw, gs)
    maxSlope = math.tan(math.radians(p["maxValveTilt"]))
    if slope > maxSlope:
        gw, gs = gw * maxSlope / slope, gs * maxSlope / slope
        a = float(np.average(ends[:, 2] - gw * ends[:, 0] - gs * ends[:, 1], weights=weights + 1e-9))
    return (float(np.clip(a, zMin, zMax)), gw, gs), ends


def _beyondBase(grid, mid, plane):
    """Cylinder rows of each column on the atrial side of the valve plane, [rows, angles] bool."""
    a, gw, gs = plane
    slope = gw * np.cos(grid.theta) + gs * np.sin(grid.theta)
    beyond = np.zeros(mid.shape, dtype=bool)
    beyond[:grid.nCyl] = grid.z[:, None] < a + slope[None, :] * mid[:grid.nCyl]
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


def valvePlane(result):
    """(a, gw, gs) of the valve plane z = a + gw * x + gs * y after the user's offset and tilt."""
    grid = result["grid"]
    p = grid.params
    a, gw, gs = result["valvePlane"]
    gw = math.tan(np.clip(math.atan(gw) + math.radians(p["valveTiltAnterior"]), -1.2, 1.2))
    gs = math.tan(np.clip(math.atan(gs) + math.radians(p["valveTiltSeptal"]), -1.2, 1.2))
    return (float(np.clip(a + p["valveOffset"], grid.z[0], -grid.dz)), gw, gs)


def valvePlaneZ(result):
    """Valve plane position on the long axis after the user's offset (mm from the LV center)."""
    return valvePlane(result)[0]


def valveTilt(result):
    """Angle between the valve plane and the short-axis plane, degrees."""
    _, gw, gs = valvePlane(result)
    return math.degrees(math.atan(math.hypot(gw, gs)))


def planeZAt(result, x, y):
    """Valve plane z at frame coordinates (x along w, y along s), kept between the base end of the
    sampled region and just below the LV center."""
    grid = result["grid"]
    a, gw, gs = valvePlane(result)
    return np.clip(a + gw * x + gs * y, grid.z[0], -grid.dz)


# ---------------------------------------------------------------------------
# Labels and surfaces
# ---------------------------------------------------------------------------
LABEL_MYOCARDIUM = 1
LABEL_CAVITY = 2


def classify(result, points):
    """Label of world points: 0 outside, LABEL_MYOCARDIUM, LABEL_CAVITY (closed by the valve plane)."""
    grid = result["grid"]
    row, col, dist, z = grid.toGrid(points)
    endo = _bilinear(result["endo"], row, col)
    epi = _bilinear(result["epi"], row, col)
    labels = np.zeros(len(dist), dtype=np.uint8)
    d = np.asarray(points, dtype=float) - grid.center
    inside = (z >= planeZAt(result, d @ grid.w, d @ grid.s)) & (row <= grid.nRows - 1)
    labels[inside & (dist <= epi)] = LABEL_MYOCARDIUM
    labels[inside & (dist < endo)] = LABEL_CAVITY
    return labels


def surfaceGrid(result, which):
    """World points [rows, angles, 3] of one surface ("endo", "mid", "epi") from the valve plane to
    the apex. In every column the cylinder rows are spread evenly from the point where the surface
    meets the (possibly oblique) valve plane up to the LV center, so the first row lies in the plane."""
    grid = result["grid"]
    distance = result[which]
    nCyl = grid.nCyl
    a, gw, gs = valvePlane(result)
    slope = gw * np.cos(grid.theta) + gs * np.sin(grid.theta)          # plane z change per mm of radius
    cylinder = distance[:nCyl]
    zBase = np.full(grid.nTheta, a)
    for _ in range(6):                                                 # surface radius depends on z
        radius = np.array([np.interp(zBase[c], grid.z, cylinder[:, c]) for c in range(grid.nTheta)])
        zBase = np.clip(a + slope * radius, grid.z[0], -grid.dz)
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
    a, gw, gs = valvePlane(result)
    slope = gw * np.cos(grid.theta) + gs * np.sin(grid.theta)
    mid = result["mid"][:nCyl]
    zBase = np.full(grid.nTheta, a)
    for _ in range(6):
        radius = np.array([np.interp(zBase[c], grid.z, mid[:, c]) for c in range(grid.nTheta)])
        zBase = np.clip(a + slope * radius, grid.z[0], -grid.dz)
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
    (epi - distance), (distance - endo) and the height above the valve plane, in mm. Its zero level
    is the whole wall as one closed surface: epicardium, endocardium and the rim in the valve plane."""
    grid = result["grid"]
    _, _, endo, epi, dist, z = _wallRays(result, points)
    d = np.asarray(points, dtype=float) - grid.center
    plane = z - planeZAt(result, d @ grid.w, d @ grid.s)
    return np.minimum(np.minimum(epi - dist, dist - endo), plane)


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
    """Region at bullseye radius / angle: the segment number (1..17) or, for "vessels", the index
    of its territory in VESSELS plus one; 0 outside the bullseye."""
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
    """Mean polar map value of each region of an overlay (17 segments or the VESSELS territories)."""
    if overlay != "vessels":
        return segmentValues(polar, resolution)
    image, segments = bullseye(polar, resolution)
    values = []
    for vessel in VESSELS:
        inside = np.isin(segments, VESSEL_SEGMENTS[vessel])
        values.append(float(np.nanmean(image[inside])) if np.any(inside) else float("nan"))
    return np.array(values)


def regionLabelPlaces(overlay="segments"):
    return VESSEL_LABEL_PLACES if overlay == "vessels" else SEGMENT_LABEL_PLACES


def bullseye(polar, size):
    """Polar map as a [size, size] image (row 0 at the top; NaN outside the circle). Anterior up,
    septum left, lateral right, inferior down; apex in the center, valve plane at the rim.
    Also returns the segment number of every pixel."""
    coordinate = (np.arange(size) + 0.5) / size * 2.0 - 1.0
    x = coordinate[None, :]                     # right
    y = -coordinate[:, None]                    # up
    radius = np.sqrt(x ** 2 + y ** 2) * np.ones((size, size))
    angle = np.arctan2(-x, y) * np.ones((size, size))     # anterior (up) 0, septum (left) +90 deg
    image = _polarLookup(polar, np.clip(radius, 0.0, 1.0).ravel(), angle.ravel()).reshape(size, size)
    image[radius > 1.0] = np.nan
    return image, segmentIndex(radius, angle)


def segmentValues(polar, resolution=181):
    """Mean polar map value of each of the 17 segments (index 0 = segment 1), by bullseye area."""
    image, segments = bullseye(polar, resolution)
    return np.array([float(np.nanmean(image[segments == n])) if np.any(segments == n) else float("nan")
                     for n in range(1, 18)])


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
    a, gw, gs = valvePlane(result)
    radius = result["mid"][:grid.nCyl]
    planeZ = a + radius * (gw * np.cos(grid.theta) + gs * np.sin(grid.theta))[None, :]
    inWall = np.ones(result["mid"].shape, dtype=bool)
    inWall[:grid.nCyl] = grid.z[:, None] >= planeZ
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


def _fwhm(profiles, t, position, peak, reach):
    """Full width at half maximum around the peak, searched up to reach mm on each side."""
    step = 0.5 * (t[1] - t[0])
    halfValue = 0.5 * peak
    offsets = np.arange(step, reach + 1e-6, step)
    widths = np.zeros_like(position)
    for sign in (-1.0, 1.0):
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
        widths += found
    return widths


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
