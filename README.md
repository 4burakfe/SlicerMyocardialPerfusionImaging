# Tarvos - Myocardial Perfusion Imaging (3D Slicer)

Tarvos is a 3D Slicer extension for myocardial perfusion SPECT (and PET). It takes reconstructed
stress, rest and gated studies through the whole reading workflow in one place:

1. sort out which volume is stress, rest or gated, and whose exam it is,
2. reorient each study to the cardiac axes (short axis, horizontal and vertical long axis),
3. detect the left ventricular wall automatically,
4. review stress and rest perfusion as slices, polar maps and 3D views, with TID,
5. check the gated frames (counts, ECG gating, segmentation),
6. review the gated study as a cine with thickening and motion maps, LV volumes and EF.

Author: Burak Demir, MD, FEBNM · License: [MIT](LICENSE)

> **Intended use.** Tarvos is research software. It is not a certified medical device and must not be
> the sole basis for a clinical decision; compare its results with your clinical software.

## Contents

- [Highlights](#highlights)
- [Installation](#installation)
- [Quick start](#quick-start)
- [The interface: hub and workflow toolbar](#the-interface-hub-and-workflow-toolbar)
- [The modules](#the-modules)
  - [1. Data roles](#1-data-roles)
  - [2. Processing](#2-processing)
  - [3. Myocardium](#3-myocardium)
  - [4. Perfusion](#4-perfusion)
  - [5. Gated QC](#5-gated-qc)
  - [6. Gated Review](#6-gated-review)
- [Parameters](#parameters)
- [Saved scenes](#saved-scenes)
- [For developers](#for-developers)

## Highlights

- **One module, one workflow.** A single entry in Slicer's module list (*Nuclear Medicine > Tarvos -
  Myocardial Perfusion Imaging*) holds six steps, with a workflow toolbar that shows each step's status
  and the most used controls of the step on screen.
- **Studies sorted for you.** Stress, rest, redistribution and viability roles are guessed from the
  names (English and Turkish keywords) and gated sequences are paired with their static study. The
  patient name and ID are read from DICOM, and you are warned when the studies belong to more than one
  patient.
- **Standard reorientation.** ASNC/SNMMI SA / HLA / VLA slices from three axis lines, with an ellipsoid
  mask against liver and gut activity and sharp Lanczos (windowed sinc) reslicing.
- **Automatic LV wall.** Endocardium, epicardium, mid-wall and an oblique valve plane, LV cavity volume,
  myocardial volume and mass, as models and a segmentation.
- **Perfusion review.** Stress and rest slices side by side or interleaved, polar maps with coronary
  territories or the 17 AHA segments, rest - stress difference maps, 3D uptake views, and the TID ratio.
- **Gated review.** Per-frame segmentation and QC, a cine of every slice, count-based thickening and
  wall motion polar maps, LV volume curve with EDV, ESV and EF, and a beating 3D wall.
- **Dual monitor layouts** put the slices on the main screen and the plots on a second one.
- **Nuclear medicine color maps**: PET-DICOM, Hel, PET-HotMetalBlue, PET-HotIron, Jotun, Vanaheim,
  Valhalla and Trugold.
- **Adjustable and repeatable.** Every step has *Advanced parameters*, *Save parameters* and *Restore
  defaults*; everything is saved with the Slicer scene.

## Installation

Requirements: 3D Slicer 5.x. Nothing else needs installing; the extension uses the numpy, VTK and Qt that come
with Slicer.

- **From source, without building:** download or clone this repository, add its inner
  `MyocardialPerfusionSPECT` folder in *Edit > Application Settings > Modules > Additional module
  paths*, and restart Slicer.
- **Built as an extension:** the top-level `CMakeLists.txt` is a standard Slicer extension.

Recommended companion: the [DICOM SPECT Recon](https://github.com/4burakfe/SlicerDICOMSpectRecon)
extension reconstructs SPECT projections in Slicer (static and gated). Tarvos recognises its output,
pairs the gated sequence with its static image automatically and reads the ECG gating information it
stores, which Gated QC shows.

Any reconstructed scalar SPECT or PET volume can be used, for example loaded from DICOM or NRRD. PET
studies (Rb-82, N-13 ammonia) are recognised by their DICOM modality and get a 7 mm system resolution
in Myocardium instead of 12 mm for SPECT.

## Quick start

1. Load the reconstructed stress and rest studies (and their gated sequences) into Slicer.
2. Open *Nuclear Medicine > Tarvos - Myocardial Perfusion Imaging*. The workflow toolbar appears at the
   top of the window.
3. **Data roles**: check the roles (pick one to confirm it, None leaves a volume out) and the case name
   and ID, then press **Process ›**.
4. **Processing**: click the left ventricle in a MIP, align the three axis lines and press **Create SA /
   HLA / VLA volumes**. With stress and rest, the *Stress + rest* screen does both at once.
5. **Myocardium** detects the wall on its own when it opens. Check the contours and press
   **Perfusion ›**.
6. **Perfusion**: read the slices and polar maps, then press **Gated QC ›**.
7. **Gated QC** segments every frame; check the counts and gating, then press **Gated Review ›**.
8. **Gated Review**: play the cine and read the thickening, motion and EF.

## The interface: hub and workflow toolbar

**Hub.** The listed module is the hub. Its overview page shows the active study and the six steps with
their state (done, available, locked) and an *Open* button for each. The steps are hidden modules whose
panels the hub shows in its own panel, so you never need to look for them in the module list.

**Workflow toolbar.** While the hub is the current module, a toolbar at the top of the Slicer window
shows:

- the Tarvos logo (back to the overview),
- the case (patient name · ID), in orange with ⚠ when the studies belong to more than one patient; a
  click opens Data roles,
- the six steps with a badge for their state,
- *Hide panel*, which hides Slicer's module panel to give the views the room (it comes back when you
  leave the module).

A second row holds the most used controls of the step on screen, so you can work with the panel hidden:

| Step | Controls in the toolbar |
|---|---|
| Processing | Place ROI, Reset axes, Mask, Create SA / HLA / VLA |
| Myocardium | Detect, Mask, the surfaces shown, layouts |
| Perfusion | Max % (top of the slice window), FOV, Contours, Sync, Vessels / 17 seg, Scores, layouts |
| Gated QC | Segment again, Max %, All frames / Each frame, Contours, the study |
| Gated Review | the Perfusion controls plus Play and All frames / Each frame |

The main actions (Place ROI, Create, Detect, Segment again, Play) have the colors of their panel buttons.
*Mask* in Myocardium makes the SA / HLA / VLA of the studies on screen again with the Processing
ellipsoid mask (or without it), in the frame they were made in, and detects their walls again.
*All frames* puts every gate frame of a study on one color scale, so the count changes over the cycle
show; *Each frame* scales every frame to its own maximum.

Only the layouts of the studies that are ready appear: the stress-rest ones when both are ready, else
those of the one study. Perfusion and Gated Review open on the layout with slices and plots of the
studies there (stress-rest when both are ready, the dual monitor one when a second screen is connected);
a layout picked by hand is kept while the same studies are ready.

**Study 1 and Study 2.** The two-study screens show Study 1 (stress, top or first) and Study 2 (rest,
bottom or second). They are picked from the roles and can be changed at the top of every step; *⇅*
swaps them and *By role* picks them from the roles again.

## The modules

### 1. Data roles

*Which volume is which, and whose exam it is.*

- **Case.** Patient name and ID, read automatically from the DICOM headers of the studies (from the
  subject hierarchy, else the DICOM database). You can type over them; typed values are kept until you
  press *Read from DICOM*. When the studies carry different names or IDs, an orange warning lists which
  volumes belong to whom, and a message box appears once. Volumes set to None are left out of this check,
  so setting another patient's volume to None clears the warning.
- **Studies.** Every static (or summed-gate) SPECT or PET volume in the scene, with its role (Stress,
  Rest, Redistribution, Viability, Other, or None to leave it out) and its gated sequence. Roles in
  italics are guesses from the volume, series and study names; picking a role confirms it. The selected
  row is the active study of all steps.
- **Gated sequences** are recognised as sequences of 4 to 32 volumes with the same grid (DICOM SPECT Recon
  gates always), and paired with the static study from the same reconstruction, else the same subject
  hierarchy study and the longest common name, else the same grid. The pairing can be changed per row.
- **Create static from gates** sums the frames of a gated-only study into a static volume.

Roles, pairings and the case are stored on the nodes (`MPI.StudyRole`, `MPI.GatedSequence`, and
`MPI.CaseName` / `MPI.CaseID` on the session node) and saved with the scene.

### 2. Processing

*Reorientation to the cardiac axes.*

1. The active study loads when the step opens. Click the left ventricle in the anterior or left lateral
   MIP on the left: the click snaps to the count-weighted center along the projection line. Drag the
   yellow point to move the spherical heart ROI and the orange point to resize it.
2. Align the three axis lines in the top row of slices (long axis direction, elevation, rotation). The
   bottom row is a live SA / HLA / VLA preview.
3. Press **Create SA / HLA / VLA volumes**.

Options:

- **Ellipsoid mask** aligned with the cardiac axes hides extracardiac activity (liver, gut), in the
  preview and optionally in the created volumes; Myocardium then detects the wall on the masked volume.
- **Pixel size and slice thickness** of the reoriented volumes; thick slices average the counts over the
  whole slab.
- **Interpolation**: Lanczos windowed sinc (default), cubic B-spline or linear. Lanczos is the sharpest
  and can ring slightly next to hot edges; the cubic B-spline is smooth and still passes through the
  original voxel values; linear interpolation of coarse SPECT voxels leaves faceted, blocky slices. The
  slices in Perfusion, Gated QC and Gated Review are resized to the screen with the same method.
- **Stress + rest** stacks two of these screens, stress above rest, and creates the volumes of both.

The slices follow the ASNC/SNMMI display standard: SA from apex to base, HLA from inferior to anterior,
VLA from septum to lateral.

### 3. Myocardium

*Automatic detection of the left ventricular wall* (prototype).

The step runs on its own when it opens after the volumes are created, and again when they are made
again. It works on the SA volume from Processing, so the Processing mask (when on) keeps the liver and
gut out.

What you get:

- endocardium, epicardium and mid-wall models, a *Myocardium* and an *LV cavity* segmentation,
- LV cavity volume, myocardial volume and mass,
- overlays on the SA / HLA / VLA slices and a 3D view of the LV as one model (epicardium, endocardium and
  the rim in the valve plane), shown orthographic, without perspective distortion,
- layouts *STRESS-REST* (stress study on the top row, rest below: SA, HLA, VLA and 3D each) and *ACTIVE
  STUDY*.

How it works:

- Rays are cast from the LV center, cylindrical in the body and spherical at the apex. The peak along
  each ray gives the mid-wall. How much a ray's own peak counts depends on how clearly the wall stands
  out: its counts above the blood pool of the cavity, against those of the typical (median) wall. A hot
  spot such as a bright lateral wall raises only its own rays, so a dimmer but plainly seen wall keeps
  its own peak. A ray far from its neighbours, or from the robust circle through its row (when a wide
  defect lets its neighbours share a wrong peak), follows the others.
- Rays through a deep defect have no peak of their own. They follow a shape model of the whole LV: a
  truncated ellipsoid fitted to the walls that are seen (the border of the defect left out, where the
  blur pulls the peaks off the wall). Its long axis may tilt up to 25 deg against the LV axis when part
  of the apex is seen (with none of it seen, only when that fits the walls clearly better). The surface
  is the ellipsoid plus the peaks' smoothed deviations from it, which fade out into a defect, so the
  walls next to a defect keep the ellipsoid's curve instead of the level of their neighbours. A defect,
  also one over the apex, takes the curve of the walls around it instead of being bridged straight
  across. The wall thickness inside a defect comes from the walls that are seen. *Shape model through
  defects* (Advanced parameters, Defects) turns it off.
- The wall thickness comes from the profile width corrected for the system resolution, and the endo-
  and epicardium are placed half a thickness on either side.
- At every angle the wall ends where its counts fall below 55 % of that angle's wall counts, or where
  the cavity fills up (the rays run into the atria). The valve plane is fitted through these ends, so it
  can be oblique to the long axis; sliders shift and tilt it.
- Only the long axis and its direction are taken from Processing. After a first fit the ray center is
  moved along the axis between the valve plane and the apex (toward the apex when a steep valve plane
  needs room) and the wall is fitted again, so the result does not depend on where the LV center was
  placed.
- Surface smoothing is 1.25 by default; higher values bridge defects more.

### 4. Perfusion

*Stress and rest perfusion: slices, polar maps and 3D.*

- **Slices** of the reoriented stress and rest studies, rendered once and kept in memory, with optional
  myocardium contours and a min-max color window per study.
- **Polar maps** (bullseye): the maximum counts across the detected wall, apex in the center and valve
  plane at the rim, with the coronary territories (LAD, RCA, LCX; default) or the 17 segments and their
  values in % of the map maximum.
- **3D views** of the LV with the same counts projected on its surface.
- The polar maps and 3D views are annotated with the LV cavity and myocardium volumes. Layouts with both
  studies write the **TID ratio** (transient ischemic dilation: stress / rest LV cavity volume) in the
  upper left corner of the stress polar map and 3D view.

Layouts:

| Layout | Shows |
|---|---|
| STRESS PERF / REST PERF | slices, polar map and 3D of one study |
| STRESS SLICES / REST SLICES | slices only |
| STRESS-REST PERF | stress and rest rows interleaved (with several grid rows, every stress row is followed by the same rest row), both polar maps and 3D views |
| STRESS-REST SLICES | the interleaved rows only |
| STRESS REST POLAR | stress, rest and rest - stress difference maps, each with its own window |
| STRESS DUAL / REST DUAL / STRESS-REST DUAL | the slices on the main window; the polar maps, difference map and 3D views in a second window that opens maximized on the second monitor |

Navigation:

- Every tile shows the same field of view (13 cm by default) centered on the LV of its own study, so
  stress and rest show at the same zoom.
- **Fit to the myocardium** (on by default) picks the slices holding the segmented myocardium and the
  grid that gives the largest tiles for the screen (the fewest rows), then fills the rows with the
  neighbouring slices as far as tiles of that size fit, again one second after every resize. The mouse
  wheel over a row moves that row and keeps the fit (past the last slice, as black tiles, until 8 slices
  are left); changing the grid or the slices by hand turns it off.
- **Sync** makes the rest rows follow the stress rows, matched on the center of each study's segmented
  myocardium; off, each row scrolls on its own.
- **Link 3D views** (on by default): rotating, zooming or panning the stress or rest 3D view moves the
  other the same way, around its own heart.

### 5. Gated QC

*Segmentation of every gate frame and quality checks, before the gated review.*

For the gated study of Study 1 or Study 2:

- **Frame segmentation.** The myocardium is segmented in every frame, starting from the static wall of
  Myocardium. The mid-wall of each ray is searched near the static one; rays through a defect (dim in the
  static image) are searched only within a third of that distance and follow the motion of the walls
  around them, so noise or a neighbouring organ in an infarct does not pull the wall. The valve plane is
  fitted again with the static tilt, at most 25 mm away (the base descends about 1 cm in systole).
- **Frames on screen.** Every frame is resliced to the SA / HLA / VLA grids of Processing and shown in
  three rows, one tile per frame, all at the same slice (the mouse wheel over a row scrolls), with its
  contours and one color scale for all frames (or each frame on its own).
- **Counts per frame** in % of the mean, from the projections, the image and the segmented myocardium;
  frames beyond ±10 % are ringed.
- **ECG gating information** stored by DICOM SPECT Recon: accepted R-R window, accepted and rejected
  beats, heart rate, frame width, and R-R per projection view when the scanner stores it (e.g. Mediso).
- A per-frame table with the LV cavity volume and a preliminary EF, and the QC findings. The per-frame
  results are kept on the gated sequence (`MPI.Gate*` attributes).

Wall thickness of the frames (*Wall thickness* in the advanced section):

- **Constant myocardial volume, ES reference** (default): every frame keeps the myocardial volume of
  end-systole, where the wall is thickest and brightest, with the ES wall thickness from its own count
  profiles and the image blur estimated from the frames.
- **Constant myocardial volume, ED reference**: the same with end-diastole as the reference (QGS-like).
- **Per ray, from the static wall**: every ray keeps its wall volume of the static fit. The summed static
  image blurs the wall over the cycle, so the walls come out thicker and the EF higher, most in small
  hearts.
- **Mid-wall only**: only the mid-wall (the count peak, the surface a gate shows best) is taken from each
  frame and every ray keeps its end-diastolic wall thickness, so the endocardium follows the mid-wall at
  a fixed distance and no thickening model enters the volumes. It reads lower than the other models in
  hearts that thicken well, and it does not overestimate when the gated reconstruction is too smooth to
  show the thickening (gate profiles wider than about 20 mm keep the same width over the whole cycle, as
  with a Butterworth 0.30 cycles/cm gated filter).

In the constant-volume models, rays through a defect keep the wall thickness of the static fit in every
frame: scar does not thicken, and its gate profiles show no wall to measure. A ray counts as defect by
its visibility in the static image (counts above the cavity's blood pool against the typical wall's):
no wall below 0.15, full wall from 0.60 (*Defect: no wall below* / *full wall from*). Surface smoothing of the
frames is 1.25 by default.

### 6. Gated Review

*The gated study as a cine, with function maps, volumes and EF.*

STRESS GATED shows the stress gated study:

- **Left:** the SA / HLA / VLA rows of the gate frames as a cine, with the same grid, slices, colors and
  window as Perfusion (taken over when the screen opens), one color scale for all frames (or each frame
  on its own, *Scale*) and each frame's endo- and epicardium. The
  slices use the field of view, *Fit to the myocardium* and *Sync stress and rest* of Perfusion.
- **Right, three polar maps** side by side, each with its own window:
  - perfusion,
  - **wall thickening**, count-based (ES / ED wall counts of every ray, first harmonic over the cycle),
    smoothed more than the others since a count ratio is noisy. *Thickening map* paints it as the mean
    of each of the five walls (default), the mean of each of the 17 AHA segments, or per pixel. The five
    walls are drawn on the bullseye itself, not grouped from the segments: the apex is the central disk
    (the size of segment 17) and the rest is cut by an X into anterior, septal, inferior and lateral
    quarters (borders at 45, 135, 225 and 315 deg). With an overlay shown, the per-wall map draws these
    five walls and writes their scores instead of the coronary territories,
  - **wall motion**, mm the endocardium moves from ED to ES, valve descent included; outward
    (dyskinetic) motion below 0 goes from black at 0 to full turquoise at -4 mm (red with PET-DICOM, Jotun
    and Vanaheim).
- **Below them:** the LV cavity volume of every frame (EDV, ESV, EF, the frame shown marked; the curve
  starts at 0 ml) and the beating 3D wall, as a surface (default), wireframe or points, with the ED wall
  drawn on top in white as a faint wireframe (default, 3 % opacity), points or surface. *Color by* paints
  the wall with the perfusion (default; in its own color map, Inferno by default, over the perfusion
  window), the wall motion from ED to ES at every point (the motion map's values, over the motion
  window) or one color of your choice; *Wall opacity* sets how see-through it is (0.5 by default). The
  EF is written large in the cine 3D view. Settings saved earlier keep their values: *Restore defaults*
  under Advanced parameters brings in the new ones.
- The cine has play / stop, a speed and a frame slider.

REST GATED is the same for the rest study. STRESS-REST GATED shows the stress and rest rows interleaved,
with the thickening and motion maps and the 3D wall of each study (the perfusion map and the volume curve
are left out to make room); the two 3D views are linked as in Perfusion and the cine runs both together.

Every one of the three comes in five layouts:

| Layout | Shows |
|---|---|
| GATED | three representative slices per axis and all plots |
| SLICES | every slice, with the grid, fit and scrolling of Perfusion |
| PLOTS | only the plots (one row per study for stress-rest) |
| DUAL MONITOR | the GATED slices on the main window, all plots on a second window that opens maximized on the second monitor |
| DUAL FULL SLICES | the same with every slice |

The three representative slices spare the graphics card: apical, mid and basal for SA, inferior, mid and
anterior for HLA, septal, mid and lateral for VLA, at 25, 50 and 75 % of the myocardium found in
Myocardium. SA is top left, HLA bottom left and VLA bottom right; top right a static 3D view of the wall
(one per study) seen from anteroseptal, raised 30 degrees toward the base, with semi-transparent planes
at the slices shown (SA yellow, HLA magenta, VLA green).

## Parameters

Every step ends with **Advanced parameters** (collapsed) and two buttons. The advanced section lists the
constants of that step that change a result or a display, grouped, each with its built-in default in the
tooltip. Examples are the wall fit grid, valve and thickness limits, defect handling, QC thresholds, map
smoothing, contour and plane colors, the three-slice positions and cache sizes. A change applies at once
to the displays. Analysis parameters take effect the next time the step runs (Detect myocardium, Segment
frames again). Parameters marked in their tooltip need a Slicer restart.

- **Save parameters** keeps the step's own controls (color maps, windows, grid, smoothing, thickness
  model, ...) and its advanced parameters as the starting values from then on, also after restarting
  Slicer. The values go to the Slicer settings under `MyocardialPerfusionSPECT/Parameters/<step>`.
- **Restore defaults** sets them back to the built-in values and forgets the saved ones. Saved values
  win over new built-in defaults after an update, so press it once to pick those up.

Saved analysis parameters are applied as soon as the extension loads, so a step uses them even before
its panel is opened. The registry of all parameters is `MyocardialPerfusionSPECTLib/parameters.py`.

## Saved scenes

Saved scenes keep the roles, the case, the active study and study pair, the reoriented volumes (with
their axes), the myocardium segmentation, surfaces and statistics, and the per-frame gated values on the
gated sequence. The wall fit, the polar maps and the gate frame segmentation are made again from them
when a step needs them. Processing's interaction helpers (ROI center, rings, axis lines, mask handles and
preview) are not saved.

Closing the scene (or loading one) brings the hub back to its overview page and, when one of the
extension's layouts is shown, Slicer's Four-Up layout.

## For developers

```
CMakeLists.txt                       extension
MyocardialPerfusionSPECT/
  CMakeLists.txt                     hub and step modules
  MyocardialPerfusionSPECT.py        hub (the only listed module)
  MPIDataRoles.py                    1. Data roles (hidden)
  MPIProcessing.py                   2. Processing: reorientation logic, widget and tests (hidden)
  MPIMyocardium.py                   3. Myocardium: automatic LV wall detection (hidden)
  MPIReview.py                       4. Perfusion (hidden)
  MPIGatedQC.py                      5. Gated QC: per-frame segmentation, counts, gating checks (hidden)
  MPIGated.py                        6. Gated Review (hidden)
  MyocardialPerfusionSPECTLib/
    geometry.py                      cardiac frame vectors, matrices, ellipsoid masks
    myocardium.py                    LV wall fitting (rays, mid-wall, thickness, valve plane; per gate frame), no Slicer imports
    gating.py                        ECG gating information and per-frame count checks, no Slicer imports
    resampling.py                    B-spline / sinc / linear reslicing, in-plane zoom
    views.py                         layouts and view helpers
    display.py                       colour maps, the slice panel, the polar map panel and the Gated QC charts
    colortables.py                   the nuclear medicine color maps
    studies.py                       study roles, gated sequence recognition and pairing
    session.py                       active study shared by the modules, workflow steps
    case.py                          case (patient) name and ID from DICOM, mixed-patient warning
    toolbar.py                       workflow toolbar
    parameters.py                    adjustable constants of every step, Save parameters / Restore defaults
    widgets.py                       coloured action buttons, study pair selector, wrapping labels
  Resources/Icons/, Resources/Images/
  Testing/
colors/                              color map tables (CSV)
```

- The steps are hidden modules (MPIDataRoles, MPIProcessing, MPIMyocardium, MPIReview, MPIGatedQC,
  MPIGated); the hub shows their panels. Shared state (active study, Study 1 / 2, case) lives on a
  singleton scripted module node, so it is saved with the scene.
- With Slicer's developer mode on, *Reload* in a module also reloads the helper package.
- Tests: enable developer mode and press **Reload and Test** in each module.
