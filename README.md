# Myocardial Perfusion SPECT (3D Slicer)

Myocardial perfusion SPECT processing in 3D Slicer. The first step, available now, reorients a
reconstructed SPECT volume to the cardiac axes and shows the standard short axis, horizontal long
axis and vertical long axis slices (ASNC/SNMMI orientation).

Author: Burak Demir, MD, FEBNM

Works with volumes reconstructed by the [DICOM SPECT Recon](https://github.com/4burakfe/SlicerDICOMSpectRecon)
extension (static or summed-gate images); any scalar SPECT or PET volume can be used (PET studies,
e.g. Rb-82 or N-13 ammonia, are recognised by their DICOM modality and get a 7 mm system resolution in
Myocardium instead of 12 mm).

## Features

- **Studies**: the SPECT and PET volumes in the scene are listed with a study role (stress, rest,
  redistribution, viability, other) guessed from their names, and the gated sequence that goes with
  each (DICOM SPECT Recon gated output is recognised and paired automatically). Confirmed roles are
  stored on the nodes and saved with the scene. A gated-only study can be summed into a static volume.
- **Reorientation layout**: fixed anterior and left lateral MIPs to place a spherical heart ROI (the
  click snaps to the count-weighted depth), three adjustment views with one axis line each (long axis
  direction, elevation, rotation) and a live SA / HLA / VLA preview. Stress and rest can be stacked.
- **Ellipsoid mask** aligned with the cardiac axes to hide extracardiac activity (liver, gut), applied
  to the preview and optionally to the created volumes.
- **Reoriented volumes** with chosen pixel size and slice thickness. Thick slices average the counts
  over the whole slab.
- **Interpolation**: cubic B-spline (default), Lanczos windowed sinc or linear. Linear interpolation
  of coarse SPECT voxels leaves faceted, blocky slices; the cubic B-spline is smooth and still passes
  through the original voxel values.
- **Slice panel**: rows of consecutive SA / HLA / VLA slices, rendered at screen resolution.

## Hub, steps and workflow toolbar

Only one module appears in Slicer's module list: *Nuclear Medicine > Myocardial Perfusion SPECT*. It shows
an overview (active study, step status) and the five workflow steps inside its own panel:

1. **Data roles**: studies in the scene, their roles (None leaves a volume out) and gated sequences. Study 1 (stress) and Study 2 (rest) come from the roles and can be changed at the top of every step.
2. **Processing**: the active study loads when the step opens. Click the left ventricle in the anterior or
   left lateral MIP on the left (the depth goes to the count-weighted center along the projection line;
   drag the yellow point to move the ROI and the orange point to resize it), align the three axis lines in
   the top row of slices, then **Create SA / HLA / VLA volumes**. With a Stress and a Rest study the
   **Stress + rest** screen stacks two of these screens, stress above rest, and creates the volumes of both.
3. **Myocardium** (prototype): opens after the volumes are created and detects the left ventricular wall
   automatically on the SA volume made in Processing, so the Processing mask (when on) keeps the liver and
   gut out; it runs again when the volumes are made again. Rays are cast from the LV center (cylindrical in the body, spherical at the apex); the
   count-weighted peak along each ray gives the mid-wall, the wall thickness comes from the profile width
   corrected for system resolution, and endo/epicardium are placed half a thickness either side. At every
   angle the wall ends where its counts fall below 55 % of that angle's wall counts or where the cavity fills up (the rays run into the atria),
   and the valve plane is
   fitted through these ends, so it can be oblique to the long axis; sliders shift and tilt it. Output:
   endo/epi/mid-wall models, a *Myocardium* and *LV cavity* segmentation, LV cavity volume, myocardial
   volume and mass. Overlays on the SA/HLA/VLA slices and a 3D view of the LV as one model (epicardium,
   endocardium and the rim in the valve plane). Layout *STRESS-REST* shows the stress study on the top row
   and the rest study below (SA, HLA, VLA, 3D each); *ACTIVE STUDY* only the active one.
4. **Perfusion**: the reoriented slices of the stress and rest studies (roles from Data roles; rendered
   once and kept in memory, optional myocardium contours, a min-max color window per study; PET-DICOM,
   Hel, PET-HotMetalBlue, PET-HotIron, Jotun, Vanaheim, Valhalla and Trugold color maps, plus Slicer's
   own PET-DICOM), polar maps (bullseye: the maximum counts across the detected wall, apex in the center
   and valve plane at the rim, with the coronary territories (LAD, RCA, LCX; default) or the 17 segments and their values in % of the map maximum) and 3D
   views of the LV with the same counts projected on its surface, both annotated with the LV cavity and
   myocardium volumes. Layouts: *STRESS PERF* / *REST PERF* (slices, polar map and 3D of one study),
   *STRESS SLICES* / *REST SLICES*, *STRESS-REST PERF* (stress and rest rows interleaved, both polar maps
   and 3D views), *STRESS-REST SLICES* and *STRESS REST POLAR* (stress, rest and rest - stress difference
   maps, each with its own window). With *Sync* on, the rest rows follow the stress rows, matched on the
   center of each study's segmented myocardium (slice and panning); off, each row scrolls on its own.
5. **Gated QC**: checks the gated study of Study 1 or Study 2 before the gated review. The myocardium
   is segmented in every gate frame, starting from the static wall of Myocardium: the mid-wall of each ray
   is searched near the static one, the valve plane is fitted again (static tilt, at most 25 mm away, the
   base descends about 1 cm in systole). By default every frame keeps the myocardial volume of
   end-systole, where the wall is thickest and brightest: the ES wall thickness comes from its own count
   profiles, with the image blur estimated from the frames rather than assumed (the summed static image
   is blurred by the motion and overestimates the wall, which collapses the cavity at ES and inflates the
   EF). Selectable instead: the ED volume as reference (QGS-like) or a per-ray wall volume taken from the
   static fit. Every frame is resliced to the SA / HLA / VLA grids of Processing and shown in three rows,
   one tile per frame, all at the same slice (mouse wheel over a row scrolls), with its contours and one
   color scale for all frames (or each frame on its own). Top right: counts per frame in % of the mean
   (projections, image, segmented myocardium; frames beyond ±10 % are ringed). Bottom right: the ECG gating
   information stored by the DICOM SPECT Recon extension (accepted R-R window, accepted and rejected
   beats, heart rate, frame width, R-R per projection view when the scanner stores it, e.g. Mediso),
   a per-frame table with the LV cavity volume and a preliminary EF, and the QC findings. The per-frame
   results are kept on the gated sequence (MPI.Gate* attributes).
6. **Gated Review**: STRESS GATED shows the stress gated study. On the left are the SA / HLA / VLA rows of
   the gate frames as a cine, with the same grid, slices, colors and window as Perfusion (taken over when
   the screen opens) and each frame's endo- and epicardium. On the right are three polar maps side by side: perfusion,
   count-based wall thickening (ES / ED wall counts of every ray, first harmonic over the cycle, smoothed
   more than the others since a count ratio is noisy) and wall motion (mm the endocardium moves from ED
   to ES, valve descent included), each with its own window. Below them
   are the LV cavity volume of every frame (EDV, ESV, EF, the frame shown marked) and the beating 3D wall in
   its own color map (Hel by default), shown as a wireframe (default), surface or points, with the ED wall in white
   drawn on top as points, wireframe or surface with its own opacity. The cine has play /
   stop, a speed and a frame slider. The frames are segmented in Gated QC.

The steps are hidden modules (MPIDataRoles, MPIProcessing, MPIMyocardium, MPIReview, MPIGatedQC, MPIGated). While the hub is open, a workflow
toolbar at the top of the window shows the active study and the steps with their status; it is hidden in
all other modules.

## Layout

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
    studies.py                       study roles, gated sequence recognition and pairing
    session.py                       active study shared by the modules, workflow steps
    toolbar.py                       workflow toolbar
    widgets.py                       coloured action buttons
  Resources/Icons/
  Testing/
```

Without building, add the inner `MyocardialPerfusionSPECT` folder to *Edit > Application Settings >
Modules > Additional module paths* and restart Slicer. Tests: enable developer mode and press
**Reload and Test** in each module.
