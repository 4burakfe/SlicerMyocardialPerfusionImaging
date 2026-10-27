"""Helper package of the Myocardial Perfusion SPECT hub (MyocardialPerfusionSPECT) and its hidden
step modules (MPIDataRoles, MPIProcessing, MPIMyocardium, MPIReview, MPIGatedQC, MPIGated).

    geometry    vectors, matrices and ellipsoid masks of the cardiac frame (numpy/VTK only)
    resampling  B-spline / windowed sinc / linear reslicing and in-plane zoom (numpy/VTK only)
    views       layouts and slice/3D view helpers
    display     colour maps, in-memory slice images and the slice panel of the result screen
    myocardium  LV wall fit of the static study and of every gate frame (numpy only)
    gating      ECG gating information and per-frame count checks of Gated QC
    gatedpet    gated PET DICOM series: gate ordering, geometry, SUV (used by DICOMGatedPETPlugin)
    studies     study roles (stress, rest, ...) and gated sequence recognition/pairing
    session     active study shared by the modules, workflow steps
    case        case (patient) name and ID from the DICOM headers, mixed-patient warning
    widgets     coloured action buttons
    dicomexport saving: scene (.mrb), screenshot, screen and scene as DICOM (toolbar buttons)
    toolbar     workflow toolbar (shown while the hub is open)
    parameters  adjustable constants of every step (Advanced parameters, Save parameters / Restore defaults)
"""

import logging
import traceback

SUBMODULES = ("geometry", "myocardium", "gating", "gatedpet", "resampling", "views", "colortables", "display", "studies", "session", "case", "widgets", "dicomexport", "toolbar", "parameters")  # dependency order


def reloadInDeveloperMode():
    """With Slicer's developer mode on, pick up edits to this package when a module is reloaded
    (Reload only re-runs the module file, not the modules it imports)."""
    try:
        import slicer
        if not slicer.util.settingsValue("Developer/DeveloperMode", False, converter=slicer.util.toBool):
            return
        import importlib
        import sys
        # Reload each one before importing the next, so a module is never
        # imported against a stale (not yet reloaded) dependency
        for name in SUBMODULES:
            fullName = f"{__name__}.{name}"
            if fullName in sys.modules:
                importlib.reload(sys.modules[fullName])
            else:
                importlib.import_module(fullName)
    except Exception:
        logging.warning("Could not reload MyocardialPerfusionSPECTLib:\n" + traceback.format_exc())
