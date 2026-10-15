"""Helper package of the Myocardial Perfusion SPECT hub (MyocardialPerfusionSPECT) and its hidden
step modules (MPIDataRoles, MPIProcessing, MPIReview, MPIGated).

    geometry    vectors, matrices and ellipsoid masks of the cardiac frame (numpy/VTK only)
    resampling  B-spline / windowed sinc / linear reslicing and in-plane zoom (numpy/VTK only)
    views       layouts and slice/3D view helpers
    display     colour maps, in-memory slice images and the slice panel of the result screen
    studies     study roles (stress, rest, ...) and gated sequence recognition/pairing
    session     active study shared by the modules, workflow steps
    widgets     coloured action buttons
    toolbar     workflow toolbar (shown while the hub is open)
"""

import logging
import traceback

SUBMODULES = ("geometry", "myocardium", "resampling", "views", "colortables", "display", "studies", "session", "widgets", "toolbar")  # dependency order


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
