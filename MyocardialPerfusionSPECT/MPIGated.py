"""
Myocardial Perfusion SPECT - 6. Gated Review

STRESS GATED layout of Study 1 (stress), REST GATED the same of Study 2 (rest):
  left           SA / HLA / VLA rows of the gate frames as in Perfusion (same grid, slices, colors and
                 window), played as a cine: every tile shows the same frame, with that frame's
                 endo- and epicardium;
  top right      the perfusion, wall thickening (count-based: ES / ED wall counts of every ray, first
                 harmonic over the cycle; five wall means by default, or 17-segment means or per pixel)
                 and wall motion (mm the endocardium moves from ED to ES) polar maps side by side;
  bottom right   the LV cavity volume of every frame (EDV, ESV, EF; the frame shown is marked) next to
                 the cine 3D wall coloured by the perfusion (SSP style), as a surface, wireframe or points.
STRESS-REST GATED shows the stress and rest rows interleaved, with the thickening and motion maps and the
3D wall of each study (the perfusion map and the volume curve left out to make room). Every layout also
comes as SLICES (only the slice rows), PLOTS (only the plots, all of them) and DUAL MONITOR (the slices
on the main window, all plots in a second window on the second monitor).
The frames and their walls come from Gated QC (MPIGatedQC.py, segmented there on first use). The slice
and polar map settings are taken over from Perfusion (MPIReview.py) when the screen opens and can be
changed here; the cine has its own speed and play / stop.
"""

import os
import sys
import logging
import traceback

import numpy as np
import qt
import ctk
import vtk
import slicer
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleWidget,
                                           ScriptedLoadableModuleTest)
from slicer.util import VTKObservationMixin

_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from MyocardialPerfusionSPECTLib import session  # noqa: E402
from MyocardialPerfusionSPECTLib import parameters as parametersLib  # noqa: E402
from MyocardialPerfusionSPECTLib import myocardium as wallLib  # noqa: E402
from MyocardialPerfusionSPECTLib.resampling import DEFAULT_INTERPOLATION, zoomInPlane  # noqa: E402
from MyocardialPerfusionSPECTLib.views import (  # noqa: E402
    GATED_LAYOUT_IDS, ORIENTATIONS, ORIENTATION_TITLES, gatedLayoutXml, gatedPlotKinds, gatedTags,
    placeMonitor2Window, registerLayout, threeDWidgetByTag, CameraLink, secondaryScreen, notSaved)
from MyocardialPerfusionSPECTLib.geometry import worldToLocalMatrix  # noqa: E402
from MyocardialPerfusionSPECTLib.display import (  # noqa: E402
    DEFAULT_GRID_ROWS, DEFAULT_GRID_COLUMNS, COLORMAPS, colorTableArray, colorize, colorNodeForLabel,
    rgbToQImage, gatedReviewMosaicPanel, gatedReviewMosaic2Panel, volumeCurvePanel, volumeCurvePanelState, polarPanel, polarPanelState,
    existingPanel, DEFAULT_FOV_CM, REFIT_DELAY_MS, lastFirstSlice, fovCrop, cropStack, bestGrid, signedMotionTable)
from MyocardialPerfusionSPECTLib.widgets import breakable, narrowLabel, styleButton, StudyPairSelector  # noqa: E402

STRESS, REST = "stress", "rest"
SLOT_TITLES = {STRESS: "STRESS", REST: "REST"}
# key: (button text, studies, layout kind, tooltip); kinds as in views.gatedLayoutXml
MODES = {}
for _prefix, _slots, _title in (("stress", (STRESS,), "STRESS"), ("rest", (REST,), "REST"),
                                ("stressRest", (STRESS, REST), "STRESS-REST")):
    _what = "Stress" if _slots == (STRESS,) else "Rest" if _slots == (REST,) else "Stress and rest"
    _plots = ("thickening and motion polar maps and cine 3D walls" if len(_slots) > 1
              else "perfusion, thickening and motion polar maps, LV volume curve and cine 3D wall")
    MODES[_prefix + "Gated"] = (f"{_title} GATED", _slots, "full",
                                f"{_what} gated: apical, mid and basal cine slices of each axis, {_plots}")
    MODES[_prefix + "GatedSlices"] = (f"{_title} GATED SLICES", _slots, "slices", f"{_what} gated: only the cine slices")
    MODES[_prefix + "GatedPlots"] = (f"{_title} GATED PLOTS", _slots, "plots",
                                     f"{_what} gated: all polar maps, the LV volume curve and the cine 3D wall")
    MODES[_prefix + "GatedDual"] = (f"{_title} GATED DUAL", _slots, "dual",
                                    f"{_what} gated on two monitors: apical, mid and basal cine slices of each axis "
                                    "on this window, all plots in a second window on the second monitor")
    MODES[_prefix + "GatedDualFull"] = (f"{_title} GATED DUAL FULL SLICES", _slots, "dualFull",
                                        f"{_what} gated on two monitors: all cine slices on this window, all plots "
                                        "in a second window on the second monitor (heavier on the graphics card)")
MODE_GRID = tuple(tuple(p + k for k in ("Gated", "GatedSlices", "GatedPlots", "GatedDual", "GatedDualFull"))
                  for p in ("stress", "rest", "stressRest"))   # columns of layout buttons
KIND_LABELS = {"full": "GATED", "slices": "SLICES", "plots": "PLOTS", "dual": "DUAL MONITOR",
               "dualFull": "DUAL FULL SLICES"}
TOOLBAR_KIND_LABELS = {"full": "GATED", "slices": "SLICES", "plots": "PLOTS", "dual": "DUAL", "dualFull": "DUAL FULL"}
DUAL_KINDS = ("dual", "dualFull")          # slices on this window, plots on the second monitor
# Layouts showing only three representative slices per axis (a 3 x 3 grid per study): every frame of every
# slice is a lot of tiles for the graphics card. SA runs apex -> base, HLA inferior -> anterior,
# VLA septum -> lateral (MPIProcessing.orientationAxes).
REPRESENTATIVE_KINDS = ("full", "dual")   # as views.REPRESENTATIVE_LAYOUT_KINDS
REPRESENTATIVE_LABELS = {"SA": ("APICAL", "MID", "BASAL"), "HLA": ("INF", "MID", "ANT"),
                         "VLA": ("SEPT", "MID", "LAT")}
POLAR_KINDS = ("polar", "thickening", "motion")
EF_FONT_SIZE = 26   # px, the EF in the 3D views
_BIG_TEXT_ACTORS = {}   # 3D view node ID -> the text actor of its EF
MONITOR2_PLACE_DELAY_MS = 300     # the second window is placed after the layout manager made its views
MOSAIC_STATE = "_mpiGatedReviewMosaicState"
MOSAIC2_STATE = "_mpiGatedReviewMosaic2State"   # HLA and VLA rows of the three-slices-per-axis layouts
REPRESENTATIVE_FRACTIONS = (0.25, 0.5, 0.75)    # where the three slices lie along the myocardium of an axis
SLICE_PLANE_COLORS = {"SA": (0.99, 0.85, 0.21), "HLA": (0.85, 0.11, 0.38), "VLA": (0.26, 0.63, 0.28)}
SLICE_PLANE_OPACITY = 0.14
SLICE_PLANE_MARGIN_MM = 8.0       # the planes reach this far beyond the epicardium
STATIC_VIEW_ELEVATION_DEG = 30.0   # static 3D wall: seen from anteroseptal, raised this much toward the base
CONTOUR_COLORS = {"epi": (64, 220, 90), "endo": (60, 190, 255)}
OVERLAY_CHOICES = (("Vessels (LAD / RCA / LCX)", "vessels"), ("17 segments", "segments"), ("None", ""))
REPRESENTATIONS = (("Surface", "surface"), ("Wireframe", "wireframe"), ("Points", "points"))
THICKENING_VIEWS = (("5 walls (apex, anterior, septal, inferior, lateral)", "walls"),
                   ("17 segments", "segments"), ("Per pixel", "pixels"))   # wallLib.REGION_VIEWS
DEFAULT_THICKENING_VIEW = "walls"
THICKENING_WINDOW = (-20.0, 150.0, 0.0, 40.0)   # range and default window of the thickening map, %
MOTION_WINDOW = (0.0, 25.0, 0.0, 10.0)   # inward wall motion window, mm (QGS: 0-10 mm); outward has its own ramp
DEFAULT_FPS = 8
VOLUME_COLOR = (255, 200, 80)
LABEL_DISTANCE_MM = 45.0
DEFAULT_WALL_COLORMAP = "Hel"     # 3D wall colors (its own setting, not taken over from Perfusion)
DEFAULT_REPRESENTATION = "wireframe"
GHOST_OPACITY = 0.05                # ED reference (drawn on top, so it need not be opaque)
GHOST_REPRESENTATION = "wireframe"
GHOST_COLOR = (1.0, 1.0, 1.0)       # ED reference points: white on every color map
WALL_OPACITY = {"surface": 0.5, "wireframe": 0.5, "points": 1.0}   # beating wall in the 3D view


class GhostLayer:
    """The ED reference wall drawn in its own renderer layer over the 3D view, sharing its camera,
    so the beating wall never hides it. Shown as points, wireframe or surface; points and lines are
    drawn once each (the triangles would draw a point once per triangle that shares it, about 6 times,
    and an edge twice, so a translucent reference would add up to nearly opaque)."""

    def __init__(self):
        self.window = self.renderer = self.actor = None
        self.key = None

    @staticmethod
    def _mainRenderer(view):
        try:
            return view.renderer()
        except AttributeError:
            return view.renderWindow().GetRenderers().GetFirstRenderer()

    def _attach(self, widget):
        view = widget.threeDView()
        window = view.renderWindow()
        if self.renderer is not None and self.window is window:
            return view
        self.detach()
        layer = window.GetNumberOfLayers()
        window.SetNumberOfLayers(layer + 1)
        renderer = vtk.vtkRenderer()
        renderer.SetLayer(layer)
        renderer.SetInteractive(False)
        renderer.SetPreserveColorBuffer(True)    # draw over the view
        renderer.SetPreserveDepthBuffer(False)   # ... ignoring its depth: always on top
        mapper = vtk.vtkPolyDataMapper()
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetPointSize(1.0)
        prop.SetLineWidth(1.0)
        prop.LightingOff()
        renderer.AddActor(actor)
        window.AddRenderer(renderer)
        self.window, self.renderer, self.actor = window, renderer, actor
        return view

    @staticmethod
    def _geometry(polyData, representation):
        if representation == "points":
            source = vtk.vtkVertexGlyphFilter()
        elif representation == "wireframe":
            source = vtk.vtkExtractEdges()           # every edge once
        else:
            return polyData
        source.SetInputData(polyData)
        source.Update()
        return source.GetOutput()

    def update(self, widget, polyData, color, representation=GHOST_REPRESENTATION, opacity=GHOST_OPACITY):
        """Show polyData in the 3D view of widget (None hides it)."""
        if widget is None or polyData is None:
            if self.actor is not None:
                self.actor.SetVisibility(False)
                self.window.Render()
            return
        view = self._attach(widget)
        camera = self._mainRenderer(view).GetActiveCamera()
        if self.renderer.GetActiveCamera() is not camera:
            self.renderer.SetActiveCamera(camera)
        key = (id(polyData), representation)
        if self.key != key:
            self.actor.GetMapper().SetInputData(self._geometry(polyData, representation))
            self.key = key
        prop = self.actor.GetProperty()
        prop.SetRepresentation({"points": 0, "wireframe": 1}.get(representation, 2))
        prop.SetColor(*color)
        prop.SetOpacity(float(opacity))
        self.actor.SetVisibility(True)
        view.scheduleRender()

    def detach(self):
        if self.renderer is not None and self.window is not None:
            self.window.RemoveRenderer(self.renderer)
        self.window = self.renderer = self.actor = None
        self.key = None


def _installToolbar():
    try:
        from MyocardialPerfusionSPECTLib import toolbar
        toolbar.install()
    except Exception:
        logging.exception("MPI: could not install the workflow toolbar")


def gatedSequenceOf(volumeNode):
    """The gated sequence paired with a static volume in Data roles, or None."""
    import MPIGatedQC
    return MPIGatedQC.gatedSequenceOf(volumeNode)


def ejectionFraction(volumes):
    """(EDV, ESV, EF %, ED index, ES index) from the cavity volume of every frame, or None."""
    phases = wallLib.endSystoleDiastole(volumes)
    if phases is None:
        return None
    ed, es = phases
    edv, esv = float(volumes[ed]), float(volumes[es])
    ef = 100.0 * (edv - esv) / edv if edv > 0 else float("nan")
    return edv, esv, ef, ed, es


class MPIGated(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MPI 6 · Gated Review"
        self.parent.categories = ["Nuclear Medicine.Myocardial Perfusion"]
        self.parent.hidden = True   # opened from the Myocardial Perfusion SPECT hub and the workflow toolbar
        self.parent.dependencies = []
        self.parent.contributors = ["Burak Demir"]
        self.parent.helpText = (
            "Gated review of the stress study: the SA / HLA / VLA slices as a cine (settings taken over from "
            "Perfusion), the perfusion and wall thickening polar maps, the LV cavity volume of every frame "
            "(EDV, ESV, EF) and the beating 3D wall coloured by the perfusion. The frames are segmented in "
            "Gated QC.")
        self.parent.acknowledgementText = ""
        iconPath = os.path.join(_MODULE_DIR, "Resources", "Icons", "MyocardialPerfusionSPECT.png")
        if os.path.exists(iconPath):
            self.parent.icon = qt.QIcon(iconPath)
        if not slicer.app.commandOptions().noMainWindow:
            slicer.app.connect("startupCompleted()", _installToolbar)


class GatedReviewData:
    """What Gated Review shows for one study, on top of the Gated QC frames (GatedStudy): the wall
    counts of every frame on the polar map rays, the thickening map, the volumes and the 3D wall of
    every frame (made on first use)."""

    def __init__(self, gated, staticResult, frame, staticPolar):
        self.gated = gated                 # MPIGatedQC.GatedStudy
        self.staticResult = staticResult
        self.frame = frame
        self.staticPolar = staticPolar     # perfusion polar map of the static study
        self.volumes = gated.cavityMl()
        self.phases = ejectionFraction(self.volumes)
        self.thickening = None
        self.motion = None                 # wall motion polar map (mm), made with the thickening
        self.thickeningMessage = ""
        self.surfaces = {}                 # frame index -> vtkPolyData of the wall with "Uptake"

    @property
    def count(self):
        return len(self.gated.labels)

    def result(self, index):
        """Wall of a frame (the static wall for a frame that could not be segmented)."""
        result = self.gated.results[index]
        return result if result is not None else self.staticResult

    def buildThickening(self):
        import MPIGatedQC
        if self.phases is None:
            self.thickeningMessage = "Needs the LV cavity volume of at least two frames."
            return
        gates = MPIGatedQC.gateNodes(self.gated.sequence)
        polars = []
        for index, (_, node) in enumerate(gates):
            array = slicer.util.arrayFromVolume(node).astype(np.float32)
            sample = MPIGatedQC.sampler(array, MPIGatedQC.worldToIjk(node, self.gated.study))
            polars.append(wallLib.polarMap(self.result(index), sample))
        _, _, _, ed, es = self.phases
        self.thickening = wallLib.thickeningMap(polars, ed, es)
        meanCounts = np.mean([p["values"] for p in polars], axis=0)
        self.motion = wallLib.motionMap([self.result(i) for i in range(len(gates))], ed, es, weights=meanCounts)

    def surface(self, index, staticSample):
        """3D wall of one frame (the closed shell, shown as a surface, wireframe or points by its display
        node), with the perfusion of the static study at its points (% of the polar map maximum) as "Uptake"."""
        if index in self.surfaces:
            return self.surfaces[index]
        import MPIMyocardium
        from vtk.util import numpy_support
        polyData = MPIMyocardium.wallShellPolyData(self.result(index), self.frame)
        points = numpy_support.vtk_to_numpy(polyData.GetPoints().GetData())
        maximum = float(np.nanmax(self.staticPolar["values"])) or 1.0
        # the rays through the frame's wall points carry the static wall's counts: perfusion colours
        uptake = 100.0 * wallLib.uptakeAtPoints(self.staticResult, staticSample, points) / maximum
        scalars = numpy_support.numpy_to_vtk(uptake.astype(np.float32), deep=1)
        scalars.SetName("Uptake")
        polyData.GetPointData().AddArray(scalars)
        polyData.GetPointData().SetActiveScalars("Uptake")
        self.surfaces[index] = polyData
        return polyData


class MPIGatedWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.mode = "stressGated"
        self.chosenMode = None      # (layout the user picked, studies ready then): kept while those do not change
        self.data = {}             # slot -> (key, GatedReviewData)
        self.tick = 0              # cine position; every study shows frame tick % its frame count
        self.cameraLink = CameraLink()   # stress and rest 3D views moved together
        self.scrollOffsets = {}    # (slot, orientation) -> slices the mouse wheel moved a fitted row
        self.slicePositions = {}   # (slot, orientation) -> slices shown: the planes of the static 3D wall
        self.tiles = {}            # (slot, orientation) -> (key, [frame][cell] (QImage or None, slice number or None), title)
        self.polarImages = {}      # (slot, polar map tag) -> (key, (QImage, region values))
        self.inherited = None      # Perfusion settings last taken over
        self.staticSamples = {}    # study ID -> (key, sampler of the volume the wall was detected on)
        self.surfaceKeys = {}      # slot -> key of the 3D models made for its data
        self.ghostLayers = {STRESS: GhostLayer(), REST: GhostLayer()}
        self.grids = {"single": [DEFAULT_GRID_ROWS, DEFAULT_GRID_COLUMNS], "dual": [1, DEFAULT_GRID_COLUMNS]}

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.studySelector = StudyPairSelector(self.layout)
        self.studyLabel = narrowLabel(qt.QLabel())
        self.layout.addWidget(self.studyLabel)

        # Cine
        cine = ctk.ctkCollapsibleButton()
        cine.text = "Cine"
        self.layout.addWidget(cine)
        cineForm = qt.QFormLayout(cine)
        playRow = qt.QHBoxLayout()
        self.playButton = styleButton(qt.QPushButton("▶  Play"), "primary")
        self.playButton.checkable = True
        self.playButton.toolTip = "Start / stop the cine of the slices, the volume curves and the 3D walls"
        self.speedSpin = qt.QSpinBox()
        self.speedSpin.setRange(1, 30)
        self.speedSpin.setValue(DEFAULT_FPS)
        self.speedSpin.setSuffix(" frames/s")
        self.speedSpin.toolTip = "Cine speed"
        playRow.addWidget(self.playButton, 1)
        playRow.addWidget(self.speedSpin)
        cineForm.addRow(playRow)
        frameRow = qt.QHBoxLayout()
        self.frameSlider = qt.QSlider(qt.Qt.Horizontal)
        self.frameSlider.setRange(0, 0)
        self.frameSlider.toolTip = "Frame shown (stops the cine)"
        self.frameLabel = qt.QLabel("-")
        self.frameLabel.setMinimumWidth(70)
        frameRow.addWidget(self.frameSlider, 1)
        frameRow.addWidget(self.frameLabel)
        cineForm.addRow("Frame:", frameRow)

        # Slices
        section = ctk.ctkCollapsibleButton()
        section.text = "Slices"
        self.layout.addWidget(section)
        form = qt.QFormLayout(section)
        self.colormapCombo = qt.QComboBox()
        self.colormapCombo.addItems([label for label, _ in COLORMAPS])
        form.addRow("Color map:", self.colormapCombo)
        self.windowRange = self._windowWidget(
            "Color window (min - max) of the slices in % of the hottest frame near the LV (all frames of a study "
            "on one scale, so the count changes over the cycle show)")
        form.addRow("Window:", self.windowRange)
        gridRow = qt.QHBoxLayout()
        self.gridRowsSpin = qt.QSpinBox()
        self.gridRowsSpin.setRange(1, 8)
        self.gridRowsSpin.setValue(DEFAULT_GRID_ROWS)
        self.gridRowsSpin.setSuffix(" rows")
        self.gridColumnsSpin = qt.QSpinBox()
        self.gridColumnsSpin.setRange(2, 30)
        self.gridColumnsSpin.setValue(DEFAULT_GRID_COLUMNS)
        self.gridColumnsSpin.setSuffix(" columns")
        gridRow.addWidget(self.gridRowsSpin)
        gridRow.addWidget(qt.QLabel("×"))
        gridRow.addWidget(self.gridColumnsSpin)
        form.addRow("Grid per row:", gridRow)
        self.autoFitCheck = qt.QCheckBox("Fit to the myocardium")
        self.autoFitCheck.checked = True
        self.autoFitCheck.toolTip = ("Show every slice with myocardium on the grid giving the largest tiles for the "
                                     "screen (fewest rows), each row filled with the neighbouring slices as far as "
                                     "the tiles fit (set again when the view is resized). "
                                     "Changing the grid or the slices by hand turns it off")
        form.addRow("", self.autoFitCheck)
        self.fovSpin = qt.QDoubleSpinBox()
        self.fovSpin.setRange(4.0, 30.0)
        self.fovSpin.setSingleStep(0.5)
        self.fovSpin.setDecimals(1)
        self.fovSpin.setSuffix(" cm")
        self.fovSpin.setValue(DEFAULT_FOV_CM)
        self.fovSpin.toolTip = ("Field of view of every tile, centered on the LV: the same for stress and rest, so "
                                "both show at the same zoom")
        form.addRow("Field of view:", self.fovSpin)
        self.syncCheck = qt.QCheckBox("Sync stress and rest")
        self.syncCheck.checked = True
        self.syncCheck.toolTip = ("The rest rows follow the stress rows, matched on the center of each study's "
                                  "segmented myocardium (as in Perfusion)")
        form.addRow("", self.syncCheck)
        self.stepSpin = qt.QSpinBox()
        self.stepSpin.setRange(1, 10)
        self.stepSpin.setValue(1)
        self.stepSpin.toolTip = "Slices advanced from one tile to the next (1 = consecutive slices)"
        form.addRow("Slice step:", self.stepSpin)
        self.firstSliceSpins = {}
        for slot in (STRESS, REST):
            for name in ORIENTATIONS:
                spin = qt.QSpinBox()
                spin.setRange(1, 1)
                spin.toolTip = (f"First slice of the {SLOT_TITLES[slot].lower()} {ORIENTATION_TITLES[name]} row "
                                "(mouse wheel over the row also scrolls)")
                self.firstSliceSpins[(slot, name)] = spin
                form.addRow(f"First {name} slice ({SLOT_TITLES[slot].lower()}):", spin)
        self.contoursButton = qt.QPushButton("Myocardium segmentation")
        self.contoursButton.checkable = True
        self.contoursButton.checked = True
        self.contoursButton.toolTip = "Epicardium (green) and endocardium (blue) of every frame, from Gated QC"
        form.addRow("Show:", self.contoursButton)
        self.inheritButton = qt.QPushButton("Take over the Perfusion settings")
        self.inheritButton.toolTip = ("Color map, windows, grid, slices and polar map overlays as set in Perfusion "
                                      "(done by itself when this screen opens after they changed there)")
        form.addRow(self.inheritButton)

        # Polar maps
        polarSection = ctk.ctkCollapsibleButton()
        polarSection.text = "Polar maps"
        self.layout.addWidget(polarSection)
        polarForm = qt.QFormLayout(polarSection)
        self.polarLinkCheck = qt.QCheckBox("Same as the slices")
        self.polarLinkCheck.checked = True
        self.polarLinkCheck.toolTip = ("Use the slice window for the perfusion polar maps and the 3D walls (in % of "
                                       "the polar map maximum); untick to window them on their own")
        polarForm.addRow("Perfusion window:", self.polarLinkCheck)
        self.polarWindowRange = self._windowWidget(
            "Color window of the perfusion polar maps and the 3D walls, % of the polar map maximum")
        polarForm.addRow("", self.polarWindowRange)
        self.thickeningRange = self._windowWidget(
            "Color window of the wall thickening maps, % thickening (ES / ED wall counts)", *THICKENING_WINDOW)
        polarForm.addRow("Thickening window:", self.thickeningRange)
        self.thickeningViewCombo = qt.QComboBox()
        for text, value in THICKENING_VIEWS:
            self.thickeningViewCombo.addItem(text, value)
        self.thickeningViewCombo.setCurrentIndex(max(0, self.thickeningViewCombo.findData(DEFAULT_THICKENING_VIEW)))
        self.thickeningViewCombo.toolTip = (
            "How the wall thickening map is painted: the mean of each of the five walls (default), the mean of "
            "each of the 17 segments, or every pixel (a count ratio per pixel is noisy)")
        polarForm.addRow("Thickening map:", self.thickeningViewCombo)
        self.motionRange = self._windowWidget(
            "Color window of the inward wall motion, mm of endocardial motion from ED to ES (outward motion: "
            "black at 0 to full color at -4 mm)", *MOTION_WINDOW,
            suffix=" mm")
        polarForm.addRow("Motion window:", self.motionRange)
        overlayRow = qt.QHBoxLayout()
        self.overlayCombo = qt.QComboBox()
        for text, value in OVERLAY_CHOICES:
            self.overlayCombo.addItem(text, value)
        self.overlayCombo.toolTip = "Regions drawn on the polar maps: coronary territories, the 17 segments or none"
        self.scoresButton = qt.QPushButton("Scores")
        self.scoresButton.checkable = True
        self.scoresButton.checked = True
        self.scoresButton.toolTip = "Show the region values (perfusion in % of the maximum, thickening in %)"
        overlayRow.addWidget(self.overlayCombo, 1)
        overlayRow.addWidget(self.scoresButton)
        polarForm.addRow("Show:", overlayRow)

        # 3D
        threeD = ctk.ctkCollapsibleButton()
        threeD.text = "3D wall"
        self.layout.addWidget(threeD)
        threeDForm = qt.QFormLayout(threeD)
        self.representationCombo = qt.QComboBox()
        for text, value in REPRESENTATIONS:
            self.representationCombo.addItem(text, value)
        self.representationCombo.setCurrentIndex(max(0, self.representationCombo.findData(DEFAULT_REPRESENTATION)))
        self.representationCombo.toolTip = "How the 3D wall is drawn: surface, wireframe or points"
        threeDForm.addRow("Display:", self.representationCombo)
        self.wallColormapCombo = qt.QComboBox()
        self.wallColormapCombo.addItems([label for label, _ in COLORMAPS])
        self.wallColormapCombo.setCurrentIndex(max(0, self.wallColormapCombo.findText(DEFAULT_WALL_COLORMAP)))
        self.wallColormapCombo.toolTip = "Color map of the 3D wall (the slices and polar maps have their own)"
        threeDForm.addRow("Color map:", self.wallColormapCombo)
        self.ghostCheck = qt.QCheckBox("End-diastole as reference")
        self.ghostCheck.checked = True
        self.ghostCheck.toolTip = "The ED wall in white, drawn over the beating wall"
        threeDForm.addRow("Show:", self.ghostCheck)
        ghostRow = qt.QHBoxLayout()
        self.ghostRepresentationCombo = qt.QComboBox()
        for text, value in REPRESENTATIONS:
            self.ghostRepresentationCombo.addItem(text, value)
        self.ghostRepresentationCombo.setCurrentIndex(
            max(0, self.ghostRepresentationCombo.findData(GHOST_REPRESENTATION)))
        self.ghostRepresentationCombo.toolTip = "How the ED reference is drawn: surface, wireframe or points"
        self.ghostOpacitySlider = ctk.ctkSliderWidget()
        self.ghostOpacitySlider.minimum, self.ghostOpacitySlider.maximum = 0.01, 1.0
        self.ghostOpacitySlider.singleStep = 0.01
        self.ghostOpacitySlider.decimals = 2
        self.ghostOpacitySlider.value = GHOST_OPACITY
        self.ghostOpacitySlider.toolTip = "Opacity of the ED reference"
        ghostRow.addWidget(self.ghostRepresentationCombo)
        ghostRow.addWidget(self.ghostOpacitySlider, 1)
        threeDForm.addRow("ED reference:", ghostRow)
        self.link3DCheck = qt.QCheckBox("Link the stress and rest 3D views")
        self.link3DCheck.checked = True
        self.link3DCheck.toolTip = ("Rotating, zooming or panning one 3D view moves the other the same way, around "
                                    "its own heart")
        threeDForm.addRow("", self.link3DCheck)

        # Layouts: one column per study (stress, rest, both), one row per kind (five kinds side by side
        # would make the panel wider than Slicer lets it be)
        self.layoutButtons = {}
        self.layoutGroup = qt.QButtonGroup()
        self.layoutGroup.exclusive = True
        layoutBox = qt.QGroupBox("Layout")
        layoutGrid = qt.QGridLayout(layoutBox)
        for column, keys in enumerate(MODE_GRID):
            title = qt.QLabel(f"<b>{MODES[keys[0]][0].replace(' GATED', '')}</b>")
            title.alignment = qt.Qt.AlignCenter
            layoutGrid.addWidget(title, 0, column)
            for row, key in enumerate(keys):
                text, _, kind, toolTip = MODES[key]
                button = qt.QPushButton(KIND_LABELS[kind])
                button.checkable = True
                button.toolTip = f"{text}: {toolTip}"
                self.layoutGroup.addButton(button)
                layoutGrid.addWidget(button, row + 1, column)
                self.layoutButtons[key] = button
        self.layout.addWidget(layoutBox)
        self.backButton = qt.QPushButton("‹ Gated QC")
        self.layout.addWidget(self.backButton)
        self.parameterPanel = parametersLib.ParameterPanel(
            "gated", self.layout, onChanged=self.onParameterChanged,
            hint=("Map smoothing applies when a study is loaded again (e.g. after Segment frames again in Gated QC); the other parameters at once."))
        self.parameterPanel.bind("speed", self.speedSpin)
        self.parameterPanel.bind("colormap", self.colormapCombo)
        self.parameterPanel.bind("window", self.windowRange)
        self.parameterPanel.bind("gridRows", self.gridRowsSpin)
        self.parameterPanel.bind("gridColumns", self.gridColumnsSpin)
        self.parameterPanel.bind("autoFit", self.autoFitCheck)
        self.parameterPanel.bind("fov", self.fovSpin)
        self.parameterPanel.bind("sync", self.syncCheck)
        self.parameterPanel.bind("sliceStep", self.stepSpin)
        self.parameterPanel.bind("contours", self.contoursButton)
        self.parameterPanel.bind("polarLink", self.polarLinkCheck)
        self.parameterPanel.bind("polarWindow", self.polarWindowRange)
        self.parameterPanel.bind("thickeningWindow", self.thickeningRange)
        self.parameterPanel.bind("thickeningView", self.thickeningViewCombo)
        self.parameterPanel.bind("motionWindow", self.motionRange)
        self.parameterPanel.bind("overlay", self.overlayCombo)
        self.parameterPanel.bind("scores", self.scoresButton)
        self.parameterPanel.bind("representation", self.representationCombo)
        self.parameterPanel.bind("wallColormap", self.wallColormapCombo)
        self.parameterPanel.bind("ghost", self.ghostCheck)
        self.parameterPanel.bind("ghostRepresentation", self.ghostRepresentationCombo)
        self.parameterPanel.bind("ghostOpacity", self.ghostOpacitySlider)
        self.parameterPanel.bind("link3D", self.link3DCheck)
        self.layout.addStretch(1)

        self.cineTimer = qt.QTimer()
        self.cineTimer.setInterval(int(1000 / DEFAULT_FPS))
        self.cineTimer.connect("timeout()", self.nextFrame)
        self.resizeTimer = qt.QTimer()
        self.resizeTimer.setSingleShot(True)
        self.resizeTimer.setInterval(REFIT_DELAY_MS)
        self.resizeTimer.connect("timeout()", lambda: self.refresh(rebuild3D=False))

        self.playButton.connect("toggled(bool)", self.onPlayToggled)
        self.speedSpin.connect("valueChanged(int)", lambda v: self.cineTimer.setInterval(int(1000 / max(v, 1))))
        self.frameSlider.connect("valueChanged(int)", self.onFrameSliderMoved)
        self.colormapCombo.connect("currentIndexChanged(int)", lambda _: self.refresh())
        self.windowRange.connect("valuesChanged(double,double)", lambda a, b: self.refresh(rebuild3D=False))
        for spin in (self.gridRowsSpin, self.gridColumnsSpin):
            spin.connect("valueChanged(int)", lambda _: self.onGridChanged())
        for spin in [self.stepSpin] + list(self.firstSliceSpins.values()):
            spin.connect("valueChanged(int)", lambda _: self.onSlicesChangedByHand())
        self.autoFitCheck.connect("toggled(bool)", lambda _: self.onAutoFitToggled())
        self.fovSpin.connect("valueChanged(double)", lambda _: self.refreshSlices())
        self.syncCheck.connect("toggled(bool)", lambda _: self.onSyncToggled())
        self.contoursButton.connect("toggled(bool)", lambda _: self.refreshSlices())
        self.inheritButton.connect("clicked(bool)", lambda: self.inheritPerfusion(force=True))
        self.polarLinkCheck.connect("toggled(bool)", lambda _: self.refresh(rebuild3D=False))
        self.polarWindowRange.connect("valuesChanged(double,double)", lambda a, b: self.refresh(rebuild3D=False))
        self.thickeningRange.connect("valuesChanged(double,double)", lambda a, b: self.refreshPolars())
        self.thickeningViewCombo.connect("currentIndexChanged(int)", lambda _: self.refreshPolars())
        self.motionRange.connect("valuesChanged(double,double)", lambda a, b: self.refreshPolars())
        self.overlayCombo.connect("currentIndexChanged(int)", lambda _: self.onOverlaysChanged())
        self.scoresButton.connect("toggled(bool)", lambda _: self.onOverlaysChanged())
        self.representationCombo.connect("currentIndexChanged(int)", lambda _: self.updateRepresentation())
        self.ghostCheck.connect("toggled(bool)", lambda _: self.updateGhost())
        self.ghostRepresentationCombo.connect("currentIndexChanged(int)", lambda _: self.updateGhost())
        self.ghostOpacitySlider.connect("valueChanged(double)", lambda _: self.updateGhost())
        self.link3DCheck.connect("toggled(bool)", lambda _: self.updateCameraLink())
        self.wallColormapCombo.connect("currentIndexChanged(int)", lambda _: self.update3DColors())
        for key, button in self.layoutButtons.items():
            button.connect("clicked(bool)", lambda _, k=key: self.chooseMode(k))
        self.backButton.connect("clicked(bool)", lambda: session.selectStep("gatedqc"))
        self.parameterPanel.loadSaved()   # saved settings of this step, as if set by hand
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneClosed)
        session.addListener(self.onSessionChanged)
        _installToolbar()
        self.updateState()

    @staticmethod
    def _windowWidget(toolTip, minimum=0.0, maximum=150.0, low=0.0, high=100.0, suffix=" %"):
        widget = ctk.ctkRangeWidget()
        widget.minimum, widget.maximum = minimum, maximum
        widget.singleStep = 1.0
        widget.decimals = 0
        widget.suffix = suffix
        widget.minimumValue, widget.maximumValue = low, high
        widget.toolTip = toolTip
        return widget

    # --- workflow toolbar (second row) -----------------------------------------------------------------
    def buildToolbar(self, bar):
        """Most used controls in the toolbar (toolbar.StepBar); they drive the controls of this panel."""
        self.toolbarBar = bar
        self.tbPlay = bar.toggle("▶ Play", self.playButton.toolTip,
                                 lambda c: self.fromToolbar(lambda: setattr(self.playButton, "checked", c)))
        bar.separator()
        self.tbMax = bar.spin("Max", 10, 150, 5, " %", "Top of the slice color window (% of the hottest frame)",
                              self.onToolbarMax)
        self.tbFov = bar.spin("FOV", 4.0, 30.0, 0.5, " cm", self.fovSpin.toolTip,
                              lambda v: self.fromToolbar(lambda: self.fovSpin.setValue(v)), decimals=1)
        bar.separator()
        self.tbContours = bar.toggle("Contours", self.contoursButton.toolTip,
                                     lambda c: self.fromToolbar(lambda: setattr(self.contoursButton, "checked", c)))
        self.tbSync = bar.toggle("Sync", self.syncCheck.toolTip,
                                 lambda c: self.fromToolbar(lambda: setattr(self.syncCheck, "checked", c)))
        bar.separator()
        self.tbOverlay = bar.choice([("Vessels", "vessels", "Coronary territories (LAD / RCA / LCX) on the polar maps"),
                                     ("17 seg", "segments", "The 17 segments on the polar maps")],
                                    self.onToolbarOverlay)
        self.tbScores = bar.toggle("Scores", self.scoresButton.toolTip,
                                   lambda c: self.fromToolbar(lambda: setattr(self.scoresButton, "checked", c)))
        bar.separator()
        self.tbLayouts = bar.buttons([(key, TOOLBAR_KIND_LABELS[MODES[key][2]], f"{MODES[key][0]}: {MODES[key][3]}")
                                      for row in MODE_GRID for key in row],
                                     lambda key: self.fromToolbar(lambda: self.chooseMode(key)))
        if not getattr(self, "_toolbarFollows", False):   # the toolbar follows this panel's controls
            self._toolbarFollows = True
            self.windowRange.connect("valuesChanged(double,double)", lambda a, b: self.syncToolbar())
            self.fovSpin.connect("valueChanged(double)", lambda _: self.syncToolbar())
            for button in (self.playButton, self.contoursButton, self.syncCheck, self.scoresButton):
                button.connect("toggled(bool)", lambda _: self.syncToolbar())
            self.overlayCombo.connect("currentIndexChanged(int)", lambda _: self.syncToolbar())

    def toolbarLayouts(self):
        """Layouts offered in the toolbar: the stress-rest ones when both gated studies are ready, else those
        of the one that is."""
        for row in reversed(MODE_GRID):    # stress-rest, rest, stress
            if row[0].startswith("stressRest") and self.available(row[0])[0]:
                return list(row)
        for row in MODE_GRID:
            if not row[0].startswith("stressRest") and self.available(row[0])[0]:
                return list(row)
        return []

    def fromToolbar(self, action):
        action()
        self.syncToolbar()   # e.g. a click on the checked overlay button changed nothing: check it again

    def onToolbarMax(self, value):
        self.windowRange.maximumValue = max(float(value), self.windowRange.minimumValue + 1.0)
        self.syncToolbar()

    def onToolbarOverlay(self, value):
        index = self.overlayCombo.findData(value)
        if index >= 0:
            self.overlayCombo.setCurrentIndex(index)
        self.syncToolbar()

    def syncToolbar(self):
        bar = getattr(self, "toolbarBar", None)
        if bar is None:
            return
        try:
            bar.setChecked(self.tbPlay, self.playButton.checked)
            self.tbPlay.setText("■ Stop" if self.playButton.checked else "▶ Play")
            bar.setValue(self.tbMax, self.windowRange.maximumValue)
            bar.setValue(self.tbFov, self.fovSpin.value)
            bar.setChecked(self.tbContours, self.contoursButton.checked)
            bar.setChecked(self.tbSync, self.syncCheck.checked)
            self.tbSync.enabled = self.syncCheck.enabled
            bar.setChoice(self.tbOverlay, self.overlay())
            bar.setChecked(self.tbScores, self.scoresButton.checked)
            offered = self.toolbarLayouts()
            for key, button in self.tbLayouts.items():
                button.setVisible(key in offered)
                bar.setChecked(button, key == self.mode and self.onScreen())
        except Exception:
            logging.debug("MPI: could not update the toolbar:\n" + traceback.format_exc())

    def enter(self):
        self.studySelector.refresh()
        self.updateState()
        mode = self.preferredMode()
        if self.available(mode)[0]:
            self.show(mode)

    # --- default layout ---------------------------------------------------------------
    def readyStudies(self):
        return tuple(slot for slot in (STRESS, REST) if self.available(f"{slot}Gated")[0])

    def defaultMode(self):
        """Slices and plots of the gated studies there: stress-rest when both are ready, on two monitors when
        there is a second screen."""
        dual = secondaryScreen() is not None
        ready = self.readyStudies()
        prefix = "stressRest" if len(ready) == 2 else ready[0] if ready else None
        if prefix is None:
            return self.mode
        return prefix + ("GatedDual" if dual else "Gated")

    def preferredMode(self):
        """The layout the user picked while the same studies are ready, else the default one."""
        if self.chosenMode is not None:
            mode, ready = self.chosenMode
            if ready == self.readyStudies() and self.available(mode)[0]:
                return mode
        return self.defaultMode()

    def chooseMode(self, mode):
        self.chosenMode = (mode, self.readyStudies())
        self.show(mode)

    def exit(self):
        self.playButton.checked = False
        self.cameraLink.unlink()

    @staticmethod
    def panelAttributes():
        """slicer attributes of every Gated Review panel (both studies)."""
        attributes = [MOSAIC_STATE, MOSAIC2_STATE]
        for slot in (STRESS, REST):
            tags = gatedTags(slot)
            attributes += [polarPanelState(tags[kind]) for kind in POLAR_KINDS]
            attributes.append(volumeCurvePanelState(tags["volume"]))
        return attributes

    def cleanup(self):
        self.cineTimer.stop()
        self.cameraLink.unlink()
        self.resizeTimer.stop()
        session.removeListener(self.onSessionChanged)
        self.studySelector.cleanup()
        for layer in self.ghostLayers.values():
            layer.detach()
        for attribute in self.panelAttributes():
            panel = existingPanel(attribute)
            if panel is not None:
                panel.onResized = None
                if hasattr(panel, "onScroll"):
                    panel.onScroll = None
        self.removeObservers()

    def onSceneClosed(self, caller=None, event=None):
        self.playButton.checked = False
        self.data, self.tiles, self.polarImages, self.staticSamples = {}, {}, {}, {}
        self.slicePositions = {}
        self.surfaceKeys = {}
        for layer in self.ghostLayers.values():
            layer.update(None, None, None)
        for attribute in self.panelAttributes():
            panel = existingPanel(attribute)
            if panel is not None:
                panel.clear()
        self.updateState()

    def onSessionChanged(self):
        self.updateState()
        mode = self.preferredMode()
        if self.parent.isVisible() and self.onScreen() and self.available(mode)[0]:
            self.show(mode)

    # --- studies -------------------------------------------------------------------
    def slots(self, mode=None):
        """The studies a layout shows: (STRESS,), (REST,) or (STRESS, REST)."""
        return MODES[mode or self.mode][1]

    def kind(self, mode=None):
        return MODES[mode or self.mode][2]

    def representative(self, mode=None):
        """Only the apical, mid and basal slices of each axis (REPRESENTATIVE_KINDS)."""
        return self.kind(mode) in REPRESENTATIVE_KINDS

    def panelGroups(self):
        """[(panel state attribute, rows [(row key suffix, orientations)], columns)] of the slice panels: one
        panel with a row per orientation on the grid of the spins (columns None), or with three slices per
        axis SA (beside the static 3D wall) and below it HLA and VLA side by side."""
        if self.representative():
            n = len(REPRESENTATIVE_FRACTIONS)
            return [(MOSAIC_STATE, [("SA", ("SA",))], n), (MOSAIC2_STATE, [("HLA+VLA", ("HLA", "VLA"))], 2 * n)]
        return [(MOSAIC_STATE, [(name, (name,)) for name in ORIENTATIONS], None)]

    @staticmethod
    def makePanel(attribute):
        return gatedReviewMosaicPanel() if attribute == MOSAIC_STATE else gatedReviewMosaic2Panel()

    def panelGrid(self, columns=None):
        if columns:
            return 1, columns
        return self.gridRowsSpin.value, self.gridColumnsSpin.value

    def representativeSlices(self, slot, name, data):
        """Slice indices (0-based) at REPRESENTATIVE_FRACTIONS of the myocardium along an axis (of all slices
        when Myocardium found none): apical, mid, basal for SA."""
        count = data.gated.slices[name][0].shape[0]
        extent = None
        try:
            extent = self.reviewWidget().myocardiumSlices(slot, name, data.gated.reference[name])
        except Exception:
            logging.debug("MPI: no myocardium extent:\n" + traceback.format_exc())
        first, last = extent if extent is not None else (0, count - 1)
        return [int(min(max(first + int(round(f * (last - first))), 0), count - 1)) for f in REPRESENTATIVE_FRACTIONS]

    def plotKinds(self, mode=None):
        return gatedPlotKinds(self.slots(mode), self.kind(mode))

    def layoutId(self, mode=None):
        return GATED_LAYOUT_IDS[(self.slots(mode), self.kind(mode))]

    @staticmethod
    def studyOf(slot):
        return session.stressRestStudies()[slot]

    @staticmethod
    def qcWidget():
        return slicer.modules.mpigatedqc.widgetRepresentation().self()

    def available(self, mode=None):
        """(ready, reason) for a layout: every study it shows must be ready."""
        for slot in self.slots(mode):
            ready, reason = self.qcWidget().available(slot)
            if not ready:
                return False, (f"{SLOT_TITLES[slot].lower()}: {reason}" if len(self.slots(mode)) > 1 else reason)
        return True, ""

    def onScreen(self):
        return slicer.app.layoutManager().layout == self.layoutId()

    def updateState(self):
        lines = []
        for slot, title in ((STRESS, "Study 1"), (REST, "Study 2")):
            study = self.studyOf(slot)
            if study is None:
                continue
            ready, reason = self.qcWidget().available(slot)
            sequence = gatedSequenceOf(study)
            text = f"{title}: <b>{breakable(study.GetName())}</b>"
            if sequence is not None:
                text += (f", gated <b>{breakable(sequence.GetName())}</b> "
                         f"({sequence.GetNumberOfDataNodes()} frames)")
            if not ready:
                text += f" <span style='color:#d08030'>({reason})</span>"
            lines.append(text)
        if not lines:
            lines = ["No study selected. Pick Study 1 / Study 2 above or set the roles in Data roles."]
        self.studyLabel.text = "<br>".join(lines)
        for key, button in self.layoutButtons.items():
            ready, reason = self.available(key)
            button.enabled = ready
            button.toolTip = f"{MODES[key][0]}: {MODES[key][3]}" + ("" if ready else f" (not available: {reason})")
        if self.mode in self.layoutButtons:
            self.layoutButtons[self.mode].checked = True
        self.polarWindowRange.enabled = not self.polarLinkCheck.checked
        shown = self.slots()
        fixed = self.representative()
        for (slot, _), spin in self.firstSliceSpins.items():
            spin.enabled = slot in shown and not (slot == REST and self.synced()) and not fixed
        for widget in (self.gridRowsSpin, self.gridColumnsSpin, self.stepSpin, self.autoFitCheck):
            widget.enabled = not fixed
        self.syncCheck.enabled = len(shown) > 1 and not fixed
        self.syncToolbar()

    def reviewData(self, slot):
        """GatedReviewData of a slot, made again when the Gated QC frames or the static wall change."""
        import MPIMyocardium
        study = self.studyOf(slot)
        gated = self.qcWidget().gatedStudy(slot)
        staticResult = slicer.modules.mpimyocardium.widgetRepresentation().self().resultForStudy(study)
        frame = MPIMyocardium.cardiacFrame(study)
        if staticResult is None or frame is None:
            raise ValueError("Detect the myocardium of this study first (Myocardium step).")
        key = (study.GetID(), id(gated), id(staticResult))
        cached = self.data.get(slot)
        if cached and cached[0] == key:
            return cached[1]
        staticPolar, message = self.reviewWidget().polarMap(study)
        if staticPolar is None:
            raise ValueError(message)
        data = GatedReviewData(gated, staticResult, frame, staticPolar)
        data.buildThickening()
        self.data[slot] = (key, data)
        self.tiles = {k: v for k, v in self.tiles.items() if k[0] != slot}
        self.polarImages = {k: v for k, v in self.polarImages.items() if k[0] != slot}
        self.surfaceKeys.pop(slot, None)
        self.fitFirstSlices(data, slot)
        return data

    def dataOf(self, slot):
        cached = self.data.get(slot)
        return cached[1] if cached else None

    def shown(self):
        """[(slot, GatedReviewData)] of the studies of the layout that have their data."""
        return [(slot, self.dataOf(slot)) for slot in self.slots() if self.dataOf(slot) is not None]

    def frameOf(self, data):
        return self.tick % data.count if data is not None and data.count else 0

    def staticSample(self, study):
        import MPIMyocardium
        source = MPIMyocardium.detectionVolume(study)
        key = (source.GetID(), source.GetImageData().GetMTime() if source.GetImageData() else 0)
        cached = self.staticSamples.get(study.GetID())
        if cached is None or cached[0] != key:
            cached = (key, MPIMyocardium.volumeSampler(source))
            self.staticSamples[study.GetID()] = cached
        return cached[1]

    # --- settings taken over from Perfusion -----------------------------------------------
    @staticmethod
    def reviewWidget():
        return slicer.modules.mpireview.widgetRepresentation().self()

    def gridKind(self, mode=None):
        return "dual" if len(self.slots(mode)) > 1 else "single"

    def perfusionSettings(self, slots):
        review = self.reviewWidget()
        first = slots[0]
        return {"colormap": review.colormapCombo.currentText,
                "window": (review.windowRanges[first].minimumValue, review.windowRanges[first].maximumValue),
                "grid": tuple(review.grids["dual" if len(slots) > 1 else "single"]), "step": review.stepSpin.value,
                "first": {(slot, name): (review.firstSliceSpins[(slot, name)].value,
                                         review.firstSliceSpins[(slot, name)].maximum)
                          for slot in slots for name in ORIENTATIONS},
                "polarLinked": bool(review.polarLinkCheck.checked),
                "polarWindow": (review.polarWindowRanges[first].minimumValue,
                                review.polarWindowRanges[first].maximumValue),
                "overlay": review.overlayCombo.currentIndex, "scores": bool(review.scoresButton.checked),
                "fov": float(review.fovSpin.value), "sync": bool(review.syncButton.checked)}

    def inheritPerfusion(self, force=False):
        """Take over the Perfusion slice and polar map settings of the shown studies when they changed
        there since last time (or always with force)."""
        try:
            settings = self.perfusionSettings(self.slots())
        except Exception:
            logging.warning("MPI: could not read the Perfusion settings:\n" + traceback.format_exc())
            return
        if settings == self.inherited and not force:
            return
        self.inherited = settings
        widgets = [self.colormapCombo, self.windowRange, self.gridRowsSpin, self.gridColumnsSpin, self.stepSpin,
                   self.polarLinkCheck, self.polarWindowRange, self.overlayCombo, self.scoresButton, self.fovSpin,
                   self.syncCheck] + \
            list(self.firstSliceSpins.values())
        blocked = [w.blockSignals(True) for w in widgets]
        try:
            self.fovSpin.setValue(settings["fov"])
            self.syncCheck.checked = settings["sync"]
            index = self.colormapCombo.findText(settings["colormap"])
            if index >= 0:
                self.colormapCombo.setCurrentIndex(index)
            self.windowRange.minimumValue, self.windowRange.maximumValue = settings["window"]
            self.grids[self.gridKind()] = list(settings["grid"])
            self.gridRowsSpin.setValue(settings["grid"][0])
            self.gridColumnsSpin.setValue(settings["grid"][1])
            self.stepSpin.setValue(settings["step"])
            self.polarLinkCheck.checked = settings["polarLinked"]
            self.polarWindowRange.minimumValue, self.polarWindowRange.maximumValue = settings["polarWindow"]
            self.overlayCombo.setCurrentIndex(settings["overlay"])
            self.scoresButton.checked = settings["scores"]
            for key, (value, maximum) in settings["first"].items():
                spin = self.firstSliceSpins[key]
                if maximum > 1:      # the gate frames are resliced on the Perfusion grids: same slice numbers
                    spin.setRange(1, max(maximum, spin.maximum))
                    spin.setValue(value)
        finally:
            for w, b in zip(widgets, blocked):
                w.blockSignals(b)
        self.updateState()
        if force:
            self.refresh(rebuild3D=False)

    def fitFirstSlices(self, data, slot):
        """Slice ranges of the frames; rows not set by Perfusion start around the LV center."""
        for name in ORIENTATIONS:
            n = data.gated.slices[name][0].shape[0]
            spin = self.firstSliceSpins[(slot, name)]
            wasBlocked = spin.blockSignals(True)
            unset = spin.maximum <= 1
            spin.setRange(1, lastFirstSlice(n))   # past the end, until a few slices are left
            if unset:
                spin.setValue(max(0, (n - self.cellsSpan()) // 2) + 1)
            spin.blockSignals(wasBlocked)

    def cellCount(self):
        return self.gridRowsSpin.value * self.gridColumnsSpin.value

    def cellsSpan(self):
        return (self.cellCount() - 1) * self.stepSpin.value + 1

    def stopAutoFit(self):
        wasBlocked = self.autoFitCheck.blockSignals(True)
        self.autoFitCheck.checked = False
        self.autoFitCheck.blockSignals(wasBlocked)

    def onGridChanged(self):
        self.stopAutoFit()
        self.grids[self.gridKind()] = [self.gridRowsSpin.value, self.gridColumnsSpin.value]
        self.refreshSlices()

    def synced(self):
        """Rest rows follow the stress rows (both shown, Sync on)."""
        return bool(self.syncCheck.checked) and len(self.slots()) > 1 and len(self.shown()) > 1

    def onSyncToggled(self):
        self.updateState()
        self.refreshSlices()

    def syncMapping(self, name):
        """Stress to rest slice mapping of one orientation (MPIReview.syncMapping: the gate frames are on the
        Perfusion grids), or None."""
        try:
            return self.reviewWidget().syncMapping(name)
        except Exception:
            logging.debug("MPI: no stress / rest slice mapping:\n" + traceback.format_exc())
            return None

    def onAutoFitToggled(self):
        self.scrollOffsets = {}
        self.refreshSlices()

    def onSlicesChangedByHand(self):
        self.stopAutoFit()
        self.refreshSlices()

    def tileCrop(self, slot, name, data):
        """Window of a row shown in the tiles: the field of view around the study's LV center (as in
        Perfusion; the gate frames are on the Perfusion grids)."""
        center = None
        try:
            center = self.reviewWidget().centerIjk(slot, name)
        except Exception:
            logging.debug("MPI: no LV center for the gated tiles:\n" + traceback.format_exc())
        node = data.gated.reference[name]
        return fovCrop(data.gated.slices[name][0].shape, node.GetSpacing()[:2],
                       None if center is None else center[:2], 10.0 * float(self.fovSpin.value))

    def autoFit(self, panel):
        """Fit to the myocardium (as in Perfusion): every slice with myocardium on the grid
        with the largest tiles for the rows' area, the rows filled with neighbouring slices (bestGrid)."""
        rows = {}
        review = self.reviewWidget()
        for slot, data in self.shown():
            if slot == REST and self.synced():
                continue    # the rest rows follow the stress rows
            for name in ORIENTATIONS:
                try:
                    extent = review.myocardiumSlices(slot, name, data.gated.reference[name])
                except Exception:
                    logging.debug("MPI: no myocardium extent:\n" + traceback.format_exc())
                    extent = None
                if extent is None:
                    continue
                count = data.gated.slices[name][0].shape[0]
                rows[(slot, name)] = (extent[0], extent[1], count)
        width, height = panel.rowArea()
        if not rows or width < 20 or height < 20:
            return
        needed = max(last - first + 1 for first, last, _ in rows.values())
        gridRows, gridColumns = bestGrid(needed, width, height, rowOverhead=panel.stripOverhead())
        cells = gridRows * gridColumns
        widgets = [self.gridRowsSpin, self.gridColumnsSpin, self.stepSpin] + list(self.firstSliceSpins.values())
        blocked = [w.blockSignals(True) for w in widgets]
        try:
            self.gridRowsSpin.setValue(gridRows)
            self.gridColumnsSpin.setValue(gridColumns)
            self.stepSpin.setValue(1)
            for (slot, name), (first, last, count) in rows.items():
                centered = first - (cells - (last - first + 1)) // 2
                start = centered + self.scrollOffsets.get((slot, name), 0)
                start = int(min(max(start, 0), lastFirstSlice(count) - 1))   # past the end: black tiles
                self.scrollOffsets[(slot, name)] = start - centered     # no scrolling past the ends
                self.firstSliceSpins[(slot, name)].setValue(start + 1)
        finally:
            for w, b in zip(widgets, blocked):
                w.blockSignals(b)
        self.grids[self.gridKind()] = [gridRows, gridColumns]

    def setGridSpins(self, mode):
        rows, columns = self.grids[self.gridKind(mode)]
        for spin, value in ((self.gridRowsSpin, rows), (self.gridColumnsSpin, columns)):
            wasBlocked = spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(wasBlocked)

    def rowOf(self, slot, name):
        """(panel state attribute, row key) showing one orientation of a study."""
        for attribute, groups, _ in self.panelGroups():
            for key, names in groups:
                if name in names:
                    return attribute, f"{slot}:{key}"
        return MOSAIC_STATE, f"{slot}:{name}"

    def rowKeys(self, groups=None):
        groups = self.panelGroups()[0][1] if groups is None else groups
        return [(f"{slot}:{key}", (f"{SLOT_TITLES[slot]}  ·  " if len(self.slots()) > 1 else "")
                 + "   |   ".join(ORIENTATION_TITLES[name] for name in names))
                for key, names in groups for slot in self.slots()]

    # --- showing ---------------------------------------------------------------------
    def show(self, mode=None):
        mode = mode or self.mode
        ready, reason = self.available(mode)
        if not ready:
            slicer.util.warningDisplay(f"The gated review cannot be shown: {reason}.", windowTitle="Gated Review")
            return
        previousSlots = self.slots()
        self.mode = mode
        self.layoutButtons[mode].checked = True
        self.setGridSpins(mode)
        self.inheritPerfusion()
        with slicer.util.tryWithErrorDisplay("Could not prepare the gated review.", waitCursor=True):
            for slot in self.slots():
                self.reviewData(slot)
            if self.slots() != previousSlots or self.tick >= max(d.count for _, d in self.shown()):
                first = self.shown()[0][1]
                self.tick = first.phases[3] if first.phases else 0
            wasBlocked = self.frameSlider.blockSignals(True)
            self.frameSlider.setRange(0, max(max(d.count for _, d in self.shown()) - 1, 0))
            self.frameSlider.setValue(self.tick)
            self.frameSlider.blockSignals(wasBlocked)
            if self.kind() != "plots":
                for attribute, groups, columns in self.panelGroups():
                    panel = self.makePanel(attribute)
                    panel.onScroll = self.onScroll
                    panel.onResized = self.resizeTimer.start
                    panel.pairRows = len(self.slots()) > 1     # each grid row of stress followed by that of rest
                    panel.gridRows, panel.gridColumns = self.panelGrid(columns)
                    panel.setRowKeys(self.rowKeys(groups))
            for slot in self.slots():
                tags = gatedTags(slot)
                for kind in self.plotKinds():
                    if kind in POLAR_KINDS:
                        polar = polarPanel(tags[kind])
                        polar.onResized = self.resizeTimer.start
                        polar.overlay, polar.showScores = self.overlay(), self.scoresButton.checked
                    elif kind == "volume":
                        volumeCurvePanel(tags["volume"])
            layoutId = self.layoutId()
            registerLayout(layoutId, gatedLayoutXml(self.slots(), self.kind()))
            slicer.app.layoutManager().setLayout(layoutId)
            slicer.app.processEvents()
            if self.kind() in DUAL_KINDS:
                viewTag = gatedTags(self.slots()[0])["3d"]
                qt.QTimer.singleShot(MONITOR2_PLACE_DELAY_MS, lambda: placeMonitor2Window(viewTag))
            self.surfaceKeys = {}
            self.updateState()
            self.refresh()

    def onParameterChanged(self, key):
        """An Advanced parameter changed (None: all were restored): slices, plots and 3D drawn again."""
        self.tiles = {}
        self.refresh()
        self.updateRepresentation()
        self.updateGhost()

    def refresh(self, rebuild3D=True):
        if not self.shown() or not self.onScreen():
            return
        self.refreshSlices(show=False)
        self.refreshPolars()
        self.refreshVolumeCurves()
        for slot, _ in self.shown():
            if rebuild3D or self.surfaceKeys.get(slot) is None:
                self.refresh3D(slot, show=False)
            else:
                self.update3DColors(slot)
        # link the 3D views on every refresh (not only when a camera is placed: exit() unlinks them)
        self.updateCameraLink()
        qt.QTimer.singleShot(500, self.updateCameraLink)   # again once the layout has made its views
        self.showFrame(self.tick)

    def onScroll(self, key, direction):
        if self.representative():
            return    # fixed apical / mid / basal slices
        row = tuple(key.split(":", 1))
        if len(row) == 2 and self.synced():
            row = (STRESS, row[1])   # the rest rows follow
        if self.autoFitCheck.checked and row in self.firstSliceSpins:   # keep fitting with the row moved
            self.scrollOffsets[row] = self.scrollOffsets.get(row, 0) + direction
            self.refreshSlices()
            return
        spin = self.firstSliceSpins.get(row)
        if spin:
            spin.setValue(spin.value + direction * self.stepSpin.value)

    def overlay(self):
        return self.overlayCombo.itemData(self.overlayCombo.currentIndex) or None

    def onOverlaysChanged(self):
        for slot in (STRESS, REST):
            tags = gatedTags(slot)
            for kind in POLAR_KINDS:
                panel = existingPanel(polarPanelState(tags[kind]))
                if panel is not None:
                    panel.setOverlays(self.overlay(), self.scoresButton.checked)

    # --- cine ----------------------------------------------------------------------------
    def onPlayToggled(self, playing):
        self.playButton.text = "■  Stop" if playing else "▶  Play"
        if playing and self.shown():
            self.cineTimer.start()
        else:
            self.cineTimer.stop()

    def nextFrame(self):
        shown = self.shown()
        if not shown or not self.onScreen():
            self.playButton.checked = False
            return
        self.showFrame((self.tick + 1) % max(d.count for _, d in shown))

    def onFrameSliderMoved(self, value):
        self.playButton.checked = False
        self.showFrame(int(value))

    def showFrame(self, tick):
        """Show one cine position everywhere: the slice rows, the marks on the volume curves and the 3D
        walls (each study at frame tick modulo its frame count)."""
        shown = [(slot, data) for slot, data in self.shown() if data.count]
        if not shown:
            return
        self.tick = int(tick) % max(data.count for _, data in shown)
        wasBlocked = self.frameSlider.blockSignals(True)
        self.frameSlider.setValue(self.tick)
        self.frameSlider.blockSignals(wasBlocked)
        self.frameLabel.text = "  ·  ".join(
            (f"{SLOT_TITLES[slot][0]} " if len(shown) > 1 else "")
            + f"{data.gated.labels[self.frameOf(data)]} of {data.count}" for slot, data in shown)
        panels = [(existingPanel(attribute), groups) for attribute, groups, _ in self.panelGroups()]
        for slot, data in shown:
            index = self.frameOf(data)
            frameText = f"frame {data.gated.labels[index]} of {data.count}"
            for panel, groups in panels:
                if panel is None or self.kind() == "plots":
                    continue
                for key, names in groups:
                    parts = [self.tiles.get((slot, name)) for name in names]
                    if not all(parts):
                        continue
                    tiles = [tile for _, frames, _ in parts for tile in frames[index]]
                    titles = [title if i == 0 else title.split("  ·  ", 1)[-1] for i, (_, _, title) in enumerate(parts)]
                    panel.setRow(f"{slot}:{key}", tiles, "   |   ".join(titles) + f"   ·   {frameText}")
            tags = gatedTags(slot)
            curve = existingPanel(volumeCurvePanelState(tags["volume"]))
            if curve is not None and "volume" in self.plotKinds():
                curve.chart.setMarker(index)
            model = self.surfaceModel(slot)
            if model is not None and "3d" in self.plotKinds():
                polyData = data.surfaces.get(index)
                if polyData is not None and model.GetPolyData() is not polyData:
                    model.SetAndObservePolyData(polyData)
                widget = threeDWidgetByTag(tags["3d"])
                if widget is not None:
                    volume = data.volumes[index]
                    self.setCornerText(widget, f"{SLOT_TITLES[slot]} GATED   frame {data.gated.labels[index]} of "
                                               f"{data.count}" + ("" if volume is None else f"\nLV cavity {volume:.0f} ml"),
                                       f"EF {data.phases[2]:.0f} %" if data.phases else "")

    # --- slices -----------------------------------------------------------------------------
    def sliceWindow(self, data):
        reference = max(data.gated.peaks) if data.gated.peaks else 1.0
        low = reference * self.windowRange.minimumValue / 100.0
        return low, max(reference * self.windowRange.maximumValue / 100.0, low + 1e-9)

    def refreshSlices(self, show=True):
        """Tiles of every frame of every row at the current settings, made once (the cine only swaps them)."""
        shownSlots = set(self.slots()) if self.kind() != "plots" else set()
        self.tiles = {k: v for k, v in self.tiles.items() if k[0] in shownSlots}   # all frames: free the others
        panels = {attribute: (existingPanel(attribute), groups, columns)
                  for attribute, groups, columns in self.panelGroups()}
        if any(p is None for p, _, _ in panels.values()) or not self.onScreen() or self.kind() == "plots":
            return
        for panel, groups, _ in panels.values():
            panel.pairRows = len(self.slots()) > 1
            panel.setRowKeys(self.rowKeys(groups))
        fixed = self.representative()
        if self.autoFitCheck.checked and not fixed:
            self.autoFit(panels[MOSAIC_STATE][0])
        for panel, _, columns in panels.values():
            panel.gridRows, panel.gridColumns = self.panelGrid(columns)
        table = colorTableArray(self.colormapCombo.currentText)
        step, cells = self.stepSpin.value, self.cellCount()
        contours = bool(self.contoursButton.checked)
        qc = self.qcWidget()
        synced = self.synced() and not fixed   # each study's own apical / mid / basal thirds
        mappings = {name: self.syncMapping(name) for name in ORIENTATIONS} if synced else {}
        for slot, data in self.shown():
            window = self.sliceWindow(data)
            for name in ORIENTATIONS:
                attribute, rowKey = self.rowOf(slot, name)
                panel = panels[attribute][0]
                size = panel.tileSize(rowKey)
                count = data.gated.slices[name][0].shape[0]
                if fixed:
                    shown = self.representativeSlices(slot, name, data)
                elif synced and slot == REST:   # the rest slices at the distances of the stress ones
                    stressFirst = self.firstSliceSpins[(STRESS, name)].value - 1
                    restSlice = self.reviewWidget().restSlice
                    shown = [restSlice(mappings[name], stressFirst + cell * step) for cell in range(cells)]
                    spin = self.firstSliceSpins[(REST, name)]
                    wasBlocked = spin.blockSignals(True)
                    spin.setValue(min(max(shown[0], 0), count - 1) + 1)
                    spin.blockSignals(wasBlocked)
                else:
                    first = self.firstSliceSpins[(slot, name)].value - 1
                    shown = [first + cell * step for cell in range(cells)]
                crop = self.tileCrop(slot, name, data)
                self.slicePositions[(slot, name)] = tuple(int(k) for k in shown)
                key = (id(data), size, self.colormapCombo.currentText, tuple(round(v, 6) for v in window),
                       tuple(shown), contours, round(float(panel.pixelRatio()), 2), tuple(crop))
                cached = self.tiles.get((slot, name))
                if cached and cached[0] == key:
                    continue
                valid = [k for k in shown if 0 <= k < count]
                title = (f"{SLOT_TITLES[slot]}  ·  " if len(self.slots()) > 1 else "") + ORIENTATION_TITLES[name]
                if valid and fixed:
                    title += "   —   " + ", ".join(f"{label.lower()} {k + 1}"
                                                   for label, k in zip(REPRESENTATIVE_LABELS[name], shown))
                elif valid:
                    title += (f"   —   slices {valid[0] + 1}–{valid[-1] + 1} of {count}"
                              + (f", every {step}" if step > 1 else ""))
                interpolation = data.gated.reference[name].GetAttribute("MPI.Interpolation") or DEFAULT_INTERPOLATION
                frames = []
                for index in range(data.count):
                    images = {}
                    if size and valid:
                        stack = cropStack(data.gated.slices[name][index][valid], crop)[:, ::-1, :]   # top = highest j
                        zoomed = zoomInPlane(np.asarray(stack), size, size, interpolation)
                        for k, values in zip(valid, zoomed):
                            rgb = colorize(values, window[1], table, window[0])
                            if contours and data.gated.results[index] is not None:
                                epi, endo = qc.contourMasks(data.gated, name, index, k, size, panel.pixelRatio(), crop)
                                rgb[epi] = CONTOUR_COLORS["epi"]
                                rgb[endo] = CONTOUR_COLORS["endo"]
                            images[k] = rgbToQImage(np.ascontiguousarray(rgb))
                    numbers = ([f"{label} {k + 1}" for label, k in zip(REPRESENTATIVE_LABELS[name], shown)]
                               if fixed else [k + 1 for k in shown])
                    frames.append([(images[k], number) if k in images else (None, None)
                                   for k, number in zip(shown, numbers)])
                self.tiles[(slot, name)] = (key, frames, title)
        low, high = self.windowRange.minimumValue, self.windowRange.maximumValue
        for panel, _, _ in panels.values():
            panel.setLegend(table, low, high)
        if fixed:
            for slot, _ in self.shown():
                self.refreshStatic3D(slot)
        if show:
            self.showFrame(self.tick)

    # --- polar maps -------------------------------------------------------------------------
    def perfusionWindow(self):
        source = self.windowRange if self.polarLinkCheck.checked else self.polarWindowRange
        low = float(source.minimumValue)
        return low, max(float(source.maximumValue), low + 1e-6)

    def functionWindow(self, kind):
        source = self.motionRange if kind == "motion" else self.thickeningRange
        low = float(source.minimumValue)
        return low, max(float(source.maximumValue), low + 1e-6)

    def refreshPolars(self):
        if not self.onScreen():
            return
        table = colorTableArray(self.colormapCombo.currentText)
        kinds = self.plotKinds()
        for slot, data in self.shown():
            title = SLOT_TITLES[slot]
            tags = gatedTags(slot)
            panel = existingPanel(polarPanelState(tags["polar"])) if "polar" in kinds else None
            if panel is not None:
                low, top = self.perfusionWindow()
                # its own color bar when the slice legend is not on the screen or shows another window
                ownLegend = not self.polarLinkCheck.checked or self.kind() in ("plots",) + DUAL_KINDS
                panel.setLegend(table if ownLegend else None, low, top)
                size = panel.polarSize()
                if size:
                    key = (id(data), int(size), self.colormapCombo.currentText, low, top)
                    cached = self.polarImages.get((slot, tags["polar"]))
                    if cached is None or cached[0] != key:
                        image, regions = self.reviewWidget().polarPercent(data.staticPolar, size)
                        rgb = colorize(np.nan_to_num(image), top, table, low)
                        rgb[np.isnan(image)] = 0
                        cached = (key, (rgbToQImage(rgb), {k: [float(v) for v in values] for k, values in regions.items()}))
                        self.polarImages[(slot, tags["polar"])] = cached
                    panel.setPolar(cached[1][0], cached[1][1], f"{title} perfusion",
                                   note="Static study, in % of the polar map maximum.")
            for kind in ("thickening", "motion"):
                if kind in kinds:
                    self.refreshFunctionPolar(slot, data, tags[kind], kind, table, title)

    def thickeningView(self):
        view = self.thickeningViewCombo.itemData(self.thickeningViewCombo.currentIndex)
        return view if view in wallLib.REGION_VIEWS else DEFAULT_THICKENING_VIEW

    def refreshFunctionPolar(self, slot, data, tag, kind, table, title):
        """The wall thickening or the wall motion polar map."""
        panel = existingPanel(polarPanelState(tag))
        if panel is None:
            return
        polar, name, unit = ((data.motion, "wall motion", "mm") if kind == "motion"
                             else (data.thickening, "wall thickening", "%"))
        if polar is None:
            panel.setLegend(None)
            panel.setPolar(None, None, f"{title} {name}", message=data.thickeningMessage)
            return
        low, top = self.functionWindow(kind)
        if kind == "motion":   # outward (dyskinetic) motion below 0 in its own color
            table, low, top = signedMotionTable(table, self.colormapCombo.currentText, low, top)
        panel.setLegend(table, low, top, unit=unit)
        size = panel.polarSize()
        if size:
            view = self.thickeningView() if kind == "thickening" else "pixels"
            key = (id(data), int(size), self.colormapCombo.currentText, low, top, view)
            cached = self.polarImages.get((slot, tag))
            if cached is None or cached[0] != key:
                image = wallLib.regionBullseye(polar, size, view)   # a count ratio per pixel is noisy
                rgb = colorize(np.nan_to_num(image, nan=low), top, table, low)
                rgb[np.isnan(image)] = 0
                regions = {overlay: [float(v) for v in wallLib.regionValues(polar, overlay)]
                           for overlay in wallLib.OVERLAYS}
                cached = (key, (rgbToQImage(rgb), regions))
                self.polarImages[(slot, tag)] = cached
            _, _, _, ed, es = data.phases
            frames = f"ED frame {data.gated.labels[ed]} to ES frame {data.gated.labels[es]}"
            note = (f"mm the endocardium moves inward (below 0: outward), {frames} (valve descent included)."
                    if kind == "motion"
                    else f"% thickening from the wall counts, {frames} (first harmonic over the cycle)"
                         + {"walls": ", mean of each of the five walls.", "segments": ", mean of each of the 17 segments.",
                            "pixels": ", per pixel."}[view])
            panel.setPolar(cached[1][0], cached[1][1], f"{title} {name}", note=note)

    def refreshVolumeCurves(self):
        if "volume" not in self.plotKinds():
            return
        for slot, data in self.shown():
            panel = existingPanel(volumeCurvePanelState(gatedTags(slot)["volume"]))
            if panel is None:
                continue
            chart = panel.chart
            title = f"{SLOT_TITLES[slot]}  ·  LV volume (ml)" if len(self.slots()) > 1 else "LV volume (ml)"
            if data.phases is None:
                chart.pointLabels = {}
                chart.overlayText = ""
                chart.setCurves([], [], title, "No LV cavity volumes (see Gated QC).")
                continue
            edv, esv, ef, ed, es = data.phases
            chart.pointLabels = {ed: "ED", es: "ES"}
            chart.overlayText = f"EF {ef:.0f} %"
            chart.marker = self.frameOf(data)
            chart.fromZero = True   # the volume axis from 0: the curve's swing shows the EF
            chart.setCurves(data.gated.labels, [("LV cavity", data.volumes, VOLUME_COLOR, "")],
                            f"{title}   EDV {edv:.0f} · ESV {esv:.0f}", mode="absolute")

    # --- 3D wall ---------------------------------------------------------------------------
    def representation(self):
        return self.representationCombo.itemData(self.representationCombo.currentIndex) or "surface"

    @staticmethod
    def viewNodes(viewTag):
        return [node for className in ("vtkMRMLModelNode", "vtkMRMLMarkupsFiducialNode")
                for node in slicer.util.getNodesByClass(className)
                if node.GetAttribute("MPI.GatedReviewView") == viewTag]

    def surfaceModel(self, slot):
        """The beating wall of a study's 3D view (the ED reference is not a model: see GhostLayer)."""
        return next((n for n in self.viewNodes(gatedTags(slot)["3d"]) if n.GetAttribute("MPI.Surface") == "gated"),
                    None)

    def refresh3D(self, slot, show=True):
        """The wall of every frame (made once), the floating wall labels and the camera of the static wall."""
        data = self.dataOf(slot)
        viewTag = gatedTags(slot)["3d"]
        widget = threeDWidgetByTag(viewTag) if "3d" in self.plotKinds() else None
        if data is None or widget is None:
            return
        try:
            import MPIMyocardium
            study = self.studyOf(slot)
            sample = self.staticSample(study)
            for index in range(data.count):
                data.surface(index, sample)
            viewNode = widget.mrmlViewNode()
            key = (id(data), study.GetID(), viewNode.GetID())
            model = self.surfaceModel(slot)
            if model is None or model.GetAttribute("MPI.DataKey") != repr(key):
                for node in self.viewNodes(viewTag):
                    slicer.mrmlScene.RemoveNode(node)
                model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{study.GetName()} gated LV 3D")
                model.CreateDefaultDisplayNodes()
                model.SetAttribute("MPI.Surface", "gated")
                model.SetAttribute("MPI.DataKey", repr(key))
                display = model.GetDisplayNode()
                display.SetVisibility2D(False)
                display.SetViewNodeIDs([viewNode.GetID()])
                for node in [model] + MPIMyocardium.addWallLabelNodes(data.staticResult, study, [viewNode.GetID()]):
                    node.SetAttribute("MPI.GatedReviewView", viewTag)
                    node.SetAttribute("MPI.GatedReviewOf", study.GetID())
                self.placeCamera(slot, data, study)
            model.SetAndObservePolyData(data.surfaces[self.frameOf(data)])
            display = model.GetDisplayNode()
            try:
                display.SetActiveScalar("Uptake", vtk.vtkAssignAttribute.POINT_DATA)
            except AttributeError:
                display.SetActiveScalarName("Uptake")
            display.SetScalarVisibility(True)
            display.SetSpecular(0.0)
            display.SetPointSize(1.0)
            display.SetLineWidth(1.0)
            display.SetBackfaceCulling(False)
            self.surfaceKeys[slot] = key
            self.updateRepresentation()
            self.update3DColors(slot)
            if show:
                self.showFrame(self.tick)
        except Exception:
            logging.error("MPI: could not make the gated 3D wall:\n" + traceback.format_exc())

    # --- static 3D wall with the slice planes (three slices per axis) ------------------------------
    def refreshStatic3D(self, slot):
        """The static wall of a study as in Myocardium (wireframe shell and wall labels) with semi-transparent
        planes at the slices shown, seen from anteroseptal and raised toward the base."""
        data = self.dataOf(slot)
        viewTag = gatedTags(slot)["static3d"]
        widget = threeDWidgetByTag(viewTag)
        if data is None or widget is None:
            return
        try:
            import MPIMyocardium
            study = self.studyOf(slot)
            viewNode = widget.mrmlViewNode()
            key = repr((id(data), study.GetID(), viewNode.GetID()))
            nodes = self.viewNodes(viewTag)
            shell = next((n for n in nodes if n.GetAttribute("MPI.Surface") == "shell"), None)
            if shell is None or shell.GetAttribute("MPI.DataKey") != key:
                for node in nodes:
                    slicer.mrmlScene.RemoveNode(node)
                shell = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{study.GetName()} gated slices 3D")
                shell.SetAndObservePolyData(MPIMyocardium.wallShellPolyData(data.staticResult, data.frame))
                shell.CreateDefaultDisplayNodes()
                display = shell.GetDisplayNode()
                display.SetColor(*MPIMyocardium.SHELL_COLOR)
                display.SetRepresentation(slicer.vtkMRMLDisplayNode.WireframeRepresentation)
                display.SetOpacity(MPIMyocardium.SHELL_OPACITY)
                self._flatDisplay(display, viewNode)
                shell.SetAttribute("MPI.Surface", "shell")
                shell.SetAttribute("MPI.DataKey", key)
                for node in [shell] + MPIMyocardium.addWallLabelNodes(data.staticResult, study, [viewNode.GetID()]):
                    node.SetAttribute("MPI.GatedReviewView", viewTag)
                    node.SetAttribute("MPI.GatedReviewOf", study.GetID())
                    notSaved(node)
                self.placeStaticCamera(slot, data)
            self.updateSlicePlanes(slot, data, viewTag, viewNode)
        except Exception:
            logging.error("MPI: could not make the static 3D wall:\n" + traceback.format_exc())

    @staticmethod
    def _flatDisplay(display, viewNode):
        display.SetVisibility2D(False)
        display.SetAmbient(1.0)
        display.SetDiffuse(0.0)
        display.SetSpecular(0.0)
        display.SetBackfaceCulling(False)
        display.SetScalarVisibility(False)
        display.SetViewNodeIDs([viewNode.GetID()])

    def updateSlicePlanes(self, slot, data, viewTag, viewNode):
        """One model per orientation: its three slices as filled planes, and their outlines."""
        import MPIMyocardium
        nodes = self.viewNodes(viewTag)
        center, _ = MPIMyocardium.cameraAnchor(data.frame, data.staticResult)
        half = wallLib.wallReach(data.staticResult) + SLICE_PLANE_MARGIN_MM
        for name in ORIENTATIONS:
            positions = self.slicePositions.get((slot, name))
            if positions is None:
                continue
            planes, outlines = self.slicePlanePolyData(data, name, positions, center, half)
            for part, polyData, opacity in (("plane", planes, SLICE_PLANE_OPACITY), ("outline", outlines, 0.9)):
                surface = f"{part}{name}"
                model = next((n for n in nodes if n.GetAttribute("MPI.Surface") == surface), None)
                if model is not None and model.GetAttribute("MPI.PlaneKey") == repr(positions):
                    continue
                if model is None:
                    model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode",
                                                               f"{self.studyOf(slot).GetName()} {name} slice {part}s")
                    model.CreateDefaultDisplayNodes()
                    display = model.GetDisplayNode()
                    display.SetColor(*SLICE_PLANE_COLORS[name])
                    display.SetOpacity(opacity)
                    display.SetLineWidth(2.0)
                    self._flatDisplay(display, viewNode)
                    model.SetAttribute("MPI.Surface", surface)
                    model.SetAttribute("MPI.GatedReviewView", viewTag)
                    model.SetAttribute("MPI.GatedReviewOf", self.studyOf(slot).GetID())
                    notSaved(model)
                model.SetAndObservePolyData(polyData)
                model.SetAttribute("MPI.PlaneKey", repr(positions))

    @staticmethod
    def slicePlanePolyData(data, name, shown, center, half):
        """(planes, outlines) in world coordinates of the slices shown of one orientation (slice indices of the
        reoriented volume of Processing): squares of 2 x half mm around the LV center (world)."""
        node = data.gated.reference[name]
        ijkToRas = vtk.vtkMatrix4x4()
        node.GetIJKToRASMatrix(ijkToRas)
        localToWorld = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(worldToLocalMatrix(node), localToWorld)
        matrix = slicer.util.arrayFromVTKMatrix(localToWorld) @ slicer.util.arrayFromVTKMatrix(ijkToRas)
        a = matrix[:3, 0] / max(np.linalg.norm(matrix[:3, 0]), 1e-9)   # in-plane axes of the slices
        b = matrix[:3, 1] / max(np.linalg.norm(matrix[:3, 1]), 1e-9)
        normal = np.cross(a, b)
        normal /= max(np.linalg.norm(normal), 1e-9)
        center = np.asarray(center, dtype=float)
        planes, outlines = vtk.vtkAppendPolyData(), vtk.vtkAppendPolyData()
        for k in sorted(set(shown)):
            point = (matrix @ np.array([0.0, 0.0, float(k), 1.0]))[:3]
            middle = center - np.dot(center - point, normal) * normal   # the LV center on this slice
            source = vtk.vtkPlaneSource()
            source.SetOrigin(*(middle - half * a - half * b))
            source.SetPoint1(*(middle + half * a - half * b))
            source.SetPoint2(*(middle - half * a + half * b))
            source.Update()
            edges = vtk.vtkFeatureEdges()
            edges.SetInputData(source.GetOutput())
            edges.BoundaryEdgesOn()
            edges.FeatureEdgesOff()
            edges.NonManifoldEdgesOff()
            edges.ManifoldEdgesOff()
            edges.Update()
            planes.AddInputData(source.GetOutput())
            outlines.AddInputData(edges.GetOutput())
        result = []
        for append in (planes, outlines):
            if append.GetNumberOfInputConnections(0) == 0:
                result.append(vtk.vtkPolyData())
                continue
            append.Update()
            polyData = vtk.vtkPolyData()
            polyData.DeepCopy(append.GetOutput())
            result.append(polyData)
        return tuple(result)

    def placeStaticCamera(self, slot, data):
        """From anteroseptal (septum and anterior wall), raised STATIC_VIEW_ELEVATION_DEG toward the base,
        the base up."""
        viewTag = gatedTags(slot)["static3d"]
        frame = data.frame
        elevation = np.radians(STATIC_VIEW_ELEVATION_DEG)
        anteroseptal = np.asarray(frame["w"], dtype=float) + np.asarray(frame["s"], dtype=float)
        anteroseptal /= max(np.linalg.norm(anteroseptal), 1e-9)
        base = -np.asarray(frame["u"], dtype=float)
        direction = np.cos(elevation) * anteroseptal + np.sin(elevation) * base

        def place():
            widget = threeDWidgetByTag(viewTag)
            if widget is None or self.dataOf(slot) is not data:
                return
            try:
                import MPIMyocardium
                viewNode = widget.mrmlViewNode()
                MPIMyocardium.style3DView(viewNode)
                scale = 0.75 * (wallLib.wallReach(data.staticResult) + LABEL_DISTANCE_MM) + 10.0
                MPIMyocardium.placeCamera(viewNode, frame, data.staticResult, parallelScale=scale,
                                          direction=direction, up=base)
            except Exception:
                logging.warning("MPI: could not place the static 3D camera:\n" + traceback.format_exc())
        place()
        qt.QTimer.singleShot(400, place)    # again after the layout and the new models have settled
        qt.QTimer.singleShot(1200, place)

    def updateGhost(self, slots=None):
        """The ED wall as the reference of the beating wall (shown when ticked and the volumes are known)."""
        representation = self.ghostRepresentationCombo.itemData(self.ghostRepresentationCombo.currentIndex)
        for slot in (slots or (STRESS, REST)):
            data = self.dataOf(slot)
            widget = threeDWidgetByTag(gatedTags(slot)["3d"])
            polyData = None
            if (data is not None and widget is not None and slot in self.slots() and "3d" in self.plotKinds()
                    and self.surfaceModel(slot) is not None and self.ghostCheck.checked):
                ed = data.phases[3] if data.phases else None
                polyData = data.surfaces.get(ed) if ed is not None else None
            self.ghostLayers[slot].update(widget, polyData, GHOST_COLOR, representation or GHOST_REPRESENTATION,
                                          float(self.ghostOpacitySlider.value))

    def updateRepresentation(self):
        """Surface, wireframe or points: only the rendering of the same models changes."""
        representation = self.representation()
        for slot in (STRESS, REST):
            model = self.surfaceModel(slot)
            if model is None or model.GetDisplayNode() is None:
                continue
            display = model.GetDisplayNode()
            display.SetRepresentation({"points": 0, "wireframe": 1}.get(representation, 2))
            display.SetAmbient(1.0 if representation == "surface" else 0.7)   # flat colors: the counts, not the shading
            display.SetDiffuse(0.0 if representation == "surface" else 0.3)
            display.SetOpacity(WALL_OPACITY.get(representation, 1.0))

    def update3DColors(self, slot=None):
        colorNode = colorNodeForLabel(self.wallColormapCombo.currentText)
        for current in ((slot,) if slot else (STRESS, REST)):
            model = self.surfaceModel(current)
            if model is None or model.GetDisplayNode() is None:
                continue
            display = model.GetDisplayNode()
            if colorNode:
                display.SetAndObserveColorNodeID(colorNode.GetID())
            display.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
            display.SetScalarRange(*self.perfusionWindow())
        self.updateGhost((slot,) if slot else None)

    def placeCamera(self, slot, data, study):
        viewTag = gatedTags(slot)["3d"]

        def place():
            widget = threeDWidgetByTag(viewTag)
            if widget is None or self.dataOf(slot) is not data:
                return
            try:
                import MPIMyocardium
                viewNode = widget.mrmlViewNode()
                MPIMyocardium.style3DView(viewNode)
                viewNode.SetRenderMode(slicer.vtkMRMLViewNode.Orthographic)
                scale = 0.75 * (wallLib.wallReach(data.staticResult) + LABEL_DISTANCE_MM) + 10.0
                MPIMyocardium.placeCamera(viewNode, data.frame, data.staticResult, parallelScale=scale)
                self.updateGhost((slot,))   # the overlay layer follows the view's (possibly new) camera
            except Exception:
                logging.warning("MPI: could not place the gated 3D camera:\n" + traceback.format_exc())
            self.updateCameraLink()
        place()
        qt.QTimer.singleShot(400, place)    # again after the layout and the new models have settled
        qt.QTimer.singleShot(1200, place)

    def updateCameraLink(self):
        """Link the stress and rest 3D views of the layout (when both are shown), each around its own heart."""
        anchors = {}
        try:
            import MPIMyocardium
            if self.link3DCheck.checked and "3d" in self.plotKinds():
                for slot, data in self.shown():
                    tag = gatedTags(slot)["3d"]
                    if threeDWidgetByTag(tag) is not None:
                        anchors[tag] = MPIMyocardium.cameraAnchor(data.frame, data.staticResult)
        except Exception:
            logging.warning("MPI: could not link the 3D views:\n" + traceback.format_exc())
        self.cameraLink.link(anchors)

    @staticmethod
    def setCornerText(widget, text, upperLeft=""):
        """Study, frame and cavity volume in the lower left corner; upperLeft (the EF) large and bold in the
        upper left (its own text actor: the corner annotation has one font for all corners)."""
        try:
            view = widget.threeDView()
            corner = view.cornerAnnotation()
            corner.SetText(vtk.vtkCornerAnnotation.LowerLeft, text)
            corner.SetText(vtk.vtkCornerAnnotation.UpperLeft, "")
            renderer = view.renderWindow().GetRenderers().GetFirstRenderer()
            if renderer is not None:
                actor = _BIG_TEXT_ACTORS.get(widget.mrmlViewNode().GetID())
                if actor is None or not renderer.HasViewProp(actor):
                    actor = vtk.vtkTextActor()
                    _BIG_TEXT_ACTORS[widget.mrmlViewNode().GetID()] = actor
                    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
                    actor.SetPosition(0.02, 0.97)
                    prop = actor.GetTextProperty()
                    prop.SetFontSize(EF_FONT_SIZE)
                    prop.SetBold(True)
                    prop.SetShadow(True)
                    prop.SetColor(1.0, 1.0, 1.0)
                    prop.SetJustificationToLeft()
                    prop.SetVerticalJustificationToTop()
                    renderer.AddViewProp(actor)
                actor.SetInput(upperLeft or "")
                actor.SetVisibility(bool(upperLeft))
            corner.SetMaximumFontSize(14)
            corner.SetMinimumFontSize(9)
            corner.GetTextProperty().SetColor(0.9, 0.9, 0.9)
            corner.SetVisibility(True)
            view.forceRender()
        except Exception:
            logging.debug("MPI: no corner annotation in this 3D view:\n" + traceback.format_exc())


class MPIGatedTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_Thickening()

    def test_Thickening(self):
        """A wall that gets brighter toward the frame with the smallest cavity thickens."""
        volumes = [100, 90, 70, 55, 50, 58, 75, 92]
        self.assertEqual(ejectionFraction(volumes)[3:], (0, 4))
        rho, theta = np.linspace(1, 0, 10), np.linspace(0, 2 * np.pi, 24, endpoint=False)
        polars = [{"values": np.full((10, 24), 100.0 * (1.0 + 0.4 * (100 - v) / 50.0)), "rho": rho, "theta": theta}
                  for v in volumes]
        thickening = wallLib.thickeningMap(polars, 0, 4)
        self.assertGreater(float(np.median(thickening["values"])), 25.0)
        self.delayDisplay("Thickening test passed")

    def test_Motion(self):
        """An LV whose endocardium moves 8 mm inward from ED to ES (same valve plane) shows about 8 mm."""
        grid = wallLib.WallGrid(np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), 55.0)
        shape = (grid.nRows, grid.nTheta)

        def wall(endo):
            return {"grid": grid, "endo": np.full(shape, endo), "mid": np.full(shape, endo + 5.0),
                    "epi": np.full(shape, endo + 10.0), "valvePlane": (-30.0, 0.0, 0.0)}
        endo = [22.0 - 8.0 * (0.5 - 0.5 * np.cos(2 * np.pi * k / 8)) for k in range(8)]
        motion = wallLib.motionMap([wall(e) for e in endo], 0, 4)
        self.assertAlmostEqual(float(np.median(motion["values"])), 8.0, delta=0.5)
        self.delayDisplay("Motion test passed")

# saved Advanced parameters of every step, this module's constants included (see parameters.py)
parametersLib.applySaved()
