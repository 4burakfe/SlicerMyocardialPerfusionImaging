"""
Myocardial Perfusion SPECT - 6. Gated Review

STRESS GATED layout of Study 1 (stress):
  left           SA / HLA / VLA rows of the gate frames as in Perfusion (same grid, slices, colors and
                 window), played as a cine: every tile shows the same frame, with that frame's
                 endo- and epicardium;
  top right      the perfusion polar map (as in Perfusion) next to a function polar map: the wall
                 thickening (count-based: ES / ED wall counts of every ray, first harmonic over the
                 cycle) or the wall motion (mm the endocardium moves from ED to ES);
  bottom right   the LV cavity volume of every frame (EDV, ESV, EF; the frame shown is marked) next to
                 the cine 3D wall coloured by the perfusion (SSP style), as a surface, wireframe or points.
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
from MyocardialPerfusionSPECTLib import myocardium as wallLib  # noqa: E402
from MyocardialPerfusionSPECTLib.resampling import DEFAULT_INTERPOLATION, zoomInPlane  # noqa: E402
from MyocardialPerfusionSPECTLib.views import (  # noqa: E402
    GATED_REVIEW_LAYOUT_ID, GATED_POLAR_TAG, GATED_THICKENING_TAG, GATED_MOTION_TAG, GATED_3D_VIEW, ORIENTATIONS,
    ORIENTATION_TITLES, gatedReviewLayoutXml, registerLayout, threeDWidgetByTag)
from MyocardialPerfusionSPECTLib.display import (  # noqa: E402
    DEFAULT_GRID_ROWS, DEFAULT_GRID_COLUMNS, COLORMAPS, colorTableArray, colorize, colorNodeForLabel,
    rgbToQImage, gatedReviewMosaicPanel, volumeCurvePanel, polarPanel, polarPanelState, existingPanel)
from MyocardialPerfusionSPECTLib.widgets import styleButton, StudyPairSelector  # noqa: E402

STRESS, REST = "stress", "rest"
SLOT_TITLES = {STRESS: "STRESS", REST: "REST"}
MODES = {"stressGated": ("STRESS GATED", STRESS, "Stress gated study: cine slices, perfusion and thickening "
                                                 "polar maps, LV volume curve and cine 3D wall")}
MOSAIC_STATE, VOLUME_STATE = "_mpiGatedReviewMosaicState", "_mpiGatedVolumeState"
CONTOUR_COLORS = {"epi": (64, 220, 90), "endo": (60, 190, 255)}
OVERLAY_CHOICES = (("Vessels (LAD / RCA / LCX)", "vessels"), ("17 segments", "segments"), ("None", ""))
REPRESENTATIONS = (("Surface", "surface"), ("Wireframe", "wireframe"), ("Points", "points"))
THICKENING_WINDOW = (-20.0, 150.0, 0.0, 40.0)   # range and default window of the thickening map, %
MOTION_WINDOW = (-10.0, 25.0, 0.0, 10.0)        # the same of the wall motion map, mm (QGS: 0-10 mm)
POLAR_TAGS = (GATED_POLAR_TAG, GATED_THICKENING_TAG, GATED_MOTION_TAG)
DEFAULT_FPS = 8
VOLUME_COLOR = (255, 200, 80)
LABEL_DISTANCE_MM = 45.0
DEFAULT_WALL_COLORMAP = "Hel"     # 3D wall colors (its own setting, not taken over from Perfusion)
DEFAULT_REPRESENTATION = "wireframe"
GHOST_OPACITY = 0.35                # ED reference (drawn on top, so it need not be opaque)
GHOST_REPRESENTATION = "points"
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
        self.data = {}             # slot -> (key, GatedReviewData)
        self.frameIndex = 0
        self.tiles = {}            # orientation -> (key, [frame][cell] (QImage or None, slice number or None), title)
        self.polarImages = {}      # polar map tag -> (key, (QImage, region values))
        self.inherited = None      # Perfusion settings last taken over
        self.staticSamples = {}    # study ID -> (key, sampler of the volume the wall was detected on)
        self.surfaceKey = None     # 3D models made for this data
        self.ghostLayer = GhostLayer()

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.studySelector = StudyPairSelector(self.layout)
        self.studyLabel = qt.QLabel()
        self.studyLabel.wordWrap = True
        self.layout.addWidget(self.studyLabel)

        # Cine
        cine = ctk.ctkCollapsibleButton()
        cine.text = "Cine"
        self.layout.addWidget(cine)
        cineForm = qt.QFormLayout(cine)
        playRow = qt.QHBoxLayout()
        self.playButton = styleButton(qt.QPushButton("▶  Play"), "primary")
        self.playButton.checkable = True
        self.playButton.toolTip = "Start / stop the cine of the slices, the volume curve and the 3D wall"
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
            "Color window (min - max) of the slices in % of the hottest frame near the LV (all frames on "
            "one scale, so the count changes over the cycle show)")
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
        self.stepSpin = qt.QSpinBox()
        self.stepSpin.setRange(1, 10)
        self.stepSpin.setValue(1)
        self.stepSpin.toolTip = "Slices advanced from one tile to the next (1 = consecutive slices)"
        form.addRow("Slice step:", self.stepSpin)
        self.firstSliceSpins = {}
        for name in ORIENTATIONS:
            spin = qt.QSpinBox()
            spin.setRange(1, 1)
            spin.toolTip = f"First slice of the {ORIENTATION_TITLES[name]} row (mouse wheel over the row also scrolls)"
            self.firstSliceSpins[name] = spin
            form.addRow(f"First {name} slice:", spin)
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
        self.polarLinkCheck.toolTip = ("Use the slice window for the perfusion polar map and the 3D wall (in % of the "
                                       "polar map maximum); untick to window them on their own")
        polarForm.addRow("Perfusion window:", self.polarLinkCheck)
        self.polarWindowRange = self._windowWidget(
            "Color window of the perfusion polar map and the 3D wall, % of the polar map maximum")
        polarForm.addRow("", self.polarWindowRange)
        self.thickeningRange = self._windowWidget(
            "Color window of the wall thickening map, % thickening (ES / ED wall counts)", *THICKENING_WINDOW)
        polarForm.addRow("Thickening window:", self.thickeningRange)
        self.motionRange = self._windowWidget(
            "Color window of the wall motion map, mm of endocardial motion from ED to ES", *MOTION_WINDOW, suffix=" mm")
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
        self.ghostOpacitySlider.minimum, self.ghostOpacitySlider.maximum = 0.05, 1.0
        self.ghostOpacitySlider.singleStep = 0.05
        self.ghostOpacitySlider.decimals = 2
        self.ghostOpacitySlider.value = GHOST_OPACITY
        self.ghostOpacitySlider.toolTip = "Opacity of the ED reference"
        ghostRow.addWidget(self.ghostRepresentationCombo)
        ghostRow.addWidget(self.ghostOpacitySlider, 1)
        threeDForm.addRow("ED reference:", ghostRow)

        # Layouts
        self.layoutButtons = {}
        self.layoutGroup = qt.QButtonGroup()
        self.layoutGroup.exclusive = True
        layoutBox = qt.QGroupBox("Layout")
        layoutGrid = qt.QVBoxLayout(layoutBox)
        for key, (text, _, toolTip) in MODES.items():
            button = qt.QPushButton(text)
            button.checkable = True
            button.toolTip = toolTip
            self.layoutGroup.addButton(button)
            layoutGrid.addWidget(button)
            self.layoutButtons[key] = button
        self.layout.addWidget(layoutBox)
        self.backButton = qt.QPushButton("‹ Gated QC")
        self.layout.addWidget(self.backButton)
        self.layout.addStretch(1)

        self.cineTimer = qt.QTimer()
        self.cineTimer.setInterval(int(1000 / DEFAULT_FPS))
        self.cineTimer.connect("timeout()", self.nextFrame)
        self.resizeTimer = qt.QTimer()
        self.resizeTimer.setSingleShot(True)
        self.resizeTimer.setInterval(150)
        self.resizeTimer.connect("timeout()", lambda: self.refresh(rebuild3D=False))

        self.playButton.connect("toggled(bool)", self.onPlayToggled)
        self.speedSpin.connect("valueChanged(int)", lambda v: self.cineTimer.setInterval(int(1000 / max(v, 1))))
        self.frameSlider.connect("valueChanged(int)", self.onFrameSliderMoved)
        self.colormapCombo.connect("currentIndexChanged(int)", lambda _: self.refresh())
        self.windowRange.connect("valuesChanged(double,double)", lambda a, b: self.refresh(rebuild3D=False))
        for spin in [self.gridRowsSpin, self.gridColumnsSpin, self.stepSpin] + list(self.firstSliceSpins.values()):
            spin.connect("valueChanged(int)", lambda _: self.refreshSlices())
        self.contoursButton.connect("toggled(bool)", lambda _: self.refreshSlices())
        self.inheritButton.connect("clicked(bool)", lambda: self.inheritPerfusion(force=True))
        self.polarLinkCheck.connect("toggled(bool)", lambda _: self.refresh(rebuild3D=False))
        self.polarWindowRange.connect("valuesChanged(double,double)", lambda a, b: self.refresh(rebuild3D=False))
        self.thickeningRange.connect("valuesChanged(double,double)", lambda a, b: self.refreshPolars())
        self.motionRange.connect("valuesChanged(double,double)", lambda a, b: self.refreshPolars())
        self.overlayCombo.connect("currentIndexChanged(int)", lambda _: self.onOverlaysChanged())
        self.scoresButton.connect("toggled(bool)", lambda _: self.onOverlaysChanged())
        self.representationCombo.connect("currentIndexChanged(int)", lambda _: self.updateRepresentation())
        self.ghostCheck.connect("toggled(bool)", lambda _: self.updateGhost())
        self.ghostRepresentationCombo.connect("currentIndexChanged(int)", lambda _: self.updateGhost())
        self.ghostOpacitySlider.connect("valueChanged(double)", lambda _: self.updateGhost())
        self.wallColormapCombo.connect("currentIndexChanged(int)", lambda _: self.update3DColors())
        for key, button in self.layoutButtons.items():
            button.connect("clicked(bool)", lambda _, k=key: self.show(k))
        self.backButton.connect("clicked(bool)", lambda: session.selectStep("gatedqc"))
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

    def enter(self):
        self.studySelector.refresh()
        self.updateState()
        if self.available(self.slot())[0]:
            self.show(self.mode)

    def exit(self):
        self.playButton.checked = False

    def cleanup(self):
        self.cineTimer.stop()
        self.resizeTimer.stop()
        session.removeListener(self.onSessionChanged)
        self.studySelector.cleanup()
        self.ghostLayer.detach()
        for attribute in (MOSAIC_STATE, VOLUME_STATE, polarPanelState(GATED_POLAR_TAG), polarPanelState(GATED_THICKENING_TAG),
                          polarPanelState(GATED_MOTION_TAG)):
            panel = existingPanel(attribute)
            if panel is not None:
                panel.onResized = None
                if hasattr(panel, "onScroll"):
                    panel.onScroll = None
        self.removeObservers()

    def onSceneClosed(self, caller=None, event=None):
        self.playButton.checked = False
        self.data, self.tiles, self.polarImages, self.staticSamples = {}, {}, {}, {}
        self.surfaceKey = None
        self.ghostLayer.update(None, None, None)
        for attribute in (MOSAIC_STATE, VOLUME_STATE, polarPanelState(GATED_POLAR_TAG), polarPanelState(GATED_THICKENING_TAG),
                          polarPanelState(GATED_MOTION_TAG)):
            panel = existingPanel(attribute)
            if panel is not None:
                panel.clear()
        self.updateState()

    def onSessionChanged(self):
        self.updateState()
        if self.parent.isVisible() and self.onScreen() and self.available(self.slot())[0]:
            self.show(self.mode)

    # --- studies -------------------------------------------------------------------
    def slot(self, mode=None):
        return MODES[mode or self.mode][1]

    @staticmethod
    def studyOf(slot):
        return session.stressRestStudies()[slot]

    @staticmethod
    def qcWidget():
        return slicer.modules.mpigatedqc.widgetRepresentation().self()

    def available(self, slot):
        return self.qcWidget().available(slot)

    def onScreen(self):
        return slicer.app.layoutManager().layout == GATED_REVIEW_LAYOUT_ID

    def updateState(self):
        lines = []
        for slot, title in ((STRESS, "Study 1"), (REST, "Study 2")):
            study = self.studyOf(slot)
            if study is None:
                continue
            ready, reason = self.available(slot)
            sequence = gatedSequenceOf(study)
            text = f"{title}: <b>{study.GetName()}</b>"
            if sequence is not None:
                text += f", gated <b>{sequence.GetName()}</b> ({sequence.GetNumberOfDataNodes()} frames)"
            if not ready:
                text += f" <span style='color:#d08030'>({reason})</span>"
            lines.append(text)
        if not lines:
            lines = ["No study selected. Pick Study 1 / Study 2 above or set the roles in Data roles."]
        self.studyLabel.text = "<br>".join(lines)
        for key, button in self.layoutButtons.items():
            button.enabled = self.available(self.slot(key))[0]
        if self.mode in self.layoutButtons:
            self.layoutButtons[self.mode].checked = True
        self.polarWindowRange.enabled = not self.polarLinkCheck.checked

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
        self.tiles, self.polarImages, self.surfaceKey = {}, {}, None
        self.frameIndex = data.phases[3] if data.phases else 0
        wasBlocked = self.frameSlider.blockSignals(True)
        self.frameSlider.setRange(0, max(data.count - 1, 0))
        self.frameSlider.setValue(self.frameIndex)
        self.frameSlider.blockSignals(wasBlocked)
        self.fitFirstSlices(data)
        return data

    def current(self):
        cached = self.data.get(self.slot())
        return cached[1] if cached else None

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

    def perfusionSettings(self, slot):
        review = self.reviewWidget()
        return {"colormap": review.colormapCombo.currentText,
                "window": (review.windowRanges[slot].minimumValue, review.windowRanges[slot].maximumValue),
                "grid": tuple(review.grids["single"]), "step": review.stepSpin.value,
                "first": {name: (review.firstSliceSpins[(slot, name)].value, review.firstSliceSpins[(slot, name)].maximum)
                          for name in ORIENTATIONS},
                "polarLinked": bool(review.polarLinkCheck.checked),
                "polarWindow": (review.polarWindowRanges[slot].minimumValue, review.polarWindowRanges[slot].maximumValue),
                "overlay": review.overlayCombo.currentIndex, "scores": bool(review.scoresButton.checked)}

    def inheritPerfusion(self, force=False):
        """Take over the Perfusion slice and polar map settings of the slot's study when they changed
        there since last time (or always with force)."""
        try:
            settings = self.perfusionSettings(self.slot())
        except Exception:
            logging.warning("MPI: could not read the Perfusion settings:\n" + traceback.format_exc())
            return
        if settings == self.inherited and not force:
            return
        self.inherited = settings
        widgets = [self.colormapCombo, self.windowRange, self.gridRowsSpin, self.gridColumnsSpin, self.stepSpin,
                   self.polarLinkCheck, self.polarWindowRange, self.overlayCombo, self.scoresButton] + \
            list(self.firstSliceSpins.values())
        blocked = [w.blockSignals(True) for w in widgets]
        try:
            index = self.colormapCombo.findText(settings["colormap"])
            if index >= 0:
                self.colormapCombo.setCurrentIndex(index)
            self.windowRange.minimumValue, self.windowRange.maximumValue = settings["window"]
            self.gridRowsSpin.setValue(settings["grid"][0])
            self.gridColumnsSpin.setValue(settings["grid"][1])
            self.stepSpin.setValue(settings["step"])
            self.polarLinkCheck.checked = settings["polarLinked"]
            self.polarWindowRange.minimumValue, self.polarWindowRange.maximumValue = settings["polarWindow"]
            self.overlayCombo.setCurrentIndex(settings["overlay"])
            self.scoresButton.checked = settings["scores"]
            for name, (value, maximum) in settings["first"].items():
                spin = self.firstSliceSpins[name]
                if maximum > 1:      # the gate frames are resliced on the Perfusion grids: same slice numbers
                    spin.setRange(1, max(maximum, spin.maximum))
                    spin.setValue(value)
        finally:
            for w, b in zip(widgets, blocked):
                w.blockSignals(b)
        self.updateState()
        if force:
            self.refresh(rebuild3D=False)

    def fitFirstSlices(self, data):
        """Slice ranges of the frames; rows not set by Perfusion start around the LV center."""
        for name in ORIENTATIONS:
            n = data.gated.slices[name][0].shape[0]
            spin = self.firstSliceSpins[name]
            wasBlocked = spin.blockSignals(True)
            unset = spin.maximum <= 1
            spin.setRange(1, n)
            if unset:
                spin.setValue(max(0, (n - self.cellsSpan()) // 2) + 1)
            spin.blockSignals(wasBlocked)

    def cellCount(self):
        return self.gridRowsSpin.value * self.gridColumnsSpin.value

    def cellsSpan(self):
        return (self.cellCount() - 1) * self.stepSpin.value + 1

    # --- showing ---------------------------------------------------------------------
    def show(self, mode=None):
        mode = mode or self.mode
        ready, reason = self.available(self.slot(mode))
        if not ready:
            slicer.util.warningDisplay(f"The gated review cannot be shown: {reason}.", windowTitle="Gated Review")
            return
        self.mode = mode
        self.layoutButtons[mode].checked = True
        self.inheritPerfusion()
        with slicer.util.tryWithErrorDisplay("Could not prepare the gated review.", waitCursor=True):
            self.reviewData(self.slot())
            panel = gatedReviewMosaicPanel()
            panel.onScroll = self.onScroll
            panel.onResized = self.resizeTimer.start
            panel.gridRows, panel.gridColumns = self.gridRowsSpin.value, self.gridColumnsSpin.value
            panel.setRowKeys([(name, ORIENTATION_TITLES[name]) for name in ORIENTATIONS])
            for tag in POLAR_TAGS:
                polar = polarPanel(tag)
                polar.onResized = self.resizeTimer.start
                polar.overlay, polar.showScores = self.overlay(), self.scoresButton.checked
            volumeCurvePanel()
            registerLayout(GATED_REVIEW_LAYOUT_ID, gatedReviewLayoutXml())
            slicer.app.layoutManager().setLayout(GATED_REVIEW_LAYOUT_ID)
            slicer.app.processEvents()
            self.surfaceKey = None
            self.refresh()

    def refresh(self, rebuild3D=True):
        if self.current() is None or not self.onScreen():
            return
        self.refreshSlices(show=False)
        self.refreshPolars()
        self.refreshVolumeCurve()
        if rebuild3D or self.surfaceKey is None:
            self.refresh3D(show=False)
        else:
            self.update3DColors()
        self.showFrame(self.frameIndex)

    def onScroll(self, key, direction):
        spin = self.firstSliceSpins.get(key)
        if spin:
            spin.setValue(spin.value + direction * self.stepSpin.value)

    def overlay(self):
        return self.overlayCombo.itemData(self.overlayCombo.currentIndex) or None

    def onOverlaysChanged(self):
        for tag in POLAR_TAGS:
            panel = existingPanel(polarPanelState(tag))
            if panel is not None:
                panel.setOverlays(self.overlay(), self.scoresButton.checked)

    # --- cine ----------------------------------------------------------------------------
    def onPlayToggled(self, playing):
        self.playButton.text = "■  Stop" if playing else "▶  Play"
        if playing and self.current() is not None:
            self.cineTimer.start()
        else:
            self.cineTimer.stop()

    def nextFrame(self):
        data = self.current()
        if data is None or not self.onScreen() or not data.count:
            self.playButton.checked = False
            return
        self.showFrame((self.frameIndex + 1) % data.count)

    def onFrameSliderMoved(self, value):
        self.playButton.checked = False
        self.showFrame(int(value))

    def showFrame(self, index):
        """Show one frame everywhere: the slice rows, the mark on the volume curve and the 3D wall."""
        data = self.current()
        if data is None or not data.count:
            return
        index = int(index) % data.count
        self.frameIndex = index
        wasBlocked = self.frameSlider.blockSignals(True)
        self.frameSlider.setValue(index)
        self.frameSlider.blockSignals(wasBlocked)
        volume = data.volumes[index]
        self.frameLabel.text = f"{data.gated.labels[index]} of {data.count}"
        panel = existingPanel(MOSAIC_STATE)
        if panel is not None:
            for name in ORIENTATIONS:
                cached = self.tiles.get(name)
                if cached:
                    _, frames, title = cached
                    panel.setRow(name, frames[index], f"{title}   ·   frame {data.gated.labels[index]} of {data.count}")
        curve = existingPanel(VOLUME_STATE)
        if curve is not None:
            curve.chart.setMarker(index)
        model = self.surfaceModel()
        if model is not None:
            polyData = data.surfaces.get(index)
            if polyData is not None and model.GetPolyData() is not polyData:
                model.SetAndObservePolyData(polyData)
            widget = threeDWidgetByTag(GATED_3D_VIEW)
            if widget is not None:
                self.setCornerText(widget, f"{SLOT_TITLES[self.slot()]} GATED   frame {data.gated.labels[index]} of "
                                           f"{data.count}" + ("" if volume is None else f"\nLV cavity {volume:.0f} ml"),
                                   f"EF {data.phases[2]:.0f} %" if data.phases else "")

    # --- slices -----------------------------------------------------------------------------
    def sliceWindow(self, data):
        reference = max(data.gated.peaks) if data.gated.peaks else 1.0
        low = reference * self.windowRange.minimumValue / 100.0
        return low, max(reference * self.windowRange.maximumValue / 100.0, low + 1e-9)

    def refreshSlices(self, show=True):
        """Tiles of every frame of every row at the current settings, made once (the cine only swaps them)."""
        data = self.current()
        panel = existingPanel(MOSAIC_STATE)
        if data is None or panel is None or not self.onScreen():
            return
        panel.gridRows, panel.gridColumns = self.gridRowsSpin.value, self.gridColumnsSpin.value
        table = colorTableArray(self.colormapCombo.currentText)
        window = self.sliceWindow(data)
        step, cells = self.stepSpin.value, self.cellCount()
        contours = bool(self.contoursButton.checked)
        qc = self.qcWidget()
        for name in ORIENTATIONS:
            size = panel.tileSize(name)
            first = self.firstSliceSpins[name].value - 1
            count = data.gated.slices[name][0].shape[0]
            shown = [first + cell * step for cell in range(cells)]
            key = (id(data), size, self.colormapCombo.currentText, tuple(round(v, 6) for v in window), tuple(shown),
                   contours, round(float(panel.pixelRatio()), 2))
            cached = self.tiles.get(name)
            if cached and cached[0] == key:
                continue
            valid = [k for k in shown if 0 <= k < count]
            title = ORIENTATION_TITLES[name]
            if valid:
                title += f"   —   slices {valid[0] + 1}–{valid[-1] + 1} of {count}" + (f", every {step}" if step > 1 else "")
            interpolation = data.gated.reference[name].GetAttribute("MPI.Interpolation") or DEFAULT_INTERPOLATION
            frames = []
            for index in range(data.count):
                images = {}
                if size and valid:
                    stack = data.gated.slices[name][index][valid][:, ::-1, :]   # top row = highest j
                    zoomed = zoomInPlane(np.asarray(stack), size, size, interpolation)
                    for k, values in zip(valid, zoomed):
                        rgb = colorize(values, window[1], table, window[0])
                        if contours and data.gated.results[index] is not None:
                            epi, endo = qc.contourMasks(data.gated, name, index, k, size, panel.pixelRatio())
                            rgb[epi] = CONTOUR_COLORS["epi"]
                            rgb[endo] = CONTOUR_COLORS["endo"]
                        images[k] = rgbToQImage(np.ascontiguousarray(rgb))
                frames.append([(images[k], k + 1) if k in images else (None, None) for k in shown])
            self.tiles[name] = (key, frames, title)
        low, high = self.windowRange.minimumValue, self.windowRange.maximumValue
        panel.setLegend(table, low, high)
        if show:
            self.showFrame(self.frameIndex)

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
        data = self.current()
        if data is None:
            return
        table = colorTableArray(self.colormapCombo.currentText)
        title = SLOT_TITLES[self.slot()]
        panel = existingPanel(polarPanelState(GATED_POLAR_TAG))
        if panel is not None:
            low, top = self.perfusionWindow()
            panel.setLegend(None if self.polarLinkCheck.checked else table, low, top)
            size = panel.polarSize()
            if size:
                key = (id(data), int(size), self.colormapCombo.currentText, low, top)
                cached = self.polarImages.get(GATED_POLAR_TAG)
                if cached is None or cached[0] != key:
                    image, regions = self.reviewWidget().polarPercent(data.staticPolar, size)
                    rgb = colorize(np.nan_to_num(image), top, table, low)
                    rgb[np.isnan(image)] = 0
                    cached = (key, (rgbToQImage(rgb), {k: [float(v) for v in values] for k, values in regions.items()}))
                    self.polarImages[GATED_POLAR_TAG] = cached
                panel.setPolar(cached[1][0], cached[1][1], f"{title} perfusion",
                               note="Static study, in % of the polar map maximum.")
        for tag, kind in ((GATED_THICKENING_TAG, "thickening"), (GATED_MOTION_TAG, "motion")):
            self.refreshFunctionPolar(data, tag, kind, table, title)

    def refreshFunctionPolar(self, data, tag, kind, table, title):
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
        panel.setLegend(table, low, top, unit=unit)
        size = panel.polarSize()
        if size:
            key = (id(data), int(size), self.colormapCombo.currentText, low, top)
            cached = self.polarImages.get(tag)
            if cached is None or cached[0] != key:
                image, _ = wallLib.bullseye(polar, size)
                rgb = colorize(np.nan_to_num(image, nan=low), top, table, low)
                rgb[np.isnan(image)] = 0
                regions = {overlay: [float(v) for v in wallLib.regionValues(polar, overlay)]
                           for overlay in wallLib.OVERLAYS}
                cached = (key, (rgbToQImage(rgb), regions))
                self.polarImages[tag] = cached
            _, _, _, ed, es = data.phases
            frames = f"ED frame {data.gated.labels[ed]} to ES frame {data.gated.labels[es]}"
            note = (f"mm the endocardium moves inward, {frames} (valve descent included)." if kind == "motion"
                    else f"% thickening from the wall counts, {frames} (first harmonic over the cycle).")
            panel.setPolar(cached[1][0], cached[1][1], f"{title} {name}", note=note)

    def refreshVolumeCurve(self):
        data = self.current()
        panel = existingPanel(VOLUME_STATE)
        if data is None or panel is None:
            return
        chart = panel.chart
        if data.phases is None:
            chart.pointLabels = {}
            chart.overlayText = ""
            chart.setCurves([], [], "LV cavity volume", "No LV cavity volumes (see Gated QC).")
            return
        edv, esv, ef, ed, es = data.phases
        chart.pointLabels = {ed: "ED", es: "ES"}
        chart.overlayText = f"EF {ef:.0f} %"
        chart.marker = self.frameIndex
        chart.setCurves(data.gated.labels, [("LV cavity", data.volumes, VOLUME_COLOR, "")],
                        f"LV volume (ml)   EDV {edv:.0f} · ESV {esv:.0f}", mode="absolute")

    # --- 3D wall ---------------------------------------------------------------------------
    def representation(self):
        return self.representationCombo.itemData(self.representationCombo.currentIndex) or "surface"

    @staticmethod
    def viewNodes():
        return [node for className in ("vtkMRMLModelNode", "vtkMRMLMarkupsFiducialNode")
                for node in slicer.util.getNodesByClass(className)
                if node.GetAttribute("MPI.GatedReviewView") == GATED_3D_VIEW]

    def surfaceModel(self, kind="gated"):
        """The beating wall of the 3D view (the ED reference is not a model: see GhostLayer)."""
        return next((n for n in self.viewNodes() if n.GetAttribute("MPI.Surface") == kind), None)

    def refresh3D(self, show=True):
        """The wall of every frame (made once), the floating wall labels and the camera of the static wall."""
        data = self.current()
        widget = threeDWidgetByTag(GATED_3D_VIEW)
        if data is None or widget is None:
            return
        try:
            import MPIMyocardium
            study = self.studyOf(self.slot())
            sample = self.staticSample(study)
            for index in range(data.count):
                data.surface(index, sample)
            viewNode = widget.mrmlViewNode()
            key = (id(data), study.GetID())
            model = self.surfaceModel()
            if model is None or model.GetAttribute("MPI.DataKey") != repr(key):
                for node in self.viewNodes():
                    slicer.mrmlScene.RemoveNode(node)
                model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"{study.GetName()} gated LV 3D")
                model.CreateDefaultDisplayNodes()
                model.SetAttribute("MPI.Surface", "gated")
                model.SetAttribute("MPI.DataKey", repr(key))
                display = model.GetDisplayNode()
                display.SetVisibility2D(False)
                display.SetViewNodeIDs([viewNode.GetID()])
                for node in [model] + MPIMyocardium.addWallLabelNodes(data.staticResult, study, [viewNode.GetID()]):
                    node.SetAttribute("MPI.GatedReviewView", GATED_3D_VIEW)
                    node.SetAttribute("MPI.GatedReviewOf", study.GetID())
                self.placeCamera(data, study)
            model.SetAndObservePolyData(data.surfaces[self.frameIndex])
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
            self.surfaceKey = key
            self.updateRepresentation()
            self.updateGhost()
            self.update3DColors()
            if show:
                self.showFrame(self.frameIndex)
        except Exception:
            logging.error("MPI: could not make the gated 3D wall:\n" + traceback.format_exc())

    def updateGhost(self):
        """The ED wall as the reference of the beating wall (shown when ticked and the volumes are known)."""
        data, widget = self.current(), threeDWidgetByTag(GATED_3D_VIEW)
        polyData = None
        if data is not None and widget is not None and self.surfaceModel() is not None and self.ghostCheck.checked:
            ed = data.phases[3] if data.phases else None
            polyData = data.surfaces.get(ed) if ed is not None else None
        representation = self.ghostRepresentationCombo.itemData(self.ghostRepresentationCombo.currentIndex)
        self.ghostLayer.update(widget, polyData, GHOST_COLOR, representation or GHOST_REPRESENTATION,
                               float(self.ghostOpacitySlider.value))

    def updateRepresentation(self):
        """Surface, wireframe or points: only the rendering of the same model changes."""
        model = self.surfaceModel()
        if model is None or model.GetDisplayNode() is None:
            return
        representation = self.representation()
        display = model.GetDisplayNode()
        display.SetRepresentation({"points": 0, "wireframe": 1}.get(representation, 2))
        display.SetAmbient(1.0 if representation == "surface" else 0.7)   # flat colors: the counts, not the shading
        display.SetDiffuse(0.0 if representation == "surface" else 0.3)
        display.SetOpacity(WALL_OPACITY.get(representation, 1.0))

    def update3DColors(self):
        model = self.surfaceModel()
        if model is None or model.GetDisplayNode() is None:
            return
        display = model.GetDisplayNode()
        colorNode = colorNodeForLabel(self.wallColormapCombo.currentText)
        if colorNode:
            display.SetAndObserveColorNodeID(colorNode.GetID())
        self.updateGhost()
        display.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
        display.SetScalarRange(*self.perfusionWindow())

    def placeCamera(self, data, study):
        def place():
            widget = threeDWidgetByTag(GATED_3D_VIEW)
            if widget is None or self.current() is not data:
                return
            try:
                import MPIMyocardium
                viewNode = widget.mrmlViewNode()
                MPIMyocardium.style3DView(viewNode)
                viewNode.SetRenderMode(slicer.vtkMRMLViewNode.Orthographic)
                scale = 0.75 * (float(np.max(data.staticResult["epi"])) + LABEL_DISTANCE_MM) + 10.0
                MPIMyocardium.placeCamera(viewNode, data.frame, data.staticResult, parallelScale=scale)
                self.updateGhost()   # the overlay layer follows the view's (possibly new) camera
            except Exception:
                logging.warning("MPI: could not place the gated 3D camera:\n" + traceback.format_exc())
        place()
        qt.QTimer.singleShot(400, place)    # again after the layout and the new models have settled
        qt.QTimer.singleShot(1200, place)

    @staticmethod
    def setCornerText(widget, text, upperLeft=""):
        try:
            view = widget.threeDView()
            corner = view.cornerAnnotation()
            corner.SetText(vtk.vtkCornerAnnotation.LowerLeft, text)
            corner.SetText(vtk.vtkCornerAnnotation.UpperLeft, upperLeft)
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
