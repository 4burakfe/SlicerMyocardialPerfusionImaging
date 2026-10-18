"""Layouts and slice/3D view helpers of the reorientation and result screens."""

import logging

import numpy as np
import qt
import vtk
import slicer

# ---------------------------------------------------------------------------
# View / layout identifiers
# ---------------------------------------------------------------------------
REORIENT_LAYOUT_ID = 5710        # one study
RESULT_LAYOUT_ID = 5711          # slices, polar map and 3D uptake of one study (STRESS PERF / REST PERF)
SLICES_LAYOUT_ID = 5720          # slices only (STRESS SLICES / REST SLICES / STRESS-REST SLICES)
DUAL_RESULT_LAYOUT_ID = 5721     # stress and rest slices, polar maps and 3D uptake (STRESS-REST PERF)
POLAR_LAYOUT_ID = 5722           # stress, rest and difference polar maps (STRESS REST POLAR)
PERF_DUAL_LAYOUT_ID = 5723       # one study on two monitors: slices / polar map and 3D uptake
DUAL_PERF_DUAL_LAYOUT_ID = 5724  # stress and rest on two monitors: slices / polar maps, difference, 3D uptake
DUAL_REORIENT_LAYOUT_ID = 5712   # two studies stacked (stress over rest)
MYOCARDIUM_LAYOUT_ID = 5713      # wall contours on SA / HLA / VLA and the 3D wall
DUAL_MYOCARDIUM_LAYOUT_ID = 5714  # the same for stress (top row) and rest (bottom row)
GATED_QC_LAYOUT_ID = 5730        # gate frames in SA / HLA / VLA rows, count curve and gating information
GATED_REVIEW_LAYOUT_ID = 5731    # cine slices, perfusion / thickening / motion polar maps, LV volume curve, cine 3D
# Gated Review layouts: (studies, kind) -> layout ID. Kinds: "full" (slices and plots), "slices", "plots",
# "dual" (slices on the main window, the plots in a second window on the second monitor), "dualFull" (the same
# with all slices; "full" and "dual" show three representative slices per axis, see MPIGated)
GATED_LAYOUT_IDS = {(("stress",), "full"): GATED_REVIEW_LAYOUT_ID, (("stress",), "slices"): 5732,
                    (("stress",), "plots"): 5733, (("stress",), "dual"): 5734,
                    (("rest",), "full"): 5735, (("rest",), "slices"): 5736,
                    (("rest",), "plots"): 5737, (("rest",), "dual"): 5738,
                    (("stress", "rest"), "full"): 5739, (("stress", "rest"), "slices"): 5740,
                    (("stress", "rest"), "plots"): 5741, (("stress", "rest"), "dual"): 5742,
                    (("stress",), "dualFull"): 5743, (("rest",), "dualFull"): 5744,
                    (("stress", "rest"), "dualFull"): 5745}
GATED_MONITOR2_TITLE = "Gated Review - Monitor 2"
PERFUSION_MONITOR2_TITLE = "Perfusion - Monitor 2"

EDIT_ORDER = ("TA", "VLA", "SA")
EDIT_VIEWS = {"TA": "MPIEditTA", "VLA": "MPIEditVLA", "SA": "MPIEditSA"}
ORIENTATIONS = ("SA", "HLA", "VLA")
PREVIEW_VIEWS = {"SA": "MPIPreviewSA", "HLA": "MPIPreviewHLA", "VLA": "MPIPreviewVLA"}
MIP_VIEW = "MPIMIP"                 # anterior MIP (top of the left column)
MIP_VIEW_LATERAL = "MPIMIPLateral"  # left lateral MIP (below it)
MIP_VIEWS = (MIP_VIEW, MIP_VIEW_LATERAL)
MIP_KINDS = ("ANT", "LAT")      # anterior, left lateral
LANE_SUFFIXES = ("", "Rest")    # view tag suffix of the top and bottom study of the processing screen
MYOCARDIUM_VIEWS = {"SA": "MPIMyoSA", "HLA": "MPIMyoHLA", "VLA": "MPIMyoVLA"}
MYOCARDIUM_3D_VIEW = "MPIMyo3D"
MOSAIC_TAG = "mpislicemosaic"   # XML tag of the embedded slice panel
POLAR_TAG = "mpipolarmap"       # XML tag of the embedded polar map (the stress or single study)
POLAR_REST_TAG = "mpipolarmaprest"   # polar map of the rest study (two-study layouts)
POLAR_DIFF_TAG = "mpipolardiff"      # rest - stress difference polar map
PERFUSION_3D_VIEW = "MPIPerf3D"   # surface uptake (SSP-style) 3D view under the polar map
PERFUSION_3D_VIEW_REST = "MPIPerf3DRest"   # the same for the rest study (STRESS-REST PERF)
GATED_MOSAIC_TAG = "mpigatedmosaic"   # Gated QC: one tile per gate frame in SA / HLA / VLA rows
GATED_CURVE_TAG = "mpigatedcurve"     # Gated QC: counts per frame
GATED_INFO_TAG = "mpigatedinfo"       # Gated QC: gating information (R-R window, beats, frame timing)
GATED_REVIEW_MOSAIC_TAG = "mpigatedreviewmosaic"   # Gated Review: cine SA / HLA / VLA rows
GATED_REVIEW_MOSAIC2_TAG = "mpigatedreviewmosaic2"   # Gated Review, three slices per axis: the HLA and VLA rows
GATED_STATIC_3D_VIEW = "MPIGatedStatic3D"          # Gated Review, three slices per axis: static wall, slice planes
GATED_POLAR_TAG = "mpigatedpolar"                  # Gated Review: perfusion polar map
GATED_THICKENING_TAG = "mpigatedthickening"        # Gated Review: wall thickening polar map
GATED_MOTION_TAG = "mpigatedmotion"                # Gated Review: wall motion polar map
GATED_VOLUME_TAG = "mpigatedvolume"                # Gated Review: LV cavity volume per frame
GATED_EF_TAG = "mpigatedregionalef"                # Gated Review: regional EF polar map
GATED_3D_VIEW = "MPIGated3D"                       # Gated Review: cine 3D wall with the perfusion colors
GATED_PLOT_KINDS = ("polar", "thickening", "motion", "volume", "3d", "ef")   # the plots of one study, in layout order

ORIENTATION_TITLES = {
    "SA": "Short axis (apex \u2192 base)",
    "HLA": "Horizontal long axis (inferior \u2192 anterior)",
    "VLA": "Vertical long axis (septum \u2192 lateral)",
}


def _sliceViewXml(tag, label, color, orientation="Axial"):
    return (
        f'<view class="vtkMRMLSliceNode" singletontag="{tag}">'
        f'<property name="orientation" action="default">{orientation}</property>'
        f'<property name="viewlabel" action="default">{label}</property>'
        f'<property name="viewcolor" action="default">{color}</property>'
        "</view>"
    )


def _horizontalRow(views):
    return ('<item><layout type="horizontal" split="true">'
            + "".join(f"<item>{v}</item>" for v in views)
            + "</layout></item>")


def laneViews(suffix=""):
    """View tags of one study's processing screen: {"edit": {TA, VLA, SA}, "preview": {SA, HLA, VLA},
    "mip": {ANT, LAT}}. The first study uses the plain tags, the second adds a suffix."""
    return {"edit": {k: v + suffix for k, v in EDIT_VIEWS.items()},
            "preview": {k: v + suffix for k, v in PREVIEW_VIEWS.items()},
            "mip": {"ANT": MIP_VIEW + suffix, "LAT": MIP_VIEW_LATERAL + suffix}}


def _laneXml(views, title=""):
    """Two MIPs stacked on the left, then axis adjustment (top) and resliced preview (bottom)."""
    prefix = f"{title} " if title else ""
    row1 = [_sliceViewXml(views["edit"]["TA"], f"{prefix}Adjust TA", "#D81B60"),
            _sliceViewXml(views["edit"]["VLA"], f"{prefix}Adjust VLA", "#43A047"),
            _sliceViewXml(views["edit"]["SA"], f"{prefix}Adjust SA", "#FDD835")]
    row2 = [_sliceViewXml(views["preview"][n], f"{prefix}{n}", "#1E88E5") for n in ORIENTATIONS]
    grid = ('<layout type="vertical" split="true">'
            + _horizontalRow(row1) + _horizontalRow(row2)
            + "</layout>")
    mips = ('<layout type="vertical" split="true">'
            + "".join(f'<item><view class="vtkMRMLViewNode" singletontag="{views["mip"][kind]}">'
                      f'<property name="viewlabel" action="default">{prefix}{label}</property></view></item>'
                      for kind, label in (("ANT", "ANT"), ("LAT", "LLAT")))
            + "</layout>")
    return ('<layout type="horizontal" split="true">'
            f'<item splitSize="300">{mips}</item>'
            f'<item splitSize="780">{grid}</item>'
            "</layout>")


def reorientLayoutXml():
    return _laneXml(laneViews(LANE_SUFFIXES[0]))


def dualReorientLayoutXml(titles=("S", "R")):
    """Two processing screens stacked top-bottom: the first study (stress) above the second (rest)."""
    lanes = [_laneXml(laneViews(suffix), title) for suffix, title in zip(LANE_SUFFIXES, titles)]
    return ('<layout type="vertical" split="true">'
            + "".join(f"<item>{lane}</item>" for lane in lanes)
            + "</layout>")


def myocardiumViews(suffix=""):
    """({orientation: slice view tag}, 3D view tag) of one study's row of the Myocardium layouts."""
    return {n: tag + suffix for n, tag in MYOCARDIUM_VIEWS.items()}, MYOCARDIUM_3D_VIEW + suffix


def dualMyocardiumLayoutXml(titles=("S", "R")):
    """Two rows of SA, HLA, VLA and the 3D wall: the stress study on top, the rest study below."""
    rows = []
    for suffix, title in zip(LANE_SUFFIXES, titles):
        slices, threeDTag = myocardiumViews(suffix)
        views = [_sliceViewXml(slices[n], f"{title} {n}", "#1E88E5") for n in ORIENTATIONS]
        views.append(f'<view class="vtkMRMLViewNode" singletontag="{threeDTag}">'
                     f'<property name="viewlabel" action="default">{title} 3D</property></view>')
        rows.append(_horizontalRow(views))
    return '<layout type="vertical" split="true">' + "".join(rows) + "</layout>"


def myocardiumLayoutXml():
    """SA and HLA on top, VLA and the 3D wall below."""
    views = {n: _sliceViewXml(MYOCARDIUM_VIEWS[n], n, "#1E88E5") for n in ORIENTATIONS}
    threeD = (f'<view class="vtkMRMLViewNode" singletontag="{MYOCARDIUM_3D_VIEW}">'
              '<property name="viewlabel" action="default">3D</property></view>')
    return ('<layout type="vertical" split="true">'
            + _horizontalRow([views["SA"], views["HLA"]])
            + _horizontalRow([views["VLA"], threeD])
            + "</layout>")


def _threeDXml(tag, label="3D"):
    return (f'<view class="vtkMRMLViewNode" singletontag="{tag}">'
            f'<property name="viewlabel" action="default">{label}</property></view>')


def resultLayoutXml():
    """STRESS PERF / REST PERF: the slice rows on the left; the polar map and the 3D uptake view stacked on the right."""
    return ('<layout type="horizontal" split="true">'
            f'<item splitSize="760"><{MOSAIC_TAG}></{MOSAIC_TAG}></item>'
            '<item splitSize="260"><layout type="vertical" split="true">'
            f'<item splitSize="500"><{POLAR_TAG}></{POLAR_TAG}></item>'
            f'<item splitSize="500">{_threeDXml(PERFUSION_3D_VIEW)}</item>'
            '</layout></item>'
            '</layout>')


def slicesLayoutXml():
    """Only the slice rows (one study, or stress and rest interleaved)."""
    return f'<layout type="horizontal"><item><{MOSAIC_TAG}></{MOSAIC_TAG}></item></layout>'


def dualResultLayoutXml():
    """STRESS-REST PERF: stress and rest slice rows on the left; on the right a 2 x 2 region with
    the stress polar map and 3D uptake (top) and the rest ones (bottom)."""
    def pair(polarTag, viewTag, label):
        return ('<item splitSize="500"><layout type="horizontal" split="true">'
                f'<item splitSize="500"><{polarTag}></{polarTag}></item>'
                f'<item splitSize="500">{_threeDXml(viewTag, label)}</item>'
                '</layout></item>')
    return ('<layout type="horizontal" split="true">'
            f'<item splitSize="640"><{MOSAIC_TAG}></{MOSAIC_TAG}></item>'
            '<item splitSize="380"><layout type="vertical" split="true">'
            + pair(POLAR_TAG, PERFUSION_3D_VIEW, "Stress")
            + pair(POLAR_REST_TAG, PERFUSION_3D_VIEW_REST, "Rest")
            + '</layout></item>'
            '</layout>')


def _twoMonitorXml(main, second, name, title, mainViewport=None):
    """Main window layout and a floating second window (a dock widget, as in Slicer's own dual monitor
    layouts), placed on the second monitor by placeMonitor2Window."""
    attributes = _mainViewportAttributes() if mainViewport is None else mainViewport
    return ('<viewports>'
            f'<layout type="horizontal"{attributes}><item>{main}</item></layout>'
            f'<layout name="{name}" type="horizontal" label="{title}" dockable="true" dockPosition="floating">'
            f'<item>{second}</item></layout>'
            '</viewports>')


def perfusionDualMonitorLayoutXml(twoStudies=False, mainViewport=None):
    """STRESS / REST PERF DUAL: the slice rows on the main window; the polar map next to the 3D uptake
    on the second monitor. STRESS-REST PERF DUAL: the stress, rest and difference polar maps (top) over
    the stress and rest 3D uptake views (bottom) on the second monitor."""
    if twoStudies:
        second = ('<layout type="vertical" split="true">'
                  '<item splitSize="500"><layout type="horizontal" split="true">'
                  + "".join(f'<item splitSize="500"><{tag}></{tag}></item>'
                            for tag in (POLAR_TAG, POLAR_REST_TAG, POLAR_DIFF_TAG))
                  + '</layout></item>'
                  '<item splitSize="500"><layout type="horizontal" split="true">'
                  f'<item splitSize="500">{_threeDXml(PERFUSION_3D_VIEW, "Stress")}</item>'
                  f'<item splitSize="500">{_threeDXml(PERFUSION_3D_VIEW_REST, "Rest")}</item>'
                  '</layout></item>'
                  '</layout>')
    else:
        second = ('<layout type="horizontal" split="true">'
                  f'<item splitSize="500"><{POLAR_TAG}></{POLAR_TAG}></item>'
                  f'<item splitSize="500">{_threeDXml(PERFUSION_3D_VIEW)}</item>'
                  '</layout>')
    return _twoMonitorXml(f"<{MOSAIC_TAG}></{MOSAIC_TAG}>", second, "MPIPerfusionMonitor2",
                          PERFUSION_MONITOR2_TITLE, mainViewport)


def polarLayoutXml():
    """STRESS REST POLAR: stress, rest and difference polar maps in one row."""
    return ('<layout type="horizontal" split="true">'
            + "".join(f'<item splitSize="500"><{tag}></{tag}></item>' for tag in (POLAR_TAG, POLAR_REST_TAG, POLAR_DIFF_TAG))
            + '</layout>')


def gatedQcLayoutXml():
    """Gated QC: the frame rows on the left; the count curve (top) and the gating information
    (bottom) on the right."""
    return ('<layout type="horizontal" split="true">'
            f'<item splitSize="760"><{GATED_MOSAIC_TAG}></{GATED_MOSAIC_TAG}></item>'
            '<item splitSize="300"><layout type="vertical" split="true">'
            f'<item splitSize="450"><{GATED_CURVE_TAG}></{GATED_CURVE_TAG}></item>'
            f'<item splitSize="550"><{GATED_INFO_TAG}></{GATED_INFO_TAG}></item>'
            '</layout></item>'
            '</layout>')


def gatedTags(slot):
    """Panel / view tags of the Gated Review plots of one study: {kind: tag}. The stress study uses the
    plain tags and the rest study its own ones, so both can be shown together."""
    suffix = "rest" if slot == "rest" else ""
    return {"polar": GATED_POLAR_TAG + suffix, "thickening": GATED_THICKENING_TAG + suffix,
            "motion": GATED_MOTION_TAG + suffix, "volume": GATED_VOLUME_TAG + suffix, "ef": GATED_EF_TAG + suffix,
            "3d": GATED_3D_VIEW + ("Rest" if suffix else ""),
            "static3d": GATED_STATIC_3D_VIEW + ("Rest" if suffix else "")}


def gatedPlotKinds(slots, kind):
    """The plots shown of every study in a Gated Review layout. STRESS-REST GATED (slices and plots on
    one screen) leaves out the perfusion polar map and the volume curve to make room; the plots-only
    and dual monitor layouts show all of them."""
    if kind == "slices":
        return ()
    if kind == "full" and len(slots) > 1:
        return ("thickening", "motion", "3d")
    return GATED_PLOT_KINDS


def _gatedPlotXml(slot, kind):
    tag = gatedTags(slot)[kind]
    if kind == "3d":
        return _threeDXml(tag, f"{'Rest' if slot == 'rest' else 'Stress'} gated 3D")
    return f"<{tag}></{tag}>"


def _items(views, size=500):
    return "".join(f'<item splitSize="{size}">{view}</item>' for view in views)


def _gatedPlotsXml(slots, kinds):
    """The plots of the studies: one study as a 2 x 3 grid, its perfusion, thickening and motion polar maps
    (top row) over the volume curve, the 3D wall and the regional EF polar map (bottom row); two studies as
    one row per study."""
    if len(slots) == 1:
        top = [_gatedPlotXml(slots[0], k) for k in kinds if k in ("polar", "thickening", "motion")]
        bottom = [_gatedPlotXml(slots[0], k) for k in kinds if k in ("volume", "3d", "ef")]
        rows = [(top, 500), (bottom, 500)]
    else:
        rows = [([_gatedPlotXml(slot, k) for k in kinds], 500) for slot in slots]
    return ('<layout type="vertical" split="true">'
            + "".join(f'<item splitSize="{size}"><layout type="horizontal" split="true">{_items(views)}</layout></item>'
                      for views, size in rows if views)
            + '</layout>')


def _mosaicXml(tag=GATED_REVIEW_MOSAIC_TAG):
    return f"<{tag}></{tag}>"


REPRESENTATIVE_LAYOUT_KINDS = ("full", "dual")   # three slices per axis (MPIGated.REPRESENTATIVE_KINDS)


def _representativeSlicesXml(slots):
    """The slice area of the three-slices-per-axis layouts: SA (top left) beside the static 3D wall with the
    slice planes (top right, one view per study), HLA and VLA side by side below."""
    views = "".join(f'<item splitSize="500">{_threeDXml(gatedTags(slot)["static3d"], f"{slot.capitalize()} slices 3D")}</item>'
                    for slot in slots)
    return ('<layout type="vertical" split="true">'
            '<item splitSize="500"><layout type="horizontal" split="true">'
            f'<item splitSize="500">{_mosaicXml()}</item>'
            f'<item splitSize="500"><layout type="vertical" split="true">{views}</layout></item>'
            '</layout></item>'
            f'<item splitSize="500">{_mosaicXml(GATED_REVIEW_MOSAIC2_TAG)}</item>'
            '</layout>')


def _mainViewportAttributes():
    """XML attributes of the main window viewport, copied from Slicer's own dual monitor layouts (as
    in SlicerPETReviewCompare), so the first <layout> of <viewports> is shown in the main window."""
    import xml.etree.ElementTree as ElementTree
    try:
        layoutNode = slicer.app.layoutManager().layoutLogic().GetLayoutNode()
        for attributeName in dir(slicer.vtkMRMLLayoutNode):
            if "DualMonitor" not in attributeName:
                continue
            layoutId = getattr(slicer.vtkMRMLLayoutNode, attributeName)
            if not isinstance(layoutId, int) or not layoutNode.IsLayoutDescription(layoutId):
                continue
            root = ElementTree.fromstring(layoutNode.GetLayoutDescription(layoutId))
            if root.tag != "viewports":
                continue
            for element in root.findall("layout"):
                if (element.get("dockable") or "").lower() != "true":
                    name = element.get("name")
                    return f' name="{name}"' if name is not None else ""
    except Exception as e:
        logging.debug(f"MPI: could not read Slicer's dual monitor layouts: {e}")
    return ""


def gatedLayoutXml(slots, kind, mainViewport=None):
    """Gated Review layout of the studies (("stress",), ("rest",) or ("stress", "rest")):
    full    the cine slice rows on the left, the plots on the right;
    slices  only the slice rows;
    plots   only the plots;
    dual    the slice rows on the main window and the plots in a floating window (a dock widget, as in
            Slicer's own dual monitor layouts), placed on the second monitor by placeMonitor2Window;
    dualFull  the same with all slices.
    full and dual show three slices per axis: SA beside a static 3D wall with the slice planes, HLA and VLA
    below (_representativeSlicesXml)."""
    slots = tuple(slots)
    plots = _gatedPlotsXml(slots, gatedPlotKinds(slots, kind))
    slices = _representativeSlicesXml(slots) if kind in REPRESENTATIVE_LAYOUT_KINDS else _mosaicXml()
    if kind == "slices":
        return f'<layout type="horizontal"><item>{_mosaicXml()}</item></layout>'
    if kind == "plots":
        return plots
    if kind in ("dual", "dualFull"):
        return _twoMonitorXml(slices, plots, "MPIGatedMonitor2", GATED_MONITOR2_TITLE, mainViewport)
    # full: the 2 x 2 slice area (three slices per axis) needs more width than the rows of all slices
    return ('<layout type="horizontal" split="true">'
            f'<item splitSize="{720 if len(slots) == 1 else 760}">{slices}</item>'
            f'<item splitSize="{540 if len(slots) == 1 else 500}">{plots}</item>'
            '</layout>')


def gatedReviewLayoutXml():
    """STRESS GATED (see gatedLayoutXml)."""
    return gatedLayoutXml(("stress",), "full")


def secondaryScreen():
    """A screen other than the one showing the main window, or None with a single screen."""
    mainWindow = slicer.util.mainWindow()
    screens = list(qt.QGuiApplication.screens())
    if mainWindow is None or len(screens) < 2:
        return None
    center = mainWindow.frameGeometry.center()
    return next((screen for screen in screens if not screen.geometry.contains(center)), None)


def placeMonitor2Window(viewTag=None, title=GATED_MONITOR2_TITLE):
    """Give the floating second window of a dual monitor layout normal window buttons and show it
    maximized on the second screen (as in SlicerPETReviewCompare), changing only what is not so
    already; with one screen it is only shown and raised. The window is found as the one holding the
    3D view viewTag, or by its title."""
    try:
        candidates = []
        widget = threeDWidgetByTag(viewTag) if viewTag else None
        if widget is not None:
            candidates.append(widget.window())
        candidates += [w for w in qt.QApplication.topLevelWidgets() if w.windowTitle == title]
        window = next((w for w in candidates if w is not None and not w.inherits("QMainWindow")
                       and not (w.inherits("QDockWidget") and not getattr(w, "floating", True))), None)
        if window is None:
            return
        flags = (qt.Qt.Window | qt.Qt.CustomizeWindowHint | qt.Qt.WindowTitleHint | qt.Qt.WindowSystemMenuHint
                 | qt.Qt.WindowMinMaxButtonsHint | qt.Qt.WindowCloseButtonHint)
        try:
            missing = (int(window.windowFlags()) & int(flags)) != int(flags)
        except (TypeError, ValueError):
            missing = True
        if missing:
            window.setWindowFlags(flags)
        screen = secondaryScreen()
        if screen is None:
            if not window.visible:
                window.show()
            window.raise_()
            return
        onSecond = screen.geometry.contains(window.frameGeometry.center())
        if not onSecond:
            window.setGeometry(screen.availableGeometry)
        if not (onSecond and window.isMaximized() and window.visible):
            window.showMaximized()
        window.raise_()
    except Exception:
        logging.exception("MPI: could not place the second window; drag it to the second monitor")


OWN_LAYOUT_IDS = range(5700, 5800)   # every layout of this extension (the IDs above)


def isOwnLayout(layoutId):
    return int(layoutId) in OWN_LAYOUT_IDS


def resetOwnLayout():
    """Back to Slicer's Four-Up layout when one of this extension's layouts is shown (after a scene
    close or load its panels and views show nothing: the studies are gone or not yet reviewed)."""
    try:
        layoutManager = slicer.app.layoutManager()
        if layoutManager is not None and isOwnLayout(layoutManager.layout):
            layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    except Exception:
        logging.warning("MPI: could not reset the layout", exc_info=True)


def notSaved(node):
    """Leave a helper node (and its display nodes) out of saved scenes."""
    if node is None:
        return
    node.SetSaveWithScene(False)
    try:
        for index in range(node.GetNumberOfDisplayNodes()):
            display = node.GetNthDisplayNode(index)
            if display is not None:
                display.SetSaveWithScene(False)
    except AttributeError:
        pass


def registerLayout(layoutId, xml):
    layoutNode = slicer.app.layoutManager().layoutLogic().GetLayoutNode()
    if layoutNode.IsLayoutDescription(layoutId):
        layoutNode.SetLayoutDescription(layoutId, xml)
    else:
        layoutNode.AddLayoutDescription(layoutId, xml)


def sliceWidgetByTag(tag):
    return slicer.app.layoutManager().sliceWidget(tag)


def sliceNodeByTag(tag):
    widget = sliceWidgetByTag(tag)
    return widget.mrmlSliceNode() if widget else None


def threeDWidgetByTag(tag):
    layoutManager = slicer.app.layoutManager()
    for index in range(layoutManager.threeDViewCount):
        widget = layoutManager.threeDWidget(index)
        if widget and widget.mrmlViewNode().GetLayoutName() == tag:
            return widget
    return None


class CameraLink:
    """3D views moved together (stress and rest): a rotation, zoom or pan of one view is copied to the
    others in each view's own anchor coordinates (center and axes of its study's cardiac frame, see
    MPIMyocardium.cameraAnchor), so every view shows its heart from the same side at the same zoom."""

    TOLERANCE = 1e-4   # pose change (mm, unit vectors) below which a camera event is not passed on

    def __init__(self):
        self.anchors = {}    # view tag -> (center, 3 x 3 axes as columns)
        self.observed = {}   # view tag -> (vtkCamera, observer id)
        self.poses = {}      # view tag -> last pose seen or set, in anchor coordinates
        self.busy = False

    def link(self, anchors):
        """anchors: {view tag: (center, axes)}; with fewer than two views nothing is linked."""
        self.anchors = {tag: (np.asarray(center, dtype=float), np.asarray(axes, dtype=float))
                        for tag, (center, axes) in anchors.items()}
        if len(self.anchors) < 2:
            self.unlink()
            return
        for tag in list(self.observed):
            if tag not in self.anchors or self._renderer(tag)[1] is None \
                    or self._renderer(tag)[1].GetActiveCamera() is not self.observed[tag][0]:
                self._release(tag)
        for tag in self.anchors:
            if tag in self.observed:
                continue
            renderer = self._renderer(tag)[1]
            if renderer is None:
                continue
            camera = renderer.GetActiveCamera()
            observer = camera.AddObserver(vtk.vtkCommand.ModifiedEvent,
                                          lambda caller, event, tag=tag: self.onModified(tag))
            self.observed[tag] = (camera, observer)
        self.poses = {}

    def unlink(self):
        for tag in list(self.observed):
            self._release(tag)
        self.anchors, self.poses = {}, {}

    def _release(self, tag):
        camera, observer = self.observed.pop(tag)
        try:
            camera.RemoveObserver(observer)
        except Exception:
            pass

    @staticmethod
    def _renderer(tag):
        widget = threeDWidgetByTag(tag)
        if widget is None:
            return None, None
        return widget, widget.threeDView().renderWindow().GetRenderers().GetFirstRenderer()

    def _pose(self, tag, camera):
        center, axes = self.anchors[tag]
        local = lambda point: axes.T @ (np.asarray(point, dtype=float) - center)   # noqa: E731
        return np.concatenate([local(camera.GetFocalPoint()), local(camera.GetPosition()),
                               axes.T @ np.asarray(camera.GetViewUp(), dtype=float), [camera.GetParallelScale()]])

    def onModified(self, tag):
        if self.busy or tag not in self.anchors or tag not in self.observed:
            return
        pose = self._pose(tag, self.observed[tag][0])
        last = self.poses.get(tag)
        if last is not None and np.allclose(pose, last, atol=self.TOLERANCE):
            return   # e.g. only the clipping range changed while rendering
        self.poses[tag] = pose
        self.busy = True
        try:
            for other, (center, axes) in self.anchors.items():
                if other == tag:
                    continue
                widget, renderer = self._renderer(other)
                if renderer is None:
                    continue
                camera = renderer.GetActiveCamera()
                camera.SetFocalPoint(*(center + axes @ pose[0:3]))
                camera.SetPosition(*(center + axes @ pose[3:6]))
                camera.SetViewUp(*(axes @ pose[6:9]))
                camera.SetParallelScale(float(pose[9]))
                renderer.ResetCameraClippingRange()
                self.poses[other] = pose
                widget.threeDView().scheduleRender()
        except Exception:
            logging.debug("MPI: could not link the 3D cameras", exc_info=True)
        finally:
            self.busy = False


def showVolumeInView(tag, volumeNode):
    widget = sliceWidgetByTag(tag)
    if not widget:
        return
    composite = widget.sliceLogic().GetSliceCompositeNode()
    composite.SetLinkedControl(False)
    composite.SetHotLinkedControl(False)
    composite.SetBackgroundVolumeID(volumeNode.GetID() if volumeNode else None)
    composite.SetForegroundVolumeID(None)
    composite.SetLabelVolumeID(None)


def setSliceGeometry(sliceNode, right, up, normal, origin, fieldOfView):
    """Point a slice view at an arbitrary plane.
    right/up are the screen directions in RAS, normal is the slice stepping direction."""
    if sliceNode is None:
        return
    matrix = vtk.vtkMatrix4x4()
    for r in range(3):
        matrix.SetElement(r, 0, float(right[r]))
        matrix.SetElement(r, 1, float(up[r]))
        matrix.SetElement(r, 2, float(normal[r]))
        matrix.SetElement(r, 3, float(origin[r]))
    dims = sliceNode.GetDimensions()
    aspect = dims[0] / float(dims[1]) if dims[0] > 0 and dims[1] > 0 else 1.0
    fovX, fovY = (fieldOfView * aspect, fieldOfView) if aspect >= 1.0 else (fieldOfView, fieldOfView / aspect)
    wasModifying = sliceNode.StartModify()
    sliceNode.GetSliceToRAS().DeepCopy(matrix)
    sliceNode.SetXYZOrigin(0, 0, 0)
    sliceNode.SetFieldOfView(fovX, fovY, sliceNode.GetFieldOfView()[2])
    sliceNode.UpdateMatrices()
    sliceNode.EndModify(wasModifying)

