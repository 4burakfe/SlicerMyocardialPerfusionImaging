"""Layouts and slice/3D view helpers of the reorientation and result screens."""

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
DUAL_REORIENT_LAYOUT_ID = 5712   # two studies stacked (stress over rest)
MYOCARDIUM_LAYOUT_ID = 5713      # wall contours on SA / HLA / VLA and the 3D wall
DUAL_MYOCARDIUM_LAYOUT_ID = 5714  # the same for stress (top row) and rest (bottom row)

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


def polarLayoutXml():
    """STRESS REST POLAR: stress, rest and difference polar maps in one row."""
    return ('<layout type="horizontal" split="true">'
            + "".join(f'<item splitSize="500"><{tag}></{tag}></item>' for tag in (POLAR_TAG, POLAR_REST_TAG, POLAR_DIFF_TAG))
            + '</layout>')


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

