"""Colour maps and the slice panel of the result screen.

Tiles are drawn directly: each slice is first resized to the tile's size on screen in the data
domain (the interpolation chosen in Processing, see resampling.zoomInPlane), then colour mapped with numpy, then handed to
Qt as an uncompressed in-memory PPM (the same route as the DICOM SPECT Recon previews). Qt never
scales the colour image, so there is no PNG encoding, no temporary file and no bilinear blending
of colours. The panel is painted at the screen's device pixel ratio, so a scaled Windows display
(125 %, 150 %) does not stretch it with nearest-neighbour sampling.
"""

import logging
import traceback

import math

import numpy as np
import qt
import slicer

from MyocardialPerfusionSPECTLib import colortables
from MyocardialPerfusionSPECTLib.views import ORIENTATIONS, ORIENTATION_TITLES

DEFAULT_GRID_ROWS = 2
DEFAULT_GRID_COLUMNS = 12
MOSAIC_PANEL_VERSION = 13  # bump when SliceMosaicPanel changes, so a reload replaces the old panel

# (combo label, color node name or ID); the first one is the default. "MPI ..." tables are this
# extension's own (colortables.py), made in the scene when first used.
COLORMAPS = [
    ("PET-DICOM", "MPI PET-DICOM"),
    ("Hel", "MPI Hel"),
    ("Inferno", "Inferno"),       # Slicer's own (matplotlib) Inferno
    ("PET-HotMetalBlue", "PET-HotMetalBlue"),
    ("PET-HotIron", "MPI PET-HotIron"),
    ("Jotun", "MPI Jotun"),
    ("Vanaheim", "MPI Vanaheim"),
    ("Valhalla", "MPI Valhalla"),
    ("Trugold", "MPI Trugold"),
    ("PET-Rainbow2", "PET-Rainbow2"),
    ("Grey", "vtkMRMLColorTableNodeGrey"),
    ("Inverted grey", "vtkMRMLColorTableNodeInvertedGrey"),
    ("PET-DICOM (Slicer)", "PET-DICOM"),
]


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------
def _sampleColorNode(node, positions):
    """RGB floats [n, 3] of a Slicer color node at positions 0..1 of its range (grey without a node)."""
    positions = np.asarray(positions, dtype=float)
    table = np.repeat(positions[:, None], 3, axis=1)
    scalarsToColors = node.GetScalarsToColors() if node else None
    if scalarsToColors:
        low, high = scalarsToColors.GetRange()
        rgb = [0.0, 0.0, 0.0]
        for i, t in enumerate(positions):
            scalarsToColors.GetColor(low + (high - low) * float(t), rgb)
            table[i] = rgb
    return table


def _builtInNode(key):
    """A Slicer color node by ID or name; color files (e.g. Inferno) also by their file node ID."""
    return (slicer.mrmlScene.GetNodeByID(key) or slicer.mrmlScene.GetFirstNodeByName(key)
            or slicer.mrmlScene.GetNodeByID(f"vtkMRMLColorTableNodeFile{key}.txt")
            or slicer.mrmlScene.GetNodeByID(f"vtkMRMLProceduralColorNodeFile{key}.txt"))


def _customTable(nodeName):
    """256 x 3 RGB floats of the tables of this extension ("MPI <label>"), or None."""
    return colortables.table(nodeName[4:]) if nodeName.startswith("MPI ") else None


def customColorNode(nodeName):
    """One of this extension's color tables (256 colors) in the scene, made when missing (e.g. after
    a scene close)."""
    node = slicer.mrmlScene.GetFirstNodeByName(nodeName)
    if node is not None and node.IsA("vtkMRMLColorTableNode"):
        return node
    table = _customTable(nodeName)
    if table is None:
        return None
    node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLColorTableNode")
    node.UnRegister(None)
    node.SetName(nodeName)
    node.SetTypeToUser()
    node.SetNumberOfColors(256)
    try:
        node.SetNamesInitialised(True)
    except AttributeError:
        pass
    for i, (r, g, b) in enumerate(table):
        node.SetColor(i, f"{i}", float(r), float(g), float(b), 1.0)
    node.SetHideFromEditors(False)
    slicer.mrmlScene.AddNode(node)
    return node


def colorNodeForLabel(label):
    for name, key in COLORMAPS:
        if name == label:
            if key.startswith("MPI "):
                node = customColorNode(key)
                if node:
                    return node
            node = _builtInNode(key)
            if node:
                return node
    return slicer.mrmlScene.GetNodeByID("vtkMRMLColorTableNodeGrey")


def colorTableArray(label):
    """256 x 3 uint8 RGB table of a color map (this extension's tables from colortables.py, the others
    sampled from the Slicer color node)."""
    key = dict(COLORMAPS).get(label, "")
    table = _customTable(key) if key.startswith("MPI ") else None
    if table is None:
        table = _sampleColorNode(colorNodeForLabel(label), np.linspace(0.0, 1.0, 256))
    return np.clip(np.round(table * 255.0), 0, 255).astype(np.uint8)


OUTWARD_MOTION_MM = 4.0          # outward (dyskinetic) wall motion shown at full color from this many mm
OUTWARD_COLORS = {"Jotun": (255, 0, 0), "Vanaheim": (255, 0, 0), "PET-DICOM": (255, 0, 0)}  # maps with turquoise
OUTWARD_COLOR = (64, 224, 208)  # turquoise


def signedMotionTable(table, label, low, top):
    """(table, low, top) of the wall motion maps: the color map over the window from max(low, 0) to top for
    inward motion, and below 0 a ramp from black at 0 to full turquoise (red for color maps that hold
    turquoise) at -OUTWARD_MOTION_MM for outward (dyskinetic) motion; the returned window starts there."""
    table = np.asarray(table, dtype=float)
    color = np.asarray(OUTWARD_COLORS.get(label, OUTWARD_COLOR), dtype=float)
    start, top = max(float(low), 0.0), float(top)
    top = max(top, start + 1e-6)
    values = np.linspace(-OUTWARD_MOTION_MM, top, len(table))
    index = np.clip(np.round((values - start) / (top - start) * (len(table) - 1)), 0, len(table) - 1).astype(int)
    combined = table[index]
    outward = values < 0.0
    combined[outward] = np.clip(-values[outward] / OUTWARD_MOTION_MM, 0.0, 1.0)[:, None] * color
    return np.round(combined).astype(np.uint8), -OUTWARD_MOTION_MM, top


def legendTicks(low, high):
    """Values marked under a color bar: five evenly spaced ones, or for a window across 0 (the signed wall
    motion) the ends, 0 itself and the middle of the positive side, so the 0 mark sits on the black."""
    low, high = float(low), float(high)
    if low < 0.0 < high:
        return [low, 0.0, high / 2.0, high]
    return [low + k * (high - low) / 4.0 for k in range(5)]


def tickText(value):
    text = f"{value:.0f}" if abs(value - round(value)) < 0.05 else f"{value:.1f}"
    return "0" if text in ("-0", "-0.0") else text


def colorize(values, top, table, low=0.0):
    """Scalar array -> RGB uint8 [..., 3]: low and below get the first color, top and above the last
    (the min / max window of the Volumes module)."""
    low = float(low)
    scale = 255.0 / max(float(top) - low, 1e-9)
    indices = np.clip((np.asarray(values, dtype=np.float32) - low) * scale, 0, 255)
    return table[indices.astype(np.uint8)]


def setVolumeDisplay(volumeNode, colorLabel, windowMin=None, windowMax=None):
    if not volumeNode:
        return
    volumeNode.CreateDefaultDisplayNodes()
    displayNode = volumeNode.GetDisplayNode()
    if not displayNode:
        return
    displayNode.SetInterpolate(True)
    colorNode = colorNodeForLabel(colorLabel)
    if colorNode:
        displayNode.SetAndObserveColorNodeID(colorNode.GetID())
    if windowMin is not None and windowMax is not None and windowMax > windowMin:
        displayNode.AutoWindowLevelOff()
        displayNode.SetWindowLevelMinMax(windowMin, windowMax)


# ---------------------------------------------------------------------------
# RGB array -> QImage, in memory
# ---------------------------------------------------------------------------
def ppmBytes(rgb):
    """Binary PPM (P6) encoding of an RGB uint8 array whose row 0 is the top row."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[0], rgb.shape[1]
    return b"P6\n%d %d\n255\n" % (w, h) + rgb.tobytes()


def imageBytes(image):
    """Memory of a QImage (or of a numpy array / tuple of them) in bytes, for the cache budgets."""
    if isinstance(image, (tuple, list)):
        return sum(imageBytes(item) for item in image)
    if isinstance(image, np.ndarray):
        return int(image.nbytes)
    for method in ("sizeInBytes", "byteCount"):
        try:
            return int(getattr(image, method)())
        except Exception:
            pass
    try:
        return int(image.width()) * int(image.height()) * 4
    except Exception:
        return 0


class BoundedCache:
    """Least recently used cache with a memory budget (bytes, measured with imageBytes): the oldest entries
    go when a new one would exceed it, so the cache cannot grow with every window, size or slice change."""

    def __init__(self, budgetBytes):
        import collections
        self.budget = int(budgetBytes)
        self.entries = collections.OrderedDict()   # key -> (value, bytes)
        self.bytes = 0

    def get(self, key, default=None):
        entry = self.entries.get(key)
        if entry is None:
            return default
        self.entries.move_to_end(key)
        return entry[0]

    def __setitem__(self, key, value):
        self.pop(key)
        size = imageBytes(value)
        self.entries[key] = (value, size)
        self.bytes += size
        while self.bytes > self.budget and len(self.entries) > 1:
            _, (_, old) = self.entries.popitem(last=False)
            self.bytes -= old

    def __getitem__(self, key):
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __contains__(self, key):
        return key in self.entries

    def __len__(self):
        return len(self.entries)

    def pop(self, key, default=None):
        entry = self.entries.pop(key, None)
        if entry is None:
            return default
        self.bytes -= entry[1]
        return entry[0]

    def clear(self):
        self.entries.clear()
        self.bytes = 0


_MISSING = object()


def packMask(mask):
    """Boolean mask -> (shape, bits): 8 times smaller in a cache."""
    mask = np.asarray(mask, dtype=bool)
    return mask.shape, np.packbits(mask, axis=None)


def unpackMask(packed):
    shape, bits = packed
    return np.unpackbits(bits, count=int(np.prod(shape))).astype(bool).reshape(shape)


def rgbToQImage(rgb):
    """uint8 [h, w, 3] array (row 0 at the top) -> QImage, decoded by Qt's built-in PPM reader."""
    image = qt.QImage.fromData(ppmBytes(rgb), "PPM")
    if image is None or image.isNull():
        raise RuntimeError("Qt could not decode the in-memory slice image (PPM)")
    return image


def devicePixelRatio(widget):
    try:
        ratio = widget.devicePixelRatioF()
    except Exception:
        try:
            ratio = widget.devicePixelRatio()
        except Exception:
            ratio = 1.0
    ratio = float(ratio() if callable(ratio) else ratio)
    return ratio if ratio > 0 else 1.0


# ---------------------------------------------------------------------------
# Slice panel embedded in the layout (replaces the removed lightbox mode)
# ---------------------------------------------------------------------------
class _EventForwarder(qt.QObject):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def eventFilter(self, obj, event):
        try:
            return bool(self.callback(obj, event))
        except Exception:
            logging.error(traceback.format_exc())
            return False


def paintColorBar(painter, table, low, high, ratio, font, top, bottom, width):
    """Vertical color bar (window maximum at the top) with ticks at 0, 25, 50, 75 and 100 %."""
    margin = int(round(font.pixelSize() * 0.8))
    barWidth = int(round(16 * ratio))
    barTop, barBottom = top + margin, bottom - margin
    barHeight = max(barBottom - barTop, 2)
    index = np.clip(np.round((1.0 - (np.arange(barHeight) + 0.5) / barHeight) * 255), 0, 255).astype(int)
    column = np.ascontiguousarray(np.repeat(table[index][:, None, :], barWidth, axis=1))
    painter.drawImage(int(round(4 * ratio)), barTop, rgbToQImage(column))
    pen = qt.QPen(qt.QColor(200, 200, 200))
    pen.setWidthF(max(1.0, ratio))
    painter.setPen(pen)
    x0 = int(round(4 * ratio)) + barWidth
    for k in range(5):
        y = barTop + int(round(k * (barHeight - 1) / 4.0))
        value = high - k * (high - low) / 4.0
        painter.drawLine(x0, y, x0 + int(round(4 * ratio)), y)
        painter.drawText(qt.QRectF(x0 + 6 * ratio, y - font.pixelSize(), width - x0 - 6 * ratio,
                                   2 * font.pixelSize()), qt.Qt.AlignLeft | qt.Qt.AlignVCenter, f"{value:.0f}%")


class SliceMosaicPanel:
    """Rows of slice tiles with slice numbers: SA, HLA and VLA of one study, or of two studies
    interleaved (setRowKeys). Mouse wheel over a row scrolls that row (onScroll(key, direction)). Tiles are supplied at tileSize() device pixels;
    after a resize the panel shows the old tiles rescaled and calls onResized so the owner
    can render them again at the new size.
    With pairRows (two studies, the row keys in stress / rest pairs) and a grid of more than one row,
    each grid row of a study is followed by the same grid row of the other study (every row of a pair is
    split into one-row strips, shown alternately)."""

    GAP = 3
    LEGEND_WIDTH = 62
    TITLE_PIXELS = 11      # row title font size (logical pixels)

    def __init__(self):
        self.onScroll = None
        self.onResized = None
        self._gridRows = DEFAULT_GRID_ROWS
        self.gridColumns = DEFAULT_GRID_COLUMNS
        self._pairRows = False
        self.widget = qt.QWidget()
        self.widget.setObjectName("MPISliceMosaic")
        self.widget.setStyleSheet("QWidget#MPISliceMosaic { background-color: black; }"
                                  "QLabel { color: #e0e0e0; background-color: black; }")
        outer = qt.QHBoxLayout(self.widget)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)
        self.filter = _EventForwarder(self._onEvent)
        # color legend down the whole left side
        self.legendImage = qt.QLabel()
        self.legendImage.setObjectName("MPILegend")
        self.legendImage.setSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Ignored)
        self.legendImage.setFixedWidth(self.LEGEND_WIDTH)
        self.legendImage.installEventFilter(self.filter)
        outer.addWidget(self.legendImage)
        self.legend = None   # (256 x 3 uint8 table, low %, high %)
        self.rowsLayout = qt.QVBoxLayout()
        self.rowsLayout.setSpacing(1)
        self.rowsLayout.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(self.rowsLayout, 1)
        self.rows = {}         # strip key -> {"title", "image", "tiles", "row" (logical key), "part"}
        self.rowKeys = []      # logical rows (one per study and orientation)
        self.titles = {}       # logical row -> title
        self.strips = {}       # logical row -> [strip keys] (one strip, or one per grid row when split)
        self.stripKeys = None  # strip layout built (to rebuild only when it changes)
        self.setRowKeys([(name, ORIENTATION_TITLES[name]) for name in ORIENTATIONS])

    # --- rows and strips ----------------------------------------------------------------
    @property
    def gridRows(self):
        return self._gridRows

    @gridRows.setter
    def gridRows(self, value):
        value = int(value)
        if value != self._gridRows:
            self._gridRows = value
            self._buildStrips()

    @property
    def pairRows(self):
        return self._pairRows

    @pairRows.setter
    def pairRows(self, value):
        if bool(value) != self._pairRows:
            self._pairRows = bool(value)
            self._buildStrips()

    def split(self):
        return self._pairRows and self._gridRows > 1 and len(self.rowKeys) % 2 == 0 and len(self.rowKeys) > 0

    def stripRows(self):
        """Grid rows of one strip."""
        return 1 if self.split() else self._gridRows

    def setRowKeys(self, keys):
        """keys: [(row key, title)] from top to bottom (in stress / rest pairs for pairRows)."""
        self.titles = {k: t for k, t in keys}
        self.rowKeys = [k for k, _ in keys]
        self._buildStrips()

    def _layoutOrder(self):
        """[(strip key, logical row, part)] from top to bottom."""
        if not self.split():
            return [(key, key, 0) for key in self.rowKeys]
        order = []
        for first, second in zip(self.rowKeys[0::2], self.rowKeys[1::2]):
            for part in range(self._gridRows):
                order += [(f"{first}#{part}", first, part), (f"{second}#{part}", second, part)]
        return order

    def _buildStrips(self):
        order = self._layoutOrder()
        if [k for k, _, _ in order] == self.stripKeys:
            return
        old = {row["row"]: row["tiles"] for row in self.rows.values() if row["part"] == 0}
        for row in self.rows.values():
            for widget in (row["title"], row["image"]):
                self.rowsLayout.removeWidget(widget)
                widget.removeEventFilter(self.filter)
                widget.setParent(None)
                widget.deleteLater()
        self.rows, self.strips = {}, {key: [] for key in self.rowKeys}
        self.stripKeys = [k for k, _, _ in order]
        for strip, key, part in order:
            title = qt.QLabel(self._stripTitle(key, part))
            title.setStyleSheet(f"font-weight: bold; font-size: {self.TITLE_PIXELS}px; padding: 0px; margin: 0px;")
            title.setContentsMargins(0, 0, 0, 0)
            title.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
            image = qt.QLabel()
            image.setObjectName(f"MPIMosaic_{strip}")
            image.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Ignored)  # pixmap must not drive the size
            image.setMinimumSize(50, 20)
            image.setAlignment(qt.Qt.AlignCenter)
            image.installEventFilter(self.filter)
            self.rowsLayout.addWidget(title)
            self.rowsLayout.addWidget(image, 1)
            self.rows[strip] = {"title": title, "image": image, "tiles": [], "row": key, "part": part}
            self.strips[key].append(strip)
        # lay the new strips out now, so that tileSize is right before the next event loop
        for row in self.rows.values():
            row["title"].show()
            row["image"].show()
        self.widget.layout().activate()
        for key, tiles in old.items():
            if key in self.strips:
                self._distribute(key, tiles)

    def _stripTitle(self, key, part):
        title = self.titles.get(key, "")
        if part > 0:   # later strips of a split row: only the study ("STRESS  ·  ...")
            title = title.split("  ·  ")[0]
        return title

    def _distribute(self, key, tiles):
        strips = self.strips.get(key, [])
        columns = self.gridColumns
        for strip in strips:
            row = self.rows[strip]
            row["tiles"] = tiles[row["part"] * columns:(row["part"] + 1) * columns] if len(strips) > 1 else tiles
            self.paintRow(strip)

    def _geometry(self, strip):
        """(tile size in logical pixels, device pixel ratio) for a strip, or (0, ratio) if too small."""
        label = self.rows[strip]["image"]
        ratio = devicePixelRatio(label)
        rect = label.contentsRect()
        width, height = rect.width(), rect.height()
        if self.split():   # one tile size for all strips (their heights can differ by a pixel)
            height = min(row["image"].contentsRect().height() for row in self.rows.values())
        if width < 10 or height < 10:
            return 0, ratio
        gap, rows, columns = self.GAP, self.stripRows(), self.gridColumns
        tile = min((width - gap * (columns - 1)) // columns, (height - gap * (rows - 1)) // rows)
        return max(tile, 4), ratio

    def _firstStrip(self, name):
        if name is None:
            name = self.rowKeys[0] if self.rowKeys else None
        strips = self.strips.get(name) or ([name] if name in self.rows else [])
        return strips[0] if strips else None

    def rowArea(self, name=None):
        """(width, height) in logical pixels for all grid rows of a row (the first row by default), as if
        it were one block: a split row counts its strips and their titles but one."""
        strip = self._firstStrip(name)
        if strip is None:
            return 0, 0
        strips = self.strips.get(self.rows[strip]["row"], [strip])
        width = self.rows[strip]["image"].contentsRect().width()
        height = sum(self.rows[s]["image"].contentsRect().height() for s in strips)
        if len(strips) > 1:
            height += (len(strips) - 1) * self.stripOverhead()
        return width, height

    def stripOverhead(self):
        """Height (logical pixels) one more grid row costs beyond its tiles when the rows are split into
        strips (its title and the layout spacing), else 0 (the tile gap is counted by bestGrid)."""
        if not (self._pairRows and len(self.rowKeys) % 2 == 0 and self.rowKeys):
            return 0
        strip = self._firstStrip(None)
        title = self.rows[strip]["title"] if strip else None
        titleHeight = title.contentsRect().height() if title is not None else 0
        if titleHeight <= 0:
            titleHeight = self.TITLE_PIXELS + 3
        spacing = self.rowsLayout.spacing   # a property in PythonQt, a method in PySide
        return titleHeight + 2 * int(spacing() if callable(spacing) else spacing)

    def tileSize(self, name):
        """Edge of one tile in device pixels: render tile images at this size."""
        strip = self._firstStrip(name)
        if strip is None:
            return 0
        tile, ratio = self._geometry(strip)
        return int(round(tile * ratio)) if tile else 0

    def pixelRatio(self):
        strip = self._firstStrip(None)
        return devicePixelRatio(self.rows[strip]["image"]) if strip else devicePixelRatio(self.widget)

    def setRow(self, name, tiles, titleText):
        """tiles: list of (QImage or None, slice number or None), row-major."""
        if name not in self.strips:
            return
        self.titles[name] = titleText
        for strip in self.strips[name]:
            self.rows[strip]["title"].setText(self._stripTitle(name, self.rows[strip]["part"]))
        self._distribute(name, tiles)

    def setGrid(self, rows, columns):
        self.gridColumns = int(columns)
        self.gridRows = rows
        for strip in self.rows:
            self.paintRow(strip)

    def clear(self):
        self.pairRows = False
        self.setRowKeys([(name, ORIENTATION_TITLES[name]) for name in ORIENTATIONS])
        for name in ORIENTATIONS:
            self.setRow(name, [], ORIENTATION_TITLES[name])
        self.legend = None
        self.paintLegend()

    def paintRow(self, strip):
        row = self.rows.get(strip)
        if row is None:
            return
        label = row["image"]
        tile, ratio = self._geometry(strip)
        if not tile:
            return
        gap, rows, columns = self.GAP, self.stripRows(), self.gridColumns
        # Paint in device pixels, so the label shows the pixmap 1:1 on scaled displays
        tileDevice = int(round(tile * ratio))
        gapDevice = int(round(gap * ratio))
        pixmap = qt.QPixmap(tileDevice * columns + gapDevice * (columns - 1),
                            tileDevice * rows + gapDevice * (rows - 1))
        pixmap.fill(qt.QColor(0, 0, 0))
        painter = qt.QPainter(pixmap)
        # Only used for stale tiles after a resize; fresh tiles already have the exact size
        painter.setRenderHint(qt.QPainter.SmoothPixmapTransform, True)
        painter.setRenderHint(qt.QPainter.TextAntialiasing, True)
        font = painter.font()
        font.setPixelSize(max(9, int(round(max(9, tile // 9) * ratio))))
        painter.setFont(font)
        for index, (qimage, number) in enumerate(row["tiles"][:rows * columns]):
            r, c = divmod(index, columns)
            x, y = c * (tileDevice + gapDevice), r * (tileDevice + gapDevice)
            if qimage is not None and not qimage.isNull():
                if qimage.width() == tileDevice and qimage.height() == tileDevice:
                    painter.drawImage(x, y, qimage)
                else:
                    painter.drawImage(qt.QRect(x, y, tileDevice, tileDevice), qimage)
            if number is not None:
                shift = max(1, int(round(ratio)))
                painter.setPen(qt.QColor(0, 0, 0))
                painter.drawText(x + 3 * shift + shift, y + 2 * shift + shift + font.pixelSize(), str(number))
                painter.setPen(qt.QColor(255, 255, 255))
                painter.drawText(x + 3 * shift, y + 2 * shift + font.pixelSize(), str(number))
        painter.end()
        try:
            pixmap.setDevicePixelRatio(ratio)
        except Exception:
            pass
        label.setPixmap(pixmap)

    # --- color legend ---------------------------------------------------------------
    def setLegend(self, table, low, high):
        """table: 256 x 3 uint8 colors from the window minimum (bottom) to its maximum (top);
        low / high: the window in % (labels)."""
        self.setLegends([(table, low, high, "")])

    def setLegends(self, entries):
        """Several legends stacked top to bottom (stress and rest with their own windows):
        [(table, low %, high %, caption)]."""
        self.legend = [(np.asarray(t, dtype=np.uint8), float(lo), float(hi), str(c or "")) for t, lo, hi, c in entries]
        self.paintLegend()

    def paintLegend(self):
        label = self.legendImage
        ratio = devicePixelRatio(label)
        rect = label.contentsRect()
        if rect.width() < 20 or rect.height() < 60:
            return
        width, height = int(round(rect.width() * ratio)), int(round(rect.height() * ratio))
        pixmap = qt.QPixmap(width, height)
        pixmap.fill(qt.QColor(0, 0, 0))
        if self.legend:
            painter = qt.QPainter(pixmap)
            painter.setRenderHint(qt.QPainter.TextAntialiasing, True)
            font = painter.font()
            font.setPixelSize(max(9, int(round(11 * ratio))))
            painter.setFont(font)
            count = len(self.legend)
            for index, (table, low, high, caption) in enumerate(self.legend):
                top, bottom = height * index // count, height * (index + 1) // count
                if caption:
                    font.setBold(True)
                    painter.setFont(font)
                    painter.setPen(qt.QColor(230, 230, 230))
                    painter.drawText(qt.QRectF(0, top, width, 1.6 * font.pixelSize()), qt.Qt.AlignCenter, caption)
                    font.setBold(False)
                    painter.setFont(font)
                    top += int(round(1.6 * font.pixelSize()))
                paintColorBar(painter, table, low, high, ratio, font, top, bottom, width)
            painter.end()
        try:
            pixmap.setDevicePixelRatio(ratio)
        except Exception:
            pass
        label.setPixmap(pixmap)

    def _onEvent(self, obj, event):
        objectName = obj.objectName if isinstance(obj.objectName, str) else obj.objectName()
        if objectName == "MPILegend":
            if event.type() == qt.QEvent.Resize:
                self.paintLegend()
            return False
        if not objectName.startswith("MPIMosaic_"):
            return False
        strip = objectName.split("_", 1)[1]
        row = self.rows.get(strip)
        name = row["row"] if row else strip
        if event.type() == qt.QEvent.Resize:
            self.paintRow(strip)
            if self.onResized:
                self.onResized()
            return False
        if event.type() == qt.QEvent.Wheel:
            try:
                delta = event.angleDelta().y()
            except AttributeError:
                delta = event.delta()
            if delta != 0 and self.onScroll:
                self.onScroll(name, -1 if delta > 0 else 1)
            return True
        return False


class PolarPanel:
    """Polar map (bullseye with the 17 segments), in its own place of the layout above the 3D view.
    Region overlay: the coronary territories ("vessels", the default), the 17 segments, or None
    (overlay); their values can be switched off (showScores). A map painted per wall sets
    regionOverride = "walls": whenever an overlay is shown, the five walls (apex and the four quarters of
    the X) are drawn and scored instead."""

    POLAR_MARGIN = 0.09    # share of the side left around the disk for the ANT / SEPT / LAT / INF labels

    def __init__(self):
        self.onResized = None
        self.overlay = "vessels"
        self.regionOverride = None
        self.showScores = True
        self.widget = qt.QWidget()
        self.widget.setObjectName("MPIPolarPanel")
        self.widget.setStyleSheet("QWidget#MPIPolarPanel { background-color: black; }"
                                  "QLabel { color: #e0e0e0; background-color: black; }")
        layout = qt.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        self.filter = _EventForwarder(self._onEvent)
        self.polarTitle = qt.QLabel("Polar map")
        self.polarTitle.setStyleSheet("font-weight: bold;")
        self.polarTitle.setWordWrap(True)
        self.polarImage = qt.QLabel()
        self.polarImage.setObjectName("MPIPolar")
        self.polarImage.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Ignored)
        self.polarImage.setMinimumSize(80, 80)
        self.polarImage.setAlignment(qt.Qt.AlignCenter)
        self.polarImage.installEventFilter(self.filter)
        self.polarNote = qt.QLabel("")
        self.polarNote.setWordWrap(True)
        self.polarNote.setStyleSheet("color: #9aa3a9;")
        self.legendImage = qt.QLabel()
        self.legendImage.setObjectName("MPIPolarLegend")
        self.legendImage.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        self.legendImage.setFixedHeight(34)
        self.legendImage.installEventFilter(self.filter)
        self.legendImage.hide()
        self.legend = None   # (256 x 3 uint8 table, low %, high %) of the color bar under the map
        layout.addWidget(self.polarTitle)
        layout.addWidget(self.polarImage, 1)
        layout.addWidget(self.legendImage)
        layout.addWidget(self.polarNote)
        self.polar = {"image": None, "segments": None, "message": ""}

    def setLegend(self, table, low=0.0, high=100.0, unit="%"):
        """Horizontal color bar under the map (when its window differs from the slice legend), its labels
        in unit; table None hides it."""
        self.legend = None if table is None else (np.asarray(table, dtype=np.uint8), float(low), float(high), unit)
        self.legendImage.setVisible(self.legend is not None)
        self.paintLegend()

    def paintLegend(self):
        label = self.legendImage
        if self.legend is None:
            return
        ratio = devicePixelRatio(label)
        rect = label.contentsRect()
        if rect.width() < 60 or rect.height() < 10:
            return
        width, height = int(round(rect.width() * ratio)), int(round(rect.height() * ratio))
        pixmap = qt.QPixmap(width, height)
        pixmap.fill(qt.QColor(0, 0, 0))
        table, low, high, unit = self.legend
        painter = qt.QPainter(pixmap)
        painter.setRenderHint(qt.QPainter.TextAntialiasing, True)
        font = painter.font()
        font.setPixelSize(max(9, int(round(10 * ratio))))
        painter.setFont(font)
        margin = int(round(width * 0.08))
        barHeight = int(round(10 * ratio))
        barWidth = max(width - 2 * margin, 2)
        index = np.clip(np.round((np.arange(barWidth) + 0.5) / barWidth * 255), 0, 255).astype(int)
        strip = np.ascontiguousarray(np.repeat(table[index][None, :, :], barHeight, axis=0))
        painter.drawImage(margin, int(round(2 * ratio)), rgbToQImage(strip))
        pen = qt.QPen(qt.QColor(200, 200, 200))
        pen.setWidthF(max(1.0, ratio))
        painter.setPen(pen)
        y0 = int(round(2 * ratio)) + barHeight
        for value in legendTicks(low, high):
            x = margin + int(round((value - low) / max(high - low, 1e-9) * (barWidth - 1)))
            painter.drawLine(x, y0, x, y0 + int(round(3 * ratio)))
            painter.drawText(qt.QRectF(x - 3 * font.pixelSize(), y0 + 3 * ratio, 6 * font.pixelSize(),
                                       1.4 * font.pixelSize()), qt.Qt.AlignCenter, f"{tickText(value)}{unit}")
        painter.end()
        try:
            pixmap.setDevicePixelRatio(ratio)
        except Exception:
            pass
        label.setPixmap(pixmap)

    def _polarGeometry(self):
        """(disk diameter in logical pixels, device pixel ratio, side of the square)."""
        label = self.polarImage
        ratio = devicePixelRatio(label)
        rect = label.contentsRect()
        side = min(rect.width(), rect.height())
        if side < 40:
            return 0, ratio, side
        return int(side * (1.0 - 2.0 * self.POLAR_MARGIN)), ratio, side

    def polarSize(self):
        """Diameter of the bullseye in device pixels: render the polar image at this size."""
        diameter, ratio, _ = self._polarGeometry()
        return int(round(diameter * ratio)) if diameter else 0

    def setOverlays(self, overlay, scores):
        """overlay: "vessels", "segments" or None (no borders and values)."""
        self.overlay, self.showScores = overlay or None, bool(scores)
        self.paintPolar()

    def clear(self):
        self.setLegend(None)
        self.setPolar(None, None, "Polar map")

    def setPolar(self, qimage, segments, titleText, note="", message="", corner=""):
        """qimage: bullseye (square, outside the disk black) or None; segments: the values written
        on the regions, {overlay: values} (17 for "segments", one per vessel for "vessels"), or None
        to hide them; message: shown instead of the map when there is no image; corner: text in the
        upper left corner of the map (e.g. the TID ratio)."""
        self.polar = {"image": qimage, "segments": segments, "message": message, "corner": corner}
        self.polarTitle.setText(titleText)
        self.polarNote.setText(note)
        self.paintPolar()

    def paintPolar(self):
        diameter, ratio, side = self._polarGeometry()
        if not diameter:
            return
        sideDevice = int(round(side * ratio))
        disk = int(round(diameter * ratio))
        offset = (sideDevice - disk) // 2
        pixmap = qt.QPixmap(sideDevice, sideDevice)
        pixmap.fill(qt.QColor(0, 0, 0))
        painter = qt.QPainter(pixmap)
        painter.setRenderHint(qt.QPainter.Antialiasing, True)
        painter.setRenderHint(qt.QPainter.SmoothPixmapTransform, True)
        painter.setRenderHint(qt.QPainter.TextAntialiasing, True)
        font = painter.font()
        font.setPixelSize(max(9, int(round(disk / 22))))
        painter.setFont(font)
        image = self.polar.get("image")
        center = qt.QPointF(offset + disk / 2.0, offset + disk / 2.0)
        radius = disk / 2.0
        if image is not None and not image.isNull():
            painter.drawImage(qt.QRect(offset, offset, disk, disk), image)
            from MyocardialPerfusionSPECTLib import myocardium as wallLib

            def point(r, degrees):   # angle from anterior (up) toward the septum (left)
                a = math.radians(degrees)
                return qt.QPointF(center.x() - r * radius * math.sin(a), center.y() - r * radius * math.cos(a))
            overlay = self.regionOverride if self.overlay and self.regionOverride else self.overlay
            bold = overlay in ("vessels", "walls")   # few large regions: heavier borders, larger values
            if overlay:
                pen = qt.QPen(qt.QColor(255, 255, 255, 190 if bold else 170))
                pen.setWidthF(max(1.0, ratio) * (1.6 if bold else 1.0))
                painter.setPen(pen)
                painter.setBrush(qt.QBrush(qt.Qt.NoBrush))
                for line in wallLib.regionBorders(overlay):
                    if len(line) == 2:
                        painter.drawLine(point(*line[0]), point(*line[1]))
                    else:
                        painter.drawPolyline(qt.QPolygonF([point(r, degrees) for r, degrees in line]))
            values = self.polar.get("segments")
            if isinstance(values, dict):
                values = values.get(overlay)
            elif overlay != "segments":
                values = None
            if overlay and values is not None and self.showScores:
                big = bold
                box = font.pixelSize() * (4 if big else 3)
                names = wallLib.VESSELS if overlay == "vessels" else [None] * len(values)
                valueFont = qt.QFont(font)
                if big:
                    valueFont.setPixelSize(int(round(font.pixelSize() * 1.35)))
                    valueFont.setBold(True)
                for value, name, (r, degrees) in zip(values, names, wallLib.regionLabelPlaces(overlay)):
                    if value is None or value != value:
                        continue
                    p = point(r, degrees)
                    rect = qt.QRectF(p.x() - box / 2.0, p.y() - box / 4.0, box, box / 2.0)
                    painter.setFont(valueFont)
                    painter.setPen(qt.QColor(0, 0, 0))
                    painter.drawText(rect.translated(max(1.0, ratio), max(1.0, ratio)), qt.Qt.AlignCenter, f"{value:.0f}")
                    painter.setPen(qt.QColor(255, 255, 255))
                    painter.drawText(rect, qt.Qt.AlignCenter, f"{value:.0f}")
                    if name:
                        painter.setFont(font)
                        below = rect.translated(0, valueFont.pixelSize() * 1.1)
                        painter.setPen(qt.QColor(0, 0, 0))
                        painter.drawText(below.translated(max(1.0, ratio), max(1.0, ratio)), qt.Qt.AlignCenter, name)
                        painter.setPen(qt.QColor(230, 230, 230))
                        painter.drawText(below, qt.Qt.AlignCenter, name)
                painter.setFont(font)
            painter.setPen(qt.QColor(224, 224, 224))
            margin = offset
            painter.drawText(qt.QRectF(0, 0, sideDevice, margin), qt.Qt.AlignCenter, "ANT")
            painter.drawText(qt.QRectF(0, sideDevice - margin, sideDevice, margin), qt.Qt.AlignCenter, "INF")
            for text, x, angle in (("SEPT", margin / 2.0, -90.0), ("LAT", sideDevice - margin / 2.0, 90.0)):
                painter.save()   # side labels run along the rim
                painter.translate(x, sideDevice / 2.0)
                painter.rotate(angle)
                painter.drawText(qt.QRectF(-sideDevice / 2.0, -margin / 2.0, sideDevice, margin), qt.Qt.AlignCenter, text)
                painter.restore()
            corner = self.polar.get("corner")
            if corner:
                cornerFont = qt.QFont(font)
                cornerFont.setPixelSize(int(round(font.pixelSize() * 1.2)))
                cornerFont.setBold(True)
                painter.setFont(cornerFont)
                painter.setPen(qt.QColor(240, 240, 240))
                pad = max(2.0, 3.0 * ratio)
                painter.drawText(qt.QRectF(pad, pad, sideDevice / 2.0, sideDevice / 4.0),
                                 qt.Qt.AlignLeft | qt.Qt.AlignTop, corner)
                painter.setFont(font)
        else:
            painter.setPen(qt.QColor(160, 160, 160))
            painter.drawText(qt.QRectF(0, 0, sideDevice, sideDevice), qt.Qt.AlignCenter | qt.Qt.TextWordWrap,
                             self.polar.get("message") or "")
        painter.end()
        try:
            pixmap.setDevicePixelRatio(ratio)
        except Exception:
            pass
        self.polarImage.setPixmap(pixmap)

    def _onEvent(self, obj, event):
        if event.type() == qt.QEvent.Resize:
            if obj is self.legendImage:
                self.paintLegend()
                return False
            self.paintPolar()
            if self.onResized:
                self.onResized()
        return False


class CurveChart:
    """Line chart painted into a label: frames or views on the x axis.
    mode "percent": every series in % of its own mean (the image and the myocardium counts share one
    axis), with dashed lines at 100 +/- band %; points outside the band are ringed.
    mode "absolute": the values themselves, in unit."""

    def __init__(self):
        self.onResized = None
        self.series = []        # [(name, values, (r, g, b), legend text, dashed)]
        self.labels = []
        self.titleText = ""
        self.message = ""
        self.mode = "percent"
        self.band = 10.0
        self.unit = ""
        self.xTitle = "frame"
        self.marker = None      # index of the frame marked with a vertical line (cine), or None
        self.pointLabels = {}   # {index: text} written above the points of the first series (ED, ES)
        self.overlayText = ""   # large text in the top left corner of the plot (e.g. the EF)
        self.fromZero = False   # "absolute": the y axis starts at 0 (volumes)
        self.label = qt.QLabel()
        self.label.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Ignored)
        self.label.setMinimumSize(80, 60)
        self.filter = _EventForwarder(self._onEvent)
        self.label.installEventFilter(self.filter)

    def setCurves(self, labels, series, titleText="", message="", mode="percent", band=10.0, unit="", xTitle="frame"):
        """labels: x axis labels; series: [(name, values, (r, g, b), legend text[, dashed])]."""
        self.labels = [str(v) for v in labels]
        self.series = []
        for entry in series:
            name, values, color, legend = entry[:4]
            dashed = bool(entry[4]) if len(entry) > 4 else False
            values = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
            self.series.append((name, values, color, legend, dashed))
        self.titleText, self.message, self.mode = titleText, message, mode
        self.band, self.unit, self.xTitle = float(band), unit, xTitle
        self.paint()

    def setMarker(self, index):
        """Mark one x position (the frame shown in a cine) with a vertical line; None removes it."""
        if index != self.marker:
            self.marker = index
            self.paint()

    def clear(self):
        self.pointLabels = {}
        self.overlayText = ""
        self.marker = None
        self.setCurves([], [], "", "")

    def plotted(self):
        """[(series, values on the axis)] of the series with numbers."""
        out = []
        for entry in self.series:
            values = entry[1]
            if not len(values) or not np.any(np.isfinite(values)):
                continue
            if self.mode == "percent":
                mean = float(np.nanmean(values))
                values = 100.0 * values / mean if mean > 0 else values * np.nan
            out.append((entry, values))
        return out

    def paint(self):
        label = self.label
        ratio = devicePixelRatio(label)
        rect = label.contentsRect()
        if rect.width() < 40 or rect.height() < 40:
            return
        width, height = int(round(rect.width() * ratio)), int(round(rect.height() * ratio))
        pixmap = qt.QPixmap(width, height)
        pixmap.fill(qt.QColor(0, 0, 0))
        painter = qt.QPainter(pixmap)
        painter.setRenderHint(qt.QPainter.Antialiasing, True)
        painter.setRenderHint(qt.QPainter.TextAntialiasing, True)
        font = painter.font()
        fontSize = max(9, int(round(11 * ratio)))
        font.setPixelSize(fontSize)
        line = 1.5 * fontSize
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(qt.QColor(230, 230, 230))
        painter.drawText(qt.QRectF(0, 0, width, line), qt.Qt.AlignCenter, self.titleText)
        font.setBold(False)
        painter.setFont(font)
        plotted = self.plotted()
        if not plotted or not self.labels:
            if self.message:
                painter.setPen(qt.QColor(170, 170, 170))
                painter.drawText(qt.QRectF(fontSize, line, width - 2 * fontSize, height - line),
                                 qt.Qt.AlignCenter | qt.Qt.TextWordWrap, self.message)
            painter.end()
            self._show(pixmap, ratio)
            return
        finite = np.concatenate([v[np.isfinite(v)] for _, v in plotted])
        low, high = float(finite.min()), float(finite.max())
        if self.mode == "percent":
            low, high = min(low, 100.0 - 1.5 * self.band), max(high, 100.0 + 1.5 * self.band)
        span = max(high - low, 1e-6)
        if self.mode != "percent":
            low, high = low - 0.08 * span, high + 0.08 * span
            if self.fromZero and low > 0.0:
                low = 0.0
            span = high - low
        rawStep = span / 5.0
        magnitude = 10.0 ** math.floor(math.log10(rawStep))
        step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= rawStep)
        low, high = math.floor(low / step) * step, math.ceil(high / step) * step
        legendEntries = [entry for entry, _ in plotted if entry[0]]
        legendHeight = line * len(legendEntries) + 0.3 * line
        suffix = "%" if self.mode == "percent" else ""
        tickText = lambda v: f"{v:.0f}{suffix}" if step >= 1 else f"{v:.1f}{suffix}"
        left = max(painter.fontMetrics().horizontalAdvance(tickText(t)) if hasattr(painter.fontMetrics(), "horizontalAdvance")
                   else painter.fontMetrics().width(tickText(t)) for t in (low, high)) + 1.2 * fontSize
        right = width - 0.8 * fontSize
        top, bottom = 1.6 * line, height - 1.7 * line - legendHeight
        overlayTop = top
        if self.overlayText:   # its own band above the plot, so it never covers a point
            top += 2.0 * fontSize
        if bottom - top < 20 or right - left < 20:
            painter.end()
            self._show(pixmap, ratio)
            return
        n = len(self.labels)

        def x(index):
            return left + (right - left) * (index + 0.5) / n

        def y(value):
            return bottom - (bottom - top) * (value - low) / max(high - low, 1e-9)

        grid = qt.QPen(qt.QColor(65, 65, 65))
        grid.setWidthF(max(1.0, 0.8 * ratio))
        tick = low
        while tick <= high + 1e-6 * step:
            painter.setPen(grid)
            painter.drawLine(qt.QPointF(left, y(tick)), qt.QPointF(right, y(tick)))
            painter.setPen(qt.QColor(180, 180, 180))
            painter.drawText(qt.QRectF(0, y(tick) - fontSize, left - 0.4 * fontSize, 2 * fontSize),
                             qt.Qt.AlignRight | qt.Qt.AlignVCenter, tickText(tick))
            tick += step
        if self.mode == "percent":
            for level, style in ((100.0, qt.Qt.SolidLine), (100.0 - self.band, qt.Qt.DashLine),
                                 (100.0 + self.band, qt.Qt.DashLine)):
                pen = qt.QPen(qt.QColor(150, 150, 150))
                pen.setStyle(style)
                pen.setWidthF(max(1.0, ratio))
                painter.setPen(pen)
                painter.drawLine(qt.QPointF(left, y(level)), qt.QPointF(right, y(level)))
        painter.setPen(qt.QColor(180, 180, 180))
        every = max(1, int(math.ceil(n / max(1.0, (right - left) / (3.0 * fontSize)))))
        for index, text in enumerate(self.labels):
            if index % every and index != n - 1:
                continue
            painter.drawText(qt.QRectF(x(index) - 2 * fontSize, bottom + 0.2 * fontSize, 4 * fontSize, line),
                             qt.Qt.AlignHCenter | qt.Qt.AlignTop, text)
        axisTitle = self.xTitle + (f"   ({self.unit})" if self.unit else "")
        painter.drawText(qt.QRectF(left, bottom + line, right - left, line), qt.Qt.AlignCenter, axisTitle)
        if self.marker is not None and 0 <= self.marker < n:
            pen = qt.QPen(qt.QColor(255, 255, 255, 150))
            pen.setWidthF(max(1.0, 1.5 * ratio))
            painter.setPen(pen)
            painter.drawLine(qt.QPointF(x(self.marker), top), qt.QPointF(x(self.marker), bottom))
        markers = n <= 32
        radius = 3.0 * ratio
        for (name, _, color, legend, dashed), values in plotted:
            pen = qt.QPen(qt.QColor(*color))
            pen.setWidthF((1.3 if dashed else 2.0) * ratio)
            if dashed:
                pen.setStyle(qt.Qt.DashLine)
            painter.setPen(pen)
            points = [qt.QPointF(x(i), y(v)) if np.isfinite(v) else None for i, v in enumerate(values)]
            for a, b in zip(points[:-1], points[1:]):
                if a is not None and b is not None:
                    painter.drawLine(a, b)
            if markers and not dashed:
                painter.setBrush(qt.QBrush(qt.QColor(*color)))
                for i, point in enumerate(points):
                    if point is None:
                        continue
                    outside = self.mode == "percent" and abs(values[i] - 100.0) > self.band
                    painter.setPen(qt.QPen(qt.QColor(255, 255, 255) if outside else qt.QColor(*color)))
                    r = radius * (1.6 if outside else 1.0)
                    painter.drawEllipse(point, r, r)
                painter.setBrush(qt.QBrush())
        if self.pointLabels and plotted:
            values = plotted[0][1]
            painter.setPen(qt.QColor(235, 235, 235))
            for index, text in self.pointLabels.items():
                if 0 <= index < n and np.isfinite(values[index]):
                    yy = y(values[index])
                    above = yy - 1.4 * line > top - line
                    rectTop = yy - 1.3 * line if above else yy + 0.3 * line
                    painter.drawText(qt.QRectF(x(index) - 3 * fontSize, rectTop, 6 * fontSize, line),
                                     qt.Qt.AlignCenter, text)
        if self.overlayText:
            big = qt.QFont(font)
            big.setPixelSize(int(round(fontSize * 1.6)))
            big.setBold(True)
            painter.setFont(big)
            painter.setPen(qt.QColor(255, 255, 255))
            painter.drawText(qt.QRectF(left, overlayTop - 0.6 * fontSize, right - left, 2.0 * big.pixelSize()),
                             qt.Qt.AlignLeft | qt.Qt.AlignTop, self.overlayText)
            painter.setFont(font)
        legendTop = height - legendHeight
        for index, (name, _, color, legend, dashed) in enumerate(legendEntries):
            yy = legendTop + index * line
            painter.fillRect(qt.QRectF(left, yy + 0.4 * line, 1.4 * fontSize, 0.2 * line), qt.QColor(*color))
            painter.setPen(qt.QColor(220, 220, 220))
            painter.drawText(qt.QRectF(left + 2.0 * fontSize, yy, width - left, line), qt.Qt.AlignLeft | qt.Qt.AlignVCenter,
                             f"{name}   {legend}".rstrip())
        painter.end()
        self._show(pixmap, ratio)

    def _show(self, pixmap, ratio):
        try:
            pixmap.setDevicePixelRatio(ratio)
        except Exception:
            pass
        self.label.setPixmap(pixmap)

    def _onEvent(self, obj, event):
        if event.type() == qt.QEvent.Resize:
            self.paint()
            if self.onResized:
                self.onResized()
        return False


class CountCurvePanel:
    """A CurveChart on its own place of a layout: Gated QC counts per gate frame (in % of the mean),
    Gated Review LV cavity volume per frame."""

    def __init__(self):
        self.widget = qt.QWidget()
        self.widget.setObjectName("MPICountCurve")
        self.widget.setStyleSheet("QWidget#MPICountCurve { background-color: black; }")
        layout = qt.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)
        self.chart = CurveChart()
        layout.addWidget(self.chart.label, 1)

    def setCurves(self, *args, **kwargs):
        self.chart.setCurves(*args, **kwargs)

    def clear(self):
        self.chart.clear()


class GatingInfoPanel:
    """Gated QC, bottom right: a chart of the gating over the acquisition (R-R per projection view,
    shown when the scanner stores it per view) above the gating information as text."""

    def __init__(self):
        self.widget = qt.QWidget()
        self.widget.setObjectName("MPIGatingInfo")
        self.widget.setStyleSheet("QWidget#MPIGatingInfo { background-color: black; }")
        layout = qt.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self.chart = CurveChart()
        layout.addWidget(self.chart.label, 2)
        self.text = qt.QTextBrowser()
        self.text.setOpenLinks(False)
        self.text.setStyleSheet("QTextBrowser { background-color: black; color: #e0e0e0; border: none; }")
        layout.addWidget(self.text, 3)

    def setInfo(self, html, chart=None):
        """html: the information; chart: CurveChart.setCurves keyword arguments, or None to hide it."""
        self.text.setHtml(html or "")
        self.chart.label.setVisible(chart is not None)
        if chart is not None:
            self.chart.setCurves(**chart)

    def clear(self):
        self.setInfo("", None)


DEFAULT_FOV_CM = 13.0     # field of view of the slice tiles (stress and rest alike)
MIN_VISIBLE_SLICES = 8   # a row scrolls past its last slice (black tiles) until this many slices are left
REFIT_DELAY_MS = 1000    # wait after the last resize before the slice tiles are fitted and made again


def fovCrop(shape, spacing, center, fovMm):
    """In-plane window of a slice stack [k, j, i] showing fovMm around center (i, j) (the LV center;
    None: the middle of the slices): (i0, j0, ni, nj) in pixels, partly outside the stack when the
    field of view is larger than the volume."""
    _, h, w = shape
    ni = max(2, int(round(float(fovMm) / max(float(spacing[0]), 1e-6))))
    nj = max(2, int(round(float(fovMm) / max(float(spacing[1]), 1e-6))))
    ci, cj = (0.5 * (w - 1), 0.5 * (h - 1)) if center is None else (float(center[0]), float(center[1]))
    return int(round(ci - 0.5 * (ni - 1))), int(round(cj - 0.5 * (nj - 1))), ni, nj


def cropStack(stack, crop):
    """The window of fovCrop cut out of a stack [k, j, i] (zero outside the volume)."""
    stack = np.asarray(stack)
    i0, j0, ni, nj = crop
    k, h, w = stack.shape
    out = np.zeros((k, nj, ni), dtype=stack.dtype)
    si0, si1 = max(i0, 0), min(i0 + ni, w)
    sj0, sj1 = max(j0, 0), min(j0 + nj, h)
    if si1 > si0 and sj1 > sj0:
        out[:, sj0 - j0:sj1 - j0, si0 - i0:si1 - i0] = stack[:, sj0:sj1, si0:si1]
    return out


def tilePixelsToIjk(size, crop, shape):
    """(i, j) volume coordinates of the centers of an n x n tile (rows from the top = highest j), the
    mapping of cropStack + zoomInPlane; crop None: the whole slice."""
    _, h, w = shape
    i0, j0, ni, nj = crop if crop is not None else (0, 0, w, h)
    n = int(size)
    i = i0 + (np.arange(n) + 0.5) * (ni / float(n)) - 0.5
    j = j0 + (nj - 1) - ((np.arange(n) + 0.5) * (nj / float(n)) - 0.5)
    return np.meshgrid(i, j)


def lastFirstSlice(count):
    """Highest first slice (1-based) of a row of a count-slice volume: past the end until MIN_VISIBLE_SLICES
    are left."""
    return max(int(count) - min(MIN_VISIBLE_SLICES, int(count)) + 1, 1)


def bestGrid(count, width, height, gap=SliceMosaicPanel.GAP, maxRows=8, maxColumns=30, rowOverhead=0):
    """(rows, columns) of a tile grid showing count tiles in a width x height area: the largest tiles (of the
    grids with nearly those tiles, the one with the fewest rows), then as many columns as fit at that tile
    size, so that the rows show more slices without making the tiles smaller.
    rowOverhead: height taken between grid rows instead of the gap (a strip title, see
    SliceMosaicPanel.stripOverhead), 0 for the gap."""
    count = max(int(count), 1)
    candidates = []
    for rows in range(1, maxRows + 1):
        columns = max(2, int(np.ceil(count / rows)))
        if columns > maxColumns:
            continue
        between = rowOverhead if rowOverhead > 0 and rows > 1 else gap
        tile = min((width - gap * (columns - 1)) / columns, (height - between * (rows - 1)) / rows)
        candidates.append((tile, rows, columns))
    if not candidates:
        return 1, min(count, maxColumns)
    largest = max(c[0] for c in candidates)
    tile, rows, columns = min((c for c in candidates if c[0] >= 0.95 * largest), key=lambda c: c[1])
    if tile > 0:   # fill the rows: more columns at the same tile size
        columns = max(columns, min(int((width + gap) // (tile + gap)), maxColumns))
    return rows, columns


def _singletonPanel(attribute, tag, panelClass):
    """A panel and its view factory live for the whole Slicer session (also across module reloads),
    because the layout manager keeps using the factory registered first. After a module update the
    factory is pointed at a new panel; restarting Slicer is the most reliable way to pick it up."""
    state = getattr(slicer, attribute, None)
    if state is None:
        factory = slicer.qSlicerSingletonViewFactory()
        factory.setTagName(tag)
        slicer.app.layoutManager().registerViewFactory(factory)
        state = {"panel": None, "factory": factory, "version": None}
        setattr(slicer, attribute, state)
    if state.get("version") != MOSAIC_PANEL_VERSION or state.get("panel") is None:
        state["panel"] = panelClass()
        state["factory"].setWidget(state["panel"].widget)
        state["version"] = MOSAIC_PANEL_VERSION
    return state["panel"]


def mosaicPanel():
    from MyocardialPerfusionSPECTLib.views import MOSAIC_TAG
    return _singletonPanel("_mpiSliceMosaicState", MOSAIC_TAG, SliceMosaicPanel)


def gatedMosaicPanel():
    """Slice rows of Gated QC: its own panel, so the Perfusion slice rows keep their state."""
    from MyocardialPerfusionSPECTLib.views import GATED_MOSAIC_TAG
    return _singletonPanel("_mpiGatedMosaicState", GATED_MOSAIC_TAG, SliceMosaicPanel)


def countCurvePanel():
    from MyocardialPerfusionSPECTLib.views import GATED_CURVE_TAG
    return _singletonPanel("_mpiCountCurveState", GATED_CURVE_TAG, CountCurvePanel)


def gatingInfoPanel():
    from MyocardialPerfusionSPECTLib.views import GATED_INFO_TAG
    return _singletonPanel("_mpiGatingInfoState", GATED_INFO_TAG, GatingInfoPanel)


def gatedReviewMosaicPanel():
    """Cine slice rows of Gated Review (its own panel: Perfusion and Gated QC keep theirs)."""
    from MyocardialPerfusionSPECTLib.views import GATED_REVIEW_MOSAIC_TAG
    return _singletonPanel("_mpiGatedReviewMosaicState", GATED_REVIEW_MOSAIC_TAG, SliceMosaicPanel)


def gatedReviewMosaic2Panel():
    """HLA and VLA rows of the Gated Review layouts with three slices per axis (the SA rows are in
    gatedReviewMosaicPanel, beside the static 3D wall)."""
    from MyocardialPerfusionSPECTLib.views import GATED_REVIEW_MOSAIC2_TAG
    return _singletonPanel("_mpiGatedReviewMosaic2State", GATED_REVIEW_MOSAIC2_TAG, SliceMosaicPanel)


def volumeCurvePanelState(tag=None):
    """Attribute name of slicer holding the LV volume curve panel of a tag (stress: GATED_VOLUME_TAG)."""
    from MyocardialPerfusionSPECTLib.views import GATED_VOLUME_TAG
    return "_mpiGatedVolumeState" if tag in (None, GATED_VOLUME_TAG) else "_mpiGatedVolumeState_" + tag


def volumeCurvePanel(tag=None):
    """LV volume curve of Gated Review: GATED_VOLUME_TAG (stress, the default) or the rest study's tag."""
    from MyocardialPerfusionSPECTLib.views import GATED_VOLUME_TAG
    tag = tag or GATED_VOLUME_TAG
    return _singletonPanel(volumeCurvePanelState(tag), tag, CountCurvePanel)


def polarPanelState(tag):
    """Attribute name of slicer holding the panel of a polar map tag (see polarPanel)."""
    from MyocardialPerfusionSPECTLib.views import POLAR_TAG
    return "_mpiPolarPanelState" if tag == POLAR_TAG else "_mpiPolarPanelState_" + tag


def polarPanel(tag=None):
    """Panel of one polar map place of the layouts: POLAR_TAG (stress or single study, the default),
    POLAR_REST_TAG or POLAR_DIFF_TAG."""
    from MyocardialPerfusionSPECTLib.views import POLAR_TAG
    tag = tag or POLAR_TAG
    return _singletonPanel(polarPanelState(tag), tag, PolarPanel)


def existingPanel(attribute):
    """The panel made by _singletonPanel under this attribute, or None (does not make one)."""
    state = getattr(slicer, attribute, None)
    return state.get("panel") if state else None
