"""Colour maps and the slice panel of the result screen.

Tiles are drawn directly: each slice is first resized to the tile's size on screen in the data
domain (cubic B-spline, see resampling.zoomInPlane), then colour mapped with numpy, then handed to
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
MOSAIC_PANEL_VERSION = 8   # bump when SliceMosaicPanel changes, so a reload replaces the old panel

# (combo label, color node name or ID); the first one is the default. "MPI ..." tables are this
# extension's own (colortables.py), made in the scene when first used.
COLORMAPS = [
    ("PET-DICOM", "MPI PET-DICOM"),
    ("Hel", "MPI Hel"),
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
    return slicer.mrmlScene.GetNodeByID(key) or slicer.mrmlScene.GetFirstNodeByName(key)


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
    can render them again at the new size."""

    GAP = 3
    LEGEND_WIDTH = 62

    def __init__(self):
        self.onScroll = None
        self.onResized = None
        self.gridRows = DEFAULT_GRID_ROWS
        self.gridColumns = DEFAULT_GRID_COLUMNS
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
        self.rowsLayout.setSpacing(2)
        outer.addLayout(self.rowsLayout, 1)
        self.rows = {}
        self.rowKeys = []
        self.setRowKeys([(name, ORIENTATION_TITLES[name]) for name in ORIENTATIONS])

    def setRowKeys(self, keys):
        """keys: [(row key, title)] from top to bottom. Rows are made again only when the keys change."""
        if [k for k, _ in keys] == self.rowKeys:
            return
        for row in self.rows.values():
            for widget in (row["title"], row["image"]):
                self.rowsLayout.removeWidget(widget)
                widget.removeEventFilter(self.filter)
                widget.setParent(None)
                widget.deleteLater()
        self.rows = {}
        self.rowKeys = [k for k, _ in keys]
        for key, titleText in keys:
            title = qt.QLabel(titleText)
            title.setStyleSheet("font-weight: bold;")
            image = qt.QLabel()
            image.setObjectName(f"MPIMosaic_{key}")
            image.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Ignored)  # pixmap must not drive the size
            image.setMinimumSize(50, 20)
            image.setAlignment(qt.Qt.AlignCenter)
            image.installEventFilter(self.filter)
            self.rowsLayout.addWidget(title)
            self.rowsLayout.addWidget(image, 1)
            self.rows[key] = {"title": title, "image": image, "tiles": []}

    def _geometry(self, name):
        """(tile size in logical pixels, device pixel ratio) for a row, or (0, ratio) if too small."""
        label = self.rows[name]["image"]
        ratio = devicePixelRatio(label)
        rect = label.contentsRect()
        width, height = rect.width(), rect.height()
        if width < 10 or height < 10:
            return 0, ratio
        gap, rows, columns = self.GAP, self.gridRows, self.gridColumns
        tile = min((width - gap * (columns - 1)) // columns, (height - gap * (rows - 1)) // rows)
        return max(tile, 4), ratio

    def tileSize(self, name):
        """Edge of one tile in device pixels: render tile images at this size."""
        tile, ratio = self._geometry(name)
        return int(round(tile * ratio)) if tile else 0

    def pixelRatio(self):
        return devicePixelRatio(self.rows[self.rowKeys[0]]["image"]) if self.rowKeys else devicePixelRatio(self.widget)

    def setRow(self, name, tiles, titleText):
        """tiles: list of (QImage or None, slice number or None), row-major."""
        row = self.rows.get(name)
        if row is None:
            return
        row["tiles"] = tiles
        row["title"].setText(titleText)
        self.paintRow(name)

    def setGrid(self, rows, columns):
        self.gridRows, self.gridColumns = int(rows), int(columns)
        for name in self.rowKeys:
            self.paintRow(name)

    def clear(self):
        self.setRowKeys([(name, ORIENTATION_TITLES[name]) for name in ORIENTATIONS])
        for name in ORIENTATIONS:
            self.setRow(name, [], ORIENTATION_TITLES[name])
        self.legend = None
        self.paintLegend()

    def paintRow(self, name):
        row = self.rows[name]
        label = row["image"]
        tile, ratio = self._geometry(name)
        if not tile:
            return
        gap, rows, columns = self.GAP, self.gridRows, self.gridColumns
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
        name = objectName.split("_", 1)[1]
        if event.type() == qt.QEvent.Resize:
            self.paintRow(name)
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
    Segment borders and segment scores can be switched off (showSegments, showScores)."""

    POLAR_MARGIN = 0.09    # share of the side left around the disk for the ANT / SEPT / LAT / INF labels

    def __init__(self):
        self.onResized = None
        self.showSegments = True
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

    def setLegend(self, table, low=0.0, high=100.0):
        """Horizontal color bar under the map (when its window differs from the slice legend);
        table None hides it."""
        self.legend = None if table is None else (np.asarray(table, dtype=np.uint8), float(low), float(high))
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
        table, low, high = self.legend
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
        for k in range(5):
            x = margin + int(round(k * (barWidth - 1) / 4.0))
            value = low + k * (high - low) / 4.0
            painter.drawLine(x, y0, x, y0 + int(round(3 * ratio)))
            painter.drawText(qt.QRectF(x - 3 * font.pixelSize(), y0 + 3 * ratio, 6 * font.pixelSize(),
                                       1.4 * font.pixelSize()), qt.Qt.AlignCenter, f"{value:.0f}%")
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

    def setOverlays(self, segments, scores):
        self.showSegments, self.showScores = bool(segments), bool(scores)
        self.paintPolar()

    def clear(self):
        self.setLegend(None)
        self.setPolar(None, None, "Polar map")

    def setPolar(self, qimage, segments, titleText, note="", message=""):
        """qimage: bullseye (square, outside the disk black) or None; segments: 17 values shown on
        the segments (None to hide); message: shown instead of the map when there is no image."""
        self.polar = {"image": qimage, "segments": segments, "message": message}
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
            from MyocardialPerfusionSPECTLib.myocardium import POLAR_RING_EDGES
            inner, apical, middle, outer = POLAR_RING_EDGES[1:]

            def point(r, degrees):   # angle from anterior (up) toward the septum (left)
                a = math.radians(degrees)
                return qt.QPointF(center.x() - r * radius * math.sin(a), center.y() - r * radius * math.cos(a))
            if self.showSegments:
                pen = qt.QPen(qt.QColor(255, 255, 255, 170))
                pen.setWidthF(max(1.0, ratio))
                painter.setPen(pen)
                painter.setBrush(qt.QBrush(qt.Qt.NoBrush))
                for edge in POLAR_RING_EDGES[1:]:
                    painter.drawEllipse(center, radius * edge, radius * edge)
                for k in range(4):
                    painter.drawLine(point(inner, 45 + 90 * k), point(apical, 45 + 90 * k))
                for k in range(6):
                    painter.drawLine(point(apical, 30 + 60 * k), point(outer, 30 + 60 * k))
            segments = self.polar.get("segments")
            if segments is not None and self.showScores:
                places = ([(0.5 * (middle + outer), 60 * k) for k in range(6)]
                          + [(0.5 * (apical + middle), 60 * k) for k in range(6)]
                          + [(0.5 * (inner + apical), 90 * k) for k in range(4)] + [(0.0, 0)])
                box = font.pixelSize() * 3
                for value, (r, degrees) in zip(segments, places):
                    if value is None or value != value:
                        continue
                    p = point(r, degrees)
                    rect = qt.QRectF(p.x() - box / 2.0, p.y() - box / 4.0, box, box / 2.0)
                    painter.setPen(qt.QColor(0, 0, 0))
                    painter.drawText(rect.translated(max(1.0, ratio), max(1.0, ratio)), qt.Qt.AlignCenter, f"{value:.0f}")
                    painter.setPen(qt.QColor(255, 255, 255))
                    painter.drawText(rect, qt.Qt.AlignCenter, f"{value:.0f}")
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
