"""Interactive page canvas for BarForge.

The canvas shows one decoded page and edits the bars and the crop of a ``Document``
in place. All geometry is in original image pixels, integers only: ``x``/``y`` is the
inclusive top-left corner and ``x + width`` / ``y + height`` the exclusive bottom-right
one, exactly as ``imaging.render_pair`` reads them. The page is painted in
``drawBackground`` and the overlay (bars, crop dimming, handles) in ``drawForeground``;
the scene itself stays empty. Exports are rendered by ``imaging`` from the ``Document``,
so nothing painted here (margins, outlines, handles, dimming) can leak into a file, and
clearing the scene cannot invalidate canvas state.

Mouse
    select  click a bar to select it, drag it to move it; drag the square handles around
            the selected bar to change its width (left/right) and thickness (top/bottom).
            A selection only exists in this tool and is dropped when another one is chosen
    bar     click the start, then click the end: length and angle follow those two points;
            thickness is the brush thickness. A detached preview follows the cursor between
            clicks. New bars stay unselected, so brush changes never rewrite the last bar
    brush   drag with the left button: a round freehand stroke with the shared thickness/colour;
            one completed stroke is one undo step and remains outside the document until release
    crop    drag a rectangle; drag a handle to move an edge, drag inside to move the
            frame, Shift+drag to start a new frame from inside the old one
    always  wheel zooms at the cursor, middle button pans, held XButton1/XButton2 plus horizontal
            movement adjusts thickness, Delete removes the selected bar, Esc cancels the current gesture
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

from PIL import Image
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QResizeEvent,
    QShowEvent,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QWidget

from .i18n import tr
from .models import BAR_COLORS, Bar, BrushStroke, Document, Rect, bar_polygon

__all__ = ["ImageCanvas", "TOOLS"]

TOOLS = ("select", "bar", "brush", "crop")

#: left, top, right, bottom; right/bottom exclusive.
Box = tuple[int, int, int, int]

MIN_SCALE = 0.01
MAX_SCALE = 64.0
ZOOM_STEP = 1.25  # per wheel notch and per zoom_in()/zoom_out()
VIEW_PAD = 28  # screen px of free room around the page (the handles live there)
SCROLL_STEP = 90  # screen px per wheel notch for Shift/Alt scrolling

DRAG_THRESHOLD = 3  # screen px before a press turns into a drag
HANDLE_SIZE = 9  # drawn side of a resize handle, screen px
HANDLE_GAP = 8  # handle centre distance from the box edge; handles sit outside the box
HANDLE_HIT = 14  # side of the square that grabs a handle
BAR_HIT_PAD = 3  # extra screen px around a bar that still selects it
PAINT_PAD = HANDLE_GAP + HANDLE_SIZE + 4  # repaint margin around a changed box

MIN_BAR = 1  # smallest bar side, image px
MIN_CROP = 8  # smallest crop side, image px
DEFAULT_THICKNESS = 24
MAX_THICKNESS = 8192
SIDE_DRAG_STEP = 8  # horizontal viewport pixels per one source pixel of thickness
HISTORY_LIMIT = 200
MERGE_WINDOW = 1.0  # seconds in which repeated update_selected() edits share one undo step

#: handle name -> (dx, dy): the sign says which edge it moves (-1 left/top, +1 right/bottom).
_HANDLES: dict[str, tuple[int, int]] = {
    "nw": (-1, -1),
    "n": (0, -1),
    "ne": (1, -1),
    "w": (-1, 0),
    "e": (1, 0),
    "sw": (-1, 1),
    "s": (0, 1),
    "se": (1, 1),
}
_HANDLE_CURSORS = {
    "n": Qt.CursorShape.SizeVerCursor,
    "s": Qt.CursorShape.SizeVerCursor,
    "e": Qt.CursorShape.SizeHorCursor,
    "w": Qt.CursorShape.SizeHorCursor,
    "nw": Qt.CursorShape.SizeFDiagCursor,
    "se": Qt.CursorShape.SizeFDiagCursor,
    "ne": Qt.CursorShape.SizeBDiagCursor,
    "sw": Qt.CursorShape.SizeBDiagCursor,
}

_TOOL_HINTS = {
    "select": tr("Выбор: клик — выделить полосу, перетаскивание — переместить, квадратные маркеры — ширина и толщина, Delete — удалить"),
    "bar": tr("Полоса: первый клик — начало, второй — конец; направление по двум точкам, толщина — настройкой кисти; Esc — отмена"),
    "crop": tr("Кроп: нарисуйте рамку; маркеры двигают края, перетаскивание внутри — сдвиг, Shift — новая рамка поверх старой"),
    "brush": tr("Кисть [P]: протягивание ЛКМ рисует штрих текущего цвета и толщины; один штрих — один Undo; Esc — отмена"),
}

_BAR_FILL = {name: QColor(*rgb) for name, rgb in BAR_COLORS.items()}
_BAD_FILL = QColor(255, 0, 255)  # a colour imaging would reject: make it impossible to miss
_BACKGROUND = QColor(38, 38, 38)
_DIM = QColor(0, 0, 0, 150)
_ACCENT = QColor(255, 140, 0)
_HANDLE_FILL = QColor(255, 255, 255)
_PEN_FRAME = QPen(QColor(100, 100, 100), 1)
_PEN_OUTLINE = QPen(QColor(0, 150, 255), 1)
_PEN_SELECTED = QPen(_ACCENT, 2)
_PEN_HANDLE = QPen(_ACCENT, 1)
_PEN_CROP_DARK = QPen(QColor(0, 0, 0, 230), 1)
_PEN_CROP_LIGHT = QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine)


def _snap(value: float) -> int:
    """Round half up (Python's round() rounds half to even)."""
    return math.floor(value + 0.5)


def _clamp(value, low, high):
    return max(low, min(value, high))


def _bar_box(bar: Bar) -> Box:
    return (bar.x, bar.y, bar.x + bar.width, bar.y + bar.height)


def _bar_bounds(bar: Bar) -> Box:
    points = bar_polygon(bar)
    return (math.floor(min(x for x, _ in points)), math.floor(min(y for _, y in points)),
            math.ceil(max(x for x, _ in points)), math.ceil(max(y for _, y in points)))


def _moved_rotated(box: Box, angle: float, dx: float, dy: float, size: tuple[int, int]) -> Box:
    bar = Bar(box[0], box[1], box[2] - box[0], box[3] - box[1], angle=angle)
    left, top, right, bottom = _bar_bounds(bar)
    if right - left <= size[0]:
        dx = _clamp(dx, -left, size[0] - right)
    else:
        dx = _clamp(dx, -(left + right) / 2, size[0] - (left + right) / 2)
    if bottom - top <= size[1]:
        dy = _clamp(dy, -top, size[1] - bottom)
    else:
        dy = _clamp(dy, -(top + bottom) / 2, size[1] - (top + bottom) / 2)
    dx, dy = _snap(dx), _snap(dy)
    return box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy


def _resized_rotated(box: Box, handle: str, dx: float, dy: float, angle: float,
                     size: tuple[int, int]) -> Box:
    """Resize in the bar's local axes, preserving the opposite edge in image coordinates."""
    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    local_x, local_y = dx * cosine + dy * sine, -dx * sine + dy * cosine
    hx, hy = _HANDLES[handle]
    old_w, old_h = box[2] - box[0], box[3] - box[1]
    limit = max(1, math.ceil(math.hypot(*size)))
    width = _clamp(_snap(old_w + hx * local_x), 1, limit) if hx else old_w
    height = _clamp(_snap(old_h + hy * local_y), 1, limit) if hy else old_h
    shift_x, shift_y = hx * (width - old_w) / 2, hy * (height - old_h) / 2
    cx = (box[0] + box[2]) / 2 + shift_x * cosine - shift_y * sine
    cy = (box[1] + box[3]) / 2 + shift_x * sine + shift_y * cosine
    x, y = _snap(cx - width / 2), _snap(cy - height / 2)
    return _moved_rotated((x, y, x + width, y + height), angle, 0, 0, size)


def _crop_box(crop: Rect) -> Box:
    return (crop.x, crop.y, crop.x + crop.width, crop.y + crop.height)


def _set_bar_box(bar: Bar, box: Box) -> None:
    bar.x, bar.y, bar.width, bar.height = box[0], box[1], box[2] - box[0], box[3] - box[1]


def _make_crop(box: Box) -> Rect:
    return Rect(x=box[0], y=box[1], width=box[2] - box[0], height=box[3] - box[1])


def _clip(box: Box, width: int, height: int) -> Box | None:
    """Intersect ``box`` with the image; None when nothing is left."""
    left, top, right, bottom = box
    left, top, right, bottom = max(left, 0), max(top, 0), min(right, width), min(bottom, height)
    return (left, top, right, bottom) if left < right and top < bottom else None


def _at_least_one(low: int, high: int, limit: int) -> tuple[int, int]:
    """Widen an empty span [low, high) to one pixel inside [0, limit]."""
    if high > low:
        return low, high
    return (low, low + 1) if low < limit else (limit - 1, limit)


def _moved(box: Box, dx: int, dy: int, size: tuple[int, int]) -> Box:
    """``box`` shifted by (dx, dy), size kept, pushed back inside the image."""
    left, top, right, bottom = box
    width, height = size
    box_w, box_h = min(right - left, width), min(bottom - top, height)
    left = _clamp(left + dx, 0, width - box_w)
    top = _clamp(top + dy, 0, height - box_h)
    return left, top, left + box_w, top + box_h


def _resized(box: Box, handle: str, dx: int, dy: int, size: tuple[int, int], floor: int) -> Box:
    """``box`` with the edges owned by ``handle`` moved by (dx, dy), inside the image, >= ``floor``."""
    left, top, right, bottom = box
    hx, hy = _HANDLES[handle]
    if hx < 0:
        left = _clamp(left + dx, 0, right - floor)
    elif hx > 0:
        right = _clamp(right + dx, left + floor, size[0])
    if hy < 0:
        top = _clamp(top + dy, 0, bottom - floor)
    elif hy > 0:
        bottom = _clamp(bottom + dy, top + floor, size[1])
    return left, top, right, bottom


def _to_device(box: Box, sx: float, sy: float, dx: float, dy: float) -> Box:
    """Scene box -> viewport pixels, snapped so outlines and hit tests agree."""
    left, top, right, bottom = box
    return (
        _snap(left * sx + dx),
        _snap(top * sy + dy),
        _snap(right * sx + dx),
        _snap(bottom * sy + dy),
    )


def _handle_center(name: str, box: Box) -> tuple[float, float]:
    """Centre of a handle for a box given in viewport pixels."""
    left, top, right, bottom = box
    hx, hy = _HANDLES[name]
    x = left - HANDLE_GAP if hx < 0 else right + HANDLE_GAP if hx > 0 else (left + right) / 2
    y = top - HANDLE_GAP if hy < 0 else bottom + HANDLE_GAP if hy > 0 else (top + bottom) / 2
    return x, y


def _normalize_color(color: str) -> str:
    name = str(color).lower()
    if name not in BAR_COLORS:
        raise ValueError(tr("Недопустимый цвет полосы: {color!r} (допустимы #ffffff и #000000)", color=color))
    return name


def _to_pixmap(image: Image.Image) -> QPixmap:
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    width, height = rgb.size
    data = rgb.tobytes()
    packed = QImage(data, width, height, width * 3, QImage.Format.Format_RGB888)
    # convertToFormat() copies into Qt-owned memory, so ``data`` may be released afterwards.
    pixmap = QPixmap.fromImage(packed.convertToFormat(QImage.Format.Format_RGB32))
    if pixmap.isNull():
        raise ValueError(tr("Не удалось подготовить изображение к показу (не хватает памяти?)"))
    return pixmap


@dataclass(frozen=True, slots=True)
class _State:
    """What undo/redo restores: the markup and the index of the selected bar."""

    bars: tuple[tuple[int, int, int, int, str, float], ...]
    strokes: tuple[tuple[tuple[tuple[int, int], ...], int, str], ...]
    crop: Box | None
    selected: int | None

    def same_markup(self, other: _State) -> bool:
        return self.bars == other.bars and self.strokes == other.strokes and self.crop == other.crop


@dataclass(slots=True)
class _Gesture:
    """A mouse gesture in progress. Draw previews stay outside the document until the second click."""

    kind: str  # draw | move | resize | crop_new | crop_move | crop_resize
    before: _State
    press_view: QPointF
    press_scene: QPointF
    bar: Bar | None = None
    origin: Box = (0, 0, 0, 0)  # box of the bar / crop at press time
    handle: str = ""
    thickness: int = 0  # brush values are frozen at press time
    color: str = "#ffffff"
    stroke: BrushStroke | None = None
    angle: float = 0.0
    started: bool = False  # moved past DRAG_THRESHOLD


class ImageCanvas(QGraphicsView):
    """Zoomable page view that edits ``Document.bars`` and ``Document.crop`` directly."""

    #: The document's bars or crop were modified (drawing, move/resize, delete, undo, ...).
    changed = Signal()
    #: The selected Bar or None; a selection only exists in the select tool. Re-sent with the
    #: same Bar whenever its geometry or colour changes, so panels bound to it stay in sync.
    selection_changed = Signal(object)
    #: Short user-facing hint or result for the status bar.
    status = Signal(str)
    #: The user changed brush thickness with a held side mouse button; value is in source pixels.
    brush_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._document: Document | None = None
        self._pixmap: QPixmap | None = None
        self._size = (0, 0)
        self._tool = "select"
        self._brush_thickness = DEFAULT_THICKNESS
        self._brush_color = "#ffffff"
        self._selected: Bar | None = None
        self._reported: Bar | None = None  # what selection_changed last announced
        self._reported_values: tuple | None = None
        self._gesture: _Gesture | None = None
        self._undo: deque[_State] = deque(maxlen=HISTORY_LIMIT)
        self._redo: list[_State] = []
        self._merge_key: tuple | None = None
        self._merge_time = 0.0
        self._fit_mode = True  # keep the page fitted on resize until the user zooms or pans
        self._pan_last: QPointF | None = None
        self._side_adjust: tuple[Qt.MouseButton, float, int, float] | None = None
        self._cursor_shape: Qt.CursorShape | None = None

        # The scene stays empty on purpose (see the module docstring).
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setBackgroundBrush(_BACKGROUND)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        # Zoom anchoring is done by hand (see _zoom_by); Qt's anchors fight the scroll bars.
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    # ------------------------------------------------------------------ properties

    @property
    def document(self) -> Document | None:
        return self._document

    @property
    def selected_bar(self) -> Bar | None:
        return self._selected

    @property
    def tool(self) -> str:
        return self._tool

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def zoom(self) -> float:
        """View pixels per image pixel (1.0 is 100 %)."""
        return self.transform().m11()

    # ------------------------------------------------------------------ document

    def set_document(self, image: Image.Image, document: Document) -> None:
        """Show ``image`` and edit ``document`` in place.

        Resets selection, undo history and zoom (the page is fitted). Geometry in the
        document that lies outside the image is clamped (bars left with no area are
        dropped); ``changed`` is emitted only in that case. Raises ValueError when the
        image is empty or cannot be turned into a pixmap, leaving the canvas untouched.
        """
        width, height = image.size
        if width < 1 or height < 1:
            raise ValueError(tr("Изображение пустое"))
        pixmap = _to_pixmap(image)
        self._end_gesture()
        self._pixmap = pixmap
        self._size = (width, height)
        self._document = document
        self._selected = None
        self._reset_history()
        repaired = self._sanitize()
        self._fit_view()
        self.viewport().update()
        self._flush_selection()
        if repaired:
            self.status.emit(tr("Часть геометрии выходила за пределы изображения и была исправлена"))
            self.changed.emit()

    def clear_document(self) -> None:
        """Drop the page, the document reference, selection and undo history."""
        self._end_gesture()
        self._document = None
        self._pixmap = None
        self._size = (0, 0)
        self._selected = None
        self._pan_last = None
        self._side_adjust = None
        self._reset_history()
        self.resetTransform()
        self.setSceneRect(QRectF())
        self._cursor_shape = None
        self.viewport().unsetCursor()
        self.viewport().update()
        self._flush_selection()

    def refresh_overlays(self) -> None:
        """Repaint after the Document was modified behind the canvas' back."""
        if self._document is None:
            return
        if self._selected is not None and self._index_of(self._selected) is None:
            self._selected = None
        self.viewport().update()
        self._flush_selection()

    # ------------------------------------------------------------------ tools and brush

    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            raise ValueError(tr("Неизвестный инструмент: {tool!r}", tool=tool))
        self._end_gesture()
        self._side_adjust = None
        self._tool = tool
        if tool != "select":
            self._select(None)  # a selection only exists in the select tool
        self.viewport().update()  # handles of the old tool disappear, the new ones appear
        self._refresh_cursor()
        self.status.emit(_TOOL_HINTS[tool])

    def set_brush(self, thickness: int, color: str) -> None:
        """Thickness and colour shared by new bars and freehand strokes; existing bars are unchanged."""
        self._brush_thickness = _clamp(int(thickness), MIN_BAR, MAX_THICKNESS)
        self._brush_color = _normalize_color(color)

    def update_selected(self, thickness: int | None = None, color: str | None = None,
                        angle: float | None = None) -> None:
        """Edit thickness/colour/angle without changing the bar's centre."""
        self._end_gesture()
        bar = self._current_bar()
        if bar is None or (thickness is None and color is None and angle is None):
            return
        fill = None if color is None else _normalize_color(color)
        if angle is not None and (not math.isfinite(angle) or not -180 <= angle <= 180):
            raise ValueError(tr("Угол линии должен быть от -180 до 180 градусов"))
        before = self._capture()
        if thickness is not None:
            height = self._size[1]
            new_height = _clamp(int(thickness), MIN_BAR, height)
            bar.y = (2 * bar.y + bar.height - new_height) // 2
            bar.height = new_height
        if fill is not None:
            bar.color = fill
        if angle is not None:
            bar.angle = float(angle)
        if bar.angle:
            _set_bar_box(bar, _moved_rotated(_bar_box(bar), bar.angle, 0, 0, self._size))
        else:
            bar.y = _clamp(bar.y, 0, self._size[1] - bar.height)
        kind = ("t" if thickness is not None else "") + ("c" if color is not None else "") + ("a" if angle is not None else "")
        self._finish(before, merge=(kind, id(bar)))

    def delete_selected(self) -> None:
        self._end_gesture()
        bar = self._current_bar()
        if bar is None:
            return
        before = self._capture()
        del self._document.bars[self._index_of(bar)]
        self._selected = None
        self._finish(before, tr("Полоса удалена"))

    def clear_bars(self) -> None:
        """Delete all bars and brush strokes as one undo step, retaining the crop."""
        self._end_gesture()
        doc = self._document
        if doc is None or not (doc.bars or doc.strokes):
            return
        before = self._capture()
        count = len(doc.bars) + len(doc.strokes)
        doc.bars.clear()
        doc.strokes.clear()
        self._selected = None
        self._finish(before, tr("Удалено элементов разметки: {count}", count=count))

    def reset_crop(self) -> None:
        self._end_gesture()
        doc = self._document
        if doc is None or doc.crop is None:
            return
        before = self._capture()
        doc.crop = None
        self._finish(before, tr("Кроп сброшен"))

    def add_bars(self, bars: list[Bar], replace: bool = False) -> None:
        """Insert ``bars`` as one undo step; with ``replace`` the existing bars are discarded.

        Every bar is copied and clipped to the image; bars with nothing left inside are
        skipped. Raises ValueError for a colour other than #ffffff / #000000.
        """
        self._end_gesture()
        doc = self._document
        if doc is None:
            return
        width, height = self._size
        fresh: list[Bar] = []
        for bar in bars:
            fill = _normalize_color(bar.color)
            if bar.angle:
                fresh_bar = Bar(int(bar.x), int(bar.y), int(bar.width), int(bar.height), fill, float(bar.angle))
                if _clip(_bar_bounds(fresh_bar), width, height) is not None:
                    fresh.append(fresh_bar)
                continue
            x, y = int(bar.x), int(bar.y)
            box = _clip((x, y, x + int(bar.width), y + int(bar.height)), width, height)
            if box is not None:
                fresh.append(Bar(x=box[0], y=box[1], width=box[2] - box[0], height=box[3] - box[1], color=fill))
        if not fresh and not (replace and doc.bars):
            if bars:
                self.status.emit(tr("Все полосы лежат вне изображения — ничего не добавлено"))
            return
        before = self._capture()
        if replace:
            doc.bars[:] = fresh
            self._selected = None
        else:
            doc.bars.extend(fresh)
        if replace and not fresh:
            message = tr("Удалено полос: {value}", value=len(before.bars))
        else:
            skipped = len(bars) - len(fresh)
            note = tr(" (вне изображения пропущено: {skipped})", skipped=skipped) if skipped else ""
            message = tr("Добавлено полос: {value}{note}", value=len(fresh), note=note)
        self._finish(before, message)

    # ------------------------------------------------------------------ history

    def undo(self) -> None:
        self._step(self._undo, self._redo, tr("Отмена действия"), tr("Нечего отменять"))

    def redo(self) -> None:
        self._step(self._redo, self._undo, tr("Повтор действия"), tr("Нечего повторять"))

    def _step(self, source, target, done: str, empty: str) -> None:
        self._end_gesture()
        if self._document is None:
            return
        if not source:
            self.status.emit(empty)
            return
        target.append(self._capture())
        self._restore(source.pop())
        self._merge_key = None
        self.viewport().update()
        self.changed.emit()
        self._flush_selection()
        self.status.emit(done)

    def _reset_history(self) -> None:
        self._undo.clear()
        self._redo.clear()
        self._merge_key = None

    def _capture(self) -> _State:
        doc = self._document
        crop = doc.crop
        return _State(
            bars=tuple((b.x, b.y, b.width, b.height, b.color, b.angle) for b in doc.bars),
            crop=None if crop is None else _crop_box(crop),
            strokes=tuple((stroke.points, stroke.thickness, stroke.color) for stroke in doc.strokes),
            selected=None if self._selected is None else self._index_of(self._selected),
        )

    def _restore(self, state: _State) -> None:
        """Write ``state`` back into the document, reusing Bar objects so references stay valid."""
        doc = self._document
        bars = doc.bars
        for index, (x, y, width, height, color, angle) in enumerate(state.bars):
            if index < len(bars):
                bar = bars[index]
                bar.x, bar.y, bar.width, bar.height, bar.color, bar.angle = x, y, width, height, color, angle
            else:
                bars.append(Bar(x=x, y=y, width=width, height=height, color=color, angle=angle))
        del bars[len(state.bars):]
        strokes = doc.strokes
        for index, (points, thickness, color) in enumerate(state.strokes):
            if index < len(strokes):
                strokes[index].points, strokes[index].thickness, strokes[index].color = points, thickness, color
            else:
                strokes.append(BrushStroke(points, thickness, color))
        del strokes[len(state.strokes):]
        doc.crop = None if state.crop is None else _make_crop(state.crop)
        index = state.selected
        keep = self._tool == "select" and index is not None and index < len(bars)
        self._selected = bars[index] if keep else None

    def _record(self, before: _State, merge: tuple | None) -> bool:
        """Push ``before`` as an undo step if the markup changed since; True if it did."""
        if self._capture().same_markup(before):
            return False
        now = time.monotonic()
        merged = (
            merge is not None
            and merge == self._merge_key
            and bool(self._undo)
            and now - self._merge_time <= MERGE_WINDOW
        )
        if not merged:
            self._undo.append(before)
        self._redo.clear()
        self._merge_key, self._merge_time = merge, now
        return True

    def _finish(self, before: _State, message: str = "", merge: tuple | None = None) -> bool:
        """Close an edit: record undo, repaint, then notify (changed, status, selection)."""
        modified = self._record(before, merge)
        self.viewport().update()
        if modified:
            self.changed.emit()
            if message:
                self.status.emit(message)
        self._flush_selection()
        return modified

    # ------------------------------------------------------------------ selection

    def _index_of(self, bar: Bar) -> int | None:
        for index, candidate in enumerate(self._document.bars):
            if candidate is bar:  # Bar compares by value; identity is what matters here
                return index
        return None

    def _current_bar(self) -> Bar | None:
        """The selected bar if it is still in the document."""
        if self._document is None or self._selected is None:
            return None
        if self._index_of(self._selected) is None:
            self._selected = None
            self._flush_selection()
            return None
        return self._selected

    def _select(self, bar: Bar | None) -> None:
        if bar is self._selected:
            return
        self._invalidate_bar(self._selected)
        self._selected = bar
        self._invalidate_bar(bar)
        if self._gesture is None:
            self._flush_selection()

    def _flush_selection(self) -> None:
        """Announce the selection if it, or the selected bar's values, differ from the last report."""
        bar = self._selected
        values = None if bar is None else (bar.x, bar.y, bar.width, bar.height, bar.color, bar.angle)
        if bar is self._reported and values == self._reported_values:
            return
        self._reported, self._reported_values = bar, values
        self.selection_changed.emit(bar)

    # ------------------------------------------------------------------ view

    def fit_image(self) -> None:
        self._fit_view()
        self.status.emit(tr("Масштаб: {value}%", value=round(self.zoom * 100)))

    def zoom_in(self) -> None:
        self._zoom_by(ZOOM_STEP)

    def zoom_out(self) -> None:
        self._zoom_by(1 / ZOOM_STEP)

    def _to_scene(self, point: QPointF) -> QPointF:
        t = self.viewportTransform()
        return QPointF((point.x() - t.dx()) / t.m11(), (point.y() - t.dy()) / t.m22())

    def _to_view(self, x: float, y: float) -> tuple[float, float]:
        t = self.viewportTransform()
        return x * t.m11() + t.dx(), y * t.m22() + t.dy()

    def _device_box(self, box: Box) -> Box:
        t = self.viewportTransform()
        return _to_device(box, t.m11(), t.m22(), t.dx(), t.dy())

    def _set_scale(self, scale: float) -> None:
        self.setTransform(QTransform.fromScale(scale, scale))
        # A fixed screen margin around the page keeps edge handles reachable at any zoom.
        margin = VIEW_PAD / scale
        width, height = self._size
        self.setSceneRect(QRectF(-margin, -margin, width + 2 * margin, height + 2 * margin))

    def _fit_view(self) -> None:
        self._fit_mode = True
        if self._document is None:
            return
        width, height = self._size
        free = self.maximumViewportSize()  # as if no scroll bar were shown, so fitting is stable
        free_w = free.width() - 2 * VIEW_PAD - 1
        free_h = free.height() - 2 * VIEW_PAD - 1
        if free_w < 1 or free_h < 1:
            return  # not laid out yet; showEvent/resizeEvent will fit again
        self._set_scale(_clamp(min(free_w / width, free_h / height), MIN_SCALE, MAX_SCALE))
        self.centerOn(width / 2, height / 2)
        self.viewport().update()

    def _zoom_by(self, factor: float, anchor: QPointF | None = None) -> None:
        """Zoom by ``factor`` keeping the page point under ``anchor`` (viewport px; default: centre) fixed."""
        if self._document is None:
            return
        old = self.zoom
        new = _clamp(old * factor, MIN_SCALE, MAX_SCALE)
        if new == old:
            return
        if anchor is None:
            anchor = QPointF(self.viewport().width() / 2, self.viewport().height() / 2)
        focus = self._to_scene(anchor)
        self._fit_mode = False
        self._set_scale(new)
        fx, fy = self._to_view(focus.x(), focus.y())
        hbar, vbar = self.horizontalScrollBar(), self.verticalScrollBar()
        hbar.setValue(hbar.value() + _snap(fx - anchor.x()))
        vbar.setValue(vbar.value() + _snap(fy - anchor.y()))
        self.viewport().update()
        self.status.emit(tr("Масштаб: {value}%", value=round(new * 100)))

    def _pan(self, pos: QPointF) -> None:
        last = self._pan_last
        dx, dy = _snap(pos.x() - last.x()), _snap(pos.y() - last.y())
        if not dx and not dy:
            return
        hbar, vbar = self.horizontalScrollBar(), self.verticalScrollBar()
        before = (hbar.value(), vbar.value())
        hbar.setValue(before[0] - dx)
        vbar.setValue(before[1] - dy)
        if (hbar.value(), vbar.value()) != before:
            self._fit_mode = False
        self._pan_last = QPointF(last.x() + dx, last.y() + dy)  # keep sub-pixel remainders

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._document is not None and self._fit_mode:
            self._fit_view()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if self._document is not None and self._fit_mode:
            self._fit_view()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._document is None:
            super().wheelEvent(event)
            return
        delta = event.angleDelta()
        modifiers = event.modifiers()
        sideways = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        vertical = bool(modifiers & Qt.KeyboardModifier.AltModifier)
        if sideways or vertical or (delta.y() == 0 and delta.x() != 0):
            step = _snap((delta.y() or delta.x()) / 120 * SCROLL_STEP)
            scroll = self.verticalScrollBar() if vertical and not sideways else self.horizontalScrollBar()
            before = scroll.value()
            scroll.setValue(before - step)
            if scroll.value() != before:
                self._fit_mode = False
            event.accept()
        elif delta.y() != 0:
            self._zoom_by(ZOOM_STEP ** (delta.y() / 120.0), event.position())
            event.accept()
        else:
            event.ignore()

    # ------------------------------------------------------------------ painting

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawBackground(painter, rect)
        pixmap = self._pixmap
        if pixmap is None:
            return
        width, height = self._size
        # One extra source pixel around the exposed part (as QGraphicsPixmapItem does)
        # keeps smooth scaling seamless between partial repaints.
        exposed = QRectF(rect).adjusted(-1, -1, 1, 1).intersected(QRectF(0, 0, width, height))
        if exposed.isEmpty():
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom < 1.0)
        painter.drawPixmap(exposed, pixmap, exposed)

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawForeground(painter, rect)
        doc = self._document
        if doc is None or self._pixmap is None:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self._paint_markup(painter, rect, doc)
        painter.save()
        try:
            painter.resetTransform()  # outlines and handles keep their size at any zoom
            self._paint_overlay(painter, doc)
        finally:
            painter.restore()

    def _paint_markup(self, painter: QPainter, rect: QRectF, doc: Document) -> None:
        """Bars and crop dimming, in image pixels so they line up with the page at any zoom."""
        painter.save()
        painter.setClipRect(QRectF(0, 0, *self._size), Qt.ClipOperation.IntersectClip)
        g = self._gesture
        for bar in doc.bars:
            self._paint_bar(painter, rect, bar)
        for stroke in doc.strokes:
            self._paint_stroke(painter, rect, stroke)
        if g is not None and g.kind == "brush" and g.stroke is not None:
            self._paint_stroke(painter, rect, g.stroke)
        if g is not None and g.kind == "draw" and g.bar is not None:
            self._paint_bar(painter, rect, g.bar)
        painter.restore()
        crop = doc.crop
        if crop is not None:
            width, height = self._size
            c_left, c_top, c_right, c_bottom = _crop_box(crop)
            for dim in (
                QRectF(0, 0, width, c_top),
                QRectF(0, c_bottom, width, height - c_bottom),
                QRectF(0, c_top, c_left, c_bottom - c_top),
                QRectF(c_right, c_top, width - c_right, c_bottom - c_top),
            ):
                if dim.width() > 0 and dim.height() > 0:
                    painter.fillRect(dim, _DIM)

    @staticmethod
    def _paint_bar(painter: QPainter, rect: QRectF, bar: Bar) -> None:
        bounds = _bar_bounds(bar)
        if not QRectF(bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]).intersects(rect):
            return
        fill = _BAR_FILL.get(str(bar.color).lower(), _BAD_FILL)
        if bar.angle:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in bar_polygon(bar)]))
        else:
            painter.fillRect(QRectF(bar.x, bar.y, bar.width, bar.height), fill)

    @staticmethod
    def _paint_stroke(painter: QPainter, rect: QRectF, stroke: BrushStroke) -> None:
        if not stroke.points:
            return
        points = [QPointF(x, y) for x, y in stroke.points]
        radius = stroke.thickness / 2
        bounds = QRectF(min(point.x() for point in points)-radius, min(point.y() for point in points)-radius,
                        max(point.x() for point in points)-min(point.x() for point in points)+stroke.thickness,
                        max(point.y() for point in points)-min(point.y() for point in points)+stroke.thickness)
        if not bounds.intersects(rect):
            return
        fill = _BAR_FILL.get(str(stroke.color).lower(), _BAD_FILL)
        if len(points) == 1:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            painter.drawEllipse(points[0], radius, radius)
            return
        pen = QPen(fill, stroke.thickness, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolyline(QPolygonF(points))

    def _paint_overlay(self, painter: QPainter, doc: Document) -> None:
        """Frames and handles in viewport pixels."""
        t = self.viewportTransform()
        sx, sy, dx, dy = t.m11(), t.m22(), t.dx(), t.dy()
        view_w, view_h = self.viewport().width(), self.viewport().height()
        painter.setBrush(Qt.BrushStyle.NoBrush)

        page = _to_device((0, 0, self._size[0], self._size[1]), sx, sy, dx, dy)
        painter.setPen(_PEN_FRAME)
        painter.drawRect(QRectF(page[0] - 0.5, page[1] - 0.5, page[2] - page[0] + 1, page[3] - page[1] + 1))

        painter.setPen(_PEN_OUTLINE)
        selected = self._selected
        for bar in doc.bars:
            if bar is not selected:
                painter.drawPolygon(self._bar_view_polygon(bar))
        g = self._gesture
        if g is not None and g.kind == "draw":
            painter.setPen(_PEN_SELECTED)
            if g.bar is not None:
                painter.drawPolygon(self._bar_view_polygon(g.bar))
            painter.drawEllipse(t.map(g.press_scene), 3, 3)


        crop = doc.crop
        if crop is not None:
            box = _to_device(_crop_box(crop), sx, sy, dx, dy)
            ring = QRectF(box[0] + 0.5, box[1] + 0.5, max(box[2] - box[0] - 1, 0), max(box[3] - box[1] - 1, 0))
            painter.setPen(_PEN_CROP_DARK)
            painter.drawRect(ring)
            painter.setPen(_PEN_CROP_LIGHT)
            painter.drawRect(ring)
            if self._tool == "crop":
                self._paint_handles(painter, box)

        if selected is not None:
            painter.setPen(_PEN_SELECTED)
            painter.drawPolygon(self._bar_view_polygon(selected))
            if self._tool == "select":
                painter.setPen(_PEN_HANDLE)
                painter.setBrush(_HANDLE_FILL)
                half = HANDLE_SIZE // 2
                for cx, cy in self._bar_handles(selected).values():
                    painter.drawRect(QRectF(_snap(cx) - half + .5, _snap(cy) - half + .5,
                                            HANDLE_SIZE - 1, HANDLE_SIZE - 1))
                painter.setBrush(Qt.BrushStyle.NoBrush)

    def _bar_view_polygon(self, bar: Bar) -> QPolygonF:
        transform = self.viewportTransform()
        return QPolygonF([transform.map(QPointF(x, y)) for x, y in bar_polygon(bar)])

    def _bar_handles(self, bar: Bar) -> dict[str, tuple[float, float]]:
        cx, cy = self._to_view(bar.x + bar.width / 2, bar.y + bar.height / 2)
        cosine, sine = math.cos(math.radians(bar.angle)), math.sin(math.radians(bar.angle))
        result = {}
        for name, (hx, hy) in _HANDLES.items():
            lx, ly = hx * (bar.width * self.zoom / 2 + HANDLE_GAP), hy * (bar.height * self.zoom / 2 + HANDLE_GAP)
            result[name] = (cx + lx * cosine - ly * sine, cy + lx * sine + ly * cosine)
        return result

    def _hit_bar_handle(self, pos: QPointF, bar: Bar) -> str | None:
        nearest, distance = None, math.inf
        for name, (x, y) in self._bar_handles(bar).items():
            dx, dy = abs(pos.x() - x), abs(pos.y() - y)
            if dx <= HANDLE_HIT / 2 and dy <= HANDLE_HIT / 2 and dx * dx + dy * dy < distance:
                nearest, distance = name, dx * dx + dy * dy
        return nearest

    @staticmethod
    def _paint_handles(painter: QPainter, box: Box) -> None:
        painter.setPen(_PEN_HANDLE)
        painter.setBrush(_HANDLE_FILL)
        half = HANDLE_SIZE // 2
        for name in _HANDLES:
            cx, cy = _handle_center(name, box)
            painter.drawRect(QRectF(_snap(cx) - half + 0.5, _snap(cy) - half + 0.5, HANDLE_SIZE - 1, HANDLE_SIZE - 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _invalidate(self, box: Box) -> None:
        """Repaint the viewport area around a scene box, handle room included."""
        t = self.viewportTransform()
        sx, sy = t.m11(), t.m22()
        left = math.floor(box[0] * sx + t.dx()) - PAINT_PAD
        top = math.floor(box[1] * sy + t.dy()) - PAINT_PAD
        right = math.ceil(box[2] * sx + t.dx()) + PAINT_PAD
        bottom = math.ceil(box[3] * sy + t.dy()) + PAINT_PAD
        self.viewport().update(QRect(left, top, right - left, bottom - top))

    def _invalidate_bar(self, bar: Bar | None) -> None:
        if bar is not None:
            self._invalidate(_bar_bounds(bar))

    # ------------------------------------------------------------------ hit testing and cursor

    def _bar_at(self, pos: QPointF) -> Bar | None:
        """Topmost bar under a viewport point, with a few pixels of slack for thin bars."""
        scene = self._to_scene(pos)
        slack = BAR_HIT_PAD / self.zoom
        for bar in reversed(self._document.bars):
            cosine, sine = math.cos(math.radians(bar.angle)), math.sin(math.radians(bar.angle))
            dx, dy = scene.x() - bar.x - bar.width / 2, scene.y() - bar.y - bar.height / 2
            lx, ly = dx * cosine + dy * sine, -dx * sine + dy * cosine
            if abs(lx) <= bar.width / 2 + slack and abs(ly) <= bar.height / 2 + slack:
                return bar
        return None

    def _hit_handle(self, pos: QPointF, box: Box) -> str | None:
        """Handle of a scene box under a viewport point; the nearest one wins when they overlap."""
        device = self._device_box(box)
        half = HANDLE_HIT / 2
        best: str | None = None
        best_dist = 0.0
        for name in _HANDLES:
            cx, cy = _handle_center(name, device)
            off_x, off_y = abs(pos.x() - cx), abs(pos.y() - cy)
            if off_x <= half and off_y <= half:
                dist = off_x * off_x + off_y * off_y
                if best is None or dist < best_dist:
                    best, best_dist = name, dist
        return best

    def _inside_crop(self, scene: QPointF) -> bool:
        crop = self._document.crop
        if crop is None:
            return False
        left, top, right, bottom = _crop_box(crop)
        return left <= scene.x() < right and top <= scene.y() < bottom

    def _cursor_at(self, pos: QPointF) -> Qt.CursorShape:
        if self._tool in ("bar", "brush"):
            return Qt.CursorShape.CrossCursor
        if self._tool == "select":
            bar = self._selected
            if bar is not None:
                handle = self._hit_bar_handle(pos, bar)
                if handle:
                    return _HANDLE_CURSORS[handle]
            return Qt.CursorShape.SizeAllCursor if self._bar_at(pos) is not None else Qt.CursorShape.ArrowCursor
        crop = self._document.crop
        if crop is not None:
            handle = self._hit_handle(pos, _crop_box(crop))
            if handle:
                return _HANDLE_CURSORS[handle]
            if self._inside_crop(self._to_scene(pos)):
                return Qt.CursorShape.SizeAllCursor
        return Qt.CursorShape.CrossCursor

    def _set_cursor(self, shape: Qt.CursorShape) -> None:
        if shape != self._cursor_shape:
            self._cursor_shape = shape
            self.viewport().setCursor(shape)

    def _update_cursor(self, pos: QPointF) -> None:
        self._set_cursor(self._cursor_at(pos))

    def _refresh_cursor(self) -> None:
        """Re-evaluate the cursor for a stationary pointer (after a tool change)."""
        if self._document is None:
            return
        pos = self.viewport().mapFromGlobal(QCursor.pos())
        if self.viewport().rect().contains(pos):
            self._update_cursor(QPointF(pos))

    # ------------------------------------------------------------------ mouse and keys

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._document is None:
            super().mousePressEvent(event)
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        button = event.button()
        if button in (Qt.MouseButton.XButton1, Qt.MouseButton.XButton2):
            self._end_gesture()
            self._side_adjust = (button, event.position().x(), self._brush_thickness, event.position().x())
            self._set_cursor(Qt.CursorShape.SizeHorCursor)
            self.status.emit(tr("Боковая кнопка зажата: движение вправо/влево изменяет толщину"))
            event.accept()
        elif button == Qt.MouseButton.MiddleButton:
            self._end_gesture()
            self._pan_last = QPointF(event.position())
            self._set_cursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        elif button == Qt.MouseButton.LeftButton and self._pan_last is None and self._side_adjust is None:
            if self._gesture is None or self._gesture.kind != "draw":
                self._end_gesture()  # a swallowed release must not leak an old drag
            self._press(QPointF(event.position()), event.modifiers())
            event.accept()
        else:
            super().mousePressEvent(event)
    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        # A quick second click arrives as a double click; treat it as a normal press.
        self.mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._document is None:
            super().mouseMoveEvent(event)
            return
        pos = QPointF(event.position())
        if self._side_adjust is not None:
            button, start_x, start_thickness, _last_x = self._side_adjust
            if not event.buttons() & button:
                self._side_adjust = None
            else:
                value = _clamp(start_thickness+math.trunc((pos.x()-start_x)/SIDE_DRAG_STEP), MIN_BAR, MAX_THICKNESS)
                if value != self._brush_thickness:
                    self._brush_thickness = value
                    self.brush_changed.emit(value)
                    self.status.emit(tr("Толщина кисти и новых полос: {value} px", value=value))
                self._side_adjust = (button, start_x, start_thickness, pos.x())
        elif self._pan_last is not None:
            if event.buttons() & Qt.MouseButton.MiddleButton:
                self._pan(pos)
            else:
                self._pan_last = None  # the release was lost
                self._update_cursor(pos)
        elif self._gesture is not None:
            if self._gesture.kind == "draw":
                self._preview_draw(self._gesture, self._to_scene(pos))
            elif event.buttons() & Qt.MouseButton.LeftButton:
                self._drag(self._gesture, pos)
            else:
                self._end_gesture()  # the release was lost
        else:
            self._update_cursor(pos)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._document is None:
            super().mouseReleaseEvent(event)
            return
        button = event.button()
        if button in (Qt.MouseButton.XButton1, Qt.MouseButton.XButton2) and self._side_adjust is not None:
            self._side_adjust = None
        elif button == Qt.MouseButton.MiddleButton and self._pan_last is not None:
            self._pan_last = None
        elif button == Qt.MouseButton.LeftButton and self._gesture is not None:
            if self._gesture.kind != "draw":
                self._drag(self._gesture, QPointF(event.position()))
                self._end_gesture()
        else:
            super().mouseReleaseEvent(event)
            return
        self._update_cursor(QPointF(event.position()))
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Delete and self._selected is not None:
            self.delete_selected()
        elif key == Qt.Key.Key_Escape and self._gesture is not None:
            self._cancel_gesture()
        elif key == Qt.Key.Key_Escape and self._selected is not None:
            self._select(None)
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    # ------------------------------------------------------------------ gestures

    def _start(self, kind: str, view: QPointF, scene: QPointF, **fields) -> None:
        self._gesture = _Gesture(kind=kind, before=self._capture(), press_view=view, press_scene=scene, **fields)

    def _press(self, view: QPointF, modifiers: Qt.KeyboardModifier) -> None:
        scene = self._to_scene(view)
        if self._tool == "select":
            self._press_select(view, scene)
        elif self._tool == "brush":
            width, height = self._size
            point = (_clamp(_snap(scene.x()), 0, width-1), _clamp(_snap(scene.y()), 0, height-1))
            stroke = BrushStroke((point,), self._brush_thickness, self._brush_color)
            self._start("brush", view, scene, stroke=stroke, started=True)
            self.viewport().update()
        elif self._tool == "bar":
            g = self._gesture
            if g is not None and g.kind == "draw":
                self._preview_draw(g, scene)
                if g.bar is not None:
                    self._gesture = None
                    self._document.bars.append(g.bar)
                    self._finish(g.before, tr("Полоса добавлена: {value}×{value2}, угол {value3:.1f}°", value=g.bar.width, value2=g.bar.height, value3=g.bar.angle))
                else:
                    self.status.emit(tr("Выберите вторую точку, отличную от начала полосы; Esc — отмена"))
            else:
                width, height = self._size
                start = QPointF(_clamp(scene.x(), 0, width), _clamp(scene.y(), 0, height))
                self._start("draw", view, start, thickness=self._brush_thickness, color=self._brush_color)
                self.viewport().update()
                self.status.emit(tr("Начало полосы задано. Кликните по конечной точке; Esc — отмена"))
        else:
            self._press_crop(view, scene, bool(modifiers & Qt.KeyboardModifier.ShiftModifier))
        self._update_cursor(view)

    def _press_select(self, view: QPointF, scene: QPointF) -> None:
        bar = self._selected
        if bar is not None:
            handle = self._hit_bar_handle(view, bar)
            if handle:
                self._start("resize", view, scene, bar=bar, handle=handle, origin=_bar_box(bar), angle=bar.angle)
                return
        hit = self._bar_at(view)
        # Select first (this may notify the parent), then open the gesture.
        self._select(hit)
        if hit is not None:
            self._start("move", view, scene, bar=hit, origin=_bar_box(hit))

    def _press_crop(self, view: QPointF, scene: QPointF, force_new: bool) -> None:
        crop = self._document.crop
        if crop is not None:
            box = _crop_box(crop)
            handle = self._hit_handle(view, box)
            if handle:
                self._start("crop_resize", view, scene, handle=handle, origin=box)
                return
            if not force_new and self._inside_crop(scene):
                self._start("crop_move", view, scene, origin=box)
                return
        self._start("crop_new", view, scene)

    def _drag(self, g: _Gesture, pos: QPointF) -> None:
        if not g.started:
            if math.hypot(pos.x() - g.press_view.x(), pos.y() - g.press_view.y()) < DRAG_THRESHOLD:
                return
            g.started = True
        scene = self._to_scene(pos)
        if g.kind == "brush" and g.stroke is not None:
            point = (_clamp(_snap(scene.x()), 0, self._size[0]-1), _clamp(_snap(scene.y()), 0, self._size[1]-1))
            if point != g.stroke.points[-1]:
                g.stroke.points += (point,)
                self.viewport().update()
            return
        if g.kind == "crop_new":
            self._drag_crop_new(g, scene)
        else:
            dx = _snap(scene.x() - g.press_scene.x())
            dy = _snap(scene.y() - g.press_scene.y())
            if g.bar is not None and g.bar.angle and g.kind == "move":
                box = _moved_rotated(g.origin, g.bar.angle, dx, dy, self._size)
            elif g.bar is not None and g.bar.angle and g.kind == "resize":
                box = _resized_rotated(g.origin, g.handle, dx, dy, g.angle, self._size)
            elif g.kind in ("move", "crop_move"):
                box = _moved(g.origin, dx, dy, self._size)
            else:
                floor = MIN_BAR if g.kind == "resize" else min(MIN_CROP, *self._size)
                box = _resized(g.origin, g.handle, dx, dy, self._size, floor)
            self._apply_box(g, box)

    def _preview_draw(self, g: _Gesture, scene: QPointF) -> None:
        """Build the rotated strip between the first click and the pointer, without editing the document."""
        width, height = self._size
        x0, y0 = g.press_scene.x(), g.press_scene.y()
        x1, y1 = _clamp(scene.x(), 0, width), _clamp(scene.y(), 0, height)
        dx, dy = x1 - x0, y1 - y0
        length = _snap(math.hypot(dx, dy))
        self._invalidate_bar(g.bar)
        if length < MIN_BAR:
            g.bar = None
            return
        angle = math.degrees(math.atan2(dy, dx))
        g.bar = Bar(_snap((x0 + x1 - length) / 2), _snap((y0 + y1 - g.thickness) / 2),
                    length, g.thickness, g.color, angle)
        self._invalidate_bar(g.bar)

    def _drag_crop_new(self, g: _Gesture, scene: QPointF) -> None:
        width, height = self._size
        x0 = _clamp(_snap(g.press_scene.x()), 0, width)
        x1 = _clamp(_snap(scene.x()), 0, width)
        y0 = _clamp(_snap(g.press_scene.y()), 0, height)
        y1 = _clamp(_snap(scene.y()), 0, height)
        left, right = _at_least_one(min(x0, x1), max(x0, x1), width)
        top, bottom = _at_least_one(min(y0, y1), max(y0, y1), height)
        self._apply_box(g, (left, top, right, bottom))

    def _apply_box(self, g: _Gesture, box: Box) -> None:
        """Write a gesture's box into its bar (bar gestures) or into the crop (crop gestures)."""
        if g.bar is not None:
            old = _bar_box(g.bar)
            if old == box:
                return
            old_bounds = _bar_bounds(g.bar)
            _set_bar_box(g.bar, box)
            self._invalidate(old_bounds)
            self._invalidate(_bar_bounds(g.bar))
        else:
            crop = self._document.crop
            if crop is not None and _crop_box(crop) == box:
                return
            self._document.crop = _make_crop(box)
            self.viewport().update()  # the dimmed area changes everywhere
        self.status.emit(self._describe_box(g, box))

    @staticmethod
    def _describe_box(g: _Gesture, box: Box) -> str:
        left, top, right, bottom = box
        what = tr("Кроп") if g.kind.startswith("crop") else tr("Полоса")
        return f"{what}: {right - left}×{bottom - top}, x={left}, y={top}"

    def _end_gesture(self) -> None:
        """Finish a drag as one undo step, or discard an uncommitted two-click preview."""
        g = self._gesture
        if g is None:
            return
        self._gesture = None
        if g.kind == "draw":
            self.viewport().update()
            return
        if g.kind == "brush" and g.stroke is not None:
            self._document.strokes.append(g.stroke)
            self._finish(g.before, tr("Штрих кисти: {value} точек, {value2} px", value=len(g.stroke.points), value2=g.stroke.thickness))
            return
        if g.kind == "crop_new" and g.started:
            crop = self._document.crop
            floor = min(MIN_CROP, *self._size)
            if crop is None or crop.width < floor or crop.height < floor:
                self._restore(g.before)
                self.viewport().update()
                self._flush_selection()
                self.status.emit(tr("Рамка слишком мала — кроп не изменён"))
                return
        message = ""
        if g.kind.startswith("crop"):
            crop = self._document.crop
            if crop is not None:
                message = self._describe_box(g, _crop_box(crop))
        elif g.bar is not None:
            verb = {"move": tr("перемещена"), "resize": tr("изменена")}[g.kind]
            message = tr("Полоса {verb}: {value}×{value2}, x={value3}, y={value4}", verb=verb, value=g.bar.width, value2=g.bar.height, value3=g.bar.x, value4=g.bar.y)
        self._finish(g.before, message)

    def _cancel_gesture(self) -> None:
        g = self._gesture
        if g is None:
            return
        self._gesture = None
        if g.kind != "draw":
            self._restore(g.before)
        self.viewport().update()
        self._flush_selection()
        self.status.emit(tr("Действие отменено"))

    # ------------------------------------------------------------------ housekeeping

    def _sanitize(self) -> bool:
        """Clamp geometry that came from outside (session files) into the image; True if any changed.

        Sessions are read without pixels, so a bar or crop may stick out of, or lie
        entirely outside, the image. Bars with no area inside are dropped.
        """
        doc = self._document
        width, height = self._size
        repaired = False
        kept: list[Bar] = []
        for bar in doc.bars:
            if bar.angle:
                if _clip(_bar_bounds(bar), width, height) is None:
                    repaired = True
                else:
                    kept.append(bar)
                continue
            box = _clip(_bar_box(bar), width, height)
            if box is None:
                repaired = True
                continue
            if box != _bar_box(bar):
                _set_bar_box(bar, box)
                repaired = True
            kept.append(bar)
        if len(kept) != len(doc.bars):
            doc.bars[:] = kept
        for stroke in doc.strokes:
            clamped = tuple((_clamp(x, 0, width-1), _clamp(y, 0, height-1)) for x, y in stroke.points)
            if clamped != stroke.points:
                stroke.points = clamped
                repaired = True
        crop = doc.crop
        if crop is not None:
            box = _clip(_crop_box(crop), width, height)
            if box is None:
                doc.crop = None
                repaired = True
            elif box != _crop_box(crop):
                doc.crop = _make_crop(box)
                repaired = True
        return repaired
