"""Local neural auto-censor: ONNX YOLO detectors turned into opaque censor bars.

Two bundled detectors (``_MODELS``) examine the whole image and, optionally, overlapping square tiles (a
detector only sees 640 px, so small regions need the zoom). Every window is also analysed turned by 90, 180
and 270 degrees (test-time augmentation). Candidates are mapped back to image pixels, merged per kind across
windows, rotations and models; survivors define zones for thin parallel lines with open gaps. No Qt,
no network; onnxruntime is imported only on the first analysis.

Detections are candidates for human review: scores are model confidences, not guarantees.
"""

from __future__ import annotations

import ast
import hashlib
import math
import os
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
from PIL import Image

from .models import BAR_COLORS, Bar, Rect

__all__ = [
    "CancelledError",
    "CensorOptions",
    "Detection",
    "DetectionResult",
    "LocalCensorEngine",
    "MAX_THICKNESS",
    "MAX_VARIATION",
    "bars_from_detections",
]

#: Largest nominal line thickness and largest width-deviation bound (px) that ``CensorOptions`` accepts.
MAX_THICKNESS = 64
MAX_VARIATION = 64

#: Bundled model file -> class names its ONNX metadata must declare.
_MODELS: dict[str, tuple[str, ...]] = {
    "anime-censor-v1-s.onnx": ("nipple_f", "penis", "pussy"),
    "erax-anti-nsfw-v11.onnx": ("anus", "make_love", "nipple", "penis", "vagina"),
}
#: Model class name -> kind of the detection; classes of a model that are not listed are never censored.
_KIND_OF: dict[str, str] = {
    "anus": "anus",
    "make_love": "act",
    "nipple": "nipple",
    "nipple_f": "nipple",
    "penis": "genital",
    "pussy": "genital",
    "vagina": "genital",
}
_KINDS = ("genital", "anus", "nipple", "act")
_LABELS = tuple(_KIND_OF)  # candidate label id -> class name
_CANONICAL_LABELS = {"pussy": "vagina", "nipple_f": "nipple"}
_LABEL_OBJECT = np.array([_LABELS.index(_CANONICAL_LABELS.get(name, name)) for name in _LABELS], dtype=np.int8)

_INPUT = 640  # side of the square model input, px
_PAD_VALUE = 114  # letterbox grey, as in YOLO training
_TILE_RATIO = 0.65  # tile edge relative to the shorter image side
_TILE_MIN = 384  # smallest tile edge, px
_TILE_OVERLAP = 0.25  # share of a tile that neighbouring tiles must cover again
_MAX_VIEWS = 25  # whole image + tiles
_EDGE_MARGIN = 3.0  # a box this close (model px) to an inner tile edge counts as cut off by it
_DUP_IOU = 0.5  # same-kind boxes overlapping at least this much (IoU) are one object
_DUP_INSIDE = 0.8  # ... as is a weaker box with this share of its area inside a stronger one
_MAX_CANDIDATES = 1000  # per inference, best scores first; guards against noise at tiny thresholds

#: One decoded box: x0, y0, x1, y1 in image pixels, model score, class label id (index into ``_LABELS``).
_CANDIDATE = np.dtype([("box", np.float64, (4,)), ("score", np.float32), ("label", np.int8)])


class CancelledError(Exception):
    """Raised by ``LocalCensorEngine.analyze`` when its ``cancelled`` callback reports True."""


@dataclass(frozen=True)
class CensorOptions:
    """What to look for and how the bars are drawn.

    ``confidence`` filters anatomical targets; ``act_confidence`` filters optional scene context only.
    ``padding`` / ``size_scale`` change only the placement zone. ``thickness`` is nominal: each line is
    thicker or thinner than it (the side is random) by a whole number of pixels from ``variation_min`` to
    ``variation_max``, both inclusive; 0 and 0 keep every line at the nominal thickness. The result stays
    bounded by the placement zone, the spacing and a 1 px minimum, so a small zone or spacing cuts it short.
    ``seed`` keeps the pattern stable while reviewing. ``tiles`` and ``rotations`` affect detection only.
    ``auto_angle`` places stripes across each foreground's long axis; otherwise ``angle`` is used.
    """

    color: str = "#ffffff"
    confidence: float = 0.12
    act_confidence: float = 0.40
    padding: float = 0.0
    size_scale: float = 1.0
    thickness: int = 6
    variation_min: int = 3
    variation_max: int = 8
    spacing: int = 48
    angle: float = 65.0
    auto_angle: bool = True
    include_nipples: bool = False
    include_anus: bool = False
    include_acts: bool = True
    tiles: bool = True
    rotations: bool = True
    seed: int = 0


@dataclass
class Detection:
    """One candidate region: ``kind`` is 'genital', 'anus', 'nipple' or 'act'; ``label`` is the model's class name."""

    kind: str
    label: str
    confidence: float
    rect: Rect
    outline: tuple[tuple[float, float], ...] = ()  # foreground contour in source pixels
    holes: tuple[tuple[tuple[float, float], ...], ...] = ()
    islands: tuple[tuple[tuple[float, float], ...], ...] = ()


@dataclass
class DetectionResult:
    """Outcome of one analysis; geometry is in pixels of the analysed image.

    ``views`` counts the image windows (whole image + tiles), ``model_names`` the models that ran.
    ``warnings`` are notes for the user, e.g. that nothing was found.
    """

    detections: list[Detection]
    bars: list[Bar]
    model_names: tuple[str, ...]
    views: int
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)


class _Head(NamedTuple):
    """One loaded detector."""

    name: str  # model file stem
    session: Any  # onnxruntime.InferenceSession
    input_name: str
    classes: np.ndarray  # class id -> index into _LABELS, -1 for classes that are never censored


class _Window(NamedTuple):
    """One analysed part of the image and where its resized copy sits inside the model input."""

    x: int
    y: int
    width: int
    height: int
    left: int
    top: int
    fit_w: int
    fit_h: int
    inner: tuple[bool, bool, bool, bool]  # left, top, right, bottom edge of the window is not an image edge


class LocalCensorEngine:
    """Runs the bundled ONNX detectors on the CPU.

    The sessions are created on the first analysis and then shared by all later ones; ``analyze`` may be
    called from any thread.
    """

    def __init__(self, models_dir: Path | None = None) -> None:
        self._dir = Path(models_dir) if models_dir is not None else Path(__file__).parent / "assets"
        self._heads: list[_Head] | None = None
        self._lock = threading.Lock()

    def analyze(
        self,
        image: Image.Image,
        options: CensorOptions,
        progress: Callable[[int, int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> DetectionResult:
        """Find censor candidates in ``image`` and build bars for them.

        Coordinates are pixels of ``image`` itself (a crop stays a crop). Returned targets are localized
        anatomy from the specialized model, never broad genital/scene boxes from the general model.
        Foreground segmentation preserves contours and background holes for clipping individual bars.
        ``include_acts`` enables scene boundaries for context only. Progress counts inference and local
        segmentation steps; cancellation is polled before every step.

        Raises ValueError for invalid options or an empty image, CancelledError when cancelled and
        RuntimeError when onnxruntime or a model file is missing or broken; there is no fallback to
        geometric guessing.
        """
        started = time.monotonic()
        _validated(options)
        rgb = image if image.mode == "RGB" else image.convert("RGB")
        width, height = rgb.size
        if width < 1 or height < 1:
            raise ValueError("Размер изображения должен быть положительным")
        views, coarse = _views(width, height, options.tiles)
        turns = range(4) if options.rotations else range(1)
        total = len(_MODELS) * len(views) * len(turns)
        if progress is not None:
            progress(0, total)
        if cancelled is not None and cancelled():
            raise CancelledError()
        heads = self._ready()
        search_options = replace(options, confidence=.05)
        cutoffs = [_cutoffs(head, search_options) for head in heads]
        pixels = np.asarray(rgb)
        blob = np.empty((1, 3, _INPUT, _INPUT), np.float32)
        found: dict[str, list[np.ndarray]] = {head.name: [] for head in heads}
        done = 0
        for left, top, right, bottom in views:
            canvas, fit_left, fit_top, fit_w, fit_h = _letterbox(pixels[top:bottom, left:right])
            window = _Window(
                left, top, right - left, bottom - top, fit_left, fit_top, fit_w, fit_h,
                (left > 0, top > 0, right < width, bottom < height),
            )
            for turn in turns:
                np.divide(np.rot90(canvas, turn).transpose(2, 0, 1), 255.0, out=blob[0], dtype=np.float32)
                for head, cutoff in zip(heads, cutoffs):
                    if cancelled is not None and cancelled():
                        raise CancelledError()
                    batch = _candidates(_run(head, blob), head, cutoff, turn, window)
                    found[head.name].append(batch[_dedupe(batch)])
                    done += 1
                    if progress is not None:
                        progress(done, total)
        if cancelled is not None and cancelled():
            raise CancelledError()
        from .anatomy import refine_anatomy
        by_head = {name: np.concatenate(batches) for name, batches in found.items()}
        detections, warnings = refine_anatomy(
            pixels, by_head, options, done, progress=progress, cancelled=cancelled
        )
        if coarse:
            warnings.append(
                "Изображение очень большое или вытянутое: фрагменты увеличены, чтобы ограничить время анализа, "
                "поэтому мелкие области могут быть пропущены."
            )
        if not detections:
            warnings.append(
                "Нейросети не нашли областей для цензуры. Автоматический поиск не гарантирует полноты: "
                "проверьте изображение и добавьте полосы вручную."
            )
        return DetectionResult(
            detections,
            bars_from_detections(detections, width, height, options),
            tuple(head.name for head in heads),
            len(views),
            time.monotonic() - started,
            warnings,
        )

    def _ready(self) -> list[_Head]:
        """The loaded detectors; loads all of them on first use (all or nothing)."""
        with self._lock:
            if self._heads is None:
                self._heads = _open_heads(self._dir)
            return self._heads


def bars_from_detections(
    detections: Iterable[Detection], width: int, height: int, options: CensorOptions
) -> list[Bar]:
    """Thin rectangular stripes clipped to each selected anatomical foreground, with open gaps.

    Outlined regions use their own transverse direction when ``auto_angle`` is enabled, preserving
    concavities and background holes. Explicit rectangular regions use ``angle`` and a shared grid.
    Scene detections are context only and are never painted.
    Each line's deviation comes from its global stripe and the seed alone (not from candidate order, colour
    or nominal thickness); new bounds only rescale it, so preview, apply and export always agree.
    """
    color = _validated(options)
    if width < 1 or height < 1:
        raise ValueError("Размер изображения должен быть положительным")
    grow = options.size_scale * (1 + 2 * options.padding)
    bars: list[Bar] = []
    for detection in detections:
        if detection.kind == "act" or not _passes(detection, options):
            continue
        if detection.outline:
            points = np.concatenate([np.asarray(contour, np.float32)
                                     for contour in (detection.outline,) + detection.islands])
            angle = options.angle
            if options.auto_angle:
                box = cv2.boxPoints(cv2.minAreaRect(points))
                edges = np.roll(box, -1, axis=0) - box
                direction = edges[np.argmax(np.sum(edges * edges, axis=1))]
                angle = math.degrees(math.atan2(float(direction[1]), float(direction[0]))) % 180 - 90
            bars.extend(_lines_in_region(detection, width, height, options, color, angle))
            continue
        rect = detection.rect
        cx, cy = rect.x + rect.width / 2, rect.y + rect.height / 2
        half_w, half_h = rect.width * grow / 2, rect.height * grow / 2
        zone = (max(0., cx - half_w), max(0., cy - half_h),
                min(float(width), cx + half_w), min(float(height), cy + half_h))
        if zone[2] > zone[0] and zone[3] > zone[1]:
            bars.extend(_lines_in_zone(zone, options, color))
    return bars


def _validated(options: CensorOptions) -> str:
    """Raise ValueError for unusable options; return the bar colour in lower case."""
    color = str(options.color).lower()
    if color not in BAR_COLORS:
        raise ValueError(f"Недопустимый цвет полосы: {options.color!r} (допустимы #ffffff и #000000)")
    for name, valid, value in (
        ("порог уверенности", 0 < options.confidence <= 1, options.confidence),
        ("порог уверенности для сцен", 0 < options.act_confidence <= 1, options.act_confidence),
        ("отступ", 0 <= options.padding <= 2, options.padding),
        ("масштаб зоны", 0.1 <= options.size_scale <= 10, options.size_scale),
        ("толщина линии", isinstance(options.thickness, int) and 1 <= options.thickness <= MAX_THICKNESS,
         options.thickness),
        ("нижняя граница отклонения толщины",
         isinstance(options.variation_min, int) and 0 <= options.variation_min <= MAX_VARIATION,
         options.variation_min),
        ("верхняя граница отклонения толщины",
         isinstance(options.variation_max, int) and 0 <= options.variation_max <= MAX_VARIATION,
         options.variation_max),
        ("шаг линий (должен превышать толщину)", isinstance(options.spacing, int) and options.spacing > options.thickness, options.spacing),
        ("угол линий", math.isfinite(options.angle) and -180 <= options.angle <= 180, options.angle),
    ):
        if not valid:
            raise ValueError(f"Недопустимое значение параметра «{name}»: {value!r}")
    if options.variation_min > options.variation_max:
        raise ValueError(
            f"Нижняя граница отклонения толщины ({options.variation_min}) больше верхней ({options.variation_max})"
        )
    return color


def _passes(detection: Detection, options: CensorOptions) -> bool:
    """Whether ``options`` ask for this detection: kind switch and the threshold of its kind."""
    kind = detection.kind
    if kind not in _KINDS:
        raise ValueError(f"Неизвестный тип области: {kind!r}")
    if kind == "act":
        return options.include_acts and detection.confidence >= options.act_confidence
    if kind == "anus" and not options.include_anus:
        return False
    if kind == "nipple" and not options.include_nipples:
        return False
    return detection.confidence >= options.confidence


def _lines_in_zone(zone: tuple[float, float, float, float], options: CensorOptions, color: str) -> list[Bar]:
    """Intersect parallel centre-lines with an axis-aligned placement zone."""
    left, top, right, bottom = zone
    # Widest line any deviation may reach: a 1 px gap to the next stripe, at most a quarter of the zone.
    limit = min(options.spacing - 1, max(1, math.floor(min(right - left, bottom - top) / 4)))
    nominal = min(options.thickness, limit)
    angle = math.radians(options.angle)
    ux, uy = math.cos(angle), math.sin(angle)
    vx, vy = -uy, ux
    cx, cy = (left + right) / 2, (top + bottom) / 2
    middle = cx * vx + cy * vy
    radius = (abs(vx) * (right - left) + abs(vy) * (bottom - top)) / 2
    first = math.ceil((middle - radius + nominal / 2) / options.spacing)
    last = math.floor((middle + radius - nominal / 2) / options.spacing)
    offsets = [index * options.spacing for index in range(first, last + 1)] if first <= last else [middle]
    lines: list[Bar] = []
    for offset in offsets:
        px, py = cx + vx * (offset - middle), cy + vy * (offset - middle)
        low, high = -math.inf, math.inf
        for origin, direction, start, end in ((px, ux, left, right), (py, uy, top, bottom)):
            if abs(direction) < 1e-9:
                if not start <= origin <= end:
                    low, high = 1., 0.
                    break
            else:
                a, b = (start - origin) / direction, (end - origin) / direction
                low, high = max(low, min(a, b)), min(high, max(a, b))
        length = high - low
        if not math.isfinite(length) or length < 1:
            continue
        line_w = max(1, round(length))
        mx, my = px + ux * (low + high) / 2, py + uy * (low + high) / 2
        thickness = max(1, min(limit, options.thickness + _variation(options, offset)))
        lines.append(Bar(round(mx - line_w / 2), round(my - thickness / 2),
                         line_w, thickness, color, float(options.angle)))
    return lines


def _variation(options: CensorOptions, offset: float) -> int:
    """Signed change of the nominal thickness for the global stripe at ``offset`` (bounds inclusive).

    The side and the place inside the range come from the seed and the stripe alone, so new bounds
    rescale an existing pattern instead of re-rolling it, and colour or nominal thickness never move it.
    """
    key = hashlib.blake2s(f"{options.seed}:{offset:.17g}".encode("ascii"), digest_size=4).digest()
    roll = int.from_bytes(key, "little")
    span = options.variation_max - options.variation_min + 1
    magnitude = options.variation_min + (((roll >> 1) * span) >> 31)
    return -magnitude if roll & 1 else magnitude


def _open_heads(directory: Path) -> list[_Head]:
    """Load every bundled detector, or raise RuntimeError saying what is wrong."""
    missing = [name for name in _MODELS if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Не найдены файлы нейросетевых моделей: {', '.join(missing)} (папка «{directory}»). "
            "Восстановите содержимое папки assets или переустановите приложение."
        )
    try:
        import onnxruntime as ort  # imported here: it is slow, and the module must import without it
    except ImportError as exc:
        raise RuntimeError(
            f"Не удалось загрузить библиотеку onnxruntime: {exc}. Установите её командой «pip install onnxruntime»."
        ) from exc
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = max(1, min(8, (os.cpu_count() or 2) // 2))  # leave cores to the GUI
    session_options.log_severity_level = 3  # errors only
    return [_open_head(ort, directory / name, expected, session_options) for name, expected in _MODELS.items()]


def _open_head(ort: Any, path: Path, expected: tuple[str, ...], session_options: Any) -> _Head:
    """Open one model and check that its input and class names are what this module decodes."""
    try:
        session = ort.InferenceSession(str(path), sess_options=session_options, providers=["CPUExecutionProvider"])
    except Exception as exc:  # onnxruntime has its own error types for unreadable or damaged files
        raise RuntimeError(f"Не удалось загрузить модель «{path.name}»: {exc}") from exc
    shape = session.get_inputs()[0].shape  # [1, 3, 640, 640] or symbolic sides
    if len(shape) != 4 or any(isinstance(side, int) and side != _INPUT for side in shape[2:]):
        raise RuntimeError(f"Модель «{path.name}» рассчитана на вход {shape}, а нужен {_INPUT}×{_INPUT}.")
    try:
        names = ast.literal_eval(session.get_modelmeta().custom_metadata_map.get("names", ""))
    except (ValueError, SyntaxError):
        names = None
    if not isinstance(names, dict) or set(names) != set(range(len(names))):
        raise RuntimeError(f"В модели «{path.name}» не удалось прочитать имена классов.")
    absent = [name for name in expected if name not in names.values()]
    if absent:
        raise RuntimeError(f"Модель «{path.name}» не содержит нужных классов: {', '.join(absent)}.")
    classes = np.array([_LABELS.index(names[i]) if names[i] in _LABELS else -1 for i in range(len(names))], np.int16)
    return _Head(path.stem, session, session.get_inputs()[0].name, classes)


def _cutoffs(head: _Head, options: CensorOptions) -> np.ndarray:
    """Score cut-off per class id of ``head``; classes that are switched off or unknown never pass."""
    by_kind = {
        "genital": options.confidence,
        "anus": options.confidence if options.include_anus else math.inf,
        "nipple": options.confidence if options.include_nipples else math.inf,
        "act": options.act_confidence if options.include_acts else math.inf,
    }
    return np.array([math.inf if c < 0 else by_kind[_KIND_OF[_LABELS[c]]] for c in head.classes], np.float64)


def _views(width: int, height: int, tiles: bool) -> tuple[list[tuple[int, int, int, int]], bool]:
    """Windows ``(left, top, right, bottom)`` to analyse: the whole image first, then overlapping tiles.

    Tiles are squares of about ``_TILE_RATIO`` of the shorter side (at least ``_TILE_MIN`` px, never larger
    than the image) that overlap by at least ``_TILE_OVERLAP``. When whole image + tiles would exceed
    ``_MAX_VIEWS`` the tile edge grows (tiles become rectangles on thin images) and the flag is True.
    """
    views = [(0, 0, width, height)]
    if not tiles:
        return views, False
    nominal = edge = max(_TILE_MIN, round(_TILE_RATIO * min(width, height)))
    while True:
        tile_w, tile_h = min(width, edge), min(height, edge)
        xs, ys = _starts(width, tile_w), _starts(height, tile_h)
        if len(xs) * len(ys) < _MAX_VIEWS:
            break
        edge = math.ceil(edge * 1.1)
    if len(xs) * len(ys) > 1:  # a single tile would just repeat the whole image
        views += [(x, y, x + tile_w, y + tile_h) for y in ys for x in xs]
    return views, edge > nominal


def _starts(length: int, size: int) -> list[int]:
    """Offsets of windows of ``size`` px covering ``length`` px, evenly spread, overlapping by >= ``_TILE_OVERLAP``."""
    if size >= length:
        return [0]
    count = math.ceil((length - size) / (size * (1 - _TILE_OVERLAP))) + 1
    return [round(i * (length - size) / (count - 1)) for i in range(count)]


def _letterbox(crop: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
    """Fit an RGB ``crop`` into the square model input, centred on grey padding.

    Returns the uint8 canvas and ``(left, top, fit_w, fit_h)``: where the resized crop sits on it.
    Shrinking uses area averaging (no aliasing on screentones), enlarging bilinear interpolation.
    """
    height, width = crop.shape[:2]
    scale = _INPUT / max(height, width)
    fit_w, fit_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(crop, (fit_w, fit_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    left, top = (_INPUT - fit_w) // 2, (_INPUT - fit_h) // 2
    canvas = np.full((_INPUT, _INPUT, 3), _PAD_VALUE, np.uint8)
    canvas[top : top + fit_h, left : left + fit_w] = resized
    return canvas, left, top, fit_w, fit_h


def _run(head: _Head, blob: np.ndarray) -> np.ndarray:
    """One forward pass; ``blob`` is the ``(1, 3, 640, 640)`` float32 RGB input in 0..1."""
    try:
        return head.session.run(None, {head.input_name: blob})[0]
    except Exception as exc:  # onnxruntime raises its own error types
        raise RuntimeError(f"Ошибка работы модели «{head.name}»: {exc}") from exc


def _unturn(x0: Any, y0: Any, x1: Any, y1: Any, turn: int) -> tuple[Any, Any, Any, Any]:
    """Boxes found in the model input turned counter-clockwise by ``turn`` x 90 degrees -> unturned input."""
    size = _INPUT
    if turn == 1:
        return size - y1, x0, size - y0, x1
    if turn == 2:
        return size - x1, size - y1, size - x0, size - y0
    if turn == 3:
        return y0, size - x1, y1, size - x0
    return x0, y0, x1, y1


def _candidates(output: np.ndarray, head: _Head, cutoff: np.ndarray, turn: int, win: _Window) -> np.ndarray:
    """Decode one raw detector output into candidates in image pixels (dtype ``_CANDIDATE``).

    The output is YOLOv8/11 style ``(1, 4 + classes, anchors)``: centre and size in model pixels, then one
    sigmoid score per class (no objectness). An anchor counts for its best class only, and only when that
    score reaches the class's ``cutoff``. Boxes are unturned, unpadded, clipped to the window and shifted by
    its origin. Boxes flush with an inner window edge are dropped: they are cut off, and the overlapping
    neighbour sees that object whole.
    """
    pred = output[0] if output.ndim == 3 else output
    channels = 4 + len(head.classes)
    if pred.ndim != 2 or channels not in pred.shape:
        raise RuntimeError(f"Модель «{head.name}» вернула неожиданный формат выхода {tuple(output.shape)}.")
    if pred.shape[0] != channels:
        pred = pred.T  # (anchors, channels) layout
    scores = pred[4:]
    best = scores.argmax(axis=0)
    conf = scores.max(axis=0)
    keep = np.flatnonzero(conf >= cutoff[best])
    if keep.size > _MAX_CANDIDATES:
        keep = keep[np.argsort(-conf[keep], kind="stable")[:_MAX_CANDIDATES]]
    if keep.size == 0:
        return np.empty(0, _CANDIDATE)
    cx, cy, w, h = pred[:4, keep].astype(np.float64)
    x0, y0, x1, y1 = _unturn(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, turn)
    scale_x, scale_y = win.fit_w / win.width, win.fit_h / win.height
    x0 = np.clip((x0 - win.left) / scale_x, 0, win.width)
    x1 = np.clip((x1 - win.left) / scale_x, 0, win.width)
    y0 = np.clip((y0 - win.top) / scale_y, 0, win.height)
    y1 = np.clip((y1 - win.top) / scale_y, 0, win.height)
    inner_left, inner_top, inner_right, inner_bottom = win.inner
    margin_x, margin_y = _EDGE_MARGIN / scale_x, _EDGE_MARGIN / scale_y
    cut = (
        (inner_left & (x0 <= margin_x))
        | (inner_right & (x1 >= win.width - margin_x))
        | (inner_top & (y0 <= margin_y))
        | (inner_bottom & (y1 >= win.height - margin_y))
    )
    ok = (x1 > x0) & (y1 > y0) & ~cut
    found = np.empty(int(ok.sum()), _CANDIDATE)
    found["box"] = np.column_stack((x0[ok] + win.x, y0[ok] + win.y, x1[ok] + win.x, y1[ok] + win.y))
    found["score"] = conf[keep][ok]
    found["label"] = head.classes[best[keep]][ok]
    return found


def _dedupe(found: np.ndarray) -> np.ndarray:
    """Indices of the ``found`` candidates that survive duplicate removal.

    Boxes are visited best score first with deterministic spatial tie breaks. Only the same anatomical
    class (including model synonyms) can suppress another prediction. A weaker contained duplicate is
    removed, but a small local detail must not suppress the whole anatomical object.
    """
    x0, y0, x1, y1 = found["box"].T
    area = (x1 - x0) * (y1 - y0)
    kind = _LABEL_OBJECT[found["label"]]
    order = np.lexsort((found["label"], y1, x1, y0, x0, -found["score"]))  # the last key sorts first
    alive = np.ones(len(found), dtype=bool)
    kept: list[int] = []
    for i in order.tolist():
        if not alive[i]:
            continue
        kept.append(i)
        overlap_w = np.minimum(x1, x1[i]) - np.maximum(x0, x0[i])
        overlap_h = np.minimum(y1, y1[i]) - np.maximum(y0, y0[i])
        inter = np.clip(overlap_w, 0, None) * np.clip(overlap_h, 0, None)
        same = (inter >= _DUP_IOU * (area + area[i] - inter)) | (inter >= _DUP_INSIDE * area)
        alive &= ~((kind == kind[i]) & same)
    return np.array(kept, dtype=np.intp)


def _detections(found: np.ndarray, width: int, height: int) -> list[Detection]:
    """Turn merged candidates into integer boxes (rounded outwards, so nothing shrinks) in reading order."""
    detections: list[Detection] = []
    for (x0, y0, x1, y1), score, label in zip(found["box"].tolist(), found["score"].tolist(), found["label"].tolist()):
        left, top = max(0, math.floor(x0)), max(0, math.floor(y0))
        right, bottom = min(width, math.ceil(x1)), min(height, math.ceil(y1))
        if right > left and bottom > top:
            rect = Rect(left, top, right - left, bottom - top)
            detections.append(Detection(_KIND_OF[_LABELS[label]], _LABELS[label], score, rect))
    detections.sort(key=lambda d: (d.rect.y, d.rect.x, d.rect.height, d.rect.width, d.kind, d.label, -d.confidence))
    return detections


def _lines_in_region(region: Detection, width: int, height: int, options: CensorOptions,
                     color: str, angle: float) -> list[Bar]:
    """Fit whole rectangular strips inside the actual foreground, without crossing background holes."""
    contours = [np.asarray(contour, np.float32) for contour in (region.outline,) + region.islands]
    points = np.concatenate(contours)
    centre = np.asarray(cv2.minAreaRect(points)[0])
    grow = options.size_scale * (1 + 2 * options.padding)
    contours = [(contour-centre)*grow+centre for contour in contours]
    holes = [(np.asarray(hole,np.float32)-centre)*grow+centre for hole in region.holes]
    points = np.concatenate(contours)
    lo = np.maximum(np.floor(points.min(axis=0)).astype(int)-1,0)
    hi = np.minimum(np.ceil(points.max(axis=0)).astype(int)+2,(width,height))
    mask_w,mask_h = int(hi[0]-lo[0]),int(hi[1]-lo[1])
    if mask_w < 1 or mask_h < 1:
        return []
    mask = np.zeros((mask_h,mask_w),np.uint8)
    for contour in contours:
        cv2.fillPoly(mask,[np.rint((contour-lo)*4).astype(np.int32)],255,shift=2)
    for hole in holes:
        cv2.fillPoly(mask,[np.rint((hole-lo)*4).astype(np.int32)],0,shift=2)
    mask = cv2.erode(mask,np.ones((3,3),np.uint8))
    matrix = cv2.getRotationMatrix2D((mask_w/2,mask_h/2),angle,1.)
    cosine,sine = abs(matrix[0,0]),abs(matrix[0,1])
    rotated_w = math.ceil(mask_w*cosine+mask_h*sine)
    rotated_h = math.ceil(mask_h*cosine+mask_w*sine)
    matrix[0,2] += (rotated_w-mask_w)/2
    matrix[1,2] += (rotated_h-mask_h)/2
    rotated = cv2.warpAffine(mask,matrix,(rotated_w,rotated_h),flags=cv2.INTER_NEAREST)
    ys,xs = np.nonzero(rotated)
    if not len(ys):
        return []
    radians = math.radians(angle)
    normal = np.asarray((-math.sin(radians),math.cos(radians)))
    base = float(normal @ lo - matrix[1,2])
    start,stop = int(ys.min()),int(ys.max())+1
    limit = min(options.spacing-1,max(1,math.floor(min(int(xs.max()-xs.min()+1),stop-start)/4)))
    nominal = min(options.thickness,limit)
    first = math.ceil((start+nominal/2+base)/options.spacing)
    last = math.floor((stop-nominal/2+base)/options.spacing)
    offsets = [index*options.spacing for index in range(first,last+1)] if first <= last else [(start+stop)/2+base]
    inverse = cv2.invertAffineTransform(matrix)
    bars = []
    for offset in offsets:
        row = offset-base
        thickness = max(1,min(limit,options.thickness+_variation(options,offset)))
        top,bottom = max(0,math.floor(row-thickness/2)),min(rotated_h,math.ceil(row+thickness/2))
        if bottom <= top:
            continue
        columns = np.all(rotated[top:bottom] != 0,axis=0)
        edges = np.flatnonzero(np.diff(np.pad(columns.astype(np.int8),(1,1))))
        runs = edges.reshape(-1,2)
        if not len(runs):
            continue
        left,right = max(runs,key=lambda pair:int(pair[1]-pair[0]))
        length = int(right-left)-2
        if length < max(2,thickness):
            continue
        midpoint = inverse @ np.asarray(((left+right)/2,row,1.)) + lo
        bars.append(Bar(round(float(midpoint[0])-length/2),round(float(midpoint[1])-thickness/2),
                        length,thickness,color,angle))
    return bars
