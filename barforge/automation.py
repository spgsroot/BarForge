"""Local, metadata-first crop/censor preparation and incremental atomic dataset export."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from collections.abc import Callable, Sequence
import uuid

import cv2
import numpy as np

from .i18n import tr
from .autocensor import CensorOptions, Detection, LocalCensorEngine, CancelledError, _passes, bars_from_detections
from .imaging import load_rgb
from .models import Document, Rect, Session, bar_polygon, render_caption_template
from .storage import export_pair, save_session


@dataclass(frozen=True)
class CropOptions:
    mode: str = "context"
    context: float = 1.35
    min_side: int = 128
    aspect_ratio: float = 0.0  # 0 keeps the anatomical region's own proportions
    max_crops: int = 8
    include_page: bool = False


@dataclass
class PageAnalysis:
    source: str
    size: tuple[int, int]
    detections: tuple[Detection, ...]
    model_names: tuple[str, ...]
    views: int
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class CropCandidate:
    document: Document
    detection_indices: tuple[int, ...]
    score: float
    reason: str


@dataclass
class DatasetExportResult:
    documents: list[Document]
    session_path: str
    errors: list[tuple[str, str]]
    cancelled: bool


def analyze_source(source: str, engine: LocalCensorEngine, options: CensorOptions, *,
                   progress: Callable[[int, int], None] | None = None,
                   cancelled: Callable[[], bool] | None = None) -> PageAnalysis:
    """Read one oriented page; retain detections at 5% so review does not need another inference."""
    if cancelled is not None and cancelled():
        raise CancelledError()
    image = load_rgb(source)
    try:
        result = engine.analyze(image, replace(options, confidence=.05,
                               include_nipples=True, include_anus=True), progress=progress, cancelled=cancelled)
        if cancelled is not None and cancelled():
            raise CancelledError()
        return PageAnalysis(str(Path(source).resolve()), image.size, tuple(result.detections),
                            result.model_names, result.views, result.elapsed_seconds, list(result.warnings))
    finally:
        image.close()


def _validate_crops(options: CropOptions) -> None:
    if options.mode not in ("context", "page"):
        raise ValueError(tr("Режим кропа должен быть «context» или «page»"))
    if not math.isfinite(options.context) or not 1 <= options.context <= 20:
        raise ValueError(tr("Множитель контекста должен быть от 1 до 20"))
    if not math.isfinite(options.aspect_ratio) or not (options.aspect_ratio == 0 or .1 <= options.aspect_ratio <= 10):
        raise ValueError(tr("Соотношение сторон: 0 (по области) либо от 0.1 до 10"))
    for name, value, maximum in ((tr("Минимальная сторона"), options.min_side, 100000),
                                  (tr("Число кропов"), options.max_crops, 100)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(tr("{name}: ожидается целое число от 1 до {maximum}", name=name, maximum=maximum))


def _intersection(a: Rect, b: Rect) -> int:
    return max(0, min(a.x+a.width, b.x+b.width)-max(a.x,b.x)) * max(0, min(a.y+a.height,b.y+b.height)-max(a.y,b.y))


def _similar(a: Rect, b: Rect) -> bool:
    intersection = _intersection(a, b)
    return intersection / (a.width*a.height+b.width*b.height-intersection) >= .75


def _context_crop(rect: Rect, width: int, height: int, options: CropOptions) -> Rect | None:
    left, top = max(0, rect.x), max(0, rect.y)
    right, bottom = min(width, rect.x+rect.width), min(height, rect.y+rect.height)
    if right <= left or bottom <= top:
        return None
    ratio = options.aspect_ratio
    if ratio == 0:
        crop_w = min(width, max(options.min_side, math.ceil((right-left)*options.context)))
        crop_h = min(height, max(options.min_side, math.ceil((bottom-top)*options.context)))
    else:
        wanted_w = max(options.min_side, (right-left)*options.context,
                       options.min_side*ratio, (bottom-top)*options.context*ratio)
        wanted_h = wanted_w / ratio
        scale = min(1., width/wanted_w, height/wanted_h)
        # Never cut an anchor to enforce an aspect ratio or leave the source image.
        crop_w = min(width, max(right-left, math.ceil(wanted_w*scale)))
        crop_h = min(height, max(bottom-top, math.ceil(wanted_h*scale)))
    x = max(0, min(width-crop_w, round((left+right-crop_w)/2)))
    y = max(0, min(height-crop_h, round((top+bottom-crop_h)/2)))
    return Rect(x, y, crop_w, crop_h)


def prepare_page(page: PageAnalysis, censor: CensorOptions, crops: CropOptions,
                 caption_template: str) -> list[CropCandidate]:
    """Propose editable crops from cached detections; bars keep a single source-anchored stripe grid."""
    _validate_crops(crops)
    width, height = page.size
    admitted = [(index, detection) for index, detection in enumerate(page.detections)
                if detection.kind != "act" and _passes(detection, censor)]
    # The existing generator performs all censor option validation, even for an empty page.
    bars = bars_from_detections((d for _, d in admitted), width, height, censor)
    if not admitted or not bars:
        return []
    full = Rect(0, 0, width, height)
    windows: list[tuple[Rect, str]] = []
    if crops.mode == "page":
        windows.append((full, tr("Вся страница")))
    else:
        anchors = list(admitted)
        anchors.sort(key=lambda pair: (-pair[1].confidence, pair[1].rect.y, pair[1].rect.x, pair[0]))
        limit = crops.max_crops - int(crops.include_page)
        for _, detection in anchors:
            if len(windows) >= limit:
                break
            crop = _context_crop(detection.rect, width, height, crops)
            if crop is not None and not any(_similar(crop, previous) for previous, _ in windows):
                windows.append((crop, tr("Контекст области")))
        if crops.include_page:
            # A whole-page request is explicit; suppress only an exact duplicate context crop.
            windows = [(crop, reason) for crop, reason in windows if crop != full]
            windows.append((full, tr("Вся страница")))
    polygons = [np.asarray(bar_polygon(bar), dtype=np.float32) for bar in bars]
    candidates = []
    for crop, reason in sorted(windows, key=lambda item: (item[0].y, item[0].x, item[0].width, item[0].height)):
        crop_polygon = np.asarray(((crop.x,crop.y), (crop.x+crop.width,crop.y),
                                   (crop.x+crop.width,crop.y+crop.height), (crop.x,crop.y+crop.height)), dtype=np.float32)
        visible = [copy.copy(bar) for bar, polygon in zip(bars, polygons)
                   if cv2.intersectConvexConvex(polygon, crop_polygon)[0] > 0]
        indices = tuple(index for index, detection in admitted if _intersection(crop, detection.rect) > 0)
        if not visible or not indices:
            continue
        document = Document(page.source, visible, copy.copy(crop), render_caption_template(caption_template, censor.color))
        candidates.append(CropCandidate(document, indices, max(page.detections[index].confidence for index in indices), reason))
    return candidates


def export_dataset(documents: Sequence[Document], output_dir: str | Path, *, source_dir: str = "",
                   progress: Callable[[int, int], None] | None = None,
                   cancelled: Callable[[], bool] | None = None, upscaler=None) -> DatasetExportResult:
    """Export private snapshots, checkpointing each completed pair; cancellation never undoes completed work.

    Every supplied snapshot becomes a NEW pair. The review UI omits unchanged exports before calling.
    Each triplet uses storage.export_pair's existing atomic publication and conflict handling.
    """
    root = Path(output_dir).resolve()
    snapshots = copy.deepcopy(list(documents))
    session_path = root / f"batch-{uuid.uuid4().hex[:12]}.barforge.json"
    result = DatasetExportResult([], str(session_path), [], False)
    session = Session(source_dir=source_dir, output_dir=str(root))
    if progress is not None:
        progress(0, len(snapshots))
    if cancelled is not None and cancelled():
        result.cancelled = True
        return result
    try:
        # Establish the checkpoint before writing any pair, so permission errors cannot hide exports.
        save_session(session, session_path)
    except (OSError, ValueError) as exc:
        result.errors.append((str(root), str(exc)))
        return result
    for index, document in enumerate(snapshots):
        if cancelled is not None and cancelled():
            result.cancelled = True
            break
        document.export_name = None
        document.exported_signature = None
        try:
            def upscale_progress(done, total, text="", index=index):
                if progress is not None:
                    progress(index+done/max(1, total), len(snapshots))
            export_pair(document, root, upscaler=upscaler, progress=upscale_progress, cancelled=cancelled)
        except Exception as exc:
            from .upscale import UpscaleCancelled
            if isinstance(exc, UpscaleCancelled):
                result.cancelled = True
                break
            result.errors.append((document.source, str(exc)))
            if isinstance(exc, RuntimeError):
                break
        else:
            result.documents.append(document)
            session.documents.append(document)
            try:
                save_session(session, session_path)
            except (OSError, ValueError) as exc:
                # Pair remains valid and is returned for adoption/recovery; stop rather than overstate durability.
                result.errors.append((str(session_path), tr("Пара {value} сохранена, но сессия не обновлена: {exc}", value=document.export_name, exc=exc)))
                break
        if progress is not None:
            progress(index+1, len(snapshots))
    return result
