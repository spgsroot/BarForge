"""Pixel pipeline: source loading, clean/censored rendering and the bar preset.

Pillow decodes (EXIF orientation, alpha flattening) and crops; OpenCV rasterises
the bars and scales 16-bit greyscale down to 8 bits, which Pillow would clip.
Source files are only ever read.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .i18n import tr
from .models import BAR_COLORS, Bar, BrushStroke, Document, Rect, bar_polygon

_WIDE_GRAY_MODES = frozenset({"I", "I;16", "I;16L", "I;16B"})
_ALPHA_MODES = frozenset({"RGBA", "LA", "PA", "RGBa", "La"})
_AUTO_BAR_WIDTH = 0.5  # share of the target area's width covered by each preset bar


def load_rgb(source: str | Path) -> Image.Image:
    """Decode ``source`` into an independent 8-bit RGB image.

    EXIF orientation is applied, so coordinates always refer to the picture as
    displayed, and transparency is flattened onto white. Any decoding problem is
    reported as ``OSError`` with a user-readable message.
    """
    name = Path(source).name
    try:
        with Image.open(source) as opened:
            opened.load()
            ImageOps.exif_transpose(opened, in_place=True)
            return _flatten_to_rgb(opened)
    except Exception as exc:  # Pillow raises OSError, SyntaxError, ValueError, DecompressionBombError, ...
        kind = type(exc) if isinstance(exc, OSError) else OSError  # FileNotFoundError and co. stay themselves
        raise kind(tr("Не удалось открыть изображение «{name}»: {exc}", name=name, exc=exc)) from exc


def render_pair(document: Document, *, upscaler=None, progress=None, cancelled=None) -> tuple[Image.Image, Image.Image]:
    """Return ``(clean, censored)`` for ``document``; the source file is not modified.

    Both images are the same crop of the oriented source (the whole image when there is no crop),
    so they always have identical size and alignment. ``censored`` additionally has bars and freehand
    strokes painted on. Geometry is in source coordinates and clipped to the image and crop.
    Out-of-bounds markup is skipped; a crop that misses the image entirely is an error. When
    ``document.upscale`` is true, both sides are independently processed by the same RealESRGAN model.
    """
    image = load_rgb(document.source)
    full = (0, 0, image.width, image.height)
    box = full
    if document.crop is not None:
        crop = document.crop
        box = _clip(crop.x, crop.y, crop.width, crop.height, full)
        if box is None:
            raise ValueError(tr("Область кадрирования пуста или лежит вне изображения"))
    left, top = box[0], box[1]
    clean = image if box == full else image.crop(box)
    pixels = np.array(clean)  # writable copy; ``clean`` itself stays untouched
    for bar in document.bars:
        fill = BAR_COLORS.get(str(bar.color).lower())
        if fill is None:
            raise ValueError(tr("Недопустимый цвет полосы: {value!r} (допустимы #ffffff и #000000)", value=bar.color))
        if bar.angle:
            polygon = np.asarray(bar_polygon(bar), dtype=np.float64)
            # Rasterize in a source-anchored ROI, so cropping never changes edge pixels.
            x0, y0 = np.maximum(np.floor(polygon.min(axis=0)).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(polygon.max(axis=0)).astype(int), image.size)
            ix0, iy0 = max(int(x0), left), max(int(y0), top)
            ix1, iy1 = min(int(x1), box[2]), min(int(y1), box[3])
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            mask = np.zeros((int(y1 - y0), int(x1 - x0)), dtype=np.uint8)
            points = np.rint((polygon - (x0, y0) - .5) * 256).astype(np.int32)
            cv2.fillConvexPoly(mask, points, 255, lineType=cv2.LINE_8, shift=8)
            visible = mask[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0].astype(bool)
            pixels[iy0 - top:iy1 - top, ix0 - left:ix1 - left][visible] = fill
            continue
        part = _clip(bar.x, bar.y, bar.width, bar.height, box)
        if part is None:
            continue
        x0, y0, x1, y1 = part
        # OpenCV rectangle corners are inclusive, our far edges are exclusive.
        cv2.rectangle(pixels, (x0 - left, y0 - top), (x1 - left - 1, y1 - top - 1), fill, thickness=cv2.FILLED)
    for stroke in document.strokes:
        _paint_stroke(pixels, stroke, box, image.size)
    control = Image.fromarray(pixels)
    if document.upscale:
        from .upscale import upscale_pair
        clean, control = upscale_pair(clean, control, engine=upscaler, progress=progress, cancelled=cancelled)
    return clean, control


def _paint_stroke(pixels: np.ndarray, stroke: BrushStroke, box: tuple[int, int, int, int],
                  image_size: tuple[int, int]) -> None:
    """Rasterize a round polyline in a small source-anchored ROI, clipped to the crop."""
    fill = BAR_COLORS.get(str(stroke.color).lower())
    if fill is None:
        raise ValueError(tr("Недопустимый цвет кисти: {value!r} (допустимы #ffffff и #000000)", value=stroke.color))
    if not stroke.points or stroke.thickness < 1:
        return
    points = np.asarray(stroke.points, dtype=np.int32)
    margin = stroke.thickness // 2 + 1
    x0, y0 = np.maximum(points.min(axis=0)-margin, 0)
    x1, y1 = np.minimum(points.max(axis=0)+margin+1, image_size)
    ix0, iy0 = max(int(x0), box[0]), max(int(y0), box[1])
    ix1, iy1 = min(int(x1), box[2]), min(int(y1), box[3])
    if ix0 >= ix1 or iy0 >= iy1:
        return
    mask = np.zeros((int(y1-y0), int(x1-x0)), np.uint8)
    local = points-(x0, y0)
    if len(local) == 1:
        cv2.circle(mask, tuple(local[0]), max(0, stroke.thickness//2), 255, thickness=cv2.FILLED, lineType=cv2.LINE_8)
    else:
        cv2.polylines(mask, [local.reshape(-1, 1, 2)], False, 255, thickness=stroke.thickness, lineType=cv2.LINE_8)
    visible = mask[iy0-y0:iy1-y0, ix0-x0:ix1-x0].astype(bool)
    pixels[iy0-box[1]:iy1-box[1], ix0-box[0]:ix1-box[0]][visible] = fill


def auto_bars(
    width: int,
    height: int,
    crop: Rect | None = None,
    *,
    count: int = 3,
    thickness: int = 24,
    color: str = "#ffffff",
) -> list[Bar]:
    """Preset of ``count`` horizontal bars stacked around the centre of the crop (or image).

    The bars are centred horizontally and cover half of the area's width; each is
    ``thickness`` tall with gaps of the same height. When the area is too small
    the thickness shrinks first and then the count, so every bar stays inside the
    area (the crop clipped to ``width`` x ``height``) with positive size.
    """
    if width < 1 or height < 1:
        raise ValueError(tr("Размер изображения должен быть положительным"))
    if count < 0 or thickness < 1:
        raise ValueError(tr("Количество полос не может быть отрицательным, а толщина должна быть не меньше 1"))
    fill = str(color).lower()
    if fill not in BAR_COLORS:
        raise ValueError(tr("Недопустимый цвет полосы: {color!r} (допустимы #ffffff и #000000)", color=color))
    area = (0, 0, width, height)
    if crop is not None:
        area = _clip(crop.x, crop.y, crop.width, crop.height, area)
        if area is None:
            raise ValueError(tr("Область кадрирования пуста или лежит вне изображения"))
    left, top, right, bottom = area
    span_w, span_h = right - left, bottom - top
    count = min(count, (span_h + 1) // 2)  # bars with one-pixel gaps (2 * count - 1 rows) must fit
    if count == 0:
        return []
    thickness = min(thickness, span_h // (2 * count - 1))
    pitch = 2 * thickness
    block = (count - 1) * pitch + thickness
    first_y = top + (span_h - block) // 2
    bar_w = max(1, min(span_w, round(span_w * _AUTO_BAR_WIDTH)))
    bar_x = left + (span_w - bar_w) // 2
    return [Bar(bar_x, first_y + index * pitch, bar_w, thickness, fill) for index in range(count)]


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    """Return a new 8-bit RGB image, compositing any transparency onto white."""
    if image.mode in _WIDE_GRAY_MODES:
        # Pillow clips 16-bit samples at 255 (the picture turns almost white); scale them instead.
        samples = np.asarray(image)
        if samples.dtype != np.uint16:  # 32-bit "I" or non-native byte order
            samples = np.clip(samples, 0, 65535).astype(np.uint16)
        gray = cv2.convertScaleAbs(samples, alpha=255 / 65535)
        return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    if image.mode in _ALPHA_MODES or "transparency" in image.info:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    return image.convert("RGB")


def _clip(
    x: int, y: int, width: int, height: int, box: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    """Intersect a rectangle with ``box`` (left, top, right, bottom; right/bottom exclusive); None if empty."""
    left, top, right, bottom = box
    x0, y0 = max(int(x), left), max(int(y), top)
    x1, y1 = min(int(x) + int(width), right), min(int(y) + int(height), bottom)
    return (x0, y0, x1, y1) if x0 < x1 and y0 < y1 else None
