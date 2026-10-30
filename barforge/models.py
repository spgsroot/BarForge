"""Plain data model shared by the GUI, the imaging pipeline and storage.

All geometry is in pixels of the full-resolution, EXIF-oriented source image:
``x``/``y`` is the inclusive top-left corner, ``x + width`` / ``y + height`` the
exclusive bottom-right corner. Nothing here touches Qt or pixel data.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

DEFAULT_CAPTION = "rustnomiconUncensor. Remove the white censor bars"

#: Allowed bar colours and the RGB triple each one is painted with.
BAR_COLORS: dict[str, tuple[int, int, int]] = {
    "#ffffff": (255, 255, 255),
    "#000000": (0, 0, 0),
}

#: Colour word that each bar colour stands for in a caption.
BAR_COLOR_NAMES: dict[str, str] = {"#ffffff": "white", "#000000": "black"}

#: Marker for the colour word inside a caption pattern.
COLOR_PLACEHOLDER = "{color}"

#: Prepared captions as ``(short label, pattern)``: the generic wording, four anatomical types and the
#: colourless wording. ``{color}`` in a pattern becomes the colour word of the bars.
CAPTION_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("Общий", "rustnomiconUncensor. Remove the {color} censor bars"),
    ("Пенис", "rustnomiconUncensor. Remove the {color} censor bars, penis, close-up"),
    ("Вагина", "rustnomiconUncensor. Remove the {color} censor bars, vagina, close-up"),
    ("Анус", "rustnomiconUncensor. Remove the {color} censor bars, anus, close-up"),
    ("Проникновение", "rustnomiconUncensor. Remove the {color} censor bars, penetration, penis penetrating vagina"),
    ("Без цвета", "rustnomiconUncensor. Remove the censor bars"),
)


def render_caption_template(template: str, color: str) -> str:
    """Replace the literal ``{color}`` of ``template`` with the colour word of ``color``.

    ``#ffffff`` / ``#000000`` (any case) give ``white`` / ``black``; any other value is inserted as given.
    Nothing else is interpreted, so other braces in hand-written text survive untouched.
    """
    return template.replace(COLOR_PLACEHOLDER, BAR_COLOR_NAMES.get(color.lower(), color))


def caption_template_for(caption: str) -> str | None:
    """Pattern of the prepared caption that ``caption`` exactly is, otherwise ``None``.

    The raw pattern and its rendering in either bar colour all match, so a prepared caption keeps its
    type while its colour changes; hand-edited text (even a single extra character) is not recognised.
    """
    for _label, template in CAPTION_TEMPLATES:
        if caption == template:
            return template
        if COLOR_PLACEHOLDER in template and any(
                caption == render_caption_template(template, color) for color in BAR_COLOR_NAMES):
            return template
    return None


@dataclass
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass
class Bar:
    x: int
    y: int
    width: int
    height: int
    color: str = "#ffffff"
    angle: float = 0.0  # clockwise degrees about the local rectangle centre


def bar_polygon(bar: Bar) -> list[tuple[float, float]]:
    """Outer-edge corners of a bar rotated clockwise in y-down image coordinates."""
    angle = float(bar.angle)
    if not math.isfinite(angle) or not -180 <= angle <= 180:
        raise ValueError("Угол линии должен быть конечным числом от -180 до 180 градусов")
    cx, cy = bar.x + bar.width / 2, bar.y + bar.height / 2
    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return [(cx + dx * cosine - dy * sine, cy + dx * sine + dy * cosine)
            for dx, dy in ((-bar.width / 2, -bar.height / 2), (bar.width / 2, -bar.height / 2),
                           (bar.width / 2, bar.height / 2), (-bar.width / 2, bar.height / 2))]


@dataclass
class BrushStroke:
    """One freehand round brush stroke in source-pixel coordinates; one undo/export unit."""

    points: tuple[tuple[int, int], ...]
    thickness: int
    color: str = "#ffffff"


@dataclass
class Document:
    """One source image with its editable markup and export bookkeeping."""

    source: str
    bars: list[Bar] = field(default_factory=list)
    crop: Rect | None = None
    caption: str = DEFAULT_CAPTION
    export_name: str | None = None
    exported_signature: str | None = None
    strokes: list[BrushStroke] = field(default_factory=list)
    upscale: bool = False
    caption_touched: bool = False

    def signature(self) -> str:
        """Deterministic fingerprint of everything that shapes the exported pair.

        Covers the source file name, caption, crop, x4 flag, ordered bars and freehand strokes
        (colour case-folded). ``caption_touched`` and export bookkeeping do not affect pixels.
        The source folder is left out, so the value survives moving a project and round-trips
        through a session file.
        """
        crop = self.crop
        payload = (
            str(self.source).replace("\\", "/").rsplit("/", 1)[-1],
            self.caption,
            None if crop is None else (int(crop.x), int(crop.y), int(crop.width), int(crop.height)),
            [(int(b.x), int(b.y), int(b.width), int(b.height), str(b.color).lower(), float(b.angle)) for b in self.bars],
        )
        if self.strokes or self.upscale:
            payload += (
                [(tuple((int(x), int(y)) for x, y in stroke.points), int(stroke.thickness),
                  str(stroke.color).lower()) for stroke in self.strokes],
                bool(self.upscale),
            )
        # ensure_ascii (the default) keeps the text pure ASCII, even for stray surrogates.
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def is_exported(self) -> bool:
        """True while the last successful export still matches the current markup."""
        return self.exported_signature is not None and self.exported_signature == self.signature()


@dataclass
class Session:
    documents: list[Document] = field(default_factory=list)
    source_dir: str = ""
    output_dir: str = ""
    current_index: int = 0
