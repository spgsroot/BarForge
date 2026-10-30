"""Persistence: image discovery, versioned session files and safe dataset export.

Nothing here keeps pixels: documents only carry paths and markup, so a session
with hundreds of sources stays small. Failures surface as ``OSError`` (disk or
decoding problems), ``ValueError`` (invalid content) and ``FileExistsError``
(an export would overwrite a pair); the messages are written for the end user.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import os
import re
import uuid
from pathlib import Path

from PIL import Image

from .i18n import tr
from .imaging import render_pair
from .models import BAR_COLORS, DEFAULT_CAPTION, Bar, BrushStroke, Document, Rect, Session

SESSION_FORMAT = "barforge-session"
SESSION_VERSION = 2
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})

_COORD_LIMIT = 1 << 20  # sanity bound for pixel values read from a session file
_EXPORT_NAME = re.compile(r"[0-9]{1,9}")  # digits only: the name becomes part of file paths
_MAX_EXPORT_NUMBER = 999_999_999
_SIGNATURE = re.compile(r"[0-9a-f]{64}")
_DIGIT_RUNS = re.compile(r"([0-9]+)")


class ExportConflictError(FileExistsError):
    """Files of the pair already exist and overwriting was not requested."""


# --------------------------------------------------------------------------- discovery


def discover_images(folder: Path) -> list[Document]:
    """List PNG/JPEG/WebP files directly inside ``folder`` as pixel-free documents, naturally sorted.

    Sub-folders and hidden (dot) files are ignored; ``2.png`` sorts before ``10.png``.
    Sources are absolute paths.
    """
    root = os.path.abspath(folder)
    try:
        with os.scandir(root) as entries:
            names = [
                entry.name
                for entry in entries
                if not entry.name.startswith(".")
                and os.path.splitext(entry.name)[1].lower() in IMAGE_SUFFIXES
                and entry.is_file()
            ]
    except OSError as exc:
        raise _same_kind(exc, tr("Не удалось прочитать папку «{root}»: {exc}", root=root, exc=exc)) from exc
    names.sort(key=_natural_key)
    return [Document(source=os.path.join(root, name)) for name in names]


def _natural_key(name: str) -> tuple[list[str | int], str]:
    """Sort key comparing digit runs as numbers; the raw name breaks ties so the order is stable."""
    parts = _DIGIT_RUNS.split(name.casefold())  # text, digits, text, digits, ..., text
    return [int(part) if index % 2 else part for index, part in enumerate(parts)], name


# --------------------------------------------------------------------------- sessions


def save_session(session: Session, path: Path) -> None:
    """Atomically write ``session`` as versioned JSON; the previous file survives any failure.

    Source, source-folder and output-folder paths are stored relative to ``path``
    when a relative form exists. Only metadata is written: no pixels, no secrets.
    """
    target = os.path.abspath(path)
    base = os.path.dirname(target)
    payload = {
        "format": SESSION_FORMAT,
        "version": SESSION_VERSION,
        "source_dir": _store_path(session.source_dir, base),
        "output_dir": _store_path(session.output_dir, base),
        "current_index": int(session.current_index),
        "documents": [_encode_document(document, base) for document in session.documents],
    }
    _decode_session(payload, base)  # never write a file that load_session would reject
    data = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    try:
        os.makedirs(base, exist_ok=True)
        _write_atomic(Path(target), data)
    except OSError as exc:
        raise _same_kind(exc, tr("Не удалось сохранить сессию «{value}»: {exc}", value=os.path.basename(target), exc=exc)) from exc


def load_session(path: Path) -> Session:
    """Read and validate a session file; relative paths are resolved against its folder.

    No image is opened, so loading stays fast for hundreds of sources and
    missing sources are not an error here. Unknown extra keys are ignored.
    """
    target = os.path.abspath(path)
    name = os.path.basename(target)
    try:
        with open(target, "rb") as handle:
            text = handle.read().decode("utf-8-sig")
    except OSError as exc:
        raise _same_kind(exc, tr("Не удалось прочитать файл сессии «{name}»: {exc}", name=name, exc=exc)) from exc
    except UnicodeDecodeError as exc:
        raise ValueError(tr("Файл сессии «{name}» не является текстом UTF-8", name=name)) from exc
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        raise ValueError(tr("Файл сессии «{name}» повреждён: {exc}", name=name, exc=exc)) from exc
    return _decode_session(data, os.path.dirname(target))


def _reject_constant(constant: str) -> None:
    raise ValueError(tr("недопустимое значение {constant}", constant=constant))  # NaN / Infinity are not valid JSON


def _encode_document(document: Document, base: str) -> dict[str, object]:
    crop = document.crop
    return {
        "source": _store_path(document.source, base),
        "bars": [{**_geometry(bar), "color": str(bar.color).lower(), "angle": float(bar.angle)}
                 for bar in document.bars],
        "strokes": [{"points": [[int(x), int(y)] for x, y in stroke.points],
                     "thickness": int(stroke.thickness), "color": str(stroke.color).lower()}
                    for stroke in document.strokes],
        "upscale": bool(document.upscale),
        "caption_touched": bool(document.caption_touched),
        "crop": None if crop is None else _geometry(crop),
        "caption": document.caption,
        "export_name": document.export_name,
        "exported_signature": document.exported_signature,
    }


def _geometry(item: Rect | Bar) -> dict[str, int]:
    """Geometry as plain ints, coerced the same way as in Document.signature() and the renderer."""
    return {"x": int(item.x), "y": int(item.y), "width": int(item.width), "height": int(item.height)}


def _decode_session(data: object, base: str) -> Session:
    root = _mapping(data, tr("корень файла"))
    if root.get("format") != SESSION_FORMAT:
        raise ValueError(tr("Файл не является сессией BarForge"))
    version = root.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise _invalid("version", tr("ожидалось целое число"))
    if version > SESSION_VERSION:
        raise ValueError(
            tr("Сессия создана более новой версией BarForge (формат {version}, поддерживается {SESSION_VERSION})", version=version, SESSION_VERSION=SESSION_VERSION)
        )
    if version < 1:
        raise ValueError(tr("Неподдерживаемая версия формата сессии: {version}", version=version))
    documents = [
        _decode_document(item, f"documents[{index}]", base)
        for index, item in enumerate(_sequence(root.get("documents", []), "documents"))
    ]
    current = _integer(root.get("current_index", 0), "current_index", -_COORD_LIMIT, _COORD_LIMIT)
    return Session(
        documents=documents,
        source_dir=_decode_path(root.get("source_dir", ""), "source_dir", base),
        output_dir=_decode_path(root.get("output_dir", ""), "output_dir", base),
        current_index=max(0, min(current, len(documents) - 1)),
    )


def _decode_document(item: object, where: str, base: str) -> Document:
    entry = _mapping(item, where)
    bars = [
        _decode_bar(bar, f"{where}.bars[{index}]")
        for index, bar in enumerate(_sequence(entry.get("bars", []), f"{where}.bars"))
    ]
    strokes = [_decode_stroke(stroke, f"{where}.strokes[{index}]")
               for index, stroke in enumerate(_sequence(entry.get("strokes", []), f"{where}.strokes"))]
    upscale, touched = entry.get("upscale", False), entry.get("caption_touched", True)
    if not isinstance(upscale, bool) or not isinstance(touched, bool):
        raise _invalid(where, tr("upscale и caption_touched должны быть логическими значениями"))
    raw_crop = entry.get("crop")
    export_name = entry.get("export_name")
    if export_name is not None and not _EXPORT_NAME.fullmatch(_text(export_name, f"{where}.export_name")):
        raise _invalid(f"{where}.export_name", tr("ожидается номер из 1–9 цифр, например 001"))
    signature = entry.get("exported_signature")
    if signature is not None and not _SIGNATURE.fullmatch(_text(signature, f"{where}.exported_signature")):
        raise _invalid(f"{where}.exported_signature", tr("ожидается SHA-256: 64 шестнадцатеричных символа"))
    return Document(
        source=_decode_path(entry.get("source"), f"{where}.source", base, required=True),
        bars=bars,
        crop=None if raw_crop is None else _decode_rect(_mapping(raw_crop, f"{where}.crop"), f"{where}.crop"),
        caption=_text(entry.get("caption", DEFAULT_CAPTION), f"{where}.caption", allow_empty=True),
        export_name=export_name,
        # a signature without an export name cannot be honoured: treat the document as not exported
        exported_signature=signature if export_name is not None else None,
        strokes=strokes,
        upscale=upscale,
        caption_touched=touched,
    )


def _decode_bar(item: object, where: str) -> Bar:
    entry = _mapping(item, where)
    rect = _decode_rect(entry, where)
    color = _text(entry.get("color", "#ffffff"), f"{where}.color").lower()
    if color not in BAR_COLORS:
        raise _invalid(f"{where}.color", tr("допустимы только #ffffff и #000000"))
    angle = entry.get("angle", 0.)
    if (isinstance(angle, bool) or not isinstance(angle, (int, float))
            or not math.isfinite(angle) or not -180 <= angle <= 180):
        raise _invalid(f"{where}.angle", tr("ожидается конечное число от -180 до 180"))
    return Bar(rect.x, rect.y, rect.width, rect.height, color, float(angle))


def _decode_stroke(item: object, where: str) -> BrushStroke:
    entry = _mapping(item, where)
    raw_points = _sequence(entry.get("points"), f"{where}.points")
    if not raw_points or len(raw_points) > _COORD_LIMIT:
        raise _invalid(f"{where}.points", tr("ожидается от 1 до {_COORD_LIMIT} точек", _COORD_LIMIT=_COORD_LIMIT))
    points = []
    for index, raw in enumerate(raw_points):
        pair = _sequence(raw, f"{where}.points[{index}]")
        if len(pair) != 2:
            raise _invalid(f"{where}.points[{index}]", tr("ожидались две координаты"))
        points.append(tuple(_integer(value, f"{where}.points[{index}]", -_COORD_LIMIT, _COORD_LIMIT)
                            for value in pair))
    thickness = _integer(entry.get("thickness"), f"{where}.thickness", 1, 8192)
    color = _text(entry.get("color", "#ffffff"), f"{where}.color").lower()
    if color not in BAR_COLORS:
        raise _invalid(f"{where}.color", tr("допустимы только #ffffff и #000000"))
    return BrushStroke(tuple(points), thickness, color)


def _decode_rect(entry: dict, where: str) -> Rect:
    return Rect(
        x=_integer(entry.get("x"), f"{where}.x", -_COORD_LIMIT, _COORD_LIMIT),
        y=_integer(entry.get("y"), f"{where}.y", -_COORD_LIMIT, _COORD_LIMIT),
        width=_integer(entry.get("width"), f"{where}.width", 1, _COORD_LIMIT),
        height=_integer(entry.get("height"), f"{where}.height", 1, _COORD_LIMIT),
    )


def _invalid(where: str, problem: str) -> ValueError:
    return ValueError(tr("Некорректная сессия: {where} — {problem}", where=where, problem=problem))


def _mapping(value: object, where: str) -> dict:
    if not isinstance(value, dict):
        raise _invalid(where, tr("ожидался объект"))
    return value


def _sequence(value: object, where: str) -> list:
    if not isinstance(value, list):
        raise _invalid(where, tr("ожидался список"))
    return value


def _text(value: object, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or not (value or allow_empty):
        raise _invalid(where, tr("ожидалась строка") if allow_empty else tr("ожидалась непустая строка"))
    if "\x00" in value:
        raise _invalid(where, tr("строка содержит нулевой символ"))
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise _invalid(where, tr("строка содержит недопустимые символы Юникода")) from None
    return value


def _integer(value: object, where: str, low: int, high: int) -> int:
    if isinstance(value, float) and value.is_integer():  # 12.0 is accepted; NaN, inf and 12.5 are not
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid(where, tr("ожидалось конечное целое число"))
    if not low <= value <= high:
        raise _invalid(where, tr("значение вне допустимого диапазона {low}…{high}", low=low, high=high))
    return value


def _store_path(path: str, base: str) -> str:
    """Path as written to a session file: relative to ``base`` when possible, always with forward slashes."""
    if not path:
        return ""
    absolute = os.path.abspath(path)
    try:
        return Path(os.path.relpath(absolute, base)).as_posix()
    except ValueError:  # another drive: only the absolute form exists
        return Path(absolute).as_posix()


def _decode_path(value: object, where: str, base: str, *, required: bool = False) -> str:
    stored = _text(value, where, allow_empty=not required)
    return os.path.abspath(os.path.join(base, stored)) if stored else ""


# --------------------------------------------------------------------------- export


def export_pair(document: Document, output_dir: Path, *, overwrite: bool = False,
                upscaler=None, progress=None, cancelled=None) -> str:
    """Write ``clean/{name}.png``, ``clean/{name}.txt`` and ``censored/{name}.png``; return ``name``.

    A document that already has an ``export_name`` keeps it, and if any of its
    files exist this raises ``ExportConflictError`` (a ``FileExistsError``)
    unless ``overwrite`` is true. A document without a name gets the next
    sequential one (001, 002, ...), one above every number already present in
    ``clean/`` or ``censored/``, so existing files are never taken over.

    The three files are staged first and swapped in together; if anything fails,
    files that were replaced are restored. ``export_name`` and
    ``exported_signature`` are updated only after all files are in place.
    """
    root = Path(os.path.abspath(output_dir))
    name = document.export_name
    if name is not None:
        if not _EXPORT_NAME.fullmatch(name):
            raise ValueError(tr("Недопустимое имя экспорта: {name!r}", name=name))
        existing = [path for path in _pair_paths(root, name) if os.path.lexists(path)]
        if existing and not overwrite:
            raise ExportConflictError(_conflict_message(name, existing, root))
    snapshot = copy.deepcopy(document)  # sign and render one consistent state
    signature = snapshot.signature()
    try:
        caption = snapshot.caption.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(tr("Подпись содержит символы, которые нельзя сохранить в UTF-8")) from exc
    clean, censored = render_pair(snapshot, upscaler=upscaler, progress=progress, cancelled=cancelled)
    clean_png, censored_png = _png_bytes(clean), _png_bytes(censored)
    try:
        (root / "clean").mkdir(parents=True, exist_ok=True)
        (root / "censored").mkdir(parents=True, exist_ok=True)
        if name is None:
            name = _next_name(root / "clean", root / "censored")
        _publish(list(zip(_pair_paths(root, name), (clean_png, caption, censored_png))), overwrite)
    except ExportConflictError:
        raise
    except OSError as exc:
        raise _same_kind(exc, tr("Не удалось записать пару в «{root}»: {exc}", root=root, exc=exc)) from exc
    document.export_name = name
    document.exported_signature = signature
    return name


def _pair_paths(root: Path, name: str) -> list[Path]:
    return [root / "clean" / f"{name}.png", root / "clean" / f"{name}.txt", root / "censored" / f"{name}.png"]


def _conflict_message(name: str, existing: list[Path], root: Path) -> str:
    shown = ", ".join(path.relative_to(root).as_posix() for path in existing)
    return tr("Пара {name} уже существует ({shown}). Для замены нужно разрешить перезапись.", name=name, shown=shown)


def _next_name(*folders: Path) -> str:
    """Next sequential export name: one above the highest number used by any file in ``folders``."""
    highest = 0
    for folder in folders:
        with os.scandir(folder) as entries:
            for entry in entries:
                stem = os.path.splitext(entry.name)[0]
                if _EXPORT_NAME.fullmatch(stem):
                    highest = max(highest, int(stem))
    if highest >= _MAX_EXPORT_NUMBER:
        raise OSError(tr("Исчерпана нумерация пар: в папке датасета уже есть номер {highest}", highest=highest))
    return f"{highest + 1:03d}"


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _publish(files: list[tuple[Path, bytes]], overwrite: bool) -> None:
    """Put every ``(final path, content)`` in place, or leave the disk as it was.

    Contents go to temporary siblings first, so a failed write never touches
    existing files. Only then are existing files moved aside (``overwrite`` only)
    and the new ones renamed into place; any failure undoes the renames and
    restores the originals.
    """
    staged: list[tuple[Path, Path]] = []  # (temporary file, final path)
    backups: list[tuple[Path, Path]] = []  # (moved-aside original, final path)
    placed: list[Path] = []  # final paths that now hold new content
    try:
        for final, data in files:
            staged.append((_write_temp(final, data), final))
        for _, final in staged:
            if os.path.lexists(final):
                if not overwrite:
                    raise ExportConflictError(tr("Файл уже существует: {value}", value=final.name))
                backup = _scratch_path(final, "bak")
                os.replace(final, backup)
                backups.append((backup, final))
        for temp, final in staged:
            os.replace(temp, final)
            placed.append(final)
    except BaseException as error:
        for temp, _ in staged:
            _discard(temp)
        for final in placed:
            _discard(final)
        stuck = []
        for backup, final in reversed(backups):
            try:
                os.replace(backup, final)
            except OSError:
                stuck.append(str(backup))
        if stuck and isinstance(error, Exception):
            raise OSError(tr("Сбой записи; прежние файлы пары сохранены под именами: ") + ", ".join(stuck)) from error
        raise
    for backup, _ in backups:
        _discard(backup)


def _write_atomic(path: Path, data: bytes) -> None:
    temp = _write_temp(path, data, sync=True)
    try:
        os.replace(temp, path)
    except BaseException:
        _discard(temp)
        raise


def _write_temp(final: Path, data: bytes, *, sync: bool = False) -> Path:
    """Write ``data`` to a fresh ``*.tmp`` sibling of ``final`` and return its path."""
    temp = _scratch_path(final, "tmp")
    try:
        with open(temp, "xb") as handle:
            handle.write(data)
            if sync:
                handle.flush()
                os.fsync(handle.fileno())
    except BaseException:
        _discard(temp)
        raise
    return temp


def _scratch_path(path: Path, kind: str) -> Path:
    return path.with_name(f"{path.name}.{uuid.uuid4().hex[:10]}.{kind}")


def _discard(path: Path) -> None:
    with contextlib.suppress(OSError):
        os.remove(path)


def _same_kind(exc: OSError, message: str) -> OSError:
    """New error with a user-readable message that keeps the type of ``exc`` (a FileNotFoundError stays one)."""
    return type(exc)(message)
