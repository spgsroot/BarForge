"""Offline RealESRGAN_x4plus_anime_6B inference with bounded-memory overlapping tiles.

The bundled ONNX graph is converted from the official v0.2.2.4 ``params_ema`` weights. Inputs and
outputs are RGB, 8-bit. The implementation follows the official right/bottom reflection pre-pad and
48-pixel tile overlap; no model service, PyTorch installation, or network request is needed. Successful
results are cached by exact RGB pixels and model fingerprint, in memory and as atomic PNG files.
"""
from __future__ import annotations

import contextlib
import hashlib
import math
import os
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

MODEL_NAME = "RealESRGAN_x4plus_anime_6B"
SCALE = 4
TILE = 128
TILE_PAD = 48  # larger than the official 10px default: materially reduces seams for RRDB's wide receptive field
PRE_PAD = 10
MAX_OUTPUT_PIXELS = 16_777_216
CACHE_VERSION = 1
CACHE_LIMIT_BYTES = 2 * 1024**3
MEMORY_CACHE_ITEMS = 4


def default_cache_dir() -> Path:
    """Per-user persistent cache; tests and portable installs may override it explicitly."""
    override = os.environ.get("BARFORGE_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home()/"AppData"/"Local"))/"BarForge"/"cache"/"realesrgan"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home()/".cache"))/"barforge"/"realesrgan"

Progress = Callable[[int, int], None]
Cancelled = Callable[[], bool]


class UpscaleCancelled(Exception):
    """Raised between tiles when the caller requested cancellation."""


class RealESRGAN:
    """One lazily loaded CPU ONNX session, safe to share between sequential worker jobs."""

    def __init__(self, model_path: str | Path | None = None, *, cache_dir: str | Path | None = None,
                 cache_limit: int = CACHE_LIMIT_BYTES):
        self.model_path = Path(model_path) if model_path is not None else Path(__file__).parent / "assets" / f"{MODEL_NAME}.onnx"
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self.cache_limit = max(0, int(cache_limit))
        self.memory_hits = 0
        self.disk_hits = 0
        self.inference_calls = 0
        self._fingerprint = None
        self._memory_cache: OrderedDict[str, Image.Image] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._session = None
        self._lock = threading.Lock()

    def _ready(self):
        with self._lock:
            if self._session is None:
                if not self.model_path.is_file():
                    raise RuntimeError(f"Модель {MODEL_NAME} не найдена: {self.model_path}")
                try:
                    import onnxruntime as ort
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
                    options.inter_op_num_threads = 1
                    session = ort.InferenceSession(str(self.model_path), sess_options=options,
                                                   providers=["CPUExecutionProvider"])
                except Exception as exc:
                    raise RuntimeError(f"Не удалось загрузить {MODEL_NAME}: {exc}") from exc
                inputs = session.get_inputs()
                outputs = session.get_outputs()
                if len(inputs) != 1 or len(outputs) != 1:
                    raise RuntimeError(f"Некорректный интерфейс модели {MODEL_NAME}")
                self._session = session
            return self._session

    def _model_hash(self) -> str:
        with self._lock:
            if self._fingerprint is None:
                digest = hashlib.sha256()
                try:
                    with self.model_path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024*1024), b""):
                            digest.update(chunk)
                except OSError as exc:
                    raise RuntimeError(f"Не удалось прочитать модель {MODEL_NAME}: {exc}") from exc
                self._fingerprint = digest.hexdigest()
            return self._fingerprint

    def _cache_key(self, image: Image.Image) -> str:
        header = f"{CACHE_VERSION}:{self._model_hash()}:{TILE}:{TILE_PAD}:{PRE_PAD}:{image.width}x{image.height}:".encode("ascii")
        return hashlib.sha256(header+image.tobytes()).hexdigest()

    def _remember(self, key: str, image: Image.Image, *, memory_hit: bool = False,
                  disk_hit: bool = False) -> Image.Image:
        with self._cache_lock:
            self._memory_cache[key] = image.copy()
            self._memory_cache.move_to_end(key)
            while len(self._memory_cache) > MEMORY_CACHE_ITEMS:
                self._memory_cache.popitem(last=False)
            self.memory_hits += int(memory_hit)
            self.disk_hits += int(disk_hit)
        return image.copy()

    def _cache_get(self, key: str, size: tuple[int, int]) -> Image.Image | None:
        with self._cache_lock:
            cached = self._memory_cache.get(key)
            if cached is not None:
                self._memory_cache.move_to_end(key)
                self.memory_hits += 1
                return cached.copy()
        path = self.cache_dir/f"{key}.png"
        if not path.is_file():
            return None
        try:
            with Image.open(path) as opened:
                opened.load()
                if opened.size != size or opened.mode != "RGB":
                    raise ValueError("cache dimensions or mode differ")
                result = opened.copy()
            os.utime(path, None)
        except (OSError, ValueError):
            with contextlib.suppress(OSError):
                path.unlink()
            return None
        return self._remember(key, result, disk_hit=True)

    def _cache_put(self, key: str, image: Image.Image) -> None:
        path = self.cache_dir/f"{key}.png"
        temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            image.save(temp, format="PNG")
            os.replace(temp, path)
        except OSError:
            with contextlib.suppress(OSError):
                temp.unlink()
            return  # a cache failure must not discard the successfully computed pair
        self._remember(key, image)
        entries = []
        with contextlib.suppress(OSError):
            for candidate in self.cache_dir.glob("*.png"):
                with contextlib.suppress(OSError):
                    stat = candidate.stat()
                    entries.append((stat.st_atime, stat.st_size, candidate))
        total = sum(size for _, size, _ in entries)
        for _, size, candidate in sorted(entries):
            if total <= self.cache_limit:
                break
            with contextlib.suppress(OSError):
                candidate.unlink()
            total -= size

    def upscale(self, image: Image.Image, *, progress: Progress | None = None,
                cancelled: Cancelled | None = None) -> Image.Image:
        """Return an exact-pixel-cached x4 image, computing only cache misses."""
        if image.width < 1 or image.height < 1:
            raise ValueError("Изображение для апскейла пустое")
        if image.width * image.height * SCALE * SCALE > MAX_OUTPUT_PIXELS:
            raise ValueError(
                f"Кроп слишком велик для x4: {image.width*SCALE}×{image.height*SCALE}. "
                f"Уменьшите его до {MAX_OUTPUT_PIXELS:,} выходных пикселей или меньше.".replace(",", " ")
            )
        if cancelled is not None and cancelled():
            raise UpscaleCancelled()
        rgb_image = image if image.mode == "RGB" else image.convert("RGB")
        width, height = rgb_image.size
        total = math.ceil((height+PRE_PAD)/TILE)*math.ceil((width+PRE_PAD)/TILE)
        if progress is not None:
            progress(0, total)
        key = self._cache_key(rgb_image)
        cached = self._cache_get(key, (width*SCALE, height*SCALE))
        if cached is not None:
            if progress is not None:
                progress(total, total)
            return cached
        # A second check after waiting prevents duplicate work within this engine.
        with self._inference_lock:
            cached = self._cache_get(key, (width*SCALE, height*SCALE))
            if cached is not None:
                if progress is not None:
                    progress(total, total)
                return cached
            result = self._infer(rgb_image, progress=progress, cancelled=cancelled)
            self._cache_put(key, result)
            return result

    def _infer(self, image: Image.Image, *, progress: Progress | None,
               cancelled: Cancelled | None) -> Image.Image:
        rgb = np.asarray(image, dtype=np.float32)/255.0
        height, width = rgb.shape[:2]
        # Official RealESRGANer.pre_process: reflect only the right and bottom edges by 10 pixels.
        rgb = cv2.copyMakeBorder(rgb, 0, PRE_PAD, 0, PRE_PAD, cv2.BORDER_REFLECT_101)
        padded_h, padded_w = rgb.shape[:2]
        output = np.empty((height*SCALE, width*SCALE, 3), np.uint8)
        count_x, count_y = math.ceil(padded_w/TILE), math.ceil(padded_h/TILE)
        session = self._ready()
        done = 0
        for top in range(0, padded_h, TILE):
            for left in range(0, padded_w, TILE):
                if cancelled is not None and cancelled():
                    raise UpscaleCancelled()
                right, bottom = min(left+TILE, padded_w), min(top+TILE, padded_h)
                tile = rgb[max(0, top-TILE_PAD):min(padded_h, bottom+TILE_PAD),
                           max(0, left-TILE_PAD):min(padded_w, right+TILE_PAD)]
                tensor = np.ascontiguousarray(tile.transpose(2, 0, 1)[None])
                prediction = session.run(None, {"input": tensor})[0][0].transpose(1, 2, 0)
                self.inference_calls += 1
                x0, y0 = left*SCALE, top*SCALE
                x1, y1 = min(right*SCALE, output.shape[1]), min(bottom*SCALE, output.shape[0])
                if x0 >= x1 or y0 >= y1:
                    done += 1
                    continue
                offset_x = (left-max(0, left-TILE_PAD))*SCALE
                offset_y = (top-max(0, top-TILE_PAD))*SCALE
                part = prediction[offset_y:offset_y+y1-y0, offset_x:offset_x+x1-x0]
                output[y0:y1, x0:x1] = np.rint(np.clip(part, 0., 1.)*255.).astype(np.uint8)
                done += 1
                if progress is not None:
                    progress(done, count_x*count_y)
        return Image.fromarray(output, "RGB")


_shared: RealESRGAN | None = None
_shared_lock = threading.Lock()


def shared_upscaler() -> RealESRGAN:
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = RealESRGAN()
        return _shared


def upscale_pair(clean: Image.Image, censored: Image.Image, *, engine: RealESRGAN | None = None,
                 progress: Callable[[int, int, str], None] | None = None,
                 cancelled: Cancelled | None = None) -> tuple[Image.Image, Image.Image]:
    """Upscale both sides independently with identical geometry and progress/cancellation semantics."""
    upscaler = engine if engine is not None else shared_upscaler()
    if clean.size != censored.size:
        raise ValueError("Стороны пары имеют разные размеры")
    result = []
    for index, (name, image) in enumerate((("clean", clean), ("censored", censored))):
        if cancelled is not None and cancelled():
            raise UpscaleCancelled()
        def report(done, total, index=index, name=name):
            if progress is not None:
                progress(index*100+round(done*100/max(1, total)), 200, f"RealESRGAN x4 · {name} · {done}/{total}")
        result.append(upscaler.upscale(image, progress=report, cancelled=cancelled))
    return result[0], result[1]
