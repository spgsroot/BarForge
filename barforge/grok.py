"""Optional xAI Grok integration: OAuth Responses API client, settings and background jobs.

Nothing here touches widgets (see grok_ui.py). Only the "enabled" flag and the model name are
written to settings. OAuth credentials are managed separately by xai_oauth.py and never enter
dataset sessions, logs or request error messages.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import math
import os
import re
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from PIL import Image
from PySide6.QtCore import QRunnable, QStandardPaths, QThreadPool

from .xai_oauth import OAuthError, OAuthManager

API_URL = "https://api.x.ai/v1/responses"
DEFAULT_MODEL = "grok-4.7"
TASKS = ("caption", "bars", "scene")
MAX_IMAGE_SIDE = 2048  # longer side of the uploaded image, px
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # JPEG size; its base64 form stays under xAI's 20 MiB image limit
JPEG_QUALITY = 90
MAX_BARS = 24
MIN_BAR_SIDE = 0.002  # smaller boxes are noise (0.2 % of the image side)
EDGE_TOLERANCE = 0.02  # a box may overshoot the image edge by this much (then it is clamped), more is rejected
REQUEST_TIMEOUT = httpx.Timeout(connect=15.0, read=240.0, write=60.0, pool=15.0)
USER_AGENT = "BarForge/1.0"

SYSTEM_PROMPT = (
    "You help prepare training pairs for an image-editing model from manga panels: a clean panel and the "
    "same panel with opaque white censor bars. Be factual, brief and neutral, and output exactly what the "
    "task asks for, nothing else."
)
CAPTION_PROMPT = """Current caption of this training pair:
<caption>
{caption}
</caption>

Rewrite it as one precise, natural instruction for an image-editing model that has to restore the clean panel from the censored one.
- Keep a leading trigger phrase (for example "rustnomiconUncensor.") exactly as written.
- Keep the language and the imperative style of the current caption; one or two short sentences, at most 300 characters.
- Mention only what matters for the edit; no explicit detail.
- If the current caption is empty, write a suitable one from scratch.
Output the caption text only: no quotes, no markdown, no explanation."""
BARS_PROMPT = """This clean manga panel is {width}x{height} pixels. Find the regions that published manga conventionally hides behind opaque censor bars (explicit anatomy) and return them as JSON only.
- Each item: "x" and "y" are the top-left corner of the box, "width" and "height" are its size. All four are fractions of the image width/height between 0 and 1; the origin is the top-left corner and y grows downward; x+width <= 1 and y+height <= 1.
- One tight rectangle per region, fully covering it; usually wider than tall, like a horizontal bar. At most {max_bars} items.
- "label": 1-4 words in Russian naming the region.
- Never box faces, speech text or ordinary clothed areas. If nothing needs a bar, return an empty list."""
SCENE_PROMPT = (
    "Describe this manga panel neutrally and briefly, in Russian, in 2-4 sentences: how many characters there "
    "are, their apparent pose and action, the state of their clothing, which body regions are visible, and the "
    "setting. Use clinical, non-erotic wording. Do not guess identities and do not transcribe dialogue. "
    "Plain text only."
)
BARS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "bars": {
            "type": "array",
            "maxItems": MAX_BARS,
            "items": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "minimum": 0, "maximum": 1},
                    "y": {"type": "number", "minimum": 0, "maximum": 1},
                    "width": {"type": "number", "minimum": 0, "maximum": 1},
                    "height": {"type": "number", "minimum": 0, "maximum": 1},
                    "label": {"type": "string", "maxLength": 60},
                },
                "required": ["x", "y", "width", "height", "label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["bars"],
    "additionalProperties": False,
}
_MAX_OUTPUT_TOKENS = {"caption": 1024, "bars": 2048, "scene": 1024}  # visible tokens only, reasoning is not counted


class GrokError(Exception):
    """Failure with a safe, user-facing Russian message (never contains OAuth credentials)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class GrokCancelled(Exception):
    """The request was cancelled; its outcome must be ignored."""


_BEARER = re.compile(r"(?i)\bbearer\s+\S+")
_XAI_KEY = re.compile(r"\bxai-[A-Za-z0-9_\-]{4,}")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/\-]{0,99}")


def scrub(value: object, secret: str | None = None, limit: int = 400) -> str:
    """One-line, length-capped text with credentials and Bearer values masked out."""
    text = str(value)
    if secret:
        text = text.replace(secret, "***")
    text = _XAI_KEY.sub("xai-***", _BEARER.sub("Bearer ***", text))
    text = " ".join("".join(ch if ch.isprintable() else " " for ch in text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def clean_model(value: str) -> str:
    model = value.strip()
    if not _MODEL.fullmatch(model):
        raise GrokError("Название модели: латиница, цифры и символы . _ : / - (до 100 знаков), без пробелов.")
    return model


# ---------------------------------------------------------------- settings (non-secret)


@dataclass(frozen=True)
class GrokSettings:
    enabled: bool = False
    model: str = DEFAULT_MODEL


def default_settings_path() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericConfigLocation)
    return Path(base or Path.home()) / "BarForge" / "grok_settings.json"


class SettingsStore:
    """The only data written to disk: the enabled flag and the model name, replaced atomically."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_settings_path()

    def load(self) -> GrokSettings:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return GrokSettings()
        if not isinstance(raw, dict):
            return GrokSettings()
        model = raw.get("model")
        try:
            model = clean_model(model) if isinstance(model, str) else DEFAULT_MODEL
        except GrokError:
            model = DEFAULT_MODEL
        return GrokSettings(enabled=raw.get("enabled") is True, model=model)

    def save(self, settings: GrokSettings) -> None:
        data = {"version": 1, "enabled": settings.enabled, "model": settings.model}
        tmp: str | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".grok_settings-", suffix=".tmp", dir=self.path.parent)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except OSError as exc:
            raise GrokError(f"Не удалось сохранить настройки Grok: {scrub(exc)}") from None
        finally:
            if tmp is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)  # nothing left to remove after a successful replace


# ---------------------------------------------------------------- results


@dataclass(frozen=True)
class BarSuggestion:
    """A box in fractions (0..1) of the submitted image; origin is its top-left corner, y grows downward."""

    x: float
    y: float
    width: float
    height: float
    label: str = ""

    def as_box(self) -> dict[str, float]:
        """The shape handed to the main window's apply_bars callback."""
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class GrokResult:
    task: str
    text: str = ""  # caption / scene description
    bars: tuple[BarSuggestion, ...] = ()
    notes: tuple[str, ...] = ()
    model: str = ""  # the model that actually answered, as echoed by xAI
    tokens_in: int = 0
    tokens_out: int = 0


# ---------------------------------------------------------------- image, prompt, payload


def to_rgb(image: Image.Image) -> Image.Image:
    """RGB version of ``image`` (transparency goes onto white); the image itself when it already is RGB."""
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        rgb = Image.new("RGB", rgba.size, "white")
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        return rgb
    return image.convert("RGB")


def upload_size(width: int, height: int) -> tuple[int, int]:
    """Size an image is scaled to before upload: longer side at most MAX_IMAGE_SIDE, never enlarged."""
    scale = min(1.0, MAX_IMAGE_SIDE / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def fit_for_upload(image: Image.Image) -> Image.Image:
    """RGB image scaled to upload_size(); only read from ``image``, which may be returned as is."""
    rgb = to_rgb(image)
    size = upload_size(*rgb.size)
    return rgb if size == rgb.size else rgb.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)


def encode_data_url(image: Image.Image) -> str:
    """JPEG + base64 data URL for an RGB image (xAI accepts jpg/jpeg/png up to 20 MiB)."""
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=JPEG_QUALITY, optimize=True)
    data = buffer.getvalue()
    if len(data) > MAX_IMAGE_BYTES:
        raise GrokError("Изображение слишком большое для отправки в xAI.")
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def build_payload(task: str, model: str, image_url: str, caption: str, size: tuple[int, int]) -> dict[str, Any]:
    """Body of POST /v1/responses: one user turn with the image and the task prompt, store=false."""
    if task == "caption":
        prompt = CAPTION_PROMPT.format(caption=caption.strip()[:2000])
    elif task == "bars":
        prompt = BARS_PROMPT.format(width=size[0], height=size[1], max_bars=MAX_BARS)
    elif task == "scene":
        prompt = SCENE_PROMPT
    else:
        raise ValueError(f"unknown Grok task: {task!r}")
    payload: dict[str, Any] = {
        "model": model,
        "store": False,
        "max_output_tokens": _MAX_OUTPUT_TOKENS[task],
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": image_url, "detail": "high"},
                    {"type": "input_text", "text": prompt},
                ],
            },
        ],
    }
    if task == "bars":
        payload["text"] = {
            "format": {"type": "json_schema", "name": "censor_bars", "schema": BARS_SCHEMA, "strict": True}
        }
    return payload


# ---------------------------------------------------------------- response parsing


def _output_text(data: dict[str, Any]) -> tuple[str, str]:
    """(assistant text, refusal text) collected from a Responses API payload."""
    texts: list[str] = []
    refusals: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif part.get("type") == "refusal":
                refusals.append(str(part.get("refusal") or part.get("text") or ""))
    if not texts and isinstance(data.get("output_text"), str):
        texts.append(data["output_text"])
    return "\n".join(texts).strip(), "\n".join(r for r in refusals if r).strip()


def _tokens(usage: Any, *keys: str) -> int:
    if isinstance(usage, dict):
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return max(value, 0)
    return 0


def clean_caption(text: str) -> str:
    """Single-line caption without fences, 'Caption:' prefixes or wrapping quotes."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*\s*|\s*```$", "", text)
    text = " ".join(text.split())
    text = re.sub(r"(?i)^(caption|капшен|подпись)\s*:\s*", "", text)
    if len(text) >= 2 and text[0] + text[-1] in ('""', "''", "«»", "“”", "``"):
        text = text[1:-1].strip()
    return text


def _load_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*\s*|\s*```$", "", text).strip()

    def reject(constant: str) -> Any:  # NaN / Infinity are not valid JSON numbers for us
        raise ValueError(constant)

    for candidate in (text, text[text.find("{") : text.rfind("}") + 1]):
        try:
            return json.loads(candidate, parse_constant=reject)
        except ValueError:
            continue
    raise GrokError("Grok вернул не JSON: " + scrub(text, limit=200))


def _validate_bar(item: Any) -> BarSuggestion | None:
    if not isinstance(item, dict):
        return None
    values: list[float] = []
    for key in ("x", "y", "width", "height"):
        value = item.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            number = float(value)
        except OverflowError:  # an int too large for a float
            return None
        if not math.isfinite(number):
            return None
        values.append(number)
    x, y, width, height = values
    eps = 1e-6
    if min(x, y) < -eps or max(x, y) > 1 + eps or width <= 0 or height <= 0 or max(width, height) > 1 + eps:
        return None
    x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
    if x + width > 1 + EDGE_TOLERANCE or y + height > 1 + EDGE_TOLERANCE:
        return None
    width, height = min(width, 1.0 - x), min(height, 1.0 - y)
    if width < MIN_BAR_SIDE or height < MIN_BAR_SIDE:
        return None
    label = item.get("label")
    return BarSuggestion(x, y, width, height, scrub(label, limit=60) if isinstance(label, str) else "")


def parse_bars(text: str) -> tuple[list[BarSuggestion], int]:
    """Validated boxes (at most MAX_BARS) and the number of items that were dropped."""
    data = _load_json(text)
    items = data.get("bars") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise GrokError("В ответе Grok нет списка «bars».")
    bars: list[BarSuggestion] = []
    dropped = 0
    for item in items:
        bar = _validate_bar(item)
        if bar is None or len(bars) >= MAX_BARS:
            dropped += 1
        else:
            bars.append(bar)
    if items and not bars:
        raise GrokError("Все полосы в ответе Grok некорректны (координаты должны быть долями 0…1).")
    return bars, dropped


def parse_response(task: str, data: Any, model: str = "") -> GrokResult:
    """Turn a decoded Responses API body into a GrokResult, or raise GrokError."""
    if not isinstance(data, dict):
        raise GrokError("Ответ xAI имеет неожиданный формат.")
    error = data.get("error")
    if isinstance(error, dict) and (error.get("message") or error.get("code")):
        raise GrokError("xAI сообщил об ошибке: " + scrub(error.get("message") or error.get("code"), limit=300))
    status = data.get("status")
    if status == "failed":
        raise GrokError("Запрос не выполнен на стороне xAI.")
    details = data.get("incomplete_details")
    reason = scrub(details.get("reason") or "", limit=60) if isinstance(details, dict) else ""
    text, refusal = _output_text(data)
    if refusal and not text:
        raise GrokError("Grok отказался отвечать: " + scrub(refusal, limit=300))
    if not text:
        raise GrokError("Grok вернул пустой ответ" + (f" (причина: {reason})" if reason else "") + ".")
    notes: list[str] = []
    if status == "incomplete":
        notes.append("Ответ неполный" + (f" ({reason})" if reason else "") + ".")
    bars: list[BarSuggestion] = []
    if task == "bars":
        try:
            bars, dropped = parse_bars(text)
        except GrokError:
            if status == "incomplete":
                raise GrokError(
                    "Ответ Grok оборван" + (f" ({reason})" if reason else "") + ": список полос неполный."
                ) from None
            raise
        text = ""
        if dropped:
            notes.append(f"Отброшено полос: {dropped} (некорректные координаты или лимит {MAX_BARS}).")
    elif task == "caption":
        text = clean_caption(text)
        if not text:
            raise GrokError("Grok вернул пустой капшен.")
    usage = data.get("usage")
    return GrokResult(
        task=task,
        text=text,
        bars=tuple(bars),
        notes=tuple(notes),
        model=scrub(data.get("model") or model, limit=100),
        tokens_in=_tokens(usage, "input_tokens", "prompt_tokens"),
        tokens_out=_tokens(usage, "output_tokens", "completion_tokens"),
    )


# ---------------------------------------------------------------- client


_HTTP_TEXT = {
    400: "xAI отклонил запрос (400)",
    401: "xAI отклонил OAuth-сессию (401): войдите повторно через «Настройки Grok»",
    403: "Доступ запрещён (403): OAuth-вход выполнен, но аккаунту или подписке недоступна эта модель или операция",
    404: "Модель или адрес не найдены (404): проверьте название модели в настройках",
    413: "Изображение слишком велико для xAI (413)",
    422: "xAI не смог обработать запрос (422)",
    429: "Превышена квота OAuth-подписки или лимит запросов xAI (429)",
}


class GrokClient:
    """One blocking call to the xAI Responses API; run it in a worker thread.

    cancel() may be called from any thread: it flags the call and closes the in-flight HTTP
    client so a blocked read fails at once. ``transport`` (e.g. httpx.MockTransport) is injectable
    for verification. Credentials are sent only to the fixed xAI inference endpoint.
    """

    def __init__(
        self,
        oauth: OAuthManager,
        model: str = DEFAULT_MODEL,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: httpx.Timeout = REQUEST_TIMEOUT,
    ) -> None:
        self._oauth = oauth
        self._access_token: str | None = None
        self.model = clean_model(model)
        self._transport = transport
        self._timeout = timeout
        self._lock = threading.Lock()
        self._http: httpx.Client | None = None
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            http = self._http
        if http is not None:
            with contextlib.suppress(Exception):
                http.close()

    def run(self, task: str, image: Image.Image, caption: str = "") -> GrokResult:
        if task not in TASKS:
            raise ValueError(f"unknown Grok task: {task!r}")
        prepared = fit_for_upload(image)
        self._raise_if_cancelled()
        payload = build_payload(task, self.model, encode_data_url(prepared), caption, prepared.size)
        self._raise_if_cancelled()
        return parse_response(task, self._post(payload), self.model)

    def _raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise GrokCancelled()

    def _post(self, payload: dict[str, Any]) -> Any:
        self._raise_if_cancelled()
        try:
            self._access_token = self._oauth.get_access_token()
        except OAuthError as exc:
            raise GrokError(exc.message) from None
        self._raise_if_cancelled()
        headers = {"Authorization": f"Bearer {self._access_token}", "Accept": "application/json", "User-Agent": USER_AGENT}
        try:
            with httpx.Client(transport=self._transport, timeout=self._timeout, follow_redirects=False) as http:
                with self._lock:
                    if self._cancelled:
                        raise GrokCancelled()
                    self._http = http
                try:
                    response = http.post(API_URL, json=payload, headers=headers)
                finally:
                    with self._lock:
                        self._http = None
        except GrokCancelled:
            raise
        except httpx.TimeoutException:
            raise GrokError("xAI не ответил за отведённое время. Повторите запрос позже.") from None
        except Exception as exc:
            if self.cancelled:
                raise GrokCancelled() from None
            if isinstance(exc, httpx.HTTPError):
                raise GrokError(
                    f"Не удалось связаться с xAI ({type(exc).__name__}). Проверьте интернет, прокси и брандмауэр."
                ) from None
            raise
        self._raise_if_cancelled()
        if response.status_code == 401:
            try:
                self._oauth.invalidate(access_token=self._access_token)
            except OAuthError as exc:
                raise GrokError(f"{_HTTP_TEXT[401]}. {exc.message}") from None
        if not response.is_success:
            raise GrokError(self._http_error(response))
        try:
            return response.json()
        except ValueError:
            raise GrokError("xAI вернул ответ, который не является JSON.") from None

    def _http_error(self, response: httpx.Response) -> str:
        status = response.status_code
        if status in _HTTP_TEXT:
            base = _HTTP_TEXT[status]
        elif status >= 500:
            base = f"Сервис xAI временно недоступен ({status}). Повторите позже"
        else:
            base = f"Неожиданный ответ xAI ({status})"
        detail = ""
        try:
            body = response.json()
        except ValueError:
            body = None
        error = body.get("error", body) if isinstance(body, dict) else None
        if isinstance(error, dict):
            error = error.get("message") or error.get("code") or error.get("detail")
        if isinstance(error, str):
            detail = scrub(error, self._access_token, 300)
        return f"{base}. {detail}" if detail else f"{base}."


# ---------------------------------------------------------------- background job


class GrokJob:
    """One request in flight.

    A plain object, not a QObject: the worker fills in result/error and the controller announces
    completion through its own signal, so no Qt object is ever destroyed from a worker thread and a
    closed dialog cannot leave the worker emitting into a deleted object.
    """

    def __init__(self, client: GrokClient, task: str, image: Image.Image, caption: str) -> None:
        self.client = client
        self.task = task
        self.image: Image.Image | None = image
        self.caption = caption
        self.result: GrokResult | None = None
        self.error: str | None = None

    @property
    def cancelled(self) -> bool:
        return self.client.cancelled

    def cancel(self) -> None:
        self.client.cancel()

    def start(self, pool: QThreadPool, notify: Callable[[GrokJob], object]) -> None:
        """Queue the request; ``notify(job)`` is called from the worker thread when it ends (also if cancelled)."""
        pool.start(_Worker(self, notify))


class _Worker(QRunnable):
    def __init__(self, job: GrokJob, notify: Callable[[GrokJob], object]) -> None:
        super().__init__()
        self._job = job
        self._notify = notify

    def run(self) -> None:
        job = self._job
        try:
            job.result = job.client.run(job.task, job.image, job.caption)
        except GrokCancelled:
            pass
        except GrokError as exc:
            job.error = exc.message
        except Exception as exc:  # a bug must surface as a message, not vanish in the pool thread
            job.error = f"Непредвиденная ошибка при запросе к Grok ({type(exc).__name__})."
        job.image = None
        with contextlib.suppress(RuntimeError):  # the receiving QObject is already gone at shutdown
            self._notify(job)
