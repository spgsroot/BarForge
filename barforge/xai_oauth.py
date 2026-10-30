"""xAI sign-in for BarForge: OAuth 2.0 device authorization grant (RFC 8628).

BarForge has no OAuth registration of its own. By explicit user decision it signs in as the public client of
the official Grok CLI (the identity other agent harnesses use as well): no client secret and no PKCE for the
device grant. Nothing here touches widgets (see grok_ui.py) and this module never writes a file.

Where tokens may live, and nowhere else:
- the OS keyring, only after an explicit choice: one record that holds just the long-lived refresh token. Access
  tokens are short-lived bearer secrets, so they are not kept at rest; they are re-minted on first use after a
  restart (which also keeps the record far below the 2560-byte blob limit of Windows Credential Manager);
- this process's memory ("this session only"), also an explicit choice.
There is no settings-file, environment-variable or plaintext fallback for tokens.

Every provider URL is pinned to xAI hosts (the discovery result included) and redirects are never followed, so
credentials can only reach the pinned endpoints. Cancellation is cooperative: a threading.Event plus generation
counters guarantee that a login that was cancelled, superseded by a newer login or overtaken by logout never
stores anything, and that a stale refresh never overwrites a newer credential.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import re
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx
from .i18n import tr

__all__ = [
    "CLIENT_ID",
    "ISSUER",
    "SCOPES",
    "DeviceCode",
    "OAuthCancelled",
    "OAuthError",
    "OAuthManager",
    "OAuthTokenStore",
    "OAuthTokens",
    "validate_verification_url",
]

ISSUER = "https://auth.x.ai"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"
CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"  # public client of the official Grok CLI
SCOPES = "openid profile email offline_access grok-cli:access api:access"
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
KEYRING_SERVICE = "BarForge xAI OAuth"
KEYRING_USER = "tokens"
USER_AGENT = "BarForge/1.0"
HTTP_TIMEOUT = httpx.Timeout(15.0)
REFRESH_MARGIN = 60.0  # seconds before expiry at which an access token is renewed
DEFAULT_INTERVAL = 5.0  # RFC 8628 polling interval when xAI names none
SLOW_DOWN_STEP = 5.0  # RFC 8628: added to the polling interval on every slow_down

_ENDPOINT_HOSTS = frozenset({"auth.x.ai"})
_VERIFICATION_HOSTS = frozenset({"accounts.x.ai", "auth.x.ai"})
_HEADERS = {"Accept": "application/json", "User-Agent": USER_AGENT}
_RECORD_VERSION = 1

_MSG_NOT_SIGNED_IN = tr("Вход через xAI OAuth не выполнен. Войдите в настройках Grok.")
_MSG_SIGN_IN_AGAIN = tr("Доступ xAI истёк или отозван. Войдите через xAI OAuth заново.")
_MSG_SESSION_CHANGED = tr("Вход xAI был сброшен или заменён во время обновления доступа. Повторите действие.")
_MSG_CODE_EXPIRED = tr("Срок действия кода подтверждения истёк. Начните вход заново.")
_MSG_DENIED = tr("Доступ отклонён на странице xAI. Чтобы войти, начните заново и подтвердите доступ.")
_MSG_NO_STORAGE = tr("Защищённое хранилище ОС недоступно: войти можно только на время сессии.")
_MSG_TIMEOUT = tr("xAI не ответил за отведённое время. Повторите позже.")
_MSG_BAD_URL = tr("xAI вернул недопустимый адрес подтверждения; вход прерван из соображений безопасности.")
_MSG_REVOKE_MANUALLY = tr("Отзовите доступ вручную в настройках аккаунта xAI.")


class OAuthError(Exception):
    """Failure with a safe, user-facing Russian message (never contains a token, device code or secret)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class OAuthCancelled(Exception):
    """The login was cancelled, superseded or overtaken by logout; its outcome must be ignored."""


@dataclass(frozen=True)
class OAuthTokens:
    """One credential. ``access_token`` is '' for a credential restored from the keyring (minted on demand)."""

    access_token: str = field(repr=False)
    refresh_token: str | None = field(repr=False)
    expires_at: float  # epoch seconds at which the access token expires; 0.0 while there is none
    scope: str = ""
    token_type: str = "Bearer"


@dataclass(frozen=True)
class DeviceCode:
    """What the user needs to approve a login, plus the secret device code used for polling."""

    device_code: str = field(repr=False)
    user_code: str
    verification_uri: str
    verification_uri_complete: str  # opens with the code pre-filled; equals verification_uri when xAI gives none
    expires_at: float
    interval: float
    generation: int = 0  # internal: the login epoch that issued this code (stale-login guard)


@dataclass(frozen=True)
class _Endpoints:
    device: str
    token: str
    revocation: str | None


# ---------------------------------------------------------------- validation helpers

_BEARER = re.compile(r"(?i)\bbearer\s+\S+")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_\-]{6,}(?:\.[A-Za-z0-9_\-]*){0,2}")
_OPAQUE = re.compile(r"[\x21-\x7e]{1,16384}")  # tokens and codes: printable ASCII, no whitespace
_URL_TEXT = re.compile(r"[\x21-\x7e]{1,2048}")
_USER_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.\-]{0,63}")
_ERROR_CODE = re.compile(r"[A-Za-z0-9_.\-]{1,64}")


def _scrub(value: object, *secrets: str, limit: int = 300) -> str:
    """One-line, length-capped text with the given secrets and anything token-shaped masked out."""
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    text = _JWT.sub("***", _BEARER.sub("Bearer ***", text))
    text = " ".join("".join(ch if ch.isprintable() else " " for ch in text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _number(value: object) -> float | None:
    """A finite positive JSON number (a bool is not one), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and number > 0 else None


def _opaque(value: object) -> str | None:
    return value if isinstance(value, str) and _OPAQUE.fullmatch(value) else None


def _pinned_url(value: object, hosts: frozenset[str]) -> str | None:
    """``value`` when it is a plain https URL on one of ``hosts``: default port, no userinfo, no fragment."""
    if not isinstance(value, str) or not _URL_TEXT.fullmatch(value) or "\\" in value or "#" in value:
        return None
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return None
    if parts.scheme != "https" or parts.hostname not in hosts or "@" in parts.netloc or port not in (None, 443):
        return None
    return value


def validate_verification_url(url: object) -> str:
    """The verification URL if it is safe to open in a browser (https on accounts.x.ai or auth.x.ai)."""
    checked = _pinned_url(url, _VERIFICATION_HOSTS)
    if checked is None:
        raise OAuthError(_MSG_BAD_URL)
    return checked


def _endpoint(document: dict[str, Any], name: str) -> str:
    url = _pinned_url(document.get(name), _ENDPOINT_HOSTS)
    if url is None:
        raise OAuthError(
            tr("Настройки входа xAI не прошли проверку: адрес «{name}» отсутствует или не принадлежит auth.x.ai. Вход прерван.", name=name)
        )
    return url


def _error_code(body: dict[str, Any] | None) -> str:
    """The OAuth ``error`` code of a response ('' when there is none or it is not a plain code)."""
    code = body.get("error") if body is not None else None
    return code if isinstance(code, str) and _ERROR_CODE.fullmatch(code) else ""


def _detail(body: dict[str, Any] | None, *secrets: str) -> str:
    if body is None:
        return ""
    text = body.get("error_description")
    if not isinstance(text, str):
        error = body.get("error")
        text = error.get("message") if isinstance(error, dict) else None
    return _scrub(text, *secrets, limit=200) if isinstance(text, str) else ""


def _status_text(status: int) -> str:
    if status == 400:
        return tr("xAI отклонил запрос (400)")
    if status == 401:
        return tr("xAI не принял авторизацию (401)")
    if status == 403:
        return (
            tr("доступ запрещён (403): аккаунту или подписке xAI недоступен OAuth-доступ к API, хотя вход выполнен корректно")
        )
    if status == 429:
        return tr("слишком много запросов к xAI (429), подождите и повторите")
    if status >= 500:
        return tr("сервис xAI временно недоступен ({status})", status=status)
    return tr("неожиданный ответ xAI ({status})", status=status)


def _failure(action: str, status: int, body: dict[str, Any] | None, *secrets: str) -> OAuthError:
    """A safe error for a non-success provider answer: the action, the HTTP status, the OAuth code, a scrubbed note."""
    text = f"{action}: {_status_text(status)}"
    code = _error_code(body)
    if code:
        text += f" [{code}]"
    detail = _detail(body, *secrets)
    if detail:
        text += f": {detail}"
    return OAuthError(text if text.endswith((".", "…")) else text + ".")


def _malformed(what: str) -> OAuthError:
    return OAuthError(tr("xAI вернул некорректный ответ ({what}); операция прервана.", what=what))


def _no_storage_error(problem: str) -> OAuthError:
    return OAuthError(tr("{_MSG_NO_STORAGE} Причина: {problem}.", _MSG_NO_STORAGE=_MSG_NO_STORAGE, problem=problem) if problem else _MSG_NO_STORAGE)


def _parse_device(body: dict[str, Any], requested_at: float, generation: int) -> DeviceCode:
    device_code = _opaque(body.get("device_code"))
    if device_code is None:
        raise _malformed("device_code")
    user_code = body.get("user_code")
    if not isinstance(user_code, str) or not _USER_CODE.fullmatch(user_code):
        raise _malformed("user_code")
    uri = _pinned_url(body.get("verification_uri"), _VERIFICATION_HOSTS)
    complete = body.get("verification_uri_complete")
    complete = uri if complete in (None, "") else _pinned_url(complete, _VERIFICATION_HOSTS)
    if uri is None or complete is None:
        raise OAuthError(_MSG_BAD_URL)
    lifetime = _number(body.get("expires_in"))
    if lifetime is None:
        raise _malformed("expires_in")
    interval = DEFAULT_INTERVAL if "interval" not in body else _number(body["interval"])
    if interval is None:
        raise _malformed("interval")
    return DeviceCode(device_code, user_code, uri, complete, requested_at + lifetime, interval, generation)


def _parse_tokens(body: dict[str, Any], requested_at: float, previous: OAuthTokens | None) -> OAuthTokens:
    """Tokens from a token-endpoint answer. A refresh answer without refresh_token keeps ``previous``'s one.

    ``requested_at`` is read before the request, so the expiry is never overestimated by network latency.
    The id_token (if any) is deliberately ignored: it is never needed and never stored.
    """
    access = _opaque(body.get("access_token"))
    if access is None:
        raise _malformed("access_token")
    refresh = body.get("refresh_token")
    if refresh in (None, ""):
        refresh = previous.refresh_token if previous is not None else None
    elif _opaque(refresh) is None:
        raise _malformed("refresh_token")
    kind = body.get("token_type")
    if kind is not None and not (isinstance(kind, str) and kind.lower() == "bearer"):
        raise OAuthError(tr("xAI выдал токен неподдерживаемого типа; операция прервана."))
    lifetime = _number(body.get("expires_in"))
    if lifetime is None:
        raise _malformed("expires_in")
    granted = body.get("scope")
    if isinstance(granted, str) and granted.strip():
        scope = _scrub(granted, limit=1024)
    else:
        scope = previous.scope if previous is not None and previous.scope else SCOPES
    return OAuthTokens(access, refresh, requested_at + lifetime, scope)


# ---------------------------------------------------------------- storage


def _encode(tokens: OAuthTokens) -> str:
    """The single keyring record. The refresh token is the durable credential; the access token is re-minted.

    Only a credential without a refresh token (it cannot be renewed) carries its access token and expiry.
    """
    record: dict[str, Any] = {"v": _RECORD_VERSION}
    if tokens.refresh_token:
        record["refresh_token"] = tokens.refresh_token
    else:
        record["access_token"] = tokens.access_token
        record["expires_at"] = tokens.expires_at
    record["scope"] = tokens.scope
    return json.dumps(record, separators=(",", ":"))


def _decode(text: str) -> OAuthTokens:
    """Inverse of _encode; ValueError for anything that is not a record written by this module."""
    record = json.loads(text)
    if not isinstance(record, dict) or record.get("v") != _RECORD_VERSION:
        raise ValueError("unsupported record")
    scope = record.get("scope", "")
    if not isinstance(scope, str):
        raise ValueError("scope")
    refresh = record.get("refresh_token")
    if refresh is not None:
        if _opaque(refresh) is None:
            raise ValueError("refresh_token")
        return OAuthTokens("", refresh, 0.0, scope)
    access = record.get("access_token")
    expires_at = _number(record.get("expires_at"))
    if _opaque(access) is None or expires_at is None:
        raise ValueError("access_token")
    return OAuthTokens(access, None, expires_at, scope)


def _find_secure_backend() -> tuple[Any | None, str, str]:
    """(backend, display title, problem) for a real OS secret store.

    keyring's own selection is bypassed on purpose: it may pick the plaintext/file backends from
    keyrings.alt, the silent "null" backend or the chainer that can fall through to them. Only the
    allowlisted OS stores below are ever used.
    """
    if sys.platform == "win32":
        candidates = [("keyring.backends.Windows", "WinVaultKeyring", tr("Диспетчер учётных данных Windows"))]
    elif sys.platform == "darwin":
        candidates = [("keyring.backends.macOS", "Keyring", tr("Связка ключей macOS"))]
    else:
        candidates = [
            ("keyring.backends.SecretService", "Keyring", "Secret Service"),
            ("keyring.backends.kwallet", "DBusKeyring", "KDE Wallet"),
            ("keyring.backends.libsecret", "Keyring", "libsecret"),
        ]
    problem = tr("в этой ОС нет поддерживаемого защищённого хранилища")
    for module_name, class_name, title in candidates:
        try:
            backend_class = getattr(importlib.import_module(module_name), class_name)
            if backend_class.priority <= 0:  # raises RuntimeError when the backend cannot work here
                raise RuntimeError("unusable")
            backend = backend_class()
        except Exception as exc:
            problem = tr("«{title}» недоступно ({value})", title=title, value=type(exc).__name__)
            continue
        if module_name.endswith(".Windows"):
            with contextlib.suppress(Exception):
                backend.persist = "local machine"  # keep the refresh token off enterprise credential roaming
        return backend, title, ""
    return None, "", problem


class OAuthTokenStore:
    """The current credential: this process's memory plus, by explicit choice, one OS-keyring record.

    ``backend`` (anything with get_password/set_password/delete_password) is injectable for verification; None
    selects an allowlisted OS store (see _find_secure_backend). The keyring is read once, when the store is
    created. Reading properties and load() never block (no lock, no keyring access); mutations are serialized.
    """

    def __init__(self, backend: Any | None = None, backend_title: str = "") -> None:
        if backend is None:
            backend, backend_title, self.secure_problem = _find_secure_backend()
        else:
            self.secure_problem = ""
        self._backend = backend
        self.secure_title = backend_title
        self.read_error = ""  # why a stored login could not be restored at start ('' if it could / none exists)
        self.write_error = ""  # why the last keyring update after a refresh failed ('' if it did not)
        self._lock = threading.Lock()
        self._tokens: OAuthTokens | None = None
        self._source = ""
        self._record = False  # the keyring is known to hold a record: found at start or written by us
        self._read_keyring()

    @property
    def secure_available(self) -> bool:
        return self._backend is not None

    @property
    def source(self) -> str:
        """'keyring' (stored across restarts), 'session' (memory only) or '' (no credential)."""
        return self._source

    def load(self) -> OAuthTokens | None:
        """The current credential, or None. A restored keyring credential has no access token yet."""
        return self._tokens

    def save(self, tokens: OAuthTokens, *, persist: bool) -> None:
        """Make ``tokens`` the credential: in the keyring if ``persist``, else memory only.

        All or nothing: on OAuthError the previous credential is unchanged. A session-only save removes an
        older keyring record, so that nothing outlives the session against the user's choice.
        """
        with self._lock:
            if persist:
                if self._backend is None:
                    raise _no_storage_error(self.secure_problem)
                self._write(tokens)
                self._source = "keyring"
            else:
                if self._record:
                    try:
                        self._delete()
                    except OAuthError as exc:
                        raise OAuthError(
                            tr("{value} Вход на время сессии не выполнен: прежняя запись вернулась бы после перезапуска.", value=exc.message)
                        ) from None
                self._source = "session"
            self._tokens = tokens
            self.read_error = ""
            self.write_error = ""

    def update(self, tokens: OAuthTokens) -> None:
        """Replace the credential after a refresh, keeping its storage mode; never raises.

        The rotated refresh token is the only live one, so a failed keyring write must not lose it: the new
        tokens stay in use in memory and the failure is reported through write_error.
        """
        with self._lock:
            if self._tokens is None:
                return
            if self._source == "keyring":
                try:
                    self._write(tokens)
                except OAuthError as exc:
                    self.write_error = exc.message
                else:
                    self.write_error = ""
            self._tokens = tokens

    def clear(self) -> None:
        """Forget the credential: memory first, then the keyring record (OAuthError if that cannot be removed)."""
        with self._lock:
            self._tokens = None
            self._source = ""
            self.write_error = ""
            if self._backend is not None and (self._record or self.read_error):
                self._delete()
            self.read_error = ""

    def _read_keyring(self) -> None:
        if self._backend is None:
            return
        try:
            text = self._backend.get_password(KEYRING_SERVICE, KEYRING_USER)
        except Exception as exc:
            self.read_error = tr("Не удалось прочитать сохранённый вход из хранилища ОС ({value}).", value=type(exc).__name__)
            return
        if not text:
            return
        self._record = True
        try:
            self._tokens = _decode(text)
        except (ValueError, TypeError):
            self.read_error = tr("Сохранённый в хранилище ОС вход повреждён: войдите заново.")
            return
        self._source = "keyring"

    def _write(self, tokens: OAuthTokens) -> None:
        record = _encode(tokens)
        try:
            self._backend.set_password(KEYRING_SERVICE, KEYRING_USER, record)
        except Exception as exc:
            raise OAuthError(tr("Не удалось сохранить вход в хранилище ОС ({value}).", value=type(exc).__name__)) from None
        self._record = True

    def _delete(self) -> None:
        try:
            self._backend.delete_password(KEYRING_SERVICE, KEYRING_USER)
        except Exception as exc:
            # Backends raise PasswordDeleteError for a missing entry too, so look before complaining.
            try:
                still_there = self._backend.get_password(KEYRING_SERVICE, KEYRING_USER) is not None
            except Exception:
                still_there = True
            if still_there:
                raise OAuthError(tr("Не удалось удалить вход из хранилища ОС ({value}).", value=type(exc).__name__)) from None
        self._record = False


# ---------------------------------------------------------------- manager


class OAuthManager:
    """Device-grant sign-in, refresh and sign-out against the pinned xAI authorization server.

    Thread-safe; the network methods block and belong in worker threads. ``transport`` (e.g.
    httpx.MockTransport) and ``clock`` are injectable for verification; the provider URLs and the client
    identity are not configurable. Two counters guard against stale writes: the login epoch (bumped by
    begin_login and logout) cancels older logins, the credential generation (bumped whenever the credential
    is replaced or dropped) discards the result of a refresh that started before the change. A new login
    does not disturb a refresh in flight for the still-valid previous session.
    """

    def __init__(
        self,
        store: OAuthTokenStore | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store if store is not None else OAuthTokenStore()
        self._transport = transport
        self._clock = clock
        self._lock = threading.Lock()  # credential state: both generations and every store mutation
        self._refresh_lock = threading.Lock()  # one refresh at a time: the refresh token rotates
        self._login_epoch = 0
        self._cred_gen = 0
        self._endpoints: _Endpoints | None = None

    # -- state

    @property
    def has_session(self) -> bool:
        """A credential exists that can still produce an access token (never blocks, no network)."""
        tokens = self.store.load()
        return tokens is not None and self._usable(tokens)

    @property
    def status_text(self) -> str:
        """Russian one-liner about the sign-in state, ready for the settings dialog."""
        store = self.store
        tokens = store.load()
        if tokens is None:
            text = tr("Вход через xAI OAuth не выполнен.")
            if store.read_error:
                text += f" {store.read_error}"
        elif not self._usable(tokens):
            text = tr("Срок действия входа xAI истёк: войдите заново.")
        elif store.source == "keyring":
            where = f" ({store.secure_title})" if store.secure_title else ""
            text = tr("Выполнен вход через xAI OAuth: токен хранится в защищённом хранилище ОС{where}.", where=where)
        else:
            text = tr("Выполнен вход через xAI OAuth на время сессии: токены хранятся только в памяти.")
        if store.write_error:
            text += tr(" Внимание: {value} После перезапуска может потребоваться войти заново.", value=store.write_error)
        return text

    # -- login

    def begin_login(self, stop: threading.Event | None = None) -> DeviceCode:
        """Discover the endpoints and ask xAI for a device code; supersedes a login still in progress.

        Raises OAuthError (message for the user) or OAuthCancelled (``stop`` was set or a newer login began).
        """
        with self._lock:
            if stop is not None and stop.is_set():
                raise OAuthCancelled()  # check and bump are atomic: a cancelled login must not outrun a newer one
            self._login_epoch += 1
            epoch = self._login_epoch
        self._check_login(stop, epoch)
        endpoints = self._discover(stop)
        self._endpoints = endpoints
        self._check_login(stop, epoch)
        requested_at = self._clock()
        status, body = self._request("POST", endpoints.device, {"client_id": CLIENT_ID, "scope": SCOPES}, stop)
        self._check_login(stop, epoch)
        if status != 200:
            raise _failure(tr("Не удалось начать вход через xAI"), status, body)
        if body is None:
            raise _malformed(tr("ответ не является JSON"))
        return _parse_device(body, requested_at, epoch)

    def complete_login(self, device: DeviceCode, stop: threading.Event, *, persist: bool = True) -> OAuthTokens:
        """Poll until the user approves ``device``, then store the credential (keyring if ``persist``, else memory).

        Blocks in the caller's worker thread. OAuthCancelled when ``stop`` is set or the login was superseded
        (logout, a newer begin_login): nothing is stored then. OAuthError: denied, expired, network, malformed
        answer, or the credential could not be stored (it is discarded and the previous credential stays).
        """
        if persist and not self.store.secure_available:
            raise _no_storage_error(self.store.secure_problem)  # fail before the user spends time approving
        self._check_login(stop, device.generation)
        endpoints = self._endpoints_for(stop)
        interval = device.interval
        while True:
            remaining = device.expires_at - self._clock()
            if remaining <= 0:
                raise OAuthError(_MSG_CODE_EXPIRED)
            if stop.wait(min(interval, remaining)):
                raise OAuthCancelled()
            self._check_login(stop, device.generation)
            if device.expires_at <= self._clock():
                raise OAuthError(_MSG_CODE_EXPIRED)
            requested_at = self._clock()
            form = {"grant_type": DEVICE_GRANT, "device_code": device.device_code, "client_id": CLIENT_ID}
            status, body = self._request("POST", endpoints.token, form, stop)
            self._check_login(stop, device.generation)
            error = _error_code(body)
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                interval += SLOW_DOWN_STEP
                continue
            if error in ("access_denied", "authorization_denied"):
                raise OAuthError(_MSG_DENIED)
            if error == "expired_token":
                raise OAuthError(_MSG_CODE_EXPIRED)
            if status != 200 or error:
                raise _failure(tr("Не удалось завершить вход через xAI"), status, body, device.device_code)
            if body is None:
                raise _malformed(tr("ответ не является JSON"))
            tokens = _parse_tokens(body, requested_at, None)
            break
        with self._lock:
            self._check_login(stop, device.generation)  # final word, atomic with the store below
            self.store.save(tokens, persist=persist)
            self._cred_gen += 1
        return tokens

    # -- credential use

    def get_access_token(self) -> str:
        """A valid access token, refreshed first when it is missing or within a minute of expiry.

        Refreshes are serialized (the refresh token rotates, so a second use of the old one must never
        happen): concurrent callers wait and then share the result. Raises OAuthError when not signed in, when
        xAI no longer honours the grant (the credential is then dropped and a new sign-in is required), or when
        the refresh failed for another reason (network, 5xx, 403, ...: the credential is kept). No retries.
        """
        tokens = self.store.load()
        if tokens is None:
            raise OAuthError(_MSG_NOT_SIGNED_IN)
        if self._fresh(tokens):
            return tokens.access_token
        with self._refresh_lock:
            with self._lock:
                tokens = self.store.load()
                generation = self._cred_gen
            if tokens is None:
                raise OAuthError(_MSG_NOT_SIGNED_IN)
            if self._fresh(tokens):
                return tokens.access_token
            return self._refresh(tokens, generation)

    def logout(self) -> None:
        """Forget the credential locally first, then ask xAI to revoke it.

        Local state is gone whatever happens next. OAuthError is raised only afterwards, to report that the
        remote revocation or the keyring cleanup did not succeed.
        """
        with self._lock:
            self._login_epoch += 1  # cancels a login in progress
            self._cred_gen += 1  # discards a refresh in flight
            tokens = self.store.load()
            try:
                self.store.clear()
                local_problem = ""
            except OAuthError as exc:
                local_problem = exc.message
        remote_problem = self._revoke(tokens) if tokens is not None else ""
        problems = [text for text in (local_problem, remote_problem) if text]
        if problems:
            raise OAuthError(tr("Вы вышли в программе, но есть замечания. ") + " ".join(problems))

    def invalidate(self, *, access_token: str | None = None) -> None:
        """Forget the local credential without contacting xAI (the provider rejected it).

        With ``access_token`` the call applies only while that token is still the current one, so a request
        that was signed before a refresh or a new sign-in cannot drop the newer credential. Memory is cleared
        first; OAuthError reports only that the keyring record could not be removed.
        """
        if access_token is None:
            with self._lock:
                problem = self._drop_locked()
        else:
            with self._refresh_lock:  # a refresh already in flight settles first, so its result is compared too
                with self._lock:
                    current = self.store.load()
                    if current is None or current.access_token != access_token:
                        return
                    problem = self._drop_locked()
        if problem:
            raise OAuthError(problem)

    # -- internals

    def _usable(self, tokens: OAuthTokens) -> bool:
        return bool(tokens.refresh_token) or (bool(tokens.access_token) and tokens.expires_at > self._clock())

    def _fresh(self, tokens: OAuthTokens) -> bool:
        return bool(tokens.access_token) and tokens.expires_at - self._clock() > REFRESH_MARGIN

    def _check_login(self, stop: threading.Event | None, epoch: int) -> None:
        """Raise OAuthCancelled when the login was cancelled or superseded (logout, a newer login)."""
        if (stop is not None and stop.is_set()) or epoch != self._login_epoch:
            raise OAuthCancelled()

    def _drop_locked(self) -> str:
        """Forget the credential (the caller holds _lock); the keyring problem text, '' if there was none."""
        self._cred_gen += 1
        try:
            self.store.clear()
        except OAuthError as exc:
            return exc.message
        return ""

    def _expired(self, generation: int) -> OAuthError:
        """Drop the credential a refresh was based on (unless replaced meanwhile); the error asks for a new sign-in."""
        with self._lock:
            problem = self._drop_locked() if self._cred_gen == generation else ""
        return OAuthError(f"{_MSG_SIGN_IN_AGAIN} {problem}" if problem else _MSG_SIGN_IN_AGAIN)

    def _refresh(self, tokens: OAuthTokens, generation: int) -> str:
        refresh_token = tokens.refresh_token
        if not refresh_token:
            if tokens.access_token and tokens.expires_at > self._clock():
                return tokens.access_token  # nothing to renew with: used to the end of its lifetime
            raise self._expired(generation)
        requested_at = self._clock()
        endpoints = self._endpoints_for()
        form = {"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": CLIENT_ID}
        status, body = self._request("POST", endpoints.token, form)
        if _error_code(body) == "invalid_grant":
            raise self._expired(generation)
        if status != 200:
            raise _failure(tr("Не удалось обновить доступ xAI"), status, body, refresh_token)
        if body is None:
            raise _malformed(tr("ответ не является JSON"))
        fresh = _parse_tokens(body, requested_at, tokens)
        with self._lock:
            if self._cred_gen != generation:
                raise OAuthError(_MSG_SESSION_CHANGED)  # logged out or signed in again meanwhile: not ours to store
            self.store.update(fresh)
        return fresh.access_token

    def _revoke(self, tokens: OAuthTokens) -> str:
        """Ask xAI to revoke the credential (RFC 7009). '' on success, else a Russian problem description."""
        token = tokens.refresh_token or tokens.access_token
        if not token:
            return ""
        hint = "refresh_token" if tokens.refresh_token else "access_token"
        try:
            endpoint = self._endpoints_for().revocation
            if endpoint is None:
                return tr("xAI не сообщил адрес отзыва токена. {_MSG_REVOKE_MANUALLY}", _MSG_REVOKE_MANUALLY=_MSG_REVOKE_MANUALLY)
            form = {"client_id": CLIENT_ID, "token": token, "token_type_hint": hint}
            status, body = self._request("POST", endpoint, form)
        except OAuthError as exc:
            return tr("Не удалось отозвать доступ в xAI: {value} {_MSG_REVOKE_MANUALLY}", value=exc.message, _MSG_REVOKE_MANUALLY=_MSG_REVOKE_MANUALLY)
        if 200 <= status < 300:
            return ""
        return f"{_failure(tr("Не удалось отозвать доступ в xAI"), status, body, token).message} {_MSG_REVOKE_MANUALLY}"

    def _endpoints_for(self, stop: threading.Event | None = None) -> _Endpoints:
        endpoints = self._endpoints
        if endpoints is None:
            endpoints = self._endpoints = self._discover(stop)
        return endpoints

    def _discover(self, stop: threading.Event | None) -> _Endpoints:
        """Fetch the OIDC discovery document and accept it only if issuer and endpoints are the pinned xAI ones."""
        status, body = self._request("GET", DISCOVERY_URL, None, stop)
        if status != 200:
            raise _failure(tr("Не удалось получить настройки входа xAI"), status, body)
        if body is None:
            raise _malformed(tr("настройки входа не являются JSON"))
        if body.get("issuer") != ISSUER:
            raise OAuthError(tr("Настройки входа xAI не прошли проверку: неожиданный издатель. Вход прерван."))
        device = _endpoint(body, "device_authorization_endpoint")
        token = _endpoint(body, "token_endpoint")
        revocation = _endpoint(body, "revocation_endpoint") if body.get("revocation_endpoint") is not None else None
        return _Endpoints(device, token, revocation)

    def _request(
        self,
        method: str,
        url: str,
        fields: dict[str, str] | None = None,
        stop: threading.Event | None = None,
    ) -> tuple[int, dict[str, Any] | None]:
        """One request to a pinned URL, redirects not followed: (HTTP status, decoded JSON object or None)."""
        try:
            with httpx.Client(transport=self._transport, timeout=HTTP_TIMEOUT, follow_redirects=False) as http:
                response = http.request(method, url, data=fields, headers=_HEADERS)
        except Exception as exc:
            if stop is not None and stop.is_set():
                raise OAuthCancelled() from None
            if isinstance(exc, httpx.TimeoutException):
                raise OAuthError(_MSG_TIMEOUT) from None
            if isinstance(exc, httpx.HTTPError):
                raise OAuthError(
                    tr("Не удалось связаться с xAI ({value}). Проверьте интернет, прокси и брандмауэр.", value=type(exc).__name__)
                ) from None
            raise
        try:
            body = response.json()
        except ValueError:
            body = None
        return response.status_code, body if isinstance(body, dict) else None
