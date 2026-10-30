"""Qt side of the Grok integration (Russian UI): GrokController, settings dialog, request dialog.

API used by the main window:

    self.grok = GrokController(self)
    self.grok.available                    # bool property; False until signed in via xAI OAuth, or while disabled
    self.grok.availability_changed         # Signal(bool)
    self.grok.settings_dialog(self)        # modal; blocks until closed; returns True if settings were saved
    self.grok.ask(self, image, caption, apply_caption, apply_bars)   # window-modal; returns at once

Sign-in is the OAuth device flow of xai_oauth.py (public Grok CLI client, no API key). Its network steps run in
worker threads; the settings dialog drives the methods below, tests and other code may as well:

    self.grok.login(persist=True)          # tokens to the OS keyring (True) or to this session's memory (False)
    self.grok.cancel_login()               # also done by closing the settings dialog
    self.grok.logout()                     # forgets the session at once, then asks xAI to revoke the tokens
    self.grok.open_login_page()            # (re)opens the verification page of the pending sign-in in the browser
    self.grok.auth                         # AuthState snapshot; self.grok.auth_changed (Signal()) announces changes

``image`` is the clean crop to upload. ``apply_caption(text: str)`` and
``apply_bars(boxes: list[dict])`` run only when the user presses the matching apply button in the
dialog. Each box is ``{"x", "y", "width", "height"}``: fractions (0..1) of ``image``, origin at its
top-left corner, x+width <= 1 and y+height <= 1. A callback returning ``False`` means "not applied"
(stale document); any other return value means applied.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from PIL import Image
from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPointF, QRectF, Qt, QThreadPool, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .i18n import tr
from .ui import PageNavigation, fit_window

from .grok import (
    DEFAULT_MODEL,
    BarSuggestion,
    GrokClient,
    GrokError,
    GrokJob,
    GrokResult,
    GrokSettings,
    SettingsStore,
    clean_model,
    to_rgb,
    upload_size,
)
from .xai_oauth import CLIENT_ID, DeviceCode, OAuthCancelled, OAuthError, OAuthManager, validate_verification_url

_TASKS = (
    ("caption", tr("Улучшить капшен"), tr("Grok перепишет капшен как точную инструкцию для редактирующей модели.")),
    ("bars", tr("Предложить полосы"), tr("Grok предложит области, которые обычно закрывают полосами.")),
    ("scene", tr("Описать сцену"), tr("Краткое нейтральное описание сцены и видимой анатомии (только для чтения).")),
)


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setStyleSheet("color: gray;")
    return label


def _thumbnail(image: Image.Image, side: int) -> QImage:
    """Small RGB QImage of a PIL image; the full-size pixels are only read, never copied."""
    rgb = to_rgb(image)
    scale = min(1.0, side / max(rgb.size))
    size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
    small = rgb if size == rgb.size else rgb.resize(size, Image.Resampling.BILINEAR, reducing_gap=2.0)
    data = small.tobytes()
    return QImage(data, small.width, small.height, small.width * 3, QImage.Format.Format_RGB888).copy()


AuthPhase = Literal["idle", "requesting", "waiting", "signing_out"]


@dataclass(frozen=True)
class AuthState:
    """What the sign-in UI shows, as a snapshot: a late signal can never make a dialog render stale data.

    ``phase``: "idle"; "requesting" (asking xAI for a code); "waiting" (the code is shown and must be approved in
    the browser: NOT signed in yet); "signing_out". ``notice`` is the outcome of the last sign-in/sign-out (empty
    once the next one starts) and ``failed`` marks it as a problem. Nothing secret is in here.
    """

    phase: AuthPhase = "idle"
    user_code: str = ""
    verification_uri: str = ""
    expires_at: float = 0.0  # time.time() deadline of the user code
    notice: str = ""
    failed: bool = False


class _AuthJob:
    """One sign-in or sign-out in its worker thread.

    A plain object like GrokJob: the worker fills it in and the controller announces progress through its own
    signals, so a worker never touches a widget and a closed dialog cannot be hit by one.
    """

    def __init__(self, kind: Literal["login", "logout"], persist: bool = False) -> None:
        self.kind = kind
        self.persist = persist
        self.stop = threading.Event()  # set = abandoned: the manager stops polling and stores no tokens
        self.signed_in = False
        self.error: str | None = None
        self.thread: threading.Thread | None = None


class GrokController(QObject):
    """Settings, xAI OAuth session, worker threads and dialogs of the Grok integration; one instance per main window.

    ``settings_path``, ``oauth`` (an OAuthManager, e.g. over a test token store) and ``transport`` (an httpx
    transport such as httpx.MockTransport; it serves the requests and, when no ``oauth`` is given, that manager's
    sign-in traffic) can be injected for verification; the defaults are the real thing.
    """

    availability_changed = Signal(bool)
    request_finished = Signal(object)  # GrokJob; delivered in the GUI thread for every job, cancelled ones too
    auth_changed = Signal()  # the sign-in/sign-out state changed (GUI thread): read ``auth``
    _job_done = Signal(object)  # worker thread -> GUI thread: a request ended
    _auth_code = Signal(object, object)  # worker thread -> GUI thread: (_AuthJob, DeviceCode) the user must approve
    _auth_done = Signal(object)  # worker thread -> GUI thread: a sign-in/sign-out worker ended

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        settings_path: Path | None = None,
        oauth: OAuthManager | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(parent)
        self._store = SettingsStore(settings_path)
        self._settings = self._store.load()
        self.oauth = oauth if oauth is not None else OAuthManager(transport=transport)
        self._transport = transport
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(2)
        self._jobs: set[GrokJob] = set()
        self._auth: _AuthJob | None = None  # the sign-in/out the UI is about; abandoned workers live on in _auth_jobs
        self._auth_jobs: set[_AuthJob] = set()
        self._device: DeviceCode | None = None
        self._notice = ""
        self._notice_failed = False
        self._ask: GrokAskDialog | None = None
        self._available = self.available
        self._job_done.connect(self._on_job_done)
        self._auth_code.connect(self._on_auth_code)
        self._auth_done.connect(self._on_auth_done)
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    @property
    def settings(self) -> GrokSettings:
        return self._settings

    @property
    def available(self) -> bool:
        """True only when the integration is enabled AND an xAI OAuth session exists, with no sign-in/out running.

        The last condition makes it turn true only in the GUI thread, right before availability_changed says so.
        """
        return self._settings.enabled and self._auth is None and self.oauth.has_session

    @property
    def status_text(self) -> str:
        if not self._settings.enabled:
            return tr("Grok выключен.")
        if self._auth is not None:
            return tr("Идёт выход из xAI.") if self._auth.kind == "logout" else tr("Идёт вход через xAI OAuth.")
        if not self.oauth.has_session:
            return tr("Grok включён. {value}", value=self.oauth.status_text)
        return tr("Grok готов (модель {value}).", value=self._settings.model)

    @property
    def auth(self) -> AuthState:
        """Snapshot of the sign-in state (GUI thread); ``waiting`` is never a signed-in state."""
        job, device = self._auth, self._device
        if job is None:
            phase: AuthPhase = "idle"
        elif job.kind == "logout":
            phase = "signing_out"
        else:
            phase = "waiting" if device is not None else "requesting"
        return AuthState(
            phase,
            device.user_code if device is not None else "",
            device.verification_uri if device is not None else "",
            device.expires_at if device is not None else 0.0,
            self._notice,
            self._notice_failed,
        )

    def settings_dialog(self, parent: QWidget | None = None) -> bool:
        """Run the settings dialog modally; True if the user saved."""
        dialog = GrokSettingsDialog(self, parent)
        saved = bool(dialog.exec())
        dialog.deleteLater()
        self._notice, self._notice_failed = "", False  # the dialog has shown the last outcome
        return saved

    def ask(
        self,
        parent: QWidget | None,
        image: Image.Image,
        caption: str,
        on_caption: Callable[[str], object],
        on_bars: Callable[[list[dict[str, float]]], object],
    ) -> GrokAskDialog | None:
        """Open the request dialog for this frame (window-modal, returns immediately)."""
        if not self.available:
            self._explain_unavailable(parent)
            return None
        if self._ask is not None:
            self._ask.raise_()
            self._ask.activateWindow()
            return self._ask
        dialog = GrokAskDialog(self, image, caption, on_caption, on_bars, parent)
        dialog.finished.connect(lambda _result, current=dialog: self._ask_closed(current))
        dialog.destroyed.connect(lambda _object, current=dialog: self._ask_closed(current))
        self._ask = dialog
        dialog.open()
        return dialog

    def start(self, task: str, image: Image.Image, caption: str) -> GrokJob:
        """Queue one request after the user presses the send button.

        Completion arrives as request_finished(job): job.result is a GrokResult, job.error a Russian
        message; a cancelled job reports neither and must be ignored.
        """
        if not self.available:
            raise GrokError(self.status_text)
        client = GrokClient(self.oauth, self._settings.model, transport=self._transport)
        job = GrokJob(client, task, image, caption)
        self._jobs.add(job)
        job.start(self._pool, self._notify)
        return job

    def _notify(self, job: GrokJob) -> None:
        """Runs in a worker thread when its job ends; the queued signal hands the job to the GUI thread."""
        self._job_done.emit(job)

    def cancel_all(self) -> None:
        for job in tuple(self._jobs):
            job.cancel()

    @Slot()
    def shutdown(self) -> None:
        """Cancel running requests and sign-ins and give the workers a moment to stop (wired to application exit)."""
        self.cancel_all()
        unfinished = tuple(self._auth_jobs)
        for job in unfinished:
            job.stop.set()
        deadline = time.monotonic() + 3.0
        self._pool.waitForDone(3000)
        for job in unfinished:
            if job.thread is not None:
                job.thread.join(max(0.0, deadline - time.monotonic()))

    def apply_settings(self, *, enabled: bool, model: str) -> None:
        """Validate and store the enable flag and the model name. Raises GrokError."""
        settings = GrokSettings(enabled=enabled, model=clean_model(model))
        self._store.save(settings)
        self._settings = settings
        self._refresh_availability()

    # ---- xAI OAuth sign-in / sign-out

    def login(self, *, persist: bool) -> None:
        """Start the OAuth device sign-in in a worker thread; raises GrokError when it cannot start.

        The tokens go to the OS keyring (persist=True) or stay in this session's memory (persist=False).
        Progress arrives through auth_changed: phase "requesting", then "waiting" with the code (the browser opens
        on the verification page by itself), then "idle" with ``notice`` telling how it ended. Only a finished
        sign-in makes ``oauth.has_session`` (and so ``available``) true.
        """
        if self._auth is not None and self._auth.kind == "login":
            raise GrokError(tr("Вход уже выполняется."))
        if self.oauth.has_session:  # a sign-out may still be revoking an earlier session: no obstacle once it is gone
            raise GrokError(
                tr("Выход ещё выполняется.") if self._auth is not None else tr("Вход уже выполнен: сначала нажмите «Выйти».")
            )
        if persist and not self.oauth.store.secure_available:
            raise GrokError(tr("Защищённое хранилище ОС недоступно: войдите с хранением только на время сессии."))
        self.cancel_all()  # requests of an earlier identity must not finish into the new one
        self._begin(_AuthJob("login", persist))

    def cancel_login(self) -> None:
        """Abandon a pending sign-in at once; its worker stops by itself and can no longer store tokens."""
        job = self._auth
        if job is None or job.kind != "login":
            return
        job.stop.set()
        self._auth, self._device = None, None
        self._notice, self._notice_failed = tr("Вход отменён."), False
        self.auth_changed.emit()

    def logout(self) -> None:
        """Sign out in a worker thread: the local session is forgotten first, then xAI is asked to revoke the tokens."""
        if self._auth is not None and self._auth.kind == "logout":
            return
        self.cancel_login()
        self.cancel_all()  # requests of the identity being left must not deliver anything
        self._begin(_AuthJob("logout"))
        self._refresh_availability()  # unavailable from this moment on

    def open_login_page(self) -> bool:
        """Open the pending sign-in's verification page in the browser; False if there is none or it did not open."""
        device = self._device
        if device is None:
            return False
        try:  # xai_oauth accepts only https pages of xAI's own hosts; nothing else is ever handed to the OS
            opened = QDesktopServices.openUrl(QUrl(validate_verification_url(device.verification_uri_complete)))
            notice = tr("Не удалось открыть браузер: откройте адрес вручную и введите код.")
        except OAuthError as exc:
            opened, notice = False, exc.message
        self._notice, self._notice_failed = ("", False) if opened else (notice, True)
        self.auth_changed.emit()
        return bool(opened)

    def _begin(self, job: _AuthJob) -> None:
        self._auth, self._device = job, None
        self._notice, self._notice_failed = "", False
        self._auth_jobs.add(job)
        job.thread = threading.Thread(target=self._run_auth, args=(job,), name=f"xai-oauth-{job.kind}", daemon=True)
        job.thread.start()
        self.auth_changed.emit()

    def _run_auth(self, job: _AuthJob) -> None:
        """Worker thread: talks to the OAuth manager and signals the controller; never touches a widget."""
        try:
            if job.kind == "logout":
                self.oauth.logout()
            else:
                if job.stop.is_set():  # cancelled before the worker got going: never touch the manager
                    raise OAuthCancelled()
                device = self.oauth.begin_login(job.stop)
                try:
                    self._auth_code.emit(job, device)
                except RuntimeError:  # the controller is gone (shutdown): nobody can approve this code any more
                    raise OAuthCancelled() from None
                self.oauth.complete_login(device, job.stop, persist=job.persist)
                job.signed_in = True
        except OAuthCancelled:
            pass
        except OAuthError as exc:
            job.error = exc.message
        except Exception as exc:  # a bug must surface as a message, not vanish in the worker thread
            what = tr("входе в") if job.kind == "login" else tr("выходе из")
            job.error = tr("Непредвиденная ошибка при {what} xAI ({value}).", what=what, value=type(exc).__name__)
        with contextlib.suppress(RuntimeError):
            self._auth_done.emit(job)

    @Slot(object, object)
    def _on_auth_code(self, job: _AuthJob, device: DeviceCode) -> None:
        if job is not self._auth:
            return  # abandoned: nobody is waiting for this code any more
        self._device = device
        self.open_login_page()

    @Slot(object)
    def _on_auth_done(self, job: _AuthJob) -> None:
        self._auth_jobs.discard(job)
        if job is self._auth:  # an abandoned job must not touch the state of a newer one
            self._auth, self._device = None, None
            self._notice, self._notice_failed = self._outcome(job)
        self._refresh_availability()  # whoever finished, the session may have changed
        self.auth_changed.emit()

    def _outcome(self, job: _AuthJob) -> tuple[str, bool]:
        """(notice, failed) for a finished sign-in/sign-out, judged by what the manager really holds."""
        if job.kind == "login":
            if job.signed_in:
                return tr("Вход выполнен."), False
            return (job.error, True) if job.error else (tr("Вход отменён."), False)
        if self.oauth.has_session:
            return " ".join(part for part in (tr("Не удалось выйти из xAI."), job.error) if part), True
        if job.error:  # the manager already forgot the session; the message tells what else did not work out
            return job.error, True
        return tr("Вы вышли из xAI: сохранённый вход удалён."), False

    def _refresh_availability(self) -> None:
        available = self.available
        if not available:
            self.cancel_all()
        if available != self._available:
            self._available = available
            self.availability_changed.emit(available)

    @Slot(object)
    def _on_job_done(self, job: GrokJob) -> None:
        self._jobs.discard(job)
        self.request_finished.emit(job)  # the request dialog shows the outcome first ...
        self._refresh_availability()  # ... then learns that a 401 or a failed refresh dropped the session meanwhile

    def _ask_closed(self, dialog: GrokAskDialog) -> None:
        if self._ask is dialog:
            self._ask = None

    def _explain_unavailable(self, parent: QWidget | None) -> None:
        box = QMessageBox(
            QMessageBox.Icon.Information,
            tr("Grok недоступен"),
            tr("{status_text}\n\nВключите помощника и войдите через xAI OAuth в настройках Grok.", status_text=self.status_text),
            QMessageBox.StandardButton.NoButton,
            parent,
        )
        settings = box.addButton(tr("Настройки…"), QMessageBox.ButtonRole.AcceptRole)
        box.addButton(tr("Закрыть"), QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() == settings:
            self.settings_dialog(parent)


_CLIENT_DISCLOSURE = (
    tr("Вход идёт по протоколу Device Authorization через публичный OAuth-клиент Grok CLI (agent, client_id {CLIENT_ID}), который xAI выпустила для своего CLI и которым пользуются сторонние оболочки; это не собственная регистрация BarForge, и на странице xAI будет указан Grok CLI. Доступ определяется вашим аккаунтом и тарифом: даже после успешного входа xAI может ответить 403, а ключ API программа не подставляет. Пароль xAI программа не видит, cookies браузера не читает.", CLIENT_ID=CLIENT_ID)
)
_STORAGE_HINT = (
    tr("Токены доступа и обновления хранятся только в защищённом хранилище ОС или в памяти программы; в настройках на диске остаются лишь флажок и название модели. Вход и выход действуют сразу, без «Сохранить».")
)


class GrokSettingsDialog(QDialog):
    """Enable switch, model name and xAI OAuth sign-in/out (device code in the browser). Nothing secret is typed here.

    Sign-in and sign-out act at once, like the old key deletion did; "Сохранить" stores only the flag and the model
    and is unavailable while a sign-in is unfinished. Closing the dialog abandons an unfinished sign-in; a sign-out
    in progress carries on in the controller.
    """

    def __init__(self, controller: GrokController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        store = controller.oauth.store
        self._phase = controller.auth.phase
        self.setWindowTitle(tr("Настройки Grok (xAI)"))
        self.setMinimumWidth(600)

        self.enabled_check = QCheckBox(tr("Включить помощника Grok (xAI)"))
        self.enabled_check.setChecked(controller.settings.enabled)

        self.model_edit = QLineEdit(controller.settings.model)
        self.model_edit.setPlaceholderText(DEFAULT_MODEL)
        reset = QPushButton(tr("По умолчанию"))
        reset.clicked.connect(lambda: self.model_edit.setText(DEFAULT_MODEL))
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel(tr("Модель:")))
        model_row.addWidget(self.model_edit, 1)
        model_row.addWidget(reset)

        self.auth_status = QLabel()
        self.auth_status.setWordWrap(True)
        self.auth_status.setTextFormat(Qt.TextFormat.PlainText)
        self.auth_notice = QLabel()
        self.auth_notice.setWordWrap(True)
        self.auth_notice.setTextFormat(Qt.TextFormat.PlainText)
        self.to_keyring = QRadioButton(tr("Защищённое хранилище ОС"))
        self.to_session = QRadioButton(tr("Только до закрытия программы"))
        self.to_keyring.setToolTip(tr("Сохранить OAuth-вход между запусками в защищённом хранилище ОС."))
        self.to_session.setToolTip(tr("Токены только в памяти: после закрытия программы потребуется войти заново."))
        (self.to_keyring if store.secure_available else self.to_session).setChecked(True)
        self.login_button = QPushButton(tr("Войти через xAI OAuth"))
        self.login_button.clicked.connect(self._login)
        self.logout_button = QPushButton(tr("Выйти"))
        self.logout_button.clicked.connect(self._logout)
        self.cancel_login_button = QPushButton(tr("Отменить вход"))
        self.cancel_login_button.clicked.connect(self._cancel_login)
        self.auth_progress = QProgressBar()
        self.auth_progress.setRange(0, 0)
        self.auth_progress.setTextVisible(False)
        self.auth_progress.setMaximumHeight(8)

        self.url_label = QLabel()
        self.url_label.setTextFormat(Qt.TextFormat.PlainText)
        self.url_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.code_label = QLabel()
        self.url_label.setWordWrap(True)
        self.url_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.code_label.setTextFormat(Qt.TextFormat.PlainText)
        self.code_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.code_label.setStyleSheet(  # the application style sheet fixes every font size, so this one must too
            "font-family: Consolas, 'Courier New', monospace; font-size: 26px; font-weight: bold;"
        )
        self.countdown = _hint("")
        self.copy_code_button = QPushButton(tr("Копировать код"))
        self.copy_code_button.clicked.connect(self._copy_code)
        self.open_page_button = QPushButton(tr("Открыть страницу входа"))
        self.open_page_button.clicked.connect(self._open_page)
        open_hint = QLabel(
            tr("Подтвердите вход на странице xAI. При необходимости введите код:")
        )
        open_hint.setWordWrap(True)
        code_row = QHBoxLayout()
        code_row.addWidget(self.copy_code_button)
        code_row.addWidget(self.open_page_button)
        code_row.addStretch(1)
        self.code_panel = QWidget()
        panel = QVBoxLayout(self.code_panel)
        panel.setContentsMargins(0, 0, 0, 0)
        panel.addWidget(open_hint)
        panel.addWidget(self.url_label)
        panel.addWidget(self.code_label)
        panel.addLayout(code_row)
        panel.addWidget(_hint(tr("Сверьте код на странице xAI с кодом в этом окне.")))
        panel.addWidget(self.countdown)

        account = QWidget()
        account_layout = QVBoxLayout(account)
        account_layout.setContentsMargins(8, 8, 8, 8)
        account_layout.setSpacing(6)
        account_layout.addWidget(self.auth_status)
        account_layout.addWidget(self.auth_notice)
        account_layout.addWidget(_hint(tr("Клиент: Grok CLI, не собственная регистрация BarForge. Подробнее — «О клиенте».")))
        storage_row = QHBoxLayout()
        storage_row.addWidget(self.to_keyring)
        storage_row.addWidget(self.to_session)
        account_layout.addLayout(storage_row)
        auth_row = QHBoxLayout()
        auth_row.addWidget(self.login_button)
        auth_row.addWidget(self.logout_button)
        auth_row.addWidget(self.cancel_login_button)
        auth_row.addStretch(1)
        account_layout.addLayout(auth_row)
        account_layout.addWidget(self.auth_progress)
        account_layout.addWidget(self.code_panel)
        account_layout.addStretch(1)
        details = QWidget()
        details_layout = QVBoxLayout(details)
        details_layout.addWidget(_hint(_CLIENT_DISCLOSURE))
        details_layout.addWidget(_hint(_STORAGE_HINT))
        details_layout.addWidget(_hint(tr("Модель должна поддерживать изображения и структурированный вывод: docs.x.ai/developers/models")))
        details_layout.addStretch(1)
        self.settings_tabs = QTabWidget()
        self.settings_tabs.addTab(account, tr("Вход"))
        self.settings_tabs.addTab(details, tr("О клиенте"))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        self.save_button.setText(tr("Сохранить"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr("Отмена"))
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.enabled_check)
        self.enabled_check.setToolTip(tr("Изображение отправляется только кнопкой «Отправить в Grok». На диске сохраняются лишь включение интеграции и название модели."))
        layout.addLayout(model_row)
        layout.addWidget(self.settings_tabs, 1)
        layout.addWidget(buttons)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(lambda: fit_window(self, (760, 640)))
        self._watching = True
        controller.auth_changed.connect(self._render_auth)
        self._render_auth()
        fit_window(self, (760, 640))

    def done(self, result: int) -> None:
        self._timer.stop()
        self._layout_timer.stop()
        if self._watching:  # once only: a closed dialog gets no callbacks and cancels nothing a newer one started
            self._watching = False
            self._controller.auth_changed.disconnect(self._render_auth)
            self._controller.cancel_login()  # an unfinished sign-in ends with the dialog; its worker winds down alone
        super().done(result)

    @Slot()
    def _render_auth(self) -> None:
        controller = self._controller
        state = controller.auth
        oauth = controller.oauth
        store = oauth.store
        idle = state.phase == "idle"
        pending = state.phase in ("requesting", "waiting")
        signed_in = oauth.has_session
        if state.phase == "requesting":
            lines = [tr("Запрашиваем код входа у xAI…")]
        elif state.phase == "waiting":
            lines = [tr("Ждём подтверждения на странице xAI. Вход ещё не выполнен.")]
        elif state.phase == "signing_out":
            lines = [tr("Выходим из xAI…")]
        else:
            lines = [oauth.status_text]
            if signed_in and not controller.settings.enabled:
                lines.append(tr("Помощник пока выключен: отметьте «Включить помощника Grok» и нажмите «Сохранить»."))
            if not signed_in and not store.secure_available:
                reason = f" ({store.secure_problem})" if store.secure_problem else ""
                lines.append(
                    tr("Защищённое хранилище ОС недоступно{reason}: вход можно держать только в памяти на время сессии; токены на диск не записываются.", reason=reason)
                )
        self.auth_status.setText("\n".join(lines))
        self.auth_notice.setText(state.notice)
        self.auth_notice.setVisible(bool(state.notice))
        self.auth_notice.setStyleSheet("color: #d32f2f;" if state.failed else "")

        free = state.phase in ("idle", "signing_out") and not signed_in  # a session already forgotten may be replaced
        self.to_keyring.setEnabled(free and store.secure_available)
        self.to_session.setEnabled(free)
        if signed_in and store.source in ("keyring", "session"):  # show where the current sign-in lives
            (self.to_keyring if store.source == "keyring" else self.to_session).setChecked(True)
        self.login_button.setEnabled(free)
        self.logout_button.setEnabled(idle and signed_in)
        self.cancel_login_button.setVisible(pending)
        self.auth_progress.setVisible(not idle)
        self.code_panel.setVisible(state.phase == "waiting")
        self.save_button.setEnabled(not pending)
        self.save_button.setToolTip(tr("Сначала завершите или отмените вход.") if pending else "")
        if state.phase == "waiting":
            self.url_label.setText(state.verification_uri)
            self.code_label.setText(state.user_code)
            self._tick()
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()
        if self._phase in ("requesting", "waiting") and idle and signed_in and not state.failed:
            self.enabled_check.setChecked(True)  # signing in signals the intent to use Grok; "Сохранить" still decides
        if self._phase != state.phase:
            if state.phase == "waiting":
                self.settings_tabs.setCurrentIndex(0)
            self._layout_timer.start(0)
        self._phase = state.phase

    @Slot()
    def _tick(self) -> None:
        left = int(self._controller.auth.expires_at - time.time())
        self.countdown.setText(
            tr("Код действует ещё {value}:{value2:02d}.", value=left // 60, value2=left % 60)
            if left > 0
            else tr("Срок действия кода истёк или истекает; ждём ответ xAI…")
        )

    def _login(self) -> None:
        try:
            self._controller.login(persist=self.to_keyring.isChecked())
        except GrokError as exc:
            QMessageBox.warning(self, tr("Вход через xAI"), exc.message)

    def _cancel_login(self) -> None:
        self._controller.cancel_login()

    def _logout(self) -> None:
        self._controller.logout()

    def _copy_code(self) -> None:
        QApplication.clipboard().setText(self._controller.auth.user_code)

    def _open_page(self) -> None:
        self._controller.open_login_page()

    def _save(self) -> None:
        try:
            self._controller.apply_settings(enabled=self.enabled_check.isChecked(), model=self.model_edit.text())
        except GrokError as exc:
            QMessageBox.warning(self, tr("Настройки Grok"), exc.message)
            return
        self.accept()


class GrokAskDialog(QDialog):
    """Ask Grok about one frame and review the answer before anything reaches the editor.

    The dialog is bound to the frame it was opened for: the image snapshot and both apply callbacks
    are fixed at construction, answers reach the callbacks only through an apply button, and closing
    the dialog cancels the request and drops its answer.
    """

    def __init__(
        self,
        controller: GrokController,
        image: Image.Image,
        caption: str,
        on_caption: Callable[[str], object],
        on_bars: Callable[[list[dict[str, float]]], object],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(tr("Grok — помощник по кадру"))
        self.setMinimumSize(760, 480)
        self._controller = controller
        self._on_caption = on_caption
        self._on_bars = on_bars
        self._caption = caption
        image.load()  # a lazily loaded file image must not be read for the first time in the worker
        self._image = image  # only read: the worker scales and JPEG-encodes it, the dialog makes a thumbnail
        self._job: GrokJob | None = None
        self._bars: tuple[BarSuggestion, ...] = ()
        self._bars_delivered = False
        self._available = True
        self._status_error = False

        self._thumb = _thumbnail(image, 380)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.preview.setMinimumSize(160, 140)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._render_preview)
        self.preview.installEventFilter(self)
        source, sent = image.size, upload_size(*image.size)
        info = tr("Кадр: {value}×{value2} px", value=source[0], value2=source[1])
        if sent != source:
            info += tr("; будет отправлен уменьшенным до {value}×{value2} px", value=sent[0], value2=sent[1])
        left = QVBoxLayout()
        left.addWidget(self.preview)
        left.addWidget(_hint(info + " (JPEG)."))
        left.addStretch(1)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._caption_page())
        self.pages.addWidget(self._bars_page())
        self.pages.addWidget(self._scene_page())
        tasks_row = QHBoxLayout()
        self._radios: list[tuple[str, QRadioButton]] = []
        for task, title, tip in _TASKS:
            radio = QRadioButton(title)
            radio.setToolTip(tip)
            self._radios.append((task, radio))
            tasks_row.addWidget(radio)
        tasks_row.addStretch(1)
        right = QVBoxLayout()
        right.addWidget(QLabel(tr("Что сделать:")))
        right.addLayout(tasks_row)
        right.addWidget(self.pages, 1)

        top = QHBoxLayout()
        top.addLayout(left)
        top.addLayout(right, 1)

        transmission_info = QLabel(
            tr("В xAI отправятся показанный кадр и капшен (модель «{value}»). Действуют квота аккаунта и правила xAI; store=false.", value=controller.settings.model)
        )
        transmission_info.setWordWrap(True)
        transmission_info.setTextFormat(Qt.TextFormat.PlainText)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(8)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.send_button = QPushButton(tr("Отправить в Grok"))
        self.send_button.clicked.connect(self._send)
        self.stop_button = QPushButton(tr("Остановить"))
        self.stop_button.clicked.connect(self._stop)
        close_button = QPushButton(tr("Закрыть"))
        close_button.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addWidget(self.send_button)
        buttons.addWidget(self.stop_button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addLayout(top, 1)
        layout.addWidget(transmission_info)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addLayout(buttons)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)

        for _task, radio in self._radios:
            radio.toggled.connect(self._task_changed)
        self._radios[0][1].setChecked(True)
        controller.request_finished.connect(self._on_finished)
        controller.availability_changed.connect(self._on_availability)
        self._listening = True
        self._show_status(tr("Нажмите «Отправить в Grok». Ничего не применяется без вашей команды."))
        self._update_controls()
        fit_window(self, (1000, 640))

    def eventFilter(self, watched, event) -> bool:
        if watched is self.preview and event.type() == QEvent.Type.Resize:
            self._preview_timer.start(0)
        return super().eventFilter(watched, event)

    # ---- pages

    def _caption_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        current = QPlainTextEdit(self._caption)
        current.setReadOnly(True)
        current.setMinimumHeight(42)
        current.setMaximumHeight(60)
        current_pages = PageNavigation(current)
        current_pages.setMaximumHeight(100)
        self.caption_edit = QPlainTextEdit()
        self.caption_edit.setPlaceholderText(tr("Предложенный капшен появится здесь; его можно править перед применением."))
        self.caption_edit.textChanged.connect(self._update_controls)
        self.apply_caption_button = QPushButton(tr("Применить капшен"))
        self.apply_caption_button.clicked.connect(self._apply_caption)
        layout.addWidget(QLabel(tr("Текущий капшен:")))
        layout.addWidget(current_pages)
        layout.addWidget(QLabel(tr("Предложение Grok (редактируется):")))
        layout.addWidget(PageNavigation(self.caption_edit), 1)
        layout.addWidget(self.apply_caption_button, 0, Qt.AlignmentFlag.AlignRight)
        return page

    def _bars_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.bars_list = QListWidget()
        self.bars_list.itemChanged.connect(self._bars_changed)
        self.bars_note = _hint("")
        self.apply_bars_button = QPushButton(tr("Применить отмеченные полосы"))
        self.apply_bars_button.clicked.connect(self._apply_bars)
        description = QLabel(tr("Предложенные полосы: снимите отметки с лишних областей."))
        description.setWordWrap(True)
        layout.addWidget(description)
        layout.addWidget(PageNavigation(self.bars_list), 1)
        layout.addWidget(self.bars_note)
        layout.addWidget(self.apply_bars_button, 0, Qt.AlignmentFlag.AlignRight)
        return page

    def _scene_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.scene_edit = QPlainTextEdit()
        self.scene_edit.setReadOnly(True)
        self.scene_edit.setPlaceholderText(tr("Нейтральное описание сцены появится здесь (только для чтения)."))
        self.scene_edit.textChanged.connect(self._update_controls)
        self.copy_scene_button = QPushButton(tr("Копировать"))
        self.copy_scene_button.clicked.connect(lambda: QApplication.clipboard().setText(self.scene_edit.toPlainText()))
        layout.addWidget(QLabel(tr("Описание сцены:")))
        layout.addWidget(PageNavigation(self.scene_edit), 1)
        layout.addWidget(self.copy_scene_button, 0, Qt.AlignmentFlag.AlignRight)
        return page

    # ---- state

    def _task(self) -> str:
        for task, radio in self._radios:
            if radio.isChecked():
                return task
        return _TASKS[0][0]

    def _checked_boxes(self) -> list[dict[str, float]]:
        return [
            bar.as_box()
            for index, bar in enumerate(self._bars)
            if self.bars_list.item(index).checkState() == Qt.CheckState.Checked
        ]

    def _update_controls(self) -> None:
        busy = self._job is not None
        ready = self._available and not busy
        self.send_button.setEnabled(ready)
        self.stop_button.setEnabled(busy)
        self.progress.setVisible(busy)
        for _task, radio in self._radios:
            radio.setEnabled(ready)
        self.apply_caption_button.setEnabled(ready and bool(self.caption_edit.toPlainText().strip()))
        self.apply_bars_button.setEnabled(ready and not self._bars_delivered and bool(self._checked_boxes()))
        self.copy_scene_button.setEnabled(bool(self.scene_edit.toPlainText()))

    def _show_status(self, text: str, *, error: bool = False) -> None:
        self._status_error = error
        self.status.setText(text)
        self.status.setStyleSheet("color: #d32f2f;" if error else "")

    @Slot(bool)
    def _task_changed(self, checked: bool) -> None:
        if checked:
            self.pages.setCurrentIndex([task for task, _title, _tip in _TASKS].index(self._task()))
            self._render_preview()

    @Slot(QListWidgetItem)
    def _bars_changed(self, _item: QListWidgetItem) -> None:
        self._render_preview()
        self._update_controls()

    def _render_preview(self) -> None:
        pixmap = QPixmap.fromImage(self._thumb)
        if self._task() == "bars" and self._bars:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            width, height = pixmap.width(), pixmap.height()
            for index, bar in enumerate(self._bars):
                checked = self.bars_list.item(index).checkState() == Qt.CheckState.Checked
                color = QColor(220, 40, 40) if checked else QColor(140, 140, 140)
                fill = QColor(color)
                fill.setAlpha(80 if checked else 30)
                rect = QRectF(bar.x * width, bar.y * height, bar.width * width, bar.height * height)
                painter.setPen(QPen(color, 2))
                painter.setBrush(fill)
                painter.drawRect(rect)
                painter.drawText(rect.topLeft() + QPointF(3, 13), str(index + 1))
            painter.end()
        self.preview.setPixmap(pixmap.scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))

    # ---- request lifecycle

    def _send(self) -> None:
        if self._job is not None or not self._available:
            return
        try:
            self._job = self._controller.start(self._task(), self._image, self._caption)
        except GrokError as exc:
            self._show_status(exc.message, error=True)
            return
        self._show_status(tr("Запрос отправлен. Ожидание ответа Grok…"))
        self._update_controls()

    def _stop(self) -> None:
        self._cancel_job()
        self._show_status(tr("Запрос остановлен; ответ будет проигнорирован."))
        self._update_controls()

    def _cancel_job(self) -> None:
        job, self._job = self._job, None
        if job is not None:
            job.cancel()

    @Slot(object)
    def _on_finished(self, job: GrokJob) -> None:
        if job is not self._job:  # an older, cancelled or foreign request: its answer is not ours to show
            return
        self._job = None
        if job.error is not None:
            self._show_status(job.error, error=True)
        elif job.result is not None:
            self._show_result(job.result)
        elif job.cancelled:  # stopped by the controller (sign-out, shutdown), not by this dialog
            self._show_status(tr("Запрос остановлен; ответ будет проигнорирован."))
        self._update_controls()

    @Slot(bool)
    def _on_availability(self, available: bool) -> None:
        self._available = available
        if not available:
            stopped = self._job is not None
            self._cancel_job()
            if stopped or not self._status_error:  # an error just shown (e.g. "sign in again") already says why
                self._show_status(
                    tr("Grok отключён или вход в xAI недействителен: запрос остановлен, отправка недоступна."), error=True
                )
        self._update_controls()

    def done(self, result: int) -> None:
        self._cancel_job()
        if self._listening:  # a closed dialog gets no more controller callbacks, even before Qt has deleted it
            self._listening = False
            with contextlib.suppress(RuntimeError):  # the controller may already be gone
                self._controller.request_finished.disconnect(self._on_finished)
                self._controller.availability_changed.disconnect(self._on_availability)
        super().done(result)

    # ---- results

    def _show_result(self, result: GrokResult) -> None:
        if result.task == "caption":
            self.caption_edit.setPlainText(result.text)
        elif result.task == "scene":
            self.scene_edit.setPlainText(result.text)
        else:
            self._fill_bars(result)
        parts = [tr("Готово. Модель: {value}.", value=result.model or self._controller.settings.model)]
        if result.tokens_in or result.tokens_out:
            parts.append(tr("Токены: вход {value}, выход {value2}.", value=result.tokens_in, value2=result.tokens_out))
        parts.extend(result.notes)
        if result.task == "bars":
            parts.append(tr("Полос: {value}.", value=len(result.bars)) if result.bars else tr("Grok не нашёл областей для полос."))
        self._show_status(" ".join(parts))
        self._render_preview()

    def _fill_bars(self, result: GrokResult) -> None:
        self._bars = result.bars
        self._bars_delivered = False
        self.bars_list.blockSignals(True)
        self.bars_list.clear()
        for index, bar in enumerate(result.bars, 1):
            text = (
                tr("{index}. {value} — x {value2:.1f}%, y {value3:.1f}%, ширина {value4:.1f}%, высота {value5:.1f}%", index=index, value=bar.label or tr("без названия"), value2=bar.x * 100, value3=bar.y * 100, value4=bar.width * 100, value5=bar.height * 100)
            )
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.bars_list.addItem(item)
        self.bars_list.blockSignals(False)
        self.bars_note.setText(" ".join(result.notes))

    def _apply_caption(self) -> None:
        text = self.caption_edit.toPlainText().strip()
        if text:
            self._deliver(lambda: self._on_caption(text), tr("Капшен передан в редактор."))

    def _apply_bars(self) -> None:
        boxes = self._checked_boxes()
        if not boxes:
            self._show_status(tr("Не отмечено ни одной полосы."), error=True)
            return
        if self._deliver(lambda: self._on_bars(boxes), tr("Полосы переданы в редактор: {value}.", value=len(boxes))):
            self._bars_delivered = True  # one delivery per answer, so a second click cannot duplicate the bars
            self._update_controls()

    def _deliver(self, call: Callable[[], object], done_message: str) -> bool:
        if call() is False:
            self._show_status(
                tr("Редактор не принял результат (кадр или документ изменились). Ничего не применено."), error=True
            )
            return False
        self._show_status(done_message)
        return True
