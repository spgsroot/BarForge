"""Qt side of the local neural auto-censor (Russian UI): AutoCensorDialog.

API used by the main window:

    dialog = AutoCensorDialog(image, color, on_apply, parent=self)    # engine=None: one shared LocalCensorEngine
    dialog.open()    # window-modal, returns at once; keep a reference until dialog.finished is emitted

``image`` is the clean crop to analyse; the dialog works on a private copy. ``on_apply(bars, replace)`` runs only
when the user presses an apply button: ``bars`` are ``barforge.models.Bar`` in pixel coordinates of ``image`` (origin:
its top-left corner, so the caller shifts them by the crop offset) and ``replace`` is the state of the "replace
existing bars" check box. A callback returning ``False`` means "not applied" (stale document); any other return value
means applied. The dialog deletes itself when it is closed.

The models run once per search with the thresholds at their floor and every category on; thresholds, categories, bar
colour, thickness with its deviation range, spacing, angle, placement zone and per-region ticks are applied live
through ``bars_from_detections``. Only "tiles" and "rotations" change the search itself and need a new run.

The window fits its screen without scroll bars: settings sit on tabs, the list of regions turns pages
(``barforge.ui.PageNavigation``) and the preview takes the remaining space.
"""

from __future__ import annotations

import contextlib
import dataclasses
import secrets
import threading
from collections.abc import Callable

import shiboken6
from PIL import Image
from PySide6.QtCore import QCoreApplication, QObject, QPointF, QRect, QRunnable, QSize, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .i18n import current_language, tr
from .autocensor import (
    MAX_THICKNESS,
    MAX_VARIATION,
    CancelledError,
    CensorOptions,
    Detection,
    DetectionResult,
    LocalCensorEngine,
    bars_from_detections,
)
from .models import BAR_COLORS, Bar, Rect, bar_polygon
from .ui import PageNavigation, fit_window

__all__ = ["AutoCensorDialog"]

_PREVIEW_SIDE = 1280  # longest side of the raster the GUI thread keeps for the preview
_FLOOR = 0.05  # search threshold of the one model run; the user's thresholds (>= 5 %) filter its result live
_CATEGORIES = {
    "genital": tr("Интимная область"),
    "anus": tr("Интимная область (сзади)"),
    "nipple": tr("Соски"),
    "act": tr("Интимное действие"),
}
_COLOR_NAMES = {"#ffffff": tr("Белые полосы"), "#000000": tr("Чёрные полосы")}
_PRESETS = ((tr("Обычный (24 %)"), 24), (tr("Чувствительный для манги (12 %)"), 12))
_SCALE_PROFILES = ((tr("Компактная зона (90 %)"), 90), (tr("Базовая зона (100 %)"), 100), (tr("Широкая зона (115 %)"), 115))
_ERROR_STYLE = "color: #ff7b7b;"
_PREFERRED_SIZE = (1100, 780)  # on a roomy screen; fit_window shrinks it to the real one


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setStyleSheet("color: gray;")
    return label


def _plural(count: int, one: str, few: str, many: str) -> str:
    """Plural form of a noun for ``count``: English has two forms, Russian three."""
    if current_language() == "en":
        return one if count == 1 else few
    if count % 10 == 1 and count % 100 != 11:
        return one
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return few
    return many


def _spin(low: int, high: int, value: int, suffix: str, tip: str) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(low, high)
    spin.setValue(min(max(value, low), high))
    spin.setSuffix(suffix)
    spin.setToolTip(tip)
    spin.setKeyboardTracking(False)  # one change per committed value, not one per typed digit
    return spin


def _check(text: str, checked: bool, tip: str) -> QCheckBox:
    box = QCheckBox(text)
    box.setChecked(checked)
    box.setToolTip(tip)
    return box


def _grid() -> QGridLayout:
    """Label / field / label / field columns; spare width goes to a last empty column, so fields stay compact."""
    grid = QGridLayout()
    grid.setHorizontalSpacing(8)
    grid.setVerticalSpacing(6)
    grid.setColumnStretch(4, 1)
    return grid


def _cell(grid: QGridLayout, row: int, column: int, text: str, field: QWidget) -> None:
    """``text: field`` in the left (column 0) or the right (column 2) half of a settings grid row."""
    grid.addWidget(QLabel(text), row, column)
    grid.addWidget(field, row, column + 1)


def _page(grid: QGridLayout, note: str) -> QWidget:
    """A tab page: the settings grid, a short note below it and the spare height underneath."""
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.addLayout(grid)
    layout.addWidget(_hint(note))
    layout.addStretch(1)
    return page


def _sync_combo(combo: QComboBox, value: int) -> None:
    """Show the profile whose value equals ``value``, or the last (custom) entry when none does."""
    index = combo.findData(value)
    combo.setCurrentIndex(index if index >= 0 else combo.count() - 1)


def _preview_raster(image: Image.Image, side: int) -> QImage:
    """Small RGB raster of ``image`` for the preview; the full-size pixels are only read."""
    scale = min(1.0, side / max(image.size))
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    small = image if size == image.size else image.resize(size, Image.Resampling.BILINEAR, reducing_gap=2.0)
    data = small.tobytes()  # QImage borrows the buffer, so it must outlive copy()
    return QImage(data, small.width, small.height, small.width * 3, QImage.Format.Format_RGB888).copy()


def _admitted(detection: Detection, options: CensorOptions) -> bool:
    """The engine's own rule: do the thresholds and category switches in ``options`` keep this detection?"""
    if detection.kind == "act":
        return options.include_acts and detection.confidence >= options.act_confidence
    if detection.kind == "anus" and not options.include_anus:
        return False
    if detection.kind == "nipple" and not options.include_nipples:
        return False
    return detection.confidence >= options.confidence


def _failure_text(exc: Exception) -> str:
    """Russian message for a failed analysis; the engine words its own RuntimeError/ValueError for the user."""
    text = str(exc).strip()
    if isinstance(exc, MemoryError):
        return tr("Не хватило памяти для анализа. Отключите проверку поворотов и разбиение на фрагменты и повторите.")
    if isinstance(exc, (RuntimeError, ValueError)) and text:
        return text
    return tr("Не удалось выполнить анализ ({value}){value2}.", value=type(exc).__name__, value2=': ' + text if text else '')


# ---------------------------------------------------------------- background analysis


class _Job:
    """One analysis in flight.

    A plain object, not a QObject: the worker fills in result/error and the hub announces completion, so no
    Qt object is ever touched or destroyed from the worker thread and a closed dialog cannot be signalled.
    """

    def __init__(self, engine: LocalCensorEngine, image: Image.Image, options: CensorOptions) -> None:
        self.engine = engine
        self.image: Image.Image | None = image  # the dialog's private snapshot; analyze() only reads it
        self.options = options
        self.result: DetectionResult | None = None
        self.error: str | None = None
        self._stop = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._stop.is_set()

    def cancel(self) -> None:
        self._stop.set()


class _Runner(QRunnable):
    def __init__(self, job: _Job, hub: _Hub) -> None:
        super().__init__()
        self._job = job
        self._hub = hub

    def run(self) -> None:
        job = self._job
        try:
            if not job.cancelled:
                job.result = job.engine.analyze(
                    job.image, job.options, progress=self._report, cancelled=lambda: job.cancelled
                )
        except CancelledError:
            pass  # the dialog asked for it and has already moved on
        except Exception as exc:  # a bug must surface as a message, not vanish in the pool thread
            if not job.cancelled:
                job.error = _failure_text(exc)
        finally:
            job.image = None
            with contextlib.suppress(RuntimeError):  # the hub is already gone at shutdown
                self._hub.finished.emit(job)

    def _report(self, done: int, total: int) -> None:
        with contextlib.suppress(RuntimeError):
            self._hub.progressed.emit(self._job, int(done), int(total))


class _Hub(QObject):
    """Pool and signals that outlive every dialog (one per application).

    Dialogs connect to ``progressed``/``finished`` and ignore jobs that are not theirs, so a worker that
    ends after its dialog closed only emits into this object, never into a deleted one.
    """

    progressed = Signal(object, int, int)  # job, steps done, steps total
    finished = Signal(object)  # job; delivered for every job, cancelled and failed ones too

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)  # one analysis at a time; onnxruntime brings its own CPU threads
        self._jobs: set[_Job] = set()
        self.finished.connect(self._forget)

    def start(self, job: _Job) -> None:
        self._jobs.add(job)
        self._pool.start(_Runner(job, self))

    @Slot(object)
    def _forget(self, job: _Job) -> None:
        self._jobs.discard(job)

    @Slot()
    def shutdown(self) -> None:
        """Cancel running analyses and give the worker a moment to stop (wired to application exit)."""
        for job in tuple(self._jobs):
            job.cancel()
        self._pool.waitForDone(3000)


_hub_instance: _Hub | None = None
_engine_instance: LocalCensorEngine | None = None


def _hub() -> _Hub:
    global _hub_instance
    if _hub_instance is None or not shiboken6.isValid(_hub_instance):
        app = QCoreApplication.instance()
        _hub_instance = _Hub(app)
        if app is not None:
            app.aboutToQuit.connect(_hub_instance.shutdown)
    return _hub_instance


def _shared_engine() -> LocalCensorEngine:
    """The engine used when the caller passes none: one set of loaded sessions for every dialog."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LocalCensorEngine()
    return _engine_instance


# ---------------------------------------------------------------- widgets


class _Preview(QWidget):
    """Scaled source with the same thin rotated line polygons as the exported image."""

    def __init__(self, raster: QImage, size: tuple[int, int], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(240, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap = QPixmap.fromImage(raster)
        self._size = size
        self._bars: list[Bar] = []
        self._marker: Rect | None = None
        self._outlines = True

    def sizeHint(self) -> QSize:
        return QSize(520, 440)

    def set_overlay(self, bars: list[Bar], marker: Rect | None, outlines: bool) -> None:
        self._bars, self._marker, self._outlines = bars, marker, outlines
        self.update()

    def release(self) -> None:
        self._pixmap = QPixmap()
        self._bars = []
        self.update()

    def _target(self) -> QRect:
        width, height = self._size
        scale = min((self.width() - 2) / width, (self.height() - 2) / height)
        w, h = max(1, int(width * scale)), max(1, int(height * scale))
        return QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def _device_rect(self, target: QRect, rect: Rect | Bar) -> QRect:
        full_w, full_h = self._size
        x0 = target.x() + rect.x * target.width() // full_w
        y0 = target.y() + rect.y * target.height() // full_h
        x1 = target.x() - (-(rect.x + rect.width) * target.width() // full_w)  # ceiling division
        y1 = target.y() - (-(rect.y + rect.height) * target.height() // full_h)
        return QRect(x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    def paintEvent(self, _event: object) -> None:
        if self._pixmap.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        target = self._target()
        painter.drawPixmap(target, self._pixmap)
        painter.setPen(QPen(QColor("#46566a"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRect(target.x() - 1, target.y() - 1, target.width() + 1, target.height() + 1))
        painter.setClipRect(target)
        for bar in self._bars:
            polygon = QPolygonF([
                QPointF(target.x() + x * target.width() / self._size[0],
                        target.y() + y * target.height() / self._size[1])
                for x, y in bar_polygon(bar)
            ])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(*BAR_COLORS.get(str(bar.color).lower(), (255, 255, 255))))
            painter.drawPolygon(polygon)
            if self._outlines:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(229, 72, 77), 1))
                painter.drawPolygon(polygon)
        if self._marker is not None:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(76, 201, 240), 2, Qt.PenStyle.DashLine))
            painter.drawRect(self._device_rect(target, self._marker).adjusted(1, 1, -1, -1))
        painter.end()


class AutoCensorDialog(QDialog):
    """Find sensitive regions with the local neural models, review them, hand the bars to the editor.

    The dialog is bound to the frame it was opened for: the pixels are copied at construction, the apply
    callback is fixed, results reach it only through an apply button, and closing the dialog cancels a running
    analysis and drops its outcome. The models run in a pool thread; the GUI thread only builds one small
    preview raster.
    """

    def __init__(
        self,
        image: Image.Image,
        color: str,
        on_apply: Callable[[list[Bar], bool], object],
        parent: QWidget | None = None,
        engine: LocalCensorEngine | None = None,
        variation_min: int = 3,
        variation_max: int = 8,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(tr("Автоцензура — локальная нейросеть"))
        self._on_apply = on_apply
        self._engine = engine if engine is not None else _shared_engine()
        # Private copy: the worker reads it on every run while the caller may reuse or close ``image``.
        self._snapshot: Image.Image | None = image.convert("RGB")
        self._size = self._snapshot.size
        self._job: _Job | None = None
        self._detections: tuple[Detection, ...] = ()  # everything the last run found, down to the floor
        self._wanted: list[bool] = []  # the user's tick per detection; thresholds only hide, never untick
        self._shown: list[int] = []  # rows of ``_detections`` the thresholds and categories currently admit
        self._run_options: CensorOptions | None = None  # what the shown detections were searched with
        self._bars: list[Bar] = []  # what Apply would deliver right now
        self._delivered: tuple | None = None
        self._seed = secrets.randbits(64)  # one stable width pattern throughout this review

        color = color.lower() if color.lower() in BAR_COLORS else "#ffffff"
        defaults = CensorOptions(color=color, seed=self._seed, variation_min=variation_min, variation_max=variation_max)

        # ---- what to keep (applied live to the finished search)
        self.preset_combo = QComboBox()
        for title, percent in _PRESETS:
            self.preset_combo.addItem(title, percent)
        self.preset_combo.addItem(tr("Свой порог"), None)
        self.preset_combo.setToolTip(tr("Пресет меняет только порог областей; порог действий остаётся прежним."))
        self.confidence_spin = _spin(
            5, 95, round(defaults.confidence * 100), " %",
            tr("Минимальная уверенность нейросети для областей. Ниже — чувствительнее, но больше лишних срабатываний."),
        )
        self.act_confidence_spin = _spin(
            5, 95, round(defaults.act_confidence * 100), " %", tr("Порог подсказок сцен для локального поиска, не область рисования.")
        )
        self.acts_check = _check(
            tr("Уточнять поиск по сценам"), defaults.include_acts, tr("Боксы сцен не закрашиваются: они только подсказывают, где искать анатомию.")
        )
        self.nipples_check = _check(tr("Закрывать соски"), defaults.include_nipples, tr("По умолчанию выключено."))
        self.anus_check = _check(tr("Закрывать анус (дополнительная модель)"), defaults.include_anus,
                                 tr("По умолчанию выключено; результаты дополнительной модели требуют проверки."))

        # ---- bar style (applied live to the finished search)
        self.color_combo = QComboBox()
        for value in BAR_COLORS:
            self.color_combo.addItem(_COLOR_NAMES.get(value, value), value)
        self.color_combo.setCurrentIndex(max(0, self.color_combo.findData(color)))
        self.padding_spin = _spin(
            0, 100, round(defaults.padding * 100), " %", tr("Запас с каждой стороны, в процентах от размера области.")
        )
        self.scale_combo = QComboBox()
        for title, percent in _SCALE_PROFILES:
            self.scale_combo.addItem(title, percent)
        self.scale_combo.addItem(tr("Свой размер"), None)
        self.scale_combo.setToolTip(tr("Меняет только зону размещения; толщина и шаг линий остаются прежними."))
        self.scale_spin = _spin(
            10, 400, round(defaults.size_scale * 100), " %", tr("Масштаб зоны, в которой размещаются тонкие линии.")
        )
        self.thickness_spin = _spin(
            1, MAX_THICKNESS, defaults.thickness, " px",
            tr("Базовая толщина; отклонение каждой полосы от неё задают границы «Отклонение от» и «до»."),
        )
        deviation_tip = (
            tr("Отклонение толщины каждой полосы от базовой: целое число px от нижней до верхней границы включительно, знак (±) случайный.\nРавные границы дают одно и то же отклонение, 0 и 0 отключают его. Если одну границу сдвинуть за другую, вторая сдвинется следом.")
        )
        self.variation_min_spin = _spin(0, MAX_VARIATION, defaults.variation_min, " px", deviation_tip)
        self.variation_max_spin = _spin(0, MAX_VARIATION, defaults.variation_max, " px", deviation_tip)
        for spin in (self.variation_min_spin, self.variation_max_spin):
            spin.setPrefix("±")
        self.spacing_spin = _spin(
            defaults.thickness + 1, 512, defaults.spacing, " px", tr("Шаг между центрами линий, измеряется поперёк них.")
        )
        self.angle_spin = _spin(
            -180, 180, round(defaults.angle), "°", tr("Ручной угол: применяется только при выключенном автоугле.")
        )
        self.auto_angle_check = _check(tr("Поперёк органа (автоугол)"), defaults.auto_angle,
                                      tr("Свой угол для каждой уточнённой анатомической области вместо одной сетки на всю страницу."))
        self.angle_spin.setEnabled(not defaults.auto_angle)

        # ---- the search itself (needs a new run)
        self.tiles_check = _check(
            tr("Разбивать на фрагменты (точнее, медленнее)"),
            defaults.tiles,
            tr("Дополнительно анализировать перекрывающиеся фрагменты кадра: находятся мелкие области.\nУчитывается при следующем нажатии «Найти области»."),
        )
        self.rotations_check = _check(
            tr("Проверять повороты (точнее, медленнее)"),
            defaults.rotations,
            tr("Анализировать и повёрнутые копии кадра: помогает с нестандартными позами и рисунком.\nУчитывается при следующем нажатии «Найти области»."),
        )
        self._run_widgets = (self.tiles_check, self.rotations_check, self.acts_check, self.act_confidence_spin)

        # ---- results
        self.detections_list = QListWidget()
        self.detections_list.setEnabled(False)
        self.detections_pages = PageNavigation(self.detections_list)  # page buttons instead of scroll bars
        self.select_all_button = QPushButton(tr("Отметить все"))
        self.clear_button = QPushButton(tr("Снять все"))
        self.summary_label = _hint("")
        self.replace_check = _check(
            tr("Заменить существующие полосы (иначе добавить к ним)"),
            False,
            tr("Включено: все полосы документа заменяются результатом; выключено: результат добавляется."),
        )

        # ---- preview, progress, status
        self.preview = _Preview(_preview_raster(self._snapshot, _PREVIEW_SIDE), self._size)
        self.outline_check = _check(
            tr("Показывать контуры полос на превью"), False, tr("Только на превью: контуры помогают видеть белые линии на белом.")
        )
        self.progress = QProgressBar()
        self.progress.setFormat(tr("шаг %v из %m"))
        self.progress.setVisible(False)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.setMinimumHeight(self.fontMetrics().lineSpacing() * 2)
        self.rerun_hint = _hint(tr("Параметры поиска изменены после анализа. Нажмите «Найти области», чтобы учесть их."))
        self.rerun_hint.setVisible(False)

        self.run_button = QPushButton(tr("Найти области"))
        self.stop_button = QPushButton(tr("Остановить"))
        self.apply_button = QPushButton(tr("Применить"))
        self.apply_close_button = QPushButton(tr("Применить и закрыть"))
        self.close_button = QPushButton(tr("Закрыть"))

        # ---- layout
        left = QVBoxLayout()
        left.addWidget(
            _hint(
                tr("Кадр: {value}×{value2} px. Анализ идёт локально, на этом компьютере: изображение никуда не отправляется. Нейросеть может пропустить область или принять лишнее за область — проверьте превью. Ничего не применяется без вашей команды.", value=self._size[0], value2=self._size[1])
            )
        )
        left.addWidget(self.preview, 1)
        left.addWidget(self.outline_check)

        keep_grid = _grid()
        keep_grid.addWidget(QLabel(tr("Пресет:")), 0, 0)
        keep_grid.addWidget(self.preset_combo, 0, 1, 1, 3)
        _cell(keep_grid, 1, 0, tr("Порог областей:"), self.confidence_spin)
        _cell(keep_grid, 1, 2, tr("Порог сцен:"), self.act_confidence_spin)
        keep_grid.addWidget(self.acts_check, 2, 0, 1, 4)
        keep_grid.addWidget(self.nipples_check, 3, 0, 1, 4)
        keep_grid.addWidget(self.anus_check, 4, 0, 1, 4)
        keep_page = _page(keep_grid, tr("Порог анатомии и категории — сразу. Подсказки сцен — при новом поиске."))
        bars_grid = _grid()
        _cell(bars_grid, 0, 0, tr("Цвет:"), self.color_combo)
        _cell(bars_grid, 0, 2, tr("Запас вокруг:"), self.padding_spin)
        _cell(bars_grid, 1, 0, tr("Размер зоны:"), self.scale_combo)
        _cell(bars_grid, 1, 2, tr("Масштаб зоны:"), self.scale_spin)
        _cell(bars_grid, 2, 0, tr("Базовая толщина:"), self.thickness_spin)
        _cell(bars_grid, 2, 2, tr("Шаг линий:"), self.spacing_spin)
        _cell(bars_grid, 3, 0, tr("Отклонение от:"), self.variation_min_spin)
        _cell(bars_grid, 3, 2, tr("до:"), self.variation_max_spin)
        _cell(bars_grid, 4, 0, tr("Угол вручную:"), self.angle_spin)
        bars_grid.addWidget(self.auto_angle_check, 4, 2, 1, 2)
        bars_page = _page(
            bars_grid,
            tr("Отклонение толщины — целое число px между границами «от» и «до» включительно, знак случайный; 0 и 0 — без отклонения. Минимум 1 px; на малых областях и при малом шаге отклонение ограничено, чтобы сохранялись промежутки."),
        )
        list_buttons = QHBoxLayout()
        list_buttons.addWidget(self.select_all_button)
        list_buttons.addWidget(self.clear_button)
        list_buttons.addStretch(1)
        review_page = QWidget()
        review = QVBoxLayout(review_page)
        review.addWidget(QLabel(tr("Найденные области (снимите отметку с лишних):")))
        review.addWidget(self.detections_pages, 1)
        review.addLayout(list_buttons)
        review.addWidget(self.summary_label)
        self.tabs = QTabWidget()
        self.tabs.addTab(keep_page, tr("Что закрывать"))
        self.tabs.addTab(bars_page, tr("Полосы"))
        self.tabs.addTab(review_page, tr("Области"))
        self._keep_page, self._review_page = keep_page, review_page

        columns = QHBoxLayout()
        columns.addLayout(left, 1)
        columns.addWidget(self.tabs)
        run_row = QHBoxLayout()
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.stop_button)
        run_row.addSpacing(12)
        run_row.addWidget(self.tiles_check)
        run_row.addWidget(self.rotations_check)
        run_row.addStretch(1)
        apply_row = QHBoxLayout()
        apply_row.addWidget(self.replace_check)
        apply_row.addStretch(1)
        apply_row.addWidget(self.apply_button)
        apply_row.addWidget(self.apply_close_button)
        apply_row.addWidget(self.close_button)
        layout = QVBoxLayout(self)
        layout.addLayout(columns, 1)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(self.rerun_hint)
        layout.addLayout(run_row)
        layout.addLayout(apply_row)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)

        # ---- wiring
        self.thickness_spin.valueChanged.connect(lambda value: self.spacing_spin.setMinimum(value + 1))
        for spin in (
            self.confidence_spin, self.act_confidence_spin, self.padding_spin, self.scale_spin,
            self.thickness_spin, self.spacing_spin, self.angle_spin
        ):
            spin.valueChanged.connect(self._options_changed)
        for spin in (self.variation_min_spin, self.variation_max_spin):
            spin.valueChanged.connect(lambda _value, moved=spin: self._variation_changed(moved))
        for check in (self.acts_check, self.nipples_check, self.anus_check, self.auto_angle_check, self.tiles_check, self.rotations_check):
            check.toggled.connect(self._options_changed)
        self.auto_angle_check.toggled.connect(lambda checked: self.angle_spin.setEnabled(not checked))
        self.color_combo.currentIndexChanged.connect(self._options_changed)
        self.preset_combo.activated.connect(lambda index: self._pick(self.preset_combo, self.confidence_spin, index))
        self.scale_combo.activated.connect(lambda index: self._pick(self.scale_combo, self.scale_spin, index))
        self.detections_list.itemChanged.connect(self._item_changed)
        self.detections_list.currentRowChanged.connect(self._render_preview)
        self.select_all_button.clicked.connect(lambda: self._set_shown(True))
        self.clear_button.clicked.connect(lambda: self._set_shown(False))
        self.outline_check.toggled.connect(self._render_preview)
        self.replace_check.toggled.connect(self._refresh)
        self.run_button.clicked.connect(self._run)
        self.stop_button.clicked.connect(self._stop)
        self.apply_button.clicked.connect(lambda: self._apply(close=False))
        self.apply_close_button.clicked.connect(lambda: self._apply(close=True))
        self.close_button.clicked.connect(self.reject)
        hub = _hub()
        hub.progressed.connect(self._on_progress)
        hub.finished.connect(self._on_finished)

        self._sync_combos()
        self._show_status(tr("Задайте параметры и нажмите «Найти области». Ничего не применяется без вашей команды."))
        self._recompute()
        fit_window(self, _PREFERRED_SIZE)
        self.run_button.setFocus()

    # ---- state

    @property
    def busy(self) -> bool:
        """True while an analysis is running."""
        return self._job is not None

    @property
    def bars(self) -> list[Bar]:
        """The bars an apply button would deliver right now (copies)."""
        return [dataclasses.replace(bar) for bar in self._bars]

    def _options(self) -> CensorOptions:
        return CensorOptions(
            color=self.color_combo.currentData(),
            confidence=self.confidence_spin.value() / 100,
            act_confidence=self.act_confidence_spin.value() / 100,
            padding=self.padding_spin.value() / 100,
            thickness=self.thickness_spin.value(),
            variation_min=self.variation_min_spin.value(),
            variation_max=self.variation_max_spin.value(),
            spacing=self.spacing_spin.value(),
            angle=float(self.angle_spin.value()),
            auto_angle=self.auto_angle_check.isChecked(),
            size_scale=self.scale_spin.value() / 100,
            include_nipples=self.nipples_check.isChecked(),
            include_anus=self.anus_check.isChecked(),
            include_acts=self.acts_check.isChecked(),
            tiles=self.tiles_check.isChecked(),
            rotations=self.rotations_check.isChecked(),
            seed=self._seed,
        )

    def _delivery_key(self) -> tuple:
        bars = tuple((bar.x, bar.y, bar.width, bar.height, bar.color, bar.angle) for bar in self._bars)
        return bars, self.replace_check.isChecked()

    def _show_status(self, text: str, *, error: bool = False) -> None:
        self.status.setText(text)
        self.status.setStyleSheet(_ERROR_STYLE if error else "")

    def _refresh(self, *_args: object) -> None:
        busy = self._job is not None
        for widget in self._run_widgets:
            widget.setEnabled(not busy)
        reviewable = bool(self._shown) and not busy
        self.detections_list.setEnabled(bool(self._detections) and not busy)
        self.select_all_button.setEnabled(reviewable)
        self.clear_button.setEnabled(reviewable)
        self.run_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        self.progress.setVisible(busy)
        count = len(self._bars)
        ready = not busy and count > 0 and self._delivered != self._delivery_key()
        self.apply_button.setEnabled(ready)
        self.apply_close_button.setEnabled(ready)
        self.apply_button.setText(tr("Применить ({count})", count=count) if count else tr("Применить"))

    def _sync_combos(self) -> None:
        _sync_combo(self.preset_combo, self.confidence_spin.value())
        _sync_combo(self.scale_combo, self.scale_spin.value())

    def _pick(self, combo: QComboBox, spin: QSpinBox, index: int) -> None:
        """The user chose a profile: copy its value into ``spin`` (the custom entry changes nothing)."""
        value = combo.itemData(index)
        if value is not None:
            spin.setValue(value)
        self._sync_combos()

    def _options_changed(self, *_args: object) -> None:
        self._sync_combos()
        self._recompute()

    def _variation_changed(self, moved: QSpinBox) -> None:
        """Keep the deviation bounds ordered: moving one past the other carries the other along."""
        low, high = self.variation_min_spin, self.variation_max_spin
        if low.value() > high.value():
            follower = high if moved is low else low
            follower.blockSignals(True)  # one recompute below, never one for a half-ordered pair
            try:
                follower.setValue(moved.value())
            finally:
                follower.blockSignals(False)
        self._options_changed()

    # ---- review (never runs the models)

    def _recompute(self) -> None:
        """Rebuild the visible list, the bars and the preview from the cached search and the current options."""
        options: CensorOptions | None
        try:
            options = self._options()
            self._shown = [row for row, found in enumerate(self._detections) if _admitted(found, options)]
            self._sync_items()
            chosen = [self._detections[row] for row in self._shown if self._wanted[row]]
            self._bars = list(bars_from_detections(chosen, *self._size, options)) if chosen else []
        except ValueError as exc:
            options = None
            self._bars = []
            self._show_status(tr("Параметры недопустимы: {exc}", exc=exc), error=True)
        total = len(self._detections)
        if total:
            ticked = sum(1 for row in self._shown if self._wanted[row])
            summary = tr("Отмечено: {ticked} из {value}. Полос к применению: {value2}.", ticked=ticked, value=len(self._shown), value2=len(self._bars))
            if self._bars:
                summary += tr(" Толщина: {value}–{value2} px.", value=min(bar.height for bar in self._bars), value2=max(bar.height for bar in self._bars))
            if len(self._shown) != total:
                summary += tr(" Скрыто порогами и категориями: {value}.", value=total - len(self._shown))
            self.summary_label.setText(summary)
        else:
            self.summary_label.setText("")
        run = self._run_options
        self.rerun_hint.setVisible(
            options is not None
            and run is not None
            and (options.tiles != run.tiles or options.rotations != run.rotations
                 or options.include_acts != run.include_acts or options.act_confidence != run.act_confidence)
        )
        self._render_preview()
        self._refresh()

    def _sync_items(self) -> None:
        """Show only the admitted detections, numbered in order, each ticked as the user wants."""
        lst = self.detections_list
        shown = set(self._shown)
        number = 0
        lst.blockSignals(True)
        try:
            for row, found in enumerate(self._detections):
                item = lst.item(row)
                item.setHidden(row not in shown)
                if row in shown:
                    number += 1
                    rect = found.rect
                    category = _CATEGORIES.get(found.kind, tr("Область"))
                    score = f"{found.confidence * 100:.0f} %"
                    item.setText(f"{number}. {category} — {score}, {rect.width}×{rect.height} px")
                item.setCheckState(Qt.CheckState.Checked if self._wanted[row] else Qt.CheckState.Unchecked)
        finally:
            lst.blockSignals(False)

    @Slot(QListWidgetItem)
    def _item_changed(self, item: QListWidgetItem) -> None:
        row = self.detections_list.row(item)
        if 0 <= row < len(self._wanted):
            self._wanted[row] = item.checkState() == Qt.CheckState.Checked
            self._recompute()

    def _set_shown(self, wanted: bool) -> None:
        for row in self._shown:
            self._wanted[row] = wanted
        self._recompute()

    def _render_preview(self, *_args: object) -> None:
        row = self.detections_list.currentRow()
        marker = None
        if row in self._shown:
            marker = self._detections[row].rect
        self.preview.set_overlay(self._bars, marker, self.outline_check.isChecked())

    # ---- analysis lifecycle

    def _run(self) -> None:
        if self._job is not None or self._snapshot is None:
            return
        try:
            options = self._options()
        except ValueError as exc:
            self._show_status(tr("Параметры недопустимы: {exc}", exc=exc), error=True)
            return
        self._clear_results()
        # Cache every anatomical category at the floor; scene hints retain their own search controls.
        search = dataclasses.replace(
            options, confidence=_FLOOR, include_nipples=True, include_anus=True
        )
        job = _Job(self._engine, self._snapshot, search)
        _hub().start(job)
        self._job = job
        self.progress.setRange(0, 0)  # busy until the engine reports its first step
        self._show_status(tr("Загрузка моделей и анализ… Окно остаётся доступным; анализ можно остановить."))
        self._refresh()

    def _stop(self) -> None:
        if self._drop_job():
            self._show_status(tr("Анализ остановлен; его результат будет проигнорирован."))
        self._refresh()

    def _drop_job(self) -> bool:
        job, self._job = self._job, None
        if job is None:
            return False
        job.cancel()
        return True

    def _clear_results(self) -> None:
        self._detections = ()
        self._wanted = []
        self._shown = []
        self._run_options = None
        self._delivered = None
        self.detections_list.clear()
        self._recompute()

    @Slot(object, int, int)
    def _on_progress(self, job: _Job, done: int, total: int) -> None:
        if job is not self._job or total <= 0:
            return
        self.progress.setRange(0, total)
        self.progress.setValue(max(0, min(done, total)))
        self._show_status(tr("Загрузка моделей…") if done == 0 else tr("Анализ: шаг {done} из {total}…", done=done, total=total))

    @Slot(object)
    def _on_finished(self, job: _Job) -> None:
        if job is not self._job:  # stopped, closed or foreign: its outcome is not ours to show
            return
        self._job = None
        if job.error is not None:
            self._show_status(job.error, error=True)
            self._refresh()
        elif job.result is None:
            self._show_status(tr("Анализ прерван без результата."), error=True)
            self._refresh()
        else:
            self._show_result(job.result, job.options)

    def done(self, result: int) -> None:
        self._drop_job()
        self._snapshot = None  # the worker keeps its own reference for as long as it still needs the pixels
        self.preview.release()
        super().done(result)

    def _show_result(self, result: DetectionResult, options: CensorOptions) -> None:
        self._detections = tuple(result.detections)
        self._wanted = [True] * len(self._detections)
        self._run_options = options
        self._delivered = None
        lst = self.detections_list
        lst.blockSignals(True)
        lst.clear()
        for found in self._detections:
            item = QListWidgetItem()
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable
            )
            rect = found.rect
            item.setToolTip(
                tr("Положение: x {value}, y {value2}; размер {value3}×{value4} px\nУверенность нейросети: {value5:.3f}", value=rect.x, value2=rect.y, value3=rect.width, value4=rect.height, value5=found.confidence)
            )
            lst.addItem(item)
        lst.blockSignals(False)
        count = len(self._detections)
        spent = tr("Готово за {value:.1f} с", value=result.elapsed_seconds)
        if count:
            lines = [
                tr("{spent}. Найдено кандидатов: {count} (поиск идёт с порогом {value} %); в списке — те, что проходят ваши пороги.", spent=spent, count=count, value=round(_FLOOR * 100))
            ]
        else:
            lines = [tr("{spent}: областей не найдено даже при пороге {value} %.", spent=spent, value=round(_FLOOR * 100))]
        technical = [tr("Окон анализа: {value}", value=result.views)]
        if result.model_names:
            technical.append(tr("модели: ") + ", ".join(result.model_names))
        lines.append("; ".join(technical) + ".")
        lines.extend(result.warnings)
        self._show_status("\n".join(lines))
        self._recompute()
        if count and self.tabs.currentWidget() is self._keep_page:
            self.tabs.setCurrentWidget(self._review_page)  # the user is waiting for exactly this list

    # ---- delivery

    def _apply(self, *, close: bool) -> None:
        if self._job is not None or not self._bars:
            return
        key = self._delivery_key()
        if self._delivered == key:  # a second click must not duplicate the bars
            return
        if self._on_apply(self.bars, self.replace_check.isChecked()) is False:
            self._show_status(
                tr("Редактор не принял результат (кадр или документ изменились). Ничего не применено."), error=True
            )
            return
        self._delivered = key
        if close:
            self.accept()
            return
        count = len(self._bars)
        how = tr("существующие заменены") if self.replace_check.isChecked() else tr("добавлены к существующим")
        self._show_status(tr("Передано в редактор: {count} {value} ({how}).", count=count, value=_plural(count, tr("полоса"), tr("полосы"), tr("полос")), how=how))
        self._refresh()
