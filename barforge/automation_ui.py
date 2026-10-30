"""Window-modal review of local crop/censor candidates, with bounded-memory background jobs."""
from __future__ import annotations

import contextlib
import copy
from dataclasses import replace
from pathlib import Path
import threading
from collections.abc import Callable, Sequence

import shiboken6
from PySide6.QtCore import QCoreApplication, QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from .i18n import tr
from .automation import CropCandidate, CropOptions, DatasetExportResult, PageAnalysis, analyze_source, export_dataset, prepare_page
from .autocensor import CancelledError, CensorOptions, LocalCensorEngine
from .canvas import ImageCanvas
from .imaging import load_rgb, render_pair
from .models import CAPTION_TEMPLATES, Document
from .ui import PageNavigation, fit_window


class _Job:
    def __init__(self, kind: str, **values):
        self.kind = kind
        self.values = values
        self.stop = threading.Event()
        self.pages: list[PageAnalysis] = []
        self.errors: list[tuple[str, str]] = []
        self.pair = None


class _Runner(QRunnable):
    def __init__(self, job: _Job, hub: _Hub):
        super().__init__()
        self.job, self.hub = job, hub

    def report(self, done: int, total: int, text: str):
        with contextlib.suppress(RuntimeError):
            self.hub.progressed.emit(self.job, round(done), total, text)

    def run(self):
        job = self.job
        try:
            if job.kind == "analyze":
                sources = job.values["sources"]
                for index, source in enumerate(sources):
                    if job.stop.is_set():
                        break
                    def progress(done, total, index=index, source=source):
                        self.report(index*100 + round(done*100/max(1,total)), len(sources)*100,
                                    tr("{value}/{value2}: {value3} · поиск {done}/{total}", value=index+1, value2=len(sources), value3=Path(source).name, done=done, total=total))
                    try:
                        page = analyze_source(source, job.values["engine"], job.values["options"],
                                              progress=progress, cancelled=job.stop.is_set)
                        job.pages.append(page)
                    except CancelledError:
                        break
                    except (OSError, ValueError, RuntimeError) as exc:
                        job.errors.append((source, str(exc)))
                        if isinstance(exc, RuntimeError):
                            break  # a broken/missing model is not a per-image failure
                    self.report((index+1)*100, len(sources)*100, tr("Обработано {value}/{value2}", value=index+1, value2=len(sources)))
            elif job.kind == "preview":
                if not job.stop.is_set():
                    job.pair = render_pair(job.values["document"], upscaler=job.values["upscaler"],
                        progress=lambda done,total,text: self.report(done,total,text), cancelled=job.stop.is_set)
            elif not job.stop.is_set():
                job.result = export_dataset(job.values["documents"], job.values["output_dir"],
                    source_dir=job.values["source_dir"], cancelled=job.stop.is_set, upscaler=job.values["upscaler"],
                    progress=lambda done,total: self.report(done,total,tr("Экспорт пар: {done}/{total}", done=done, total=total)))
        except Exception as exc:
            job.errors.append((tr("Обработка"), f"{type(exc).__name__}: {exc}"))
        finally:
            with contextlib.suppress(RuntimeError):
                self.hub.finished.emit(job)


class _Hub(QObject):
    progressed = Signal(object, int, int, str)
    finished = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.jobs: set[_Job] = set()
        self.finished.connect(self._forget)

    def start(self, job):
        self.jobs.add(job)
        self.pool.start(_Runner(job, self))

    @Slot(object)
    def _forget(self, job):
        self.jobs.discard(job)

    @Slot()
    def shutdown(self):
        for job in tuple(self.jobs):
            job.stop.set()
        self.pool.waitForDone(3000)


_hub_instance = None


def _hub():
    global _hub_instance
    if _hub_instance is None or not shiboken6.isValid(_hub_instance):
        app = QCoreApplication.instance()
        _hub_instance = _Hub(app)
        if app is not None:
            app.aboutToQuit.connect(_hub_instance.shutdown)
    return _hub_instance


def _number(low, high, value, suffix="", decimals=None):
    widget = QDoubleSpinBox() if decimals is not None else QSpinBox()
    widget.setRange(low, high)
    if decimals is not None:
        widget.setDecimals(decimals)
        widget.setSingleStep(.1)
    widget.setValue(value)
    widget.setSuffix(suffix)
    return widget


def _field(grid, row, column, title, widget):
    grid.addWidget(QLabel(title), row, column)
    grid.addWidget(widget, row, column+1)


class BatchDatasetDialog(QDialog):
    def __init__(self, sources: Sequence[str], on_accept: Callable[[list[Document]], object], parent=None, *,
                 output_dir: str = "", engine: LocalCensorEngine | None = None, upscaler=None,
                 censor_options: CensorOptions | None = None,
                 caption_template: str = CAPTION_TEMPLATES[0][1], current_source: str | None = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(tr("Автодатасет — кропы и полосы"))
        self._sources = list(dict.fromkeys(str(Path(source).resolve()) for source in sources))
        self._current_source = str(Path(current_source).resolve()) if current_source else None
        self._on_accept = on_accept
        self._engine = engine if engine is not None else LocalCensorEngine()
        self._upscale_engine = upscaler
        self._base_options = censor_options or CensorOptions()
        self._job: _Job | None = None
        self._pages: list[PageAnalysis] = []
        self._candidates: list[CropCandidate] = []
        self._loading = False
        self._manual_edits = False
        self._search_signature = None
        self._export_selection = []
        self._bookkeeping_root = output_dir
        self._exported = {}  # (dataset root, source, content signature) -> published pair bookkeeping
        self._shortcuts = []
        self._build(caption_template, output_dir)
        self._hub = _hub()
        self._hub.progressed.connect(self._on_progress)
        self._hub.finished.connect(self._on_finished)
        self.finished.connect(self._dispose)
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self._fit)
        self._refresh()
        fit_window(self, (1150, 760))

    @property
    def output_dir(self):
        return self.output_edit.text().strip()

    def _build(self, caption_template, output_dir):
        options = self._base_options
        layout = QVBoxLayout(self)
        self.current_only_check = QCheckBox(tr("Только текущий файл"))
        self.current_only_check.setEnabled(self._current_source is not None)
        head = QHBoxLayout()
        head.addWidget(QLabel(tr("Исходных файлов: {value} · локально, без отправки изображений", value=len(self._sources))), 1)
        head.addWidget(self.current_only_check)
        layout.addLayout(head)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        review = QWidget()
        review_layout = QHBoxLayout(review)
        self.candidates_list = QListWidget()
        self.candidates_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        listing = QWidget()
        listing.setMinimumWidth(155)
        listing.setMaximumWidth(255)
        list_layout = QVBoxLayout(listing)
        list_layout.setContentsMargins(0,0,0,0)
        list_layout.addWidget(QLabel(tr("Кропы для датасета")))
        list_layout.addWidget(PageNavigation(self.candidates_list),1)
        buttons = QHBoxLayout()
        for text, checked in ((tr("Все"),True),(tr("Снять"),False)):
            button = QPushButton(text)
            button.clicked.connect(lambda _checked=False, value=checked: self._check_all(value))
            buttons.addWidget(button)
        list_layout.addLayout(buttons)
        review_layout.addWidget(listing)
        panel = QVBoxLayout()
        tools = QHBoxLayout()
        self.canvas = ImageCanvas()
        self.canvas.setMinimumSize(280,180)
        self._tool_buttons = []
        for name,text in (("select",tr("Выбор [V]")),("bar",tr("Полоса [B]")),("brush",tr("Кисть [P]")),("crop",tr("Кроп [C]"))):
            button = QPushButton(text)
            button.clicked.connect(lambda _checked=False, tool=name: self.canvas.set_tool(tool))
            tools.addWidget(button)
            self._tool_buttons.append(button)
        for text, callback in ((tr("Вписать"),self.canvas.fit_image),("↶",self.canvas.undo),("↷",self.canvas.redo)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            tools.addWidget(button)
        self.preview_button = QPushButton(tr("Превью пары"))
        self.preview_button.clicked.connect(self._preview)
        tools.addWidget(self.preview_button)
        panel.addLayout(tools)
        panel.addWidget(self.canvas,1)
        self.caption_edit = QPlainTextEdit()
        self.caption_edit.setMaximumHeight(65)
        caption_pages = PageNavigation(self.caption_edit)
        caption_pages.setMaximumHeight(100)
        panel.addWidget(caption_pages)
        review_layout.addLayout(panel,1)
        self.tabs.addTab(review,tr("Результаты"))
        crop_page = QWidget()
        grid = QGridLayout(crop_page)
        self.crop_mode_combo = QComboBox()
        self.crop_mode_combo.addItem(tr("Контекст вокруг областей"),"context")
        self.crop_mode_combo.addItem(tr("Только целая страница"),"page")
        crop_defaults = CropOptions()
        self.context_spin = _number(1,20,crop_defaults.context," ×",decimals=2)
        self.min_side_spin = _number(32,8192,crop_defaults.min_side," px")
        self.aspect_combo = QComboBox()
        for text, ratio in ((tr("По анатомической области"),0.),(tr("Квадрат 1:1"),1.),(tr("Вертикальный 3:4"),.75),(tr("Горизонтальный 4:3"),4/3)):
            self.aspect_combo.addItem(text,ratio)
        self.max_crops_spin = _number(1,100,8)
        self.include_page_check = QCheckBox(tr("Добавить вариант целой страницы"))
        _field(grid,0,0,tr("Режим"),self.crop_mode_combo)
        _field(grid,0,2,tr("Контекст"),self.context_spin)
        _field(grid,1,0,tr("Мин. сторона"),self.min_side_spin)
        _field(grid,1,2,tr("Форма"),self.aspect_combo)
        _field(grid,2,0,tr("Макс. кропов на файл"),self.max_crops_spin)
        grid.addWidget(self.include_page_check,2,2,1,2)
        crop_note = QLabel(tr("Контекстный кроп — не распознавание рамок панелей. Область сохраняется целиком, соседние найденные зоны тоже получают полосы. Без растяжения и увеличения изображения."))
        crop_note.setWordWrap(True)
        grid.addWidget(crop_note,3,0,1,4)
        grid.setRowStretch(4,1)
        self.tabs.addTab(crop_page,tr("Автокроп"))
        bars_page = QWidget()
        grid = QGridLayout(bars_page)
        self.color_combo = QComboBox()
        self.color_combo.addItem(tr("Белый"),"#ffffff")
        self.color_combo.addItem(tr("Чёрный"),"#000000")
        self.color_combo.setCurrentIndex(max(0,self.color_combo.findData(options.color)))
        self.thickness_spin = _number(1,64,options.thickness," px")
        self.spacing_spin = _number(options.thickness+1,512,options.spacing," px")
        self.variation_min_spin = _number(0,64,options.variation_min," px")
        self.variation_max_spin = _number(0,64,options.variation_max," px")
        self.angle_spin = _number(-180,180,options.angle,"°",decimals=1)
        self.auto_angle_check = QCheckBox(tr("Поперёк органа (автоугол)"))
        self.auto_angle_check.setChecked(options.auto_angle)
        self.angle_spin.setEnabled(not options.auto_angle)
        self.padding_spin = _number(0,100,round(options.padding*100)," %")
        self.scale_spin = _number(10,400,round(options.size_scale*100)," %")
        for row, entries in enumerate(((tr("Цвет"),self.color_combo,tr("Толщина"),self.thickness_spin),
                                       (tr("Шаг"),self.spacing_spin,tr("Угол"),self.angle_spin),
                                       (tr("Отклонение от"),self.variation_min_spin,tr("до"),self.variation_max_spin),
                                       (tr("Запас зоны"),self.padding_spin,tr("Масштаб зоны"),self.scale_spin))):
            _field(grid,row,0,entries[0],entries[1])
            _field(grid,row,2,entries[2],entries[3])
        note = QLabel(tr("Тонкие линии с промежутками. Отклонение имеет случайный знак; 0/0 отключает вариацию. Настройки не переписывают ручные правки до перестроения предложений."))
        note.setWordWrap(True)
        grid.addWidget(self.auto_angle_check,4,0,1,4)
        grid.addWidget(note,5,0,1,4)
        selected_button = QPushButton(tr("Толщину, цвет и угол — выделенной полосе"))
        selected_button.clicked.connect(lambda: self.canvas.update_selected(
            thickness=self.thickness_spin.value(),color=self.color_combo.currentData(),
            angle=None if self.auto_angle_check.isChecked() else self.angle_spin.value()))
        grid.addWidget(selected_button,6,0,1,4)
        grid.setRowStretch(7,1)
        self.tabs.addTab(bars_page,tr("Полосы"))
        search_page = QWidget()
        grid = QGridLayout(search_page)
        self.confidence_spin = _number(5,95,round(options.confidence*100)," %")
        self.act_confidence_spin = _number(5,95,round(options.act_confidence*100)," %")
        self.acts_check = QCheckBox(tr("Уточнять анатомию по подсказкам сцен"))
        self.acts_check.setChecked(options.include_acts)
        self.nipples_check = QCheckBox(tr("Учитывать соски"))
        self.nipples_check.setChecked(options.include_nipples)
        self.anus_check = QCheckBox(tr("Анус (дополнительная модель)"))
        self.anus_check.setChecked(options.include_anus)
        self.tiles_check = QCheckBox(tr("Анализировать фрагменты"))
        self.tiles_check.setChecked(options.tiles)
        self.rotations_check = QCheckBox(tr("Проверять повороты"))
        self.rotations_check.setChecked(options.rotations)
        _field(grid,0,0,tr("Порог областей"),self.confidence_spin)
        _field(grid,0,2,tr("Порог подсказок сцен"),self.act_confidence_spin)
        grid.addWidget(self.acts_check,1,0,1,2)
        grid.addWidget(self.nipples_check,1,2,1,2)
        grid.addWidget(self.tiles_check,2,0,1,2)
        grid.addWidget(self.rotations_check,2,2,1,2)
        grid.addWidget(self.anus_check,3,0,1,4)
        note = QLabel(tr("Сцены — только подсказки локального поиска, не зоны рисования. Полосы строятся по уточнённой анатомии. Пороги анатомии и категории применяются к кешу; изменение подсказок сцен, фрагментов, поворотов или файлов требует нового поиска."))
        note.setWordWrap(True)
        grid.addWidget(note,4,0,1,4)
        grid.setRowStretch(5,1)
        self.tabs.addTab(search_page,tr("Поиск"))
        captions_page = QWidget()
        captions_layout = QVBoxLayout(captions_page)
        self.template_combo = QComboBox()
        for title,pattern in CAPTION_TEMPLATES:
            self.template_combo.addItem(title,pattern)
        self.pattern_edit = QPlainTextEdit(caption_template)
        self.pattern_edit.setMinimumHeight(70)
        captions_layout.addWidget(QLabel(tr("Шаблон новых предложений: {color} → white / black")))
        captions_layout.addWidget(self.template_combo)
        captions_layout.addWidget(PageNavigation(self.pattern_edit),1)
        template_button = QPushButton(tr("Использовать выбранный шаблон"))
        template_button.clicked.connect(lambda: self.pattern_edit.setPlainText(self.template_combo.currentData()))
        captions_layout.addWidget(template_button)
        self.tabs.addTab(captions_page,tr("Капшен"))
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.tabs.addTab(PageNavigation(self.log_edit),tr("Отчёт"))
        self.config_hint = QLabel(tr("Найти области → проверить кропы → экспортировать выбранные пары."))
        self.config_hint.setWordWrap(True)
        self.config_hint.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)
        self.config_hint.setMaximumHeight(self.fontMetrics().lineSpacing()*3)
        layout.addWidget(self.config_hint)
        output_row = QHBoxLayout()
        self.output_edit = QLineEdit(output_dir)
        self.output_edit.setPlaceholderText(tr("Каталог датасета"))
        output_row.addWidget(self.output_edit,1)
        choose = QPushButton(tr("Папка…"))
        choose.clicked.connect(self._choose_output)
        output_row.addWidget(choose)
        layout.addLayout(output_row)
        self.upscale_check = QCheckBox(tr("RealESRGAN x4 · anime_6B для выбранного кропа"))
        self.upscale_check.toggled.connect(self._upscale_changed)
        layout.addWidget(self.upscale_check)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.status = QLabel(tr("Готово к поиску."))
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)
        self.status.setMaximumHeight(self.fontMetrics().lineSpacing()*3)
        layout.addWidget(self.status)
        row = QHBoxLayout()
        self.run_button = QPushButton(tr("Найти и подготовить"))
        self.stop_button = QPushButton(tr("Остановить"))
        self.rebuild_button = QPushButton(tr("Перестроить из кеша"))
        self.export_button = QPushButton(tr("Экспортировать выбранные"))
        self.accept_button = QPushButton(tr("В редактор"))
        close = QPushButton(tr("Закрыть"))
        for button in (self.run_button,self.stop_button,self.rebuild_button,self.export_button,self.accept_button,close):
            button.setAutoDefault(False)
            row.addWidget(button)
        layout.addLayout(row)
        self.run_button.clicked.connect(self._run)
        self.stop_button.clicked.connect(self._stop)
        self.rebuild_button.clicked.connect(self._rebuild)
        self.export_button.clicked.connect(self._export)
        self.accept_button.clicked.connect(self._accept_selected)
        close.clicked.connect(self.reject)
        self.candidates_list.currentRowChanged.connect(self._select)
        self.candidates_list.itemChanged.connect(lambda _item: self._refresh())
        self.caption_edit.textChanged.connect(self._caption_changed)
        self.canvas.changed.connect(self._canvas_changed)
        self.canvas.brush_changed.connect(self.thickness_spin.setValue)
        self.output_edit.textChanged.connect(self._destination_changed)
        self.thickness_spin.valueChanged.connect(lambda value: self.spacing_spin.setMinimum(value+1))
        self.variation_min_spin.valueChanged.connect(lambda value: self._bound(True,value))
        self.variation_max_spin.valueChanged.connect(lambda value: self._bound(False,value))
        self._settings = (crop_page,bars_page,search_page,captions_page,self.current_only_check,self.output_edit,choose,self.upscale_check)
        for widget in self.tabs.findChildren(QComboBox):
            widget.currentIndexChanged.connect(self._settings_changed)
        for widget in self.tabs.findChildren(QDoubleSpinBox):
            widget.valueChanged.connect(self._settings_changed)
        for widget in self.tabs.findChildren(QSpinBox):
            widget.valueChanged.connect(self._settings_changed)
        for widget in (self.acts_check,self.nipples_check,self.anus_check,self.auto_angle_check,self.tiles_check,self.rotations_check,self.include_page_check,self.current_only_check):
            widget.toggled.connect(self._settings_changed)
        self.pattern_edit.textChanged.connect(self._settings_changed)
        for key,callback in (("V",lambda:self.canvas.set_tool("select")),("B",lambda:self.canvas.set_tool("bar")),
                             ("P",lambda:self.canvas.set_tool("brush")),("C",lambda:self.canvas.set_tool("crop")),
                             ("F",self.canvas.fit_image),("Ctrl+Z",self.canvas.undo),("Ctrl+Shift+Z",self.canvas.redo)):
            shortcut = QShortcut(QKeySequence(key),self.canvas)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    def _options(self):
        return replace(self._base_options, color=self.color_combo.currentData(), thickness=self.thickness_spin.value(),
            spacing=self.spacing_spin.value(),variation_min=self.variation_min_spin.value(),variation_max=self.variation_max_spin.value(),
            angle=self.angle_spin.value(),auto_angle=self.auto_angle_check.isChecked(),
            padding=self.padding_spin.value()/100,size_scale=self.scale_spin.value()/100,
            confidence=self.confidence_spin.value()/100,act_confidence=self.act_confidence_spin.value()/100,
            include_acts=self.acts_check.isChecked(),include_nipples=self.nipples_check.isChecked(),include_anus=self.anus_check.isChecked(),
            tiles=self.tiles_check.isChecked(),rotations=self.rotations_check.isChecked())

    def _crop_options(self):
        return CropOptions(self.crop_mode_combo.currentData(),self.context_spin.value(),self.min_side_spin.value(),
                           self.aspect_combo.currentData(),self.max_crops_spin.value(),self.include_page_check.isChecked())

    def _sources_chosen(self):
        return [self._current_source] if self.current_only_check.isChecked() and self._current_source else self._sources

    def _search_key(self):
        return (tuple(self._sources_chosen()),self.tiles_check.isChecked(),self.rotations_check.isChecked(),
                self.acts_check.isChecked(),self.act_confidence_spin.value())

    def _bound(self, lower, value):
        other = self.variation_max_spin if lower else self.variation_min_spin
        if (lower and value > other.value()) or (not lower and value < other.value()):
            other.setValue(value)

    def _settings_changed(self, *_):
        self.canvas.set_brush(self.thickness_spin.value(),self.color_combo.currentData())
        self.angle_spin.setEnabled(not self.auto_angle_check.isChecked())
        if self._pages:
            hint = tr("Параметры изменены: перестроение заменит предложения и ручные правки; повторный поиск не нужен.")
            if self._search_key() != self._search_signature:
                hint = tr("Набор файлов или параметры поиска изменены: нужен новый поиск. Кеш относится к прежнему поиску.")
            self.config_hint.setText(hint)
        self._refresh()

    def _status(self, text):
        self.status.setText(text)
        self.status.setToolTip(text)

    def _refresh(self):
        if self._loading:
            return
        busy = self._job is not None
        checked = self._checked()
        for widget in self._settings:
            widget.setEnabled(not busy)
        self.tabs.widget(0).setEnabled(not busy)
        self.candidates_list.setEnabled(not busy)
        self.canvas.setEnabled(not busy and self.canvas._document is not None)
        self.caption_edit.setEnabled(not busy and self.canvas._document is not None)
        self.run_button.setEnabled(not busy and bool(self._sources))
        self.stop_button.setEnabled(busy)
        self.rebuild_button.setEnabled(not busy and bool(self._pages) and self._search_key() == self._search_signature)
        self.accept_button.setEnabled(not busy and bool(checked))
        remaining = [item for item in checked if not item.document.is_exported]
        self.export_button.setEnabled(not busy and bool(remaining))
        self.preview_button.setEnabled(not busy and self.canvas._document is not None)
        self.progress.setVisible(busy)

    def _checked(self):
        return [candidate for index,candidate in enumerate(self._candidates)
                if self.candidates_list.item(index).checkState() == Qt.CheckState.Checked]

    def _check_all(self, checked):
        if self._job is not None:
            return
        for index in range(self.candidates_list.count()):
            self.candidates_list.item(index).setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    def _run(self):
        if self._job is not None:
            return
        if self._manual_edits and QMessageBox.question(self,tr("Новый поиск?"),tr("Новый поиск заменит предложения и ручные правки. Продолжить?")) != QMessageBox.StandardButton.Yes:
            return
        self._search_signature = self._search_key()
        self._job = _Job("analyze",sources=list(self._sources_chosen()),engine=self._engine,options=self._options())
        self._status(tr("Начат локальный поиск…"))
        self._refresh()
        self._hub.start(self._job)

    def _stop(self):
        if self._job is not None:
            self._job.stop.set()
            self._status(tr("Остановка после текущего шага; готовые пары и страницы сохранятся."))

    @Slot(object,int,int,str)
    def _on_progress(self,job,done,total,text):
        if job is self._job:
            self.progress.setRange(0,max(1,total))
            self.progress.setValue(done)
            self._status(text)

    @Slot(object)
    def _on_finished(self,job):
        if job is not self._job:
            return
        self._job = None
        if job.kind == "analyze":
            self._pages = job.pages
            self._build_candidates()
            lines = [tr("{value}: {value2} кандидатов при 5 %, {value3:.2f} с; модели: {value4}", value=Path(page.source).name, value2=len(page.detections), value3=page.elapsed_seconds, value4=', '.join(page.model_names)) + ("\n"+"\n".join(page.warnings) if page.warnings else "") for page in self._pages]
            lines.extend(tr("Ошибка {source}: {error}", source=source, error=error) for source,error in job.errors)
            self.log_edit.setPlainText("\n".join(lines))
            prefix = tr("Поиск остановлен") if job.stop.is_set() else tr("Поиск завершён")
            self._status(tr("{prefix}. Страниц: {value}/{value2}; кропов: {value3}; ошибок: {value4}. Подробности — «Отчёт».", prefix=prefix, value=len(self._pages), value2=len(job.values['sources']), value3=len(self._candidates), value4=len(job.errors)))
        elif job.kind == "preview":
            if job.pair is not None:
                self._show_preview_result(*job.pair, job.values["document"].caption)
            else:
                self._status(tr("Превью остановлено.") if job.stop.is_set() else tr("Превью не выполнено. Подробности — «Отчёт»."))
        elif job.kind == "export":
            result = job.result
            if result is not None:
                for exported in result.documents:
                    key = (Path(result.session_path).parent.resolve(), Path(exported.source).resolve(), exported.signature())
                    self._exported[key] = exported.export_name, exported.exported_signature
                    for document in self._export_selection:
                        if Path(document.source) == Path(exported.source) and document.signature() == exported.signature():
                            document.export_name,document.exported_signature = exported.export_name,exported.exported_signature
                            break
                adopted = True
                if result.documents:
                    adopted = self._on_accept(copy.deepcopy(result.documents)) is not False
                errors = result.errors + job.errors
                self.log_edit.appendPlainText("\n"+tr("Экспорт: {value} пар. Сессия: {value2}", value=len(result.documents), value2=result.session_path))
                for source,error in errors:
                    self.log_edit.appendPlainText(tr("Ошибка {source}: {error}", source=source, error=error))
                note = tr(" Остановлено.") if result.cancelled else ""
                if not adopted:
                    note += tr(" В главную сессию не добавлено; откройте сохранённую пакетную сессию.")
                self._status(tr("Сохранено пар: {value}; ошибок: {value2}.{note} Сессия: {value3}", value=len(result.documents), value2=len(errors), note=note, value3=Path(result.session_path).name))
                self._bookkeeping_root = self.output_dir
            else:
                self._status(tr("Экспорт остановлен.") if job.stop.is_set() else tr("Экспорт не выполнен. Подробности — «Отчёт»."))
                self.log_edit.appendPlainText("\n".join(f"{source}: {error}" for source,error in job.errors))
            self._update_labels()
        self._refresh()

    def _rebuild(self):
        if self._job is not None or not self._pages or self._search_key() != self._search_signature:
            return
        if self._manual_edits and QMessageBox.question(self,tr("Перестроить?"),tr("Перестроение заменит ручные правки кропов, полос и капшенов. Продолжить?")) != QMessageBox.StandardButton.Yes:
            return
        self._build_candidates()
        self._status(tr("Предложения перестроены из кеша без поиска. Кропов: {value}.", value=len(self._candidates)))

    def _build_candidates(self):
        try:
            candidates = [candidate for page in self._pages for candidate in prepare_page(
                page,self._options(),self._crop_options(),self.pattern_edit.toPlainText())]
        except ValueError as exc:
            self._status(str(exc))
            return
        self._restore_exports(candidates)
        self._loading = True
        self._candidates = candidates
        self.candidates_list.clear()
        self.canvas.clear_document()
        self.caption_edit.clear()
        for candidate in candidates:
            item = QListWidgetItem()
            item.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsSelectable|Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.candidates_list.addItem(item)
        self._loading = False
        self._manual_edits = False
        self._update_labels()
        if candidates:
            self.candidates_list.setCurrentRow(0)
            self.tabs.setCurrentIndex(0)
        self.config_hint.setText(tr("Снимите отметки с лишних кропов. Геометрию и капшен можно исправить в редакторе."))
        self._refresh()

    def _restore_exports(self, candidates):
        if not self.output_dir:
            return
        root = Path(self.output_dir).resolve()
        for candidate in candidates:
            document = candidate.document
            published = self._exported.get((root, Path(document.source).resolve(), document.signature()))
            if published is not None:
                document.export_name, document.exported_signature = published


    def _update_labels(self):
        self.candidates_list.blockSignals(True)
        for index,candidate in enumerate(self._candidates):
            doc = candidate.document
            crop = doc.crop
            dimensions = f"{crop.width}×{crop.height} [{crop.x},{crop.y}]" if crop is not None else tr("вся страница")
            item = self.candidates_list.item(index)
            item.setText(f"{'✓' if doc.is_exported else '·'} {Path(doc.source).name} · {dimensions}")
            item.setToolTip(tr("{value}\n{value2}; {dimensions}\nПолос: {value3}; уверенность: {value4:.0%}", value=doc.source, value2=candidate.reason, dimensions=dimensions, value3=len(doc.bars), value4=candidate.score))
        self.candidates_list.blockSignals(False)

    def _select(self,index):
        if self._loading:
            return
        self._loading = True
        try:
            if not 0 <= index < len(self._candidates):
                self.canvas.clear_document()
                self.caption_edit.clear()
            else:
                document = self._candidates[index].document
                image = load_rgb(document.source)
                try:
                    self.canvas.set_document(image,document)
                finally:
                    image.close()
                self.canvas.set_brush(self.thickness_spin.value(),self.color_combo.currentData())
                self.caption_edit.setPlainText(document.caption)
                self.upscale_check.blockSignals(True)
                self.upscale_check.setChecked(document.upscale)
                self.upscale_check.blockSignals(False)
        except (OSError,ValueError) as exc:
            self.canvas.clear_document()
            self._status(str(exc))
        finally:
            self._loading = False
        self._refresh()

    def _upscale_changed(self, checked):
        if not self._loading and self.canvas._document is not None:
            self.canvas._document.upscale = bool(checked)
            self._canvas_changed()

    def _caption_changed(self):
        if not self._loading and self.canvas._document is not None:
            self.canvas._document.caption = self.caption_edit.toPlainText()
            self._canvas_changed()

    def _canvas_changed(self):
        if not self._loading:
            self._manual_edits = True
            self._update_labels()
            self._refresh()

    def _choose_output(self):
        folder = QFileDialog.getExistingDirectory(self,tr("Папка датасета"),self.output_dir)
        if folder:
            self.output_edit.setText(folder)

    def _destination_changed(self):
        if self.output_dir != self._bookkeeping_root:
            for candidate in self._candidates:
                candidate.document.export_name = None
                candidate.document.exported_signature = None
            self._bookkeeping_root = self.output_dir
            self._restore_exports(self._candidates)
            self._update_labels()
        self._refresh()

    def _export(self):
        if self._job is not None:
            return
        documents = [candidate.document for candidate in self._checked() if not candidate.document.is_exported]
        if not documents:
            return
        if not self.output_dir:
            self._choose_output()
            if not self.output_dir:
                return
        self._export_selection = documents
        self._job = _Job("export",documents=copy.deepcopy(documents),output_dir=self.output_dir,upscaler=self._upscale_engine,
                         source_dir=str(Path(self._sources[0]).parent) if self._sources else "")
        self._status(tr("Начат экспорт выбранных пар…"))
        self._refresh()
        self._hub.start(self._job)

    def _accept_selected(self):
        if self._job is None:
            documents = [copy.deepcopy(candidate.document) for candidate in self._checked()]
            if documents and self._on_accept(documents) is not False:
                self.accept()

    def _preview(self):
        document = self.canvas._document
        if document is None or self._job is not None:
            return
        snapshot = copy.deepcopy(document)
        if snapshot.upscale:
            self._job = _Job("preview",document=snapshot,upscaler=self._upscale_engine)
            self._status(tr("RealESRGAN x4: подготовка превью…"))
            self._refresh()
            self._hub.start(self._job)
            return
        try:
            clean,censored = render_pair(snapshot)
        except (OSError,ValueError) as exc:
            self._status(str(exc))
            return
        self._show_preview_result(clean,censored,snapshot.caption)

    def _show_preview_result(self,clean,censored,caption_text):
        from .main_window import PreviewView
        dialog = QDialog(self)
        dialog.setWindowTitle(tr("Превью пары · {value}×{value2} px", value=clean.width, value2=clean.height))
        layout = QVBoxLayout(dialog)
        images = QHBoxLayout()
        for title,image in (("CLEAN · TARGET",clean),("CENSORED · CONTROL",censored)):
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            column.addWidget(PreviewView(image),1)
            images.addLayout(column)
        layout.addLayout(images,1)
        caption = QPlainTextEdit(caption_text)
        caption.setReadOnly(True)
        caption.setMaximumHeight(65)
        layout.addWidget(PageNavigation(caption))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        fit_window(dialog,(1000,650))
        dialog.exec()
    @Slot(int)
    def _dispose(self,_result):
        if self._job is not None:
            self._job.stop.set()
            self._job = None

    def showEvent(self,event):
        super().showEvent(event)
        self._fit_timer.start(0)

    def _fit(self):
        fit_window(self)
