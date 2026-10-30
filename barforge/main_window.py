"""Desktop workflow; image geometry and export stay outside the window."""
from __future__ import annotations

import copy
from dataclasses import replace

from pathlib import Path

from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QGraphicsScene,
    QGraphicsView, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy, QSpinBox, QSplitter,
    QTabWidget, QToolBar, QVBoxLayout, QWidget,
)

from .canvas import ImageCanvas
from .imaging import auto_bars, load_rgb, render_pair
from .models import (
    Bar, CAPTION_TEMPLATES, COLOR_PLACEHOLDER, DEFAULT_CAPTION, Document, Rect, Session,
    caption_template_for, render_caption_template,
)
from .storage import discover_images, export_pair, load_session, save_session
from .ui import PageNavigation, fit_window


class PreviewView(QGraphicsView):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        scene = QGraphicsScene(self)
        self.setScene(scene)
        scene.addPixmap(QPixmap.fromImage(ImageQt(image)))
        self.setMinimumSize(280, 220)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class ElidedLabel(QLabel):
    """One-line label that elides the middle of long text instead of widening its window.

    ``text()`` keeps returning the full text; the tooltip shows it while the display is shortened.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def text(self) -> str:
        return self._full

    def setText(self, text: str) -> None:
        self._full = text
        self._elide()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        shown = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideMiddle, self.contentsRect().width())
        QLabel.setText(self, shown)
        self.setToolTip(self._full if shown != self._full else "")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.session = Session()
        self.session_path: Path | None = None
        self.session_dirty = False
        self._loading = False
        self._image = None
        self._active_index = -1
        self._disk_pair_count = 0
        self._shortcuts: list[QShortcut] = []
        self._auto_dialog = None
        self._auto_engine = None
        self._upscale_engine = None
        self._image_task = None
        self._last_caption = DEFAULT_CAPTION
        self._batch_dialog = None
        self.setWindowTitle("BarForge — парный датасет")
        self._build_ui()
        self._build_shortcuts()
        # Importing credentials is separate from opening source images.
        from .grok_ui import GrokController
        self.grok = GrokController(self)
        self.grok.availability_changed.connect(lambda _available: self._refresh_grok())
        self._refresh_grok()
        self._update_progress()
        self.statusBar().showMessage("Откройте папку PNG/JPG/WEBP или сохранённую сессию.")
        fit_window(self, (1440, 920))

    @property
    def document(self) -> Document | None:
        if 0 <= self._active_index < len(self.session.documents):
            return self.session.documents[self._active_index]
        return None

    def _button(self, text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _build_ui(self):
        toolbar = QToolBar("Проект", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        for text, callback, shortcut in (
            ("Открыть папку", self.open_folder, "Ctrl+O"),
            ("Загрузить сессию", self.open_session, "Ctrl+L"),
            ("Сохранить сессию", self.save_current_session, "Ctrl+Shift+S"),
        ):
            action = QAction(text, self)
            action.triggered.connect(callback)
            action.setShortcut(shortcut)
            toolbar.addAction(action)
        toolbar.addSeparator()
        self.output_action = QAction("Папка датасета…", self)
        self.output_action.triggered.connect(self.choose_output)
        toolbar.addAction(self.output_action)
        settings = QAction("Настройки Grok…", self)
        settings.triggered.connect(self.grok_settings)
        toolbar.addAction(settings)
        help_action = QAction("Горячие клавиши", self)
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 8, 12, 10)
        header = QHBoxLayout()
        title = QLabel("BARFORGE")
        title.setObjectName("brand")
        header.addWidget(title)
        header.addWidget(QLabel("CLEAN / CONTROL · DATASET EDITOR"))
        self.project_label = ElidedLabel("Нет открытой папки")
        self.project_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self.project_label, 1)
        outer.addLayout(header)
        split = QSplitter()
        outer.addWidget(split, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.addWidget(QLabel("ИЗОБРАЖЕНИЯ / КРОПЫ"))
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Фильтр по имени…")
        self.filter.textChanged.connect(self._filter_files)
        left_layout.addWidget(self.filter)
        self.files = QListWidget()
        # Without a horizontal scroll bar long names are shortened in the middle, keeping the distinguishing tail.
        self.files.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.files.currentRowChanged.connect(self._select_document)
        left_layout.addWidget(PageNavigation(self.files), 1)
        nav = QHBoxLayout()
        self._button("← Назад", lambda: self.navigate(-1), nav)
        self._button("Далее →", lambda: self.navigate(1), nav)
        left_layout.addLayout(nav)
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m документов")
        left_layout.addWidget(self.progress)
        self.pair_count = QLabel("Пар на диске: 0")
        self.pair_count.setWordWrap(True)
        left_layout.addWidget(self.pair_count)
        split.addWidget(left)

        middle = QWidget()
        middle_layout = QVBoxLayout(middle)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        tools = QToolBar("Инструменты")
        tools.setMovable(False)
        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_actions = {}
        for name, text in (("select", "Выбор [V]"), ("bar", "Полоса [B]"), ("brush", "Кисть [P]"), ("crop", "Кроп [C]")):
            action = QAction(text, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, tool=name: self.canvas.set_tool(tool))
            self.tool_group.addAction(action)
            tools.addAction(action)
            self.tool_actions[name] = action
        self.tool_actions["bar"].setChecked(True)
        middle_layout.addWidget(tools)
        self.canvas = ImageCanvas()
        self.canvas.setMinimumSize(320, 160)
        self.canvas.set_tool("bar")
        self.canvas.changed.connect(self._document_changed)
        self.canvas.selection_changed.connect(self._selection_changed)
        self.canvas.status.connect(lambda message: self.statusBar().showMessage(message, 6000))
        middle_layout.addWidget(self.canvas, 1)
        view_tools = QHBoxLayout()
        self._button("−", self.canvas.zoom_out, view_tools)
        self._button("Вписать [F]", self.canvas.fit_image, view_tools)
        self._button("+", self.canvas.zoom_in, view_tools)
        self.image_info = ElidedLabel("Колесо — зум · СКМ — панорама · Боковая кнопка + движение — толщина")
        self.image_info.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        view_tools.addWidget(self.image_info, 1)
        middle_layout.addLayout(view_tools)
        split.addWidget(middle)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 0, 0, 0)
        # Tabs instead of a scroll area: every setting stays reachable and the export actions never leave the view.
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_bars_page(), "Полосы")
        self.tabs.addTab(self._build_caption_page(), "Капшен")
        self.tabs.setTabToolTip(1, "Капшен сохраняется в clean/*.txt")
        sidebar_layout.addWidget(self.tabs, 1)
        sidebar_layout.addWidget(self._build_export_group())
        split.addWidget(sidebar)
        split.setSizes([225, 850, 340])
        split.setStretchFactor(1, 1)
        split.setCollapsible(0, False)
        split.setCollapsible(1, False)
        split.setCollapsible(2, False)
        self.setCentralWidget(root)
        self.canvas.brush_changed.connect(self.thickness.setValue)
        self._brush_changed()

    def _build_bars_page(self):
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(5)
        self.thickness = QSpinBox()
        self.thickness.setRange(1, 8192)
        self.thickness.setValue(6)
        self.thickness.setSuffix(" px")
        self.thickness.valueChanged.connect(self._brush_changed)
        grid.addWidget(QLabel("Толщина"), 0, 0)
        grid.addWidget(self.thickness, 0, 1)
        self.bar_angle = QDoubleSpinBox()
        self.bar_angle.setRange(-180, 180)
        self.bar_angle.setDecimals(1)
        self.bar_angle.setSingleStep(5)
        self.bar_angle.setSuffix("°")
        self.bar_angle.valueChanged.connect(self._brush_changed)
        angle_label = QLabel("Угол")
        angle_label.setToolTip("Угол выбранной полосы")
        self.bar_angle.setToolTip("Угол выбранной полосы")
        grid.addWidget(angle_label, 0, 2)
        grid.addWidget(self.bar_angle, 0, 3)
        self.color = QComboBox()
        self.color.addItem("Белый", "#ffffff")
        self.color.addItem("Чёрный", "#000000")
        self.color.currentIndexChanged.connect(self._brush_changed)
        self.color.currentIndexChanged.connect(self._recolor_caption)
        grid.addWidget(QLabel("Цвет"), 1, 0)
        grid.addWidget(self.color, 1, 1)
        self.presets = QComboBox()
        self.presets.addItem("Пресеты…", None)
        for label, value in (("Белый · 4 px", (4, "#ffffff")),
                             ("Белый · 6 px", (6, "#ffffff")),
                             ("Белый · 9 px", (9, "#ffffff")),
                             ("Чёрный · 4 px", (4, "#000000")),
                             ("Чёрный · 6 px", (6, "#000000")),
                             ("Чёрный · 9 px", (9, "#000000"))):
            self.presets.addItem(label, value)
        self.presets.activated.connect(self._apply_preset)
        grid.addWidget(self.presets, 1, 2, 1, 2)
        row = QHBoxLayout()
        self._button("Удалить", self.canvas.delete_selected, row)
        self._button("Очистить разметку", self._clear_bars, row)
        grid.addLayout(row, 2, 0, 1, 4)
        neural_button = QPushButton("Нейроавтоцензор… [Ctrl+Shift+A]")
        neural_button.clicked.connect(self.automatic_censor)
        grid.addWidget(neural_button, 3, 0, 1, 4)
        self.batch_button = QPushButton("Автодатасет… [Ctrl+Alt+A]")
        self.batch_button.setToolTip("Пакетно предложить кропы и полосы, проверить и экспортировать пары.")
        self.batch_button.setEnabled(False)
        self.batch_button.clicked.connect(self.automatic_dataset)
        grid.addWidget(self.batch_button, 4, 0, 1, 4)
        variation_row = QHBoxLayout()
        variation_row.addWidget(QLabel("Случайная толщина ±"))
        self.variation_min = QSpinBox()
        self.variation_max = QSpinBox()
        for spin in (self.variation_min, self.variation_max):
            spin.setRange(0, 64)
            spin.setSuffix(" px")
            spin.valueChanged.connect(lambda _value, moved=spin: self._variation_changed(moved))
            spin.setToolTip("Диапазон для автоматических линий; ручная кисть использует точную толщину.")
            variation_row.addWidget(spin)
        self.variation_min.setValue(3)
        self.variation_max.setValue(8)
        grid.addLayout(variation_row, 4, 0, 1, 4)
        grid.addWidget(self.batch_button, 5, 0, 1, 4)
        self.auto_count = QSpinBox()
        self.auto_count.setRange(1, 12)
        self.auto_count.setValue(3)
        row = QHBoxLayout()
        row.addWidget(QLabel("Полос в пресете"))
        row.addWidget(self.auto_count)
        self._button("+ Центр", self.add_auto_bars, row)
        self._button("Сбросить кроп", self.canvas.reset_crop, row)
        grid.addLayout(row, 6, 0, 1, 4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setRowStretch(7, 1)
        return page

    def _build_caption_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        row = QHBoxLayout()
        self.templates = QComboBox()
        for label, pattern in CAPTION_TEMPLATES:
            self.templates.addItem(label, pattern)
            self.templates.setItemData(self.templates.count() - 1, pattern, Qt.ItemDataRole.ToolTipRole)
        self.templates.setToolTip("Готовый капшен: цвет полос подставляется как white / black. При смене цвета "
                                  "готовый капшен перекрашивается, собственный текст не меняется.")
        row.addWidget(self.templates, 1)
        self.apply_template_button = self._button("Применить шаблон", self.apply_template, row)
        layout.addLayout(row)
        self.caption = QPlainTextEdit()
        self.caption.setPlaceholderText("Инструкция для обучения или собственный капшен…")
        self.caption.setToolTip(f"{COLOR_PLACEHOLDER} в тексте заменяется на white или black по выбранному цвету полос.")
        self.caption.setMinimumHeight(72)
        self.caption.textChanged.connect(self._caption_changed)
        layout.addWidget(PageNavigation(self.caption), 1)
        self.ask_button = self._button("Спросить Grok…", self.ask_grok, layout)
        self.grok_hint = QLabel()
        self.grok_hint.setWordWrap(True)
        # Reserve the wrapped lines: a word-wrapped label would otherwise squeeze the editor at the minimum height.
        self.grok_hint.setMinimumHeight(self.grok_hint.fontMetrics().lineSpacing() * 4)
        layout.addWidget(self.grok_hint)
        return page

    def _build_export_group(self):
        export = QGroupBox("ЭКСПОРТ ПАРЫ")
        layout = QVBoxLayout(export)
        layout.setSpacing(5)
        # A read-only field keeps the whole path selectable and never forces the sidebar wider.
        self.output_label = QLineEdit()
        self.output_label.setReadOnly(True)
        self.output_label.setPlaceholderText("Папка датасета не выбрана")
        layout.addWidget(self.output_label)
        self._show_output_dir()
        self.new_pair = QCheckBox("Сохранить как новую пару")
        self.new_pair.setToolTip("Не заменять прошлый экспорт этого источника; удобно для нескольких панелей.")
        self.new_pair.setChecked(True)
        options = QHBoxLayout()
        options.addWidget(self.new_pair)
        layout.addLayout(options)
        self.upscale_check = QCheckBox("RealESRGAN x4 · anime_6B")
        self.upscale_check.setToolTip("Локальный апскейл обеих сторон кропа; до 16 777 216 выходных пикселей.")
        self.upscale_check.toggled.connect(self._upscale_changed)
        options.addWidget(self.upscale_check)
        self.upscale_progress = QProgressBar()
        self.upscale_progress.setVisible(False)
        layout.addWidget(self.upscale_progress)
        self.cancel_upscale = self._button("Остановить апскейл", self._cancel_image_task, layout)
        self.cancel_upscale.setVisible(False)
        actions = QHBoxLayout()
        self.preview_button = self._button("Превью [Ctrl+P]", self.preview_pair, actions)
        self.save_button = self._button("Сохранить [Ctrl+S]", self.save_pair, actions)
        layout.addLayout(actions)
        self.save_button.setObjectName("primary")
        return export

    def _shortcut(self, sequence, callback, canvas_only=False):
        target = self.canvas if canvas_only else self
        shortcut = QShortcut(QKeySequence(sequence), target)
        if canvas_only:
            shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        shortcut.activated.connect(callback)
        self._shortcuts.append(shortcut)

    def _build_shortcuts(self):
        for sequence, callback in (("Ctrl+S", self.save_pair), ("Ctrl+P", self.preview_pair),
                                   ("Ctrl+Shift+A", self.automatic_censor),
                                   ("Ctrl+Alt+A", self.automatic_dataset),
                                   ("Alt+Left", lambda: self.navigate(-1)),
                                   ("Alt+Right", lambda: self.navigate(1))):
            self._shortcut(sequence, callback)
        for sequence, callback in (("V", lambda: self._set_tool("select")),
                                   ("B", lambda: self._set_tool("bar")),
                                   ("P", lambda: self._set_tool("brush")),
                                   ("C", lambda: self._set_tool("crop")),
                                   ("F", self.canvas.fit_image),
                                   ("Ctrl+Z", self.canvas.undo), ("Ctrl+Shift+Z", self.canvas.redo)):
            self._shortcut(sequence, callback, True)

    def _set_tool(self, tool):
        self.tool_actions[tool].setChecked(True)
        self.canvas.set_tool(tool)

    def _error(self, title, error):
        QMessageBox.critical(self, title, str(error))

    def _touch(self):
        if not self._loading:
            self.session_dirty = True
            self._title()

    def _title(self):
        name = self.session_path.name if self.session_path else "Новая сессия"
        self.setWindowTitle(f"{'* ' if self.session_dirty else ''}{name} — BarForge")

    def _maybe_save_session(self):
        if not self.session_dirty:
            return True
        answer = QMessageBox.question(
            self, "Сохранить сессию?", "Есть изменения в разметке или настройках сессии.",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save)
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        return self.save_current_session() if answer == QMessageBox.StandardButton.Save else True

    def open_folder(self):
        if self._image_task is not None:
            self.statusBar().showMessage("Сначала дождитесь или остановите текущий апскейл.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Папка исходных PNG/JPG/WEBP", self.session.source_dir)
        if not folder:
            return
        try:
            documents = discover_images(Path(folder))
        except (OSError, ValueError) as exc:
            self._error("Не удалось открыть папку", exc)
            return
        if not documents:
            QMessageBox.information(self, "Нет изображений", "В этой папке нет PNG/JPG/JPEG/WEBP. Подпапки не включаются.")
            return
        if not self._maybe_save_session():
            return
        self.session = Session(documents=documents, source_dir=folder)
        self.session_path = None
        self._populate_files()
        self._touch()

    def open_session(self):
        if self._image_task is not None:
            self.statusBar().showMessage("Сначала дождитесь или остановите текущий апскейл.")
            return
        filename, _ = QFileDialog.getOpenFileName(self, "Загрузить сессию", "", "BarForge (*.barforge.json *.json)")
        if not filename:
            return
        try:
            session = load_session(Path(filename))
        except (OSError, ValueError, TypeError) as exc:
            self._error("Не удалось загрузить сессию", exc)
            return
        if not self._maybe_save_session():
            return
        self.session = session
        if session.documents:
            current = max(0,min(session.current_index,len(session.documents)-1))
            self._last_caption = session.documents[current].caption
        self.session_path = Path(filename)
        self.session_dirty = False
        self._populate_files()
        self._title()
        missing = sum(not Path(doc.source).is_file() for doc in session.documents)
        if missing:
            QMessageBox.warning(self, "Исходники недоступны", f"Не найдено изображений: {missing}. Восстановите файлы по путям в сессии или перенесите папку вместе с сессией.")

    def save_current_session(self):
        path = self.session_path
        if path is None:
            filename, _ = QFileDialog.getSaveFileName(self, "Сохранить сессию", "session.barforge.json", "BarForge (*.barforge.json)")
            if not filename:
                return False
            path = Path(filename)
            if not path.name.lower().endswith(".json"):
                path = path.with_name(path.name + ".barforge.json")
        self._sync_caption()
        try:
            save_session(self.session, path)
        except (OSError, ValueError) as exc:
            self._error("Ошибка сохранения сессии", exc)
            return False
        self.session_path = path
        self.session_dirty = False
        self._title()
        self.statusBar().showMessage(f"Сессия сохранена: {path}", 6000)
        return True

    def _populate_files(self):
        self._loading = True
        self._active_index = -1
        self._image = None
        self.filter.clear()
        self.files.clear()
        for doc in self.session.documents:
            self.files.addItem(self._file_label(doc))
            self.files.item(self.files.count() - 1).setToolTip(doc.source)
        self.project_label.setText(Path(self.session.source_dir).name or "Сессия")
        self._show_output_dir()
        self._loading = False
        self.batch_button.setEnabled(bool(self.session.documents))
        self.upscale_check.blockSignals(True)
        self.upscale_check.setChecked(False)
        self.upscale_check.blockSignals(False)
        if self.session.documents:
            self.files.setCurrentRow(min(max(self.session.current_index, 0), len(self.session.documents) - 1))
        else:
            self.canvas.clear_document()
            self.caption.clear()
        self._update_progress(refresh_disk=True)

    def _file_label(self, doc):
        state = "✓" if doc.is_exported else ("●" if doc.export_name else "·")
        crop = doc.crop
        suffix = f" · {crop.width}×{crop.height} [{crop.x},{crop.y}]" if crop is not None else ""
        return f"{state}  {Path(doc.source).name}{suffix}"

    def _filter_files(self, text):
        for index, doc in enumerate(self.session.documents):
            self.files.item(index).setHidden(text.casefold() not in Path(doc.source).name.casefold())

    def _select_document(self, index):
        if self._loading or index < 0 or index >= len(self.session.documents):
            return
        doc = self.session.documents[index]
        if index != self._active_index:
            self._sync_caption()
        if not doc.caption_touched:
            doc.caption = self._last_caption
        self._active_index = index
        if self.session.current_index != index:
            self.session.current_index = index
            self._touch()
        self._loading = True
        try:
            self._image = load_rgb(doc.source)
            self.canvas.set_document(self._image, doc)
            self._brush_changed()
            self.canvas.setEnabled(True)
            self.image_info.setText(f"{self._image.width} × {self._image.height} px · Колесо: зум · СКМ: панорама")
        except (OSError, ValueError) as exc:
            self._image = None
            self.canvas.setEnabled(False)
            self.canvas.clear_document()
            self.image_info.setText("Изображение недоступно")
            self.statusBar().showMessage(f"Ошибка чтения {Path(doc.source).name}: {exc}")
        self.upscale_check.blockSignals(True)
        self.upscale_check.setChecked(doc.upscale)
        self.upscale_check.blockSignals(False)
        self.caption.setPlainText(doc.caption)
        self._loading = False
        self.save_button.setEnabled(self._image is not None)
        self._refresh_grok()

    def navigate(self, delta):
        if self._image_task is not None:
            return
        index = self.files.currentRow() + delta
        while 0 <= index < self.files.count():
            if not self.files.item(index).isHidden():
                self.files.setCurrentRow(index)
                return
            index += delta

    def _document_changed(self):
        if self.document is None:
            return
        self._touch()
        item = self.files.item(self._active_index)
        if item:
            item.setText(self._file_label(self.document))
        self._update_progress()

    def apply_template(self):
        """Put the selected prepared caption into the editor, rendered in the colour chosen for the bars."""
        self.caption.setPlainText(render_caption_template(self.templates.currentData(), self.color.currentData()))

    def _caption_changed(self):
        if not self._loading:
            self._sync_caption()

    def _variation_changed(self, moved):
        low, high = self.variation_min, self.variation_max
        if low.value() > high.value():
            other = high if moved is low else low
            other.blockSignals(True)
            other.setValue(moved.value())
            other.blockSignals(False)

    def _upscale_changed(self, checked):
        if not self._loading and self.document is not None:
            self.document.upscale = bool(checked)
            self._document_changed()

    def _sync_caption(self):
        """Mirror the editor into the document with ``{color}`` resolved; the editor keeps the typed text."""
        doc = self.document
        if doc is None:
            return
        caption = render_caption_template(self.caption.toPlainText(), self.color.currentData())
        self._last_caption = caption
        if caption != doc.caption:
            doc.caption = caption
            doc.caption_touched = True
            self._document_changed()

    def _recolor_caption(self, *_):
        """Follow the colour selector: a prepared caption keeps its type and changes colour, custom text stays."""
        if self._loading:
            return
        text = self.caption.toPlainText()
        template = None if COLOR_PLACEHOLDER in text else caption_template_for(text)
        if template is None:
            self._sync_caption()  # a typed {color} follows the selector without rewriting the editor
            return
        recolored = render_caption_template(template, self.color.currentData())
        if recolored != text:
            self.caption.setPlainText(recolored)

    def _follow_bar_color(self, doc):
        """After bars were added: one colour on the page moves the selector and recolours a prepared caption.

        Mixed colours leave both alone, and a hand-written caption is never rewritten.
        """
        colors = {str(bar.color).lower() for bar in doc.bars} | {stroke.color.lower() for stroke in doc.strokes}
        if len(colors) != 1:
            return
        color = colors.pop()
        index = self.color.findData(color)
        if index < 0:
            return
        if index != self.color.currentIndex():
            blockers = [QSignalBlocker(self.color)]  # moving the brush colour must not recolour a selected bar
            self.color.setCurrentIndex(index)
            del blockers
            self.canvas.set_brush(self.thickness.value(), color)
        self._recolor_caption()

    def _brush_changed(self, *_):
        if not hasattr(self, "canvas"):
            return
        self.canvas.set_brush(self.thickness.value(), self.color.currentData())
        if not self._loading:
            self.canvas.update_selected(thickness=self.thickness.value(), color=self.color.currentData(),
                                        angle=self.bar_angle.value())

    def _selection_changed(self, bar):
        if bar is not None:
            blockers = [QSignalBlocker(self.thickness), QSignalBlocker(self.color), QSignalBlocker(self.bar_angle)]
            self.thickness.setValue(bar.height)
            self.color.setCurrentIndex(max(0, self.color.findData(bar.color)))
            self.bar_angle.setValue(bar.angle)
            del blockers
            self.canvas.set_brush(self.thickness.value(), self.color.currentData())

    def _apply_preset(self, index):
        value = self.presets.itemData(index)
        if value:
            blockers = [QSignalBlocker(self.thickness), QSignalBlocker(self.color)]
            self.thickness.setValue(value[0])
            self.color.setCurrentIndex(self.color.findData(value[1]))
            del blockers
            self._brush_changed()
            self._recolor_caption()

    def _clear_bars(self):
        if self.document and (self.document.bars or self.document.strokes):
            if QMessageBox.question(self, "Очистить разметку?", "Удалить все полосы и штрихи кисти текущего изображения? Это действие можно отменить на канвасе.") == QMessageBox.StandardButton.Yes:
                self.canvas.clear_bars()

    def add_auto_bars(self):
        if self.document is None or self._image is None:
            return
        bars = auto_bars(self._image.width, self._image.height, self.document.crop,
                         count=self.auto_count.value(), thickness=self.thickness.value(), color=self.color.currentData())
        self.canvas.add_bars(bars)
        self.statusBar().showMessage("Добавлены центральные полосы. Перейдите в «Выбор» для правки; Ctrl+Z — отмена.", 6000)

    def automatic_censor(self):
        doc = self.document
        if doc is None or self._image is None or self._image_task is not None:
            return
        if self._batch_dialog is not None:
            self._batch_dialog.raise_()
            self._batch_dialog.activateWindow()
            return
        if self._auto_dialog is not None:
            self._auto_dialog.raise_()
            self._auto_dialog.activateWindow()
            return
        from .autocensor import LocalCensorEngine
        from .autocensor_ui import AutoCensorDialog
        if self._auto_engine is None:
            self._auto_engine = LocalCensorEngine()
        area = doc.crop or Rect(0, 0, self._image.width, self._image.height)
        left, top = max(0, area.x), max(0, area.y)
        right = min(self._image.width, area.x + area.width)
        bottom = min(self._image.height, area.y + area.height)
        image = self._image.crop((left, top, right, bottom))
        signature = doc.signature()

        def apply(bars, replace):
            nonlocal signature
            if self.document is not doc or doc.signature() != signature:
                return False
            placed = [Bar(bar.x + left, bar.y + top, bar.width, bar.height, bar.color, bar.angle) for bar in bars]
            self.canvas.add_bars(placed, replace=replace)
            self._follow_bar_color(doc)
            signature = doc.signature()
            return True

        dialog = AutoCensorDialog(image, self.color.currentData(), apply, self, engine=self._auto_engine,
                                  variation_min=self.variation_min.value(), variation_max=self.variation_max.value())
        self._auto_dialog = dialog
        dialog.finished.connect(lambda _result, current=dialog: self._release_auto_dialog(current))
        dialog.destroyed.connect(lambda _object, current=dialog: self._release_auto_dialog(current))
        dialog.open()

    def _release_auto_dialog(self, dialog):
        if self._auto_dialog is dialog:
            self._auto_dialog = None

    def automatic_dataset(self):
        if not self.session.documents:
            return
        if self._batch_dialog is not None:
            self._batch_dialog.raise_()
            self._batch_dialog.activateWindow()
            return
        if self._auto_dialog is not None:
            self._auto_dialog.raise_()
            self._auto_dialog.activateWindow()
            return
        from .autocensor import CensorOptions, LocalCensorEngine
        from .automation_ui import BatchDatasetDialog
        if self._auto_engine is None:
            self._auto_engine = LocalCensorEngine()
        session = self.session
        caption = self.caption.toPlainText()
        template = caption_template_for(caption) or caption or CAPTION_TEMPLATES[0][1]

        def adopt(documents):
            if self.session is not session:
                return False
            return self._adopt_prepared_documents(documents, dialog.output_dir)

        dialog = BatchDatasetDialog(
            list(dict.fromkeys(doc.source for doc in session.documents)), adopt, self,
            output_dir=session.output_dir, engine=self._auto_engine, upscaler=self._upscale_engine,
            censor_options=CensorOptions(color=self.color.currentData(), thickness=min(64, self.thickness.value()),
                                         variation_min=self.variation_min.value(), variation_max=self.variation_max.value()),
            current_source=self.document.source if self.document is not None else None,
        )
        self._batch_dialog = dialog
        dialog.finished.connect(lambda _result, current=dialog: self._release_batch_dialog(current))
        dialog.destroyed.connect(lambda _object, current=dialog: self._release_batch_dialog(current))
        dialog.open()

    def _release_batch_dialog(self, dialog):
        if self._batch_dialog is dialog:
            self._batch_dialog = None

    def _adopt_prepared_documents(self, documents, output_dir):
        """Keep each prepared crop editable without sharing mutable state with the batch dialog."""
        if not documents:
            return True
        changed = False
        if output_dir and any(doc.export_name is not None for doc in documents):
            old_root = Path(self.session.output_dir).resolve() if self.session.output_dir else None
            if old_root != Path(output_dir).resolve():
                if any(doc.export_name for doc in self.session.documents):
                    answer = QMessageBox.question(
                        self._batch_dialog or self, "Сменить датасет сессии?",
                        "Пакет сохранён в другой папке. Переключить сессию на неё? Привязки к прежним "
                        "экспортам сбросятся; сами файлы останутся. Отдельная пакетная сессия уже сохранена.",
                    )
                    if answer != QMessageBox.StandardButton.Yes:
                        return False
                self.session.output_dir = str(Path(output_dir).resolve())
                for doc in self.session.documents:
                    doc.export_name = None
                    doc.exported_signature = None
                changed = True
        existing = {(Path(doc.source).resolve(), doc.signature()): index
                    for index, doc in enumerate(self.session.documents)}
        first = None
        for prepared in documents:
            key = (Path(prepared.source).resolve(), prepared.signature())
            index = existing.get(key)
            if index is None:
                index = len(self.session.documents)
                self.session.documents.append(copy.deepcopy(prepared))
                existing[key] = index
                changed = True
            else:
                current = self.session.documents[index]
                if prepared.export_name is not None and (
                        current.export_name != prepared.export_name
                        or current.exported_signature != prepared.exported_signature):
                    current.export_name = prepared.export_name
                    current.exported_signature = prepared.exported_signature
                    changed = True
            if first is None:
                first = index
        if changed:
            self.session.current_index = first
            self._populate_files()
            self._touch()
        self.statusBar().showMessage(f"Подготовленные документы в сессии: {len(documents)}.", 7000)
        return True


    def _show_output_dir(self):
        folder = self.session.output_dir
        self.output_label.setText(folder)
        self.output_label.setToolTip(folder or "Выберите папку датасета в верхней панели («Папка датасета…»)")

    def choose_output(self):
        if self._image_task is not None:
            return False
        folder = QFileDialog.getExistingDirectory(self, "Папка датасета (содержит clean и censored)", self.session.output_dir)
        if not folder or folder == self.session.output_dir:
            return bool(folder)
        if any(doc.export_name for doc in self.session.documents):
            answer = QMessageBox.question(self, "Сменить датасет?", "Привязки к прошлым экспортам будут сброшены. Файлы прежнего датасета останутся на диске.")
            if answer != QMessageBox.StandardButton.Yes:
                return False
        self.session.output_dir = folder
        for doc in self.session.documents:
            doc.export_name = None
            doc.exported_signature = None
        self._show_output_dir()
        for index, doc in enumerate(self.session.documents):
            self.files.item(index).setText(self._file_label(doc))
        self._touch()
        self._update_progress(refresh_disk=True)
        return True

    def _update_progress(self, *, refresh_disk=False):
        count = len(self.session.documents)
        self.progress.setRange(0, max(count, 1))
        ready = sum(doc.is_exported for doc in self.session.documents)
        self.progress.setValue(ready)
        if refresh_disk:
            self._disk_pair_count = 0
            if self.session.output_dir:
                root = Path(self.session.output_dir)
                try:
                    self._disk_pair_count = sum(
                        (root / "censored" / path.name).is_file() and path.with_suffix(".txt").is_file()
                        for path in (root / "clean").glob("*.png"))
                except OSError:
                    pass
        self.pair_count.setText(f"Пар на диске: {self._disk_pair_count}\nГотово документов: {ready} / {count}")

    def _set_image_busy(self, busy):
        self.files.setEnabled(not busy)
        self.preview_button.setEnabled(not busy and self._image is not None)
        self.save_button.setEnabled(not busy and self._image is not None)
        self.upscale_check.setEnabled(not busy)
        self.upscale_progress.setVisible(busy)
        self.cancel_upscale.setVisible(busy)

    def _start_image_task(self, work, on_done, on_error=None):
        if self._image_task is not None:
            return False
        from .upscale_ui import ImageTask
        task = ImageTask(work, self)
        self._image_task = task
        task.progressed.connect(self._image_task_progress)
        task.finished.connect(lambda current=task: self._finish_image_task(current, on_done, on_error))
        self._set_image_busy(True)
        task.start()
        return True

    def _image_task_progress(self, done, total, text):
        self.upscale_progress.setRange(0, max(1, total))
        self.upscale_progress.setValue(done)
        if text:
            self.statusBar().showMessage(text)

    def _finish_image_task(self, task, on_done, on_error):
        if task is not self._image_task:
            return
        self._image_task = None
        self._set_image_busy(False)
        if task.cancelled:
            self.statusBar().showMessage("Апскейл остановлен; незавершённая пара не сохранена.", 7000)
        elif task.error is not None:
            self._error("Операция с изображением не выполнена", task.error)
        if (task.cancelled or task.error is not None) and on_error is not None:
            on_error()
        elif not task.cancelled and task.error is None:
            on_done(task.result)
        task.deleteLater()

    def _cancel_image_task(self):
        if self._image_task is not None:
            self._image_task.cancel()
            self.statusBar().showMessage("Остановка после текущего тайла RealESRGAN…")

    def preview_pair(self):
        doc = self.document
        if doc is None or self._image is None or self._image_task is not None:
            return
        snapshot = copy.deepcopy(doc)
        if snapshot.upscale:
            self._start_image_task(
                lambda progress, cancelled: (*render_pair(snapshot, upscaler=self._upscale_engine,
                    progress=progress, cancelled=cancelled), snapshot.caption),
                lambda result: self._show_preview_result(*result))
            return
        try:
            clean, censored = render_pair(snapshot)
        except (OSError, ValueError) as exc:
            self._error("Ошибка превью", exc)
            return
        self._show_preview_result(clean, censored, snapshot.caption)

    def _show_preview_result(self, clean, censored, caption_text):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Превью экспорта · {clean.width} × {clean.height} px")
        layout = QVBoxLayout(dialog)
        columns = QHBoxLayout()
        for title, image in (("CLEAN · TARGET", clean), ("CENSORED · CONTROL", censored)):
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            column.addWidget(PreviewView(image), 1)
            columns.addLayout(column)
        layout.addLayout(columns, 1)
        caption = QPlainTextEdit(caption_text)
        caption.setReadOnly(True)
        caption.setMaximumHeight(90)
        pages = PageNavigation(caption)
        pages.setMaximumHeight(124)
        layout.addWidget(pages)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        fit_window(dialog, (1150, 760))
        dialog.exec()

    def save_pair(self):
        doc = self.document
        if doc is None or self._image is None or self._image_task is not None:
            return
        if not doc.bars and not doc.strokes:
            if QMessageBox.question(self, "Нет разметки", "Полос и штрихов нет: clean и censored будут одинаковыми. Сохранить такую пару?") != QMessageBox.StandardButton.Yes:
                return
        if not doc.caption.strip():
            if QMessageBox.question(self, "Пустой капшен", "Сохранить пару с пустым .txt?") != QMessageBox.StandardButton.Yes:
                return
        if not self.session.output_dir and not self.choose_output():
            return
        old_name = doc.export_name
        old_signature = doc.exported_signature
        overwrite = False
        if self.new_pair.isChecked():
            doc.export_name = None
        elif doc.export_name:
            answer = QMessageBox.question(self, "Обновить пару?", f"Перезаписать пару {doc.export_name}? Для другой панели включите «Сохранить как новую пару».")
            if answer != QMessageBox.StandardButton.Yes:
                return
            overwrite = True

        def restore():
            doc.export_name = old_name
            doc.exported_signature = old_signature

        if doc.upscale:
            snapshot = copy.deepcopy(doc)
            expected = snapshot.signature()
            output = Path(self.session.output_dir)
            def work(progress, cancelled):
                export_pair(snapshot, output, overwrite=overwrite, upscaler=self._upscale_engine,
                            progress=progress, cancelled=cancelled)
                return snapshot.export_name, snapshot.exported_signature
            def published(result):
                stem, signature = result
                if self.document is doc and doc.signature() == expected:
                    doc.export_name, doc.exported_signature = stem, signature
                    self._document_changed()
                    self.statusBar().showMessage(f"Сохранены x4: clean/{stem}.png, clean/{stem}.txt, censored/{stem}.png", 9000)
                else:
                    QMessageBox.warning(self, "Документ изменился", f"Пара {stem} сохранена, но текущая разметка изменилась во время апскейла. Она не связана с этим экспортом.")
                self._update_progress(refresh_disk=True)
            self._start_image_task(work, published, restore)
            return
        try:
            stem = export_pair(doc, Path(self.session.output_dir), overwrite=overwrite)
        except (OSError, ValueError) as exc:
            restore()
            self._error("Пара не сохранена", exc)
            return
        self._update_progress(refresh_disk=True)
        self._document_changed()
        self.statusBar().showMessage(f"Сохранены clean/{stem}.png, clean/{stem}.txt, censored/{stem}.png", 9000)

    def _refresh_grok(self):
        available = bool(self.grok.available)
        self.ask_button.setEnabled(available and self._image is not None)
        self.grok_hint.setText("Изображение отправляется по кнопке «Отправить в Grok»." if available
                               else "Grok выключен или вход не выполнен. Войдите через xAI OAuth в «Настройки Grok». Локальные инструменты работают без сети.")

    def grok_settings(self):
        self.grok.settings_dialog(self)
        self._refresh_grok()

    def ask_grok(self):
        doc = self.document
        if doc is None or self._image is None:
            return
        try:
            image, _ = render_pair(replace(doc, upscale=False))
        except (OSError, ValueError) as exc:
            self._error("Ошибка чтения изображения", exc)
            return
        signature = doc.signature()
        crop = doc.crop or Rect(0, 0, self._image.width, self._image.height)
        # Loaded session rectangles may extend beyond the current source.
        left, top = max(0, crop.x), max(0, crop.y)
        crop = Rect(left, top, image.width, image.height)

        def still_current():
            if self.document is not doc or doc.signature() != signature:
                QMessageBox.warning(self, "Изображение изменилось", "Ответ относится к предыдущему состоянию. Повторите запрос для текущего изображения.")
                return False
            return True

        def apply_caption(text):
            if not still_current():
                return False
            self.caption.setPlainText(text)
            return True

        def apply_bars(boxes):
            if not still_current():
                return False
            bars = []
            for box in boxes:
                x = crop.x + round(box["x"] * crop.width)
                y = crop.y + round(box["y"] * crop.height)
                width = max(1, round(box["width"] * crop.width))
                height = max(1, round(box["height"] * crop.height))
                bars.append(Bar(x, y, width, height, self.color.currentData()))
            self.canvas.add_bars(bars)
            return True

        self.grok.ask(self, image, doc.caption, apply_caption, apply_bars)

    def show_help(self):
        QMessageBox.information(self, "Горячие клавиши", "Ctrl+O — открыть папку\nCtrl+L — загрузить сессию\nCtrl+Shift+S — сохранить сессию\nCtrl+S — сохранить пару\nCtrl+P — превью\nCtrl+Shift+A — локальный нейроавтоцензор\nCtrl+Alt+A — автодатасет\nAlt+← / Alt+→ — предыдущий / следующий\n\nПри фокусе на канвасе:\nV — выбор\nB — полоса: первый клик — начало, второй — конец\nP — кисть: протягивание ЛКМ\nC — кроп\nF — вписать\nDelete — удалить выделенную полосу\nEsc — отменить текущий жест\nCtrl+Z / Ctrl+Shift+Z — отмена / повтор\nКолесо — зум; средняя кнопка — панорама\nБоковая кнопка + движение влево/вправо — толщина\n\n✓ — экспорт актуален; ● — экспорт есть, разметка изменена.")

    def closeEvent(self, event: QCloseEvent):
        if self._image_task is not None:
            self._image_task.cancel()
            self.statusBar().showMessage("Дождитесь остановки апскейла и закройте окно повторно.")
            event.ignore()
            return
        if self._maybe_save_session():
            if self._auto_dialog is not None:
                self._auto_dialog.reject()
            if self._batch_dialog is not None:
                self._batch_dialog.reject()
            event.accept()
        else:
            event.ignore()
