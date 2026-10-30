"""Screen-bounded windows and explicit page navigation for long lists/text, without scroll bars."""
from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QSize, Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractScrollArea, QAbstractSlider, QHBoxLayout, QLabel, QPlainTextEdit,
    QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)


def fit_window(window: QWidget, preferred: tuple[int, int] | None = None) -> None:
    """Fit a laid-out window to its screen; callers must keep their content's minimum size usable."""
    screen = window.screen()
    if screen is None:
        return
    area = screen.availableGeometry().adjusted(12, 12, -12, -12)
    if window.layout() is not None:
        window.layout().activate()
    hint = window.sizeHint()
    requested = QSize(*preferred) if preferred is not None else hint
    minimum = window.minimumSizeHint().expandedTo(window.minimumSize())
    frame = window.frameGeometry()
    extra = QSize(max(8, frame.width() - window.width()), max(32, frame.height() - window.height()))
    maximum = QSize(max(1, area.width() - extra.width()), max(1, area.height() - extra.height()))
    target = requested.expandedTo(minimum).boundedTo(maximum)
    window.resize(target)
    # Do not recenter a window the user has placed; just keep its frame on this screen.
    frame = window.frameGeometry()
    left = max(area.left(), min(frame.left(), area.right() - frame.width() + 1))
    top = max(area.top(), min(frame.top(), area.bottom() - frame.height() + 1))
    if window.isVisible() and (frame.left() != left or frame.top() != top):
        window.move(left, top)


class PageNavigation(QWidget):
    """Keep a native list/editor and its selection semantics, exposing page buttons instead of scroll bars.

    Widget indices, selection signals, editing and keyboard navigation remain the original widget's.
    Navigation appears only when the contents do not fit. All content stays reachable, including text
    entered later and list items added/filtered after construction.
    """

    def __init__(self, area: QAbstractScrollArea, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.area = area
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        if isinstance(area, QPlainTextEdit):
            area.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(area, 1)
        self.navigation = QWidget(self)
        row = QHBoxLayout(self.navigation)
        row.setContentsMargins(0, 0, 0, 0)
        self.previous_button = QToolButton()
        self.previous_button.setText("←")
        self.previous_button.setToolTip("Предыдущая страница")
        self.next_button = QToolButton()
        self.next_button.setText("→")
        self.next_button.setToolTip("Следующая страница")
        self.page_label = QLabel()
        self.page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self.previous_button)
        row.addWidget(self.page_label, 1)
        row.addWidget(self.next_button)
        layout.addWidget(self.navigation)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        bar = area.verticalScrollBar()
        self.previous_button.clicked.connect(
            lambda: bar.triggerAction(QAbstractSlider.SliderAction.SliderPageStepSub))
        self.next_button.clicked.connect(
            lambda: bar.triggerAction(QAbstractSlider.SliderAction.SliderPageStepAdd))
        bar.valueChanged.connect(self._refresh)
        bar.rangeChanged.connect(self._refresh)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._refresh)
        area.viewport().installEventFilter(self)
        self._refresh()

    def _refresh(self, *_args: object) -> None:
        bar = self.area.verticalScrollBar()
        low, high, value = bar.minimum(), bar.maximum(), bar.value()
        step = max(1, bar.pageStep())
        overflow = high > low
        self.navigation.setVisible(overflow)
        self.previous_button.setEnabled(value > low)
        self.next_button.setEnabled(value < high)
        pages = max(1, math.ceil((high - low) / step) + 1)
        current = pages if value == high else min(pages, (value - low) // step + 1)
        self.page_label.setText(f"Страница {current} / {pages}")

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Resize:
            self._refresh_timer.start(0)
        return super().eventFilter(watched, event)
