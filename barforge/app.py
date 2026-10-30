"""Application entry point, independent of the current working directory."""
from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication


STYLE = """
QMainWindow, QDialog { background: #181c23; }
QWidget { color: #e3e8f0; font-size: 13px; }
QToolBar { background: #202630; border: none; spacing: 6px; padding: 7px; }
QToolBar QToolButton { padding: 7px 9px; border-radius: 4px; }
QToolBar QToolButton:hover, QToolBar QToolButton:checked { background: #354555; }
QGroupBox { border: 1px solid #384352; border-radius: 7px; margin-top: 14px; padding: 12px 8px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #a4b7cd; }
QPushButton { background: #2b3542; border: 1px solid #46566a; border-radius: 5px; padding: 7px 9px; }
QPushButton:hover { background: #37485c; border-color: #7ca7c4; }
QPushButton:pressed { background: #202b36; }
QPushButton:disabled { color: #75808d; background: #232a33; border-color: #343e49; }
QPushButton#primary { background: #427b77; border-color: #63aaa0; color: white; font-weight: bold; padding: 11px; }
QPushButton#primary:hover { background: #4d9188; }
QLineEdit, QPlainTextEdit, QSpinBox, QComboBox { background: #12171e; border: 1px solid #3a4758; border-radius: 4px; padding: 5px; selection-background-color: #436b80; }
QComboBox QAbstractItemView { background: #202630; selection-background-color: #436b80; }
QListWidget { background: #12171e; border: 1px solid #303d4b; border-radius: 5px; outline: 0; }
QListWidget::item { padding: 7px 4px; }
QListWidget::item:selected { background: #304c5a; color: #eaf9f7; }
QGraphicsView { background: #10151c; border: 1px solid #303d4b; border-radius: 5px; }
QProgressBar { border: 1px solid #3a4758; border-radius: 4px; background: #12171e; text-align: center; min-height: 20px; }
QProgressBar::chunk { background: #386e69; }
QStatusBar { color: #a4b7cd; }
QLabel#brand { color: #86d7c8; font-size: 22px; font-weight: bold; padding: 6px 12px 6px 0; }
QSplitter::handle { background: #202630; width: 5px; }
QToolTip { color: #edf3f9; background: #293747; border: 1px solid #637e99; }
"""


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("BarForge")
    app.setOrganizationName("BarForge")
    app.setApplicationVersion("1.1.0")
    from .i18n import initialize_language
    initialize_language()
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#181c23"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e3e8f0"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#12171e"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e3e8f0"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#2b3542"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e3e8f0"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#436b80"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#293747"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#edf3f9"))
    app.setPalette(palette)
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLE)
    from .main_window import MainWindow
    window = MainWindow()
    window.show()
    return app.exec()
