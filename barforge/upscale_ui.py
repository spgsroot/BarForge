"""Small owner-scoped Qt worker for expensive single-image preview/export tasks."""
from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from .upscale import UpscaleCancelled


class _Runner(QRunnable):
    def __init__(self, task: "ImageTask"):
        super().__init__()
        self.task = task

    def run(self):
        task = self.task
        try:
            def progress(done, total, text=""):
                with contextlib.suppress(RuntimeError):
                    task.progressed.emit(done, total, text)
            task.result = task.work(progress, task.stop.is_set)
        except UpscaleCancelled:
            task.cancelled = True
        except Exception as exc:
            task.error = exc
        finally:
            with contextlib.suppress(RuntimeError):
                task.finished.emit()


class ImageTask(QObject):
    """Run one callable in a private single-thread pool; the owner decides whether to accept its result."""

    progressed = Signal(int, int, str)
    finished = Signal()

    def __init__(self, work: Callable, parent=None):
        super().__init__(parent)
        self.work = work
        self.stop = threading.Event()
        self.result = None
        self.error = None
        self.cancelled = False
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)

    def start(self):
        self.pool.start(_Runner(self))

    def cancel(self):
        self.stop.set()

    def shutdown(self):
        self.cancel()
        self.pool.waitForDone(3000)
