"""Consumer-visible batch review transitions; neutral images and no external inference/network."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import threading
import time
import unittest

from PIL import Image
import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from barforge.autocensor import Detection, DetectionResult
from barforge.automation_ui import BatchDatasetDialog
from barforge.models import Bar, Rect


class FixedEngine:
    def __init__(self):
        self.calls = 0

    def analyze(self, image, options, progress=None, cancelled=None):
        self.calls += 1
        return DetectionResult([Detection("genital", "penis", .9, Rect(250, 200, 100, 90))], [],
                               ("neutral-fixture",), 1, .001)


class GatedEngine(FixedEngine):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def analyze(self, *args, **kwargs):
        self.entered.set()
        try:
            if not self.release.wait(3):
                raise RuntimeError("Test did not release its engine")
            return super().analyze(*args, **kwargs)
        finally:
            self.finished.set()


class SecondPageGate(GatedEngine):
    def analyze(self, *args, **kwargs):
        if self.calls == 0:
            return FixedEngine.analyze(self, *args, **kwargs)
        return super().analyze(*args, **kwargs)


class BatchReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "neutral.png"
        Image.new("RGB", (800, 600), (80, 80, 80)).save(self.source)
        self.dialogs = []

    def tearDown(self):
        for dialog in self.dialogs:
            if shiboken6.isValid(dialog):
                dialog.reject()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.temp.cleanup()

    def make_dialog(self, engine):
        dialog = BatchDatasetDialog([str(self.source)], lambda documents: True,
                                    output_dir=str(self.root / "dataset"), engine=engine)
        self.dialogs.append(dialog)
        dialog.open()
        return dialog

    def pump(self, milliseconds=10):
        loop = QEventLoop()
        QTimer.singleShot(milliseconds, loop.quit)
        loop.exec()

    def until(self, predicate):
        deadline = time.monotonic() + 5
        while not predicate() and time.monotonic() < deadline:
            self.pump()
        self.assertTrue(predicate(), "Batch operation did not reach its expected state")

    def test_settings_do_not_discard_manual_review_edits(self):
        engine = FixedEngine()
        dialog = self.make_dialog(engine)
        dialog.run_button.click()
        self.until(lambda: dialog._job is None and bool(dialog._candidates))
        dialog.candidates_list.setCurrentRow(0)
        document = dialog._candidates[0].document
        dialog.canvas.add_bars([Bar(250, 220, 100, 4, "#000000", 30)])
        manual_signature = document.signature()
        dialog.context_spin.setValue(4.0)
        dialog.color_combo.setCurrentIndex(dialog.color_combo.findData("#000000"))
        dialog.variation_min_spin.setValue(1)
        self.pump()
        self.assertEqual(document.signature(), manual_signature)
        self.assertIs(dialog._candidates[0].document, document)
        self.assertEqual(engine.calls, 1)

    def test_repeated_export_keeps_exactly_the_original_dataset_files(self):
        dialog = self.make_dialog(FixedEngine())
        dialog.run_button.click()
        self.until(lambda: dialog._job is None and bool(dialog._candidates))
        dialog.export_button.click()
        self.until(lambda: dialog._job is None and dialog._candidates[0].document.is_exported)
        dataset = self.root / "dataset"
        before = {path.relative_to(dataset).as_posix(): path.read_bytes() for path in dataset.rglob("*") if path.is_file()}
        self.assertEqual({name for name in before if name.endswith(".png")}, {"clean/001.png", "censored/001.png"})
        self.assertEqual(before["clean/001.txt"].decode("utf-8"), dialog._candidates[0].document.caption)
        dialog.export_button.click()
        self.pump(50)
        after = {path.relative_to(dataset).as_posix(): path.read_bytes() for path in dataset.rglob("*") if path.is_file()}
        self.assertEqual(after, before)

    def test_closed_analysis_cannot_populate_a_new_review_window(self):
        engine = GatedEngine()
        old = self.make_dialog(engine)
        old.run_button.click()
        self.until(engine.entered.is_set)
        old.reject()
        new = self.make_dialog(FixedEngine())
        engine.release.set()
        self.until(engine.finished.is_set)
        self.pump(50)
        self.assertEqual(new._pages, [])
        self.assertEqual(new._candidates, [])
        self.assertEqual(list((self.root / "dataset").glob("**/*.png")), [])

    def test_stopping_a_folder_keeps_completed_pages_available_for_review(self):
        second_source = self.root / "second.png"
        Image.new("RGB", (800, 600), (90, 90, 90)).save(second_source)
        engine = SecondPageGate()
        dialog = BatchDatasetDialog([str(self.source), str(second_source)], lambda documents: True,
                                    output_dir=str(self.root / "dataset"), engine=engine)
        self.dialogs.append(dialog)
        dialog.open()
        dialog.run_button.click()
        self.until(engine.entered.is_set)
        dialog.stop_button.click()
        engine.release.set()
        self.until(lambda: dialog._job is None)
        self.assertEqual([page.source for page in dialog._pages], [str(self.source.resolve())])
        self.assertEqual([item.document.source for item in dialog._candidates], [str(self.source.resolve())])
        self.assertTrue(dialog.export_button.isEnabled())

    def test_rebuild_remembers_exports_but_changed_geometry_becomes_a_new_pair(self):
        dialog = self.make_dialog(FixedEngine())
        dialog.run_button.click()
        self.until(lambda: dialog._job is None and bool(dialog._candidates))
        dialog.export_button.click()
        self.until(lambda: dialog._job is None and dialog._candidates[0].document.is_exported)
        output = self.root / "dataset"
        original = (output / "censored" / "001.png").read_bytes()
        dialog.rebuild_button.click()
        self.assertTrue(dialog._candidates[0].document.is_exported)
        self.assertFalse(dialog.export_button.isEnabled())
        dialog.angle_spin.setValue(0)
        dialog.rebuild_button.click()
        self.assertFalse(dialog._candidates[0].document.is_exported)
        dialog.export_button.click()
        self.until(lambda: dialog._job is None and dialog._candidates[0].document.is_exported)
        self.assertEqual(dialog._candidates[0].document.export_name, "002")
        self.assertEqual((output / "censored" / "001.png").read_bytes(), original)
        self.assertNotEqual((output / "censored" / "002.png").read_bytes(), original)



if __name__ == "__main__":
    unittest.main()
