"""Batch preview/export with x4 uses the same edited document and stays aligned."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from barforge.autocensor import Detection, DetectionResult
from barforge.automation_ui import BatchDatasetDialog
from barforge.models import Rect
from barforge.storage import load_session


class Engine:
    def analyze(self,image,options,progress=None,cancelled=None):
        return DetectionResult([Detection("genital","penis",.9,Rect(20,20,40,30))],[],("fixture",),1,.001)


class NearestX4:
    def upscale(self,image,**kwargs):
        return image.resize((image.width*4,image.height*4),Image.Resampling.NEAREST)


class BatchUpscaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.source=self.root/"neutral.png"
        Image.new("RGB",(80,60),(80,80,80)).save(self.source)
        self.dialog=BatchDatasetDialog([str(self.source)],lambda documents:True,
            output_dir=str(self.root/"dataset"),engine=Engine(),upscaler=NearestX4())
        self.dialog.open()
        self.dialog.run_button.click()
        self.until(lambda:self.dialog._job is None and bool(self.dialog._candidates))

    def tearDown(self):
        self.dialog.reject()
        self.dialog.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def until(self,predicate):
        deadline=time.monotonic()+5
        while not predicate() and time.monotonic()<deadline:
            loop=QEventLoop();QTimer.singleShot(10,loop.quit);loop.exec()
        self.assertTrue(predicate())

    def test_background_preview_uses_x4_and_keeps_caption(self):
        document=self.dialog._candidates[0].document
        self.dialog.upscale_check.setChecked(True)
        self.dialog.caption_edit.setPlainText("batch caption")
        with patch.object(BatchDatasetDialog,"_show_preview_result") as preview:
            self.dialog._preview()
            self.until(lambda:preview.called)
            clean,control,caption=preview.call_args.args
            self.assertEqual(clean.size,control.size)
            self.assertEqual(clean.width,document.crop.width*4)
            self.assertEqual(caption,"batch caption")

    def test_batch_export_checkpoints_upscaled_pair_and_edited_caption(self):
        self.dialog.upscale_check.setChecked(True)
        self.dialog.caption_edit.setPlainText("batch caption")
        document=self.dialog._candidates[0].document
        self.dialog._export()
        self.until(lambda:self.dialog._job is None)
        self.assertTrue(document.is_exported)
        output=self.root/"dataset"
        with Image.open(output/"clean"/"001.png") as clean, Image.open(output/"censored"/"001.png") as control:
            self.assertEqual(clean.size,control.size)
            self.assertEqual(clean.width,document.crop.width*4)
        self.assertEqual((output/"clean"/"001.txt").read_text(encoding="utf-8"),"batch caption")
        checkpoints=list(output.glob("batch-*.barforge.json"))
        self.assertEqual(len(checkpoints),1)
        restored=load_session(checkpoints[0]).documents[0]
        self.assertTrue(restored.upscale)
        self.assertEqual(restored.caption,"batch caption")


if __name__=="__main__":
    unittest.main()
