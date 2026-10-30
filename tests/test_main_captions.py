"""Caption lifecycle across pages and session reopen, without external services."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image
from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

from barforge.main_window import MainWindow
from barforge.models import BrushStroke, Document, Rect, Session
from barforge.storage import save_session


class OfflineGrok(QObject):
    availability_changed = Signal(bool)
    available = False
    def __init__(self, *args, **kwargs):
        super().__init__()
    def shutdown(self):
        pass


class CaptionNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.windows = []
        self.grok_patch = patch("barforge.grok_ui.GrokController",OfflineGrok)
        self.grok_patch.start()

    def tearDown(self):
        for window in self.windows:
            window.session_dirty = False
            window.close()
            window.deleteLater()
        self.grok_patch.stop()
        self.app.processEvents()
        self.temp.cleanup()

    def window(self):
        window = MainWindow()
        self.windows.append(window)
        return window

    def test_pages_keep_their_caption_and_unedited_pages_inherit_the_last_one(self):
        documents=[]
        for name in ("001.png","002.png","003.png"):
            path=self.root/name
            Image.new("RGB",(20,20),(80,80,80)).save(path)
            documents.append(Document(str(path)))
        window=self.window()
        window.session=Session(documents=documents,source_dir=str(self.root))
        window._populate_files()
        window.caption.setPlainText("caption for first")
        window.files.setCurrentRow(1)
        self.assertEqual(documents[0].caption,"caption for first")
        self.assertEqual(documents[1].caption,"caption for first")
        window.caption.setPlainText("caption for second")
        window.files.setCurrentRow(0)
        self.assertEqual(window.caption.toPlainText(),"caption for first")
        window.files.setCurrentRow(1)
        self.assertEqual(window.caption.toPlainText(),"caption for second")
        window.files.setCurrentRow(2)
        self.assertEqual(documents[2].caption,"caption for second")
        window.files.setCurrentRow(1)
        self.assertEqual(window.caption.toPlainText(),"caption for second")

    def test_session_reopen_does_not_recolor_saved_captions(self):
        source=self.root/"panel.png"
        Image.new("RGB",(20,20),(80,80,80)).save(source)
        path=self.root/"work.barforge.json"
        save_session(Session(documents=[Document(str(source),caption="{color} stays literal",caption_touched=True)]),path)
        window=self.window()
        window.color.setCurrentIndex(window.color.findData("#000000"))
        with patch("barforge.main_window.QFileDialog.getOpenFileName",return_value=(str(path),"BarForge")):
            window.open_session()
        self.assertEqual(window.document.caption,"{color} stays literal")
        self.assertEqual(window.caption.toPlainText(),"{color} stays literal")
        self.assertFalse(window.session_dirty)

    def test_upscale_and_random_range_are_per_document_and_ordered(self):
        source=self.root/"panel.png"
        Image.new("RGB",(20,20),(80,80,80)).save(source)
        window=self.window()
        window.session=Session(documents=[Document(str(source))])
        window._populate_files()
        window.upscale_check.setChecked(True)
        self.assertTrue(window.document.upscale)
        self.assertNotEqual(window.document.exported_signature,window.document.signature())
        window.variation_min.setValue(9)
        self.assertEqual(window.variation_max.value(),9)
        window.variation_max.setValue(2)
        self.assertEqual(window.variation_min.value(),2)
        window.upscale_check.setChecked(False)
        self.assertFalse(window.document.upscale)

    def test_upscaled_export_runs_in_background_and_publishes_aligned_pair(self):
        class NearestX4:
            def upscale(self,image,**kwargs):
                return image.resize((image.width*4,image.height*4),Image.Resampling.NEAREST)
        source=self.root/"panel.png"
        Image.new("RGB",(80,60),(80,80,80)).save(source)
        document=Document(str(source),strokes=[BrushStroke(((15,15),(35,25)),5,"#000000")],
                          crop=Rect(10,10,40,30),caption="worker",upscale=True,caption_touched=True)
        window=self.window()
        window._upscale_engine=NearestX4()
        window.session=Session(documents=[document],output_dir=str(self.root/"dataset"))
        window._populate_files()
        window.save_pair()
        self.assertIsNotNone(window._image_task)
        deadline=time.monotonic()+5
        while window._image_task is not None and time.monotonic()<deadline:
            loop=QEventLoop()
            QTimer.singleShot(10,loop.quit)
            loop.exec()
        self.assertIsNone(window._image_task)
        self.assertTrue(document.is_exported)
        name=document.export_name
        with Image.open(self.root/"dataset"/"clean"/f"{name}.png") as clean, Image.open(self.root/"dataset"/"censored"/f"{name}.png") as control:
            self.assertEqual(clean.size,(160,120))
            self.assertEqual(control.size,clean.size)
            self.assertNotEqual(clean.tobytes(),control.tobytes())


if __name__ == "__main__":
    unittest.main()
