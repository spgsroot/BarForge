"""Freehand strokes, side-button thickness and exact crop/export persistence."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import tempfile
import unittest

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from barforge.canvas import ImageCanvas
from barforge.imaging import render_pair
from barforge.models import Document


class BrushCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.source = Path(self.temp.name)/"neutral.png"
        from PIL import Image
        self.image = Image.new("RGB", (512,512), (80,80,80))
        self.image.save(self.source)
        self.doc = Document(str(self.source))
        self.canvas = ImageCanvas()
        self.canvas.resize(700,620)
        self.canvas.set_document(self.image,self.doc)
        self.canvas.show()
        self.app.processEvents()

    def tearDown(self):
        self.canvas.close()
        self.canvas.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def event(self, kind, x, y, button, buttons):
        point = self.canvas.mapFromScene(QPointF(x,y))
        return QMouseEvent(kind,QPointF(point),QPointF(self.canvas.viewport().mapToGlobal(point)),
                           button,buttons,Qt.KeyboardModifier.NoModifier)

    def test_freehand_stroke_commits_on_release_and_is_one_undo_step(self):
        self.canvas.set_tool("brush")
        self.canvas.set_brush(9,"#000000")
        QApplication.sendEvent(self.canvas.viewport(),self.event(QEvent.Type.MouseButtonPress,100,100,Qt.MouseButton.LeftButton,Qt.MouseButton.LeftButton))
        for x,y in ((140,120),(180,160),(220,180)):
            QApplication.sendEvent(self.canvas.viewport(),self.event(QEvent.Type.MouseMove,x,y,Qt.MouseButton.NoButton,Qt.MouseButton.LeftButton))
        self.assertEqual(self.doc.strokes,[],"Preview must not leak into export")
        QApplication.sendEvent(self.canvas.viewport(),self.event(QEvent.Type.MouseButtonRelease,220,180,Qt.MouseButton.LeftButton,Qt.MouseButton.NoButton))
        stroke, = self.doc.strokes
        self.assertEqual((stroke.thickness,stroke.color),(9,"#000000"))
        self.assertGreaterEqual(len(stroke.points),4)
        clean,control = render_pair(self.doc)
        self.assertEqual(control.getpixel((180,160)),(0,0,0))
        self.assertEqual(clean.getpixel((180,160)),(80,80,80))
        self.canvas.undo()
        self.assertEqual(self.doc.strokes,[])
        self.assertFalse(self.canvas.can_undo)
        self.canvas.redo()
        self.assertEqual(self.doc.strokes,[stroke])
        self.canvas.clear_bars()
        self.assertEqual(self.doc.strokes,[])
        self.canvas.undo()
        self.assertEqual(self.doc.strokes,[stroke])

    def test_held_side_buttons_adjust_thickness_in_both_directions(self):
        self.canvas.set_tool("brush")
        self.canvas.set_brush(12,"#ffffff")
        changes=[]
        self.canvas.brush_changed.connect(changes.append)
        QApplication.sendEvent(self.canvas.viewport(),self.event(QEvent.Type.MouseButtonPress,100,100,Qt.MouseButton.XButton1,Qt.MouseButton.XButton1))
        QApplication.sendEvent(self.canvas.viewport(),self.event(QEvent.Type.MouseMove,132,100,Qt.MouseButton.NoButton,Qt.MouseButton.XButton1))
        self.assertEqual(changes[-1],16)
        QApplication.sendEvent(self.canvas.viewport(),self.event(QEvent.Type.MouseButtonRelease,132,100,Qt.MouseButton.XButton1,Qt.MouseButton.NoButton))
        QApplication.sendEvent(self.canvas.viewport(),self.event(QEvent.Type.MouseButtonPress,200,100,Qt.MouseButton.XButton2,Qt.MouseButton.XButton2))
        QApplication.sendEvent(self.canvas.viewport(),self.event(QEvent.Type.MouseMove,168,100,Qt.MouseButton.NoButton,Qt.MouseButton.XButton2))
        self.assertEqual(changes[-1],12)
        QApplication.sendEvent(self.canvas.viewport(),self.event(QEvent.Type.MouseButtonRelease,168,100,Qt.MouseButton.XButton2,Qt.MouseButton.NoButton))
        self.assertEqual(self.doc.strokes,[])
        self.assertEqual(self.canvas._brush_thickness,12)


if __name__ == "__main__":
    unittest.main()
