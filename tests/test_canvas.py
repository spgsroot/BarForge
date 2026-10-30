"""Two-click drawing must not commit previews or lose the chosen endpoints."""
import math
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from barforge.canvas import ImageCanvas
from barforge.imaging import render_pair
from barforge.models import Document, bar_polygon


class CanvasDrawingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.source = Path(self.folder.name) / "source.png"
        self.image = Image.new("RGB", (512, 512), (80, 80, 80))
        self.image.save(self.source)
        self.doc = Document(str(self.source))
        self.canvas = ImageCanvas()
        self.canvas.resize(700, 620)
        self.canvas.set_document(self.image, self.doc)
        self.canvas.set_tool("bar")
        self.canvas.set_brush(12, "#ffffff")
        self.canvas.show()
        self.app.processEvents()

    def tearDown(self):
        self.canvas.close()
        self.canvas.deleteLater()
        self.app.processEvents()
        self.folder.cleanup()

    def click(self, x, y):
        point = self.canvas.mapFromScene(QPointF(x, y))
        actual = self.canvas.mapToScene(point)
        QTest.mouseClick(self.canvas.viewport(), Qt.MouseButton.LeftButton, pos=point)
        return actual

    def move(self, x, y):
        point = self.canvas.mapFromScene(QPointF(x, y))
        local = QPointF(point)
        event = QMouseEvent(QMouseEvent.Type.MouseMove, local,
                            QPointF(self.canvas.viewport().mapToGlobal(point)),
                            Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(self.canvas.viewport(), event)
        self.app.processEvents()

    def assert_endpoints(self, bar, start, end):
        corners = bar_polygon(bar)
        actual = [((corners[a][0] + corners[b][0]) / 2, (corners[a][1] + corners[b][1]) / 2)
                  for a, b in ((0, 3), (1, 2))]
        for found, wanted in zip(actual, (start, end)):
            self.assertLessEqual(math.hypot(found[0] - wanted.x(), found[1] - wanted.y()), 1.1)

    def test_diagonal_commits_only_on_second_click_and_exports_as_one_undo_step(self):
        signature = self.doc.signature()
        start = self.click(100, 100)
        self.move(340, 280)
        self.assertEqual(self.doc.bars, [], "Moving after the first click must remain a preview")
        self.assertEqual(self.doc.signature(), signature)
        self.assertFalse(self.canvas.can_undo)
        clean, preview_export = render_pair(self.doc)
        self.assertEqual(preview_export.tobytes(), clean.tobytes(), "Pending preview leaked into export")

        end = self.click(340, 280)
        bar, = self.doc.bars
        self.assert_endpoints(bar, start, end)
        self.assertEqual(bar.height, 12)
        self.assertAlmostEqual(bar.angle, math.degrees(math.atan2(end.y() - start.y(), end.x() - start.x())))
        clean, output = render_pair(self.doc)
        self.assertEqual(clean.getpixel((220, 190)), (80, 80, 80))
        self.assertEqual(output.getpixel((220, 190)), (255, 255, 255))
        self.assertEqual(output.getpixel((300, 100)), (80, 80, 80), "The strip must not stay horizontal")
        self.canvas.undo()
        self.assertEqual(self.doc.bars, [])
        self.assertFalse(self.canvas.can_undo, "A completed strip must have exactly one history step")
        self.canvas.redo()
        restored, = self.doc.bars
        self.assert_endpoints(restored, start, end)

    def test_coincident_second_click_does_not_create_a_square_and_upward_line_is_vertical(self):
        self.canvas.set_brush(8, "#000000")
        start = self.click(240, 360)
        self.click(240, 360)
        self.assertEqual(self.doc.bars, [])
        end = self.click(240, 120)
        bar, = self.doc.bars
        self.assert_endpoints(bar, start, end)
        self.assertAlmostEqual(bar.angle, -90)
        self.assertEqual((bar.height, bar.color), (8, "#000000"))
        _clean, output = render_pair(self.doc)
        self.assertEqual(output.getpixel((240, 240)), (0, 0, 0))
        self.assertEqual(output.getpixel((250, 240)), (80, 80, 80))

    def test_cancel_tool_change_and_new_image_discard_the_pending_first_point(self):
        self.click(100, 100)
        self.move(340, 280)
        QTest.keyClick(self.canvas, Qt.Key.Key_Escape)
        self.assertEqual(self.doc.bars, [])
        self.assertFalse(self.canvas.can_undo)

        self.click(120, 160)
        self.canvas.set_tool("select")
        self.canvas.set_tool("bar")
        self.click(300, 300)
        self.assertEqual(self.doc.bars, [], "Switching tools must discard the old first point")

        next_doc = Document(str(self.source))
        self.canvas.set_document(self.image, next_doc)
        start = self.click(350, 140)
        self.assertEqual(next_doc.bars, [], "A new image must not reuse the old first point")
        end = self.click(150, 300)
        bar, = next_doc.bars
        self.assert_endpoints(bar, start, end)
        self.assertEqual(self.doc.bars, [])


if __name__ == "__main__":
    unittest.main()
