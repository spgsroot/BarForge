"""Brush persistence and pair-aligned x4 export with a deterministic test upscaler."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from barforge.imaging import render_pair
from barforge.models import Bar, BrushStroke, Document, Rect, Session
from barforge.storage import export_pair, load_session, save_session
from barforge.upscale import UpscaleCancelled


class NearestX4:
    def __init__(self):
        self.sizes = []
    def upscale(self, image, *, progress=None, cancelled=None):
        self.sizes.append(image.size)
        if progress is not None:
            progress(0,1)
            progress(1,1)
        return image.resize((image.width*4,image.height*4),Image.Resampling.NEAREST)


class CancelEngine:
    def upscale(self, image, **kwargs):
        raise UpscaleCancelled()


class UpscaleDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root/"panel.png"
        yy,xx = np.mgrid[:60,:80]
        self.pixels = np.stack((xx,yy,xx+yy),axis=-1).astype(np.uint8)
        Image.fromarray(self.pixels).save(self.source)

    def tearDown(self):
        self.temp.cleanup()

    def document(self):
        return Document(str(self.source),bars=[Bar(20,20,20,3,"#ffffff")],
            strokes=[BrushStroke(((15,15),(35,25),(50,20)),5,"#000000")],
            crop=Rect(10,10,40,30),caption="same crop",upscale=True,caption_touched=True)

    def test_both_sides_use_exact_x4_geometry_and_persist_brush_and_flag(self):
        document = self.document()
        engine = NearestX4()
        clean,control = render_pair(document,upscaler=engine)
        self.assertEqual(engine.sizes,[(40,30),(40,30)])
        self.assertEqual(clean.size,control.size)
        self.assertEqual(clean.size,(160,120))
        expected = self.pixels[10:40,10:50].repeat(4,axis=0).repeat(4,axis=1)
        np.testing.assert_array_equal(np.asarray(clean),expected)
        self.assertEqual(control.getpixel((100,40)),(255,255,255))
        self.assertNotEqual(clean.tobytes(),control.tobytes())
        path = self.root/"work.barforge.json"
        save_session(Session(documents=[document]),path)
        restored = load_session(path).documents[0]
        self.assertEqual(restored.strokes,document.strokes)
        self.assertTrue(restored.upscale)
        self.assertTrue(restored.caption_touched)
        self.assertEqual(restored.signature(),document.signature())
        restored.upscale = False
        self.assertNotEqual(restored.signature(),document.signature())

    def test_export_uses_same_upscaler_for_both_pngs_and_caption(self):
        document = self.document()
        output = self.root/"dataset"
        engine = NearestX4()
        name = export_pair(document,output,upscaler=engine)
        with Image.open(output/"clean"/f"{name}.png") as clean, Image.open(output/"censored"/f"{name}.png") as control:
            self.assertEqual(clean.size,(160,120))
            self.assertEqual(control.size,clean.size)
            self.assertNotEqual(clean.tobytes(),control.tobytes())
        self.assertEqual((output/"clean"/f"{name}.txt").read_text(encoding="utf-8"),"same crop")
        self.assertEqual(engine.sizes,[(40,30),(40,30)])
        self.assertTrue(document.is_exported)

    def test_cancellation_publishes_no_partial_pair(self):
        document = self.document()
        output = self.root/"dataset"
        with self.assertRaises(UpscaleCancelled):
            export_pair(document,output,upscaler=CancelEngine())
        self.assertFalse(output.exists())
        self.assertIsNone(document.export_name)
        self.assertIsNone(document.exported_signature)

    def test_version_one_document_keeps_legacy_signature_and_opens(self):
        document=Document(str(self.source),bars=[Bar(5,6,20,4,"#000000")],caption="legacy")
        payload=(Path(document.source).name,document.caption,None,
                 [(5,6,20,4,"#000000",0.)])
        legacy=hashlib.sha256(json.dumps(payload,separators=(",",":"),allow_nan=False).encode("ascii")).hexdigest()
        self.assertEqual(document.signature(),legacy)
        path=self.root/"old.barforge.json"
        path.write_text(json.dumps({"format":"barforge-session","version":1,"documents":[{
            "source":document.source,"bars":[{"x":5,"y":6,"width":20,"height":4,"color":"#000000"}],
            "caption":"legacy","export_name":"001","exported_signature":legacy}]}),encoding="utf-8")
        restored=load_session(path).documents[0]
        self.assertTrue(restored.is_exported)
        self.assertFalse(restored.upscale)
        self.assertEqual(restored.strokes,[])


if __name__ == "__main__":
    unittest.main()
