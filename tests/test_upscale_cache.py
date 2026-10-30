"""Exact-pixel memory/disk caching for preview and export, without real neural inference."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from barforge.imaging import render_pair
from barforge.models import Bar, Document, Rect
from barforge.storage import export_pair
from barforge.upscale import RealESRGAN, UpscaleCancelled


class FakeSession:
    def __init__(self):
        self.calls = 0
    def run(self, _outputs, inputs):
        self.calls += 1
        tile = inputs["input"][0].transpose(1,2,0)
        output = np.repeat(np.repeat(tile,4,axis=0),4,axis=1)
        return [output.transpose(2,0,1)[None]]


class UpscaleCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root/"model.onnx"
        self.model.write_bytes(b"deterministic fake model identity")
        self.cache = self.root/"cache"
        self.source = self.root/"panel.png"
        Image.new("RGB",(24,20),(80,80,80)).save(self.source)

    def tearDown(self):
        self.temp.cleanup()

    def engine(self):
        engine = RealESRGAN(self.model,cache_dir=self.cache)
        engine._session = FakeSession()
        return engine

    def document(self):
        return Document(str(self.source),bars=[Bar(4,4,12,3,"#ffffff")],
                        crop=Rect(2,2,16,12),caption="first",upscale=True)

    def test_preview_export_and_caption_change_reuse_exact_pixel_results(self):
        engine = self.engine()
        document = self.document()
        clean,control = render_pair(document,upscaler=engine)
        self.assertEqual(clean.size,control.size)
        self.assertEqual(engine.inference_calls,2)
        render_pair(document,upscaler=engine)
        self.assertEqual(engine.inference_calls,2)
        self.assertEqual(engine.memory_hits,2)
        export_pair(document,self.root/"dataset",upscaler=engine)
        self.assertEqual(engine.inference_calls,2)
        document.caption = "only the caption changed"
        export_pair(document,self.root/"new-caption",upscaler=engine)
        self.assertEqual(engine.inference_calls,2)
        self.assertEqual((self.root/"new-caption"/"clean"/"001.txt").read_text(encoding="utf-8"),"only the caption changed")
        self.assertEqual(len(list(self.cache.glob("*.png"))),2)

    def test_new_engine_uses_disk_and_changed_markup_recomputes_only_control(self):
        first = self.engine()
        document = self.document()
        render_pair(document,upscaler=first)
        second = self.engine()
        render_pair(document,upscaler=second)
        self.assertEqual(second.inference_calls,0)
        self.assertEqual(second.disk_hits,2)
        document.bars[0].color = "#000000"
        render_pair(document,upscaler=second)
        self.assertEqual(second.inference_calls,1,"Only changed censored pixels need inference")
        self.assertEqual(second.memory_hits,1,"Clean was promoted to memory after its first disk hit")
        self.assertEqual(len(list(self.cache.glob("*.png"))),3)

    def test_corrupt_entry_is_recomputed_and_cancellation_writes_nothing(self):
        engine = self.engine()
        image = Image.new("RGB",(5,4),(1,2,3))
        engine.upscale(image)
        path = next(self.cache.glob("*.png"))
        path.write_bytes(b"not a png")
        recovered = self.engine()
        recovered.upscale(image)
        self.assertEqual(recovered.inference_calls,1)
        self.assertEqual(recovered.disk_hits,0)
        with Image.open(path) as repaired:
            self.assertEqual(repaired.size,(20,16))
        with self.assertRaises(UpscaleCancelled):
            recovered.upscale(Image.new("RGB",(5,4),(9,9,9)),cancelled=lambda:True)
        self.assertEqual(len(list(self.cache.glob("*.png"))),1)


if __name__ == "__main__":
    unittest.main()
