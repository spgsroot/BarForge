"""Consumer-visible dataset invariants; no Qt or network required."""
from pathlib import Path
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from barforge.imaging import load_rgb, render_pair
from barforge.models import Bar, Document, Rect, Session
from barforge.storage import discover_images, export_pair, load_session, save_session


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "panel.png"
        yy, xx = np.mgrid[:80, :100]
        self.pixels = np.stack([xx, yy, xx + yy], axis=-1).astype(np.uint8)
        Image.fromarray(self.pixels).save(self.source)

    def tearDown(self):
        self.temporary.cleanup()

    def test_crop_preserves_clean_and_clips_bar_in_source_coordinates(self):
        doc = Document(str(self.source), bars=[Bar(5, 20, 30, 10, "#000000")],
                       crop=Rect(10, 15, 60, 40))
        clean, control = render_pair(doc)
        expected_clean = self.pixels[15:55, 10:70]
        expected_control = expected_clean.copy()
        expected_control[5:15, :25] = 0
        np.testing.assert_array_equal(np.asarray(clean), expected_clean)
        np.testing.assert_array_equal(np.asarray(control), expected_control)
        np.testing.assert_array_equal(np.asarray(Image.open(self.source)), self.pixels)

    def test_export_triplet_collision_protection_and_revision_tracking(self):
        output = self.root / "dataset"
        (output / "censored").mkdir(parents=True)
        (output / "censored" / "001.png").write_bytes(b"unrelated incomplete export")
        doc = Document(str(self.source), bars=[Bar(4, 5, 25, 6, "#ffffff")], caption="Убрать белые полосы — café")
        stem = export_pair(doc, output)
        self.assertNotEqual(stem, "001")
        expected_clean, expected_control = render_pair(doc)
        np.testing.assert_array_equal(np.asarray(Image.open(output / "clean" / f"{stem}.png")), np.asarray(expected_clean))
        np.testing.assert_array_equal(np.asarray(Image.open(output / "censored" / f"{stem}.png")), np.asarray(expected_control))
        self.assertEqual((output / "clean" / f"{stem}.txt").read_text(encoding="utf-8"), doc.caption)
        self.assertTrue(doc.is_exported)
        prior = (output / "clean" / f"{stem}.png").read_bytes()
        doc.caption = "Другая инструкция"
        self.assertFalse(doc.is_exported)
        with self.assertRaises(FileExistsError):
            export_pair(doc, output)
        self.assertEqual((output / "clean" / f"{stem}.png").read_bytes(), prior)
        self.assertEqual((output / "censored" / "001.png").read_bytes(), b"unrelated incomplete export")
        export_pair(doc, output, overwrite=True)
        self.assertTrue(doc.is_exported)
        self.assertEqual((output / "clean" / f"{stem}.txt").read_text(encoding="utf-8"), "Другая инструкция")

    def test_session_remains_portable_after_bundle_move(self):
        bundle = self.root / "bundle"
        (bundle / "panels").mkdir(parents=True)
        target = bundle / "panels" / "page.png"
        shutil.copyfile(self.source, target)
        doc = Document(str(target), bars=[Bar(8, 9, 20, 4, "#000000")],
                       crop=Rect(4, 5, 65, 50), caption="Переносимая сессия")
        session = Session(documents=[doc], source_dir=str(target.parent), output_dir=str(bundle / "dataset"))
        path = bundle / "work.barforge.json"
        save_session(session, path)
        moved = self.root / "moved"
        shutil.move(str(bundle), moved)
        restored = load_session(moved / path.name)
        self.assertEqual(Path(restored.documents[0].source), moved / "panels" / "page.png")
        self.assertEqual(restored.documents[0].bars, doc.bars)
        self.assertEqual(restored.documents[0].crop, doc.crop)
        self.assertEqual(restored.documents[0].caption, doc.caption)
        clean, _ = render_pair(restored.documents[0])
        np.testing.assert_array_equal(np.asarray(clean), self.pixels[5:55, 4:69])

    def test_transparent_input_is_composited_on_white(self):
        path = self.root / "transparent.png"
        image = Image.new("RGBA", (2, 1))
        image.putdata([(255, 0, 0, 0), (0, 0, 0, 128)])
        image.save(path)
        loaded = load_rgb(path)
        self.assertEqual(loaded.mode, "RGB")
        self.assertEqual([loaded.getpixel((x, 0)) for x in range(2)], [(255, 255, 255), (127, 127, 127)])

    def test_webp_sources_are_discovered_and_exported(self):
        sources = self.root / "webp-sources"
        sources.mkdir()
        Image.fromarray(self.pixels).save(sources / "2.WEBP", lossless=True)
        Image.new("RGB", (4, 3), (1, 2, 3)).save(sources / "10.webp", lossless=True)

        documents = discover_images(sources)
        self.assertEqual([Path(document.source).name for document in documents], ["2.WEBP", "10.webp"])

        stem = export_pair(documents[0], self.root / "dataset")
        with Image.open(self.root / "dataset" / "clean" / f"{stem}.png") as clean:
            np.testing.assert_array_equal(np.asarray(clean), self.pixels)

    def test_failed_overwrite_restores_all_three_previous_files(self):
        output = self.root / "dataset"
        doc = Document(str(self.source), bars=[Bar(5, 6, 30, 8)], caption="original")
        stem = export_pair(doc, output)
        paths = [output / "clean" / f"{stem}.png", output / "clean" / f"{stem}.txt",
                 output / "censored" / f"{stem}.png"]
        original = {path: path.read_bytes() for path in paths}
        prior_signature = doc.exported_signature
        doc.caption = "changed"
        replace = os.replace
        injected = False

        def fail_once(source, target):
            nonlocal injected
            if not injected and str(source).endswith(".tmp") and Path(target).suffix == ".txt":
                injected = True
                raise OSError("simulated write failure")
            return replace(source, target)

        with patch("barforge.storage.os.replace", side_effect=fail_once):
            with self.assertRaises(OSError):
                export_pair(doc, output, overwrite=True)
        self.assertTrue(injected)
        self.assertEqual({path: path.read_bytes() for path in paths}, original)
        self.assertEqual(doc.exported_signature, prior_signature)
        self.assertFalse(doc.is_exported)

    def test_session_rejects_export_name_path_traversal(self):
        path = self.root / "session.json"
        save_session(Session(documents=[Document(str(self.source))]), path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["documents"][0]["export_name"] = "../outside"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_session(path)


if __name__ == "__main__":
    unittest.main()
