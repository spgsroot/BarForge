"""Real foreground refinement and painting boundaries; no neural weights or network traffic."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from barforge.anatomy import refine_anatomy, _merge_foregrounds
from barforge import autocensor as ac
from barforge.imaging import render_pair
from barforge.models import Document, Rect


class AnatomicalPlacementTests(unittest.TestCase):
    def rows(self, *items):
        return np.asarray([(box, score, ac._LABELS.index(label)) for box, score, label in items],
                          dtype=ac._CANDIDATE)

    def test_broad_secondary_genital_boxes_never_replace_local_anatomical_predictions(self):
        pixels = np.full((180, 240, 3), 230, np.uint8)
        pixels[50:150, 95:145] = (180, 100, 100)
        broad = self.rows(((10, 10, 230, 170), .99, "penis"))
        primary = self.rows(((90, 45, 150, 155), .9, "penis"))
        options = ac.CensorOptions(thickness=4, spacing=24, variation_min=0, variation_max=0)
        regions, _ = refine_anatomy(pixels, {"anime-censor-v1-s":primary, "erax-anti-nsfw-v11":broad},
                                   options, 0)
        self.assertEqual(len(regions), 1)
        bars = ac.bars_from_detections(regions, 240, 180, options)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/"neutral.png"
            Image.fromarray(pixels).save(source)
            clean, output = render_pair(Document(str(source), bars=bars))
            self.assertEqual(output.getpixel((120, 96)), (255, 255, 255))
            np.testing.assert_array_equal(np.asarray(output)[:, :90], np.asarray(clean)[:, :90])
            np.testing.assert_array_equal(np.asarray(output)[:45], np.asarray(clean)[:45])
        missing, _ = refine_anatomy(pixels, {"anime-censor-v1-s":self.rows(), "erax-anti-nsfw-v11":broad},
                                   options, 0)
        self.assertEqual(missing, [])

    def test_unseparable_foreground_is_omitted_not_replaced_with_the_box(self):
        pixels = np.full((180, 240, 3), 230, np.uint8)
        regions, notes = refine_anatomy(
            pixels, {"anime-censor-v1-s":self.rows(((90, 45, 150, 155), .9, "penis")),
                     "erax-anti-nsfw-v11":self.rows()}, ac.CensorOptions(), 0)
        self.assertEqual(regions, [])
        self.assertEqual(ac.bars_from_detections(regions, 240, 180, ac.CensorOptions()), [])

    def test_repeated_foreground_union_preserves_holes_and_distinct_neighbor_objects(self):
        first = ac.Detection("genital", "penis", .9, Rect(20, 20, 180, 130),
                             ((20., 20.), (200., 20.), (200., 150.), (20., 150.)),
                             holes=(((90., 50.), (130., 50.), (130., 110.), (90., 110.)),))
        nested = ac.Detection("genital", "penis", .8, Rect(20, 20, 60, 130),
                              ((20., 20.), (80., 20.), (80., 150.), (20., 150.)))
        merged = _merge_foregrounds(first, nested)
        self.assertIsNotNone(merged)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/"neutral.png"
            Image.new("RGB", (240, 180), (80, 80, 80)).save(source)
            options = ac.CensorOptions(angle=0, auto_angle=False, thickness=4, spacing=24,
                                       variation_min=0, variation_max=0)
            _, output = render_pair(Document(str(source), bars=ac.bars_from_detections([merged], 240, 180, options)))
            self.assertEqual(output.getpixel((40, 72)), (255, 255, 255))
            self.assertEqual(output.getpixel((110, 72)), (80, 80, 80))
        neighbor = ac.Detection("genital", "penis", .8, Rect(190, 20, 40, 130),
                                ((190., 20.), (230., 20.), (230., 150.), (190., 150.)))
        self.assertIsNone(_merge_foregrounds(first, neighbor))


if __name__ == "__main__":
    unittest.main()
