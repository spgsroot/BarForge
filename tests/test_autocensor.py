"""Thin line gaps, configurable width deviation, rotated rasterization and editable-session geometry."""
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from barforge.autocensor import CensorOptions, Detection, MAX_VARIATION, bars_from_detections, _unturn
from barforge.imaging import render_pair
from barforge.models import Bar, Document, Rect, Session
from barforge.storage import save_session, load_session


class ThinLineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.source = Path(self.temp.name) / "neutral.png"
        Image.new("RGB", (240, 180), (80, 80, 80)).save(self.source)

    def tearDown(self):
        self.temp.cleanup()

    def test_generated_lines_leave_source_visible_between_them(self):
        region = Detection("genital", "penis", .9, Rect(20, 20, 180, 120))
        options = CensorOptions(thickness=4, spacing=24, angle=0, padding=0)
        bars = bars_from_detections([region], 240, 180, options)
        clean, censored = render_pair(Document(str(self.source), bars=bars))
        self.assertEqual(censored.getpixel((100, 24)), (255, 255, 255))
        self.assertEqual(censored.getpixel((100, 36)), (80, 80, 80))
        self.assertEqual(censored.getpixel((220, 24)), (80, 80, 80))

    def test_tiny_region_gets_a_thinner_line_instead_of_full_cover(self):
        region = Detection("genital", "penis", .9, Rect(11, 11, 8, 8))
        for low, high in ((3, 8), (0, 0), (40, 64)):
            with self.subTest(deviation=(low, high)):
                options = CensorOptions(thickness=6, spacing=48, angle=0, padding=0,
                                        variation_min=low, variation_max=high)
                bars = bars_from_detections([region], 240, 180, options)
                self.assertEqual(len(bars), 1)
                self.assertLessEqual(bars[0].height, 2)
                self.assertGreater(bars[0].width, bars[0].height)

    def test_overlapping_zones_do_not_interleave_stripe_grids(self):
        regions = [Detection("genital", "penis", .9, Rect(20, 20, 100, 120)),
                   Detection("genital", "vagina", .8, Rect(60, 35, 120, 110))]
        options = CensorOptions(thickness=4, spacing=24, angle=0, padding=0)
        bars = bars_from_detections(regions, 240, 180, options)
        _clean, combined = render_pair(Document(str(self.source), bars=bars))
        first = bars_from_detections(regions[:1], 240, 180, options)
        _clean, single = render_pair(Document(str(self.source), bars=first))
        np.testing.assert_array_equal(np.asarray(combined)[40:130, 70:110],
                                      np.asarray(single)[40:130, 70:110])

    def test_chosen_deviation_range_is_inclusive_random_sided_and_reaches_the_export(self):
        Image.new("RGB", (1024, 1024), (80, 80, 80)).save(self.source)
        region = Detection("genital", "penis", .9, Rect(20, 20, 960, 960))
        options = CensorOptions(thickness=20, spacing=32, angle=0, padding=0, variation_min=2, variation_max=5)
        deltas = [bar.height - options.thickness
                  for seed in range(8)
                  for bar in bars_from_detections([region], 1024, 1024, replace(options, seed=seed))]
        self.assertEqual({abs(delta) for delta in deltas}, {2, 3, 4, 5})
        self.assertEqual({delta < 0 for delta in deltas}, {False, True})
        bars = bars_from_detections([region], 1024, 1024, replace(options, seed=3))
        _clean, output = render_pair(Document(str(self.source), bars=bars))
        column = np.asarray(output)[:, 512, 0] == 255
        edges = np.diff(np.concatenate(([False], column, [False])).astype(np.int8))
        heights = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        self.assertEqual(sorted(heights.tolist()), sorted(bar.height for bar in bars))

    def test_equal_bounds_fix_the_amplitude_and_zero_range_disables_the_deviation(self):
        region = Detection("genital", "penis", .9, Rect(20, 20, 960, 960))
        options = CensorOptions(thickness=20, spacing=32, angle=0, padding=0)

        def heights(low, high):
            return [bar.height
                    for seed in range(8)
                    for bar in bars_from_detections(
                        [region], 1024, 1024, replace(options, seed=seed, variation_min=low, variation_max=high))]

        fixed = heights(4, 4)
        self.assertEqual({abs(height - options.thickness) for height in fixed}, {4})
        self.assertEqual({height < options.thickness for height in fixed}, {False, True})
        self.assertEqual(set(heights(0, 0)), {options.thickness})

    def test_editing_the_bounds_rescales_the_pattern_without_rerolling_it(self):
        region = Detection("genital", "penis", .9, Rect(20, 20, 960, 960))
        options = CensorOptions(thickness=20, spacing=32, angle=0, padding=0, seed=5,
                                variation_min=3, variation_max=8)
        narrow = bars_from_detections([region], 1024, 1024, options)
        wide = bars_from_detections([region], 1024, 1024, replace(options, variation_min=1, variation_max=11))
        self.assertEqual(len(narrow), len(wide))
        self.assertEqual([bar.height > options.thickness for bar in narrow],
                         [bar.height > options.thickness for bar in wide])

    def test_deviation_bounds_must_be_ordered_whole_pixels_within_the_maximum(self):
        for low, high in ((9, 8), (-1, 8), (3, -1), (0, MAX_VARIATION + 1), (2.5, 8), (3, "8"), (None, 8)):
            with self.subTest(rejected=(low, high)):
                options = CensorOptions(variation_min=low, variation_max=high)
                with self.assertRaises(ValueError):
                    bars_from_detections([], 240, 180, options)

    def test_nominal_and_colour_changes_do_not_reroll_width_variation(self):
        Image.new("RGB", (1024, 1024), (80, 80, 80)).save(self.source)
        region = Detection("genital", "penis", .9, Rect(20, 20, 960, 960))
        options = CensorOptions(thickness=12, spacing=48, angle=0, padding=0, seed=42)
        original = bars_from_detections([region], 1024, 1024, options)
        thicker = bars_from_detections([region], 1024, 1024, replace(options, thickness=20))
        self.assertEqual([bar.height - 12 for bar in original], [bar.height - 20 for bar in thicker])
        black = bars_from_detections([region], 1024, 1024, replace(options, color="#000000"))
        clean, white_output = render_pair(Document(str(self.source), bars=original))
        _clean, black_output = render_pair(Document(str(self.source), bars=black))
        np.testing.assert_array_equal(np.asarray(white_output) != np.asarray(clean),
                                      np.asarray(black_output) != np.asarray(clean))

    def test_variation_respects_minimum_width_and_small_spacing_in_the_export(self):
        region = Detection("genital", "penis", .9, Rect(20, 20, 180, 120))
        for low, high in ((3, 8), (20, 64)):
            with self.subTest(deviation=(low, high)):
                options = CensorOptions(thickness=6, spacing=7, angle=0, padding=0, seed=0,
                                        variation_min=low, variation_max=high)
                bars = bars_from_detections([region], 240, 180, options)
                self.assertTrue(all(1 <= bar.height < options.spacing for bar in bars))
                _clean, output = render_pair(Document(str(self.source), bars=bars))
                gap_rows = 3 + np.arange(4, 19) * options.spacing
                np.testing.assert_array_equal(np.asarray(output)[gap_rows, 100],
                                              np.full((len(gap_rows), 3), 80, dtype=np.uint8))

    def test_rotated_bar_is_a_thin_diagonal_not_a_filled_bounding_box(self):
        bar = Bar(50, 65, 100, 4, "#ffffff", 30.)
        clean, output = render_pair(Document(str(self.source), bars=[bar]))
        yy, xx = np.nonzero(np.any(np.asarray(clean) != np.asarray(output), axis=2))
        covariance = np.cov(np.column_stack((xx, yy)), rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, values.argmax()]
        angle = np.degrees(np.arctan2(direction[1], direction[0])) % 180
        self.assertAlmostEqual(angle, 30., delta=1.5)
        self.assertLess(len(xx), 700)
        self.assertGreater(len(xx), 300)
        self.assertEqual(output.getpixel((58, 85)), (80, 80, 80))

    def test_rotated_crop_matches_the_same_full_image_region(self):
        bars = [Bar(40, 70, 140, 5, "#000000", 65.)]
        _, full = render_pair(Document(str(self.source), bars=bars))
        _, cropped = render_pair(Document(str(self.source), bars=bars, crop=Rect(65, 40, 80, 90)))
        np.testing.assert_array_equal(np.asarray(cropped), np.asarray(full.crop((65, 40, 145, 130))))

    def test_session_preserves_angle_and_detects_angle_only_edits(self):
        doc = Document(str(self.source), bars=[Bar(30, 60, 100, 5, "#ffffff", 65.)])
        signature = doc.signature()
        path = Path(self.temp.name) / "work.json"
        save_session(Session(documents=[doc]), path)
        restored = load_session(path).documents[0]
        self.assertEqual(restored.bars, doc.bars)
        self.assertEqual(restored.signature(), signature)
        restored.bars[0].angle = 10.
        self.assertNotEqual(restored.signature(), signature)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["documents"][0]["bars"][0]["angle"] = "not-a-number"
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_session(path)

    def test_scene_context_is_never_painted_even_when_enabled(self):
        activity = Detection("act", "make_love", .85, Rect(10, 10, 220, 150))
        anatomy = Detection("genital", "penis", .9, Rect(80, 60, 40, 80))
        for enabled in (True, False):
            options = CensorOptions(include_acts=enabled, angle=0, padding=0,
                                    thickness=4, spacing=24, variation_min=0, variation_max=0)
            bars = bars_from_detections([activity, anatomy], 240, 180, options)
            _, output = render_pair(Document(str(self.source), bars=bars))
            self.assertEqual(output.getpixel((100, 72)), (255, 255, 255))
            self.assertEqual(output.getpixel((40, 72)), (80, 80, 80))
            self.assertEqual(output.getpixel((100, 24)), (80, 80, 80))

    def test_each_anatomical_contour_has_its_own_transverse_direction(self):
        horizontal = Detection("genital", "penis", .9, Rect(20, 20, 180, 40),
                               ((20., 20.), (200., 20.), (200., 60.), (20., 60.)))
        vertical = Detection("genital", "penis", .9, Rect(40, 80, 40, 90),
                             ((40., 80.), (80., 80.), (80., 170.), (40., 170.)))
        options = CensorOptions(thickness=4, spacing=24, variation_min=0, variation_max=0)
        across_horizontal = bars_from_detections([horizontal], 240, 180, options)
        across_vertical = bars_from_detections([vertical], 240, 180, options)
        self.assertEqual({bar.angle for bar in across_horizontal}, {-90.})
        self.assertEqual({bar.angle for bar in across_vertical}, {0.})
        _, output = render_pair(Document(str(self.source), bars=across_horizontal+across_vertical))
        self.assertEqual(output.getpixel((96, 40)), (255, 255, 255))
        self.assertEqual(output.getpixel((60, 120)), (255, 255, 255))
        self.assertEqual(output.getpixel((96, 80)), (80, 80, 80))

    def test_rectangular_bars_stay_inside_concave_foreground_and_leave_background_holes(self):
        region = Detection("genital", "penis", .9, Rect(20, 20, 200, 140),
                           ((20., 20.), (220., 20.), (220., 160.), (140., 160.),
                            (140., 110.), (100., 110.), (100., 160.), (20., 160.)),
                           holes=(((70., 45.), (170., 45.), (170., 85.), (70., 85.)),))
        options = CensorOptions(thickness=4, spacing=24, angle=0, auto_angle=False,
                                variation_min=0, variation_max=0)
        bars = bars_from_detections([region], 240, 180, options)
        _, output = render_pair(Document(str(self.source), bars=bars))
        self.assertEqual(output.getpixel((110, 24)), (255, 255, 255))
        self.assertEqual(output.getpixel((40, 72)), (255, 255, 255))
        np.testing.assert_array_equal(np.asarray(output)[46:85, 71:170], np.full((39, 99, 3), 80))
        np.testing.assert_array_equal(np.asarray(output)[111:160, 101:140], np.full((49, 39, 3), 80))

    def test_rotation_inverse_restores_edges_not_pixel_centres(self):
        mask = np.zeros((640, 640), np.uint8)
        mask[201:246, 13:112] = 1
        for turn in range(4):
            with self.subTest(turn=turn):
                yy, xx = np.nonzero(np.rot90(mask, turn))
                found = (int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1)
                self.assertEqual(_unturn(*found, turn), (13, 201, 112, 246))

    def test_spacing_must_leave_gaps_and_empty_detection_never_invents_lines(self):
        self.assertEqual(bars_from_detections([], 240, 180, CensorOptions()), [])
        with self.assertRaises(ValueError):
            bars_from_detections([], 240, 180, CensorOptions(thickness=6, spacing=6))


if __name__ == "__main__":
    unittest.main()
