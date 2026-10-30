"""Boundary checks for untrusted model geometry."""
import json
import unittest

from barforge.grok import GrokError, parse_bars


class GrokBoundaryTests(unittest.TestCase):
    def test_model_boxes_reject_nonfinite_and_unusable_geometry(self):
        with self.assertRaises(GrokError):
            parse_bars('{"bars":[{"x":NaN,"y":0.2,"width":0.3,"height":0.04}]}')
        invalid = [
            {"x": 0.8, "y": 0.2, "width": 0.6, "height": 0.05},
            {"x": 0.2, "y": 0.2, "width": -0.1, "height": 0.05},
            {"x": True, "y": 0.2, "width": 0.3, "height": 0.05},
        ]
        for box in invalid:
            with self.subTest(box=box), self.assertRaises(GrokError):
                parse_bars(json.dumps({"bars": [box]}))

    def test_valid_model_box_survives_invalid_neighbor_without_coordinate_drift(self):
        good = {"x": 0.25, "y": 0.4, "width": 0.5, "height": 0.03, "label": "region"}
        bars, dropped = parse_bars(json.dumps({"bars": [good, {**good, "x": -5}]}))
        self.assertEqual([bar.as_box() for bar in bars], [{key: good[key] for key in ("x", "y", "width", "height")}])
        self.assertEqual(dropped, 1)


if __name__ == "__main__":
    unittest.main()
