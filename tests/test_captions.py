"""Caption substitution must preserve user edits and prepared anatomical metadata."""
import unittest

from barforge.models import caption_template_for, render_caption_template


class CaptionTemplateTests(unittest.TestCase):
    def test_colour_substitution_does_not_interpret_other_user_braces(self):
        pattern = ('rustnomiconUncensor. Remove the {color} censor bars, penis, close-up; '
                   '{"camera": "front"}, {unrelated}')
        result = render_caption_template(pattern, "#000000")
        self.assertIn("black censor bars, penis, close-up", result)
        self.assertIn('{"camera": "front"}, {unrelated}', result)
        self.assertNotIn("{color}", result)

    def test_colour_roundtrip_keeps_the_prepared_subject_and_close_up(self):
        original = "rustnomiconUncensor. Remove the white censor bars, vagina, close-up"
        pattern = caption_template_for(original)
        self.assertIsNotNone(pattern)
        black = render_caption_template(pattern, "#000000")
        self.assertIn("black censor bars, vagina, close-up", black)
        white = render_caption_template(caption_template_for(black), "#FFFFFF")
        self.assertEqual(white, original)

    def test_recognition_does_not_claim_a_manually_extended_caption(self):
        edited = ("rustnomiconUncensor. Remove the white censor bars, penis, close-up; "
                  "keep the lettering unchanged")
        self.assertIsNone(caption_template_for(edited))


if __name__ == "__main__":
    unittest.main()
