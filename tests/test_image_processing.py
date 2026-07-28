"""Tests for rejecting non-digit canvas drawings before classification."""

import unittest

from PIL import Image, ImageDraw

from image_processing import InvalidDigitDrawingError, prepare_digit_image


class DigitDrawingValidationTests(unittest.TestCase):
    def make_canvas(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        image = Image.new("L", (560, 560), color=255)
        return image, ImageDraw.Draw(image)

    def test_centered_digit_is_accepted(self) -> None:
        image, draw = self.make_canvas()
        draw.line(
            [(335, 105), (240, 205), (225, 330), (330, 330)],
            fill=15,
            width=32,
            joint="curve",
        )
        draw.line([(315, 165), (315, 455)], fill=15, width=32)

        prepared = prepare_digit_image(image)

        self.assertEqual(prepared.size, (28, 28))
        self.assertGreater(prepared.getbbox()[2], 0)

    def test_edge_spanning_scribble_is_rejected(self) -> None:
        image, draw = self.make_canvas()
        draw.line(
            [(0, 280), (559, 280), (310, 10), (220, 535), (465, 85)],
            fill=15,
            width=32,
            joint="curve",
        )

        with self.assertRaisesRegex(
            InvalidDigitDrawingError,
            "nearly the whole canvas",
        ):
            prepare_digit_image(image)

    def test_dense_scribble_is_rejected(self) -> None:
        image, draw = self.make_canvas()
        for y_coordinate in range(70, 500, 45):
            draw.line(
                [(35, y_coordinate), (525, y_coordinate)],
                fill=15,
                width=42,
            )

        with self.assertRaisesRegex(
            InvalidDigitDrawingError,
            "fills too much",
        ):
            prepare_digit_image(image)

    def test_separate_marks_are_rejected(self) -> None:
        image, draw = self.make_canvas()
        for center_x, center_y in [(160, 180), (350, 170), (180, 385), (380, 390)]:
            draw.ellipse(
                (
                    center_x - 18,
                    center_y - 18,
                    center_x + 18,
                    center_y + 18,
                ),
                fill=15,
            )

        with self.assertRaisesRegex(
            InvalidDigitDrawingError,
            "several separate marks",
        ):
            prepare_digit_image(image)


if __name__ == "__main__":
    unittest.main()
