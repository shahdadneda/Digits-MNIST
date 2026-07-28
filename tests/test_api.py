"""Tests for accepted and abstained prediction API results."""

import unittest
from io import BytesIO
from unittest.mock import patch

import torch
from PIL import Image, ImageDraw
from starlette.datastructures import Headers, UploadFile

from api import predict


class PredictionApiTests(unittest.IsolatedAsyncioTestCase):
    async def post_image(self, image: Image.Image):
        payload = BytesIO()
        image.save(payload, format="PNG")
        payload.seek(0)
        upload = UploadFile(
            file=payload,
            filename="drawing.png",
            headers=Headers({"content-type": "image/png"}),
        )
        return await predict(upload)

    async def test_clear_digit_returns_prediction(self) -> None:
        image = Image.new("L", (560, 560), color=255)
        draw = ImageDraw.Draw(image)
        draw.line([(280, 100), (280, 460)], fill=15, width=32)

        response = await self.post_image(image)

        self.assertTrue(response.accepted)
        self.assertEqual(response.digit, 1)
        self.assertIsNotNone(response.confidence)
        self.assertEqual(len(response.probabilities), 10)

    async def test_fragmented_drawing_abstains_without_digit_scores(self) -> None:
        image = Image.new("L", (560, 560), color=255)
        draw = ImageDraw.Draw(image)
        for center_x, center_y in [
            (160, 180),
            (350, 170),
            (180, 385),
            (380, 390),
        ]:
            draw.ellipse(
                (
                    center_x - 18,
                    center_y - 18,
                    center_x + 18,
                    center_y + 18,
                ),
                fill=15,
            )

        response = await self.post_image(image)

        self.assertFalse(response.accepted)
        self.assertIsNone(response.digit)
        self.assertIsNone(response.confidence)
        self.assertEqual(response.probabilities, {})

    async def test_low_confidence_drawing_still_returns_closest_digit(self) -> None:
        image = Image.new("L", (560, 560), color=255)
        draw = ImageDraw.Draw(image)
        draw.line([(280, 100), (280, 460)], fill=15, width=32)

        with patch("api.model", return_value=torch.zeros((1, 10))):
            response = await self.post_image(image)

        self.assertTrue(response.accepted)
        self.assertEqual(response.digit, 0)
        self.assertAlmostEqual(response.confidence, 0.10)
        self.assertEqual(len(response.probabilities), 10)


if __name__ == "__main__":
    unittest.main()
