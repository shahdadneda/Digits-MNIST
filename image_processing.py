"""Convert an ordinary digit image into the format expected by MNIST."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms


CANVAS_SIZE = 28
DIGIT_SIZE = 20


def _flatten_transparency(image: Image.Image) -> Image.Image:
    """Place a transparent drawing on a contrasting solid background."""
    image = ImageOps.exif_transpose(image)
    if "A" not in image.getbands():
        return image.convert("L")

    rgba = image.convert("RGBA")
    rgba_pixels = np.asarray(rgba, dtype=np.uint8)
    alpha = rgba_pixels[:, :, 3]

    if int(alpha.min()) == 255:
        return rgba.convert("L")

    visible_pixels = alpha >= max(8, round(int(alpha.max()) * 0.1))
    if not visible_pixels.any():
        raise ValueError("The image appears blank. Draw a visible digit and try again.")

    red = rgba_pixels[:, :, 0].astype(np.float32)
    green = rgba_pixels[:, :, 1].astype(np.float32)
    blue = rgba_pixels[:, :, 2].astype(np.float32)
    luminance = (0.299 * red) + (0.587 * green) + (0.114 * blue)
    visible_luminance = float(np.median(luminance[visible_pixels]))

    # Dark strokes need a white background; light strokes need a black one.
    background_value = 255 if visible_luminance < 127 else 0
    background = Image.new(
        "RGBA",
        rgba.size,
        color=(background_value, background_value, background_value, 255),
    )
    return Image.alpha_composite(background, rgba).convert("L")


def _move_center_of_mass_to_middle(pixels: np.ndarray) -> np.ndarray:
    """Shift the digit without wrapping pixels around the canvas edges."""
    total_brightness = pixels.sum()
    if total_brightness == 0:
        return pixels

    y_positions, x_positions = np.indices(pixels.shape)
    center_x = float((pixels * x_positions).sum() / total_brightness)
    center_y = float((pixels * y_positions).sum() / total_brightness)
    target_center = (CANVAS_SIZE - 1) / 2
    shift_x = round(target_center - center_x)
    shift_y = round(target_center - center_y)

    shifted = np.zeros_like(pixels)

    source_x_start = max(0, -shift_x)
    source_x_end = min(CANVAS_SIZE, CANVAS_SIZE - shift_x)
    source_y_start = max(0, -shift_y)
    source_y_end = min(CANVAS_SIZE, CANVAS_SIZE - shift_y)

    destination_x_start = max(0, shift_x)
    destination_x_end = destination_x_start + (source_x_end - source_x_start)
    destination_y_start = max(0, shift_y)
    destination_y_end = destination_y_start + (source_y_end - source_y_start)

    shifted[
        destination_y_start:destination_y_end,
        destination_x_start:destination_x_end,
    ] = pixels[source_y_start:source_y_end, source_x_start:source_x_end]
    return shifted


def prepare_digit_image(image: Image.Image) -> Image.Image:
    """Return a centered 28x28 white-on-black version of a digit image."""
    grayscale = _flatten_transparency(image)
    pixels = np.asarray(grayscale, dtype=np.uint8)

    # Estimate the background from the outside border. A normal drawing is
    # usually black ink on white, while MNIST uses white ink on black.
    border = np.concatenate(
        [
            pixels[0, :],
            pixels[-1, :],
            pixels[:, 0],
            pixels[:, -1],
        ]
    )
    if float(np.median(border)) > 127:
        pixels = 255 - pixels

    # Remove faint background noise and locate the meaningful digit pixels.
    strongest_pixel = int(pixels.max())
    if strongest_pixel < 20:
        raise ValueError("The image appears blank. Draw a visible digit and try again.")

    threshold = max(20, round(strongest_pixel * 0.15))
    mask = pixels >= threshold
    y_coordinates, x_coordinates = np.where(mask)
    if len(x_coordinates) == 0:
        raise ValueError("No digit could be found in the image.")

    left = int(x_coordinates.min())
    right = int(x_coordinates.max()) + 1
    top = int(y_coordinates.min())
    bottom = int(y_coordinates.max()) + 1
    cropped = Image.fromarray(pixels[top:bottom, left:right], mode="L")

    # Preserve the digit's proportions while making its longest side 20 pixels.
    scale = DIGIT_SIZE / max(cropped.size)
    resized_width = max(1, round(cropped.width * scale))
    resized_height = max(1, round(cropped.height * scale))
    resized = cropped.resize(
        (resized_width, resized_height),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
    paste_x = (CANVAS_SIZE - resized_width) // 2
    paste_y = (CANVAS_SIZE - resized_height) // 2
    canvas.paste(resized, (paste_x, paste_y))

    centered_pixels = _move_center_of_mass_to_middle(
        np.asarray(canvas, dtype=np.uint8)
    )
    return Image.fromarray(centered_pixels, mode="L")


def image_to_tensor(
    image: Image.Image,
    mean: tuple[float, ...],
    standard_deviation: tuple[float, ...],
) -> tuple[torch.Tensor, Image.Image]:
    """Convert an open image into both its model tensor and visual preview."""
    prepared_image = prepare_digit_image(image)
    to_model_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, standard_deviation),
        ]
    )
    return to_model_tensor(prepared_image), prepared_image


def image_file_to_tensor(
    image_path: Path,
    mean: tuple[float, ...],
    standard_deviation: tuple[float, ...],
) -> tuple[torch.Tensor, Image.Image]:
    """Load an image file and return both its model tensor and visual preview."""
    with Image.open(image_path) as image:
        return image_to_tensor(image, mean, standard_deviation)
