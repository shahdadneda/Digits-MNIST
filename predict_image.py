"""Predict a handwritten digit stored in an ordinary image file."""

import argparse
from pathlib import Path

import torch

from image_processing import image_file_to_tensor
from model import MnistCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to an image containing one digit.")
    parser.add_argument(
        "--preview",
        type=Path,
        help="Optional path where the processed 28x28 image should be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    model_path = project_dir / "models" / "mnist_cnn.pt"

    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if args.preview is not None and args.preview.exists():
        raise FileExistsError(
            f"Preview already exists: {args.preview}. Choose a new filename."
        )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = MnistCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_tensor, prepared_image = image_file_to_tensor(
        args.image,
        checkpoint["mnist_mean"],
        checkpoint["mnist_std"],
    )

    with torch.inference_mode():
        logits = model(image_tensor.unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_digit = int(probabilities.argmax().item())
    confidence = float(probabilities[predicted_digit].item())
    top_probabilities, top_digits = probabilities.topk(3)

    print(f"Image: {args.image}")
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2%}")
    print("Top three guesses:")
    for digit, probability in zip(top_digits.tolist(), top_probabilities.tolist()):
        print(f"  {digit}: {probability:.2%}")

    if args.preview is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        prepared_image.save(args.preview)
        print(f"Saved model-input preview to: {args.preview}")


if __name__ == "__main__":
    main()
