"""Load the trained MNIST model and predict one test image."""

import argparse
from pathlib import Path

import torch
from torchvision import datasets, transforms

from model import MnistCNN


def parse_args() -> argparse.Namespace:
    """Read options supplied after `python predict.py` in the terminal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the MNIST test image to predict (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Because this file lives in the project folder, its parent is the project.
    project_dir = Path(__file__).resolve().parent
    model_path = project_dir / "models" / "mnist_cnn.pt"
    data_dir = project_dir / "data"

    # The checkpoint contains the numbers learned during training.
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

    # First recreate the empty CNN structure, then put the learned numbers into it.
    model = MnistCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Use the same image normalization that was used during training.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                checkpoint["mnist_mean"],
                checkpoint["mnist_std"],
            ),
        ]
    )

    test_data = datasets.MNIST(
        root=data_dir,
        train=False,
        download=False,
        transform=transform,
    )

    if not 0 <= args.index < len(test_data):
        raise ValueError(
            f"--index must be between 0 and {len(test_data) - 1}, "
            f"but received {args.index}."
        )

    image, actual_digit = test_data[args.index]

    # unsqueeze(0) adds a batch dimension: [1 image, 1 channel, 28, 28].
    with torch.inference_mode():
        logits = model(image.unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_digit = int(probabilities.argmax().item())
    confidence = float(probabilities[predicted_digit].item())
    top_probabilities, top_digits = probabilities.topk(3)

    print(f"Test image index: {args.index}")
    print(f"Actual digit: {actual_digit}")
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2%}")
    print("Top three guesses:")
    for digit, probability in zip(top_digits.tolist(), top_probabilities.tolist()):
        print(f"  {digit}: {probability:.2%}")


if __name__ == "__main__":
    main()
