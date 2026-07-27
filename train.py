"""Download MNIST, train the CNN, evaluate it, and save its weights."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import MnistCNN


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_data_loaders(
    data_dir: Path,
    batch_size: int,
    limit_train: int | None = None,
    limit_test: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )

    training_data = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    if limit_train is not None:
        training_data = Subset(training_data, range(min(limit_train, len(training_data))))
    if limit_test is not None:
        test_data = Subset(test_data, range(min(limit_test, len(test_data))))

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_function(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Use only this many training images (useful for a quick test).",
    )
    parser.add_argument(
        "--limit-test",
        type=int,
        default=None,
        help="Use only this many test images (useful for a quick test).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    model_dir = project_dir / "models"
    model_path = model_dir / "mnist_cnn.pt"

    torch.manual_seed(42)
    device = choose_device()
    print(f"Using device: {device}")

    train_loader, test_loader = make_data_loaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )
    print(
        f"Loaded {len(train_loader.dataset):,} training images and "
        f"{len(test_loader.dataset):,} test images."
    )

    model = MnistCNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, loss_function, optimizer, device
        )
        test_loss, test_accuracy = evaluate(
            model, test_loader, loss_function, device
        )
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train accuracy {train_accuracy:.2%} | "
            f"test loss {test_loss:.4f} | "
            f"test accuracy {test_accuracy:.2%}"
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "mnist_mean": MNIST_MEAN,
            "mnist_std": MNIST_STD,
        },
        model_path,
    )
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()
