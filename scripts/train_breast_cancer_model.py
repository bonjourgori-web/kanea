from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

CLASSES = ["Normal", "Benign", "Malignant"]


class MammographyFolderDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None) -> None:
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform
        for idx, label in enumerate(CLASSES):
            class_dir = root_dir / label
            if not class_dir.exists():
                continue
            for path in class_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    self.samples.append((path, idx))
        if not self.samples:
            raise ValueError(f"No mammography images found in {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_model() -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASSES))
    return model


def train_model(dataset_dir: Path, export_path: Path, epochs: int, batch_size: int, learning_rate: float, valid_pct: float) -> None:
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = MammographyFolderDataset(dataset_dir, transform=transform)
    valid_size = max(1, int(len(dataset) * valid_pct))
    train_size = len(dataset) - valid_size
    train_set, valid_set = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)

    device = torch.device("cpu")
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total if total else 0.0
        avg_loss = train_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1}/{epochs} - loss={avg_loss:.4f} - valid_acc={accuracy:.4f}")

    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), export_path)
    print(f"Model exported to: {export_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the KANEA breast cancer model.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/breast_cancer"))
    parser.add_argument("--export-path", type=Path, default=Path("models/deep_learning/breast_cancer_model.pth"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--valid-pct", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(args.dataset_dir, args.export_path, args.epochs, args.batch_size, args.learning_rate, args.valid_pct)


if __name__ == "__main__":
    main()
