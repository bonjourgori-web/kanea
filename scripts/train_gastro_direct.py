"""
GastroAI — EfficientNet-B0 sur Nerthus bowel cleanliness (4 classes)
Usage: python scripts/train_gastro_direct.py --dataset data/gastro --epochs 12
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, f1_score

ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "deep_learning"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Score 0 (propre)", "Score 1 (residus mineurs)",
           "Score 2 (residus moderes)", "Score 3 (obstruction)"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/gastro")
    p.add_argument("--epochs",  type=int, default=12)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--lr",      type=float, default=1e-4)
    return p.parse_args()


def find_image_root(data_dir: Path) -> Path:
    for candidate in [
        data_dir / "nerthus-dataset-frames" / "nerthus-dataset-frames",
        data_dir / "nerthus-dataset-frames",
        data_dir,
    ]:
        subdirs = [d for d in candidate.iterdir() if d.is_dir()] if candidate.exists() else []
        if subdirs and any((candidate / s).glob("*.jpg") or list((candidate / s).glob("*.png"))
                           for s in [d.name for d in subdirs]):
            return candidate
    return data_dir


def train(args):
    t0 = time.time()
    data_dir = ROOT / args.dataset
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_root = find_image_root(data_dir)
    print(f"Images dans: {img_root}")

    tfm_train = T.Compose([
        T.Resize((224, 224)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3),
        T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tfm_val = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_ds = ImageFolder(str(img_root), transform=tfm_train)
    n_classes = len(full_ds.classes)
    print(f"Classes: {full_ds.classes} ({n_classes}) | Images: {len(full_ds)}")

    n_val   = max(1, int(len(full_ds) * 0.2))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    val_ds.dataset.transform = tfm_val

    train_loader = DataLoader(train_ds, args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   args.batch, shuffle=False, num_workers=0)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_state, best_acc = 0.0, None, 0.0

    for epoch in range(args.epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(imgs), labels).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                out = model(imgs.to(device))
                preds.extend(out.argmax(1).cpu().numpy())
                trues.extend(labels.numpy())

        f1  = f1_score(trues, preds, average="weighted", zero_division=0)
        acc = accuracy_score(trues, preds)
        print(f"Epoch {epoch+1}/{args.epochs} — Acc:{acc:.3f}  F1:{f1:.3f}")
        if f1 > best_f1:
            best_f1, best_acc = f1, acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    onnx_path = MODEL_DIR / "gastro_model.onnx"
    dummy = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(model, dummy, onnx_path, opset_version=18,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "b"}, "output": {0: "b"}})
    np.save(MODEL_DIR / "gastro_fc_weights.npy",
            model.classifier[1].weight.data.cpu().numpy())

    metrics = {
        "accuracy": round(float(best_acc), 4), "f1_weighted": round(float(best_f1), 4),
        "classes": full_ds.classes, "architecture": "EfficientNet-B0",
        "dataset": "Nerthus Bowel Cleanliness", "samples_train": n_train,
        "training_time_s": round(time.time() - t0, 1),
    }
    (MODEL_DIR / "gastro_model.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"ONNX -> {onnx_path}")
    return metrics


if __name__ == "__main__":
    train(parse_args())
