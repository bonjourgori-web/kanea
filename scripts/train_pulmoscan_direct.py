"""
PulmoScan AI — Entraînement DenseNet121 sur chest X-ray pneumonia
=================================================================
Dataset : data/pulmoscan/chest_xray/ (Kaggle Chest X-Ray Pneumonia)
Classes : Normal / Pneumonia
Export  : ONNX Runtime CPU

Usage : python scripts/train_pulmoscan_direct.py --dataset data/pulmoscan --epochs 15
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "deep_learning"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Normal", "Pneumonie bactérienne", "Pneumonie virale",
    "COVID-19", "Tuberculose", "Autre pathologie pulmonaire",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/pulmoscan")
    p.add_argument("--epochs",  type=int, default=15)
    p.add_argument("--batch",   type=int, default=32)
    p.add_argument("--lr",      type=float, default=1e-4)
    return p.parse_args()


class ChestXrayDataset(Dataset):
    LABEL_MAP = {"NORMAL": 0, "PNEUMONIA": 1, "normal": 0, "pneumonia": 1,
                 "COVID19": 3, "TURBERCULOSIS": 4, "TUBERCULOSIS": 4}

    def __init__(self, root: Path, split: str, transform=None):
        self.samples = []
        self.transform = transform
        # Chercher les sous-dossiers train/test/val
        for search in [root / split, root / "chest_xray" / split,
                       root / split.upper(), root]:
            if not search.exists():
                continue
            for class_dir in sorted(search.iterdir()):
                if not class_dir.is_dir():
                    continue
                label = self.LABEL_MAP.get(class_dir.name.upper(), 1)
                for ext in ["*.jpeg","*.jpg","*.png"]:
                    for img in class_dir.glob(ext):
                        self.samples.append((img, label))
            if self.samples:
                break
        counts = {}
        for _, l in self.samples:
            counts[l] = counts.get(l, 0) + 1
        print(f"  {split}: {len(self.samples)} images — {counts}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(train: bool):
    if train:
        return T.Compose([
            T.Resize((256, 256)), T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def train(args):
    t0 = time.time()
    data_dir = ROOT / args.dataset
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    train_ds = ChestXrayDataset(data_dir, "train", get_transforms(True))
    val_ds   = ChestXrayDataset(data_dir, "val",   get_transforms(False))

    if not train_ds.samples:
        print("ERREUR : aucune image trouvée dans data/pulmoscan/")
        return

    n_classes = len(set(l for _, l in train_ds.samples))
    n_classes = max(n_classes, 2)
    print(f"Nombre de classes détectées : {n_classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    # DenseNet121
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier.in_features, n_classes),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_state, best_acc, best_auc = 0.0, None, 0.0, 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if not val_ds.samples:
            print(f"Epoch {epoch+1}/{args.epochs} — loss:{total_loss/len(train_loader):.4f}")
            continue

        model.eval()
        all_preds, all_labels, all_proba = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                out  = model(imgs.to(device))
                prob = torch.softmax(out, dim=1).cpu().numpy()
                pred = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(pred); all_labels.extend(labels.numpy())
                all_proba.append(prob)

        f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        try:
            proba_mat = np.vstack(all_proba)
            auc = roc_auc_score(all_labels, proba_mat, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.0
        print(f"Epoch {epoch+1}/{args.epochs} — loss:{total_loss/len(train_loader):.4f}  Acc:{acc:.3f}  F1:{f1:.3f}  AUC:{auc:.3f}")
        if f1 > best_f1:
            best_f1, best_acc, best_auc = f1, acc, auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    onnx_path = MODEL_DIR / "pulmoscan_model.onnx"
    dummy = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(model, dummy, onnx_path,
                      input_names=["input"], output_names=["output"],
                      opset_version=17, dynamic_axes={"input":{0:"b"},"output":{0:"b"}})

    fc = model.classifier[1]
    np.save(MODEL_DIR / "pulmoscan_fc_weights.npy",
            fc.weight.data.cpu().numpy())

    metrics = {
        "accuracy": round(float(best_acc), 4), "f1_weighted": round(float(best_f1), 4),
        "auc_roc": round(float(best_auc), 4),
        "classes": CLASSES[:n_classes], "architecture": "DenseNet121",
        "dataset": "Chest X-Ray Pneumonia (Kaggle)",
        "samples_train": len(train_ds), "training_time_s": round(time.time() - t0, 1),
    }
    (MODEL_DIR / "pulmoscan_model.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nONNX → {onnx_path}")
    return metrics


if __name__ == "__main__":
    train(parse_args())
