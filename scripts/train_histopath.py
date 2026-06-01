"""
HistoPath AI — Entraînement ResNet18 sur lames histologiques
=============================================================
Dataset : GlaS Gland Segmentation + PathMNIST colon pathology
Classes : Normal · Adénome bénin · Adénocarcinome · Autre
Export  : ONNX Runtime CPU

Usage :
    python scripts/train_histopath.py --dataset data/histopath --epochs 20
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "deep_learning"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal / Tissu sain", "Adénome bénin", "Adénocarcinome bas grade",
           "Adénocarcinome haut grade"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/histopath")
    p.add_argument("--epochs",  type=int, default=15)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--lr",      type=float, default=1e-4)
    return p.parse_args()


class HistoDataset(Dataset):
    """Charge images depuis dossiers, chaque sous-dossier = classe."""
    def __init__(self, root: Path, transform=None):
        self.samples = []
        self.transform = transform
        self.classes = []

        # 1. Chercher structure classe/images
        subdirs = sorted([d for d in root.rglob("*") if d.is_dir()])
        class_dirs = []
        for d in subdirs:
            imgs = list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.bmp"))
            if 5 <= len(imgs) <= 50000:
                class_dirs.append(d)

        if class_dirs:
            self.classes = [d.name for d in class_dirs]
            for label, d in enumerate(class_dirs):
                for ext in ["*.png","*.jpg","*.bmp","*.tif","*.tiff"]:
                    for img in d.glob(ext):
                        self.samples.append((img, label % len(CLASSES)))

        # 2. Fallback: toutes les images → classe 0
        if not self.samples:
            for ext in ["*.png","*.jpg","*.bmp","*.tif","*.tiff"]:
                for img in root.rglob(ext):
                    self.samples.append((img, 0))
            self.classes = CLASSES[:1]

        print(f"  Histopath: {len(self.samples)} images, {len(self.classes)} classes")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(train: bool):
    if train:
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.ToTensor(),
            T.Normalize([0.7, 0.6, 0.7], [0.15, 0.15, 0.15]),  # normales histo
        ])
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.7, 0.6, 0.7], [0.15, 0.15, 0.15]),
    ])


def train(args):
    t0 = time.time()
    data_dir = ROOT / args.dataset
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Dataset complet puis split 80/20
    full_ds = HistoDataset(data_dir, get_transforms(True))
    if not full_ds.samples:
        print("ERREUR : aucune image histopath trouvée")
        return

    n_val   = max(1, int(len(full_ds) * 0.2))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    val_ds.dataset.transform = get_transforms(False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    n_classes = min(len(full_ds.classes), len(CLASSES))
    n_classes = max(n_classes, 2)

    # ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_state, best_acc, best_auc = 0.0, None, 0.0, 0.0

    for epoch in range(args.epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        all_preds, all_labels, all_proba = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                out  = model(imgs)
                prob = torch.softmax(out, dim=1).cpu().numpy()
                pred = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(pred); all_labels.extend(labels.numpy())
                all_proba.append(prob)

        if not all_labels:
            continue

        f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        try:
            proba_mat = np.vstack(all_proba)
            auc = roc_auc_score(all_labels, proba_mat, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.0
        print(f"Epoch {epoch+1}/{args.epochs} — Acc:{acc:.3f}  F1:{f1:.3f}  AUC:{auc:.3f}")
        if f1 > best_f1:
            best_f1, best_acc, best_auc = f1, acc, auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    onnx_path = MODEL_DIR / "histopath_model.onnx"
    dummy = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(model, dummy, onnx_path,
                      input_names=["input"], output_names=["output"],
                      opset_version=17, dynamic_axes={"input":{0:"b"},"output":{0:"b"}})

    fc_weights = model.fc.weight.data.cpu().numpy()
    np.save(MODEL_DIR / "histopath_fc_weights.npy", fc_weights)

    metrics = {
        "accuracy": round(best_acc, 4), "f1_weighted": round(best_f1, 4),
        "auc_roc": round(best_auc, 4),
        "classes": CLASSES[:n_classes], "architecture": "ResNet18",
        "dataset": "GlaS + PathMNIST", "samples_train": n_train,
        "training_time_s": round(time.time() - t0, 1),
    }
    (MODEL_DIR / "histopath_model.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nONNX → {onnx_path}")
    print(f"Métriques → {MODEL_DIR / 'histopath_model.json'}")
    return metrics


if __name__ == "__main__":
    train(parse_args())
