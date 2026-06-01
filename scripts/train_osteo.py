"""
OsteoDetect AI — Entraînement EfficientNet-B0 sur X-ray fractures
==================================================================
Dataset : X-Ray Bone Fracture (train/fractured + train/not fractured)
Classes : Fracture / Normal
Export  : ONNX Runtime CPU

Usage :
    python scripts/train_osteo.py --dataset data/osteo --epochs 20
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
import onnxruntime

ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "deep_learning"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Fracture"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/osteo")
    p.add_argument("--epochs",  type=int, default=15)
    p.add_argument("--batch",   type=int, default=32)
    p.add_argument("--lr",      type=float, default=1e-4)
    return p.parse_args()


class FractureDataset(Dataset):
    def __init__(self, root: Path, split: str, transform=None):
        self.samples = []
        self.transform = transform
        # Chercher train/ ou val/ ou directement fractured/not fractured
        for split_dir in [root / split, root / "xray_bone_fracture" / split, root]:
            for label, name in enumerate(["not fractured", "fractured"]):
                for candidate in [split_dir / name, split_dir / name.replace(" ","_"),
                                  split_dir / name.title()]:
                    if candidate.exists():
                        for ext in ["*.jpg","*.jpeg","*.png","*.bmp"]:
                            for img in candidate.rglob(ext):
                                self.samples.append((img, label))
            if self.samples:
                break
        print(f"  {split}: {len(self.samples)} images — "
              f"Fracture:{sum(1 for _,l in self.samples if l==1)} "
              f"Normal:{sum(1 for _,l in self.samples if l==0)}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(train: bool):
    if train:
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
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

    train_ds = FractureDataset(data_dir, "train", get_transforms(True))
    val_ds   = FractureDataset(data_dir, "val",   get_transforms(False))

    if not train_ds.samples:
        print("ERREUR : aucune image trouvée dans data/osteo/")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    # EfficientNet-B0 pré-entraîné
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_state = 0.0, None

    for epoch in range(args.epochs):
        # Train
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Val
        model.eval()
        all_preds, all_labels, all_proba = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                out  = model(imgs)
                prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                pred = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(pred); all_labels.extend(labels.numpy())
                all_proba.extend(prob)

        if not all_labels:
            continue
        f1  = f1_score(all_labels, all_preds, average="binary", zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, all_proba)
        except Exception:
            auc = 0.0
        print(f"Epoch {epoch+1}/{args.epochs} — Acc:{acc:.3f}  F1:{f1:.3f}  AUC:{auc:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # Export ONNX
    onnx_path = MODEL_DIR / "osteo_model.onnx"
    dummy = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(model, dummy, onnx_path,
                      input_names=["input"], output_names=["output"],
                      opset_version=17, dynamic_axes={"input":{0:"batch"},"output":{0:"batch"}})

    # Sauvegarder poids FC pour KANEA
    fc_weights = model.classifier[1].weight.data.cpu().numpy()
    np.save(MODEL_DIR / "osteo_fc_weights.npy", fc_weights)

    metrics = {
        "accuracy": round(acc, 4), "f1_binary": round(best_f1, 4), "auc_roc": round(auc, 4),
        "classes": CLASSES, "architecture": "EfficientNet-B0",
        "dataset": "X-Ray Bone Fracture", "samples_train": len(train_ds),
        "training_time_s": round(time.time() - t0, 1),
    }
    (MODEL_DIR / "osteo_model.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nONNX → {onnx_path}")
    print(f"Métriques → {MODEL_DIR / 'osteo_model.json'}")
    return metrics


if __name__ == "__main__":
    train(parse_args())
