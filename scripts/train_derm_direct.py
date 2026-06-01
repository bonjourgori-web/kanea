"""
DermAI — Entraînement EfficientNet-B0 sur ISIC MILK10k
=======================================================
Dataset : data/derm/MILK10k_Training_Input/ + GroundTruth.csv
Classes : 10 pathologies dermatologiques (AKIEC, BCC, MEL, NV, ...)
Export  : ONNX Runtime CPU

Usage : python scripts/train_derm_direct.py --dataset data/derm --epochs 15
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "deep_learning"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    "AKIEC":   "Kératose actinique / CIS",
    "BCC":     "Carcinome basocellulaire",
    "BEN_OTH": "Lésion bénigne autre",
    "BKL":     "Kératose bénigne",
    "DF":      "Dermatofibrome",
    "INF":     "Inflammation / Infection",
    "MAL_OTH": "Malignité autre",
    "MEL":     "Mélanome",
    "NV":      "Nævus bénin",
    "SCCKA":   "Carcinome épidermoïde",
    "VASC":    "Lésion vasculaire",
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/derm")
    p.add_argument("--epochs",  type=int, default=15)
    p.add_argument("--batch",   type=int, default=32)
    p.add_argument("--lr",      type=float, default=1e-4)
    return p.parse_args()


class ISICDataset(Dataset):
    def __init__(self, gt_df: pd.DataFrame, img_root: Path, label_cols: list, transform=None):
        self.transform   = transform
        self.label_cols  = label_cols
        self.samples     = []

        for _, row in gt_df.iterrows():
            lesion_id = row["lesion_id"]
            # Chercher l'image dans le sous-dossier du patient
            patient_dir = img_root / lesion_id
            img_path = None
            if patient_dir.exists():
                for ext in ["*.jpg","*.jpeg","*.png","*.bmp"]:
                    imgs = list(patient_dir.glob(ext))
                    if imgs:
                        img_path = imgs[0]
                        break
            # Chercher directement dans img_root
            if img_path is None:
                for ext in [".jpg",".jpeg",".png"]:
                    p = img_root / f"{lesion_id}{ext}"
                    if p.exists():
                        img_path = p
                        break

            if img_path is None:
                continue

            label_vec = row[label_cols].values.astype(np.float32)
            label = int(np.argmax(label_vec))
            self.samples.append((img_path, label))

        print(f"  Dataset: {len(self.samples)} images chargées sur {len(gt_df)} lesions")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (180, 140, 120))
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(train: bool):
    if train:
        return T.Compose([
            T.Resize((256, 256)), T.RandomCrop(224),
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
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

    gt_path = data_dir / "MILK10k_Training_GroundTruth.csv"
    if not gt_path.exists():
        print(f"ERREUR : {gt_path} introuvable")
        return
    gt_df = pd.read_csv(gt_path)
    label_cols = [c for c in gt_df.columns if c != "lesion_id"]
    print(f"Classes : {label_cols}")
    print(f"Distribution : {dict(zip(label_cols, gt_df[label_cols].sum().astype(int)))}")

    img_root = data_dir / "MILK10k_Training_Input"
    # Gérer le dossier imbriqué MILK10k_Training_Input/MILK10k_Training_Input/
    nested = img_root / "MILK10k_Training_Input"
    if nested.exists():
        img_root = nested
    full_ds  = ISICDataset(gt_df, img_root, label_cols, get_transforms(True))

    if not full_ds.samples:
        print("ERREUR : aucune image trouvée")
        return

    n_classes = len(label_cols)
    n_val   = max(1, int(len(full_ds) * 0.2))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    val_ds.dataset.transform = get_transforms(False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    # EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_state, best_acc = 0.0, None, 0.0

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

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                out  = model(imgs.to(device))
                pred = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(pred); all_labels.extend(labels.numpy())

        f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{args.epochs} — loss:{total_loss/len(train_loader):.4f}  Acc:{acc:.3f}  F1:{f1:.3f}")
        if f1 > best_f1:
            best_f1, best_acc = f1, acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    onnx_path = MODEL_DIR / "derm_model.onnx"
    dummy = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(model, dummy, onnx_path,
                      input_names=["input"], output_names=["output"],
                      opset_version=17, dynamic_axes={"input":{0:"b"},"output":{0:"b"}})
    np.save(MODEL_DIR / "derm_fc_weights.npy",
            model.classifier[1].weight.data.cpu().numpy())

    metrics = {
        "accuracy": round(float(best_acc), 4), "f1_weighted": round(float(best_f1), 4),
        "classes": label_cols,
        "class_names": [CLASS_NAMES.get(c, c) for c in label_cols],
        "architecture": "EfficientNet-B0", "dataset": "ISIC MILK10k",
        "samples_train": n_train, "training_time_s": round(time.time() - t0, 1),
    }
    (MODEL_DIR / "derm_model.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nONNX → {onnx_path}")
    return metrics


if __name__ == "__main__":
    train(parse_args())
