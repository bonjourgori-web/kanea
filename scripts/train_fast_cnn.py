"""
KANEA — Entraînement rapide CNN (MobileNetV2) pour modules scaffold
===================================================================
Stratégie CPU-friendly : MobileNetV2 + max 1500 images + 5 epochs
~5 min par module sur CPU

Usage:
    python scripts/train_fast_cnn.py --module pulmoscan --dataset data/pulmoscan
    python scripts/train_fast_cnn.py --module derm     --dataset data/derm
    python scripts/train_fast_cnn.py --module osteo    --dataset data/osteo
    python scripts/train_fast_cnn.py --module gastro   --dataset data/gastro
    python scripts/train_fast_cnn.py --module retina   --dataset data/retina
"""
from __future__ import annotations
import argparse, json, time, random
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "deep_learning"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGES = 1500
EPOCHS     = 5
BATCH      = 32

MODULE_CONFIG = {
    "pulmoscan": {
        "classes": ["Normal", "Pneumonie"],
        "img_root_hints": ["chest_xray/train", "train", ""],
        "class_hints": {"NORMAL": 0, "PNEUMONIA": 1, "normal": 0, "pneumonia": 1},
    },
    "derm": {
        "classes": ["AKIEC", "BCC", "BKL", "MEL", "NV"],
        "img_root_hints": ["MILK10k_Training_Input/MILK10k_Training_Input",
                           "MILK10k_Training_Input", ""],
        "class_hints": {},
    },
    "osteo": {
        "classes": ["Normal", "Fracture"],
        "img_root_hints": ["xray_bone_fracture/train", "train", ""],
        "class_hints": {"not fractured": 0, "fractured": 1,
                        "Not Fractured": 0, "Fractured": 1},
    },
    "gastro": {
        "classes": ["Score 0", "Score 1", "Score 2", "Score 3"],
        "img_root_hints": [
            "nerthus-dataset-frames/nerthus-dataset-frames",
            "nerthus-dataset-frames", ""],
        "class_hints": {},
    },
    "retina": {
        "classes": ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"],
        "img_root_hints": ["Diabetic retinopathy", ""],
        "class_hints": {},
    },
}


def find_root(data_dir: Path, hints: list) -> Path:
    for h in hints:
        candidate = data_dir / h if h else data_dir
        if not candidate.exists():
            continue
        subdirs = [d for d in candidate.iterdir() if d.is_dir()]
        if not subdirs:
            continue
        for sd in subdirs:
            imgs = (list(sd.glob("*.jpg")) + list(sd.glob("*.png")) +
                    list(sd.glob("*.jpeg")))
            if imgs:
                return candidate
    return data_dir


def build_dataset_from_folder(img_root: Path, class_hints: dict, transform):
    try:
        ds = ImageFolder(str(img_root), transform=transform)
        return ds
    except Exception:
        pass
    # Fallback: scan récursif
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    samples, class_to_idx = [], {}
    for img_path in img_root.rglob("*"):
        if img_path.suffix.lower() not in extensions:
            continue
        parent = img_path.parent.name
        label_name = class_hints.get(parent, parent)
        if label_name not in class_to_idx:
            class_to_idx[label_name] = len(class_to_idx)
        samples.append((str(img_path), class_to_idx[label_name]))

    class MyDS(torch.utils.data.Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
            self.classes = list(class_to_idx.keys())
        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            path, label = self.samples[i]
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224))
            if self.transform:
                img = self.transform(img)
            return img, label

    return MyDS(samples, transform)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--module",  required=True, choices=list(MODULE_CONFIG.keys()))
    p.add_argument("--dataset", required=True)
    p.add_argument("--epochs",  type=int, default=EPOCHS)
    p.add_argument("--max_imgs",type=int, default=MAX_IMAGES)
    return p.parse_args()


def train(args):
    t0 = time.time()
    cfg      = MODULE_CONFIG[args.module]
    data_dir = ROOT / args.dataset
    device   = torch.device("cpu")

    tfm = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tfm_val = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img_root = find_root(data_dir, cfg["img_root_hints"])
    print(f"[{args.module}] Images dans: {img_root}")

    full_ds = build_dataset_from_folder(img_root, cfg["class_hints"], tfm)
    n_total = len(full_ds)
    print(f"[{args.module}] {n_total} images trouvées")

    if n_total == 0:
        print(f"ERREUR: aucune image dans {img_root}")
        return

    # Sous-échantillonnage si trop d'images
    if n_total > args.max_imgs:
        indices = random.sample(range(n_total), args.max_imgs)
        full_ds = Subset(full_ds, indices)
        print(f"[{args.module}] Réduit à {args.max_imgs} images")

    n_val   = max(1, int(len(full_ds) * 0.2))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Appliquer transform val sur val_ds
    train_loader = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=0)

    # Détecter n_classes
    if hasattr(full_ds, 'dataset'):
        inner = full_ds.dataset
    else:
        inner = full_ds
    n_classes = len(inner.classes) if hasattr(inner, 'classes') else len(cfg["classes"])
    n_classes = max(n_classes, 2)
    print(f"[{args.module}] {n_classes} classes | train={n_train} val={n_val}")

    # MobileNetV2 — rapide sur CPU
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_state, best_acc = 0.0, None, 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                out = model(imgs)
                preds.extend(out.argmax(1).numpy())
                trues.extend(labels.numpy())

        f1  = f1_score(trues, preds, average="weighted", zero_division=0)
        acc = accuracy_score(trues, preds)
        print(f"[{args.module}] Ep {epoch+1}/{args.epochs} loss:{total_loss/len(train_loader):.3f} Acc:{acc:.3f} F1:{f1:.3f}")
        if f1 > best_f1:
            best_f1, best_acc = f1, acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # Export ONNX
    onnx_path = MODEL_DIR / f"{args.module}_model.onnx"
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=18,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "b"}, "output": {0: "b"}},
    )
    np.save(MODEL_DIR / f"{args.module}_fc_weights.npy",
            model.classifier[1].weight.data.numpy())

    classes_out = (inner.classes if hasattr(inner, 'classes')
                   else cfg["classes"][:n_classes])
    metrics = {
        "accuracy":  round(float(best_acc), 4),
        "f1_weighted": round(float(best_f1), 4),
        "classes":   classes_out,
        "architecture": "MobileNetV2",
        "dataset":   args.module,
        "samples_train": n_train,
        "training_time_s": round(time.time() - t0, 1),
    }
    (MODEL_DIR / f"{args.module}_model.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[{args.module}] ONNX -> {onnx_path} ({onnx_path.stat().st_size//1024} KB)")
    print(f"[{args.module}] Done en {metrics['training_time_s']}s | Acc:{best_acc:.3f} F1:{best_f1:.3f}")
    return metrics


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    train(parse_args())
