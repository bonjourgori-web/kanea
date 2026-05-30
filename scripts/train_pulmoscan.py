"""
PulmoScan AI — Pipeline d'entraînement DenseNet121
===================================================
Architecture : DenseNet121 pré-entraîné ImageNet → fine-tuning multi-label
Datasets     : NIH ChestXray14 + CheXpert + COVIDx
Classes      : 16 pathologies pulmonaires + Normal
Export       : ONNX Runtime (CPU-compatible, sans PyTorch en prod)

Usage :
    python scripts/train_pulmoscan.py --dataset data/pulmoscan --epochs 30
    python scripts/train_pulmoscan.py --dataset data/pulmoscan --resume --checkpoint models/deep_learning/pulmoscan_checkpoint.pth
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Classes et configuration
# ═══════════════════════════════════════════════════════════════════════════════

CLASSES = [
    "Normal", "Pneumonie bactérienne", "Pneumonie virale", "COVID-19",
    "Tuberculose pulmonaire", "Cancer pulmonaire", "Nodule pulmonaire",
    "Fibrose pulmonaire", "BPCO / Emphysème", "Bronchiectasies",
    "Atélectasie", "Épanchement pleural", "Pneumothorax", "Œdème pulmonaire",
    "Hypertension pulmonaire", "Maladie interstitielle",
]
NUM_CLASSES = len(CLASSES)

# Hyperparamètres optimaux (issus de la littérature)
CONFIG = {
    "model":          "densenet121",
    "pretrained":     True,
    "input_size":     224,
    "batch_size":     32,
    "lr_head":        1e-3,
    "lr_backbone":    1e-5,
    "weight_decay":   1e-4,
    "epochs":         50,
    "warmup_epochs":  3,
    "patience":       8,
    "label_smoothing": 0.1,
    "mixup_alpha":    0.3,
    "cutmix_alpha":   1.0,
    "tta_augments":   5,    # Test-Time Augmentation
    "grad_clip":      1.0,
    "amp":            True, # Automatic Mixed Precision
}


# ═══════════════════════════════════════════════════════════════════════════════
# Datasets référencés
# ═══════════════════════════════════════════════════════════════════════════════

DATASETS = {
    "NIH ChestXray14": {
        "url":          "https://nihcc.app.box.com/v/ChestXray-NIHCC",
        "images":       112_120,
        "patients":     30_805,
        "classes":      14,
        "labels":       "Multi-label (15 pathologies + Finding Labels)",
        "format":       "PNG 1024×1024",
        "license":      "NIH Open License — usage recherche uniquement",
        "download":     "kaggle datasets download -d nih-chest-xrays/data",
        "size_gb":      42,
        "notes":        "Dataset de référence mondial — utilisé dans CheXNet (Stanford)",
    },
    "CheXpert": {
        "url":          "https://stanfordmlgroup.github.io/competitions/chexpert/",
        "images":       224_316,
        "patients":     65_240,
        "classes":      14,
        "labels":       "Multi-label avec incertitude (U-label)",
        "format":       "JPEG",
        "license":      "Stanford Research License",
        "download":     "Inscription sur le site Stanford requis",
        "size_gb":      439,
        "notes":        "Meilleur dataset pour pneumonie + atélectasie",
    },
    "MIMIC-CXR": {
        "url":          "https://physionet.org/content/mimic-cxr/2.0.0/",
        "images":       377_110,
        "patients":     65_379,
        "classes":      14,
        "labels":       "Multi-label + rapports radiologiques libres",
        "format":       "DICOM + JPEG",
        "license":      "PhysioNet Credentialed Access",
        "download":     "wget -r -N -c -np https://physionet.org/files/mimic-cxr/2.0.0/",
        "size_gb":      560,
        "notes":        "Unique dataset avec radiology reports — NLP possible",
    },
    "COVIDx": {
        "url":          "https://github.com/lindawangg/COVID-Net",
        "images":       30_386,
        "patients":     None,
        "classes":      3,
        "labels":       "Normal / Non-COVID / COVID-19",
        "format":       "PNG",
        "license":      "GNU AGPL v3",
        "download":     "python create_COVIDx_dataset.py",
        "size_gb":      4,
        "notes":        "Meilleur dataset COVID radiographie thoracique",
    },
    "Montgomery TB": {
        "url":          "https://openi.nlm.nih.gov/faq#collection",
        "images":       138,
        "patients":     138,
        "classes":      2,
        "labels":       "Normal / Tuberculose",
        "format":       "PNG 4020×4892",
        "license":      "NIH Open License",
        "download":     "https://ceb.nlm.nih.gov/repositories/tuberculosis-chest-x-ray-image-data-sets/",
        "size_gb":      1.5,
        "notes":        "Petite taille — idéal en complément de Shenzhen",
    },
    "Shenzhen TB": {
        "url":          "https://openi.nlm.nih.gov/faq#collection",
        "images":       662,
        "patients":     662,
        "classes":      2,
        "labels":       "Normal / Tuberculose",
        "format":       "PNG",
        "license":      "NIH Open License",
        "download":     "https://ceb.nlm.nih.gov/repositories/tuberculosis-chest-x-ray-image-data-sets/",
        "size_gb":      4,
        "notes":        "Dataset TB adapté contexte africain — Guangzhou",
    },
    "LIDC-IDRI": {
        "url":          "https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI",
        "images":       244_527,
        "patients":     1_010,
        "classes":      "Nodules annotés (4 niveaux de malignité)",
        "labels":       "Nodule segmentation + malignancy score 1–5",
        "format":       "DICOM CT",
        "license":      "TCIA Public License",
        "download":     "TCIA Downloader ou NBIA Data Retriever",
        "size_gb":      125,
        "notes":        "Gold standard pour détection de nodules pulmonaires",
    },
    "VinDr-CXR": {
        "url":          "https://physionet.org/content/vindr-cxr/1.0.0/",
        "images":       18_000,
        "patients":     None,
        "classes":      28,
        "labels":       "Bounding boxes + 28 pathologies",
        "format":       "DICOM",
        "license":      "PhysioNet Open Access",
        "download":     "wget -r -N -c -np https://physionet.org/files/vindr-cxr/1.0.0/",
        "size_gb":      40,
        "notes":        "Annotations par 17 radiologistes — haute qualité",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline d'augmentation
# ═══════════════════════════════════════════════════════════════════════════════

AUGMENTATION_TRAIN = """
# Augmentation entraînement (torchvision + albumentations)
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.GaussNoise(var_limit=(10, 50), p=0.4),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
"""

TRAINING_SCRIPT = '''#!/usr/bin/env python
"""
PulmoScan AI — Script d'entraînement DenseNet121 complet
=========================================================
Nécessite : torch >= 2.0, torchvision, albumentations, timm, onnx, onnxruntime
"""
import os, json, time, argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

CLASSES = {classes}
NUM_CLASSES = len(CLASSES)

# ── Dataset ──────────────────────────────────────────────────────────────────
class ChestXRayDataset(Dataset):
    def __init__(self, image_dir, labels_csv, transform=None, multi_label=True):
        import pandas as pd
        self.df = pd.read_csv(labels_csv)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.multi_label = multi_label
        self.classes = CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row["filename"]
        img = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        if self.multi_label:
            label = torch.FloatTensor([row.get(c, 0) for c in self.classes])
        else:
            label = torch.LongTensor([row["label"]])
        return img, label


# ── Modèle ────────────────────────────────────────────────────────────────────
class PulmoScanModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.backbone = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        # Grad-CAM hooks
        self.feature_maps = None
        self.backbone.features.denseblock4.register_forward_hook(
            lambda m, i, o: setattr(self, "feature_maps", o)
        )

    def forward(self, x):
        return self.backbone(x)


# ── Loss multi-label ──────────────────────────────────────────────────────────
class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pt = torch.exp(-bce_loss)
        focal = (1 - pt) ** self.gamma * bce_loss
        return focal.mean()


# ── Mixup ─────────────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return x_mix, y_mix


# ── Entraînement ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler, device, clip=1.0):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs, labels = mixup_data(imgs, labels, alpha=0.3)
        optimizer.zero_grad()
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Évaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    preds  = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    auc = roc_auc_score(labels, preds, average="macro", multi_output="ovr")
    ap  = average_precision_score(labels, preds, average="macro")
    return total_loss / len(loader), auc, ap


# ── Export ONNX ───────────────────────────────────────────────────────────────
def export_onnx(model, output_path, input_size=224):
    import onnx, onnxruntime
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={{"input": {{0: "batch"}}, "output": {{0: "batch"}}}},
        opset_version=17,
        do_constant_folding=True,
    )
    # Vérification
    onnx.checker.check_model(str(output_path))
    sess = onnxruntime.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {{"input": dummy.numpy()}})[0]
    print(f"ONNX export OK — shape: {{out.shape}}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {{device}}")

    ROOT = Path(__file__).resolve().parents[1]
    MODEL_DIR = ROOT / "models" / "deep_learning"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Transforms
    train_tf = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.7),
        A.GaussNoise(var_limit=(10, 50), p=0.4),
        A.CLAHE(clip_limit=2.0, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(256, 256), A.CenterCrop(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    dataset_path = Path(args.dataset)
    train_ds = ChestXRayDataset(dataset_path/"images", dataset_path/"train.csv", train_tf)
    val_ds   = ChestXRayDataset(dataset_path/"images", dataset_path/"val.csv",   val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = PulmoScanModel(NUM_CLASSES, pretrained=True).to(device)

    # Optim 2 groupes (backbone lr < head lr)
    optimizer = optim.AdamW([
        {{"params": model.backbone.features.parameters(), "lr": args.lr * 0.01}},
        {{"params": model.backbone.classifier.parameters(), "lr": args.lr}},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = FocalBCELoss(gamma=2.0)
    scaler    = GradScaler()

    if args.resume and args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Resumed from {{args.checkpoint}}")

    best_auc = 0.0
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_auc, val_ap = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {{epoch:3d}}/{args.epochs} | "
              f"loss {{train_loss:.4f}} | val_auc {{val_auc:.4f}} | "
              f"val_ap {{val_ap:.4f}} | {{elapsed:.1f}}s")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_count = 0
            torch.save({{
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "classes": CLASSES,
            }}, str(MODEL_DIR / "pulmoscan_best.pth"))
            print(f"  → Meilleur modèle sauvegardé (AUC: {{best_auc:.4f}})")
        else:
            patience_count += 1
            if patience_count >= 8:
                print("Early stopping déclenché.")
                break

    # Export ONNX
    best_ckpt = torch.load(MODEL_DIR / "pulmoscan_best.pth", map_location="cpu")
    model.load_state_dict(best_ckpt["model_state_dict"])
    export_onnx(model, MODEL_DIR / "pulmoscan_model.onnx")
    print(f"ONNX exporté → models/deep_learning/pulmoscan_model.onnx")

    # Sauvegarder métadonnées
    meta = {{
        "classes": CLASSES, "architecture": "DenseNet121",
        "input_size": 224, "best_auc": best_auc,
        "model_version": "v2.0", "normalization": {{"mean": [0.485,0.456,0.406], "std": [0.229,0.224,0.225]}},
    }}
    with open(MODEL_DIR / "pulmoscan_model.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Entraînement terminé.")


if __name__ == "__main__":
    main()
'''.format(classes=repr(CLASSES))


def main():
    parser = argparse.ArgumentParser(
        description="PulmoScan AI — génère le script d'entraînement ou affiche les datasets"
    )
    parser.add_argument("--generate-script", action="store_true",
                        help="Génère le script train_pulmoscan_runner.py")
    parser.add_argument("--list-datasets", action="store_true",
                        help="Liste les datasets disponibles avec leurs métadonnées")
    parser.add_argument("--export-config", action="store_true",
                        help="Exporte la configuration d'hyperparamètres")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parent.parent

    if args.generate_script:
        out = ROOT / "scripts" / "train_pulmoscan_runner.py"
        out.write_text(TRAINING_SCRIPT, encoding="utf-8")
        print(f"Script généré : {out}")
        print("Usage : python scripts/train_pulmoscan_runner.py --dataset data/pulmoscan --epochs 50")

    elif args.list_datasets:
        print("\n=== Datasets PulmoScan AI ===\n")
        for name, info in DATASETS.items():
            print(f"{'─'*60}")
            print(f"  {name}")
            print(f"  Images   : {info['images']:,}")
            print(f"  Format   : {info['format']}")
            print(f"  Licence  : {info['license']}")
            print(f"  Taille   : {info.get('size_gb', '?')} GB")
            print(f"  Download : {info['download']}")
            print(f"  Note     : {info['notes']}")
        print()

    elif args.export_config:
        out = ROOT / "scripts" / "pulmoscan_config.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({
                "classes": CLASSES,
                "num_classes": NUM_CLASSES,
                "training_config": CONFIG,
                "datasets": {k: {kk: vv for kk, vv in v.items() if kk not in ("download",)}
                             for k, v in DATASETS.items()},
            }, f, indent=2, ensure_ascii=False)
        print(f"Config exportée : {out}")

    else:
        print("PulmoScan AI — Training Pipeline")
        print(f"  Classes     : {NUM_CLASSES}")
        print(f"  Architecture: {CONFIG['model']}")
        print(f"  Datasets    : {len(DATASETS)}")
        print()
        print("Options :")
        print("  --generate-script   Générer le script d'entraînement complet")
        print("  --list-datasets     Lister tous les datasets avec métadonnées")
        print("  --export-config     Exporter la configuration JSON")


if __name__ == "__main__":
    main()
