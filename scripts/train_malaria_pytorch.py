"""
KANÉA — Entraînement malaria deep learning (PyTorch / ResNet34)
═══════════════════════════════════════════════════════════════
Dataset  : NIH Malaria Cell Images (27 560 images, 2 classes)
Modèle   : ResNet34 pré-entraîné ImageNet, fine-tuning complet
Sortie   : models/deep_learning/malaria_model.pth  (state_dict)
          models/deep_learning/malaria_model.json (métadonnées)

Usage :
    python scripts/train_malaria_pytorch.py
    python scripts/train_malaria_pytorch.py --epochs 15 --batch-size 64
    python scripts/train_malaria_pytorch.py --data-dir data/malaria --fast-dev
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── Imports ──────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, models, transforms
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, recall_score
    )
    import numpy as np
except ImportError as exc:
    print(f"[ERREUR] Dépendance manquante : {exc}")
    print("  → pip install torch torchvision scikit-learn numpy")
    sys.exit(1)

# ─── Constantes ───────────────────────────────────────────────────────────────
CLASSES        = ["Parasitised", "Uninfected"]
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]
IMG_SIZE       = 224


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════════

def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class MalariaDataset(torch.utils.data.Dataset):
    """Wrapper sur ImageFolder avec transform dynamique train/val."""

    def __init__(self, root: Path, train: bool = True):
        self.dataset   = datasets.ImageFolder(root=str(root))
        self.transform = get_transforms(train)
        self.classes   = self.dataset.classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        return self.transform(img), label


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    valid_pct: float,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, list[str]]:
    full_ds = MalariaDataset(data_dir, train=True)
    classes = full_ds.classes

    n_total = len(full_ds)
    n_valid = int(n_total * valid_pct)
    n_train = n_total - n_valid

    train_ds, val_ds = random_split(
        full_ds, [n_train, n_valid],
        generator=torch.Generator().manual_seed(42),
    )

    # Le split garde le même transform — on reconstruit val_ds avec train=False
    val_ds_proper = torch.utils.data.Subset(
        MalariaDataset(data_dir, train=False),
        val_ds.indices,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds_proper, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"  Dataset  : {n_total:,} images ({classes})")
    print(f"  Train    : {n_train:,} | Val : {n_valid:,}")
    return train_loader, val_loader, classes


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLE
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int = 2, freeze_backbone: bool = False) -> nn.Module:
    """ResNet34 pré-entraîné ImageNet, tête de classification remplacée."""
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Remplacement de la couche finale
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# BOUCLES D'ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, float, float]:
    """Retourne (loss_moy, accuracy, recall_parasitised)."""
    model.train(train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    n      = len(all_labels)
    acc    = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    return total_loss / n, acc, recall


def train_model(
    data_dir: Path,
    export_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    valid_pct: float,
    num_workers: int,
    freeze_epochs: int,
    fast_dev: bool,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*60}")
    print(f"  KANÉA — Entraînement Malaria DL (ResNet34)")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {epochs}  |  Batch : {batch_size}  |  LR : {lr}")
    print(f"{'═'*60}")

    train_loader, val_loader, classes = build_dataloaders(
        data_dir, batch_size, valid_pct, num_workers
    )

    if fast_dev:
        print("  ⚡ Mode fast-dev : 2 batches seulement")
        epochs = 1

    # Poids de classes inversés pour maximiser le recall (paludisme = classe 0)
    class_weights = torch.tensor([1.5, 1.0], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(num_classes=len(classes), freeze_backbone=(freeze_epochs > 0))
    model = model.to(device)

    # Phase 1 : backbone gelé (si demandé)
    if freeze_epochs > 0:
        print(f"\n  Phase 1 — Backbone gelé ({freeze_epochs} époques)")
        optimizer_frozen = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )
        scheduler_frozen = optim.lr_scheduler.OneCycleLR(
            optimizer_frozen, max_lr=lr,
            steps_per_epoch=len(train_loader), epochs=freeze_epochs,
        )
        for ep in range(freeze_epochs):
            t0 = time.time()
            tr_loss, tr_acc, tr_rec = _run_epoch(
                model, train_loader, criterion, optimizer_frozen, device, train=True
            )
            vl_loss, vl_acc, vl_rec = _run_epoch(
                model, val_loader, criterion, None, device, train=False
            )
            scheduler_frozen.step()
            print(
                f"  Ep {ep+1:02d}/{freeze_epochs}  "
                f"loss={tr_loss:.4f}/{vl_loss:.4f}  "
                f"acc={tr_acc:.3f}/{vl_acc:.3f}  "
                f"recall={tr_rec:.3f}/{vl_rec:.3f}  "
                f"({time.time()-t0:.0f}s)"
            )
        # Dégel complet
        for param in model.parameters():
            param.requires_grad = True

    # Phase 2 : fine-tuning complet
    finetune_epochs = epochs - freeze_epochs if freeze_epochs > 0 else epochs
    print(f"\n  Phase 2 — Fine-tuning complet ({finetune_epochs} époques)")

    optimizer = optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)

    best_recall = 0.0
    best_state  = None
    history     = []

    for ep in range(finetune_epochs):
        t0 = time.time()
        tr_loss, tr_acc, tr_rec = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        vl_loss, vl_acc, vl_rec = _run_epoch(
            model, val_loader, criterion, None, device, train=False
        )
        scheduler.step()

        history.append({
            "epoch": ep + 1,
            "train_loss": round(tr_loss, 4),
            "val_loss": round(vl_loss, 4),
            "train_acc": round(tr_acc, 4),
            "val_acc": round(vl_acc, 4),
            "val_recall_parasitised": round(vl_rec, 4),
        })

        flag = ""
        if vl_rec > best_recall:
            best_recall = vl_rec
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            flag = "  ← 🏆 best recall"

        print(
            f"  Ep {ep+1:02d}/{finetune_epochs}  "
            f"loss={tr_loss:.4f}/{vl_loss:.4f}  "
            f"acc={tr_acc:.3f}/{vl_acc:.3f}  "
            f"recall={tr_rec:.3f}/{vl_rec:.3f}  "
            f"({time.time()-t0:.0f}s){flag}"
        )

        if fast_dev:
            break

    # ── Évaluation finale ──────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            preds = model(images.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(f"\n{'─'*60}")
    print("  Rapport de classification final :")
    print(classification_report(all_labels, all_preds, target_names=classes))
    print("  Matrice de confusion :")
    print(confusion_matrix(all_labels, all_preds))

    final_acc    = round(accuracy_score(all_labels, all_preds), 4)
    final_recall = round(recall_score(all_labels, all_preds, pos_label=0, zero_division=0), 4)

    # ── Sauvegarde ─────────────────────────────────────────────────────────────
    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, export_path)
    print(f"\n  ✅ Modèle sauvegardé : {export_path}")

    metadata = {
        "model_name":     "resnet34_malaria_kanea",
        "architecture":   "ResNet34",
        "library":        "pytorch",
        "classes":        classes,
        "num_classes":    len(classes),
        "img_size":       IMG_SIZE,
        "val_accuracy":   final_acc,
        "val_recall_parasitised": final_recall,
        "epochs_trained": epochs,
        "batch_size":     batch_size,
        "device_used":    str(device),
        "trained_at":     datetime.now().isoformat(),
        "dataset":        "NIH Malaria Cell Images (27 560 images)",
        "history":        history,
    }
    meta_path = export_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"  ✅ Métadonnées : {meta_path}")
    print(f"\n  Accuracy finale : {final_acc:.1%}  |  Recall Parasitised : {final_recall:.1%}")
    print(f"{'═'*60}\n")
    return metadata


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraîne le modèle malaria ResNet34 pour KANÉA."
    )
    parser.add_argument("--data-dir",   type=Path, default=PROJECT_ROOT / "data" / "malaria")
    parser.add_argument("--export-path",type=Path, default=PROJECT_ROOT / "models" / "deep_learning" / "malaria_model.pth")
    parser.add_argument("--epochs",     type=int,  default=12)
    parser.add_argument("--batch-size", type=int,  default=32)
    parser.add_argument("--lr",         type=float,default=1e-3)
    parser.add_argument("--valid-pct",  type=float,default=0.15)
    parser.add_argument("--num-workers",type=int,  default=0)
    parser.add_argument("--freeze-epochs",type=int,default=3,
                        help="Époques backbone gelé avant fine-tuning complet")
    parser.add_argument("--fast-dev",   action="store_true",
                        help="Test rapide : 2 batches seulement")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        print(f"[ERREUR] Dataset introuvable : {args.data_dir}")
        print("  → Vérifier que data/malaria/Parasitised/ et data/malaria/Uninfected/ existent")
        sys.exit(1)

    train_model(
        data_dir=args.data_dir,
        export_path=args.export_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        valid_pct=args.valid_pct,
        num_workers=args.num_workers,
        freeze_epochs=args.freeze_epochs,
        fast_dev=args.fast_dev,
    )


if __name__ == "__main__":
    main()
