# -*- coding: utf-8 -*-
"""
MalariaScan AI - Entrainement ResNet34 sur NIH dataset (27 560 images).

Usage :
    cd c:/Users/HP/gori/KANEA
    python scripts/train_malaria_resnet34.py

Sortie :
    models/deep_learning/malaria_model.onnx
    models/deep_learning/malaria_model.pth
    scripts/malaria_training_metrics.json
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

ROOT        = Path(__file__).resolve().parents[1]
DATA_PARA   = ROOT / "data" / "malaria" / "Parasitised"
DATA_UNINF  = ROOT / "data" / "malaria" / "Uninfected"
MODEL_DIR   = ROOT / "models" / "deep_learning"
ONNX_OUT    = MODEL_DIR / "malaria_model.onnx"
PTH_OUT     = MODEL_DIR / "malaria_model.pth"
METRICS_OUT = ROOT / "scripts" / "malaria_training_metrics.json"

IMG_SIZE         = 224
BATCH_SIZE       = 32
VALID_PCT        = 0.20
FREEZE_EPOCHS    = 3
UNFREEZE_EPOCHS  = 10
BASE_LR          = 3e-3
BACKBONE_LR      = 1e-5
LABEL_SMOOTHING  = 0.10
SEED             = 42
CLASSES          = ["Parasitized", "Uninfected"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MalariaDataset(Dataset):
    def __init__(self, paths: list[Path], labels: list[int], transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def build_split(valid_pct: float = 0.20, seed: int = 42):
    rng = random.Random(seed)

    para  = sorted(DATA_PARA.glob("*"))
    uninf = sorted(DATA_UNINF.glob("*"))

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    para  = [p for p in para  if p.suffix.lower() in exts]
    uninf = [p for p in uninf if p.suffix.lower() in exts]

    rng.shuffle(para)
    rng.shuffle(uninf)

    n_valid_p = int(len(para)  * valid_pct)
    n_valid_u = int(len(uninf) * valid_pct)

    train_paths  = para[n_valid_p:]  + uninf[n_valid_u:]
    train_labels = [0] * len(para[n_valid_p:]) + [1] * len(uninf[n_valid_u:])

    valid_paths  = para[:n_valid_p]  + uninf[:n_valid_u]
    valid_labels = [0] * n_valid_p   + [1] * n_valid_u

    combined_train = list(zip(train_paths, train_labels))
    combined_valid = list(zip(valid_paths, valid_labels))
    rng.shuffle(combined_train)
    rng.shuffle(combined_valid)

    t_paths, t_labels = zip(*combined_train)
    v_paths, v_labels = zip(*combined_valid)
    return list(t_paths), list(t_labels), list(v_paths), list(v_labels)


def make_transforms(augment: bool = True):
    if augment:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model(device: torch.device) -> nn.Module:
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc")


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def compute_metrics(model, loader, device) -> dict:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(1)
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs[:, 0].cpu().tolist())

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "accuracy":    round(accuracy_score(all_labels, all_preds), 4),
        "precision":   round(precision_score(all_labels, all_preds, zero_division=0), 4),
        "recall":      round(recall_score(all_labels, all_preds, zero_division=0), 4),
        "f1_score":    round(f1_score(all_labels, all_preds, zero_division=0), 4),
        "auc_roc":     round(roc_auc_score(all_labels, all_probs), 4) if len(set(all_labels)) > 1 else 0.0,
        "sensitivity": round(float(tp / (tp + fn)) if (tp + fn) else 0.0, 4),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            all_labels, all_preds, target_names=CLASSES, output_dict=True, zero_division=0
        ),
    }


def export_onnx(model: nn.Module, path: Path) -> bool:
    model.cpu().eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    try:
        torch.onnx.export(
            model, dummy, str(path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        print("  [OK] ONNX exporte : %s  (%d MB)" % (path, path.stat().st_size // 1_000_000))
        return True
    except Exception as exc:
        print("  [ERR] ONNX export echoue : %s" % exc)
        return False


def main() -> None:

    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*60)
    print("  MalariaScan AI - Entrainement ResNet34")
    print("  Device : %s" % device)
    print("="*60 + "\n")

    print("[*] Chargement du dataset NIH (27 560 images)...")
    t_paths, t_labels, v_paths, v_labels = build_split(VALID_PCT, SEED)
    print("  Train : %d images  |  Valid : %d images" % (len(t_paths), len(v_paths)))
    print("  Classes - Parasitized: %d  |  Uninfected: %d\n" % (
        t_labels.count(0) + v_labels.count(0),
        t_labels.count(1) + v_labels.count(1)
    ))

    train_ds = MalariaDataset(t_paths, t_labels, make_transforms(augment=True))
    valid_ds = MalariaDataset(v_paths, v_labels, make_transforms(augment=False))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = build_model(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Phase 1 : backbone gele
    print("[*] Phase 1 : backbone gele - %d epochs  (lr=%.4f)" % (FREEZE_EPOCHS, BASE_LR))
    freeze_backbone(model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

    best_loss = float("inf")
    history   = []
    t0        = time.time()

    for epoch in range(FREEZE_EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = eval_epoch(model,  valid_loader, criterion, device)
        scheduler.step(vl_loss)
        history.append({"epoch": epoch, "phase": "frozen",
                         "train_loss": round(tr_loss, 4), "valid_loss": round(vl_loss, 4),
                         "valid_acc": round(vl_acc, 4)})
        print("  Epoch %02d/%d  train_loss=%.4f  valid_loss=%.4f  valid_acc=%.4f" % (
            epoch+1, FREEZE_EPOCHS, tr_loss, vl_loss, vl_acc))
        if vl_loss < best_loss:
            best_loss = vl_loss
            torch.save(model.state_dict(), PTH_OUT)
            print("    -> best model sauve (loss=%.4f)" % best_loss)

    # Phase 2 : backbone degele
    print("\n[*] Phase 2 : backbone degele - %d epochs  (lr=%.0e)" % (UNFREEZE_EPOCHS, BACKBONE_LR))
    unfreeze_all(model)
    optimizer = torch.optim.Adam([
        {"params": model.fc.parameters(), "lr": BASE_LR},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("fc")], "lr": BACKBONE_LR},
    ])
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    patience_cnt = 0

    for epoch in range(UNFREEZE_EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = eval_epoch(model,  valid_loader, criterion, device)
        scheduler.step(vl_loss)
        ep = FREEZE_EPOCHS + epoch
        history.append({"epoch": ep, "phase": "unfrozen",
                         "train_loss": round(tr_loss, 4), "valid_loss": round(vl_loss, 4),
                         "valid_acc": round(vl_acc, 4)})
        print("  Epoch %02d/%d  train_loss=%.4f  valid_loss=%.4f  valid_acc=%.4f" % (
            ep+1, FREEZE_EPOCHS+UNFREEZE_EPOCHS, tr_loss, vl_loss, vl_acc))

        if vl_loss < best_loss:
            best_loss    = vl_loss
            patience_cnt = 0
            torch.save(model.state_dict(), PTH_OUT)
            print("    -> best model sauve (loss=%.4f)" % best_loss)
        else:
            patience_cnt += 1
            if patience_cnt >= 3:
                print("  EarlyStopping declenche.")
                break

    elapsed = time.time() - t0
    print("\n[*] Entrainement termine en %.1f min" % (elapsed / 60))

    # Evaluation finale
    model.load_state_dict(torch.load(PTH_OUT, map_location=device))
    print("\n[*] Evaluation du meilleur modele...")
    metrics = compute_metrics(model, valid_loader, device)
    print("  Accuracy   : %.4f" % metrics["accuracy"])
    print("  F1-Score   : %.4f" % metrics["f1_score"])
    print("  AUC-ROC    : %.4f" % metrics["auc_roc"])
    print("  Sensitivity: %.4f" % metrics["sensitivity"])
    print("  Specificity: %.4f" % metrics["specificity"])

    # Export ONNX
    print("\n[*] Export ONNX...")
    model.load_state_dict(torch.load(PTH_OUT, map_location="cpu"))
    export_onnx(model, ONNX_OUT)

    # Sauvegarde metriques
    output = {
        "model":              "ResNet34",
        "dataset":            "NIH Malaria Cell Images (27 560 images)",
        "epochs":             len(history),
        "best_valid_loss":    round(best_loss, 4),
        "training_time_min":  round(elapsed / 60, 1),
        "metrics":            metrics,
        "history":            history,
        "onnx_path":          str(ONNX_OUT),
    }
    METRICS_OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\n[OK] Metriques  : %s" % METRICS_OUT)
    print("[OK] ONNX       : %s" % ONNX_OUT)
    print("[OK] Checkpoint : %s" % PTH_OUT)
    print("\nProchaine etape : git add models/deep_learning/malaria_model.onnx && git push\n")


if __name__ == "__main__":
    import traceback as _tb
    try:
        main()
    except Exception as _e:
        _lf = open(ROOT / "scripts" / "training_log.txt", "a", encoding="utf-8")
        _lf.write("\n=== CRASH ===\n")
        _lf.write(_tb.format_exc())
        _lf.flush()
        raise
