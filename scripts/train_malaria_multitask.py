# -*- coding: utf-8 -*-
"""
KANEA MalariaScan AI - Entrainement multi-tache (infection + espece + stade).

Architecture : ResNet50 backbone partage + 3 tetes de classification.
Datasets requis (lancer d'abord download_malaria_datasets.py) :
  - data/malaria_advanced/nih/       -> Tache 1 (infection binaire)
  - data/malaria_advanced/species/   -> Tache 2 (espece Plasmodium)
  - data/malaria_advanced/stages/    -> Tache 3 (stade parasitaire)

Usage :
    cd c:/Users/HP/gori/KANEA
    python scripts/train_malaria_multitask.py [--tasks infection,species,stage] [--epochs 25]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from sklearn.metrics import accuracy_score, f1_score

ROOT         = Path(__file__).resolve().parents[1]
DATA_ADV     = ROOT / "data" / "malaria_advanced"
DATA_NIH_INF = ROOT / "data" / "malaria" / "Parasitised"
DATA_NIH_UNI = ROOT / "data" / "malaria" / "Uninfected"
MODEL_DIR    = ROOT / "models" / "deep_learning"

INFECTION_CLASSES = ["Parasitized", "Uninfected"]
SPECIES_CLASSES   = ["falciparum", "vivax", "malariae", "ovale", "knowlesi"]
STAGE_CLASSES     = ["ring", "trophozoite", "schizont", "gametocyte"]

IMG_SIZE        = 224
BATCH_SIZE      = 32
VALID_PCT       = 0.20
FREEZE_EPOCHS   = 5
UNFREEZE_EPOCHS = 20
BASE_LR         = 3e-3
BACKBONE_LR     = 1e-5
LABEL_SMOOTHING = 0.10
SEED            = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Datasets ──────────────────────────────────────────────────────────────────

class MalariaTaskDataset(Dataset):
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


def make_transforms(augment: bool = True):
    aug = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    ] if augment else [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    aug += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(aug)


def load_task_data(task: str, max_per_class: int = 5000) -> tuple | None:
    """
    Charge les donnees pour une tache.
    Retourne (t_paths, t_labels, v_paths, v_labels, n_classes) ou None.
    """
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    rng  = random.Random(SEED)

    if task == "infection":
        para_dir  = DATA_ADV / "nih" / "infected"
        uninf_dir = DATA_ADV / "nih" / "uninfected"
        # Fallback sur dossiers NIH originaux
        if not any(para_dir.glob("*")):
            para_dir  = DATA_NIH_INF
            uninf_dir = DATA_NIH_UNI
        para  = [p for p in sorted(para_dir.glob("*"))  if p.suffix.lower() in exts][:max_per_class]
        uninf = [p for p in sorted(uninf_dir.glob("*")) if p.suffix.lower() in exts][:max_per_class]
        if len(para) < 50 or len(uninf) < 50:
            print(f"  [!!] Donnees infection insuffisantes (para={len(para)}, uninf={len(uninf)})")
            return None
        all_paths  = para + uninf
        all_labels = [0] * len(para) + [1] * len(uninf)
        n_classes  = 2

    elif task == "species":
        all_paths, all_labels = [], []
        for i, sp in enumerate(SPECIES_CLASSES):
            sp_dir = DATA_ADV / "species" / sp
            imgs   = [p for p in sp_dir.glob("*") if p.suffix.lower() in exts][:max_per_class]
            if len(imgs) < 20:
                print(f"  [skip] {sp} : seulement {len(imgs)} images")
                continue
            all_paths.extend(imgs)
            all_labels.extend([i] * len(imgs))
            print(f"  {sp} : {len(imgs)} images")
        if len(all_paths) < 100:
            print("  [!!] Donnees species insuffisantes - lancer download_malaria_datasets.py")
            return None
        n_classes = len(SPECIES_CLASSES)

    elif task == "stage":
        all_paths, all_labels = [], []
        for i, st in enumerate(STAGE_CLASSES):
            st_dir = DATA_ADV / "stages" / st
            imgs   = [p for p in st_dir.glob("*") if p.suffix.lower() in exts][:max_per_class]
            if len(imgs) < 20:
                print(f"  [skip] {st} : seulement {len(imgs)} images")
                continue
            all_paths.extend(imgs)
            all_labels.extend([i] * len(imgs))
            print(f"  {st} : {len(imgs)} images")
        if len(all_paths) < 100:
            print("  [!!] Donnees stage insuffisantes - lancer download_malaria_datasets.py")
            return None
        n_classes = len(STAGE_CLASSES)

    else:
        return None

    # Split train/val
    combined = list(zip(all_paths, all_labels))
    rng.shuffle(combined)
    n_val    = max(1, int(len(combined) * VALID_PCT))
    val      = combined[:n_val]
    train    = combined[n_val:]
    rng.shuffle(train)

    t_p, t_l = zip(*train)
    v_p, v_l = zip(*val)
    return list(t_p), list(t_l), list(v_p), list(v_l), n_classes


# ── Modele multi-tache ────────────────────────────────────────────────────────

class MalariaMultiTask(nn.Module):
    """
    ResNet50 backbone + N tetes de classification.
    Chaque tete est un MLP : [2048 -> 512 -> n_classes].
    """

    def __init__(self, task_sizes: dict[str, int]):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Backbone = tout sauf la derniere couche FC
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # [B, 2048, 1, 1]

        self.heads = nn.ModuleDict()
        for task_name, n_classes in task_sizes.items():
            self.heads[task_name] = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, n_classes),
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x).flatten(1)  # [B, 2048]
        return {name: head(feat) for name, head in self.heads.items()}

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True


# ── Boucles d'entrainement ────────────────────────────────────────────────────

def train_epoch_multitask(
    model: MalariaMultiTask,
    loaders: dict[str, DataLoader],
    criterions: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
) -> dict[str, float]:
    model.train()
    task_losses: dict[str, list] = {t: [] for t in loaders}
    task_accs:   dict[str, list] = {t: [] for t in loaders}

    # Creer des iterateurs cycliques
    iters = {t: iter(dl) for t, dl in loaders.items()}
    max_batches = max(len(dl) for dl in loaders.values())

    for _ in range(max_batches):
        total_loss = torch.tensor(0.0, device=device)
        optimizer.zero_grad()

        for task, it in iters.items():
            try:
                imgs, labels = next(it)
            except StopIteration:
                iters[task] = iter(loaders[task])
                imgs, labels = next(iters[task])

            imgs, labels = imgs.to(device), labels.to(device)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(imgs)
                loss    = criterions[task](outputs[task], labels)

            total_loss = total_loss + loss
            preds = outputs[task].argmax(1)
            acc   = (preds == labels).float().mean().item()
            task_losses[task].append(loss.item())
            task_accs[task].append(acc)

        if device.type == "cuda":
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

    return {t: {"loss": np.mean(task_losses[t]), "acc": np.mean(task_accs[t])} for t in loaders}


@torch.no_grad()
def eval_epoch_multitask(
    model: MalariaMultiTask,
    loaders: dict[str, DataLoader],
    criterions: dict[str, nn.Module],
    device: torch.device,
) -> dict[str, dict]:
    model.eval()
    results = {}
    for task, loader in loaders.items():
        all_labels, all_preds = [], []
        total_loss = 0.0
        n = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss    = criterions[task](outputs[task], labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs[task].argmax(1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            n += imgs.size(0)
        results[task] = {
            "loss": total_loss / n,
            "acc":  accuracy_score(all_labels, all_preds),
            "f1":   f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        }
    return results


# ── Export ONNX multi-tache ───────────────────────────────────────────────────

def export_multitask_onnx(model: MalariaMultiTask, task_names: list[str], output_path: Path) -> bool:
    """Export ONNX avec toutes les tetes comme sorties nommees."""
    import os
    os.environ["PYTHONIOENCODING"] = "utf-8"
    model.cpu().eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    try:
        # Wrapper pour retourner un tuple (ONNX ne supporte pas les dicts)
        class ONNXWrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x):
                out = self.m(x)
                return tuple(out[t] for t in task_names)

        wrapper = ONNXWrapper(model)
        torch.onnx.export(
            wrapper, dummy, str(output_path),
            input_names=["input"],
            output_names=task_names,
            dynamic_axes={"input": {0: "batch"}, **{t: {0: "batch"} for t in task_names}},
            opset_version=17,
            dynamo=False,
        )
        print(f"[OK] ONNX multi-tache : {output_path}  ({output_path.stat().st_size // 1_000_000} MB)")
        return True
    except Exception as exc:
        print(f"[ERR] Export ONNX : {exc}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="infection",
                        help="Taches separees par virgules : infection,species,stage")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Nombre total d'epochs (override FREEZE+UNFREEZE)")
    parser.add_argument("--max-per-class", type=int, default=5000)
    args = parser.parse_args()

    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    print()
    print("=" * 60)
    print("  MalariaScan AI - Entrainement Multi-Tache ResNet50")
    print(f"  Device : {device}")
    print("=" * 60)

    task_list  = [t.strip() for t in args.tasks.split(",")]
    task_data: dict[str, tuple] = {}
    task_sizes: dict[str, int]  = {}

    for task in task_list:
        print(f"\n[*] Chargement donnees : {task}")
        data = load_task_data(task, args.max_per_class)
        if data is None:
            print(f"  [skip] {task} : donnees insuffisantes")
            continue
        task_data[task]  = data
        task_sizes[task] = data[4]
        t_paths, _, _, _, _ = data
        print(f"  Train : {len(data[0])} | Val : {len(data[2])} | Classes : {data[4]}")

    if not task_data:
        print("[ERR] Aucune tache disponible.")
        print("  Lancer : python scripts/download_malaria_datasets.py")
        return

    # Dataloaders
    train_loaders: dict[str, DataLoader] = {}
    val_loaders:   dict[str, DataLoader] = {}
    criterions:    dict[str, nn.Module]  = {}

    for task, (t_p, t_l, v_p, v_l, n_cls) in task_data.items():
        tf_tr = make_transforms(augment=True)
        tf_vl = make_transforms(augment=False)
        train_loaders[task] = DataLoader(
            MalariaTaskDataset(t_p, t_l, tf_tr), batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        val_loaders[task] = DataLoader(
            MalariaTaskDataset(v_p, v_l, tf_vl), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )
        criterions[task] = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Modele
    model = MalariaMultiTask(task_sizes).to(device)
    print(f"\n[*] Modele ResNet50 multi-tache : {list(task_sizes.keys())}")

    freeze_ep   = FREEZE_EPOCHS   if args.epochs is None else max(1, args.epochs // 4)
    unfreeze_ep = UNFREEZE_EPOCHS if args.epochs is None else args.epochs - freeze_ep
    best_loss   = float("inf")
    t0          = time.time()
    history     = []

    # Phase 1 : backbone gele
    print(f"\n[*] Phase 1 : backbone gele ({freeze_ep} epochs)")
    model.freeze_backbone()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=freeze_ep)

    for ep in range(freeze_ep):
        te = time.time()
        tr = train_epoch_multitask(model, train_loaders, criterions, optimizer, device, scaler)
        vl = eval_epoch_multitask(model, val_loaders, criterions, device)
        scheduler.step()

        avg_loss = np.mean([v["loss"] for v in vl.values()])
        metrics  = " | ".join(f"{t} acc={vl[t]['acc']:.4f}" for t in vl)
        print(f"  Epoch {ep+1:02d}/{freeze_ep}  {metrics}  ({time.time()-te:.0f}s)", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_DIR / "malaria_multitask.pth")
            print(f"    -> checkpoint sauve (loss={best_loss:.4f})", flush=True)

        entry = {"epoch": ep, "phase": "frozen", "val_avg_loss": round(avg_loss, 4)}
        entry.update({f"{t}_acc": round(vl[t]["acc"], 4) for t in vl})
        history.append(entry)

    # Phase 2 : backbone degele
    print(f"\n[*] Phase 2 : backbone degele ({unfreeze_ep} epochs)")
    model.unfreeze_backbone()
    optimizer = torch.optim.Adam([
        {"params": model.heads.parameters(),    "lr": BASE_LR},
        {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
    ])
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=unfreeze_ep)
    patience    = 0

    for ep in range(unfreeze_ep):
        te = time.time()
        tr = train_epoch_multitask(model, train_loaders, criterions, optimizer, device, scaler)
        vl = eval_epoch_multitask(model, val_loaders, criterions, device)
        scheduler.step()

        avg_loss = np.mean([v["loss"] for v in vl.values()])
        ep_num   = freeze_ep + ep
        metrics  = " | ".join(f"{t} acc={vl[t]['acc']:.4f}" for t in vl)
        print(f"  Epoch {ep_num+1:02d}/{freeze_ep+unfreeze_ep}  {metrics}  ({time.time()-te:.0f}s)", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience  = 0
            torch.save(model.state_dict(), MODEL_DIR / "malaria_multitask.pth")
            print(f"    -> checkpoint sauve (loss={best_loss:.4f})", flush=True)
        else:
            patience += 1
            if patience >= 5:
                print("  EarlyStopping declenche.", flush=True)
                break

        entry = {"epoch": ep_num, "phase": "unfrozen", "val_avg_loss": round(avg_loss, 4)}
        entry.update({f"{t}_acc": round(vl[t]["acc"], 4) for t in vl})
        history.append(entry)

    elapsed = time.time() - t0
    print(f"\n[*] Entrainement termine en {elapsed/60:.1f} min")

    # Evaluation finale
    model.load_state_dict(torch.load(MODEL_DIR / "malaria_multitask.pth", map_location=device))
    final = eval_epoch_multitask(model, val_loaders, criterions, device)

    print("\n[*] Metriques finales :")
    for t, m in final.items():
        print(f"  {t:15s} : accuracy={m['acc']:.4f}  f1={m['f1']:.4f}")

    # Export ONNX
    onnx_path = MODEL_DIR / "malaria_multitask.onnx"
    export_multitask_onnx(model, list(task_sizes.keys()), onnx_path)

    # Sauvegarde metriques
    metrics_out = ROOT / "scripts" / "malaria_multitask_metrics.json"
    metrics_out.write_text(json.dumps({
        "model":     "ResNet50_MultiTask",
        "tasks":     list(task_sizes.keys()),
        "epochs":    len(history),
        "best_loss": round(best_loss, 4),
        "time_min":  round(elapsed / 60, 1),
        "metrics":   {t: {k: round(v, 4) for k, v in m.items()} for t, m in final.items()},
        "history":   history,
        "onnx_path": str(onnx_path),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[OK] Metriques  : {metrics_out}")
    print(f"[OK] Checkpoint : {MODEL_DIR / 'malaria_multitask.pth'}")
    print(f"[OK] ONNX       : {onnx_path}")
    print("\nProchaine etape : git add models/ && git push")


if __name__ == "__main__":
    main()
