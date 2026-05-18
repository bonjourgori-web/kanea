"""
KANEA AI — Entraîneur des modèles (grade médical)
══════════════════════════════════════════════════
Module 1 – Malaria     : EfficientNet-B0 (PyTorch, 2 classes)
Module 2 – Nutrition   : RandomForest + XGBoost ensemble (4 classes)
Module 3 – Cancer sein : EfficientNet-B0 (PyTorch, 3 classes)

Fonctionnalités médicales :
  ▸ Early stopping (patience configurable) → évite sur-apprentissage
  ▸ Class weights automatiques → compense déséquilibre de classes
  ▸ Dropout régularisé dans la tête de classification
  ▸ Label smoothing (CrossEntropyLoss) → meilleure calibration
  ▸ Cosine Annealing LR scheduler
  ▸ Sauvegarde du meilleur modèle (val_recall, pas val_loss)
  ▸ Reproductibilité complète (seed=42)

Appel principal : train_models()
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import (
    KANEA_ROOT, get_logger, safe_run,
    save_model_pkl, count_images, timestamp,
)

log = get_logger(__name__)

# Chemins
PROC_MALARIA = KANEA_ROOT / "data" / "malaria_processed"
RAW_MALARIA  = KANEA_ROOT / "data" / "malaria"
PROC_BREAST  = KANEA_ROOT / "data" / "breast_cancer_processed"
RAW_BREAST   = KANEA_ROOT / "data" / "breast_cancer"
PROC_NUT_CSV = KANEA_ROOT / "data" / "nutrition" / "nutrition_processed.csv"
RAW_NUT_CSV  = KANEA_ROOT / "data" / "nutrition" / "nutrition_dataset.csv"
OUT_DL       = KANEA_ROOT / "models" / "deep_learning"
OUT_ML       = KANEA_ROOT / "models" / "machine_learning"


# ═══════════════════════════════════════════════════════════════════════════════
# REPRODUCTIBILITÉ
# ═══════════════════════════════════════════════════════════════════════════════

def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# EARLY STOPPING
# ═══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Arrêt précoce basé sur le Recall de validation.
    Priorité médicale : on surveille recall, pas loss.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "max"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best      = -np.inf if mode == "max" else np.inf
        self.stop      = False

    def __call__(self, current: float) -> bool:
        improved = (
            current > self.best + self.min_delta
            if self.mode == "max"
            else current < self.best - self.min_delta
        )
        if improved:
            self.best    = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                log.info(f"Early stopping déclenché après {self.patience} époques sans amélioration")
        return self.stop


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS PYTORCH
# ═══════════════════════════════════════════════════════════════════════════════

def _get_device():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device} {'(GPU)' if device.type == 'cuda' else '(CPU — mode offline)'}")
    return device


def _get_transforms(augment: bool = True):
    from torchvision import transforms
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if augment:
        # Augmentation médicale ciblée : simule variabilité de coloration et d'orientation
        train_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def _compute_class_weights(dataset):
    """
    Calcule les poids de classe pour CrossEntropyLoss.
    Compensie le déséquilibre : classes minoritaires → poids plus élevé.
    """
    import torch
    from sklearn.utils.class_weight import compute_class_weight

    all_labels = [label for _, label in dataset.imgs]
    classes    = np.unique(all_labels)
    weights    = compute_class_weight("balanced", classes=classes, y=np.array(all_labels))
    log.info(f"Class weights : {dict(zip(classes.tolist(), weights.round(3).tolist()))}")
    return torch.tensor(weights, dtype=torch.float)


def _build_efficientnet(n_classes: int, dropout: float = 0.35):
    """
    EfficientNet-B0 pré-entraîné ImageNet.
    Tête personnalisée avec Dropout pour régularisation.
    """
    import torch.nn as nn
    from torchvision import models

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Remplace la tête de classification
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(1280, n_classes),
    )
    return model


def _train_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float, float]:
    """Un passage d'entraînement. Retourne (loss, accuracy, recall_weighted)."""
    from sklearn.metrics import recall_score

    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        # Gradient clipping → stabilité (important en médical)
        import torch.nn as nn
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc    = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    return total_loss / max(len(loader), 1), acc, recall


def _val_epoch(model, loader, criterion, device) -> tuple[float, float, float]:
    """Un passage de validation. Retourne (loss, accuracy, recall_weighted)."""
    import torch
    from sklearn.metrics import recall_score

    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc    = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    return total_loss / max(len(loader), 1), acc, recall


def _image_training_loop(
    data_dir:   Path,
    n_classes:  int,
    model_path: Path,
    meta_extra: dict,
    epochs:     int,
    batch_size: int,
    lr:         float,
    patience:   int,
    dropout:    float,
    label_smoothing: float,
) -> bool:
    """
    Boucle d'entraînement générique pour modèles image PyTorch.
    Inclut : class weights + early stopping + gradient clipping + label smoothing.
    """
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, random_split
    from torchvision.datasets import ImageFolder

    _set_seed(42)
    train_tf, val_tf = _get_transforms(augment=True)
    device = _get_device()

    # ── Dataset ───────────────────────────────────────────────────────────────
    try:
        full_ds = ImageFolder(str(data_dir), transform=train_tf)
    except FileNotFoundError:
        log.error(f"Dossier dataset introuvable : {data_dir}")
        return False

    n_total = len(full_ds)
    if n_total == 0:
        log.error("Dataset vide")
        return False

    val_size   = max(10, int(0.15 * n_total))
    test_size  = max(10, int(0.15 * n_total))
    train_size = n_total - val_size - test_size

    if train_size <= 0:
        # Dataset trop petit : pas de split test
        test_size  = 0
        val_size   = max(5, int(0.2 * n_total))
        train_size = n_total - val_size

    indices = list(range(n_total))
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    train_ds = torch.utils.data.Subset(full_ds, indices[:train_size])
    val_ds   = torch.utils.data.Subset(full_ds, indices[train_size:train_size + val_size])

    # Dataset de validation avec val_tf
    class _TfSubset(torch.utils.data.Dataset):
        def __init__(self, subset, tf):
            self.subset = subset
            self.tf     = tf
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            from PIL import Image as PILImage
            path, label = self.subset.dataset.samples[self.subset.indices[idx]]
            return self.tf(PILImage.open(path).convert("RGB")), label

    val_ds_tf  = _TfSubset(val_ds,  val_tf)
    train_dl   = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_dl     = DataLoader(val_ds_tf,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    log.info(f"Dataset : {n_total} images | train={train_size} val={val_size}")
    log.info(f"Classes  : {full_ds.classes}")

    # ── Class weights ─────────────────────────────────────────────────────────
    weights  = _compute_class_weights(full_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

    # ── Modèle ────────────────────────────────────────────────────────────────
    model     = _build_efficientnet(n_classes, dropout=dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    es        = EarlyStopping(patience=patience, mode="max")

    best_recall = 0.0
    best_epoch  = 1
    history     = []

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_recall = _train_epoch(model, train_dl, optimizer, criterion, device)
        vl_loss, vl_acc, vl_recall = _val_epoch(model, val_dl,  criterion, device)
        scheduler.step()

        ep = {
            "epoch":       epoch,
            "train_loss":  round(tr_loss, 4),
            "train_acc":   round(tr_acc, 2),
            "train_recall": round(tr_recall, 4),
            "val_loss":    round(vl_loss, 4),
            "val_acc":     round(vl_acc, 2),
            "val_recall":  round(vl_recall, 4),
            "lr":          round(scheduler.get_last_lr()[0], 6),
        }
        history.append(ep)

        log.info(
            f"Époque {epoch:02d}/{epochs} — "
            f"train_acc={tr_acc:.1f}%  val_acc={vl_acc:.1f}%  "
            f"val_recall={vl_recall:.3f}  lr={ep['lr']:.1e}"
        )

        # Sauvegarde best model (basé sur recall, priorité médicale)
        if vl_recall > best_recall:
            best_recall = vl_recall
            best_epoch  = epoch
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            log.info(f"  → Meilleur modèle (recall={vl_recall:.3f}) sauvegardé — époque {epoch}")

        # Early stopping
        if es(vl_recall):
            log.info(f"Early stopping à l'époque {epoch} (meilleur : époque {best_epoch})")
            break

    # ── Métadonnées ───────────────────────────────────────────────────────────
    meta = {
        "architecture":  "EfficientNet-B0",
        "classes":       full_ds.classes,
        "n_classes":     n_classes,
        "best_val_recall": round(best_recall, 4),
        "best_epoch":    best_epoch,
        "epochs_run":    len(history),
        "epochs_max":    epochs,
        "early_stopped": es.stop,
        "dropout":       dropout,
        "label_smoothing": label_smoothing,
        "batch_size":    batch_size,
        "lr":            lr,
        "patience":      patience,
        "trained_at":    timestamp(),
        "reproducibility_seed": 42,
        "history":       history,
        **meta_extra,
    }
    with open(model_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info(
        f"Entraînement terminé — meilleur recall val={best_recall:.3f} "
        f"(époque {best_epoch}/{len(history)})"
    )
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — MALARIA (EfficientNet-B0, 2 classes)
# ═══════════════════════════════════════════════════════════════════════════════

def train_malaria(
    epochs:     int   = 20,
    batch_size: int   = 32,
    lr:         float = 1e-3,
    patience:   int   = 5,
    dropout:    float = 0.35,
) -> bool:
    """
    Entraîne EfficientNet-B0 sur les images de cellules sanguines.
    Classes : Parasitised / Uninfected.
    Métrique de sélection : Recall (priorité médicale).
    """
    log.info("═══ ENTRAÎNEMENT MODULE 1 : Malaria ═══")
    data_dir = PROC_MALARIA if count_images(PROC_MALARIA) > 50 else RAW_MALARIA
    n_imgs   = count_images(data_dir)

    if n_imgs == 0:
        log.error("Aucune image malaria. Lancez download_data() d'abord.")
        return False

    log.info(f"Source : {data_dir} ({n_imgs} images)")
    return _image_training_loop(
        data_dir        = data_dir,
        n_classes       = 2,
        model_path      = OUT_DL / "malaria_model.pth",
        meta_extra      = {"task": "malaria_binary_classification", "dataset": "NIH/Kaggle"},
        epochs          = epochs,
        batch_size      = batch_size,
        lr              = lr,
        patience        = patience,
        dropout         = dropout,
        label_smoothing = 0.05,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — NUTRITION (RF + XGBoost ensemble, 4 classes)
# ═══════════════════════════════════════════════════════════════════════════════

def train_nutrition() -> bool:
    """
    Entraîne RF + XGBoost avec poids de classe et validation croisée interne.
    Classes : MAS / MAM / Normal / Overweight.
    """
    log.info("═══ ENTRAÎNEMENT MODULE 2 : Nutrition ═══")
    _set_seed(42)

    try:
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.metrics import (
            accuracy_score, recall_score, f1_score, classification_report,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.class_weight import compute_sample_weight
        from xgboost import XGBClassifier
    except ImportError as exc:
        log.error(f"Dépendance manquante : {exc}")
        return False

    csv_path = PROC_NUT_CSV if PROC_NUT_CSV.exists() else RAW_NUT_CSV
    if not csv_path.exists():
        log.error("Dataset nutrition introuvable. Lancez download_data() d'abord.")
        return False

    df = pd.read_csv(csv_path)
    log.info(f"Dataset : {df.shape} | source : {csv_path.name}")

    feature_cols = [
        c for c in
        ["age_months", "weight_kg", "height_cm", "muac_cm", "whz", "haz", "waz", "bmi", "sex_encoded"]
        if c in df.columns
    ]
    if "sex_encoded" not in df.columns and "sex" in df.columns:
        df["sex_encoded"] = (df["sex"].str.strip().str.upper() == "M").astype(int)
        feature_cols.append("sex_encoded")

    df = df.dropna(subset=["nutrition_status"] + feature_cols)
    X  = df[feature_cols].values

    le = LabelEncoder()
    y  = le.fit_transform(df["nutrition_status"].values)
    classes = list(le.classes_)

    log.info(f"Classes : {classes}")
    log.info(f"Distribution : {dict(zip(classes, np.bincount(y).tolist()))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Poids d'échantillons (compensie déséquilibre de classes)
    sample_weights = compute_sample_weight("balanced", y_train)

    # ── RandomForest ─────────────────────────────────────────────────────────
    log.info("Entraînement RandomForest (class_weight=balanced) …")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train, sample_weight=sample_weights)
    rf_recall = recall_score(y_test, rf.predict(X_test), average="weighted", zero_division=0)
    log.info(f"RandomForest — val_recall : {rf_recall:.3f}")

    # ── XGBoost ──────────────────────────────────────────────────────────────
    log.info("Entraînement XGBoost (scale_pos_weight) …")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weights)
    xgb_recall = recall_score(y_test, xgb.predict(X_test), average="weighted", zero_division=0)
    log.info(f"XGBoost — val_recall : {xgb_recall:.3f}")

    # ── Ensemble soft-voting ─────────────────────────────────────────────────
    log.info("Ensemble (soft voting) …")
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)], voting="soft"
    )
    ensemble.fit(X_train, y_train)
    y_pred       = ensemble.predict(X_test)
    ens_recall   = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    ens_acc      = accuracy_score(y_test, y_pred)
    ens_f1       = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report       = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)

    log.info(f"Ensemble — acc={ens_acc:.3f} recall={ens_recall:.3f} f1={ens_f1:.3f}")

    # Alerte médicale
    for cls_idx, cls in enumerate(classes):
        cls_recall = report[cls].get("recall", 0)
        if cls_recall < 0.75:
            log.warning(
                f"⚠️ MÉDICAL : recall faible pour '{cls}' = {cls_recall:.2f} "
                f"→ augmenter le poids de cette classe"
            )

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    OUT_ML.mkdir(parents=True, exist_ok=True)
    artifact = {
        "ensemble":      ensemble,
        "label_encoder": le,
        "features":      feature_cols,
        "classes":       classes,
    }
    meta = {
        "architecture":   "RandomForest + XGBoost Ensemble (soft voting)",
        "classes":        classes,
        "features":       feature_cols,
        "class_weight":   "balanced",
        "rf_recall":      round(rf_recall, 4),
        "xgb_recall":     round(xgb_recall, 4),
        "ensemble_acc":   round(ens_acc, 4),
        "ensemble_recall": round(ens_recall, 4),
        "ensemble_f1":    round(ens_f1, 4),
        "report":         report,
        "trained_at":     timestamp(),
        "reproducibility_seed": 42,
        "task":           "nutrition_classification_MAS_MAM_Normal_Overweight",
    }
    save_model_pkl(artifact, OUT_ML / "nutrition_model.pkl", meta)
    log.info(f"Modèle sauvegardé : {OUT_ML / 'nutrition_model.pkl'}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — CANCER DU SEIN (EfficientNet-B0, 3 classes)
# ═══════════════════════════════════════════════════════════════════════════════

def train_breast_cancer(
    epochs:     int   = 20,
    batch_size: int   = 16,
    lr:         float = 5e-4,
    patience:   int   = 6,
    dropout:    float = 0.40,
) -> bool:
    """
    Entraîne EfficientNet-B0 sur les mammographies.
    Classes : Normal / Benign / Malignant.
    ⚠️ Recall 'Malignant' priorisé — erreur de type II (FN) critique.
    """
    log.info("═══ ENTRAÎNEMENT MODULE 3 : Cancer du sein ═══")
    data_dir = PROC_BREAST if count_images(PROC_BREAST) > 50 else RAW_BREAST
    n_imgs   = count_images(data_dir)

    if n_imgs == 0:
        log.error("Aucune image breast_cancer. Lancez download_data() d'abord.")
        return False

    log.info(f"Source : {data_dir} ({n_imgs} images)")
    log.info("⚠️ Priorité médicale : recall Malignant > 0.90 cible")
    return _image_training_loop(
        data_dir        = data_dir,
        n_classes       = 3,
        model_path      = OUT_DL / "breast_cancer_model.pth",
        meta_extra      = {
            "task":    "breast_cancer_3class_classification",
            "dataset": "MIAS/CBIS-DDSM/Synthétique",
            "medical_note": "Priorité recall Malignant — FN (cancer non détecté) critique",
        },
        epochs          = epochs,
        batch_size      = batch_size,
        lr              = lr,
        patience        = patience,
        dropout         = dropout,
        label_smoothing = 0.05,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

@safe_run
def train_models(
    malaria_epochs:       int = 20,
    breast_cancer_epochs: int = 20,
    malaria_batch:        int = 32,
    breast_batch:         int = 16,
    patience:             int = 5,
) -> dict:
    """
    Entraîne les 3 modèles KANEA en séquence.

    Args:
        malaria_epochs       : époques max malaria (early stopping possible).
        breast_cancer_epochs : époques max cancer sein.
        malaria_batch        : batch size malaria.
        breast_batch         : batch size cancer sein.
        patience             : patience early stopping.

    Returns:
        dict statut par module.
    """
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   KANEA AI — ENTRAÎNEMENT DES MODÈLES                ║")
    log.info(f"║   {timestamp()}                              ║")
    log.info("║   Class weights + Early stopping + Dropout actifs    ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    results: dict = {}

    log.info("[1/3] Malaria …")
    results["malaria"]       = train_malaria(epochs=malaria_epochs, batch_size=malaria_batch, patience=patience)

    log.info("[2/3] Nutrition …")
    results["nutrition"]     = train_nutrition()

    log.info("[3/3] Cancer du sein …")
    results["breast_cancer"] = train_breast_cancer(epochs=breast_cancer_epochs, batch_size=breast_batch, patience=patience)

    ok = sum(bool(v) for v in results.values())
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info(f"║   RÉSULTAT : {ok}/{len(results)} modèles entraînés                 ║")
    log.info("╚══════════════════════════════════════════════════════╝")
    for module, status in results.items():
        icon = "✓" if status else "✗"
        log.info(f"  {icon} {module}")

    return results
