"""
KANEA AI — Évaluateur médical (grade clinique)
═══════════════════════════════════════════════
Métriques calculées pour chaque modèle :
  ▸ Accuracy, Precision, Recall (sensibilité), F1-score
  ▸ ROC-AUC (binaire et multi-classe)
  ▸ Matrice de confusion (PNG)
  ▸ Courbe ROC (PNG)
  ▸ Taux de faux négatifs (⚠️ priorité médicale)
  ▸ K-fold cross-validation (nutrition uniquement)

⚠️ Priorité médicale : MINIMISER les faux négatifs
   → un patient malade non détecté est plus grave qu'un faux positif.

Appel principal : evaluate_models()
Sorties         : reports/*.json + reports/*.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # pas d'interface graphique requise
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import label_binarize

from src.utils import KANEA_ROOT, get_logger, load_model_pkl, safe_run, timestamp

log = get_logger(__name__)

REPORTS_DIR = KANEA_ROOT / "reports"
OUT_ML      = KANEA_ROOT / "models" / "machine_learning"
OUT_DL      = KANEA_ROOT / "models" / "deep_learning"

# Seuil d'alerte taux de faux négatifs (médical)
FN_ALERT_THRESHOLD = 0.15


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — GRAPHIQUES
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_confusion_matrix(cm: np.ndarray, classes: list, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(classes)), yticks=range(len(classes)),
        xticklabels=classes, yticklabels=classes,
        xlabel="Prédit", ylabel="Réel",
        title=f"{title}\n⚠️ Priorité : minimiser faux négatifs (diagonale basse)",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=12)

    # Mettre en évidence faux négatifs (hors diagonale, même ligne)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=2, edgecolor="red", facecolor="none", alpha=0.6,
                ))

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Matrice de confusion → {path.name}")


def _plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    classes: list,
    title: str,
    path: Path,
) -> dict:
    """
    Trace la courbe ROC.
    Binaire  → une courbe.
    Multi    → une courbe par classe (OvR).
    Retourne : { classe: auc_value }
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    aucs: dict[str, float] = {}

    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc_val:.3f})")
        aucs[classes[1]] = round(roc_auc_val, 4)
    else:
        y_bin = label_binarize(y_true, classes=list(range(len(classes))))
        colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
        for i, (cls, color) in enumerate(zip(classes, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, color=color, label=f"{cls} (AUC = {roc_auc_val:.3f})")
            aucs[cls] = round(roc_auc_val, 4)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aléatoire (AUC=0.5)")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set(
        xlabel="Taux de Faux Positifs (1 - Spécificité)",
        ylabel="Taux de Vrais Positifs (Sensibilité / Recall)",
        title=title,
        xlim=(-0.02, 1.02), ylim=(-0.02, 1.02),
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Courbe ROC → {path.name}")
    return aucs


# ═══════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES MÉDICALES
# ═══════════════════════════════════════════════════════════════════════════════

def _medical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    classes: list,
    average: str = "weighted",
) -> dict:
    """
    Calcule toutes les métriques médicales.
    Priorité : Recall (sensibilité) → minimiser faux négatifs.
    """
    metrics: dict = {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, average=average, zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_true, y_pred, average=average, zero_division=0)), 4),
    }

    # ROC-AUC
    try:
        if len(classes) == 2:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_score[:, 1])), 4)
        else:
            metrics["roc_auc"] = round(float(
                roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")
            ), 4)
    except Exception as exc:
        log.warning(f"ROC-AUC non calculable : {exc}")
        metrics["roc_auc"] = None

    # ── Analyse des faux négatifs par classe (critique médical) ───────────────
    cm = confusion_matrix(y_true, y_pred)
    fn_analysis = []
    critical_classes = []

    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fn_rate = fn / max(tp + fn, 1)
        entry = {
            "class":              cls,
            "true_positives":     int(tp),
            "false_negatives":    int(fn),
            "false_negative_rate": round(fn_rate, 4),
            "sensitivity":        round(1 - fn_rate, 4),
            "medical_risk":       "ÉLEVÉ" if fn_rate > FN_ALERT_THRESHOLD else "ACCEPTABLE",
        }
        fn_analysis.append(entry)

        if fn_rate > FN_ALERT_THRESHOLD:
            critical_classes.append(cls)
            log.warning(
                f"⚠️ MÉDICAL : '{cls}' → FN rate = {fn_rate*100:.1f}% "
                f"({fn} patients malades non détectés)"
            )

    metrics["false_negative_analysis"] = fn_analysis
    metrics["critical_classes"]        = critical_classes

    # Rapport scikit-learn complet
    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# ÉVALUATION — MALARIA (EfficientNet-B0)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_malaria() -> dict:
    """Évalue le modèle MalariaScan AI sur un ensemble de test hors-entraînement."""
    log.info("═══ Évaluation Malaria ═══")

    model_path = OUT_DL / "malaria_model.pth"
    meta_path  = OUT_DL / "malaria_model.json"

    if not model_path.exists():
        log.error("Modèle malaria introuvable — lancez train_models() d'abord")
        return {"module": "malaria", "error": "Modèle non entraîné"}

    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader, random_split
    except ImportError as exc:
        return {"module": "malaria", "error": str(exc)}

    # Métadonnées du modèle
    classes = ["Parasitised", "Uninfected"]
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        classes = meta.get("classes", classes)

    # Dataset
    data_dir = next(
        (d for d in [
            KANEA_ROOT / "data" / "malaria_processed",
            KANEA_ROOT / "data" / "malaria",
        ] if d.exists() and any(d.iterdir())),
        None,
    )
    if not data_dir:
        return {"module": "malaria", "error": "Données introuvables"}

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    full_ds = ImageFolder(str(data_dir), transform=tf)
    val_size = max(10, int(0.2 * len(full_ds)))
    _, test_ds = random_split(
        full_ds, [len(full_ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # Chargement modèle en mode CPU (compatible offline)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, len(full_ds.classes))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            out   = model(imgs)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
            all_probs.extend(probs.numpy())

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_score = np.array(all_probs)
    classes = full_ds.classes

    metrics = _medical_metrics(y_true, y_pred, y_score, classes)

    # Plots
    _plot_confusion_matrix(
        confusion_matrix(y_true, y_pred), classes,
        "MalariaScan AI — Matrice de confusion",
        REPORTS_DIR / "malaria_confusion_matrix.png",
    )
    auc_per_class = _plot_roc_curves(
        y_true, y_score, classes,
        "MalariaScan AI — Courbe ROC (Recall vs FPR)",
        REPORTS_DIR / "malaria_roc_curve.png",
    )
    metrics["auc_per_class"] = auc_per_class

    log.info(
        f"Malaria — Acc={metrics['accuracy']:.3f} | "
        f"Recall={metrics['recall']:.3f} | "
        f"F1={metrics['f1_score']:.3f} | "
        f"AUC={metrics.get('roc_auc', 'N/A')}"
    )
    return {
        "module":  "malaria",
        "n_test":  int(len(y_true)),
        "classes": list(classes),
        "metrics": metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ÉVALUATION — NUTRITION (RF + XGBoost + K-fold)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_nutrition(k_folds: int = 5) -> dict:
    """
    Évalue NutriTrack AI avec métriques médicales + cross-validation k=5.
    La CV confirme la robustesse du modèle (pas de sur-apprentissage).
    """
    log.info("═══ Évaluation Nutrition ═══")

    model_path = OUT_ML / "nutrition_model.pkl"
    if not model_path.exists():
        log.error("Modèle nutrition introuvable")
        return {"module": "nutrition", "error": "Modèle non entraîné"}

    artifact = load_model_pkl(model_path)
    if not isinstance(artifact, dict):
        return {"module": "nutrition", "error": "Format modèle invalide"}

    ensemble = artifact["ensemble"]
    le       = artifact["label_encoder"]
    features = artifact["features"]

    # Dataset
    csv_path = next(
        (p for p in [
            KANEA_ROOT / "data" / "nutrition" / "nutrition_processed.csv",
            KANEA_ROOT / "data" / "nutrition" / "nutrition_dataset.csv",
        ] if p.exists()),
        None,
    )
    if not csv_path:
        return {"module": "nutrition", "error": "Dataset introuvable"}

    df = pd.read_csv(csv_path)
    if "sex_encoded" not in df.columns and "sex" in df.columns:
        df["sex_encoded"] = (df["sex"].str.strip().str.upper() == "M").astype(int)

    features = [f for f in features if f in df.columns]
    df = df.dropna(subset=["nutrition_status"] + features)

    X = df[features].values
    y = le.transform(df["nutrition_status"].values)
    classes = list(le.classes_)

    # Split test reproductible (même seed que training)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred  = ensemble.predict(X_test)
    y_score = ensemble.predict_proba(X_test)

    metrics = _medical_metrics(y_test, y_pred, y_score, classes)

    # ── K-Fold Cross-Validation k=5 ───────────────────────────────────────────
    log.info(f"Cross-validation StratifiedKFold k={k_folds} …")
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    cv  = cross_validate(
        ensemble, X, y, cv=skf, n_jobs=-1,
        scoring=["accuracy", "f1_weighted", "recall_weighted", "precision_weighted"],
    )
    metrics["cross_validation"] = {
        "k":              k_folds,
        "accuracy_mean":  round(float(cv["test_accuracy"].mean()), 4),
        "accuracy_std":   round(float(cv["test_accuracy"].std()), 4),
        "recall_mean":    round(float(cv["test_recall_weighted"].mean()), 4),
        "recall_std":     round(float(cv["test_recall_weighted"].std()), 4),
        "f1_mean":        round(float(cv["test_f1_weighted"].mean()), 4),
        "f1_std":         round(float(cv["test_f1_weighted"].std()), 4),
        "precision_mean": round(float(cv["test_precision_weighted"].mean()), 4),
        "precision_std":  round(float(cv["test_precision_weighted"].std()), 4),
        "interpretation": (
            "Modèle robuste (faible variance)"
            if cv["test_accuracy"].std() < 0.03
            else "⚠️ Variance élevée — risque sur-apprentissage"
        ),
    }
    log.info(
        f"  CV k={k_folds} — "
        f"Acc={metrics['cross_validation']['accuracy_mean']:.3f}"
        f"±{metrics['cross_validation']['accuracy_std']:.3f} | "
        f"Recall={metrics['cross_validation']['recall_mean']:.3f}"
        f"±{metrics['cross_validation']['recall_std']:.3f}"
    )

    # Plots
    _plot_confusion_matrix(
        confusion_matrix(y_test, y_pred), classes,
        "NutriTrack AI — Matrice de confusion",
        REPORTS_DIR / "nutrition_confusion_matrix.png",
    )
    auc_per_class = _plot_roc_curves(
        y_test, y_score, classes,
        "NutriTrack AI — Courbe ROC (Recall vs FPR)",
        REPORTS_DIR / "nutrition_roc_curve.png",
    )
    metrics["auc_per_class"] = auc_per_class

    log.info(
        f"Nutrition — Acc={metrics['accuracy']:.3f} | "
        f"Recall={metrics['recall']:.3f} | "
        f"F1={metrics['f1_score']:.3f} | "
        f"AUC={metrics.get('roc_auc', 'N/A')}"
    )
    return {
        "module":   "nutrition",
        "n_test":   int(len(y_test)),
        "classes":  classes,
        "features": features,
        "metrics":  metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ÉVALUATION — CANCER DU SEIN (EfficientNet-B0, 3 classes)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_breast_cancer() -> dict:
    """Évalue Breast Cancer AI avec métriques médicales. Priorité : Malignant recall."""
    log.info("═══ Évaluation Cancer du sein ═══")

    model_path = OUT_DL / "breast_cancer_model.pth"
    meta_path  = OUT_DL / "breast_cancer_model.json"

    if not model_path.exists():
        log.error("Modèle breast_cancer introuvable")
        return {"module": "breast_cancer", "error": "Modèle non entraîné"}

    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader, random_split
    except ImportError as exc:
        return {"module": "breast_cancer", "error": str(exc)}

    classes = ["Benign", "Malignant", "Normal"]
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        classes = meta.get("classes", classes)

    data_dir = next(
        (d for d in [
            KANEA_ROOT / "data" / "breast_cancer_processed",
            KANEA_ROOT / "data" / "breast_cancer",
        ] if d.exists() and any(d.iterdir())),
        None,
    )
    if not data_dir:
        return {"module": "breast_cancer", "error": "Données introuvables"}

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    full_ds = ImageFolder(str(data_dir), transform=tf)
    val_size = max(10, int(0.2 * len(full_ds)))
    _, test_ds = random_split(
        full_ds, [len(full_ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    test_dl = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, len(full_ds.classes))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            out   = model(imgs)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
            all_probs.extend(probs.numpy())

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_score = np.array(all_probs)
    classes = full_ds.classes

    metrics = _medical_metrics(y_true, y_pred, y_score, classes)

    # Alerte spécifique cancer : recall Malignant critique
    mal_idx = list(classes).index("Malignant") if "Malignant" in classes else None
    if mal_idx is not None:
        mal_fn = next(
            (x for x in metrics["false_negative_analysis"] if x["class"] == "Malignant"), None
        )
        if mal_fn and mal_fn["false_negative_rate"] > 0.10:
            log.warning(
                f"⚠️ CRITIQUE MÉDICAL : Malignant FN = {mal_fn['false_negative_rate']*100:.1f}% — "
                f"Augmenter le poids de classe Malignant ou abaisser le seuil de décision"
            )

    _plot_confusion_matrix(
        confusion_matrix(y_true, y_pred), classes,
        "Breast Cancer AI — Matrice de confusion",
        REPORTS_DIR / "breast_cancer_confusion_matrix.png",
    )
    auc_per_class = _plot_roc_curves(
        y_true, y_score, classes,
        "Breast Cancer AI — Courbe ROC (Recall vs FPR)",
        REPORTS_DIR / "breast_cancer_roc_curve.png",
    )
    metrics["auc_per_class"] = auc_per_class

    log.info(
        f"Cancer sein — Acc={metrics['accuracy']:.3f} | "
        f"Recall={metrics['recall']:.3f} | "
        f"F1={metrics['f1_score']:.3f} | "
        f"AUC={metrics.get('roc_auc', 'N/A')}"
    )
    return {
        "module":  "breast_cancer",
        "n_test":  int(len(y_true)),
        "classes": list(classes),
        "metrics": metrics,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RAPPORT GLOBAL + DASHBOARD MÉDICAL
# ═══════════════════════════════════════════════════════════════════════════════

def _print_medical_dashboard(results: dict) -> None:
    """Affiche un tableau de bord médical lisible dans les logs."""
    log.info("╔══════════════════════════════════════════════════════════════════╗")
    log.info("║   KANEA AI — TABLEAU DE BORD MÉDICAL                            ║")
    log.info("╠══════════════════════════════════════════════════════════════════╣")
    log.info("║  Module          Acc    Recall   F1     AUC    Risque FN         ║")
    log.info("╠══════════════════════════════════════════════════════════════════╣")

    for module in ["malaria", "nutrition", "breast_cancer"]:
        r = results.get(module, {})
        if "error" in r:
            log.info(f"║  {module:15s}  [ERREUR : {r['error'][:35]}] ║")
            continue
        m = r.get("metrics", {})
        acc    = m.get("accuracy", 0)
        recall = m.get("recall", 0)
        f1     = m.get("f1_score", 0)
        auc_v  = m.get("roc_auc") or 0

        critical = m.get("critical_classes", [])
        risk = f"⚠️ {critical[0]}" if critical else "✓ OK"

        log.info(
            f"║  {module:15s}  "
            f"{acc:.3f}  {recall:.3f}    {f1:.3f}  {auc_v:.3f}  {risk:20s} ║"
        )

    log.info("╚══════════════════════════════════════════════════════════════════╝")
    log.info("Priorité médicale : Recall élevé = moins de malades non détectés")


# ═══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

@safe_run
def evaluate_models(k_folds: int = 5) -> dict:
    """
    Évalue les 3 modèles KANEA avec métriques médicales complètes.

    Sorties :
        ▸ reports/evaluation_report.json
        ▸ reports/*_confusion_matrix.png  (avec faux négatifs encadrés en rouge)
        ▸ reports/*_roc_curve.png         (AUC par classe)

    Args:
        k_folds: nombre de folds pour la CV du modèle nutrition (défaut=5).

    Returns:
        dict complet avec métriques par module.
    """
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   KANEA AI — ÉVALUATION MÉDICALE                     ║")
    log.info(f"║   {timestamp()}                              ║")
    log.info("║   Priorité : minimiser FAUX NÉGATIFS (Recall ↑)      ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "malaria":          evaluate_malaria(),
        "nutrition":        evaluate_nutrition(k_folds=k_folds),
        "breast_cancer":    evaluate_breast_cancer(),
        "evaluated_at":     timestamp(),
        "medical_priority": "Minimiser faux négatifs — recall > precision",
        "k_folds":          k_folds,
    }

    _print_medical_dashboard(results)

    # Sauvegarde JSON
    report_path = REPORTS_DIR / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    log.info(f"Rapport complet : {report_path}")
    log.info(f"Graphiques      : {REPORTS_DIR}/*.png")

    return results
