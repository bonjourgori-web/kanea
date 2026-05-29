# -*- coding: utf-8 -*-
"""
NutriTrack AI v3.0 - Pipeline d'entrainement complet.

Modeles : RandomForest + XGBoost + LightGBM -> VotingClassifier (soft)
Dataset : data/biometry/biometry_dataset.csv
Output  : models/machine_learning/nutrition_model.pkl

Usage :
    cd c:/Users/HP/gori/KANEA
    python scripts/train_nutritrack.py [--csv data/biometry/biometry_dataset.csv]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    recall_score, precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    _LGBM_OK = True
except ImportError:
    _LGBM_OK = False
    print("[warn] lightgbm non installe — VotingClassifier sans LightGBM")

from modules.nutrition_ml.features import build_feature_payload

FEATURE_COLS = [
    "age_months", "weight_kg", "height_cm", "muac_cm",
    "whz", "haz", "waz", "bmi", "sex_encoded",
]
TEST_SIZE     = 0.20
RANDOM_STATE  = 42
CV_FOLDS      = 5
MODEL_DIR     = PROJECT_ROOT / "models" / "machine_learning"
VIZ_DIR       = MODEL_DIR


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = [build_feature_payload(row.to_dict()) for _, row in df.iterrows()]
    return pd.DataFrame(rows)[FEATURE_COLS].fillna(0.0)


def build_estimators(random_state: int) -> list:
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=2,
        class_weight="balanced", random_state=random_state, n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        objective="multi:softprob", eval_metric="mlogloss",
        use_label_encoder=False, random_state=random_state,
    )
    estimators = [("rf", rf), ("xgb", xgb)]
    if _LGBM_OK:
        lgbm = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            class_weight="balanced", random_state=random_state, verbose=-1,
        )
        estimators.append(("lgbm", lgbm))
    return estimators


def generate_visualizations(
    voting: VotingClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    classes: list[str],
) -> None:
    """Génère les visualisations de performance et SHAP."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Matrice de confusion
    from sklearn.metrics import confusion_matrix
    y_pred = voting.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax)
    ticks = range(len(classes))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                    fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "#1A2B3C")
    ax.set_xlabel("Prédiction", fontsize=10); ax.set_ylabel("Réel", fontsize=10)
    ax.set_title(
        f"Matrice de confusion — acc={accuracy_score(y_test,y_pred):.1%}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(str(VIZ_DIR / "nutrition_confusion_matrix.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [OK] nutrition_confusion_matrix.png")

    # 2. Feature importance RF
    rf = dict(voting.estimators)["rf"]
    imp = rf.feature_importances_
    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(idx)))
    ax.barh([FEATURE_COLS[i] for i in idx], imp[idx], color=colors, edgecolor="white")
    ax.set_xlabel("Importance (RF)", fontsize=10)
    ax.set_title("NutriTrack AI — Feature Importance", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(str(VIZ_DIR / "nutrition_feature_importance.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  [OK] nutrition_feature_importance.png")

    # 3. SHAP summary (100 samples)
    try:
        import shap
        X_sample = X_test[:100] if len(X_test) >= 100 else X_test
        explainer = shap.TreeExplainer(rf)
        sv        = explainer.shap_values(X_sample)
        sv_mean   = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)
        idx2      = np.argsort(sv_mean)
        fig, ax   = plt.subplots(figsize=(6, 4))
        colors2   = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(idx2)))
        ax.barh([FEATURE_COLS[i] for i in idx2], sv_mean[idx2], color=colors2, edgecolor="white")
        ax.set_xlabel("|SHAP value| moyen", fontsize=10)
        ax.set_title("NutriTrack AI — SHAP Summary", fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        plt.savefig(str(VIZ_DIR / "nutrition_shap_summary.png"), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print("  [OK] nutrition_shap_summary.png")
    except Exception as exc:
        print(f"  [skip] SHAP summary : {exc}")


def train(csv_path: Path, export_path: Path, random_state: int) -> None:
    t0 = time.time()
    print(f"\n{'='*55}")
    print("  NutriTrack AI v3.0 — Entrainement")
    print(f"{'='*55}\n")

    # Données
    print(f"[*] Chargement : {csv_path}")
    df = pd.read_csv(csv_path)
    if "nutrition_status" not in df.columns:
        raise ValueError("Colonne 'nutrition_status' manquante dans le CSV")

    X      = prepare_features(df)
    labels = df["nutrition_status"].astype(str)
    le     = LabelEncoder()
    y      = le.fit_transform(labels)
    classes = list(le.classes_)

    print(f"  Echantillons : {len(X)} | Classes : {classes}")
    print(f"  Distribution :")
    for cls, cnt in zip(*np.unique(y, return_counts=True)):
        print(f"    {le.inverse_transform([cls])[0]:12s} : {cnt}")

    # Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state, stratify=y,
    )
    print(f"\n  Train : {len(X_train)} | Test : {len(X_test)}")

    # Modèles
    estimators = build_estimators(random_state)
    names = [n for n, _ in estimators]
    print(f"\n[*] Estimateurs : {names}")

    # Cross-validation individuelle
    for name, est in estimators:
        cv_scores = cross_val_score(est, X_train, y_train, cv=CV_FOLDS, scoring="f1_weighted")
        print(f"  {name:6s} CV F1 : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Entraînement individuel
    print(f"\n[*] Entrainement des {len(estimators)} estimateurs...")
    for name, est in estimators:
        est.fit(X_train, y_train)
        pred  = est.predict(X_test)
        acc   = accuracy_score(y_test, pred)
        f1    = f1_score(y_test, pred, average="weighted")
        print(f"  {name:6s} — acc={acc:.4f}  f1={f1:.4f}")

    # VotingClassifier soft
    print("\n[*] Assemblage VotingClassifier (soft voting)...")
    voting = VotingClassifier(estimators=estimators, voting="soft")
    voting.fit(X_train, y_train)
    y_pred_v = voting.predict(X_test)
    acc_v    = accuracy_score(y_test, y_pred_v)
    f1_v     = f1_score(y_test, y_pred_v, average="weighted")
    print(f"  VotingClassifier — acc={acc_v:.4f}  f1={f1_v:.4f}")
    print(f"\n{classification_report(y_test, y_pred_v, target_names=classes, zero_division=0)}")

    # Visualisations
    print("[*] Generation des visualisations...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    generate_visualizations(voting, X_test.values, y_test, le, classes)

    # Sauvegarde bundle
    bundle = {
        "ensemble":      voting,
        "label_encoder": le,
        "features":      FEATURE_COLS,
        "classes":       classes,
        "model_version": "v3.0",
        "n_estimators":  names,
        "metrics": {
            "accuracy": round(acc_v, 4),
            "f1_score": round(f1_v, 4),
        },
    }
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("wb") as fh:
        pickle.dump(bundle, fh)
    print(f"\n[OK] Modele sauvegarde : {export_path}")

    # Métriques JSON
    metrics_out = PROJECT_ROOT / "scripts" / "nutritrack_metrics.json"
    metrics_out.write_text(json.dumps({
        "model": "NutriTrack_AI_VotingClassifier",
        "version": "v3.0",
        "estimators": names,
        "classes": classes,
        "accuracy": round(acc_v, 4),
        "f1_weighted": round(f1_v, 4),
        "recall_weighted": round(recall_score(y_test, y_pred_v, average="weighted"), 4),
        "precision_weighted": round(precision_score(y_test, y_pred_v, average="weighted", zero_division=0), 4),
        "training_time_s": round(time.time() - t0, 1),
        "n_samples": len(X),
        "test_size": TEST_SIZE,
    }, indent=2), encoding="utf-8")
    print(f"[OK] Metriques JSON : {metrics_out}")
    print(f"\nDuree totale : {(time.time()-t0)/60:.1f} min")
    print("\nProchaine etape :")
    print("  git add models/machine_learning/nutrition_model.pkl && git push")


def main() -> None:
    parser = argparse.ArgumentParser(description="NutriTrack AI v3.0 — Entrainement")
    parser.add_argument("--csv",    type=Path, default=Path("data/biometry/biometry_dataset.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/machine_learning/nutrition_model.pkl"))
    parser.add_argument("--seed",   type=int, default=RANDOM_STATE)
    args = parser.parse_args()
    train(args.csv, args.output, args.seed)


if __name__ == "__main__":
    main()
