"""
CardioSense AI — Entraînement XGBoost sur MIT-BIH Arrhythmia
=============================================================
Dataset : MIT-BIH (mitbih_train.csv / mitbih_test.csv)
Classes : Normal · Supraventriculaire · Ventriculaire · Fusion · Inconnu
Export  : PKL + JSON metrics

Usage :
    python scripts/train_cardio.py --dataset data/cardio
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                              accuracy_score, f1_score)
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
import joblib

ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "machine_learning"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "Supraventriculaire", "Ventriculaire", "Fusion", "Inconnu"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/cardio")
    p.add_argument("--epochs",  type=int, default=300)
    return p.parse_args()


def load_mitbih(data_dir: Path):
    train_path = data_dir / "mitbih_train.csv"
    test_path  = data_dir / "mitbih_test.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {train_path}")
    df_train = pd.read_csv(train_path, header=None)
    df_test  = pd.read_csv(test_path,  header=None)
    X_train, y_train = df_train.iloc[:, :-1].values, df_train.iloc[:, -1].astype(int).values
    X_test,  y_test  = df_test.iloc[:,  :-1].values, df_test.iloc[:,  -1].astype(int).values
    print(f"Train : {X_train.shape}  Test : {X_test.shape}")
    print(f"Classes : {dict(zip(*np.unique(y_train, return_counts=True)))}")
    return X_train, y_train, X_test, y_test


def extract_features(X: np.ndarray) -> np.ndarray:
    """Extraction de features statistiques sur les 187 points ECG."""
    feats = np.column_stack([
        X.mean(axis=1), X.std(axis=1), X.min(axis=1), X.max(axis=1),
        np.percentile(X, 25, axis=1), np.percentile(X, 75, axis=1),
        X.max(axis=1) - X.min(axis=1),
        np.abs(np.diff(X, axis=1)).mean(axis=1),
        (X > X.mean(axis=1, keepdims=True)).sum(axis=1),
        X[:, :50].mean(axis=1), X[:, 50:100].mean(axis=1), X[:, 100:].mean(axis=1),
        X[:, X.shape[1]//2],
    ])
    return np.hstack([X, feats])


def train(args):
    t0 = time.time()
    data_dir = ROOT / args.dataset

    print("Chargement MIT-BIH...")
    X_train, y_train, X_test, y_test = load_mitbih(data_dir)

    print("Feature engineering...")
    X_train = extract_features(X_train)
    X_test  = extract_features(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── XGBoost principal ──────────────────────────────────────────────────────
    print("Entraînement XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=args.epochs,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=50)

    # ── Évaluation ─────────────────────────────────────────────────────────────
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0

    print(f"\nAccuracy : {acc:.4f}  F1 : {f1:.4f}  AUC : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASSES))

    # ── Sauvegarde ─────────────────────────────────────────────────────────────
    joblib.dump({"model": model, "scaler": scaler}, MODEL_DIR / "cardio_model.pkl")
    metrics = {
        "accuracy": round(acc, 4), "f1_weighted": round(f1, 4),
        "auc_roc": round(auc, 4), "classes": CLASSES,
        "dataset": "MIT-BIH Arrhythmia", "samples_train": len(X_train),
        "training_time_s": round(time.time() - t0, 1),
    }
    (MODEL_DIR / "cardio_model.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nModèle sauvegardé → {MODEL_DIR / 'cardio_model.pkl'}")
    print(f"Métriques        → {MODEL_DIR / 'cardio_model.json'}")
    return metrics


if __name__ == "__main__":
    train(parse_args())
