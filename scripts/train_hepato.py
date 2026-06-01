"""
HepatoScan AI — Entraînement XGBoost sur cirrhose hépatique
============================================================
Dataset : Cirrhosis Prediction Dataset (fedesoriano/kaggle)
Cible   : Stade cirrhose / survie patient
Usage   : python scripts/train_hepato.py --dataset data/hepato
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb
import joblib

ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "machine_learning"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Stade C (sévère)", "Stade B (modéré)", "Stade A (léger)", "Compensée / Stable"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/hepato")
    p.add_argument("--epochs",  type=int, default=400)
    return p.parse_args()


def load_data(data_dir: Path):
    csv_files = list(data_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Aucun CSV dans {data_dir}")
    df = pd.read_csv(csv_files[0])
    print(f"Dataset : {csv_files[0].name} — {df.shape}")
    print(f"Colonnes : {list(df.columns)}")

    # Identifier la cible (Status ou Stage ou similaire)
    target_candidates = [c for c in df.columns if any(k in c.lower()
                         for k in ["status","stage","outcome","surviv","class","label","target"])]
    target_col = target_candidates[0] if target_candidates else df.columns[-1]
    print(f"Cible : {target_col}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Encoder les colonnes catégorielles (sauf la cible)
    for c in cat_cols:
        if c != target_col:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    feat_cols = [c for c in df.columns if c != target_col]
    X = df[feat_cols].fillna(df[feat_cols].median(numeric_only=True))
    y_raw = df[target_col]
    le = LabelEncoder()
    y  = le.fit_transform(y_raw.astype(str))
    print(f"Features : {len(feat_cols)} | Classes : {dict(zip(le.classes_, np.bincount(y)))}")
    return X.values, y, list(feat_cols), le


def train(args):
    t0 = time.time()
    X, y, feat_cols, le = load_data(ROOT / args.dataset)
    n_classes = len(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                          random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("Entraînement XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=args.epochs, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        n_jobs=-1, random_state=42, tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro") \
              if n_classes > 2 else roc_auc_score(y_test, y_proba[:,1])
    except Exception:
        auc = 0.0

    print(f"\nAccuracy : {acc:.4f}  F1 : {f1:.4f}  AUC : {auc:.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump({"model": model, "scaler": scaler, "features": feat_cols, "encoder": le},
                MODEL_DIR / "hepato_model.pkl")
    metrics = {
        "accuracy": round(float(acc), 4), "f1_weighted": round(float(f1), 4),
        "auc_roc": round(float(auc), 4),
        "classes": list(le.classes_.astype(str)), "architecture": "XGBoost",
        "dataset": "Cirrhosis Prediction Dataset", "samples_train": int(len(X_train)),
        "training_time_s": round(time.time() - t0, 1),
    }
    (MODEL_DIR / "hepato_model.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nModele -> {MODEL_DIR / 'hepato_model.pkl'}")
    return metrics


if __name__ == "__main__":
    train(parse_args())
