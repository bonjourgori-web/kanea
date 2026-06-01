"""
GynoCare AI — Entraînement XGBoost sur dépistage cancer col utérin
===================================================================
Dataset : GynoCare Colposcopy Ratings (hinselmann, schiller, green CSVs)
Cible   : Classification CIN1 / CIN2 / CIN3 / Normal
Export  : PKL + JSON metrics

Usage :
    python scripts/train_gyno.py --dataset data/gyno
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              f1_score, classification_report)
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "machine_learning"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Normal", "CIN1 (légère)", "CIN2 (modérée)", "CIN3 (sévère)"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/gyno")
    p.add_argument("--epochs",  type=int, default=400)
    return p.parse_args()


def load_data(data_dir: Path):
    dfs = []
    for csv_name in ["hinselmann.csv", "schiller.csv", "green.csv"]:
        f = data_dir / csv_name
        if f.exists():
            df = pd.read_csv(f)
            df["source"] = csv_name.replace(".csv","")
            dfs.append(df)
            print(f"  {csv_name}: {df.shape}")

    if not dfs:
        # Chercher tout CSV disponible
        for f in data_dir.rglob("*.csv"):
            df = pd.read_csv(f)
            if len(df) > 50:
                dfs.append(df)
                print(f"  {f.name}: {df.shape}")
        if not dfs:
            raise FileNotFoundError(f"Aucune donnée CSV dans {data_dir}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total : {len(df):,} lignes × {df.shape[1]} colonnes")
    print(f"Colonnes : {list(df.columns)}")

    # Détecter la colonne cible
    target_candidates = [c for c in df.columns if any(k in c.lower() for k in ["class","label","target","result","diagnosis","cin"])]
    if not target_candidates:
        target_col = df.columns[-1]
    else:
        target_col = target_candidates[0]
    print(f"Cible : {target_col}")

    # Encodage
    df = df.dropna(subset=[target_col])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    X = df[numeric_cols].fillna(df[numeric_cols].median()).values
    y_raw = df[target_col].values
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    print(f"Classes : {dict(zip(le.classes_, np.bincount(y)))}")
    return X, y, numeric_cols, le


def train(args):
    t0 = time.time()
    data_dir = ROOT / args.dataset

    X, y, feat_names, le = load_data(data_dir)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                          stratify=y if len(np.unique(y)) > 1 else None)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("Entraînement XGBoost...")
    n_classes = len(np.unique(y))
    model = xgb.XGBClassifier(
        n_estimators=args.epochs,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        n_jobs=-1, random_state=42, tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    try:
        y_proba = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro") if n_classes > 2 \
              else roc_auc_score(y_test, y_proba[:, 1])
    except Exception:
        auc = 0.0

    print(f"\nAccuracy : {acc:.4f}  F1 : {f1:.4f}  AUC : {auc:.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump({"model": model, "scaler": scaler, "features": feat_names, "encoder": le},
                MODEL_DIR / "gyno_model.pkl")
    metrics = {
        "accuracy": round(acc, 4), "f1_weighted": round(f1, 4), "auc_roc": round(auc, 4),
        "classes": list(le.classes_.astype(str)), "features": feat_names,
        "dataset": "GynoCare Colposcopy Ratings", "samples_train": len(X_train),
        "training_time_s": round(time.time() - t0, 1),
    }
    (MODEL_DIR / "gyno_model.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nModèle sauvegardé → {MODEL_DIR / 'gyno_model.pkl'}")
    return metrics


if __name__ == "__main__":
    train(parse_args())
