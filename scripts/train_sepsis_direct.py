"""
SepsisPredict AI — Entraînement direct XGBoost sur PhysioNet Sepsis 2019
========================================================================
Dataset : data/sepsis/Dataset.csv (PhysioNet Challenge 2019)
Usage   : python scripts/train_sepsis_direct.py --dataset data/sepsis
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              f1_score, classification_report)
import xgboost as xgb
import lightgbm as lgb
import joblib

ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "machine_learning"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Pas de sepsis", "SIRS", "Sepsis", "Sepsis sévère", "Choc septique"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/sepsis")
    p.add_argument("--epochs",  type=int, default=500)
    return p.parse_args()


def load_data(data_dir: Path):
    # Essayer Dataset.csv d'abord
    main_csv = data_dir / "Dataset.csv"
    if main_csv.exists():
        df = pd.read_csv(main_csv, nrows=100_000)
        print(f"Dataset.csv : {df.shape}")
        print(f"Colonnes : {list(df.columns[:15])}...")
        return df

    # Sinon charger les PSV PhysioNet
    psv_files = list(data_dir.glob("*.psv"))[:5000]
    if psv_files:
        frames = []
        for f in psv_files[:5000]:
            try:
                d = pd.read_csv(f, sep="|")
                if len(d) > 0:
                    d["patient_id"] = f.stem
                    frames.append(d.tail(1))  # Dernière ligne = état final
            except Exception:
                pass
        if frames:
            df = pd.concat(frames, ignore_index=True)
            print(f"PSV agrégé : {df.shape}")
            return df

    raise FileNotFoundError(f"Aucune données dans {data_dir}")


def prepare_features(df: pd.DataFrame):
    # Exclure colonnes non-numériques et identifiants
    exclude = ["patient_id", "SepsisLabel", "Unit1", "Unit2", "ICULOS",
               "HospAdmTime", "Gender", "Age"]
    target_col = next((c for c in ["SepsisLabel", "sepsis_label", "label", "target",
                                   "Sepsis", "outcome"] if c in df.columns), None)
    if target_col is None:
        # Chercher colonne binaire 0/1
        for c in df.columns:
            if df[c].nunique() == 2 and df[c].dtype in [np.int64, np.float64]:
                vals = sorted(df[c].dropna().unique())
                if vals == [0, 1] or vals == [0.0, 1.0]:
                    target_col = c
                    break

    if target_col is None:
        # Utiliser dernière colonne
        target_col = df.columns[-1]

    print(f"Colonne cible : {target_col}")

    feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in exclude and c != target_col]

    X = df[feat_cols].fillna(df[feat_cols].median())
    y = (df[target_col].fillna(0) > 0).astype(int)
    print(f"Features : {len(feat_cols)}  |  Taux sepsis : {y.mean():.1%}")
    return X.values, y.values, feat_cols


def train(args):
    t0 = time.time()
    data_dir = ROOT / args.dataset

    df = load_data(data_dir)
    X, y, feat_cols = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    pos_weight = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))
    print(f"pos_weight = {pos_weight:.1f}")

    print("Entraînement XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=args.epochs, max_depth=6,
        learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=pos_weight,
        eval_metric="auc", n_jobs=-1, random_state=42, tree_method="hist",
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=100)

    print("Entraînement LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05,
        scale_pos_weight=pos_weight, random_state=42, n_jobs=-1,
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.log_evaluation(100)])

    # Ensemble
    p_xgb = model.predict_proba(X_test)[:, 1]
    p_lgb = lgb_model.predict_proba(X_test)[:, 1]
    p_ens = 0.6 * p_xgb + 0.4 * p_lgb
    y_pred = (p_ens > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_test, p_ens)
    except Exception:
        auc = 0.0

    print(f"\nAccuracy : {acc:.4f}  F1 : {f1:.4f}  AUC : {auc:.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump({"xgboost": model, "lightgbm": lgb_model, "scaler": scaler,
                 "features": feat_cols},
                MODEL_DIR / "sepsis_model.pkl")

    metrics = {
        "accuracy": round(float(acc), 4), "f1_binary": round(float(f1), 4),
        "auc_roc": round(float(auc), 4), "classes": CLASSES,
        "architecture": "XGBoost + LightGBM Ensemble",
        "dataset": "PhysioNet Sepsis Challenge 2019",
        "samples_train": int(len(X_train)), "training_time_s": round(time.time() - t0, 1),
    }
    (MODEL_DIR / "sepsis_model.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nModèle → {MODEL_DIR / 'sepsis_model.pkl'}")
    return metrics


if __name__ == "__main__":
    train(parse_args())
