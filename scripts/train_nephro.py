"""
NephroAI — Entraînement XGBoost sur indicateurs de santé rénale/diabète
========================================================================
Dataset : BRFSS2015 Diabetes Health Indicators
Cible   : Prédiction risque maladie rénale chronique / diabète
Export  : PKL + JSON metrics

Usage :
    python scripts/train_nephro.py --dataset data/nephro
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              f1_score, classification_report)
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import shap
import joblib

ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "machine_learning"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Pas de risque", "Pré-diabète / risque modéré", "Diabète / risque élevé"]

FEATURES = [
    "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack",
    "PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare",
    "NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/nephro")
    p.add_argument("--epochs",  type=int, default=400)
    return p.parse_args()


def load_data(data_dir: Path):
    # Préférer le fichier 5050split (classes équilibrées)
    f = data_dir / "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
    if not f.exists():
        f = next(data_dir.glob("*.csv"), None)
    if not f:
        raise FileNotFoundError(f"Aucun CSV dans {data_dir}")
    df = pd.read_csv(f)
    print(f"Dataset : {f.name} — {len(df):,} lignes × {df.shape[1]} colonnes")

    target = "Diabetes_binary" if "Diabetes_binary" in df.columns else df.columns[0]
    avail   = [c for c in FEATURES if c in df.columns]
    X = df[avail].values
    y = df[target].astype(int).values
    print(f"Features utilisées : {len(avail)}  |  Classes : {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, avail


def train(args):
    t0 = time.time()
    data_dir = ROOT / args.dataset

    X, y, feat_names = load_data(data_dir)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("Entraînement XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=args.epochs,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = 0.0

    print(f"\nAccuracy : {acc:.4f}  F1 : {f1:.4f}  AUC : {auc:.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump({"model": model, "scaler": scaler, "features": feat_names},
                MODEL_DIR / "nephro_model.pkl")
    metrics = {
        "accuracy": round(acc, 4), "f1_weighted": round(f1, 4),
        "auc_roc": round(auc, 4), "classes": CLASSES,
        "features": feat_names, "dataset": "BRFSS2015 Diabetes Health Indicators",
        "samples_train": len(X_train), "training_time_s": round(time.time() - t0, 1),
    }
    (MODEL_DIR / "nephro_model.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nModèle sauvegardé → {MODEL_DIR / 'nephro_model.pkl'}")
    return metrics


if __name__ == "__main__":
    train(parse_args())
