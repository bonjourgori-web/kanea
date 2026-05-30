"""
SepsisPredict AI — Pipeline d'entraînement XGBoost + LightGBM + LSTM
======================================================================
Datasets : MIMIC-IV, PhysioNet Sepsis Challenge 2019, eICU CRD, HiRID.
Cible    : Prédiction précoce du sepsis (6–12h avant critères cliniques).

Usage :
    python scripts/train_sepsis.py --dataset data/sepsis --model xgboost
    python scripts/train_sepsis.py --dataset data/sepsis --model lstm --seq_len 24
    python scripts/train_sepsis.py --list-datasets
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ── Datasets de référence ─────────────────────────────────────────────────────

DATASETS = {
    "MIMIC-IV": {
        "patients":  50_000,
        "variables": "200+ (vitaux, biologie, médicaments, procédures)",
        "format":    "CSV / PostgreSQL",
        "access":    "PhysioNet Credentialed Access — https://physionet.org/content/mimiciv/",
        "license":   "PhysioNet Credentialed Health Data License 1.5.0",
        "size_gb":   7,
        "notes":     "Gold standard réanimation — 2008–2019 BIDMC Boston. Sepsis labelling avec critères Sepsis-3.",
        "sepsis_prevalence": "12–14% des admissions ICU",
    },
    "PhysioNet Sepsis Challenge 2019": {
        "patients":  40_336,
        "variables": "40 variables cliniques + temporelles (horaires)",
        "format":    "PSV (pipe-separated values) — séries temporelles horaires",
        "access":    "Téléchargement libre — https://physionet.org/content/challenge-2019/",
        "license":   "Open Database License (ODbL)",
        "size_gb":   0.3,
        "notes":     "Benchmark standard détection précoce sepsis. Labels Sepsis-3. Séries temporelles 40h.",
        "sepsis_prevalence": "8% (3282 cas / 40336 patients)",
    },
    "eICU CRD": {
        "patients":  200_000,
        "variables": "Multiple ICU units (208 hospitals USA)",
        "format":    "CSV / PostgreSQL",
        "access":    "PhysioNet Credentialed Access — https://physionet.org/content/eicu-crd/",
        "license":   "PhysioNet Credentialed Health Data License",
        "size_gb":   7,
        "notes":     "200 000 patients UCM 2014–2015. Données multi-centre. Bonne généralisation.",
    },
    "HiRID": {
        "patients":  34_000,
        "variables": "712 variables — ICU Berne 2008–2016",
        "format":    "HDF5 + CSV",
        "access":    "PhysioNet Credentialed Access — https://physionet.org/content/hirid/",
        "license":   "PhysioNet Credentialed Health Data License",
        "size_gb":   4,
        "notes":     "Haute fréquence d'acquisition (2 min). Idéal pour LSTM / Transformer temporel.",
    },
    "AmsterdamUMCdb": {
        "patients":  23_106,
        "variables": "Données ICU Amsterdam 2003–2016",
        "format":    "CSV",
        "access":    "Accès restreint — https://amsterdammedicaldatascience.nl/",
        "license":   "Amsterdam UMC Data License",
        "size_gb":   3,
        "notes":     "Données européennes — complémentaires MIMIC pour généralisation.",
    },
}

# ── Features engineering ──────────────────────────────────────────────────────

FEATURE_GROUPS = {
    "vitals": ["temperature", "heart_rate", "resp_rate", "systolic_bp",
               "diastolic_bp", "map", "spo2", "gcs"],
    "labs": ["wbc", "neutrophils", "platelets", "lactate", "crp",
             "procalcitonin", "creatinine", "urea", "bilirubin",
             "ast", "alt", "albumin", "inr", "d_dimers", "glucose", "pao2_fio2"],
    "derived": ["shock_index", "neutrophil_lymphocyte_ratio", "sirs_count",
                "infection_score", "sofa_score", "qsofa_score", "news2_score",
                "lactate_norm", "pao2_fio2_norm"],
    "temporal_deltas": ["delta_lactate_1h", "delta_map_1h", "delta_wbc_1h",
                        "delta_creatinine_6h", "delta_temp_1h"],
    "patient_info": ["age", "sex_binary", "comorbidity_score", "icu_los_hours",
                     "on_ventilator", "vasopressors_flag"],
}

# ── Script d'entraînement XGBoost ─────────────────────────────────────────────

XGBOOST_SCRIPT = '''"""
SepsisPredict AI — Entraînement XGBoost + LightGBM
===================================================
Nécessite : xgboost, lightgbm, scikit-learn, optuna, shap, pandas, numpy
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.impute import KNNImputer
import xgboost as xgb
import lightgbm as lgb
import optuna
import shap
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "machine_learning"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "sepsis_label"     # 0=no sepsis, 1=SIRS, 2=Sepsis, 3=Severe, 4=Shock
TARGET_BINARY = "sepsis_binary" # 0=no sepsis, 1=any sepsis

# Feature engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["shock_index"] = df["heart_rate"] / df["systolic_bp"].clip(lower=1)
    df["nlr"] = df["neutrophils"] / (df["wbc"] - df["neutrophils"]).clip(lower=0.1)
    df["sirs_count"] = (
        ((df["temperature"] > 38) | (df["temperature"] < 36)).astype(int) +
        (df["heart_rate"] > 90).astype(int) +
        (df["resp_rate"] > 20).astype(int) +
        ((df["wbc"] > 12) | (df["wbc"] < 4)).astype(int)
    )
    df["infection_score"] = (
        (df["procalcitonin"] / 10).clip(0,1) * 0.5 +
        (df["crp"] / 200).clip(0,1) * 0.3 +
        (df["d_dimers"] / 5).clip(0,1) * 0.2
    )
    df["map_deficit"] = (65 - df["map"]).clip(lower=0)
    df["lactate_excess"] = (df["lactate"] - 2.0).clip(lower=0)
    df["pao2_fio2_deficit"] = (400 - df["pao2_fio2"]).clip(lower=0) / 400
    df["age_norm"] = df["age"] / 100
    return df

# Imputation valeurs manquantes (KNN)
def impute_missing(X_train, X_test):
    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp  = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    return X_train_imp, X_test_imp, imputer

# Objective Optuna XGBoost
def objective_xgb(trial, X, y, cv):
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 200, 1000),
        "max_depth":       trial.suggest_int("max_depth", 3, 8),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":trial.suggest_int("min_child_weight", 1, 10),
        "gamma":           trial.suggest_float("gamma", 0, 5),
        "scale_pos_weight":trial.suggest_float("scale_pos_weight", 1, 20),
        "use_label_encoder":False, "eval_metric":"auc", "random_state":42,
    }
    aucs = []
    for train_idx, val_idx in cv.split(X, y):
        model = xgb.XGBClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx],
                  eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                  early_stopping_rounds=30, verbose=False)
        pred = model.predict_proba(X.iloc[val_idx])[:,1]
        aucs.append(roc_auc_score(y.iloc[val_idx], pred))
    return np.mean(aucs)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="xgboost", choices=["xgboost","lightgbm","ensemble"])
    parser.add_argument("--n_trials", type=int, default=50)
    args = parser.parse_args()

    print("Chargement des données...")
    data_path = Path(args.dataset)
    df = pd.read_csv(data_path / "sepsis_features.csv")
    df = engineer_features(df)

    feature_cols = [c for c in df.columns if c not in [LABEL_COL, TARGET_BINARY, "patient_id", "time"]]
    X = df[feature_cols]
    y = df[TARGET_BINARY] if TARGET_BINARY in df.columns else (df[LABEL_COL] >= 2).astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    print(f"Dataset : {len(df)} patients, {len(feature_cols)} features")
    print(f"Sepsis rate : {y.mean():.1%}")

    # Optimisation Optuna
    print("Optimisation hyperparamètres (Optuna)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective_xgb(t, X_scaled, y, cv), n_trials=args.n_trials)
    best_params = study.best_params
    print(f"Best AUC : {study.best_value:.4f}")
    print(f"Best params : {best_params}")

    # Entraînement final
    print("Entraînement modèle final...")
    best_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric="auc", random_state=42)
    best_model.fit(X_scaled, y)

    # LightGBM (ensemble)
    lgb_model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42)
    lgb_model.fit(X_scaled, y)

    # Évaluation finale
    pred_xgb = best_model.predict_proba(X_scaled)[:,1]
    pred_lgb = lgb_model.predict_proba(X_scaled)[:,1]
    pred_ens = (pred_xgb * 0.6 + pred_lgb * 0.4)

    for name, pred in [("XGBoost", pred_xgb), ("LightGBM", pred_lgb), ("Ensemble", pred_ens)]:
        auc = roc_auc_score(y, pred)
        ap  = average_precision_score(y, pred)
        f1  = f1_score(y, pred > 0.5)
        print(f"{name}: AUC={auc:.4f} | AP={ap:.4f} | F1={f1:.4f}")

    # SHAP explanability
    print("Calcul SHAP...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_scaled[:1000])
    feature_importance = dict(zip(feature_cols, np.abs(shap_values).mean(axis=0)))

    # Sauvegarde
    bundle = {
        "xgboost": best_model,
        "lightgbm": lgb_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "best_params": best_params,
        "feature_importance": feature_importance,
    }
    joblib.dump(bundle, MODEL_DIR / "sepsis_model.pkl")

    meta = {
        "model_version": "v2.0",
        "architecture": "XGBoost + LightGBM Ensemble",
        "features": feature_cols,
        "best_auc": study.best_value,
        "classes": ["Pas de sepsis", "SIRS", "Sepsis", "Sepsis sévère", "Choc septique"],
        "dataset": str(data_path),
        "feature_importance": {k: float(v) for k,v in sorted(feature_importance.items(), key=lambda x: -x[1])[:20]},
    }
    with open(MODEL_DIR / "sepsis_model.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Modèle sauvegardé → {MODEL_DIR / 'sepsis_model.pkl'}")

if __name__ == "__main__":
    main()
'''

# ── Script LSTM (séries temporelles) ─────────────────────────────────────────

LSTM_SCRIPT = '''"""
SepsisPredict AI — Entraînement LSTM (Série temporelle)
=========================================================
Architecture : BiLSTM + Attention — PhysioNet Sepsis Challenge 2019.
Prédiction 6h, 12h et 24h avant critères sepsis.
Nécessite : torch >= 2.0, numpy, pandas, scikit-learn.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SepsisLSTMDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.FloatTensor(sequences)
        self.y = torch.FloatTensor(labels)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class BiLSTMAttention(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(hidden_size*2, num_heads=4, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size*2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        self.norm = nn.LayerNorm(hidden_size*2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out)
        # Self-attention
        out_t = out.transpose(0, 1)
        attn_out, _ = self.attention(out_t, out_t, out_t)
        context = attn_out.mean(dim=0)
        return self.classifier(context).squeeze(-1)

def train_lstm(dataset_path, epochs=50, lr=1e-3, batch_size=64, seq_len=24):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = BiLSTMAttention(input_size=40, hidden_size=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    print("Modèle LSTM BiLSTM+Attention prêt — lancer avec données PhysioNet Sepsis 2019")
    return model

if __name__ == "__main__":
    model = train_lstm("data/sepsis_physionet")
'''


def main():
    parser = argparse.ArgumentParser(description="SepsisPredict AI — Training Pipeline")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--generate-xgboost", action="store_true")
    parser.add_argument("--generate-lstm", action="store_true")
    parser.add_argument("--export-config", action="store_true")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parent.parent

    if args.list_datasets:
        print("\n=== Datasets SepsisPredict AI ===\n")
        for name, info in DATASETS.items():
            print(f"{'─'*60}")
            print(f"  {name}")
            print(f"  Patients : {info['patients']:,}")
            print(f"  Variables: {info['variables']}")
            print(f"  Licence  : {info['license']}")
            print(f"  Accès    : {info['access']}")
            print(f"  Note     : {info['notes']}")
        print()

    elif args.generate_xgboost:
        out = ROOT / "scripts" / "train_sepsis_xgboost.py"
        out.write_text(XGBOOST_SCRIPT, encoding="utf-8")
        print(f"Script XGBoost généré : {out}")
        print("Usage : python scripts/train_sepsis_xgboost.py --dataset data/sepsis")

    elif args.generate_lstm:
        out = ROOT / "scripts" / "train_sepsis_lstm.py"
        out.write_text(LSTM_SCRIPT, encoding="utf-8")
        print(f"Script LSTM généré : {out}")

    elif args.export_config:
        config = {
            "module": "SepsisPredict AI v2.0",
            "classes": ["Pas de sepsis","SIRS","Sepsis","Sepsis sévère","Choc septique"],
            "features": FEATURE_GROUPS,
            "datasets": {k: {kk:vv for kk,vv in v.items() if kk != "access"} for k,v in DATASETS.items()},
            "scores": ["SOFA","qSOFA","NEWS2","APACHE II","SAPS II","MEWS","MODS","SSC Bundle"],
            "models": ["XGBoost","LightGBM","BiLSTM-Attention"],
        }
        out = ROOT / "scripts" / "sepsis_config.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Config exportée : {out}")

    else:
        print("SepsisPredict AI — Training Pipeline")
        print(f"  Classes  : 5 (SIRS → Choc septique)")
        print(f"  Modèles  : XGBoost + LightGBM + BiLSTM")
        print(f"  Datasets : {len(DATASETS)}")
        print(f"  Scores   : SOFA, qSOFA, NEWS2, APACHE II, SAPS II, MEWS, MODS")
        print()
        print("Options :")
        print("  --list-datasets       Lister les datasets")
        print("  --generate-xgboost    Générer script XGBoost + Optuna + SHAP")
        print("  --generate-lstm       Générer script BiLSTM + Attention")
        print("  --export-config       Exporter config JSON")


if __name__ == "__main__":
    main()
