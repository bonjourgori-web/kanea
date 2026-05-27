"""
train.py — Script d'entraînement complet du bundle BioID AI.

Usage :
    python bioid_ai/training/train.py

Génère des données synthétiques réalistes si aucune donnée réelle n'est disponible,
entraîne 4 modèles (sexe, âge, ascendance, stature) et sauvegarde le bundle
dans KANEA/models/machine_learning/bioid_bundle.pkl.
"""
from __future__ import annotations

import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("bioid_training")

# ── Résolution des chemins ────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent          # bioid_ai/training/
BIOID_DIR   = SCRIPT_DIR.parent                         # bioid_ai/
KANEA_ROOT  = BIOID_DIR.parent                          # KANEA/
MODELS_DIR  = KANEA_ROOT / "models" / "machine_learning"
VIZ_DIR     = BIOID_DIR / "visualizations"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Ajouter KANEA au path pour les imports
if str(KANEA_ROOT) not in sys.path:
    sys.path.insert(0, str(KANEA_ROOT))

# ── Imports scikit-learn ──────────────────────────────────────────────────────
try:
    import joblib
    from sklearn.decomposition import PCA
    from sklearn.ensemble import (
        GradientBoostingClassifier, GradientBoostingRegressor,
        RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor,
    )
    from sklearn.linear_model import ElasticNet, LinearRegression
    from sklearn.metrics import (
        accuracy_score, classification_report, f1_score,
        mean_absolute_error, r2_score,
    )
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    _SK_OK = True
except ImportError as e:
    log.error("scikit-learn manquant : %s", e)
    sys.exit(1)

# ── MLflow (optionnel) ────────────────────────────────────────────────────────
try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_OK = True
    log.info("MLflow disponible — logs activés")
except ImportError:
    _MLFLOW_OK = False
    log.info("MLflow non disponible — logs MLflow désactivés")

# ── Visualisations (optionnel) ────────────────────────────────────────────────
try:
    from bioid_ai.utils.visualizations import plot_confusion_matrix, plot_feature_importance
    _VIZ_OK = True
except ImportError:
    try:
        sys.path.insert(0, str(BIOID_DIR.parent))
        from bioid_ai.utils.visualizations import plot_confusion_matrix, plot_feature_importance
        _VIZ_OK = True
    except ImportError:
        _VIZ_OK = False
        log.info("Module visualizations non disponible — graphiques désactivés")

# ── Clés des features ─────────────────────────────────────────────────────────
CRANIAL_KEYS = [
    "GOL", "XCB", "BBH", "ZYB", "AUB", "ASB",
    "BNL", "BPL", "NLH", "NLB", "OBH", "OBB",
    "MAB", "FOL", "FOB",
]
POSTCRANIAL_KEYS = [
    "femur_max_length", "femur_bicondylar", "tibia_length",
    "humerus_max_length", "radius_max_length", "fibula_max_length",
    "femur_head_diam", "humerus_head_diam",
]
DERIVED_KEYS = ["cephalic_idx", "nasal_idx", "orbital_idx", "femur_ratio"]
AIMS_KEYS    = ["AIM_PC1", "AIM_PC2", "AIM_PC3"]
ALL_FEATURES = CRANIAL_KEYS + POSTCRANIAL_KEYS + DERIVED_KEYS + AIMS_KEYS


# ── Génération de données synthétiques ───────────────────────────────────────
def generate_synthetic_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Génère des données synthétiques réalistes basées sur la littérature FORDISC.

    Paramètres basés sur :
    - Ousley & Jantz (1996) FORDISC 3.0 database
    - Trotter & Gleser (1958) stature regressions
    """
    rng = np.random.default_rng(seed)
    n_male   = n_samples // 2
    n_female = n_samples - n_male

    records: list[dict[str, Any]] = []

    # ── Moyennes et std par sexe (mm) ─────────────────────────────────────────
    MALE_PARAMS: dict[str, tuple[float, float]] = {
        "GOL": (185.0, 8.0),  "XCB": (139.0, 7.0),  "BBH": (136.0, 7.5),
        "ZYB": (133.0, 7.0),  "AUB": (123.0, 6.5),  "ASB": (112.0, 6.0),
        "BNL": (102.0, 6.0),  "BPL": (100.0, 6.5),  "NLH": (53.0, 4.5),
        "NLB": (27.0, 3.0),   "OBH": (33.5, 2.5),   "OBB": (41.5, 2.5),
        "MAB": (65.0, 4.5),   "FOL": (37.5, 3.0),   "FOB": (31.0, 2.5),
        "femur_max_length":  (467.0, 28.0),
        "femur_bicondylar":  (462.0, 28.0),
        "tibia_length":      (378.0, 23.0),
        "humerus_max_length":(328.0, 22.0),
        "radius_max_length": (252.0, 17.0),
        "fibula_max_length": (370.0, 22.0),
        "femur_head_diam":   (47.0, 3.5),
        "humerus_head_diam": (47.0, 3.5),
    }
    FEMALE_PARAMS: dict[str, tuple[float, float]] = {
        "GOL": (178.0, 7.5),  "XCB": (134.0, 6.5),  "BBH": (128.0, 7.0),
        "ZYB": (125.0, 6.5),  "AUB": (117.0, 6.0),  "ASB": (107.0, 5.5),
        "BNL": (97.0, 5.5),   "BPL": (94.0, 6.0),   "NLH": (49.0, 4.0),
        "NLB": (25.0, 2.8),   "OBH": (34.5, 2.5),   "OBB": (40.5, 2.5),
        "MAB": (62.0, 4.0),   "FOL": (36.0, 2.8),   "FOB": (29.5, 2.5),
        "femur_max_length":  (427.0, 25.0),
        "femur_bicondylar":  (422.0, 25.0),
        "tibia_length":      (344.0, 21.0),
        "humerus_max_length":(302.0, 20.0),
        "radius_max_length": (226.0, 15.0),
        "fibula_max_length": (337.0, 20.0),
        "femur_head_diam":   (41.0, 3.0),
        "humerus_head_diam": (41.0, 3.0),
    }

    # Groupes ancestraux : Africaine, Européenne, Mixte
    ANCESTRY_GROUPS = ["Africaine", "Europeenne", "Mixte"]

    def _sample_individual(sex: str, params: dict) -> dict[str, Any]:
        row: dict[str, Any] = {"sex": sex}
        for k, (mu, sd) in params.items():
            # ~5% de données manquantes simulées (remplacées par 0 pour l'entraînement)
            if rng.random() < 0.05:
                row[k] = 0.0
            else:
                row[k] = float(rng.normal(mu, sd))

        # Âge au décès : distribution bimodale réaliste (adultes)
        age_group = rng.choice(["young", "middle", "old"], p=[0.25, 0.50, 0.25])
        if age_group == "young":
            row["age_at_death"] = float(rng.normal(32, 7))
        elif age_group == "middle":
            row["age_at_death"] = float(rng.normal(52, 10))
        else:
            row["age_at_death"] = float(rng.normal(68, 8))
        row["age_at_death"] = max(18.0, min(90.0, row["age_at_death"]))

        # Ascendance
        row["ancestry"] = rng.choice(ANCESTRY_GROUPS)

        # Indices dérivés
        gol = row.get("GOL", 0)
        xcb = row.get("XCB", 0)
        nlh = row.get("NLH", 0)
        nlb = row.get("NLB", 0)
        obh = row.get("OBH", 0)
        obb = row.get("OBB", 0)
        fem = row.get("femur_max_length", 0)
        fhd = row.get("femur_head_diam", 0)

        row["cephalic_idx"] = (xcb / gol * 100) if gol > 0 else 75.0
        row["nasal_idx"]    = (nlb / nlh * 100) if nlh > 0 else 51.0
        row["orbital_idx"]  = (obh / obb * 100) if obb > 0 else 80.0
        row["femur_ratio"]  = (fem / fhd) if fhd > 0 else 10.0

        # AIMs synthétiques (corrélés avec l'ascendance)
        if row["ancestry"] == "Africaine":
            aims_base = np.array([2.0, -1.0, 0.5])
        elif row["ancestry"] == "Europeenne":
            aims_base = np.array([-1.5, 1.5, -0.5])
        else:
            aims_base = np.array([0.2, 0.3, 0.1])
        aims_noise = rng.normal(0, 0.8, size=3)
        aims = aims_base + aims_noise
        row["AIM_PC1"] = float(aims[0])
        row["AIM_PC2"] = float(aims[1])
        row["AIM_PC3"] = float(aims[2])

        # Stature (formule Trotter & Gleser, en cm)
        fbc = row.get("femur_bicondylar", 0)
        fbc_cm = fbc / 10.0 if fbc > 100 else fbc
        if sex == "Male":
            row["stature_cm"] = 2.32 * fbc_cm + 65.53 + rng.normal(0, 3.0)
        else:
            row["stature_cm"] = 2.47 * fbc_cm + 54.10 + rng.normal(0, 3.0)
        row["stature_cm"] = max(140.0, min(210.0, row["stature_cm"]))

        return row

    for _ in range(n_male):
        records.append(_sample_individual("Male", MALE_PARAMS))
    for _ in range(n_female):
        records.append(_sample_individual("Female", FEMALE_PARAMS))

    df = pd.DataFrame(records)
    log.info("Données synthétiques générées : %d lignes, %d colonnes", len(df), len(df.columns))
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ── Entraînement des modèles ──────────────────────────────────────────────────
def train_models(df: pd.DataFrame) -> dict[str, Any]:
    """
    Entraîne les 4 modèles BioID sur le DataFrame fourni.

    Retourne le bundle complet prêt à être sérialisé.
    """
    log.info("Début de l'entraînement sur %d échantillons", len(df))

    # ── Encodeurs ────────────────────────────────────────────────────────────
    sex_encoder      = LabelEncoder()
    ancestry_encoder = LabelEncoder()
    y_sex      = sex_encoder.fit_transform(df["sex"])
    y_age      = df["age_at_death"].values
    y_ancestry = ancestry_encoder.fit_transform(df["ancestry"])
    y_stature  = df["stature_cm"].values

    X = df[ALL_FEATURES].fillna(0).values

    # ── Normalisation ─────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Split train/test ──────────────────────────────────────────────────────
    (X_tr, X_te,
     ys_tr, ys_te,
     ya_tr, ya_te,
     yanc_tr, yanc_te,
     yst_tr, yst_te) = train_test_split(
        X_scaled, y_sex, y_age, y_ancestry, y_stature,
        test_size=0.2, random_state=42, stratify=y_sex,
    )

    # ── PCA sur les AIMs ──────────────────────────────────────────────────────
    aims_idx = [ALL_FEATURES.index(k) for k in AIMS_KEYS if k in ALL_FEATURES]
    aims_data = X_scaled[:, aims_idx] if aims_idx else X_scaled[:, :3]
    pca_scaler = StandardScaler()
    aims_scaled = pca_scaler.fit_transform(aims_data)
    n_aims_comp = min(3, aims_data.shape[1], aims_data.shape[0])
    pca = PCA(n_components=n_aims_comp)
    pca.fit(aims_scaled)
    log.info("PCA AIMs — variance expliquée : %s", np.round(pca.explained_variance_ratio_, 3))

    results: dict[str, Any] = {}

    # ── 1. MODÈLE SEXE — VotingClassifier(RF + GB) ───────────────────────────
    log.info("Entraînement du modèle de sexe...")
    rf_sex = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    gb_sex = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    sex_model = VotingClassifier(
        estimators=[("rf", rf_sex), ("gb", gb_sex)],
        voting="soft",
    )
    sex_model.fit(X_tr, ys_tr)

    ys_pred = sex_model.predict(X_te)
    sex_acc = accuracy_score(ys_te, ys_pred)
    sex_f1  = f1_score(ys_te, ys_pred, average="weighted")
    log.info("Sexe — Accuracy=%.4f, F1=%.4f", sex_acc, sex_f1)
    print(f"\n{'='*50}")
    print(f"SEXE BIOLOGIQUE — Accuracy: {sex_acc:.4f}, F1: {sex_f1:.4f}")
    print(classification_report(ys_te, ys_pred, target_names=sex_encoder.classes_))

    cv_sex = cross_val_score(sex_model, X_scaled, y_sex, cv=5, scoring="accuracy")
    log.info("CV Sexe — mean=%.4f ± %.4f", cv_sex.mean(), cv_sex.std())
    print(f"Cross-Val Sexe — {cv_sex.mean():.4f} ± {cv_sex.std():.4f}")
    results["sex_metrics"] = {"accuracy": sex_acc, "f1": sex_f1, "cv_mean": cv_sex.mean()}

    # ── 2. MODÈLE ÂGE — VotingRegressor(RF + GB + ElasticNet) ───────────────
    log.info("Entraînement du modèle d'âge...")
    rf_age = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    gb_age = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    en_age = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=42)
    age_model = VotingRegressor(
        estimators=[("rf", rf_age), ("gb", gb_age), ("en", en_age)],
    )
    age_model.fit(X_tr, ya_tr)

    ya_pred = age_model.predict(X_te)
    age_mae = mean_absolute_error(ya_te, ya_pred)
    age_r2  = r2_score(ya_te, ya_pred)
    log.info("Âge — MAE=%.2f ans, R²=%.4f", age_mae, age_r2)
    print(f"\n{'='*50}")
    print(f"ÂGE AU DÉCÈS — MAE: {age_mae:.2f} ans, R²: {age_r2:.4f}")

    cv_age = cross_val_score(age_model, X_scaled, y_age, cv=5, scoring="neg_mean_absolute_error")
    log.info("CV Âge — MAE mean=%.2f ± %.2f", -cv_age.mean(), cv_age.std())
    print(f"Cross-Val Âge — MAE: {-cv_age.mean():.2f} ± {cv_age.std():.2f}")
    results["age_metrics"] = {"mae": age_mae, "r2": age_r2, "cv_mae": -cv_age.mean()}

    # ── 3. MODÈLE ASCENDANCE — RandomForestClassifier ────────────────────────
    log.info("Entraînement du modèle d'ascendance...")
    ancestry_model = RandomForestClassifier(
        n_estimators=150, max_depth=10, random_state=42, n_jobs=-1,
        class_weight="balanced",
    )
    ancestry_model.fit(X_tr, yanc_tr)

    yanc_pred = ancestry_model.predict(X_te)
    anc_acc = accuracy_score(yanc_te, yanc_pred)
    anc_f1  = f1_score(yanc_te, yanc_pred, average="weighted")
    log.info("Ascendance — Accuracy=%.4f, F1=%.4f", anc_acc, anc_f1)
    print(f"\n{'='*50}")
    print(f"ASCENDANCE — Accuracy: {anc_acc:.4f}, F1: {anc_f1:.4f}")
    print(classification_report(yanc_te, yanc_pred, target_names=ancestry_encoder.classes_))

    cv_anc = cross_val_score(ancestry_model, X_scaled, y_ancestry, cv=5, scoring="accuracy")
    log.info("CV Ascendance — mean=%.4f ± %.4f", cv_anc.mean(), cv_anc.std())
    print(f"Cross-Val Ascendance — {cv_anc.mean():.4f} ± {cv_anc.std():.4f}")
    results["ancestry_metrics"] = {"accuracy": anc_acc, "f1": anc_f1, "cv_mean": cv_anc.mean()}

    # ── 4. MODÈLE STATURE — LinearRegression sur femur_bicondylar ────────────
    log.info("Entraînement du modèle de stature...")
    stature_model = LinearRegression()
    # On entraîne sur toutes les features pour plus de robustesse
    stature_model.fit(X_tr, yst_tr)

    yst_pred = stature_model.predict(X_te)
    stat_mae = mean_absolute_error(yst_te, yst_pred)
    stat_r2  = r2_score(yst_te, yst_pred)
    log.info("Stature — MAE=%.2f cm, R²=%.4f", stat_mae, stat_r2)
    print(f"\n{'='*50}")
    print(f"STATURE — MAE: {stat_mae:.2f} cm, R²: {stat_r2:.4f}")
    results["stature_metrics"] = {"mae": stat_mae, "r2": stat_r2}

    # ── Visualisations ────────────────────────────────────────────────────────
    if _VIZ_OK:
        log.info("Génération des visualisations...")
        try:
            # Confusion matrix — Sexe (décoder les entiers en labels)
            ys_te_labels   = sex_encoder.inverse_transform(ys_te)
            ys_pred_labels = sex_encoder.inverse_transform(ys_pred)
            plot_confusion_matrix(
                ys_te_labels, ys_pred_labels,
                labels=list(sex_encoder.classes_),
                title="Matrice de Confusion — Sexe Biologique",
                save_path=VIZ_DIR / "confusion_sex.png",
            )
            # Confusion matrix — Ascendance (décoder les entiers en labels)
            yanc_te_labels   = ancestry_encoder.inverse_transform(yanc_te)
            yanc_pred_labels = ancestry_encoder.inverse_transform(yanc_pred)
            plot_confusion_matrix(
                yanc_te_labels, yanc_pred_labels,
                labels=list(ancestry_encoder.classes_),
                title="Matrice de Confusion — Ascendance",
                save_path=VIZ_DIR / "confusion_ancestry.png",
            )
            # Feature importance (RF ascendance)
            fi_dict = {
                ALL_FEATURES[i]: ancestry_model.feature_importances_[i]
                for i in range(len(ALL_FEATURES))
            }
            plot_feature_importance(
                fi_dict,
                title="Importance des Features — Ascendance",
                save_path=VIZ_DIR / "feature_importance_ancestry.png",
            )
            log.info("Visualisations sauvegardées dans %s", VIZ_DIR)
        except Exception as exc:
            log.warning("Erreur lors de la génération des visualisations : %s", exc)

    # ── Bundle final ──────────────────────────────────────────────────────────
    bundle = {
        "sex_model":         sex_model,
        "age_model":         age_model,
        "ancestry_model":    ancestry_model,
        "stature_model":     stature_model,
        "sex_encoder":       sex_encoder,
        "ancestry_encoder":  ancestry_encoder,
        "feature_columns":   ALL_FEATURES,
        "scaler":            scaler,
        "pca":               pca,
        "pca_scaler":        pca_scaler,
        "metadata": {
            "trained_at":  datetime.now().isoformat(),
            "n_samples":   len(df),
            "features":    ALL_FEATURES,
            "sex_classes":      list(sex_encoder.classes_),
            "ancestry_classes": list(ancestry_encoder.classes_),
            "metrics":     results,
            "version":     "2.0.0",
            "framework":   "scikit-learn",
        },
    }
    return bundle


def save_bundle(bundle: dict[str, Any], path: Path) -> None:
    """Sauvegarde le bundle avec joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    size_mb = path.stat().st_size / 1024 / 1024
    log.info("Bundle sauvegardé : %s (%.2f MB)", path, size_mb)
    print(f"\nBundle sauvegardé : {path} ({size_mb:.2f} MB)")


def run_mlflow_logging(bundle: dict[str, Any], run_name: str = "bioid_training") -> None:
    """Log les métriques dans MLflow si disponible."""
    if not _MLFLOW_OK:
        return
    try:
        mlflow.set_experiment("KANEA_BioID_AI")
        with mlflow.start_run(run_name=run_name):
            metrics = bundle.get("metadata", {}).get("metrics", {})
            for model_name, m in metrics.items():
                for metric_key, metric_val in m.items():
                    mlflow.log_metric(f"{model_name}_{metric_key}", float(metric_val))
            mlflow.log_param("n_samples",   bundle["metadata"]["n_samples"])
            mlflow.log_param("n_features",  len(bundle["feature_columns"]))
            mlflow.log_param("version",     bundle["metadata"]["version"])
            mlflow.sklearn.log_model(bundle["sex_model"],      "sex_model")
            mlflow.sklearn.log_model(bundle["age_model"],      "age_model")
            mlflow.sklearn.log_model(bundle["ancestry_model"], "ancestry_model")
            log.info("Métriques loggées dans MLflow")
    except Exception as exc:
        log.warning("MLflow logging échoué (non bloquant) : %s", exc)


# ── Point d'entrée ────────────────────────────────────────────────────────────
def main() -> None:
    print("\n" + "=" * 60)
    print("  KANÉA — BioID AI — Entraînement des modèles v2.0")
    print("=" * 60)
    print(f"  Répertoire modèles : {MODELS_DIR}")
    print(f"  Visualisations     : {VIZ_DIR}")
    print("=" * 60 + "\n")

    # Vérifier si des données réelles existent
    real_data_path = KANEA_ROOT / "data" / "bioid_dataset.csv"
    if real_data_path.exists():
        log.info("Données réelles trouvées : %s", real_data_path)
        df = pd.read_csv(real_data_path)
        log.info("Chargement de %d échantillons réels", len(df))
    else:
        log.info("Aucune donnée réelle — génération de données synthétiques (N=500)")
        df = generate_synthetic_data(n_samples=500, seed=42)

    bundle = train_models(df)

    bundle_path = MODELS_DIR / "bioid_bundle.pkl"
    save_bundle(bundle, bundle_path)
    run_mlflow_logging(bundle)

    print("\n" + "=" * 60)
    print("  Entraînement terminé avec succès !")
    print(f"  Bundle : {bundle_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
