"""
bioid_service.py — Service principal orchestrateur du module BioID AI.

Orchestre :
- Chargement des modèles depuis le bundle pkl
- predict_full() — 4 prédictions (sexe, âge, ascendance, stature)
- compute_shap() — valeurs SHAP pour interprétabilité
- generate_pdf_report() — délègue au générateur PDF
- export_json() / export_csv()
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Résolution des chemins ────────────────────────────────────────────────────
_SERVICE_DIR = Path(__file__).resolve().parent          # api/services/
_BIOID_DIR   = _SERVICE_DIR.parent.parent               # bioid_ai/
_KANEA_ROOT  = _BIOID_DIR.parent                         # KANEA/
_BUNDLE_PATH = _KANEA_ROOT / "models" / "machine_learning" / "bioid_bundle.pkl"

if str(_KANEA_ROOT) not in sys.path:
    sys.path.insert(0, str(_KANEA_ROOT))

# ── Imports locaux ────────────────────────────────────────────────────────────
try:
    from bioid_ai.utils.feature_engineering import (
        ALL_FEATURE_KEYS, engineer_features, estimate_stature, payload_from_schema,
    )
    from bioid_ai.utils.pdf_generator import generate_bioid_report
    from bioid_ai.utils.visualizations import (
        plot_biological_radar, plot_feature_importance, plot_prediction_summary,
    )
    _LOCAL_OK = True
except ImportError:
    try:
        from utils.feature_engineering import (
            ALL_FEATURE_KEYS, engineer_features, estimate_stature, payload_from_schema,
        )
        from utils.pdf_generator import generate_bioid_report
        from utils.visualizations import (
            plot_biological_radar, plot_feature_importance, plot_prediction_summary,
        )
        _LOCAL_OK = True
    except ImportError:
        _LOCAL_OK = False
        log.warning("Modules locaux bioid_ai non disponibles — mode dégradé")

try:
    import joblib
    _JOBLIB_OK = True
except ImportError:
    import pickle as _pickle
    _JOBLIB_OK = False

try:
    import shap as _shap_lib
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False
    log.info("shap non disponible — valeurs SHAP désactivées")


# ── Cache des modèles (singleton en mémoire) ──────────────────────────────────
_MODEL_BUNDLE: Optional[dict[str, Any]] = None
_BUNDLE_LOAD_ERROR: Optional[str]       = None


def load_bundle(bundle_path: Optional[Path] = None) -> dict[str, Any]:
    """
    Charge le bundle de modèles en mémoire (lazy loading).

    Retourne le bundle chargé.
    Lève RuntimeError si le bundle ne peut pas être chargé.
    """
    global _MODEL_BUNDLE, _BUNDLE_LOAD_ERROR

    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE

    path = bundle_path or _BUNDLE_PATH

    if not Path(path).exists():
        msg = (
            f"Bundle introuvable : {path}. "
            f"Exécutez d'abord : python bioid_ai/training/train.py"
        )
        _BUNDLE_LOAD_ERROR = msg
        raise FileNotFoundError(msg)

    try:
        if _JOBLIB_OK:
            bundle = joblib.load(str(path))
        else:
            with open(path, "rb") as fh:
                bundle = _pickle.load(fh)  # type: ignore[name-defined]

        _MODEL_BUNDLE = bundle
        _BUNDLE_LOAD_ERROR = None
        log.info(
            "Bundle chargé : %s | modèles=%s",
            path,
            [k for k in bundle.keys() if "model" in k],
        )
        return bundle

    except Exception as exc:
        _BUNDLE_LOAD_ERROR = str(exc)
        raise RuntimeError(f"Erreur lors du chargement du bundle : {exc}") from exc


def is_bundle_loaded() -> bool:
    """Vérifie si le bundle est chargé."""
    return _MODEL_BUNDLE is not None


def reload_bundle(bundle_path: Optional[Path] = None) -> dict[str, Any]:
    """Force le rechargement du bundle."""
    global _MODEL_BUNDLE
    _MODEL_BUNDLE = None
    return load_bundle(bundle_path)


# ── Prédiction complète ───────────────────────────────────────────────────────
def predict_full(
    cranial: Optional[dict]     = None,
    postcranial: Optional[dict] = None,
    aims_raw: Optional[list]    = None,
    bundle: Optional[dict]      = None,
) -> dict[str, Any]:
    """
    Effectue les 4 prédictions du profil biologique forensique.

    Paramètres
    ----------
    cranial : dict
        Mesures crâniennes (clés = noms des variables, valeurs = float ou None).
    postcranial : dict
        Mesures post-crâniennes.
    aims_raw : list[float]
        Marqueurs AIMs bruts (sera transformé en PC1/PC2/PC3 via PCA).
    bundle : dict optionnel
        Bundle pré-chargé (si None, charge depuis le fichier).

    Retourne un dict avec toutes les prédictions et métadonnées.
    """
    try:
        b = bundle or load_bundle()
    except (FileNotFoundError, RuntimeError) as exc:
        log.warning("Bundle indisponible — prédiction placeholder : %s", exc)
        return _placeholder_prediction(str(exc))

    try:
        # ── Transformation AIMs bruts → composantes PCA ─────────────────────
        aims_pcs: dict[str, float] = {}
        if aims_raw and len(aims_raw) > 0:
            pca       = b.get("pca")
            pca_scaler = b.get("pca_scaler")
            if pca is not None and pca_scaler is not None:
                try:
                    if _LOCAL_OK:
                        from bioid_ai.utils.pca_analysis import transform_aims
                    else:
                        from utils.pca_analysis import transform_aims  # type: ignore
                    valid_aims = [float(v) if v is not None else 0.0 for v in aims_raw]
                    aims_pcs = transform_aims(valid_aims, pca, pca_scaler)
                except Exception as e:
                    log.warning("Transformation AIMs échouée : %s", e)
                    aims_pcs = {"AIM_PC1": 0.0, "AIM_PC2": 0.0, "AIM_PC3": 0.0}

        # ── Construction du payload ──────────────────────────────────────────
        payload: dict[str, Any] = {
            "cranial":    cranial    or {},
            "postcranial": postcranial or {},
            "aims":       aims_pcs,
        }

        # ── Feature engineering ──────────────────────────────────────────────
        if _LOCAL_OK:
            df = engineer_features(payload)
        else:
            # Fallback minimal
            all_keys = b.get("feature_columns", ALL_FEATURE_KEYS if _LOCAL_OK else [])
            row = {}
            cranial_d = cranial or {}
            post_d    = postcranial or {}
            for k in all_keys:
                val = cranial_d.get(k) or post_d.get(k) or aims_pcs.get(k)
                row[k] = float(val) if val is not None else 0.0
            df = pd.DataFrame([row])

        feature_cols = b.get("feature_columns", list(df.columns))
        # Aligner les colonnes avec le modèle
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        X_raw = df[feature_cols].fillna(0.0).values

        # ── Normalisation ────────────────────────────────────────────────────
        scaler = b.get("scaler")
        if scaler is not None:
            try:
                X = scaler.transform(X_raw)
            except Exception:
                X = X_raw
        else:
            X = X_raw

        # ── Modèles ──────────────────────────────────────────────────────────
        sex_model        = b["sex_model"]
        age_model        = b["age_model"]
        ancestry_model   = b["ancestry_model"]
        stature_model    = b["stature_model"]
        sex_encoder      = b["sex_encoder"]
        ancestry_encoder = b["ancestry_encoder"]

        # ── Prédictions ──────────────────────────────────────────────────────
        sex_pred_enc  = sex_model.predict(X)[0]
        sex_proba     = sex_model.predict_proba(X)[0]
        sex_label     = str(sex_encoder.inverse_transform([sex_pred_enc])[0])
        sex_conf      = float(sex_proba.max())

        age_pred      = float(age_model.predict(X)[0])
        age_pred      = max(0.0, min(120.0, age_pred))

        anc_pred_enc  = ancestry_model.predict(X)[0]
        anc_proba     = ancestry_model.predict_proba(X)[0]
        anc_label     = str(ancestry_encoder.inverse_transform([anc_pred_enc])[0])
        anc_conf      = float(anc_proba.max())

        # Probabilités complètes pour l'ascendance
        anc_classes   = list(ancestry_encoder.classes_)
        anc_proba_dict = {
            str(anc_classes[i]): round(float(anc_proba[i]), 4)
            for i in range(len(anc_classes))
        }

        stat_pred     = float(stature_model.predict(X)[0])
        stat_pred     = max(100.0, min(250.0, stat_pred))

        # Stature alternative (formule directe) si modèle non fiable
        femur_bic = (postcranial or {}).get("femur_bicondylar")
        if femur_bic and _LOCAL_OK:
            stat_formula = estimate_stature(femur_bic, sex_label)
            stat_method  = "Trotter & Gleser (1958)"
            # Moyenne pondérée modèle + formule
            if stat_formula:
                stat_pred = round((stat_pred * 0.6 + stat_formula * 0.4), 1)
        else:
            stat_method = "Random Forest Regressor"
            if femur_bic is None:
                stat_pred = None  # type: ignore
                stat_method = "Données insuffisantes (fémur manquant)"

        # Intervalle d'âge
        age_min   = max(0, round(age_pred - 8))
        age_max   = round(age_pred + 8)
        age_range = f"{age_min}–{age_max} ans"

        # Importance des features (RF ascendance)
        feature_importances: dict[str, float] = {}
        try:
            rf_estimators = getattr(ancestry_model, "estimators_", None)
            if rf_estimators and hasattr(rf_estimators[0], "feature_importances_"):
                fi = rf_estimators[0].feature_importances_
            elif hasattr(ancestry_model, "feature_importances_"):
                fi = ancestry_model.feature_importances_
            else:
                fi = None
            if fi is not None and len(fi) == len(feature_cols):
                feature_importances = {
                    feature_cols[i]: round(float(fi[i]), 6)
                    for i in range(len(feature_cols))
                }
        except Exception as e:
            log.debug("feature_importances extraction: %s", e)

        # Variance PCA
        pca_variance: Optional[list[float]] = None
        pca_obj = b.get("pca")
        if pca_obj is not None:
            try:
                pca_variance = [round(float(v), 4) for v in pca_obj.explained_variance_ratio_]
            except Exception:
                pass

        result: dict[str, Any] = {
            "biological_sex":       sex_label,
            "sex_confidence":       round(sex_conf, 4),
            "age_at_death":         round(age_pred, 1),
            "age_range":            age_range,
            "ancestry":             anc_label,
            "ancestry_confidence":  round(anc_conf, 4),
            "ancestry_probabilities": anc_proba_dict,
            "stature_cm":           round(stat_pred, 1) if stat_pred is not None else None,
            "stature_method":       stat_method,
            "feature_importances":  feature_importances,
            "pca_variance":         pca_variance,
            "model_version":        b.get("metadata", {}).get("version", "2.0.0"),
            "status":               "success",
        }
        log.info(
            "predict_full OK → sex=%s(%.0f%%), age=%.0f, ancestry=%s(%.0f%%)",
            sex_label, sex_conf * 100, age_pred, anc_label, anc_conf * 100,
        )
        return result

    except Exception as exc:
        log.error("predict_full error: %s", exc, exc_info=True)
        return {**_placeholder_prediction(str(exc)), "status": "inference_error"}


def _placeholder_prediction(error_msg: str = "") -> dict[str, Any]:
    """Retourne une prédiction placeholder en cas d'indisponibilité des modèles."""
    return {
        "biological_sex":       "Inconnu",
        "sex_confidence":       0.0,
        "age_at_death":         0.0,
        "age_range":            "0–0 ans",
        "ancestry":             "Inconnu",
        "ancestry_confidence":  0.0,
        "ancestry_probabilities": {},
        "stature_cm":           None,
        "stature_method":       "Modèle indisponible",
        "feature_importances":  {},
        "pca_variance":         None,
        "model_version":        "N/A",
        "status":               "placeholder",
        "error":                error_msg,
    }


# ── Valeurs SHAP ──────────────────────────────────────────────────────────────
def compute_shap(
    X: np.ndarray,
    model: Any,
    feature_names: list[str],
    max_samples: int = 1,
) -> Optional[np.ndarray]:
    """
    Calcule les valeurs SHAP pour un échantillon ou un ensemble d'échantillons.

    Paramètres
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    model : modèle scikit-learn compatible shap
    feature_names : list[str]
    max_samples : int

    Retourne np.ndarray des valeurs SHAP, ou None si shap non disponible.
    """
    if not _SHAP_OK:
        log.info("shap non disponible — valeurs SHAP ignorées")
        return None

    try:
        X_sample = X[:max_samples]
        explainer = _shap_lib.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        log.debug("compute_shap → shape=%s", np.array(shap_values).shape)
        return np.array(shap_values)
    except Exception as exc:
        log.warning("compute_shap error (non bloquant) : %s", exc)
        return None


# ── Génération du rapport PDF ─────────────────────────────────────────────────
def generate_pdf_report(
    case_id: str,
    examiner: str,
    predictions: dict[str, Any],
    charts: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
) -> str:
    """
    Génère un rapport PDF médico-légal complet.

    Retourne le chemin du PDF ou "" si erreur.
    """
    try:
        case_data = {
            "case_id":   case_id,
            "examiner":  examiner,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status":    predictions.get("status", "Analysé"),
        }

        # Génération des graphiques si demandé
        chart_paths = list(charts) if charts else []

        if not chart_paths:
            # Générer les graphiques automatiquement
            viz_dir = _BIOID_DIR / "visualizations" / case_id
            viz_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Radar chart
                radar_path = plot_biological_radar(
                    predictions,
                    save_path=viz_dir / "radar.png",
                    title=f"Profil Biologique — {case_id}",
                )
                if radar_path:
                    chart_paths.append(radar_path)

                # Summary chart
                confidences = {
                    "sex_confidence":      predictions.get("sex_confidence", 0),
                    "ancestry_confidence": predictions.get("ancestry_confidence", 0),
                }
                summary_path = plot_prediction_summary(
                    predictions, confidences,
                    save_path=viz_dir / "summary.png",
                    title=f"Résumé BioID — {case_id}",
                )
                if summary_path:
                    chart_paths.append(summary_path)

                # Feature importance
                if predictions.get("feature_importances"):
                    fi_path = plot_feature_importance(
                        predictions["feature_importances"],
                        title="Importance des Features",
                        save_path=viz_dir / "feature_importance.png",
                    )
                    if fi_path:
                        chart_paths.append(fi_path)

            except Exception as e:
                log.warning("Génération des graphiques échouée : %s", e)

        if output_dir is None:
            output_dir = _KANEA_ROOT / "reports"

        pdf_path = generate_bioid_report(case_data, predictions, chart_paths, output_dir)
        if pdf_path:
            log.info("Rapport PDF généré : %s", pdf_path)
        return pdf_path

    except Exception as exc:
        log.error("generate_pdf_report error: %s", exc, exc_info=True)
        return ""


# ── Export JSON ───────────────────────────────────────────────────────────────
def export_json(
    case_id: str,
    predictions: dict[str, Any],
    output_dir: Optional[Path] = None,
) -> str:
    """
    Exporte les prédictions en format JSON.

    Retourne le chemin du fichier JSON ou "" si erreur.
    """
    try:
        if output_dir is None:
            output_dir = _KANEA_ROOT / "reports"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bioid_{case_id}_{ts}.json"
        filepath = output_dir / filename

        export_data = {
            "case_id":    case_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "predictions": predictions,
            "system":     "BioID AI v2.0 — KANÉA",
        }

        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(export_data, fh, ensure_ascii=False, indent=2, default=str)

        log.info("Export JSON : %s", filepath)
        return str(filepath)

    except Exception as exc:
        log.error("export_json error: %s", exc)
        return ""


# ── Export CSV ────────────────────────────────────────────────────────────────
def export_csv(
    case_id: str,
    predictions: dict[str, Any],
    output_dir: Optional[Path] = None,
) -> str:
    """
    Exporte les prédictions en format CSV (une ligne par cas).

    Retourne le chemin du fichier CSV ou "" si erreur.
    """
    try:
        if output_dir is None:
            output_dir = _KANEA_ROOT / "reports"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bioid_{case_id}_{ts}.csv"
        filepath = output_dir / filename

        row = {
            "case_id":              case_id,
            "exported_at":          datetime.now(timezone.utc).isoformat(),
            "biological_sex":       predictions.get("biological_sex", ""),
            "sex_confidence":       predictions.get("sex_confidence", ""),
            "age_at_death":         predictions.get("age_at_death", ""),
            "age_range":            predictions.get("age_range", ""),
            "ancestry":             predictions.get("ancestry", ""),
            "ancestry_confidence":  predictions.get("ancestry_confidence", ""),
            "stature_cm":           predictions.get("stature_cm", ""),
            "stature_method":       predictions.get("stature_method", ""),
            "status":               predictions.get("status", ""),
            "model_version":        predictions.get("model_version", ""),
        }

        df = pd.DataFrame([row])
        df.to_csv(filepath, index=False, encoding="utf-8")

        log.info("Export CSV : %s", filepath)
        return str(filepath)

    except Exception as exc:
        log.error("export_csv error: %s", exc)
        return ""
