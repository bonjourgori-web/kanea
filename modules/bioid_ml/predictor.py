from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from modules.bioid_ml.features import build_feature_payload, ALL_FEATURE_KEYS

# ── Racine KANEA dans le path ─────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Chemins des deux bundles (nouveau + ancien fallback)
_BIOID_BUNDLE   = _ROOT / "models" / "machine_learning" / "bioid_bundle.pkl"
_LEGACY_BUNDLE  = _ROOT / "models" / "machine_learning" / "forensic_model.pkl"


# ─── Tentative d'import du nouveau service bioid_ai ──────────────────────────
try:
    from bioid_ai.api.services.bioid_service import predict_full as _bioid_predict_full
    from bioid_ai.api.services.bioid_service import load_bundle  as _bioid_load_bundle
    _BIOID_AI_OK = True
except ImportError:
    _BIOID_AI_OK = False


def _placeholder_response(payload: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    return {
        "module":               "module_3_forensic",
        "task":                 "forensic_biological_profile_estimation",
        "model_family":         "machine_learning",
        "model_name":           "bioid_ai_placeholder",
        "library":              "scikit-learn",
        "expected_model_path":  str(_BIOID_BUNDLE).replace("\\", "/"),
        "inputs_summary": {
            "cranial_measurements":     payload.get("cranial_measurements", {}),
            "postcranial_measurements": payload.get("postcranial_measurements", {}),
            "aims_pcs":                 payload.get("aims_pcs", {}),
        },
        "flattened_features": features,
        "prediction": {
            "biological_sex": None,
            "age_at_death":   None,
            "ancestry":       None,
            "stature_cm":     None,
        },
        "confidence": None,
        "methods_reference": ["PCA", "osteometric_regressions", "craniometric_analysis"],
        "status": "scaffold_ready",
    }


def _via_bioid_ai(payload: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    """Délègue au nouveau service bioid_ai (bundle bioid_bundle.pkl)."""
    cranial     = payload.get("cranial_measurements", {}) or {}
    postcranial = payload.get("postcranial_measurements", {}) or {}
    aims_pcs    = payload.get("aims_pcs", {}) or {}

    # Convertit aims_pcs déjà calculés en liste brute pour la PCA interne
    # Si l'appelant a fourni AIM_PC1/2/3, on les passe directement dans postcranial
    # et on laisse aims_raw=None (la PCA ne sera pas ré-appliquée)
    try:
        _bioid_load_bundle()
    except Exception:
        pass

    result = _bioid_predict_full(
        cranial=cranial,
        postcranial=postcranial,
        aims_raw=None,        # AIMs bruts non disponibles ici — PC déjà calculées
    )

    # Injecte les AIM_PCs dans le résultat si la prédiction est ok
    if result.get("status") == "success":
        return {
            "module":               "module_3_forensic",
            "task":                 "forensic_biological_profile_estimation",
            "model_family":         "machine_learning",
            "model_name":           "bioid_ai_v2",
            "library":              "scikit-learn",
            "expected_model_path":  str(_BIOID_BUNDLE).replace("\\", "/"),
            "inputs_summary": {
                "cranial_measurements":     cranial,
                "postcranial_measurements": postcranial,
                "aims_pcs":                 aims_pcs,
            },
            "flattened_features": features,
            "prediction": {
                "biological_sex": result.get("biological_sex"),
                "age_at_death":   result.get("age_at_death"),
                "ancestry":       result.get("ancestry"),
                "stature_cm":     result.get("stature_cm"),
            },
            "confidence": {
                "biological_sex": result.get("sex_confidence"),
                "ancestry":       result.get("ancestry_confidence"),
            },
            # Données enrichies accessibles au dashboard
            "bioid_ai": {
                "age_range":               result.get("age_range"),
                "ancestry_probabilities":  result.get("ancestry_probabilities", {}),
                "stature_method":          result.get("stature_method"),
                "feature_importances":     result.get("feature_importances", {}),
                "pca_variance":            result.get("pca_variance", []),
                "model_version":           result.get("model_version", "2.0.0"),
            },
            "methods_reference": ["PCA", "VotingClassifier", "Trotter_Gleser", "osteometric_regressions"],
            "status": "model_loaded",
        }

    # Fallback si le service retourne une erreur
    resp = _placeholder_response(payload, features)
    resp["status"] = result.get("status", "inference_error")
    return resp


def _via_legacy_bundle(payload: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    """Utilise l'ancien bundle forensic_model.pkl (compatibilité ascendante)."""
    import pickle

    try:
        with _LEGACY_BUNDLE.open("rb") as fh:
            bundle = pickle.load(fh)
    except Exception as exc:
        resp = _placeholder_response(payload, features)
        resp["status"] = "model_load_error"
        resp["error"]  = str(exc)
        return resp

    try:
        feature_columns  = bundle.get("feature_columns", ALL_FEATURE_KEYS)
        sex_encoder      = bundle["sex_encoder"]
        ancestry_encoder = bundle["ancestry_encoder"]
        sex_model        = bundle["sex_model"]
        age_model        = bundle["age_model"]
        ancestry_model   = bundle["ancestry_model"]
        stature_model    = bundle["stature_model"]

        row = [float(features.get(col) or 0.0) for col in feature_columns]
        X   = np.array([row])

        sex_pred      = sex_model.predict(X)[0]
        age_pred      = age_model.predict(X)[0]
        ancestry_pred = ancestry_model.predict(X)[0]
        stature_pred  = stature_model.predict(X)[0]

        sex_proba      = sex_model.predict_proba(X)[0]
        ancestry_proba = ancestry_model.predict_proba(X)[0]

        return {
            "module":               "module_3_forensic",
            "task":                 "forensic_biological_profile_estimation",
            "model_family":         "machine_learning",
            "model_name":           "scikit_learn_multi_model_legacy",
            "library":              "scikit-learn",
            "expected_model_path":  str(_LEGACY_BUNDLE).replace("\\", "/"),
            "inputs_summary": {
                "cranial_measurements":     payload.get("cranial_measurements", {}),
                "postcranial_measurements": payload.get("postcranial_measurements", {}),
                "aims_pcs":                 payload.get("aims_pcs", {}),
            },
            "flattened_features": features,
            "prediction": {
                "biological_sex": str(sex_encoder.inverse_transform([sex_pred])[0]),
                "age_at_death":   round(float(age_pred), 1),
                "ancestry":       str(ancestry_encoder.inverse_transform([ancestry_pred])[0]),
                "stature_cm":     round(float(stature_pred), 1),
            },
            "confidence": {
                "biological_sex": round(float(sex_proba.max()), 4),
                "ancestry":       round(float(ancestry_proba.max()), 4),
            },
            "methods_reference": ["PCA", "osteometric_regressions", "craniometric_analysis"],
            "status": "model_loaded",
        }

    except Exception as exc:
        resp = _placeholder_response(payload, features)
        resp["status"] = "inference_error"
        resp["error"]  = str(exc)
        return resp


def predict_bioid(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Estimation du profil biologique forensique — Module 3.

    Ordre de priorité :
    1. bioid_ai (bundle bioid_bundle.pkl — nouveau, riche)
    2. forensic_model.pkl (ancien bundle — compatibilité)
    3. placeholder si aucun bundle disponible
    """
    features = build_feature_payload(payload)

    # 1. Nouveau service bioid_ai
    if _BIOID_AI_OK and _BIOID_BUNDLE.exists():
        return _via_bioid_ai(payload, features)

    # 2. Ancien bundle forensic
    if _LEGACY_BUNDLE.exists():
        return _via_legacy_bundle(payload, features)

    # 3. Placeholder
    return _placeholder_response(payload, features)
