from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from modules.nutrition_ml.features import build_feature_payload

MODEL_PATH = Path("models/machine_learning/nutrition_model.pkl")

# Correspondance classes du modèle entraîné → labels standardisés
_CLASS_MAP = {
    "MAS":       "severe_undernutrition",
    "MAM":       "moderate_undernutrition",
    "Normal":    "normal",
    "Overweight": "overweight",
    "Obesity":   "obesity",
}


def _placeholder_response(features: dict[str, Any]) -> dict[str, Any]:
    return {
        "module":         "module_2_biometry",
        "task":           "biometric_nutritional_status_prediction",
        "model_family":   "machine_learning",
        "model_name":     "VotingClassifier_RF_XGBoost_placeholder",
        "library":        "scikit-learn_xgboost",
        "expected_model_path": str(MODEL_PATH).replace("\\", "/"),
        "inputs_summary": {
            "age_months": features.get("age_months"),
            "weight_kg":  features.get("weight_kg"),
            "height_cm":  features.get("height_cm"),
            "sex":        features.get("sex"),
            "muac_cm":    features.get("muac_cm"),
        },
        "derived_features": {
            "bmi": features.get("bmi"),
            "waz": features.get("waz"),
            "haz": features.get("haz"),
            "whz": features.get("whz"),
        },
        "prediction":       None,
        "confidence":       None,
        "ensemble_strategy": "voting_classifier",
        "explainability": {
            "method": "feature_importance_and_shap",
            "status": "not_generated",
        },
        "status": "scaffold_ready",
    }


def _compute_shap(bundle: dict) -> dict[str, Any]:
    """Calcule l'importance des features via le RF interne du VotingClassifier."""
    try:
        voting = bundle["ensemble"]
        rf = dict(voting.estimators).get("rf")
        if rf is None:
            return {"status": "rf_not_found"}

        importances = rf.feature_importances_
        feature_names = bundle["features"]
        ranked = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        return {
            "status":    "generated",
            "method":    "random_forest_feature_importance",
            "top_features": [
                {"feature": name, "importance": round(float(imp), 4)}
                for name, imp in ranked[:5]
            ],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def predict_nutrition(payload: dict[str, Any]) -> dict[str, Any]:
    """Inférence nutritionnelle — VotingClassifier (RF + XGBoost)."""
    features   = build_feature_payload(payload)
    model_path = Path.cwd() / MODEL_PATH
    if not model_path.exists():
        return _placeholder_response(features)

    try:
        with model_path.open("rb") as fh:
            bundle = pickle.load(fh)
    except Exception as exc:
        response = _placeholder_response(features)
        response["status"] = "model_load_error"
        response["error"]  = str(exc)
        return response

    try:
        voting        = bundle["ensemble"]
        label_encoder = bundle["label_encoder"]
        feature_cols  = bundle["features"]
        classes       = bundle["classes"]

        # Construire le vecteur de features dans l'ordre exact d'entraînement
        row = [float(features.get(col, 0.0)) for col in feature_cols]
        # numpy array évite le warning sklearn "fitted without feature names"
        X = np.array([row])

        # Prédiction VotingClassifier (soft voting)
        proba       = voting.predict_proba(X)[0]
        pred_idx    = int(np.argmax(proba))
        raw_label   = classes[pred_idx]
        std_label   = _CLASS_MAP.get(raw_label, raw_label.lower())
        confidence  = round(float(proba[pred_idx]), 4)

        # Scores par classe
        class_scores = {
            _CLASS_MAP.get(cls, cls.lower()): round(float(p), 4)
            for cls, p in zip(classes, proba)
        }

        # Contribution par estimateur
        rf_pred  = dict(voting.estimators)["rf"].predict(X)[0]
        xgb_pred = dict(voting.estimators)["xgb"].predict(X)[0]

        # Explainabilité via importances RF
        shap_result = _compute_shap(bundle)

        return {
            "module":         "module_2_biometry",
            "task":           "biometric_nutritional_status_prediction",
            "model_family":   "machine_learning",
            "model_name":     "VotingClassifier_RF_XGBoost_production",
            "library":        "scikit-learn_xgboost",
            "expected_model_path": str(MODEL_PATH).replace("\\", "/"),
            "inputs_summary": {
                "age_months": features.get("age_months"),
                "weight_kg":  features.get("weight_kg"),
                "height_cm":  features.get("height_cm"),
                "sex":        features.get("sex"),
                "muac_cm":    features.get("muac_cm"),
            },
            "derived_features": {
                "bmi": features.get("bmi"),
                "waz": features.get("waz"),
                "haz": features.get("haz"),
                "whz": features.get("whz"),
            },
            "prediction":        std_label,
            "prediction_raw":    raw_label,
            "confidence":        confidence,
            "class_probabilities": class_scores,
            "ensemble_strategy": "voting_classifier",
            "base_model_outputs": {
                "random_forest": _CLASS_MAP.get(str(label_encoder.inverse_transform([rf_pred])[0]), str(rf_pred)),
                "xgboost":       _CLASS_MAP.get(str(label_encoder.inverse_transform([xgb_pred])[0]), str(xgb_pred)),
            },
            "explainability": shap_result,
            "status": "model_loaded",
        }

    except Exception as exc:
        response = _placeholder_response(features)
        response["status"] = "inference_error"
        response["error"]  = str(exc)
        return response
