from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from modules.nutrition_ml.features import build_feature_payload

MODEL_PATH = Path("models/machine_learning/nutrition_model.pkl")


def _placeholder_response(features: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "module_2_biometry",
        "task": "biometric_nutritional_status_prediction",
        "model_family": "machine_learning",
        "model_name": "RandomForest_XGBoost_ensemble_placeholder",
        "library": "scikit-learn_xgboost",
        "expected_model_path": str(MODEL_PATH).replace("\\", "/"),
        "inputs_summary": {
            "age_months": features.get("age_months"),
            "weight_kg": features.get("weight_kg"),
            "height_cm": features.get("height_cm"),
            "sex": features.get("sex"),
            "muac_cm": features.get("muac_cm"),
        },
        "derived_features": {
            "bmi": features.get("bmi"),
            "waz": features.get("waz"),
            "haz": features.get("haz"),
            "whz": features.get("whz"),
        },
        "prediction": None,
        "confidence": None,
        "ensemble_strategy": "majority_voting",
        "explainability": {
            "method": "feature_importance_and_shap",
            "status": "not_generated",
        },
        "status": "scaffold_ready",
    }


def predict_nutrition(payload: dict[str, Any]) -> dict[str, Any]:
    """Interface d'inference du Module 2 conforme a la documentation."""
    features = build_feature_payload(payload)
    model_path = Path.cwd() / MODEL_PATH
    if not model_path.exists():
        return _placeholder_response(features)

    try:
        with model_path.open("rb") as fh:
            bundle = pickle.load(fh)
    except Exception as exc:
        response = _placeholder_response(features)
        response["status"] = "model_load_error"
        response["error"] = str(exc)
        return response

    try:
        feature_columns = bundle["feature_columns"]
        label_encoder = bundle["label_encoder"]
        rf_model = bundle["random_forest"]
        xgb_model = bundle["xgboost"]

        frame = pd.DataFrame([features], columns=feature_columns)
        rf_prediction = rf_model.predict(frame)[0]
        xgb_prediction = xgb_model.predict(frame)[0]

        rf_proba = rf_model.predict_proba(frame)[0]
        xgb_proba = xgb_model.predict_proba(frame)[0]

        rf_label = label_encoder.inverse_transform([rf_prediction])[0]
        xgb_label = label_encoder.inverse_transform([xgb_prediction])[0]

        if rf_prediction == xgb_prediction:
            final_prediction = rf_prediction
            confidence = float((rf_proba.max() + xgb_proba.max()) / 2.0)
        else:
            final_prediction = rf_prediction
            confidence = float(rf_proba.max())

        return {
            "module": "module_2_biometry",
            "task": "biometric_nutritional_status_prediction",
            "model_family": "machine_learning",
            "model_name": "RandomForest_XGBoost_ensemble_production",
            "library": "scikit-learn_xgboost",
            "expected_model_path": str(MODEL_PATH).replace("\\", "/"),
            "inputs_summary": {
                "age_months": features.get("age_months"),
                "weight_kg": features.get("weight_kg"),
                "height_cm": features.get("height_cm"),
                "sex": features.get("sex"),
                "muac_cm": features.get("muac_cm"),
            },
            "derived_features": {
                "bmi": features.get("bmi"),
                "waz": features.get("waz"),
                "haz": features.get("haz"),
                "whz": features.get("whz"),
            },
            "prediction": str(label_encoder.inverse_transform([final_prediction])[0]),
            "confidence": confidence,
            "ensemble_strategy": "majority_voting",
            "base_model_outputs": {
                "random_forest": str(rf_label),
                "xgboost": str(xgb_label),
            },
            "explainability": {
                "method": "feature_importance_and_shap",
                "status": "pending_generation",
            },
            "status": "model_loaded",
        }
    except Exception as exc:
        response = _placeholder_response(features)
        response["status"] = "inference_error"
        response["error"] = str(exc)
        return response
