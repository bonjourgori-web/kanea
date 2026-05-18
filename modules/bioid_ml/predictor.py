from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from modules.bioid_ml.features import build_feature_payload

MODEL_PATH = Path("models/machine_learning/forensic_model.pkl")


def _placeholder_response(payload: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "module_3_forensic",
        "task": "forensic_biological_profile_estimation",
        "model_family": "machine_learning",
        "model_name": "scikit_learn_ensemble_pca_placeholder",
        "library": "scikit-learn",
        "expected_model_path": str(MODEL_PATH).replace("\\", "/"),
        "inputs_summary": {
            "cranial_measurements": payload.get("cranial_measurements", {}),
            "postcranial_measurements": payload.get("postcranial_measurements", {}),
            "aims_pcs": payload.get("aims_pcs", {}),
        },
        "flattened_features": features,
        "prediction": {
            "biological_sex": None,
            "age_at_death": None,
            "ancestry": None,
            "stature_cm": None,
        },
        "confidence": None,
        "methods_reference": [
            "PCA",
            "osteometric_regressions",
            "craniometric_analysis",
        ],
        "status": "scaffold_ready",
    }


def predict_bioid(payload: dict[str, Any]) -> dict[str, Any]:
    """Interface d'inference du Module 3 conforme a la documentation."""
    features = build_feature_payload(payload)
    model_path = Path.cwd() / MODEL_PATH
    if not model_path.exists():
        return _placeholder_response(payload, features)

    try:
        with model_path.open("rb") as fh:
            bundle = pickle.load(fh)
    except Exception as exc:
        response = _placeholder_response(payload, features)
        response["status"] = "model_load_error"
        response["error"] = str(exc)
        return response

    try:
        feature_columns = bundle["feature_columns"]
        sex_encoder = bundle["sex_encoder"]
        ancestry_encoder = bundle["ancestry_encoder"]
        sex_model = bundle["sex_model"]
        age_model = bundle["age_model"]
        ancestry_model = bundle["ancestry_model"]
        stature_model = bundle["stature_model"]

        frame = pd.DataFrame([features], columns=feature_columns)

        sex_pred = sex_model.predict(frame)[0]
        age_pred = age_model.predict(frame)[0]
        ancestry_pred = ancestry_model.predict(frame)[0]
        stature_pred = stature_model.predict(frame)[0]

        sex_proba = sex_model.predict_proba(frame)[0]
        ancestry_proba = ancestry_model.predict_proba(frame)[0]

        return {
            "module": "module_3_forensic",
            "task": "forensic_biological_profile_estimation",
            "model_family": "machine_learning",
            "model_name": "scikit_learn_multi_model_production",
            "library": "scikit-learn",
            "expected_model_path": str(MODEL_PATH).replace("\\", "/"),
            "inputs_summary": {
                "cranial_measurements": payload.get("cranial_measurements", {}),
                "postcranial_measurements": payload.get("postcranial_measurements", {}),
                "aims_pcs": payload.get("aims_pcs", {}),
            },
            "flattened_features": features,
            "prediction": {
                "biological_sex": str(sex_encoder.inverse_transform([sex_pred])[0]),
                "age_at_death": round(float(age_pred), 1),
                "ancestry": str(ancestry_encoder.inverse_transform([ancestry_pred])[0]),
                "stature_cm": round(float(stature_pred), 1),
            },
            "confidence": {
                "biological_sex": float(sex_proba.max()),
                "ancestry": float(ancestry_proba.max()),
            },
            "methods_reference": [
                "PCA",
                "osteometric_regressions",
                "craniometric_analysis",
            ],
            "status": "model_loaded",
        }
    except Exception as exc:
        response = _placeholder_response(payload, features)
        response["status"] = "inference_error"
        response["error"] = str(exc)
        return response
