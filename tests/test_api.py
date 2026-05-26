"""Tests API FastAPI — KANÉA
Couvre tous les endpoints REST avec le TestClient de FastAPI (pas de serveur requis).
"""
from __future__ import annotations

import io

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ═══════════════════════════════════════════════════════════════════════════════
# Routes de base
# ═══════════════════════════════════════════════════════════════════════════════

def test_root_returns_application_name() -> None:
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["application"] == "KANEA"
    assert data["status"] == "running"


def test_health_returns_ok() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_info_contains_four_modules() -> None:
    r = client.get("/info")
    assert r.status_code == 200
    data = r.json()
    assert len(data["modules"]) == 4
    module_names = [m["name"] for m in data["modules"]]
    assert "module_1_malaria" in module_names
    assert "module_4_breast_cancer" in module_names


# ═══════════════════════════════════════════════════════════════════════════════
# Prédiction nutrition (tabular)
# ═══════════════════════════════════════════════════════════════════════════════

_NUTRITION_PAYLOAD = {
    "age_months": 24,
    "weight_kg": 12.0,
    "height_cm": 85.0,
    "sex": "F",
    "muac_cm": 13.5,
    "waz": -1.1,
    "haz": -0.8,
    "whz": -0.3,
}


def test_predict_biometry_returns_module_2() -> None:
    r = client.post("/predict/biometry", json=_NUTRITION_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert data["module"] == "module_2_biometry"
    assert "status" in data


def test_predict_biometry_has_explainability() -> None:
    r = client.post("/predict/biometry", json=_NUTRITION_PAYLOAD)
    assert r.status_code == 200
    expl = r.json().get("explainability", {})
    assert "method" in expl
    assert "status" in expl


def test_predict_biometry_model_loaded_has_prediction() -> None:
    r = client.post("/predict/biometry", json=_NUTRITION_PAYLOAD)
    data = r.json()
    if data["status"] == "model_loaded":
        assert data["prediction"] is not None
        assert 0.0 <= data["confidence"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Prédiction forensique (tabular)
# ═══════════════════════════════════════════════════════════════════════════════

_BIOID_PAYLOAD = {
    "cranial_measurements": {
        "GOL": 183.0, "XCB": 143.0, "BBH": 135.0,
        "ZYB": 133.0, "AUB": 122.0, "ASB": 108.0,
        "BNL": 99.0,  "BPL": 98.0,  "NLH": 53.0,
        "NLB": 27.0,  "OBH": 35.0,  "OBB": 43.0,
        "MAB": 65.0,  "FOL": 37.0,  "FOB": 31.0,
    },
    "postcranial_measurements": {
        "femur_max_length":   452.0,
        "femur_bicondylar":   449.0,
        "tibia_length":       372.0,
        "humerus_max_length": 327.0,
        "radius_max_length":  247.0,
        "fibula_max_length":  367.0,
        "femur_head_diam":     47.5,
        "humerus_head_diam":   46.0,
    },
    "aims_pcs": {"AIM_PC1": 1.2, "AIM_PC2": 0.4, "AIM_PC3": 0.3},
}


def test_predict_forensic_returns_module_3() -> None:
    r = client.post("/predict/forensic", json=_BIOID_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert data["module"] == "module_3_forensic"


def test_predict_forensic_model_loaded_has_full_profile() -> None:
    r = client.post("/predict/forensic", json=_BIOID_PAYLOAD)
    data = r.json()
    if data["status"] == "model_loaded":
        pred = data.get("prediction", {})
        assert "biological_sex" in pred
        assert "age_at_death" in pred
        assert "ancestry" in pred
        assert "stature_cm" in pred


# ═══════════════════════════════════════════════════════════════════════════════
# Prédiction malaria — path (sans vrai fichier → scaffold)
# ═══════════════════════════════════════════════════════════════════════════════

def test_predict_malaria_from_path_scaffold() -> None:
    r = client.post("/predict/malaria", json={"image_path": "nonexistent.png"})
    assert r.status_code == 200
    data = r.json()
    assert data["module"] == "module_1_malaria"
    assert data["status"] in ("scaffold_ready", "inference_error", "model_loaded")


def test_predict_malaria_upload_png() -> None:
    # Image PNG 1×1 pixel valide (minimal)
    png_1x1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
        b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    files = {"file": ("test.png", io.BytesIO(png_1x1), "image/png")}
    r = client.post("/predict/malaria/upload", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["module"] == "module_1_malaria"


# ═══════════════════════════════════════════════════════════════════════════════
# Prédiction cancer du sein — upload
# ═══════════════════════════════════════════════════════════════════════════════

def test_predict_breast_cancer_upload_png() -> None:
    png_1x1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
        b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    files = {"file": ("mammo.png", io.BytesIO(png_1x1), "image/png")}
    r = client.post("/predict/breast-cancer/upload", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["module"] == "module_4_breast_cancer"


def test_predict_breast_cancer_from_path_scaffold() -> None:
    r = client.post("/predict/breast-cancer", json={"image_path": "nonexistent.png"})
    assert r.status_code == 200
    data = r.json()
    assert data["module"] == "module_4_breast_cancer"


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint multimodal /predict
# ═══════════════════════════════════════════════════════════════════════════════

def test_predict_multimodal_nutrition_only() -> None:
    r = client.post("/predict", json={"nutrition_data": _NUTRITION_PAYLOAD})
    assert r.status_code == 200
    data = r.json()
    assert data["application"] == "KANEA"
    assert "module_2_biometry" in data["results"]


def test_predict_multimodal_empty_returns_no_input() -> None:
    r = client.post("/predict", json={})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "no_input_provided"


def test_predict_multimodal_forensic_and_nutrition() -> None:
    r = client.post("/predict", json={
        "nutrition_data": _NUTRITION_PAYLOAD,
        "bioid_data":     _BIOID_PAYLOAD,
    })
    assert r.status_code == 200
    data = r.json()
    assert "module_2_biometry"  in data["results"]
    assert "module_3_forensic"  in data["results"]
