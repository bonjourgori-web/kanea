from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# ── Intègre malaria_scan_ai (multibio-ai) dans le path ───────────────────────
_MULTIBIO   = Path(__file__).resolve().parents[3] / "multibio-ai"
_KANEA_ROOT = Path(__file__).resolve().parents[2]

if _MULTIBIO.exists() and str(_MULTIBIO) not in sys.path:
    sys.path.insert(0, str(_MULTIBIO))

ONNX_PATH = _KANEA_ROOT / "models" / "deep_learning" / "malaria_model.onnx"
PTH_PATH  = _KANEA_ROOT / "models" / "deep_learning" / "malaria_model.pth"
CLASSES   = ["Parasitized", "Uninfected"]

_IMG_SIZE = 224
_MEAN     = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD      = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _placeholder_response(image_name: str | None, status: str = "scaffold_ready") -> dict[str, Any]:
    return {
        "module":                   "module_1_malaria",
        "task":                     "automatic_malaria_detection",
        "model_family":             "deep_learning",
        "model_name":               "efficientnet_b0_malaria",
        "library":                  "onnxruntime",
        "expected_model_path":      str(ONNX_PATH).replace("\\", "/"),
        "input_image":              image_name,
        "prediction":               None,
        "confidence":               None,
        "supported_output_classes": CLASSES,
        "explainability":           {"method": "Grad-CAM", "status": "not_generated"},
        "deployment_mode":          "streamlit_cloud_ready",
        "status":                   status,
    }


def _preprocess_numpy(image_path: str) -> np.ndarray:
    """Preprocessing PIL + numpy uniquement — ne requiert pas torch."""
    img = Image.open(image_path).convert("RGB").resize((_IMG_SIZE, _IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)


def _softmax(logits: np.ndarray) -> dict[str, float]:
    exps = np.exp(logits - np.max(logits))
    probs = exps / np.sum(exps)
    return {CLASSES[i]: round(float(p), 4) for i, p in enumerate(probs)}


def _run_onnx(image_path: str) -> dict[str, Any]:
    """Inférence ONNX directe via numpy — aucune dépendance torch."""
    # Chemin absolu (depuis la racine KANEA) ou relatif au CWD (Streamlit Cloud)
    onnx_path = ONNX_PATH if ONNX_PATH.exists() else Path("models/deep_learning/malaria_model.onnx")

    if not onnx_path.exists():
        return {"error": f"ONNX model introuvable : {ONNX_PATH}"}

    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime non installé"}

    try:
        session   = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        tensor    = _preprocess_numpy(image_path)
        input_nm  = session.get_inputs()[0].name
        logits    = session.run(None, {input_nm: tensor})[0][0]
        probs     = _softmax(logits)
        pred      = max(probs, key=probs.get)
        return {"prediction": pred, "confidence": probs[pred], "probabilities": probs}
    except Exception as exc:
        return {"error": str(exc)}


def predict_malaria(image_path: str | None = None) -> dict[str, Any]:
    """Inférence malaria.

    Priorité :
    1. ONNX direct avec preprocessing numpy (sans torch)
    2. MalariaInferenceEngine (multibio-ai, nécessite torch)
    3. Fallback placeholder
    """
    image_name = Path(image_path).name if image_path else None
    if not image_path:
        return _placeholder_response(image_name)

    # ── Voie 1 : ONNX direct (numpy + PIL, pas de torch) ────────────────────
    onnx_result = _run_onnx(image_path)
    if "prediction" in onnx_result:
        r = onnx_result
        return {
            "module":                   "module_1_malaria",
            "task":                     "automatic_malaria_detection",
            "model_family":             "deep_learning",
            "model_name":               "efficientnet_b0_malaria_onnx",
            "library":                  "onnxruntime",
            "expected_model_path":      str(ONNX_PATH).replace("\\", "/"),
            "input_image":              image_name,
            "prediction":               r["prediction"],
            "confidence":               r["confidence"],
            "probabilities":            r["probabilities"],
            "supported_output_classes": CLASSES,
            "explainability":           {"method": "Grad-CAM", "status": "not_generated"},
            "deployment_mode":          "streamlit_cloud_ready",
            "inference_backend":        "onnx",
            "model_version":            "v2.0",
            "status":                   "model_loaded",
        }

    # ── Voie 2 : MalariaInferenceEngine (multibio-ai) ───────────────────────
    try:
        from malaria_scan_ai.training.inference import MalariaInferenceEngine

        model_dir = ONNX_PATH.parent
        engine    = MalariaInferenceEngine(model_dir=model_dir)
        result    = engine.predict_file(Path(image_path))
        return {
            "module":                   "module_1_malaria",
            "task":                     "automatic_malaria_detection",
            "model_family":             "deep_learning",
            "model_name":               f"efficientnet_b0_malaria_{result.backend}",
            "library":                  result.backend,
            "expected_model_path":      str(ONNX_PATH).replace("\\", "/"),
            "input_image":              image_name,
            "prediction":               result.prediction,
            "confidence":               result.confidence,
            "probabilities":            result.probabilities,
            "supported_output_classes": CLASSES,
            "explainability":           {"method": "Grad-CAM", "status": "not_generated"},
            "deployment_mode":          "streamlit_cloud_ready",
            "inference_backend":        result.backend,
            "model_version":            result.model_version,
            "status":                   "model_loaded",
        }
    except Exception:
        pass

    # ── Voie 3 : fallback ────────────────────────────────────────────────────
    resp = _placeholder_response(image_name, "inference_error")
    resp["error"] = onnx_result.get("error", "Tous les backends ont échoué")
    return resp
