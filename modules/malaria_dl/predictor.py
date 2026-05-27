from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ── Intègre malaria_scan_ai (multibio-ai) dans le path ───────────────────────
_MULTIBIO = Path(__file__).resolve().parents[3] / "multibio-ai"
if _MULTIBIO.exists() and str(_MULTIBIO) not in sys.path:
    sys.path.insert(0, str(_MULTIBIO))

ONNX_PATH = Path("models/deep_learning/malaria_model.onnx")
PTH_PATH  = Path("models/deep_learning/malaria_model.pth")
CLASSES   = ["Parasitized", "Uninfected"]


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


def predict_malaria(image_path: str | None = None) -> dict[str, Any]:
    """Délègue à MalariaInferenceEngine (malaria_scan_ai).

    Priorité : ONNX → TorchScript → PyTorch .pth → fallback.
    Même logique que multibio-ai/malaria_scan_ai/training/inference.py.
    """
    image_name = Path(image_path).name if image_path else None
    if not image_path:
        return _placeholder_response(image_name)

    try:
        from malaria_scan_ai.training.inference import MalariaInferenceEngine

        # Pointe sur le dossier models/ de KANEA
        model_dir = Path.cwd() / "models" / "deep_learning"
        engine = MalariaInferenceEngine(model_dir=model_dir)
        result = engine.predict_file(Path(image_path))

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

    except Exception as exc:
        resp = _placeholder_response(image_name, "inference_error")
        resp["error"] = str(exc)
        return resp
