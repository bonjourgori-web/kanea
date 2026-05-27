from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_KANEA_ROOT = Path(__file__).resolve().parents[2]

ONNX_PATH = _KANEA_ROOT / "models" / "deep_learning" / "malaria_model.onnx"
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


def _resolve_onnx_path() -> Path | None:
    """Retourne le chemin vers le fichier ONNX si valide (pas un pointeur LFS)."""
    candidates = [
        ONNX_PATH,
        Path("models/deep_learning/malaria_model.onnx"),  # relatif CWD
    ]
    for p in candidates:
        if not p.exists():
            continue
        # Détecter un pointeur LFS (fichier < 200 bytes ou commence par "version https://git-lfs")
        if p.stat().st_size < 500:
            return None  # pointeur LFS — fichier non résolu
        return p
    return None


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


def predict_malaria(image_path: str | None = None) -> dict[str, Any]:
    """Inférence malaria via ONNX Runtime (numpy/PIL, sans torch)."""
    image_name = Path(image_path).name if image_path else None
    if not image_path:
        return _placeholder_response(image_name)

    # ── Résoudre le modèle ONNX ──────────────────────────────────────────────
    onnx_path = _resolve_onnx_path()

    if onnx_path is None:
        resp = _placeholder_response(image_name, "inference_error")
        # Vérifier si c'est un pointeur LFS
        lfs_check = ONNX_PATH if ONNX_PATH.exists() else Path("models/deep_learning/malaria_model.onnx")
        if lfs_check.exists() and lfs_check.stat().st_size < 500:
            resp["error"] = (
                "Modèle ONNX non résolu — fichier Git LFS non téléchargé. "
                "Le modèle (81 MB) doit être un blob git régulier, pas un pointeur LFS."
            )
        else:
            resp["error"] = f"Modèle ONNX introuvable : {ONNX_PATH}"
        return resp

    # ── Inférence ONNX ───────────────────────────────────────────────────────
    try:
        import onnxruntime as ort
    except ImportError:
        resp = _placeholder_response(image_name, "inference_error")
        resp["error"] = "onnxruntime non installé — ajoutez-le dans requirements.txt"
        return resp

    try:
        session  = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        tensor   = _preprocess_numpy(image_path)
        input_nm = session.get_inputs()[0].name
        logits   = session.run(None, {input_nm: tensor})[0][0]
        probs    = _softmax(logits)
        pred     = max(probs, key=probs.get)
        conf     = probs[pred]
    except Exception as exc:
        resp = _placeholder_response(image_name, "inference_error")
        resp["error"] = f"Erreur ONNX : {exc}"
        return resp

    return {
        "module":                   "module_1_malaria",
        "task":                     "automatic_malaria_detection",
        "model_family":             "deep_learning",
        "model_name":               "efficientnet_b0_malaria_onnx",
        "library":                  "onnxruntime",
        "expected_model_path":      str(ONNX_PATH).replace("\\", "/"),
        "input_image":              image_name,
        "prediction":               pred,
        "confidence":               conf,
        "probabilities":            probs,
        "supported_output_classes": CLASSES,
        "explainability":           {"method": "Grad-CAM", "status": "not_generated"},
        "deployment_mode":          "streamlit_cloud_ready",
        "inference_backend":        "onnx",
        "model_version":            "v2.0",
        "status":                   "model_loaded",
    }
