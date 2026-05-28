from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_KANEA_ROOT = Path(__file__).resolve().parents[2]

ONNX_PATH     = _KANEA_ROOT / "models" / "deep_learning" / "malaria_model.onnx"
FC_WEIGHTS_PATH = _KANEA_ROOT / "models" / "deep_learning" / "malaria_fc_weights.npy"
CLASSES       = ["Parasitized", "Uninfected"]

_IMG_SIZE = 224
_MEAN     = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD      = np.array([0.229, 0.224, 0.225],  dtype=np.float32)


_MAX_FILE_MB   = 10
_ALLOWED_EXTS  = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_MIN_BRIGHTNESS = 8.0
_MIN_BLUR_SCORE = 15.0


def _validate_image(image_path: str) -> dict[str, Any]:
    """Validation pré-inférence : taille, format, corruption, flou, image noire."""
    path = Path(image_path)

    if not path.exists():
        return {"valid": False, "error": f"Fichier introuvable : {path.name}"}

    size_mb = path.stat().st_size / 1_000_000
    if size_mb > _MAX_FILE_MB:
        return {"valid": False, "error": f"Fichier trop volumineux ({size_mb:.1f} MB, max {_MAX_FILE_MB} MB)"}

    if path.suffix.lower() not in _ALLOWED_EXTS:
        return {"valid": False, "error": f"Format non supporté : {path.suffix}. Acceptés : JPEG, PNG, TIFF"}

    try:
        img = Image.open(image_path)
        img.verify()
    except Exception as exc:
        return {"valid": False, "error": f"Image corrompue ou illisible : {exc}"}

    try:
        arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    except Exception as exc:
        return {"valid": False, "error": f"Impossible de décoder l'image : {exc}"}

    mean_brightness = float(arr.mean())
    if mean_brightness < _MIN_BRIGHTNESS:
        return {"valid": False, "error": f"Image noire ou vide (luminosité : {mean_brightness:.1f}/255)"}

    # Blur score via variance du gradient (Laplacian approx)
    gray    = arr.mean(axis=2)
    padded  = np.pad(gray, 1, mode="edge")
    lap     = (-4 * gray
               + padded[:-2, 1:-1] + padded[2:, 1:-1]
               + padded[1:-1, :-2] + padded[1:-1, 2:])
    blur_score = float(lap.var())
    if blur_score < _MIN_BLUR_SCORE:
        return {"valid": False, "error": f"Image trop floue (score : {blur_score:.1f}, min requis : {_MIN_BLUR_SCORE})"}

    return {
        "valid":       True,
        "size_mb":     round(size_mb, 2),
        "format":      path.suffix.lower(),
        "brightness":  round(mean_brightness, 1),
        "blur_score":  round(blur_score, 1),
    }


def _placeholder_response(image_name: str | None, status: str = "scaffold_ready") -> dict[str, Any]:
    return {
        "module":                   "module_1_malaria",
        "task":                     "automatic_malaria_detection",
        "model_family":             "deep_learning",
        "model_name":               "resnet34_malaria_onnx",
        "library":                  "onnxruntime",
        "expected_model_path":      str(ONNX_PATH).replace("\\", "/"),
        "input_image":              image_name,
        "prediction":               None,
        "confidence":               None,
        "supported_output_classes": CLASSES,
        "explainability":           {"method": "CAM", "status": "not_generated"},
        "deployment_mode":          "streamlit_cloud_ready",
        "status":                   status,
    }


def _resolve_onnx_path() -> Path | None:
    for p in [ONNX_PATH, Path("models/deep_learning/malaria_model.onnx")]:
        if not p.exists():
            continue
        if p.stat().st_size < 500:
            return None  # pointeur LFS non résolu
        return p
    return None


def _preprocess(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (tensor NCHW float32, RGB array HWC uint8 224×224)."""
    img = Image.open(image_path).convert("RGB").resize((_IMG_SIZE, _IMG_SIZE))
    rgb = np.asarray(img, dtype=np.uint8)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[np.newaxis], rgb  # (1,3,H,W) + (H,W,3)


def _softmax(logits: np.ndarray) -> dict[str, float]:
    exps = np.exp(logits - np.max(logits))
    probs = exps / np.sum(exps)
    return {CLASSES[i]: round(float(p), 4) for i, p in enumerate(probs)}


def _generate_cam_b64(
    features: np.ndarray,
    class_idx: int,
    orig_rgb: np.ndarray,
) -> str | None:
    """
    Class Activation Map (CAM) = equivalent visuel de Grad-CAM pour ResNet+GAP.
    features : [1, 512, 7, 7]
    class_idx: 0=Parasitized, 1=Uninfected
    Retourne l'overlay base64 PNG ou None si poids FC manquants.
    """
    try:
        # Charger les poids FC sauvegardés [2, 512]
        candidates = [FC_WEIGHTS_PATH, Path("models/deep_learning/malaria_fc_weights.npy")]
        fc_w = None
        for p in candidates:
            if p.exists():
                fc_w = np.load(str(p))
                break
        if fc_w is None:
            return None

        weights = fc_w[class_idx]          # [512]
        feat    = features[0]              # [512, 7, 7]

        # CAM = somme pondérée des feature maps par les poids FC
        cam = np.tensordot(weights, feat, axes=([0], [0]))  # [7, 7]
        cam = np.maximum(cam, 0)           # ReLU

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Upsample 7×7 → 224×224
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (_IMG_SIZE, _IMG_SIZE), Image.LANCZOS
        )
        cam_arr = np.asarray(cam_pil, dtype=np.float32) / 255.0  # [224, 224]

        # Colormap jet via matplotlib (disponible sur Streamlit Cloud)
        try:
            import matplotlib.cm as cm
            heatmap = (cm.jet(cam_arr)[:, :, :3] * 255).astype(np.uint8)
        except ImportError:
            # Fallback: colormap rouge manuel
            heatmap = np.zeros((_IMG_SIZE, _IMG_SIZE, 3), dtype=np.uint8)
            heatmap[:, :, 0] = (cam_arr * 255).astype(np.uint8)

        # Fusion 55% original + 45% heatmap
        overlay = (0.55 * orig_rgb.astype(np.float32) + 0.45 * heatmap.astype(np.float32))
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        buf = BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    except Exception:
        return None


def predict_malaria(image_path: str | None = None) -> dict[str, Any]:
    """Inférence malaria via ONNX Runtime + CAM explainability (numpy/PIL, sans torch)."""
    image_name = Path(image_path).name if image_path else None
    if not image_path:
        return _placeholder_response(image_name)

    # Validation image avant inférence
    validation = _validate_image(image_path)
    if not validation["valid"]:
        resp = _placeholder_response(image_name, "validation_error")
        resp["error"]      = validation["error"]
        resp["validation"] = validation
        return resp

    onnx_path = _resolve_onnx_path()
    if onnx_path is None:
        resp = _placeholder_response(image_name, "inference_error")
        lfs_check = ONNX_PATH if ONNX_PATH.exists() else Path("models/deep_learning/malaria_model.onnx")
        if lfs_check.exists() and lfs_check.stat().st_size < 500:
            resp["error"] = (
                "Modèle ONNX non résolu — fichier Git LFS non téléchargé. "
                "Le modèle (81 MB) doit être un blob git régulier."
            )
        else:
            resp["error"] = f"Modèle ONNX introuvable : {ONNX_PATH}"
        return resp

    try:
        import onnxruntime as ort
    except ImportError:
        resp = _placeholder_response(image_name, "inference_error")
        resp["error"] = "onnxruntime non installé — ajoutez-le dans requirements.txt"
        return resp

    try:
        session  = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        tensor, orig_rgb = _preprocess(image_path)
        input_nm = session.get_inputs()[0].name
        outputs  = session.run(None, {input_nm: tensor})

        logits   = outputs[0][0]           # [2]
        features = outputs[1] if len(outputs) > 1 else None  # [1, 512, 7, 7]

        probs    = _softmax(logits)
        pred     = max(probs, key=probs.get)
        conf     = probs[pred]
        class_idx = CLASSES.index(pred)

    except Exception as exc:
        resp = _placeholder_response(image_name, "inference_error")
        resp["error"] = f"Erreur ONNX : {exc}"
        return resp

    # Génération CAM
    heatmap_b64 = None
    if features is not None:
        heatmap_b64 = _generate_cam_b64(features, class_idx, orig_rgb)

    expl_status = "generated" if heatmap_b64 else "not_generated"

    return {
        "module":                   "module_1_malaria",
        "task":                     "automatic_malaria_detection",
        "model_family":             "deep_learning",
        "model_name":               "resnet34_malaria_onnx",
        "library":                  "onnxruntime",
        "expected_model_path":      str(ONNX_PATH).replace("\\", "/"),
        "input_image":              image_name,
        "prediction":               pred,
        "confidence":               conf,
        "probabilities":            probs,
        "supported_output_classes": CLASSES,
        "explainability":           {
            "method":      "CAM",
            "status":      expl_status,
            "heatmap_b64": heatmap_b64,
        },
        "deployment_mode":          "streamlit_cloud_ready",
        "inference_backend":        "onnx",
        "model_version":            "v2.1",
        "status":                   "model_loaded",
    }
