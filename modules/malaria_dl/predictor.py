from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_KANEA_ROOT     = Path(__file__).resolve().parents[2]
ONNX_PATH       = _KANEA_ROOT / "models" / "deep_learning" / "malaria_model.onnx"
FC_WEIGHTS_PATH = _KANEA_ROOT / "models" / "deep_learning" / "malaria_fc_weights.npy"
CLASSES         = ["Parasitized", "Uninfected"]

_IMG_SIZE       = 224
_MEAN           = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD            = np.array([0.229, 0.224, 0.225],  dtype=np.float32)
_MAX_FILE_MB    = 10
_ALLOWED_EXTS   = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_MIN_BRIGHTNESS = 8.0
_MIN_BLUR_SCORE = 15.0


# ── Validation image ──────────────────────────────────────────────────────────

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

    gray   = arr.mean(axis=2)
    padded = np.pad(gray, 1, mode="edge")
    lap    = (-4 * gray
              + padded[:-2, 1:-1] + padded[2:, 1:-1]
              + padded[1:-1, :-2] + padded[1:-1, 2:])
    blur_score = float(lap.var())
    if blur_score < _MIN_BLUR_SCORE:
        return {"valid": False, "error": f"Image trop floue (score : {blur_score:.1f}, min requis : {_MIN_BLUR_SCORE})"}

    return {
        "valid":      True,
        "size_mb":    round(size_mb, 2),
        "format":     path.suffix.lower(),
        "brightness": round(mean_brightness, 1),
        "blur_score": round(blur_score, 1),
    }


# ── CAM (Class Activation Map) ────────────────────────────────────────────────

def _generate_cam(
    features: np.ndarray,
    class_idx: int,
    orig_rgb: np.ndarray,
) -> tuple[str | None, np.ndarray | None]:
    """
    CAM = equivalent Grad-CAM pour ResNet+GAP.
    Retourne (overlay_b64, cam_arr [0-1]) ou (None, None).
    features : [1, 512, 7, 7]
    """
    try:
        candidates = [FC_WEIGHTS_PATH, Path("models/deep_learning/malaria_fc_weights.npy")]
        fc_w = None
        for p in candidates:
            if p.exists():
                fc_w = np.load(str(p))
                break
        if fc_w is None:
            return None, None

        weights = fc_w[class_idx]
        feat    = features[0]
        cam     = np.tensordot(weights, feat, axes=([0], [0]))
        cam     = np.maximum(cam, 0)

        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min) if cam_max > cam_min else np.zeros_like(cam)

        cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (_IMG_SIZE, _IMG_SIZE), Image.LANCZOS
        )
        cam_arr = np.asarray(cam_pil, dtype=np.float32) / 255.0

        try:
            import matplotlib.cm as mcm
            heatmap = (mcm.jet(cam_arr)[:, :, :3] * 255).astype(np.uint8)
        except ImportError:
            heatmap = np.zeros((_IMG_SIZE, _IMG_SIZE, 3), dtype=np.uint8)
            heatmap[:, :, 0] = (cam_arr * 255).astype(np.uint8)

        overlay = np.clip(0.55 * orig_rgb.astype(np.float32) + 0.45 * heatmap.astype(np.float32), 0, 255).astype(np.uint8)

        buf = BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii"), cam_arr

    except Exception:
        return None, None


# ── Analyses cliniques avancées ───────────────────────────────────────────────

def _estimate_parasitemia(cam_arr: np.ndarray, pred: str, conf: float) -> dict[str, Any]:
    """
    Estimation de la parasitémie via la surface d'activation CAM.
    Méthode : surface activée (cam > 0.5) / surface totale × facteur RBC.
    Non certifiée cliniquement — estimation algorithmique.
    """
    if pred == "Uninfected":
        return {"infected_cells": 0, "total_cells": 120, "percentage": 0.0,
                "severity": "Aucune infection", "method": "CAM_estimation"}

    activated      = (cam_arr > 0.5).sum()
    total_px       = cam_arr.size
    infection_frac = float(activated) / total_px

    estimated_total  = 120
    infected         = max(1, int(estimated_total * infection_frac * conf))
    infected         = min(infected, estimated_total)
    pct              = round((infected / estimated_total) * 100, 2)

    if pct < 1:       severity = "Faible (< 1%)"
    elif pct < 5:     severity = "Modérée (1–5%)"
    elif pct < 10:    severity = "Sévère (5–10%)"
    else:             severity = "Critique (> 10%)"

    return {
        "infected_cells": infected,
        "total_cells":    estimated_total,
        "percentage":     pct,
        "severity":       severity,
        "method":         "CAM_estimation",
    }


def _estimate_species(conf: float) -> dict[str, Any]:
    """
    Probabilités d'espèce basées sur les priors épidémiologiques Afrique subsaharienne.
    P. falciparum représente ~90-95% des cas en Afrique.
    Ces probabilités sont des estimations — non issues d'un modèle espèces dédié.
    """
    pf       = round(min(0.94, max(0.72, 0.72 + (conf - 0.5) * 0.44)), 3)
    remain   = round(1.0 - pf, 3)
    return {
        "plasmodium_falciparum": pf,
        "plasmodium_vivax":      round(remain * 0.40, 3),
        "plasmodium_malariae":   round(remain * 0.30, 3),
        "plasmodium_ovale":      round(remain * 0.20, 3),
        "plasmodium_knowlesi":   round(remain * 0.10, 3),
        "dominant_species":      "Plasmodium falciparum",
        "epidemiological_note":  "Estimation basée sur épidémiologie Afrique subsaharienne",
    }


def _estimate_stage(cam_arr: np.ndarray) -> dict[str, Any]:
    """
    Estimation du stade parasitaire via analyse spatiale du pattern CAM.
    - Activation concentrée/petite → Ring stage (anneau)
    - Activation moyenne → Trophozoïte
    - Activation diffuse/large → Schizonte
    - Pattern allongé → Gamétocyte
    """
    activated = cam_arr > 0.55
    if not activated.any():
        return {"ring": 0.65, "trophozoite": 0.20, "schizont": 0.10,
                "gametocyte": 0.05, "dominant_stage": "Ring", "method": "CAM_spatial"}

    ys, xs  = np.where(activated)
    n_px    = len(ys)
    spread  = float(np.sqrt(np.var(xs) + np.var(ys))) / (_IMG_SIZE / 2) if n_px >= 3 else 0.0
    density = n_px / cam_arr.size

    if spread < 0.15 and density < 0.08:
        probs = {"ring": 0.72, "trophozoite": 0.18, "schizont": 0.07, "gametocyte": 0.03}
        stage = "Ring"
    elif spread < 0.30:
        probs = {"ring": 0.25, "trophozoite": 0.52, "schizont": 0.18, "gametocyte": 0.05}
        stage = "Trophozoite"
    elif spread < 0.50:
        probs = {"ring": 0.10, "trophozoite": 0.28, "schizont": 0.47, "gametocyte": 0.15}
        stage = "Schizont"
    else:
        probs = {"ring": 0.05, "trophozoite": 0.20, "schizont": 0.35, "gametocyte": 0.40}
        stage = "Gametocyte"

    return {**probs, "dominant_stage": stage, "spread_score": round(spread, 3),
            "method": "CAM_spatial"}


def _clinical_safety_flag(pred: str, conf: float) -> dict[str, Any]:
    """Signalement de sécurité clinique selon le niveau de confiance."""
    if pred == "Uninfected" and conf < 0.85:
        return {
            "flag":    "ATTENTION",
            "level":   "warning",
            "message": f"Confiance faible ({conf:.1%}) — Faux négatif possible. Validation biologiste recommandée.",
        }
    if conf < 0.70:
        return {
            "flag":    "INCERTAIN",
            "level":   "low_confidence",
            "message": f"Confiance insuffisante ({conf:.1%}) — Résultat non fiable. Répéter l'analyse.",
        }
    return {
        "flag":    "OK",
        "level":   "normal",
        "message": "Confiance suffisante pour aide à la décision.",
    }


# ── Réponse placeholder ───────────────────────────────────────────────────────

def _placeholder_response(image_name: str | None, status: str = "scaffold_ready") -> dict[str, Any]:
    return {
        "module":       "module_1_malaria",
        "task":         "automatic_malaria_detection",
        "model_name":   "resnet34_malaria_onnx",
        "input_image":  image_name,
        "prediction":   None,
        "confidence":   None,
        "status":       status,
    }


def _resolve_onnx_path() -> Path | None:
    for p in [ONNX_PATH, Path("models/deep_learning/malaria_model.onnx")]:
        if not p.exists():
            continue
        if p.stat().st_size < 500:
            return None
        return p
    return None


def _preprocess(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(image_path).convert("RGB").resize((_IMG_SIZE, _IMG_SIZE))
    rgb = np.asarray(img, dtype=np.uint8)
    arr = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[np.newaxis], rgb


def _softmax(logits: np.ndarray) -> dict[str, float]:
    exps  = np.exp(logits - np.max(logits))
    probs = exps / np.sum(exps)
    return {CLASSES[i]: round(float(p), 4) for i, p in enumerate(probs)}


# ── Point d'entrée principal ──────────────────────────────────────────────────

def predict_malaria(image_path: str | None = None) -> dict[str, Any]:
    """
    Pipeline clinique complet MalariaScan AI v3.0 :
    inférence ONNX + CAM + parasitémie + espèce + stade + sécurité clinique.
    """
    image_name = Path(image_path).name if image_path else None
    if not image_path:
        return _placeholder_response(image_name)

    validation = _validate_image(image_path)
    if not validation["valid"]:
        resp = _placeholder_response(image_name, "validation_error")
        resp["error"]      = validation["error"]
        resp["validation"] = validation
        return resp

    onnx_path = _resolve_onnx_path()
    if onnx_path is None:
        resp = _placeholder_response(image_name, "inference_error")
        lfs = ONNX_PATH if ONNX_PATH.exists() else Path("models/deep_learning/malaria_model.onnx")
        resp["error"] = (
            "Modèle ONNX non résolu — pointeur LFS non téléchargé."
            if (lfs.exists() and lfs.stat().st_size < 500)
            else f"Modèle ONNX introuvable : {ONNX_PATH}"
        )
        return resp

    try:
        import onnxruntime as ort
    except ImportError:
        resp = _placeholder_response(image_name, "inference_error")
        resp["error"] = "onnxruntime non installé"
        return resp

    try:
        session  = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        tensor, orig_rgb = _preprocess(image_path)
        outputs  = session.run(None, {session.get_inputs()[0].name: tensor})
        logits   = outputs[0][0]
        features = outputs[1] if len(outputs) > 1 else None
        probs    = _softmax(logits)
        pred     = max(probs, key=probs.get)
        conf     = probs[pred]
        class_idx = CLASSES.index(pred)
    except Exception as exc:
        resp = _placeholder_response(image_name, "inference_error")
        resp["error"] = f"Erreur ONNX : {exc}"
        return resp

    # CAM
    heatmap_b64, cam_arr = _generate_cam(features, class_idx, orig_rgb) if features is not None else (None, None)

    # Analyses cliniques avancées (uniquement si infecté ou si utile cliniquement)
    parasitemia    = _estimate_parasitemia(cam_arr if cam_arr is not None else np.zeros((_IMG_SIZE, _IMG_SIZE)), pred, conf)
    species        = _estimate_species(conf)         if pred == "Parasitized" else None
    stage          = _estimate_stage(cam_arr)         if (pred == "Parasitized" and cam_arr is not None) else None
    safety         = _clinical_safety_flag(pred, conf)

    return {
        "module":          "module_1_malaria",
        "task":            "automatic_malaria_detection",
        "model_name":      "resnet34_malaria_onnx",
        "model_version":   "v3.0",
        "library":         "onnxruntime",
        "input_image":     image_name,

        # Résultat principal
        "prediction":      pred,
        "confidence":      conf,
        "probabilities":   probs,

        # Analyses cliniques
        "parasitemia":     parasitemia,
        "species_prediction": species,
        "parasite_stage":  stage,

        # Sécurité clinique
        "clinical_safety": safety,

        # Explainability CAM
        "explainability":  {
            "method":      "CAM",
            "status":      "generated" if heatmap_b64 else "not_generated",
            "heatmap_b64": heatmap_b64,
        },

        "deployment_mode": "streamlit_cloud_ready",
        "inference_backend": "onnx",
        "status":          "model_loaded",
    }
