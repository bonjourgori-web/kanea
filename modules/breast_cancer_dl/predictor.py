from __future__ import annotations

import base64
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_KANEA_ROOT   = Path(__file__).resolve().parents[2]
ONNX_PATH     = _KANEA_ROOT / "models" / "deep_learning" / "breast_cancer_model.onnx"
FC_PATH       = _KANEA_ROOT / "models" / "deep_learning" / "breast_cancer_fc_weights.npy"
MODEL_PATH    = Path("models/deep_learning/breast_cancer_model.pth")   # kept for compat
CLASSES       = ["Normal", "Benign", "Malignant"]
IMG_SIZE      = 224
_MEAN         = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD          = np.array([0.229, 0.224, 0.225], dtype=np.float32)

RISK_COLORS = {"Normal": "#27AE60", "Benign": "#F39C12", "Malignant": "#E74C3C"}

_MAX_FILE_MB   = 20
_ALLOWED_EXTS  = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".dcm"}


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_image(image_path: str) -> dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        return {"valid": False, "error": f"Fichier introuvable : {path.name}"}
    size_mb = path.stat().st_size / 1_000_000
    if size_mb > _MAX_FILE_MB:
        return {"valid": False, "error": f"Fichier trop volumineux ({size_mb:.1f} MB, max {_MAX_FILE_MB} MB)"}
    if path.suffix.lower() not in _ALLOWED_EXTS:
        return {"valid": False, "error": f"Format non supporté : {path.suffix}. Acceptés : JPEG, PNG, TIFF, DICOM"}
    try:
        img = Image.open(image_path); img.verify()
    except Exception as exc:
        return {"valid": False, "error": f"Image corrompue : {exc}"}
    try:
        arr = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        if arr.mean() < 5:
            return {"valid": False, "error": "Image noire ou vide — vérifier le fichier mammographie"}
    except Exception as exc:
        return {"valid": False, "error": f"Lecture impossible : {exc}"}
    return {"valid": True, "size_mb": round(size_mb, 2)}


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (tensor NCHW float32, RGB array HWC uint8)."""
    img   = Image.open(image_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    gray  = np.asarray(img, dtype=np.uint8)
    rgb   = np.stack([gray, gray, gray], axis=2)          # grayscale → 3 canaux
    arr   = rgb.astype(np.float32) / 255.0
    arr   = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[np.newaxis], rgb


# ── CAM ───────────────────────────────────────────────────────────────────────

def _generate_cam(
    features: np.ndarray,
    class_idx: int,
    orig_rgb: np.ndarray,
) -> tuple[str | None, np.ndarray | None]:
    """CAM via EfficientNet-B0 features [1, 1280, 7, 7] + FC weights [3, 1280]."""
    try:
        candidates = [FC_PATH, Path("models/deep_learning/breast_cancer_fc_weights.npy")]
        fc_w = None
        for p in candidates:
            if p.exists():
                fc_w = np.load(str(p)); break
        if fc_w is None:
            return None, None

        weights = fc_w[class_idx]                                      # [1280]
        feat    = features[0]                                           # [1280, 7, 7]
        cam     = np.tensordot(weights, feat, axes=([0], [0]))         # [7, 7]
        cam     = np.maximum(cam, 0)
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min) if cam_max > cam_min else np.zeros_like(cam)

        cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        cam_arr = np.asarray(cam_pil, dtype=np.float32) / 255.0

        try:
            import matplotlib.cm as mcm
            heatmap = (mcm.jet(cam_arr)[:, :, :3] * 255).astype(np.uint8)
        except ImportError:
            heatmap = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            heatmap[:, :, 0] = (cam_arr * 255).astype(np.uint8)

        overlay = np.clip(0.55 * orig_rgb.astype(np.float32) + 0.45 * heatmap.astype(np.float32), 0, 255).astype(np.uint8)
        buf = BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii"), cam_arr
    except Exception:
        return None, None


# ── Analyses cliniques avancées ───────────────────────────────────────────────

def _estimate_grade(cam_arr: np.ndarray | None, conf: float) -> dict[str, Any]:
    """Grade tumoral (1/2/3) via pattern spatial CAM + confiance."""
    if cam_arr is None:
        spread = 0.3
    else:
        activated = cam_arr > 0.55
        if activated.any():
            ys, xs = np.where(activated)
            spread = float(np.sqrt(np.var(xs) + np.var(ys))) / (IMG_SIZE / 2) if len(ys) >= 3 else 0.2
        else:
            spread = 0.1

    if spread < 0.20 and conf < 0.90:
        probs = {"grade1": 0.60, "grade2": 0.30, "grade3": 0.10}
        grade = "Grade 1"
    elif spread < 0.40:
        probs = {"grade1": 0.15, "grade2": 0.55, "grade3": 0.30}
        grade = "Grade 2"
    else:
        probs = {"grade1": 0.05, "grade2": 0.25, "grade3": 0.70}
        grade = "Grade 3"

    grade_notes = {
        "Grade 1": "Bien différencié — croissance lente, pronostic favorable",
        "Grade 2": "Modérément différencié — croissance intermédiaire",
        "Grade 3": "Peu différencié — croissance rapide, traitement agressif requis",
    }
    return {**probs, "dominant_grade": grade, "spread_score": round(spread, 3),
            "clinical_note": grade_notes[grade], "method": "CAM_spatial_estimation"}


def _estimate_tnm(conf: float, grade: str) -> dict[str, Any]:
    """Estimation TNM stage selon confiance + grade."""
    if conf > 0.95 and grade == "Grade 3":
        T, N, M, stage = "T2", "N1", "M0", "Stage II"
    elif conf > 0.85:
        T, N, M, stage = "T1", "N1", "M0", "Stage II"
    elif conf > 0.70:
        T, N, M, stage = "T1", "N0", "M0", "Stage I"
    else:
        T, N, M, stage = "Tis", "N0", "M0", "Stage 0"

    stage_notes = {
        "Stage 0":  "Carcinome in situ — traitement chirurgical conservateur",
        "Stage I":  "Tumeur localisée < 2 cm — excellent pronostic (survie 5 ans > 95%)",
        "Stage II": "Tumeur 2–5 cm ou N+ — traitement multimodal (chir + chimio/radio)",
        "Stage III": "Tumeur avancée locale — traitement néoadjuvant recommandé",
        "Stage IV": "Métastases à distance — traitement palliatif/systémique",
    }
    return {
        "T": T, "N": N, "M": M,
        "clinical_stage": stage,
        "stage_note": stage_notes.get(stage, "—"),
        "method": "epidemiological_estimation",
    }


def _estimate_receptors(conf: float, grade: str) -> dict[str, Any]:
    """
    Statut récepteurs hormonaux (ER/PR/HER2) — priors épidémiologiques.
    ~70% cancers du sein sont ER+, ~60% PR+, ~15-20% HER2+.
    """
    is_grade3 = grade == "Grade 3"
    er_pct  = round(max(35, 90 - (conf * 20) - (15 if is_grade3 else 0)))
    pr_pct  = round(max(25, 75 - (conf * 15) - (10 if is_grade3 else 0)))
    her2    = "Positif" if (is_grade3 and conf > 0.85) else "Négatif"
    triple_neg = er_pct < 10 and pr_pct < 10 and her2 == "Négatif"

    return {
        "ER_percentage":   er_pct,
        "PR_percentage":   pr_pct,
        "HER2_status":     her2,
        "ER_status":       "Positif" if er_pct >= 10 else "Négatif",
        "PR_status":       "Positif" if pr_pct >= 10 else "Négatif",
        "triple_negative": triple_neg,
        "phenotype":       "Triple négatif" if triple_neg else
                           ("HER2+" if her2 == "Positif" else "Luminal A/B"),
        "note":            "Estimation probabiliste — IHC obligatoire pour confirmation",
    }


def _estimate_ki67(grade: str, conf: float) -> dict[str, Any]:
    """Ki67 index de prolifération selon grade estimé."""
    if grade == "Grade 1":
        pct, risk = round(max(3, 14 - int(conf * 8))), "Faible prolifération (< 14%)"
    elif grade == "Grade 2":
        pct, risk = round(14 + int(conf * 16)), "Prolifération intermédiaire (14–30%)"
    else:
        pct, risk = round(min(95, 30 + int(conf * 50))), "Haute prolifération (> 30%)"
    return {
        "percentage": pct,
        "risk_category": risk,
        "interpretation": "Haute agressivité — traitement intensif" if pct > 30 else
                          "Agressivité modérée — surveillance renforcée" if pct > 14 else
                          "Faible agressivité — suivi standard",
        "method": "grade_based_estimation",
    }


def _estimate_subtype(grade: str) -> dict[str, Any]:
    """Sous-type histologique — distribution épidémiologique (NST ductal ~75%)."""
    if grade == "Grade 3":
        probs = {"ductal_nst": 0.78, "lobular": 0.10, "mucinous": 0.04, "papillary": 0.03, "other": 0.05}
    else:
        probs = {"ductal_nst": 0.70, "lobular": 0.15, "mucinous": 0.08, "papillary": 0.04, "other": 0.03}
    dominant = max(probs, key=probs.get)
    labels = {"ductal_nst": "Carcinome canalaire NST", "lobular": "Carcinome lobulaire",
              "mucinous": "Carcinome mucineux", "papillary": "Carcinome papillaire", "other": "Autre"}
    return {
        **probs,
        "dominant_subtype":  labels.get(dominant, dominant),
        "dominant_key":      dominant,
        "epidemiological_note": "Distribution basée sur priors SEER/épidémio — biopsie nécessaire",
    }


def _clinical_safety_flag(pred: str, conf: float) -> dict[str, Any]:
    """Alerte sécurité clinique."""
    if pred == "Normal" and conf < 0.85:
        return {"flag": "ATTENTION", "level": "warning",
                "message": f"Confiance faible ({conf:.1%}) — Faux négatif possible. Mammographie de contrôle recommandée."}
    if pred == "Benign" and conf < 0.75:
        return {"flag": "INCERTAIN", "level": "low_confidence",
                "message": f"Résultat incertain ({conf:.1%}) — Biopsie à envisager pour confirmation."}
    if pred == "Malignant":
        return {"flag": "URGENT", "level": "critical",
                "message": "Signe potentiel de malignité — Consultation oncologique urgente requise."}
    return {"flag": "OK", "level": "normal", "message": "Confiance suffisante — suivi standard."}


# ── Recommandations cliniques ─────────────────────────────────────────────────

_RECOMMENDATIONS = {
    "Normal": {
        "urgency": "FAIBLE",
        "actions": [
            "Mammographie de dépistage annuelle recommandée après 40 ans",
            "Auto-examen mensuel des seins",
            "Consultation gynécologique annuelle",
        ],
        "follow_up": "12 mois",
    },
    "Benign": {
        "urgency": "MODÉRÉE",
        "actions": [
            "Échographie mammaire complémentaire pour caractériser la lésion",
            "Surveillance mammographique tous les 6 mois pendant 2 ans",
            "Consultation sénologie si doute clinique",
            "Biopsie si taille > 2 cm ou aspect atypique",
        ],
        "follow_up": "6 mois",
    },
    "Malignant": {
        "urgency": "CRITIQUE",
        "actions": [
            "Consultation oncologique URGENTE dans les 72h",
            "Biopsie percutanée (micro/macrobiopsie) pour confirmation histologique",
            "IRM mammaire bilatérale pour bilan d'extension local",
            "Bilan d'extension général (scanner thoraco-abdomino-pelvien, os)",
            "RCP multidisciplinaire (chirurgien, oncologue, radiothérapeute)",
            "Dosage marqueurs tumoraux (CA 15-3, ACE)",
        ],
        "follow_up": "IMMÉDIAT",
    },
}


# ── Placeholder ───────────────────────────────────────────────────────────────

def _placeholder_response(image_name: str | None, status: str = "scaffold_ready") -> dict[str, Any]:
    return {
        "module":       "module_4_breast_cancer",
        "task":         "breast_cancer_detection",
        "model_name":   "BreastCancer_AI_EfficientNet_B0",
        "model_version": "v3.0",
        "input_image":  image_name,
        "prediction":   None,
        "confidence":   None,
        "status":       status,
    }


# ── Inférence principale ──────────────────────────────────────────────────────

def predict_breast_cancer(image_path: str | None = None) -> dict[str, Any]:
    """
    Pipeline BreastCancer AI v3.0 :
    validation → ONNX inférence → CAM → grade/TNM/ER-PR-HER2/Ki67/subtype.
    """
    import time
    t0 = time.time()
    image_name = Path(image_path).name if image_path else None
    if not image_path:
        return _placeholder_response(image_name)

    validation = _validate_image(image_path)
    if not validation["valid"]:
        resp = _placeholder_response(image_name, "validation_error")
        resp["error"] = validation["error"]
        return resp

    # Résolution ONNX
    onnx_path = None
    for p in [ONNX_PATH, Path("models/deep_learning/breast_cancer_model.onnx")]:
        if p.exists() and p.stat().st_size > 500:
            onnx_path = p; break

    if onnx_path is None:
        # Fallback PyTorch si ONNX absent
        return _predict_pytorch(image_path, image_name, t0)

    try:
        import onnxruntime as ort
    except ImportError:
        return _predict_pytorch(image_path, image_name, t0)

    try:
        session  = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        tensor, orig_rgb = _preprocess(image_path)
        outputs  = session.run(None, {session.get_inputs()[0].name: tensor})
        logits   = outputs[0][0]
        features = outputs[1] if len(outputs) > 1 else None

        exps  = np.exp(logits - logits.max())
        probs = exps / exps.sum()
        pred_idx = int(probs.argmax())
        pred     = CLASSES[pred_idx]
        conf     = round(float(probs[pred_idx]), 4)
        probs_d  = {CLASSES[i]: round(float(p), 4) for i, p in enumerate(probs)}
    except Exception as exc:
        resp = _placeholder_response(image_name, "inference_error")
        resp["error"] = f"Erreur ONNX : {exc}"
        return resp

    # CAM
    heatmap_b64, cam_arr = _generate_cam(features, pred_idx, orig_rgb) if features is not None else (None, None)

    return _build_response(pred, conf, probs_d, heatmap_b64, cam_arr, image_name, t0)


def _build_response(
    pred: str, conf: float, probs_d: dict,
    heatmap_b64: str | None, cam_arr, image_name: str | None, t0: float,
) -> dict[str, Any]:
    import time

    # Analyses cliniques
    is_malignant = pred == "Malignant"
    grade  = _estimate_grade(cam_arr, conf)          if is_malignant else None
    tnm    = _estimate_tnm(conf, (grade or {}).get("dominant_grade", "Grade 2")) if is_malignant else None
    recep  = _estimate_receptors(conf, (grade or {}).get("dominant_grade", "Grade 2")) if is_malignant else None
    ki67   = _estimate_ki67((grade or {}).get("dominant_grade", "Grade 2"), conf) if is_malignant else None
    subtyp = _estimate_subtype((grade or {}).get("dominant_grade", "Grade 2")) if is_malignant else None
    safety = _clinical_safety_flag(pred, conf)
    reco   = _RECOMMENDATIONS.get(pred, {})

    return {
        "module":         "module_4_breast_cancer",
        "task":           "breast_cancer_detection",
        "model_name":     "BreastCancer_AI_EfficientNet_B0",
        "model_version":  "v3.0",
        "library":        "onnxruntime",
        "request_id":     str(uuid.uuid4()),
        "input_image":    image_name,

        # Résultat principal
        "prediction":     pred,
        "confidence":     conf,
        "probabilities":  probs_d,
        "risk_level":     pred,

        # Analyses cliniques
        "tumor_grade":        grade,
        "tnm_stage":          tnm,
        "receptor_status":    recep,
        "ki67":               ki67,
        "tumor_subtype":      subtyp,

        # Sécurité
        "clinical_safety":    safety,
        "recommendations":    reco,

        # Explainability
        "explainability": {
            "method":      "CAM",
            "status":      "generated" if heatmap_b64 else "not_generated",
            "heatmap_b64": heatmap_b64,
            "target_layer": "features[-1]",
        },

        "supported_output_classes": ["Normal", "Benign", "Malignant"],
        "deployment_mode": "streamlit_cloud_ready",
        "processing_ms":   round((time.time() - t0) * 1000),
        "status":          "model_loaded",
    }


def _predict_pytorch(image_path: str, image_name: str | None, t0: float) -> dict[str, Any]:
    """Fallback PyTorch si ONNX absent."""
    try:
        import torch
        from torchvision import models, transforms

        model_path = Path.cwd() / MODEL_PATH
        if not model_path.exists():
            return _placeholder_response(image_name)

        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)
        model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        model.eval()

        tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN.tolist(), _STD.tolist()),
        ])
        img    = Image.open(image_path).convert("L")
        tensor = tf(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax())

        pred  = CLASSES[pred_idx]
        conf  = round(float(probs[pred_idx]), 4)
        probs_d = {CLASSES[i]: round(float(p), 4) for i, p in enumerate(probs)}
        return _build_response(pred, conf, probs_d, None, None, image_name, t0)
    except Exception as exc:
        resp = _placeholder_response(image_name, "inference_error")
        resp["error"] = str(exc)
        return resp
