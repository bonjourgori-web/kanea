from __future__ import annotations

import sys
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ── Racine KANEA dans le path ─────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

log = logging.getLogger("kanea.api")

from api.schemas import BioIDInput, BreastCancerPathInput, MalariaPathInput, MultibioRequest, NutritionInput
from modules.bioid_ml.predictor import predict_bioid
from modules.breast_cancer_dl.predictor import predict_breast_cancer
from modules.malaria_dl.predictor import predict_malaria
from modules.integration.engine import multibio_predict
from modules.nutrition_ml.predictor import predict_nutrition

# ── Import optionnel du service bioid_ai enrichi ──────────────────────────────
try:
    from bioid_ai.api.services.bioid_service import (
        predict_full as _bioid_predict_full,
        generate_pdf_report as _bioid_generate_pdf,
        load_bundle as _bioid_load_bundle,
    )
    _BIOID_AI_OK = True
    _bioid_load_bundle()
    log.info("bioid_ai service chargé — bundle bioid_bundle.pkl actif")
except Exception as _e:
    _BIOID_AI_OK = False
    log.warning("bioid_ai service indisponible : %s", _e)

# ── Import optionnel du router bioid_ai ──────────────────────────────────────
try:
    from bioid_ai.api.routes.bioid import router as _bioid_router
    _BIOID_ROUTER_OK = True
except Exception:
    _BIOID_ROUTER_OK = False

app = FastAPI(
    title="KANÉA API",
    description=(
        "API d'aide à la décision biomédicale et médico-légale — "
        "Knowledge Anthropology & Neural Engine for Africa"
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monte le router BioID AI avancé sous /api/v2/bioid
if _BIOID_ROUTER_OK:
    app.include_router(_bioid_router, prefix="/api/v2", tags=["BioID AI v2"])


# ═══════════════════════════════════════════════════════════════════════════════
# RACINE & SANTÉ
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root() -> dict:
    return {
        "application":  "KANÉA",
        "version":      "2.0.0",
        "status":       "running",
        "message":      "Knowledge Anthropology & Neural Engine for Africa",
        "bioid_ai_v2":  _BIOID_AI_OK,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "bioid_bundle": _BIOID_AI_OK}


@app.get("/info")
def info() -> dict:
    return {
        "application": "KANÉA",
        "full_name":   "Knowledge Anthropology & Neural Engine for Africa",
        "modules": [
            {"name": "module_1_malaria",      "approach": "deep_learning",    "technology": "fast.ai + PyTorch + ResNet34"},
            {"name": "module_2_biometry",     "approach": "machine_learning", "technology": "RandomForest + XGBoost"},
            {"name": "module_3_forensic",     "approach": "machine_learning", "technology": "BioID AI v2 — VotingClassifier + PCA + Trotter-Gleser"},
            {"name": "module_4_breast_cancer","approach": "deep_learning",    "technology": "PyTorch + EfficientNet-B0"},
        ],
        "integration": "multibio_predict",
        "bioid_ai_v2_routes": "/api/v2/bioid/" if _BIOID_ROUTER_OK else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — MALARIA
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/predict/malaria")
def predict_malaria_from_path(request: MalariaPathInput) -> dict:
    return predict_malaria(request.image_path)


@app.post("/predict/malaria/upload")
async def predict_malaria_from_upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    return predict_malaria(temp_path)


@app.post("/predict/malaria/species")
async def predict_malaria_species(file: UploadFile = File(...)) -> dict:
    """Identification de l'espece Plasmodium (estimation epidemiologique)."""
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    result = predict_malaria(temp_path)
    return {
        "prediction":        result.get("prediction"),
        "confidence":        result.get("confidence"),
        "species_prediction": result.get("species_prediction"),
        "clinical_safety":   result.get("clinical_safety"),
        "model_version":     result.get("model_version"),
        "status":            result.get("status"),
    }


@app.post("/predict/malaria/stage")
async def predict_malaria_stage(file: UploadFile = File(...)) -> dict:
    """Classification du stade parasitaire (anneau, trophozoite, schizonte, gametocyte)."""
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    result = predict_malaria(temp_path)
    return {
        "prediction":      result.get("prediction"),
        "confidence":      result.get("confidence"),
        "parasite_stage":  result.get("parasite_stage"),
        "clinical_safety": result.get("clinical_safety"),
        "model_version":   result.get("model_version"),
        "status":          result.get("status"),
    }


@app.post("/predict/malaria/parasitemia")
async def predict_malaria_parasitemia(file: UploadFile = File(...)) -> dict:
    """Estimation de la parasitemie (% cellules infectees, severite)."""
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    result = predict_malaria(temp_path)
    return {
        "prediction":      result.get("prediction"),
        "confidence":      result.get("confidence"),
        "parasitemia":     result.get("parasitemia"),
        "clinical_safety": result.get("clinical_safety"),
        "model_version":   result.get("model_version"),
        "status":          result.get("status"),
    }


@app.post("/malaria/report")
async def generate_malaria_report(file: UploadFile = File(...)) -> dict:
    """Analyse complete + generation rapport PDF medical (base64)."""
    import base64
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    result = predict_malaria(temp_path)
    try:
        from dashboard.export_pdf import build_pdf_report
        pdf_bytes, _ = build_pdf_report(result, module="malaria")
        pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    except Exception as exc:
        pdf_b64 = None
        log.warning("PDF generation failed: %s", exc)
    return {
        **{k: v for k, v in result.items() if k != "explainability"},
        "pdf_report_b64": pdf_b64,
        "pdf_available":  pdf_b64 is not None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — BREAST CANCER
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/predict/breast-cancer")
def predict_breast_cancer_from_path(request: BreastCancerPathInput) -> dict:
    return predict_breast_cancer(request.image_path)


@app.post("/predict/breast-cancer/upload")
async def predict_breast_cancer_from_upload(file: UploadFile = File(...)) -> dict:
    """Analyse complète BreastCancer AI v3.0 — grade, TNM, ER/PR/HER2, Ki67, CAM."""
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    return predict_breast_cancer(temp_path)


@app.post("/predict/breast-cancer/stage")
async def predict_bc_stage(file: UploadFile = File(...)) -> dict:
    """Estimation stade TNM et grade tumoral."""
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    result = predict_breast_cancer(temp_path)
    return {
        "request_id":    result.get("request_id"),
        "prediction":    result.get("prediction"),
        "confidence":    result.get("confidence"),
        "tumor_grade":   result.get("tumor_grade"),
        "tnm_stage":     result.get("tnm_stage"),
        "tumor_subtype": result.get("tumor_subtype"),
        "clinical_safety": result.get("clinical_safety"),
        "processing_ms": result.get("processing_ms"),
        "status":        result.get("status"),
    }


@app.post("/predict/breast-cancer/receptors")
async def predict_bc_receptors(file: UploadFile = File(...)) -> dict:
    """Prédiction statut récepteurs ER/PR/HER2 et phénotype moléculaire."""
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    result = predict_breast_cancer(temp_path)
    return {
        "request_id":       result.get("request_id"),
        "prediction":       result.get("prediction"),
        "confidence":       result.get("confidence"),
        "receptor_status":  result.get("receptor_status"),
        "clinical_safety":  result.get("clinical_safety"),
        "processing_ms":    result.get("processing_ms"),
        "status":           result.get("status"),
    }


@app.post("/predict/breast-cancer/ki67")
async def predict_bc_ki67(file: UploadFile = File(...)) -> dict:
    """Estimation index Ki67 de prolifération tumorale."""
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    result = predict_breast_cancer(temp_path)
    return {
        "request_id":      result.get("request_id"),
        "prediction":      result.get("prediction"),
        "confidence":      result.get("confidence"),
        "ki67":            result.get("ki67"),
        "tumor_grade":     result.get("tumor_grade"),
        "clinical_safety": result.get("clinical_safety"),
        "processing_ms":   result.get("processing_ms"),
        "status":          result.get("status"),
    }


@app.post("/breast-cancer/report")
async def generate_bc_report(file: UploadFile = File(...)) -> dict:
    """Analyse complète + rapport PDF médical BreastCancer AI (base64)."""
    import base64
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    result = predict_breast_cancer(temp_path)
    try:
        from dashboard.export_pdf import build_pdf_report
        pdf_bytes, _ = build_pdf_report(result, module="breast_cancer")
        pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    except Exception as exc:
        pdf_b64 = None
        log.warning("BreastCancer PDF generation failed: %s", exc)
    return {
        **{k: v for k, v in result.items() if k not in ("explainability",)},
        "pdf_report_b64": pdf_b64,
        "pdf_available":  pdf_b64 is not None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — PULMOSCAN AI v2.0 (16 pathologies pulmonaires)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from modules.pulmoscan_ai.predictor import predict_pulmoscan
    from modules.pulmoscan_ai.report import build_pulmoscan_pdf_report
    _PULMOSCAN_OK = True
    log.info("PulmoScan AI v2.0 chargé — 16 pathologies pulmonaires")
except Exception as _ps_err:
    _PULMOSCAN_OK = False
    log.warning("PulmoScan AI indisponible : %s", _ps_err)


@app.post("/predict/pulmoscan/upload", tags=["PulmoScan AI"])
async def predict_pulmoscan_upload(file: UploadFile = File(...)) -> dict:
    """
    PulmoScan AI v2.0 — Analyse complète radiographie/TDM thoracique.
    Détecte 16 pathologies pulmonaires avec scores cliniques et Grad-CAM.
    """
    if not _PULMOSCAN_OK:
        from fastapi import HTTPException
        raise HTTPException(503, "PulmoScan AI non disponible")
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    return predict_pulmoscan(image_path=temp_path)


@app.post("/predict/pulmoscan/severity", tags=["PulmoScan AI"])
async def predict_pulmoscan_severity(file: UploadFile = File(...)) -> dict:
    """Sévérité + urgence + CURB-65/PSI/COVID CT Severity selon pathologie détectée."""
    if not _PULMOSCAN_OK:
        from fastapi import HTTPException
        raise HTTPException(503, "PulmoScan AI non disponible")
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    result = predict_pulmoscan(image_path=temp_path)
    return {
        "request_id":      result.get("request_id"),
        "prediction":      result.get("prediction"),
        "confidence":      result.get("confidence"),
        "severity":        result.get("severity"),
        "clinical_profile": result.get("clinical_profile"),
        "clinical_scores": result.get("clinical_scores"),
        "clinical_safety": result.get("clinical_safety"),
        "processing_ms":   result.get("processing_ms"),
        "status":          result.get("status"),
    }


@app.post("/predict/pulmoscan/differential", tags=["PulmoScan AI"])
async def predict_pulmoscan_differential(file: UploadFile = File(...)) -> dict:
    """Diagnostic différentiel Top-3 avec probabilités et CIM-10."""
    if not _PULMOSCAN_OK:
        from fastapi import HTTPException
        raise HTTPException(503, "PulmoScan AI non disponible")
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    result = predict_pulmoscan(image_path=temp_path)
    return {
        "request_id":           result.get("request_id"),
        "prediction":           result.get("prediction"),
        "confidence":           result.get("confidence"),
        "differential_diagnosis": result.get("differential_diagnosis"),
        "probabilities":        result.get("probabilities"),
        "processing_ms":        result.get("processing_ms"),
        "status":               result.get("status"),
    }


@app.post("/pulmoscan/report", tags=["PulmoScan AI"])
async def generate_pulmoscan_report(
    file: UploadFile = File(...),
    patient_id: str = "KANEA-AUTO",
    examiner: str = "KANÉA System",
) -> dict:
    """Analyse complète + génération rapport PDF médical A4 (base64)."""
    import base64
    if not _PULMOSCAN_OK:
        from fastapi import HTTPException
        raise HTTPException(503, "PulmoScan AI non disponible")
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    result = predict_pulmoscan(image_path=temp_path)
    pdf_b64 = None
    try:
        pdf_bytes = build_pulmoscan_pdf_report(result, patient_id=patient_id, examiner=examiner)
        if pdf_bytes:
            pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    except Exception as exc:
        log.warning("PulmoScan PDF generation failed: %s", exc)
    return {
        **{k: v for k, v in result.items() if k not in ("explainability",)},
        "pdf_report_b64": pdf_b64,
        "pdf_available":  pdf_b64 is not None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULES 6–17 — SCAFFOLD AI MODULES
# ═══════════════════════════════════════════════════════════════════════════════

from modules.kanea_modules.scaffold import scaffold_predict, MODULES as _SCAFFOLD_MODULES

def _make_scaffold_routes() -> None:
    """Enregistre dynamiquement les routes pour tous les modules scaffold."""
    _ROUTE_MAP = {
        "derm":      "/predict/derm",
        "retina":    "/predict/retina",
        "cardio":    "/predict/cardio",
        "neuro":     "/predict/neuro",
        "gastro":    "/predict/gastro",
        "histopath": "/predict/histopath",
        "osteo":     "/predict/osteo",
        "sepsis":    "/predict/sepsis",
        "hepato":    "/predict/hepato",
        "nephro":    "/predict/nephro",
        "hemato":    "/predict/hemato",
        "gyno":      "/predict/gyno",
    }
    for module_key, route in _ROUTE_MAP.items():
        cfg = _SCAFFOLD_MODULES.get(module_key, {})

        # Route upload (image modules)
        if cfg.get("input_type") == "image":
            async def _upload_route(file: UploadFile = File(...), _mk=module_key) -> dict:
                suffix = Path(file.filename or "upload.png").suffix or ".png"
                with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(await file.read())
                    temp_path = tmp.name
                return scaffold_predict(_mk, image_path=temp_path)
            _upload_route.__name__ = f"predict_{module_key}_upload"
            app.post(route + "/upload", tags=[cfg.get("name", module_key)])(_upload_route)
        else:
            # Route paramètres (JSON body)
            def _param_route(payload: dict = {}, _mk=module_key) -> dict:
                return scaffold_predict(_mk, params=payload)
            _param_route.__name__ = f"predict_{module_key}"
            app.post(route, tags=[cfg.get("name", module_key)])(_param_route)

_make_scaffold_routes()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — BIOMETRY / NUTRITION
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/predict/biometry")
def predict_biometry(request: NutritionInput) -> dict:
    return predict_nutrition(request.model_dump())


@app.post("/predict/nutrition")
def predict_nutrition_route(request: NutritionInput) -> dict:
    """Alias NutriTrack AI — statut nutritionnel complet avec SHAP et recommandations."""
    return predict_nutrition(request.model_dump())


@app.post("/predict/nutrition/risk")
def predict_nutrition_risk(request: NutritionInput) -> dict:
    """Niveau de risque nutritionnel et score composite."""
    result = predict_nutrition(request.model_dump())
    return {
        "request_id":           result.get("request_id"),
        "prediction":           result.get("prediction"),
        "confidence":           result.get("confidence"),
        "risk_level":           result.get("risk_level"),
        "nutrition_risk_score": (result.get("derived_features") or {}).get("nutrition_risk_score"),
        "clinical_urgency":     (result.get("recommendations") or {}).get("urgency"),
        "bmi":                  (result.get("derived_features") or {}).get("bmi"),
        "processing_ms":        result.get("processing_ms"),
        "status":               result.get("status"),
    }


@app.post("/nutrition/report")
async def generate_nutrition_report(request: NutritionInput) -> dict:
    """Analyse complète + rapport PDF médical NutriTrack AI (base64)."""
    import base64
    result = predict_nutrition(request.model_dump())
    try:
        from dashboard.export_pdf import build_pdf_report
        pdf_bytes, _ = build_pdf_report(result, module="nutrition")
        pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    except Exception as exc:
        pdf_b64 = None
        log.warning("NutriTrack PDF generation failed: %s", exc)
    return {
        **{k: v for k, v in result.items() if k not in ("explainability",)},
        "pdf_report_b64": pdf_b64,
        "pdf_available":  pdf_b64 is not None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — FORENSIC / BIOID
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/predict/forensic")
def predict_forensic(request: BioIDInput) -> dict:
    """
    Estimation du profil biologique médico-légal.

    Utilise automatiquement BioID AI v2 si disponible,
    sinon fallback vers le modèle forensic classique.
    """
    payload = {
        "cranial_measurements":     request.cranial_measurements,
        "postcranial_measurements": request.postcranial_measurements,
        "aims_pcs":                 request.aims_pcs,
    }

    # Voie enrichie bioid_ai v2
    if _BIOID_AI_OK and request.aims_raw is not None:
        try:
            cranial     = request.cranial_measurements
            postcranial = request.postcranial_measurements
            aims_raw    = request.aims_raw

            result = _bioid_predict_full(
                cranial=cranial,
                postcranial=postcranial,
                aims_raw=aims_raw,
            )

            pdf_path = None
            if request.generate_pdf and result.get("status") == "success":
                case_id  = request.case_id  or "KANEA-AUTO"
                examiner = request.examiner or "KANÉA System"
                try:
                    pdf_path = _bioid_generate_pdf(case_id, examiner, result)
                except Exception as pdf_err:
                    log.warning("PDF non généré : %s", pdf_err)

            return {
                "module":    "module_3_forensic",
                "engine":    "bioid_ai_v2",
                "result":    result,
                "pdf_path":  pdf_path,
                "status":    result.get("status", "ok"),
            }
        except Exception as exc:
            log.error("bioid_ai v2 error : %s", exc)

    # Voie classique (fallback)
    result = predict_bioid(payload)

    # Génération PDF si demandée et modèle chargé
    pdf_path = None
    if request.generate_pdf and result.get("status") == "model_loaded" and _BIOID_AI_OK:
        try:
            case_id  = request.case_id  or "KANEA-AUTO"
            examiner = request.examiner or "KANÉA System"
            preds = {
                "biological_sex":     result["prediction"].get("biological_sex"),
                "sex_confidence":     (result.get("confidence") or {}).get("biological_sex"),
                "age_at_death":       result["prediction"].get("age_at_death"),
                "ancestry":           result["prediction"].get("ancestry"),
                "ancestry_confidence":(result.get("confidence") or {}).get("ancestry"),
                "stature_cm":         result["prediction"].get("stature_cm"),
                "status":             "success",
            }
            pdf_path = _bioid_generate_pdf(case_id, examiner, preds)
        except Exception as pdf_err:
            log.warning("PDF non généré : %s", pdf_err)

    result["pdf_path"] = pdf_path
    return result


@app.post("/predict/forensic/report")
def generate_forensic_report(request: BioIDInput) -> dict:
    """Génère uniquement le rapport PDF pour un dossier BioID."""
    if not _BIOID_AI_OK:
        raise HTTPException(status_code=503, detail="bioid_ai non disponible — installez les dépendances")

    case_id  = request.case_id  or "KANEA-AUTO"
    examiner = request.examiner or "KANÉA System"

    cranial     = request.cranial_measurements
    postcranial = request.postcranial_measurements
    aims_raw    = request.aims_raw

    try:
        predictions = _bioid_predict_full(cranial=cranial, postcranial=postcranial, aims_raw=aims_raw)
        pdf_path    = _bioid_generate_pdf(case_id, examiner, predictions)
        return {"case_id": case_id, "pdf_path": pdf_path, "status": "ok"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur génération rapport : {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# MULTIMODAL
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/predict")
def predict(request: MultibioRequest) -> dict:
    nutrition_data = request.nutrition_data.model_dump() if request.nutrition_data else None
    bioid_data     = request.bioid_data.model_dump()     if request.bioid_data     else None

    return multibio_predict(
        image_path=request.image_path,
        breast_cancer_image_path=request.breast_cancer_image_path,
        nutrition_data=nutrition_data,
        bioid_data=bioid_data,
    )
