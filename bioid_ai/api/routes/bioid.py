"""
bioid.py — Router FastAPI pour l'API BioID AI.

Endpoints :
- POST /predict/bioid    — analyse complète, retourne BioidResponse
- POST /generate-report  — génère PDF, retourne le chemin
- GET  /health           — healthcheck
- GET  /models/info      — info sur les modèles chargés
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, status
    from fastapi.responses import FileResponse, JSONResponse
    _FASTAPI_OK = True
except ImportError:
    log.error("fastapi non disponible — routes bioid désactivées")
    _FASTAPI_OK = False

if not _FASTAPI_OK:
    raise ImportError("fastapi est requis pour les routes bioid")

# ── Imports locaux ─────────────────────────────────────────────────────────────
_ROUTE_DIR  = Path(__file__).resolve().parent
_BIOID_DIR  = _ROUTE_DIR.parent.parent.parent
_KANEA_ROOT = _BIOID_DIR.parent

for p in [str(_KANEA_ROOT), str(_BIOID_DIR.parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from bioid_ai.api.schemas.bioid_schemas import (
        BioidRequest, BioidResponse, HealthResponse,
        ModelInfo, PredictionResult, ReportRequest,
    )
    from bioid_ai.api.security import get_current_user, log_audit_event, sanitize_input
    from bioid_ai.api.services.bioid_service import (
        generate_pdf_report, is_bundle_loaded, load_bundle, predict_full,
    )
    _IMPORTS_OK = True
except ImportError as e:
    log.error("Imports locaux bioid routes échoués : %s", e)
    try:
        from api.schemas.bioid_schemas import (  # type: ignore
            BioidRequest, BioidResponse, HealthResponse,
            ModelInfo, PredictionResult, ReportRequest,
        )
        from api.security import get_current_user, log_audit_event, sanitize_input  # type: ignore
        from api.services.bioid_service import (  # type: ignore
            generate_pdf_report, is_bundle_loaded, load_bundle, predict_full,
        )
        _IMPORTS_OK = True
    except ImportError as e2:
        log.critical("Impossible d'importer les dépendances des routes : %s", e2)
        _IMPORTS_OK = False

# ── Router ─────────────────────────────────────────────────────────────────────
router = APIRouter(
    prefix="/bioid",
    tags=["BioID AI — Profil Biologique Forensique"],
)

API_VERSION = "2.0.0"


# ── GET /health ───────────────────────────────────────────────────────────────
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Vérifie l'état du service BioID AI.",
)
async def health_check() -> HealthResponse:
    """Healthcheck endpoint — retourne le statut du service et des modèles."""
    try:
        models_loaded = is_bundle_loaded()
        if not models_loaded:
            try:
                load_bundle()
                models_loaded = True
            except (FileNotFoundError, RuntimeError):
                models_loaded = False

        return HealthResponse(
            status="healthy" if models_loaded else "degraded",
            version=API_VERSION,
            models_loaded=models_loaded,
            timestamp=datetime.now(timezone.utc),
        )
    except Exception as exc:
        log.error("health_check error: %s", exc)
        return HealthResponse(
            status="error",
            version=API_VERSION,
            models_loaded=False,
            timestamp=datetime.now(timezone.utc),
        )


# ── GET /models/info ──────────────────────────────────────────────────────────
@router.get(
    "/models/info",
    response_model=ModelInfo,
    summary="Information sur les modèles",
    description="Retourne les informations sur les modèles chargés.",
)
async def get_models_info(
    current_user: dict = Depends(get_current_user),
) -> ModelInfo:
    """Retourne les métadonnées du bundle de modèles."""
    try:
        bundle = load_bundle()
        meta   = bundle.get("metadata", {})
        return ModelInfo(
            bundle_path      = str(_KANEA_ROOT / "models" / "machine_learning" / "bioid_bundle.pkl"),
            models_available = [k for k in bundle.keys() if "model" in k],
            feature_columns  = bundle.get("feature_columns", []),
            sex_classes      = list(bundle["sex_encoder"].classes_) if "sex_encoder" in bundle else None,
            ancestry_classes = list(bundle["ancestry_encoder"].classes_) if "ancestry_encoder" in bundle else None,
            trained_at       = meta.get("trained_at"),
            n_samples        = meta.get("n_samples"),
            version          = meta.get("version"),
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Bundle de modèles indisponible : {exc}",
        )
    except Exception as exc:
        log.error("get_models_info error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des informations modèles : {exc}",
        )


# ── POST /predict/bioid ───────────────────────────────────────────────────────
@router.post(
    "/predict/bioid",
    response_model=BioidResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse BioID complète",
    description=(
        "Effectue l'estimation complète du profil biologique forensique : "
        "sexe, âge au décès, ascendance biogéographique et stature."
    ),
)
async def predict_bioid(
    request: BioidRequest,
    generate_report: bool = Query(False, description="Générer automatiquement le rapport PDF"),
    current_user: dict = Depends(get_current_user),
) -> BioidResponse:
    """
    Endpoint principal BioID — analyse complète du profil biologique.

    Corps de la requête : BioidRequest avec mesures crâniennes et/ou post-crâniennes.
    Retourne : BioidResponse avec toutes les prédictions.
    """
    try:
        # Sanitisation des inputs textuels
        safe_case_id  = sanitize_input(request.case_id, max_length=100)
        safe_examiner = sanitize_input(request.examiner, max_length=100)

        log_audit_event(
            action="predict_bioid",
            user=str(current_user.get("sub", "anonymous")),
            case_id=safe_case_id,
            details=f"examiner={safe_examiner}",
        )
        log.info("predict_bioid → case_id='%s', examiner='%s'", safe_case_id, safe_examiner)

        # Extraction des données
        cranial_dict     = request.cranial.to_dict()     if request.cranial     else {}
        postcranial_dict = request.postcranial.to_dict() if request.postcranial else {}
        aims_list        = request.aims.to_list()        if request.aims        else None

        # Appel au service de prédiction
        preds = predict_full(
            cranial=cranial_dict,
            postcranial=postcranial_dict,
            aims_raw=aims_list,
        )

        # Construction du PredictionResult
        prediction_result = PredictionResult(
            biological_sex      = preds.get("biological_sex", "Inconnu"),
            sex_confidence      = float(preds.get("sex_confidence", 0.0)),
            age_at_death        = float(preds.get("age_at_death", 0.0)),
            age_range           = str(preds.get("age_range", "0–0 ans")),
            ancestry            = preds.get("ancestry", "Inconnu"),
            ancestry_confidence = float(preds.get("ancestry_confidence", 0.0)),
            stature_cm          = preds.get("stature_cm"),
            stature_method      = preds.get("stature_method"),
        )

        # Génération optionnelle du rapport PDF
        report_path: Optional[str] = None
        if generate_report:
            try:
                report_path = generate_pdf_report(
                    case_id=safe_case_id,
                    examiner=safe_examiner,
                    predictions=preds,
                )
            except Exception as e:
                log.warning("Génération rapport PDF échouée (non bloquant) : %s", e)

        response = BioidResponse(
            case_id            = safe_case_id,
            timestamp          = datetime.now(timezone.utc),
            predictions        = prediction_result,
            feature_importances = preds.get("feature_importances"),
            pca_variance       = preds.get("pca_variance"),
            report_path        = report_path,
            status             = preds.get("status", "success"),
            warnings           = _build_warnings(preds, cranial_dict, postcranial_dict),
            model_version      = preds.get("model_version"),
        )

        log_audit_event(
            action="predict_bioid_complete",
            user=str(current_user.get("sub", "anonymous")),
            case_id=safe_case_id,
            success=True,
        )
        return response

    except HTTPException:
        raise
    except ValueError as exc:
        log.warning("predict_bioid validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Erreur de validation : {exc}",
        )
    except Exception as exc:
        log.error("predict_bioid error: %s", exc, exc_info=True)
        log_audit_event(
            action="predict_bioid",
            user=str(current_user.get("sub", "anonymous") if current_user else "anonymous"),
            case_id=str(getattr(request, "case_id", "unknown")),
            details=str(exc),
            success=False,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne lors de l'analyse BioID : {str(exc)[:200]}",
        )


# ── POST /generate-report ─────────────────────────────────────────────────────
@router.post(
    "/generate-report",
    summary="Génération de rapport PDF",
    description="Génère un rapport médico-légal PDF complet depuis des données de prédiction.",
)
async def generate_report(
    request: ReportRequest,
    download: bool = Query(False, description="Retourner le fichier directement en téléchargement"),
    current_user: dict = Depends(get_current_user),
) -> Any:
    """
    Génère un rapport PDF médico-légal BioID.

    Si download=True, retourne le fichier PDF directement.
    Sinon, retourne le chemin du fichier généré.
    """
    try:
        safe_case_id  = sanitize_input(request.case_id, max_length=100)
        safe_examiner = sanitize_input(request.examiner, max_length=100)

        log_audit_event(
            action="generate_report",
            user=str(current_user.get("sub", "anonymous")),
            case_id=safe_case_id,
        )

        chart_paths = request.chart_paths or []

        pdf_path = generate_pdf_report(
            case_id=safe_case_id,
            examiner=safe_examiner,
            predictions=request.predictions,
            charts=chart_paths,
        )

        if not pdf_path:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="La génération du rapport PDF a échoué.",
            )

        if download and Path(pdf_path).exists():
            return FileResponse(
                path=pdf_path,
                media_type="application/pdf",
                filename=Path(pdf_path).name,
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status":      "success",
                "report_path": pdf_path,
                "case_id":     safe_case_id,
                "message":     "Rapport PDF généré avec succès.",
            },
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Erreur de validation : {exc}",
        )
    except Exception as exc:
        log.error("generate_report error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la génération du rapport : {str(exc)[:200]}",
        )


# ── Helpers ────────────────────────────────────────────────────────────────────
def _build_warnings(
    preds: dict[str, Any],
    cranial: dict,
    postcranial: dict,
) -> list[str]:
    """Construit la liste des avertissements selon la qualité des données."""
    warnings: list[str] = []

    # Confiance sexe
    sex_conf = float(preds.get("sex_confidence", 1.0))
    if sex_conf < 0.70:
        warnings.append(
            f"Confiance faible pour le sexe biologique ({sex_conf:.1%}). "
            "Vérifiez les mesures crâniennes."
        )

    # Confiance ascendance
    anc_conf = float(preds.get("ancestry_confidence", 1.0))
    if anc_conf < 0.60:
        warnings.append(
            f"Confiance faible pour l'ascendance ({anc_conf:.1%}). "
            "Les marqueurs AIMs manquants réduisent la précision."
        )

    # Stature
    if preds.get("stature_cm") is None:
        warnings.append(
            "Stature non estimée : mesures post-crâniennes manquantes "
            "(femur_bicondylar requis)."
        )

    # Données crâniennes insuffisantes
    cranial_count = sum(1 for v in cranial.values() if v is not None)
    if cranial_count < 5:
        warnings.append(
            f"Seulement {cranial_count} mesures crâniennes disponibles (minimum recommandé : 5). "
            "La précision des prédictions peut être réduite."
        )

    # Statut placeholder
    if preds.get("status") in ("placeholder", "inference_error"):
        warnings.append(
            "Les modèles BioID ne sont pas chargés. "
            "Exécutez python bioid_ai/training/train.py pour générer le bundle."
        )

    return warnings
