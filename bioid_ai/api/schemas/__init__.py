"""
bioid_ai.api.schemas — Schémas Pydantic v2 pour l'API BioID AI.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    from bioid_ai.api.schemas.bioid_schemas import (
        CranialMeasurements,
        PostcranialMeasurements,
        AIMarkers,
        BioidRequest,
        PredictionResult,
        BioidResponse,
        ReportRequest,
        HealthResponse,
        ModelInfo,
    )
    __all__ = [
        "CranialMeasurements",
        "PostcranialMeasurements",
        "AIMarkers",
        "BioidRequest",
        "PredictionResult",
        "BioidResponse",
        "ReportRequest",
        "HealthResponse",
        "ModelInfo",
    ]
except ImportError as e:
    log.debug("Schemas non disponibles : %s", e)
    __all__ = []
