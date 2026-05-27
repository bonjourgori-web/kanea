"""
bioid_ai — Module BioID AI de la plateforme KANÉA.

Estimation du profil biologique forensique :
- Sexe biologique
- Âge au décès
- Ascendance biogéographique
- Stature

Sous-modules :
- bioid_ai.api         — API REST FastAPI
- bioid_ai.utils       — Utilitaires (features, PCA, visualisations, PDF)
- bioid_ai.training    — Script d'entraînement des modèles
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

__version__   = "2.0.0"
__author__    = "KANÉA — Plateforme IA Médicale"
__license__   = "Propriétaire"

# ── Exports principaux ────────────────────────────────────────────────────────
try:
    from bioid_ai.utils.feature_engineering import (
        ALL_FEATURE_KEYS,
        CRANIAL_KEYS,
        POSTCRANIAL_KEYS,
        AIMS_KEYS,
        DERIVED_KEYS,
        engineer_features,
        estimate_stature,
        payload_from_schema,
    )
    from bioid_ai.api.services.bioid_service import (
        predict_full,
        load_bundle,
        is_bundle_loaded,
        generate_pdf_report,
        export_json,
        export_csv,
    )
    _EXPORTS_OK = True
except ImportError as e:
    log.debug("Exports complets non disponibles (dépendances manquantes) : %s", e)
    _EXPORTS_OK = False

    # Exports minimaux depuis les utils
    try:
        from bioid_ai.utils.feature_engineering import (  # type: ignore
            ALL_FEATURE_KEYS, CRANIAL_KEYS, POSTCRANIAL_KEYS,
            AIMS_KEYS, DERIVED_KEYS, engineer_features,
        )
    except ImportError:
        pass


__all__ = [
    "__version__",
    "__author__",
    # Feature engineering
    "ALL_FEATURE_KEYS",
    "CRANIAL_KEYS",
    "POSTCRANIAL_KEYS",
    "AIMS_KEYS",
    "DERIVED_KEYS",
    "engineer_features",
    "estimate_stature",
    "payload_from_schema",
    # Service
    "predict_full",
    "load_bundle",
    "is_bundle_loaded",
    "generate_pdf_report",
    "export_json",
    "export_csv",
]
