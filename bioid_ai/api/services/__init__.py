"""
bioid_ai.api.services — Services métier du module BioID AI.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    from bioid_ai.api.services.bioid_service import (
        predict_full,
        load_bundle,
        reload_bundle,
        is_bundle_loaded,
        compute_shap,
        generate_pdf_report,
        export_json,
        export_csv,
    )
    __all__ = [
        "predict_full",
        "load_bundle",
        "reload_bundle",
        "is_bundle_loaded",
        "compute_shap",
        "generate_pdf_report",
        "export_json",
        "export_csv",
    ]
except ImportError as e:
    log.debug("Services non disponibles : %s", e)
    __all__ = []
