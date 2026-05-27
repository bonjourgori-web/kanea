"""
bioid_ai.utils — Utilitaires du module BioID AI.

Sous-modules :
- feature_engineering : Construction et enrichissement des features
- pca_analysis        : Analyse PCA sur les marqueurs AIMs
- visualizations      : Graphiques médicaux professionnels
- pdf_generator       : Génération de rapports PDF médico-légaux
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    from bioid_ai.utils.feature_engineering import (
        engineer_features,
        estimate_stature,
        compute_derived_indices,
        payload_from_schema,
        ALL_FEATURE_KEYS,
        CRANIAL_KEYS,
        POSTCRANIAL_KEYS,
        AIMS_KEYS,
        DERIVED_KEYS,
        REFERENCE_MEDIANS,
    )
    _FEAT_OK = True
except ImportError as e:
    log.debug("feature_engineering non disponible : %s", e)
    _FEAT_OK = False

try:
    from bioid_ai.utils.pca_analysis import (
        fit_aims_pca,
        transform_aims,
        plot_pca_variance,
        plot_pca_scatter,
    )
    _PCA_OK = True
except ImportError as e:
    log.debug("pca_analysis non disponible : %s", e)
    _PCA_OK = False

try:
    from bioid_ai.utils.visualizations import (
        plot_confusion_matrix,
        plot_roc_curves,
        plot_feature_importance,
        plot_shap_summary,
        plot_biological_radar,
        plot_prediction_summary,
    )
    _VIZ_OK = True
except ImportError as e:
    log.debug("visualizations non disponible : %s", e)
    _VIZ_OK = False

try:
    from bioid_ai.utils.pdf_generator import (
        BioidPDFReport,
        generate_bioid_report,
    )
    _PDF_OK = True
except ImportError as e:
    log.debug("pdf_generator non disponible : %s", e)
    _PDF_OK = False

__all__ = [
    # feature_engineering
    "engineer_features", "estimate_stature", "compute_derived_indices",
    "payload_from_schema", "ALL_FEATURE_KEYS", "CRANIAL_KEYS",
    "POSTCRANIAL_KEYS", "AIMS_KEYS", "DERIVED_KEYS", "REFERENCE_MEDIANS",
    # pca_analysis
    "fit_aims_pca", "transform_aims", "plot_pca_variance", "plot_pca_scatter",
    # visualizations
    "plot_confusion_matrix", "plot_roc_curves", "plot_feature_importance",
    "plot_shap_summary", "plot_biological_radar", "plot_prediction_summary",
    # pdf_generator
    "BioidPDFReport", "generate_bioid_report",
]
