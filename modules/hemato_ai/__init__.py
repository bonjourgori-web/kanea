"""
HematoVision AI — Module hématologie KANÉA
==========================================
Détection, classification et stadification des maladies hématologiques.
WHO 2022 · ICC 2022 · ELN 2022 · ISS/R-ISS · IPSS-R · ISTH DIC · OMS Paludisme.
"""
from modules.hemato_ai.predictor import predict_hemato, HEMATO_CLASSES
from modules.hemato_ai.report import build_hemato_pdf_report, build_hemato_html_report
from modules.hemato_ai.clinical_scores import (
    compute_all_hemato_scores,
    compute_nfs,
    compute_eln_aml,
    compute_ann_arbor,
    compute_ipi,
    compute_iss_myeloma,
    compute_ipss_r,
    compute_isth_dic,
    compute_parasitemia,
)

__all__ = [
    "predict_hemato",
    "HEMATO_CLASSES",
    "build_hemato_pdf_report",
    "build_hemato_html_report",
    "compute_all_hemato_scores",
    "compute_nfs",
    "compute_eln_aml",
    "compute_ann_arbor",
    "compute_ipi",
    "compute_iss_myeloma",
    "compute_ipss_r",
    "compute_isth_dic",
    "compute_parasitemia",
]
