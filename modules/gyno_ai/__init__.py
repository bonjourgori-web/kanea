"""
GynoCare AI — Module gynécologie-oncologie KANÉA
=================================================
FIGO 2018/2023 · ESGO 2020 · O-RADS · rASRM · Rotterdam SOPK · ROMA Score.
"""
from modules.gyno_ai.predictor import predict_gyno, GYNO_CLASSES
from modules.gyno_ai.report import build_gyno_pdf_report, build_gyno_html_report
from modules.gyno_ai.clinical_scores import (
    compute_all_gyno_scores,
    compute_figo_cervix,
    compute_figo_endometrium,
    compute_figo_ovary,
    compute_orads,
    compute_rasrm_endometriosis,
    compute_sopk_rotterdam,
    compute_roma_score,
    compute_tumor_markers,
)

__all__ = [
    "predict_gyno",
    "GYNO_CLASSES",
    "build_gyno_pdf_report",
    "build_gyno_html_report",
    "compute_all_gyno_scores",
    "compute_figo_cervix",
    "compute_figo_endometrium",
    "compute_figo_ovary",
    "compute_orads",
    "compute_rasrm_endometriosis",
    "compute_sopk_rotterdam",
    "compute_roma_score",
    "compute_tumor_markers",
]
