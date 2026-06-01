"""GastroAI v1.0 — Gastroentérologie · Endoscopie IA · 25 pathologies digestives."""
from modules.gastro_ai.predictor import predict_gastro
from modules.gastro_ai.report import generate_gastro_report

__all__ = ["predict_gastro", "generate_gastro_report"]
