"""
KANÉA — Universal Medical Data Acquisition & Auto-Training Pipeline
====================================================================
11 phases automatisées pour acquisition, prétraitement, annotation,
entraînement, réentraînement et validation des 15 modules médicaux IA.
"""
from pipeline.orchestrator import run_full_pipeline, get_pipeline_status
from pipeline.registry import REGISTRY, get_module, list_modules
from pipeline.catalog import get_catalog

__all__ = [
    "run_full_pipeline",
    "get_pipeline_status",
    "REGISTRY",
    "get_module",
    "list_modules",
    "get_catalog",
]
