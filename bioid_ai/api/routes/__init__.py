"""
bioid_ai.api.routes — Routers FastAPI du module BioID AI.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    from bioid_ai.api.routes.bioid import router as bioid_router
    __all__ = ["bioid_router"]
except ImportError as e:
    log.debug("bioid_router non disponible : %s", e)
    __all__ = []
