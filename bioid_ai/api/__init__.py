"""
bioid_ai.api — Package API REST FastAPI du module BioID AI.

Contient :
- main.py           : Application FastAPI principale
- security.py       : JWT, OAuth2, chiffrement, sanitisation
- routes/bioid.py   : Endpoints BioID (predict, report, health, models/info)
- schemas/          : Modèles Pydantic v2
- services/         : Logique métier et orchestration
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)
