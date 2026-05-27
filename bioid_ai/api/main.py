"""
main.py — Application FastAPI principale du module BioID AI.

Configure :
- CORS
- Lifespan (chargement des modèles au démarrage)
- Rate limiting avec slowapi
- Router bioid
- Middleware de logging
- Gestion globale des exceptions
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

log = logging.getLogger(__name__)

# ── FastAPI ────────────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException, Request, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    _FASTAPI_OK = True
except ImportError:
    log.critical("fastapi non disponible — impossible de démarrer l'API")
    sys.exit(1)

# ── Rate limiting ──────────────────────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address
    limiter = Limiter(key_func=get_remote_address)
    _SLOWAPI_OK = True
except ImportError:
    _SLOWAPI_OK = False
    limiter = None
    log.info("slowapi non disponible — rate limiting désactivé")

# ── Résolution des chemins ─────────────────────────────────────────────────────
_MAIN_DIR   = Path(__file__).resolve().parent    # api/
_BIOID_DIR  = _MAIN_DIR.parent                   # bioid_ai/
_KANEA_ROOT = _BIOID_DIR.parent                  # KANEA/

for p in [str(_KANEA_ROOT), str(_BIOID_DIR.parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Imports locaux ─────────────────────────────────────────────────────────────
try:
    from bioid_ai.api.routes.bioid import router as bioid_router
    from bioid_ai.api.services.bioid_service import load_bundle, is_bundle_loaded
    _IMPORTS_OK = True
except ImportError:
    try:
        from api.routes.bioid import router as bioid_router  # type: ignore
        from api.services.bioid_service import load_bundle, is_bundle_loaded  # type: ignore
        _IMPORTS_OK = True
    except ImportError as e:
        log.error("Imports locaux main.py échoués : %s", e)
        _IMPORTS_OK = False

# ── Configuration CORS ─────────────────────────────────────────────────────────
CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8501",    # Streamlit
    "http://127.0.0.1:8501",
    "http://127.0.0.1:8000",
]

# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Gestionnaire du cycle de vie de l'application.
    Charge les modèles au démarrage et libère les ressources à l'arrêt.
    """
    log.info("=== BioID AI — Démarrage ===")

    # Chargement des modèles
    try:
        bundle = load_bundle()
        meta = bundle.get("metadata", {})
        log.info(
            "Bundle chargé : v%s | %d samples | features=%d",
            meta.get("version", "?"),
            meta.get("n_samples", 0),
            len(bundle.get("feature_columns", [])),
        )
        print(f"[BioID AI] Modèles chargés : v{meta.get('version', '?')}")
    except FileNotFoundError:
        log.warning(
            "Bundle introuvable — service en mode dégradé. "
            "Exécutez : python bioid_ai/training/train.py"
        )
        print("[BioID AI] Avertissement : modèles non disponibles (mode dégradé)")
    except Exception as exc:
        log.error("Erreur chargement modèles : %s", exc)
        print(f"[BioID AI] Erreur chargement modèles : {exc}")

    yield  # Application en cours d'exécution

    # Nettoyage à l'arrêt
    log.info("=== BioID AI — Arrêt ===")
    print("[BioID AI] Arrêt du service")


# ── Application FastAPI ────────────────────────────────────────────────────────
app = FastAPI(
    title="KANÉA — BioID AI API",
    description=(
        "API REST du module BioID AI de la plateforme KANÉA.\n\n"
        "Estimation du profil biologique forensique (sexe, âge, ascendance, stature) "
        "à partir de mesures anthropométriques crâniennes et post-crâniennes.\n\n"
        "**Plateforme IA Médicale — Côte d'Ivoire**"
    ),
    version="2.0.0",
    contact={
        "name": "KANÉA — Équipe IA Médicale",
        "email": "support@kanea-ai.ci",
    },
    license_info={
        "name": "Propriétaire — KANÉA",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── Rate Limiting ──────────────────────────────────────────────────────────────
if _SLOWAPI_OK and limiter:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    log.info("Rate limiting activé (slowapi)")

# ── CORS Middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ── Middleware de logging des requêtes ────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log chaque requête HTTP avec durée et statut."""
    start_time = time.perf_counter()
    client_ip  = getattr(request.client, "host", "unknown")

    log.info(
        "→ %s %s | IP=%s | UA=%s",
        request.method,
        request.url.path,
        client_ip,
        request.headers.get("user-agent", "unknown")[:80],
    )

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000
        log.info(
            "← %s %s | status=%d | %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
        return response
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000
        log.error(
            "← %s %s | ERROR | %.1fms | %s",
            request.method, request.url.path, duration_ms, exc,
        )
        raise


# ── Gestion globale des exceptions ────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handler pour les HTTPException FastAPI."""
    log.warning("HTTPException %d : %s | path=%s", exc.status_code, exc.detail, request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error":   "HTTP_ERROR",
            "detail":  exc.detail,
            "status":  exc.status_code,
            "path":    str(request.url.path),
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handler pour les ValueError (validation)."""
    log.warning("ValueError : %s | path=%s", exc, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error":   "VALIDATION_ERROR",
            "detail":  str(exc),
            "status":  422,
            "path":    str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler générique pour toutes les exceptions non gérées."""
    log.error("Exception non gérée : %s | path=%s", exc, request.url.path, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error":   "INTERNAL_SERVER_ERROR",
            "detail":  "Une erreur interne s'est produite. Contactez l'équipe KANÉA.",
            "status":  500,
            "path":    str(request.url.path),
        },
    )


# ── Routes de base ─────────────────────────────────────────────────────────────
@app.get("/", summary="Accueil API", tags=["Général"])
async def root():
    """Page d'accueil de l'API BioID AI."""
    return {
        "service":     "KANÉA — BioID AI API",
        "version":     "2.0.0",
        "description": "Estimation du profil biologique forensique",
        "docs":        "/docs",
        "health":      "/bioid/health",
        "status":      "running",
        "models":      "loaded" if is_bundle_loaded() else "not_loaded",
    }


@app.get("/api/version", summary="Version de l'API", tags=["Général"])
async def api_version():
    """Retourne la version de l'API."""
    return {
        "api_version": "2.0.0",
        "module":      "BioID AI",
        "platform":    "KANÉA",
        "framework":   "FastAPI",
    }


# ── Inclusion des routers ──────────────────────────────────────────────────────
if _IMPORTS_OK:
    app.include_router(bioid_router, prefix="/api/v2")
    log.info("Router BioID inclus sous /api/v2/bioid/")
else:
    log.error("Router BioID non disponible — endpoints désactivés")

    @app.get("/api/v2/bioid/health", tags=["BioID AI"])
    async def health_fallback():
        return {"status": "error", "detail": "Module BioID AI non disponible"}


# ── Point d'entrée ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    print("\n" + "=" * 60)
    print("  KANÉA — BioID AI API v2.0.0")
    print("  http://localhost:8000/docs")
    print("=" * 60 + "\n")

    uvicorn.run(
        "bioid_ai.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
