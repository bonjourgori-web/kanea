"""
KANEA AI — Utilitaires partagés
Logging, sauvegarde modèles, gestion erreurs.
"""

import json
import logging
import pickle
from datetime import datetime
from functools import wraps
from pathlib import Path

KANEA_ROOT = Path(__file__).parent.parent


# ─── Logger ────────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Fichier
    log_dir = KANEA_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / f"kanea_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─── Décorateur sécurisé ───────────────────────────────────────────────────────

def safe_run(func):
    """Capture toutes les exceptions sans planter le pipeline."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        log = get_logger(func.__module__ or "kanea")
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            log.error(f"{func.__name__} a échoué : {exc}", exc_info=True)
            return None
    return wrapper


# ─── Sauvegarde / chargement modèles ──────────────────────────────────────────

def save_model_pkl(obj, path: Path, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    if meta:
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


def save_model_pth(state_dict, path: Path, meta: dict | None = None) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
    if meta:
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


def load_model_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Helpers divers ────────────────────────────────────────────────────────────

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def count_images(directory: Path, exts=("png", "jpg", "jpeg")) -> int:
    total = 0
    for ext in exts:
        total += len(list(directory.glob(f"**/*.{ext}")))
    return total


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")
