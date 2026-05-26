from __future__ import annotations

from typing import Any

# ── 15 mesures crâniennes FORDISC (standard international) ──────────────────
CRANIAL_KEYS = [
    "GOL",  # Glabello-occipital length
    "XCB",  # Maximum cranial breadth
    "BBH",  # Basion-bregma height
    "ZYB",  # Bizygomatic breadth
    "AUB",  # Biauricular breadth
    "ASB",  # Biasterionic breadth
    "BNL",  # Basion-nasion length
    "BPL",  # Basion-prosthion length
    "NLH",  # Nasal height
    "NLB",  # Nasal breadth
    "OBH",  # Orbital height
    "OBB",  # Orbital breadth
    "MAB",  # Maxillary (palate) breadth
    "FOL",  # Foramen magnum length
    "FOB",  # Foramen magnum breadth
]

# ── 8 mesures post-crâniennes (toutes en mm) ─────────────────────────────────
POSTCRANIAL_KEYS = [
    "femur_max_length",    # Longueur max fémur
    "femur_bicondylar",    # Longueur bicondylaire fémur
    "tibia_length",        # Longueur tibia
    "humerus_max_length",  # Longueur max humérus
    "radius_max_length",   # Longueur max radius
    "fibula_max_length",   # Longueur max fibula
    "femur_head_diam",     # Diamètre tête fémur
    "humerus_head_diam",   # Diamètre tête humérus
]

AIMS_KEYS = ["AIM_PC1", "AIM_PC2", "AIM_PC3"]

# Toutes les features dans l'ordre attendu par le modèle
ALL_FEATURE_KEYS = CRANIAL_KEYS + POSTCRANIAL_KEYS + AIMS_KEYS


def build_feature_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cranial     = payload.get("cranial_measurements", {}) or {}
    postcranial = payload.get("postcranial_measurements", {}) or {}
    aims        = payload.get("aims_pcs", {}) or {}

    features: dict[str, Any] = {}
    for key in CRANIAL_KEYS:
        features[key] = cranial.get(key)
    for key in POSTCRANIAL_KEYS:
        features[key] = postcranial.get(key)
    for key in AIMS_KEYS:
        features[key] = aims.get(key)
    return features
