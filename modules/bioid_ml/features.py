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

# ── 3 composantes ancestrales (AIMs PCA) ─────────────────────────────────────
AIMS_KEYS = ["AIM_PC1", "AIM_PC2", "AIM_PC3"]

# ── 4 indices dérivés (engineer_bioid_features) ───────────────────────────────
# Identiques à ceux calculés lors de l'entraînement — DOIVENT être présents
# dans le vecteur de features soumis au modèle.
DERIVED_KEYS = [
    "cephalic_idx",  # XCB / GOL × 100
    "nasal_idx",     # NLB / NLH × 100
    "orbital_idx",   # OBH / OBB × 100
    "femur_ratio",   # femur_max_length / femur_head_diam
]

# Ordre exact du modèle entraîné (ALL_MORPHO = CRANIAL + POSTCRANIAL puis dérivés + AIMs)
ALL_FEATURE_KEYS = CRANIAL_KEYS + POSTCRANIAL_KEYS + DERIVED_KEYS + AIMS_KEYS


def _safe_div(a: float | None, b: float | None) -> float | None:
    """Division sûre : retourne None si l'un des opérandes est absent ou nul."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def build_feature_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Construit le vecteur de features complet (mesures brutes + indices dérivés).
    Reproduit exactement engineer_bioid_features() utilisé à l'entraînement.
    """
    cranial     = payload.get("cranial_measurements", {}) or {}
    postcranial = payload.get("postcranial_measurements", {}) or {}
    aims        = payload.get("aims_pcs", {}) or {}

    features: dict[str, Any] = {}

    # ── Mesures brutes ────────────────────────────────────────────────────────
    for key in CRANIAL_KEYS:
        features[key] = cranial.get(key)
    for key in POSTCRANIAL_KEYS:
        features[key] = postcranial.get(key)

    # ── Indices dérivés (même formules que engineer_bioid_features) ───────────
    GOL = features.get("GOL")
    XCB = features.get("XCB")
    NLH = features.get("NLH")
    NLB = features.get("NLB")
    OBH = features.get("OBH")
    OBB = features.get("OBB")
    fem = features.get("femur_max_length")
    fhd = features.get("femur_head_diam")

    features["cephalic_idx"] = (_safe_div(XCB, GOL) * 100) if (XCB and GOL) else None
    features["nasal_idx"]    = (_safe_div(NLB, NLH) * 100) if (NLB and NLH) else None
    features["orbital_idx"]  = (_safe_div(OBH, OBB) * 100) if (OBH and OBB) else None
    features["femur_ratio"]  = _safe_div(fem, fhd)

    # ── Marqueurs ancestraux (PCA déjà calculée côté client) ─────────────────
    for key in AIMS_KEYS:
        features[key] = aims.get(key)

    return features
