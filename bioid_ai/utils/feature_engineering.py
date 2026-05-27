"""
feature_engineering.py — Construction et enrichissement des features BioID AI.

Réutilise la logique de modules/bioid_ml/features.py et ajoute :
- engineer_features() pour construire le DataFrame complet depuis un payload dict
- Imputation des NaN par médiane
- Calcul des formules de stature
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Clés des mesures (reprises depuis modules/bioid_ml/features.py) ──────────
CRANIAL_KEYS = [
    "GOL", "XCB", "BBH", "ZYB", "AUB", "ASB",
    "BNL", "BPL", "NLH", "NLB", "OBH", "OBB",
    "MAB", "FOL", "FOB",
]

POSTCRANIAL_KEYS = [
    "femur_max_length", "femur_bicondylar", "tibia_length",
    "humerus_max_length", "radius_max_length", "fibula_max_length",
    "femur_head_diam", "humerus_head_diam",
]

AIMS_KEYS = ["AIM_PC1", "AIM_PC2", "AIM_PC3"]

DERIVED_KEYS = [
    "cephalic_idx",  # XCB / GOL × 100
    "nasal_idx",     # NLB / NLH × 100
    "orbital_idx",   # OBH / OBB × 100
    "femur_ratio",   # femur_max_length / femur_head_diam
]

ALL_FEATURE_KEYS = CRANIAL_KEYS + POSTCRANIAL_KEYS + DERIVED_KEYS + AIMS_KEYS

# Médianes de référence (basées sur la littérature FORDISC — population mixte)
REFERENCE_MEDIANS: dict[str, float] = {
    "GOL": 182.0, "XCB": 137.0, "BBH": 132.0, "ZYB": 130.0,
    "AUB": 121.0, "ASB": 111.0, "BNL": 100.0, "BPL": 98.0,
    "NLH": 51.0,  "NLB": 26.0,  "OBH": 33.0,  "OBB": 41.0,
    "MAB": 64.0,  "FOL": 37.0,  "FOB": 30.0,
    "femur_max_length": 450.0, "femur_bicondylar": 445.0,
    "tibia_length": 365.0, "humerus_max_length": 320.0,
    "radius_max_length": 245.0, "fibula_max_length": 358.0,
    "femur_head_diam": 45.0, "humerus_head_diam": 43.0,
    "AIM_PC1": 0.0, "AIM_PC2": 0.0, "AIM_PC3": 0.0,
    "cephalic_idx": 75.0, "nasal_idx": 51.0,
    "orbital_idx": 80.0, "femur_ratio": 10.0,
}


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Division sûre : retourne None si un opérande est absent ou nul."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def compute_derived_indices(features: dict[str, Any]) -> dict[str, Any]:
    """Calcule les 4 indices dérivés à partir des mesures brutes."""
    GOL = features.get("GOL")
    XCB = features.get("XCB")
    NLH = features.get("NLH")
    NLB = features.get("NLB")
    OBH = features.get("OBH")
    OBB = features.get("OBB")
    fem = features.get("femur_max_length")
    fhd = features.get("femur_head_diam")

    cephalic = (_safe_div(XCB, GOL) * 100) if (XCB and GOL) else None
    nasal    = (_safe_div(NLB, NLH) * 100) if (NLB and NLH) else None
    orbital  = (_safe_div(OBH, OBB) * 100) if (OBH and OBB) else None
    femur_r  = _safe_div(fem, fhd)

    features["cephalic_idx"] = cephalic
    features["nasal_idx"]    = nasal
    features["orbital_idx"]  = orbital
    features["femur_ratio"]  = femur_r
    return features


def estimate_stature(femur_bicondylar: Optional[float], sex: str = "unknown") -> Optional[float]:
    """
    Estime la stature à partir de la longueur bicondylaire du fémur.

    Formules de Trotter & Gleser (1958) adaptées :
      Masculin  : 2.32 × femur_bicondylar + 65.53
      Féminin   : 2.47 × femur_bicondylar + 54.10
      Inconnu   : moyenne des deux formules
    Résultat en cm (l'entrée doit être en mm → conversion automatique).
    """
    if femur_bicondylar is None or femur_bicondylar <= 0:
        return None

    # Conversion mm → cm si la valeur semble être en mm (> 100)
    val_cm = femur_bicondylar / 10.0 if femur_bicondylar > 100 else femur_bicondylar

    sex_lower = sex.lower() if sex else "unknown"
    if "male" in sex_lower and "fe" not in sex_lower:
        return round(2.32 * val_cm + 65.53, 1)
    elif "fe" in sex_lower or "femm" in sex_lower or "woman" in sex_lower:
        return round(2.47 * val_cm + 54.10, 1)
    else:
        male_est   = 2.32 * val_cm + 65.53
        female_est = 2.47 * val_cm + 54.10
        return round((male_est + female_est) / 2, 1)


def engineer_features(payload: dict[str, Any]) -> pd.DataFrame:
    """
    Construit le DataFrame de features complet depuis un payload dict.

    Le payload peut avoir les formes :
      - {cranial: {...}, postcranial: {...}, aims: {...}}
      - {cranial_measurements: {...}, postcranial_measurements: {...}, aims_pcs: {...}}
      - clé plate directement au niveau racine

    Retourne un DataFrame à 1 ligne avec toutes les colonnes ALL_FEATURE_KEYS.
    Les valeurs manquantes sont imputées par les médianes de référence.
    """
    try:
        # Support des deux formats de clé
        cranial: dict = (
            payload.get("cranial") or
            payload.get("cranial_measurements") or
            {}
        )
        postcranial: dict = (
            payload.get("postcranial") or
            payload.get("postcranial_measurements") or
            {}
        )
        aims_raw: dict = (
            payload.get("aims") or
            payload.get("aims_pcs") or
            {}
        )

        features: dict[str, Any] = {}

        # Mesures brutes
        for key in CRANIAL_KEYS:
            val = cranial.get(key) or payload.get(key)
            features[key] = float(val) if val is not None else None

        for key in POSTCRANIAL_KEYS:
            val = postcranial.get(key) or payload.get(key)
            features[key] = float(val) if val is not None else None

        # AIMs PCA
        for key in AIMS_KEYS:
            val = aims_raw.get(key) or payload.get(key)
            features[key] = float(val) if val is not None else None

        # Indices dérivés
        features = compute_derived_indices(features)

        # Imputation par médiane de référence
        for key in ALL_FEATURE_KEYS:
            if features.get(key) is None or (
                isinstance(features.get(key), float) and np.isnan(features[key])
            ):
                features[key] = REFERENCE_MEDIANS.get(key, 0.0)

        # Construction du DataFrame dans l'ordre exact du modèle
        row = {k: features.get(k, 0.0) for k in ALL_FEATURE_KEYS}
        df = pd.DataFrame([row], columns=ALL_FEATURE_KEYS)
        log.debug("engineer_features shape=%s", df.shape)
        return df

    except Exception as exc:
        log.error("engineer_features error: %s", exc)
        # Retourne un DataFrame vide mais valide
        return pd.DataFrame([[0.0] * len(ALL_FEATURE_KEYS)], columns=ALL_FEATURE_KEYS)


def payload_from_schema(cranial: dict, postcranial: dict, aims: dict) -> dict[str, Any]:
    """
    Construit un payload normalisé depuis les données des schémas Pydantic.
    Compatible avec build_feature_payload() du module existant.
    """
    return {
        "cranial_measurements": cranial,
        "postcranial_measurements": postcranial,
        "aims_pcs": aims,
        # Alias pour engineer_features()
        "cranial": cranial,
        "postcranial": postcranial,
        "aims": aims,
    }
