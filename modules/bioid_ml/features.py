from __future__ import annotations

from typing import Any


CRANIAL_KEYS = [
    "max_cranial_length_mm",
    "max_cranial_breadth_mm",
    "bizygomatic_breadth_mm",
    "nasal_height_mm",
    "nasal_breadth_mm",
    "basion_nasion_length_mm",
]

POSTCRANIAL_KEYS = [
    "femur_length_cm",
    "tibia_length_cm",
    "humerus_length_cm",
    "radius_length_cm",
]

AIMS_KEYS = ["AIM_PC1", "AIM_PC2", "AIM_PC3"]


def build_feature_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cranial = payload.get("cranial_measurements", {}) or {}
    postcranial = payload.get("postcranial_measurements", {}) or {}
    aims = payload.get("aims_pcs", {}) or {}

    features: dict[str, Any] = {}
    for key in CRANIAL_KEYS:
        features[key] = cranial.get(key)
    for key in POSTCRANIAL_KEYS:
        features[key] = postcranial.get(key)
    for key in AIMS_KEYS:
        features[key] = aims.get(key)
    return features

