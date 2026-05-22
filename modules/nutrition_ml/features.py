from __future__ import annotations

from typing import Any


def compute_bmi(weight_kg: float | None, height_cm: float | None) -> float | None:
    if weight_kg is None or height_cm is None:
        return None
    if weight_kg <= 0 or height_cm <= 0:
        return None
    height_m = height_cm / 100.0
    return round(weight_kg / (height_m * height_m), 2)


def normalize_sex(sex: str | None) -> str | None:
    if not sex:
        return None
    sex = str(sex).strip().upper()
    if sex in {"M", "MALE", "H"}:
        return "M"
    if sex in {"F", "FEMALE"}:
        return "F"
    return sex


def encode_sex(sex: str | None) -> int:
    """Encode le sexe en entier : F=0, M=1 (convention du modèle entraîné)."""
    if sex == "M":
        return 1
    return 0  # F par défaut


def build_feature_payload(payload: dict[str, Any]) -> dict[str, Any]:
    weight_kg = payload.get("weight_kg")
    height_cm = payload.get("height_cm")
    sex_raw   = normalize_sex(payload.get("sex"))

    features = {
        "age_months":  payload.get("age_months"),
        "weight_kg":   weight_kg,
        "height_cm":   height_cm,
        "sex":         sex_raw,
        "muac_cm":     payload.get("muac_cm"),
        "waz":         payload.get("waz"),
        "haz":         payload.get("haz"),
        "whz":         payload.get("whz"),
        "bmi":         compute_bmi(weight_kg, height_cm),
        "sex_encoded": encode_sex(sex_raw),
    }
    return features
