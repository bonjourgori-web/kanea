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
    if sex in {"M", "MALE", "H", "HOMME"}:
        return "M"
    if sex in {"F", "FEMALE", "FEMME"}:
        return "F"
    return sex


def encode_sex(sex: str | None) -> int:
    """F=0, M=1 (convention du modèle)."""
    return 1 if sex == "M" else 0


def compute_nutrition_risk_score(
    waz: float | None,
    haz: float | None,
    whz: float | None,
    bmi: float | None,
    muac_cm: float | None,
) -> float:
    """
    Score de risque nutritionnel composite [0–10].
    Basé sur seuils OMS pour les Z-scores et MUAC.
    """
    score = 0.0

    if whz is not None:
        if whz < -3:        score += 4.0
        elif whz < -2:      score += 2.5
        elif whz < -1:      score += 1.0
        elif whz > 2:       score += 1.5  # surpoids

    if waz is not None:
        if waz < -3:        score += 2.0
        elif waz < -2:      score += 1.0

    if haz is not None:
        if haz < -3:        score += 2.0
        elif haz < -2:      score += 1.0

    if muac_cm is not None:
        if muac_cm < 11.5:  score += 4.0
        elif muac_cm < 12.5: score += 2.0
        elif muac_cm < 13.5: score += 1.0

    if bmi is not None:
        if bmi < 16:        score += 2.0
        elif bmi < 18.5:    score += 1.0
        elif bmi > 30:      score += 1.5
        elif bmi > 25:      score += 0.5

    return round(min(score, 10.0), 2)


def get_age_group(age_months: float | None) -> str:
    """Groupe d'âge clinique."""
    if age_months is None:
        return "unknown"
    if age_months < 6:
        return "infant_0_6m"
    if age_months < 24:
        return "infant_6_24m"
    if age_months < 60:
        return "child_2_5y"
    if age_months < 144:
        return "child_5_12y"
    if age_months < 216:
        return "adolescent"
    return "adult"


def classify_bmi(bmi: float | None, age_months: float | None) -> str:
    """Classification IMC selon OMS (adulte et enfant simplifié)."""
    if bmi is None:
        return "unknown"
    is_adult = age_months is None or age_months >= 216
    if is_adult:
        if bmi < 16:    return "severe_thinness"
        if bmi < 17:    return "moderate_thinness"
        if bmi < 18.5:  return "mild_thinness"
        if bmi < 25:    return "normal"
        if bmi < 30:    return "overweight"
        if bmi < 35:    return "obese_class1"
        if bmi < 40:    return "obese_class2"
        return "obese_class3"
    # Enfant — seuils approximatifs (référence CDC/OMS simplifiée)
    if bmi < 14:    return "severe_thinness"
    if bmi < 15.5:  return "mild_thinness"
    if bmi < 22:    return "normal"
    if bmi < 25:    return "overweight"
    return "obese"


def build_feature_payload(payload: dict[str, Any]) -> dict[str, Any]:
    weight_kg  = payload.get("weight_kg")
    height_cm  = payload.get("height_cm")
    age_months = payload.get("age_months")
    sex_raw    = normalize_sex(payload.get("sex"))
    muac_cm    = payload.get("muac_cm")
    waz        = payload.get("waz")
    haz        = payload.get("haz")
    whz        = payload.get("whz")
    bmi        = compute_bmi(weight_kg, height_cm)

    risk_score = compute_nutrition_risk_score(waz, haz, whz, bmi, muac_cm)
    age_group  = get_age_group(age_months)
    bmi_class  = classify_bmi(bmi, age_months)

    return {
        "age_months":           age_months,
        "weight_kg":            weight_kg,
        "height_cm":            height_cm,
        "sex":                  sex_raw,
        "muac_cm":              muac_cm,
        "waz":                  waz,
        "haz":                  haz,
        "whz":                  whz,
        "bmi":                  bmi,
        "sex_encoded":          encode_sex(sex_raw),
        "nutrition_risk_score": risk_score,
        "age_group":            age_group,
        "bmi_class":            bmi_class,
    }
