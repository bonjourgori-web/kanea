"""
KANÉA — Génération de datasets synthétiques réalistes
======================================================
Génère deux CSV prêts pour l'entraînement :
  1. data/nutrition/nutrition_dataset.csv  (1000 lignes, 5 classes)
  2. data/forensic/forensic_dataset.csv    (800 lignes, profils biologiques)

Cohérence anthropométrique :
  ▸ Nutrition : normes OMS/UNICEF pour enfants 0-60 mois (Côte d'Ivoire)
  ▸ Forensique : mesures craniométriques et post-crâniennes (pop. africaine subsaharienne)

Usage :
    python scripts/generate_synthetic_training_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RNG  = np.random.default_rng(42)
ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — NUTRITION
# ═══════════════════════════════════════════════════════════════════════════════

def _zscore_to_weight(age_months: float, sex: str, waz: float) -> float:
    """Convertit WAZ → poids (kg) selon références OMS simplifées CIV."""
    # Médianes OMS approximées pour 0-60 mois
    median_weight_f = 3.2 + age_months * 0.178
    median_weight_m = 3.4 + age_months * 0.182
    sd = 0.8 + age_months * 0.02
    median = median_weight_f if sex == "F" else median_weight_m
    return round(max(1.5, median + waz * sd), 1)


def _zscore_to_height(age_months: float, sex: str, haz: float) -> float:
    """Convertit HAZ → taille (cm) selon références OMS simplifiées."""
    median_height_f = 49.0 + age_months * 0.83
    median_height_m = 49.8 + age_months * 0.85
    sd = 1.5 + age_months * 0.03
    median = median_height_f if sex == "F" else median_height_m
    return round(max(40.0, median + haz * sd), 1)


def generate_nutrition_dataset(n: int = 1000) -> pd.DataFrame:
    """
    Génère un dataset nutritionnel cohérent avec les normes OMS.

    Classes :
      severe_undernutrition   (WHZ < -3 ou MUAC < 11.5)
      moderate_undernutrition (WHZ -3 à -2 ou MUAC 11.5–12.5)
      normal                  (WHZ -2 à +2)
      overweight              (WHZ +2 à +3)
      obesity                 (WHZ > +3)
    """
    rows = []

    # Distribution des classes : réaliste pour la CIV (SMART Survey 2022)
    class_config = {
        "severe_undernutrition":   {"n": int(n * 0.08), "waz_range": (-5.0, -3.0), "whz_range": (-5.0, -3.0)},
        "moderate_undernutrition": {"n": int(n * 0.18), "waz_range": (-3.0, -2.0), "whz_range": (-3.0, -2.0)},
        "normal":                  {"n": int(n * 0.60), "waz_range": (-2.0,  2.0), "whz_range": (-2.0,  2.0)},
        "overweight":              {"n": int(n * 0.10), "waz_range": ( 2.0,  3.0), "whz_range": ( 2.0,  3.0)},
        "obesity":                 {"n": int(n * 0.04), "waz_range": ( 3.0,  5.0), "whz_range": ( 3.0,  5.0)},
    }

    for label, cfg in class_config.items():
        n_class  = cfg["n"]
        waz_min, waz_max = cfg["waz_range"]
        whz_min, whz_max = cfg["whz_range"]

        for _ in range(n_class):
            sex        = RNG.choice(["F", "M"])
            age_months = RNG.integers(0, 61)
            waz        = round(float(RNG.uniform(waz_min, waz_max)), 2)
            haz        = round(float(RNG.uniform(max(-5.0, waz - 1.5), min(5.0, waz + 1.5))), 2)
            whz        = round(float(RNG.uniform(whz_min, whz_max)), 2)

            weight_kg  = _zscore_to_weight(age_months, sex, waz)
            height_cm  = _zscore_to_height(age_months, sex, haz)
            bmi        = round(weight_kg / ((height_cm / 100) ** 2), 2)

            # MUAC corrélé à l'état nutritionnel
            muac_base  = {"severe_undernutrition": 10.5, "moderate_undernutrition": 12.0,
                          "normal": 14.5, "overweight": 17.0, "obesity": 20.0}[label]
            muac_cm    = round(muac_base + RNG.normal(0, 0.8), 1)
            muac_cm    = round(max(7.0, min(30.0, muac_cm)), 1)

            rows.append({
                "age_months": int(age_months),
                "weight_kg":  weight_kg,
                "height_cm":  height_cm,
                "sex":        sex,
                "muac_cm":    muac_cm,
                "waz":        waz,
                "haz":        haz,
                "whz":        whz,
                "bmi":        bmi,
                "nutritional_status": label,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — FORENSIQUE
# ═══════════════════════════════════════════════════════════════════════════════

# Paramètres anthropométriques pour population africaine subsaharienne
# Sources : Trotter & Gleser (1952), Bass (2005), adaptations CIV
_FORENSIC_PROFILES = {
    "M": {
        "max_cranial_length_mm":    (183.0, 7.5),
        "max_cranial_breadth_mm":   (143.0, 6.5),
        "bizygomatic_breadth_mm":   (133.0, 6.0),
        "nasal_height_mm":          ( 53.0, 4.5),
        "nasal_breadth_mm":         ( 27.0, 3.0),
        "basion_nasion_length_mm":  ( 99.0, 5.0),
        "femur_length_cm":          ( 46.5, 2.5),
        "tibia_length_cm":          ( 38.5, 2.2),
        "humerus_length_cm":        ( 33.0, 2.0),
        "radius_length_cm":         ( 25.0, 1.6),
        "stature_cm":               (170.0, 6.5),
    },
    "F": {
        "max_cranial_length_mm":    (176.0, 7.0),
        "max_cranial_breadth_mm":   (138.0, 6.0),
        "bizygomatic_breadth_mm":   (123.0, 5.5),
        "nasal_height_mm":          ( 49.5, 4.0),
        "nasal_breadth_mm":         ( 25.0, 2.8),
        "basion_nasion_length_mm":  ( 94.0, 4.8),
        "femur_length_cm":          ( 42.0, 2.3),
        "tibia_length_cm":          ( 34.5, 2.0),
        "humerus_length_cm":        ( 30.0, 1.8),
        "radius_length_cm":         ( 22.5, 1.5),
        "stature_cm":               (159.0, 5.8),
    },
}

_ANCESTRY_GROUPS = ["West_African", "East_African", "Central_African", "Admixed"]
_AGE_RANGES = {
    "young_adult":  (20, 35),
    "middle_adult": (35, 50),
    "old_adult":    (50, 70),
    "elderly":      (70, 90),
}


def generate_forensic_dataset(n: int = 800) -> pd.DataFrame:
    """
    Génère un dataset forensique réaliste pour populations africaines subsahariennes.

    Cibles multi-sorties :
      biological_sex   : M / F
      age_at_death     : années (continu)
      ancestry         : West_African / East_African / Central_African / Admixed
      stature_cm       : taille estimée en cm (continu)
    """
    rows = []

    for _ in range(n):
        sex      = RNG.choice(["M", "F"])
        ancestry = RNG.choice(_ANCESTRY_GROUPS, p=[0.50, 0.20, 0.20, 0.10])
        age_cat  = RNG.choice(list(_AGE_RANGES.keys()), p=[0.30, 0.30, 0.25, 0.15])
        age_min, age_max = _AGE_RANGES[age_cat]
        age      = round(float(RNG.uniform(age_min, age_max)), 1)

        profile  = _FORENSIC_PROFILES[sex]
        row: dict = {"biological_sex": sex, "ancestry": ancestry, "age_at_death": age}

        for feat, (mean, sd) in profile.items():
            if feat == "stature_cm":
                continue
            # Légère variation inter-individus + variation ancestrale
            anc_offset = RNG.normal(0, sd * 0.15)
            val = round(float(RNG.normal(mean + anc_offset, sd)), 1)
            # Contraintes physiques
            row[feat] = round(max(mean - 4 * sd, min(mean + 4 * sd, val)), 1)

        # Stature estimée via formule Trotter & Gleser adaptée
        femur = row.get("femur_length_cm", profile["femur_length_cm"][0])
        tibia = row.get("tibia_length_cm", profile["tibia_length_cm"][0])
        if sex == "M":
            stature = round(2.10 * femur + 2.06 * tibia + 72.22 + RNG.normal(0, 3.0), 1)
        else:
            stature = round(1.96 * femur + 1.96 * tibia + 72.65 + RNG.normal(0, 3.5), 1)
        stature_m, stature_sd = profile["stature_cm"]
        row["stature_cm"] = round(max(stature_m - 4 * stature_sd, min(stature_m + 4 * stature_sd, stature)), 1)

        # AIMs (Principal Components ancestraux simulés)
        anc_map = {
            "West_African":    ( 1.5, 0.4,  0.2),
            "East_African":    ( 0.5, 1.8,  0.3),
            "Central_African": ( 0.3, 0.4,  1.6),
            "Admixed":         ( 0.8, 0.7,  0.6),
        }
        pc_means = anc_map[ancestry]
        row["AIM_PC1"] = round(float(RNG.normal(pc_means[0], 0.6)), 3)
        row["AIM_PC2"] = round(float(RNG.normal(pc_means[1], 0.6)), 3)
        row["AIM_PC3"] = round(float(RNG.normal(pc_means[2], 0.5)), 3)

        rows.append(row)

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("KANEA - Generation des datasets synthetiques d'entrainement")
    print("=" * 60)

    # Nutrition
    nutri_dir = ROOT / "data" / "nutrition"
    nutri_dir.mkdir(parents=True, exist_ok=True)
    nutri_path = nutri_dir / "nutrition_dataset.csv"
    df_nutri = generate_nutrition_dataset(n=1000)
    df_nutri.to_csv(nutri_path, index=False)
    print(f"[OK] Nutrition dataset  : {len(df_nutri)} lignes -> {nutri_path}")
    print(f"     Distribution : {df_nutri['nutritional_status'].value_counts().to_dict()}")

    # Forensique
    forensic_dir = ROOT / "data" / "forensic"
    forensic_dir.mkdir(parents=True, exist_ok=True)
    forensic_path = forensic_dir / "forensic_dataset.csv"
    df_forensic = generate_forensic_dataset(n=800)
    df_forensic.to_csv(forensic_path, index=False)
    print(f"[OK] Forensic dataset   : {len(df_forensic)} lignes -> {forensic_path}")
    print(f"     Sexes : {df_forensic['biological_sex'].value_counts().to_dict()}")
    print(f"     Ancestry : {df_forensic['ancestry'].value_counts().to_dict()}")

    print("\n[>] Pour entrainer les modeles :")
    print("    python scripts/train_biometry_model.py")
    print("    python scripts/train_forensic_model.py")


if __name__ == "__main__":
    main()
