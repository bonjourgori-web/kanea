"""
NephroAI — Scores cliniques néphrologiques de référence
========================================================
eGFR CKD-EPI 2021 · MDRD · Cockcroft-Gault
KDIGO CKD Staging (G1–G5 + A1–A3) · KDIGO AKI (Stades 1–3)
RIFLE Classification · AKIN Staging
KFRE (Kidney Failure Risk Equation) · Électrolytes critiques

Sources :
  - KDIGO Clinical Practice Guidelines 2022 (CKD, AKI, Diabetes)
  - NKF KDOQI CKD Guidelines 2023
  - Levey et al., CKD-EPI 2021 (Ann Intern Med)
  - Tangri et al., KFRE — JAMA 2016
  - RIFLE — Bellomo et al., Crit Care 2004
  - AKIN — Mehta et al., Crit Care 2007
  - ERA — European Renal Association Best Practice Guidelines 2023
  - MDRD Study — Levey et al., Ann Intern Med 1999
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 1. eGFR CKD-EPI 2021 — Race-free (Levey et al., Ann Intern Med 2021)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EGFRResult:
    egfr_ml_min: float
    formula: str
    ckd_stage: str
    ckd_stage_code: str
    prognosis: str
    dialysis_risk: str
    interpretation: str
    recommendation: str


def compute_egfr_ckd_epi(
    creatinine_umol: float = 88.0,
    age: int = 50,
    sex: str = "M",
) -> EGFRResult:
    """
    CKD-EPI 2021 (race-free) — Levey et al., Ann Intern Med 2021.
    Équation de référence pour estimation DFG (KDIGO 2022, NKF 2023).
    Scr en mg/dL. Résultat en mL/min/1.73m².
    """
    scr = max(0.1, creatinine_umol / 88.4)  # µmol/L → mg/dL
    a   = int(age)
    sex_f = str(sex).upper() == "F"

    if not sex_f:  # Homme
        kappa, alpha = 0.9, -0.302
        if scr <= kappa:
            egfr = 142 * ((scr / kappa) ** alpha) * (0.9938 ** a)
        else:
            egfr = 142 * ((scr / kappa) ** -1.200) * (0.9938 ** a)
    else:          # Femme
        kappa, alpha = 0.7, -0.241
        if scr <= kappa:
            egfr = 142 * ((scr / kappa) ** alpha) * (0.9938 ** a) * 1.012
        else:
            egfr = 142 * ((scr / kappa) ** -1.200) * (0.9938 ** a) * 1.012

    egfr = round(egfr, 1)
    stage, code, prog, dial_risk, interp, reco = _ckd_stage_from_egfr(egfr)

    return EGFRResult(
        egfr_ml_min=egfr, formula="CKD-EPI 2021 (sans race)",
        ckd_stage=stage, ckd_stage_code=code,
        prognosis=prog, dialysis_risk=dial_risk,
        interpretation=interp, recommendation=reco,
    )


def compute_egfr_mdrd(
    creatinine_umol: float = 88.0,
    age: int = 50,
    sex: str = "M",
) -> EGFRResult:
    """
    MDRD 4 variables (Levey et al., Ann Intern Med 1999).
    Moins précis que CKD-EPI pour DFG ≥ 60. Utile en pratique hospitalière.
    eGFR = 175 × Scr^(-1.154) × Age^(-0.203) × [0.742 si Femme].
    """
    scr   = max(0.1, creatinine_umol / 88.4)
    egfr  = 175 * (scr ** -1.154) * (age ** -0.203)
    if str(sex).upper() == "F":
        egfr *= 0.742
    egfr = round(egfr, 1)
    stage, code, prog, dial_risk, interp, reco = _ckd_stage_from_egfr(egfr)

    return EGFRResult(
        egfr_ml_min=egfr, formula="MDRD 4 variables",
        ckd_stage=stage, ckd_stage_code=code,
        prognosis=prog, dialysis_risk=dial_risk,
        interpretation=interp, recommendation=reco,
    )


def compute_cockcroft_gault(
    creatinine_umol: float = 88.0,
    age: int = 50,
    sex: str = "M",
    weight_kg: float = 70.0,
) -> EGFRResult:
    """
    Cockcroft-Gault (Cockcroft & Gault, Nephron 1976).
    CrCl = [(140 - âge) × poids / (72 × Scr mg/dL)] × 0.85 si Femme.
    Utile pour adaptation posologie médicaments néphrotoxiques.
    """
    scr    = max(0.1, creatinine_umol / 88.4)
    sex_f  = str(sex).upper() == "F"
    crcl   = ((140 - age) * weight_kg) / (72 * scr)
    if sex_f:
        crcl *= 0.85
    crcl = round(max(crcl, 1.0), 1)
    stage, code, prog, dial_risk, interp, reco = _ckd_stage_from_egfr(crcl)

    return EGFRResult(
        egfr_ml_min=crcl, formula="Cockcroft-Gault",
        ckd_stage=stage, ckd_stage_code=code,
        prognosis=prog, dialysis_risk=dial_risk,
        interpretation=interp, recommendation=reco,
    )


def _ckd_stage_from_egfr(egfr: float):
    """Détermine le stade CKD KDIGO à partir du DFG estimé."""
    if egfr >= 90:
        return ("MRC Stade 1 (G1)", "G1", "Excellent si pas de marqueurs lésion",
                "< 1%", f"DFG = {egfr} mL/min/1.73m² — Fonction rénale préservée (G1).",
                "Traitement étiologique — contrôle PA < 130/80 — protéinurie si DM/HTA")
    elif egfr >= 60:
        return ("MRC Stade 2 (G2)", "G2", "Bon avec prise en charge précoce",
                "1–2%", f"DFG = {egfr} mL/min/1.73m² — Légère diminution (G2).",
                "Néphroprotection — IEC/ARA2 si protéinurie — HbA1c < 7% si DM")
    elif egfr >= 45:
        return ("MRC Stade 3a (G3a)", "G3a", "Modéré — progression possible",
                "3–5%", f"DFG = {egfr} mL/min/1.73m² — Réduction modérée (G3a).",
                "Consultation néphrologue — éviter AINS/néphrotoxiques — corriger anémie")
    elif egfr >= 30:
        return ("MRC Stade 3b (G3b)", "G3b", "Modéré-sévère — surveillance rapprochée",
                "8–12%", f"DFG = {egfr} mL/min/1.73m² — Réduction modérée-sévère (G3b).",
                "Néphrologue tous les 3–6 mois — préparer épuration extrarénale — vaccins")
    elif egfr >= 15:
        return ("MRC Stade 4 (G4)", "G4", "Sévère — préparation dialyse/transplantation",
                "25–40%", f"DFG = {egfr} mL/min/1.73m² — Réduction sévère (G4).",
                "Éducation dialyse — FAV — inscription transplantation — SGLT2i si DM")
    else:
        return ("MRC Stade 5 (G5) — IRC Terminale", "G5", "IRC terminale",
                "> 80%", f"DFG = {egfr} mL/min/1.73m² — IRC terminale (G5) — dialyse imminente.",
                "Dialyse (HD/DP) ou transplantation rénale — soins palliatifs si non éligible")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. KDIGO CKD Staging + Albuminurie A1–A3
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KDIGOCKDResult:
    egfr: float
    ckd_g_stage: str
    albuminuria_mg_g: float
    albuminuria_stage: str
    combined_risk: str
    progression_risk: str
    cv_risk: str
    recommendation: str


def compute_kdigo_ckd(
    egfr: float = 60.0,
    albuminuria_mg_g: float = 30.0,     # ACR mg/g ou albumine mg/g créatinine
    proteinuria_g_24h: float | None = None,
) -> KDIGOCKDResult:
    """
    KDIGO CKD Classification 2022 — DFG (G1–G5) + Albuminurie (A1–A3).
    Matrice pronostique CGA (Cause + G-stage + A-stage).
    """
    # Albuminurie stage
    if proteinuria_g_24h is not None:
        # Conversion approximative protéinurie → ACR (ratio 1 g/g ≈ 1000 mg/g)
        acr = proteinuria_g_24h * 1000
    else:
        acr = albuminuria_mg_g

    if acr < 30:
        a_stage = "A1 — Normale à légèrement augmentée (< 30 mg/g)"
        a_code  = "A1"
        a_prog  = "Risque de progression CKD faible"
    elif acr < 300:
        a_stage = "A2 — Modérément augmentée (30–300 mg/g)"
        a_code  = "A2"
        a_prog  = "Risque de progression CKD intermédiaire"
    else:
        a_stage = "A3 — Sévèrement augmentée (> 300 mg/g)"
        a_code  = "A3"
        a_prog  = "Risque de progression CKD élevé"

    # DFG stage (simplified)
    if egfr >= 90:    g_code = "G1"
    elif egfr >= 60:  g_code = "G2"
    elif egfr >= 45:  g_code = "G3a"
    elif egfr >= 30:  g_code = "G3b"
    elif egfr >= 15:  g_code = "G4"
    else:             g_code = "G5"

    # Risque combiné (matrice KDIGO 2022)
    risk_matrix = {
        ("G1","A1"):"Faible",  ("G1","A2"):"Modéré", ("G1","A3"):"Élevé",
        ("G2","A1"):"Faible",  ("G2","A2"):"Modéré", ("G2","A3"):"Élevé",
        ("G3a","A1"):"Modéré", ("G3a","A2"):"Élevé", ("G3a","A3"):"Très élevé",
        ("G3b","A1"):"Élevé",  ("G3b","A2"):"Très élevé","G3b_A3":"Très élevé",
        ("G4","A1"):"Très élevé",("G4","A2"):"Très élevé",("G4","A3"):"Très élevé",
        ("G5","A1"):"Très élevé",("G5","A2"):"Très élevé",("G5","A3"):"Très élevé",
    }
    combined_risk = risk_matrix.get((g_code, a_code), "Élevé")

    cv_risk = "Très élevé" if egfr < 45 or acr > 300 else ("Élevé" if egfr < 60 or acr > 30 else "Modéré")

    if combined_risk == "Faible":
        reco = "Surveillance annuelle — contrôle HTA/DM — IEC/ARA2 si protéinurie"
    elif combined_risk == "Modéré":
        reco = "Surveillance semestrielle — néphrologue si progression — SGLT2i si DM"
    elif combined_risk == "Élevé":
        reco = "Néphrologue tous les 3–4 mois — SGLT2i + finerenone + IEC/ARA2 — éviter AINS"
    else:
        reco = ("Néphrologue mensuel — préparer EER (FAV) — inscription transplantation — "
                "phosphate, vitamine D, érythropoïétine si anémie")

    return KDIGOCKDResult(
        egfr=round(egfr, 1), ckd_g_stage=f"{g_code}",
        albuminuria_mg_g=round(acr, 1), albuminuria_stage=a_stage,
        combined_risk=combined_risk, progression_risk=a_prog,
        cv_risk=cv_risk, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. KDIGO AKI Classification — IRA Stades 1–3 (KDIGO 2012)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KDIGOAKIResult:
    aki_stage: int
    label: str
    mortality_risk: str
    dialysis_indication: bool
    trigger: str
    interpretation: str
    recommendation: str


def compute_kdigo_aki(
    creatinine_current_umol: float = 88.0,
    creatinine_baseline_umol: float | None = None,
    creatinine_48h_increase: bool = False,  # augmentation ≥ 26.5 µmol/L en 48h
    urine_output_ml_kg_h: float | None = None,  # diurèse ml/kg/h
    on_dialysis: bool = False,
) -> KDIGOAKIResult:
    """
    KDIGO AKI Classification 2012 — Stades 1–3.
    Critères : créatinine (fold-change vs baseline) + diurèse.
    Stade 3 si dialyse.
    """
    if on_dialysis:
        return KDIGOAKIResult(3, "AKI Stade 3 — Dialyse en cours", "> 50%", True,
                               "Traitement de suppléance rénal actif",
                               "Stade 3 KDIGO — dialyse en cours.",
                               "Néphrologue — bilan récupération — sevrage dialyse si possible")

    stage = 0
    trigger = "Pas d'IRA détectée"

    if creatinine_baseline_umol is not None:
        baseline = max(creatinine_baseline_umol, 44.0)
        ratio    = creatinine_current_umol / baseline
        increase = creatinine_current_umol - baseline

        if ratio >= 3.0 or creatinine_current_umol >= 354:
            stage = 3; trigger = f"Créatinine × {ratio:.1f} ou ≥ 354 µmol/L"
        elif ratio >= 2.0:
            stage = 2; trigger = f"Créatinine × {ratio:.1f} (×2–3 baseline)"
        elif ratio >= 1.5 or (creatinine_48h_increase and increase >= 26.5):
            stage = 1; trigger = f"Créatinine × {ratio:.1f} ou +{increase:.0f} µmol/L en 48h"
    elif creatinine_48h_increase:
        stage   = 1
        trigger = "Augmentation créatinine ≥ 26.5 µmol/L en 48h"

    # Diurèse
    if urine_output_ml_kg_h is not None:
        uo = float(urine_output_ml_kg_h)
        if uo < 0.3:   # anuria > 12h
            stage = max(stage, 3); trigger += " + Anurie < 0.3 mL/kg/h"
        elif uo < 0.5 and stage < 2:  # oligurie
            stage = max(stage, 1); trigger += " + Oligurie < 0.5 mL/kg/h"

    data = {
        0: ("Pas d'IRA KDIGO", "—", False, "Fonction rénale préservée.", "Surveillance hydratation"),
        1: ("IRA Stade 1", "10–20%", False,
            "IRA légère — perte créatinine × 1.5–1.9 ou +26.5 µmol/L/48h.",
            "Arrêt néphrotoxiques — hydratation IV — bilan étiologique (pré/intra/post-rénal)"),
        2: ("IRA Stade 2", "20–40%", False,
            "IRA modérée — créatinine × 2–2.9 baseline.",
            "Néphrologue + réanimateur — hydratation + diurétiques si surcharge — ECBU + écho rénale"),
        3: ("IRA Stade 3 — CRITIQUE", "> 50%", True,
            "IRA sévère — créatinine × 3 ou ≥ 354 µmol/L ou dialyse ou anurie.",
            "URGENCE RÉANIMATION — EER si anurie/surcharge/hyperkaliémie — ICU — néphrologue urgent"),
    }[stage]

    label, mort, dial, interp, reco = data
    return KDIGOAKIResult(
        aki_stage=stage, label=label, mortality_risk=mort,
        dialysis_indication=dial, trigger=trigger,
        interpretation=f"{interp} Déclencheur : {trigger}.",
        recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RIFLE Classification — AKI en réanimation (Bellomo 2004)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RIFLEResult:
    category: str
    code: str
    mortality: str
    creatinine_criteria: str
    urine_criteria: str
    recommendation: str


def compute_rifle(
    egfr_reduction_pct: float = 0.0,
    creatinine_increase_fold: float = 1.0,
    urine_output_ml_kg_h: float | None = None,
    on_dialysis: bool = False,
    recovery: bool = False,
) -> RIFLEResult:
    """
    RIFLE Classification (Bellomo et al., Crit Care 2004).
    Risk · Injury · Failure · Loss · End-stage kidney disease.
    """
    if on_dialysis and not recovery:
        return RIFLEResult("Failure/Loss", "F", "> 50%",
                           "Dialyse en cours", "Anurie ou oligurie persistante",
                           "EER active — réanimation — surveillance SRAA/diurétiques")
    if recovery:
        return RIFLEResult("Loss", "L", "30–40%",
                           "Perte > 4 semaines", "Variable",
                           "Néphrologue — suivi CKD — recherche récupération")

    code = "—"
    uo_crit = "—"

    if urine_output_ml_kg_h is not None:
        uo = float(urine_output_ml_kg_h)
        if uo < 0.3:    uo_crit = "Anurie > 12h — FAILURE"
        elif uo < 0.5:  uo_crit = "< 0.5 mL/kg/h × 12h — INJURY"
        else:           uo_crit = f"{uo:.2f} mL/kg/h — normal"

    if creatinine_increase_fold >= 3.0 or egfr_reduction_pct >= 75:
        code = "F"
        return RIFLEResult("Failure", code, "> 50%",
                           f"Scr × {creatinine_increase_fold:.1f} ou DFG −75%",
                           uo_crit, "Dialyse à envisager — réanimation néphro — surveillance IOA")
    elif creatinine_increase_fold >= 2.0 or egfr_reduction_pct >= 50:
        code = "I"
        return RIFLEResult("Injury", code, "20–40%",
                           f"Scr × {creatinine_increase_fold:.1f} ou DFG −50%",
                           uo_crit, "Néphrologue — arrêt néphrotoxiques — EER si oligurie réfractaire")
    elif creatinine_increase_fold >= 1.5 or egfr_reduction_pct >= 25:
        code = "R"
        return RIFLEResult("Risk", code, "10–20%",
                           f"Scr × {creatinine_increase_fold:.1f} ou DFG −25%",
                           uo_crit, "Surveillance horaire diurèse — hydratation — bilan étiologique")
    return RIFLEResult("No AKI", "—", "< 5%", "Pas de critère", uo_crit,
                       "Surveillance hydratation et créatinine")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. KFRE — Kidney Failure Risk Equation (Tangri 2016)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KFREResult:
    risk_2yr_pct: float
    risk_5yr_pct: float
    risk_category: str
    dialysis_planning: str
    interpretation: str
    recommendation: str


def compute_kfre(
    age: int = 60,
    sex: str = "M",
    egfr: float = 30.0,
    acr_mg_g: float = 300.0,
) -> KFREResult:
    """
    KFRE 4-variable (Tangri et al., JAMA Intern Med 2016).
    Prédit le risque de progression vers ESRD (dialyse/transplantation) à 2 et 5 ans.
    Validé chez patients MRC stades 3–5.
    """
    sex_male = 1 if str(sex).upper() == "M" else 0
    acr      = max(1.0, float(acr_mg_g))

    # 2-year equation
    lp2 = (-0.2201 * (age / 10 - 7.036) +
            0.2467 * (sex_male - 0.5642) -
            0.5567 * (egfr / 5 - 7.222) +
            0.4510 * (math.log(acr / 1000 - (-4.0228) if acr >= 1 else 0.01)))
    risk_2 = round((1 - 0.9832 ** math.exp(lp2)) * 100, 1)

    # 5-year equation (approx. avec survival 0.9365)
    risk_5 = round(min((1 - 0.9365 ** math.exp(lp2)) * 100, 95.0), 1)

    risk_2 = max(0.1, min(risk_2, 99.9))
    risk_5 = max(0.5, min(risk_5, 99.9))

    if risk_2 < 5:
        cat = "Faible risque progression"
        plan = "Surveillance CKD standard tous les 6 mois"
        reco = "IEC/ARA2 + SGLT2i — contrôle PA — HbA1c — néphrologue annuel"
    elif risk_2 < 20:
        cat = "Risque intermédiaire"
        plan = "Préparer l'EER — éducation dialyse/transplantation — FAV 1–2 ans"
        reco = "Néphrologue tous les 3 mois — FAV si HD envisagée — évaluer transplantation préemptive"
    elif risk_2 < 40:
        cat = "Risque élevé"
        plan = "Inscription transplantation — créer FAV maintenant"
        reco = "Inscription transplantation urgente — création FAV — DP si préféré — vaccins"
    else:
        cat = "Très haut risque — ESRD imminent"
        plan = "EER à initier < 1 an — FAV fonctionnelle requise"
        reco = "Initiation EER planifiée — FAV/cathéter — DP si CI HD — soins palliatifs si non-dialyse"

    return KFREResult(
        risk_2yr_pct=risk_2, risk_5yr_pct=risk_5,
        risk_category=cat, dialysis_planning=plan,
        interpretation=f"KFRE : risque dialyse/transplantation à 2 ans = {risk_2}%, à 5 ans = {risk_5}%.",
        recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ÉLECTROLYTES — Analyse critique K+, Na+, Ca²+, Phosphate
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ElectrolyteResult:
    potassium_mmol: float
    sodium_mmol: float
    calcium_mmol: Optional[float]
    phosphate_mmol: Optional[float]
    potassium_status: str
    sodium_status: str
    calcium_status: str
    phosphate_status: str
    critical_flags: list
    urgency: str
    recommendation: str


def compute_electrolytes(
    potassium: float = 4.5,     # mmol/L  (normal 3.5–5.0)
    sodium: float = 140.0,      # mmol/L  (normal 135–145)
    calcium: float | None = None,   # mmol/L  (normal 2.2–2.6)
    phosphate: float | None = None, # mmol/L  (normal 0.87–1.45)
    bicarbonate: float | None = None,  # mmol/L
) -> ElectrolyteResult:
    """
    Analyse électrolytique critique — alertes K+, Na+, Ca²+, Phosphate.
    Seuils KDIGO 2022 + urgences cliniques réanimation.
    """
    flags = []
    urgency = "Faible"

    # Potassium
    k = float(potassium)
    if k >= 6.5:
        k_status = f"Hyperkaliémie sévère critique ({k:.1f} mmol/L)"
        flags.append({"param":"K+","value":k,"status":"CRITIQUE","detail":"ECG + Ca gluconate + kayexalate + EER si réfractaire"})
        urgency = "Critique"
    elif k >= 5.5:
        k_status = f"Hyperkaliémie modérée ({k:.1f} mmol/L)"
        flags.append({"param":"K+","value":k,"status":"ÉLEVÉ","detail":"Régime pauvre K+ — résines échangeuses — surveiller ECG"})
        urgency = "Élevée"
    elif k <= 2.5:
        k_status = f"Hypokaliémie sévère ({k:.1f} mmol/L)"
        flags.append({"param":"K+","value":k,"status":"CRITIQUE","detail":"Supplémentation IV urgente — risque arythmies"})
        urgency = "Critique"
    elif k <= 3.5:
        k_status = f"Hypokaliémie légère ({k:.1f} mmol/L)"
        flags.append({"param":"K+","value":k,"status":"MODÉRÉ","detail":"Supplémentation orale KCl"})
        if urgency == "Faible": urgency = "Modérée"
    else:
        k_status = f"Potassium normal ({k:.1f} mmol/L)"

    # Sodium
    na = float(sodium)
    if na > 155 or na < 120:
        na_status = f"{'Hypernatrémie' if na>155 else 'Hyponatrémie'} sévère ({na:.0f} mmol/L)"
        flags.append({"param":"Na+","value":na,"status":"CRITIQUE","detail":"Correction lente — risque engagement cérébral si correction rapide"})
        urgency = "Critique"
    elif na > 148 or na < 130:
        na_status = f"{'Hypernatrémie' if na>148 else 'Hyponatrémie'} modérée ({na:.0f} mmol/L)"
        flags.append({"param":"Na+","value":na,"status":"ÉLEVÉ","detail":"Correction progressive 6–8 mmol/L/24h"})
        if urgency == "Faible": urgency = "Élevée"
    else:
        na_status = f"Sodium normal ({na:.0f} mmol/L)"

    # Calcium
    ca_status = "Non évalué"
    if calcium is not None:
        ca = float(calcium)
        if ca > 3.0 or ca < 1.75:
            ca_status = f"{'Hypercalcémie' if ca>3.0 else 'Hypocalcémie'} sévère ({ca:.2f} mmol/L)"
            flags.append({"param":"Ca²+","value":ca,"status":"CRITIQUE","detail":"Perfusion saline + furosémide (hypercalc) ou Ca gluconate IV (hypocalc)"})
            urgency = "Critique"
        elif ca > 2.65 or ca < 2.15:
            ca_status = f"{'Hypercalcémie' if ca>2.65 else 'Hypocalcémie'} légère ({ca:.2f} mmol/L)"
        else:
            ca_status = f"Calcium normal ({ca:.2f} mmol/L)"

    # Phosphate
    ph_status = "Non évalué"
    if phosphate is not None:
        ph = float(phosphate)
        if ph > 2.0:
            ph_status = f"Hyperphosphatémie ({ph:.2f} mmol/L)"
            flags.append({"param":"PO4","value":ph,"status":"MODÉRÉ","detail":"Chélateurs phosphate alimentaire — restriction phosphore"})
        elif ph < 0.4:
            ph_status = f"Hypophosphatémie sévère ({ph:.2f} mmol/L)"
            flags.append({"param":"PO4","value":ph,"status":"ÉLEVÉ","detail":"Supplémentation IV phosphate"})
        else:
            ph_status = f"Phosphate normal ({ph:.2f} mmol/L)"

    if urgency == "Critique":
        reco = "URGENCE métabolique — ECG — bilan complet — réanimation si instable"
    elif urgency == "Élevée":
        reco = "Bilan ionique complet — traitement électrolytique — surveillance monitoring"
    elif urgency == "Modérée":
        reco = "Surveillance électrolytes — correction progressive — régime adapté"
    else:
        reco = "Ionogramme normal — contrôle selon CKD stage"

    return ElectrolyteResult(
        potassium_mmol=k, sodium_mmol=na,
        calcium_mmol=float(calcium) if calcium else None,
        phosphate_mmol=float(phosphate) if phosphate else None,
        potassium_status=k_status, sodium_status=na_status,
        calcium_status=ca_status, phosphate_status=ph_status,
        critical_flags=flags, urgency=urgency, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ORCHESTRATEUR — compute_all_nephro_scores()
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_nephro_scores(params: dict[str, Any]) -> dict[str, Any]:
    """Orchestre tous les scores néphrologiques."""
    age     = int(params.get("age", 50))
    sex     = str(params.get("sex", "M"))
    creat   = float(params.get("creatinine", 88.0))
    weight  = float(params.get("weight", 70.0))
    acr     = float(params.get("albuminuria_mg_g") or params.get("proteinuria_g_24h", 0) * 1000 or 30)
    baseline = params.get("creatinine_baseline")
    egfr_provided = params.get("egfr")

    results: dict[str, Any] = {}

    # eGFR CKD-EPI
    ckd_epi = compute_egfr_ckd_epi(creat, age, sex)
    results["egfr_ckdepi"] = vars(ckd_epi)
    egfr_val = float(egfr_provided) if egfr_provided else ckd_epi.egfr_ml_min

    # MDRD
    results["egfr_mdrd"] = vars(compute_egfr_mdrd(creat, age, sex))

    # Cockcroft-Gault
    results["cockcroft_gault"] = vars(compute_cockcroft_gault(creat, age, sex, weight))

    # KDIGO CKD
    prot = params.get("proteinuria_g_24h")
    results["kdigo_ckd"] = vars(compute_kdigo_ckd(
        egfr=egfr_val,
        albuminuria_mg_g=acr,
        proteinuria_g_24h=float(prot) if prot else None,
    ))

    # KDIGO AKI
    results["kdigo_aki"] = vars(compute_kdigo_aki(
        creatinine_current_umol=creat,
        creatinine_baseline_umol=float(baseline) if baseline else None,
        creatinine_48h_increase=bool(params.get("creatinine_rapid_rise", False)),
        urine_output_ml_kg_h=params.get("urine_output_ml_kg_h"),
        on_dialysis=bool(params.get("on_dialysis", False)),
    ))

    # RIFLE
    baseline_v = float(baseline) if baseline else creat * 0.7
    fold = creat / max(baseline_v, 44.0)
    egfr_red = max(0.0, (1 - egfr_val / max(_ref_egfr(age, sex), 1)) * 100)
    results["rifle"] = vars(compute_rifle(
        egfr_reduction_pct=egfr_red,
        creatinine_increase_fold=fold,
        urine_output_ml_kg_h=params.get("urine_output_ml_kg_h"),
    ))

    # KFRE (CKD stades 3–5)
    if egfr_val < 60:
        results["kfre"] = vars(compute_kfre(age=age, sex=sex, egfr=egfr_val, acr_mg_g=acr))

    # Électrolytes
    results["electrolytes"] = vars(compute_electrolytes(
        potassium=float(params.get("potassium", 4.5)),
        sodium=float(params.get("sodium", 140)),
        calcium=params.get("calcium"),
        phosphate=params.get("phosphate"),
        bicarbonate=params.get("bicarbonate"),
    ))

    return results


def _ref_egfr(age: int, sex: str) -> float:
    """DFG de référence estimé pour un adulte sain (approximation)."""
    base = 120.0 if str(sex).upper() == "M" else 110.0
    return max(30.0, base - max(0, age - 40) * 0.8)
