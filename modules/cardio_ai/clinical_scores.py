"""
CardioSense AI — Scores cliniques cardiovasculaires de référence
================================================================
SCORE2 · Framingham Risk Score · ASCVD Pooled Cohort Equations
CHA₂DS₂-VASc · HAS-BLED · GRACE Score · TIMI (STEMI/NSTEMI)
NYHA Classification · ECG QTc (Bazett/Fridericia/Hodges) · MAGGIC

Sources :
  - ESC Guidelines 2021 (SCORE2 — Visseren et al., Eur Heart J 2021)
  - Framingham Heart Study (D'Agostino et al., Circulation 2008)
  - ACC/AHA Pooled Cohort Equations (Goff et al., JACC 2014)
  - GRACE Registry (Fox et al., BMJ 2006 — ESC NSTEMI Guidelines 2020)
  - TIMI Risk Score (Morrow et al., JAMA 2000 — Antman et al., JAMA 2000)
  - ESC AF Guidelines 2020 (CHA₂DS₂-VASc — Lip et al.)
  - HAS-BLED (Pisters et al., Chest 2010)
  - NYHA Functional Classification (NYHA 1994)
  - Bazett (1920) · Fridericia (1920) · Hodges (1983) — QTc corrections
  - ESC Heart Failure Guidelines 2021
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SCORE2 — Risque cardiovasculaire à 10 ans (ESC 2021)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SCORE2Result:
    risk_percent: float
    risk_category: str
    region: str
    event_type: str
    interpretation: str
    treatment_threshold: str
    recommendation: str


def compute_score2(
    age: int = 55,
    sex: str = "M",
    smoking: bool = False,
    systolic_bp: float = 130.0,
    non_hdl_cholesterol: float = 3.9,   # mmol/L (TC − HDL)
    region: str = "high",               # low, moderate, high, very_high
) -> SCORE2Result:
    """
    SCORE2 — European Systematic COronary Risk Evaluation (ESC 2021).
    Estime le risque d'événements cardiovasculaires fatals + non fatals à 10 ans.
    Validé chez les adultes de 40–69 ans sans maladie cardiovasculaire préexistante.
    Regions : low (Western Europe) · moderate · high (Eastern Europe, Africa) · very_high.
    """
    a = max(40, min(69, int(age)))
    sex_f = str(sex).upper() == "F"

    # Facteurs de risque normalisés
    age_norm  = (a - 60) / 5.0
    sbp_norm  = (systolic_bp - 120) / 20.0
    chol_norm = (non_hdl_cholesterol - 3.9) / 1.0
    smk_val   = 1.0 if smoking else 0.0

    # Coefficients log-risque simplifiés (adaptés de l'article ESC 2021)
    if not sex_f:  # Homme
        lp = (0.3742 * age_norm +
              0.6012 * smk_val +
              0.2777 * sbp_norm +
              0.1458 * chol_norm)
    else:          # Femme
        lp = (0.4648 * age_norm +
              0.7744 * smk_val +
              0.3131 * sbp_norm +
              0.1002 * chol_norm)

    # Calibration par région (multiplicateurs approximatifs)
    region_mult = {"low": 0.65, "moderate": 1.0, "high": 1.45, "very_high": 2.0}
    mult = region_mult.get(region.lower(), 1.45)

    # Probabilité de base (baseline hazard simplifié)
    base_risk = 0.05 if not sex_f else 0.03
    risk_raw  = base_risk * math.exp(lp) * mult
    risk_pct  = round(min(risk_raw * 100, 40.0), 1)

    # Catégorisation selon ESC 2021
    age_group = "< 50 ans" if a < 50 else ("50–69 ans" if a < 70 else "≥ 70 ans")

    if a < 50:
        if risk_pct < 2.5:
            cat = "Faible"
        elif risk_pct < 7.5:
            cat = "Modéré"
        elif risk_pct < 10:
            cat = "Élevé"
        else:
            cat = "Très élevé"
    else:
        if risk_pct < 5:
            cat = "Faible"
        elif risk_pct < 10:
            cat = "Modéré"
        elif risk_pct < 20:
            cat = "Élevé"
        else:
            cat = "Très élevé"

    thresholds = {
        "Faible":      "Modification du style de vie — statines non systématiques",
        "Modéré":      "Statines si LDL > 2.6 mmol/L — objectif LDL < 2.6 mmol/L",
        "Élevé":       "Statines haute intensité — objectif LDL < 1.8 mmol/L",
        "Très élevé":  "Statines + ézétimibe ± iPCSK9 — objectif LDL < 1.4 mmol/L",
    }
    recos = {
        "Faible":      "Contrôle des FRCV (PA, glycémie, cholestérol, tabac). Exercice 150 min/semaine.",
        "Modéré":      "Statines modérées (atorvastatine 10–20 mg). Régime méditerranéen. Objectif LDL < 2.6.",
        "Élevé":       "Statines haute intensité (atorvastatine 40–80 mg). Bilan cardiologique. PA < 130/80.",
        "Très élevé":  "Traitement médicamenteux intensif. Consultation cardiologue urgente. Évaluer iPCSK9.",
    }

    return SCORE2Result(
        risk_percent=risk_pct, risk_category=cat,
        region=region.title(), event_type="Événements CV fatals + non fatals à 10 ans",
        interpretation=f"SCORE2 = {risk_pct}% ({cat}) — {age_group}, région {region}.",
        treatment_threshold=thresholds.get(cat, "—"),
        recommendation=recos.get(cat, "—"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FRAMINGHAM RISK SCORE — Risque coronarien à 10 ans
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FraminghamResult:
    risk_percent: float
    risk_category: str
    heart_age: int
    sex: str
    interpretation: str
    ldl_target: str
    recommendation: str


def compute_framingham(
    age: int = 55,
    sex: str = "M",
    total_cholesterol: float = 5.2,   # mmol/L
    hdl_cholesterol: float = 1.3,     # mmol/L
    systolic_bp: float = 130.0,
    bp_treated: bool = False,
    smoking: bool = False,
    diabetes: bool = False,
) -> FraminghamResult:
    """
    Framingham Risk Score (D'Agostino et al., Circulation 2008).
    Risque d'événement coronarien à 10 ans chez adultes 30–74 ans sans ATCD CV.
    Utilise les équations de régression de Cox validées sur 8 491 patients.
    """
    a = max(30, min(74, int(age)))
    tc  = total_cholesterol * 38.67     # mmol/L → mg/dL
    hdl = hdl_cholesterol   * 38.67
    sbp = float(systolic_bp)
    smk = 1 if smoking else 0
    dm  = 1 if diabetes else 0

    sex_f = str(sex).upper() == "F"

    if not sex_f:  # Homme
        ln_age  = math.log(a)
        ln_tc   = math.log(tc)
        ln_hdl  = math.log(hdl)
        ln_sbp  = math.log(sbp) * (1.764 if bp_treated else 1.764)
        ln_sbp_t = math.log(sbp) if bp_treated else 0.0
        ln_smk  = math.log(1) if not smk else math.log(2)

        score = (
            3.06117 * ln_age +
            1.12370 * ln_tc  -
            0.93263 * ln_hdl +
            1.93303 * (ln_sbp_t if bp_treated else 0) +
            1.99881 * (math.log(sbp) if not bp_treated else 0) +
            0.65451 * smk +
            0.57367 * dm
        )
        baseline_surv = 0.88936
        mean_coeff    = 23.9802
        risk_raw = 1 - baseline_surv ** math.exp(score - mean_coeff)
        heart_age_base = 45
    else:          # Femme
        ln_age  = math.log(a)
        ln_tc   = math.log(tc)
        ln_hdl  = math.log(hdl)

        score = (
            2.32888 * ln_age +
            1.20904 * ln_tc  -
            0.70833 * ln_hdl +
            2.76157 * (math.log(sbp) if bp_treated else 0) +
            2.82263 * (math.log(sbp) if not bp_treated else 0) +
            0.52873 * smk +
            0.69154 * dm
        )
        baseline_surv = 0.94833
        mean_coeff    = 26.1931
        risk_raw = 1 - baseline_surv ** math.exp(score - mean_coeff)
        heart_age_base = 40

    risk_pct  = round(min(max(risk_raw * 100, 0.5), 50.0), 1)

    # Âge cardiaque estimé
    heart_age = int(a + (risk_pct - 10) * 0.8)
    heart_age = max(30, min(90, heart_age))

    if risk_pct < 10:
        cat  = "Faible risque (< 10%)"
        ldlt = "LDL < 3.0 mmol/L"
        reco = "Modification style de vie. Statines si LDL > 3.0 mmol/L avec FRCV additionnels."
    elif risk_pct <= 20:
        cat  = "Risque intermédiaire (10–20%)"
        ldlt = "LDL < 2.6 mmol/L"
        reco = "Statines modérées. Aspirine 100 mg si > 50 ans. PA < 140/90 mmHg."
    else:
        cat  = "Risque élevé (> 20%)"
        ldlt = "LDL < 1.8 mmol/L"
        reco = "Statines haute intensité + ézétimibe si LDL non contrôlé. Bilan coronarien."

    return FraminghamResult(
        risk_percent=risk_pct, risk_category=cat,
        heart_age=heart_age, sex="Femme" if sex_f else "Homme",
        interpretation=f"Framingham {risk_pct}% ({cat}). Âge cardiaque estimé : {heart_age} ans.",
        ldl_target=ldlt, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CHA₂DS₂-VASc — Risque thromboembolique en FA (ESC 2020)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CHA2DS2VAScResult:
    score: int
    components: dict
    annual_stroke_risk: str
    anticoagulation_recommendation: str
    preferred_anticoagulant: str
    interpretation: str


def compute_cha2ds2_vasc(
    chf: bool = False,             # Insuffisance cardiaque (C)
    hypertension: bool = True,     # HTA (H)
    age: int = 70,                 # ≥ 75 ans = 2 pts (A₂) ; 65-74 = 1 pt (A)
    diabetes: bool = False,        # Diabète (D)
    prior_stroke_tia: bool = False,# ATCD AVC/AIT (S₂ = 2 pts)
    vascular_disease: bool = False,# Maladie vasculaire (V)
    sex_female: bool = False,      # Sexe féminin (Sc) — uniquement si score ≥ 1
) -> CHA2DS2VAScResult:
    """
    CHA₂DS₂-VASc Score (Lip et al., Chest 2010 — ESC AF Guidelines 2020).
    Stratification du risque d'AVC et thromboembolisme en fibrillation auriculaire.
    Score 0–9. Anticoagulation si ≥ 1 (homme) ou ≥ 2 (femme).
    """
    comps: dict[str, int] = {}
    s = 0

    comps["CHF (C)"]              = 1 if chf else 0;              s += comps["CHF (C)"]
    comps["HTA (H)"]              = 1 if hypertension else 0;     s += comps["HTA (H)"]
    comps["Âge ≥ 75 (A₂)"]       = 2 if age >= 75 else 0;        s += comps["Âge ≥ 75 (A₂)"]
    comps["Diabète (D)"]          = 1 if diabetes else 0;         s += comps["Diabète (D)"]
    comps["AVC/AIT (S₂)"]         = 2 if prior_stroke_tia else 0; s += comps["AVC/AIT (S₂)"]
    comps["Maladie vasc. (V)"]    = 1 if vascular_disease else 0; s += comps["Maladie vasc. (V)"]
    comps["Âge 65–74 (A)"]        = 1 if 65 <= age < 75 else 0;   s += comps["Âge 65–74 (A)"]
    comps["Sexe féminin (Sc)"]    = 1 if sex_female else 0;        s += comps["Sexe féminin (Sc)"]

    # Risque annuel d'AVC selon score (Lip et al.)
    risk_table = {
        0: ("0%",     "Pas d'anticoagulation — risque nul"),
        1: ("1.3%",   "Anticoagulation recommandée chez l'homme"),
        2: ("2.2%",   "Anticoagulation fortement recommandée"),
        3: ("3.2%",   "Anticoagulation obligatoire"),
        4: ("4.0%",   "Anticoagulation obligatoire"),
        5: ("6.7%",   "Anticoagulation obligatoire — risque très élevé"),
        6: ("9.8%",   "Anticoagulation obligatoire — risque très élevé"),
        7: ("9.6%",   "Anticoagulation obligatoire"),
        8: ("6.7%",   "Anticoagulation obligatoire"),
        9: ("15.2%",  "Anticoagulation obligatoire — risque critique"),
    }
    annual_risk, anticoag_reco = risk_table.get(min(s, 9), ("—", "—"))

    # Anticoagulant préféré
    if s == 0 or (s == 1 and sex_female):
        preferred = "Pas d'anticoagulation (risque faible)"
    elif prior_stroke_tia:
        preferred = "AOD : apixaban 5 mg ×2/j ou rivaroxaban 20 mg/j (1er choix ESC 2020)"
    else:
        preferred = "AOD préféré : apixaban · rivaroxaban · dabigatran · édoxaban"

    return CHA2DS2VAScResult(
        score=s, components=comps, annual_stroke_risk=annual_risk,
        anticoagulation_recommendation=anticoag_reco,
        preferred_anticoagulant=preferred,
        interpretation=(
            f"CHA₂DS₂-VASc = {s}/9 — Risque AVC annuel ≈ {annual_risk}. "
            f"{anticoag_reco}."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HAS-BLED — Risque hémorragique sous anticoagulation (ESC 2020)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HASBLEDResult:
    score: int
    components: dict
    annual_bleeding_risk: str
    risk_category: str
    modifiable_factors: list
    interpretation: str
    recommendation: str


def compute_has_bled(
    hypertension_uncontrolled: bool = False,   # PAS > 160 mmHg
    renal_dysfunction: bool = False,            # Dialyse, transplantation, créat > 200 µmol/L
    liver_dysfunction: bool = False,            # Cirrhose, bilirubine > 2×, ASAT/ALAT > 3×
    prior_stroke: bool = False,                 # ATCD AVC (hors hémorragique)
    prior_bleeding: bool = False,               # ATCD hémorragie ou prédisposition
    labile_inr: bool = False,                   # INR instable (TTR < 60%)
    age_gt65: bool = False,                     # Âge > 65 ans
    antiplatelet_nsaid: bool = False,           # Antiplaquettaires ou AINS
    alcohol_excess: bool = False,               # Alcool ≥ 8 unités/semaine
) -> HASBLEDResult:
    """
    HAS-BLED Score (Pisters et al., Chest 2010 — ESC 2020).
    Évalue le risque annuel d'hémorragie majeure sous anticoagulation en FA.
    Score ≥ 3 = risque élevé → ne contre-indique pas l'anticoagulation,
    mais impose correction des facteurs modifiables.
    """
    comps: dict[str, int] = {}
    s = 0

    comps["HTA non contrôlée (H)"] = 1 if hypertension_uncontrolled else 0; s += comps["HTA non contrôlée (H)"]
    comps["Dysfonction rénale (A)"] = 1 if renal_dysfunction else 0;         s += comps["Dysfonction rénale (A)"]
    comps["Dysfonction hépatique (A)"] = 1 if liver_dysfunction else 0;      s += comps["Dysfonction hépatique (A)"]
    comps["ATCD AVC (S)"]           = 1 if prior_stroke else 0;              s += comps["ATCD AVC (S)"]
    comps["ATCD hémorragie (B)"]    = 1 if prior_bleeding else 0;            s += comps["ATCD hémorragie (B)"]
    comps["INR labile (L)"]         = 1 if labile_inr else 0;                s += comps["INR labile (L)"]
    comps["Âge > 65 ans (E)"]       = 1 if age_gt65 else 0;                 s += comps["Âge > 65 ans (E)"]
    comps["Médicaments (D)"]        = 1 if antiplatelet_nsaid else 0;        s += comps["Médicaments (D)"]
    comps["Alcool (D)"]             = 1 if alcohol_excess else 0;            s += comps["Alcool (D)"]

    risk_table = {
        0: ("0.9%",  "Faible"),
        1: ("3.4%",  "Faible"),
        2: ("4.1%",  "Modéré"),
        3: ("5.8%",  "Élevé — corriger facteurs modifiables"),
        4: ("8.9%",  "Élevé"),
        5: ("9.1%",  "Très élevé"),
    }
    annual_risk, risk_cat = risk_table.get(min(s, 5), ("≥ 10%", "Très élevé"))

    modifiable = []
    if hypertension_uncontrolled: modifiable.append("Contrôler HTA (PA < 140/90 mmHg)")
    if labile_inr:                modifiable.append("Optimiser TTR AVK > 65% ou switcher vers AOD")
    if antiplatelet_nsaid:        modifiable.append("Arrêter AINS/antiplaquettaires si possible")
    if alcohol_excess:             modifiable.append("Réduire consommation alcool < 8 unités/sem")

    if s <= 2:
        reco = "Anticoagulation maintenue. Risque hémorragique acceptable vs bénéfice thromboprofilactique."
    elif s == 3:
        reco = (
            "Risque élevé — ne pas arrêter anticoagulation. "
            "Corriger facteurs modifiables. Suivi INR mensuel si AVK. Préférer AOD."
        )
    else:
        reco = (
            "Risque très élevé — revue complète par cardiologue. "
            "Corriger tous facteurs modifiables. AOD préféré. Évaluer rapport bénéfice/risque."
        )

    return HASBLEDResult(
        score=s, components=comps, annual_bleeding_risk=annual_risk,
        risk_category=risk_cat, modifiable_factors=modifiable,
        interpretation=(
            f"HAS-BLED = {s}/9 — Risque hémorragique annuel ≈ {annual_risk} ({risk_cat}). "
            f"{len(modifiable)} facteur(s) modifiable(s) identifié(s)."
        ),
        recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GRACE SCORE — Stratification du risque en SCA (ESC 2020)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GRACEResult:
    score: int
    mortality_30d: str
    mortality_6m: str
    risk_category: str
    invasive_strategy: str
    interpretation: str
    recommendation: str


def compute_grace(
    age: int = 65,
    heart_rate: float = 88.0,
    systolic_bp: float = 130.0,
    creatinine: float = 120.0,     # µmol/L
    killip_class: int = 1,         # 1=pas IC, 2=crépitants, 3=OAP, 4=choc
    cardiac_arrest: bool = False,
    st_deviation: bool = False,    # sus-décalage ou sous-décalage
    elevated_troponin: bool = False,
) -> GRACEResult:
    """
    GRACE Score 2.0 (Fox et al., BMJ 2006 — ESC NSTEMI/STEMI Guidelines 2020).
    Stratification du risque à court terme en syndrome coronarien aigu.
    Prédit la mortalité intra-hospitalière et à 6 mois.
    """
    s = 0

    # Âge (0–100 pts)
    age_pts = {(0,29):0,(30,39):0,(40,49):18,(50,59):36,(60,69):55,(70,79):73,(80,89):91,(90,200):100}
    for (lo, hi), pts in age_pts.items():
        if lo <= age <= hi:
            s += pts; break

    # FC (0–46 pts)
    hr_pts = {(0,49):0,(50,69):3,(70,89):9,(90,109):15,(110,149):24,(150,199):38,(200,999):46}
    hr = float(heart_rate)
    for (lo, hi), pts in hr_pts.items():
        if lo <= hr <= hi:
            s += pts; break

    # PAS (0–58 pts)
    sbp_pts = {(0,79):58,(80,99):53,(100,119):43,(120,139):34,(140,159):24,(160,199):10,(200,999):0}
    sbp = float(systolic_bp)
    for (lo, hi), pts in sbp_pts.items():
        if lo <= sbp <= hi:
            s += pts; break

    # Créatinine (0–28 pts)
    crea = float(creatinine) / 88.4  # µmol/L → mg/dL
    crea_pts = {(0,0.39):1,(0.40,0.79):4,(0.80,1.19):7,(1.20,1.59):10,(1.60,1.99):13,(2.00,3.99):21,(4.00,999):28}
    for (lo, hi), pts in crea_pts.items():
        if lo <= crea <= hi:
            s += pts; break

    # Killip (0–24 pts)
    killip_pts = {1:0, 2:20, 3:39, 4:59}
    s += killip_pts.get(killip_class, 0)

    # Arrêt cardiaque
    s += 39 if cardiac_arrest else 0

    # Déviation ST
    s += 28 if st_deviation else 0

    # Troponine
    s += 14 if elevated_troponin else 0

    # Interprétation GRACE 2.0
    if s < 109:
        cat    = "Faible risque"
        mort30 = "< 1%"
        mort6m = "< 3%"
        inv    = "Stratégie conservatrice ou invasive élective (≤ 72h)"
        reco   = (
            "Coronarographie non urgente (≤ 72h si NSTEMI). "
            "Bithérapie antiplaquettaire. Statines haute intensité. BB + IEC."
        )
    elif s <= 140:
        cat    = "Risque intermédiaire"
        mort30 = "1–3%"
        mort6m = "3–8%"
        inv    = "Stratégie invasive précoce (< 24h)"
        reco   = (
            "Coronarographie dans les 24h. DAPT (aspirine + ticagrélor/prasugrel). "
            "Anticoagulation (fondaparinux ou HBPM). Bêtabloquant. Statines."
        )
    else:
        cat    = "Haut risque"
        mort30 = "> 3%"
        mort6m = "> 8%"
        inv    = "Stratégie invasive très précoce (< 2h si STEMI ou instabilité)"
        reco   = (
            "Coronarographie urgente (< 2h si STEMI/instabilité, < 24h si NSTEMI haut risque). "
            "ICP primaire si STEMI. DAPT + anticoagulation. Aspirine 325 mg + ticagrélor."
        )

    return GRACEResult(
        score=s, mortality_30d=mort30, mortality_6m=mort6m,
        risk_category=cat, invasive_strategy=inv,
        interpretation=f"GRACE {s} pts — {cat} — Mortalité 30j ≈ {mort30}.",
        recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TIMI SCORE — Risque dans le SCA (NSTEMI et STEMI)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TIMIResult:
    score: int
    score_type: str
    risk_percent: str
    risk_category: str
    interpretation: str
    recommendation: str


def compute_timi_nstemi(
    age_ge65: bool = False,
    cad_risk_factors: int = 3,      # ≥ 3 FRCV parmi : ATCD familiaux, HTA, hyperlipidémie, tabac, diabète
    known_cad: bool = False,         # sténose coronarienne ≥ 50% documentée
    aspirin_use_7days: bool = False, # usage aspirine < 7j (paradoxal)
    severe_angina: bool = False,     # ≥ 2 épisodes angor < 24h
    st_changes: bool = False,        # sus- ou sous-décalage ST ≥ 0.5 mm
    positive_troponin: bool = False, # marqueurs cardiaques élevés
) -> TIMIResult:
    """
    TIMI Risk Score for UA/NSTEMI (Antman et al., JAMA 2000).
    Prédit le risque d'événement à 14 jours (décès, IDM, revascularisation urgente).
    Score 0–7.
    """
    s = sum([
        1 if age_ge65 else 0,
        1 if cad_risk_factors >= 3 else 0,
        1 if known_cad else 0,
        1 if aspirin_use_7days else 0,
        1 if severe_angina else 0,
        1 if st_changes else 0,
        1 if positive_troponin else 0,
    ])

    risk_table = {
        0: ("4.7%",   "Faible"),
        1: ("4.7%",   "Faible"),
        2: ("8.3%",   "Faible"),
        3: ("13.2%",  "Intermédiaire"),
        4: ("19.9%",  "Intermédiaire"),
        5: ("26.2%",  "Élevé"),
        6: ("40.9%",  "Élevé"),
        7: ("40.9%",  "Élevé"),
    }
    risk_pct, cat = risk_table.get(min(s, 7), ("—", "—"))

    if cat == "Faible":
        reco = "Stratégie conservatrice. Coronarographie élective si symptômes persistants."
    elif cat == "Intermédiaire":
        reco = "Stratégie invasive dans les 24–48h. DAPT + anticoagulation. Monitoring."
    else:
        reco = "Stratégie invasive urgente (< 2–24h). DAPT + anticoagulation. Coronarographie."

    return TIMIResult(
        score=s, score_type="TIMI NSTEMI/UA",
        risk_percent=risk_pct, risk_category=cat,
        interpretation=f"TIMI UA/NSTEMI = {s}/7 — {cat} — Risque événement J14 ≈ {risk_pct}.",
        recommendation=reco,
    )


def compute_timi_stemi(
    age: int = 65,
    systolic_bp: float = 100.0,
    heart_rate: float = 100.0,
    killip_class: int = 2,
    anterior_stemi: bool = True,    # STEMI antérieur ou BBG
    diabetes_or_htn: bool = True,
    weight_le67: bool = False,       # Poids ≤ 67 kg
    time_to_treatment_ge4h: bool = False,  # Délai > 4h
    prior_angina_mi: bool = False,
) -> TIMIResult:
    """
    TIMI Risk Score for STEMI (Morrow et al., Circulation 2000).
    Prédit la mortalité à 30 jours après STEMI traité par fibrinolyse.
    Score 0–14.
    """
    s = 0

    # Âge
    if 65 <= age < 75:    s += 2
    elif age >= 75:        s += 3

    # Killip > I
    if killip_class > 1:  s += 2

    # Signes vitaux
    if systolic_bp < 100: s += 3
    if heart_rate > 100:  s += 2

    # STEMI antérieur ou BBG
    if anterior_stemi:    s += 1

    # Diabète ou HTA
    if diabetes_or_htn:   s += 1

    # Poids ≤ 67 kg
    if weight_le67:        s += 1

    # Délai > 4h
    if time_to_treatment_ge4h: s += 1

    # ATCD angor ou IDM
    if prior_angina_mi:   s += 1

    risk_table = {
        0: ("0.8%", "Faible"), 1: ("1.6%", "Faible"), 2: ("2.2%", "Faible"),
        3: ("4.4%", "Modéré"), 4: ("7.3%", "Modéré"), 5: ("12.4%", "Élevé"),
        6: ("16.1%", "Élevé"), 7: ("23.4%", "Élevé"), 8: ("26.8%", "Élevé"),
        9: ("35.9%", "Critique"), 10: ("35.9%", "Critique"),
        11: ("46.0%", "Critique"), 12: ("52.5%", "Critique"),
        13: ("52.5%", "Critique"), 14: ("52.5%", "Critique"),
    }
    risk_pct, cat = risk_table.get(min(s, 14), ("—", "—"))

    if cat in ("Faible", "Modéré"):
        reco = "ICP primaire < 90 min. Aspirine + clopidogrel/ticagrélor. Héparine IV."
    else:
        reco = (
            "ICP primaire urgente < 60 min. Double DAPT. Anticoagulation. "
            "Thrombus aspiration si nécessaire. USI coronaire."
        )

    return TIMIResult(
        score=s, score_type="TIMI STEMI",
        risk_percent=risk_pct, risk_category=cat,
        interpretation=f"TIMI STEMI = {s}/14 — {cat} — Mortalité J30 ≈ {risk_pct}.",
        recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. NYHA CLASSIFICATION — Insuffisance cardiaque (NYHA 1994)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NYHAResult:
    class_number: int
    class_label: str
    symptoms: str
    one_year_mortality: str
    ef_category: str
    interpretation: str
    recommendation: str


def compute_nyha(
    nyha_class: int = 2,
    ejection_fraction: float | None = None,
    nt_probnp: float | None = None,
) -> NYHAResult:
    """
    NYHA Functional Classification (New York Heart Association 1994).
    Classification fonctionnelle de l'insuffisance cardiaque sur 4 classes.
    Intègre FEVG et NT-proBNP pour préciser le type (HFrEF/HFmrEF/HFpEF).
    """
    nc = max(1, min(4, int(nyha_class)))

    data = {
        1: {
            "label":    "Classe I — Pas de limitation",
            "symptoms": "Activité physique habituelle sans symptômes. IC asymptomatique.",
            "mort1y":   "5–10%",
            "interp":   "Insuffisance cardiaque asymptomatique. Excellent pronostic à court terme.",
            "reco":     "IEC/ARA2 + bêtabloquant. Optimisation des FRCV. Suivi cardiologique 6 mois.",
        },
        2: {
            "label":    "Classe II — Limitation légère",
            "symptoms": "Dyspnée à l'effort intense. Confort au repos.",
            "mort1y":   "10–15%",
            "interp":   "Limitation modérée de l'activité physique. Traitement médical optimal indiqué.",
            "reco":     "IEC/ARA2 + bêtabloquant + diurétiques si rétention. Restriction sel < 5 g/j.",
        },
        3: {
            "label":    "Classe III — Limitation marquée",
            "symptoms": "Dyspnée à l'effort minimal. Confort seulement au repos.",
            "mort1y":   "20–30%",
            "interp":   "Insuffisance cardiaque décompensée. Risque d'hospitalisation élevé.",
            "reco":     (
                "IEC/ARA2 ou sacubitril-valsartan + bêtabloquant + spironolactone + diurétiques. "
                "SGLT2i (dapagliflozine). Évaluer ICD ou TRC."
            ),
        },
        4: {
            "label":    "Classe IV — Limitation sévère",
            "symptoms": "Dyspnée au repos. Incapacité à toute activité sans symptômes.",
            "mort1y":   "40–60%",
            "interp":   "IC terminale. Hospitalisation souvent requise. Pronostic sévère.",
            "reco":     (
                "Traitement IV : diurétiques IV (furosémide), inotropes (dobutamine). "
                "Évaluer transplantation cardiaque ou LVAD. Soins palliatifs si irréversible."
            ),
        },
    }

    d = data[nc]

    # EF category
    ef = float(ejection_fraction) if ejection_fraction is not None else None
    if ef is not None:
        if ef < 40:     ef_cat = f"HFrEF — FEVG réduite ({ef:.0f}%)"
        elif ef < 50:   ef_cat = f"HFmrEF — FEVG légèrement réduite ({ef:.0f}%)"
        else:           ef_cat = f"HFpEF — FEVG préservée ({ef:.0f}%)"
    else:
        ef_cat = "FEVG non renseignée — échocardiographie recommandée"

    bnp_note = ""
    if nt_probnp is not None:
        if nt_probnp > 2000:
            bnp_note = f" NT-proBNP = {nt_probnp:.0f} pg/mL — IC décompensée (très élevé)."
        elif nt_probnp > 900:
            bnp_note = f" NT-proBNP = {nt_probnp:.0f} pg/mL — IC probable."
        elif nt_probnp > 300:
            bnp_note = f" NT-proBNP = {nt_probnp:.0f} pg/mL — IC possible."

    return NYHAResult(
        class_number=nc, class_label=d["label"], symptoms=d["symptoms"],
        one_year_mortality=d["mort1y"], ef_category=ef_cat,
        interpretation=d["interp"] + bnp_note,
        recommendation=d["reco"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. QTc — Correction de l'intervalle QT (Bazett, Fridericia, Hodges)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QTcResult:
    qt_ms: float
    rr_ms: float
    heart_rate: float
    qtc_bazett: float
    qtc_fridericia: float
    qtc_hodges: float
    qtc_used: float
    qtc_status: str
    prolongation_risk: str
    interpretation: str
    recommendation: str


def compute_qtc(
    qt_ms: float = 380.0,
    heart_rate: float = 75.0,
    sex: str = "M",
    method: str = "bazett",
) -> QTcResult:
    """
    Correction de l'intervalle QT selon Bazett (1920), Fridericia (1920), Hodges (1983).
    QTc normal : < 440 ms (H) / < 460 ms (F).
    QTc ≥ 500 ms : risque de torsades de pointes (TdP) critique.
    """
    hr   = max(20.0, min(300.0, float(heart_rate)))
    qt   = max(200.0, min(800.0, float(qt_ms)))
    rr   = 60000.0 / hr        # RR en ms
    rr_s = rr / 1000.0         # RR en secondes

    qtc_baz  = qt / math.sqrt(rr_s)
    qtc_fri  = qt / (rr_s ** (1/3))
    qtc_hod  = qt + 1.75 * (hr - 60)

    qtc_map  = {"bazett": qtc_baz, "fridericia": qtc_fri, "hodges": qtc_hod}
    qtc_used = qtc_map.get(method.lower(), qtc_baz)

    sex_f    = str(sex).upper() == "F"
    upper_n  = 460 if sex_f else 440

    if qtc_used >= 500:
        status  = "QTc CRITIQUE ≥ 500 ms"
        prolif  = "Risque élevé de torsades de pointes (TdP)"
        reco    = (
            "ARRÊTER médicaments allongeant le QT immédiatement. "
            "Surveillance ECG continue. Corriger hypokaliémie/hypomagnésémie. "
            "Sulfate de magnésium IV si TdP. Cardiologue urgent."
        )
    elif qtc_used >= 480:
        status  = "QTc Très prolongé ≥ 480 ms"
        prolif  = "Risque modéré-élevé de TdP"
        reco    = (
            "Réduire/arrêter médicaments QT-prolongateurs. "
            "ECG de contrôle. Corriger électrolytes. Avis cardiologique."
        )
    elif qtc_used > upper_n:
        status  = f"QTc Prolongé > {upper_n} ms"
        prolif  = "Risque faible mais surveillance requise"
        reco    = (
            f"Surveillance ECG. Vérifier médicaments QT-prolongateurs. "
            "Corriger électrolytes. Cardiologue si symptômes."
        )
    else:
        status  = f"QTc Normal (≤ {upper_n} ms)"
        prolif  = "Risque de TdP faible"
        reco    = "Pas d'intervention requise. Surveillance standard si traitement QT-actif."

    method_name = method.capitalize()
    return QTcResult(
        qt_ms=round(qt, 1), rr_ms=round(rr, 1), heart_rate=round(hr, 1),
        qtc_bazett=round(qtc_baz, 1), qtc_fridericia=round(qtc_fri, 1),
        qtc_hodges=round(qtc_hod, 1), qtc_used=round(qtc_used, 1),
        qtc_status=status, prolongation_risk=prolif,
        interpretation=(
            f"QTc ({method_name}) = {qtc_used:.0f} ms — {status}. "
            f"Valeur normale : < {upper_n} ms. RR = {rr:.0f} ms (FC = {hr:.0f} bpm)."
        ),
        recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. ANALYSE ECG — Paramètres et détections automatiques
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ECGAnalysisResult:
    heart_rate: float
    rhythm: str
    pr_interval_ms: float
    qrs_duration_ms: float
    qt_ms: float
    axis_degrees: Optional[float]
    findings: list
    abnormalities: list
    urgency: str
    interpretation: str
    recommendation: str


def compute_ecg_analysis(
    heart_rate: float = 75.0,
    pr_interval: float = 160.0,     # ms (normal 120–200)
    qrs_duration: float = 90.0,     # ms (normal 60–120)
    qt_interval: float = 380.0,     # ms
    axis_degrees: float | None = None,
    p_waves_present: bool = True,
    regular_rhythm: bool = True,
    st_elevation_leads: list | None = None,   # ex: ["V1","V2","V3"]
    st_depression_leads: list | None = None,
    t_wave_inversion_leads: list | None = None,
    lbbb: bool = False,
    rbbb: bool = False,
    lvh_criteria: bool = False,
    sex: str = "M",
) -> ECGAnalysisResult:
    """
    Analyse automatisée des paramètres ECG 12 dérivations.
    Détecte arythmies, troubles de conduction, ischémie, hypertrophie.
    """
    hr = float(heart_rate)
    pr = float(pr_interval)
    qrs = float(qrs_duration)
    qt  = float(qt_interval)
    st_elev  = st_elevation_leads or []
    st_dep   = st_depression_leads or []
    t_inv    = t_wave_inversion_leads or []

    findings     = []
    abnormalities = []
    urgency_level = "Faible"

    # Rythme
    if not p_waves_present and not regular_rhythm:
        rhythm = "Fibrillation auriculaire probable"
        abnormalities.append("FA — Absence ondes P, rythme irrégulier")
        urgency_level = "Élevée"
    elif not p_waves_present and regular_rhythm:
        rhythm = "Rythme jonctionnel ou flutter probable"
        abnormalities.append("Absence ondes P avec rythme régulier")
        urgency_level = "Modérée"
    elif hr < 60:
        rhythm = f"Bradycardie sinusale ({hr:.0f} bpm)"
        if hr < 40:
            abnormalities.append(f"Bradycardie sévère {hr:.0f} bpm")
            urgency_level = "Critique"
        else:
            findings.append(f"Bradycardie sinusale {hr:.0f} bpm")
    elif hr > 100:
        rhythm = f"Tachycardie ({hr:.0f} bpm)"
        if hr > 150:
            abnormalities.append(f"Tachycardie rapide {hr:.0f} bpm — TSV ou TV?")
            urgency_level = "Élevée"
        else:
            findings.append(f"Tachycardie sinusale {hr:.0f} bpm")
    else:
        rhythm = f"Rythme sinusal ({hr:.0f} bpm)"

    # Intervalle PR
    if pr > 200:
        abnormalities.append(f"BAV 1er degré (PR = {pr:.0f} ms > 200 ms)")
        if urgency_level == "Faible": urgency_level = "Modérée"
    elif pr > 250:
        abnormalities.append(f"BAV 1er degré marqué (PR = {pr:.0f} ms)")
        urgency_level = "Modérée"

    # QRS
    if qrs > 120:
        if lbbb:
            abnormalities.append(f"Bloc de branche gauche complet (QRS {qrs:.0f} ms)")
            urgency_level = "Élevée"
        elif rbbb:
            abnormalities.append(f"Bloc de branche droit (QRS {qrs:.0f} ms)")
            if urgency_level == "Faible": urgency_level = "Modérée"
        else:
            abnormalities.append(f"QRS élargi ({qrs:.0f} ms > 120 ms) — BBG/BBD/TV?")
            urgency_level = "Modérée"

    # ST
    if st_elev:
        abnormalities.append(f"Sus-décalage ST : {', '.join(st_elev)} — STEMI POSSIBLE")
        urgency_level = "Critique"
    if st_dep:
        abnormalities.append(f"Sous-décalage ST : {', '.join(st_dep)} — Ischémie/NSTEMI")
        if urgency_level not in ("Critique",): urgency_level = "Élevée"
    if t_inv:
        findings.append(f"Inversion onde T : {', '.join(t_inv)}")

    # QTc
    rr_s = 60.0 / hr
    qtc  = qt / math.sqrt(rr_s) if rr_s > 0 else qt
    upper_qtc = 460 if str(sex).upper() == "F" else 440
    if qtc >= 500:
        abnormalities.append(f"QTc CRITIQUE = {qtc:.0f} ms — Risque TdP")
        urgency_level = "Critique"
    elif qtc > upper_qtc:
        findings.append(f"QTc prolongé = {qtc:.0f} ms (normal < {upper_qtc} ms)")

    # Axe électrique
    axis_note = ""
    if axis_degrees is not None:
        ax = float(axis_degrees)
        if ax < -30:
            axis_note = f"Déviation axiale gauche ({ax:.0f}°)"
            findings.append(axis_note)
        elif ax > 110:
            axis_note = f"Déviation axiale droite ({ax:.0f}°)"
            findings.append(axis_note)

    # LVH
    if lvh_criteria:
        findings.append("Critères d'hypertrophie ventriculaire gauche (HVG)")

    # Interprétation finale
    all_issues = abnormalities + findings
    if urgency_level == "Critique":
        interp = f"ECG ANORMAL CRITIQUE — {'; '.join(abnormalities[:3])}."
        reco   = "URGENCE CARDIOLOGIQUE — Cardiologue de garde immédiatement. ECG continu."
    elif urgency_level == "Élevée":
        interp = f"ECG anormal — {'; '.join(all_issues[:3])}."
        reco   = "Consultation cardiologue urgente. ECG de contrôle 1h. Monitoring."
    elif urgency_level == "Modérée":
        interp = f"ECG avec anomalies modérées — {'; '.join(all_issues[:2])}."
        reco   = "Bilan cardiologique. ECG Holter 24h si symptômes. Écho-Doppler cardiaque."
    elif all_issues:
        interp = f"ECG avec variations mineures — {'; '.join(all_issues[:2])}."
        reco   = "Surveillance standard. ECG annuel si ≥ 50 ans ou FRCV."
    else:
        interp = f"ECG normal — Rythme sinusal {hr:.0f} bpm, axe normal."
        reco   = "Pas d'anomalie ECG détectée. Contrôle selon indication clinique."

    return ECGAnalysisResult(
        heart_rate=round(hr, 1), rhythm=rhythm,
        pr_interval_ms=round(pr, 1), qrs_duration_ms=round(qrs, 1),
        qt_ms=round(qt, 1), axis_degrees=axis_degrees,
        findings=findings, abnormalities=abnormalities,
        urgency=urgency_level, interpretation=interp, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 10. ORCHESTRATEUR — compute_all_cardio_scores()
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_cardio_scores(params: dict[str, Any]) -> dict[str, Any]:
    """
    Orchestre le calcul de tous les scores cardiovasculaires selon les paramètres fournis.
    """
    results: dict[str, Any] = {}
    age     = int(params.get("age", 55))
    sex     = str(params.get("sex", "M"))
    hr      = float(params.get("heart_rate", 75))
    sbp     = float(params.get("systolic_bp", 130))
    tc      = float(params.get("cholesterol_total", 5.2))
    hdl     = float(params.get("hdl", 1.3))
    smk     = str(params.get("smoking", "Non")).lower() in ("oui", "true", "yes")
    dm      = bool(params.get("diabetes", False))
    htn     = bool(params.get("hypertension", False))
    bp_tx   = bool(params.get("bp_treatment", False))

    # SCORE2
    non_hdl = max(0.5, tc - hdl)
    results["score2"] = vars(compute_score2(
        age=age, sex=sex, smoking=smk,
        systolic_bp=sbp, non_hdl_cholesterol=non_hdl,
        region=params.get("score2_region", "high"),
    ))

    # Framingham
    results["framingham"] = vars(compute_framingham(
        age=age, sex=sex, total_cholesterol=tc, hdl_cholesterol=hdl,
        systolic_bp=sbp, bp_treated=bp_tx, smoking=smk, diabetes=dm,
    ))

    # CHA₂DS₂-VASc (si FA documentée ou suspectée)
    if params.get("prior_af") or params.get("af_rhythm") or params.get("flutter"):
        results["cha2ds2_vasc"] = vars(compute_cha2ds2_vasc(
            chf=bool(params.get("heart_failure", False)),
            hypertension=htn,
            age=age,
            diabetes=dm,
            prior_stroke_tia=bool(params.get("prior_stroke", False)),
            vascular_disease=bool(params.get("vascular_disease", False)),
            sex_female=str(sex).upper() == "F",
        ))
        results["has_bled"] = vars(compute_has_bled(
            hypertension_uncontrolled=sbp > 160,
            renal_dysfunction=bool(params.get("renal_disease", False)),
            liver_dysfunction=bool(params.get("liver_disease", False)),
            prior_stroke=bool(params.get("prior_stroke", False)),
            prior_bleeding=bool(params.get("bleeding_history", False)),
            labile_inr=bool(params.get("labile_inr", False)),
            age_gt65=age > 65,
            antiplatelet_nsaid=bool(params.get("on_antiplatelet", False)),
            alcohol_excess=bool(params.get("alcohol_drugs", False)),
        ))

    # GRACE (si SCA)
    trop = float(params.get("troponin", 0.01))
    trop_elevated = trop > 0.04
    st_elev = bool(params.get("st_elevation", False))
    if trop_elevated or st_elev or params.get("chest_pain"):
        results["grace"] = vars(compute_grace(
            age=age, heart_rate=hr, systolic_bp=sbp,
            creatinine=float(params.get("creatinine", 88.0)),
            killip_class=int(params.get("killip_class", 1)),
            cardiac_arrest=bool(params.get("cardiac_arrest", False)),
            st_deviation=st_elev or bool(params.get("st_depression", False)),
            elevated_troponin=trop_elevated,
        ))
        # TIMI NSTEMI
        if not st_elev:
            results["timi_nstemi"] = vars(compute_timi_nstemi(
                age_ge65=age >= 65,
                cad_risk_factors=sum([smk, dm, htn, bool(params.get("family_hx_cvd")), False]),
                known_cad=bool(params.get("prior_mi", False)),
                aspirin_use_7days=bool(params.get("on_aspirin", False)),
                severe_angina=bool(params.get("chest_pain", False)),
                st_changes=bool(params.get("st_depression", False)),
                positive_troponin=trop_elevated,
            ))
        else:
            results["timi_stemi"] = vars(compute_timi_stemi(
                age=age, systolic_bp=sbp, heart_rate=hr,
                killip_class=int(params.get("killip_class", 1)),
                anterior_stemi=bool(params.get("anterior_stemi", False)),
                diabetes_or_htn=dm or htn,
                weight_le67=float(params.get("weight", 75)) <= 67,
                time_to_treatment_ge4h=bool(params.get("late_presentation", False)),
                prior_angina_mi=bool(params.get("prior_mi", False)) or bool(params.get("prior_angina", False)),
            ))

    # NYHA (si IC)
    ef = params.get("ef")
    nt_bnp = params.get("nt_probnp")
    if params.get("heart_failure") or params.get("dyspnea") or (ef is not None and float(ef) < 50):
        results["nyha"] = vars(compute_nyha(
            nyha_class=int(params.get("nyha_symptoms", 2)),
            ejection_fraction=float(ef) if ef is not None else None,
            nt_probnp=float(nt_bnp) if nt_bnp is not None else None,
        ))

    # QTc
    qt  = float(params.get("qt_interval", 380))
    results["qtc"] = vars(compute_qtc(
        qt_ms=qt, heart_rate=hr, sex=sex,
        method=str(params.get("qtc_method", "bazett")),
    ))

    # ECG Analysis
    results["ecg"] = vars(compute_ecg_analysis(
        heart_rate=hr,
        pr_interval=float(params.get("pr_interval", 160)),
        qrs_duration=float(params.get("qrs_duration", 90)),
        qt_interval=qt,
        axis_degrees=params.get("ecg_axis"),
        p_waves_present=not bool(params.get("af_rhythm", False)),
        regular_rhythm=not bool(params.get("af_rhythm", False)),
        st_elevation_leads=params.get("st_elevation_leads", []) if st_elev else [],
        st_depression_leads=params.get("st_depression_leads", []) if params.get("st_depression") else [],
        t_wave_inversion_leads=params.get("t_wave_inversion_leads", []),
        lbbb=bool(params.get("lbbb", False)),
        rbbb=bool(params.get("rbbb", False)),
        lvh_criteria=bool(params.get("lvh", False)),
        sex=sex,
    ))

    return results
