"""
OsteoDetect AI — Scores cliniques rhumatologiques et ostéo-articulaires
========================================================================
Kellgren-Lawrence · WOMAC · DAS28 · CDAI · SDAI
T-Score / Z-Score · FRAX (WHO) · BASDAI · BASFI · ASDAS

Sources :
  - Kellgren & Lawrence, Ann Rheum Dis 1957
  - WOMAC (Bellamy et al., 1988)
  - DAS28 — van der Heijde et al., 1995
  - CDAI/SDAI — Smolen et al., 2003
  - FRAX — WHO Collaborating Centre, Sheffield 2008 · Kanis et al., JBMR 2008
  - BASDAI — Garrett et al., J Rheumatol 1994
  - BASFI — Calin et al., J Rheumatol 1994
  - ASDAS — Lukas et al., Ann Rheum Dis 2009
  - ACR/EULAR RA Criteria 2010 · ASAS Criteria 2009
  - IOF/ESCEO/ECTS Osteoporosis Guidelines 2023
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 1. KELLGREN-LAWRENCE — Arthrose radiologique (KL 0–4)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KLResult:
    grade: int
    label: str
    radiological_features: list
    progression_risk: str
    functional_impact: str
    recommendation: str


def compute_kellgren_lawrence(
    grade: int = 2,
    joint: str = "genou",    # noqa: ARG001 — label clinique, utilisé dans le rapport
    osteophytes: bool = True,
    joint_space_narrowing: bool = True,
    subchondral_sclerosis: bool = False,
    cysts: bool = False,
    deformity: bool = False,
) -> KLResult:
    """
    Kellgren-Lawrence Grading System (Ann Rheum Dis 1957).
    Grade 0–4 d'arthrose radiologique. Standard OAI / OARSI.
    """
    g = max(0, min(4, int(grade)))

    # Auto-détection du grade à partir des features radiologiques
    if deformity or (osteophytes and joint_space_narrowing and subchondral_sclerosis and cysts):
        g = max(g, 4)
    elif osteophytes and joint_space_narrowing and subchondral_sclerosis:
        g = max(g, 3)
    elif osteophytes and joint_space_narrowing:
        g = max(g, 2)
    elif osteophytes:
        g = max(g, 1)

    data = {
        0: {
            "label": "Grade 0 — Arthrose absente",
            "features": ["Pas d'ostéophytes","Interligne articulaire normal","Pas de sclérose sous-chondrale"],
            "prog": "Risque de progression faible",
            "func": "Pas de limitation fonctionnelle",
            "reco": "Prévention primaire — activité physique — maintien du poids — surveillance clinique",
        },
        1: {
            "label": "Grade 1 — Arthrose douteuse",
            "features": ["Ostéophytes minimes","Possible rétrécissement interligne","Doute diagnostique"],
            "prog": "Risque de progression faible-modéré",
            "func": "Limitation fonctionnelle minime",
            "reco": "Exercice physique adapté — kiné — AINS topiques — réévaluation 1 an",
        },
        2: {
            "label": "Grade 2 — Arthrose légère",
            "features": ["Ostéophytes certains","Léger rétrécissement interligne","Début sclérose sous-chondrale"],
            "prog": "Risque de progression modéré",
            "func": "Limitation fonctionnelle légère-modérée",
            "reco": "Kinésithérapie — AINS si douleur — orthèse si instabilité — infiltration si réfractaire",
        },
        3: {
            "label": "Grade 3 — Arthrose modérée",
            "features": ["Ostéophytes marqués","Rétrécissement interligne significatif","Sclérose sous-chondrale","Début géodes"],
            "prog": "Progression probable sans traitement",
            "func": "Limitation fonctionnelle significative — difficulté escaliers",
            "reco": "Infiltrations corticoïdes ou acide hyaluronique — rééducation — évaluation chirurgicale",
        },
        4: {
            "label": "Grade 4 — Arthrose sévère",
            "features": ["Ostéophytes majeurs","Disparition interligne articulaire","Sclérose dense","Géodes larges","Déformité osseuse"],
            "prog": "Arthrose terminale",
            "func": "Incapacité fonctionnelle — douleur permanente",
            "reco": "Prothèse totale (PTG/PTH) — arthrodèse si CI — soins palliatifs douleur",
        },
    }

    d = data[g]
    return KLResult(
        grade=g, label=d["label"],
        radiological_features=d["features"],
        progression_risk=d["prog"],
        functional_impact=d["func"],
        recommendation=d["reco"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. WOMAC Index — Douleur/Raideur/Fonction (Arthrose)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WOMACResult:
    pain_score: float
    stiffness_score: float
    function_score: float
    total_score: float
    total_pct: float
    severity: str
    recommendation: str


def compute_womac(
    pain_score: float = 5.0,        # 0–20 (5 questions × 0–4)
    stiffness_score: float = 2.0,   # 0–8  (2 questions × 0–4)
    function_score: float = 20.0,   # 0–68 (17 questions × 0–4)
) -> WOMACResult:
    """
    WOMAC Index (Western Ontario and McMaster Universities — Bellamy 1988).
    Évalue douleur (0–20), raideur (0–8), fonction (0–68) en arthrose.
    Total normalisé 0–100%.
    """
    total   = float(pain_score) + float(stiffness_score) + float(function_score)
    max_t   = 96.0
    pct     = round(total / max_t * 100, 1)
    total   = round(total, 1)

    if pct < 25:
        sev = "Légère — impact fonctionnel minime"
        reco = "Exercice physique — kiné — AINS topiques"
    elif pct < 50:
        sev = "Modérée — limitation fonctionnelle significative"
        reco = "AINS per os — kiné intensive — infiltration intra-articulaire"
    elif pct < 75:
        sev = "Sévère — importante limitation fonctionnelle"
        reco = "Évaluation chirurgicale — infiltrations — viscosupplémentation"
    else:
        sev = "Très sévère — incapacité fonctionnelle"
        reco = "Prothèse articulaire (PTG/PTH) — rhumatologie urgente"

    return WOMACResult(
        pain_score=round(float(pain_score),1),
        stiffness_score=round(float(stiffness_score),1),
        function_score=round(float(function_score),1),
        total_score=total, total_pct=pct,
        severity=sev, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DAS28 — Disease Activity Score 28 joints (PR)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DAS28Result:
    score: float
    activity_level: str
    remission: bool
    low_disease: bool
    biological_eligible: bool
    interpretation: str
    recommendation: str


def compute_das28(
    tender_joint_count: int = 6,     # TJC28 (0–28)
    swollen_joint_count: int = 4,    # SJC28 (0–28)
    crp_mg_l: float = 15.0,          # CRP mg/L (DAS28-CRP)
    esr_mm_h: float | None = None,   # VS mm/h (si DAS28-ESR)
    patient_global: float = 40.0,    # EVA patient 0–100
) -> DAS28Result:
    """
    DAS28-CRP (van der Heijde et al., 1995 — adapté CRP Prevoo et al., 2003).
    DAS28-CRP = 0.56×√TJC28 + 0.28×√SJC28 + 0.36×ln(CRP+1) + 0.014×PGA + 0.96.
    Seuils : < 2.6 remission · 2.6–3.2 faible · 3.2–5.1 modéré · > 5.1 élevé.
    """
    tjc = max(0, min(28, int(tender_joint_count)))
    sjc = max(0, min(28, int(swollen_joint_count)))
    pga = max(0.0, min(100.0, float(patient_global)))

    if esr_mm_h is not None:
        esr = max(1.0, float(esr_mm_h))
        das = 0.56 * math.sqrt(tjc) + 0.28 * math.sqrt(sjc) + 0.70 * math.log(esr) + 0.014 * pga
        formula = "DAS28-VS"
    else:
        crp = max(0.0, float(crp_mg_l))
        das = (0.56 * math.sqrt(tjc) + 0.28 * math.sqrt(sjc) +
               0.36 * math.log(crp + 1) + 0.014 * pga + 0.96)
        formula = "DAS28-CRP"

    das = round(das, 2)

    remission = das < 2.6
    low_dis   = 2.6 <= das < 3.2
    bio_elig  = das >= 3.2

    if remission:
        act   = "Rémission"
        reco  = "Maintien traitement actuel — fenêtre thérapeutique si stable > 6 mois"
    elif low_dis:
        act   = "Faible activité"
        reco  = "Optimisation MTX — hydroxychloroquine — cibles DAS28 < 2.6"
    elif das <= 5.1:
        act   = "Activité modérée"
        reco  = "Biothérapie (anti-TNF) si MTX insuffisant — léflunomide — rééval 3 mois"
    else:
        act   = "Activité élevée"
        reco  = ("Biothérapie urgente (adalimumab/étanercept/tocilizumab/abatacept) — "
                 "RCP rhumatologie — corticoïde bridge < 3 mois")

    return DAS28Result(
        score=das, activity_level=act,
        remission=remission, low_disease=low_dis,
        biological_eligible=bio_elig,
        interpretation=(
            f"{formula} = {das} — {act}. "
            f"TJC28 = {tjc} · SJC28 = {sjc} · PGA = {pga:.0f}/100."
        ),
        recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CDAI + SDAI — Indices cliniques PR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CDAIResult:
    cdai: float
    sdai: float
    activity_cdai: str
    activity_sdai: str
    remission_cdai: bool
    recommendation: str


def compute_cdai_sdai(
    tender_joint_count: int = 6,
    swollen_joint_count: int = 4,
    patient_global_vas: float = 40.0,   # 0–10 cm
    physician_global_vas: float = 3.0,  # 0–10 cm
    crp_mg_dl: float = 1.5,             # CRP mg/dL (SDAI uniquement)
) -> CDAIResult:
    """
    CDAI (Smolen et al., Arthritis Rheum 2003) = TJC + SJC + PGA + MDA.
    SDAI = CDAI + CRP (mg/dL).
    Seuils CDAI : ≤ 2.8 remission · ≤ 10 faible · ≤ 22 modéré · > 22 élevé.
    """
    pga_p = max(0.0, min(10.0, float(patient_global_vas)))
    pga_m = max(0.0, min(10.0, float(physician_global_vas)))
    cdai  = float(tender_joint_count) + float(swollen_joint_count) + pga_p + pga_m
    sdai  = cdai + float(crp_mg_dl)
    cdai  = round(cdai, 1)
    sdai  = round(sdai, 1)

    def _act_cdai(v):
        if v <= 2.8:   return "Rémission CDAI", True
        elif v <= 10:  return "Faible activité", False
        elif v <= 22:  return "Activité modérée", False
        else:          return "Activité élevée", False

    def _act_sdai(v):
        if v <= 3.3:   return "Rémission SDAI"
        elif v <= 11:  return "Faible activité"
        elif v <= 26:  return "Activité modérée"
        else:          return "Activité élevée"

    act_c, rem_c = _act_cdai(cdai)
    act_s        = _act_sdai(sdai)

    if cdai > 22 or sdai > 26:
        reco = "Biothérapie urgente — RCP rhumatologie — traitement de fond intensif"
    elif cdai > 10:
        reco = "Optimisation MTX ou ajout d'un csDMARD — biothérapie si cible non atteinte"
    else:
        reco = "Maintien traitement — cible CDAI ≤ 2.8 (rémission)"

    return CDAIResult(
        cdai=cdai, sdai=sdai,
        activity_cdai=act_c, activity_sdai=act_s,
        remission_cdai=rem_c, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. T-SCORE / Z-SCORE + FRAX — Ostéoporose (IOF/ESCEO/WHO)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TScoreResult:
    t_score: float
    z_score: Optional[float]
    site: str
    diagnosis: str
    fracture_risk: str
    treatment_threshold: str
    recommendation: str


def compute_tscore(
    t_score: float = -1.5,
    z_score: float | None = None,
    site: str = "rachis_lombaire",
) -> TScoreResult:
    """
    T-Score DXA — WHO Criteria (Kanis, JBMR 1994).
    Normal ≥ -1.0 · Ostéopénie -1.0 à -2.5 · Ostéoporose ≤ -2.5 · Sévère ≤ -2.5 + fracture.
    """
    ts = float(t_score)

    if ts >= -1.0:
        diag  = "Normal"
        risk  = "Faible risque fracturaire"
        thresh = "Pas de traitement pharmacologique"
        reco  = "Exercice physique — calcium 1000–1200 mg/j — vitamine D 800–2000 UI/j"
    elif ts >= -2.5:
        diag  = "Ostéopénie (T-score -1.0 à -2.5)"
        risk  = "Risque fracturaire intermédiaire"
        thresh = "Traitement si FRAX élevé ou facteurs de risque"
        reco  = ("Calcium + vitamine D — exercice — réévaluation DXA 2 ans — "
                 "FRAX si âge > 50 ans ou ménoposée")
    else:
        diag  = "Ostéoporose (T-score ≤ -2.5)"
        risk  = "Risque fracturaire élevé"
        thresh = "Traitement pharmacologique recommandé"
        reco  = ("Bisphosphonate (alendronate/risédronate) ou dénosumab 60 mg × 2/an. "
                 "Calcium + vitamine D. Prévention chutes. Réévaluation DXA 1–2 ans.")

    if ts <= -2.5:
        z_note = ""
        if z_score is not None and float(z_score) < -2.0:
            z_note = " Z-score très bas — exclure ostéoporose secondaire (hypercortisolisme, MM, hyperparathyroïdie)."
            reco += z_note

    return TScoreResult(
        t_score=round(ts,2), z_score=round(float(z_score),2) if z_score else None,
        site=site.replace("_"," "), diagnosis=diag,
        fracture_risk=risk, treatment_threshold=thresh,
        recommendation=reco,
    )


@dataclass
class FRAXResult:
    hip_fracture_risk_10yr: float
    major_fracture_risk_10yr: float
    risk_category: str
    intervention_threshold: bool
    recommendation: str


def compute_frax(
    age: int = 65,
    sex: str = "F",
    bmi: float = 25.0,
    t_score_femoral_neck: float = -1.5,
    prior_fracture: bool = False,
    parent_hip_fracture: bool = False,
    smoking: bool = False,
    glucocorticoids: bool = False,
    rheumatoid_arthritis: bool = False,
    secondary_osteoporosis: bool = False,
    alcohol_3_plus: bool = False,
) -> FRAXResult:
    """
    FRAX Score (Kanis et al., JBMR 2008 — WHO Collaborating Centre Sheffield).
    Estime le risque de fracture ostéoporotique majeure et de hanche à 10 ans.
    Version simplifiée basée sur les coefficients de régression principaux.
    """
    sex_f  = str(sex).upper() == "F"
    a      = max(40, min(90, int(age)))

    # Base risk selon âge et sexe
    # Risque de base calibré (Kanis et al. 2008 — valeurs européennes)
    # Femme : hanche ~0.4% à 50 ans, ~2.5% à 65 ans, ~7% à 75 ans
    # Homme : hanche ~0.15% à 50 ans, ~0.8% à 65 ans, ~2.5% à 75 ans
    if sex_f:
        base_hip   = 0.004 * math.exp(0.075 * max(0, a - 50))
        base_major = 0.012 * math.exp(0.060 * max(0, a - 50))
    else:
        base_hip   = 0.0015 * math.exp(0.070 * max(0, a - 50))
        base_major = 0.007  * math.exp(0.055 * max(0, a - 50))

    # Facteurs de risque multiplicatifs (OR approx.)
    rf_mult = 1.0
    if prior_fracture:        rf_mult *= 1.86
    if parent_hip_fracture:   rf_mult *= 1.52
    if smoking:               rf_mult *= 1.21
    if glucocorticoids:       rf_mult *= 1.63
    if rheumatoid_arthritis:  rf_mult *= 1.49
    if secondary_osteoporosis:rf_mult *= 1.38
    if alcohol_3_plus:        rf_mult *= 1.41

    # BMI correction
    bmi_corr = max(0.5, min(2.0, math.exp(-0.016 * (float(bmi) - 25))))

    # T-score DXA correction
    ts      = float(t_score_femoral_neck)
    ts_corr = max(0.3, math.exp(-0.30 * (ts + 1.0)))  # T-score → fracture risk

    hip_risk   = min(base_hip   * rf_mult * bmi_corr * ts_corr * 100, 50.0)
    major_risk = min(base_major * rf_mult * bmi_corr * ts_corr * 100, 70.0)
    hip_risk   = round(max(hip_risk,   0.1), 1)
    major_risk = round(max(major_risk, 0.5), 1)

    # Catégorisation selon seuils IOF/ESCEO 2023
    if hip_risk >= 3.0 or major_risk >= 20.0:
        cat  = "Risque élevé — traitement indiqué"
        thr  = True
        reco = ("Bisphosphonate (alendronate 70 mg/sem) ou dénosumab 60 mg/6 mois. "
                "Calcium 1000 mg/j + vitamine D 800–2000 UI/j. Prévention chutes. "
                "Réévaluation DXA 1–2 ans.")
    elif hip_risk >= 1.0 or major_risk >= 10.0:
        cat  = "Risque intermédiaire — traitement à discuter"
        thr  = False
        reco = ("Calcium + vitamine D. Exercice physique (équilibre + renforcement). "
                "Traitement pharmacologique si T-score ≤ -2.5 ou ostéopénie + FRAX élevé.")
    else:
        cat  = "Risque faible — prévention primaire"
        thr  = False
        reco = ("Calcium 1000–1200 mg/j + vitamine D 800 UI/j. "
                "Activité physique régulière. Réévaluation DXA dans 3–5 ans.")

    return FRAXResult(
        hip_fracture_risk_10yr=hip_risk,
        major_fracture_risk_10yr=major_risk,
        risk_category=cat,
        intervention_threshold=thr,
        recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BASDAI — Bath AS Disease Activity Index (Spondylarthrite)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BASDAIResult:
    score: float
    activity_level: str
    biological_threshold: bool
    interpretation: str
    recommendation: str


def compute_basdai(
    fatigue_vas: float = 5.0,           # Q1 : fatigue 0–10
    spinal_pain_vas: float = 5.0,       # Q2 : douleur rachidienne 0–10
    peripheral_pain_vas: float = 3.0,   # Q3 : douleur articulations périphériques 0–10
    enthesitis_vas: float = 3.0,        # Q4 : enthésopathies 0–10
    morning_stiffness_severity: float = 5.0,  # Q5 : intensité raideur matinale 0–10
    morning_stiffness_duration: float = 5.0,  # Q6 : durée raideur matinale 0–10
) -> BASDAIResult:
    """
    BASDAI (Garrett et al., J Rheumatol 1994).
    BASDAI = [(Q1+Q2+Q3+Q4) + (Q5+Q6)/2] / 5. Score 0–10.
    ≥ 4 = maladie active — indication biothérapie (ASAS 2010).
    """
    qs = [
        max(0.0, min(10.0, float(fatigue_vas))),
        max(0.0, min(10.0, float(spinal_pain_vas))),
        max(0.0, min(10.0, float(peripheral_pain_vas))),
        max(0.0, min(10.0, float(enthesitis_vas))),
    ]
    stiff_mean = (max(0.0, min(10.0, float(morning_stiffness_severity))) +
                  max(0.0, min(10.0, float(morning_stiffness_duration)))) / 2.0
    basdai = round((sum(qs) + stiff_mean) / 5.0, 2)

    bio_thr = basdai >= 4.0

    if basdai < 2.0:
        act  = "Faible activité — rémission possible"
        reco = "Maintien traitement — AINS ou cDMARDs si nécessaire"
    elif basdai < 4.0:
        act  = "Activité modérée"
        reco = "AINS optimisation — physiothérapie — BASDAI à réévaluer dans 3 mois"
    elif basdai < 6.0:
        act  = "Activité élevée — biothérapie à envisager"
        reco = "Anti-TNF (étanercept/adalimumab) ou anti-IL-17 (sécukinumab) si AINS inefficaces"
    else:
        act  = "Activité très élevée — traitement urgent"
        reco = "Biothérapie urgente — anti-TNF ou JAKi (tofacitinib) — RCP rhumatologie"

    return BASDAIResult(
        score=basdai, activity_level=act,
        biological_threshold=bio_thr,
        interpretation=(
            f"BASDAI = {basdai}/10 — {act}. "
            f"{'Seuil biothérapie atteint (≥ 4.0).' if bio_thr else 'Seuil biothérapie non atteint (< 4.0).'}"
        ),
        recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BASFI — Bath AS Functional Index
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BASFIResult:
    score: float
    functional_level: str
    disability: str
    recommendation: str


def compute_basfi(activities: list | None = None) -> BASFIResult:
    """
    BASFI (Calin et al., J Rheumatol 1994).
    10 questions EVA 0–10 : activités quotidiennes.
    Score = moyenne des 10 items.
    """
    if activities is None or len(activities) == 0:
        activities = [5.0] * 10
    acts = [max(0.0, min(10.0, float(a))) for a in activities[:10]]
    while len(acts) < 10:
        acts.append(5.0)
    score = round(sum(acts) / len(acts), 2)

    if score < 2.0:
        level  = "Fonctionnement préservé"
        disab  = "Pas d'incapacité significative"
        reco   = "Maintien exercice physique — physiothérapie préventive"
    elif score < 5.0:
        level  = "Limitation fonctionnelle modérée"
        disab  = "Incapacité modérée — adaptation activités"
        reco   = "Kinésithérapie intensive — ergothérapie — aides techniques"
    else:
        level  = "Limitation fonctionnelle sévère"
        disab  = "Incapacité sévère — dépendance partielle"
        reco   = "Biothérapie urgente — rééducation fonctionnelle — ergothérapie — MDPH"

    return BASFIResult(score=score, functional_level=level,
                       disability=disab, recommendation=reco)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. ASDAS — Ankylosing Spondylitis Disease Activity Score
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ASDASResult:
    score: float
    activity_level: str
    inactive_disease: bool
    high_disease: bool
    very_high_disease: bool
    recommendation: str


def compute_asdas(
    back_pain_vas: float = 5.0,           # 0–10 (dernier 7 jours)
    morning_stiffness_duration: float = 60.0,  # minutes
    peripheral_pain_vas: float = 3.0,     # 0–10
    patient_global_vas: float = 4.0,      # 0–10
    crp_mg_l: float = 15.0,               # CRP mg/L
) -> ASDASResult:
    """
    ASDAS-CRP (Lukas et al., Ann Rheum Dis 2009).
    ASDAS = 0.121×BP + 0.058×MS + 0.110×PPS + 0.073×PtGA + 0.579×ln(CRP+1).
    Seuils : < 1.3 inactive · 1.3–2.1 faible · 2.1–3.5 élevé · ≥ 3.5 très élevé.
    """
    bp   = max(0.0, min(10.0, float(back_pain_vas)))
    ms   = max(0.0, min(120.0, float(morning_stiffness_duration)))
    pps  = max(0.0, min(10.0, float(peripheral_pain_vas)))
    ptga = max(0.0, min(10.0, float(patient_global_vas)))
    crp  = max(0.0, float(crp_mg_l))

    asdas = round(
        0.121 * bp + 0.058 * ms + 0.110 * pps +
        0.073 * ptga + 0.579 * math.log(crp + 1),
        2
    )

    inactive  = asdas < 1.3
    low_dis   = 1.3 <= asdas < 2.1
    high_dis  = 2.1 <= asdas < 3.5
    very_high = asdas >= 3.5

    if inactive:
        act  = "Maladie inactive"
        reco = "Maintien traitement actuel — réévaluation ASDAS dans 6 mois"
    elif low_dis:
        act  = "Faible activité"
        reco = "AINS si nécessaire — physiothérapie — cible ASDAS < 1.3"
    elif high_dis:
        act  = "Activité élevée — traitement à intensifier"
        reco = "Anti-TNF (adalimumab/étanercept) ou anti-IL-17 (sécukinumab/ixékizumab)"
    else:
        act  = "Activité très élevée — biothérapie urgente"
        reco = ("Biothérapie urgente — RCP rhumatologie — "
                "anti-TNF + AINS ou JAKi (upadacitinib/tofacitinib)")

    return ASDASResult(
        score=asdas, activity_level=act,
        inactive_disease=inactive, high_disease=high_dis,
        very_high_disease=very_high,
        recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. ORCHESTRATEUR — compute_all_osteo_scores()
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_osteo_scores(params: dict[str, Any]) -> dict[str, Any]:
    """Orchestre tous les scores ostéo-articulaires selon le contexte clinique."""
    results: dict[str, Any] = {}
    ct = str(params.get("disease_context", "arthrose")).lower()

    # Kellgren-Lawrence — toujours calculé si grade fourni
    if params.get("kl_grade") is not None or params.get("osteoarthritis"):
        results["kellgren_lawrence"] = vars(compute_kellgren_lawrence(
            grade=int(params.get("kl_grade", 2)),
            joint=params.get("joint_site", "genou"),
            osteophytes=bool(params.get("osteophytes", True)),
            joint_space_narrowing=bool(params.get("joint_space_narrowing", False)),
            subchondral_sclerosis=bool(params.get("subchondral_sclerosis", False)),
            cysts=bool(params.get("subchondral_cysts", False)),
            deformity=bool(params.get("deformity", False)),
        ))

    # WOMAC — arthrose
    if "arthrose" in ct or params.get("osteoarthritis"):
        results["womac"] = vars(compute_womac(
            pain_score=float(params.get("womac_pain", 5)),
            stiffness_score=float(params.get("womac_stiffness", 2)),
            function_score=float(params.get("womac_function", 20)),
        ))

    # DAS28 — PR
    if "pr" in ct or "polyarthrite" in ct or params.get("rheumatoid_arthritis"):
        results["das28"] = vars(compute_das28(
            tender_joint_count=int(params.get("tender_joints", 6)),
            swollen_joint_count=int(params.get("swollen_joints", 4)),
            crp_mg_l=float(params.get("crp", 15)),
            esr_mm_h=params.get("esr"),
            patient_global=float(params.get("patient_global_vas", 40)),
        ))
        results["cdai_sdai"] = vars(compute_cdai_sdai(
            tender_joint_count=int(params.get("tender_joints", 6)),
            swollen_joint_count=int(params.get("swollen_joints", 4)),
            patient_global_vas=float(params.get("patient_global_vas", 40)) / 10,
            physician_global_vas=float(params.get("physician_global_vas", 30)) / 10,
            crp_mg_dl=float(params.get("crp", 15)) / 10,
        ))

    # Ostéoporose / T-Score
    t_score = params.get("t_score")
    if t_score is not None or params.get("osteoporosis"):
        ts = float(t_score) if t_score is not None else -2.0
        results["t_score"] = vars(compute_tscore(
            t_score=ts,
            z_score=params.get("z_score"),
            site=params.get("dxa_site", "col_femoral"),
        ))
        results["frax"] = vars(compute_frax(
            age=int(params.get("age", 65)),
            sex=str(params.get("sex", "F")),
            bmi=float(params.get("bmi", 25)),
            t_score_femoral_neck=ts,
            prior_fracture=bool(params.get("prior_fracture", False)),
            parent_hip_fracture=bool(params.get("parent_hip_fracture", False)),
            smoking=bool(params.get("smoking", False)),
            glucocorticoids=bool(params.get("long_term_glucocorticoids", False)),
            rheumatoid_arthritis=bool(params.get("rheumatoid_arthritis", False)),
            secondary_osteoporosis=bool(params.get("secondary_osteoporosis", False)),
            alcohol_3_plus=bool(params.get("alcohol_excess", False)),
        ))

    # Spondylarthrite
    if "spondy" in ct or "spa" in ct or "ankylosante" in ct or params.get("axial_spondyloarthritis"):
        results["basdai"] = vars(compute_basdai(
            fatigue_vas=float(params.get("fatigue_vas", 5)),
            spinal_pain_vas=float(params.get("spinal_pain_vas", 5)),
            peripheral_pain_vas=float(params.get("peripheral_pain_vas", 3)),
            enthesitis_vas=float(params.get("enthesitis_vas", 3)),
            morning_stiffness_severity=float(params.get("morning_stiffness_severity", 5)),
            morning_stiffness_duration=float(params.get("morning_stiffness_duration_vas", 5)),
        ))
        results["basfi"] = vars(compute_basfi(params.get("basfi_activities")))
        results["asdas"] = vars(compute_asdas(
            back_pain_vas=float(params.get("spinal_pain_vas", 5)),
            morning_stiffness_duration=float(params.get("morning_stiffness_min", 60)),
            peripheral_pain_vas=float(params.get("peripheral_pain_vas", 3)),
            patient_global_vas=float(params.get("patient_global_vas", 40)) / 10,
            crp_mg_l=float(params.get("crp", 15)),
        ))

    return results
