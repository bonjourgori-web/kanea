"""
PulmoScan AI — Scores cliniques & algorithmes médicaux
=======================================================
Implémentation de référence de tous les scores pneumologiques :
CURB-65, PSI, Lung-RADS, TNM NSCLC, GOLD BPCO, BODE Index,
COVID-19 CT Severity Score, CT Severity Index, Fleischner Guidelines.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# CURB-65 — Pneumonia Severity Index (BTS Guidelines)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CURB65Result:
    score: int
    components: dict[str, bool]
    severity: str
    mortality_30d: str
    recommendation: str
    hospitalization: str

def compute_curb65(
    confusion: bool = False,
    urea_mmol: float = 0.0,
    resp_rate: int = 0,
    systolic_bp: int = 120,
    diastolic_bp: int = 80,
    age: int = 50,
) -> CURB65Result:
    """
    CURB-65 Score (British Thoracic Society).
    C = Confusion, U = Urea > 7 mmol/L, R = RR >= 30/min,
    B = BP systolique < 90 ou diastolique <= 60, 65 = âge >= 65.
    """
    c = confusion
    u = urea_mmol > 7.0
    r = resp_rate >= 30
    b = systolic_bp < 90 or diastolic_bp <= 60
    age_flag = age >= 65

    score = sum([c, u, r, b, age_flag])

    if score <= 1:
        severity = "Faible"
        mortality = "< 3%"
        recommendation = "Traitement ambulatoire possible"
        hosp = "Ambulatoire"
    elif score == 2:
        severity = "Modérée"
        mortality = "3–15%"
        recommendation = "Hospitalisation courte durée recommandée"
        hosp = "Hospitalisation"
    else:
        severity = "Sévère"
        mortality = "> 15%"
        recommendation = "Hospitalisation urgente — évaluer USI si score ≥ 4"
        hosp = "USI / Réanimation" if score >= 4 else "Hospitalisation urgente"

    return CURB65Result(
        score=score,
        components={"Confusion": c, "Urée > 7 mmol/L": u, "FR ≥ 30/min": r,
                    "PAS < 90 ou PAD ≤ 60": b, "Âge ≥ 65 ans": age_flag},
        severity=severity,
        mortality_30d=mortality,
        recommendation=recommendation,
        hospitalization=hosp,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PSI — Pneumonia Severity Index (Fine Score)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PSIResult:
    score: int
    risk_class: str
    mortality_30d: str
    recommendation: str

def compute_psi(
    age: int,
    sex: str = "M",
    nursing_home: bool = False,
    neoplasm: bool = False,
    liver_disease: bool = False,
    heart_failure: bool = False,
    cerebrovascular: bool = False,
    renal_disease: bool = False,
    altered_mental: bool = False,
    resp_rate: int = 16,
    systolic_bp: int = 120,
    temp_celsius: float = 37.0,
    pulse: int = 80,
    pH: float = 7.40,
    bun_mmol: float = 5.0,
    sodium_meq: float = 140.0,
    glucose_mmol: float = 5.5,
    hematocrit: float = 40.0,
    pao2_mmhg: float = 90.0,
    pleural_effusion: bool = False,
) -> PSIResult:
    """PSI / Fine Score — Classe I à V."""
    score = age if sex == "M" else age - 10
    if nursing_home:     score += 10
    if neoplasm:         score += 30
    if liver_disease:    score += 20
    if heart_failure:    score += 10
    if cerebrovascular:  score += 10
    if renal_disease:    score += 10
    if altered_mental:   score += 20
    if resp_rate >= 30:  score += 20
    if systolic_bp < 90: score += 20
    if temp_celsius < 35 or temp_celsius >= 40: score += 15
    if pulse >= 125:     score += 10
    if pH < 7.35:        score += 30
    if bun_mmol >= 10.7: score += 20
    if sodium_meq < 130: score += 20
    if glucose_mmol >= 13.9: score += 10
    if hematocrit < 30:  score += 10
    if pao2_mmhg < 60:   score += 10
    if pleural_effusion: score += 10

    if score <= 50:
        risk_class, mortality, rec = "I", "0.1%", "Ambulatoire"
    elif score <= 70:
        risk_class, mortality, rec = "II", "0.6%", "Ambulatoire"
    elif score <= 90:
        risk_class, mortality, rec = "III", "2.8%", "Observation courte durée"
    elif score <= 130:
        risk_class, mortality, rec = "IV", "8.2%", "Hospitalisation"
    else:
        risk_class, mortality, rec = "V", "29.2%", "Hospitalisation — USI à envisager"

    return PSIResult(score=score, risk_class=risk_class,
                     mortality_30d=mortality, recommendation=rec)


# ═══════════════════════════════════════════════════════════════════════════════
# Lung-RADS — ACR Lung Imaging Reporting and Data System (v2022)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LungRADSResult:
    category: str
    descriptor: str
    malignancy_probability: str
    management: str
    follow_up: str

def compute_lung_rads(
    nodule_present: bool = False,
    nodule_size_mm: float = 0.0,
    nodule_type: str = "solid",        # solid | part-solid | ground-glass
    nodule_growth: bool = False,
    new_nodule: bool = False,
    prior_category: str = "0",
    suspicious_features: bool = False,
) -> LungRADSResult:
    """
    Lung-RADS v2022 (ACR) — Catégories 0 à 4X.
    nodule_type: 'solid', 'part-solid', 'ground-glass'
    """
    if not nodule_present:
        return LungRADSResult(
            category="1",
            descriptor="Négatif",
            malignancy_probability="< 1%",
            management="Dépistage annuel de routine",
            follow_up="CT annuel si eligibilité au dépistage",
        )

    if nodule_size_mm < 6.0 and nodule_type == "solid" and not new_nodule:
        category = "2"
        desc = "Bénin — probabilité de malignité très faible"
        prob = "< 1%"
        mgmt = "Dépistage annuel de routine"
        fu = "CT annuel"

    elif nodule_type == "solid" and 6.0 <= nodule_size_mm < 8.0:
        category = "3"
        desc = "Probablement bénin — suivi à court terme"
        prob = "1–2%"
        mgmt = "CT thoracique 6 mois"
        fu = "CT à 6 mois puis annuel si stable"

    elif nodule_type == "solid" and nodule_size_mm >= 8.0:
        category = "4A" if nodule_size_mm < 15.0 else "4B"
        desc = "Suspect — évaluation complémentaire"
        prob = "5–15%" if nodule_size_mm < 15.0 else "> 15%"
        mgmt = "PET-scan + évaluation multidisciplinaire"
        fu = "TEP-TDM / biopsie selon clinique"

    elif nodule_type == "part-solid":
        if nodule_size_mm < 6.0:
            category, desc, prob, mgmt = "2", "Probablement bénin", "< 1%", "CT annuel"
            fu = "CT annuel"
        elif nodule_size_mm < 8.0:
            category, desc, prob, mgmt = "3", "Probablement bénin", "< 1%", "CT 6 mois"
            fu = "CT 6 mois"
        else:
            category, desc, prob, mgmt = "4A", "Suspect", "5–15%", "CT 3 mois"
            fu = "CT 3 mois + avis oncologie"

    elif nodule_type == "ground-glass":
        if nodule_size_mm < 30.0:
            category, desc, prob, mgmt = "2", "Probablement bénin", "< 1%", "CT annuel"
            fu = "CT 12 mois"
        else:
            category, desc, prob, mgmt = "3", "Probablement bénin", "1–2%", "CT 6 mois"
            fu = "CT 6 mois"
    else:
        category, desc, prob, mgmt = "2", "Indéterminé", "< 1%", "CT annuel"
        fu = "CT annuel"

    if suspicious_features or nodule_growth:
        category = category.replace("3", "4A").replace("2", "3")
        prob = "> 15%"
        mgmt = "TEP-TDM + biopsie urgente"
        fu = "Bilan oncologique urgent"

    return LungRADSResult(
        category=category,
        descriptor=desc,
        malignancy_probability=prob,
        management=mgmt,
        follow_up=fu,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TNM — Staging Cancer Pulmonaire (IASLC 9e édition 2024)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TNMResult:
    T: str
    N: str
    M: str
    stage: str
    stage_description: str
    median_survival: str
    five_year_survival: str
    treatment_recommendation: str

def compute_tnm_lung(
    tumor_size_cm: float = 0.0,
    invades_visceral_pleura: bool = False,
    main_bronchus_involvement: bool = False,
    atelectasis: bool = False,
    chest_wall_invasion: bool = False,
    mediastinum_invasion: bool = False,
    diaphragm_invasion: bool = False,
    heart_invasion: bool = False,
    trachea_invasion: bool = False,
    separate_tumor_nodule_same_lobe: bool = False,
    separate_tumor_nodule_diff_lobe: bool = False,
    lymph_nodes_ipsilateral_hilar: bool = False,
    lymph_nodes_mediastinal: bool = False,
    lymph_nodes_contralateral: bool = False,
    distant_metastasis: bool = False,
    metastasis_single_site: bool = False,
) -> TNMResult:
    """TNM NSCLC — IASLC 9e édition."""
    # Déterminer T
    if tumor_size_cm == 0:
        T = "T0"
    elif mediastinum_invasion or heart_invasion or trachea_invasion:
        T = "T4"
    elif chest_wall_invasion or diaphragm_invasion or separate_tumor_nodule_diff_lobe:
        T = "T3" if not (mediastinum_invasion or heart_invasion) else "T4"
    elif separate_tumor_nodule_same_lobe:
        T = "T3"
    elif tumor_size_cm <= 1.0:
        T = "T1a"
    elif tumor_size_cm <= 2.0:
        T = "T1b"
    elif tumor_size_cm <= 3.0:
        T = "T1c"
    elif tumor_size_cm <= 4.0:
        T = "T2a"
    elif tumor_size_cm <= 5.0:
        T = "T2b"
    elif tumor_size_cm <= 7.0:
        T = "T3"
    else:
        T = "T4"

    if main_bronchus_involvement or invades_visceral_pleura or atelectasis:
        if T in ("T1a", "T1b", "T1c"):
            T = "T2a"

    # Déterminer N
    if lymph_nodes_contralateral:
        N = "N3"
    elif lymph_nodes_mediastinal:
        N = "N2"
    elif lymph_nodes_ipsilateral_hilar:
        N = "N1"
    else:
        N = "N0"

    # Déterminer M
    if distant_metastasis:
        M = "M1b" if not metastasis_single_site else "M1a"
        if metastasis_single_site:
            M = "M1b"
        else:
            M = "M1c"
    else:
        M = "M0"

    # Déterminer le stade
    stage_map = {
        ("T1a","N0","M0"): ("IA1","5 ans: 92%","Lobectomie + curage ganglionnaire"),
        ("T1b","N0","M0"): ("IA2","5 ans: 83%","Lobectomie ± adjuvant"),
        ("T1c","N0","M0"): ("IA3","5 ans: 77%","Lobectomie ± adjuvant"),
        ("T2a","N0","M0"): ("IB","5 ans: 68%","Lobectomie + bilan adjuvant"),
        ("T2b","N0","M0"): ("IIA","5 ans: 60%","Lobectomie + chimio adjuvante"),
        ("T1a","N1","M0"): ("IIB","5 ans: 53%","Chirurgie + chimio adjuvante"),
        ("T2a","N1","M0"): ("IIB","5 ans: 53%","Chirurgie + chimio adjuvante"),
        ("T2b","N1","M0"): ("IIB","5 ans: 53%","Chirurgie + chimio adjuvante"),
        ("T3","N0","M0"): ("IIB","5 ans: 46%","Chirurgie + chimio adjuvante"),
    }
    key = (T, N, M)
    if key in stage_map:
        stage, survival, treatment = stage_map[key]
    elif M != "M0":
        stage = "IV"
        survival = "5 ans: 6%" if M == "M1c" else "5 ans: 10%"
        treatment = "Chimiothérapie ± immunothérapie ± thérapie ciblée"
    elif N == "N2":
        stage = "IIIA" if T not in ("T3","T4") else "IIIB"
        survival = "5 ans: 13–26%"
        treatment = "Chimio-radiothérapie concomitante ± chirurgie"
    elif N == "N3":
        stage = "IIIB" if T not in ("T3","T4") else "IIIC"
        survival = "5 ans: 7–13%"
        treatment = "Chimio-radiothérapie concomitante + immunothérapie"
    else:
        stage = "IIIA"
        survival = "5 ans: 22%"
        treatment = "RCP multidisciplinaire — chirurgie + chimio adjuvante"

    desc = {
        "IA1": "Cancer localisé < 1 cm — résection limitée possible",
        "IA2": "Cancer localisé 1–2 cm — lobectomie standard",
        "IA3": "Cancer localisé 2–3 cm — lobectomie",
        "IB":  "Cancer localisé 3–4 cm — lobectomie",
        "IIA": "Cancer localisé 4–5 cm — lobectomie + chimio adjuvante",
        "IIB": "Extension régionale — chirurgie + traitement adjuvant",
        "IIIA":"Extension ganglionnaire médiastinale — traitement multimodal",
        "IIIB":"Extension ganglionnaire controlatérale — radio-chimio",
        "IIIC":"Très avancé localement — traitement palliatif à discuter",
        "IV":  "Métastatique — traitement systémique",
    }.get(stage, "Évaluation multidisciplinaire requise")

    return TNMResult(
        T=T, N=N, M=M,
        stage=stage,
        stage_description=desc,
        median_survival={"IA1":"NR","IA2":"NR","IA3":"NR","IB":"NR","IIA":"NR",
                         "IIB":"60 mois","IIIA":"28 mois","IIIB":"18 mois",
                         "IIIC":"12 mois","IV":"10–14 mois"}.get(stage,"—"),
        five_year_survival=survival,
        treatment_recommendation=treatment,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GOLD — BPCO Classification (GOLD 2024)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GOLDResult:
    grade: str
    group: str
    fev1_predicted: str
    exacerbation_risk: str
    symptoms: str
    treatment: str

def compute_gold(
    fev1_percent_predicted: float = 80.0,
    fev1_fvc_ratio: float = 0.75,
    cat_score: int = 10,
    mmrc_score: int = 1,
    exacerbations_last_year: int = 0,
    hospitalizations_last_year: int = 0,
) -> GOLDResult:
    """GOLD 2024 — Classification BPCO."""
    if fev1_fvc_ratio >= 0.70:
        return GOLDResult(
            grade="Non-BPCO",
            group="—",
            fev1_predicted=f"{fev1_percent_predicted:.0f}%",
            exacerbation_risk="Non applicable",
            symptoms="VEMS/CVF ≥ 0.70 — pas d'obstruction bronchique",
            treatment="Spirométrie à répéter si symptômes"
        )

    if fev1_percent_predicted >= 80:
        grade = "GOLD 1 — Légère"
    elif fev1_percent_predicted >= 50:
        grade = "GOLD 2 — Modérée"
    elif fev1_percent_predicted >= 30:
        grade = "GOLD 3 — Sévère"
    else:
        grade = "GOLD 4 — Très sévère"

    high_symptoms = cat_score >= 10 or mmrc_score >= 2
    high_exacerbation = exacerbations_last_year >= 2 or hospitalizations_last_year >= 1
    moderate_exacerbation = exacerbations_last_year == 1 and hospitalizations_last_year == 0

    if not high_symptoms and not high_exacerbation and not moderate_exacerbation:
        group = "A"
        treatment = "Bronchodilatateur à courte durée d'action (SABA/SAMA)"
    elif high_symptoms and not high_exacerbation:
        group = "B"
        treatment = "LABA ou LAMA (bronchodilatateur à longue durée d'action)"
    elif high_exacerbation or moderate_exacerbation:
        if high_symptoms:
            group = "E"
            treatment = "LABA + LAMA + ICS (triple thérapie si éosinophiles ≥ 300)"
        else:
            group = "E"
            treatment = "LABA + LAMA"
    else:
        group = "B"
        treatment = "LAMA ou LABA + LAMA"

    return GOLDResult(
        grade=grade,
        group=group,
        fev1_predicted=f"{fev1_percent_predicted:.0f}%",
        exacerbation_risk="Élevé" if high_exacerbation else "Faible",
        symptoms="Très symptomatique" if high_symptoms else "Peu symptomatique",
        treatment=treatment,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BODE Index — BPCO Pronostic (Celli 2004)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BODEResult:
    score: int
    quartile: str
    four_year_survival: str
    components: dict[str, int]

def compute_bode(
    bmi: float = 25.0,
    fev1_percent_predicted: float = 60.0,
    mmrc_dyspnea: int = 2,
    six_min_walk_meters: float = 350.0,
) -> BODEResult:
    """BODE Index — Body mass index, airflow Obstruction, Dyspnea, Exercise capacity."""
    # FEV1 score (0-3)
    if fev1_percent_predicted >= 65: fev1_score = 0
    elif fev1_percent_predicted >= 50: fev1_score = 1
    elif fev1_percent_predicted >= 36: fev1_score = 2
    else: fev1_score = 3

    # BMI score (0-1)
    bmi_score = 0 if bmi > 21 else 1

    # mMRC score (0-3)
    mmrc_score = min(mmrc_dyspnea, 3)

    # 6MWT score (0-3)
    if six_min_walk_meters >= 350: walk_score = 0
    elif six_min_walk_meters >= 250: walk_score = 1
    elif six_min_walk_meters >= 150: walk_score = 2
    else: walk_score = 3

    total = bmi_score + fev1_score + mmrc_score + walk_score

    if total <= 2:
        quartile = "Q1 (0–2)"
        survival = "80%"
    elif total <= 4:
        quartile = "Q2 (3–4)"
        survival = "67%"
    elif total <= 6:
        quartile = "Q3 (5–6)"
        survival = "57%"
    else:
        quartile = "Q4 (7–10)"
        survival = "18%"

    return BODEResult(
        score=total,
        quartile=quartile,
        four_year_survival=survival,
        components={"BMI": bmi_score, "FEV1": fev1_score, "mMRC": mmrc_score, "6MWT": walk_score},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# COVID-19 CT Severity Score (Pan et al., European Radiology 2020)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class COVIDCTSeverityResult:
    total_score: int
    severity: str
    lung_involvement_percent: float
    pattern: str
    recommendation: str
    icu_probability: str

def compute_covid_ct_severity(
    right_upper_lobe: int = 0,
    right_middle_lobe: int = 0,
    right_lower_lobe: int = 0,
    left_upper_lobe: int = 0,
    left_lower_lobe: int = 0,
    dominant_pattern: str = "normal",  # normal|ground_glass|consolidation|mixed
) -> COVIDCTSeverityResult:
    """
    COVID-19 CT Severity Score — 5 lobes × 0–5 = max 25 points.
    0 = aucune atteinte, 1 = <5%, 2 = 5–25%, 3 = 26–50%, 4 = 51–75%, 5 = >75%.
    """
    total = right_upper_lobe + right_middle_lobe + right_lower_lobe + left_upper_lobe + left_lower_lobe
    involvement = total / 25 * 100

    if total == 0:
        severity = "Normal"
        recommendation = "Pas d'image évocatrice de COVID-19"
        icu = "< 1%"
    elif total <= 7:
        severity = "Léger"
        recommendation = "Isolement — suivi ambulatoire + oxymétrie"
        icu = "< 5%"
    elif total <= 17:
        severity = "Modéré"
        recommendation = "Hospitalisation — O2 + anticoagulation + corticoïdes si SpO2 < 94%"
        icu = "10–25%"
    else:
        severity = "Sévère / Critique"
        recommendation = "USI — ventilation mécanique à envisager"
        icu = "> 50%"

    pattern_desc = {
        "normal": "Aspect normal",
        "ground_glass": "Verre dépoli bilatéral — typique COVID-19",
        "consolidation": "Consolidations bilatérales — forme sévère",
        "mixed": "Verre dépoli + consolidations — forme modérée à sévère",
    }.get(dominant_pattern, "Indéterminé")

    return COVIDCTSeverityResult(
        total_score=total,
        severity=severity,
        lung_involvement_percent=round(involvement, 1),
        pattern=pattern_desc,
        recommendation=recommendation,
        icu_probability=icu,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Fleischner Society Guidelines — Nodules pulmonaires incidentels
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FleischnerResult:
    guideline_year: str
    nodule_type: str
    size_category: str
    low_risk_management: str
    high_risk_management: str
    follow_up_low: str
    follow_up_high: str

def compute_fleischner(
    nodule_size_mm: float = 0.0,
    nodule_type: str = "solid",    # solid | subsolid | ground-glass
    multiple_nodules: bool = False,
    high_risk: bool = False,       # tabagisme, antécédent néoplasique
) -> FleischnerResult:
    """Fleischner Society Guidelines 2017."""
    if nodule_type == "solid":
        if nodule_size_mm < 6.0:
            lr_mgmt = "Pas de suivi routinier"
            hr_mgmt = "CT optionnel à 12 mois"
            lr_fu = "Aucun si < 6 mm"
            hr_fu = "CT 12 mois (optionnel)"
        elif nodule_size_mm < 8.0:
            lr_mgmt = "CT à 6–12 mois puis CT 18–24 mois si stable"
            hr_mgmt = "CT à 6–12 mois puis CT 18–24 mois"
            lr_fu = "CT 6–12 mois"
            hr_fu = "CT 6–12 mois"
        else:
            lr_mgmt = "CT 3 mois — TEP-TDM ou biopsie à considérer"
            hr_mgmt = "CT 3 mois — TEP-TDM ou biopsie"
            lr_fu = "CT 3 mois + évaluation oncologique"
            hr_fu = "CT 3 mois + évaluation oncologique urgente"
        size_cat = f"{nodule_size_mm:.0f} mm — Nodule solide"

    elif nodule_type == "ground-glass":
        if nodule_size_mm < 6.0:
            lr_mgmt = "Pas de suivi routinier"
            hr_mgmt = "Pas de suivi routinier"
            lr_fu = "Aucun"
            hr_fu = "Aucun"
        else:
            lr_mgmt = "CT 6–12 mois pour confirmer la persistance puis CT annuel × 5 ans"
            hr_mgmt = "CT 6–12 mois puis annuel × 5 ans"
            lr_fu = "CT 6–12 mois"
            hr_fu = "CT 6–12 mois"
        size_cat = f"{nodule_size_mm:.0f} mm — Verre dépoli pur"

    else:  # subsolid / part-solid
        if nodule_size_mm < 6.0:
            lr_mgmt = "Pas de suivi routinier"
            hr_mgmt = "Pas de suivi routinier"
            lr_fu = "Aucun"
            hr_fu = "Aucun"
        elif nodule_size_mm < 8.0:
            lr_mgmt = "CT 3–6 mois puis annuel × 5 ans si stable"
            hr_mgmt = "CT 3–6 mois puis annuel × 5 ans"
            lr_fu = "CT 3–6 mois"
            hr_fu = "CT 3–6 mois"
        else:
            lr_mgmt = "CT 3–6 mois — si persistant TEP ou biopsie"
            hr_mgmt = "CT 3–6 mois — si persistant TEP ou biopsie urgente"
            lr_fu = "CT 3–6 mois + évaluation oncologique"
            hr_fu = "CT 3–6 mois + biopsie à considérer"
        size_cat = f"{nodule_size_mm:.0f} mm — Nodule sous-solide (part-solid)"

    if multiple_nodules:
        lr_mgmt += " (nodules multiples — comparer au plus suspect)"
        hr_mgmt += " (nodules multiples — comparer au plus suspect)"

    return FleischnerResult(
        guideline_year="2017",
        nodule_type=nodule_type,
        size_category=size_cat,
        low_risk_management=lr_mgmt,
        high_risk_management=hr_mgmt,
        follow_up_low=lr_fu,
        follow_up_high=hr_fu,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tuberculose — Classification OMS + Sévérité
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TBSeverityResult:
    form: str
    severity: str
    bacillary_load: str
    drug_resistance_risk: str
    treatment_regimen: str
    duration_months: int
    inh_prophylaxis: str

def compute_tb_severity(
    cavitary: bool = False,
    bilateral: bool = False,
    multilobar: bool = False,
    smear_positive: bool = False,
    culture_positive: bool = False,
    previous_tb_treatment: bool = False,
    immunocompromised: bool = False,
    mdr_contact: bool = False,
) -> TBSeverityResult:
    """Classification sévérité tuberculose selon OMS 2022."""
    if cavitary and bilateral:
        form = "Extensive Pulmonary TB"
        severity = "Sévère"
    elif cavitary or (bilateral and multilobar):
        form = "Cavitary Pulmonary TB"
        severity = "Modérée à sévère"
    elif bilateral:
        form = "Bilateral Pulmonary TB"
        severity = "Modérée"
    else:
        form = "Unilateral Pulmonary TB"
        severity = "Légère"

    bacillary = "Élevée" if smear_positive else "Faible à modérée"

    dr_risk = "Élevé" if (mdr_contact or previous_tb_treatment) else "Faible"

    if dr_risk == "Élevé" or previous_tb_treatment:
        regimen = "DST + régime individualisé (anti-TB de 2e ligne)"
        duration = 18
        inh = "Non applicable — traitement curatif en cours"
    elif immunocompromised:
        regimen = "2HRZE/10HR (12 mois) — surveillance rapprochée"
        duration = 12
        inh = "Isoniazide prophylactique post-guérison si VIH+"
    else:
        regimen = "2HRZE/4HR — Protocole OMS standard"
        duration = 6
        inh = "6H ou 3HP (rifapentine + isoniazide) — contacts proches"

    return TBSeverityResult(
        form=form,
        severity=severity,
        bacillary_load=bacillary,
        drug_resistance_risk=dr_risk,
        treatment_regimen=regimen,
        duration_months=duration,
        inh_prophylaxis=inh,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Sévérité Pneumonie — Intégration CURB-65 + PSI
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pneumonia_severity(curb65_score: int, psi_class: str) -> dict[str, Any]:
    """Agrège CURB-65 + PSI pour une décision clinique intégrée."""
    if curb65_score >= 3 or psi_class in ("IV","V"):
        overall = "Sévère"
        recommendation = "Hospitalisation urgente — évaluer USI"
        color = "#C0392B"
        urgency = "Critique"
    elif curb65_score == 2 or psi_class == "III":
        overall = "Modérée"
        recommendation = "Hospitalisation — antibiothérapie IV"
        color = "#E67E22"
        urgency = "Élevée"
    else:
        overall = "Légère"
        recommendation = "Traitement ambulatoire — amoxicilline PO"
        color = "#27AE60"
        urgency = "Faible"

    return {
        "overall_severity": overall,
        "recommendation": recommendation,
        "color": color,
        "urgency": urgency,
        "antibiotic_first_line": {
            "Légère": "Amoxicilline 1g × 3/j PO × 5–7j (ou doxycycline si atypique)",
            "Modérée": "Amoxicilline-clavulanate IV + azithromycine IV",
            "Sévère": "C3G IV (ceftriaxone) + azithromycine IV ± antipseudomonal si facteur de risque",
        }.get(overall, "—"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Rapport consolidé — tous les scores disponibles
# ═══════════════════════════════════════════════════════════════════════════════

def build_clinical_summary(prediction: str, confidence: float, image_features: dict[str, Any]) -> dict[str, Any]:
    """
    Construit un résumé clinique complet en fonction de la prédiction IA.
    image_features: dict contenant les paramètres extraits de l'image ou saisis.
    """
    pred_lower = prediction.lower()
    summary: dict[str, Any] = {"prediction": prediction, "confidence": confidence}

    if "pneumonie" in pred_lower or "pneumonia" in pred_lower:
        curb = compute_curb65(
            confusion=image_features.get("confusion", False),
            urea_mmol=image_features.get("urea_mmol", 5.0),
            resp_rate=image_features.get("resp_rate", 18),
            systolic_bp=image_features.get("systolic_bp", 120),
            diastolic_bp=image_features.get("diastolic_bp", 80),
            age=image_features.get("age", 50),
        )
        psi = compute_psi(age=image_features.get("age", 50))
        summary["curb65"] = {"score": curb.score, "severity": curb.severity,
                              "mortality": curb.mortality_30d, "recommendation": curb.recommendation}
        summary["psi"] = {"class": psi.risk_class, "score": psi.score, "mortality": psi.mortality_30d}
        summary["integrated_severity"] = compute_pneumonia_severity(curb.score, psi.risk_class)

    elif "covid" in pred_lower:
        covid = compute_covid_ct_severity(
            right_upper_lobe=image_features.get("right_upper", 1),
            right_middle_lobe=image_features.get("right_middle", 2),
            right_lower_lobe=image_features.get("right_lower", 2),
            left_upper_lobe=image_features.get("left_upper", 1),
            left_lower_lobe=image_features.get("left_lower", 2),
            dominant_pattern=image_features.get("pattern", "ground_glass"),
        )
        summary["covid_ct_severity"] = {
            "total_score": covid.total_score,
            "severity": covid.severity,
            "involvement": f"{covid.lung_involvement_percent:.1f}%",
            "pattern": covid.pattern,
            "recommendation": covid.recommendation,
            "icu_probability": covid.icu_probability,
        }

    elif "cancer" in pred_lower or "nodule" in pred_lower or "masse" in pred_lower:
        size = image_features.get("nodule_size_mm", 10.0)
        lung_rads = compute_lung_rads(
            nodule_present=True,
            nodule_size_mm=size,
            nodule_type=image_features.get("nodule_type", "solid"),
        )
        fleischner = compute_fleischner(
            nodule_size_mm=size,
            nodule_type=image_features.get("nodule_type", "solid"),
            high_risk=image_features.get("smoker", False),
        )
        summary["lung_rads"] = {"category": lung_rads.category, "probability": lung_rads.malignancy_probability,
                                 "management": lung_rads.management}
        summary["fleischner"] = {"low_risk": fleischner.low_risk_management,
                                  "high_risk": fleischner.high_risk_management}
        if "cancer" in pred_lower:
            tnm = compute_tnm_lung(
                tumor_size_cm=image_features.get("tumor_size_cm", size / 10),
            )
            summary["tnm"] = {"T": tnm.T, "N": tnm.N, "M": tnm.M, "stage": tnm.stage,
                               "survival": tnm.five_year_survival, "treatment": tnm.treatment_recommendation}

    elif "bpco" in pred_lower or "emphysème" in pred_lower or "copd" in pred_lower:
        gold = compute_gold(
            fev1_percent_predicted=image_features.get("fev1_pct", 55.0),
            fev1_fvc_ratio=image_features.get("fev1_fvc", 0.62),
            cat_score=image_features.get("cat_score", 18),
            mmrc_score=image_features.get("mmrc", 2),
            exacerbations_last_year=image_features.get("exacerbations", 1),
        )
        bode = compute_bode(
            bmi=image_features.get("bmi", 22.0),
            fev1_percent_predicted=image_features.get("fev1_pct", 55.0),
            mmrc_dyspnea=image_features.get("mmrc", 2),
            six_min_walk_meters=image_features.get("six_min_walk", 280.0),
        )
        summary["gold"] = {"grade": gold.grade, "group": gold.group,
                            "exacerbation_risk": gold.exacerbation_risk, "treatment": gold.treatment}
        summary["bode"] = {"score": bode.score, "quartile": bode.quartile,
                           "survival_4yr": bode.four_year_survival}

    elif "tuberculose" in pred_lower or "tb" in pred_lower:
        tb = compute_tb_severity(
            cavitary=image_features.get("cavitary", False),
            bilateral=image_features.get("bilateral", False),
            multilobar=image_features.get("multilobar", False),
            smear_positive=image_features.get("smear_positive", False),
        )
        summary["tuberculosis"] = {"form": tb.form, "severity": tb.severity,
                                    "bacillary_load": tb.bacillary_load,
                                    "treatment": tb.treatment_regimen,
                                    "duration_months": tb.duration_months}

    return summary
