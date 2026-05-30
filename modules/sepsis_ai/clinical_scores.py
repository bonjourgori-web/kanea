"""
SepsisPredict AI — Scores cliniques de réanimation
====================================================
Implémentation de référence complète :
SOFA, qSOFA, NEWS2, APACHE II, SAPS II, MEWS, MODS Score.

Sources : Surviving Sepsis Campaign 2021, SCCM, ESICM, Singer et al. JAMA 2016.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# SOFA Score — Sequential Organ Failure Assessment
# (Singer et al., JAMA 2016 — Définition Sepsis-3)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SOFAResult:
    total: int
    components: dict[str, int]
    organ_failures: list[str]
    predicted_mortality: str
    severity: str
    sepsis_definition: str
    interpretation: str

def compute_sofa(
    pao2_fio2: float = 400.0,       # Rapport PaO2/FiO2 (mmHg)
    on_ventilator: bool = False,
    platelets_g_l: float = 150.0,   # Plaquettes G/L
    bilirubin_umol_l: float = 20.0, # Bilirubine µmol/L
    map_mmhg: float = 75.0,         # Pression artérielle moyenne
    dopamine_mcg_kg_min: float = 0.0,
    dobutamine: bool = False,
    norepinephrine_mcg_kg_min: float = 0.0,
    epinephrine_mcg_kg_min: float = 0.0,
    gcs: int = 15,
    creatinine_umol_l: float = 88.0,
    urine_output_ml_24h: float = 1500.0,
    baseline_sofa: int = 0,         # SOFA de base (pour delta SOFA)
) -> SOFAResult:
    """
    SOFA Score complet (0–24 pts).
    Sepsis-3 : SOFA ≥ 2 au-dessus de la ligne de base.
    Choc septique : vasopresseurs + lactate > 2 mmol/L.
    """
    scores: dict[str, int] = {}

    # 1. Respiration — PaO2/FiO2
    if pao2_fio2 >= 400:
        scores["respiration"] = 0
    elif pao2_fio2 >= 300:
        scores["respiration"] = 1
    elif pao2_fio2 >= 200:
        scores["respiration"] = 2 if not on_ventilator else 2
    elif pao2_fio2 >= 100 and on_ventilator:
        scores["respiration"] = 3
    else:
        scores["respiration"] = 4 if on_ventilator else 3

    # 2. Coagulation — Plaquettes
    if platelets_g_l >= 150:
        scores["coagulation"] = 0
    elif platelets_g_l >= 100:
        scores["coagulation"] = 1
    elif platelets_g_l >= 50:
        scores["coagulation"] = 2
    elif platelets_g_l >= 20:
        scores["coagulation"] = 3
    else:
        scores["coagulation"] = 4

    # 3. Foie — Bilirubine (µmol/L) [×17.1 pour mg/dL]
    bili_mg = bilirubin_umol_l / 17.1
    if bili_mg < 1.2:
        scores["liver"] = 0
    elif bili_mg < 2.0:
        scores["liver"] = 1
    elif bili_mg < 6.0:
        scores["liver"] = 2
    elif bili_mg < 12.0:
        scores["liver"] = 3
    else:
        scores["liver"] = 4

    # 4. Cardiovasculaire — MAP + vasopresseurs
    if map_mmhg >= 70 and not dobutamine and dopamine_mcg_kg_min == 0:
        scores["cardiovascular"] = 0
    elif map_mmhg < 70:
        scores["cardiovascular"] = 1
    elif dobutamine or dopamine_mcg_kg_min <= 5:
        scores["cardiovascular"] = 2
    elif dopamine_mcg_kg_min <= 15 or (norepinephrine_mcg_kg_min > 0 and norepinephrine_mcg_kg_min <= 0.1):
        scores["cardiovascular"] = 3
    else:
        scores["cardiovascular"] = 4

    # 5. Neurologique — GCS
    if gcs == 15:
        scores["neurological"] = 0
    elif gcs >= 13:
        scores["neurological"] = 1
    elif gcs >= 10:
        scores["neurological"] = 2
    elif gcs >= 6:
        scores["neurological"] = 3
    else:
        scores["neurological"] = 4

    # 6. Rénal — Créatinine (µmol/L) + diurèse
    creat_mg = creatinine_umol_l / 88.4
    if creat_mg < 1.2:
        cr_score = 0
    elif creat_mg < 2.0:
        cr_score = 1
    elif creat_mg < 3.5:
        cr_score = 2
    elif creat_mg < 5.0 or urine_output_ml_24h < 500:
        cr_score = 3
    else:
        cr_score = 4
    if urine_output_ml_24h < 200:
        cr_score = max(cr_score, 4)
    scores["renal"] = cr_score

    total = sum(scores.values())
    delta = total - baseline_sofa

    # Organes défaillants (score ≥ 3)
    organ_map = {"respiration": "Pulmonaire", "coagulation": "Coagulation",
                 "liver": "Hépatique", "cardiovascular": "Cardiovasculaire",
                 "neurological": "Neurologique", "renal": "Rénal"}
    failures = [organ_map[k] for k, v in scores.items() if v >= 3]

    # Mortalité estimée (Ferreira et al., ICM 2001)
    if total <= 1:
        mortality = "< 1%"
        severity = "Normal"
    elif total <= 3:
        mortality = "2–4%"
        severity = "Léger"
    elif total <= 5:
        mortality = "15–20%"
        severity = "Modéré"
    elif total <= 8:
        mortality = "20–40%"
        severity = "Sévère"
    elif total <= 11:
        mortality = "40–60%"
        severity = "Critique"
    else:
        mortality = "> 80%"
        severity = "Extrême"

    # Définition Sepsis-3
    if delta >= 2:
        sep_def = f"SEPSIS confirmé (ΔSOFA = {delta:+d} ≥ 2)"
    elif delta >= 1:
        sep_def = f"Dysfonction d'organe suspectée (ΔSOFA = {delta:+d})"
    else:
        sep_def = f"SOFA stable (ΔSOFA = {delta:+d})"

    return SOFAResult(
        total=total,
        components=scores,
        organ_failures=failures,
        predicted_mortality=mortality,
        severity=severity,
        sepsis_definition=sep_def,
        interpretation=f"SOFA {total}/24 — {len(failures)} organe(s) en défaillance — Mortalité estimée {mortality}",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# qSOFA — Quick SOFA (Bedside Screening Tool)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QSOFAResult:
    score: int
    components: dict[str, bool]
    high_risk: bool
    recommendation: str
    action: str

def compute_qsofa(
    resp_rate: int = 16,
    systolic_bp: int = 120,
    altered_mentation: bool = False,   # GCS < 15 ou confusion aiguë
) -> QSOFAResult:
    """
    qSOFA (Quick SOFA) — outil de dépistage rapide au lit du patient.
    Score ≥ 2 → risque élevé de sepsis → SOFA complet requis.
    """
    rr_flag = resp_rate >= 22
    bp_flag = systolic_bp <= 100
    men_flag = altered_mentation

    score = sum([rr_flag, bp_flag, men_flag])

    if score >= 2:
        high_risk = True
        rec = "RISQUE ÉLEVÉ — Évaluer SOFA complet + hémocultures + lactate"
        action = "Alerte sepsis — mesures urgentes protocole Surviving Sepsis Campaign"
    elif score == 1:
        high_risk = False
        rec = "Risque modéré — surveillance rapprochée recommandée"
        action = "Réévaluation clinique toutes les 2h — bilan biologique complet"
    else:
        high_risk = False
        rec = "Risque faible — poursuite surveillance standard"
        action = "Monitoring paramètres vitaux standard"

    return QSOFAResult(
        score=score,
        components={"FR ≥ 22/min": rr_flag, "PAS ≤ 100 mmHg": bp_flag, "Altération mentale": men_flag},
        high_risk=high_risk,
        recommendation=rec,
        action=action,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS2 — National Early Warning Score 2 (Royal College of Physicians 2017)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NEWS2Result:
    total: int
    components: dict[str, int]
    clinical_risk: str
    response: str
    monitoring_frequency: str
    escalation: str

def compute_news2(
    resp_rate: int = 16,
    spo2: float = 98.0,
    spo2_scale2: bool = False,         # Vrai pour patients BPCO (cible 88–92%)
    on_supplemental_o2: bool = False,
    systolic_bp: int = 120,
    pulse: int = 75,
    consciousness: str = "A",          # A=Alert, C=Confused, V=Voice, P=Pain, U=Unresponsive (ACVPU)
    temperature: float = 37.0,
) -> NEWS2Result:
    """
    NEWS2 Score — National Early Warning Score 2.
    Score 0–4 faible, 5–6 moyen, ≥7 élevé, tout ≥3 dans une catégorie → réponse urgente.
    """
    s: dict[str, int] = {}

    # Fréquence respiratoire
    if resp_rate <= 8:         s["resp_rate"] = 3
    elif resp_rate <= 11:      s["resp_rate"] = 1
    elif resp_rate <= 20:      s["resp_rate"] = 0
    elif resp_rate <= 24:      s["resp_rate"] = 2
    else:                      s["resp_rate"] = 3

    # SpO2 (Scale 1 par défaut, Scale 2 pour BPCO)
    if not spo2_scale2:
        if spo2 <= 91:         s["spo2"] = 3
        elif spo2 <= 93:       s["spo2"] = 2
        elif spo2 <= 94:       s["spo2"] = 1
        else:                  s["spo2"] = 0
    else:  # Scale 2 (BPCO — cible 88–92%)
        if spo2 <= 83:         s["spo2"] = 3
        elif spo2 <= 85:       s["spo2"] = 2
        elif spo2 <= 87:       s["spo2"] = 1
        elif spo2 <= 92:       s["spo2"] = 0
        elif spo2 == 93:       s["spo2"] = 1
        elif spo2 == 94:       s["spo2"] = 2
        else:                  s["spo2"] = 3

    # O2 supplémentaire
    s["supplemental_o2"] = 2 if on_supplemental_o2 else 0

    # Pression artérielle systolique
    if systolic_bp <= 90:      s["systolic_bp"] = 3
    elif systolic_bp <= 100:   s["systolic_bp"] = 2
    elif systolic_bp <= 110:   s["systolic_bp"] = 1
    elif systolic_bp <= 219:   s["systolic_bp"] = 0
    else:                      s["systolic_bp"] = 3

    # Pouls
    if pulse <= 40:            s["pulse"] = 3
    elif pulse <= 50:          s["pulse"] = 1
    elif pulse <= 90:          s["pulse"] = 0
    elif pulse <= 110:         s["pulse"] = 1
    elif pulse <= 130:         s["pulse"] = 2
    else:                      s["pulse"] = 3

    # Conscience (ACVPU)
    consciousness_scores = {"A": 0, "C": 3, "V": 3, "P": 3, "U": 3}
    s["consciousness"] = consciousness_scores.get(consciousness.upper(), 0)

    # Température
    if temperature <= 35.0:    s["temperature"] = 3
    elif temperature <= 36.0:  s["temperature"] = 1
    elif temperature <= 38.0:  s["temperature"] = 0
    elif temperature <= 39.0:  s["temperature"] = 1
    else:                      s["temperature"] = 2

    total = sum(s.values())
    any_single_3 = any(v >= 3 for v in s.values())

    # Niveaux de risque RCP
    if total <= 4 and not any_single_3:
        risk = "Faible"
        response = "Évaluation infirmière"
        frequency = "Au minimum toutes les 12h"
        escalation = "Informer l'infirmière responsable"
    elif total in (5, 6) or (total <= 4 and any_single_3):
        risk = "Moyen"
        response = "Revue médicale urgente"
        frequency = "Toutes les 1–2h minimum"
        escalation = "Alerter le médecin en charge — évaluation dans l'heure"
    else:  # ≥ 7 ou toute catégorie ≥ 3
        risk = "Élevé"
        response = "Réponse d'urgence"
        frequency = "Monitoring continu"
        escalation = "ALERTE CRITIQUE — Équipe de réanimation d'urgence"

    return NEWS2Result(
        total=total,
        components=s,
        clinical_risk=risk,
        response=response,
        monitoring_frequency=frequency,
        escalation=escalation,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# APACHE II — Acute Physiology and Chronic Health Evaluation II
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class APACHEIIResult:
    score: int
    aps_score: int            # Acute Physiology Score (0–60)
    age_score: int
    chronic_score: int
    predicted_mortality: str
    severity: str

def compute_apache_ii(
    temperature: float = 37.0,
    map_mmhg: float = 75.0,
    heart_rate: int = 75,
    resp_rate: int = 16,
    fio2: float = 0.21,
    pao2: float = 80.0,        # mmHg
    paco2: float = 40.0,       # mmHg
    arterial_ph: float = 7.40,
    sodium_meq: float = 140.0,
    potassium_meq: float = 4.0,
    creatinine_mg: float = 1.0,
    acute_renal_failure: bool = False,
    hematocrit: float = 40.0,
    wbc_thousands: float = 8.0,
    gcs: int = 15,
    age: int = 50,
    chronic_organ_failure: bool = False,
    nonoperative: bool = True,         # True si non chirurgical ou urgence
) -> APACHEIIResult:
    """APACHE II Score (Knaus et al., 1985). Score 0–71."""

    def _range_score(val: float, thresholds: list) -> int:
        """Interpolation des scores par plages."""
        for score, low, high in thresholds:
            if low <= val < high:
                return score
        return 4

    # Temperature (°C)
    temp_s = _range_score(temperature, [
        (4,41,float('inf')),(3,39,41),(1,38.5,39),(0,36,38.5),
        (1,34,36),(2,32,34),(3,30,32),(4,-float('inf'),30)
    ])

    # MAP
    map_s = _range_score(map_mmhg, [
        (4,160,float('inf')),(3,130,160),(2,110,130),(0,70,110),
        (2,50,70),(4,-float('inf'),50)
    ])

    # HR
    hr_s = _range_score(heart_rate, [
        (4,180,float('inf')),(3,140,180),(2,110,140),(0,70,110),
        (2,55,70),(3,40,55),(4,-float('inf'),40)
    ])

    # RR
    rr_s = _range_score(resp_rate, [
        (4,50,float('inf')),(3,35,50),(1,25,35),(0,12,25),
        (1,10,12),(2,6,10),(4,-float('inf'),6)
    ])

    # Oxygénation
    if fio2 >= 0.50:
        a_a_gradient = (713 * fio2 - paco2 / 0.8) - pao2
        oxy_s = _range_score(a_a_gradient, [
            (4,500,float('inf')),(3,350,500),(2,200,350),(0,-float('inf'),200)
        ])
    else:
        oxy_s = _range_score(pao2, [
            (4,-float('inf'),55),(3,55,60),(1,61,70),(0,71,float('inf'))
        ])

    # pH artériel
    ph_s = _range_score(arterial_ph, [
        (4,7.70,float('inf')),(3,7.60,7.70),(1,7.50,7.60),(0,7.33,7.50),
        (2,7.25,7.33),(3,7.15,7.25),(4,-float('inf'),7.15)
    ])

    # Sodium
    na_s = _range_score(sodium_meq, [
        (4,180,float('inf')),(3,160,180),(2,155,160),(1,150,155),
        (0,130,150),(2,120,130),(3,111,120),(4,-float('inf'),111)
    ])

    # Potassium
    k_s = _range_score(potassium_meq, [
        (4,7.0,float('inf')),(3,6.0,7.0),(1,5.5,6.0),(0,3.5,5.5),
        (1,3.0,3.5),(2,2.5,3.0),(4,-float('inf'),2.5)
    ])

    # Créatinine (doublement si IRA)
    cr_s = _range_score(creatinine_mg, [
        (4,3.5,float('inf')),(3,2.0,3.5),(2,1.5,2.0),(0,0.6,1.5),(2,-float('inf'),0.6)
    ])
    if acute_renal_failure:
        cr_s = min(cr_s * 2, 8)

    # Hématocrite
    ht_s = _range_score(hematocrit, [
        (4,60,float('inf')),(2,50,60),(1,46,50),(0,30,46),
        (2,20,30),(4,-float('inf'),20)
    ])

    # Leucocytes (×1000/mm³)
    wbc_s = _range_score(wbc_thousands, [
        (4,40,float('inf')),(2,20,40),(1,15,20),(0,3,15),
        (2,1,3),(4,-float('inf'),1)
    ])

    # GCS — points = 15 − GCS
    gcs_s = 15 - gcs

    aps = temp_s + map_s + hr_s + rr_s + oxy_s + ph_s + na_s + k_s + cr_s + ht_s + wbc_s + gcs_s

    # Score âge
    if age < 45:       age_s = 0
    elif age < 55:     age_s = 2
    elif age < 65:     age_s = 3
    elif age < 75:     age_s = 5
    else:              age_s = 6

    # Score insuffisance chronique (A=nonoperative/B=postop urgence, C=postop programmé)
    chronic_s = 5 if (nonoperative or True) and chronic_organ_failure else (2 if chronic_organ_failure else 0)

    total = aps + age_s + chronic_s

    # Mortalité estimée (Knaus nomogram — régression logistique)
    logit = -3.517 + total * 0.146
    mort_pct = 100 / (1 + math.exp(-logit))

    if mort_pct < 10:
        severity = "Faible"
    elif mort_pct < 25:
        severity = "Modéré"
    elif mort_pct < 50:
        severity = "Sévère"
    elif mort_pct < 75:
        severity = "Critique"
    else:
        severity = "Extrême"

    return APACHEIIResult(
        score=total,
        aps_score=aps,
        age_score=age_s,
        chronic_score=chronic_s,
        predicted_mortality=f"{mort_pct:.1f}%",
        severity=severity,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SAPS II — Simplified Acute Physiology Score II (Le Gall, JAMA 1993)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SAPSIIResult:
    score: int
    predicted_mortality: str
    severity: str

def compute_saps_ii(
    age: int = 50,
    heart_rate: int = 75,
    systolic_bp: int = 120,
    temperature: float = 37.0,
    pao2_fio2: float = 400.0,
    on_ventilator: bool = False,
    urine_output_l_24h: float = 1.5,
    bun_mmol_l: float = 5.0,
    wbc_thousands: float = 8.0,
    potassium_meq: float = 4.0,
    sodium_meq: float = 140.0,
    bicarbonate_meq: float = 24.0,
    bilirubin_umol_l: float = 17.0,
    gcs: int = 15,
    chronic_disease: str = "none",   # none | metastatic_cancer | hematologic | aids
    admission_type: str = "medical", # medical | scheduled_surgery | unscheduled_surgery
) -> SAPSIIResult:
    """SAPS II Score — Mortalité hospitalière ICU."""
    s = 0

    # Âge
    s += {(0,40):0,(40,60):7,(60,70):12,(70,75):15,(75,80):16,(80,200):18}.get(
        next(((a,b) for a,b in [(0,40),(40,60),(60,70),(70,75),(75,80),(80,200)] if a<=age<b), (0,40)), 0)

    # FC
    if heart_rate < 40 or heart_rate >= 160:  s += 11
    elif heart_rate < 70 or heart_rate >= 120: s += 4 if heart_rate >= 120 else 2
    elif heart_rate < 120:                      s += 0

    # PAS
    if systolic_bp < 70:    s += 13
    elif systolic_bp < 100: s += 5
    elif systolic_bp < 200: s += 0
    else:                   s += 2

    # Température
    if temperature >= 39.0: s += 3

    # PaO2/FiO2 (seulement si ventilé ou CPAP)
    if on_ventilator:
        if pao2_fio2 < 100: s += 11
        elif pao2_fio2 < 200: s += 9
        elif pao2_fio2 < 400 - 1: s += 6

    # Diurèse
    if urine_output_l_24h < 0.5:   s += 11
    elif urine_output_l_24h < 1.0: s += 4

    # Urée
    bun_mg = bun_mmol_l * 2.8
    if bun_mg >= 84:   s += 10
    elif bun_mg >= 28: s += 6

    # Leucocytes
    if wbc_thousands < 1.0 or wbc_thousands >= 20: s += 3 if wbc_thousands >= 20 else 12

    # Potassium
    if potassium_meq < 3.0 or potassium_meq >= 5.0: s += 3

    # Sodium
    if sodium_meq < 125:    s += 5
    elif sodium_meq < 145:  s += 0
    else:                   s += 1

    # Bicarbonates
    if bicarbonate_meq < 15:   s += 6
    elif bicarbonate_meq < 20: s += 3

    # Bilirubine
    bili_mg = bilirubin_umol_l / 17.1
    if bili_mg >= 6.0:   s += 9
    elif bili_mg >= 4.0: s += 4

    # GCS
    if gcs < 6:    s += 26
    elif gcs < 9:  s += 13
    elif gcs < 11: s += 7
    elif gcs < 14: s += 5

    # Maladie chronique
    chronic_map = {"metastatic_cancer": 9, "hematologic": 10, "aids": 17, "none": 0}
    s += chronic_map.get(chronic_disease, 0)

    # Type d'admission
    adm_map = {"medical": 6, "unscheduled_surgery": 8, "scheduled_surgery": 0}
    s += adm_map.get(admission_type, 6)

    # Mortalité (Le Gall regression logistique)
    logit = -7.7631 + 0.0737 * s + 0.9971 * math.log(s + 1)
    mort_pct = 100 * math.exp(logit) / (1 + math.exp(logit))

    if mort_pct < 10:     severity = "Faible"
    elif mort_pct < 25:   severity = "Modéré"
    elif mort_pct < 50:   severity = "Sévère"
    elif mort_pct < 75:   severity = "Critique"
    else:                 severity = "Extrême"

    return SAPSIIResult(score=s, predicted_mortality=f"{mort_pct:.1f}%", severity=severity)


# ═══════════════════════════════════════════════════════════════════════════════
# MEWS — Modified Early Warning Score
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MEWSResult:
    score: int
    risk_level: str
    action: str
    monitor_frequency: str

def compute_mews(
    systolic_bp: int = 120,
    heart_rate: int = 75,
    resp_rate: int = 16,
    temperature: float = 37.0,
    consciousness: str = "A",  # A/V/P/U (AVPU)
    urine_output: str = "normal",  # normal|reduced|nil|catheter
) -> MEWSResult:
    """MEWS — 5 paramètres vitaux + conscience + diurèse."""
    s = 0

    # PAS
    if systolic_bp <= 70:         s += 3
    elif systolic_bp <= 80:       s += 2
    elif systolic_bp <= 100:      s += 1
    elif systolic_bp <= 199:      s += 0
    else:                         s += 2

    # FC
    if heart_rate < 40:           s += 2
    elif heart_rate <= 50:        s += 1
    elif heart_rate <= 100:       s += 0
    elif heart_rate <= 110:       s += 1
    elif heart_rate <= 129:       s += 2
    else:                         s += 3

    # FR
    if resp_rate < 9:             s += 2
    elif resp_rate <= 14:         s += 0
    elif resp_rate <= 20:         s += 1
    elif resp_rate <= 29:         s += 2
    else:                         s += 3

    # Température
    if temperature < 35.0:        s += 2
    elif temperature <= 36.0:     s += 1
    elif temperature <= 38.0:     s += 0
    elif temperature <= 38.5:     s += 1
    else:                         s += 2

    # Conscience (AVPU)
    avpu_scores = {"A": 0, "V": 1, "P": 2, "U": 3}
    s += avpu_scores.get(consciousness.upper(), 0)

    if s <= 2:
        risk, action, freq = "Faible", "Surveillance standard", "Toutes les 4–6h"
    elif s <= 4:
        risk, action, freq = "Modéré", "Revue médicale dans l'heure", "Toutes les 1–2h"
    elif s <= 6:
        risk, action, freq = "Élevé", "Revue médicale urgente", "Toutes les 30 min"
    else:
        risk, action, freq = "Critique", "URGENCE — Réanimation immédiate", "Monitoring continu"

    return MEWSResult(score=s, risk_level=risk, action=action, monitor_frequency=freq)


# ═══════════════════════════════════════════════════════════════════════════════
# MODS — Multiple Organ Dysfunction Score (Marshall, CCM 1995)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MODSResult:
    total: int
    components: dict[str, int]
    organ_systems_failing: int
    predicted_mortality: str
    severity: str

def compute_mods(
    pao2_fio2: float = 400.0,
    platelets_g_l: float = 150.0,
    bilirubin_umol_l: float = 17.0,
    pressure_adjusted_hr: float = 10.0,  # PAR = HR × CVP / MAP
    gcs: int = 15,
    creatinine_umol_l: float = 88.0,
) -> MODSResult:
    """MODS Score (Marshall 1995) — 0–24 points (6 systèmes × 4 pts)."""
    s: dict[str, int] = {}

    # Pulmonaire
    if pao2_fio2 > 300:    s["pulmonary"] = 0
    elif pao2_fio2 > 226:  s["pulmonary"] = 1
    elif pao2_fio2 > 151:  s["pulmonary"] = 2
    elif pao2_fio2 > 75:   s["pulmonary"] = 3
    else:                  s["pulmonary"] = 4

    # Rénal
    cr_mg = creatinine_umol_l / 88.4
    if cr_mg <= 1.1:       s["renal"] = 0
    elif cr_mg <= 2.3:     s["renal"] = 1
    elif cr_mg <= 3.9:     s["renal"] = 2
    elif cr_mg <= 6.6:     s["renal"] = 3
    else:                  s["renal"] = 4

    # Hépatique
    bili_mg = bilirubin_umol_l / 17.1
    if bili_mg <= 1.2:     s["hepatic"] = 0
    elif bili_mg <= 3.5:   s["hepatic"] = 1
    elif bili_mg <= 7.0:   s["hepatic"] = 2
    elif bili_mg <= 14.0:  s["hepatic"] = 3
    else:                  s["hepatic"] = 4

    # Cardiovasculaire (PAR)
    if pressure_adjusted_hr <= 10:    s["cardiovascular"] = 0
    elif pressure_adjusted_hr <= 15:  s["cardiovascular"] = 1
    elif pressure_adjusted_hr <= 20:  s["cardiovascular"] = 2
    elif pressure_adjusted_hr <= 30:  s["cardiovascular"] = 3
    else:                             s["cardiovascular"] = 4

    # Hématologique (plaquettes)
    if platelets_g_l > 120:   s["hematologic"] = 0
    elif platelets_g_l > 81:  s["hematologic"] = 1
    elif platelets_g_l > 51:  s["hematologic"] = 2
    elif platelets_g_l > 21:  s["hematologic"] = 3
    else:                     s["hematologic"] = 4

    # Neurologique (GCS)
    if gcs == 15:             s["neurological"] = 0
    elif gcs >= 13:           s["neurological"] = 1
    elif gcs >= 10:           s["neurological"] = 2
    elif gcs >= 7:            s["neurological"] = 3
    else:                     s["neurological"] = 4

    total = sum(s.values())
    failing = sum(1 for v in s.values() if v >= 3)

    # Mortalité estimée (Marshall 1995 nomogram)
    if total == 0:      mortality = "< 1%"
    elif total <= 4:    mortality = "1–7%"
    elif total <= 8:    mortality = "3–17%"
    elif total <= 12:   mortality = "25–47%"
    elif total <= 16:   mortality = "50–67%"
    elif total <= 20:   mortality = "67–87%"
    else:               mortality = "> 87%"

    severity_map = {0: "Normal", (1,4): "Léger", (5,8): "Modéré",
                    (9,12): "Sévère", (13,20): "Critique", (21,24): "Extrême"}
    severity = "Léger"
    for rng, sev in severity_map.items():
        if isinstance(rng, int):
            if total == rng: severity = sev
        elif rng[0] <= total <= rng[1]:
            severity = sev

    return MODSResult(
        total=total, components=s,
        organ_systems_failing=failing,
        predicted_mortality=mortality,
        severity=severity,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Surviving Sepsis Campaign Bundle — Checklist 1h et 3h
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ssc_bundle_compliance(
    lactate_measured: bool = False,
    blood_cultures_before_abx: bool = False,
    broad_spectrum_antibiotics: bool = False,
    crystalloids_30ml_kg: bool = False,
    vasopressors_if_hypotensive: bool = False,
    lactate_mmol_l: float = 0.0,
    repeat_lactate_if_high: bool = False,
) -> dict[str, Any]:
    """
    Surviving Sepsis Campaign 2021 — Bundle 1h.
    Compliance : tous les items doivent être réalisés dans l'heure du diagnostic.
    """
    items = {
        "Mesure lactate": lactate_measured,
        "Hémocultures avant antibiotiques": blood_cultures_before_abx,
        "Antibiotiques large spectre": broad_spectrum_antibiotics,
        "Remplissage 30 mL/kg cristalloïdes": crystalloids_30ml_kg,
        "Vasopresseurs si hypotension persistante": vasopressors_if_hypotensive,
    }
    if lactate_mmol_l >= 2.0:
        items["Mesure lactate à 2h si initial élevé"] = repeat_lactate_if_high

    completed = sum(items.values())
    total = len(items)
    compliance = completed / total * 100

    return {
        "bundle_items": items,
        "completed": completed,
        "total": total,
        "compliance_pct": round(compliance, 1),
        "status": "Complet" if compliance == 100 else "Incomplet",
        "missing_items": [k for k, v in items.items() if not v],
        "lactate_interpretation": (
            "Choc septique cryptique probable (lactate ≥ 4 mmol/L)" if lactate_mmol_l >= 4.0 else
            "Hypoperfusion tissulaire (lactate ≥ 2 mmol/L)" if lactate_mmol_l >= 2.0 else
            "Lactate normal (< 2 mmol/L)"
        ) if lactate_mmol_l > 0 else "Non mesuré",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Résumé consolidé — tous les scores
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_sepsis_scores(params: dict[str, Any]) -> dict[str, Any]:
    """
    Calcule tous les scores de réanimation à partir d'un dict de paramètres.
    """
    results: dict[str, Any] = {}

    # SOFA
    try:
        sofa = compute_sofa(
            pao2_fio2=params.get("pao2_fio2", 400),
            on_ventilator=params.get("on_ventilator", False),
            platelets_g_l=params.get("platelets", 150),
            bilirubin_umol_l=params.get("bilirubin", 17),
            map_mmhg=params.get("map", 75),
            dopamine_mcg_kg_min=params.get("dopamine", 0),
            dobutamine=params.get("dobutamine", False),
            norepinephrine_mcg_kg_min=params.get("norepinephrine", 0),
            gcs=params.get("gcs", 15),
            creatinine_umol_l=params.get("creatinine", 88),
            urine_output_ml_24h=params.get("urine_output", 1500),
            baseline_sofa=params.get("baseline_sofa", 0),
        )
        results["sofa"] = {
            "total": sofa.total,
            "components": sofa.components,
            "organ_failures": sofa.organ_failures,
            "mortality": sofa.predicted_mortality,
            "severity": sofa.severity,
            "sepsis_definition": sofa.sepsis_definition,
        }
    except Exception as e:
        results["sofa"] = {"error": str(e)}

    # qSOFA
    try:
        qsofa = compute_qsofa(
            resp_rate=params.get("resp_rate", 16),
            systolic_bp=params.get("systolic_bp", 120),
            altered_mentation=params.get("gcs", 15) < 15,
        )
        results["qsofa"] = {
            "score": qsofa.score,
            "high_risk": qsofa.high_risk,
            "components": qsofa.components,
            "recommendation": qsofa.recommendation,
        }
    except Exception as e:
        results["qsofa"] = {"error": str(e)}

    # NEWS2
    try:
        news2 = compute_news2(
            resp_rate=params.get("resp_rate", 16),
            spo2=params.get("spo2", 98),
            on_supplemental_o2=params.get("on_o2", False),
            systolic_bp=params.get("systolic_bp", 120),
            pulse=params.get("heart_rate", 75),
            consciousness="A" if params.get("gcs", 15) == 15 else ("C" if params.get("gcs", 15) >= 13 else "V"),
            temperature=params.get("temperature", 37.0),
        )
        results["news2"] = {
            "total": news2.total,
            "clinical_risk": news2.clinical_risk,
            "response": news2.response,
            "escalation": news2.escalation,
        }
    except Exception as e:
        results["news2"] = {"error": str(e)}

    # APACHE II
    try:
        apache = compute_apache_ii(
            temperature=params.get("temperature", 37.0),
            map_mmhg=params.get("map", 75),
            heart_rate=params.get("heart_rate", 75),
            resp_rate=params.get("resp_rate", 16),
            gcs=params.get("gcs", 15),
            age=params.get("age", 50),
            creatinine_mg=params.get("creatinine", 88) / 88.4,
        )
        results["apache_ii"] = {
            "score": apache.score,
            "mortality": apache.predicted_mortality,
            "severity": apache.severity,
        }
    except Exception as e:
        results["apache_ii"] = {"error": str(e)}

    # MEWS
    try:
        mews = compute_mews(
            systolic_bp=params.get("systolic_bp", 120),
            heart_rate=params.get("heart_rate", 75),
            resp_rate=params.get("resp_rate", 16),
            temperature=params.get("temperature", 37.0),
            consciousness="A" if params.get("gcs", 15) == 15 else "V",
        )
        results["mews"] = {
            "score": mews.score,
            "risk_level": mews.risk_level,
            "action": mews.action,
            "monitor_frequency": mews.monitor_frequency,
        }
    except Exception as e:
        results["mews"] = {"error": str(e)}

    # SSC Bundle
    try:
        ssc = compute_ssc_bundle_compliance(
            lactate_measured=params.get("lactate", 0) > 0,
            blood_cultures_before_abx=params.get("blood_cultures_done", False),
            broad_spectrum_antibiotics=params.get("antibiotics_given", False),
            crystalloids_30ml_kg=params.get("fluid_resuscitation", False),
            vasopressors_if_hypotensive=params.get("vasopressors", False) or params.get("norepinephrine", 0) > 0,
            lactate_mmol_l=params.get("lactate", 0),
        )
        results["ssc_bundle"] = ssc
    except Exception as e:
        results["ssc_bundle"] = {"error": str(e)}

    return results
