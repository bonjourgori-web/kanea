"""
GastroAI — Scores cliniques gastro-entérologiques
===================================================
Child-Pugh · MELD · MELD-Na · CDAI · Harvey-Bradshaw · Mayo Score
Ranson · BISAP · Glasgow-Blatchford · Rockall · TNM Colorectal · Paris Classification.

Sources :
  - American Gastroenterological Association (AGA) 2023
  - European Society of Gastrointestinal Endoscopy (ESGE) 2023
  - American Society for Gastrointestinal Endoscopy (ASGE) 2023
  - United European Gastroenterology (UEG) 2023
  - AASLD Liver Guidelines 2023 · EASL 2023
  - ACG Pancreatitis Guidelines 2023
  - The Lancet Gastroenterology & Hepatology · PubMed
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Child-Pugh Score — Cirrhose hépatique (Child & Turcotte 1964 / Pugh 1972)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChildPughResult:
    score: int
    grade: str
    one_year_survival: str
    two_year_survival: str
    transplant_indication: bool
    recommendation: str

def compute_child_pugh(
    bilirubin_umol: float = 17.0,    # µmol/L (normale < 17)
    albumin_g: float = 40.0,         # g/L (normale > 35)
    inr: float = 1.0,                # (normale < 1.7)
    ascites: int = 0,                # 0=absente, 1=légère, 2=modérée-sévère
    encephalopathy: int = 0,         # 0=absente, 1=grade I-II, 2=grade III-IV
) -> ChildPughResult:
    """
    Child-Pugh Score — 5 paramètres · Score 5–15.
    Classe A (5–6) : cirrhose compensée · B (7–9) : décompensée · C (10–15) : terminale.
    """
    pts = 0
    # Bilirubine
    if bilirubin_umol < 34:    pts += 1
    elif bilirubin_umol < 51:  pts += 2
    else:                      pts += 3
    # Albumine
    if albumin_g > 35:         pts += 1
    elif albumin_g >= 28:      pts += 2
    else:                      pts += 3
    # INR
    if inr < 1.7:              pts += 1
    elif inr < 2.3:            pts += 2
    else:                      pts += 3
    # Ascite
    pts += [1, 2, 3][min(ascites, 2)]
    # Encéphalopathie
    pts += [1, 2, 3][min(encephalopathy, 2)]

    if pts <= 6:
        grade = "A"
        s1y, s2y = "100%", "85%"
        transplant = False
        rec = ("Traitement étiologique — surveillance semestrielle (écho+AFP) — "
               "endoscopie varices — pas d'indication transplantation immédiate")
    elif pts <= 9:
        grade = "B"
        s1y, s2y = "81%", "57%"
        transplant = True
        rec = ("Évaluation transplantation hépatique — traitement complications — "
               "diurétiques ascite — beta-bloquants varices — TIPS si réfractaire")
    else:
        grade = "C"
        s1y, s2y = "45%", "35%"
        transplant = True
        rec = ("TRANSPLANTATION URGENTE — soins palliatifs si contre-indication — "
               "traitement intensif complications — TIPS — encéphalopathie : lactulose + rifaximine")

    return ChildPughResult(
        score=pts, grade=grade,
        one_year_survival=s1y, two_year_survival=s2y,
        transplant_indication=transplant, recommendation=rec,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MELD & MELD-Na Score — Insuffisance hépatique (Kamath 2001 / Kim 2008)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MELDResult:
    meld: float
    meld_na: float
    mortality_90d: str
    transplant_priority: str
    recommendation: str

def compute_meld(
    creatinine_umol: float = 88.0,   # µmol/L
    bilirubin_umol: float = 17.0,    # µmol/L
    inr: float = 1.0,
    sodium_mmol: float = 140.0,      # mmol/L (pour MELD-Na)
    dialysis: bool = False,
) -> MELDResult:
    """
    MELD Score (UNOS 2016 formula) + MELD-Na (Kim 2008).
    Prédiction mortalité à 90 jours — liste d'attente transplantation.
    """
    # Valeurs minimales UNOS
    creat_mg = max(1.0, creatinine_umol / 88.4)
    bili_mg  = max(1.0, bilirubin_umol / 17.1)
    inr_v    = max(1.0, inr)
    if dialysis:
        creat_mg = 4.0

    meld = round(3.78 * math.log(bili_mg) + 11.2 * math.log(inr_v) + 9.57 * math.log(creat_mg) + 6.43, 1)
    meld = max(6.0, min(meld, 40.0))

    # MELD-Na
    na = max(125.0, min(sodium_mmol, 140.0))
    meld_na = round(meld - na - (0.025 * meld * (140 - na)) + 140, 1)
    meld_na = max(6.0, min(meld_na, 40.0))

    if meld < 10:
        mort = "< 2%"
        prio = "Faible"
        rec  = "Traitement ambulatoire — surveillance trimestrielle"
    elif meld < 20:
        mort = "6–20%"
        prio = "Modérée"
        rec  = "Évaluation transplantation — centre hépatologie — optimisation traitement"
    elif meld < 30:
        mort = "20–50%"
        prio = "Élevée — liste urgence"
        rec  = "Inscription urgente transplantation — soins intensifs hépatologie"
    else:
        mort = "> 50%"
        prio = "Très élevée — super-urgence"
        rec  = "SUPER-URGENCE transplantation — réanimation hépatologie — MARS/albumine dialyse"

    return MELDResult(meld=meld, meld_na=meld_na, mortality_90d=mort,
                      transplant_priority=prio, recommendation=rec)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CDAI — Crohn's Disease Activity Index (Best 1976)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CDAIResult:
    score: int
    activity: str
    remission: bool
    biologic_indication: bool
    recommendation: str

def compute_cdai(
    stools_per_week: int = 7,           # selles liquides × 2
    abdominal_pain_score: int = 0,      # 0–3/j × 7 jours
    general_wellbeing: int = 0,         # 0–4/j × 7 jours
    extraintestinal: int = 0,           # nb manifestations × 20
    antidiarrheal: bool = False,        # × 30
    abdominal_mass: int = 0,            # 0=absent, 2=douteux, 5=certain
    hematocrit_deficit: float = 0.0,   # (norme-Ht) × 6
    weight_pct_below_normal: float = 0.0,  # × 1
) -> CDAIResult:
    """
    Crohn's Disease Activity Index (Best 1976) — Score 0–600+.
    Rémission < 150 · Légère 150–220 · Modérée 220–450 · Sévère > 450.
    """
    score = (
        stools_per_week * 2 +
        abdominal_pain_score * 5 +
        general_wellbeing * 7 +
        extraintestinal * 20 +
        (30 if antidiarrheal else 0) +
        abdominal_mass +
        hematocrit_deficit * 6 +
        weight_pct_below_normal
    )
    score = int(max(0, score))

    if score < 150:
        activity = "Rémission clinique"
        rem, bio = True, False
        rec = "Maintien traitement — azathioprine/6-MP ou biologiques si rémission sous biologiques"
    elif score < 220:
        activity = "Maladie de Crohn légère"
        rem, bio = False, False
        rec = "Budésonide orale 9 mg/j — mésalazine selon localisation — aminosalicylés"
    elif score < 450:
        activity = "Maladie de Crohn modérée"
        rem, bio = False, True
        rec = "Anti-TNF (infliximab/adalimumab) ou vedolizumab — corticoïdes court terme — nutrition entérale"
    else:
        activity = "Maladie de Crohn sévère"
        rem, bio = False, True
        rec = "Hospitalisation — corticoïdes IV — anti-TNF biologique — chirurgie si complication"

    return CDAIResult(score=score, activity=activity, remission=rem,
                      biologic_indication=bio, recommendation=rec)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Harvey-Bradshaw Index (HBI) — Crohn simplifié (Harvey 1980)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HBIResult:
    score: int
    activity: str
    recommendation: str

def compute_hbi(
    general_wellbeing: int = 0,      # 0=très bien … 4=très mal
    abdominal_pain: int = 0,         # 0=absent … 3=sévère
    abdominal_mass: int = 0,         # 0=absent … 3=certaine
    liquid_stools: int = 0,          # nombre/jour
    complications: int = 0,          # nombre manifestations extrait.
) -> HBIResult:
    score = general_wellbeing + abdominal_pain + abdominal_mass + liquid_stools + complications
    if score < 5:
        return HBIResult(score, "Rémission", "Maintien traitement de fond")
    if score < 8:
        return HBIResult(score, "Légère", "Budésonide orale — réévaluation 4 semaines")
    if score < 16:
        return HBIResult(score, "Modérée", "Biologiques (anti-TNF/vedolizumab) — consultation IBD urgente")
    return HBIResult(score, "Sévère", "Hospitalisation — corticoïdes IV — avis chirurgical si besoin")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Mayo Score — Rectocolite hémorragique (Schroeder 1987 / Mayo Clinic)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MayoResult:
    score: int
    partial_mayo: int
    activity: str
    endoscopic_remission: bool
    biologic_indication: bool
    recommendation: str

def compute_mayo(
    stool_frequency: int = 0,        # 0–3
    rectal_bleeding: int = 0,        # 0–3
    physician_global: int = 0,       # 0–3
    endoscopy_score: int = 0,        # 0–3 (Mayo Endoscopic Subscore)
) -> MayoResult:
    """
    Mayo Score — Score 0–12 (complet) ou 0–9 (partiel sans endoscopie).
    Rémission : score ≤ 2 sans sous-score > 1.
    """
    total   = stool_frequency + rectal_bleeding + physician_global + endoscopy_score
    partial = stool_frequency + rectal_bleeding + physician_global

    if total <= 2 and endoscopy_score <= 1:
        activity = "Rémission"
        rem, bio = True, False
        rec = "Maintien traitement — colonoscopie de surveillance selon protocole"
    elif total <= 5:
        activity = "RCH légère"
        rem, bio = False, False
        rec = "Mésalazine topique + orale — corticoïdes locaux"
    elif total <= 10:
        activity = "RCH modérée"
        rem, bio = False, True
        rec = "Anti-TNF (infliximab) ou vedolizumab ou JAK inhibiteurs (tofacitinib/upadacitinib)"
    else:
        activity = "RCH sévère"
        rem, bio = False, True
        rec = "Hospitalisation — corticoïdes IV — infliximab intensifié — colectomie si échec 5–7 jours"

    return MayoResult(score=total, partial_mayo=partial, activity=activity,
                      endoscopic_remission=rem, biologic_indication=bio, recommendation=rec)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Ranson Score — Pancréatite aiguë (Ranson 1974)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RansonResult:
    score: int
    severity: str
    mortality: str
    icu_indication: bool
    recommendation: str

def compute_ranson(
    age_over_55: bool = False,
    wbc_over_16: bool = False,         # GB > 16 000/mm³
    glucose_over_11: bool = False,     # Glycémie > 11 mmol/L
    ldh_over_350: bool = False,        # LDH > 350 UI/L
    ast_over_250: bool = False,        # ASAT > 250 UI/L
    # À 48h :
    hematocrit_drop_10: bool = False,  # Chute Ht > 10%
    bun_rise_18: bool = False,         # Urée ↑ > 1.8 mmol/L
    calcium_under_2: bool = False,     # Ca²⁺ < 2 mmol/L
    pao2_under_60: bool = False,       # PaO₂ < 60 mmHg
    base_deficit_4: bool = False,      # Déficit base > 4 mEq/L
    fluid_over_6l: bool = False,       # Séquestration liquidienne > 6L
) -> RansonResult:
    """
    Ranson Score — 11 critères (5 à l'admission + 6 à 48h).
    Score 0–2 : légère · 3–4 : modérée · ≥ 5 : sévère (mortalité 25–40%).
    """
    score = sum([
        age_over_55, wbc_over_16, glucose_over_11, ldh_over_350, ast_over_250,
        hematocrit_drop_10, bun_rise_18, calcium_under_2, pao2_under_60,
        base_deficit_4, fluid_over_6l,
    ])

    if score < 3:
        return RansonResult(score, "Légère", "< 1%", False,
                            "Traitement médical — jeûne — SNG si vomissements — renutrition précoce J3")
    if score < 5:
        return RansonResult(score, "Modérée", "10–15%", True,
                            "USIC — antibiotiques si nécrose infectée — CPRE si lithiase — TDM à 72h")
    return RansonResult(score, "Sévère", "25–40%", True,
                        "RÉANIMATION — antibiotiques (imipénem) — nécrosectomie endoscopique/chirurgicale — nutrition parentérale")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BISAP Score — Pancréatite (Singh 2009)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BISAPResult:
    score: int
    severity: str
    mortality: str
    recommendation: str

def compute_bisap(
    bun_over_25: bool = False,         # Urée > 25 mg/dL
    impaired_mental: bool = False,     # Glasgow < 15
    sirs: bool = False,                # Syndrome inflammatoire systémique
    age_over_60: bool = False,
    pleural_effusion: bool = False,
) -> BISAPResult:
    score = sum([bun_over_25, impaired_mental, sirs, age_over_60, pleural_effusion])
    if score < 2:
        return BISAPResult(score, "Légère", "< 1%",
                           "Traitement médical — renutrition entérale précoce")
    if score < 3:
        return BISAPResult(score, "Modérée", "1–5%",
                           "Surveillance rapprochée — TDM à 72h si aggravation")
    return BISAPResult(score, "Sévère", "> 15%",
                       "RÉANIMATION urgente — antibiothérapie — geste endoscopique si indication")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Glasgow-Blatchford Score — Hémorragie digestive haute (Blatchford 2000)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BlatchfordResult:
    score: int
    risk: str
    endoscopy_urgency: str
    transfusion_risk: str
    recommendation: str

def compute_blatchford(
    urea_mmol: float = 5.0,          # Urée sanguine mmol/L
    hemoglobin_g: float = 13.0,      # g/dL
    systolic_bp: int = 120,          # mmHg
    pulse_over_100: bool = False,
    melena: bool = False,
    syncope: bool = False,
    hepatic_disease: bool = False,
    cardiac_failure: bool = False,
    sex_male: bool = True,
) -> BlatchfordResult:
    """
    Glasgow-Blatchford Score — Hémorragie digestive haute (HDH).
    Score 0 = risque très faible (sortie immédiate possible).
    Score ≥ 6 = risque élevé (endoscopie urgente).
    """
    pts = 0
    # Urée
    if urea_mmol >= 6.5:   pts += 2
    if urea_mmol >= 8.0:   pts += 1
    if urea_mmol >= 10.0:  pts += 1
    if urea_mmol >= 25.0:  pts += 2
    # Hémoglobine (homme vs femme)
    if sex_male:
        if hemoglobin_g < 12: pts += 3
        elif hemoglobin_g < 13: pts += 1
    else:
        if hemoglobin_g < 10: pts += 6
        elif hemoglobin_g < 12: pts += 1
    # PAS
    if systolic_bp < 90:   pts += 3
    elif systolic_bp < 100: pts += 2
    elif systolic_bp < 110: pts += 1
    # Autres
    if pulse_over_100:   pts += 1
    if melena:           pts += 1
    if syncope:          pts += 2
    if hepatic_disease:  pts += 2
    if cardiac_failure:  pts += 2

    if pts == 0:
        risk = "Très faible"
        endo = "Électif — peut sortir sans endoscopie immédiate"
        transf = "< 2%"
        rec = "Sortie possible — IPP oral — endoscopie ambulatoire dans les 24h"
    elif pts < 6:
        risk = "Faible à modéré"
        endo = "Endoscopie dans les 24h"
        transf = "5–15%"
        rec = "Hospitalisation — IPP IV — endoscopie urgente < 24h — cross-match sang"
    else:
        risk = "Élevé"
        endo = "Endoscopie URGENTE < 12h"
        transf = "> 30%"
        rec = "RÉANIMATION — transfusion VR si Hb < 7 — endoscopie urgente — clip/sclérose/TIPS"

    return BlatchfordResult(score=pts, risk=risk, endoscopy_urgency=endo,
                            transfusion_risk=transf, recommendation=rec)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Rockall Score — Hémorragie digestive haute (Rockall 1996)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RockallResult:
    pre_score: int
    full_score: int
    rebleed_risk: str
    mortality_risk: str
    recommendation: str

def compute_rockall(
    age: int = 50,
    shock: int = 0,              # 0=pas de choc, 1=FC>100/TA≥100, 2=TA<100
    comorbidity: int = 0,        # 0=aucune, 1=cardio/autre, 2=IR/cirrho/néo
    diagnosis: int = 0,          # 0=Mallory-Weiss/pas de lésion, 1=autre cause, 2=cancer
    stigmata: int = 0,           # 0=aucun/tache noire, 1=sang dans estomac, 2=vaisseau/actif
) -> RockallResult:
    """Rockall Score post-endoscopique — Risque récidive et mortalité HDH."""
    pre = (0 if age < 60 else 1 if age < 80 else 2) + shock + comorbidity
    full = pre + diagnosis + stigmata

    if full <= 2:
        return RockallResult(pre, full, "< 5%", "< 1%",
                             "Sortie précoce possible après endoscopie normale")
    if full <= 4:
        return RockallResult(pre, full, "10–15%", "5–10%",
                             "Hospitalisation — observation 48–72h — IPP IV haute dose")
    return RockallResult(pre, full, "> 20%", "> 15%",
                         "SOINS INTENSIFS — surveillance continue — 2e look endoscopique à 24h — chirurgie si récidive")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Paris Classification — Polypes colorectaux (Paris 2002)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParisResult:
    classification: str
    morphology: str
    cancer_risk: str
    resection_technique: str

def compute_paris(polyp_type: str = "Is") -> ParisResult:
    """
    Classification de Paris — Polypes colorectaux (ESGE/ASGE 2003).
    Guide la technique de résection endoscopique.
    """
    types = {
        "Ip":  ("Pédiculé", "Risque faible < 5%", "Polypectomie à l'anse froide/chaude"),
        "Is":  ("Sessile", "Risque modéré 5–10%", "EMR (mucosectomie) ou polypectomie"),
        "IIa": ("Surélevé plat", "Risque faible 3–5%", "EMR ou ESD si > 20 mm"),
        "IIb": ("Plan", "Risque modéré 5–15%", "ESD (dissection sous-muqueuse) recommandée"),
        "IIc": ("Déprimé", "Risque ÉLEVÉ > 30%", "ESD ou chirurgie — suspicion malignité"),
        "III": ("Excavé ulcéré", "Risque très élevé > 50%", "CHIRURGIE — biopsie urgente"),
    }
    t = types.get(polyp_type, types["Is"])
    return ParisResult(classification=polyp_type, morphology=t[0],
                       cancer_risk=t[1], resection_technique=t[2])


# ═══════════════════════════════════════════════════════════════════════════════
# 11. TNM + AJCC Staging — Cancer colorectal (AJCC 8e édition 2017)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ColorectalTNMResult:
    T: str; N: str; M: str
    stage: str
    five_year_survival: str
    treatment: str

def compute_colorectal_tnm(T: int = 1, N: int = 0, M: int = 0) -> ColorectalTNMResult:
    """
    TNM AJCC 8e — Cancer colorectal.
    T1–T4b · N0–N2b · M0–M1c → Stades I–IV.
    """
    T_str = {0:"Tis",1:"T1",2:"T2",3:"T3",4:"T4a",5:"T4b"}.get(min(T,5),"T?")
    N_str = {0:"N0",1:"N1",2:"N1b",3:"N2a",4:"N2b"}.get(min(N,4),"N?")
    M_str = {0:"M0",1:"M1a",2:"M1b",3:"M1c"}.get(min(M,3),"M?")

    if M > 0:
        stage = "IV" + ["A","B","C"][min(M-1,2)]
        surv5y = ["45%","20%","5%"][min(M-1,2)]
        treat = "Chimiothérapie palliative (FOLFOX/FOLFIRI) ± bevacizumab/cetuximab — soins de support"
    elif N > 2:
        stage = "IIIC"
        surv5y = "28%"
        treat = "Chirurgie curative + chimio adjuvante FOLFOX 6 mois — RCC pelvienne si rectal"
    elif N > 0:
        stage = "IIIA" if T <= 2 else "IIIB"
        surv5y = "70%" if T <= 2 else "45%"
        treat = "Colectomie + curage + chimiothérapie adjuvante CAPOX/FOLFOX"
    elif T >= 4:
        stage = "IIC"
        surv5y = "58%"
        treat = "Résection étendue ± chimiothérapie adjuvante si facteurs risque"
    elif T >= 3:
        stage = "IIA"
        surv5y = "72%"
        treat = "Colectomie avec marges chirurgicales — surveillance CEA + coloscopie 1 an"
    else:
        stage = "I" if T >= 1 else "0"
        surv5y = "93%"
        treat = "Colectomie segmentaire (T1–T2) ou résection endoscopique (Tis/T1 sm)"

    return ColorectalTNMResult(T=T_str, N=N_str, M=M_str, stage=stage,
                               five_year_survival=surv5y, treatment=treat)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Synthèse clinique complète GastroAI
# ═══════════════════════════════════════════════════════════════════════════════

def build_gastro_clinical_summary(
    prediction: str,
    confidence: float,
    img_feats: dict[str, float],
    clinical_params: dict[str, Any],
) -> dict[str, Any]:
    p = clinical_params

    # Child-Pugh (foie/cirrhose)
    cp = compute_child_pugh(
        bilirubin_umol=p.get("bilirubin", 20.0),
        albumin_g=p.get("albumin", 38.0),
        inr=p.get("inr", 1.1),
        ascites=int(img_feats.get("ascites_ratio", 0) * 2),
        encephalopathy=int(img_feats.get("liver_lesion_ratio", 0) * 2),
    )

    # MELD
    meld = compute_meld(
        creatinine_umol=p.get("creatinine", 88.0),
        bilirubin_umol=p.get("bilirubin", 20.0),
        inr=p.get("inr", 1.1),
        sodium_mmol=p.get("sodium", 140.0),
    )

    # CDAI (Crohn)
    cdai = compute_cdai(
        stools_per_week=int(img_feats.get("mucosal_inflammation", 0) * 20),
        abdominal_pain_score=int(img_feats.get("lesion_severity", 0) * 15),
        extraintestinal=int(img_feats.get("lesion_severity", 0) * 2),
    )

    # HBI
    hbi = compute_hbi(
        general_wellbeing=int(img_feats.get("lesion_severity", 0) * 3),
        abdominal_pain=int(img_feats.get("lesion_severity", 0) * 2),
        liquid_stools=int(img_feats.get("mucosal_inflammation", 0) * 5),
    )

    # Mayo (RCH)
    mayo = compute_mayo(
        stool_frequency=min(3, int(img_feats.get("mucosal_inflammation", 0) * 3)),
        rectal_bleeding=min(3, int(img_feats.get("bleeding_ratio", 0) * 3)),
        physician_global=min(3, int(img_feats.get("lesion_severity", 0) * 3)),
        endoscopy_score=min(3, int(img_feats.get("lesion_severity", 0) * 3)),
    )

    # Ranson (pancréatite)
    ranson = compute_ranson(
        age_over_55=p.get("age_over_55", False),
        wbc_over_16=p.get("wbc_over_16", False),
        glucose_over_11=p.get("glucose_over_11", False),
    )

    # BISAP
    bisap = compute_bisap(
        bun_over_25=p.get("bun_over_25", False),
        sirs=img_feats.get("lesion_severity", 0) > 0.5,
        age_over_60=p.get("age_over_60", False),
    )

    # Blatchford (hémorragie)
    blatch = compute_blatchford(
        hemoglobin_g=p.get("hemoglobin", 13.0),
        systolic_bp=p.get("systolic_bp", 120),
        melena=img_feats.get("bleeding_ratio", 0) > 0.05,
        hepatic_disease="cirrhose" in prediction.lower() or "hépatite" in prediction.lower(),
    )

    # Rockall
    rock = compute_rockall(
        age=p.get("age", 50),
        shock=1 if p.get("systolic_bp", 120) < 100 else 0,
    )

    # TNM colorectal
    is_colorectal = any(k in prediction.lower() for k in ("colorectal", "côlon", "polype", "colon"))
    t_val = int(img_feats.get("lesion_severity", 0) * 4) + 1 if is_colorectal else 1
    tnm = compute_colorectal_tnm(T=min(t_val, 4), N=0, M=0)

    # Paris classification
    paris = compute_paris("IIa" if img_feats.get("polyp_ratio", 0) > 0.05 else "Is")

    return {
        "child_pugh": {
            "score": cp.score, "grade": cp.grade,
            "survival_1y": cp.one_year_survival, "survival_2y": cp.two_year_survival,
            "transplant": cp.transplant_indication, "recommendation": cp.recommendation,
        },
        "meld": {
            "meld": meld.meld, "meld_na": meld_na if (meld_na := meld.meld_na) else meld.meld,
            "mortality_90d": meld.mortality_90d,
            "priority": meld.transplant_priority, "recommendation": meld.recommendation,
        },
        "cdai": {
            "score": cdai.score, "activity": cdai.activity,
            "remission": cdai.remission, "biologic": cdai.biologic_indication,
            "recommendation": cdai.recommendation,
        },
        "hbi": {
            "score": hbi.score, "activity": hbi.activity, "recommendation": hbi.recommendation,
        },
        "mayo": {
            "score": mayo.score, "partial": mayo.partial_mayo,
            "activity": mayo.activity, "biologic": mayo.biologic_indication,
            "recommendation": mayo.recommendation,
        },
        "ranson": {
            "score": ranson.score, "severity": ranson.severity,
            "mortality": ranson.mortality, "icu": ranson.icu_indication,
            "recommendation": ranson.recommendation,
        },
        "bisap": {
            "score": bisap.score, "severity": bisap.severity,
            "mortality": bisap.mortality, "recommendation": bisap.recommendation,
        },
        "blatchford": {
            "score": blatch.score, "risk": blatch.risk,
            "endoscopy": blatch.endoscopy_urgency, "transfusion_risk": blatch.transfusion_risk,
            "recommendation": blatch.recommendation,
        },
        "rockall": {
            "pre_score": rock.pre_score, "full_score": rock.full_score,
            "rebleed": rock.rebleed_risk, "mortality": rock.mortality_risk,
            "recommendation": rock.recommendation,
        },
        "tnm_colorectal": {
            "T": tnm.T, "N": tnm.N, "M": tnm.M, "stage": tnm.stage,
            "survival_5y": tnm.five_year_survival, "treatment": tnm.treatment,
        },
        "paris": {
            "classification": paris.classification, "morphology": paris.morphology,
            "cancer_risk": paris.cancer_risk, "resection": paris.resection_technique,
        },
        "overall_urgency": _gastro_urgency(prediction, blatch.score, ranson.score, cp.grade),
    }


def _gastro_urgency(pred: str, blatch: int, ranson: int, cp_grade: str) -> str:
    p = pred.lower()
    if blatch >= 6 or any(k in p for k in ("hémorragie digestive", "varices œsophagiennes", "hématémèse")):
        return "Urgente"
    if ranson >= 5 or "pancréatite" in p and ranson >= 3:
        return "Urgente"
    if any(k in p for k in ("cancer", "carcinome", "cirrhose")):
        return "Élevée"
    if cp_grade == "C":
        return "Urgente"
    if cp_grade == "B":
        return "Élevée"
    if any(k in p for k in ("crohn", "rch", "polype", "barrett")):
        return "Modérée"
    return "Faible"
