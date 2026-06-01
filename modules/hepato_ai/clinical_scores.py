"""
HepatoScan AI — Scores cliniques hépatologiques
================================================
Child-Pugh · MELD · MELD-Na · METAVIR · Ishak · BCLC Staging
LI-RADS · CAP Score · CLIF-C ACLF · ALBI Score · FIB-4 · APRI.

Sources :
  - EASL Clinical Practice Guidelines 2023
  - AASLD Practice Guidance 2023
  - BCLC Barcelona Clinic Liver Cancer 2022
  - LI-RADS v2018 — ACR
  - METAVIR Cooperative Study Group 1994
  - Journal of Hepatology · Hepatology · The Lancet Gastroenterology
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# 1. METAVIR Score — Fibrose hépatique (METAVIR 1994)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class METAVIRResult:
    F: int
    A: int
    label: str
    fibrosis_description: str
    antiviral_indication: bool
    recommendation: str

def compute_metavir(
    fibrosis_grade: int = 0,   # F0–F4
    activity_grade: int = 0,   # A0–A3
) -> METAVIRResult:
    """
    METAVIR Scoring System — Fibrose (F0–F4) + Activité (A0–A3).
    Standard de référence pour biopsie hépatique (VHC, VHB, auto-immune).
    F0=Pas de fibrose · F1=Portale sans septa · F2=Quelques septa · F3=Nombreux septa · F4=Cirrhose.
    """
    F = max(0, min(fibrosis_grade, 4))
    A = max(0, min(activity_grade, 3))

    F_desc = {
        0: "Pas de fibrose — foie normal",
        1: "Fibrose portale sans septa — légère",
        2: "Fibrose portale avec quelques septa — modérée",
        3: "Fibrose avec nombreux septa sans cirrhose — sévère",
        4: "Cirrhose hépatique — stade terminal fibrose",
    }[F]

    antiviral = F >= 2

    if F == 4:
        rec = ("Cirrhose établie — surveillance CHC semestrielle (écho + AFP) — "
               "EOGD varices — bêtabloquants — évaluation transplantation si Child B/C")
    elif F >= 2:
        rec = ("Traitement antiviral recommandé — VHC : sofosbuvir/daclatasvir 12 semaines — "
               "VHB : entécavir/ténofovir à vie — réévaluation METAVIR à 48 semaines")
    elif F == 1:
        rec = ("Traitement à discuter si activité A2–A3 — surveillance biologique annuelle — "
               "éviction alcool et facteurs de progression")
    else:
        rec = "Pas de traitement fibrose — surveiller facteurs risque — bilan hépatique annuel"

    return METAVIRResult(F=F, A=A, label=f"F{F} A{A}", fibrosis_description=F_desc,
                         antiviral_indication=antiviral, recommendation=rec)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FIB-4 Index — Fibrose non invasive (Sterling 2006)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FIB4Result:
    score: float
    fibrosis_stage: str
    cirrhosis_risk: str
    biopsy_needed: bool
    recommendation: str

def compute_fib4(
    age: int = 50,
    asat_iu: float = 40.0,      # UI/L
    alat_iu: float = 40.0,      # UI/L
    platelets_g: float = 200.0, # Giga/L
) -> FIB4Result:
    """
    FIB-4 Index = (Age × ASAT) / (Plaquettes × √ALAT).
    Seuils EASL 2023 : < 1.30 = Faible · 1.30–2.67 = Intermédiaire · > 2.67 = Élevé (F3–F4).
    """
    if alat_iu <= 0 or platelets_g <= 0:
        return FIB4Result(0.0, "Non calculable", "—", False, "Données biologiques incomplètes")

    score = round((age * asat_iu) / (platelets_g * math.sqrt(alat_iu)), 2)

    if score < 1.30:
        return FIB4Result(score, "F0–F1 — Fibrose légère ou absente", "< 5%", False,
                          "Surveillance biologique annuelle — FIB-4 à 3 ans si facteurs risque")
    if score <= 2.67:
        return FIB4Result(score, "F2–F3 — Fibrose modérée à sévère (zone grise)", "10–25%", True,
                          "Élastographie hépatique (FibroScan) recommandée — bilan hépatologie")
    return FIB4Result(score, "F3–F4 — Fibrose avancée / Cirrhose probable", "> 40%", True,
                      "Consultation hépatologie urgente — élastographie — biopsie si doute")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. APRI Score — Fibrose VHC (Wai 2003)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class APRIResult:
    score: float
    interpretation: str
    metavir_estimate: str
    recommendation: str

def compute_apri(
    asat_iu: float = 40.0,
    asat_upper_normal: float = 40.0,
    platelets_g: float = 200.0,
) -> APRIResult:
    """
    APRI (AST-to-Platelet Ratio Index) = (ASAT/ULN × 100) / Plaquettes.
    Seuils : < 0.5 = F0–F1 · 0.5–1.5 = F1–F2 · > 1.5 = F3–F4.
    Outil WHO recommandé pour ressources limitées (Guide WHO HCV 2022).
    """
    if platelets_g <= 0:
        return APRIResult(0.0, "Non calculable", "—", "Données manquantes")
    score = round((asat_iu / asat_upper_normal * 100) / platelets_g, 2)

    if score < 0.5:
        return APRIResult(score, "Fibrose faible probable", "F0–F1",
                          "Surveillance annuelle — pas de traitement fibrose urgent")
    if score < 1.5:
        return APRIResult(score, "Fibrose modérée probable", "F1–F2",
                          "Élastographie ou biopsie hépatique recommandée")
    return APRIResult(score, "Fibrose sévère / Cirrhose probable", "F3–F4",
                      "Hépatologie urgente — traitement antiviral + surveillance CHC")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CAP Score — Stéatose (FibroScan · Echosens)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CAPResult:
    cap_db_m: float
    steatosis_grade: str
    hepatocyte_fat_pct: str
    recommendation: str

def compute_cap(cap_db_m: float = 250.0) -> CAPResult:
    """
    CAP Score (Controlled Attenuation Parameter) — FibroScan Echosens.
    Unité : dB/m · Seuils EASL 2023 : < 248 = S0 · 248–267 = S1 · 268–279 = S2 · ≥ 280 = S3.
    """
    cap = max(100.0, min(cap_db_m, 400.0))
    if cap < 248:
        return CAPResult(cap, "S0 — Pas de stéatose", "< 11%",
                         "Pas de stéatose significative — surveiller facteurs métaboliques")
    if cap < 268:
        return CAPResult(cap, "S1 — Stéatose légère", "11–33%",
                         "Stéatose légère — régime alimentaire + activité physique — réévaluation 1 an")
    if cap < 280:
        return CAPResult(cap, "S2 — Stéatose modérée", "34–66%",
                         "Stéatose modérée — perte poids ≥ 7% — traitement MetS — bilan NASH")
    return CAPResult(cap, "S3 — Stéatose sévère", "> 66%",
                     "Stéatose sévère — risque NASH/fibrose élevé — semaglutide si éligible — biopsie si fibrose avancée")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BCLC Staging — Carcinome hépatocellulaire (BCLC 2022 · EASL)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BCLCResult:
    stage: str
    label: str
    tumor_description: str
    ps_ecog: str
    child_pugh_required: str
    treatment: str
    median_survival: str

def compute_bclc(
    n_tumors: int = 1,
    max_diameter_cm: float = 2.0,
    vascular_invasion: bool = False,
    extrahepatic: bool = False,
    child_pugh_grade: str = "A",
    ecog_ps: int = 0,
) -> BCLCResult:
    """
    BCLC Staging System 2022 (Reig M. et al., Journal of Hepatology 2022).
    Stage 0/A/B/C/D → Traitement curatif → TACE → Systémique → Soins palliatifs.
    """
    cp = child_pugh_grade.upper()

    if cp == "C" or ecog_ps >= 4:
        return BCLCResult("D", "Terminal", "Tout type lésion — Child C ou PS 3–4",
                          "PS 3–4", "Child C",
                          "Soins palliatifs — traitement symptomatique uniquement",
                          "< 3 mois")

    if extrahepatic or vascular_invasion:
        return BCLCResult("C", "Avancé", "Invasion vasculaire OU métastases extra-hépatiques",
                          "PS 0–2", "Child A–B",
                          "Atézolizumab + bevacizumab (IMbrave150) OU sorafenib/lenvatinib",
                          "13–19 mois")

    if n_tumors > 3 or (n_tumors > 1 and max_diameter_cm > 3.0):
        return BCLCResult("B", "Intermédiaire", "Multifocal sans invasion vasculaire",
                          "PS 0", "Child A–B",
                          "TACE (chimioembolisation trans-artérielle) ± ablation",
                          "26–45 mois")

    if n_tumors == 1 and max_diameter_cm <= 2.0 and cp == "A":
        return BCLCResult("0", "Très précoce", "Nodule unique ≤ 2 cm — Child A — PS 0",
                          "PS 0", "Child A",
                          "Résection chirurgicale OU RFA (ablation radiofréquence) OU transplantation",
                          "Survie 5 ans > 70%")

    if n_tumors <= 3 and max_diameter_cm <= 3.0 and cp in ("A", "B"):
        return BCLCResult("A", "Précoce", "1–3 nodules ≤ 3 cm OU nodule unique ≤ 5 cm",
                          "PS 0", "Child A–B",
                          "Résection OU ablation (RFA/MWA) OU transplantation (critères Milan/UCSF)",
                          "Survie 5 ans 50–70%")

    return BCLCResult("A", "Précoce", "1 nodule > 2 cm ou critères Child B",
                      "PS 0", "Child A–B",
                      "Résection OU ablation OU évaluation transplantation",
                      "Survie 5 ans 40–60%")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LI-RADS — Caractérisation nodule hépatique IRM/CT (ACR 2018)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LIRADSResult:
    category: str
    malignancy_probability: str
    hcc_probability: str
    action: str
    major_features: list[str]

def compute_lirads(
    arterial_enhancement: bool = False,
    washout: bool = False,
    enhancing_capsule: bool = False,
    threshold_growth: bool = False,
    diameter_mm: float = 10.0,
    cirrhosis_context: bool = True,
) -> LIRADSResult:
    """
    LI-RADS v2018 — Liver Imaging Reporting and Data System (ACR).
    Applicable uniquement en contexte de cirrhose ou risque CHC élevé.
    LR-1 = Définitivement bénin → LR-5 = Définitivement CHC · LR-M = Probablement malin non-CHC.
    """
    if not cirrhosis_context:
        return LIRADSResult("LR-NC", "Non applicable", "—",
                            "LI-RADS s'applique uniquement aux patients à risque CHC élevé (cirrhose)",
                            [])

    major_features = []
    if arterial_enhancement: major_features.append("Rehaussement artériel non rim-like")
    if washout:              major_features.append("Wash-out portal/tardif")
    if enhancing_capsule:    major_features.append("Capsule rehaussée")
    if threshold_growth:     major_features.append("Croissance ≥ 50% en ≤ 6 mois")

    n_major = len(major_features)

    if n_major >= 2 or (n_major == 1 and diameter_mm >= 20):
        return LIRADSResult("LR-5", "> 95%", "> 95%",
                            "Diagnostic CHC sans biopsie — traitement selon BCLC — TACE/chirurgie/systémique",
                            major_features)
    if n_major >= 1 or diameter_mm >= 20:
        return LIRADSResult("LR-4", "60–94%", "60–94%",
                            "Haute probabilité CHC — multidisciplinaire — biopsie si doute — suivi 3 mois",
                            major_features)
    if diameter_mm >= 10:
        return LIRADSResult("LR-3", "26–59%", "33–60%",
                            "Probabilité intermédiaire — IRM avec contraste 3–6 mois",
                            major_features)
    if diameter_mm >= 5:
        return LIRADSResult("LR-2", "< 26%", "< 33%",
                            "Probablement bénin — suivi IRM 6 mois",
                            major_features)
    return LIRADSResult("LR-1", "< 5%", "< 5%",
                        "Définitivement bénin — suivi standard CHC tous les 6 mois",
                        major_features)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ALBI Score — Fonction hépatique / CHC (Johnson 2015)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ALBIResult:
    score: float
    grade: int
    label: str
    child_pugh_correlation: str
    recommendation: str

def compute_albi(
    albumin_g_l: float = 40.0,
    bilirubin_umol_l: float = 17.0,
) -> ALBIResult:
    """
    ALBI Score (Albumin-Bilirubin) = log10(bilirubin µmol/L × 0.66) + (albumin g/L × -0.085).
    Grade 1 ≤ -2.60 · -2.60 < Grade 2 ≤ -1.39 · Grade 3 > -1.39.
    Standard pour évaluation fonction hépatique CHC (AASLD/EASL 2022).
    """
    score = round(math.log10(bilirubin_umol_l + 0.001) * 0.66 + albumin_g_l * (-0.085), 3)

    if score <= -2.60:
        return ALBIResult(score, 1, "ALBI Grade 1 — Fonction hépatique préservée",
                          "Correspond globalement à Child-Pugh A (bien conservée)",
                          "Résection chirurgicale ou RFA possible — survie favorable")
    if score <= -1.39:
        return ALBIResult(score, 2, "ALBI Grade 2 — Altération modérée",
                          "Correspond Child-Pugh A-B — hétérogène",
                          "TACE ou ablation — surveillance rapprochée — évaluer transplantation")
    return ALBIResult(score, 3, "ALBI Grade 3 — Insuffisance hépatique sévère",
                      "Correspond Child-Pugh B-C",
                      "Soins de support — traitement systémique palliatif — transplantation si éligible")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CLIF-C ACLF Score — Insuffisance hépatique aiguë-sur-chronique
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CLIFACLFResult:
    aclf_grade: int
    label: str
    mortality_28d: str
    organ_failures: list[str]
    recommendation: str

def compute_clif_aclf(
    creatinine_umol: float = 88.0,
    bilirubin_umol: float = 17.0,
    inr: float = 1.1,
    sodium_mmol: float = 140.0,    # noqa: ARG001 — utilisé dans MELD-Na externe
    hepatic_encephalopathy: int = 0,  # grade 0–4
    renal_failure: bool = False,
    respiratory_failure: bool = False,
    circulatory_failure: bool = False,
) -> CLIFACLFResult:
    """
    CLIF-C ACLF (Chronic Liver Failure Consortium ACLF) — Mortalité J28.
    Grades 0–3 selon nombre de défaillances d'organes.
    EASL-CLIF Definition (Moreau et al. Hepatology 2013).
    """
    failures = []
    if creatinine_umol >= 353:   failures.append("Rénale sévère (Cr ≥ 353 µmol/L)")
    elif creatinine_umol >= 176: failures.append("Rénale modérée (Cr ≥ 176 µmol/L)")
    if bilirubin_umol >= 204:    failures.append("Hépatique sévère (Bili ≥ 204 µmol/L)")
    if inr >= 2.5:               failures.append("Coagulopathie (INR ≥ 2.5)")
    if hepatic_encephalopathy >= 3: failures.append(f"Encéphalopathie grade {hepatic_encephalopathy}")
    if renal_failure:            failures.append("Défaillance rénale")
    if respiratory_failure:      failures.append("Défaillance respiratoire")
    if circulatory_failure:      failures.append("Défaillance circulatoire (vasopresseurs)")

    n = len(failures)

    if n == 0:
        return CLIFACLFResult(0, "Pas d'ACLF", "< 5%", failures,
                              "Traitement étiologique — surveillance USIC hépatologie")
    if n == 1:
        return CLIFACLFResult(1, "ACLF Grade 1 — 1 défaillance d'organe", "22%", failures,
                              "USIC hépatologie — traitement précipitant — évaluation transplantation urgente")
    if n == 2:
        return CLIFACLFResult(2, "ACLF Grade 2 — 2 défaillances d'organes", "32%", failures,
                              "Réanimation — support organe — transplantation en urgence si éligible")
    return CLIFACLFResult(3, "ACLF Grade 3 — ≥ 3 défaillances d'organes", "> 73%", failures,
                          "Réanimation intensive — pronostic très sombre — évaluation transplantation super-urgence")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Child-Pugh (repris de GastroAI pour cohérence HepatoScan)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChildPughHResult:
    score: int; grade: str
    one_year_survival: str; transplant: bool; recommendation: str

def compute_child_pugh_h(
    bilirubin_umol: float = 17.0, albumin_g: float = 40.0,
    inr: float = 1.0, ascites: int = 0, encephalopathy: int = 0,
) -> ChildPughHResult:
    pts = 0
    pts += 1 if bilirubin_umol < 34 else 2 if bilirubin_umol < 51 else 3
    pts += 1 if albumin_g > 35 else 2 if albumin_g >= 28 else 3
    pts += 1 if inr < 1.7 else 2 if inr < 2.3 else 3
    pts += [1,2,3][min(ascites,2)]; pts += [1,2,3][min(encephalopathy,2)]
    if pts <= 6:
        return ChildPughHResult(pts,"A","100%",False,
                                "Surveillance CHC — bêtabloquants si varices — traitement étiologique")
    if pts <= 9:
        return ChildPughHResult(pts,"B","81%",True,
                                "Évaluation transplantation — diurétiques — prophylaxie bactérienne")
    return ChildPughHResult(pts,"C","45%",True,
                            "Transplantation urgente — soins palliatifs si CI — TIPS si réfractaire")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MELD Score (repris pour contexte hépatologique complet)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MELDHResult:
    meld: float; meld_na: float
    mortality_90d: str; priority: str; recommendation: str

def compute_meld_h(
    creatinine_umol: float = 88.0, bilirubin_umol: float = 17.0,
    inr: float = 1.0, sodium_mmol: float = 140.0, dialysis: bool = False,
) -> MELDHResult:
    creat = max(1.0, creatinine_umol/88.4); bili = max(1.0, bilirubin_umol/17.1)
    inr_v = max(1.0, inr)
    if dialysis: creat = 4.0
    meld = round(3.78*math.log(bili) + 11.2*math.log(inr_v) + 9.57*math.log(creat) + 6.43, 1)
    meld = max(6.0, min(meld, 40.0))
    na   = max(125.0, min(sodium_mmol, 140.0))
    meld_na = round(meld - na - (0.025*meld*(140-na)) + 140, 1)
    meld_na = max(6.0, min(meld_na, 40.0))
    if meld < 10:   mort,prio,rec = "< 2%","Faible","Ambulatoire — suivi trimestriel"
    elif meld < 20: mort,prio,rec = "6–20%","Modérée","Évaluation transplantation — hépatologie"
    elif meld < 30: mort,prio,rec = "20–50%","Élevée — urgence","Inscription urgente transplantation"
    else:           mort,prio,rec = "> 50%","Super-urgence","RÉANIMATION — transplantation super-urgence"
    return MELDHResult(meld=meld, meld_na=meld_na, mortality_90d=mort, priority=prio, recommendation=rec)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Synthèse clinique globale HepatoScan
# ═══════════════════════════════════════════════════════════════════════════════

def build_hepato_clinical_summary(
    prediction: str,
    confidence: float,    # noqa: ARG001 — réservé pour seuil de confiance futur
    img_feats: dict[str, float],
    clinical_params: dict[str, Any],
) -> dict[str, Any]:
    p = clinical_params
    pred_lower = prediction.lower()

    # METAVIR
    fib_grade = int(img_feats.get("fibrosis_index", 0) * 4)
    act_grade = int(img_feats.get("liver_texture", 0.1) * 3)
    metavir = compute_metavir(fib_grade, act_grade)

    # FIB-4
    fib4 = compute_fib4(
        age=p.get("age", 50),
        asat_iu=p.get("asat", 40.0), alat_iu=p.get("alat", 40.0),
        platelets_g=p.get("platelets", 200.0),
    )

    # APRI
    apri = compute_apri(asat_iu=p.get("asat",40.0), asat_upper_normal=40.0,
                        platelets_g=p.get("platelets",200.0))

    # CAP Score (stéatose)
    cap_val = p.get("cap_score") or (200 + img_feats.get("hepatic_brightness", 0) * 200)
    cap = compute_cap(cap_val)

    # BCLC
    bclc = compute_bclc(
        n_tumors=int(img_feats.get("focal_count", 0) + 1),
        max_diameter_cm=img_feats.get("lesion_size_cm", 2.0),
        vascular_invasion=img_feats.get("vascular_invasion", 0) > 0.3,
        extrahepatic=p.get("extrahepatic", False),
        child_pugh_grade=p.get("child_pugh_grade") or "A",
        ecog_ps=p.get("ecog_ps", 0),
    )

    # LI-RADS
    lirads = compute_lirads(
        arterial_enhancement=img_feats.get("arterial_enhancement", 0) > 0.3,
        washout=img_feats.get("washout_ratio", 0) > 0.2,
        enhancing_capsule=img_feats.get("capsule_ratio", 0) > 0.2,
        threshold_growth=p.get("threshold_growth", False),
        diameter_mm=img_feats.get("lesion_size_cm", 1.0) * 10,
        cirrhosis_context=any(k in pred_lower for k in ("cirrhose","hépatite","stéatose","nash")),
    )

    # ALBI
    albi = compute_albi(albumin_g_l=p.get("albumin",40.0), bilirubin_umol_l=p.get("bilirubin",17.0))

    # Child-Pugh
    cp = compute_child_pugh_h(
        bilirubin_umol=p.get("bilirubin",17.0), albumin_g=p.get("albumin",40.0),
        inr=p.get("inr",1.0), ascites=int(img_feats.get("ascites_ratio",0)*2),
        encephalopathy=p.get("encephalopathy",0),
    )

    # MELD
    meld = compute_meld_h(
        creatinine_umol=p.get("creatinine",88.0), bilirubin_umol=p.get("bilirubin",17.0),
        inr=p.get("inr",1.0), sodium_mmol=p.get("sodium",140.0),
    )

    # CLIF-C ACLF
    clif = compute_clif_aclf(
        creatinine_umol=p.get("creatinine",88.0), bilirubin_umol=p.get("bilirubin",17.0),
        inr=p.get("inr",1.0), sodium_mmol=p.get("sodium",140.0),
        hepatic_encephalopathy=p.get("encephalopathy",0),
    )

    return {
        "metavir":  {"F":metavir.F,"A":metavir.A,"label":metavir.label,
                     "description":metavir.fibrosis_description,
                     "antiviral":metavir.antiviral_indication,"recommendation":metavir.recommendation},
        "fib4":     {"score":fib4.score,"stage":fib4.fibrosis_stage,
                     "cirrhosis_risk":fib4.cirrhosis_risk,"biopsy":fib4.biopsy_needed,
                     "recommendation":fib4.recommendation},
        "apri":     {"score":apri.score,"interpretation":apri.interpretation,
                     "metavir_estimate":apri.metavir_estimate,"recommendation":apri.recommendation},
        "cap":      {"score":cap.cap_db_m,"grade":cap.steatosis_grade,
                     "fat_pct":cap.hepatocyte_fat_pct,"recommendation":cap.recommendation},
        "bclc":     {"stage":bclc.stage,"label":bclc.label,
                     "tumor_desc":bclc.tumor_description,"treatment":bclc.treatment,
                     "median_survival":bclc.median_survival},
        "lirads":   {"category":lirads.category,"malignancy":lirads.malignancy_probability,
                     "hcc_prob":lirads.hcc_probability,"action":lirads.action,
                     "major_features":lirads.major_features},
        "albi":     {"score":albi.score,"grade":albi.grade,"label":albi.label,
                     "cp_correlation":albi.child_pugh_correlation,"recommendation":albi.recommendation},
        "child_pugh":{"score":cp.score,"grade":cp.grade,
                      "survival_1y":cp.one_year_survival,"transplant":cp.transplant,
                      "recommendation":cp.recommendation},
        "meld":     {"meld":meld.meld,"meld_na":meld.meld_na,
                     "mortality_90d":meld.mortality_90d,"priority":meld.priority,
                     "recommendation":meld.recommendation},
        "clif_aclf":{"grade":clif.aclf_grade,"label":clif.label,
                     "mortality_28d":clif.mortality_28d,"organ_failures":clif.organ_failures,
                     "recommendation":clif.recommendation},
        "overall_urgency": _hepato_urgency(prediction, cp.grade, meld.meld, bclc.stage, clif.aclf_grade),
    }


def _hepato_urgency(pred: str, cp_grade: str, meld: float, bclc: str, clif: int) -> str:
    p = pred.lower()
    if clif >= 2 or meld >= 25:                        return "Urgente"
    if any(k in p for k in ("aclf","insuffisance hépatique aiguë","thrombose portale")): return "Urgente"
    if any(k in p for k in ("carcinome","chc","cholangiocarcinome","métastase")): return "Élevée"
    if cp_grade == "C" or bclc in ("C","D"):           return "Élevée"
    if cp_grade == "B" or bclc == "B":                 return "Modérée"
    if any(k in p for k in ("cirrhose","fibrose f3","nash","hépatite")): return "Modérée"
    return "Faible"
