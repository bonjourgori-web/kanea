"""
HistoPath AI — Scores anatomopathologiques de référence
========================================================
Nottingham Grade (Elston-Ellis) · Gleason/ISUP · TNM Pathologique (AJCC 8e)
Ki-67 Index · HER2 Score · Récepteurs Hormonaux (ER/PR)
Tumor Budding (ITBCC 2016) · Edmondson-Steiner (HCC)
CIN Grading / Bethesda System · Sous-type Moléculaire (St. Gallen 2021)

Sources :
  - College of American Pathologists (CAP) Protocols 2023
  - WHO Classification of Tumours, 5th ed. (2022)
  - AJCC Cancer Staging Manual, 8th ed. (2017)
  - ESMO/ASCO/NCCN Clinical Practice Guidelines 2023
  - Elston & Ellis, Histopathology 1991 (Nottingham Grade)
  - ISUP Consensus Conference 2014 (Gleason)
  - ITBCC 2016 (Tumor Budding — Lugli et al., Mod Pathol 2017)
  - Bethesda System 2014 (Cervical Cytology)
  - Edmondson & Steiner, Cancer 1954 (HCC Grading)
  - St. Gallen Expert Consensus 2021 (Molecular Subtypes)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 1. NOTTINGHAM HISTOLOGIC GRADE — Elston-Ellis (Cancer du sein)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NottinghamResult:
    tubule_score: int
    nuclear_score: int
    mitosis_score: int
    total_score: int
    grade: str
    grade_number: int
    five_year_survival: str
    ten_year_survival: str
    recurrence_risk: str
    interpretation: str
    recommendation: str


def compute_nottingham_grade(
    tubule_formation: int = 2,      # 1=>75%, 2=10–75%, 3=<10% tubules
    nuclear_pleomorphism: int = 2,  # 1=uniforme, 2=modéré, 3=marqué
    mitotic_count: int = 2,         # 1=faible, 2=modéré, 3=élevé (par 10 HPF)
) -> NottinghamResult:
    """
    Nottingham Histologic Grade selon Elston & Ellis (Histopathology 1991).
    Score 3–9 → Grade I (3–5), Grade II (6–7), Grade III (8–9).
    Standard CAP/WHO pour tous les carcinomes mammaires infiltrants.
    """
    t = max(1, min(3, int(tubule_formation)))
    n = max(1, min(3, int(nuclear_pleomorphism)))
    m = max(1, min(3, int(mitotic_count)))
    total = t + n + m

    if total <= 5:
        grade_num = 1
        grade     = "Grade I — Bien différencié"
        surv5     = "88–92%"
        surv10    = "75–82%"
        recurrence = "Faible (10–15%)"
        interp     = (
            "Tumeur bien différenciée (score {}/9). Cellules tumorales proches du tissu "
            "mammaire normal. Formation tubulaire > 75%. Faible activité mitotique. "
            "Excellent pronostic avec traitement standard.".format(total)
        )
        reco = (
            "Chirurgie conservatrice (tumorectomie) + radiothérapie adjuvante. "
            "Hormonothérapie 5–10 ans si HR+. Chimiothérapie non recommandée sauf Ki-67 > 30%."
        )
    elif total <= 7:
        grade_num = 2
        grade     = "Grade II — Modérément différencié"
        surv5     = "72–82%"
        surv10    = "58–68%"
        recurrence = "Intermédiaire (20–30%)"
        interp     = (
            "Différenciation intermédiaire (score {}/9). Pronostic modéré dépendant du "
            "statut HR/HER2. Recommander test génomique (Oncotype DX / Mammaprint) "
            "pour décision chimiothérapie si HR+ / HER2−.".format(total)
        )
        reco = (
            "Chirurgie + radiothérapie ± chimiothérapie selon Oncotype DX. "
            "Si Oncotype DX ≥ 26 : chimiothérapie + hormonothérapie. "
            "Protocole AC-T ou FEC-D si chimiothérapie indiquée."
        )
    else:
        grade_num = 3
        grade     = "Grade III — Peu différencié"
        surv5     = "52–65%"
        surv10    = "40–52%"
        recurrence = "Élevé (35–50%)"
        interp     = (
            "Tumeur agressive peu différenciée (score {}/9). Forte activité mitotique, "
            "pléomorphisme nucléaire marqué. Risque de rechute élevé. "
            "Corrélé avec Triple Négatif, HER2+ et Luminal B.".format(total)
        )
        reco = (
            "Chimiothérapie néoadjuvante recommandée (AC-T ou FEC-D). "
            "Si HER2+ : trastuzumab + pertuzumab. Si TNBC : pembrolizumab si PD-L1+. "
            "RCP oncologique obligatoire avant décision thérapeutique."
        )

    return NottinghamResult(
        tubule_score=t, nuclear_score=n, mitosis_score=m,
        total_score=total, grade=grade, grade_number=grade_num,
        five_year_survival=surv5, ten_year_survival=surv10,
        recurrence_risk=recurrence, interpretation=interp, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. KI-67 INDEX — Marqueur de prolifération cellulaire
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Ki67Result:
    percent: float
    category: str
    proliferation_rate: str
    prognosis_impact: str
    luminal_hint: str
    recommendation: str


def compute_ki67(ki67_percent: float = 20.0) -> Ki67Result:
    """
    Ki-67 Index — Anticorps MIB-1, marqueur de prolifération cellulaire.
    Seuils selon consensus St. Gallen 2021 et ESMO Breast Cancer Guidelines 2023.
    Applicable : sein, lymphomes, tumeurs neuroendocrines, sarcomes.
    """
    pct = max(0.0, min(100.0, float(ki67_percent)))

    if pct < 14:
        cat   = "Faible (< 14%)"
        prolif = "Faible prolifération cellulaire"
        prog   = "Favorable — Tumeur à croissance lente (Luminal A probable)"
        hint   = "Compatible Luminal A — hormonothérapie seule probable"
        reco   = (
            "Hormonothérapie seule (tamoxifène 5 ans ou inhibiteur d'aromatase 5–10 ans). "
            "Chimiothérapie non recommandée si Oncotype DX < 11."
        )
    elif pct < 30:
        cat   = "Intermédiaire (14–30%)"
        prolif = "Prolifération modérée"
        prog   = "Intermédiaire — Décision chimiothérapie selon test génomique"
        hint   = "Luminal B possible — Oncotype DX recommandé"
        reco   = (
            "Discussion RCP. Oncotype DX ou Mammaprint pour décision chimiothérapie. "
            "Si score RS 11–25 : hormonothérapie ± chimiothérapie selon âge et ménopause."
        )
    else:
        cat   = "Élevé (≥ 30%)"
        prolif = "Forte prolifération — Tumeur agressive"
        prog   = "Défavorable — Corrélé Grade III, TNBC, HER2+, Luminal B"
        hint   = "Luminal B HER2− ou Triple Négatif — chimiothérapie indiquée"
        reco   = (
            "Chimiothérapie adjuvante recommandée (AC-T, FEC-D ou TC). "
            "Anti-HER2 si HER2+. Capécitabine adjuvante si résidu post-NAC (TNBC)."
        )

    return Ki67Result(
        percent=pct, category=cat, proliferation_rate=prolif,
        prognosis_impact=prog, luminal_hint=hint, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HER2 SCORE — Récepteur HER2/neu (ASCO/CAP 2018)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HER2Result:
    ihc_score: str
    fish_ratio: Optional[float]
    status: str
    amplification: bool
    her2_low: bool
    treatment_eligible: bool
    interpretation: str
    recommendation: str


def compute_her2(
    ihc_score: str = "2+",
    fish_ratio: float | None = None,
) -> HER2Result:
    """
    HER2 Score selon ASCO/CAP Guidelines 2018 (Wolff et al., JCO 2018).
    IHC 3+ = positif. IHC 2+ = équivoque → FISH/ISH obligatoire.
    HER2-Low (IHC 1+ ou 2+/FISH−) : éligible T-DXd (DESTINY-Breast04).
    """
    ihc   = str(ihc_score).strip().upper().replace(" ", "")
    amp   = False
    low   = False
    elig  = False

    if ihc == "3+":
        status = "HER2 Positif — Surexpression forte"
        amp    = True
        elig   = True
        interp = (
            "Surexpression HER2 forte et homogène (IHC 3+). "
            "Amplification confirmée sans nécessité de FISH selon ASCO/CAP 2018. "
            "> 10% de cellules tumorales avec marquage membranaire complet intense."
        )
        reco = (
            "Double blocage HER2 : trastuzumab (Herceptin) + pertuzumab (Perjeta). "
            "Protocole TCHP (néoadjuvant) ou AC-THP (adjuvant). "
            "T-DM1 (Kadcyla) si résidu tumoral post-NAC."
        )

    elif ihc == "2+":
        if fish_ratio is not None:
            if fish_ratio >= 2.0:
                status = "HER2 Positif — Amplification FISH confirmée"
                amp    = True
                elig   = True
                interp = (
                    f"IHC 2+ équivoque confirmée positif par FISH : "
                    f"ratio HER2/CEP17 = {fish_ratio:.2f} ≥ 2.0. "
                    "Amplification génique significative."
                )
                reco = (
                    "Trastuzumab + pertuzumab (double blocage HER2). "
                    "Protocole selon stade : TCHP néoadjuvant ou AC-THP adjuvant."
                )
            else:
                low    = True
                status = "HER2-Low (IHC 2+ / FISH négatif)"
                interp = (
                    f"IHC 2+ mais FISH négatif (ratio = {fish_ratio:.2f} < 2.0). "
                    "Statut HER2-Low — éligible trastuzumab déruxtécan (T-DXd)."
                )
                reco = (
                    "Pas de double blocage HER2 standard. "
                    "En métastatique : T-DXd (Enhertu) selon DESTINY-Breast04. "
                    "Traitement selon HR (hormonothérapie si HR+)."
                )
        else:
            status = "HER2 Équivoque — FISH requis"
            interp = (
                "IHC 2+ : expression HER2 intermédiaire équivoque. "
                "FISH/ISH obligatoire selon ASCO/CAP 2018 avant décision thérapeutique."
            )
            reco = (
                "Demander FISH HER2/CEP17 en urgence. "
                "Ne pas initier anti-HER2 avant confirmation FISH."
            )

    elif ihc == "1+":
        low    = True
        status = "HER2-Low (IHC 1+)"
        interp = (
            "Expression HER2 faible (HER2-Low IHC 1+). "
            "Marquage membranaire incomplet ou faible dans > 10% des cellules. "
            "Éligible T-DXd selon DESTINY-Breast04 (Nelson et al., NEJM 2022)."
        )
        reco = (
            "Pas d'anti-HER2 standard. Traitement selon HR. "
            "En rechute métastatique HR+/HER2-Low : T-DXd (Enhertu) recommandé."
        )

    else:  # "0"
        status = "HER2 Négatif (IHC 0)"
        interp = (
            "Absence totale d'expression HER2 (IHC 0). "
            "Pas d'amplification. Traitement anti-HER2 non indiqué."
        )
        reco = (
            "Traitement selon HR et Ki-67. "
            "Hormonothérapie si HR+. Chimiothérapie si TNBC (Grade III, Ki-67 élevé)."
        )

    return HER2Result(
        ihc_score=ihc, fish_ratio=fish_ratio, status=status,
        amplification=amp, her2_low=low, treatment_eligible=elig,
        interpretation=interp, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RÉCEPTEURS HORMONAUX — ER / PR (ASCO/CAP 2020)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HormoneReceptorResult:
    er_percent: float
    pr_percent: float
    er_status: str
    pr_status: str
    hr_status: str
    allred_hint: str
    luminal_subtype_hint: str
    recommendation: str


def compute_hormone_receptors(
    er_percent: float = 75.0,
    pr_percent: float = 50.0,
) -> HormoneReceptorResult:
    """
    Récepteurs hormonaux ER/PR selon ASCO/CAP Guidelines 2020.
    Seuil positivité : ≥ 1% de cellules immunoréactives (Hammond et al., JCO 2010).
    Contribue à la classification moléculaire (Luminal A/B) et décision thérapeutique.
    """
    er_pos = float(er_percent) >= 1.0
    pr_pos = float(pr_percent) >= 1.0
    hr_pos = er_pos or pr_pos

    er_status  = f"ER {'Positif' if er_pos else 'Négatif'} ({er_percent:.0f}%)"
    pr_status  = f"PR {'Positif' if pr_pos else 'Négatif'} ({pr_percent:.0f}%)"
    hr_status  = "HR Positif (HR+)" if hr_pos else "HR Négatif (HR−)"

    # Allred-like interpretation
    if er_percent >= 50:
        allred = "Expression ER forte (Allred probable ≥ 6/8) — excellent marqueur"
    elif er_percent >= 10:
        allred = "Expression ER modérée (Allred 4–5/8)"
    elif er_percent >= 1:
        allred = "Expression ER faible (Allred 3/8) — décision RCP"
    else:
        allred = "ER négatif — hormonothérapie non indiquée"

    if er_pos and pr_pos and er_percent >= 50:
        subtype = "Luminal A probable (HR++, double positivité forte)"
        reco = (
            "Hormonothérapie 5–10 ans (tamoxifène ou IA selon statut ménopausique). "
            "Chimiothérapie rarement indiquée sauf Grade III / Ki-67 élevé."
        )
    elif er_pos and pr_pos and er_percent < 50:
        subtype = "Luminal B possible (HR+, expression modérée)"
        reco = (
            "Hormonothérapie + évaluation chimiothérapie (Oncotype DX). "
            "Chimiothérapie adjuvante si RS ≥ 26 ou Ki-67 > 30%."
        )
    elif er_pos and not pr_pos:
        subtype = "Luminal B probable (ER+ PR−) — profil moins favorable"
        reco = (
            "Hormonothérapie recommandée + discussion chimiothérapie adjuvante. "
            "ER+/PR− associé à résistance hormonothérapie plus fréquente."
        )
    else:
        subtype = "HR Négatif — sous-type non Luminal (HER2+ ou TNBC)"
        reco = (
            "Pas d'hormonothérapie. "
            "Traitement selon statut HER2 (anti-HER2) ou TNBC (chimiothérapie ± immunothérapie)."
        )

    return HormoneReceptorResult(
        er_percent=float(er_percent), pr_percent=float(pr_percent),
        er_status=er_status, pr_status=pr_status, hr_status=hr_status,
        allred_hint=allred, luminal_subtype_hint=subtype, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GLEASON SCORE + ISUP GRADE GROUP — Adénocarcinome prostatique
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GleasonResult:
    primary: int
    secondary: int
    tertiary: Optional[int]
    gleason_score: int
    gleason_pattern: str
    isup_grade: int
    isup_label: str
    risk_category: str
    five_year_bfs: str
    biochemical_recurrence_risk: str
    interpretation: str
    recommendation: str


def compute_gleason(
    primary: int = 3,
    secondary: int = 4,
    tertiary: int | None = None,
) -> GleasonResult:
    """
    Gleason Score + ISUP Grade Group (Epstein et al., Eur Urol 2016).
    ISUP Consensus 2014 modifie le Gleason classique en 5 Grade Groups.
    Patterns 1–5 (1=bien différencié, 5=indifférencié).
    """
    p = max(1, min(5, int(primary)))
    s = max(1, min(5, int(secondary)))
    t = max(1, min(5, int(tertiary))) if tertiary is not None else None

    gleason  = p + s
    pattern  = f"{p}+{s}" + (f"+{t}" if t else "")

    if gleason <= 6:
        isup, label = 1, "Grade Group 1"
        risk   = "Très faible / Faible risque"
        bfs    = "> 95% à 5 ans"
        rec_risk = "< 15% à 5 ans"
        interp = (
            f"Gleason {p}+{s} = {gleason} (ISUP 1). Tumeur bien différenciée, "
            "glandes régulières uniformes. Comportement indolent. "
            "Surveillance active envisageable pour tumeur cT1–2a, PSA < 10."
        )
        reco = (
            "Surveillance active (PSA + TR 6 mois, biopsie 1–2 ans) si cT1–2a. "
            "Prostatectomie radicale ou RT si traitement actif préféré."
        )
    elif gleason == 7 and p == 3:
        isup, label = 2, "Grade Group 2"
        risk   = "Risque intermédiaire favorable"
        bfs    = "85–92% à 5 ans"
        rec_risk = "15–25% à 5 ans"
        interp = (
            f"Gleason {p}+{s} = {gleason} (ISUP 2). Prédominance bien formée (3). "
            "< 50% biopsies Gleason 4. Pronostic intermédiaire favorable."
        )
        reco = (
            "Prostatectomie radicale laparoscopique/robotique ou radiothérapie externe ± curiethérapie. "
            "Hormonothérapie courte 4–6 mois non systématique."
        )
    elif gleason == 7 and p == 4:
        isup, label = 3, "Grade Group 3"
        risk   = "Risque intermédiaire défavorable"
        bfs    = "72–82% à 5 ans"
        rec_risk = "25–40% à 5 ans"
        interp = (
            f"Gleason {p}+{s} = {gleason} (ISUP 3). Prédominance mal formée (4). "
            "≥ 50% biopsies Gleason 4. Pronostic intermédiaire défavorable."
        )
        reco = (
            "Prostatectomie radicale ou radiothérapie externe + hormonothérapie courte (6 mois). "
            "Curiethérapie à haut débit de dose en option."
        )
    elif gleason == 8:
        isup, label = 4, "Grade Group 4"
        risk   = "Haut risque"
        bfs    = "58–70% à 5 ans"
        rec_risk = "40–55% à 5 ans"
        interp = (
            f"Gleason {p}+{s} = {gleason} (ISUP 4). Tumeur de haut grade. "
            "Risque significatif de récidive biochimique et d'extension extra-prostatique."
        )
        reco = (
            "Radiothérapie externe + hormonothérapie longue (18–36 mois). "
            "Apalutamide/enzalutamide en association. "
            "Prostatectomie radicale si cT2–3a sans envahissement vésical."
        )
    else:
        isup, label = 5, "Grade Group 5"
        risk   = "Très haut risque"
        bfs    = "< 50% à 5 ans"
        rec_risk = "> 60% à 5 ans"
        interp = (
            f"Gleason {p}+{s} = {gleason} (ISUP 5). Tumeur très agressive, peu différenciée. "
            "Risque métastatique élevé. Criblage cellulaire, nécrose cométale."
        )
        reco = (
            "Hormonothérapie longue (36 mois) + RT intensifiée ± chimiothérapie (docétaxel). "
            "Abiratérone ou enzalutamide. PSMA PET-TDM pour staging métastatique."
        )

    return GleasonResult(
        primary=p, secondary=s, tertiary=t,
        gleason_score=gleason, gleason_pattern=pattern,
        isup_grade=isup, isup_label=label + f" — Gleason {gleason}",
        risk_category=risk, five_year_bfs=bfs,
        biochemical_recurrence_risk=rec_risk,
        interpretation=interp, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TNM PATHOLOGIQUE — AJCC 8e édition (multi-cancer)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TNMResult:
    cancer_type: str
    pt: str
    pn: str
    pm: str
    stage: str
    stage_roman: str
    five_year_survival: str
    r_status: str
    interpretation: str
    recommendation: str


def compute_tnm_pathological(
    cancer_type: str = "breast",
    pt: str = "T2",
    pn: str = "N0",
    pm: str = "M0",
    r_status: str = "R0",
    grade: int = 2,
) -> TNMResult:
    """
    TNM Pathologique selon AJCC Cancer Staging Manual 8e édition (2017).
    Supporte : breast, colorectal, prostate, lung, cervical, gastric, liver, general.
    """
    ct = cancer_type.lower()

    pt_clean = pt.upper().replace("P", "").strip()
    pn_clean = pn.upper().replace("P", "").strip()
    pm_clean = pm.upper().replace("P", "").strip()

    pt_num = 0
    for ch in pt_clean:
        if ch.isdigit():
            pt_num = int(ch)
            break

    pn_num = 0
    for ch in pn_clean:
        if ch.isdigit():
            pn_num = int(ch)
            break

    pm_pos = "1" in pm_clean or pm_clean.endswith("A") or pm_clean.endswith("B") or pm_clean.endswith("C")

    if pm_pos:
        stage, roman = "Stage IV", "IV"
        surv  = "< 25%"
        interp = (
            f"Maladie métastatique ({pt}, {pn}, {pm}). "
            "Prise en charge palliative. Thérapies ciblées selon profil moléculaire."
        )
        reco = (
            "RCP oncologique urgente. Chimiothérapie palliative ± thérapies ciblées. "
            "Immunothérapie si PD-L1+. Soins palliatifs précoces."
        )
    elif pn_num >= 3:
        stage, roman = "Stage IIIC", "IIIC"
        surv  = "44–58%"
        interp = (
            f"Envahissement ganglionnaire massif ({pn}). "
            "Risque élevé de dissémination systémique. Traitement systémique intensif requis."
        )
        reco = (
            "Chimiothérapie adjuvante intensive + radiothérapie. "
            "Thérapies ciblées selon profil moléculaire (HER2, EGFR, MSI-H, BRCA)."
        )
    elif pn_num >= 2:
        stage, roman = "Stage IIIB", "IIIB"
        surv  = "55–65%"
        interp = (
            f"Atteinte ganglionnaire significative ({pn}, {pt}). "
            "Traitement systémique adjuvant indispensable."
        )
        reco = (
            "Chimiothérapie adjuvante + radiothérapie selon cancer primaire. "
            "Hormonothérapie adjuvante si HR+ (sein). Thérapies ciblées si applicable."
        )
    elif pn_num == 1:
        stage, roman = "Stage IIIA", "IIIA"
        surv  = "65–83%"
        interp = (
            f"Envahissement ganglionnaire ({pn}) avec tumeur {pt}. "
            "Traitement systémique adjuvant recommandé pour réduire le risque de rechute."
        )
        reco = (
            "Chimiothérapie adjuvante ± radiothérapie. "
            "Durée hormonothérapie 10 ans si sein HR+. FOLFOX si colorectal N+."
        )
    elif pt_num >= 4:
        stage, roman = "Stage IIB", "IIB"
        surv  = "65–75%"
        interp = (
            f"Tumeur localement avancée ({pt}), sans envahissement ganglionnaire. "
            "Résection chirurgicale large recommandée."
        )
        reco = (
            "Chirurgie curative (R0 obligatoire) + radiothérapie adjuvante si marges proches. "
            "Chimiothérapie selon grade et facteurs de risque moléculaires."
        )
    elif pt_num >= 3:
        stage, roman = "Stage IIA", "IIA"
        surv  = "78–88%"
        interp = (
            f"Tumeur de taille significative ({pt}), ganglions négatifs ({pn}). "
            "Bon pronostic avec traitement adapté. Chirurgie curative recommandée."
        )
        reco = (
            "Chirurgie conservatrice si possible ± radiothérapie adjuvante. "
            "Chimiothérapie adjuvante selon grade et profil moléculaire."
        )
    elif pt_num == 2:
        stage, roman = "Stage IB", "IB"
        surv  = "85–93%"
        interp = (
            f"Tumeur {pt}, sans envahissement ganglionnaire. Pronostic favorable."
        )
        reco = (
            "Chirurgie standard (tumorectomie ou résection segmentaire). "
            "Radiothérapie adjuvante si sein ou col utérin."
        )
    else:
        stage, roman = "Stage IA", "IA"
        surv  = "92–99%"
        interp = (
            f"Tumeur précoce {pt}, {pn}, {pm}. Excellent pronostic. "
            "Résection chirurgicale curative recommandée."
        )
        reco = (
            "Chirurgie curative conservatrice (résection endoscopique si applicable). "
            "Surveillance rapprochée post-opératoire. Radiothérapie rarement requise."
        )

    r_note = ""
    if r_status == "R1":
        r_note = " Résection R1 (marges microscopiques atteintes) — ré-excision recommandée."
    elif r_status == "R2":
        r_note = " Résection R2 (résidu macroscopique) — chirurgie de complétion ou RT."

    return TNMResult(
        cancer_type=ct, pt=pt, pn=pn, pm=pm,
        stage=stage, stage_roman=roman,
        five_year_survival=surv,
        r_status=r_status,
        interpretation=interp + r_note,
        recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TUMOR BUDDING SCORE — ITBCC 2016 (Cancer colorectal)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TumorBuddingResult:
    bud_count: int
    grade: str
    grade_number: int
    hotspot_area: str
    metastasis_risk: str
    lymph_node_risk: str
    interpretation: str
    recommendation: str


def compute_tumor_budding(bud_count: int = 3) -> TumorBuddingResult:
    """
    Tumor Budding selon ITBCC 2016 (Lugli et al., Mod Pathol 2017).
    Comptage dans hotspot de 0.785 mm² (×20, champ 0.785 mm²).
    Bd1 (0–4), Bd2 (5–9), Bd3 (≥10 buds).
    Facteur pronostique indépendant validé en cancer colorectal.
    """
    bc = max(0, int(bud_count))

    if bc <= 4:
        grade, gn = "Bd1 — Faible", 1
        met_risk  = "Faible (< 5%)"
        ln_risk   = "Faible risque N+"
        interp    = (
            f"Tumor budding faible ({bc} buds/hotspot). "
            "Comportement tumoral non agressif. Absence d'infiltration péri-tumorale significative."
        )
        reco = (
            "Résection R0 standard. Surveillance post-opératoire standard. "
            "Coloscopie de contrôle 1 an post-chirurgie."
        )
    elif bc <= 9:
        grade, gn = "Bd2 — Intermédiaire", 2
        met_risk  = "Intermédiaire (5–15%)"
        ln_risk   = "Risque N+ augmenté (OR 2.1)"
        interp    = (
            f"Tumor budding modéré ({bc} buds/hotspot). "
            "Facteur de risque indépendant de rechute ganglionnaire et à distance. "
            "Corrélé avec invasion lymphovasculaire dans 40–60% des cas."
        )
        reco = (
            "Discuter chimiothérapie adjuvante selon stade T. "
            "Si pT1 : évaluation curage ganglionnaire complémentaire. "
            "Surveillance rapprochée (CEA 3 mois, scanner 6 mois)."
        )
    else:
        grade, gn = "Bd3 — Élevé", 3
        met_risk  = "Haut risque métastatique (OR 3.2)"
        ln_risk   = "Haut risque N+ (OR 3.2)"
        interp    = (
            f"Tumor budding élevé ({bc} buds/hotspot — ITBCC Bd3). "
            "Infiltration péri-tumorale très agressive. "
            "Risque de ganglions métastatiques et de rechute à distance multiplié par 3.2. "
            "Association fréquente avec invasion lymphovasculaire et périneurale."
        )
        reco = (
            "Chimiothérapie adjuvante recommandée (FOLFOX 6 mois ou CAPOX 3 mois). "
            "Si MSI-H : immunothérapie (pembrolizumab) à discuter en RCP. "
            "Surveillance intensive : CEA 2 mois, scanner thoraco-abdo 4 mois."
        )

    return TumorBuddingResult(
        bud_count=bc, grade=grade, grade_number=gn,
        hotspot_area="0.785 mm² (champ ×20)",
        metastasis_risk=met_risk, lymph_node_risk=ln_risk,
        interpretation=interp, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. EDMONDSON-STEINER GRADE — Carcinome hépatocellulaire (HCC)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EdmondsonResult:
    grade: int
    label: str
    differentiation: str
    five_year_survival: str
    recurrence_risk: str
    vascular_invasion_risk: str
    interpretation: str
    recommendation: str


def compute_edmondson_steiner(grade: int = 2) -> EdmondsonResult:
    """
    Edmondson-Steiner Grade (Edmondson & Steiner, Cancer 1954).
    Grade histologique du carcinome hépatocellulaire (I–IV).
    Corrélé à l'AFP, l'invasion vasculaire et la survie post-résection.
    """
    g = max(1, min(4, int(grade)))

    data = {
        1: {
            "label":   "Grade I — Très bien différencié",
            "diff":    "Bien différencié",
            "surv5":   "55–65%",
            "rec":     "Faible (15–25%)",
            "vasc":    "Faible (< 10%)",
            "interp":  (
                "Cellules tumorales très proches des hépatocytes normaux. "
                "Trabécules minces (1–2 couches cellulaires), cytoplasme abondant éosinophile. "
                "Excellent pronostic si résection R0."
            ),
            "reco":    (
                "Résection hépatique curative (hépatectomie partielle). "
                "RFA si tumeur < 3 cm et accessible. Suivi AFP + écho 3 mois."
            ),
        },
        2: {
            "label":   "Grade II — Bien différencié",
            "diff":    "Modérément différencié",
            "surv5":   "38–48%",
            "rec":     "Intermédiaire (30–45%)",
            "vasc":    "Modéré (20–35%)",
            "interp":  (
                "Différenciation modérée. Trabécules épaissies (2–3 couches), "
                "atypies nucléaires présentes. Pronostic intermédiaire selon Child-Pugh et AFP."
            ),
            "reco":    (
                "Résection hépatique si Child-Pugh A ou B. "
                "TACE ou RFA si inopérable. Suivi AFP 3 mois + IRM 6 mois."
            ),
        },
        3: {
            "label":   "Grade III — Modérément différencié",
            "diff":    "Peu différencié",
            "surv5":   "20–30%",
            "rec":     "Élevé (55–65%)",
            "vasc":    "Élevé (45–60%)",
            "interp":  (
                "Atypies nucléaires majeures, rapport N/C élevé, mitoses nombreuses. "
                "Trabécules irrégulières. Risque invasion vasculaire élevé. "
                "Pronostic défavorable même après résection."
            ),
            "reco":    (
                "TACE + sorafénib ou atézolizumab-bévacizumab (IMbrave150). "
                "Évaluation transplantation si critères Milan (1 nœud < 5 cm ou 3 nœuds < 3 cm). "
                "Lenvatinib en 2ème ligne."
            ),
        },
        4: {
            "label":   "Grade IV — Peu différencié (Anaplasique)",
            "diff":    "Indifférencié",
            "surv5":   "< 12%",
            "rec":     "Très élevé (> 70%)",
            "vasc":    "Très élevé (> 70%)",
            "interp":  (
                "Tumeur anaplasique indifférenciée. Cellules géantes, nécrose tumorale extensive. "
                "Invasion vasculaire presque constante. Très mauvais pronostic. "
                "Souvent multinodulaire avec thrombose porte."
            ),
            "reco":    (
                "Immunothérapie systémique (atézolizumab-bévacizumab ou tremelimumab-durvalumab). "
                "Soins palliatifs précoces. Sorafénib si contre-indication immunothérapie."
            ),
        },
    }

    d = data[g]
    return EdmondsonResult(
        grade=g, label=d["label"], differentiation=d["diff"],
        five_year_survival=d["surv5"], recurrence_risk=d["rec"],
        vascular_invasion_risk=d["vasc"],
        interpretation=d["interp"], recommendation=d["reco"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. CIN GRADING + BETHESDA SYSTEM — Néoplasie intraépithéliale cervicale
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CINResult:
    cin_grade: str
    bethesda_category: str
    hpv_status: bool
    hpv_high_risk_types: list
    malignant_transformation_risk: str
    regression_probability: str
    progression_timeline: str
    interpretation: str
    recommendation: str


def compute_cin_grade(
    cin_grade: str = "CIN II",
    hpv_status: bool = True,
    hpv_types: list | None = None,
) -> CINResult:
    """
    CIN Grading selon WHO Classification of Female Genital Tumours 2020.
    Bethesda System 2014 pour cytologie cervicale.
    HPV HR types 16/18 : 70% des cancers du col (IARC 2022).
    """
    cin    = str(cin_grade).upper().strip()
    hpvt   = hpv_types or (["HPV 16"] if hpv_status else [])

    if "III" in cin or "CIS" in cin or "IN SITU" in cin:
        cin_key = "CIN III"
        bethesda = "HSIL — Carcinome In Situ (CIS)"
        transf   = "30–50% sans traitement (suivi 10–15 ans)"
        regress  = "< 15% (régression spontanée rare)"
        timeline = "Progression vers carcinome invasif en 10–15 ans sans traitement"
        interp   = (
            "Dysplasie sévère envahissant toute l'épaisseur de l'épithélium cervical. "
            "Carcinome in situ — lésion précancéreuse à haut risque de progression. "
            "Traitement obligatoire selon guidelines OMS/ESGO 2022."
        )
        reco = (
            "Conisation diathermique LEEP/LLETZ ou conisation au bistouri froid. "
            "Marges libres indispensables (R0). Suivi colposcopique 3 mois post-conisation. "
            "Frottis + PCR HPV 6 et 12 mois. Vaccination HPV si < 26 ans."
        )
    elif "II" in cin:
        cin_key  = "CIN II"
        bethesda = "HSIL — Lésion Intraépithéliale de Haut Grade"
        transf   = "5–15% si non traité"
        regress  = "40–50% à 2 ans (surveillance possible < 25 ans)"
        timeline = "Progression vers CIN III en 2–5 ans possible"
        interp   = (
            "Dysplasie modérée envahissant les 2/3 inférieurs de l'épithélium. "
            "HSIL selon Bethesda. Traitement recommandé chez la femme > 25 ans. "
            "Surveillance acceptable chez femme jeune (< 25 ans) avec cytologie 6 mois."
        )
        reco = (
            "LEEP/LLETZ recommandé. Ou surveillance colposcopique 12 mois si < 25 ans. "
            "Biopsie dirigée colposcopique obligatoire. Test HPV 12 mois post-traitement. "
            "Vaccination HPV 9-valent recommandée si non vaccinée."
        )
    else:
        cin_key  = "CIN I"
        bethesda = "LSIL — Lésion Intraépithéliale de Bas Grade"
        transf   = "< 1–2% (très faible risque)"
        regress  = "60–70% à 2 ans (surtout HPV 6/11)"
        timeline = "Régression spontanée fréquente en 12–24 mois"
        interp   = (
            "Dysplasie légère du tiers inférieur de l'épithélium. Koïlocytes visibles. "
            "Lésion souvent transitoire liée à HPV à bas risque (6/11). "
            "Régression spontanée dans 60–70% des cas à 2 ans."
        )
        reco = (
            "Surveillance colposcopique à 12 mois. Pas de traitement actif recommandé. "
            "Frottis de contrôle à 12 mois. Traitement si persistance > 24 mois ou progression."
        )

    hpv_note = ""
    if hpv_status and any(t in ("HPV 16", "HPV 18", "HPV 31", "HPV 33", "HPV 45") for t in hpvt):
        hpv_note = " HPV HR 16/18 positif — risque de progression majoré (réduire délai surveillance)."
    elif hpv_status:
        hpv_note = " HPV HR positif — suivi renforcé recommandé."
    else:
        hpv_note = " HPV négatif — risque de progression très faible."

    return CINResult(
        cin_grade=cin_key, bethesda_category=bethesda,
        hpv_status=hpv_status, hpv_high_risk_types=hpvt,
        malignant_transformation_risk=transf,
        regression_probability=regress,
        progression_timeline=timeline,
        interpretation=interp + hpv_note,
        recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SOUS-TYPE MOLÉCULAIRE — Cancer du sein (St. Gallen 2021)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MolecularSubtypeResult:
    subtype: str
    her2_status: str
    hr_status: str
    ki67_category: str
    frequency: str
    prognosis: str
    preferred_therapy: str
    five_year_os: str


def compute_molecular_subtype(
    er_positive: bool = True,
    pr_positive: bool = True,
    her2_positive: bool = False,
    ki67_percent: float = 15.0,
) -> MolecularSubtypeResult:
    """
    Classification moléculaire cancer du sein par proxy IHC (St. Gallen 2021).
    Luminal A · Luminal B (HER2±) · HER2-enrichi · Triple Négatif (TNBC).
    """
    hr_pos    = er_positive or pr_positive
    hr_str    = "HR+" if hr_pos else "HR−"
    her2_str  = "HER2+" if her2_positive else "HER2−"
    ki67_cat  = "Élevé" if ki67_percent >= 30 else ("Intermédiaire" if ki67_percent >= 14 else "Faible")

    if hr_pos and not her2_positive and ki67_percent < 14:
        subtype  = "Luminal A"
        freq     = "40–45% des cancers du sein"
        prog     = "Excellent — meilleur pronostic"
        therapy  = "Hormonothérapie seule 5–10 ans (tamoxifène ou IA). Chimio non indiquée."
        os5      = "88–95%"
    elif hr_pos and not her2_positive and ki67_percent >= 14:
        subtype  = "Luminal B (HER2−)"
        freq     = "20–25% des cancers du sein"
        prog     = "Intermédiaire"
        therapy  = "Hormonothérapie + chimiothérapie (si Oncotype DX ≥ 26 ou Ki-67 ≥ 30%)."
        os5      = "78–88%"
    elif hr_pos and her2_positive:
        subtype  = "Luminal B (HER2+)"
        freq     = "10–12% des cancers du sein"
        prog     = "Intermédiaire-défavorable sans ciblage HER2"
        therapy  = "Chimiothérapie + double blocage HER2 (trastuzumab+pertuzumab) + hormonothérapie."
        os5      = "80–88%"
    elif not hr_pos and her2_positive:
        subtype  = "HER2-enrichi (non Luminal)"
        freq     = "10–12% des cancers du sein"
        prog     = "Défavorable sans ciblage HER2 — amélioré par anti-HER2"
        therapy  = "Chimiothérapie + trastuzumab + pertuzumab. T-DM1 adjuvant si résidu."
        os5      = "72–82%"
    else:
        subtype  = "Triple Négatif (TNBC)"
        freq     = "15–20% des cancers du sein"
        prog     = "Défavorable — sous-type le plus agressif"
        therapy  = "Chimio néoadjuvante + pembrolizumab si PD-L1 CPS ≥ 10. Capécitabine adjuvante si résidu."
        os5      = "63–75%"

    return MolecularSubtypeResult(
        subtype=subtype, her2_status=her2_str, hr_status=hr_str,
        ki67_category=ki67_cat, frequency=freq, prognosis=prog,
        preferred_therapy=therapy, five_year_os=os5,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 11. ORCHESTRATEUR — compute_all_histopath_scores()
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_histopath_scores(params: dict[str, Any]) -> dict[str, Any]:
    """
    Orchestre le calcul de tous les scores anatomopathologiques selon le type de cancer.
    Retourne un dict avec tous les scores calculés.
    """
    ct = str(params.get("cancer_type", "general")).lower()
    results: dict[str, Any] = {}

    # TNM pathologique — universel
    results["tnm"] = vars(compute_tnm_pathological(
        cancer_type=ct,
        pt=params.get("pt_stage", "T2"),
        pn=params.get("pn_stage", "N0"),
        pm=params.get("pm_stage", "M0"),
        r_status=params.get("r_status", "R0"),
        grade=params.get("histological_grade", 2),
    ))

    if ct == "breast":
        results["nottingham"] = vars(compute_nottingham_grade(
            tubule_formation=params.get("tubule_formation", 2),
            nuclear_pleomorphism=params.get("nuclear_pleomorphism", 2),
            mitotic_count=params.get("mitotic_count", 2),
        ))
        results["ki67"] = vars(compute_ki67(params.get("ki67_percent", 20.0)))
        results["her2"] = vars(compute_her2(
            ihc_score=params.get("her2_score", "2+"),
            fish_ratio=params.get("her2_fish"),
        ))
        results["hormone_receptors"] = vars(compute_hormone_receptors(
            er_percent=params.get("er_percent", 75.0),
            pr_percent=params.get("pr_percent", 50.0),
        ))
        her2_p = (str(params.get("her2_score", "0")) == "3+" or
                  (float(params.get("her2_fish") or 0) >= 2.0))
        results["molecular_subtype"] = vars(compute_molecular_subtype(
            er_positive=float(params.get("er_percent", 0)) >= 1.0,
            pr_positive=float(params.get("pr_percent", 0)) >= 1.0,
            her2_positive=her2_p,
            ki67_percent=float(params.get("ki67_percent", 20.0)),
        ))

    elif ct in ("prostate",):
        results["gleason"] = vars(compute_gleason(
            primary=params.get("gleason_primary", 3),
            secondary=params.get("gleason_secondary", 4),
            tertiary=params.get("gleason_tertiary"),
        ))

    elif ct in ("colorectal", "colon", "rectal", "crc"):
        results["tumor_budding"] = vars(compute_tumor_budding(
            bud_count=params.get("tumor_budding", 3),
        ))

    elif ct in ("cervical", "cervix", "col"):
        results["cin"] = vars(compute_cin_grade(
            cin_grade=params.get("cin_grade", "CIN II"),
            hpv_status=params.get("hpv_status", True),
            hpv_types=params.get("hpv_types"),
        ))

    elif ct in ("liver", "hepatic", "hcc", "foie"):
        results["edmondson_steiner"] = vars(compute_edmondson_steiner(
            grade=params.get("edmondson_grade", 2),
        ))

    return results
