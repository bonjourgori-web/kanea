"""
HematoVision AI — Scores cliniques hématologiques de référence
==============================================================
ELN 2022 (LAM) · Ann Arbor / Lugano (Lymphomes) · IPI Score
ISS / R-ISS (Myélome) · IPSS-R (SMD) · ISTH DIC Score
NFS complète · Analyse frottis · Parasitémie paludisme

Sources :
  - ELN AML Guidelines 2022 (Döhner et al., Blood 2022)
  - WHO Classification Haematopoietic 2022 (Khoury et al.)
  - ICC 2022 (Arber et al., Blood 2022)
  - Ann Arbor Staging 1971 · Lugano Classification 2014
  - IPI Score (NEJM 1993)
  - ISS (Greipp et al., JCO 2005) · R-ISS (Palumbo et al., JCO 2015)
  - IPSS-R (Greenberg et al., Blood 2012)
  - ISTH DIC Score (Taylor et al., Thromb Haemost 2001)
  - ASH/EHA Clinical Practice Guidelines 2023
  - NCCN Clinical Practice Guidelines 2023
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ANALYSE NFS — Numération Formule Sanguine complète
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NFSResult:
    # Hémogramme
    hemoglobin_g_dl: float
    hematocrit_pct: float
    rbc_t_l: float
    wbc_g_l: float
    platelets_g_l: float
    # Indices érythrocytaires
    vgm_fl: float
    tcmh_pg: float
    ccmh_g_dl: float
    rdw_pct: float
    # Anomalies détectées
    anemia: bool
    anemia_severity: str
    macrocytosis: bool
    microcytosis: bool
    leukocytosis: bool
    leukopenia: bool
    thrombocytopenia: bool
    thrombocytosis: bool
    neutropenia: bool
    critical_flags: list
    interpretation: str
    recommendation: str


def compute_nfs(
    hemoglobin: float = 12.0,      # g/dL
    hematocrit: float = 36.0,      # %
    rbc: float = 4.0,              # T/L (10¹²/L)
    wbc: float = 7.0,              # G/L
    platelets: float = 200.0,      # G/L
    vgm: float = 88.0,             # fL (VGM)
    tcmh: float = 28.0,            # pg (TCMH)
    ccmh: float = 32.0,            # g/dL (CCMH)
    rdw: float = 12.5,             # %
    neutrophils: float = 4.0,      # G/L
    lymphocytes: float = 2.0,      # G/L
    monocytes: float = 0.5,        # G/L
    eosinophils: float = 0.2,      # G/L
    basophils: float = 0.05,       # G/L
    sex: str = "F",
) -> NFSResult:
    """
    Analyse complète de la NFS selon valeurs de référence adulte.
    Hb normale : H > 13.0 g/dL · F > 12.0 g/dL (OMS 2011).
    """
    sex_f = str(sex).upper() == "F"
    hb_normal = 12.0 if sex_f else 13.0

    flags = []

    # Anémie
    anemia = hemoglobin < hb_normal
    if hemoglobin < 7.0:
        anemia_sev = "Sévère — transfusion indiquée"
        flags.append({"param":"Hb","value":f"{hemoglobin:.1f} g/dL","status":"CRITIQUE",
                      "detail":"Anémie sévère — transfusion si sympto ou Hb < 7"})
    elif hemoglobin < 10.0:
        anemia_sev = "Modérée"
        flags.append({"param":"Hb","value":f"{hemoglobin:.1f} g/dL","status":"ÉLEVÉ",
                      "detail":"Anémie modérée — bilan étiologique urgent"})
    elif hemoglobin < hb_normal:
        anemia_sev = "Légère"
        flags.append({"param":"Hb","value":f"{hemoglobin:.1f} g/dL","status":"MODÉRÉ",
                      "detail":"Anémie légère — bilan NFS + ferritine + B12/folates"})
    else:
        anemia_sev = "Pas d'anémie"

    # Indices érythrocytaires
    macrocytosis = vgm > 100
    microcytosis  = vgm < 80
    if macrocytosis:
        flags.append({"param":"VGM","value":f"{vgm:.0f} fL","status":"MODÉRÉ",
                      "detail":"Macrocytose — B12/folates? Alcool? Hypothyroïdie?"})
    if microcytosis:
        flags.append({"param":"VGM","value":f"{vgm:.0f} fL","status":"MODÉRÉ",
                      "detail":"Microcytose — carence fer? Thalassémie?"})

    # Leucocytes
    leukocytosis = wbc > 11.0
    leukopenia   = wbc < 4.0
    if wbc > 30.0:
        flags.append({"param":"GB","value":f"{wbc:.1f} G/L","status":"CRITIQUE",
                      "detail":"Hyperleucocytose majeure > 30 G/L — leucémie aiguë à exclure"})
    elif leukocytosis:
        flags.append({"param":"GB","value":f"{wbc:.1f} G/L","status":"ÉLEVÉ",
                      "detail":"Leucocytose — infection? LAM/LAL/LLC/LMC?"})
    elif leukopenia:
        flags.append({"param":"GB","value":f"{wbc:.1f} G/L","status":"MODÉRÉ",
                      "detail":"Leucopénie — aplasie? Chimiothérapie? VIH?"})

    # Plaquettes
    thrombocytopenia = platelets < 150
    thrombocytosis   = platelets > 400
    neutropenia      = neutrophils < 0.5
    if platelets < 20:
        flags.append({"param":"Plt","value":f"{platelets:.0f} G/L","status":"CRITIQUE",
                      "detail":"Thrombocytopénie sévère < 20 G/L — risque hémorragie spontanée"})
    elif platelets < 50:
        flags.append({"param":"Plt","value":f"{platelets:.0f} G/L","status":"ÉLEVÉ",
                      "detail":"Thrombocytopénie — transfusion si saignement ou geste invasif"})
    elif thrombocytosis:
        flags.append({"param":"Plt","value":f"{platelets:.0f} G/L","status":"MODÉRÉ",
                      "detail":"Thrombocytose > 400 G/L — TE? Réactionnelle?"})

    if neutropenia:
        flags.append({"param":"Neutrophiles","value":f"{neutrophils:.2f} G/L","status":"CRITIQUE",
                      "detail":"Neutropénie sévère < 0.5 G/L — isolement + G-CSF + antibio si fièvre"})

    # Interprétation globale
    issues = []
    if anemia:               issues.append(f"Anémie {anemia_sev}")
    if macrocytosis:         issues.append("Macrocytose")
    if microcytosis:         issues.append("Microcytose")
    if leukocytosis:         issues.append("Leucocytose")
    if leukopenia:           issues.append("Leucopénie")
    if thrombocytopenia:     issues.append("Thrombocytopénie")
    if thrombocytosis:       issues.append("Thrombocytose")
    if neutropenia:          issues.append("Neutropénie sévère")

    if not issues:
        interp = "NFS dans les limites normales."
        reco   = "Pas d'anomalie hématologique. Contrôle selon indication clinique."
    else:
        interp = f"Anomalies : {', '.join(issues)}."
        reco   = "Bilan hématologique complet — frottis sanguin — avis hématologue si besoin."

    return NFSResult(
        hemoglobin_g_dl=hemoglobin, hematocrit_pct=hematocrit, rbc_t_l=rbc,
        wbc_g_l=wbc, platelets_g_l=platelets,
        vgm_fl=vgm, tcmh_pg=tcmh, ccmh_g_dl=ccmh, rdw_pct=rdw,
        anemia=anemia, anemia_severity=anemia_sev,
        macrocytosis=macrocytosis, microcytosis=microcytosis,
        leukocytosis=leukocytosis, leukopenia=leukopenia,
        thrombocytopenia=thrombocytopenia, thrombocytosis=thrombocytosis,
        neutropenia=neutropenia, critical_flags=flags,
        interpretation=interp, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ELN 2022 — Classification du risque LAM (Döhner et al., Blood 2022)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ELNResult:
    risk_category: str
    favorable_markers: list
    adverse_markers: list
    cr_rate: str
    os_3yr: str
    transplant_recommendation: str
    targeted_therapy: list
    recommendation: str


def compute_eln_aml(
    npm1_mutated: bool = False,
    flt3_itd: bool = False,             # FLT3-ITD haute allélique
    flt3_itd_low: bool = False,         # FLT3-ITD faible allélique
    cebpa_biallelic: bool = False,
    t_8_21: bool = False,               # t(8;21) / RUNX1-RUNX1T1
    inv_16: bool = False,               # inv(16) / t(16;16)
    tp53_mutated: bool = False,
    runx1_mutated: bool = False,
    asxl1_mutated: bool = False,
    ezh2_mutated: bool = False,
    complex_karyotype: bool = False,    # ≥ 3 anomalies non récurrentes
    monosomal_karyotype: bool = False,
    del_5q: bool = False,
    del_7: bool = False,
    inv_3: bool = False,
    t_6_9: bool = False,
    bcr_abl1: bool = False,
) -> ELNResult:
    """
    ELN 2022 AML Risk Stratification (Döhner et al., Blood 2022).
    Favorable · Intermediate · Adverse.
    """
    fav_m   = []
    adv_m   = []

    if t_8_21:                 fav_m.append("t(8;21) / RUNX1-RUNX1T1")
    if inv_16:                 fav_m.append("inv(16) / t(16;16) / CBFB-MYH11")
    if npm1_mutated and not flt3_itd: fav_m.append("NPM1 muté sans FLT3-ITD")
    if cebpa_biallelic:        fav_m.append("CEBPA biallélique muté")
    if flt3_itd_low and npm1_mutated: fav_m.append("NPM1 muté + FLT3-ITD faible")

    if tp53_mutated:           adv_m.append("TP53 muté")
    if runx1_mutated:          adv_m.append("RUNX1 muté")
    if asxl1_mutated:          adv_m.append("ASXL1 muté")
    if ezh2_mutated:           adv_m.append("EH2 muté")
    if complex_karyotype:      adv_m.append("Caryotype complexe ≥ 3 anomalies")
    if monosomal_karyotype:    adv_m.append("Caryotype monosomal")
    if del_5q:                 adv_m.append("-5/del(5q)")
    if del_7:                  adv_m.append("-7/del(7q)")
    if inv_3:                  adv_m.append("inv(3) / t(3;3)")
    if t_6_9:                  adv_m.append("t(6;9) / DEK-NUP214")
    if bcr_abl1:               adv_m.append("BCR-ABL1")
    if flt3_itd and not npm1_mutated: adv_m.append("FLT3-ITD haute allélique sans NPM1")

    if adv_m and not fav_m:
        risk = "Adverse (Défavorable)"
        cr_rate = "45–60%"
        os3yr   = "15–30%"
        transpl = "Allogreffe en RC1 recommandée"
        targeted = ["Midostaurin ou Quizartinib si FLT3+","APR-246 si TP53+ (essais)"]
        reco    = "Induction 7+3 + midostaurin si FLT3 — allogreffe en RC1 si donneur disponible"
    elif fav_m:
        risk = "Favorable"
        cr_rate = "80–90%"
        os3yr   = "60–75%"
        transpl = "Allogreffe non recommandée en RC1 sauf rechute/haut risque"
        targeted = ["Midostaurin si FLT3-ITD faible","Vénétoclax en association"]
        reco    = "Induction 7+3 — consolidation HiDAC ×3–4 cycles — pas d'allogreffe en RC1"
    else:
        risk = "Intermédiaire"
        cr_rate = "60–75%"
        os3yr   = "35–55%"
        transpl = "Allogreffe à discuter en RC1 selon profil moléculaire et âge"
        targeted = ["Midostaurin si FLT3+","Ivosidenib si IDH1+","Enasidenib si IDH2+"]
        reco    = "Induction 7+3 ± midostaurin — allogreffe selon profil mutationnel"

    return ELNResult(
        risk_category=risk, favorable_markers=fav_m, adverse_markers=adv_m,
        cr_rate=cr_rate, os_3yr=os3yr,
        transplant_recommendation=transpl,
        targeted_therapy=targeted, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANN ARBOR / LUGANO — Stadification des lymphomes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnnArborResult:
    stage: str
    stage_roman: str
    bulky: bool
    b_symptoms: bool
    extranodal: bool
    five_year_os: str
    lugano_modification: str
    recommendation: str


def compute_ann_arbor(
    n_nodal_regions: int = 1,
    same_side_diaphragm: bool = True,
    extranodal_sites: int = 0,
    b_symptoms: bool = False,  # Fièvre, sueurs nocturnes, perte poids > 10%
    bulky_mass_cm: float = 0.0,  # Masse > 10 cm ou > 1/3 de diamètre thorax
    spleen_involved: bool = False,
    liver_involved: bool = False,
    bone_marrow: bool = False,
) -> AnnArborResult:
    """
    Ann Arbor Staging (1971) — Lugano Modification 2014 (Cheson et al., JCO 2014).
    Stages I–IV · B symptoms · Bulky disease.
    """
    extranodal = extranodal_sites > 0 or bone_marrow or liver_involved

    if bone_marrow or (extranodal_sites > 1) or liver_involved:
        stage_num, roman = 4, "IV"
        os5 = "55–70% (DLBCL)"
        lugano = "IV — maladie disséminée, atteinte médullaire ou hépatique"
        reco = ("R-CHOP 6 cycles — réévaluation PET-TDM après cycle 2 et 4. "
                "Rituximab maintenance si CD20+. Essai clinique recommandé.")
    elif (spleen_involved and extranodal_sites >= 1) or n_nodal_regions > 2 and not same_side_diaphragm:
        stage_num, roman = 3, "III"
        os5 = "65–80%"
        lugano = "III — régions ganglionnaires des 2 côtés du diaphragme"
        reco = "R-CHOP 6 cycles — PET-TDM assessment. Radiothérapie adjuvante si maladie bulky."
    elif n_nodal_regions >= 2 and same_side_diaphragm:
        stage_num, roman = 2, "II"
        os5 = "75–85%"
        lugano = "II — ≥ 2 régions ganglionnaires même côté diaphragme"
        reco = "R-CHOP 4–6 cycles + radiothérapie si stade II bulky."
    else:
        stage_num, roman = 1, "I"
        os5 = "85–95%"
        lugano = "I — région ganglionnaire unique ou site extranodal unique"
        reco = "R-CHOP 3–4 cycles + radiothérapie 30 Gy (traitement localisé)."

    b_suffix = "B" if b_symptoms else "A"
    stage_label = f"Stade {roman}{b_suffix}" + (" bulky" if bulky_mass_cm >= 10 else "")

    return AnnArborResult(
        stage=stage_label, stage_roman=roman,
        bulky=bulky_mass_cm >= 10, b_symptoms=b_symptoms,
        extranodal=extranodal, five_year_os=os5,
        lugano_modification=lugano, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. IPI SCORE — Index Pronostique International (LNH agressifs)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IPIResult:
    score: int
    risk_group: str
    cr_rate: str
    five_year_os: str
    components: dict
    recommendation: str


def compute_ipi(
    age: int = 60,
    ldh_elevated: bool = False,       # LDH > ULN
    ecog_ps: int = 0,                  # ECOG ≥ 2
    ann_arbor_stage: int = 2,          # III ou IV
    extranodal_sites: int = 1,         # > 1 site extranodal
) -> IPIResult:
    """
    IPI Score (International Prognostic Index — NEJM 1993).
    5 facteurs × 1 point chacun. Score 0–5.
    LNH agressifs (DLBCL et équivalents).
    """
    comps: dict[str, int] = {
        "Âge > 60 ans":           1 if age > 60 else 0,
        "LDH élevé":              1 if ldh_elevated else 0,
        "PS ECOG ≥ 2":            1 if ecog_ps >= 2 else 0,
        "Ann Arbor Stade III/IV": 1 if ann_arbor_stage >= 3 else 0,
        "Sites extranodaux > 1":  1 if extranodal_sites > 1 else 0,
    }
    score = sum(comps.values())

    data = {
        0: ("Low",              "87%","73%"),
        1: ("Low",              "87%","73%"),
        2: ("Low-Intermediate", "75%","51%"),
        3: ("High-Intermediate","56%","43%"),
        4: ("High",             "44%","26%"),
        5: ("High",             "44%","26%"),
    }
    risk, cr_rate, os5 = data[min(score, 5)]

    if score <= 1:
        reco = "R-CHOP 6 cycles — rémission attendue > 80%. PET-TDM final."
    elif score <= 2:
        reco = "R-CHOP 6 cycles ± radiothérapie consolidation. PET intermédiaire (cycle 4)."
    else:
        reco = ("R-CHOP 6 cycles + essai clinique. Considérer R-CHOEP ou rituximab polivy (Pola-R-CHP). "
                "Autogreffe en RC1 si jeune patient et haut risque.")

    return IPIResult(
        score=score, risk_group=risk, cr_rate=cr_rate,
        five_year_os=os5, components=comps, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ISS / R-ISS — Myélome Multiple (ISS 2005 · R-ISS 2015)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ISSResult:
    iss_stage: int
    r_iss_stage: int
    iss_label: str
    r_iss_label: str
    five_year_os_iss: str
    five_year_os_riss: str
    high_risk_cytogenetics: bool
    recommendation: str


def compute_iss_myeloma(
    albumin_g_dl: float = 3.5,
    beta2_microglobulin_mg_l: float = 3.0,
    ldh_elevated: bool = False,
    del_17p: bool = False,
    t_4_14: bool = False,       # t(4;14)
    t_14_16: bool = False,      # t(14;16)
    gain_1q: bool = False,      # +1q21 (gain)
) -> ISSResult:
    """
    ISS (Greipp et al., JCO 2005) + R-ISS (Palumbo et al., JCO 2015).
    ISS : albumine et β2-microglobuline.
    R-ISS : ISS + cytogénétique + LDH.
    """
    # ISS staging
    b2m = float(beta2_microglobulin_mg_l)
    alb = float(albumin_g_dl)

    if b2m < 3.5 and alb >= 3.5:
        iss = 1; iss_lbl = "ISS Stade I"; os5_iss = "62 mois médian"
    elif b2m >= 5.5:
        iss = 3; iss_lbl = "ISS Stade III"; os5_iss = "29 mois médian"
    else:
        iss = 2; iss_lbl = "ISS Stade II"; os5_iss = "44 mois médian"

    # Cytogénétique haut risque
    high_risk_cyto = del_17p or t_4_14 or t_14_16 or gain_1q

    # R-ISS
    if iss == 1 and not high_risk_cyto and not ldh_elevated:
        r_iss = 1; r_lbl = "R-ISS Stade I"; os5_riss = "82% à 5 ans"
    elif iss == 3 and (high_risk_cyto or ldh_elevated):
        r_iss = 3; r_lbl = "R-ISS Stade III"; os5_riss = "40% à 5 ans"
    else:
        r_iss = 2; r_lbl = "R-ISS Stade II"; os5_riss = "62% à 5 ans"

    if r_iss == 1:
        reco = ("Triplet VRd (bortézomib + lénalidomide + dexaméthasone) 4 cycles — "
                "autogreffe si éligible — maintenance lénalidomide.")
    elif r_iss == 2:
        reco = ("VRd ou DaraVRd (daratumumab + VRd) — autogreffe en CR/VGPR — "
                "maintenance lénalidomide ou ixazomib.")
    else:
        reco = ("Quadruplet DaraVRd ou KRd intensifié — tandem autogreffe si possible — "
                "essai clinique avec ciltacabtagene autoleucel (CARVYKTI) ou thérapies ciblées TP53.")

    return ISSResult(
        iss_stage=iss, r_iss_stage=r_iss,
        iss_label=iss_lbl, r_iss_label=r_lbl,
        five_year_os_iss=os5_iss, five_year_os_riss=os5_riss,
        high_risk_cytogenetics=high_risk_cyto,
        recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. IPSS-R — Revised IPSS Syndromes Myélodysplasiques (Greenberg 2012)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IPSSRResult:
    score: float
    risk_category: str
    median_survival_yr: str
    leukemia_risk_25pct: str
    treatment_indication: str
    recommendation: str


def compute_ipss_r(
    cytogenetics: str = "intermediate",  # very_good, good, intermediate, poor, very_poor
    bone_marrow_blasts_pct: float = 3.0,
    hemoglobin_g_dl: float = 10.0,
    platelets_g_l: float = 100.0,
    anc_g_l: float = 1.0,               # Neutrophiles absolus G/L
) -> IPSSRResult:
    """
    IPSS-R (Greenberg et al., Blood 2012).
    5 facteurs pondérés. Score 0–10+.
    Très faible · Faible · Intermédiaire · Élevé · Très élevé.
    """
    cyto_pts = {
        "very_good": 0, "good": 1, "intermediate": 2, "poor": 3, "very_poor": 4
    }.get(str(cytogenetics).lower().replace(" ","_"), 2)

    # Blastes médullaires
    if bone_marrow_blasts_pct <= 2:      blast_pts = 0
    elif bone_marrow_blasts_pct <= 4.9:  blast_pts = 1
    elif bone_marrow_blasts_pct <= 10:   blast_pts = 2
    else:                                blast_pts = 3

    # Hémoglobine
    if hemoglobin_g_dl >= 10:    hb_pts = 0
    elif hemoglobin_g_dl >= 8:   hb_pts = 1
    else:                        hb_pts = 1.5

    # Plaquettes
    if platelets_g_l >= 100:     plt_pts = 0
    elif platelets_g_l >= 50:    plt_pts = 0.5
    else:                        plt_pts = 1.0

    # Neutrophiles
    anc_pts = 0.5 if anc_g_l < 0.8 else 0

    score = round(cyto_pts + blast_pts + hb_pts + plt_pts + anc_pts, 1)

    if score <= 1.5:
        cat   = "Très faible risque"
        surv  = "8.8 ans"
        lk_25 = "NR (ne progresse pas)"
        tx    = "Observation ou G-CSF/EPO si symptômes"
        reco  = "Surveillance clinique et biologique — EPO si Hb < 10 et faible EPO endogène"
    elif score <= 3.0:
        cat   = "Faible risque"
        surv  = "5.3 ans"
        lk_25 = "10.8 ans"
        tx    = "EPO / G-CSF / Lénalidomide si del(5q)"
        reco  = ("EPO ± G-CSF — lénalidomide 10 mg si del(5q) — transfusions si Hb < 8 g/dL. "
                 "Chélation du fer si ferritine > 1000 µg/L.")
    elif score <= 4.5:
        cat   = "Risque intermédiaire"
        surv  = "3.0 ans"
        lk_25 = "3.2 ans"
        tx    = "Azacitidine — allogreffe si éligible et donneur"
        reco  = "Azacitidine 75 mg/m² × 7j/28j — évaluer allogreffe si < 70 ans et donneur."
    elif score <= 6.0:
        cat   = "Risque élevé"
        surv  = "1.6 ans"
        lk_25 = "1.7 ans"
        tx    = "Azacitidine ± vénétoclax — allogreffe urgente"
        reco  = "Azacitidine ± vénétoclax — allogreffe en urgence si éligible — essai clinique."
    else:
        cat   = "Très haut risque"
        surv  = "0.8 ans"
        lk_25 = "0.7 ans"
        tx    = "Allogreffe urgente ou azacitidine + vénétoclax"
        reco  = ("Allogreffe hématopoïétique en urgence si < 75 ans et PS conservé. "
                 "Azacitidine + vénétoclax si non éligible. Soins palliatifs si très âgé.")

    return IPSSRResult(
        score=score, risk_category=cat,
        median_survival_yr=surv, leukemia_risk_25pct=lk_25,
        treatment_indication=tx, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ISTH DIC SCORE — Coagulation Intravasculaire Disséminée
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DICResult:
    score: int
    overt_dic: bool
    mortality_risk: str
    components: dict
    interpretation: str
    recommendation: str


def compute_isth_dic(
    platelets_g_l: float = 200.0,
    pt_prolongation_sec: float = 0.0,   # allongement PT en secondes vs normal
    fibrinogen_g_l: float = 2.5,        # g/L
    d_dimers_elevated: int = 0,          # 0=normal, 1=modéré, 2=fort/fibrin
) -> DICResult:
    """
    ISTH DIC Score (Taylor et al., Thromb Haemost 2001).
    Score ≥ 5 = CIVD manifeste. Score < 5 = CIVD non manifeste.
    """
    comps: dict[str, int] = {}
    s = 0

    if platelets_g_l >= 100:      comps["Plaquettes"] = 0
    elif platelets_g_l >= 50:     comps["Plaquettes"] = 1; s += 1
    else:                          comps["Plaquettes"] = 2; s += 2

    if pt_prolongation_sec < 3:    comps["TP allongé"] = 0
    elif pt_prolongation_sec < 6:  comps["TP allongé"] = 1; s += 1
    else:                           comps["TP allongé"] = 2; s += 2

    if fibrinogen_g_l > 1.0:       comps["Fibrinogène"] = 0
    else:                           comps["Fibrinogène"] = 1; s += 1

    comps["D-Dimères"] = min(d_dimers_elevated, 3)
    s += comps["D-Dimères"]

    overt = s >= 5
    if s >= 7:
        mort  = "> 50%"
        interp = f"CIVD manifeste majeure (score {s}/9) — coagulopathie de consommation sévère."
        reco   = "URGENCE — PFC + plaquettes + fibrinogène IV — traiter cause — anticoagulation discutée"
    elif overt:
        mort  = "30–50%"
        interp = f"CIVD manifeste (score {s}/9) — coagulopathie de consommation active."
        reco   = "PFC 10–15 mL/kg — fibrinogène si < 1.5 g/L — traitement étiologique — surveillance horaire"
    else:
        mort  = "< 20%"
        interp = f"CIVD non manifeste (score {s}/9) — surveillance rapprochée."
        reco   = "Surveillance coagulation toutes les 6h — traitement cause déclenchante"

    return DICResult(
        score=s, overt_dic=overt, mortality_risk=mort,
        components=comps, interpretation=interp, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PARASITÉMIE PALUDISME — OMS grading
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MalariaResult:
    parasitemia_pct: float
    grade: str
    severity: str
    plasmodium_species: str
    treatment_urgency: str
    recommendation: str


def compute_parasitemia(
    parasitemia_pct: float = 0.0,
    species: str = "P. falciparum",
    rapid_test_positive: bool = False,
) -> MalariaResult:
    """
    Grading parasitémie paludisme (OMS 2022 Guidelines).
    < 0.1% faible · 0.1–2% modérée · > 2% élevée.
    P. falciparum > 2–5% = paludisme grave.
    """
    p = max(0.0, float(parasitemia_pct))
    falci = "falciparum" in str(species).lower()

    if p == 0 and not rapid_test_positive:
        grade  = "Négatif"
        sev    = "Pas de paludisme détecté"
        urg    = "Pas d'urgence"
        reco   = "Frottis/GE à répéter si fièvre persiste. TDR négatif. Autres diagnostics."
    elif p < 0.1 or (p == 0 and rapid_test_positive):
        grade  = "Grade 1 — Parasitémie faible"
        sev    = "Paludisme non compliqué (faible)"
        urg    = "Traitement oral urgent"
        reco   = ("P. falciparum : artémether + luméfantrine (Coartem) 3j VO. "
                  "P. vivax : chloroquine + primaquine (si G6PD ok). Suivi J3/J7/J28.")
    elif p < 2.0:
        grade  = "Grade 2 — Parasitémie modérée"
        sev    = "Paludisme non compliqué (modéré)"
        urg    = "Traitement oral intensif"
        reco   = ("Artémether-luméfantrine VO — surveillance clinique quotidienne. "
                  "Hospitalisation si vomissements, grossesse ou enfant < 5 ans.")
    elif p < 5.0:
        grade  = "Grade 3 — Parasitémie élevée"
        sev    = "Paludisme grave probable" if falci else "Parasitémie élevée"
        urg    = "Hospitalisation urgente"
        reco   = ("URGENCE — Artésunate IV (3 mg/kg) + doxycycline. "
                  "Surveillance NFS/biochimie/urines toutes les 12h. USI si défaillance.")
    else:
        grade  = "Grade 4 — Hyperparasitémie critique"
        sev    = "Paludisme grave / Hyperparasitémie"
        urg    = "USI — urgence vitale"
        reco   = ("RÉANIMATION — Artésunate IV haute dose — exsanguino-transfusion si > 10%. "
                  "Glucose IV — convulsions : diazépam IV. Surveillance conscience/anurie.")

    return MalariaResult(
        parasitemia_pct=p, grade=grade, severity=sev,
        plasmodium_species=str(species),
        treatment_urgency=urg, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. ORCHESTRATEUR — compute_all_hemato_scores()
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_hemato_scores(params: dict[str, Any]) -> dict[str, Any]:
    """Orchestre tous les scores hématologiques selon le contexte clinique."""
    results: dict[str, Any] = {}
    ct = str(params.get("disease_context","general")).lower()

    # NFS — toujours
    results["nfs"] = vars(compute_nfs(
        hemoglobin=float(params.get("hemoglobin",12)),
        hematocrit=float(params.get("hematocrit",36)),
        rbc=float(params.get("rbc",4.0)),
        wbc=float(params.get("wbc",7.0)),
        platelets=float(params.get("platelets",200)),
        vgm=float(params.get("vgm",88)),
        tcmh=float(params.get("tcmh",28)),
        ccmh=float(params.get("ccmh",32)),
        rdw=float(params.get("rdw",12.5)),
        neutrophils=float(params.get("neutrophils",4.0)),
        lymphocytes=float(params.get("lymphocytes",2.0)),
        monocytes=float(params.get("monocytes",0.5)),
        eosinophils=float(params.get("eosinophils",0.2)),
        basophils=float(params.get("basophils",0.05)),
        sex=str(params.get("sex","F")),
    ))

    # ELN (LAM)
    if "lam" in ct or "aml" in ct or params.get("blast_pct", 0) >= 20:
        results["eln_aml"] = vars(compute_eln_aml(
            npm1_mutated=bool(params.get("npm1_mutated",False)),
            flt3_itd=bool(params.get("flt3_itd",False)),
            flt3_itd_low=bool(params.get("flt3_itd_low",False)),
            cebpa_biallelic=bool(params.get("cebpa_biallelic",False)),
            t_8_21=bool(params.get("t_8_21",False)),
            inv_16=bool(params.get("inv_16",False)),
            tp53_mutated=bool(params.get("tp53_mutated",False)),
            runx1_mutated=bool(params.get("runx1_mutated",False)),
            asxl1_mutated=bool(params.get("asxl1_mutated",False)),
            complex_karyotype=bool(params.get("complex_karyotype",False)),
            del_5q=bool(params.get("del_5q",False)),
            del_7=bool(params.get("del_7",False)),
        ))

    # Ann Arbor + IPI (Lymphomes)
    if "lymphome" in ct or "hodgkin" in ct or "lnh" in ct or params.get("lymphoma"):
        results["ann_arbor"] = vars(compute_ann_arbor(
            n_nodal_regions=int(params.get("nodal_regions",1)),
            same_side_diaphragm=bool(params.get("same_side_diaphragm",True)),
            extranodal_sites=int(params.get("extranodal_sites",0)),
            b_symptoms=bool(params.get("b_symptoms",False)),
            bulky_mass_cm=float(params.get("bulky_mass_cm",0)),
            spleen_involved=bool(params.get("spleen_involved",False)),
            liver_involved=bool(params.get("liver_involved",False)),
            bone_marrow=bool(params.get("bone_marrow_involved",False)),
        ))
        results["ipi"] = vars(compute_ipi(
            age=int(params.get("age",60)),
            ldh_elevated=bool(params.get("ldh_elevated",False)),
            ecog_ps=int(params.get("ecog_ps",0)),
            ann_arbor_stage=int(params.get("nodal_regions",2)),
            extranodal_sites=int(params.get("extranodal_sites",1)),
        ))

    # ISS / R-ISS (Myélome)
    if "myelome" in ct or "myélome" in ct or "myeloma" in ct or params.get("myeloma"):
        results["iss_riss"] = vars(compute_iss_myeloma(
            albumin_g_dl=float(params.get("albumin",3.5)),
            beta2_microglobulin_mg_l=float(params.get("beta2_microglobulin",3.0)),
            ldh_elevated=bool(params.get("ldh_elevated",False)),
            del_17p=bool(params.get("del_17p",False)),
            t_4_14=bool(params.get("t_4_14",False)),
            t_14_16=bool(params.get("t_14_16",False)),
            gain_1q=bool(params.get("gain_1q",False)),
        ))

    # IPSS-R (SMD)
    if "smd" in ct or "mds" in ct or "myélodys" in ct or params.get("myelodysplasia"):
        results["ipss_r"] = vars(compute_ipss_r(
            cytogenetics=str(params.get("cytogenetics_risk","intermediate")),
            bone_marrow_blasts_pct=float(params.get("blast_pct",3.0)),
            hemoglobin_g_dl=float(params.get("hemoglobin",10)),
            platelets_g_l=float(params.get("platelets",100)),
            anc_g_l=float(params.get("neutrophils",1.0)),
        ))

    # ISTH DIC (CIVD)
    if params.get("dic_suspected") or float(params.get("d_dimers_severe",0)) > 0:
        results["isth_dic"] = vars(compute_isth_dic(
            platelets_g_l=float(params.get("platelets",200)),
            pt_prolongation_sec=float(params.get("pt_prolonged_sec",0)),
            fibrinogen_g_l=float(params.get("fibrinogen",2.5)),
            d_dimers_elevated=int(params.get("d_dimers_elevated",0)),
        ))

    # Paludisme
    if params.get("malaria_suspected") or float(params.get("parasitemia_pct",0)) > 0:
        results["malaria"] = vars(compute_parasitemia(
            parasitemia_pct=float(params.get("parasitemia_pct",0)),
            species=str(params.get("plasmodium_species","P. falciparum")),
            rapid_test_positive=bool(params.get("rdt_positive",False)),
        ))

    return results
