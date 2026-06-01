"""
GynoCare AI — Scores cliniques gynécologiques et oncologiques de référence
==========================================================================
FIGO Col Utérin 2018 · FIGO Endomètre 2023 · FIGO Ovaire 2014
ESGO Risk Groups (Endomètre) · O-RADS (Ovaire) · rASRM (Endométriose)
Rotterdam Criteria (SOPK) · ROMA Score · Marqueurs tumoraux

Sources :
  - FIGO 2018 Cervical Cancer Staging (Bhatla et al., Int J Gynaecol Obstet 2019)
  - FIGO 2023 Endometrial Cancer Classification (Berek et al., Int J Gynaecol Obstet 2023)
  - FIGO 2014 Ovarian Cancer Staging (Prat, Int J Gynaecol Obstet 2014)
  - ESGO/ESTRO/ESP Guidelines Endometrial Cancer 2020
  - O-RADS MRI Score 2020 (ACR) · O-RADS US 2020
  - rASRM Classification Endometriosis 1997 (ASRM)
  - Rotterdam Consensus SOPK 2003
  - NCCN Clinical Practice Guidelines in Oncology 2023
  - Gynecologic Oncology · The Lancet Oncology · Int J Gynecol Cancer
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FIGO 2018 — Stadification Cancer du Col de l'Utérus
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FIGOCervixResult:
    figo_stage: str
    substage: str
    tnm: str
    five_year_os: str
    tumor_size_cm: float
    parametrial_invasion: bool
    vaginal_invasion: bool
    pelvic_wall: bool
    lymph_node_positive: bool
    distant_metastasis: bool
    treatment: str
    recommendation: str


def compute_figo_cervix(
    tumor_size_cm: float = 0.0,
    microscopic_only: bool = False,
    parametrial_invasion: bool = False,
    vaginal_upper_third: bool = False,
    vaginal_lower_third: bool = False,
    pelvic_wall_invasion: bool = False,
    hydronephrosis: bool = False,
    bladder_invasion: bool = False,
    rectal_invasion: bool = False,
    lymph_node_pelvic: bool = False,   # IIIC1
    lymph_node_paraaortic: bool = False, # IIIC2
    distant_metastasis: bool = False,  # IVB
    cin_grade: str = "none",           # cin1, cin2, cin3, cis
) -> FIGOCervixResult:
    """
    FIGO 2018 Cervical Cancer Staging (Bhatla et al., Int J Gynaecol Obstet 2019).
    Inclut les modifications 2018 : stadification ganglionnaire IIIC1/IIIC2.
    """
    # CIN / In situ
    cin = str(cin_grade).lower()
    if cin in ("cin1",):
        stage, sub, tnm = "Précancéreux CIN1", "Lésion bas grade (LSIL)", "T0"
        os5 = "> 99% (régression spontanée fréquente)"
        treat = "Surveillance colposcopique — frottis 6 mois"
        reco  = "Colposcopie + biopsie — frottis de contrôle à 6 et 12 mois — pas de traitement si CIN1 persistant < 2 ans"
        return FIGOCervixResult(
            figo_stage=stage, substage=sub, tnm=tnm, five_year_os=os5,
            tumor_size_cm=0, parametrial_invasion=False, vaginal_invasion=False,
            pelvic_wall=False, lymph_node_positive=False, distant_metastasis=False,
            treatment=treat, recommendation=reco,
        )
    if cin in ("cin2",):
        stage, sub, tnm = "Précancéreux CIN2", "Lésion haut grade (HSIL)", "T0"
        os5 = "> 99% après traitement"
        treat = "Conisation LLETZ ou laser — surveillance post-traitement"
        reco  = "Conisation diagnostique et thérapeutique — HPV test + frottis à 6 mois post-traitement"
        return FIGOCervixResult(
            figo_stage=stage, substage=sub, tnm=tnm, five_year_os=os5,
            tumor_size_cm=0, parametrial_invasion=False, vaginal_invasion=False,
            pelvic_wall=False, lymph_node_positive=False, distant_metastasis=False,
            treatment=treat, recommendation=reco,
        )
    if cin in ("cin3", "cis", "carcinome_in_situ"):
        stage, sub, tnm = "Carcinome in situ (CIN3/CIS)", "HSIL sévère / CIS", "Tis"
        os5 = "> 98% après traitement complet"
        treat = "Conisation large — hystérectomie si marge positive ou désir de grossesse accompli"
        reco  = "Conisation large (marges saines) — hystérectomie si marge positive — suivi strict 5 ans"
        return FIGOCervixResult(
            figo_stage=stage, substage=sub, tnm=tnm, five_year_os=os5,
            tumor_size_cm=0, parametrial_invasion=False, vaginal_invasion=False,
            pelvic_wall=False, lymph_node_positive=False, distant_metastasis=False,
            treatment=treat, recommendation=reco,
        )

    # Cancer invasif
    if distant_metastasis:
        stage, sub, tnm = "FIGO IVB", "Métastases à distance", "M1"
        os5 = "15–20%"
        treat = "Chimiothérapie palliative (cisplatine + paclitaxel + bévacizumab) — immunothérapie pembrolizumab"
        reco  = "Pembro + bevacizumab + chimo si PD-L1+ — soins de confort si performance altérée"
    elif bladder_invasion or rectal_invasion:
        stage, sub, tnm = "FIGO IVA", "Envahissement vésical/rectal", "T4"
        os5 = "20–25%"
        treat = "Radio-chimiothérapie concomitante — exentération pelvienne si faisable"
        reco  = "RCC (cisplatine hebdo) + curiethérapie de boost — bilan pré-exentération"
    elif lymph_node_paraaortic:
        stage, sub, tnm = "FIGO IIIC2", "N+ para-aortiques", "N2"
        os5 = "30–40%"
        treat = "Radio-chimiothérapie étendue aux ganglions para-aortiques + curiethérapie"
        reco  = "RCC champ étendu (45–50 Gy) + curiethérapie — pembro si PD-L1 ≥ 1%"
    elif lymph_node_pelvic:
        stage, sub, tnm = "FIGO IIIC1", "N+ pelviens", "N1"
        os5 = "40–55%"
        treat = "Radio-chimiothérapie concomitante (cisplatine 40 mg/m²/sem) + curiethérapie"
        reco  = "RCC (45 Gy pelvis) + curiethérapie — IRM post-traitement à 3 mois"
    elif pelvic_wall_invasion or hydronephrosis:
        stage, sub, tnm = "FIGO IIIB", "Extension paroi pelvienne / hydronéphrose", "T3b"
        os5 = "35–45%"
        treat = "Radio-chimiothérapie concomitante + curiethérapie"
        reco  = "RCC (50 Gy) + curiethérapie interstitielle — néphrostomie si urétéro-hydronéphrose"
    elif vaginal_lower_third:
        stage, sub, tnm = "FIGO IIIA", "Extension tiers inférieur vaginal", "T3a"
        os5 = "40–50%"
        treat = "Radio-chimiothérapie + curiethérapie intracavitaire"
        reco  = "RCC 45–50 Gy + boost curiethérapie — PET-TDM post-traitement"
    elif parametrial_invasion:
        stage, sub, tnm = "FIGO IIB", "Envahissement paramétrial", "T2b"
        os5 = "55–65%"
        treat = "Radio-chimiothérapie concomitante (standard of care)"
        reco  = "Cisplatine 40 mg/m²/sem × 5 + RT pelvis 45–50 Gy + curiethérapie"
    elif vaginal_upper_third:
        stage, sub, tnm = "FIGO IIA", "Extension vaginale (tiers sup.)", "T2a"
        if tumor_size_cm > 4:
            sub = "FIGO IIA2 > 4 cm"
            os5 = "60–70%"
            treat = "Radio-chimiothérapie ou chirurgie (Wertheim) selon centre expert"
        else:
            sub = "FIGO IIA1 ≤ 4 cm"
            os5 = "70–80%"
            treat = "Chirurgie (Wertheim-Meigs) ou RCC selon centre expert"
        reco = "RCP — chirurgie si < 4 cm et patiente opérable — RCC si IIA2 ou doute"
    elif microscopic_only:
        stage, sub, tnm = "FIGO IA", "Micro-invasif", "T1a"
        if tumor_size_cm <= 0.3:
            sub = "FIGO IA1 ≤ 3 mm profondeur"
            os5 = "> 98%"
            treat = "Conisation (si désir grossesse) ou hystérectomie simple"
        else:
            sub = "FIGO IA2 3–5 mm profondeur"
            os5 = "> 95%"
            treat = "Trachélectomie radicale (désir grossesse) ou hystérectomie radicale"
        reco = "Conisation diagnostique avec évaluation des marges — désir de fertilité à discuter"
    else:
        stage, tnm = "FIGO IB", "T1b"
        if tumor_size_cm <= 2:
            sub = "FIGO IB1 ≤ 2 cm"
            os5 = "90–95%"
            treat = "Hystérectomie radicale (Wertheim) + curage ganglionnaire"
        elif tumor_size_cm <= 4:
            sub = "FIGO IB2 2–4 cm"
            os5 = "80–88%"
            treat = "Hystérectomie radicale ou RCC (équivalents en IB2)"
        else:
            sub = "FIGO IB3 > 4 cm"
            os5 = "70–78%"
            treat = "RCC de référence (chirurgie déconseillée si > 4 cm)"
        reco = f"{sub} — RCP multidisciplinaire gynéco-onco — IRM pelvienne + TEP-TDM"

    return FIGOCervixResult(
        figo_stage=stage, substage=sub, tnm=tnm, five_year_os=os5,
        tumor_size_cm=tumor_size_cm,
        parametrial_invasion=parametrial_invasion,
        vaginal_invasion=(vaginal_upper_third or vaginal_lower_third),
        pelvic_wall=pelvic_wall_invasion,
        lymph_node_positive=(lymph_node_pelvic or lymph_node_paraaortic),
        distant_metastasis=distant_metastasis,
        treatment=treat, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FIGO 2023 + ESGO — Stadification Cancer de l'Endomètre
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FIGOEndometriumResult:
    figo_stage: str
    esgo_risk_group: str
    histologic_type: str
    grade: str
    five_year_os: str
    adjuvant_treatment: str
    recommendation: str


def compute_figo_endometrium(
    myometrial_invasion_pct: float = 0.0,  # % invasion myomètre
    cervical_stroma: bool = False,
    adnexal_invasion: bool = False,
    vaginal_invasion: bool = False,
    lymph_node_pelvic: bool = False,
    lymph_node_paraaortic: bool = False,
    bladder_bowel_invasion: bool = False,
    peritoneal_metastasis: bool = False,
    distant_metastasis: bool = False,
    histologic_type: str = "endometrioid",   # endometrioid, serous, clear_cell, carcinosarcoma
    grade: int = 1,                          # 1, 2, 3
    lvsi: bool = False,                      # Lymphovascular Space Invasion
    mismatch_repair_deficient: bool = False, # dMMR/MSI-H
    p53_mutated: bool = False,
    her2_positive: bool = False,
) -> FIGOEndometriumResult:
    """
    FIGO 2023 Endometrial Cancer Staging (Berek et al., Int J Gynaecol Obstet 2023).
    ESGO/ESTRO/ESP 2020 Risk Groups.
    """
    htype = str(histologic_type).lower()
    aggressive = htype in ("serous", "clear_cell", "carcinosarcoma") or p53_mutated

    # Staging
    if distant_metastasis:
        stage = "FIGO IVB"
        os5   = "15–25%"
        adj   = "Chimiothérapie systémique ± immunothérapie"
        reco  = "Carboplatine + paclitaxel ± pembrolizumab (dMMR) ou lenvatinib + pembrolizumab"
    elif peritoneal_metastasis or bladder_bowel_invasion:
        stage = "FIGO IVA"
        os5   = "25–40%"
        adj   = "Chimiothérapie + radiothérapie si localement avancé"
        reco  = "RCP — chimo carboplatine/paclitaxel — immunothérapie si dMMR/MSI-H"
    elif lymph_node_paraaortic:
        stage = "FIGO IIIC2"
        os5   = "40–55%"
        adj   = "Chimiothérapie + radiothérapie pelvienne étendue"
        reco  = "Carboplatine + paclitaxel × 6 cycles + RT pelvis étendu"
    elif lymph_node_pelvic:
        stage = "FIGO IIIC1"
        os5   = "55–65%"
        adj   = "Chimiothérapie + curiethérapie vaginale"
        reco  = "Carboplatine + paclitaxel × 6 cycles + curiethérapie vaginale"
    elif vaginal_invasion or adnexal_invasion:
        stage = "FIGO IIIA/B"
        os5   = "55–70%"
        adj   = "Chimiothérapie séquentielle + radiothérapie"
        reco  = "Carboplatine + paclitaxel + RT pelvienne — RCP multidisciplinaire"
    elif cervical_stroma:
        stage = "FIGO II"
        os5   = "70–82%"
        adj   = "Radiothérapie externe + curiethérapie vaginale ± chimiothérapie"
        reco  = "RT pelvis 45 Gy + curiethérapie de boost — chimo si grade 3 ou histologie agressive"
    elif myometrial_invasion_pct >= 50:
        stage = "FIGO IB"
        os5   = "75–85%"
        if aggressive or grade == 3 or lvsi:
            adj  = "Curiethérapie vaginale ± radiothérapie externe ± chimiothérapie"
            reco = "Curiethérapie vaginale + RCP — chimo si séreux/cellules claires/G3"
        else:
            adj  = "Curiethérapie vaginale (standard)"
            reco = "Curiethérapie vaginale adjuvante — surveillance IRM à 3 mois"
    else:
        stage = "FIGO IA"
        os5   = "88–95%"
        if (grade == 1 or grade == 2) and not aggressive and not lvsi:
            adj  = "Surveillance (bas risque)"
            reco = "Pas de traitement adjuvant recommandé — surveillance clinique semestrielle"
        else:
            adj  = "Curiethérapie vaginale si haut risque"
            reco = "Curiethérapie vaginale si FIGO IA grade 3 ou histologie agressive — RCP"

    # ESGO Risk Groups (2020)
    g3 = grade == 3
    if stage in ("FIGO IVA", "FIGO IVB") or distant_metastasis:
        esgo = "Avancé / Métastatique"
    elif stage in ("FIGO IIIC1", "FIGO IIIC2", "FIGO IIIA/B"):
        esgo = "Haut risque — Avancé localement"
    elif stage == "FIGO IB" and (g3 or aggressive or lvsi):
        esgo = "Haut risque"
    elif stage == "FIGO IB" and grade == 2 and not aggressive:
        esgo = "Risque intermédiaire-élevé"
    elif stage == "FIGO IA" and (g3 or aggressive):
        esgo = "Haut risque"
    elif stage == "FIGO IA" and grade == 2:
        esgo = "Risque intermédiaire"
    elif stage == "FIGO II":
        esgo = "Haut risque"
    else:
        esgo = "Bas risque"

    grade_str = {1: "Grade 1 (bien différencié)", 2: "Grade 2 (modérément)", 3: "Grade 3 (peu différencié)"}.get(grade, "—")

    return FIGOEndometriumResult(
        figo_stage=stage, esgo_risk_group=esgo,
        histologic_type=htype, grade=grade_str,
        five_year_os=os5, adjuvant_treatment=adj, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FIGO 2014 — Stadification Cancer de l'Ovaire
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FIGOOvaryResult:
    figo_stage: str
    substage: str
    five_year_os: str
    histologic_type: str
    brca_status: str
    debulking_feasibility: str
    treatment: str
    recommendation: str


def compute_figo_ovary(
    confined_to_ovary: bool = True,
    bilateral: bool = False,
    capsule_ruptured: bool = False,
    pelvic_extension: bool = False,
    uterus_tubes_invaded: bool = False,
    peritoneal_implants: bool = False,
    retroperitoneal_nodes: bool = False,
    omental_metastasis: bool = False,
    liver_spleen_surface: bool = False,
    distant_metastasis: bool = False,
    pleural_effusion_malignant: bool = False,
    histologic_type: str = "serous_high_grade",
    brca1_mutated: bool = False,
    brca2_mutated: bool = False,
    ca125_u_ml: float = 35.0,
) -> FIGOOvaryResult:
    """
    FIGO 2014 Ovarian Cancer Staging (Prat, Int J Gynaecol Obstet 2014).
    Inclut les sous-stades IC1/IC2/IC3 · IIA/IIB · IIIA/IIIB/IIIC.
    """
    brca_s = "BRCA1+" if brca1_mutated else ("BRCA2+" if brca2_mutated else "BRCA non muté / inconnu")

    if distant_metastasis or pleural_effusion_malignant:
        stage, sub = "FIGO IVB", "Métastases extra-abdominales / épanchement pleural malin"
        os5 = "15–30%"
        deb = "Chirurgie de cytoréduction secondaire si R0 atteignable"
        treat = "Carboplatine + paclitaxel ± bévacizumab → maintenance olaparib (BRCA+)"
        reco = ("Carboplatine/paclitaxel × 6 cycles + bévacizumab — maintenance PARP-i si BRCA+"
                " — surveillance CA125 + TEP-TDM trimestrielle")
    elif liver_spleen_surface:
        stage, sub = "FIGO IVA", "Surface hépatique / splénique"
        os5 = "25–35%"
        deb = "Chirurgie de cytoréduction si centre expert"
        treat = "Chimiothérapie néoadjuvante → chirurgie d'intervalle → maintenance PARP-i"
        reco = "NACT 3 cycles → chirurgie d'intervalle → NACT 3 cycles → olaparib si BRCA+"
    elif omental_metastasis or (peritoneal_implants and retroperitoneal_nodes):
        stage, sub = "FIGO IIIC", "Implants péritonéaux > 2 cm ou N+ rétropéritonéaux"
        os5 = "35–45%"
        deb = "Cytoréduction primaire ou NACT → chirurgie d'intervalle"
        treat = "Carboplatine + paclitaxel + bévacizumab → maintenance PARP-i ou bévacizumab"
        reco = "Objectif: résidu tumoral nul (R0) — olaparib si BRCA+ ou niraparib si HRD+"
    elif peritoneal_implants:
        if ca125_u_ml > 200:
            stage, sub = "FIGO IIIB", "Implants péritonéaux ≤ 2 cm"
        else:
            stage, sub = "FIGO IIIA2", "Implants péritonéaux microscopiques"
        os5 = "45–55%"
        deb = "Chirurgie primaire recommandée"
        treat = "Chirurgie complète (R0) + carboplatine/paclitaxel + bévacizumab"
        reco = "Cytoréduction complète → carboplatine/paclitaxel × 6 → maintenance olaparib si BRCA+"
    elif retroperitoneal_nodes:
        stage, sub = "FIGO IIIA1", "N+ rétropéritonéaux seuls"
        os5 = "55–65%"
        deb = "Chirurgie complète recommandée"
        treat = "Chirurgie + carboplatine/paclitaxel — PARP-i si BRCA+"
        reco = "Chirurgie de stadification complète — carboplatine/paclitaxel × 6 cycles"
    elif pelvic_extension or uterus_tubes_invaded:
        stage, sub = "FIGO IIB", "Extension pelvienne — utérus/trompes"
        os5 = "65–75%"
        deb = "Chirurgie complète faisable"
        treat = "Chirurgie radicale + carboplatine/paclitaxel"
        reco = "Hystérectomie totale + annexectomie bilatérale + curage + carboplatine/paclitaxel × 6"
    elif confined_to_ovary:
        if capsule_ruptured:
            stage, sub = "FIGO IC2/IC3", "Capsule rompue per-op (IC2) / cellules malignes ascite (IC3)"
            os5 = "75–85%"
        elif bilateral:
            stage, sub = "FIGO IB", "Atteinte bilatérale — capsules intactes"
            os5 = "82–90%"
        else:
            stage, sub = "FIGO IA", "Ovaire unique — capsule intacte"
            os5 = "88–95%"
        deb = "Chirurgie de stadification complète"
        treat = "Chirurgie complète ± carboplatine/paclitaxel × 3–6 cycles (stade IA grade 1 : observation)"
        reco = "Chirurgie de stadification FIGO complète — chimo adjuvante si grade 2/3 ou histologie agressive"
    else:
        stage, sub = "FIGO II", "Extension pelvienne"
        os5 = "65–75%"
        deb = "Chirurgie complète recommandée"
        treat = "Chirurgie radicale + carboplatine/paclitaxel"
        reco = "RCP — cytoréduction primaire optimale — carboplatine/paclitaxel × 6"

    htype_label = {
        "serous_high_grade": "Séreux haut grade (HGSOC)",
        "serous_low_grade":  "Séreux bas grade (LGSOC)",
        "mucinous":          "Mucineux",
        "endometrioid":      "Endométrioïde",
        "clear_cell":        "Cellules claires",
    }.get(str(histologic_type).lower(), str(histologic_type))

    return FIGOOvaryResult(
        figo_stage=stage, substage=sub, five_year_os=os5,
        histologic_type=htype_label, brca_status=brca_s,
        debulking_feasibility=deb, treatment=treat, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. O-RADS — Ovarian-Adnexal Reporting & Data System (ACR 2020)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ORADSResult:
    score: int
    category: str
    malignancy_risk: str
    description: str
    recommendation: str


def compute_orads(
    lesion_present: bool = True,
    simple_cyst: bool = False,
    cyst_size_cm: float = 0.0,
    loculations: int = 1,
    solid_component: bool = False,
    solid_component_size_cm: float = 0.0,
    wall_papillary_projections: bool = False,
    acoustic_shadowing: bool = False,
    color_score: int = 1,             # 1=aucun, 2=minime, 3=modéré, 4=important
    irregular_wall: bool = False,
    ascites: bool = False,
    peritoneal_nodules: bool = False,
) -> ORADSResult:
    """
    O-RADS Ultrasound 2020 (Andreotti et al., Radiology 2020).
    Score 0–5. Risque de malignité croissant.
    """
    if not lesion_present:
        return ORADSResult(
            score=0, category="O-RADS 0 — Normal / Incomplet",
            malignancy_risk="Non applicable",
            description="Ovaires normaux ou évaluation incomplète.",
            recommendation="Évaluation complémentaire si doute clinique.",
        )

    # O-RADS 5 — Caractéristiques très suspectes
    if ascites or peritoneal_nodules or (solid_component and color_score >= 4 and cyst_size_cm > 10):
        return ORADSResult(
            score=5, category="O-RADS 5 — Très suspicieux",
            malignancy_risk="> 50%",
            description="Masse avec caractéristiques hautement suspectes de malignité.",
            recommendation="Référence chirurgicale urgente — TEP-TDM + CA-125 — RCP oncologie",
        )

    # O-RADS 4 — Probablement malin
    if (wall_papillary_projections and color_score >= 3) or (solid_component_size_cm > 3 and color_score >= 3) or (irregular_wall and solid_component):
        return ORADSResult(
            score=4, category="O-RADS 4 — Probablement malin",
            malignancy_risk="10–50%",
            description="Masse avec composante solide vasculaire ou projections papillaires.",
            recommendation="CA-125 + HE4 + IRM pelvienne — consultation gynéco-oncologie",
        )

    # O-RADS 3 — Intermédiaire
    if (wall_papillary_projections and color_score <= 2) or (solid_component and solid_component_size_cm <= 3) or (loculations > 3 and cyst_size_cm > 5):
        return ORADSResult(
            score=3, category="O-RADS 3 — Risque intermédiaire",
            malignancy_risk="1–10%",
            description="Kyste multiloculé > 10 cm ou avec composante solide non vasculaire.",
            recommendation="IRM pelvienne — CA-125 — contrôle échographique à 6–12 sem ou consultation gynéco",
        )

    # O-RADS 2 — Probablement bénin
    if simple_cyst and cyst_size_cm <= 10 and color_score <= 1:
        category = "O-RADS 2 — Presque certainement bénin"
        malignancy = "< 1%"
        desc = "Kyste uniloculaire simple ≤ 10 cm sans composante solide."
        reco = "Contrôle échographique à 6–12 semaines si < 5 cm — observation si ménopausée sans symptômes"
        return ORADSResult(score=2, category=category, malignancy_risk=malignancy,
                           description=desc, recommendation=reco)

    # O-RADS 1 — Normal
    return ORADSResult(
        score=1, category="O-RADS 1 — Normal",
        malignancy_risk="Quasi nul",
        description="Ovaires d'aspect normal.",
        recommendation="Pas de suivi spécifique requis.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. rASRM — Classification de l'Endométriose (ASRM 1997)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EndometriosisResult:
    score: int
    stage: str
    stage_label: str
    peritoneal_score: int
    ovarian_score: int
    adhesion_score: int
    cul_de_sac_obliterated: bool
    fertility_impact: str
    medical_treatment: str
    surgical_treatment: str
    recommendation: str


def compute_rasrm_endometriosis(
    peritoneal_superficial: float = 0.0,    # cm²
    peritoneal_deep: float = 0.0,           # cm²
    ovarian_right_superficial: float = 0.0, # cm²
    ovarian_right_deep: float = 0.0,
    ovarian_left_superficial: float = 0.0,
    ovarian_left_deep: float = 0.0,
    cul_de_sac_partial: bool = False,
    cul_de_sac_complete: bool = False,
    adhesion_tube_right: str = "none",     # none, filmy, dense
    adhesion_tube_left: str = "none",
    adhesion_ovary_right: str = "none",
    adhesion_ovary_left: str = "none",
    adhesion_enclosing_tube: bool = False,
) -> EndometriosisResult:
    """
    rASRM Endometriosis Classification (ASRM 1997 revision).
    Score 1–15 : Stade I/II · Score 16–40 : Stade III · > 40 : Stade IV.
    """
    s = 0

    # Implants péritonéaux
    if peritoneal_superficial > 3:    s += 6
    elif peritoneal_superficial > 0:  s += 3
    if peritoneal_deep > 3:           s += 6
    elif peritoneal_deep > 0:         s += 6
    peri_s = s

    # Implants ovariens
    ov_s = 0
    for side_sf, side_dp in [(ovarian_right_superficial, ovarian_right_deep),
                               (ovarian_left_superficial, ovarian_left_deep)]:
        if side_sf > 3:    ov_s += 20
        elif side_sf > 0:  ov_s += 4
        if side_dp > 3:    ov_s += 20
        elif side_dp > 0:  ov_s += 16
    s += ov_s

    # Cul-de-sac
    if cul_de_sac_complete:   s += 40
    elif cul_de_sac_partial:  s += 4
    cds = cul_de_sac_complete

    # Adhérences
    adh_s = 0
    adh_map = {"none": 0, "filmy": 4, "dense": 8}
    enc_map = {"none": 0, "filmy": 4, "dense": 16}
    adh_s += adh_map.get(adhesion_ovary_right, 0)
    adh_s += adh_map.get(adhesion_ovary_left, 0)
    adh_s += enc_map.get(adhesion_tube_right, 0) if not adhesion_enclosing_tube else 16
    adh_s += enc_map.get(adhesion_tube_left, 0)
    s += adh_s

    # Stade
    if s <= 5:
        stage, slabel = "I", "Minime"
        fert = "Impact minime sur la fertilité — taux de grossesse naturelle conservé"
        med  = "Progestatifs, contraceptif estro-progestatif en continu, DIU au lévonorgestrel"
        surg = "Coelioscopie diagnostique + vaporisation laser des implants si douleurs"
        reco = "Traitement médical en 1ère ligne (3–6 mois) — chirurgie si infertilité ou douleurs rebelles"
    elif s <= 15:
        stage, slabel = "II", "Légère"
        fert = "Impact modéré — stimulation ovarienne + IUI ou FIV recommandée"
        med  = "Dienogest 2 mg/j ou agoniste GnRH 3 mois — progestatifs en continu"
        surg = "Coelioscopie — exérèse des implants — libération adhérences trompes"
        reco = "Chirurgie coelioscopique + traitement médical post-op — FIV si infertilité persistante"
    elif s <= 40:
        stage, slabel = "III", "Modérée"
        fert = "Impact significatif — FIV recommandée d'emblée"
        med  = "Agoniste GnRH (leuproréline) 3–6 mois + add-back — dienogest post-op"
        surg = "Coelioscopie opératoire par chirurgien expert — kystectomie endométriome"
        reco = "Chirurgie experte (kystectomie ovaire + implants) → FIV — traitement médical 6 mois post-op"
    else:
        stage, slabel = "IV", "Sévère"
        fert = "Impact majeur — chirurgie complexe nécessaire avant FIV"
        med  = "Agoniste GnRH préopératoire 3 mois — dienogest longue durée"
        surg = "Chirurgie radicale multiviscérale en centre expert — shaving digestif si atteinte rectale"
        reco = "Centre expert endométriose profonde — IRM préop — chirurgie multidisciplinaire — FIV post-op"

    return EndometriosisResult(
        score=s, stage=stage, stage_label=slabel,
        peritoneal_score=peri_s, ovarian_score=ov_s, adhesion_score=adh_s,
        cul_de_sac_obliterated=cds,
        fertility_impact=fert, medical_treatment=med,
        surgical_treatment=surg, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ROTTERDAM CRITERIA — Syndrome des Ovaires Polykystiques (SOPK)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SOPKResult:
    criteria_met: int
    diagnosis: bool
    phenotype: str
    amh_ng_ml: float
    hormonal_profile: str
    metabolic_risk: str
    fertility_recommendation: str
    treatment: str
    recommendation: str


def compute_sopk_rotterdam(
    oligoanovulation: bool = False,
    clinical_hyperandrogenism: bool = False,  # hirsutisme, acné, alopécie
    biochemical_hyperandrogenism: bool = False,  # testostérone libre élevée
    polycystic_ovary_morphology: bool = False,   # ≥ 20 follicules < 10 mm ou vol > 10 mL
    amh_ng_ml: float = 3.0,
    lh_fsh_ratio: float = 1.0,
    testosterone_nmol_l: float = 1.0,
    fasting_insulin: float = 10.0,   # µUI/mL
    glucose: float = 5.0,            # mmol/L
    bmi: float = 23.0,
    other_causes_excluded: bool = True,  # hypothyroïdie, hyperPRL, cushing exclus
) -> SOPKResult:
    """
    Critères de Rotterdam 2003 (révisés 2018).
    2 des 3 critères requis : oligo/anovulation · hyperandrogénisme · PCOM.
    Exclusion obligatoire des autres causes.
    """
    hyperandro = clinical_hyperandrogenism or biochemical_hyperandrogenism
    criteria = sum([bool(oligoanovulation), bool(hyperandro), bool(polycystic_ovary_morphology)])

    if criteria >= 2 and other_causes_excluded:
        diagnosis = True
        if oligoanovulation and hyperandro and polycystic_ovary_morphology:
            phenotype = "Phénotype A (classique complet) — le plus sévère"
        elif oligoanovulation and hyperandro:
            phenotype = "Phénotype B (oligo-anovulation + hyperandrogénisme)"
        elif hyperandro and polycystic_ovary_morphology:
            phenotype = "Phénotype C (hyperandrogénisme + PCOM)"
        else:
            phenotype = "Phénotype D (oligo-anovulation + PCOM sans hyperandrogénisme)"
    else:
        diagnosis = False
        phenotype = "Critères insuffisants — diagnostic SOPK non retenu"

    # Profil hormonal
    horm = []
    if lh_fsh_ratio > 2:     horm.append(f"LH/FSH ratio élevé ({lh_fsh_ratio:.1f})")
    if testosterone_nmol_l > 2.8: horm.append(f"Testostérone élevée ({testosterone_nmol_l:.1f} nmol/L)")
    if amh_ng_ml > 5.0:      horm.append(f"AMH élevée ({amh_ng_ml:.1f} ng/mL)")
    horm_str = " · ".join(horm) if horm else "Profil hormonal normal"

    # Risque métabolique
    homa_ir = (fasting_insulin * glucose) / 22.5
    if homa_ir > 2.5 or bmi >= 30:
        metab_risk = f"Élevé — HOMA-IR {homa_ir:.1f} · IMC {bmi:.0f} — risque diabète T2 et MCV"
    elif homa_ir > 1.5:
        metab_risk = f"Modéré — HOMA-IR {homa_ir:.1f} — surveiller glycémie et bilan lipidique"
    else:
        metab_risk = "Faible — HOMA-IR normal"

    if not diagnosis:
        fert_reco = "Autres causes d'oligo-anovulation à explorer"
        treat = "Bilan étiologique — thyroïde, prolactine, FSH, test progestatif"
        reco  = "Diagnostic différentiel : hypothyroïdie, hyperprolactinémie, insuffisance ovarienne prématurée"
    elif oligoanovulation:
        fert_reco = "Induction de l'ovulation : létrozole 2,5–5 mg j2–j6 (1ère ligne) ou FSH recombinante + FIV si échec"
        treat = ("Metformine 1500–2000 mg/j si HOMA-IR > 2.5 — "
                 "Létrozole (1ère intention) — FIV avec stimulation douce si 3 cycles échec")
        reco  = "Corriger surpoids (objectif IMC < 25) — létrozole — FIV si échec stimulation 3–6 cycles"
    else:
        fert_reco = "Fertilité conservée — surveillance ovulation"
        treat = "Pilule estro-progestative (hirsutisme) — spironolactone si hyperandrogénisme sévère"
        reco  = "Traitement cosmétique hirsutisme — contraception si non désir — bilan métabolique annuel"

    return SOPKResult(
        criteria_met=criteria, diagnosis=diagnosis, phenotype=phenotype,
        amh_ng_ml=amh_ng_ml, hormonal_profile=horm_str,
        metabolic_risk=metab_risk, fertility_recommendation=fert_reco,
        treatment=treat, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ROMA SCORE — Risk of Ovarian Malignancy Algorithm
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ROMAResult:
    roma_score_pct: float
    high_risk: bool
    menopausal_status: str
    ca125_u_ml: float
    he4_pmol_l: float
    phe4: float
    pca125: float
    sensitivity: str
    specificity: str
    recommendation: str


def compute_roma_score(
    ca125_u_ml: float = 35.0,
    he4_pmol_l: float = 70.0,
    postmenopausal: bool = False,
) -> ROMAResult:
    """
    ROMA Score (Moore et al., Gynecol Oncol 2009 · Karlsen et al., BJOG 2012).
    Algorithme combinant CA-125 et HE4 + statut ménopausique.
    """
    import math

    phe4   = 0.0
    pca125 = 0.0

    if postmenopausal:
        # Post-ménopause
        pi = -8.309 + 1.404 * math.log(float(he4_pmol_l)) + 0.9306 * math.log(float(ca125_u_ml))
        menostatus = "Post-ménopausée"
        threshold = 29.9  # %
    else:
        # Pré-ménopause
        pi = -12.0 + 2.38 * math.log(float(he4_pmol_l)) + 0.0626 * math.log(float(ca125_u_ml))
        menostatus = "Pré-ménopausée"
        threshold = 11.4  # %

    roma = round(math.exp(pi) / (1 + math.exp(pi)) * 100, 1)
    high_risk = roma >= threshold

    if high_risk:
        reco = (f"ROMA {roma:.1f}% — RISQUE ÉLEVÉ de carcinome ovarien épithélial. "
                "Référence urgente gynéco-oncologie — IRM pelvienne + TEP-TDM — RCP")
        sens = "94% (pour les cancers épithéliaux)"
        spec = "75% (pré-ménopause) · 80% (post-ménopause)"
    else:
        reco = (f"ROMA {roma:.1f}% — Risque faible. "
                "Surveillance clinique et biologique — contrôle CA-125 + HE4 à 3 mois si kyste connu")
        sens = "94%"
        spec = "75–80%"

    return ROMAResult(
        roma_score_pct=roma, high_risk=high_risk,
        menopausal_status=menostatus,
        ca125_u_ml=float(ca125_u_ml), he4_pmol_l=float(he4_pmol_l),
        phe4=0.0, pca125=0.0,
        sensitivity=sens, specificity=spec,
        recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MARQUEURS TUMORAUX — Analyse complète
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TumorMarkersResult:
    ca125_u_ml: float
    he4_pmol_l: float
    cea_ng_ml: float
    ca19_9_u_ml: float
    afp_ng_ml: float
    bhcg_miu_ml: float
    ca125_elevated: bool
    he4_elevated: bool
    cea_elevated: bool
    bhcg_elevated: bool
    afp_elevated: bool
    flags: list
    interpretation: str
    recommendation: str


def compute_tumor_markers(
    ca125_u_ml: float = 20.0,
    he4_pmol_l: float = 60.0,
    cea_ng_ml: float = 2.0,
    ca19_9_u_ml: float = 15.0,
    afp_ng_ml: float = 5.0,
    bhcg_miu_ml: float = 0.0,
    postmenopausal: bool = False,
) -> TumorMarkersResult:
    """
    Analyse des marqueurs tumoraux gynécologiques (NCCN / ESGO 2023).
    """
    flags = []
    ca125_norm  = 35.0
    he4_norm_pre  = 140.0  # pmol/L pré-ménopause
    he4_norm_post = 70.0   # pmol/L post-ménopause
    he4_norm = he4_norm_post if postmenopausal else he4_norm_pre

    ca_elev = ca125_u_ml > ca125_norm
    he4_elev = he4_pmol_l > he4_norm
    cea_elev = cea_ng_ml > 5.0
    bhcg_elev = bhcg_miu_ml > 5.0
    afp_elev = afp_ng_ml > 10.0
    ca19_elev = ca19_9_u_ml > 37.0

    if ca125_u_ml > 500:
        flags.append({"marker":"CA-125","value":f"{ca125_u_ml:.0f} U/mL","status":"CRITIQUE",
                      "detail":"CA-125 > 500 — forte suspicion cancer ovarien avancé"})
    elif ca_elev:
        flags.append({"marker":"CA-125","value":f"{ca125_u_ml:.0f} U/mL","status":"ÉLEVÉ",
                      "detail":"CA-125 > 35 U/mL — OMS ovaire · endomètre · péritonite · endométriose"})
    if he4_elev:
        flags.append({"marker":"HE4","value":f"{he4_pmol_l:.0f} pmol/L","status":"ÉLEVÉ",
                      "detail":f"HE4 > {he4_norm:.0f} pmol/L — spécificité supérieure au CA-125 pour K ovaire"})
    if bhcg_elev:
        flags.append({"marker":"β-hCG","value":f"{bhcg_miu_ml:.0f} mUI/mL","status":"ÉLEVÉ",
                      "detail":"β-hCG élevée — grossesse? GTD? Choriocarcinome?"})
    if afp_elev:
        flags.append({"marker":"AFP","value":f"{afp_ng_ml:.0f} ng/mL","status":"MODÉRÉ",
                      "detail":"AFP élevée — tumeur germinale? Hépatopathie?"})
    if cea_elev:
        flags.append({"marker":"CEA","value":f"{cea_ng_ml:.0f} ng/mL","status":"MODÉRÉ",
                      "detail":"CEA élevé — K mucineux ovaire? Métastase digestive?"})
    if ca19_elev:
        flags.append({"marker":"CA19-9","value":f"{ca19_9_u_ml:.0f} U/mL","status":"MODÉRÉ",
                      "detail":"CA19-9 élevé — K mucineux ovaire? Pancréas?"})

    if ca_elev and he4_elev:
        interp = "CA-125 et HE4 élevés — suspicion forte de cancer ovarien épithélial. Score ROMA recommandé."
        reco   = "IRM pelvienne urgente + TEP-TDM + ROMA score + consultation gynéco-oncologie"
    elif bhcg_elev:
        interp = "β-hCG élevée — grossesse ou tumeur trophoblastique à exclure en urgence."
        reco   = "Echographie pelvienne urgente — β-hCG quantitatif sérié — consultation urgente"
    elif ca_elev:
        interp = "CA-125 isolément élevé — valeur diagnostique limitée (endométriose, kyste, menstruation)."
        reco   = "Échographie + HE4 + ROMA score + suivi à 3 mois — IRM si masse pelvienne"
    elif flags:
        interp = f"Anomalie(s) biomarqueurs : {', '.join(f['marker'] for f in flags)}."
        reco   = "Corrélation clinique et imagerie — consultation gynécologique"
    else:
        interp = "Marqueurs tumoraux gynécologiques dans les limites normales."
        reco   = "Surveillance clinique selon âge et facteurs de risque — mammographie + frottis selon calendrier"

    return TumorMarkersResult(
        ca125_u_ml=ca125_u_ml, he4_pmol_l=he4_pmol_l,
        cea_ng_ml=cea_ng_ml, ca19_9_u_ml=ca19_9_u_ml,
        afp_ng_ml=afp_ng_ml, bhcg_miu_ml=bhcg_miu_ml,
        ca125_elevated=ca_elev, he4_elevated=he4_elev,
        cea_elevated=cea_elev, bhcg_elevated=bhcg_elev, afp_elevated=afp_elev,
        flags=flags, interpretation=interp, recommendation=reco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. ORCHESTRATEUR
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_gyno_scores(params: dict[str, Any]) -> dict[str, Any]:
    """Orchestre tous les scores gynécologiques selon le contexte clinique."""
    results: dict[str, Any] = {}
    ctx = str(params.get("disease_context", "general")).lower()

    # Marqueurs tumoraux — toujours
    results["tumor_markers"] = vars(compute_tumor_markers(
        ca125_u_ml=float(params.get("ca125", 20)),
        he4_pmol_l=float(params.get("he4", 60)),
        cea_ng_ml=float(params.get("cea", 2)),
        ca19_9_u_ml=float(params.get("ca19_9", 15)),
        afp_ng_ml=float(params.get("afp", 5)),
        bhcg_miu_ml=float(params.get("bhcg", 0)),
        postmenopausal=bool(params.get("postmenopausal", False)),
    ))

    # ROMA score si CA-125 ou HE4 disponible
    if params.get("ca125") or params.get("he4"):
        results["roma"] = vars(compute_roma_score(
            ca125_u_ml=float(params.get("ca125", 35)),
            he4_pmol_l=float(params.get("he4", 70)),
            postmenopausal=bool(params.get("postmenopausal", False)),
        ))

    # FIGO Col
    if any(x in ctx for x in ("col", "cervix", "cin", "hpv", "cervical")):
        results["figo_cervix"] = vars(compute_figo_cervix(
            tumor_size_cm=float(params.get("tumor_size_cm", 0)),
            microscopic_only=bool(params.get("microscopic_only", False)),
            parametrial_invasion=bool(params.get("parametrial_invasion", False)),
            vaginal_upper_third=bool(params.get("vaginal_upper_third", False)),
            vaginal_lower_third=bool(params.get("vaginal_lower_third", False)),
            pelvic_wall_invasion=bool(params.get("pelvic_wall_invasion", False)),
            hydronephrosis=bool(params.get("hydronephrosis", False)),
            bladder_invasion=bool(params.get("bladder_invasion", False)),
            rectal_invasion=bool(params.get("rectal_invasion", False)),
            lymph_node_pelvic=bool(params.get("lymph_node_pelvic", False)),
            lymph_node_paraaortic=bool(params.get("lymph_node_paraaortic", False)),
            distant_metastasis=bool(params.get("distant_metastasis", False)),
            cin_grade=str(params.get("cin_grade", "none")),
        ))

    # FIGO Endomètre
    if any(x in ctx for x in ("endometre", "endomètre", "endometrium", "uterin", "utérin")):
        results["figo_endometrium"] = vars(compute_figo_endometrium(
            myometrial_invasion_pct=float(params.get("myometrial_invasion_pct", 0)),
            cervical_stroma=bool(params.get("cervical_stroma", False)),
            adnexal_invasion=bool(params.get("adnexal_invasion", False)),
            vaginal_invasion=bool(params.get("vaginal_invasion", False)),
            lymph_node_pelvic=bool(params.get("lymph_node_pelvic", False)),
            lymph_node_paraaortic=bool(params.get("lymph_node_paraaortic", False)),
            bladder_bowel_invasion=bool(params.get("bladder_bowel_invasion", False)),
            peritoneal_metastasis=bool(params.get("peritoneal_metastasis", False)),
            distant_metastasis=bool(params.get("distant_metastasis", False)),
            histologic_type=str(params.get("histologic_type", "endometrioid")),
            grade=int(params.get("histologic_grade", 1)),
            lvsi=bool(params.get("lvsi", False)),
            mismatch_repair_deficient=bool(params.get("mmr_deficient", False)),
            p53_mutated=bool(params.get("p53_mutated", False)),
        ))

    # FIGO Ovaire + ROMA + O-RADS
    if any(x in ctx for x in ("ovaire", "ovary", "ovarien", "ovarian")):
        results["figo_ovary"] = vars(compute_figo_ovary(
            confined_to_ovary=bool(params.get("confined_to_ovary", True)),
            bilateral=bool(params.get("bilateral_ovary", False)),
            capsule_ruptured=bool(params.get("capsule_ruptured", False)),
            pelvic_extension=bool(params.get("pelvic_extension", False)),
            peritoneal_implants=bool(params.get("peritoneal_implants", False)),
            retroperitoneal_nodes=bool(params.get("retroperitoneal_nodes", False)),
            omental_metastasis=bool(params.get("omental_metastasis", False)),
            liver_spleen_surface=bool(params.get("liver_spleen_metastasis", False)),
            distant_metastasis=bool(params.get("distant_metastasis", False)),
            histologic_type=str(params.get("histologic_type", "serous_high_grade")),
            brca1_mutated=bool(params.get("brca1_mutated", False)),
            brca2_mutated=bool(params.get("brca2_mutated", False)),
            ca125_u_ml=float(params.get("ca125", 35)),
        ))

    # O-RADS
    if params.get("ovarian_cyst_present") or params.get("orads_requested"):
        results["orads"] = vars(compute_orads(
            lesion_present=bool(params.get("ovarian_cyst_present", True)),
            simple_cyst=bool(params.get("simple_cyst", False)),
            cyst_size_cm=float(params.get("cyst_size_cm", 0)),
            loculations=int(params.get("cyst_loculations", 1)),
            solid_component=bool(params.get("solid_component", False)),
            solid_component_size_cm=float(params.get("solid_size_cm", 0)),
            wall_papillary_projections=bool(params.get("papillary_projections", False)),
            color_score=int(params.get("color_score", 1)),
            ascites=bool(params.get("ascites", False)),
            peritoneal_nodules=bool(params.get("peritoneal_nodules", False)),
        ))

    # Endométriose
    if any(x in ctx for x in ("endometriose", "endométriose", "endometriosis")):
        results["endometriosis"] = vars(compute_rasrm_endometriosis(
            peritoneal_superficial=float(params.get("peritoneal_superficial_cm2", 0)),
            peritoneal_deep=float(params.get("peritoneal_deep_cm2", 0)),
            ovarian_right_deep=float(params.get("ovarian_right_deep_cm2", 0)),
            ovarian_left_deep=float(params.get("ovarian_left_deep_cm2", 0)),
            cul_de_sac_partial=bool(params.get("cul_de_sac_partial", False)),
            cul_de_sac_complete=bool(params.get("cul_de_sac_complete", False)),
        ))

    # SOPK
    if any(x in ctx for x in ("sopk", "pcos", "polykystique", "polycystic")):
        results["sopk"] = vars(compute_sopk_rotterdam(
            oligoanovulation=bool(params.get("oligoanovulation", False)),
            clinical_hyperandrogenism=bool(params.get("clinical_hyperandrogenism", False)),
            biochemical_hyperandrogenism=bool(params.get("biochemical_hyperandrogenism", False)),
            polycystic_ovary_morphology=bool(params.get("polycystic_morphology", False)),
            amh_ng_ml=float(params.get("amh", 3.0)),
            lh_fsh_ratio=float(params.get("lh_fsh_ratio", 1.0)),
            testosterone_nmol_l=float(params.get("testosterone", 1.0)),
            fasting_insulin=float(params.get("fasting_insulin", 10)),
            glucose=float(params.get("fasting_glucose", 5.0)),
            bmi=float(params.get("bmi", 23)),
        ))

    return results
