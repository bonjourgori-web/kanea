"""
RetinaVision AI — Scores cliniques ophtalmologiques
====================================================
ETDRS · ICDR · AREDS/AREDS2 · Cup-to-Disc · VFI · MD · CRT · Volume maculaire
CURB vasculaire · ROP Score · BRVO/CRVO · MyopiaCLASS.

Sources :
  - American Academy of Ophthalmology (AAO) 2023
  - European Society of Retina Specialists (EURETINA) 2023
  - ETDRS Research Group (Early Treatment Diabetic Retinopathy Study)
  - AREDS / AREDS2 Research Group — NEI
  - International Council of Ophthalmology (ICO)
  - Ophthalmology Journal · The Lancet Ophthalmology · PubMed
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ETDRS Severity Scale — Rétinopathie diabétique (ETDRS Research Group)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ETDRSResult:
    level: int
    stage: str
    description: str
    risk_progression: str
    recommendation: str
    urgency: str

def compute_etdrs(
    microaneurysms: int = 0,       # nombre estimé
    hemorrhages: bool = False,
    hard_exudates: bool = False,
    soft_exudates: bool = False,   # nodules cotonneux
    venous_beading: bool = False,
    irma: bool = False,            # Intraretinal Microvascular Abnormalities
    neovascularization: bool = False,
    vitreous_hemorrhage: bool = False,
    traction_detachment: bool = False,
) -> ETDRSResult:
    """
    ETDRS Severity Scale — 5 niveaux (AAD/EURETINA 2023).
    Level 10 = Pas de RD · Level 20-35 = NPDR légère
    Level 43-47 = NPDR modérée · Level 53 = NPDR sévère · Level 61+ = RDP.
    """
    score = 0
    score += min(microaneurysms // 3, 4)
    score += 2 if hemorrhages else 0
    score += 1 if hard_exudates else 0
    score += 2 if soft_exudates else 0
    score += 3 if venous_beading else 0
    score += 3 if irma else 0
    score += 5 if neovascularization else 0
    score += 4 if vitreous_hemorrhage else 0
    score += 5 if traction_detachment else 0

    if traction_detachment or (neovascularization and vitreous_hemorrhage):
        return ETDRSResult(
            level=71, stage="RDP avancée",
            description="Décollement tractionnel ou hémorragie vitréenne + NV — urgence chirurgicale",
            risk_progression="Perte visuelle irréversible imminente",
            recommendation="Vitrectomie urgente + panphotocoagulation + anti-VEGF intravitréen",
            urgency="Urgence absolue"
        )
    if neovascularization:
        return ETDRSResult(
            level=61, stage="RDP (Rétinopathie Diabétique Proliférante)",
            description="Néovascularisation active — risque hémorragique majeur",
            risk_progression="Perte visuelle sévère si non traité dans les semaines",
            recommendation="Panphotocoagulation rétinienne + anti-VEGF intravitréen — urgence < 2 semaines",
            urgency="Urgente"
        )
    if venous_beading or irma:
        return ETDRSResult(
            level=53, stage="NPDR sévère (règle 4-2-1)",
            description="Hémorragies 4 quadrants OU chapelet veineux 2Q OU AMIR 1Q",
            risk_progression="50% risque progression RDP dans l'année",
            recommendation="Contrôle glycémique strict + HbA1c < 7% + suivi ophtalmologique < 3 mois",
            urgency="Élevée"
        )
    if score >= 6:
        return ETDRSResult(
            level=47, stage="NPDR modérée",
            description="Microanévrismes + hémorragies + exsudats durs ou mous",
            risk_progression="15–20% risque progression sur 1 an",
            recommendation="Suivi ophtalmologique tous les 6 mois — contrôle PA et glycémie",
            urgency="Modérée"
        )
    if score >= 2:
        return ETDRSResult(
            level=20, stage="NPDR légère",
            description="Microanévrismes seuls sans autre lésion significative",
            risk_progression="5% risque progression sur 1 an",
            recommendation="Contrôle annuel + glycémie HbA1c < 7% + TA < 130/80 mmHg",
            urgency="Faible"
        )
    return ETDRSResult(
        level=10, stage="Pas de rétinopathie diabétique",
        description="Fond d'œil normal — aucun signe de RD détecté",
        risk_progression="< 2% sur 1 an si contrôle métabolique optimal",
        recommendation="Dépistage annuel si diabète ≥ 5 ans ou HbA1c > 8%",
        urgency="Faible"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. AREDS2 Classification — DMLA (NEI / AREDS2 Research Group)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AREDS2Result:
    category: int
    stage: str
    description: str
    five_year_risk: str
    recommendation: str
    supplement_indication: str

def compute_areds2(
    no_drusen: bool = False,
    small_drusen: bool = False,        # < 63 µm
    medium_drusen: bool = False,       # 63–124 µm
    large_drusen: bool = False,        # ≥ 125 µm
    geographic_atrophy: bool = False,
    neovascular_amd: bool = False,
    pigment_changes: bool = False,
    bilateral_involvement: bool = False,
) -> AREDS2Result:
    """
    AREDS2 Classification (NEI 2013, mise à jour AAO 2023).
    Catégories 1–4 avec risque à 5 ans.
    """
    if neovascular_amd or (geographic_atrophy and bilateral_involvement):
        return AREDS2Result(
            category=4, stage="DMLA avancée",
            description="Néovascularisation choroïdienne OU atrophie géographique centrale bilatérale",
            five_year_risk="Perte visuelle établie dans au moins un œil",
            recommendation="Anti-VEGF intravitréen (ranibizumab/aflibercept) si NVC — suivi mensuel",
            supplement_indication="AREDS2 pour protéger l'œil adelphe (Lutéine 10mg + Zéaxanthine 2mg + Zn + C + E)"
        )
    if geographic_atrophy:
        return AREDS2Result(
            category=4, stage="DMLA avancée — Atrophie géographique",
            description="Plages d'atrophie géographique de l'EPR — perte visuelle centrale",
            five_year_risk="Progression bilatérale 40% à 5 ans",
            recommendation="Pegcetacoplan (Syfovre) — suivi ophtalmologique trimestriel",
            supplement_indication="AREDS2 recommandés — Lutéine 10mg + Zéaxanthine 2mg"
        )
    if large_drusen or (medium_drusen and pigment_changes):
        return AREDS2Result(
            category=3, stage="DMLA intermédiaire",
            description="Grands drusen ≥ 125 µm OU drusen moyens + altérations pigmentaires de l'EPR",
            five_year_risk="18% risque DMLA avancée à 5 ans",
            recommendation="AREDS2 (Lutéine 10mg + Zéaxanthine 2mg + Vit C 500mg + Vit E 400UI + Zn 80mg)",
            supplement_indication="Supplémentation AREDS2 indiquée — peut réduire le risque de 25%"
        )
    if medium_drusen:
        return AREDS2Result(
            category=2, stage="DMLA précoce",
            description="Drusen de taille moyenne (63–124 µm) sans altérations pigmentaires importantes",
            five_year_risk="< 5% risque DMLA avancée à 5 ans",
            recommendation="Surveillance ophtalmologique annuelle + grille d'Amsler quotidienne",
            supplement_indication="AREDS2 non recommandés à ce stade — alimentation riche en lutéine et caroténoïdes"
        )
    if small_drusen and not no_drusen:
        return AREDS2Result(
            category=1, stage="Drusen petits / signes minimes",
            description="Petits drusen < 63 µm — modifications mineures de l'EPR",
            five_year_risk="< 1% risque DMLA avancée à 5 ans",
            recommendation="Examen ophtalmologique tous les 2 ans",
            supplement_indication="Aucun supplément nécessaire — alimentation méditerranéenne"
        )
    return AREDS2Result(
        category=0, stage="Pas de DMLA",
        description="Aucun signe de dégénérescence maculaire liée à l'âge",
        five_year_risk="< 0.5%",
        recommendation="Examen ophtalmologique tous les 2 ans après 50 ans",
        supplement_indication="Non indiqué"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Cup-to-Disc Ratio (CDR) — Glaucome (AAO / EGS 2023)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CDRResult:
    cdr: float
    stage: str
    rim_loss: str
    rnfl_estimate: str
    recommendation: str
    urgency: str

def compute_cup_to_disc_ratio(
    cup_diameter: float = 0.3,   # diamètre de l'excavation (en fraction du disque)
    disc_diameter: float = 1.0,  # normalisé à 1.0
    asymmetry_index: float = 0.0,  # différence CDR entre les 2 yeux (> 0.2 = suspect)
    rnfl_thickness_um: float = 95.0,  # µm — normale ≥ 90 µm
    iop_mmhg: float = 14.0,          # Pression intraoculaire mmHg
    visual_field_md: float = 0.0,    # Mean Deviation dB (négatif = perte)
) -> CDRResult:
    """
    Cup-to-Disc Ratio (CDR) et évaluation glaucomateuse (EGS/AAO 2023).
    CDR normal < 0.5 · CDR > 0.7 = glaucome probable.
    """
    cdr = cup_diameter / disc_diameter if disc_diameter > 0 else 0.0
    cdr = round(min(max(cdr, 0.0), 1.0), 2)

    # Évaluation RNFL
    if rnfl_thickness_um >= 90:
        rnfl_status = f"Normal ({rnfl_thickness_um:.0f} µm — seuil ≥ 90 µm)"
    elif rnfl_thickness_um >= 75:
        rnfl_status = f"Amincissement précoce ({rnfl_thickness_um:.0f} µm)"
    else:
        rnfl_status = f"Amincissement sévère ({rnfl_thickness_um:.0f} µm — < 75 µm)"

    # Staging
    if cdr >= 0.8 or (cdr >= 0.7 and visual_field_md < -6):
        stage = "Glaucome avancé"
        rim = "Perte d'anneau neurorétinier > 50% — excavation majeure"
        rec = "IOP cible ≤ 12 mmHg — traitement chirurgical (trabéculectomie/stent) ou laser SLT"
        urg = "Urgente"
    elif cdr >= 0.7 or asymmetry_index > 0.2 or iop_mmhg > 21:
        stage = "Glaucome probable / précoce"
        rim = "Réduction de l'anneau neurorétinier — asymétrie excavation"
        rec = "Prostaglandines topiques — IOP cible ≤ 18 mmHg — OCT RNFL et champ visuel"
        urg = "Élevée"
    elif cdr >= 0.6:
        stage = "Suspect glaucome"
        rim = "Rapport C/D limite — surveillance recommandée"
        rec = "Périmétrie automatisée + OCT papille — contrôle IOP — suivi 6 mois"
        urg = "Modérée"
    else:
        stage = "Normal"
        rim = "Anneau neurorétinier intact — pas d'excavation pathologique"
        rec = "Examen ophtalmologique tous les 2 ans si facteurs de risque"
        urg = "Faible"

    return CDRResult(
        cdr=cdr, stage=stage, rim_loss=rim,
        rnfl_estimate=rnfl_status, recommendation=rec, urgency=urg
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Central Retinal Thickness (CRT) & Volume Maculaire — OCT (ETDRS grid)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CRTResult:
    crt_um: float
    stage: str
    edema_type: str
    volume_mm3: float
    recommendation: str
    anti_vegf_indication: bool

def compute_crt(
    crt_um: float = 260.0,       # µm — ETDRS central 1mm zone (normale 250–280 µm)
    volume_mm3: float = 8.5,     # mm³ macular cube (normale 8.0–9.5 mm³)
    subretinal_fluid: bool = False,
    intraretinal_fluid: bool = False,
    hyperreflective_foci: bool = False,
    epiretinal_membrane: bool = False,
    diabetic_context: bool = False,
    amd_context: bool = False,
) -> CRTResult:
    """
    Central Retinal Thickness — OCT Cirrus/Spectralis (DRCR.net protocoles).
    Seuil œdème : > 300 µm fovéal (DRCR.net T1, T2 études).
    """
    anti_vegf = False

    if crt_um > 400 or (intraretinal_fluid and subretinal_fluid):
        stage = "Œdème maculaire sévère"
        etype = "Fluide intra- et sous-rétinien — atteinte fovéolaire majeure"
        anti_vegf = True
        rec = "Anti-VEGF intravitréen (aflibercept/ranibizumab/faricimab) mensuel × 3 → PRN"
    elif crt_um > 320 or intraretinal_fluid:
        stage = "Œdème maculaire modéré"
        etype = "Fluide intrarétinien prédominant ± kystes"
        anti_vegf = True
        rec = "Anti-VEGF intravitréen — protocole T&E ou PRN — OCT mensuel"
    elif crt_um > 300 or subretinal_fluid:
        stage = "Œdème maculaire léger"
        etype = "Décollement séreux rétinien limité / fluide sous-rétinien"
        anti_vegf = amd_context
        rec = "Surveillance OCT mensuelle — anti-VEGF si progression"
    elif epiretinal_membrane:
        stage = "Membrane épirétinienne"
        etype = "Traction vitréo-maculaire — membrane épirétinienne"
        rec = "Vitréctomie si acuité visuelle < 5/10 ou distorsion invalidante"
    else:
        stage = "Épaisseur maculaire normale"
        etype = "Aucun signe d'œdème maculaire"
        rec = "Contrôle annuel — pas d'intervention indiquée"

    return CRTResult(
        crt_um=round(crt_um, 0),
        stage=stage,
        edema_type=etype,
        volume_mm3=round(volume_mm3, 2),
        recommendation=rec,
        anti_vegf_indication=anti_vegf
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Visual Field Index (VFI) et Mean Deviation (MD) — Glaucome (Humphrey HFA)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VFIResult:
    vfi_percent: float
    md_db: float
    psd_db: float
    stage: str
    rate_of_progression: str
    recommendation: str

def compute_visual_field(
    vfi_percent: float = 100.0,   # % de champ visuel intact (100% = normal)
    md_db: float = 0.0,           # Mean Deviation en dB (0 = normal, négatif = perte)
    psd_db: float = 1.5,          # Pattern Standard Deviation (> 2.5 dB = anormal)
    rate_db_per_year: float = 0.0,  # taux de progression dB/an (< -1 = rapide)
) -> VFIResult:
    """
    Visual Field Analysis — Humphrey Field Analyzer (HFA) 24-2 SITA-Standard.
    Hodapp-Parrish-Anderson (HPA) criteria pour staging glaucome.
    """
    # Staging HPA
    if vfi_percent < 50 or md_db < -12:
        stage = "Glaucome avancé"
        rate = "Progression rapide si > -1 dB/an"
        rec = "Chirurgie filtrante urgente — tube shunt ou trabéculectomie — IOP cible ≤ 10 mmHg"
    elif vfi_percent < 75 or md_db < -6:
        stage = "Glaucome modéré"
        rate = "Contrôle progression tous les 4 mois"
        rec = "Combinaison prostaglandines + bêtabloquant — laser SLT si pression non contrôlée"
    elif vfi_percent < 90 or md_db < -2 or psd_db > 2.5:
        stage = "Glaucome précoce"
        rate = "Contrôle champ visuel tous les 6 mois"
        rec = "Prostaglandines topiques — IOP cible = 30% réduction baseline — OCT RNFL trimestriel"
    else:
        stage = "Champ visuel normal"
        rate = "Pas de progression détectée"
        rec = "Surveillance annuelle si facteurs de risque glaucome"

    prog_desc = f"{rate_db_per_year:.1f} dB/an" if rate_db_per_year != 0 else "Non évaluable"

    return VFIResult(
        vfi_percent=round(vfi_percent, 1),
        md_db=round(md_db, 2),
        psd_db=round(psd_db, 2),
        stage=stage,
        rate_of_progression=prog_desc,
        recommendation=rec
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. International Clinical Diabetic Retinopathy (ICDR) Scale
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ICDRResult:
    level: str
    code: int
    findings: str
    follow_up: str
    treatment: str

def compute_icdr(etdrs_level: int = 10) -> ICDRResult:
    """
    ICDR Scale (Wilkinson et al., Ophthalmology 2003) — échelle simplifiée 5 niveaux.
    Corrélée avec ETDRS (American Academy of Ophthalmology).
    """
    if etdrs_level >= 61:
        return ICDRResult(
            level="Rétinopathie Diabétique Proliférante (RDP)",
            code=4,
            findings="Néovaisseaux, hémorragie vitréenne ou prérétinienne, décollement tractionnel",
            follow_up="Urgence — < 1 semaine",
            treatment="Photocoagulation panrétinienne + anti-VEGF + vitrectomie si indiqué"
        )
    if etdrs_level >= 47:
        return ICDRResult(
            level="RDNP sévère",
            code=3,
            findings="Nombreuses hémorragies, chapelet veineux, AMIR (règle 4-2-1)",
            follow_up="< 1–2 mois",
            treatment="Contrôle strict glycémie + HTA — discuter photocoagulation préventive"
        )
    if etdrs_level >= 35:
        return ICDRResult(
            level="RDNP modérée",
            code=2,
            findings="Plus que microanévrismes seuls, moins que RDNP sévère",
            follow_up="6 mois",
            treatment="Optimisation HbA1c < 7% + TA < 130/80 mmHg"
        )
    if etdrs_level >= 20:
        return ICDRResult(
            level="RDNP légère",
            code=1,
            findings="Microanévrismes seuls",
            follow_up="12 mois",
            treatment="Contrôle facteurs de risque — glycémie, lipides, PA"
        )
    return ICDRResult(
        level="Pas de rétinopathie diabétique apparente",
        code=0,
        findings="Aucun signe de rétinopathie",
        follow_up="12 mois",
        treatment="Dépistage annuel — optimisation métabolique"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Score de risque vasculaire rétinien
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetinalVascularRisk:
    risk_score: int
    risk_level: str
    findings: list[str]
    recommendation: str

def compute_vascular_risk(
    arteriovenous_nicking: bool = False,
    copper_wire: bool = False,
    silver_wire: bool = False,
    flame_hemorrhages: bool = False,
    disc_edema: bool = False,
    cotton_wool_spots: bool = False,
    venous_occlusion: bool = False,
    arterial_occlusion: bool = False,
) -> RetinalVascularRisk:
    """
    Évaluation rétinopathie hypertensive + occlusions vasculaires.
    Classification Keith-Wagener-Barker (KWB) adaptée.
    """
    score = 0
    findings = []

    if arteriovenous_nicking: score += 1; findings.append("Signe de Salus-Gunn (croisement artério-veineux)")
    if copper_wire: score += 2; findings.append("Signe du fil de cuivre (HTA grade II)")
    if silver_wire: score += 3; findings.append("Signe du fil d'argent (HTA grade III)")
    if flame_hemorrhages: score += 3; findings.append("Hémorragies en flammèches")
    if cotton_wool_spots: score += 2; findings.append("Nodules cotonneux (micro-infarctus)")
    if disc_edema: score += 4; findings.append("Œdème papillaire — HTA maligne")
    if venous_occlusion: score += 4; findings.append("Occlusion veineuse rétinienne")
    if arterial_occlusion: score += 5; findings.append("Occlusion artérielle rétinienne — urgence AVC")

    if arterial_occlusion:
        return RetinalVascularRisk(score, "Urgence absolue",
            findings, "URGENCE — AVC stroke ischémique possible — appel SAMU + IRM cérébrale")
    if score >= 6:
        return RetinalVascularRisk(score, "Sévère",
            findings, "Bilan cardiologique urgent — Holter ECG + écho cardiaque + TA 24h")
    if score >= 3:
        return RetinalVascularRisk(score, "Modérée",
            findings, "Contrôle tensionnel strict + bilan cardiovasculaire complet")
    if score >= 1:
        return RetinalVascularRisk(score, "Légère",
            findings, "Surveillance TA + traitement antihypertenseur si non contrôlé")
    return RetinalVascularRisk(0, "Normal", [], "Pas d'anomalie vasculaire rétinienne")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Score global RetinaVision — synthèse clinique
# ═══════════════════════════════════════════════════════════════════════════════

def build_retina_clinical_summary(
    prediction: str,
    confidence: float,
    img_feats: dict[str, float],
    clinical_params: dict[str, Any],
) -> dict[str, Any]:
    """
    Génère le résumé clinique complet adapté à la pathologie détectée.
    """
    p = clinical_params

    # ETDRS + ICDR (rétinopathie diabétique)
    etdrs = compute_etdrs(
        microaneurysms=int(img_feats.get("microaneurysm_count", 0)),
        hemorrhages=img_feats.get("hemorrhage_ratio", 0) > 0.02,
        hard_exudates=img_feats.get("bright_lesion_ratio", 0) > 0.05,
        soft_exudates=img_feats.get("cotton_wool_ratio", 0) > 0.01,
        venous_beading=p.get("venous_beading", False),
        irma=p.get("irma", False),
        neovascularization=p.get("neovascularization", False),
    )
    icdr = compute_icdr(etdrs.level)

    # AREDS2 (DMLA)
    is_amd = any(k in prediction for k in ("DMLA", "Maculaire", "Dégénérescence"))
    areds2 = compute_areds2(
        large_drusen=img_feats.get("drusen_ratio", 0) > 0.08 and is_amd,
        medium_drusen=img_feats.get("drusen_ratio", 0) > 0.03 and is_amd,
        small_drusen=img_feats.get("drusen_ratio", 0) > 0.01 and is_amd,
        geographic_atrophy=p.get("geographic_atrophy", False),
        neovascular_amd="humide" in prediction.lower() or p.get("neovascular_amd", False),
        pigment_changes=img_feats.get("pigment_ratio", 0) > 0.05,
        bilateral_involvement=p.get("bilateral", False),
    )

    # Cup-to-Disc (glaucome)
    cdr_est = min(0.9, 0.35 + img_feats.get("disc_pallor", 0) * 1.5)
    cup_disc = compute_cup_to_disc_ratio(
        cup_diameter=cdr_est,
        asymmetry_index=img_feats.get("asymmetry", 0) * 0.5,
        rnfl_thickness_um=max(40, 100 - img_feats.get("disc_pallor", 0) * 80),
        iop_mmhg=p.get("iop_mmhg", 15.0),
        visual_field_md=p.get("vf_md", 0.0),
    )

    # CRT / Œdème maculaire
    crt_est = 260 + img_feats.get("foveal_brightness", 0) * 200
    crt = compute_crt(
        crt_um=crt_est,
        volume_mm3=8.5 + img_feats.get("macular_volume_delta", 0),
        subretinal_fluid=img_feats.get("dark_subretinal", 0) > 0.03,
        intraretinal_fluid=img_feats.get("cystoid_ratio", 0) > 0.02,
        diabetic_context="diabétique" in prediction.lower(),
        amd_context=is_amd,
    )

    # Champ visuel
    md_est = -max(0, (img_feats.get("disc_pallor", 0) - 0.4) * 20)
    vf = compute_visual_field(
        vfi_percent=max(30, 100 + md_est * 5),
        md_db=md_est,
        psd_db=max(1.5, img_feats.get("contrast", 0.2) * 10),
    )

    # Risque vasculaire
    vasc = compute_vascular_risk(
        flame_hemorrhages=img_feats.get("hemorrhage_ratio", 0) > 0.04,
        cotton_wool_spots=img_feats.get("cotton_wool_ratio", 0) > 0.01,
        disc_edema=img_feats.get("disc_edema", 0) > 0.3,
        arteriovenous_nicking=p.get("av_nicking", False),
        venous_occlusion="Occlusion veineuse" in prediction,
        arterial_occlusion="Occlusion artérielle" in prediction,
    )

    return {
        "etdrs": {
            "level": etdrs.level,
            "stage": etdrs.stage,
            "description": etdrs.description,
            "risk": etdrs.risk_progression,
            "recommendation": etdrs.recommendation,
            "urgency": etdrs.urgency,
        },
        "icdr": {
            "level": icdr.level,
            "code": icdr.code,
            "findings": icdr.findings,
            "follow_up": icdr.follow_up,
            "treatment": icdr.treatment,
        },
        "areds2": {
            "category": areds2.category,
            "stage": areds2.stage,
            "description": areds2.description,
            "five_year_risk": areds2.five_year_risk,
            "recommendation": areds2.recommendation,
            "supplement": areds2.supplement_indication,
        },
        "cup_to_disc": {
            "cdr": cup_disc.cdr,
            "stage": cup_disc.stage,
            "rim_loss": cup_disc.rim_loss,
            "rnfl": cup_disc.rnfl_estimate,
            "recommendation": cup_disc.recommendation,
            "urgency": cup_disc.urgency,
        },
        "crt": {
            "crt_um": crt.crt_um,
            "stage": crt.stage,
            "edema_type": crt.edema_type,
            "volume_mm3": crt.volume_mm3,
            "recommendation": crt.recommendation,
            "anti_vegf": crt.anti_vegf_indication,
        },
        "visual_field": {
            "vfi_percent": vf.vfi_percent,
            "md_db": vf.md_db,
            "psd_db": vf.psd_db,
            "stage": vf.stage,
            "rate": vf.rate_of_progression,
            "recommendation": vf.recommendation,
        },
        "vascular_risk": {
            "score": vasc.risk_score,
            "level": vasc.risk_level,
            "findings": vasc.findings,
            "recommendation": vasc.recommendation,
        },
        "overall_urgency": _determine_overall_urgency(etdrs.urgency, cup_disc.urgency, vasc.risk_level, crt.anti_vegf_indication),
    }


def _determine_overall_urgency(etdrs_urg: str, cdr_urg: str, vasc_lvl: str, anti_vegf: bool) -> str:
    priority = {"Urgence absolue": 5, "Urgente": 4, "Élevée": 3, "Modérée": 2, "Faible": 1}
    max_score = max(
        priority.get(etdrs_urg, 1),
        priority.get(cdr_urg, 1),
        5 if vasc_lvl == "Urgence absolue" else priority.get(vasc_lvl, 1),
        3 if anti_vegf else 1,
    )
    return {5: "Urgence absolue", 4: "Urgente", 3: "Élevée", 2: "Modérée", 1: "Faible"}.get(max_score, "Faible")
