"""
NeuroVision AI — Scores cliniques neurologiques
================================================
NIHSS · mRS · ASPECTS · GCS · Marshall · MMSE · MoCA · CDR
Hoehn-Yahr · UPDRS · EDSS · Hunt-Hess · Fisher · ICH Score.

Sources :
  - American Academy of Neurology (AAN) 2023
  - European Academy of Neurology (EAN) 2023
  - AHA/ASA Stroke Guidelines 2023
  - NINDS Stroke Scale · Marshall Classification (JNS 1991)
  - Alzheimer's Association DSM-5 Criteria 2022
  - Multiple Sclerosis International Federation (MSIF) 2023
  - UPDRS — MDS-UPDRS v3 · Hoehn & Yahr 1967
  - The Lancet Neurology · Neurology Journal · PubMed
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# 1. NIH Stroke Scale (NIHSS) — AVC (NINDS / AHA 2023)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NIHSSResult:
    total: int
    stage: str
    thrombolysis_eligible: bool
    thrombectomy_window: str
    door_to_needle_target: str
    recommendation: str

def compute_nihss(
    consciousness: int = 0,        # 0–3
    gaze: int = 0,                 # 0–2
    visual_fields: int = 0,        # 0–3
    facial_palsy: int = 0,         # 0–3
    motor_arm_left: int = 0,       # 0–4
    motor_arm_right: int = 0,      # 0–4
    motor_leg_left: int = 0,       # 0–4
    motor_leg_right: int = 0,      # 0–4
    limb_ataxia: int = 0,          # 0–2
    sensory: int = 0,              # 0–2
    language: int = 0,             # 0–3
    dysarthria: int = 0,           # 0–2
    extinction: int = 0,           # 0–2
) -> NIHSSResult:
    """
    NIH Stroke Scale — 15 items · Score 0–42.
    AHA/ASA Acute Stroke Guidelines 2023.
    """
    total = (consciousness + gaze + visual_fields + facial_palsy +
             motor_arm_left + motor_arm_right + motor_leg_left + motor_leg_right +
             limb_ataxia + sensory + language + dysarthria + extinction)
    total = max(0, min(total, 42))

    if total == 0:
        stage = "Normal / Minime"
        thrombo = False
        rec = "Observation — anticoagulation orale si FA — bilan étiologique AVC"
    elif total <= 4:
        stage = "AVC léger"
        thrombo = True
        rec = "Thrombolyse IV si < 4h30 (si éligible) — aspirine 300 mg — USIC"
    elif total <= 15:
        stage = "AVC modéré"
        thrombo = True
        rec = "Thrombolyse IV alteplase 0.9 mg/kg (max 90mg) + thrombectomie si occlusion proximale"
    elif total <= 20:
        stage = "AVC sévère"
        thrombo = True
        rec = "Thrombectomie mécanique URGENTE (si < 24h occlusion ICA/M1) — NIHSS > 10"
    else:
        stage = "AVC très sévère"
        thrombo = True
        rec = "Thrombectomie + soins intensifs neurologique — surveillance ICP si œdème malin"

    return NIHSSResult(
        total=total, stage=stage,
        thrombolysis_eligible=thrombo,
        thrombectomy_window="0–24h si occlusion proximale (DAWN/DEFUSE-3)" if total >= 6 else "Non indiquée",
        door_to_needle_target="< 60 minutes (AHA 2023 cible < 30 min centres experts)",
        recommendation=rec,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Modified Rankin Scale (mRS) — Handicap neurologique
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class mRSResult:
    score: int
    level: str
    description: str
    autonomy: str
    clinical_use: str

def compute_mrs(score: int = 0) -> mRSResult:
    """
    Modified Rankin Scale (mRS) — Outcome AVC / neurologique.
    Score 0–6 · Bon outcome défini comme mRS 0–2 (AHA/ASA 2023).
    """
    score = max(0, min(score, 6))
    data = {
        0: ("Aucun symptôme", "Pas de handicap", "Indépendant total",
            "Outcome excellent — critère primaire essais thérapeutiques AVC"),
        1: ("Symptômes mineurs sans handicap", "Handicap minime",
            "Indépendant — retour activités préalables",
            "Bon outcome (mRS 0–2 = succès thrombolyse)"),
        2: ("Handicap léger", "Légèrement dépendant",
            "Soins personnels indépendant — aide pour activités complexes",
            "Bon outcome — seuil cliniquement signif. essais AVC"),
        3: ("Handicap modéré", "Dépendant partiel",
            "Aide pour la marche — autonome pour soins personnels",
            "Outcome défavorable — besoin de rééducation"),
        4: ("Handicap modérément sévère", "Dépendant", "Aide constante pour soins personnels",
            "Placement en structure spécialisée souvent nécessaire"),
        5: ("Handicap sévère", "Grabataire", "Soins continus 24h — lit ou fauteuil roulant",
            "Soins palliatifs / intensifs longue durée"),
        6: ("Décès", "Décédé", "—", "Critère d'arrêt essais thérapeutiques"),
    }
    lv, desc, aut, clin = data.get(score, data[6])
    return mRSResult(score=score, level=lv, description=desc, autonomy=aut, clinical_use=clin)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ASPECTS Score — AVC ischémique MCA (Alberta Stroke Programme)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ASPECTSResult:
    score: int
    outcome: str
    thrombectomy_benefit: str
    recommendation: str

def compute_aspects(
    caudate: bool = False,
    putamen: bool = False,
    internal_capsule: bool = False,
    insular_cortex: bool = False,
    m1: bool = False,  # MCA territoire M1–M6
    m2: bool = False,
    m3: bool = False,
    m4: bool = False,
    m5: bool = False,
    m6: bool = False,
) -> ASPECTSResult:
    """
    ASPECTS Score (Barber et al. 2000) — 10 régions territoire ACM.
    Score 10 = normal · 0 = infarctus MCA complet.
    Thrombectomie bénéfice si ASPECTS ≥ 6 (ESCAPE / SWIFT-PRIME / EXTENDS-IA).
    """
    lesion_count = sum([caudate, putamen, internal_capsule, insular_cortex,
                        m1, m2, m3, m4, m5, m6])
    score = 10 - lesion_count

    if score >= 8:
        outcome = "Bon pronostic"
        benefit = "Bénéfice thrombectomie maximal (ASPECTS ≥ 8)"
        rec = "Thrombectomie mécanique urgente recommandée — excellent pronostic fonctionnel"
    elif score >= 6:
        outcome = "Pronostic intermédiaire"
        benefit = "Thrombectomie bénéfique (ASPECTS 6–7 — classe IIa AHA 2023)"
        rec = "Thrombectomie si occlusion proximale — évaluation cas par cas"
    elif score >= 4:
        outcome = "Pronostic réservé"
        benefit = "Thrombectomie discutable (ASPECTS < 6 — données limitées)"
        rec = "Discussion RCP neurovasculaire — thrombectomie non recommandée en routine"
    else:
        outcome = "Pronostic très défavorable"
        benefit = "Thrombectomie non recommandée (infarctus étendu)"
        rec = "Soins de support — évaluation chirurgie décompressive si œdème malin"

    return ASPECTSResult(score=score, outcome=outcome,
                         thrombectomy_benefit=benefit, recommendation=rec)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Glasgow Coma Scale (GCS) — Traumatisme crânien (Teasdale & Jennett 1974)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GCSResult:
    total: int
    eye: int
    verbal: int
    motor: int
    severity: str
    intubation_threshold: bool
    icp_monitoring: bool
    recommendation: str

def compute_gcs(eye: int = 4, verbal: int = 5, motor: int = 6) -> GCSResult:
    """
    Glasgow Coma Scale — Score 3–15.
    Sévérité TBI : Léger 13–15 · Modéré 9–12 · Sévère ≤ 8.
    Intubation si GCS ≤ 8 (protection voies aériennes — ATLS ACSCOT 2018).
    """
    eye    = max(1, min(eye, 4))
    verbal = max(1, min(verbal, 5))
    motor  = max(1, min(motor, 6))
    total  = eye + verbal + motor

    if total >= 13:
        severity = "TBI léger"
        intub = False
        icp   = False
        rec   = "Scanner CT cérébral urgent — observation 24h — pas d'intubation systématique"
    elif total >= 9:
        severity = "TBI modéré"
        intub = False
        icp   = False
        rec   = "Scanner CT + IRM — USIC neurologique — réévaluation neurologique toutes les heures"
    else:
        severity = "TBI sévère"
        intub = True
        icp   = True
        rec   = ("INTUBATION URGENTE (GCS ≤ 8) — monitorage PIC — neurochirurgie immédiate si "
                 "hématome expansif — hyperventilation transitoire si engagement")

    return GCSResult(
        total=total, eye=eye, verbal=verbal, motor=motor,
        severity=severity, intubation_threshold=intub,
        icp_monitoring=icp, recommendation=rec,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Marshall Classification — TBI CT scan (Marshall et al. 1991)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MarshallResult:
    grade: int
    label: str
    description: str
    mortality_risk: str
    icp_risk: str
    recommendation: str

def compute_marshall(
    cisterns_compressed: bool = False,
    midline_shift_mm: float = 0.0,
    high_density_lesion: bool = False,
    volume_ml: float = 0.0,
    evacuated_lesion: bool = False,
    non_evacuated_lesion: bool = False,
) -> MarshallResult:
    """
    Marshall CT Classification — TBI (Journal of Neurosurgery 1991).
    Grades I–VI (I = normal → VI = lésions non évacuées ≥ 25 mL).
    """
    if evacuated_lesion:
        return MarshallResult(
            grade=5, label="Lésion évacuée",
            description="Toute lésion chirurgicale évacuée",
            mortality_risk="Variable selon lésion",
            icp_risk="Réduit après évacuation",
            recommendation="Post-opératoire neurochirurgie — monitoring PIC"
        )
    if non_evacuated_lesion or volume_ml >= 25:
        return MarshallResult(
            grade=6, label="Lésion non évacuée ≥ 25 mL",
            description="Haute densité ou mixte > 25 mL non évacuée",
            mortality_risk="> 50%",
            icp_risk="Très élevé",
            recommendation="Discussion neurochirurgicale urgente — monitoring PIC obligatoire"
        )
    if high_density_lesion and volume_ml >= 25:
        return MarshallResult(
            grade=4, label="Lésion focale IV",
            description="Haute densité ou mixte > 25 mL",
            mortality_risk="40–50%",
            icp_risk="Très élevé",
            recommendation="Évacuation neurochirurgicale urgente"
        )
    if (cisterns_compressed or midline_shift_mm > 5) and not high_density_lesion:
        return MarshallResult(
            grade=3, label="Tuméfaction diffuse III",
            description="Citernes périmésencéphaliques comprimées/absentes — shift ≤ 5 mm",
            mortality_risk="34%",
            icp_risk="Élevé",
            recommendation="Monitoring PIC obligatoire — osmothérapie — USIC neurologique"
        )
    if midline_shift_mm > 5:
        return MarshallResult(
            grade=2, label="Lésion diffuse II",
            description="Citernes visibles — midline > 5 mm — pas de haute densité > 25 mL",
            mortality_risk="14%",
            icp_risk="Modéré",
            recommendation="Scanner CT répété à 6h — surveillance neurologique — PIC si aggravation"
        )
    return MarshallResult(
        grade=1, label="Lésion diffuse I",
        description="Pas de pathologie visible au CT",
        mortality_risk="9.6%",
        icp_risk="Faible",
        recommendation="Observation 24h — TDM répété si aggravation GCS"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MMSE — Mini Mental State Examination (Folstein 1975)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MMSEResult:
    score: int
    severity: str
    alzheimer_stage: str
    recommendation: str

def compute_mmse(score: int = 30) -> MMSEResult:
    """
    MMSE (Folstein 1975) — Score 0–30.
    Seuil démence : < 24 (Se 87% / Sp 82% — Crum 1993).
    """
    score = max(0, min(score, 30))
    if score >= 27:
        return MMSEResult(score, "Normal", "Pas de démence",
                          "Suivi annuel si facteurs de risque — MoCA si suspicion")
    if score >= 24:
        return MMSEResult(score, "Léger (borderline)",
                          "Mild Cognitive Impairment (MCI) possible",
                          "IRM cérébrale + biomarqueurs LCR + PET amyloïde — neuropsychologue")
    if score >= 18:
        return MMSEResult(score, "Démence légère",
                          "Alzheimer probable stade léger",
                          "Inhibiteurs ChE (donépézil/rivastigmine) + stimulation cognitive")
    if score >= 10:
        return MMSEResult(score, "Démence modérée",
                          "Alzheimer modéré — perte autonomie significative",
                          "Mémantine 20 mg/j + EHPAD évaluation + soutien aidants")
    return MMSEResult(score, "Démence sévère",
                      "Alzheimer sévère — dépendance totale",
                      "Soins palliatifs — unité mémoire — équipe mobile gériatrie")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MoCA — Montreal Cognitive Assessment (Nasreddine 2005)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MoCAResult:
    score: int
    interpretation: str
    domains_affected: list[str]
    recommendation: str

def compute_moca(
    score: int = 30,
    visuospatial: int = 5,
    naming: int = 3,
    memory_recall: int = 5,
    attention: int = 6,
    language: int = 3,
    abstraction: int = 2,
    orientation: int = 6,
) -> MoCAResult:
    """
    MoCA — Seuil normal ≥ 26/30 (sensitivity 90% MCI — Nasreddine 2005).
    Ajouter 1 point si < 12 ans d'éducation.
    """
    score = max(0, min(score, 30))
    domains: list[str] = []
    if visuospatial < 4: domains.append("Visuospatial/Exécutif")
    if naming < 2:       domains.append("Dénomination")
    if memory_recall < 3: domains.append("Mémoire")
    if attention < 5:    domains.append("Attention")
    if language < 2:     domains.append("Langage")
    if orientation < 5:  domains.append("Orientation")

    if score >= 26:
        interp = "Normal"
        rec = "Pas de trouble cognitif — suivi annuel si âge > 65 ans"
    elif score >= 22:
        interp = "Mild Cognitive Impairment (MCI)"
        rec = "IRM + bilan neuropsychologique complet — biomarqueurs Alzheimer si disponibles"
    elif score >= 18:
        interp = "Déficit cognitif léger à modéré"
        rec = "Consultation neurologie — inhibiteurs ChE si Alzheimer confirmé"
    else:
        interp = "Déficit cognitif sévère"
        rec = "Évaluation urgente — placement assisté — soins spécialisés démence"

    return MoCAResult(score=score, interpretation=interp,
                      domains_affected=domains or ["Aucun déficit spécifique détecté"],
                      recommendation=rec)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CDR — Clinical Dementia Rating (Hughes 1982 / Morris 1993)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CDRResult:
    cdr: float
    stage: str
    sb_score: float
    progression_rate: str
    recommendation: str

def compute_cdr(
    memory: float = 0,          # 0, 0.5, 1, 2, 3
    orientation: float = 0,
    judgment: float = 0,
    community: float = 0,
    home_hobbies: float = 0,
    personal_care: float = 0,
) -> CDRResult:
    """
    Clinical Dementia Rating — Score global 0/0.5/1/2/3.
    Sum of Boxes (SoB) = 0–18 (sensible progression Alzheimer — Morris 1993).
    """
    items = [memory, orientation, judgment, community, home_hobbies, personal_care]
    sb = sum(items)

    # CDR global selon règle Memory-dominante (Hughes 1982)
    mem  = memory
    rest = [orientation, judgment, community, home_hobbies, personal_care]
    majority = sorted(rest)
    cdr = mem  # mémoire est le domaine principal

    # Ajustement si majorité différente
    above = sum(1 for x in rest if x > mem)
    below = sum(1 for x in rest if x < mem)
    if above >= 3: cdr = min(mem + 0.5, 3.0)
    if below >= 3: cdr = max(mem - 0.5, 0.0)

    cdr_str = {0: "0", 0.5: "0.5", 1: "1", 1.5: "1.5", 2: "2", 3: "3"}.get(cdr, "1")

    if cdr == 0:
        stage = "Normal — pas de démence"
        prog = "< 1% / an"
        rec = "Surveillance annuelle si > 70 ans"
    elif cdr == 0.5:
        stage = "MCI / Démence très légère (douteuse)"
        prog = "10–15% conversion Alzheimer/an"
        rec = "IRM + PET amyloïde + LCR Tau/Abeta — réévaluation 6 mois"
    elif cdr == 1:
        stage = "Démence légère"
        prog = "2–4 points SoB/an"
        rec = "Inhibiteurs cholinestérases + programme stimulation cognitive"
    elif cdr == 2:
        stage = "Démence modérée"
        prog = "3–5 points SoB/an"
        rec = "Mémantine + aide à domicile ou EHPAD — évaluation capacité juridique"
    else:
        stage = "Démence sévère"
        prog = "Stade terminal"
        rec = "Soins palliatifs — alimentation adaptée — prévention escarres"

    return CDRResult(cdr=cdr, stage=stage, sb_score=round(sb, 1),
                     progression_rate=prog, recommendation=rec)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Hoehn & Yahr Scale + UPDRS — Parkinson (MDS 2023)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParkinsonResult:
    hy_stage: float
    hy_description: str
    updrs_iii: int
    updrs_severity: str
    dbs_candidate: bool
    recommendation: str

def compute_parkinson_scales(
    hy_stage: float = 1.0,
    updrs_iii_score: int = 0,       # MDS-UPDRS Part III (0–132)
    tremor_dominant: bool = True,
    bradykinesia: bool = False,
    postural_instability: bool = False,
    falls: bool = False,
    bilateral: bool = False,
) -> ParkinsonResult:
    """
    Hoehn & Yahr Scale + MDS-UPDRS Part III (Movement Disorder Society 2023).
    DBS (Deep Brain Stimulation) candidat si UPDRS III > 30 + levodopa réponse > 33%.
    """
    hy_desc = {
        1.0: "Unilatéral — pas d'instabilité posturale",
        1.5: "Unilatéral + axial",
        2.0: "Bilatéral — pas d'atteinte équilibre",
        2.5: "Bilatéral léger — récupération au pull-test",
        3.0: "Bilatéral modéré — instabilité posturale — autonome",
        4.0: "Sévèrement handicapé — peut marcher/se tenir debout sans aide",
        5.0: "Fauteuil roulant ou alité — assistance totale",
    }.get(hy_stage, "Stade inconnu")

    if updrs_iii_score < 20:
        u_sev = "Léger"
    elif updrs_iii_score < 40:
        u_sev = "Modéré"
    elif updrs_iii_score < 60:
        u_sev = "Sévère"
    else:
        u_sev = "Très sévère"

    dbs = hy_stage >= 3 and updrs_iii_score >= 30

    if hy_stage <= 2:
        rec = "Lévodopa + carbidopa ou agonistes dopaminergiques — kinésithérapie — sport"
    elif hy_stage <= 3:
        rec = "Optimisation lévodopa — évaluation DBS si fluctuations — Apomorphine si indiqué"
    elif hy_stage <= 4:
        rec = "DBS STN/GPi si éligible — pompe apomorphine — aide à domicile renforcée"
    else:
        rec = "Soins palliatifs — nursing — prévention complications — alimentation adaptée"

    return ParkinsonResult(
        hy_stage=hy_stage, hy_description=hy_desc,
        updrs_iii=updrs_iii_score, updrs_severity=u_sev,
        dbs_candidate=dbs, recommendation=rec,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 10. EDSS — Expanded Disability Status Scale — SEP (Kurtzke 1983)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EDSSResult:
    score: float
    stage: str
    walking_ability: str
    treatment_indication: str
    monitoring: str

def compute_edss(
    score: float = 0.0,
    requires_cane: bool = False,
    walking_distance_m: float = 500.0,
    wheelchair: bool = False,
    bedridden: bool = False,
) -> EDSSResult:
    """
    EDSS (Kurtzke 1983) — Score 0–10 par pas de 0.5.
    Référence traitement MS — EMA/FDA endpoint primaire.
    """
    score = round(max(0.0, min(score, 10.0)) * 2) / 2

    if bedridden or score >= 9.5:
        stage = "Totalement dépendant — alité"
        walk  = "Aucune marche possible"
        treat = "Soins palliatifs — symptomatique"
        mon   = "Équipe MS dédiée"
    elif wheelchair or score >= 7.0:
        stage = "Handicap sévère — fauteuil roulant"
        walk  = "< 5 m avec aide"
        treat = "Natalizumab / Alemtuzumab / Ocrelizumab si RRMS"
        mon   = "IRM cérébrale+médullaire annuelle"
    elif requires_cane or score >= 6.0:
        stage = "Aide unilatérale requise pour 100m"
        walk  = f"{walking_distance_m:.0f} m avec canne"
        treat = "Traitements de fond hautement efficaces (HET)"
        mon   = "IRM tous les 6 mois + surveillance NMO-IgG"
    elif score >= 4.0:
        stage = "Handicap modéré — marche sans aide"
        walk  = f"> {walking_distance_m:.0f} m sans aide"
        treat = "Traitements de fond modérément efficaces"
        mon   = "IRM annuelle — Visita MS neurologique"
    elif score >= 2.0:
        stage = "Handicap léger"
        walk  = "Marche normale — déficit fonctionnel minime"
        treat = "Interférons béta / Glatiramère / Diméthylfumarate / Tériflunomide"
        mon   = "IRM annuelle — EDSS tous les 3 mois"
    else:
        stage = "Pas de handicap"
        walk  = "Marche normale"
        treat = "Surveillance — Interferon béta si poussée unique CIS"
        mon   = "IRM cérébrale initiale + suivi annuel"

    return EDSSResult(score=score, stage=stage, walking_ability=walk,
                      treatment_indication=treat, monitoring=mon)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Hunt & Hess Scale + Fisher Scale — Hémorragie Sous-Arachnoïdienne
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HSAResult:
    hunt_hess: int
    hunt_hess_desc: str
    fisher: int
    fisher_desc: str
    vasospasm_risk: str
    recommendation: str

def compute_hsa_scores(
    hunt_hess: int = 1,
    fisher: int = 1,
) -> HSAResult:
    """
    Hunt & Hess Scale + Fisher Scale — HSA par rupture anévrismale.
    Risque vasospasme : Fisher 3–4 = très élevé (Mayer 2005).
    """
    hh_map = {
        1: "Asymptomatique ou céphalée légère",
        2: "Céphalée modérée à sévère — rigidité nuque — pas de déficit neurologique",
        3: "Somnolence / Confusion / Déficit focal léger",
        4: "Stupeur — hémiparésie modérée à sévère — rigidité de décérébration",
        5: "Coma — rigidité de décérébration — moribond",
    }
    fisher_map = {
        1: "Pas de sang — risque vasospasme très faible",
        2: "Sang diffus < 1 mm — risque faible",
        3: "Caillot localisé ou couche ≥ 1 mm — RISQUE ÉLEVÉ vasospasme",
        4: "Hématome intraparenchymateux ou intraventriculaire",
    }
    hh   = max(1, min(hunt_hess, 5))
    fis  = max(1, min(fisher, 4))
    hh_d = hh_map.get(hh, "—")
    fi_d = fisher_map.get(fis, "—")

    vasosp = {1: "< 5%", 2: "5–15%", 3: "30–40%", 4: "25–35%"}.get(fis, "—")
    mort = {1: "< 5%", 2: "10–15%", 3: "30%", 4: "50–60%", 5: "> 70%"}.get(hh, "—")

    if hh <= 2:
        rec = f"Clip/coil anévrisme dans 24–72h — nimodipine 60 mg/4h — mortalité {mort}"
    elif hh <= 3:
        rec = f"Traitement endovasculaire si possible < 24h — USI — nimodipine — mortalité {mort}"
    else:
        rec = f"Réanimation — traitement anévrisme différé si stabilisation — mortalité {mort}"

    return HSAResult(
        hunt_hess=hh, hunt_hess_desc=hh_d,
        fisher=fis, fisher_desc=fi_d,
        vasospasm_risk=vasosp, recommendation=rec
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 12. ICH Score — Hémorragie IntraCérébrale (Hemphill 2001)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ICHScoreResult:
    total: int
    mortality_30d: str
    good_outcome_probability: str
    recommendation: str

def compute_ich_score(
    gcs: int = 15,
    volume_ml: float = 10.0,
    ivh: bool = False,          # hémorragie intraventriculaire
    infratentorial: bool = False,
    age_over_80: bool = False,
) -> ICHScoreResult:
    """
    ICH Score (Hemphill 2001) — Prédiction mortalité J30 hémorragie intracérébrale.
    Score 0–6.
    """
    pts = 0
    # GCS : 3–4 = 2pts ; 5–12 = 1pt ; 13–15 = 0pt
    if gcs <= 4:    pts += 2
    elif gcs <= 12: pts += 1
    # Volume : ≥ 30 mL = 1pt
    if volume_ml >= 30: pts += 1
    # IVH : 1pt
    if ivh: pts += 1
    # Infratentorial : 1pt
    if infratentorial: pts += 1
    # Age ≥ 80 ans : 1pt
    if age_over_80: pts += 1

    mort_map = {0: "0%", 1: "13%", 2: "26%", 3: "72%", 4: "97%", 5: "100%", 6: "100%"}
    good_map = {0: "72%", 1: "63%", 2: "40%", 3: "15%", 4: "0%", 5: "0%", 6: "0%"}

    mort    = mort_map.get(pts, "—")
    good_ou = good_map.get(pts, "—")

    if pts <= 1:
        rec = f"USIC neurologique — TA cible < 140 mmHg (AHA 2015) — mortalité J30 {mort}"
    elif pts <= 2:
        rec = f"Neurochirurgie si localisation favorable + volume > 30 mL — mortalité J30 {mort}"
    else:
        rec = f"Pronostic très sévère — discussion soins palliatifs précoces — mortalité J30 {mort}"

    return ICHScoreResult(total=pts, mortality_30d=mort,
                          good_outcome_probability=good_ou, recommendation=rec)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Synthèse clinique globale NeuroVision
# ═══════════════════════════════════════════════════════════════════════════════

def build_neuro_clinical_summary(
    prediction: str,
    confidence: float,
    img_feats: dict[str, float],
    clinical_params: dict[str, Any],
) -> dict[str, Any]:
    """
    Génère le résumé clinique complet adapté à la pathologie neurologique détectée.
    """
    p = clinical_params

    # NIHSS (AVC)
    nihss = compute_nihss(
        consciousness=int(img_feats.get("lesion_severity", 0) * 2),
        motor_arm_left=int(img_feats.get("left_hemisphere_ratio", 0) * 4),
        motor_arm_right=int(img_feats.get("right_hemisphere_ratio", 0) * 4),
        language=int(img_feats.get("left_hemisphere_ratio", 0) * 2),
    )

    # mRS
    mrs_score = min(5, int(nihss.total / 8))
    mrs = compute_mrs(mrs_score)

    # ASPECTS
    aspects = compute_aspects(
        caudate=img_feats.get("caudate_lesion", 0) > 0.3,
        putamen=img_feats.get("basal_ganglia_density", 0) < 0.35,
        insular_cortex=img_feats.get("left_hemisphere_ratio", 0) > 0.15,
        m1=img_feats.get("cortex_lesion_ratio", 0) > 0.05,
        m2=img_feats.get("cortex_lesion_ratio", 0) > 0.10,
    )

    # GCS
    gcs_total = max(3, 15 - int(img_feats.get("lesion_severity", 0) * 8))
    e = max(1, min(4, 4 - int(img_feats.get("lesion_severity", 0) * 3)))
    v = max(1, min(5, 5 - int(img_feats.get("lesion_severity", 0) * 3)))
    m = max(1, min(6, 6 - int(img_feats.get("lesion_severity", 0) * 4)))
    gcs = compute_gcs(eye=e, verbal=v, motor=m)

    # Marshall (traumatisme)
    marshall = compute_marshall(
        cisterns_compressed=img_feats.get("lesion_severity", 0) > 0.5,
        midline_shift_mm=img_feats.get("midline_shift_mm", 0.0),
        high_density_lesion=img_feats.get("hyperdense_ratio", 0) > 0.05,
        volume_ml=img_feats.get("lesion_volume_ml", 0.0),
    )

    # MMSE (Alzheimer)
    mmse_est = max(0, int(30 - img_feats.get("atrophy_ratio", 0) * 25))
    mmse = compute_mmse(mmse_est)

    # MoCA
    moca_est = max(0, int(30 - img_feats.get("atrophy_ratio", 0) * 20))
    moca = compute_moca(moca_est)

    # CDR
    cdr_mem = min(3, img_feats.get("hippocampal_atrophy", 0) * 3)
    cdr = compute_cdr(memory=cdr_mem, orientation=cdr_mem * 0.7,
                      judgment=cdr_mem * 0.6, home_hobbies=cdr_mem * 0.8)

    # Parkinson
    hy = min(5, max(1, img_feats.get("substantia_nigra_loss", 0) * 5))
    pk = compute_parkinson_scales(
        hy_stage=hy,
        updrs_iii_score=int(hy * 15),
        bilateral=hy >= 2,
        postural_instability=hy >= 3,
    )

    # EDSS (SEP)
    edss_score = min(10, img_feats.get("white_matter_lesion_ratio", 0) * 30)
    edss = compute_edss(
        score=edss_score,
        requires_cane=edss_score >= 6.0,
        walking_distance_m=max(50, 500 - edss_score * 40),
        wheelchair=edss_score >= 7.0,
    )

    # ICH Score
    ich = compute_ich_score(
        gcs=gcs.total,
        volume_ml=img_feats.get("lesion_volume_ml", 0.0),
        ivh=img_feats.get("ventricle_hemorrhage", 0) > 0.2,
        infratentorial="cérébelleux" in prediction.lower() or "tronc" in prediction.lower(),
        age_over_80=p.get("age_over_80", False),
    )

    return {
        "nihss": {
            "total": nihss.total, "stage": nihss.stage,
            "thrombolysis": nihss.thrombolysis_eligible,
            "thrombectomy": nihss.thrombectomy_window,
            "recommendation": nihss.recommendation,
        },
        "mrs": {
            "score": mrs.score, "level": mrs.level,
            "autonomy": mrs.autonomy, "clinical_use": mrs.clinical_use,
        },
        "aspects": {
            "score": aspects.score, "outcome": aspects.outcome,
            "thrombectomy_benefit": aspects.thrombectomy_benefit,
            "recommendation": aspects.recommendation,
        },
        "gcs": {
            "total": gcs.total, "eye": gcs.eye, "verbal": gcs.verbal, "motor": gcs.motor,
            "severity": gcs.severity, "intubation": gcs.intubation_threshold,
            "icp_monitoring": gcs.icp_monitoring, "recommendation": gcs.recommendation,
        },
        "marshall": {
            "grade": marshall.grade, "label": marshall.label,
            "description": marshall.description, "mortality": marshall.mortality_risk,
            "icp_risk": marshall.icp_risk, "recommendation": marshall.recommendation,
        },
        "mmse": {
            "score": mmse.score, "severity": mmse.severity,
            "stage": mmse.alzheimer_stage, "recommendation": mmse.recommendation,
        },
        "moca": {
            "score": moca.score, "interpretation": moca.interpretation,
            "domains": moca.domains_affected, "recommendation": moca.recommendation,
        },
        "cdr": {
            "cdr": cdr.cdr, "stage": cdr.stage, "sb": cdr.sb_score,
            "progression": cdr.progression_rate, "recommendation": cdr.recommendation,
        },
        "parkinson": {
            "hy_stage": pk.hy_stage, "hy_desc": pk.hy_description,
            "updrs_iii": pk.updrs_iii, "severity": pk.updrs_severity,
            "dbs_candidate": pk.dbs_candidate, "recommendation": pk.recommendation,
        },
        "edss": {
            "score": edss.score, "stage": edss.stage,
            "walking": edss.walking_ability, "treatment": edss.treatment_indication,
            "monitoring": edss.monitoring,
        },
        "ich_score": {
            "total": ich.total, "mortality_30d": ich.mortality_30d,
            "good_outcome": ich.good_outcome_probability,
            "recommendation": ich.recommendation,
        },
        "overall_urgency": _neuro_urgency(prediction, nihss.total, gcs.total, img_feats),
    }


def _neuro_urgency(prediction: str, nihss: int, gcs: int, feats: dict) -> str:
    p = prediction.lower()
    if any(k in p for k in ("artériel", "artérielle", "occlusion artérielle")):
        return "Urgence absolue"
    if any(k in p for k in ("avc", "ischémique", "hémorragique", "sous-arachnoïdienne", "hématome épidural")):
        return "Urgente"
    if gcs <= 8 or nihss >= 15:
        return "Urgente"
    if any(k in p for k in ("glioblastome", "traumatisme", "hématome")):
        return "Élevée"
    if any(k in p for k in ("tumeur", "anévrisme", "hydrocéphal")):
        return "Élevée"
    if any(k in p for k in ("alzheimer", "parkinson", "sep", "épilepsie")):
        return "Modérée"
    return "Faible"
