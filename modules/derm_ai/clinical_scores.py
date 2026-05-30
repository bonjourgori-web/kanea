"""
DermAI — Scores cliniques dermatologiques
==========================================
ABCDE, Breslow, Clark, TNM AJCC 8e, PASI, BSA, SCORAD, EASI, GAGS, VASI, VIDA.

Sources : AAD 2023, EADV 2022, AJCC 8e édition, WHO Skin Diseases.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# ABCDE Rule — Dépistage mélanome (AAD)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ABCDEResult:
    total_score: float
    components: dict[str, Any]
    risk_level: str
    malignancy_probability: str
    recommendation: str

def compute_abcde(
    asymmetry: int = 0,         # 0=symétrique, 1=1 axe, 2=2 axes
    border: int = 0,            # 0=régulier, 1=légèrement irrégulier, 2=très irrégulier
    color_variations: int = 1,  # nombre de teintes différentes (1–6)
    diameter_mm: float = 5.0,
    evolution: bool = False,    # changement récent documenté
) -> ABCDEResult:
    """
    Règle ABCDE (Friedman et al.) — Score dermoscopique mélanome.
    TDS (Total Dermoscopy Score) = A×1.3 + B×0.1×(segments) + C×0.5×(couleurs) + D×0.5
    """
    # Asymmetry : 0–2
    a_score = asymmetry * 1.3

    # Border : irrégularité comptée en segments (0–8)
    border_segments = border * 4
    b_score = border_segments * 0.1

    # Color : nombre de teintes (blanc, rouge, brun clair, brun foncé, bleu-gris, noir)
    c_score = min(color_variations, 6) * 0.5

    # Diameter
    d_score = 0.5 if diameter_mm >= 6.0 else 0.0

    # Evolution (non dans TDS original mais critère clinique clé)
    evolution_flag = evolution

    tds = a_score + b_score + c_score + d_score

    if tds < 4.75:
        risk = "Faible"
        prob = "< 5%"
        rec  = "Surveillance dermoscopique annuelle — photographie de référence"
    elif tds < 5.45:
        risk = "Modéré"
        prob = "5–25%"
        rec  = "Suivi dermoscopique 3–6 mois ou exérèse diagnostique à discuter"
    else:
        risk = "Élevé"
        prob = "> 25%"
        rec  = "Exérèse chirurgicale diagnostique recommandée — marges larges"

    if evolution_flag:
        risk = "Élevé"
        prob = "> 30% (évolution documentée)"
        rec  = "BIOPSIE EXCISIONNELLE URGENTE — évolution = critère de malignité majeur"

    return ABCDEResult(
        total_score=round(tds, 2),
        components={
            "Asymétrie (A)":      {"score": round(a_score, 2), "valeur": f"{asymmetry}/2"},
            "Bords (B)":          {"score": round(b_score, 2), "valeur": f"{border_segments} segments"},
            "Couleurs (C)":       {"score": round(c_score, 2), "valeur": f"{color_variations} teinte(s)"},
            "Diamètre (D)":       {"score": d_score,           "valeur": f"{diameter_mm:.1f} mm"},
            "Évolution (E)":      {"score": 1 if evolution_flag else 0, "valeur": "Oui" if evolution_flag else "Non"},
        },
        risk_level=risk,
        malignancy_probability=prob,
        recommendation=rec,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Breslow Thickness + Clark Level — Mélanome
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BreslowClarkResult:
    breslow_mm: float
    breslow_category: str
    clark_level: int
    clark_description: str
    t_stage: str
    five_year_survival: str
    sentinel_node_biopsy: bool
    surgical_margins_cm: float

def compute_breslow_clark(
    breslow_mm: float = 0.5,
    clark_level: int = 2,    # I–V
    ulceration: bool = False,
    mitotic_rate: int = 0,   # /mm²
) -> BreslowClarkResult:
    """
    Breslow thickness + Clark level — Pronostic mélanome (AJCC 8e éd.).
    """
    # Catégorie Breslow
    if breslow_mm <= 0.1:
        cat = "In situ"
    elif breslow_mm <= 0.8:
        cat = "≤ 0.8 mm (mince)"
    elif breslow_mm <= 1.0:
        cat = "0.8–1.0 mm"
    elif breslow_mm <= 2.0:
        cat = "1.0–2.0 mm"
    elif breslow_mm <= 4.0:
        cat = "2.0–4.0 mm"
    else:
        cat = "> 4.0 mm (épais)"

    # Clark level description
    clark_desc = {
        1: "Épiderme uniquement (in situ)",
        2: "Envahissement du derme papillaire",
        3: "Remplissage du derme papillaire jusqu'à l'interface",
        4: "Invasion du derme réticulaire",
        5: "Envahissement de l'hypoderme",
    }.get(clark_level, "Inconnu")

    # T-stage AJCC 8e
    if breslow_mm == 0 or breslow_mm <= 0.1:
        t = "Tis"
        survival = "97–99%"
    elif breslow_mm <= 0.8 and not ulceration:
        t = "T1a"
        survival = "97%"
    elif breslow_mm <= 1.0 and (ulceration or mitotic_rate >= 1):
        t = "T1b"
        survival = "94%"
    elif breslow_mm <= 2.0 and not ulceration:
        t = "T2a"
        survival = "89%"
    elif breslow_mm <= 2.0 and ulceration:
        t = "T2b"
        survival = "82%"
    elif breslow_mm <= 4.0 and not ulceration:
        t = "T3a"
        survival = "78%"
    elif breslow_mm <= 4.0 and ulceration:
        t = "T3b"
        survival = "71%"
    elif breslow_mm > 4.0 and not ulceration:
        t = "T4a"
        survival = "67%"
    else:
        t = "T4b"
        survival = "54%"

    # Biopsie du ganglion sentinelle
    snb = breslow_mm >= 0.8 or (breslow_mm >= 0.5 and (ulceration or mitotic_rate >= 1))

    # Marges chirurgicales recommandées
    if breslow_mm <= 1.0:
        margins = 1.0
    elif breslow_mm <= 2.0:
        margins = 1.0
    elif breslow_mm <= 4.0:
        margins = 2.0
    else:
        margins = 2.0

    return BreslowClarkResult(
        breslow_mm=breslow_mm, breslow_category=cat,
        clark_level=clark_level, clark_description=clark_desc,
        t_stage=t, five_year_survival=survival,
        sentinel_node_biopsy=snb, surgical_margins_cm=margins,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TNM Mélanome — AJCC 8e édition (2017)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MelanomaStageResult:
    T: str
    N: str
    M: str
    stage: str
    five_year_survival: str
    treatment: str

def compute_melanoma_tnm(
    breslow_mm: float = 1.0,
    ulceration: bool = False,
    mitotic_rate: int = 0,
    regional_nodes_positive: int = 0,
    in_transit_metastasis: bool = False,
    distant_metastasis: bool = False,
    ldh_elevated: bool = False,
    met_site: str = "none",  # none | skin_soft | lung | visceral_non_cns | cns
) -> MelanomaStageResult:
    """TNM Mélanome AJCC 8e édition."""
    # T
    bc = compute_breslow_clark(breslow_mm, ulceration=ulceration, mitotic_rate=mitotic_rate)
    T = bc.t_stage

    # N
    if in_transit_metastasis and regional_nodes_positive == 0:
        N = "N1c"
    elif regional_nodes_positive == 0 and not in_transit_metastasis:
        N = "N0"
    elif regional_nodes_positive == 1:
        N = "N1b" if in_transit_metastasis else "N1a"
    elif regional_nodes_positive <= 3:
        N = "N2b" if in_transit_metastasis else "N2a"
    else:
        N = "N3b" if in_transit_metastasis else "N3a"

    # M
    if not distant_metastasis:
        M = "M0"
    elif met_site == "skin_soft":
        M = "M1a"
    elif met_site == "lung":
        M = "M1b"
    elif met_site == "visceral_non_cns":
        M = "M1c" + (" (LDH↑)" if ldh_elevated else "")
    elif met_site == "cns":
        M = "M1d"
    else:
        M = "M1c"

    # Stage
    stage_5yr = {
        ("Tis","N0","M0"): ("0",  "99%"),
        ("T1a","N0","M0"): ("IA", "97%"),
        ("T1b","N0","M0"): ("IA", "94%"),
        ("T2a","N0","M0"): ("IB", "94%"),
        ("T2b","N0","M0"): ("IIA","82%"),
        ("T3a","N0","M0"): ("IIA","79%"),
        ("T3b","N0","M0"): ("IIB","68%"),
        ("T4a","N0","M0"): ("IIB","71%"),
        ("T4b","N0","M0"): ("IIC","53%"),
    }
    key = (T, N, M)
    if key in stage_5yr:
        stage, survival = stage_5yr[key]
    elif M != "M0":
        stage = "IV"
        survival = "15–30%" if met_site in ("skin_soft","lung") else "5–10%"
    elif N != "N0":
        stage = "IIIA" if regional_nodes_positive <= 1 else "IIIB" if regional_nodes_positive <= 3 else "IIIC"
        survival = {"IIIA": "78%", "IIIB": "59%", "IIIC": "40%"}.get(stage, "50%")
    else:
        stage = "IIB"
        survival = "68%"

    treatment_map = {
        "0":   "Exérèse simple marges 0.5 cm",
        "IA":  "Exérèse marges 1 cm — ganglion sentinelle si T1b",
        "IB":  "Exérèse marges 1 cm + biopsie ganglion sentinelle",
        "IIA": "Exérèse marges 2 cm + biopsie ganglion sentinelle — adjuvant à discuter",
        "IIB": "Exérèse marges 2 cm + biopsie ganglion sentinelle + pembrolizumab adjuvant",
        "IIC": "Exérèse + pembrolizumab ou nivolumab adjuvant — suivi rapproché",
        "IIIA":"Exérèse + curage ou SLNB + immunothérapie adjuvante (pembrolizumab/nivolumab)",
        "IIIB":"Traitement multimodal — immunothérapie + thérapie ciblée (BRAF/MEK si BRAF+)",
        "IIIC":"Immunothérapie systémique + thérapie ciblée — essais cliniques",
        "IV":  "Immunothérapie (pembrolizumab/nivolumab/ipilimumab) ± thérapie ciblée BRAF/MEK",
    }
    return MelanomaStageResult(
        T=T, N=N, M=M.split(" ")[0], stage=stage,
        five_year_survival=survival,
        treatment=treatment_map.get(stage, "RCP oncologique multidisciplinaire"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PASI Score — Psoriasis Area and Severity Index
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PASIResult:
    total: float
    component_scores: dict[str, float]
    severity: str
    bsa_affected: float
    treatment_recommendation: str

def compute_pasi(
    # Tête (10% surface)
    head_erythema: int = 0,     # 0–4
    head_induration: int = 0,   # 0–4
    head_desquamation: int = 0, # 0–4
    head_area: int = 0,         # 0–6 (0=0%, 1=<10%, 2=10–29%, 3=30–49%, 4=50–69%, 5=70–89%, 6=90–100%)
    # Membres supérieurs (20%)
    arms_erythema: int = 0, arms_induration: int = 0,
    arms_desquamation: int = 0, arms_area: int = 0,
    # Tronc (30%)
    trunk_erythema: int = 0, trunk_induration: int = 0,
    trunk_desquamation: int = 0, trunk_area: int = 0,
    # Membres inférieurs (40%)
    legs_erythema: int = 0, legs_induration: int = 0,
    legs_desquamation: int = 0, legs_area: int = 0,
) -> PASIResult:
    """
    PASI Score (Fredriksson & Pettersson, 1978). Score 0–72.
    PASI ≥ 10 = psoriasis modéré à sévère.
    """
    area_map = {0: 0, 1: 0.05, 2: 0.20, 3: 0.40, 4: 0.60, 5: 0.80, 6: 1.00}

    def _region(ery, ind, deq, area, weight):
        severity = (ery + ind + deq) / 3.0
        return round(weight * severity * area_map.get(area, 0) * 72 / (72 * weight) * 72, 2)

    head_score  = _region(head_erythema, head_induration, head_desquamation, head_area, 0.1)
    arms_score  = _region(arms_erythema, arms_induration, arms_desquamation, arms_area, 0.2)
    trunk_score = _region(trunk_erythema, trunk_induration, trunk_desquamation, trunk_area, 0.3)
    legs_score  = _region(legs_erythema, legs_induration, legs_desquamation, legs_area, 0.4)

    # Formule officielle PASI
    def _official(e, i, d, a_idx, w):
        a = area_map.get(a_idx, 0)
        return w * (e + i + d) * a

    h = _official(head_erythema, head_induration, head_desquamation, head_area, 0.1)
    a = _official(arms_erythema, arms_induration, arms_desquamation, arms_area, 0.2)
    t = _official(trunk_erythema, trunk_induration, trunk_desquamation, trunk_area, 0.3)
    l = _official(legs_erythema, legs_induration, legs_desquamation, legs_area, 0.4)
    total = round(h + a + t + l, 1)

    bsa = (area_map.get(head_area, 0) * 10 + area_map.get(arms_area, 0) * 20 +
           area_map.get(trunk_area, 0) * 30 + area_map.get(legs_area, 0) * 40)

    if total < 5:
        severity = "Léger"
        treatment = "Corticostéroïdes topiques ± émollients — kératolytiques si squames épaisses"
    elif total < 10:
        severity = "Modéré"
        treatment = "Corticoïdes topiques de classe III + photothérapie UVB bande étroite"
    elif total < 20:
        severity = "Modéré à sévère"
        treatment = "Photothérapie PUVA ou UVB + méthotrexate ou ciclosporine"
    else:
        severity = "Sévère"
        treatment = "Biothérapies : anti-IL-17 (sécukinumab), anti-IL-12/23 (ustékinumab), anti-TNF"

    return PASIResult(
        total=total,
        component_scores={"Tête": round(h, 2), "Membres sup.": round(a, 2),
                          "Tronc": round(t, 2), "Membres inf.": round(l, 2)},
        severity=severity,
        bsa_affected=round(bsa, 1),
        treatment_recommendation=treatment,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SCORAD — SCORing Atopic Dermatitis (Eczéma)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SCORADResult:
    total: float
    a_score: float      # Étendue (0–100)
    b_score: float      # Intensité (0–18)
    c_score: float      # Symptômes subjectifs (0–20)
    severity: str
    treatment: str

def compute_scorad(
    bsa_percent: float = 5.0,     # A : % surface corporelle atteinte
    erythema: int = 0,            # B : 0–3
    edema: int = 0,               # B : 0–3
    oozing: int = 0,              # B : 0–3
    excoriation: int = 0,         # B : 0–3
    lichenification: int = 0,     # B : 0–3
    dryness: int = 0,             # B : 0–3 (sur peau non lésée)
    pruritus_score: float = 3.0,  # C : EVA 0–10
    sleep_loss_score: float = 1.0,# C : EVA 0–10
) -> SCORADResult:
    """
    SCORAD = A/5 + 7B/2 + C
    A = étendue (0–100%), B = intensité (0–18), C = symptômes subjectifs (0–20).
    Légère < 25, modérée 25–50, sévère > 50.
    """
    a = min(bsa_percent, 100.0)
    b = erythema + edema + oozing + excoriation + lichenification + dryness
    c = pruritus_score + sleep_loss_score

    total = round(a / 5 + 7 * b / 2 + c, 1)

    if total < 25:
        severity = "Légère"
        treatment = "Émollients intensifs + dermocorticoïdes modérés (classe II) · Éviter les irritants"
    elif total <= 50:
        severity = "Modérée"
        treatment = "Dermocorticoïdes classe III–IV + inhibiteurs calcineurine (tacrolimus/pimecrolimus)"
    else:
        severity = "Sévère"
        treatment = "Dupilumab (biothérapie anti-IL-4/IL-13) + dermocorticoïdes puissants ± photothérapie"

    return SCORADResult(
        total=total, a_score=round(a/5, 2), b_score=round(7*b/2, 2), c_score=round(c, 2),
        severity=severity, treatment=treatment,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EASI — Eczema Area and Severity Index
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EASIResult:
    total: float
    severity: str
    iga_equivalent: str
    treatment: str

def compute_easi(
    head_erythema: int = 0, head_edema: int = 0,
    head_excoriation: int = 0, head_lichenification: int = 0, head_area: int = 0,
    arms_erythema: int = 0, arms_edema: int = 0,
    arms_excoriation: int = 0, arms_lichenification: int = 0, arms_area: int = 0,
    trunk_erythema: int = 0, trunk_edema: int = 0,
    trunk_excoriation: int = 0, trunk_lichenification: int = 0, trunk_area: int = 0,
    legs_erythema: int = 0, legs_edema: int = 0,
    legs_excoriation: int = 0, legs_lichenification: int = 0, legs_area: int = 0,
) -> EASIResult:
    """
    EASI Score (0–72). Utilisé dans les essais cliniques eczéma.
    EASI 0=clair, <1=presque clair, 1–7=léger, 7–21=modéré, 21–50=sévère, >50=très sévère.
    """
    area_mult = {0: 0, 1: 0.05, 2: 0.175, 3: 0.35, 4: 0.525, 5: 0.70, 6: 0.875}

    def _region(e, ed, ex, li, a, w):
        return w * (e + ed + ex + li) * area_mult.get(a, 0)

    h = _region(head_erythema, head_edema, head_excoriation, head_lichenification, head_area, 0.1)
    a = _region(arms_erythema, arms_edema, arms_excoriation, arms_lichenification, arms_area, 0.2)
    t = _region(trunk_erythema, trunk_edema, trunk_excoriation, trunk_lichenification, trunk_area, 0.3)
    l = _region(legs_erythema, legs_edema, legs_excoriation, legs_lichenification, legs_area, 0.4)

    total = round((h + a + t + l) * 72, 1)

    if total == 0:
        sev, iga, tr = "Clair", "IGA 0", "Maintien émollients"
    elif total < 1:
        sev, iga, tr = "Presque clair", "IGA 1", "Émollients + corticoïdes faibles au besoin"
    elif total < 7:
        sev, iga, tr = "Léger", "IGA 2", "Dermocorticoïdes classe II + émollients"
    elif total < 21:
        sev, iga, tr = "Modéré", "IGA 3", "Dermocorticoïdes classe III + tacrolimus topique"
    elif total < 50:
        sev, iga, tr = "Sévère", "IGA 4", "Dupilumab ou ciclosporine + corticoïdes puissants"
    else:
        sev, iga, tr = "Très sévère", "IGA 4+", "Dupilumab IV + soins intensifs dermatologiques"

    return EASIResult(total=total, severity=sev, iga_equivalent=iga, treatment=tr)


# ═══════════════════════════════════════════════════════════════════════════════
# GAGS — Global Acne Grading System
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GAGSResult:
    total: int
    severity: str
    grade: str
    treatment: str

def compute_gags(
    forehead: int = 0,     # 0–4 (0=absent, 1=comédons, 2=papules, 3=pustules, 4=nodules)
    right_cheek: int = 0,
    left_cheek: int = 0,
    nose: int = 0,
    chin: int = 0,
    chest_back: int = 0,
) -> GAGSResult:
    """
    GAGS (Doshi et al.) — Score acné global.
    Facteurs localisation × sévérité locale.
    """
    weights = {"forehead": 2, "right_cheek": 2, "left_cheek": 2,
               "nose": 1, "chin": 1, "chest_back": 3}
    scores = {
        "forehead": forehead, "right_cheek": right_cheek, "left_cheek": left_cheek,
        "nose": nose, "chin": chin, "chest_back": chest_back,
    }
    total = sum(weights[r] * scores[r] for r in weights)

    if total == 0:
        sev, grade, tr = "Absent", "Grade 0", "Soins préventifs — nettoyant doux"
    elif total <= 18:
        sev, grade, tr = "Léger", "Grade I",  "Rétinoïdes topiques + peroxyde de benzoyle"
    elif total <= 30:
        sev, grade, tr = "Modéré", "Grade II", "Rétinoïdes topiques + antibiotiques topiques ± oraux"
    elif total <= 38:
        sev, grade, tr = "Sévère", "Grade III","Isotrétinoïne orale si échec traitements précédents"
    else:
        sev, grade, tr = "Très sévère", "Grade IV", "Isotrétinoïne orale — suivi mensuel"

    return GAGSResult(total=total, severity=sev, grade=grade, treatment=tr)


# ═══════════════════════════════════════════════════════════════════════════════
# VASI + VIDA — Vitiligo Area Scoring Index
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VASIResult:
    total: float
    depigmentation_percent: float
    severity: str
    treatment: str

def compute_vasi(
    head_hands_percent: float = 0.0,    # % dépigmentation tête + mains (100 US)
    upper_extremities_percent: float = 0.0,  # 20 US
    trunk_percent: float = 0.0,               # 30 US
    lower_extremities_percent: float = 0.0,  # 40 US
    residual_depigmentation: float = 1.0,    # 0=complet | 0.1–0.9=partiel | 1.0=aucune pigmentation résiduelle
) -> VASIResult:
    """
    VASI = Σ [surface corporelle (unités main) × résidus dépigmentés].
    1 unité main = 1% surface corporelle.
    """
    hand_units = 100.0  # total toutes régions en "hand units"
    total = round(
        (head_hands_percent/100 * 10 +
         upper_extremities_percent/100 * 20 +
         trunk_percent/100 * 30 +
         lower_extremities_percent/100 * 40) * residual_depigmentation,
        2
    )
    depig_pct = (head_hands_percent * 10 + upper_extremities_percent * 20 +
                 trunk_percent * 30 + lower_extremities_percent * 40) / 100

    if total < 5:
        sev = "Léger (< 5% surface)"
        treatment = "Corticoïdes topiques classe III + tacrolimus 0.1% + photothérapie UVB NB"
    elif total < 25:
        sev = "Modéré (5–25%)"
        treatment = "Photothérapie UVB bande étroite + tacrolimus + réflexion thérapeutique"
    else:
        sev = "Étendu (> 25%)"
        treatment = "Ruxolitinib topique (JAK inhibiteur) + photothérapie systémique + soutien psychologique"

    return VASIResult(total=total, depigmentation_percent=round(depig_pct, 1),
                      severity=sev, treatment=treatment)


# ═══════════════════════════════════════════════════════════════════════════════
# Résumé consolidé — tous les scores selon pathologie
# ═══════════════════════════════════════════════════════════════════════════════

def build_derm_clinical_summary(
    prediction: str,
    confidence: float,
    image_features: dict[str, Any],
    abcde_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calcule les scores appropriés selon la pathologie prédite."""
    pred_lower = prediction.lower()
    summary: dict[str, Any] = {"prediction": prediction, "confidence": confidence}

    if "mélanome" in pred_lower or "melanome" in pred_lower:
        abcde = compute_abcde(**(abcde_params or {}))
        bc = compute_breslow_clark(
            breslow_mm=image_features.get("estimated_breslow_mm", 1.0),
            clark_level=image_features.get("clark_level", 3),
            ulceration=image_features.get("ulceration", False),
        )
        tnm = compute_melanoma_tnm(
            breslow_mm=bc.breslow_mm,
            ulceration=image_features.get("ulceration", False),
        )
        summary["abcde"] = {
            "total_score": abcde.total_score, "risk": abcde.risk_level,
            "probability": abcde.malignancy_probability, "recommendation": abcde.recommendation,
        }
        summary["breslow_clark"] = {
            "breslow_mm": bc.breslow_mm, "category": bc.breslow_category,
            "clark_level": bc.clark_level, "t_stage": bc.t_stage, "survival": bc.five_year_survival,
            "sentinel_node_biopsy": bc.sentinel_node_biopsy, "margins_cm": bc.surgical_margins_cm,
        }
        summary["tnm"] = {"T": tnm.T, "N": tnm.N, "M": tnm.M,
                          "stage": tnm.stage, "survival": tnm.five_year_survival,
                          "treatment": tnm.treatment}

    elif "carcinome" in pred_lower or "kératose" in pred_lower:
        abcde = compute_abcde(**(abcde_params or {}))
        summary["abcde"] = {"total_score": abcde.total_score, "risk": abcde.risk_level,
                            "probability": abcde.malignancy_probability,
                            "recommendation": abcde.recommendation}

    elif "psoriasis" in pred_lower:
        summary["pasi"] = {"note": "PASI calculé via formulaire clinique — saisie manuelle requise",
                           "reference": "PASI ≥ 10 = traitement systémique éligible"}

    elif "eczéma" in pred_lower or "dermatite" in pred_lower:
        scorad = compute_scorad(bsa_percent=image_features.get("bsa_estimate", 15.0))
        summary["scorad"] = {"total": scorad.total, "severity": scorad.severity,
                             "treatment": scorad.treatment}

    elif "vitiligo" in pred_lower:
        summary["vasi"] = {"note": "VASI calculé via formulaire clinique — examen dermatologique requis"}

    elif "acné" in pred_lower or "acne" in pred_lower:
        summary["gags"] = {"note": "Score GAGS — saisie régions faciales requise"}

    return summary


# ── Fitzpatrick Phototype ────────────────────────────────────────────────────

FITZPATRICK_TYPES = {
    1: {"description": "Très claire — toujours brûle, ne bronze jamais", "risk": "Très élevé", "spf_recommendation": "SPF 50+"},
    2: {"description": "Claire — brûle facilement, bronze peu",           "risk": "Élevé",     "spf_recommendation": "SPF 50+"},
    3: {"description": "Intermédiaire — brûle modérément, bronze",        "risk": "Modéré",    "spf_recommendation": "SPF 30–50"},
    4: {"description": "Olive — brûle peu, bronze facilement",            "risk": "Modéré",    "spf_recommendation": "SPF 30"},
    5: {"description": "Foncée — brûle rarement, bronze toujours",        "risk": "Faible",    "spf_recommendation": "SPF 15–30"},
    6: {"description": "Très foncée — ne brûle jamais",                   "risk": "Très faible","spf_recommendation": "SPF 15"},
}
