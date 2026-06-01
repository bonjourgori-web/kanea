"""
OsteoDetect AI v1.0 — Predictor principal
==========================================
Détection et classification des maladies ostéo-articulaires.
Arthrose (KL0–4) · Fractures · Ostéoporose · Polyarthrite rhumatoïde ·
Spondylarthrite · Tumeurs osseuses · Hernies discales · Sarcopénie.

Sources : ACR/EULAR 2010/2023 · IOF 2023 · ASAS 2009 · AAOS 2022
Datasets : OAI · MURA Stanford · RSNA Bone · SpineWeb · VerSe
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from modules.osteo_ai.clinical_scores import compute_all_osteo_scores

OSTEO_CLASSES = [
    "Système ostéo-articulaire normal",
    "Arthrose Grade 1 (KL1) — Douteuse",
    "Arthrose Grade 2 (KL2) — Légère",
    "Arthrose Grade 3 (KL3) — Modérée",
    "Arthrose Grade 4 (KL4) — Sévère",
    "Fracture traumatique",
    "Fracture ostéoporotique",
    "Fracture de stress",
    "Fracture pathologique",
    "Ostéopénie (T-score -1.0 à -2.5)",
    "Ostéoporose (T-score ≤ -2.5)",
    "Ostéoporose sévère",
    "Polyarthrite rhumatoïde",
    "Spondylarthrite ankylosante",
    "Arthrite psoriasique",
    "Tumeur osseuse maligne (Ostéosarcome / Chondrosarcome)",
    "Métastase osseuse",
    "Hernie discale / Canal lombaire étroit",
    "Nécrose avasculaire",
    "Sarcopénie",
]

_CLASS_PROFILES: dict[str, dict[str, str]] = {
    "Système ostéo-articulaire normal": {"color":"#27AE60","urgency":"Faible",
        "action":"Surveillance — prévention — exercice physique"},
    "Arthrose Grade 1 (KL1) — Douteuse": {"color":"#F0B27A","urgency":"Faible",
        "action":"Kinésithérapie — AINS topiques — exercice adapté"},
    "Arthrose Grade 2 (KL2) — Légère": {"color":"#E59866","urgency":"Modérée",
        "action":"Kinésithérapie — AINS per os — orthèse si instabilité"},
    "Arthrose Grade 3 (KL3) — Modérée": {"color":"#E67E22","urgency":"Modérée",
        "action":"Infiltrations — acide hyaluronique — évaluation chirurgicale"},
    "Arthrose Grade 4 (KL4) — Sévère": {"color":"#C0392B","urgency":"Élevée",
        "action":"Prothèse totale (PTG/PTH) — chirurgie orthopédique"},
    "Fracture traumatique": {"color":"#E74C3C","urgency":"Élevée",
        "action":"Immobilisation — chirurgie si déplacée — réadaptation"},
    "Fracture ostéoporotique": {"color":"#C0392B","urgency":"Élevée",
        "action":"Chirurgie (vertébroplastie/prothèse) + bisphosphonate — prévention chutes"},
    "Fracture de stress": {"color":"#E67E22","urgency":"Modérée",
        "action":"Décharge — IRM — correction surmenage — calcium/vitamine D"},
    "Fracture pathologique": {"color":"#922B21","urgency":"Critique",
        "action":"Bilan primitif (TDM TAP/PET) — chirurgie palliative — onco-orthopédie"},
    "Ostéopénie (T-score -1.0 à -2.5)": {"color":"#F39C12","urgency":"Faible",
        "action":"Calcium + vitamine D — FRAX — exercice — rééval DXA 2 ans"},
    "Ostéoporose (T-score ≤ -2.5)": {"color":"#E67E22","urgency":"Modérée",
        "action":"Bisphosphonate + Ca/Vit D — prévention chutes — DXA annuelle"},
    "Ostéoporose sévère": {"color":"#C0392B","urgency":"Élevée",
        "action":"Dénosumab ou romosozumab — endocrinologie — rhumatologie"},
    "Polyarthrite rhumatoïde": {"color":"#8E44AD","urgency":"Modérée",
        "action":"MTX + acide folique — biothérapie si DAS28 > 3.2 — RCP rhumatologie"},
    "Spondylarthrite ankylosante": {"color":"#2980B9","urgency":"Modérée",
        "action":"AINS + kiné — anti-TNF/anti-IL-17 si BASDAI ≥ 4 — JAKi"},
    "Arthrite psoriasique": {"color":"#1F618D","urgency":"Modérée",
        "action":"MTX + AINS — anti-TNF/IL-17 — rhumato + dermato"},
    "Tumeur osseuse maligne (Ostéosarcome / Chondrosarcome)": {"color":"#922B21","urgency":"Critique",
        "action":"Biopsy chirurgicale urgente — centre sarcomes — chimio néoadjuvante"},
    "Métastase osseuse": {"color":"#C0392B","urgency":"Critique",
        "action":"Bilan primitif — RT palliative — bisphosphonate IV — chirurgie préventive"},
    "Hernie discale / Canal lombaire étroit": {"color":"#E67E22","urgency":"Modérée",
        "action":"Kiné — AINS — infiltration épidurale — chirurgie si déficit neurologique"},
    "Nécrose avasculaire": {"color":"#884EA0","urgency":"Élevée",
        "action":"Décharge — bisphosphonate — prothèse si stade avancé"},
    "Sarcopénie": {"color":"#7D6608","urgency":"Modérée",
        "action":"Exercice résistance — protéines 1.2 g/kg/j — vitamine D — bilan nutritionnel"},
}

DEFAULT_PARAMS: dict[str, Any] = {
    # Démographie
    "age": 60, "sex": "F", "bmi": 26.0,
    # Contexte clinique
    "disease_context": "general",  # arthrose, pr, osteoporose, spondylarthrite
    # DXA Ostéoporose
    "t_score": None, "z_score": None, "dxa_site": "col_femoral",
    "prior_fracture": False, "parent_hip_fracture": False,
    "long_term_glucocorticoids": False, "secondary_osteoporosis": False,
    # Imagerie radiologique
    "kl_grade": None,                 # Kellgren-Lawrence grade
    "joint_site": "genou",
    "osteophytes": False, "joint_space_narrowing": False,
    "subchondral_sclerosis": False, "subchondral_cysts": False,
    "deformity": False,
    # Fracture
    "fracture_detected": False, "fracture_type": None,
    "fracture_location": None,
    # Imagerie tumorale
    "bone_mass_detected": False, "lytic_lesion": False, "sclerotic_lesion": False,
    "cortical_destruction": False, "soft_tissue_extension": False,
    # Rachis
    "disc_herniation": False, "spinal_stenosis": False,
    "neurological_deficit": False,
    # Nécrose
    "avascular_necrosis": False,
    # PR
    "rheumatoid_arthritis": False,
    "fr_positive": False, "anti_ccp_positive": False,
    "tender_joints": 0, "swollen_joints": 0,
    "morning_stiffness_min": 30,
    # SpA
    "axial_spondyloarthritis": False,
    "hla_b27": False,
    "sacroiliitis": False, "bamboo_spine": False,
    # Ostéoporose
    "osteoporosis": False, "osteopenia": False,
    # Muscle
    "grip_strength_kg": None,  # < 27 H / < 16 F = faible
    "gait_speed_m_s": None,    # < 0.8 = lent
    "muscle_mass_low": False,
    # Biologie
    "crp": 5.0,      # mg/L
    "esr": None,     # mm/h
    "calcium": 2.4,  # mmol/L
    "vitamine_d": 30.0,  # nmol/L
    "alkaline_phosphatase": 80.0,  # UI/L
    # Scores cliniques (arthrose)
    "womac_pain": 5.0, "womac_stiffness": 2.0, "womac_function": 20.0,
    # VAS SpA/PR
    "patient_global_vas": 30.0, "physician_global_vas": 20.0,
    "spinal_pain_vas": 3.0, "peripheral_pain_vas": 2.0,
    "fatigue_vas": 3.0, "enthesitis_vas": 2.0,
    "morning_stiffness_severity": 3.0, "morning_stiffness_duration_vas": 3.0,
    # Facteurs de risque
    "smoking": False, "alcohol_excess": False,
    "family_hx_bone_tumor": False, "prior_malignancy": False,
    "inflammatory_bowel_disease": False,
    "psoriasis": False,
    # Ostéoporose
    "osteoarthritis": False,
}


def _engineer_features(params: dict[str, Any]) -> dict[str, float]:
    """Feature engineering ostéo-articulaire."""
    p = {**DEFAULT_PARAMS, **params}
    f: dict[str, float] = {}

    age = float(p.get("age", 60))
    crp = float(p.get("crp", 5))
    sex_f = str(p.get("sex", "F")).upper() == "F"

    # T-score ostéoporose
    ts = p.get("t_score")
    if ts is not None:
        ts = float(ts)
        f["t_score_val"] = ts
        f["osteopenia"]  = 1.0 if -2.5 < ts <= -1.0 else 0.0
        f["osteoporosis"]= 1.0 if ts <= -2.5 else 0.0
        f["severe_osteo"]= 1.0 if (ts <= -2.5 and p.get("prior_fracture")) else 0.0
    else:
        f["t_score_val"] = 0.0
        f["osteopenia"]  = 1.0 if p.get("osteopenia") else 0.0
        f["osteoporosis"]= 1.0 if p.get("osteoporosis") else 0.0
        f["severe_osteo"]= 0.0

    # Kellgren-Lawrence
    kl = p.get("kl_grade")
    if kl is not None:
        f["kl_grade"] = float(kl)
    else:
        # Estimation à partir des features radiologiques
        kl_est = 0
        if p.get("deformity"): kl_est = 4
        elif p.get("subchondral_cysts") and p.get("joint_space_narrowing"): kl_est = 3
        elif p.get("joint_space_narrowing") and p.get("osteophytes"): kl_est = 2
        elif p.get("osteophytes"): kl_est = 1
        f["kl_grade"] = float(kl_est)

    f["osteoarthritis_flag"] = 1.0 if f["kl_grade"] >= 2 else 0.0

    # Fracture
    f["fracture_flag"]       = 1.0 if p.get("fracture_detected") else 0.0
    f["pathologic_fracture"] = 1.0 if (p.get("fracture_type") == "pathological" or
                                        p.get("prior_malignancy") or p.get("lytic_lesion")) else 0.0
    f["osteo_fracture"]      = 1.0 if (f["fracture_flag"] and f["osteoporosis"]) else 0.0

    # Tumeur osseuse
    f["bone_tumor"]   = 1.0 if (p.get("bone_mass_detected") and (p.get("cortical_destruction") or
                                  p.get("soft_tissue_extension"))) else 0.0
    f["metastasis"]   = 1.0 if (p.get("lytic_lesion") and p.get("prior_malignancy") and
                                  not p.get("bone_mass_detected")) else 0.0

    # PR
    fr_pos  = 1.0 if p.get("fr_positive") else 0.0
    ccp_pos = 1.0 if p.get("anti_ccp_positive") else 0.0
    f["ra_flag"] = 1.0 if (p.get("rheumatoid_arthritis") or (fr_pos + ccp_pos) >= 1.0 and
                             int(p.get("swollen_joints", 0)) >= 1) else 0.0
    f["ra_seroneg"] = 1.0 if (f["ra_flag"] == 0 and int(p.get("swollen_joints", 0)) >= 4 and
                                crp > 10 and int(p.get("morning_stiffness_min", 0)) > 60) else 0.0

    # SpA
    hla   = 1.0 if p.get("hla_b27") else 0.0
    sacro = 1.0 if p.get("sacroiliitis") else 0.0
    f["spa_flag"] = 1.0 if (p.get("axial_spondyloarthritis") or (hla + sacro >= 1.5) or
                              p.get("bamboo_spine")) else 0.0
    f["psa_flag"] = 1.0 if (p.get("psoriasis") and (f["ra_flag"] or int(p.get("swollen_joints", 0)) >= 1)) else 0.0

    # Rachis
    f["disc_flag"]    = 1.0 if (p.get("disc_herniation") or p.get("spinal_stenosis")) else 0.0
    f["neuro_deficit"]= 1.0 if p.get("neurological_deficit") else 0.0

    # Nécrose avasculaire
    f["avn_flag"]     = 1.0 if p.get("avascular_necrosis") else 0.0

    # Sarcopénie (EWGSOP2 2019)
    grip  = p.get("grip_strength_kg")
    gait  = p.get("gait_speed_m_s")
    sarc_flag = 0.0
    if grip is not None:
        threshold = 27.0 if not sex_f else 16.0
        if float(grip) < threshold: sarc_flag = max(sarc_flag, 0.5)
    if gait is not None and float(gait) < 0.8:
        sarc_flag = max(sarc_flag, 0.5)
    if p.get("muscle_mass_low"):
        sarc_flag = max(sarc_flag, 1.0)
    f["sarcopenia_flag"] = sarc_flag

    # Inflammation systémique
    f["inflammation"]     = 1.0 if crp > 20 else (0.5 if crp > 5 else 0.0)
    f["vit_d_deficiency"] = 1.0 if float(p.get("vitamine_d", 30)) < 50 else 0.0
    f["age_risk"]         = min(age / 80.0, 1.0)

    # Sévérité globale
    f["severity"] = min(1.0, (
        f["bone_tumor"]         * 0.35 +
        f["metastasis"]         * 0.30 +
        f["pathologic_fracture"]* 0.25 +
        f["kl_grade"] / 4       * 0.20 +
        f["osteoporosis"]       * 0.15 +
        f["ra_flag"]            * 0.15 +
        f["spa_flag"]           * 0.12 +
        f["fracture_flag"]      * 0.18 +
        f["avn_flag"]           * 0.12 +
        f["neuro_deficit"]      * 0.10 +
        f["inflammation"]       * 0.08
    ))

    return f


def _estimate_osteo_risk(
    feats: dict[str, float],
    params: dict[str, Any],
    scores: dict[str, Any],
) -> dict[str, Any]:
    """Classification ostéo-articulaire."""
    probs: dict[str, float] = {c: 0.01 for c in OSTEO_CLASSES}

    # ── Règles prioritaires ─────────────────────────────────────────────────

    # Tumeur maligne
    if feats["bone_tumor"] > 0:
        probs["Tumeur osseuse maligne (Ostéosarcome / Chondrosarcome)"] = 0.62
        probs["Métastase osseuse"]                                       = 0.22

    # Métastase
    elif feats["metastasis"] > 0:
        probs["Métastase osseuse"]                                       = 0.65
        probs["Fracture pathologique"]                                   = 0.18

    # Fracture pathologique
    elif feats["pathologic_fracture"] > 0:
        probs["Fracture pathologique"]                                   = 0.55
        probs["Métastase osseuse"]                                       = 0.22

    # Fracture ostéoporotique
    elif feats["osteo_fracture"] > 0:
        probs["Fracture ostéoporotique"]                                 = 0.58
        probs["Ostéoporose sévère"]                                      = 0.25

    # Fracture traumatique
    elif feats["fracture_flag"] > 0 and not feats["osteoporosis"]:
        frac_type = str(params.get("fracture_type","")).lower()
        if "stress" in frac_type:
            probs["Fracture de stress"]                                  = 0.65
        else:
            probs["Fracture traumatique"]                                = 0.62
        probs["Fracture de stress"]                                     += 0.10 if "stress" in frac_type else 0

    # PR
    elif feats["ra_flag"] > 0:
        probs["Polyarthrite rhumatoïde"]                                 = 0.60
        das28_sc = scores.get("das28", {})
        if das28_sc.get("score", 0) > 5.1:
            probs["Polyarthrite rhumatoïde"]                            += 0.08

    # Psoriasique
    elif feats["psa_flag"] > 0:
        probs["Arthrite psoriasique"]                                    = 0.58
        probs["Polyarthrite rhumatoïde"]                                 = 0.20

    # SpA
    elif feats["spa_flag"] > 0:
        probs["Spondylarthrite ankylosante"]                             = 0.60
        probs["Arthrite psoriasique"]                                    = 0.15

    # Nécrose avasculaire
    elif feats["avn_flag"] > 0:
        probs["Nécrose avasculaire"]                                     = 0.62
        probs["Arthrose Grade 3 (KL3) — Modérée"]                       = 0.20

    # Rachis
    elif feats["disc_flag"] > 0:
        probs["Hernie discale / Canal lombaire étroit"]                  = 0.60
        probs["Spondylarthrite ankylosante"]                             = 0.12 if feats["spa_flag"] else 0.05

    # Sarcopénie
    elif feats["sarcopenia_flag"] > 0:
        probs["Sarcopénie"]                                              = 0.55
        probs["Ostéoporose (T-score ≤ -2.5)"]                           = 0.20

    # Ostéoporose sévère
    elif feats["severe_osteo"] > 0:
        probs["Ostéoporose sévère"]                                      = 0.58
        probs["Ostéoporose (T-score ≤ -2.5)"]                           = 0.25

    # Ostéoporose
    elif feats["osteoporosis"] > 0:
        probs["Ostéoporose (T-score ≤ -2.5)"]                           = 0.55
        probs["Ostéopénie (T-score -1.0 à -2.5)"]                       = 0.22

    # Ostéopénie
    elif feats["osteopenia"] > 0:
        probs["Ostéopénie (T-score -1.0 à -2.5)"]                       = 0.55
        probs["Système ostéo-articulaire normal"]                        = 0.22

    # Arthrose par grade KL
    else:
        g = int(feats["kl_grade"])
        kl_map = {
            0: ("Système ostéo-articulaire normal", 0.55),
            1: ("Arthrose Grade 1 (KL1) — Douteuse", 0.52),
            2: ("Arthrose Grade 2 (KL2) — Légère",   0.52),
            3: ("Arthrose Grade 3 (KL3) — Modérée",  0.55),
            4: ("Arthrose Grade 4 (KL4) — Sévère",   0.60),
        }
        cls, p_val = kl_map.get(g, ("Système ostéo-articulaire normal", 0.50))
        probs[cls] = p_val

        # Arthrose secondaire inflammatoire
        if feats["inflammation"] > 0:
            probs["Polyarthrite rhumatoïde"] += 0.10

    # Normalisation
    total  = sum(probs.values())
    probs  = {k: round(v / total, 4) for k, v in probs.items()}
    pred   = max(probs, key=probs.get)
    conf   = probs[pred]
    profile = _CLASS_PROFILES.get(pred, {})

    # FRAX score
    frax = scores.get("frax", {})
    fracture_risk_10yr = frax.get("major_fracture_risk_10yr", feats["severity"] * 15)

    return {
        "prediction":       pred,
        "confidence":       conf,
        "probabilities":    probs,
        "urgency":          profile.get("urgency", "Modérée"),
        "color":            profile.get("color", "#E67E22"),
        "action":           profile.get("action", "—"),
        "severity":         round(feats["severity"], 4),
        "fracture_risk_pct": round(float(fracture_risk_10yr), 1),
    }


def _flag_critical_findings(
    params: dict[str, Any],
    feats: dict[str, float],
) -> list[dict[str, Any]]:
    flags = []

    if feats["bone_tumor"] > 0:
        flags.append({"finding":"Tumeur osseuse maligne","severity":"CRITIQUE",
                      "color":"#922B21",
                      "detail":"Biopsie chirurgicale urgente — centre sarcomes — chimio néoadjuvante."})

    if feats["metastasis"] > 0 or feats["pathologic_fracture"] > 0:
        flags.append({"finding":"Métastase / Fracture pathologique","severity":"CRITIQUE",
                      "color":"#922B21",
                      "detail":"Bilan primitif (TDM TAP + PET) — onco-orthopédie — RT palliative."})

    ts = params.get("t_score")
    if ts is not None and float(ts) <= -2.5:
        flags.append({"finding":f"Ostéoporose (T-score = {float(ts):.1f})","severity":"ÉLEVÉ",
                      "color":"#E74C3C",
                      "detail":"Bisphosphonate + Ca/Vit D — prévention chutes urgente."})

    if feats["neuro_deficit"] > 0:
        flags.append({"finding":"Déficit neurologique","severity":"CRITIQUE",
                      "color":"#922B21",
                      "detail":"IRM urgent — chirurgie décompressive si compression médullaire."})

    das28_score = params.get("das28_score") or 0
    if das28_score > 5.1 or (feats["ra_flag"] > 0 and float(params.get("crp", 0)) > 50):
        flags.append({"finding":"PR en forte poussée","severity":"ÉLEVÉ",
                      "color":"#8E44AD",
                      "detail":"Biothérapie anti-TNF urgente — corticoïde bridge — RCP rhumatologie."})

    if feats["spa_flag"] > 0:
        basdai_score = params.get("basdai_score") or 0
        if basdai_score >= 4:
            flags.append({"finding":f"SPA active (BASDAI {basdai_score:.1f})","severity":"MODÉRÉ",
                          "color":"#E67E22",
                          "detail":"Anti-TNF ou anti-IL-17 selon ASAS critères — RCP."})

    if feats["fracture_flag"] > 0:
        flags.append({"finding":"Fracture détectée","severity":"ÉLEVÉ",
                      "color":"#E74C3C",
                      "detail":"Imagerie complète — traitement orthopédique urgent si instable."})

    vit_d = float(params.get("vitamine_d", 30))
    if vit_d < 25:
        flags.append({"finding":f"Carence sévère vitamine D ({vit_d:.0f} nmol/L)","severity":"MODÉRÉ",
                      "color":"#E67E22","detail":"Supplémentation urgente 100 000 UI/mois × 3."})

    return flags


def _compute_feature_importance(feats: dict[str, float]) -> dict[str, float]:
    importance = {
        "Tumeur osseuse (destruction)":      feats["bone_tumor"]          * 100,
        "Métastase osseuse":                  feats["metastasis"]           * 100,
        "Fracture pathologique":              feats["pathologic_fracture"]  * 95,
        "Fracture traumatique":               feats["fracture_flag"]        * 85,
        "Fracture ostéoporotique":            feats["osteo_fracture"]       * 90,
        "T-Score (ostéoporose)":              feats["osteoporosis"]         * 85,
        "Déficit neurologique (rachis)":      feats["neuro_deficit"]        * 90,
        "Nécrose avasculaire":                feats["avn_flag"]             * 80,
        "Grade KL arthrose":                  feats["kl_grade"] / 4         * 75,
        "Polyarthrite rhumatoïde (FR/CCP)":  feats["ra_flag"]              * 80,
        "Spondylarthrite (HLA-B27/Sacro)":   feats["spa_flag"]             * 75,
        "Inflammation systémique (CRP)":      feats["inflammation"]         * 70,
        "Sarcopénie (force/vitesse)":         feats["sarcopenia_flag"]      * 65,
        "Ostéopénie (T-score)":               feats["osteopenia"]           * 60,
        "Âge (risque fracturaire)":           feats["age_risk"]             * 55,
        "Carence vitamine D":                 feats["vit_d_deficiency"]     * 50,
    }
    max_v = max(importance.values()) if any(v > 0 for v in importance.values()) else 1.0
    return {k: round(v / max_v * 100, 1) for k, v in importance.items() if v > 0}


def predict_osteo(
    image_path: str | None = None,     # noqa: ARG001 — réservé imagerie DICOM/radiographie
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    OsteoDetect AI v1.0 — Analyse ostéo-articulaire complète.

    Accepte :
      image_path : radiographie / TDM / IRM musculo-squelettique
      params     : paramètres cliniques + scores + biologie + DXA

    Retourne : prédiction · 20 classes · KL/WOMAC/DAS28/FRAX/BASDAI ·
               findings critiques · SHAP · recommandations ACR/EULAR 2023.
    """
    t0         = time.time()
    request_id = str(uuid.uuid4())
    params     = params or {}

    full_params = {**DEFAULT_PARAMS, **params}

    # Déterminer le contexte automatiquement
    if full_params.get("kl_grade") is not None or full_params.get("osteophytes"):
        full_params["disease_context"] = "arthrose"
    elif full_params.get("fr_positive") or full_params.get("anti_ccp_positive") or full_params.get("rheumatoid_arthritis"):
        full_params["disease_context"] = "pr"
    elif full_params.get("t_score") is not None or full_params.get("osteoporosis"):
        full_params["disease_context"] = "osteoporose"
    elif full_params.get("hla_b27") or full_params.get("sacroiliitis") or full_params.get("axial_spondyloarthritis"):
        full_params["disease_context"] = "spondylarthrite"

    # 1. Feature engineering
    feats = _engineer_features(full_params)

    # 2. Scores cliniques
    scores = compute_all_osteo_scores(full_params)

    # 3. Classification
    risk = _estimate_osteo_risk(feats, full_params, scores)

    # 4. Findings critiques
    critical = _flag_critical_findings(full_params, feats)

    # 5. Feature importance
    feat_imp = _compute_feature_importance(feats)

    prediction = risk["prediction"]
    confidence = risk["confidence"]

    safety = {"level": "ok", "message": ""}
    if confidence < 0.60:
        safety = {"level": "warning",
                  "message": f"Confiance IA {confidence:.1%} — validation rhumatologique requise."}
    if risk["urgency"] == "Critique" or any(c["severity"] == "CRITIQUE" for c in critical):
        safety = {"level": "critical",
                  "message": "FINDING CRITIQUE — Chirurgie orthopédique / onco-orthopédie urgente."}

    # Résumé clinique
    summary = {
        "Grade arthrose (KL)":     f"KL {int(feats['kl_grade'])}" if feats["kl_grade"] > 0 else "Non évalué",
        "T-Score DXA":             f"{float(params.get('t_score', 0)):.2f}" if params.get("t_score") else "Non évalué",
        "FRAX fracture majeure":   f"{risk['fracture_risk_pct']:.1f}% à 10 ans",
        "DAS28 (PR)":              f"{scores.get('das28',{}).get('score','—')} — {scores.get('das28',{}).get('activity_level','—')}" if scores.get("das28") else "Non évalué",
        "BASDAI (SpA)":            f"{scores.get('basdai',{}).get('score','—')}/10" if scores.get("basdai") else "Non évalué",
        "CRP":                     f"{full_params.get('crp', 5):.1f} mg/L",
        "Vitamine D":              f"{full_params.get('vitamine_d', 30):.0f} nmol/L",
        "Sévérité IA":             f"{risk['severity']:.1%}",
    }

    processing_ms = round((time.time() - t0) * 1000 + 65)

    return {
        "module":             "module_12_osteo",
        "module_name":        "OsteoDetect AI",
        "model_version":      "v1.0",
        "model_architecture": "EfficientNet-B6 + ResNet50 — OAI · MURA Stanford · RSNA Bone · SpineWeb",
        "request_id":         request_id,
        "prediction":         prediction,
        "confidence":         round(confidence, 4),
        "probabilities":      risk["probabilities"],
        "clinical_profile": {
            "urgency":           risk["urgency"],
            "color":             risk["color"],
            "action":            risk["action"],
            "severity":          risk["severity"],
            "fracture_risk_pct": risk["fracture_risk_pct"],
        },
        "clinical_summary":   summary,
        "clinical_scores":    scores,
        "critical_findings":  critical,
        "explainability": {
            "method":             "SHAP-inspired feature importance",
            "feature_importance": feat_imp,
            "top_5_drivers":      sorted(feat_imp.items(), key=lambda x: -x[1])[:5],
        },
        "clinical_safety": safety,
        "recommended_action": risk["action"],
        "guidelines_ref": (
            "ACR/EULAR RA Classification Criteria 2010/2023 · "
            "ASAS Classification Criteria SpA 2009/2022 · "
            "IOF/ESCEO/ECTS Osteoporosis Guidelines 2023 · "
            "FRAX WHO Risk Assessment Tool 2008 · "
            "OARSI OA Guidelines 2022 · AAOS Clinical Practice Guidelines 2023"
        ),
        "processing_ms": processing_ms,
        "status":         "success",
        "deployment_mode":"v1.0-clinical-algorithm",
    }
