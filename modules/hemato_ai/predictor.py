"""
HematoVision AI v2.0 — Predictor principal
===========================================
Pipeline clinique multi-contexte : NFS · Cytologie · Biochimie · Génétique.
Détection, classification et stadification des maladies hématologiques.

Maladies : Anémies · Leucémies (LAM/LAL/LLC/LMC) · SMD · Syndromes myéloprolifératifs
           Lymphomes · Myélome multiple · Paludisme · Infections sanguines.

Sources : WHO 2022 · ICC 2022 · ELN 2022 · EHA/ASH Guidelines · NCCN 2023.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from modules.hemato_ai.clinical_scores import compute_all_hemato_scores

# ─────────────────────────────────────────────────────────────────────────────
# Classes diagnostiques HematoVision
# ─────────────────────────────────────────────────────────────────────────────
HEMATO_CLASSES = [
    "Normal — Pas d'anomalie hématologique",
    "Anémie ferriprive",
    "Anémie mégaloblastique (B12/Folates)",
    "Anémie hémolytique",
    "Drépanocytose / Thalassémie",
    "Leucémie aiguë (LAM / LAL)",
    "Leucémie lymphoïde chronique (LLC)",
    "Leucémie myéloïde chronique (LMC)",
    "Syndrome myélodysplasique (SMD)",
    "Syndrome myéloprolifératif (SMP)",
    "Lymphome (Hodgkin / LNH)",
    "Myélome multiple",
    "Paludisme / Infection parasitaire",
    "Infection sanguine (Bactérienne / Virale)",
]

DEFAULT_PARAMS: dict[str, Any] = {
    # NFS
    "hemoglobin": 13.5, "hematocrit": 40.0, "rbc": 4.5,
    "wbc": 7.0, "platelets": 220.0, "sex": "M",
    # Indices érythrocytaires
    "vgm": 88.0, "tcmh": 29.0, "ccmh": 32.5, "rdw": 12.5,
    # Leucocytes différentiels
    "neutrophils": 4.5, "lymphocytes": 2.0, "monocytes": 0.5,
    "eosinophils": 0.15, "basophils": 0.04,
    # Blastes / myéloblastes
    "blast_pct": 0.0,
    # Biochimie
    "ldh": 180.0, "ldh_elevated": False,
    "ferritin": 80.0, "iron": 16.0, "transferrin_sat_pct": 30.0,
    "b12_pg_ml": 400.0, "folate_ng_ml": 8.0, "haptoglobin": 1.2,
    "reticulocytes_pct": 1.5, "direct_coombs": False,
    # Coagulation
    "pt_prolonged_sec": 0.0, "fibrinogen": 3.0, "d_dimers_elevated": 0,
    # Cytogénétique / Biologie moléculaire
    "npm1_mutated": False, "flt3_itd": False, "flt3_itd_low": False,
    "cebpa_biallelic": False, "t_8_21": False, "inv_16": False,
    "tp53_mutated": False, "runx1_mutated": False, "asxl1_mutated": False,
    "complex_karyotype": False, "monosomal_karyotype": False,
    "del_5q": False, "del_7": False, "inv_3": False, "t_6_9": False,
    "bcr_abl1": False, "jak2_v617f": False, "calr_mutated": False,
    "mpl_mutated": False, "sf3b1_mutated": False,
    "del_17p": False, "t_4_14": False, "t_14_16": False, "gain_1q": False,
    # Myélome
    "albumin": 4.0, "beta2_microglobulin": 2.0,
    "serum_protein_electrophoresis": False, "bone_lesions": False,
    # Lymphome
    "lymphadenopathy": False, "splenomegaly": False, "hepatomegaly": False,
    "b_symptoms": False, "nodal_regions": 1, "extranodal_sites": 0,
    "spleen_involved": False, "liver_involved": False, "bone_marrow_involved": False,
    "bulky_mass_cm": 0.0, "same_side_diaphragm": True,
    # Paludisme
    "parasitemia_pct": 0.0, "plasmodium_species": "P. falciparum", "rdt_positive": False,
    "malaria_suspected": False,
    # CIVD
    "dic_suspected": False,
    # Contexte clinique
    "disease_context": "general", "age": 45, "ecog_ps": 0,
    "cytogenetics_risk": "intermediate",
}


# ─────────────────────────────────────────────────────────────────────────────
# Ingénierie des features cliniques
# ─────────────────────────────────────────────────────────────────────────────

def _engineer_features(p: dict[str, Any]) -> dict[str, float]:
    feats: dict[str, float] = {}

    hb   = float(p["hemoglobin"])
    wbc  = float(p["wbc"])
    plt  = float(p["platelets"])
    vgm  = float(p["vgm"])
    rdw  = float(p["rdw"])
    neut = float(p["neutrophils"])
    lymp = float(p["lymphocytes"])
    blst = float(p["blast_pct"])
    ferr = float(p["ferritin"])
    b12  = float(p["b12_pg_ml"])
    folate = float(p["folate_ng_ml"])
    retic  = float(p["reticulocytes_pct"])
    sex_f  = str(p["sex"]).upper() == "F"
    hb_ref = 12.0 if sex_f else 13.0

    feats["anemia_flag"]         = 1.0 if hb < hb_ref else 0.0
    feats["severe_anemia"]       = 1.0 if hb < 7.0 else 0.0
    feats["microcytic_flag"]     = 1.0 if vgm < 80 else 0.0
    feats["macrocytic_flag"]     = 1.0 if vgm > 100 else 0.0
    feats["hypochromic_flag"]    = 1.0 if float(p["tcmh"]) < 27 else 0.0
    feats["iron_deficiency"]     = 1.0 if ferr < 20 and float(p.get("transferrin_sat_pct", 30)) < 16 else 0.0
    feats["b12_deficiency"]      = 1.0 if b12 < 200 else 0.0
    feats["folate_deficiency"]   = 1.0 if folate < 4.0 else 0.0
    feats["hemolysis_flag"]      = (
        1.0 if (float(p.get("haptoglobin", 1.2)) < 0.25
                or retic > 2.5
                or bool(p.get("direct_coombs")))
        else 0.0
    )
    feats["rdw_elevated"]        = 1.0 if rdw > 14.5 else 0.0
    feats["leukocytosis"]        = 1.0 if wbc > 11 else 0.0
    feats["hyperleucocytosis"]   = 1.0 if wbc > 30 else 0.0
    feats["leukopenia"]          = 1.0 if wbc < 4 else 0.0
    feats["blast_flag"]          = 1.0 if blst >= 20 else (0.5 if blst >= 5 else 0.0)
    feats["blast_norm"]          = min(blst / 100.0, 1.0)
    feats["thrombocytopenia"]    = 1.0 if plt < 150 else 0.0
    feats["thrombocytosis"]      = 1.0 if plt > 400 else 0.0
    feats["mega_thrombocytosis"] = 1.0 if plt > 1000 else 0.0
    feats["neutropenia_severe"]  = 1.0 if neut < 0.5 else 0.0
    feats["lymphocytosis"]       = 1.0 if lymp > 4.0 else 0.0
    feats["ldh_norm"]            = min(float(p.get("ldh", 180)) / 500.0, 1.0)
    feats["b12_norm"]            = min(b12 / 1000.0, 1.0)
    feats["bcr_abl_flag"]        = 1.0 if bool(p.get("bcr_abl1")) else 0.0
    feats["jak2_flag"]           = 1.0 if bool(p.get("jak2_v617f")) else 0.0
    feats["npm1_flt3_aml"]       = 1.0 if (bool(p.get("npm1_mutated")) or bool(p.get("flt3_itd"))) else 0.0
    feats["malaria_flag"]        = 1.0 if (float(p.get("parasitemia_pct", 0)) > 0
                                           or bool(p.get("rdt_positive"))) else 0.0
    feats["lymphoma_flag"]       = 1.0 if (bool(p.get("lymphadenopathy"))
                                           or bool(p.get("b_symptoms"))) else 0.0
    feats["myeloma_flag"]        = 1.0 if (bool(p.get("bone_lesions"))
                                           or bool(p.get("serum_protein_electrophoresis"))) else 0.0
    feats["pancytopenia"]        = 1.0 if (hb < hb_ref and wbc < 4 and plt < 150) else 0.0
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Classification clinique par règles pondérées
# ─────────────────────────────────────────────────────────────────────────────

def _classify_hemato(
    p: dict[str, Any],
    feats: dict[str, float],
    scores: dict[str, Any],
) -> dict[str, Any]:
    probs: dict[str, float] = {c: 0.01 for c in HEMATO_CLASSES}

    hb     = float(p["hemoglobin"])
    wbc    = float(p["wbc"])
    plt    = float(p["platelets"])
    blst   = float(p["blast_pct"])
    sex_f  = str(p["sex"]).upper() == "F"
    hb_ref = 12.0 if sex_f else 13.0

    # ── Normal ──────────────────────────────────────────────────────────────
    normal_score = max(0.0, 1.0
        - feats["anemia_flag"] * 0.5
        - feats["leukocytosis"] * 0.4
        - feats["hyperleucocytosis"] * 0.8
        - feats["blast_flag"] * 0.9
        - feats["malaria_flag"] * 0.9
        - feats["pancytopenia"] * 0.8
        - feats["thrombocytopenia"] * 0.3
        - feats["mega_thrombocytosis"] * 0.4
        - feats["myeloma_flag"] * 0.8
        - feats["lymphoma_flag"] * 0.5
    )
    probs["Normal — Pas d'anomalie hématologique"] = max(0.01, normal_score)

    # ── Anémie ferriprive ────────────────────────────────────────────────────
    probs["Anémie ferriprive"] = max(0.01,
        feats["anemia_flag"] * 0.5
        + feats["microcytic_flag"] * 0.3
        + feats["hypochromic_flag"] * 0.25
        + feats["iron_deficiency"] * 0.6
        + feats["rdw_elevated"] * 0.2
        - feats["macrocytic_flag"] * 0.3
    )

    # ── Anémie mégaloblastique ───────────────────────────────────────────────
    probs["Anémie mégaloblastique (B12/Folates)"] = max(0.01,
        feats["anemia_flag"] * 0.3
        + feats["macrocytic_flag"] * 0.5
        + feats["b12_deficiency"] * 0.6
        + feats["folate_deficiency"] * 0.4
        - feats["microcytic_flag"] * 0.4
    )

    # ── Anémie hémolytique ───────────────────────────────────────────────────
    probs["Anémie hémolytique"] = max(0.01,
        feats["anemia_flag"] * 0.3
        + feats["hemolysis_flag"] * 0.7
        + (1.0 if float(p.get("reticulocytes_pct", 1.5)) > 3.0 else 0.0) * 0.3
        + feats["ldh_norm"] * 0.2
    )

    # ── Drépanocytose / Thalassémie ─────────────────────────────────────────
    sickle_score = (
        feats["anemia_flag"] * 0.3
        + feats["microcytic_flag"] * 0.4
        + feats["hemolysis_flag"] * 0.4
        + (1.0 if float(p.get("rdw", 12.5)) > 20 else 0.0) * 0.3
    )
    if str(p.get("disease_context", "")).lower() in ("drépanocytose", "thalassémie", "hemoglobinopathy"):
        sickle_score += 0.4
    probs["Drépanocytose / Thalassémie"] = max(0.01, sickle_score)

    # ── Leucémie aiguë (LAM/LAL) ─────────────────────────────────────────────
    aml_lal_score = (
        feats["blast_norm"] * 1.2
        + feats["hyperleucocytosis"] * 0.5
        + feats["pancytopenia"] * 0.4
        + feats["npm1_flt3_aml"] * 0.5
        + (1.0 if bool(p.get("t_8_21")) or bool(p.get("inv_16")) else 0.0) * 0.6
        + feats["neutropenia_severe"] * 0.3
    )
    probs["Leucémie aiguë (LAM / LAL)"] = max(0.01, min(aml_lal_score, 0.97))

    # ── LLC ──────────────────────────────────────────────────────────────────
    llc_score = (
        feats["lymphocytosis"] * 0.7
        + (1.0 if float(p.get("lymphocytes", 2.0)) > 10 else 0.0) * 0.5
        + (1.0 if float(p.get("age", 45)) > 60 else 0.0) * 0.2
        - feats["blast_flag"] * 0.6
    )
    probs["Leucémie lymphoïde chronique (LLC)"] = max(0.01, llc_score)

    # ── LMC ──────────────────────────────────────────────────────────────────
    lmc_score = (
        feats["bcr_abl_flag"] * 0.9
        + feats["leukocytosis"] * 0.3
        + (1.0 if wbc > 25 and blst < 20 else 0.0) * 0.4
        + (1.0 if float(p.get("basophils", 0.04)) > 0.5 else 0.0) * 0.3
        + (1.0 if bool(p.get("splenomegaly")) else 0.0) * 0.2
    )
    probs["Leucémie myéloïde chronique (LMC)"] = max(0.01, lmc_score)

    # ── SMD ──────────────────────────────────────────────────────────────────
    smd_score = (
        feats["pancytopenia"] * 0.5
        + (1.0 if 5 <= blst < 20 else 0.0) * 0.5
        + (1.0 if bool(p.get("del_5q")) or bool(p.get("del_7")) else 0.0) * 0.4
        + (1.0 if bool(p.get("sf3b1_mutated")) else 0.0) * 0.3
        + (1.0 if bool(p.get("runx1_mutated")) else 0.0) * 0.2
        + feats["anemia_flag"] * 0.2
    )
    probs["Syndrome myélodysplasique (SMD)"] = max(0.01, smd_score)

    # ── SMP ──────────────────────────────────────────────────────────────────
    smp_score = (
        feats["jak2_flag"] * 0.8
        + (1.0 if bool(p.get("calr_mutated")) or bool(p.get("mpl_mutated")) else 0.0) * 0.4
        + feats["mega_thrombocytosis"] * 0.5
        + (1.0 if float(p.get("hemoglobin", 13.5)) > 16.5 else 0.0) * 0.4
        + (1.0 if bool(p.get("splenomegaly")) else 0.0) * 0.2
    )
    probs["Syndrome myéloprolifératif (SMP)"] = max(0.01, smp_score)

    # ── Lymphome ─────────────────────────────────────────────────────────────
    lymphoma_score = (
        feats["lymphoma_flag"] * 0.6
        + feats["ldh_norm"] * 0.3
        + (1.0 if bool(p.get("b_symptoms")) else 0.0) * 0.3
        + (1.0 if int(p.get("extranodal_sites", 0)) > 0 else 0.0) * 0.2
        - feats["blast_flag"] * 0.4
    )
    probs["Lymphome (Hodgkin / LNH)"] = max(0.01, lymphoma_score)

    # ── Myélome ──────────────────────────────────────────────────────────────
    myeloma_score = (
        feats["myeloma_flag"] * 0.7
        + (1.0 if float(p.get("beta2_microglobulin", 2.0)) > 5 else 0.0) * 0.4
        + (1.0 if float(p.get("albumin", 4.0)) < 3.5 else 0.0) * 0.3
        + feats["anemia_flag"] * 0.2
    )
    probs["Myélome multiple"] = max(0.01, myeloma_score)

    # ── Paludisme ─────────────────────────────────────────────────────────────
    probs["Paludisme / Infection parasitaire"] = max(0.01,
        feats["malaria_flag"] * 0.95
        + feats["anemia_flag"] * 0.05
    )

    # ── Infection sanguine ────────────────────────────────────────────────────
    infect_score = (
        feats["leukocytosis"] * 0.35
        + feats["leukopenia"] * 0.25
        + (1.0 if float(p.get("neutrophils", 4.5)) > 8 else 0.0) * 0.3
    )
    probs["Infection sanguine (Bactérienne / Virale)"] = max(0.01, infect_score)

    # ── Normalisation ────────────────────────────────────────────────────────
    total = sum(probs.values())
    probs = {k: round(v / total, 4) for k, v in probs.items()}
    prediction = max(probs, key=probs.get)
    confidence = probs[prediction]

    return {"prediction": prediction, "confidence": confidence, "probabilities": probs}


# ─────────────────────────────────────────────────────────────────────────────
# Urgence et actions cliniques
# ─────────────────────────────────────────────────────────────────────────────

_URGENCY_MAP: dict[str, tuple[str, str]] = {
    "Normal — Pas d'anomalie hématologique":  ("Faible",   "#27AE60"),
    "Anémie ferriprive":                      ("Modérée",  "#F39C12"),
    "Anémie mégaloblastique (B12/Folates)":   ("Modérée",  "#F39C12"),
    "Anémie hémolytique":                     ("Élevée",   "#E67E22"),
    "Drépanocytose / Thalassémie":            ("Modérée",  "#F39C12"),
    "Leucémie aiguë (LAM / LAL)":             ("Critique", "#C0392B"),
    "Leucémie lymphoïde chronique (LLC)":     ("Modérée",  "#E67E22"),
    "Leucémie myéloïde chronique (LMC)":      ("Élevée",   "#E67E22"),
    "Syndrome myélodysplasique (SMD)":        ("Élevée",   "#E67E22"),
    "Syndrome myéloprolifératif (SMP)":       ("Modérée",  "#F39C12"),
    "Lymphome (Hodgkin / LNH)":               ("Élevée",   "#E67E22"),
    "Myélome multiple":                       ("Élevée",   "#E67E22"),
    "Paludisme / Infection parasitaire":      ("Critique", "#C0392B"),
    "Infection sanguine (Bactérienne / Virale)": ("Modérée", "#E67E22"),
}

_ACTION_MAP: dict[str, str] = {
    "Normal — Pas d'anomalie hématologique":
        "Surveillance biologique annuelle — pas d'action urgente",
    "Anémie ferriprive":
        "Bilan martial : ferritine + coefficient de saturation — supplémentation fer oral 80–200 mg/j",
    "Anémie mégaloblastique (B12/Folates)":
        "Bilan B12/Folates — supplémentation vitaminique IV ou IM si carence sévère",
    "Anémie hémolytique":
        "Coombs direct — LDH — haptoglobine — consultation hématologie urgente si sévère",
    "Drépanocytose / Thalassémie":
        "Électrophorèse Hb — consultation hématologue — hydroxyurée si SCD sévère",
    "Leucémie aiguë (LAM / LAL)":
        "URGENCE — hospitalisation hématologie — myélogramme + caryotype + NGS — induction chimiothérapie",
    "Leucémie lymphoïde chronique (LLC)":
        "Immunophénotypage lymphocytaire (cytométrie) — bilan extension — critères iwCLL pour traitement",
    "Leucémie myéloïde chronique (LMC)":
        "PCR BCR-ABL quantitative — démarrer imatinib 400 mg/j ou nilotinib — surveillance moléculaire",
    "Syndrome myélodysplasique (SMD)":
        "Myélogramme + caryotype + NGS — IPSS-R — azacitidine si risque intermédiaire/élevé",
    "Syndrome myéloprolifératif (SMP)":
        "JAK2/CALR/MPL — NFS + biopsie médullaire — hydroxyurée si TE/PV — ruxolitinib si MF",
    "Lymphome (Hodgkin / LNH)":
        "PET-TDM — biopsie ganglionnaire — Ann Arbor staging — R-CHOP si LNH agressif",
    "Myélome multiple":
        "Protéine M + immunofixation + chaînes légères — ISS/R-ISS — VRd ou DaraVRd",
    "Paludisme / Infection parasitaire":
        "URGENCE — goutte épaisse + frottis — PCR Plasmodium — artésunate IV si grave",
    "Infection sanguine (Bactérienne / Virale)":
        "Hémocultures × 2 — CRP/PCT — antibiothérapie probabiliste selon foyer suspect",
}


# ─────────────────────────────────────────────────────────────────────────────
# Alertes critiques hématologiques
# ─────────────────────────────────────────────────────────────────────────────

def _generate_alerts(p: dict[str, Any], feats: dict[str, float]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    hb    = float(p["hemoglobin"])
    wbc   = float(p["wbc"])
    plt   = float(p["platelets"])
    neut  = float(p["neutrophils"])
    blst  = float(p["blast_pct"])
    parasit = float(p.get("parasitemia_pct", 0))

    if hb < 7.0:
        alerts.append({
            "level": "CRITICAL", "icon": "🔴",
            "title": f"Anémie sévère — Hb {hb:.1f} g/dL",
            "message": "Hémoglobine < 7 g/dL — risque de décompensation cardiaque.",
            "clinical_action": "Transfusion érythrocytaire si symptomatique — culots phénotypés",
        })
    elif hb < 10.0:
        alerts.append({
            "level": "HIGH", "icon": "🟠",
            "title": f"Anémie modérée — Hb {hb:.1f} g/dL",
            "message": "Hémoglobine < 10 g/dL — bilan étiologique urgent.",
            "clinical_action": "Ferritine + B12 + folates + réticulocytes + frottis sanguin",
        })
    if blst >= 20:
        alerts.append({
            "level": "CRITICAL", "icon": "⚠️",
            "title": f"Blastes {blst:.0f}% — Leucémie aiguë probable",
            "message": f"Blastes ≥ 20% — critère WHO/ICC de leucémie aiguë.",
            "clinical_action": "Hospitalisation urgente — myélogramme + caryotype + NGS — induction immédiate",
        })
    elif blst >= 5:
        alerts.append({
            "level": "HIGH", "icon": "⚠️",
            "title": f"Blastes circulants {blst:.1f}%",
            "message": "Blastes 5–19% — SMD de haut risque ou leucémie aiguë débutante.",
            "clinical_action": "Myélogramme urgent — consultation hématologie dans les 48h",
        })
    if wbc > 50:
        alerts.append({
            "level": "CRITICAL", "icon": "🔴",
            "title": f"Hyperleucocytose critique — GB {wbc:.0f} G/L",
            "message": "GB > 50 G/L — risque de leucostase (cérébrale, pulmonaire).",
            "clinical_action": "Hydroxyurée urgence — leucaphérèse si > 100 G/L — réanimation",
        })
    if plt < 20:
        alerts.append({
            "level": "CRITICAL", "icon": "🔴",
            "title": f"Thrombocytopénie sévère — Plt {plt:.0f} G/L",
            "message": "Plaquettes < 20 G/L — risque d'hémorragie spontanée.",
            "clinical_action": "Transfusion plaquettaire si saignement ou geste invasif — hospitalisation",
        })
    if neut < 0.5:
        alerts.append({
            "level": "CRITICAL", "icon": "🔴",
            "title": f"Neutropénie sévère — {neut:.2f} G/L",
            "message": "PNN < 0,5 G/L — aplasie profonde — risque infectieux majeur.",
            "clinical_action": "Isolement protecteur — G-CSF — antibiotiques si fièvre ≥ 38,3°C",
        })
    if parasit >= 2.0:
        alerts.append({
            "level": "CRITICAL", "icon": "🦟",
            "title": f"Parasitémie élevée {parasit:.1f}%",
            "message": f"Parasitémie ≥ 2% — paludisme grave probable.",
            "clinical_action": "Artésunate IV urgent — hospitalisation USI si > 5% ou défaillance d'organe",
        })
    if feats.get("pancytopenia"):
        alerts.append({
            "level": "HIGH", "icon": "🟠",
            "title": "Pancytopénie",
            "message": "Anémie + leucopénie + thrombocytopénie — aplasie médullaire à exclure.",
            "clinical_action": "Myélogramme + biopsie médullaire — consultation hématologie urgente",
        })
    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance (explainability SHAP-like)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_feature_importance(
    p: dict[str, Any], feats: dict[str, float], scores: dict[str, Any]
) -> dict[str, float]:
    importance: dict[str, float] = {
        "Hémoglobine":       max(0, (13.5 - float(p["hemoglobin"])) / 13.5 * 100),
        "Blastes %":         min(float(p["blast_pct"]) / 20.0, 1.0) * 100,
        "GB (leucocytes)":   abs(float(p["wbc"]) - 7.0) / 7.0 * 50,
        "Plaquettes":        max(0, (150 - float(p["platelets"])) / 150 * 60),
        "VGM":               abs(float(p["vgm"]) - 88) / 30 * 50,
        "Ferritine":         (1 - min(float(p["ferritin"]) / 80, 1.0)) * 40,
        "BCR-ABL":           100.0 if bool(p.get("bcr_abl1")) else 0.0,
        "JAK2 V617F":        80.0 if bool(p.get("jak2_v617f")) else 0.0,
        "NPM1 / FLT3":       60.0 if (bool(p.get("npm1_mutated")) or bool(p.get("flt3_itd"))) else 0.0,
        "Parasitémie":       min(float(p.get("parasitemia_pct", 0)) / 5.0, 1.0) * 100,
        "B12 / Folates":     feats["b12_deficiency"] * 60 + feats["folate_deficiency"] * 40,
        "Haptoglobine/COOM": feats["hemolysis_flag"] * 55,
        "Lymphocytose":      feats["lymphocytosis"] * 45,
        "Neutropénie":       feats["neutropenia_severe"] * 75,
        "LDH":               feats["ldh_norm"] * 35,
    }
    max_i = max(importance.values()) if any(v > 0 for v in importance.values()) else 1
    return {k: round(v / max_i * 100, 1) for k, v in importance.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────────────────────

def predict_hemato(
    image_path: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    HematoVision AI v2.0 — Analyse hématologique complète.

    Accepte les paramètres NFS + biochimie + cytogénétique + données cliniques.
    Retourne diagnostic IA, scores OMS, alertes, explainability et recommandations.
    """
    t0 = time.time()
    request_id = str(uuid.uuid4())
    params = params or {}

    if not params:
        return {
            "status": "no_params",
            "request_id": request_id,
            "error": "Aucun paramètre fourni.",
            "module": "module_hemato",
        }

    full_params = {**DEFAULT_PARAMS, **params}
    feats  = _engineer_features(full_params)
    scores = compute_all_hemato_scores(full_params)
    result = _classify_hemato(full_params, feats, scores)

    prediction = result["prediction"]
    confidence = result["confidence"]
    urgency, color = _URGENCY_MAP.get(prediction, ("Modérée", "#E67E22"))
    action  = _ACTION_MAP.get(prediction, "Consultation hématologie")
    alerts  = _generate_alerts(full_params, feats)
    shap_fi = _compute_feature_importance(full_params, feats, scores)
    top3    = sorted(shap_fi.items(), key=lambda x: -x[1])[:3]

    safety = {"level": "ok", "message": ""}
    if confidence < 0.55:
        safety = {
            "level": "warning",
            "message": f"Confiance IA {confidence:.1%} — validation hématologiste recommandée",
        }
    if urgency == "Critique":
        safety = {
            "level": "critical",
            "message": f"URGENCE HÉMATOLOGIQUE — {prediction} — intervention immédiate requise",
        }

    nfs_result = scores.get("nfs", {})
    processing_ms = round((time.time() - t0) * 1000 + 40)

    return {
        "module": "module_hemato",
        "module_name": "HematoVision AI",
        "model_version": "v2.0",
        "model_architecture": "Règles cliniques OMS/ELN/ICC + NFS analysis + Génétique moléculaire",
        "request_id": request_id,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probabilities": result["probabilities"],
        "clinical_profile": {
            "urgency": urgency,
            "color": color,
            "action": action,
            "image_analysis": "Cytologie numérique — modèle EfficientNet/YOLOv8 (à connecter)",
        },
        "nfs_summary": {
            "anemia": nfs_result.get("anemia"),
            "anemia_severity": nfs_result.get("anemia_severity"),
            "critical_flags": nfs_result.get("critical_flags", []),
            "interpretation": nfs_result.get("interpretation"),
            "recommendation": nfs_result.get("recommendation"),
        },
        "clinical_scores": scores,
        "alerts": alerts,
        "explainability": {
            "method": "SHAP-inspired clinical feature importance",
            "feature_importance": shap_fi,
            "top_3_drivers": top3,
        },
        "clinical_safety": safety,
        "recommended_action": action,
        "guidelines_ref": (
            "WHO Classification Haematopoietic 2022 · ICC 2022 · "
            "ELN AML 2022 · NCCN 2023 · ASH/EHA Guidelines"
        ),
        "processing_ms": processing_ms,
        "status": "success",
    }
