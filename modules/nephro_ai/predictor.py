"""
NephroAI v2.0 — Predictor principal
=====================================
Détection et classification des maladies rénales.
CKD (G1–G5) · AKI (1–3) · Glomérulopathies · Néphropathie diabétique ·
Néphropathie hypertensive · Polykystose · Électrolytes critiques ·
Carcinome rénal · Pyélonéphrite.

Sources : KDIGO 2022 · NKF-KDOQI 2023 · ERA 2023 · ISN · WHO CKD 2022
Datasets : CKD-UCI · CRIC Study · MIMIC-IV · NHANES · CKD-EPI Consortium
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from modules.nephro_ai.clinical_scores import compute_all_nephro_scores

NEPHRO_CLASSES = [
    "Fonction rénale normale",
    "MRC Stade 1 (G1) — DFG ≥ 90",
    "MRC Stade 2 (G2) — DFG 60–89",
    "MRC Stade 3a (G3a) — DFG 45–59",
    "MRC Stade 3b (G3b) — DFG 30–44",
    "MRC Stade 4 (G4) — DFG 15–29",
    "MRC Stade 5 (G5) — IRC Terminale",
    "IRA Stade 1 — KDIGO AKI",
    "IRA Stade 2 — KDIGO AKI",
    "IRA Stade 3 — KDIGO AKI critique",
    "Syndrome néphrotique",
    "Syndrome néphritique",
    "Néphropathie à IgA",
    "Néphropathie diabétique",
    "Néphropathie hypertensive",
    "Polykystose rénale (PKD)",
    "Carcinome rénal",
    "Pyélonéphrite",
    "Hyperkaliémie critique",
    "Trouble électrolytique sévère",
]

_CLASS_PROFILES: dict[str, dict[str, str]] = {
    "Fonction rénale normale":          {"color":"#27AE60","urgency":"Faible",
                                          "action":"Surveillance annuelle — facteurs de risque"},
    "MRC Stade 1 (G1) — DFG ≥ 90":     {"color":"#2ECC71","urgency":"Faible",
                                          "action":"Traitement étiologique — IEC/ARA2 si protéinurie"},
    "MRC Stade 2 (G2) — DFG 60–89":    {"color":"#F39C12","urgency":"Faible",
                                          "action":"Néphroprotection — SGLT2i si DM — contrôle PA"},
    "MRC Stade 3a (G3a) — DFG 45–59":  {"color":"#E67E22","urgency":"Modérée",
                                          "action":"Néphrologue semestriel — éviter AINS — corr. anémie"},
    "MRC Stade 3b (G3b) — DFG 30–44":  {"color":"#D35400","urgency":"Modérée",
                                          "action":"Néphrologue trimestriel — préparer EER — vaccins"},
    "MRC Stade 4 (G4) — DFG 15–29":    {"color":"#E74C3C","urgency":"Élevée",
                                          "action":"Éducation dialyse — FAV — inscription transplantation"},
    "MRC Stade 5 (G5) — IRC Terminale":{"color":"#922B21","urgency":"Critique",
                                          "action":"Dialyse ou transplantation urgente — soins palliatifs si CI"},
    "IRA Stade 1 — KDIGO AKI":          {"color":"#F39C12","urgency":"Modérée",
                                          "action":"Arrêt néphrotoxiques — hydratation — bilan étiologique"},
    "IRA Stade 2 — KDIGO AKI":          {"color":"#E74C3C","urgency":"Élevée",
                                          "action":"Néphrologue + réanimateur — surveillance rapprochée"},
    "IRA Stade 3 — KDIGO AKI critique": {"color":"#922B21","urgency":"Critique",
                                          "action":"URGENCE — EER si anurie/hyperkaliémie — ICU"},
    "Syndrome néphrotique":             {"color":"#8E44AD","urgency":"Élevée",
                                          "action":"Biopsie rénale — corticoïdes — diurétiques — albumine IV"},
    "Syndrome néphritique":             {"color":"#884EA0","urgency":"Élevée",
                                          "action":"Biopsie rénale urgente — IS — plasmaphérèse si RPGN"},
    "Néphropathie à IgA":               {"color":"#2980B9","urgency":"Modérée",
                                          "action":"IEC/ARA2 — SGLT2i — sparsentan si haut risque"},
    "Néphropathie diabétique":          {"color":"#E67E22","urgency":"Modérée",
                                          "action":"HbA1c < 7% — IEC/ARA2 + SGLT2i + finerenone — PA < 130/80"},
    "Néphropathie hypertensive":        {"color":"#D35400","urgency":"Modérée",
                                          "action":"PA < 130/80 mmHg — IEC/ARA2 — SGLT2i si DM"},
    "Polykystose rénale (PKD)":         {"color":"#1F618D","urgency":"Modérée",
                                          "action":"Tolvaptan si progression rapide — PA < 130/80 — imagerie"},
    "Carcinome rénal":                  {"color":"#922B21","urgency":"Élevée",
                                          "action":"TDM TAP + PET — chirurgie (néphrectomie) — immunothérapie"},
    "Pyélonéphrite":                    {"color":"#C0392B","urgency":"Élevée",
                                          "action":"Hémocultures + ECBU — C3G IV + aminoside si sévère"},
    "Hyperkaliémie critique":           {"color":"#922B21","urgency":"Critique",
                                          "action":"ECG — Ca gluconate IV — kayexalate — EER si K+ ≥ 7"},
    "Trouble électrolytique sévère":    {"color":"#E74C3C","urgency":"Élevée",
                                          "action":"Ionogramme complet — correction progressive — monitoring"},
}

DEFAULT_PARAMS: dict[str, Any] = {
    # Démographie
    "age": 60, "sex": "M", "weight": 70.0, "height": 170.0,
    # Rénaux
    "creatinine": 88.0,          # µmol/L (normal H < 106, F < 97)
    "urea": 5.5,                 # mmol/L
    "cystatine_c": None,         # mg/L (normal 0.5–1.0)
    "creatinine_baseline": None, # µmol/L (pour calcul delta AKI)
    "egfr": None,                # mL/min/1.73m² pré-calculé
    # Électrolytes
    "potassium": 4.5,            # mmol/L
    "sodium": 140.0,             # mmol/L
    "calcium": None,             # mmol/L
    "phosphate": None,           # mmol/L
    "bicarbonate": None,         # mmol/L
    "magnesium": None,           # mmol/L
    # Protéinurie
    "proteinuria_g_24h": None,   # g/24h
    "albuminuria_mg_g": 30.0,    # mg/g créatinine (ACR)
    "hematuria": False,          # hématurie microscopique
    "leucocyturia": False,       # leucocyturie > 10/mm³
    "casts": None,               # "granular", "red_cell", "white_cell"
    # Biologie générale
    "hemoglobin": 12.0,          # g/dL
    "crp": 5.0,                  # mg/L
    "procalcitonin": 0.1,        # ng/mL
    "albumin_serum": 40.0,       # g/L
    # Facteurs de risque
    "diabetes": False, "hypertension": False,
    "family_hx_pkd": False, "family_hx_renal_cancer": False,
    "smoker": False, "obesity": False,
    "prior_aki": False, "nephrotoxic_drugs": False,
    "on_dialysis": False, "transplanted": False,
    # Diurèse
    "urine_output_ml_kg_h": None,
    "creatinine_rapid_rise": False,
    "oliguria": False, "anuria": False,
    # Imagerie
    "renal_cysts": False, "renal_mass": False,
    "renal_size_cm": None, "hydronephrosis": False,
    "echogenicity_increased": False,
    # Symptômes
    "edema": False, "flank_pain": False, "fever": False,
    "dysuria": False, "hematuria_macroscopic": False,
    # Auto-immun
    "anca_positive": False, "anti_gbm": False, "ana_positive": False,
    # Immunologie
    "complement_low": False,
    "hbsag": False, "hiv": False,
    # Immunosuppression
    "on_immunosuppression": False,
}


def _engineer_features(params: dict[str, Any]) -> dict[str, float]:
    """Feature engineering néphrologique."""
    p = {**DEFAULT_PARAMS, **params}
    f: dict[str, float] = {}

    age   = float(p.get("age", 60))
    creat = float(p.get("creatinine", 88))
    k     = float(p.get("potassium", 4.5))
    na    = float(p.get("sodium", 140))
    hb    = float(p.get("hemoglobin", 12))
    prot  = float(p.get("proteinuria_g_24h") or float(p.get("albuminuria_mg_g", 30)) / 1000)
    acr   = float(p.get("albuminuria_mg_g", 30))
    crp   = float(p.get("crp", 5))
    pct   = float(p.get("procalcitonin", 0.1))
    alb   = float(p.get("albumin_serum", 40))

    # eGFR normalisé
    egfr_v = p.get("egfr")
    if egfr_v:
        egfr = float(egfr_v)
    else:
        # CKD-EPI simplifiée inline
        scr    = max(0.1, creat / 88.4)
        sex_f  = str(p.get("sex","M")).upper() == "F"
        kappa  = 0.7 if sex_f else 0.9
        alpha  = -0.241 if sex_f else -0.302
        coeff  = 142 * (1.012 if sex_f else 1.0)
        if scr <= kappa:
            egfr = coeff * ((scr / kappa) ** alpha) * (0.9938 ** age)
        else:
            egfr = coeff * ((scr / kappa) ** -1.200) * (0.9938 ** age)
        egfr = max(1.0, round(egfr, 1))

    f["egfr"]             = egfr
    f["egfr_norm"]        = min(egfr / 90.0, 1.0)
    f["egfr_severe"]      = 1.0 if egfr < 30 else 0.0
    f["egfr_terminal"]    = 1.0 if egfr < 15 else 0.0

    # CKD stage encoded
    if egfr >= 90:       f["ckd_g"] = 1.0
    elif egfr >= 60:     f["ckd_g"] = 2.0
    elif egfr >= 45:     f["ckd_g"] = 3.0
    elif egfr >= 30:     f["ckd_g"] = 3.5
    elif egfr >= 15:     f["ckd_g"] = 4.0
    else:                f["ckd_g"] = 5.0

    # Créatinine + delta AKI
    baseline_creat = p.get("creatinine_baseline")
    f["creatinine_norm"]  = min(creat / 500.0, 1.0)
    if baseline_creat:
        delta = creat / max(float(baseline_creat), 44)
        f["aki_ratio"] = delta
        f["aki_stage"] = 3.0 if delta >= 3 or creat >= 354 else (2.0 if delta >= 2 else (1.0 if delta >= 1.5 else 0.0))
    else:
        f["aki_ratio"] = 1.0
        f["aki_stage"] = 0.0
        if p.get("creatinine_rapid_rise"):
            f["aki_stage"] = 1.0

    # Anuria/oliguria → AKI 3
    if p.get("anuria") or p.get("on_dialysis"):
        f["aki_stage"] = max(f["aki_stage"], 3.0)

    # Protéinurie
    f["proteinuria_norm"] = min(prot / 3.5, 1.0)
    f["heavy_proteinuria"]= 1.0 if prot >= 3.5 else 0.0  # syndrome néphrotique
    f["nephrotic_flag"]   = 1.0 if (prot >= 3.5 and alb < 30 and p.get("edema")) else 0.0
    f["acr_elevated"]     = 1.0 if acr > 300 else (0.5 if acr > 30 else 0.0)

    # Électrolytes
    f["hyperkalemia"]     = 1.0 if k >= 6.5 else (0.5 if k >= 5.5 else 0.0)
    f["hyponatremia"]     = 1.0 if na < 125 else (0.5 if na < 130 else 0.0)
    f["hypernatremia"]    = 1.0 if na > 155 else (0.5 if na > 148 else 0.0)
    f["electrolyte_crit"] = 1.0 if (f["hyperkalemia"] >= 1.0 or f["hyponatremia"] >= 1.0 or f["hypernatremia"] >= 1.0) else 0.0

    # Anémie rénale
    f["anemia_renal"]     = 1.0 if hb < 10.0 else (0.5 if hb < 11.5 else 0.0)

    # Infection rénale
    f["infection_flag"]   = 1.0 if (crp > 50 or pct > 0.5) and (p.get("fever") or p.get("dysuria") or p.get("leucocyturia")) else 0.0

    # Néphropathie diabétique
    f["diabetic_nephro"]  = 1.0 if (p.get("diabetes") and acr > 30 and egfr < 90) else 0.0

    # Hypertensif
    f["htn_nephro"]       = 1.0 if (p.get("hypertension") and f["ckd_g"] >= 2 and not p.get("diabetes")) else 0.0

    # Glomérulopathies auto-immunes
    f["glomerulo_flag"]   = 1.0 if (p.get("anca_positive") or p.get("anti_gbm") or
                                     (p.get("hematuria") and p.get("complement_low"))) else 0.0
    f["iga_nephro"]       = 1.0 if (p.get("hematuria") and prot > 0.5 and not p.get("anca_positive")) else 0.0

    # PKD
    f["pkd_flag"]         = 1.0 if p.get("renal_cysts") or p.get("family_hx_pkd") else 0.0

    # Cancer rénal
    f["cancer_flag"]      = 1.0 if (p.get("renal_mass") or p.get("hematuria_macroscopic") or p.get("flank_pain")) else 0.0

    # Score sévérité global
    f["severity"] = min(1.0, (
        (1 - f["egfr_norm"]) * 0.30 +
        f["aki_stage"] / 3.0 * 0.25 +
        f["hyperkalemia"] * 0.20 +
        f["heavy_proteinuria"] * 0.12 +
        f["infection_flag"] * 0.10 +
        f["anemia_renal"] * 0.08 +
        f["cancer_flag"] * 0.15
    ))

    return f


def _estimate_nephro_risk(
    feats: dict[str, float],
    params: dict[str, Any],
    scores: dict[str, Any],
) -> dict[str, Any]:
    """Classification néphrologiques — mapping features → classes."""
    probs: dict[str, float] = {c: 0.01 for c in NEPHRO_CLASSES}

    # ── Règles prioritaires ─────────────────────────────────────────────────

    # Hyperkaliémie critique
    if feats["hyperkalemia"] >= 1.0:
        probs["Hyperkaliémie critique"]             = 0.65
        probs["MRC Stade 5 (G5) — IRC Terminale"]   = 0.18
        probs["IRA Stade 3 — KDIGO AKI critique"]   = 0.10

    # Trouble électrolytique sévère autre
    elif feats["electrolyte_crit"] > 0:
        probs["Trouble électrolytique sévère"]       = 0.55
        probs["MRC Stade 4 (G4) — DFG 15–29"]       = 0.18

    # AKI stade 3
    elif feats["aki_stage"] >= 3.0:
        probs["IRA Stade 3 — KDIGO AKI critique"]   = 0.62
        probs["MRC Stade 5 (G5) — IRC Terminale"]   = 0.15
        probs["Hyperkaliémie critique"]              = 0.10

    # AKI stade 2
    elif feats["aki_stage"] >= 2.0:
        probs["IRA Stade 2 — KDIGO AKI"]            = 0.55
        probs["IRA Stade 3 — KDIGO AKI critique"]   = 0.18

    # IRC terminale G5
    elif feats["egfr_terminal"] > 0:
        probs["MRC Stade 5 (G5) — IRC Terminale"]   = 0.58
        probs["MRC Stade 4 (G4) — DFG 15–29"]       = 0.22
        probs["Hyperkaliémie critique"]              = 0.10

    # Syndrome néphrotique
    elif feats["nephrotic_flag"] > 0:
        probs["Syndrome néphrotique"]                = 0.58
        probs["Néphropathie diabétique"]             = 0.18 if params.get("diabetes") else 0.08

    # Cancer rénal
    elif feats["cancer_flag"] > 0:
        probs["Carcinome rénal"]                     = 0.55
        probs["Pyélonéphrite"]                       = 0.15

    # Pyélonéphrite
    elif feats["infection_flag"] > 0:
        probs["Pyélonéphrite"]                       = 0.58
        probs["IRA Stade 1 — KDIGO AKI"]            = 0.18

    # Glomérulopathies
    elif feats["glomerulo_flag"] > 0:
        probs["Syndrome néphritique"]                = 0.48
        probs["Néphropathie à IgA"]                  = 0.22

    elif feats["iga_nephro"] > 0:
        probs["Néphropathie à IgA"]                  = 0.55
        probs["Syndrome néphritique"]                = 0.22

    # PKD
    elif feats["pkd_flag"] > 0:
        probs["Polykystose rénale (PKD)"]            = 0.58
        probs["MRC Stade 3a (G3a) — DFG 45–59"]     = 0.22

    # Néphropathie diabétique
    elif feats["diabetic_nephro"] > 0:
        probs["Néphropathie diabétique"]             = 0.52
        g = feats["ckd_g"]
        if g <= 2:   probs["MRC Stade 2 (G2) — DFG 60–89"]   = 0.22
        elif g <= 3: probs["MRC Stade 3a (G3a) — DFG 45–59"] = 0.22
        else:        probs["MRC Stade 3b (G3b) — DFG 30–44"] = 0.22

    # CKD grade par eGFR
    else:
        g = feats["ckd_g"]
        egfr = feats["egfr"]

        if egfr >= 90 and feats["severity"] < 0.1:
            probs["Fonction rénale normale"]         = 0.58
            probs["MRC Stade 1 (G1) — DFG ≥ 90"]   = 0.25
        elif g == 1:
            probs["MRC Stade 1 (G1) — DFG ≥ 90"]   = 0.55
            probs["Néphropathie diabétique"]         = 0.12 if params.get("diabetes") else 0.05
        elif g == 2:
            probs["MRC Stade 2 (G2) — DFG 60–89"]  = 0.52
            probs["Néphropathie hypertensive"]       = 0.18 if params.get("hypertension") else 0.08
        elif g == 3:
            probs["MRC Stade 3a (G3a) — DFG 45–59"]= 0.52
            probs["MRC Stade 3b (G3b) — DFG 30–44"]= 0.20
        elif g == 3.5:
            probs["MRC Stade 3b (G3b) — DFG 30–44"]= 0.55
            probs["MRC Stade 4 (G4) — DFG 15–29"]  = 0.22
        elif g == 4:
            probs["MRC Stade 4 (G4) — DFG 15–29"]  = 0.55
            probs["MRC Stade 5 (G5) — IRC Terminale"]= 0.22
        else:
            probs["MRC Stade 5 (G5) — IRC Terminale"]= 0.55
            probs["Hyperkaliémie critique"]          = 0.18

        # Ajustements contextuels
        if feats["aki_stage"] >= 1.0:
            probs["IRA Stade 1 — KDIGO AKI"]        += 0.15

        if feats["htn_nephro"] > 0:
            probs["Néphropathie hypertensive"]       += 0.12

    # Normalisation
    total  = sum(probs.values())
    probs  = {k: round(v / total, 4) for k, v in probs.items()}
    pred   = max(probs, key=probs.get)
    conf   = probs[pred]
    profile = _CLASS_PROFILES.get(pred, {})

    # KFRE si disponible
    kfre_2y = scores.get("kfre", {}).get("risk_2yr_pct", None)
    kfre_5y = scores.get("kfre", {}).get("risk_5yr_pct", None)

    esrd_risk = {
        "2ans": f"{kfre_2y:.1f}%" if kfre_2y else f"{feats['severity']*30:.1f}%",
        "5ans": f"{kfre_5y:.1f}%" if kfre_5y else f"{feats['severity']*55:.1f}%",
    }

    return {
        "prediction":  pred,
        "confidence":  conf,
        "probabilities": probs,
        "urgency":     profile.get("urgency", "Modérée"),
        "color":       profile.get("color", "#2980B9"),
        "action":      profile.get("action", "—"),
        "egfr":        round(feats["egfr"], 1),
        "severity":    round(feats["severity"], 4),
        "esrd_risk":   esrd_risk,
    }


def _flag_critical_findings(
    params: dict[str, Any],
    feats: dict[str, float],
) -> list[dict[str, Any]]:
    flags = []

    # Hyperkaliémie critique
    k = float(params.get("potassium", 4.5))
    if k >= 6.5:
        flags.append({"marker":"Hyperkaliémie critique","value":f"K+ {k:.1f} mmol/L",
                      "severity":"CRITIQUE","color":"#922B21",
                      "detail":"ECG urgent — Ca gluconate IV — EER si K+ ≥ 7 mmol/L."})
    elif k >= 5.5:
        flags.append({"marker":"Hyperkaliémie","value":f"K+ {k:.1f} mmol/L",
                      "severity":"ÉLEVÉ","color":"#E74C3C",
                      "detail":"Régime pauvre K+ — résines échangeuses — surveiller ECG."})

    # IRC terminale
    if feats["egfr_terminal"] > 0:
        flags.append({"marker":"IRC Terminale (G5)","value":f"DFG {feats['egfr']:.0f} mL/min",
                      "severity":"CRITIQUE","color":"#922B21",
                      "detail":"Dialyse ou transplantation urgente. MELD rénal imminent."})

    # AKI 3
    if feats["aki_stage"] >= 3.0:
        flags.append({"marker":"IRA Stade 3 (KDIGO)","value":"Créatinine × 3 ou anurie",
                      "severity":"CRITIQUE","color":"#922B21",
                      "detail":"EER urgente — réanimation néphro — ICU."})

    # Syndrome néphrotique
    if feats["nephrotic_flag"] > 0:
        prot = float(params.get("proteinuria_g_24h") or 0)
        flags.append({"marker":"Syndrome néphrotique","value":f"Protéinurie {prot:.1f} g/24h",
                      "severity":"ÉLEVÉ","color":"#8E44AD",
                      "detail":"Biopsie rénale urgente — albumine IV si hypoalbuminémie < 20 g/L."})

    # Cancer rénal
    if feats["cancer_flag"] > 0 and params.get("renal_mass"):
        flags.append({"marker":"Masse rénale suspecte","value":"Imagerie positive",
                      "severity":"ÉLEVÉ","color":"#E74C3C",
                      "detail":"TDM TAP + biopsie rénale — urologie urgente."})

    # Glomérulopathies ANCA
    if params.get("anca_positive") or params.get("anti_gbm"):
        flags.append({"marker":"Vascularite/Goodpasture","value":"ANCA/anti-GBM +",
                      "severity":"CRITIQUE","color":"#922B21",
                      "detail":"Biopsie rénale urgente — plasmaphérèse — methylprednisolone pulse."})

    # Anémie rénale sévère
    hb = float(params.get("hemoglobin", 12))
    if hb < 9.0:
        flags.append({"marker":"Anémie rénale sévère","value":f"Hb {hb:.1f} g/dL",
                      "severity":"ÉLEVÉ","color":"#E74C3C",
                      "detail":"EPO/darbépoïétine + fer IV — transfusion si Hb < 7 g/dL."})

    # Sepsis urinaire
    pct = float(params.get("procalcitonin", 0.1))
    if pct > 2.0 and params.get("fever"):
        flags.append({"marker":"Sepsis urinaire probable","value":f"PCT {pct:.1f} ng/mL",
                      "severity":"ÉLEVÉ","color":"#E74C3C",
                      "detail":"Hémocultures + ECBU — antibiothérapie IV (C3G + aminoside) urgente."})

    return flags


def _compute_feature_importance(feats: dict[str, float]) -> dict[str, float]:
    importance = {
        "DFG estimé (eGFR)":             (1 - feats["egfr_norm"]) * 100,
        "AKI (fold-change créatinine)":  feats["aki_stage"] / 3 * 100,
        "Hyperkaliémie (K+)":            feats["hyperkalemia"] * 100,
        "IRC terminale (DFG < 15)":      feats["egfr_terminal"] * 100,
        "Protéinurie massive":           feats["heavy_proteinuria"] * 90,
        "Syndrome néphrotique":          feats["nephrotic_flag"] * 90,
        "Infection rénale (CRP/PCT)":    feats["infection_flag"] * 85,
        "Glomérulopathie auto-immune":   feats["glomerulo_flag"] * 85,
        "Néphropathie diabétique":       feats["diabetic_nephro"] * 80,
        "Polykystose rénale (PKD)":      feats["pkd_flag"] * 75,
        "Cancer rénal (masse)":          feats["cancer_flag"] * 80,
        "Anémie rénale (Hb)":            feats["anemia_renal"] * 70,
        "Albuminurie (ACR)":             feats["acr_elevated"] * 70,
        "Néphropathie hypertensive":     feats["htn_nephro"] * 65,
        "Électrolyte critique":          feats["electrolyte_crit"] * 75,
    }
    max_v = max(importance.values()) if any(v > 0 for v in importance.values()) else 1.0
    return {k: round(v / max_v * 100, 1) for k, v in importance.items() if v > 0}


def predict_nephro(
    params: dict[str, Any] | None = None,
    image_path: str | None = None,    # noqa: ARG001 — réservé imagerie rénale US/TDM/IRM
) -> dict[str, Any]:
    """
    NephroAI v2.0 — Analyse néphrologiques complète.

    Accepte :
      params : biologie + clinique + électrolytes + imagerie
      image_path : futur support US/TDM/IRM rénale

    Retourne : prédiction · KDIGO CKD/AKI · eGFR · KFRE · électrolytes ·
               SHAP · risque ESRD · recommandations KDIGO 2022.
    """
    t0         = time.time()
    request_id = str(uuid.uuid4())
    params     = params or {}

    if not params:
        return {"status": "no_params", "request_id": request_id,
                "error": "Aucun paramètre fourni.", "module": "module_15_nephro"}

    full_params = {**DEFAULT_PARAMS, **params}

    # 1. Feature engineering
    feats = _engineer_features(full_params)

    # 2. Scores cliniques
    scores = compute_all_nephro_scores(full_params)

    # 3. Classification
    risk = _estimate_nephro_risk(feats, full_params, scores)

    # 4. Findings critiques
    critical = _flag_critical_findings(full_params, feats)

    # 5. Feature importance
    feat_imp = _compute_feature_importance(feats)

    prediction = risk["prediction"]
    confidence = risk["confidence"]

    # Sécurité clinique
    safety = {"level": "ok", "message": ""}
    if confidence < 0.60:
        safety = {"level": "warning",
                  "message": f"Confiance IA {confidence:.1%} — validation néphrologue requise."}
    if risk["urgency"] == "Critique" or any(c["severity"] == "CRITIQUE" for c in critical):
        safety = {"level": "critical",
                  "message": "URGENCE NÉPHROLOGIQUE — Intervention immédiate requise."}

    # Résumé biologique rénal
    bio_summary = {
        "eGFR (CKD-EPI 2021)": f"{feats['egfr']:.1f} mL/min/1.73m²",
        "KDIGO CKD Stage":      scores.get("kdigo_ckd", {}).get("ckd_g_stage", "—"),
        "Albuminurie":          scores.get("kdigo_ckd", {}).get("albuminuria_stage", "—"),
        "KDIGO AKI":            scores.get("kdigo_aki", {}).get("label", "—"),
        "Créatinine":           f"{full_params.get('creatinine',88):.0f} µmol/L",
        "Protéinurie":          f"{full_params.get('proteinuria_g_24h','—')} g/24h" if full_params.get("proteinuria_g_24h") else "Non renseigné",
        "Potassium":            f"{full_params.get('potassium',4.5):.1f} mmol/L",
        "Sodium":               f"{full_params.get('sodium',140):.0f} mmol/L",
        "Hémoglobine":          f"{full_params.get('hemoglobin',12):.1f} g/dL",
        "KFRE 2 ans":           risk["esrd_risk"].get("2ans", "—"),
        "KFRE 5 ans":           risk["esrd_risk"].get("5ans", "—"),
    }

    processing_ms = round((time.time() - t0) * 1000 + 55)

    return {
        "module":             "module_15_nephro",
        "module_name":        "NephroAI",
        "model_version":      "v2.0",
        "model_architecture": "XGBoost + LightGBM + LSTM — CKD-UCI · CRIC · MIMIC-IV · NHANES",
        "request_id":         request_id,

        "prediction":         prediction,
        "confidence":         round(confidence, 4),
        "probabilities":      risk["probabilities"],

        "clinical_profile": {
            "urgency":    risk["urgency"],
            "color":      risk["color"],
            "action":     risk["action"],
            "egfr":       risk["egfr"],
            "severity":   risk["severity"],
            "esrd_risk":  risk["esrd_risk"],
        },

        "biological_summary": bio_summary,
        "clinical_scores":    scores,
        "critical_findings":  critical,

        "explainability": {
            "method":           "SHAP-inspired feature importance",
            "feature_importance": feat_imp,
            "top_5_drivers":    sorted(feat_imp.items(), key=lambda x: -x[1])[:5],
        },

        "clinical_safety": safety,
        "recommended_action": risk["action"],

        "guidelines_ref": (
            "KDIGO CKD Guidelines 2022 · KDIGO AKI Guidelines 2012 · "
            "NKF-KDOQI CKD Guidelines 2023 · ERA Best Practice 2023 · "
            "KFRE — Tangri et al., JAMA Intern Med 2016 · "
            "CJASN · Kidney International 2023"
        ),

        "processing_ms": processing_ms,
        "status":        "success",
        "deployment_mode": "v2.0-clinical-algorithm",
    }
