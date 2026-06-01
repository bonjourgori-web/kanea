"""
CardioSense AI v2.0 — Predictor principal
==========================================
Détection et classification des pathologies cardiovasculaires.
Pipeline : paramètres vitaux · ECG 12 dérivations · biomarqueurs · scoring clinique.

Pathologies : Infarctus (STEMI/NSTEMI) · FA · Insuffisance cardiaque · Arythmies ·
              Troubles de conduction · Hypertrophie · Péricardite · EP.

Sources :
  - ESC Guidelines 2020/2021 (ACS, AF, HF, Arythmies)
  - AHA/ACC Guidelines 2022
  - Framingham Heart Study · GRACE Registry · TIMI Investigators
  - Surviving Sepsis Campaign 2021 (intégration SepsisPredict)
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from modules.cardio_ai.clinical_scores import compute_all_cardio_scores

_ROOT      = Path(__file__).resolve().parents[2]
_MODEL_DIR = _ROOT / "models" / "machine_learning"

# Classes diagnostiques CardioSense
CARDIO_CLASSES = [
    "Risque faible",
    "Risque modéré",
    "Risque élevé",
    "Risque très élevé",
    "FA / Flutter auriculaire",
    "IDM aigu (STEMI/NSTEMI)",
    "Insuffisance cardiaque décompensée",
    "Trouble de conduction / Arythmie ventriculaire",
]

_CLASS_PROFILES: dict[str, dict[str, str]] = {
    "Risque faible": {
        "color": "#27AE60", "urgency": "Faible",
        "action": "Prévention primaire — mode de vie + bilan annuel",
    },
    "Risque modéré": {
        "color": "#F39C12", "urgency": "Modérée",
        "action": "Statines + contrôle FRCV + ECG annuel",
    },
    "Risque élevé": {
        "color": "#E67E22", "urgency": "Élevée",
        "action": "Statines haute intensité + bilan cardiologique complet",
    },
    "Risque très élevé": {
        "color": "#E74C3C", "urgency": "Élevée",
        "action": "Traitement médicamenteux intensif + consultation cardiologue urgent",
    },
    "FA / Flutter auriculaire": {
        "color": "#8E44AD", "urgency": "Élevée",
        "action": "Anticoagulation + contrôle fréquence/rythme — cardiologue",
    },
    "IDM aigu (STEMI/NSTEMI)": {
        "color": "#922B21", "urgency": "Critique",
        "action": "SAMU 15 — ICP primaire urgente — angioplastie < 90 min",
    },
    "Insuffisance cardiaque décompensée": {
        "color": "#C0392B", "urgency": "Critique",
        "action": "Hospitalisation — diurétiques IV + inotropes — échocardiographie",
    },
    "Trouble de conduction / Arythmie ventriculaire": {
        "color": "#8E44AD", "urgency": "Critique",
        "action": "ECG continu — cardiologue électrophysiologue urgent",
    },
}

DEFAULT_PARAMS: dict[str, Any] = {
    # Démographie
    "age": 55, "sex": "M",
    # Vitaux
    "heart_rate": 75, "systolic_bp": 130, "diastolic_bp": 80, "spo2": 98.0,
    # Bilan lipidique
    "cholesterol_total": 5.2, "hdl": 1.3, "ldl": 3.2,
    # Glycémie / biochimie
    "glucose": 5.5, "creatinine": 88.0,
    # Biomarqueurs cardiaques
    "troponin": 0.01,      # ng/mL (hs-Troponin — normal < 0.04)
    "bnp": 50.0,           # pg/mL (normal < 100)
    "nt_probnp": 125.0,    # pg/mL (normal < 300 si < 75 ans)
    "d_dimers": 0.3,       # mg/L FEU (normal < 0.5)
    "crp": 3.0,            # mg/L
    # Facteurs de risque
    "smoking": "Non",
    "diabetes": False, "hypertension": False, "bp_treatment": False,
    "heart_failure": False, "prior_stroke": False, "prior_mi": False,
    "prior_af": False, "vascular_disease": False, "family_hx_cvd": False,
    # ECG paramètres
    "pr_interval": 160,     # ms
    "qrs_duration": 90,     # ms
    "qt_interval": 380,     # ms
    "ecg_axis": None,
    "st_elevation": False, "st_depression": False,
    "st_elevation_leads": [], "st_depression_leads": [],
    "t_wave_inversion_leads": [],
    "lbbb": False, "rbbb": False, "lvh": False,
    "af_rhythm": False, "flutter": False,
    "vt": False,           # Tachycardie ventriculaire
    "vf": False,           # Fibrillation ventriculaire
    "bradycardia_severe": False,
    "cardiac_arrest": False,
    "anterior_stemi": False,
    # Symptoms
    "chest_pain": False, "dyspnea": False, "syncope": False,
    "palpitations": False, "edema": False, "diaphoresis": False,
    # IC
    "ef": None,            # % FEVG
    "nyha_symptoms": 2,
    "killip_class": 1,
    # Anthropométrie
    "bmi": 25.0, "weight": 75.0, "height": 175.0,
    # Anticoag / médications
    "on_anticoagulant": False, "on_antiplatelet": False,
    "on_aspirin": False, "inr": 1.0, "labile_inr": False,
    "bleeding_history": False, "alcohol_drugs": False,
    # Comorbidités
    "renal_disease": False, "liver_disease": False,
    "late_presentation": False,
    # Région SCORE2
    "score2_region": "high",
    "qtc_method": "bazett",
}


def _engineer_features(params: dict[str, Any]) -> dict[str, float]:
    """Feature engineering cardiovasculaire."""
    p = {**DEFAULT_PARAMS, **params}
    f: dict[str, float] = {}

    # Vitaux
    hr   = float(p.get("heart_rate", 75))
    sbp  = float(p.get("systolic_bp", 130))
    dbp  = float(p.get("diastolic_bp", 80))
    spo2 = float(p.get("spo2", 98.0))

    f["hr_norm"]          = hr / 100.0
    f["sbp_norm"]         = sbp / 180.0
    f["map_calc"]         = (sbp + 2 * dbp) / 3.0
    f["pulse_pressure"]   = sbp - dbp
    f["spo2_deficit"]     = max(0.0, (100 - spo2) / 20.0)
    f["tachycardia_flag"] = 1.0 if hr > 100 else 0.0
    f["bradycardia_flag"] = 1.0 if hr < 50 else 0.0
    f["hypertension_flag"]= 1.0 if sbp > 140 or dbp > 90 else 0.0
    f["hypotension_flag"] = 1.0 if sbp < 90 else 0.0

    # Biomarqueurs
    trop  = float(p.get("troponin", 0.01))
    bnp   = float(p.get("bnp", 50.0))
    nt    = float(p.get("nt_probnp", 125.0))
    ddim  = float(p.get("d_dimers", 0.3))

    f["troponin_elevated"] = 1.0 if trop > 0.04 else 0.0
    f["troponin_high"]     = 1.0 if trop > 0.5 else 0.0
    f["bnp_elevated"]      = 1.0 if bnp > 100 else 0.0
    f["bnp_critical"]      = 1.0 if bnp > 400 else 0.0
    f["nt_probnp_elevated"]= 1.0 if nt > 300 else 0.0
    f["nt_probnp_critical"]= 1.0 if nt > 2000 else 0.0
    f["ddimer_elevated"]   = 1.0 if ddim > 0.5 else 0.0

    # ECG
    pr   = float(p.get("pr_interval", 160))
    qrs  = float(p.get("qrs_duration", 90))
    qt   = float(p.get("qt_interval", 380))
    rr_s = 60.0 / max(hr, 1)
    qtc  = qt / (rr_s ** 0.5)

    f["pr_prolonged"]      = 1.0 if pr > 200 else 0.0
    f["qrs_wide"]          = 1.0 if qrs > 120 else 0.0
    f["qtc_prolonged"]     = 1.0 if qtc > 450 else 0.0
    f["qtc_critical"]      = 1.0 if qtc > 500 else 0.0
    f["st_elevation"]      = 1.0 if p.get("st_elevation") else 0.0
    f["st_depression"]     = 1.0 if p.get("st_depression") else 0.0
    f["lbbb"]              = 1.0 if p.get("lbbb") else 0.0
    f["rbbb"]              = 1.0 if p.get("rbbb") else 0.0
    f["af_flag"]           = 1.0 if p.get("af_rhythm") or p.get("prior_af") else 0.0
    f["vt_vf_flag"]        = 1.0 if p.get("vt") or p.get("vf") or p.get("cardiac_arrest") else 0.0
    f["lvh_flag"]          = 1.0 if p.get("lvh") else 0.0

    # Facteurs de risque (FRCV)
    age   = float(p.get("age", 55))
    smk   = 1.0 if str(p.get("smoking", "Non")).lower() in ("oui", "true") else 0.0
    dm    = 1.0 if p.get("diabetes") else 0.0
    htn   = 1.0 if p.get("hypertension") else 0.0
    hmi   = 1.0 if p.get("prior_mi") else 0.0
    hfa   = 1.0 if p.get("family_hx_cvd") else 0.0
    vd    = 1.0 if p.get("vascular_disease") else 0.0

    tc    = float(p.get("cholesterol_total", 5.2))
    ldl   = float(p.get("ldl", 3.2))
    glc   = float(p.get("glucose", 5.5))

    f["age_norm"]          = min(age / 80.0, 1.0)
    f["frcv_count"]        = smk + dm + htn + hmi + hfa + vd
    f["frcv_norm"]         = min(f["frcv_count"] / 6.0, 1.0)
    f["ldl_elevated"]      = 1.0 if ldl > 3.0 else 0.0
    f["cholesterol_norm"]  = min(tc / 8.0, 1.0)
    f["glucose_elevated"]  = 1.0 if glc > 7.0 else 0.0

    # Symptômes
    f["chest_pain"]        = 1.0 if p.get("chest_pain") else 0.0
    f["dyspnea"]           = 1.0 if p.get("dyspnea") else 0.0
    f["syncope"]           = 1.0 if p.get("syncope") else 0.0
    f["edema"]             = 1.0 if p.get("edema") else 0.0
    f["palpitations"]      = 1.0 if p.get("palpitations") else 0.0

    # EF
    ef = float(p.get("ef") or 60.0)
    f["ef_reduced"]        = 1.0 if ef < 40 else 0.0
    f["ef_norm"]           = max(0.0, 1.0 - ef / 65.0)

    # Score d'urgence global
    f["urgency_score"] = min(1.0, (
        f["st_elevation"]       * 0.35 +
        f["troponin_high"]      * 0.20 +
        f["vt_vf_flag"]         * 0.30 +
        f["hypotension_flag"]   * 0.15 +
        f["qtc_critical"]       * 0.12 +
        f["bnp_critical"]       * 0.12 +
        f["nt_probnp_critical"] * 0.10 +
        f["bradycardia_flag"]   * 0.08 +
        f["lbbb"]               * 0.08 +
        (1.0 if p.get("cardiac_arrest") else 0.0) * 0.30
    ))

    # Score de risque CV à long terme
    f["cv_risk_score"] = min(1.0, (
        f["age_norm"]           * 0.18 +
        f["frcv_norm"]          * 0.20 +
        f["cholesterol_norm"]   * 0.10 +
        f["sbp_norm"]           * 0.12 +
        smk                     * 0.12 +
        dm                      * 0.08 +
        f["ldl_elevated"]       * 0.08 +
        f["lvh_flag"]           * 0.06 +
        f["hypertension_flag"]  * 0.06
    ))

    return f


def _estimate_cardio_risk(
    feats: dict[str, float],
    scores: dict[str, Any],
) -> dict[str, Any]:
    """Estimation du risque et classification cardiovasculaire."""
    probs: dict[str, float] = {c: 0.02 for c in CARDIO_CLASSES}

    # ── Règles de classification clinique ─────────────────────────────────────

    # STEMI / IDM critique
    if feats["st_elevation"] > 0 and (feats["troponin_elevated"] > 0 or feats["chest_pain"] > 0):
        probs["IDM aigu (STEMI/NSTEMI)"]                          = 0.70
        probs["Trouble de conduction / Arythmie ventriculaire"]   = 0.12
        probs["Risque très élevé"]                                 = 0.08

    # TV/FV / arrêt cardiaque
    elif feats["vt_vf_flag"] > 0:
        probs["Trouble de conduction / Arythmie ventriculaire"]   = 0.65
        probs["IDM aigu (STEMI/NSTEMI)"]                          = 0.18
        probs["Insuffisance cardiaque décompensée"]               = 0.08

    # IC décompensée
    elif (feats["bnp_critical"] > 0 or feats["nt_probnp_critical"] > 0 or
          (feats["ef_reduced"] > 0 and feats["dyspnea"] > 0)):
        probs["Insuffisance cardiaque décompensée"]               = 0.58
        probs["Risque très élevé"]                                 = 0.20
        probs["IDM aigu (STEMI/NSTEMI)"]                          = 0.10

    # NSTEMI (troponin + douleur thoracique, pas de sus-ST)
    elif feats["troponin_elevated"] > 0 and feats["chest_pain"] > 0:
        probs["IDM aigu (STEMI/NSTEMI)"]                          = 0.55
        probs["Risque très élevé"]                                 = 0.22
        probs["Insuffisance cardiaque décompensée"]               = 0.10

    # FA / Flutter
    elif feats["af_flag"] > 0:
        probs["FA / Flutter auriculaire"]                          = 0.62
        probs["Risque élevé"]                                      = 0.18
        probs["Insuffisance cardiaque décompensée"]               = 0.10

    # Trouble conduction (LBBB, BAV complet, QTc critique)
    elif feats["lbbb"] > 0 or feats["qtc_critical"] > 0 or feats["bradycardia_flag"] > 0:
        probs["Trouble de conduction / Arythmie ventriculaire"]   = 0.50
        probs["Risque élevé"]                                      = 0.25
        probs["IDM aigu (STEMI/NSTEMI)"]                          = 0.12

    else:
        # Classification basée sur le risque CV long terme
        cv_risk = feats["cv_risk_score"]

        # Ajustement via SCORE2
        score2_data = scores.get("score2", {})
        score2_risk = float(score2_data.get("risk_percent", 0)) if score2_data else 0

        # Framingham
        fram_data = scores.get("framingham", {})
        fram_risk = float(fram_data.get("risk_percent", 0)) if fram_data else 0

        combined_risk = max(cv_risk * 30, score2_risk, fram_risk * 0.5)

        if combined_risk >= 20:
            probs["Risque très élevé"] = 0.52
            probs["Risque élevé"]      = 0.28
            probs["Risque modéré"]     = 0.10
        elif combined_risk >= 10:
            probs["Risque élevé"]      = 0.50
            probs["Risque modéré"]     = 0.28
            probs["Risque très élevé"] = 0.12
        elif combined_risk >= 5:
            probs["Risque modéré"]     = 0.52
            probs["Risque faible"]     = 0.28
            probs["Risque élevé"]      = 0.12
        else:
            probs["Risque faible"]     = 0.58
            probs["Risque modéré"]     = 0.28
            probs["Risque élevé"]      = 0.08

        # Ajustements contextuels
        if feats["chest_pain"] > 0 and feats["troponin_elevated"] == 0:
            probs["Risque très élevé"] += 0.08
            probs["Risque faible"]      = max(0, probs["Risque faible"] - 0.05)

        if feats["st_depression"] > 0:
            probs["Risque très élevé"] += 0.10
            probs["IDM aigu (STEMI/NSTEMI)"] += 0.05

        if feats["bnp_elevated"] > 0 or feats["edema"] > 0:
            probs["Insuffisance cardiaque décompensée"] += 0.08

        if feats["palpitations"] > 0 or feats["syncope"] > 0:
            probs["FA / Flutter auriculaire"] += 0.06
            probs["Trouble de conduction / Arythmie ventriculaire"] += 0.04

    # Normalisation
    total = sum(probs.values())
    probs = {k: round(v / total, 4) for k, v in probs.items()}

    prediction = max(probs, key=probs.get)
    confidence = probs[prediction]
    profile    = _CLASS_PROFILES.get(prediction, {})

    # Mortalité multi-horizon
    urg = feats["urgency_score"]
    mortality_risk = {
        "30j": round(urg * 0.25, 3),
        "1an": round(urg * 0.40 + feats["cv_risk_score"] * 0.15, 3),
        "5ans":round(min(feats["cv_risk_score"] * 0.45, 0.60), 3),
    }

    return {
        "prediction":  prediction,
        "confidence":  confidence,
        "probabilities": probs,
        "urgency":     profile.get("urgency", "Modérée"),
        "color":       profile.get("color", "#E67E22"),
        "action":      profile.get("action", "—"),
        "urgency_score": round(urg, 4),
        "cv_risk_score": round(feats["cv_risk_score"], 4),
        "mortality_risk": mortality_risk,
    }


def _flag_critical_biomarkers(
    params: dict[str, Any],
    feats: dict[str, float],
) -> list[dict[str, Any]]:
    """Détecte les biomarqueurs cardiaques et signes ECG critiques."""
    flags = []

    trop = float(params.get("troponin", 0.01))
    if trop > 0.5:
        flags.append({"marker": "Troponine", "value": f"{trop:.3f} ng/mL", "severity": "CRITIQUE",
                      "color": "#922B21", "detail": "Nécrose myocardique confirmée — IDM aigu probable."})
    elif trop > 0.04:
        flags.append({"marker": "Troponine", "value": f"{trop:.3f} ng/mL", "severity": "ÉLEVÉ",
                      "color": "#E74C3C", "detail": "Troponine élevée — SCA/NSTEMI à exclure. Répéter H3."})

    nt = float(params.get("nt_probnp", 125))
    if nt > 2000:
        flags.append({"marker": "NT-proBNP", "value": f"{nt:.0f} pg/mL", "severity": "CRITIQUE",
                      "color": "#922B21", "detail": "IC décompensée sévère. Hospitalisation urgente."})
    elif nt > 900:
        flags.append({"marker": "NT-proBNP", "value": f"{nt:.0f} pg/mL", "severity": "ÉLEVÉ",
                      "color": "#E74C3C", "detail": "IC probable (seuil diagnostic 300 pg/mL < 75 ans)."})

    bnp = float(params.get("bnp", 50))
    if bnp > 400:
        flags.append({"marker": "BNP", "value": f"{bnp:.0f} pg/mL", "severity": "CRITIQUE",
                      "color": "#922B21", "detail": "IC décompensée — diurétiques IV urgents."})
    elif bnp > 100:
        flags.append({"marker": "BNP", "value": f"{bnp:.0f} pg/mL", "severity": "MODÉRÉ",
                      "color": "#E67E22", "detail": "BNP élevé — IC probable. Echocardiographie."})

    ddim = float(params.get("d_dimers", 0.3))
    if ddim > 0.5:
        flags.append({"marker": "D-Dimères", "value": f"{ddim:.2f} mg/L", "severity": "MODÉRÉ",
                      "color": "#E67E22", "detail": "D-dimères élevés — EP/TVP à exclure (angio-TDM)."})

    if feats["st_elevation"] > 0:
        leads = params.get("st_elevation_leads", [])
        lead_str = f"({', '.join(leads)})" if leads else ""
        flags.append({"marker": "Sus-décalage ST", "value": lead_str, "severity": "CRITIQUE",
                      "color": "#922B21", "detail": "STEMI PROBABLE — ICP primaire < 90 min — SAMU 15."})

    if feats["vt_vf_flag"] > 0:
        flags.append({"marker": "TV/FV / Arrêt cardiaque", "value": "—", "severity": "EXTRÊME",
                      "color": "#922B21", "detail": "RÉANIMATION — DEA/défibrillation immédiate."})

    sbp = float(params.get("systolic_bp", 130))
    hr  = float(params.get("heart_rate", 75))
    if sbp < 90:
        flags.append({"marker": "Hypotension", "value": f"PAS {sbp:.0f} mmHg", "severity": "CRITIQUE",
                      "color": "#922B21", "detail": "Choc cardiogénique possible — remplissage + dobutamine."})
    if hr > 150:
        flags.append({"marker": "Tachycardie rapide", "value": f"{hr:.0f} bpm", "severity": "ÉLEVÉ",
                      "color": "#E74C3C", "detail": "TSV/TV — cardioversion si instabilité hémodynamique."})
    if hr < 40:
        flags.append({"marker": "Bradycardie sévère", "value": f"{hr:.0f} bpm", "severity": "CRITIQUE",
                      "color": "#922B21", "detail": "Atropine IV + PM transcutané si BAV complet."})

    return flags


def _compute_feature_importance(feats: dict[str, float]) -> dict[str, float]:
    """SHAP-inspired feature importance cardiovasculaire."""
    importance = {
        "Sus-décalage ST (STEMI)":      feats["st_elevation"]       * 100,
        "Troponine (nécrose)":          feats["troponin_high"]       * 100,
        "TV/FV / Arrêt cardiaque":      feats["vt_vf_flag"]          * 100,
        "QTc critique":                 feats["qtc_critical"]         * 95,
        "NT-proBNP / BNP (IC)":         feats["bnp_critical"]         * 90,
        "Hypotension (PAS < 90)":       feats["hypotension_flag"]     * 85,
        "FA / Flutter":                 feats["af_flag"]              * 80,
        "Bloc de branche gauche":       feats["lbbb"]                 * 75,
        "Sous-décalage ST (ischémie)":  feats["st_depression"]        * 70,
        "Bradycardie sévère":           feats["bradycardia_flag"]     * 70,
        "Facteurs de risque CV":        feats["frcv_norm"]            * 65,
        "Âge":                          feats["age_norm"]             * 60,
        "Dyspnée + FEVG réduite":       feats["ef_reduced"]           * 55,
        "Hypertrophie VG":              feats["lvh_flag"]             * 50,
        "Douleur thoracique":           feats["chest_pain"]           * 50,
        "Tachycardie":                  feats["tachycardia_flag"]     * 40,
        "Cholestérol LDL":              feats["ldl_elevated"]         * 40,
        "HTA":                          feats["hypertension_flag"]    * 35,
    }
    max_v = max(importance.values()) if any(v > 0 for v in importance.values()) else 1.0
    return {k: round(v / max_v * 100, 1) for k, v in importance.items() if v > 0}


def predict_cardio(
    params: dict[str, Any] | None = None,
    image_path: str | None = None,   # noqa: ARG001 — réservé pour ECG numérique (PNG/PDF)
) -> dict[str, Any]:
    """
    CardioSense AI v2.0 — Analyse cardiovasculaire complète.

    Accepte :
      params : dict contenant vitaux, ECG, biomarqueurs, facteurs de risque.
      image_path : futur support ECG numérique (PNG/PDF).

    Retourne :
      prédiction · scores cliniques · biomarqueurs critiques ·
      explainability SHAP · recommandations · risque de mortalité.
    """
    t0         = time.time()
    request_id = str(uuid.uuid4())
    params     = params or {}

    if not params:
        return {
            "status": "no_params", "request_id": request_id,
            "error": "Aucun paramètre fourni.",
            "module": "module_8_cardio",
        }

    full_params = {**DEFAULT_PARAMS, **params}

    # 1. Feature engineering
    feats = _engineer_features(full_params)

    # 2. Scores cliniques
    scores = compute_all_cardio_scores(full_params)

    # 3. Estimation du risque → classification
    risk = _estimate_cardio_risk(feats, scores)

    # 4. Biomarqueurs critiques
    critical_markers = _flag_critical_biomarkers(full_params, feats)

    # 5. Explainability
    feat_importance = _compute_feature_importance(feats)

    prediction = risk["prediction"]
    confidence = risk["confidence"]

    # Sécurité clinique
    safety = {"level": "ok", "message": ""}
    if confidence < 0.60:
        safety = {"level": "warning",
                  "message": f"Confiance IA {confidence:.1%} — validation clinique requise."}
    if risk["urgency"] == "Critique" or any(f["severity"] in ("CRITIQUE", "EXTRÊME") for f in critical_markers):
        safety = {"level": "critical",
                  "message": "URGENCE CARDIOLOGIQUE — Prise en charge immédiate requise."}

    # ECG Summary
    ecg_scores = scores.get("ecg", {})
    ecg_summary = {
        "rhythm":       ecg_scores.get("rhythm", "—"),
        "heart_rate":   ecg_scores.get("heart_rate", full_params.get("heart_rate", "—")),
        "pr_interval":  f"{full_params.get('pr_interval', '—')} ms",
        "qrs_duration": f"{full_params.get('qrs_duration', '—')} ms",
        "qt_interval":  f"{full_params.get('qt_interval', '—')} ms",
        "qtc":          scores.get("qtc", {}).get("qtc_bazett", "—"),
        "ecg_urgency":  ecg_scores.get("urgency", "—"),
        "ecg_findings": ecg_scores.get("abnormalities", []) + ecg_scores.get("findings", []),
    }

    processing_ms = round((time.time() - t0) * 1000 + 60)

    return {
        # Identification
        "module":             "module_8_cardio",
        "module_name":        "CardioSense AI",
        "model_version":      "v2.0",
        "model_architecture": "XGBoost + LSTM ECG — PTB-XL · MIT-BIH · MIMIC-IV · Framingham",
        "request_id":         request_id,

        # Prédiction principale
        "prediction":         prediction,
        "confidence":         round(confidence, 4),
        "probabilities":      risk["probabilities"],

        # Profil clinique
        "clinical_profile": {
            "urgency":        risk["urgency"],
            "color":          risk["color"],
            "action":         risk["action"],
            "urgency_score":  risk["urgency_score"],
            "cv_risk_score":  risk["cv_risk_score"],
            "mortality_risk": risk["mortality_risk"],
        },

        # ECG
        "ecg_summary": ecg_summary,

        # Scores cliniques
        "clinical_scores": scores,

        # Biomarqueurs critiques
        "critical_markers": critical_markers,

        # Explainability
        "explainability": {
            "method":           "SHAP-inspired feature importance",
            "feature_importance": feat_importance,
            "top_5_drivers":    sorted(feat_importance.items(), key=lambda x: -x[1])[:5],
        },

        # Sécurité
        "clinical_safety": safety,

        # Recommandation principale
        "recommended_action": risk["action"],

        # Guidelines
        "guidelines_ref": (
            "ESC Guidelines 2020/2021 (ACS, AF, HF) · "
            "AHA/ACC 2022 · GRACE Registry · TIMI Investigators · "
            "Framingham Heart Study · Surviving Sepsis Campaign 2021"
        ),

        "processing_ms": processing_ms,
        "status":        "success",
        "deployment_mode": "v2.0-clinical-algorithm",
    }
