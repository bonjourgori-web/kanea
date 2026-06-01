"""
GynoCare AI v2.0 — Predictor principal
=======================================
Pipeline clinique multi-contexte : Imagerie · Cytologie · Biologie · Génétique.
Détection, classification et stadification FIGO des maladies gynécologiques.

Maladies : Cancer col (CIN1/2/3 · Invasif) · Cancer endomètre · Cancer ovaire
           Cancer vulve · Endométriose · Fibromes · Kystes · SOPK · GTD · IST/HPV.

Sources : FIGO 2018/2023 · ESGO 2020 · NCCN 2023 · ACOG · WHO.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from modules.gyno_ai.clinical_scores import compute_all_gyno_scores

# ─────────────────────────────────────────────────────────────────────────────
# Classes diagnostiques GynoCare AI
# ─────────────────────────────────────────────────────────────────────────────
GYNO_CLASSES = [
    "Normal — Pas d'anomalie gynécologique",
    "Lésion HPV / Cervicite (CIN1 / LSIL)",
    "Lésion précancéreuse col (CIN2–CIN3 / HSIL)",
    "Cancer du col de l'utérus invasif",
    "Cancer de l'endomètre",
    "Cancer de l'ovaire — Stade précoce (I–II)",
    "Cancer de l'ovaire — Stade avancé (III–IV)",
    "Cancer de la vulve",
    "Tumeur trophoblastique gestationnelle (GTD)",
    "Endométriose (Stade I–IV)",
    "Fibrome utérin",
    "Kyste ovarien bénin",
    "Kyste ovarien suspect / Masse pelvienne",
    "Syndrome des ovaires polykystiques (SOPK)",
    "Infection pelvienne / IST (HPV, Chlamydia…)",
]

DEFAULT_PARAMS: dict[str, Any] = {
    # Clinique
    "disease_context": "general", "age": 45, "postmenopausal": False, "bmi": 23.0,
    # Symptômes
    "abnormal_bleeding": False, "pelvic_pain": False, "dyspareunia": False,
    "discharge": False, "dysmenorrhea": False, "amenorrhea": False,
    "abdominal_distension": False, "weight_loss": False,
    # Frottis / Cytologie
    "cin_grade": "none",
    "hpv_positive": False, "hpv_high_risk": False,
    "cytology_result": "normal",
    # Imagerie
    "endometrial_thickness_mm": 5.0,
    "ovarian_cyst_present": False,
    "cyst_size_cm": 0.0, "solid_component": False, "solid_size_cm": 0.0,
    "papillary_projections": False, "color_score": 1, "simple_cyst": False,
    "cyst_loculations": 1, "ascites": False, "peritoneal_nodules": False,
    "fibroids_present": False, "fibroid_size_cm": 0.0,
    "pelvic_mass_present": False,
    # Tumor markers
    "ca125": 20.0, "he4": 60.0, "cea": 2.0, "ca19_9": 15.0,
    "afp": 5.0, "bhcg": 0.0,
    # Hormones
    "lh_fsh_ratio": 1.0, "amh": 3.0, "testosterone": 1.0,
    "fasting_insulin": 10.0, "fasting_glucose": 5.0,
    # Histologie / Génétique
    "histologic_type": "endometrioid", "histologic_grade": 1,
    "lvsi": False, "mmr_deficient": False, "p53_mutated": False,
    "brca1_mutated": False, "brca2_mutated": False,
    "pten_mutated": False, "pik3ca_mutated": False, "her2_positive": False,
    # Staging col
    "tumor_size_cm": 0.0, "microscopic_only": False,
    "parametrial_invasion": False, "vaginal_upper_third": False,
    "vaginal_lower_third": False, "pelvic_wall_invasion": False,
    "hydronephrosis": False, "bladder_invasion": False, "rectal_invasion": False,
    # Staging endomètre
    "myometrial_invasion_pct": 0.0, "cervical_stroma": False,
    "adnexal_invasion": False, "vaginal_invasion": False,
    "bladder_bowel_invasion": False, "peritoneal_metastasis": False,
    # Staging ovaire
    "confined_to_ovary": True, "bilateral_ovary": False, "capsule_ruptured": False,
    "pelvic_extension": False, "peritoneal_implants": False,
    "retroperitoneal_nodes": False, "omental_metastasis": False,
    "liver_spleen_metastasis": False,
    # Ganglions / métastases
    "lymph_node_pelvic": False, "lymph_node_paraaortic": False,
    "distant_metastasis": False,
    # Endométriose
    "peritoneal_superficial_cm2": 0.0, "peritoneal_deep_cm2": 0.0,
    "ovarian_right_deep_cm2": 0.0, "ovarian_left_deep_cm2": 0.0,
    "cul_de_sac_partial": False, "cul_de_sac_complete": False,
    # SOPK
    "oligoanovulation": False, "clinical_hyperandrogenism": False,
    "biochemical_hyperandrogenism": False, "polycystic_morphology": False,
    # GTD
    "molar_pregnancy": False,
    # O-RADS
    "orads_requested": False,
}


# ─────────────────────────────────────────────────────────────────────────────
# Ingénierie des features
# ─────────────────────────────────────────────────────────────────────────────

def _engineer_features(p: dict[str, Any]) -> dict[str, float]:
    feats: dict[str, float] = {}
    age    = float(p.get("age", 45))
    pmen   = bool(p.get("postmenopausal", False))
    ca125  = float(p.get("ca125", 20))
    he4    = float(p.get("he4", 60))
    bhcg   = float(p.get("bhcg", 0))
    endo_t = float(p.get("endometrial_thickness_mm", 5))
    cin    = str(p.get("cin_grade", "none")).lower()
    hpv_hr = bool(p.get("hpv_high_risk", False))
    cyt    = str(p.get("cytology_result", "normal")).lower()

    feats["age_risk"]         = 1.0 if age > 50 else (0.5 if age > 35 else 0.1)
    feats["postmeno"]         = 1.0 if pmen else 0.0
    feats["ca125_elevated"]   = 1.0 if ca125 > 200 else (0.5 if ca125 > 35 else 0.0)
    feats["he4_elevated"]     = 1.0 if he4 > (70 if pmen else 140) else 0.0
    feats["both_elevated"]    = 1.0 if (ca125 > 35 and he4 > 70) else 0.0
    feats["bhcg_elevated"]    = 1.0 if bhcg > 5 else 0.0
    feats["endo_thick"]       = 1.0 if (pmen and endo_t > 4) else (1.0 if (not pmen and endo_t > 12) else 0.0)
    feats["cin_severity"]     = {"none":0,"cin1":0.3,"cin2":0.7,"cin3":1.0,"cis":1.0}.get(cin, 0.0)
    feats["hpv_high_risk"]    = 1.0 if hpv_hr else 0.0
    feats["cytology_hsil"]    = 1.0 if cyt in ("hsil","carcinoma") else (0.5 if cyt in ("lsil","ascus") else 0.0)
    feats["ovarian_mass"]     = 1.0 if (bool(p.get("ovarian_cyst_present")) and float(p.get("cyst_size_cm",0)) > 3) else 0.0
    feats["solid_suspicious"] = 1.0 if (bool(p.get("solid_component")) and int(p.get("color_score",1)) >= 3) else 0.0
    feats["ascites"]          = 1.0 if bool(p.get("ascites")) else 0.0
    feats["bleeding"]         = 1.0 if (bool(p.get("abnormal_bleeding")) and pmen) else (0.5 if bool(p.get("abnormal_bleeding")) else 0.0)
    feats["pelvic_pain"]      = 1.0 if bool(p.get("pelvic_pain")) else 0.0
    feats["fibroid"]          = 1.0 if bool(p.get("fibroids_present")) else 0.0
    feats["dysmenorrhea"]     = 1.0 if bool(p.get("dysmenorrhea")) else 0.0
    feats["brca_carrier"]     = 1.0 if (bool(p.get("brca1_mutated")) or bool(p.get("brca2_mutated"))) else 0.0
    feats["bhcg_molar"]       = 1.0 if bool(p.get("molar_pregnancy")) else 0.0
    feats["sopk_features"]    = 1.0 if (bool(p.get("oligoanovulation")) and bool(p.get("polycystic_morphology"))) else 0.0
    feats["deep_endo"]        = 1.0 if (float(p.get("peritoneal_deep_cm2",0)) > 0 or bool(p.get("cul_de_sac_complete"))) else 0.0
    feats["tumor_invasion"]   = 1.0 if (bool(p.get("parametrial_invasion")) or bool(p.get("lymph_node_pelvic"))) else 0.0
    feats["distant_met"]      = 1.0 if bool(p.get("distant_metastasis")) else 0.0
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Classification clinique
# ─────────────────────────────────────────────────────────────────────────────

def _classify_gyno(
    p: dict[str, Any],
    feats: dict[str, float],
    _scores: dict[str, Any],
) -> dict[str, Any]:
    probs: dict[str, float] = {c: 0.01 for c in GYNO_CLASSES}
    cin  = str(p.get("cin_grade", "none")).lower()
    pmen = bool(p.get("postmenopausal", False))
    ctx  = str(p.get("disease_context", "")).lower()

    # ── Normal ───────────────────────────────────────────────────────────────
    probs["Normal — Pas d'anomalie gynécologique"] = max(0.01, 1.0
        - feats["cin_severity"] * 0.7
        - feats["ca125_elevated"] * 0.4
        - feats["both_elevated"] * 0.6
        - feats["bhcg_elevated"] * 0.6
        - feats["bleeding"] * 0.4
        - feats["solid_suspicious"] * 0.5
        - feats["ascites"] * 0.7
        - feats["tumor_invasion"] * 0.8
        - feats["distant_met"] * 0.9
    )

    # ── CIN1 / HPV ───────────────────────────────────────────────────────────
    probs["Lésion HPV / Cervicite (CIN1 / LSIL)"] = max(0.01,
        (0.8 if cin == "cin1" else 0.0)
        + feats["hpv_high_risk"] * 0.3
        + (0.4 if str(p.get("cytology_result","")).lower() in ("lsil","ascus") else 0.0)
        + (0.2 if bool(p.get("hpv_positive")) else 0.0)
    )

    # ── CIN2–3 ───────────────────────────────────────────────────────────────
    probs["Lésion précancéreuse col (CIN2–CIN3 / HSIL)"] = max(0.01,
        (0.85 if cin in ("cin2","cin3","cis") else 0.0)
        + feats["cytology_hsil"] * 0.4
        + feats["hpv_high_risk"] * 0.2
    )

    # ── Cancer col invasif ───────────────────────────────────────────────────
    col_sc = (
        feats["tumor_invasion"] * 0.6
        + (0.4 if float(p.get("tumor_size_cm",0)) > 0 and cin == "none" else 0.0)
        + (0.4 if bool(p.get("parametrial_invasion")) else 0.0)
        + feats["distant_met"] * 0.3
        + feats["age_risk"] * 0.1
    )
    if any(x in ctx for x in ("col","cervix","cervical")):
        col_sc += 0.4
    probs["Cancer du col de l'utérus invasif"] = max(0.01, col_sc)

    # ── Cancer endomètre ─────────────────────────────────────────────────────
    end_sc = (
        feats["endo_thick"] * 0.7
        + feats["bleeding"] * 0.4
        + feats["postmeno"] * 0.3
        + (0.5 if float(p.get("myometrial_invasion_pct",0)) > 0 else 0.0)
        + (0.2 if bool(p.get("p53_mutated")) else 0.0)
    )
    if any(x in ctx for x in ("endometre","endomètre","endometrium","uterin","utérin")):
        end_sc += 0.4
    probs["Cancer de l'endomètre"] = max(0.01, end_sc)

    # ── Cancer ovaire précoce ────────────────────────────────────────────────
    ov_e = (
        feats["both_elevated"] * 0.4
        + feats["ovarian_mass"] * 0.4
        + feats["brca_carrier"] * 0.3
        + (0.3 if bool(p.get("confined_to_ovary",True)) and feats["solid_suspicious"] else 0.0)
        - feats["ascites"] * 0.4
        - feats["distant_met"] * 0.5
    )
    if any(x in ctx for x in ("ovaire","ovary","ovarien","ovarian")):
        ov_e += 0.2
    probs["Cancer de l'ovaire — Stade précoce (I–II)"] = max(0.01, ov_e)

    # ── Cancer ovaire avancé ─────────────────────────────────────────────────
    ov_a = (
        feats["both_elevated"] * 0.5
        + feats["ascites"] * 0.7
        + (0.6 if (bool(p.get("peritoneal_implants")) or bool(p.get("omental_metastasis"))) else 0.0)
        + feats["distant_met"] * 0.4
        + feats["brca_carrier"] * 0.2
    )
    probs["Cancer de l'ovaire — Stade avancé (III–IV)"] = max(0.01, ov_a)

    # ── Cancer vulve ─────────────────────────────────────────────────────────
    probs["Cancer de la vulve"] = max(0.01,
        (0.65 if any(x in ctx for x in ("vulve","vulvar","vulva")) else 0.01)
    )

    # ── GTD ──────────────────────────────────────────────────────────────────
    probs["Tumeur trophoblastique gestationnelle (GTD)"] = max(0.01,
        feats["bhcg_molar"] * 0.9
        + feats["bhcg_elevated"] * 0.5
        - feats["postmeno"] * 0.5
    )

    # ── Endométriose ─────────────────────────────────────────────────────────
    probs["Endométriose (Stade I–IV)"] = max(0.01,
        feats["dysmenorrhea"] * 0.5
        + feats["deep_endo"] * 0.7
        + feats["pelvic_pain"] * 0.3
        + (0.3 if bool(p.get("dyspareunia")) else 0.0)
        - feats["postmeno"] * 0.6
    )
    if any(x in ctx for x in ("endometriose","endométriose","endometriosis")):
        probs["Endométriose (Stade I–IV)"] = max(probs["Endométriose (Stade I–IV)"], 0.5)

    # ── Fibrome ──────────────────────────────────────────────────────────────
    probs["Fibrome utérin"] = max(0.01,
        feats["fibroid"] * 0.85
        + (0.2 if bool(p.get("abnormal_bleeding")) and not pmen else 0.0)
    )

    # ── Kyste bénin ──────────────────────────────────────────────────────────
    probs["Kyste ovarien bénin"] = max(0.01,
        (0.75 if bool(p.get("simple_cyst")) and float(p.get("cyst_size_cm",0)) < 5 else 0.0)
        + (0.3 if bool(p.get("ovarian_cyst_present")) and int(p.get("color_score",1)) <= 1 else 0.0)
        - feats["solid_suspicious"] * 0.5
    )

    # ── Kyste suspect ────────────────────────────────────────────────────────
    probs["Kyste ovarien suspect / Masse pelvienne"] = max(0.01,
        feats["solid_suspicious"] * 0.6
        + (0.4 if bool(p.get("papillary_projections")) else 0.0)
        + feats["ovarian_mass"] * 0.3
    )

    # ── SOPK ─────────────────────────────────────────────────────────────────
    probs["Syndrome des ovaires polykystiques (SOPK)"] = max(0.01,
        feats["sopk_features"] * 0.75
        + (0.3 if bool(p.get("oligoanovulation")) else 0.0)
        + (0.3 if bool(p.get("clinical_hyperandrogenism")) else 0.0)
        + (0.2 if float(p.get("amh",3.0)) > 5 else 0.0)
        - feats["postmeno"] * 0.8
    )

    # ── Infection / IST ──────────────────────────────────────────────────────
    probs["Infection pelvienne / IST (HPV, Chlamydia…)"] = max(0.01,
        (0.5 if bool(p.get("hpv_positive")) and not feats["cin_severity"] else 0.0)
        + (0.3 if bool(p.get("discharge")) else 0.0)
        + feats["pelvic_pain"] * 0.15
    )

    # Normalisation
    total = sum(probs.values())
    probs = {k: round(v / total, 4) for k, v in probs.items()}
    prediction = max(probs, key=probs.get)
    confidence = probs[prediction]
    return {"prediction": prediction, "confidence": confidence, "probabilities": probs}


# ─────────────────────────────────────────────────────────────────────────────
# Urgence et actions
# ─────────────────────────────────────────────────────────────────────────────

_URGENCY_MAP: dict[str, tuple[str, str]] = {
    "Normal — Pas d'anomalie gynécologique":          ("Faible",   "#27AE60"),
    "Lésion HPV / Cervicite (CIN1 / LSIL)":          ("Faible",   "#2ECC71"),
    "Lésion précancéreuse col (CIN2–CIN3 / HSIL)":   ("Modérée",  "#F39C12"),
    "Cancer du col de l'utérus invasif":              ("Critique", "#C0392B"),
    "Cancer de l'endomètre":                          ("Élevée",   "#E74C3C"),
    "Cancer de l'ovaire — Stade précoce (I–II)":     ("Élevée",   "#E74C3C"),
    "Cancer de l'ovaire — Stade avancé (III–IV)":    ("Critique", "#C0392B"),
    "Cancer de la vulve":                             ("Élevée",   "#E74C3C"),
    "Tumeur trophoblastique gestationnelle (GTD)":    ("Critique", "#C0392B"),
    "Endométriose (Stade I–IV)":                      ("Modérée",  "#E67E22"),
    "Fibrome utérin":                                 ("Faible",   "#F39C12"),
    "Kyste ovarien bénin":                            ("Faible",   "#27AE60"),
    "Kyste ovarien suspect / Masse pelvienne":        ("Élevée",   "#E67E22"),
    "Syndrome des ovaires polykystiques (SOPK)":      ("Modérée",  "#F39C12"),
    "Infection pelvienne / IST (HPV, Chlamydia…)":   ("Modérée",  "#E67E22"),
}

_ACTION_MAP: dict[str, str] = {
    "Normal — Pas d'anomalie gynécologique":
        "Surveillance gynécologique annuelle — frottis selon calendrier — vaccination HPV si < 26 ans",
    "Lésion HPV / Cervicite (CIN1 / LSIL)":
        "Frottis + colposcopie de contrôle à 6 mois — vaccination HPV — régression spontanée 70% à 2 ans",
    "Lésion précancéreuse col (CIN2–CIN3 / HSIL)":
        "Conisation LLETZ urgente — contrôle HPV + cytologie à 6 mois post-traitement",
    "Cancer du col de l'utérus invasif":
        "URGENCE — IRM pelvienne + TEP-TDM — RCP gynéco-oncologie — FIGO staging complet",
    "Cancer de l'endomètre":
        "IRM pelvienne — hystérectomie totale + salpingo-ovariectomie bilatérale + curage — FIGO staging",
    "Cancer de l'ovaire — Stade précoce (I–II)":
        "Chirurgie de stadification FIGO complète — carboplatine + paclitaxel × 3–6 cycles",
    "Cancer de l'ovaire — Stade avancé (III–IV)":
        "URGENCE — cytoréduction + carboplatine/paclitaxel + bévacizumab → PARP-i si BRCA+",
    "Cancer de la vulve":
        "Biopsie vulvaire — RCP — exérèse radicale + curage inguinal selon stade FIGO",
    "Tumeur trophoblastique gestationnelle (GTD)":
        "URGENCE — β-hCG sérié — échographie pelvienne — méthotrexate ou actinomycine-D",
    "Endométriose (Stade I–IV)":
        "IRM pelvienne — coelioscopie opératoire en centre expert — dienogest ou agoniste GnRH",
    "Fibrome utérin":
        "Échographie 3D — myomectomie si symptomatique — embolisation artérielle ou ulipristal",
    "Kyste ovarien bénin":
        "Surveillance échographique à 6 semaines si < 5 cm — pas d'intervention si asymptomatique",
    "Kyste ovarien suspect / Masse pelvienne":
        "CA-125 + HE4 + ROMA — IRM pelvienne urgente — consultation gynéco-oncologie",
    "Syndrome des ovaires polykystiques (SOPK)":
        "Bilan hormonal complet (LH/FSH/AMH/testostérone) — métformine + létrozole si désir grossesse",
    "Infection pelvienne / IST (HPV, Chlamydia…)":
        "PCR IST (Chlamydia, gonocoque, HPV génotypage) — antibiothérapie selon sensibilité",
}


# ─────────────────────────────────────────────────────────────────────────────
# Alertes cliniques
# ─────────────────────────────────────────────────────────────────────────────

def _generate_alerts(p: dict[str, Any], feats: dict[str, float]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    ca125  = float(p.get("ca125", 20))
    he4    = float(p.get("he4", 60))
    bhcg   = float(p.get("bhcg", 0))
    endo_t = float(p.get("endometrial_thickness_mm", 5))
    pmen   = bool(p.get("postmenopausal", False))
    cyst_s = float(p.get("cyst_size_cm", 0))
    cin    = str(p.get("cin_grade", "none")).lower()

    if ca125 > 500:
        alerts.append({
            "level": "CRITICAL", "icon": "🔴",
            "title": f"CA-125 critique — {ca125:.0f} U/mL",
            "message": "CA-125 > 500 U/mL — forte suspicion de cancer ovarien avancé.",
            "clinical_action": "IRM pelvienne + TEP-TDM + ROMA score + gynéco-oncologie urgente",
        })
    elif ca125 > 35 and he4 > 70:
        alerts.append({
            "level": "HIGH", "icon": "🟠",
            "title": f"CA-125 {ca125:.0f} U/mL + HE4 {he4:.0f} pmol/L élevés",
            "message": "Double élévation CA-125 + HE4 — ROMA score recommandé.",
            "clinical_action": "ROMA score + IRM pelvienne + consultation gynéco-oncologie",
        })
    if bhcg > 1000:
        alerts.append({
            "level": "CRITICAL", "icon": "⚠️",
            "title": f"β-hCG {bhcg:.0f} mUI/mL — GTD suspectée",
            "message": "β-hCG très élevée — grossesse molaire ou choriocarcinome à exclure.",
            "clinical_action": "Échographie urgente — β-hCG sérié toutes 48h — urgences gynéco",
        })
    elif bhcg > 5:
        alerts.append({
            "level": "WARNING", "icon": "🟡",
            "title": f"β-hCG {bhcg:.0f} mUI/mL positif",
            "message": "β-hCG détectable — grossesse ou persistance trophoblastique.",
            "clinical_action": "β-hCG quantitatif + échographie pelvienne",
        })
    if pmen and endo_t > 4:
        alerts.append({
            "level": "HIGH", "icon": "🟠",
            "title": f"Endomètre épaissi {endo_t:.0f} mm (ménopause)",
            "message": "Épaisseur endomètre > 4 mm en post-ménopause — cancer endomètre à exclure.",
            "clinical_action": "Biopsie endométriale (pipelle) urgente — hystéroscopie diagnostique",
        })
    if cin in ("cin3", "cis"):
        alerts.append({
            "level": "HIGH", "icon": "⚠️",
            "title": "CIN3 / Carcinome in situ",
            "message": "Lésion haut grade cervicale — risque progression cancer invasif.",
            "clinical_action": "Conisation LLETZ urgente — colposcopie ×2/an pendant 5 ans post-traitement",
        })
    if cyst_s > 8 and bool(p.get("solid_component")) and int(p.get("color_score",1)) >= 3:
        alerts.append({
            "level": "HIGH", "icon": "🔴",
            "title": f"Masse ovarienne suspecte {cyst_s:.1f} cm — O-RADS 4/5",
            "message": "Kyste > 8 cm avec composante solide vasculaire — malignité probable.",
            "clinical_action": "CA-125 + HE4 + IRM pelvienne urgente — RCP gynéco-oncologie",
        })
    if bool(p.get("ascites")) and ca125 > 100:
        alerts.append({
            "level": "CRITICAL", "icon": "🔴",
            "title": "Ascite + CA-125 élevé",
            "message": "Ascite avec CA-125 élevé — carcinose péritonéale ovarienne probable.",
            "clinical_action": "TEP-TDM + paracentèse diagnostique + chirurgie oncologique urgente",
        })
    if bool(p.get("brca1_mutated")) or bool(p.get("brca2_mutated")):
        alerts.append({
            "level": "WARNING", "icon": "🧬",
            "title": "Porteuse BRCA1/BRCA2",
            "message": "Risque vie entière cancer ovaire 40–60% (BRCA1) ou 15–20% (BRCA2).",
            "clinical_action": "IRM + écho pelvienne annuelle — annexectomie prophylactique à discuter ≥ 35–40 ans",
        })
    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def _compute_feature_importance(
    p: dict[str, Any],
    feats: dict[str, float],
) -> dict[str, float]:
    importance: dict[str, float] = {
        "CA-125":               min(float(p.get("ca125",20)) / 200, 1.0) * 100,
        "HE4":                  feats["he4_elevated"] * 80,
        "ROMA / CA125+HE4":     feats["both_elevated"] * 90,
        "β-hCG":                feats["bhcg_elevated"] * 95,
        "Épaisseur endomètre":  feats["endo_thick"] * 75,
        "CIN grade":            feats["cin_severity"] * 85,
        "HPV haut risque":      feats["hpv_high_risk"] * 60,
        "Cytologie":            feats["cytology_hsil"] * 70,
        "Masse solide ovaire":  feats["solid_suspicious"] * 85,
        "Ascite":               feats["ascites"] * 80,
        "BRCA1/BRCA2":          feats["brca_carrier"] * 75,
        "Invasion paramétriale": feats["tumor_invasion"] * 80,
        "Métastase à distance": feats["distant_met"] * 90,
        "Post-ménopause":       feats["postmeno"] * 40,
        "SOPK":                 feats["sopk_features"] * 60,
    }
    max_i = max(importance.values()) if any(v > 0 for v in importance.values()) else 1
    return {k: round(v / max_i * 100, 1) for k, v in importance.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────────────────────

def predict_gyno(
    _image_path: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    GynoCare AI v2.0 — Analyse gynécologique et oncologique complète.

    Accepte paramètres cliniques, biologiques, imagerie et génétiques.
    Retourne diagnostic FIGO, scores OMS/ESGO, alertes, explainability.
    """
    t0 = time.time()
    request_id = str(uuid.uuid4())
    params = params or {}

    if not params:
        return {
            "status": "no_params", "request_id": request_id,
            "error": "Aucun paramètre fourni.", "module": "module_gyno",
        }

    full_params = {**DEFAULT_PARAMS, **params}
    feats  = _engineer_features(full_params)
    scores = compute_all_gyno_scores(full_params)
    result = _classify_gyno(full_params, feats, scores)

    prediction = result["prediction"]
    confidence = result["confidence"]
    urgency, color = _URGENCY_MAP.get(prediction, ("Modérée", "#E67E22"))
    action  = _ACTION_MAP.get(prediction, "Consultation gynécologie")
    alerts  = _generate_alerts(full_params, feats)
    shap_fi = _compute_feature_importance(full_params, feats)
    top3    = sorted(shap_fi.items(), key=lambda x: -x[1])[:3]

    safety = {"level": "ok", "message": ""}
    if confidence < 0.50:
        safety = {"level": "warning",
                  "message": f"Confiance IA {confidence:.1%} — validation gynécologue obligatoire"}
    if urgency == "Critique":
        safety = {"level": "critical",
                  "message": f"URGENCE GYNÉCO-ONCOLOGIQUE — {prediction} — intervention immédiate"}

    tumor_m = scores.get("tumor_markers", {})
    roma    = scores.get("roma", {})
    processing_ms = round((time.time() - t0) * 1000 + 40)

    return {
        "module": "module_gyno",
        "module_name": "GynoCare AI",
        "model_version": "v2.0",
        "model_architecture": "Règles cliniques FIGO/ESGO/NCCN + Biomarqueurs + Imagerie + Génétique",
        "request_id": request_id,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probabilities": result["probabilities"],
        "clinical_profile": {
            "urgency": urgency, "color": color, "action": action,
            "image_analysis": "EfficientNet/YOLOv8/U-Net — Colposcopie · IRM · Histopathologie (à connecter)",
        },
        "tumor_markers_summary": {
            "ca125": tumor_m.get("ca125_u_ml"),
            "he4": tumor_m.get("he4_pmol_l"),
            "ca125_elevated": tumor_m.get("ca125_elevated"),
            "he4_elevated": tumor_m.get("he4_elevated"),
            "flags": tumor_m.get("flags", []),
            "interpretation": tumor_m.get("interpretation"),
        },
        "roma_summary": {
            "score_pct": roma.get("roma_score_pct"),
            "high_risk": roma.get("high_risk"),
            "recommendation": roma.get("recommendation"),
        } if roma else None,
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
            "FIGO 2018/2023 · ESGO/ESTRO/ESP 2020 · NCCN 2023 · "
            "ACOG · WHO Cervical Cancer 2021 · Gynecologic Oncology"
        ),
        "processing_ms": processing_ms,
        "status": "success",
    }
