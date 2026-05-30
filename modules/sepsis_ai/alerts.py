"""
SepsisPredict AI — Système d'alertes médicales temps réel
===========================================================
Niveaux : INFO / WARNING / HIGH / CRITICAL
Déclenchement automatique basé sur prédiction IA + scores + biomarqueurs.
Compatible monitoring ICU continu (1 min, 5 min, configurable).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# Niveaux d'alerte
# ═══════════════════════════════════════════════════════════════════════════════

ALERT_LEVELS = {
    "INFO":     {"color": "#2E86DE", "bg": "#EBF4FD", "icon": "ℹ️",  "priority": 1},
    "WARNING":  {"color": "#E67E22", "bg": "#FEF0E7", "icon": "⚠️",  "priority": 2},
    "HIGH":     {"color": "#E74C3C", "bg": "#FDECEA", "icon": "🔴",  "priority": 3},
    "CRITICAL": {"color": "#922B21", "bg": "#F5B7B1", "icon": "🚨",  "priority": 4},
}


@dataclass
class SepsisAlert:
    level: str           # INFO | WARNING | HIGH | CRITICAL
    code: str            # Code court identifiant l'alerte
    title: str           # Titre court
    message: str         # Message détaillé
    trigger_value: Any   # Valeur qui a déclenché l'alerte
    threshold: Any       # Seuil dépassé
    clinical_action: str # Action clinique recommandée
    timestamp: str       = field(default_factory=lambda: datetime.now().isoformat())
    priority: int        = 1
    bundle_item: str     = ""  # Item SSC bundle associé si applicable

    def to_dict(self) -> dict[str, Any]:
        return {
            "level":          self.level,
            "code":           self.code,
            "title":          self.title,
            "message":        self.message,
            "trigger_value":  self.trigger_value,
            "threshold":      self.threshold,
            "clinical_action":self.clinical_action,
            "timestamp":      self.timestamp,
            "priority":       self.priority,
            "color":          ALERT_LEVELS[self.level]["color"],
            "icon":           ALERT_LEVELS[self.level]["icon"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Générateur d'alertes
# ═══════════════════════════════════════════════════════════════════════════════

def generate_sepsis_alerts(
    prediction: str,
    confidence: float,
    params: dict[str, Any],
    scores: dict[str, Any],
    critical_biomarkers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Génère la liste des alertes cliniques à partir des résultats IA.
    Retourne une liste de dicts triée par priorité décroissante.
    """
    alerts: list[SepsisAlert] = []

    sofa_total   = scores.get("sofa", {}).get("total", 0)
    qsofa_score  = scores.get("qsofa", {}).get("score", 0)
    news2_total  = scores.get("news2", {}).get("total", 0)
    mews_score   = scores.get("mews", {}).get("score", 0)

    lactate      = float(params.get("lactate", 1.0))
    pct          = float(params.get("procalcitonin", 0.1))
    systolic_bp  = float(params.get("systolic_bp", 120))
    platelets    = float(params.get("platelets", 200))
    creatinine   = float(params.get("creatinine", 88))
    spo2         = float(params.get("spo2", 98))
    resp_rate    = float(params.get("resp_rate", 16))
    heart_rate   = float(params.get("heart_rate", 75))
    temperature  = float(params.get("temperature", 37.0))
    gcs          = int(params.get("gcs", 15))
    inr          = float(params.get("inr", 1.0))

    # ── ALERTES CRITIQUES ────────────────────────────────────────────────────

    if prediction == "Choc septique":
        alerts.append(SepsisAlert(
            level="CRITICAL", code="SEPTIC_SHOCK",
            title="CHOC SEPTIQUE DÉTECTÉ",
            message=f"SepsisPredict AI détecte un choc septique (confiance {confidence:.1%}). "
                    f"SOFA={sofa_total} · qSOFA={qsofa_score} · Lactate={lactate} mmol/L.",
            trigger_value=prediction, threshold="Choc septique",
            clinical_action="RÉANIMATION IMMÉDIATE — Noradrénaline IV + hémocultures + antibiotiques < 1h + remplissage 30 mL/kg",
            priority=4,
            bundle_item="Bundle SSC 1h complet",
        ))

    if prediction == "Sepsis sévère":
        alerts.append(SepsisAlert(
            level="CRITICAL", code="SEVERE_SEPSIS",
            title="SEPSIS SÉVÈRE",
            message=f"Sepsis sévère détecté (confiance {confidence:.1%}). "
                    f"SOFA={sofa_total} — {len(scores.get('sofa',{}).get('organ_failures',[]))} organe(s) défaillant(s).",
            trigger_value=prediction, threshold="Sepsis sévère",
            clinical_action="BUNDLE SSC 1h — Hémocultures + Antibiotiques IV + Lactate + Remplissage + Évaluation USI",
            priority=4,
        ))

    if lactate >= 4.0:
        alerts.append(SepsisAlert(
            level="CRITICAL", code="HYPERLACTATEMIA_CRITICAL",
            title="HYPERLACTATEMIE CRITIQUE",
            message=f"Lactate {lactate:.1f} mmol/L ≥ 4.0 — Choc cryptique ou hypoperfusion sévère.",
            trigger_value=lactate, threshold=4.0,
            clinical_action="Remplissage vasculaire 30 mL/kg + réévaluation lactate dans 2h + vasopresseurs si MAP < 65",
            priority=4,
            bundle_item="Mesure lactate SSC Bundle",
        ))

    if sofa_total >= 6:
        alerts.append(SepsisAlert(
            level="CRITICAL", code="SOFA_CRITICAL",
            title=f"SOFA CRITIQUE — {sofa_total}/24",
            message=f"Score SOFA {sofa_total}/24 — Défaillance multiviscérale sévère. "
                    f"Organes défaillants : {', '.join(scores.get('sofa',{}).get('organ_failures',[]) or ['—'])}.",
            trigger_value=sofa_total, threshold=6,
            clinical_action="Transfert USI urgent — soutien multiviscéral — consultation réanimation",
            priority=4,
        ))

    if systolic_bp < 70:
        alerts.append(SepsisAlert(
            level="CRITICAL", code="SEVERE_HYPOTENSION",
            title=f"HYPOTENSION SÉVÈRE — PAS {systolic_bp:.0f} mmHg",
            message=f"PAS {systolic_bp:.0f} mmHg < 70 mmHg — Instabilité hémodynamique critique.",
            trigger_value=systolic_bp, threshold=70,
            clinical_action="Vasopresseurs IV en urgence + remplissage + monitoring invasif",
            priority=4,
        ))

    if platelets < 20:
        alerts.append(SepsisAlert(
            level="CRITICAL", code="THROMBOCYTOPENIA_CRITICAL",
            title=f"THROMBOPÉNIE CRITIQUE — {platelets:.0f} G/L",
            message=f"Plaquettes {platelets:.0f} G/L < 20 G/L — Risque hémorragique vital.",
            trigger_value=platelets, threshold=20,
            clinical_action="Transfusion plaquettaire urgente — avis hématologie",
            priority=4,
        ))

    # ── ALERTES HAUTES ───────────────────────────────────────────────────────

    if prediction == "Sepsis" or (sofa_total >= 2 and qsofa_score >= 2):
        alerts.append(SepsisAlert(
            level="HIGH", code="SEPSIS_DETECTED",
            title="SEPSIS DÉTECTÉ",
            message=f"Critères Sepsis-3 remplis : SOFA={sofa_total} (ΔSOFA≥2) + suspicion infectieuse. "
                    f"qSOFA={qsofa_score}/3.",
            trigger_value=sofa_total, threshold=2,
            clinical_action="Hémocultures × 2 + Antibiotiques probabilistes < 1h + Bilan biologique complet + Lactate",
            priority=3,
        ))

    if lactate >= 2.0 and lactate < 4.0:
        alerts.append(SepsisAlert(
            level="HIGH", code="HYPERLACTATEMIA",
            title=f"HYPERLACTATEMIE — {lactate:.1f} mmol/L",
            message=f"Lactate {lactate:.1f} mmol/L ≥ 2.0 — Hypoperfusion tissulaire probable.",
            trigger_value=lactate, threshold=2.0,
            clinical_action="Remplissage vasculaire + réévaluation lactate dans 2h + contrôle MAP ≥ 65",
            priority=3,
            bundle_item="Mesure lactate SSC Bundle",
        ))

    if news2_total >= 7:
        alerts.append(SepsisAlert(
            level="HIGH", code="NEWS2_HIGH",
            title=f"NEWS2 ÉLEVÉ — {news2_total}/20",
            message=f"Score NEWS2 {news2_total} — Risque de détérioration clinique rapide.",
            trigger_value=news2_total, threshold=7,
            clinical_action="Réponse d'urgence — équipe de réanimation — monitoring continu",
            priority=3,
        ))

    if pct >= 2.0:
        alerts.append(SepsisAlert(
            level="HIGH", code="PCT_HIGH",
            title=f"PROCALCITONINE ÉLEVÉE — {pct:.1f} ng/mL",
            message=f"PCT {pct:.1f} ng/mL ≥ 2.0 — Forte probabilité d'infection bactérienne sévère.",
            trigger_value=pct, threshold=2.0,
            clinical_action="Hémocultures × 2 + antibiothérapie large spectre en urgence",
            priority=3,
        ))

    if spo2 < 90:
        alerts.append(SepsisAlert(
            level="HIGH", code="HYPOXEMIA_SEVERE",
            title=f"HYPOXÉMIE SÉVÈRE — SpO2 {spo2:.0f}%",
            message=f"SpO2 {spo2:.0f}% < 90% — Insuffisance respiratoire aiguë.",
            trigger_value=spo2, threshold=90,
            clinical_action="O2 haute concentration + évaluation VNI/IOT + bilan gazométrique urgent",
            priority=3,
        ))

    if creatinine > 354:
        alerts.append(SepsisAlert(
            level="HIGH", code="AKI_SEVERE",
            title=f"IRA SÉVÈRE — Créatinine {creatinine:.0f} µmol/L",
            message=f"Créatinine {creatinine:.0f} µmol/L > 354 µmol/L — Insuffisance rénale aiguë stade 3.",
            trigger_value=creatinine, threshold=354,
            clinical_action="Bilan rénal complet + avis néphro + épuration extrarénale si indiquée",
            priority=3,
        ))

    if gcs < 10:
        alerts.append(SepsisAlert(
            level="HIGH", code="LOW_GCS",
            title=f"GCS CRITIQUE — {gcs}/15",
            message=f"GCS {gcs}/15 — Altération sévère de la conscience (encéphalopathie septique probable).",
            trigger_value=gcs, threshold=10,
            clinical_action="Protection des voies aériennes — évaluation neurologique urgente — scanner cérébral",
            priority=3,
        ))

    # ── ALERTES WARNINGS ─────────────────────────────────────────────────────

    if qsofa_score >= 2:
        alerts.append(SepsisAlert(
            level="WARNING", code="QSOFA_POSITIVE",
            title=f"qSOFA POSITIF — {qsofa_score}/3",
            message=f"qSOFA {qsofa_score}/3 ≥ 2 — Risque élevé de sepsis — SOFA complet requis.",
            trigger_value=qsofa_score, threshold=2,
            clinical_action="Évaluer SOFA complet + bilan infectieux + hémocultures",
            priority=2,
        ))

    if systolic_bp < 100 and systolic_bp >= 70:
        alerts.append(SepsisAlert(
            level="WARNING", code="HYPOTENSION",
            title=f"HYPOTENSION — PAS {systolic_bp:.0f} mmHg",
            message=f"PAS {systolic_bp:.0f} mmHg < 100 mmHg — Critère choc septique possible.",
            trigger_value=systolic_bp, threshold=100,
            clinical_action="Remplissage vasculaire + surveillance MAP + évaluer vasopresseurs",
            priority=2,
        ))

    if resp_rate >= 25:
        alerts.append(SepsisAlert(
            level="WARNING", code="TACHYPNEA",
            title=f"TACHYPNÉE — FR {resp_rate:.0f}/min",
            message=f"Fréquence respiratoire {resp_rate:.0f}/min ≥ 25 — critère qSOFA + NEWS2.",
            trigger_value=resp_rate, threshold=25,
            clinical_action="Gazométrie artérielle + saturation continue + évaluer O2",
            priority=2,
        ))

    if temperature > 38.5 or temperature < 36.0:
        direction = "Hyperthermie" if temperature > 38.5 else "Hypothermie"
        alerts.append(SepsisAlert(
            level="WARNING", code="TEMPERATURE_ALERT",
            title=f"{direction} — {temperature:.1f}°C",
            message=f"Température {temperature:.1f}°C — critère SIRS. Infection active probable.",
            trigger_value=temperature, threshold=38.5 if temperature > 38.5 else 36.0,
            clinical_action="Hémocultures + ECBU + bilan infectieux complet",
            priority=2,
        ))

    if inr >= 1.5:
        alerts.append(SepsisAlert(
            level="WARNING", code="COAGULOPATHY",
            title=f"COAGULOPATHIE — INR {inr:.1f}",
            message=f"INR {inr:.1f} ≥ 1.5 — Dysfonction hépatique ou CIVD possible.",
            trigger_value=inr, threshold=1.5,
            clinical_action="Bilan hémostase complet + D-dimères + fibrinogène — avis hématologie si CIVD",
            priority=2,
        ))

    # ── ALERTES INFO ─────────────────────────────────────────────────────────

    if not params.get("blood_cultures_done") and prediction not in ("Pas de sepsis",):
        alerts.append(SepsisAlert(
            level="INFO", code="BLOOD_CULTURES_MISSING",
            title="Hémocultures non réalisées",
            message="Les hémocultures n'ont pas été réalisées avant l'antibiothérapie.",
            trigger_value=False, threshold=True,
            clinical_action="Prélever 2 hémocultures (aérobie + anaérobie) AVANT tout antibiotique",
            priority=1,
            bundle_item="Bundle SSC 1h — Item 1",
        ))

    if not params.get("antibiotics_given") and prediction in ("Sepsis", "Sepsis sévère", "Choc septique"):
        alerts.append(SepsisAlert(
            level="INFO", code="ANTIBIOTICS_MISSING",
            title="Antibiothérapie non initiée",
            message="Les antibiotiques large spectre n'ont pas encore été administrés.",
            trigger_value=False, threshold=True,
            clinical_action="Initier antibiothérapie large spectre dans l'heure (ex: ceftriaxone + métronidazole + amikacine)",
            priority=1,
            bundle_item="Bundle SSC 1h — Item 3",
        ))

    if not params.get("fluid_resuscitation") and (systolic_bp < 90 or lactate >= 2.0):
        alerts.append(SepsisAlert(
            level="INFO", code="FLUID_MISSING",
            title="Remplissage vasculaire non réalisé",
            message="Le remplissage cristalloïde 30 mL/kg n'est pas documenté.",
            trigger_value=False, threshold=True,
            clinical_action="Remplissage NaCl 0.9% ou Ringer Lactate 30 mL/kg en 3h — réévaluer MAP et diurèse",
            priority=1,
            bundle_item="Bundle SSC 1h — Item 4",
        ))

    # Dédoublonner et trier par priorité
    seen_codes = set()
    unique_alerts = []
    for a in sorted(alerts, key=lambda x: -x.priority):
        if a.code not in seen_codes:
            seen_codes.add(a.code)
            unique_alerts.append(a.to_dict())

    return unique_alerts


def get_alert_summary(alerts: list[dict[str, Any]]) -> dict[str, Any]:
    """Résumé statistique des alertes générées."""
    counts = {lvl: 0 for lvl in ALERT_LEVELS}
    for a in alerts:
        counts[a["level"]] = counts.get(a["level"], 0) + 1

    highest = "INFO"
    for lvl in ["CRITICAL", "HIGH", "WARNING", "INFO"]:
        if counts[lvl] > 0:
            highest = lvl
            break

    return {
        "total_alerts":   len(alerts),
        "by_level":       counts,
        "highest_level":  highest,
        "highest_color":  ALERT_LEVELS[highest]["color"],
        "highest_icon":   ALERT_LEVELS[highest]["icon"],
        "requires_immediate_action": highest in ("CRITICAL", "HIGH"),
    }
