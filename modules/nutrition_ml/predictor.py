from __future__ import annotations

import base64
import pickle
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from modules.nutrition_ml.features import build_feature_payload

MODEL_PATH   = Path("models/machine_learning/nutrition_model.pkl")
CHARTS_DIR   = Path("models/machine_learning")

_CLASS_MAP = {
    "MAS":        "severe_undernutrition",
    "MAM":        "moderate_undernutrition",
    "Normal":     "normal",
    "Overweight": "overweight",
    "Obesity":    "obesity",
}

# ── Recommandations cliniques par classe ──────────────────────────────────────

_RECOMMENDATIONS: dict[str, dict[str, Any]] = {
    "severe_undernutrition": {
        "urgency":     "CRITIQUE",
        "urgency_color": "#C0392B",
        "label_fr":    "Malnutrition Aiguë Sévère (MAS)",
        "actions": [
            "Référence URGENTE en unité de nutrition thérapeutique (UNT)",
            "Initier traitement ATPE (Aliment Thérapeutique Prêt à l'Emploi) si appétit conservé",
            "Évaluer les complications : hypoglycémie, déshydratation, infections",
            "Surveillance médicale quotidienne — prise en charge hospitalière",
            "Protocole IMCI/PCIMA selon directives OMS si enfant < 5 ans",
        ],
        "diet_advice": "ATPE (plumpy'nut) 3×/j ou F75/F100 selon état clinique",
        "monitoring":  "Pesée quotidienne — objectif gain pondéral > 8 g/kg/j",
        "oms_threshold": "WHZ < -3 ou MUAC < 11.5 cm",
    },
    "moderate_undernutrition": {
        "urgency":     "ÉLEVÉE",
        "urgency_color": "#E67E22",
        "label_fr":    "Malnutrition Aiguë Modérée (MAM)",
        "actions": [
            "Inclure dans programme de supplémentation nutritionnelle (PSN)",
            "Distribuer aliment de supplémentation (CSB++, Plumpy'Sup)",
            "Contrôle mensuel du poids et MUAC",
            "Éducation nutritionnelle de la famille",
            "Vaccination et déparasitage selon calendrier national",
        ],
        "diet_advice": "Augmenter apports énergétiques de 20% — alimentation enrichie",
        "monitoring":  "Pesée hebdomadaire — contrôle MUAC toutes les 2 semaines",
        "oms_threshold": "-3 ≤ WHZ < -2 ou 11.5 ≤ MUAC < 12.5 cm",
    },
    "normal": {
        "urgency":     "FAIBLE",
        "urgency_color": "#27AE60",
        "label_fr":    "Statut nutritionnel normal",
        "actions": [
            "Maintenir alimentation équilibrée et diversifiée",
            "Suivi nutritionnel de routine tous les 3 mois",
            "Encourager activité physique adaptée à l'âge",
            "Prévention — surveiller courbe de croissance",
        ],
        "diet_advice": "Alimentation variée — 5 groupes alimentaires par jour",
        "monitoring":  "Consultation nutritionnelle tous les 3 mois",
        "oms_threshold": "-2 ≤ WHZ ≤ +2",
    },
    "overweight": {
        "urgency":     "MODÉRÉE",
        "urgency_color": "#F39C12",
        "label_fr":    "Surpoids",
        "actions": [
            "Consultation diététique pour rééquilibrage alimentaire",
            "Réduire aliments ultra-transformés et sucres ajoutés",
            "Augmenter activité physique (30 min/j minimum)",
            "Suivi mensuel du poids et IMC",
            "Évaluer risques métaboliques si surpoids persistant > 6 mois",
        ],
        "diet_advice": "Réduction calorique modérée — éviter régimes restrictifs",
        "monitoring":  "Pesée mensuelle — consultation médicale si IMC > 27",
        "oms_threshold": "+2 ≤ WHZ ≤ +3 ou 25 ≤ IMC < 30",
    },
    "obesity": {
        "urgency":     "ÉLEVÉE",
        "urgency_color": "#E74C3C",
        "label_fr":    "Obésité",
        "actions": [
            "Consultation médicale spécialisée (endocrinologie/nutrition)",
            "Bilan métabolique complet : glycémie, lipides, tension artérielle",
            "Programme de modification comportementale alimentaire",
            "Activité physique adaptée et progressive",
            "Évaluation psychologique si obésité sévère",
        ],
        "diet_advice": "Programme diététique personnalisé — accompagnement pluridisciplinaire",
        "monitoring":  "Suivi médical mensuel — évaluation comorbidités",
        "oms_threshold": "WHZ > +3 ou IMC ≥ 30",
    },
}


# ── SHAP explainability ────────────────────────────────────────────────────────

def _compute_shap_values(bundle: dict, X: np.ndarray) -> dict[str, Any]:
    """
    Calcule les SHAP values via TreeExplainer (shap library).
    Fallback sur RF feature_importances_ si shap non installé.
    """
    feature_names = bundle["features"]
    try:
        import shap
        voting = bundle["ensemble"]
        rf = dict(voting.estimators).get("rf")
        if rf is None:
            raise ValueError("RF estimator not found")

        explainer   = shap.TreeExplainer(rf)
        shap_vals   = explainer.shap_values(X)  # (n_samples, n_features, n_classes) ou list
        classes     = bundle["classes"]

        # Proba pour identifier la classe prédite
        proba     = voting.predict_proba(X)[0]
        pred_idx  = int(np.argmax(proba))

        # Gérer les deux formats SHAP (ancien: list[classes], nouveau: ndarray 3D)
        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            sv_class = shap_vals[0, :, pred_idx]          # (n_features,) pour la classe prédite
        elif isinstance(shap_vals, list):
            sv_class = shap_vals[pred_idx][0]              # ancien format
        else:
            sv_class = shap_vals[0]                        # fallback

        ranked = sorted(
            zip(feature_names, sv_class),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        top_features = [
            {
                "feature":    name,
                "shap_value": round(float(val), 4),
                "direction":  "+" if val > 0 else "-",
                "importance": round(abs(float(val)), 4),
            }
            for name, val in ranked[:7]
        ]

        # Générer image SHAP force plot (bar simple)
        shap_img_b64 = _generate_shap_bar_b64(top_features, classes[pred_idx])

        return {
            "status":        "generated",
            "method":        "shap_tree_explainer",
            "top_features":  top_features,
            "shap_img_b64":  shap_img_b64,
            "predicted_class_shap": classes[pred_idx],
        }

    except ImportError:
        # Fallback RF importance
        return _compute_rf_importance(bundle)
    except Exception as exc:
        return _compute_rf_importance(bundle, error=str(exc))


def _compute_rf_importance(bundle: dict, error: str | None = None) -> dict[str, Any]:
    """Fallback : importances globales RF + graphique barre (sans shap)."""
    try:
        voting        = bundle["ensemble"]
        rf            = dict(voting.estimators).get("rf")
        if rf is None:
            return {"status": "rf_not_found"}
        importances   = rf.feature_importances_
        feature_names = bundle["features"]
        ranked        = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        top_features  = [
            {"feature": n, "importance": round(float(v), 4), "shap_value": round(float(v), 4), "direction": "+"}
            for n, v in ranked[:7]
        ]
        # Génère quand même le graphique (importance comme proxy SHAP)
        pred_class    = bundle["classes"][0] if bundle.get("classes") else "?"
        img_b64       = _generate_shap_bar_b64(top_features, pred_class)
        return {
            "status":        "generated",
            "method":        "random_forest_feature_importance",
            "top_features":  top_features,
            "shap_img_b64":  img_b64,
            **({"fallback_reason": error} if error else {}),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _generate_shap_bar_b64(top_features: list[dict], pred_class: str) -> str | None:
    """Génère un graphique à barres SHAP horizontales en base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names  = [f["feature"] for f in top_features[:6]][::-1]
        values = [f["shap_value"] for f in top_features[:6]][::-1]
        colors = ["#C0392B" if v > 0 else "#2980B9" for v in values]

        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="#1A2B3C", linewidth=0.8)
        ax.set_xlabel("Contribution SHAP", fontsize=9)
        ax.set_title(f"Facteurs prédictifs — {pred_class}", fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        for bar, val in zip(bars, values):
            ax.text(val + (0.001 if val >= 0 else -0.001), bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right",
                    fontsize=7.5, color="#1A2B3C")
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


# ── Validation des entrées ────────────────────────────────────────────────────

def _validate_inputs(payload: dict[str, Any]) -> dict[str, Any]:
    """Validation clinique des valeurs biométriques."""
    errors, warnings = [], []

    w  = payload.get("weight_kg")
    h  = payload.get("height_cm")
    a  = payload.get("age_months")
    mu = payload.get("muac_cm")

    if w is not None and (w <= 0 or w > 300):
        errors.append(f"Poids invalide : {w} kg (attendu : 0–300 kg)")
    if h is not None and (h <= 0 or h > 250):
        errors.append(f"Taille invalide : {h} cm (attendu : 0–250 cm)")
    if a is not None and a < 0:
        errors.append(f"Âge invalide : {a} mois")
    if mu is not None and (mu < 5 or mu > 50):
        warnings.append(f"MUAC inhabituel : {mu} cm (normal : 10–30 cm)")

    # Cohérence poids/taille
    if w and h:
        bmi = w / (h / 100) ** 2
        if bmi < 8 or bmi > 70:
            warnings.append(f"IMC calculé inhabituel : {bmi:.1f} kg/m² — vérifier mesures")

    return {
        "valid":    len(errors) == 0,
        "errors":   errors,
        "warnings": warnings,
    }


# ── Placeholder ───────────────────────────────────────────────────────────────

def _placeholder_response(features: dict[str, Any]) -> dict[str, Any]:
    return {
        "module":            "module_2_nutrition",
        "task":              "nutritional_status_prediction",
        "model_family":      "machine_learning",
        "model_name":        "NutriTrack_AI_VotingClassifier",
        "model_version":     "v3.0",
        "library":           "scikit-learn + xgboost + lightgbm",
        "expected_model_path": str(MODEL_PATH).replace("\\", "/"),
        "request_id":        str(uuid.uuid4()),
        "inputs_summary": {
            "age_months":  features.get("age_months"),
            "weight_kg":   features.get("weight_kg"),
            "height_cm":   features.get("height_cm"),
            "sex":         features.get("sex"),
            "muac_cm":     features.get("muac_cm"),
        },
        "derived_features": {
            "bmi":                  features.get("bmi"),
            "waz":                  features.get("waz"),
            "haz":                  features.get("haz"),
            "whz":                  features.get("whz"),
            "nutrition_risk_score": features.get("nutrition_risk_score"),
            "age_group":            features.get("age_group"),
            "bmi_class":            features.get("bmi_class"),
        },
        "prediction":  None,
        "confidence":  None,
        "risk_level":  None,
        "explainability": {"method": "SHAP_TreeExplainer", "status": "not_generated"},
        "recommendations": None,
        "status":      "scaffold_ready",
    }


# ── Inférence principale ──────────────────────────────────────────────────────

def predict_nutrition(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Pipeline NutriTrack AI v3.0 :
    validation → feature engineering → inférence → SHAP → recommandations.
    """
    import time
    t0 = time.time()

    # Validation
    validation = _validate_inputs(payload)
    features   = build_feature_payload(payload)

    if not validation["valid"]:
        resp = _placeholder_response(features)
        resp["status"]     = "validation_error"
        resp["error"]      = " | ".join(validation["errors"])
        resp["validation"] = validation
        return resp

    model_path = Path.cwd() / MODEL_PATH
    if not model_path.exists():
        resp = _placeholder_response(features)
        if validation["warnings"]:
            resp["warnings"] = validation["warnings"]
        return resp

    try:
        with model_path.open("rb") as fh:
            bundle = pickle.load(fh)
    except Exception as exc:
        resp = _placeholder_response(features)
        resp["status"] = "model_load_error"
        resp["error"]  = str(exc)
        return resp

    try:
        voting        = bundle["ensemble"]
        label_encoder = bundle["label_encoder"]
        feature_cols  = bundle["features"]
        classes       = bundle["classes"]

        row  = [float(features.get(col, 0.0) or 0.0) for col in feature_cols]
        X    = np.array([row])

        proba      = voting.predict_proba(X)[0]
        pred_idx   = int(np.argmax(proba))
        raw_label  = classes[pred_idx]
        std_label  = _CLASS_MAP.get(raw_label, raw_label.lower())
        confidence = round(float(proba[pred_idx]), 4)

        class_scores = {
            _CLASS_MAP.get(cls, cls.lower()): round(float(p), 4)
            for cls, p in zip(classes, proba)
        }

        # Contributions par estimateur
        rf_pred  = dict(voting.estimators)["rf"].predict(X)[0]
        xgb_pred = dict(voting.estimators)["xgb"].predict(X)[0]
        try:
            rf_label  = label_encoder.inverse_transform([rf_pred])[0]
            xgb_label = label_encoder.inverse_transform([xgb_pred])[0]
        except Exception:
            rf_label = xgb_label = raw_label

        # SHAP
        shap_result = _compute_shap_values(bundle, X)

        # Niveau de risque
        risk_map = {
            "severe_undernutrition":   "critical",
            "moderate_undernutrition": "high",
            "normal":                  "low",
            "overweight":              "moderate",
            "obesity":                 "high",
        }
        risk_level = risk_map.get(std_label, "unknown")

        # Recommandations
        reco = _RECOMMENDATIONS.get(std_label, {})

        elapsed_ms = round((time.time() - t0) * 1000)

        return {
            "module":         "module_2_nutrition",
            "task":           "nutritional_status_prediction",
            "model_family":   "machine_learning",
            "model_name":     "NutriTrack_AI_VotingClassifier",
            "model_version":  "v3.0",
            "library":        "scikit-learn + xgboost",
            "request_id":     str(uuid.uuid4()),

            # Données d'entrée
            "inputs_summary": {
                "age_months": features.get("age_months"),
                "weight_kg":  features.get("weight_kg"),
                "height_cm":  features.get("height_cm"),
                "sex":        features.get("sex"),
                "muac_cm":    features.get("muac_cm"),
            },
            "derived_features": {
                "bmi":                  features.get("bmi"),
                "waz":                  features.get("waz"),
                "haz":                  features.get("haz"),
                "whz":                  features.get("whz"),
                "nutrition_risk_score": features.get("nutrition_risk_score"),
                "age_group":            features.get("age_group"),
                "bmi_class":            features.get("bmi_class"),
            },

            # Résultat
            "prediction":          std_label,
            "prediction_raw":      raw_label,
            "confidence":          confidence,
            "risk_level":          risk_level,
            "class_probabilities": class_scores,

            # Estimateurs
            "ensemble_strategy":   "voting_classifier_soft",
            "base_model_outputs": {
                "random_forest": _CLASS_MAP.get(str(rf_label), str(rf_label)),
                "xgboost":       _CLASS_MAP.get(str(xgb_label), str(xgb_label)),
            },

            # Explainability SHAP
            "explainability": shap_result,

            # Recommandations cliniques
            "recommendations": reco,

            # Validation
            "validation": {
                "valid":    True,
                "warnings": validation["warnings"],
            },

            "processing_ms": elapsed_ms,
            "status":        "model_loaded",
        }

    except Exception as exc:
        resp = _placeholder_response(features)
        resp["status"] = "inference_error"
        resp["error"]  = str(exc)
        return resp
