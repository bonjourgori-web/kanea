"""
KANÉA — Safety Validation Engine
===================================
Phase 11 : Validation sécurité avant déploiement des modèles.
Métriques : Accuracy · Precision · Recall · F1 · Sensibilité · Spécificité · ROC-AUC
Validation clinique obligatoire avant mise en production.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.catalog import get_catalog
from pipeline.registry import REGISTRY

logger = logging.getLogger("kanea.safety")


# ═══════════════════════════════════════════════════════════════════════════════
# SEUILS DE SÉCURITÉ PAR MÉTRIQUE
# ═══════════════════════════════════════════════════════════════════════════════

MIN_THRESHOLDS = {
    "accuracy":    0.80,
    "precision":   0.75,
    "recall":      0.75,
    "f1":          0.75,
    "sensitivity": 0.80,
    "specificity": 0.75,
    "auc":         0.80,
    "auroc":       0.80,
    "kappa":       0.70,
    "dice":        0.75,
    "map50":       0.70,
}

# Métriques critiques (blocage déploiement si non atteintes)
CRITICAL_METRICS = {"recall", "sensitivity", "auc", "auroc", "f1"}


@dataclass
class SafetyReport:
    module_key: str
    run_id: str
    model_path: str
    metrics: dict[str, float]
    target_metrics: dict[str, float]
    passed_metrics: list[str]
    failed_metrics: list[str]
    critical_failures: list[str]
    overall_pass: bool
    clinical_validation_required: bool
    safety_level: str  # APPROVED | CONDITIONAL | BLOCKED
    recommendations: list[str]
    validated_at: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# CALCULATEUR DE MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsCalculator:
    """Calcule toutes les métriques de classification."""

    def compute_binary(
        self,
        y_true: list[int],
        y_pred: list[int],
        y_score: list[float] | None = None,
    ) -> dict[str, float]:
        if not y_true or not y_pred:
            return {}

        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        n  = len(y_true)

        accuracy    = (tp + tn) / n if n > 0 else 0.0
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        metrics = {
            "accuracy":    round(accuracy, 4),
            "precision":   round(precision, 4),
            "recall":      round(recall, 4),
            "sensitivity": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1":          round(f1, 4),
        }

        if y_score:
            try:
                auc = self._roc_auc(y_true, y_score)
                metrics["auc"] = round(auc, 4)
            except Exception:
                pass

        return metrics

    def compute_multiclass(
        self,
        y_true: list[int],
        y_pred: list[int],
    ) -> dict[str, float]:
        try:
            from sklearn.metrics import (
                accuracy_score, f1_score, precision_score,
                recall_score, cohen_kappa_score,
            )
            return {
                "accuracy":  round(accuracy_score(y_true, y_pred), 4),
                "f1":        round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
                "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
                "recall":    round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
                "kappa":     round(cohen_kappa_score(y_true, y_pred), 4),
            }
        except ImportError:
            return {}

    def _roc_auc(self, y_true: list[int], y_score: list[float]) -> float:
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(y_true, y_score))
        except Exception:
            # Calcul manuel (trapèze)
            pairs = sorted(zip(y_score, y_true), reverse=True)
            tp = fp = 0
            prev_tp = prev_fp = 0
            auc = 0.0
            total_pos = sum(y_true)
            total_neg = len(y_true) - total_pos
            if total_pos == 0 or total_neg == 0:
                return 0.5
            for _, label in pairs:
                if label == 1:
                    tp += 1
                else:
                    fp += 1
                auc += (fp - prev_fp) * (tp + prev_tp) / 2
                prev_tp, prev_fp = tp, fp
            return auc / (total_pos * total_neg)


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATEUR DE SÉCURITÉ
# ═══════════════════════════════════════════════════════════════════════════════

class SafetyValidator:
    def __init__(self):
        self.catalog    = get_catalog()
        self.calculator = MetricsCalculator()

    def validate_run(self, run_id: str) -> SafetyReport:
        """Valide un run d'entraînement contre les seuils de sécurité."""
        runs = self.catalog.list_runs()
        run  = next((r for r in runs if r.get("run_id") == run_id), None)

        if not run:
            raise ValueError(f"Run '{run_id}' non trouvé")

        module_key = run["module_key"]
        mod        = REGISTRY.get(module_key)
        if not mod:
            raise ValueError(f"Module '{module_key}' non trouvé")

        current_metrics = json.loads(run.get("metrics") or "{}")
        target_metrics  = mod.target_metrics
        model_path      = run.get("model_path", "")

        passed_metrics   : list[str] = []
        failed_metrics   : list[str] = []
        critical_failures: list[str] = []
        recommendations  : list[str] = []

        # Évaluation de chaque métrique
        all_metrics_checked = set(current_metrics.keys()) | set(target_metrics.keys())

        for metric in all_metrics_checked:
            current_val = current_metrics.get(metric)
            target_val  = target_metrics.get(metric)
            min_val     = MIN_THRESHOLDS.get(metric, 0.75)

            if current_val is None:
                recommendations.append(f"Métrique '{metric}' non calculée — évaluation manuelle requise")
                continue

            passed = current_val >= min_val
            if passed:
                passed_metrics.append(f"{metric}={current_val:.3f} ✓")
            else:
                failed_metrics.append(
                    f"{metric}={current_val:.3f} < seuil {min_val:.3f}"
                )
                if metric in CRITICAL_METRICS:
                    critical_failures.append(metric)
                    recommendations.append(
                        f"CRITIQUE : {metric} {current_val:.3f} < {min_val:.3f} — "
                        f"augmenter le dataset ou ajuster les hyperparamètres"
                    )
                else:
                    recommendations.append(
                        f"{metric} sous le seuil ({current_val:.3f}) — amélioration recommandée"
                    )

        # Décision globale
        overall_pass = len(critical_failures) == 0

        if overall_pass and len(failed_metrics) == 0:
            safety_level = "APPROVED"
        elif overall_pass and len(failed_metrics) > 0:
            safety_level = "CONDITIONAL"
            recommendations.append(
                "Déploiement conditionnel : surveillance rapprochée post-déploiement obligatoire"
            )
        else:
            safety_level = "BLOCKED"
            recommendations.append(
                "DÉPLOIEMENT BLOQUÉ — métriques critiques insuffisantes — "
                "réentraîner avec plus de données ou ajuster le modèle"
            )

        clinical_required = safety_level in ("CONDITIONAL", "APPROVED")
        if clinical_required:
            recommendations.append(
                "VALIDATION CLINIQUE OBLIGATOIRE avant mise en production. "
                "Revue par un expert médical certifié est requise."
            )

        # Mise à jour du catalogue
        if overall_pass:
            self.catalog.update_training_run(run_id, status="validated")

        self.catalog.log_event(
            "safety_validation", module_key, "Phase 11 - Safety",
            safety_level.lower(),
            f"Run {run_id} — {safety_level} — {len(critical_failures)} échecs critiques",
            {
                "safety_level": safety_level,
                "passed": len(passed_metrics),
                "failed": len(failed_metrics),
                "critical": critical_failures,
            },
        )

        now = datetime.utcnow().isoformat()

        return SafetyReport(
            module_key=module_key,
            run_id=run_id,
            model_path=model_path,
            metrics=current_metrics,
            target_metrics=target_metrics,
            passed_metrics=passed_metrics,
            failed_metrics=failed_metrics,
            critical_failures=critical_failures,
            overall_pass=overall_pass,
            clinical_validation_required=clinical_required,
            safety_level=safety_level,
            recommendations=recommendations,
            validated_at=now,
        )

    def validate_module_latest(self, module_key: str) -> SafetyReport | None:
        """Valide le dernier run d'un module."""
        runs = self.catalog.list_runs(module_key=module_key)
        completed_runs = [
            r for r in runs
            if r.get("status") in ("completed", "simulated")
        ]
        if not completed_runs:
            logger.warning(f"[Safety] Aucun run complété pour {module_key}")
            return None
        latest = completed_runs[0]
        return self.validate_run(latest["run_id"])

    def validate_all_modules(
        self, modules: list[str] | None = None
    ) -> list[SafetyReport]:
        targets = modules or list(REGISTRY.keys())
        reports = []
        for key in targets:
            try:
                report = self.validate_module_latest(key)
                if report:
                    reports.append(report)
            except Exception as e:
                logger.error(f"[Safety] Erreur {key} : {e}")
        return reports

    def generate_safety_summary(
        self, reports: list[SafetyReport]
    ) -> dict[str, Any]:
        if not reports:
            return {"total": 0}

        approved    = [r for r in reports if r.safety_level == "APPROVED"]
        conditional = [r for r in reports if r.safety_level == "CONDITIONAL"]
        blocked     = [r for r in reports if r.safety_level == "BLOCKED"]

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total": len(reports),
            "approved": len(approved),
            "conditional": len(conditional),
            "blocked": len(blocked),
            "clinical_validation_required": len([r for r in reports if r.clinical_validation_required]),
            "modules": {
                r.module_key: {
                    "safety_level": r.safety_level,
                    "overall_pass": r.overall_pass,
                    "metrics": r.metrics,
                    "critical_failures": r.critical_failures,
                    "recommendations_count": len(r.recommendations),
                }
                for r in reports
            },
        }


def run_safety_validation(
    module_key: str | None = None,
    run_id: str | None = None,
    modules: list[str] | None = None,
) -> dict[str, Any]:
    """Point d'entrée Phase 11."""
    validator = SafetyValidator()

    if run_id:
        report = validator.validate_run(run_id)
        return {
            "module": report.module_key,
            "run_id": report.run_id,
            "safety_level": report.safety_level,
            "overall_pass": report.overall_pass,
            "metrics": report.metrics,
            "critical_failures": report.critical_failures,
            "recommendations": report.recommendations,
        }

    if module_key:
        report = validator.validate_module_latest(module_key)
        if not report:
            return {"error": f"Aucun run trouvé pour {module_key}"}
        return {
            "module": report.module_key,
            "safety_level": report.safety_level,
            "overall_pass": report.overall_pass,
            "metrics": report.metrics,
            "recommendations": report.recommendations,
        }

    reports = validator.validate_all_modules(modules=modules)
    return validator.generate_safety_summary(reports)
