"""
KANÉA — Auto Retraining Engine
================================
Phase 8 : Réentraînement automatique déclenché si :
  • Nouveau dataset détecté (croissance > 10%)
  • Baisse de performance > 3%
  • Schedule programmé
Pipeline : New Data → Training → Validation → Clinical Check → Deployment
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.catalog import get_catalog
from pipeline.registry import REGISTRY
from pipeline.trainer import TrainingEngine, TrainingResult

logger = logging.getLogger("kanea.retrainer")

GROWTH_THRESHOLD   = 0.10   # 10% de nouvelles données
PERF_DROP_THRESHOLD = 0.03  # 3% de baisse de performance


# ═══════════════════════════════════════════════════════════════════════════════
# DÉTECTEUR DE DÉCLENCHEURS
# ═══════════════════════════════════════════════════════════════════════════════

class RetrainingTrigger:
    """Analyse le catalogue pour décider si un réentraînement est nécessaire."""

    def __init__(self):
        self.catalog = get_catalog()

    def check_new_data(self, module_key: str) -> dict[str, Any]:
        """Vérifie si de nouveaux datasets ont été ajoutés depuis le dernier entraînement."""
        best_run = self.catalog.get_best_run(module_key)
        if not best_run:
            return {"trigger": True, "reason": "Aucun modèle entraîné — premier entraînement requis"}

        last_train = best_run.get("started_at", "")
        old_ds_ids = json.loads(best_run.get("dataset_ids") or "[]")

        all_datasets = self.catalog.get_ready_datasets(module_key)
        new_ds_ids = [d["id"] for d in all_datasets if d["id"] not in old_ds_ids]

        if not all_datasets or not old_ds_ids:
            growth = 0.0
        else:
            growth = len(new_ds_ids) / max(len(old_ds_ids), 1)

        trigger = growth >= GROWTH_THRESHOLD
        return {
            "trigger": trigger,
            "reason": f"Croissance données {growth:.1%} ({'≥' if trigger else '<'} seuil {GROWTH_THRESHOLD:.0%})",
            "new_datasets": len(new_ds_ids),
            "total_datasets": len(all_datasets),
            "growth_rate": growth,
        }

    def check_performance_drop(self, module_key: str) -> dict[str, Any]:
        """Vérifie si les métriques ont chuté par rapport au target."""
        mod = REGISTRY.get(module_key)
        if not mod:
            return {"trigger": False, "reason": "Module inconnu"}

        best_run = self.catalog.get_best_run(module_key)
        if not best_run:
            return {"trigger": False, "reason": "Aucun run validé"}

        current_metrics = json.loads(best_run.get("metrics") or "{}")
        target_metrics  = mod.target_metrics

        drops: list[str] = []
        trigger = False

        for metric, target_val in target_metrics.items():
            current = current_metrics.get(metric)
            if current is None:
                continue
            drop = target_val - current
            if drop > PERF_DROP_THRESHOLD:
                trigger = True
                drops.append(f"{metric}: {current:.3f} < cible {target_val:.3f} (drop={drop:.3f})")

        return {
            "trigger": trigger,
            "reason": f"Baisse performance: {'; '.join(drops)}" if drops else "Performances OK",
            "drops": drops,
        }

    def check_all_triggers(self, module_key: str) -> dict[str, Any]:
        data_check  = self.check_new_data(module_key)
        perf_check  = self.check_performance_drop(module_key)

        should_retrain = data_check["trigger"] or perf_check["trigger"]

        reasons = []
        if data_check["trigger"]:
            reasons.append(data_check["reason"])
        if perf_check["trigger"]:
            reasons.append(perf_check["reason"])

        return {
            "module_key": module_key,
            "should_retrain": should_retrain,
            "reasons": reasons,
            "data_check": data_check,
            "perf_check": perf_check,
            "checked_at": datetime.utcnow().isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GESTIONNAIRE DE RÉENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class RetrainingManager:
    def __init__(self):
        self.catalog  = get_catalog()
        self.trigger  = RetrainingTrigger()
        self.trainer  = TrainingEngine()

    def retrain_if_needed(
        self,
        module_key: str,
        force: bool = False,
        epochs: int = 30,
    ) -> dict[str, Any]:
        check = self.trigger.check_all_triggers(module_key)

        if not force and not check["should_retrain"]:
            logger.info(f"[Retrainer] {module_key} — Pas de réentraînement nécessaire")
            return {
                "module_key": module_key,
                "retrained": False,
                "reasons": check["reasons"],
                "check": check,
            }

        triggered_by = "forced" if force else (
            "new_data" if check["data_check"]["trigger"] else "performance_drop"
        )
        reasons = check["reasons"] if check["reasons"] else ["Force retrain"]

        logger.info(f"[Retrainer] {module_key} — Déclenchement : {'; '.join(reasons)}")
        self.catalog.log_event(
            "retrain_trigger", module_key, "Phase 8 - Retraining",
            "triggered", "; ".join(reasons),
            {"reasons": reasons, "check": check},
        )

        result = self.trainer.train_module(
            module_key, epochs=epochs, triggered_by=triggered_by
        )

        self.catalog.log_event(
            "retrain_complete", module_key, "Phase 8 - Retraining",
            "success" if result.status in ("completed", "simulated") else "failed",
            f"Run {result.run_id} — status={result.status} — metrics={result.metrics}",
            {"run_id": result.run_id, "metrics": result.metrics},
        )

        return {
            "module_key": module_key,
            "retrained": True,
            "triggered_by": triggered_by,
            "reasons": reasons,
            "run_id": result.run_id,
            "status": result.status,
            "metrics": result.metrics,
            "model_path": result.model_path,
        }

    def check_all_modules(
        self,
        modules: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        targets = modules or list(REGISTRY.keys())
        return [self.trigger.check_all_triggers(k) for k in targets]

    def retrain_all(
        self,
        modules: list[str] | None = None,
        force: bool = False,
        epochs: int = 30,
    ) -> list[dict[str, Any]]:
        targets = modules or list(REGISTRY.keys())
        results = []
        for key in targets:
            try:
                r = self.retrain_if_needed(key, force=force, epochs=epochs)
                results.append(r)
            except Exception as e:
                logger.error(f"[Retrainer] Erreur {key} : {e}")
                results.append({"module_key": key, "retrained": False, "error": str(e)})
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# APPRENTISSAGE CONTINU — Phase 9
# ═══════════════════════════════════════════════════════════════════════════════

class ContinuousLearningManager:
    """
    Phase 9 : Intègre les nouveaux cas validés, met à jour les datasets,
    réentraîne et archive les anciens modèles.
    """

    def __init__(self):
        self.catalog  = get_catalog()
        self.retrainer = RetrainingManager()

    def integrate_validated_case(
        self,
        module_key: str,
        case_data: dict[str, Any],
        label: str,
    ) -> dict[str, Any]:
        """Intègre un nouveau cas validé par un clinicien."""
        now = datetime.utcnow().isoformat()
        case_dir = (
            Path(__file__).resolve().parent.parent
            / "data" / "continuous_learning" / module_key
        )
        case_dir.mkdir(parents=True, exist_ok=True)

        case_id = f"case_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        case_file = case_dir / f"{case_id}.json"
        case_file.write_text(
            json.dumps({
                "case_id": case_id,
                "module_key": module_key,
                "label": label,
                "data": case_data,
                "validated_at": now,
                "source": "expert_validation",
            }),
            encoding="utf-8",
        )

        self.catalog.log_event(
            "case_integrated", module_key, "Phase 9 - Continuous Learning",
            "success", f"Cas {case_id} intégré — label={label}",
        )
        return {"case_id": case_id, "module_key": module_key, "label": label}

    def count_pending_cases(self, module_key: str) -> int:
        case_dir = (
            Path(__file__).resolve().parent.parent
            / "data" / "continuous_learning" / module_key
        )
        if not case_dir.exists():
            return 0
        return len(list(case_dir.glob("*.json")))

    def archive_old_model(self, module_key: str) -> None:
        """Archive le modèle précédent avant mise à jour."""
        archive_dir = MODELS_ROOT = (
            Path(__file__).resolve().parent.parent
            / "models" / "archive" / module_key
        )
        archive_dir.mkdir(parents=True, exist_ok=True)

        model_dir = (
            Path(__file__).resolve().parent.parent / "models" / "trained" / module_key
        )
        if model_dir.exists():
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
            dest = archive_dir / f"archived_{ts}"
            import shutil
            shutil.copytree(model_dir, dest, dirs_exist_ok=True)
            logger.info(f"[ContinuousLearning] Modèle archivé → {dest}")

    def trigger_continuous_update(
        self,
        module_key: str,
        min_new_cases: int = 50,
    ) -> dict[str, Any]:
        pending = self.count_pending_cases(module_key)
        if pending < min_new_cases:
            return {
                "module_key": module_key,
                "triggered": False,
                "reason": f"{pending}/{min_new_cases} cas — seuil non atteint",
            }

        self.archive_old_model(module_key)
        result = self.retrainer.retrain_if_needed(module_key, force=True)

        return {
            "module_key": module_key,
            "triggered": True,
            "new_cases": pending,
            "retrain_result": result,
        }


def run_retraining(
    module_key: str | None = None,
    force: bool = False,
    epochs: int = 30,
    check_only: bool = False,
) -> dict[str, Any]:
    """Point d'entrée Phases 8 & 9."""
    manager = RetrainingManager()

    if check_only:
        modules = [module_key] if module_key else list(REGISTRY.keys())
        return {
            "mode": "check_only",
            "checks": manager.check_all_modules(modules),
        }

    if module_key:
        return manager.retrain_if_needed(module_key, force=force, epochs=epochs)
    else:
        results = manager.retrain_all(force=force, epochs=epochs)
        return {
            "total": len(results),
            "retrained": sum(1 for r in results if r.get("retrained")),
            "skipped": sum(1 for r in results if not r.get("retrained")),
            "results": results,
        }
