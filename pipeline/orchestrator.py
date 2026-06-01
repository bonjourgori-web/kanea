"""
KANÉA — Universal Medical Data Acquisition & Auto-Training Orchestrator
========================================================================
Orchestrateur principal : coordonne les 11 phases du pipeline.

Phase 1  → registry       : Registre des sources
Phase 2  → discovery      : Auto-découverte des datasets
Phase 3  → downloader     : Téléchargement + intégrité
Phase 4  → quality        : Contrôle qualité (seuil 80/100)
Phase 5  → preprocessor   : Prétraitement par modalité
Phase 6  → annotator      : Auto-annotation SAM/GroundingDINO
Phase 7  → trainer        : Pipeline d'entraînement
Phase 8  → retrainer      : Réentraînement automatique
Phase 9  → retrainer      : Apprentissage continu
Phase 10 → knowledge      : Mise à jour connaissances médicales
Phase 11 → safety         : Validation sécurité avant déploiement
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.catalog import get_catalog
from pipeline.registry import REGISTRY

logger = logging.getLogger("kanea.orchestrator")

LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "pipeline"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DU PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    modules: list[str] = field(default_factory=lambda: list(REGISTRY.keys()))
    phases: list[int]  = field(default_factory=lambda: list(range(1, 12)))
    epochs: int        = 30
    batch_size: int    = 32
    lr: float          = 1e-4
    force_retrain: bool = False
    credentials: dict[str, str] = field(default_factory=dict)
    dry_run: bool      = False
    delay_between_phases: float = 1.0


@dataclass
class PipelineRun:
    run_id: str
    started_at: str
    config: PipelineConfig
    phase_results: dict[int, dict[str, Any]] = field(default_factory=dict)
    status: str = "running"
    finished_at: str = ""
    errors: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATEUR
# ═══════════════════════════════════════════════════════════════════════════════

class MedicalPipelineOrchestrator:

    PHASE_NAMES = {
        1:  "Medical Data Sources Registry",
        2:  "Auto Data Discovery",
        3:  "Auto Download",
        4:  "Data Quality Control",
        5:  "Data Preprocessing",
        6:  "Auto Annotation",
        7:  "Training Pipeline",
        8:  "Auto Retraining",
        9:  "Continuous Learning",
        10: "Knowledge Update",
        11: "Safety Validation",
    }

    def __init__(self):
        self.catalog = get_catalog()

    # ── Phases individuelles ──────────────────────────────────────────────────

    def _run_phase_1(self, cfg: PipelineConfig) -> dict[str, Any]:
        """Phase 1 — Registre des sources (statique)."""
        sources_count = sum(len(m.sources) for m in REGISTRY.values())
        return {
            "phase": 1,
            "status": "ok",
            "modules": len(REGISTRY),
            "total_sources": sources_count,
            "modules_list": list(REGISTRY.keys()),
        }

    def _run_phase_2(self, cfg: PipelineConfig) -> dict[str, Any]:
        """Phase 2 — Auto-découverte."""
        from pipeline.discovery import run_discovery
        logger.info("[Phase 2] Auto-découverte des datasets")
        return run_discovery(modules=cfg.modules)

    def _run_phase_3(self, cfg: PipelineConfig) -> dict[str, Any]:
        """Phase 3 — Téléchargement."""
        if cfg.dry_run:
            return {"phase": 3, "status": "skipped", "reason": "dry_run"}
        from pipeline.downloader import run_download
        logger.info("[Phase 3] Téléchargement des datasets")
        results = {}
        for mod in cfg.modules:
            try:
                r = run_download(module_key=mod, credentials=cfg.credentials)
                results[mod] = r
            except Exception as e:
                results[mod] = {"error": str(e)}
        return {"phase": 3, "status": "ok", "results": results}

    def _run_phase_4(self, cfg: PipelineConfig) -> dict[str, Any]:
        """Phase 4 — Contrôle qualité."""
        from pipeline.quality import run_quality_check
        logger.info("[Phase 4] Contrôle qualité")
        results = {}
        for mod in cfg.modules:
            try:
                results[mod] = run_quality_check(module_key=mod)
            except Exception as e:
                results[mod] = {"error": str(e)}
        return {"phase": 4, "status": "ok", "results": results}

    def _run_phase_5(self, cfg: PipelineConfig) -> dict[str, Any]:
        """Phase 5 — Prétraitement."""
        if cfg.dry_run:
            return {"phase": 5, "status": "skipped", "reason": "dry_run"}
        from pipeline.preprocessor import run_preprocessing
        logger.info("[Phase 5] Prétraitement des données")
        results = {}
        for mod in cfg.modules:
            try:
                results[mod] = run_preprocessing(module_key=mod)
            except Exception as e:
                results[mod] = {"error": str(e)}
        return {"phase": 5, "status": "ok", "results": results}

    def _run_phase_6(self, cfg: PipelineConfig) -> dict[str, Any]:
        """Phase 6 — Auto-annotation."""
        if cfg.dry_run:
            return {"phase": 6, "status": "skipped", "reason": "dry_run"}
        from pipeline.annotator import run_annotation
        logger.info("[Phase 6] Auto-annotation")
        results = {}
        for mod in cfg.modules:
            try:
                results[mod] = run_annotation(module_key=mod)
            except Exception as e:
                results[mod] = {"error": str(e)}
        return {"phase": 6, "status": "ok", "results": results}

    def _run_phase_7(self, cfg: PipelineConfig) -> dict[str, Any]:
        """Phase 7 — Entraînement."""
        if cfg.dry_run:
            return {"phase": 7, "status": "skipped", "reason": "dry_run"}
        from pipeline.trainer import run_training
        logger.info("[Phase 7] Pipeline d'entraînement")
        return run_training(modules=cfg.modules, epochs=cfg.epochs)

    def _run_phase_8(self, cfg: PipelineConfig) -> dict[str, Any]:
        """Phase 8 — Réentraînement automatique."""
        if cfg.dry_run:
            return {"phase": 8, "status": "skipped", "reason": "dry_run"}
        from pipeline.retrainer import run_retraining
        logger.info("[Phase 8] Réentraînement automatique")
        return run_retraining(force=cfg.force_retrain, epochs=cfg.epochs)

    def _run_phase_9(self, cfg: PipelineConfig) -> dict[str, Any]:
        """Phase 9 — Apprentissage continu."""
        from pipeline.retrainer import ContinuousLearningManager
        logger.info("[Phase 9] Apprentissage continu")
        manager = ContinuousLearningManager()
        status = {}
        for mod in cfg.modules:
            pending = manager.count_pending_cases(mod)
            status[mod] = {
                "pending_cases": pending,
                "threshold_reached": pending >= 50,
            }
        return {"phase": 9, "status": "ok", "continuous_learning_status": status}

    def _run_phase_10(self, cfg: PipelineConfig) -> dict[str, Any]:
        """Phase 10 — Mise à jour des connaissances."""
        from pipeline.knowledge import run_knowledge_update
        logger.info("[Phase 10] Mise à jour des connaissances médicales")
        return run_knowledge_update(modules=cfg.modules)

    def _run_phase_11(self, cfg: PipelineConfig) -> dict[str, Any]:
        """Phase 11 — Validation sécurité."""
        from pipeline.safety import run_safety_validation
        logger.info("[Phase 11] Validation sécurité")
        return run_safety_validation(modules=cfg.modules)

    _PHASE_HANDLERS = {
        1:  _run_phase_1,
        2:  _run_phase_2,
        3:  _run_phase_3,
        4:  _run_phase_4,
        5:  _run_phase_5,
        6:  _run_phase_6,
        7:  _run_phase_7,
        8:  _run_phase_8,
        9:  _run_phase_9,
        10: _run_phase_10,
        11: _run_phase_11,
    }

    # ── Pipeline principal ────────────────────────────────────────────────────

    def run(
        self,
        config: PipelineConfig | None = None,
        phases: list[int] | None = None,
        modules: list[str] | None = None,
    ) -> PipelineRun:
        cfg = config or PipelineConfig()
        if phases:
            cfg.phases = phases
        if modules:
            cfg.modules = modules

        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        started = datetime.utcnow().isoformat()

        run = PipelineRun(run_id=run_id, started_at=started, config=cfg)

        logger.info("=" * 70)
        logger.info(f"KANÉA — Universal Medical Pipeline — {run_id}")
        logger.info(f"Modules : {cfg.modules}")
        logger.info(f"Phases  : {cfg.phases}")
        logger.info(f"Dry-run : {cfg.dry_run}")
        logger.info("=" * 70)

        self.catalog.log_event(
            "pipeline_start", "", "Orchestrator", "started",
            f"Run {run_id} — {len(cfg.modules)} modules — phases {cfg.phases}",
            {"config": {"modules": cfg.modules, "phases": cfg.phases, "dry_run": cfg.dry_run}},
        )

        for phase_num in sorted(cfg.phases):
            handler = self._PHASE_HANDLERS.get(phase_num)
            if not handler:
                logger.warning(f"Phase {phase_num} inconnue — ignorée")
                continue

            phase_name = self.PHASE_NAMES.get(phase_num, f"Phase {phase_num}")
            logger.info(f"\n── Phase {phase_num} : {phase_name} ──")

            t0 = time.time()
            try:
                result = handler(self, cfg)
                elapsed = round(time.time() - t0, 2)
                result["_elapsed_s"] = elapsed
                run.phase_results[phase_num] = result

                logger.info(f"   ✓ Phase {phase_num} terminée en {elapsed}s")
                self.catalog.log_event(
                    f"phase_{phase_num}", "", phase_name,
                    "success", f"Terminée en {elapsed}s",
                    {"elapsed_s": elapsed},
                )

            except Exception as e:
                elapsed = round(time.time() - t0, 2)
                error_msg = f"Phase {phase_num} ({phase_name}) : {e}"
                logger.error(f"   ✗ {error_msg}")
                run.errors.append(error_msg)
                run.phase_results[phase_num] = {
                    "status": "error", "error": str(e), "_elapsed_s": elapsed
                }
                self.catalog.log_event(
                    f"phase_{phase_num}", "", phase_name,
                    "error", str(e),
                )

            time.sleep(cfg.delay_between_phases)

        run.status    = "completed" if not run.errors else "completed_with_errors"
        run.finished_at = datetime.utcnow().isoformat()

        self.catalog.log_event(
            "pipeline_complete", "", "Orchestrator", run.status,
            f"Run {run_id} — {len(run.errors)} erreurs",
            {"errors": run.errors, "phases_done": len(run.phase_results)},
        )

        # Sauvegarde du rapport
        self._save_run_report(run)
        logger.info(f"\n{'=' * 70}")
        logger.info(f"PIPELINE TERMINÉ — {run.status}")
        logger.info(f"Durée totale : {run.finished_at}")
        logger.info(f"Erreurs : {len(run.errors)}")
        logger.info("=" * 70)

        return run

    def _save_run_report(self, run: PipelineRun) -> None:
        report_path = LOG_DIR / f"pipeline_{run.run_id}.json"
        try:
            report = {
                "run_id":     run.run_id,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
                "status":     run.status,
                "config": {
                    "modules":  run.config.modules,
                    "phases":   run.config.phases,
                    "epochs":   run.config.epochs,
                    "dry_run":  run.config.dry_run,
                },
                "errors": run.errors,
                "phase_results": {
                    str(k): v for k, v in run.phase_results.items()
                },
                "catalog_stats": get_catalog().get_stats(),
            }
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(f"[Orchestrator] Rapport sauvegardé → {report_path}")
        except Exception as e:
            logger.warning(f"[Orchestrator] Erreur sauvegarde rapport : {e}")

    def get_pipeline_status(self) -> dict[str, Any]:
        """Retourne l'état global du pipeline."""
        stats = self.catalog.get_stats()
        runs  = self.catalog.list_runs()
        events = self.catalog.list_events(limit=20)

        return {
            "catalog": stats,
            "recent_runs": runs[:5],
            "recent_events": events[:10],
            "registry": {
                "modules": len(REGISTRY),
                "total_sources": sum(len(m.sources) for m in REGISTRY.values()),
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# API SIMPLIFIÉE
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    modules: list[str] | None = None,
    phases: list[int] | None = None,
    epochs: int = 30,
    dry_run: bool = False,
    credentials: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Lance le pipeline complet ou une sélection de phases."""
    cfg = PipelineConfig(
        modules=modules or list(REGISTRY.keys()),
        phases=phases or list(range(1, 12)),
        epochs=epochs,
        dry_run=dry_run,
        credentials=credentials or {},
    )
    orch = MedicalPipelineOrchestrator()
    run  = orch.run(config=cfg)

    return {
        "run_id":     run.run_id,
        "status":     run.status,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "errors":     run.errors,
        "phases_completed": len(run.phase_results),
        "catalog_stats": get_catalog().get_stats(),
    }


def get_pipeline_status() -> dict[str, Any]:
    orch = MedicalPipelineOrchestrator()
    return orch.get_pipeline_status()
