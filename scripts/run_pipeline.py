"""
KANÉA — Universal Pipeline CLI
================================
Usage :
  python scripts/run_pipeline.py --help
  python scripts/run_pipeline.py --phases 1 2 10          # Découverte + Connaissances
  python scripts/run_pipeline.py --module malaria          # Module spécifique
  python scripts/run_pipeline.py --all --dry-run           # Test sans téléchargement
  python scripts/run_pipeline.py --status                  # État du pipeline
  python scripts/run_pipeline.py --retrain --module hemato # Réentraînement
  python scripts/run_pipeline.py --safety --module hemato  # Validation sécurité
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Ajoute la racine KANEA au path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)-25s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                _ROOT / "logs" / "pipeline" / "cli.log",
                encoding="utf-8",
            ),
        ],
    )


def cmd_run(args: argparse.Namespace) -> None:
    from pipeline.orchestrator import run_full_pipeline
    from pipeline.registry import REGISTRY

    modules = [args.module] if args.module else (list(REGISTRY.keys()) if args.all else None)
    phases  = args.phases or (list(range(1, 12)) if args.all else [1, 2, 4, 10, 11])

    print(f"\n{'─'*60}")
    print(f"  KANÉA — Universal Medical Pipeline")
    print(f"  Modules : {modules or 'tous'}")
    print(f"  Phases  : {phases}")
    print(f"  Dry-run : {args.dry_run}")
    print(f"{'─'*60}\n")

    result = run_full_pipeline(
        modules=modules,
        phases=phases,
        epochs=args.epochs,
        dry_run=args.dry_run,
    )

    print(f"\n{'─'*60}")
    print(f"  Statut  : {result['status']}")
    print(f"  Run ID  : {result['run_id']}")
    print(f"  Phases  : {result['phases_completed']} terminées")
    print(f"  Erreurs : {len(result['errors'])}")
    if result["errors"]:
        print("\n  Erreurs détectées :")
        for e in result["errors"]:
            print(f"    ✗ {e}")
    print(f"{'─'*60}\n")

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Rapport sauvegardé : {args.output}")


def cmd_status(args: argparse.Namespace) -> None:
    from pipeline.orchestrator import get_pipeline_status

    status = get_pipeline_status()

    print(f"\n{'─'*60}")
    print(f"  KANÉA Pipeline Status")
    print(f"{'─'*60}")

    cat = status["catalog"]
    print(f"\n  Catalogue SQLite :")
    print(f"    Datasets total  : {cat['datasets_total']}")
    print(f"    Datasets ready  : {cat['datasets_ready']}")
    print(f"    Training runs   : {cat['training_runs_total']}")
    print(f"    Runs validés    : {cat['training_runs_validated']}")
    print(f"    Knowledge upd.  : {cat['knowledge_updates']}")
    print(f"    Événements      : {cat['pipeline_events']}")

    reg = status["registry"]
    print(f"\n  Registre :")
    print(f"    Modules         : {reg['modules']}")
    print(f"    Sources totales : {reg['total_sources']}")

    runs = status["recent_runs"]
    if runs:
        print(f"\n  Derniers entraînements :")
        for r in runs[:3]:
            metrics = json.loads(r.get("metrics") or "{}")
            print(f"    [{r['module_key']:15s}] {r['status']:12s}  {metrics}")

    events = status["recent_events"]
    if events:
        print(f"\n  Derniers événements :")
        for e in events[:5]:
            print(f"    [{e['created_at'][:16]}] {e['phase']:25s}  {e['status']:10s}  {e['message'][:50]}")

    print(f"\n{'─'*60}\n")


def cmd_discover(args: argparse.Namespace) -> None:
    from pipeline.discovery import run_discovery
    from pipeline.registry import REGISTRY

    modules = [args.module] if args.module else list(REGISTRY.keys())
    print(f"\n  Découverte — {len(modules)} modules...")
    report = run_discovery(modules=modules)
    print(f"\n  Résultats :")
    print(f"    Sources explorées : {report['total_sources']}")
    print(f"    Accessibles       : {report['reachable']}")
    print(f"    Inaccessibles     : {report['unreachable']}")
    print(f"\n  Par type d'API :")
    for api_type, stats in report.get("by_api_type", {}).items():
        print(f"    {api_type:12s} : {stats['reachable']}/{stats['total']}")
    print()


def cmd_retrain(args: argparse.Namespace) -> None:
    from pipeline.retrainer import run_retraining
    from pipeline.registry import REGISTRY

    if args.check_only:
        result = run_retraining(
            module_key=args.module, check_only=True
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    result = run_retraining(
        module_key=args.module,
        force=args.force,
        epochs=args.epochs,
    )
    print(f"\n  Réentraînement : {'déclenché' if result.get('retrained') else 'ignoré'}")
    if result.get("retrained"):
        print(f"  Run ID  : {result.get('run_id')}")
        print(f"  Statut  : {result.get('status')}")
        print(f"  Métriques : {result.get('metrics')}")
    else:
        print(f"  Raisons : {result.get('reasons', [])}")
    print()


def cmd_knowledge(args: argparse.Namespace) -> None:
    from pipeline.knowledge import run_knowledge_update
    from pipeline.registry import REGISTRY

    modules = [args.module] if args.module else list(REGISTRY.keys())
    print(f"\n  Mise à jour connaissances — {len(modules)} modules...")
    result = run_knowledge_update(modules=modules)
    print(f"\n  Mises à jour extraites : {result['total_updates']}")
    for mod, count in result.get("by_module", {}).items():
        if count > 0:
            print(f"    {mod:15s} : {count} articles")
    print()


def cmd_safety(args: argparse.Namespace) -> None:
    from pipeline.safety import run_safety_validation
    from pipeline.registry import REGISTRY

    result = run_safety_validation(
        module_key=args.module,
        run_id=args.run_id,
        modules=None if args.module else list(REGISTRY.keys()),
    )

    print(f"\n{'─'*60}")
    print(f"  KANÉA — Rapport Sécurité")
    print(f"{'─'*60}")

    if "safety_level" in result:
        # Rapport module unique
        level = result["safety_level"]
        icons = {"APPROVED": "✅", "CONDITIONAL": "⚠️", "BLOCKED": "🚫"}
        print(f"\n  Module     : {result.get('module')}")
        print(f"  Sécurité   : {icons.get(level,'?')} {level}")
        print(f"  Pass global: {result.get('overall_pass')}")
        print(f"  Métriques  : {result.get('metrics')}")
        if result.get("critical_failures"):
            print(f"  CRITIQUES  : {result['critical_failures']}")
        print(f"\n  Recommandations :")
        for r in result.get("recommendations", []):
            print(f"    → {r}")
    else:
        # Rapport multi-modules
        print(f"\n  Total validés  : {result.get('total', 0)}")
        print(f"  ✅ Approuvés   : {result.get('approved', 0)}")
        print(f"  ⚠️  Conditionnels: {result.get('conditional', 0)}")
        print(f"  🚫 Bloqués     : {result.get('blocked', 0)}")
        mods = result.get("modules", {})
        if mods:
            print(f"\n  Détail :")
            for mod, info in mods.items():
                icons = {"APPROVED": "✅", "CONDITIONAL": "⚠️", "BLOCKED": "🚫"}
                icon = icons.get(info.get("safety_level", ""), "?")
                print(f"    {icon} {mod:18s}  {info.get('safety_level')}")
    print(f"\n{'─'*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline",
        description="KANÉA — Universal Medical Data Acquisition & Auto-Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--module",  "-m", metavar="MODULE",
                   help="Module spécifique (ex: malaria, hemato, breast_cancer)")
    p.add_argument("--output",  "-o", metavar="FILE",
                   help="Fichier de sortie JSON pour le rapport")

    sub = p.add_subparsers(dest="command", help="Commande")

    # run
    run_p = sub.add_parser("run", help="Lancer le pipeline (phases sélectionnées)")
    run_p.add_argument("--phases", nargs="+", type=int, metavar="N",
                       help="Phases à exécuter (1-11)")
    run_p.add_argument("--all",      action="store_true", help="Toutes les phases")
    run_p.add_argument("--dry-run",  action="store_true", help="Sans téléchargement/entraînement")
    run_p.add_argument("--epochs",   type=int, default=30)

    # status
    sub.add_parser("status", help="État du pipeline et du catalogue")

    # discover
    sub.add_parser("discover", help="Phase 2 : Auto-découverte des sources")

    # retrain
    ret_p = sub.add_parser("retrain", help="Phases 8-9 : Réentraînement automatique")
    ret_p.add_argument("--force",      action="store_true", help="Forcer le réentraînement")
    ret_p.add_argument("--check-only", action="store_true", help="Vérifier sans entraîner")
    ret_p.add_argument("--epochs",     type=int, default=30)

    # knowledge
    sub.add_parser("knowledge", help="Phase 10 : Mise à jour connaissances médicales")

    # safety
    saf_p = sub.add_parser("safety", help="Phase 11 : Validation sécurité")
    saf_p.add_argument("--run-id", metavar="RUN_ID", help="Run ID spécifique")

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Crée le dossier logs
    (_ROOT / "logs" / "pipeline").mkdir(parents=True, exist_ok=True)
    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        return

    dispatch = {
        "run":       cmd_run,
        "status":    cmd_status,
        "discover":  cmd_discover,
        "retrain":   cmd_retrain,
        "knowledge": cmd_knowledge,
        "safety":    cmd_safety,
    }

    fn = dispatch.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
