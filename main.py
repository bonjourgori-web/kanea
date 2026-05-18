#!/usr/bin/env python3
"""
KANEA AI — Pipeline automatique multimodal (grade médical)
══════════════════════════════════════════════════════════

Pipeline complet :
  1. download_data()   → téléchargement / génération synthétique
  2. validate_data()   → intégrité, biais, balance des classes
  3. preprocess_data() → resize, normalisation, augmentation, DICOM→PNG
  4. train_models()    → early stopping, class weights, dropout
  5. evaluate_models() → Accuracy, Recall, F1, AUC, confusion matrix, ROC

⚠️ Priorité médicale : minimiser les faux négatifs (recall élevé)

Usage :
  python main.py                              # pipeline complet
  python main.py --module malaria             # un seul module
  python main.py --skip-download              # si données déjà présentes
  python main.py --skip-train                 # évaluation seule
  python main.py --epochs 30 --force          # force re-téléchargement
  python main.py --dry-run                    # plan sans exécution

Modules : malaria | nutrition | breast_cancer | all
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import get_logger, KANEA_ROOT

log = get_logger("kanea.pipeline")

BANNER = r"""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║    ██╗  ██╗ █████╗ ███╗   ██╗███████╗ █████╗                 ║
║    ██║ ██╔╝██╔══██╗████╗  ██║██╔════╝██╔══██╗                ║
║    █████╔╝ ███████║██╔██╗ ██║█████╗  ███████║                ║
║    ██╔═██╗ ██╔══██║██║╚██╗██║██╔══╝  ██╔══██║                ║
║    ██║  ██╗██║  ██║██║ ╚████║███████╗██║  ██║                ║
║    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝                ║
║                                                               ║
║    Knowledge Anthropology & Neural Engine for Africa          ║
║    ─── Multimodal Medical AI System ───                       ║
║    ⚠️  Priorité : minimiser les faux négatifs (Recall ↑)      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""


# ─── Arguments CLI ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KANEA AI — Pipeline médical automatique",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--module",
        choices=["malaria", "nutrition", "breast_cancer", "all"],
        default="all",
        help="Module(s) à traiter (défaut : all)",
    )
    p.add_argument("--skip-download",   action="store_true", help="Saute l'étape téléchargement")
    p.add_argument("--skip-validate",   action="store_true", help="Saute la validation des données")
    p.add_argument("--skip-preprocess", action="store_true", help="Saute le prétraitement")
    p.add_argument("--skip-train",      action="store_true", help="Saute l'entraînement")
    p.add_argument("--skip-evaluate",   action="store_true", help="Saute l'évaluation")
    p.add_argument("--force",           action="store_true", help="Force le re-téléchargement")
    p.add_argument("--no-augment",      action="store_true", help="Désactive l'augmentation d'images")
    p.add_argument("--epochs",          type=int, default=20, help="Époques max (défaut : 20)")
    p.add_argument("--patience",        type=int, default=5,  help="Patience early stopping (défaut : 5)")
    p.add_argument("--batch-malaria",   type=int, default=32, help="Batch malaria (défaut : 32)")
    p.add_argument("--batch-cancer",    type=int, default=16, help="Batch cancer (défaut : 16)")
    p.add_argument("--kfolds",          type=int, default=5,  help="K pour la cross-validation nutrition (défaut : 5)")
    p.add_argument("--dry-run",         action="store_true", help="Affiche le plan sans exécuter")
    return p.parse_args()


# ─── Affichage du plan ─────────────────────────────────────────────────────────

def print_plan(args: argparse.Namespace) -> None:
    log.info("─── Plan d'exécution ───")
    steps = [
        ("1. Téléchargement", not args.skip_download),
        ("2. Validation",     not args.skip_validate),
        ("3. Prétraitement",  not args.skip_preprocess),
        ("4. Entraînement",   not args.skip_train),
        ("5. Évaluation",     not args.skip_evaluate),
    ]
    for name, active in steps:
        icon = "✓" if active else "–"
        log.info(f"  {icon} {name}")
    log.info(f"  Module(s)     : {args.module}")
    log.info(f"  Époques max   : {args.epochs}")
    log.info(f"  Early stop    : patience={args.patience}")
    log.info(f"  Augmentation  : {'NON' if args.no_augment else 'OUI'}")
    log.info(f"  CV k-folds    : {args.kfolds}")
    log.info(f"  Modèles  → {KANEA_ROOT / 'models'}")
    log.info(f"  Rapports → {KANEA_ROOT / 'reports'}")
    log.info(f"  Logs     → {KANEA_ROOT / 'logs'}")


# ─── Filtrage par module ────────────────────────────────────────────────────────

def _filter(results: dict | None, module: str) -> dict:
    if not results:
        return {}
    return results if module == "all" else {k: v for k, v in results.items() if k == module}


# ─── Étapes du pipeline ────────────────────────────────────────────────────────

def _step_download(args: argparse.Namespace) -> bool:
    if args.skip_download:
        log.info("ÉTAPE 1/5 — Téléchargement : IGNORÉ")
        return True
    log.info("ÉTAPE 1/5 — Téléchargement des données")
    from src.data_loader import download_data
    results = _filter(download_data(force=args.force), args.module)
    ok = sum(bool(v) for v in results.values())
    log.info(f"  → {ok}/{len(results)} sources acquises")
    return True  # toujours continue (fallback synthétique actif)


def _step_validate(args: argparse.Namespace) -> bool:
    if args.skip_validate:
        log.info("ÉTAPE 2/5 — Validation : IGNORÉ")
        return True
    log.info("ÉTAPE 2/5 — Validation et audit des données")
    from src.validator import validate_data
    report = validate_data() or {}

    issues = []
    for module in ["malaria", "nutrition", "breast_cancer"]:
        if args.module not in ("all", module):
            continue
        r = report.get(module, {})
        if not r.get("ready"):
            issues.append(module)
        for w in r.get("bias_warnings", []):
            log.warning(f"  {w}")
    if issues:
        log.warning(f"Modules non prêts : {issues} — pipeline continue avec données disponibles")
    return True


def _step_preprocess(args: argparse.Namespace) -> bool:
    if args.skip_preprocess:
        log.info("ÉTAPE 3/5 — Prétraitement : IGNORÉ")
        return True
    log.info("ÉTAPE 3/5 — Prétraitement des données")
    from src.preprocessor import preprocess_data
    results = _filter(preprocess_data(augment=not args.no_augment), args.module)
    ok = sum(bool(v) for v in results.values())
    log.info(f"  → {ok}/{len(results)} modules prétraités")
    return ok > 0


def _step_train(args: argparse.Namespace) -> bool:
    if args.skip_train:
        log.info("ÉTAPE 4/5 — Entraînement : IGNORÉ")
        return True
    log.info("ÉTAPE 4/5 — Entraînement des modèles IA")
    from src.trainer import train_models, train_malaria, train_nutrition, train_breast_cancer

    if args.module == "all":
        results = train_models(
            malaria_epochs=args.epochs,
            breast_cancer_epochs=args.epochs,
            malaria_batch=args.batch_malaria,
            breast_batch=args.batch_cancer,
            patience=args.patience,
        ) or {}
    elif args.module == "malaria":
        results = {"malaria": train_malaria(epochs=args.epochs, batch_size=args.batch_malaria, patience=args.patience)}
    elif args.module == "nutrition":
        results = {"nutrition": train_nutrition()}
    elif args.module == "breast_cancer":
        results = {"breast_cancer": train_breast_cancer(epochs=args.epochs, batch_size=args.batch_cancer, patience=args.patience)}
    else:
        results = {}

    ok = sum(bool(v) for v in results.values())
    log.info(f"  → {ok}/{len(results)} modèles entraînés")
    return ok == len(results)


def _step_evaluate(args: argparse.Namespace) -> bool:
    if args.skip_evaluate:
        log.info("ÉTAPE 5/5 — Évaluation : IGNORÉ")
        return True
    log.info("ÉTAPE 5/5 — Évaluation médicale des modèles")
    from src.evaluator import evaluate_models
    results = evaluate_models(k_folds=args.kfolds) or {}

    ok = sum(1 for m in ["malaria", "nutrition", "breast_cancer"] if "error" not in results.get(m, {"error": True}))
    log.info(f"  → {ok}/3 modèles évalués | rapports dans {KANEA_ROOT / 'reports'}")
    return ok > 0


# ─── Pipeline principal ────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> int:
    t0      = time.time()
    all_ok  = True

    steps = [
        ("Téléchargement", _step_download),
        ("Validation",     _step_validate),
        ("Prétraitement",  _step_preprocess),
        ("Entraînement",   _step_train),
        ("Évaluation",     _step_evaluate),
    ]

    for name, fn in steps:
        try:
            ok = fn(args)
            if not ok:
                log.warning(f"Étape '{name}' partiellement échouée — pipeline continue")
                all_ok = False
        except Exception as exc:
            log.error(f"Erreur étape '{name}' : {exc}", exc_info=True)
            all_ok = False

    elapsed = time.time() - t0
    m, s = int(elapsed // 60), int(elapsed % 60)

    log.info("═" * 60)
    status = "✓ SUCCÈS COMPLET" if all_ok else "⚠️  PARTIEL (voir logs)"
    log.info(f"Pipeline terminé en {m}m {s}s — {status}")
    log.info(f"Modèles  : {KANEA_ROOT / 'models'}")
    log.info(f"Rapports : {KANEA_ROOT / 'reports'}")
    log.info(f"Logs     : {KANEA_ROOT / 'logs'}")
    log.info("═" * 60)

    return 0 if all_ok else 1


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(BANNER)
    args = parse_args()
    print_plan(args)

    if args.dry_run:
        log.info("Mode --dry-run : aucune action exécutée.")
        sys.exit(0)

    sys.exit(run_pipeline(args))
