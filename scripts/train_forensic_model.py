"""
train_forensic_model.py — Entraînement du module BioID AI (Module 3 KANÉA).

Ce script est un wrapper qui délègue à bioid_ai/training/train.py.
Il garde l'ancienne interface CLI (--csv-path, --export-path) pour la
compatibilité avec les scripts CI/CD existants.

Usage :
  python scripts/train_forensic_model.py           # données synthétiques FORDISC
  python scripts/train_forensic_model.py --csv-path data/forensic/forensic_dataset.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraîne le module BioID AI (forensic profile estimation).",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Chemin vers un CSV avec les colonnes FORDISC (optionnel — synthétique si absent)",
    )
    parser.add_argument(
        "--export-path",
        type=Path,
        default=Path("models/machine_learning/bioid_bundle.pkl"),
        help="Chemin de sortie du bundle (défaut : models/machine_learning/bioid_bundle.pkl)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-samples",    type=int, default=500,
                        help="Nombre de samples synthétiques si pas de CSV (défaut : 500)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Essai via import direct
    try:
        from bioid_ai.training.train import (
            generate_synthetic_data, train_models, save_bundle,
        )
        import pandas as pd
        from pathlib import Path as _Path

        if args.csv_path and _Path(args.csv_path).exists():
            df = pd.read_csv(args.csv_path)
            print(f"[BioID AI] {len(df)} échantillons chargés depuis {args.csv_path}")
        else:
            df = generate_synthetic_data(n_samples=args.n_samples, seed=args.random_state)
            print(f"[BioID AI] {len(df)} échantillons synthétiques générés")

        bundle = train_models(df)
        save_bundle(bundle, _Path(args.export_path))
        print(f"[BioID AI] Bundle sauvegardé : {args.export_path}")
        return
    except ImportError:
        pass

    # Fallback : subprocess sur le script d'entraînement BioID
    train_script = PROJECT_ROOT / "bioid_ai" / "training" / "train.py"
    if not train_script.exists():
        print(f"[ERREUR] Script d'entraînement introuvable : {train_script}")
        sys.exit(1)

    cmd = [sys.executable, str(train_script)]
    if args.csv_path:
        cmd += ["--csv-path", str(args.csv_path)]
    cmd += ["--export-path", str(args.export_path)]
    cmd += ["--random-state", str(args.random_state)]
    cmd += ["--n-samples", str(args.n_samples)]

    print(f"[BioID AI] Lancement : {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
