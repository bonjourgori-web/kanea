from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dirs(root: Path) -> None:
    for path in [
        root / "data" / "malaria" / "Parasitised",
        root / "data" / "malaria" / "Uninfected",
        root / "data" / "biometry",
        root / "data" / "forensic",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def generate_forensic_demo(csv_path: Path, rows: int, seed: int) -> None:
    rng = np.random.default_rng(seed)

    sexes = rng.choice(["M", "F"], size=rows)
    ancestry = rng.choice(
        ["West_African", "Central_African", "East_African"],
        size=rows,
        p=[0.4, 0.3, 0.3],
    )

    base_stature = np.where(sexes == "M", 170.0, 159.0)
    stature = np.round(base_stature + rng.normal(0, 6, rows), 1)
    age = np.round(rng.uniform(18, 75, rows), 1)

    dataframe = pd.DataFrame(
        {
            "max_cranial_length_mm": np.round(rng.normal(180, 8, rows), 1),
            "max_cranial_breadth_mm": np.round(rng.normal(139, 6, rows), 1),
            "bizygomatic_breadth_mm": np.round(rng.normal(127, 5, rows), 1),
            "nasal_height_mm": np.round(rng.normal(50, 4, rows), 1),
            "nasal_breadth_mm": np.round(rng.normal(24, 2.5, rows), 1),
            "basion_nasion_length_mm": np.round(rng.normal(97, 5, rows), 1),
            "femur_length_cm": np.round(np.where(sexes == "M", rng.normal(45, 2.8, rows), rng.normal(42, 2.5, rows)), 1),
            "tibia_length_cm": np.round(np.where(sexes == "M", rng.normal(37, 2.2, rows), rng.normal(34.5, 2.0, rows)), 1),
            "humerus_length_cm": np.round(np.where(sexes == "M", rng.normal(32, 1.8, rows), rng.normal(29.8, 1.7, rows)), 1),
            "radius_length_cm": np.round(np.where(sexes == "M", rng.normal(24.5, 1.5, rows), rng.normal(22.8, 1.4, rows)), 1),
            "AIM_PC1": np.round(rng.normal(0, 0.15, rows), 3),
            "AIM_PC2": np.round(rng.normal(0, 0.15, rows), 3),
            "AIM_PC3": np.round(rng.normal(0, 0.15, rows), 3),
            "biological_sex": sexes,
            "age_at_death": age,
            "ancestry": ancestry,
            "stature_cm": stature,
        }
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(csv_path, index=False)
    print(f"Forensic demo dataset written to: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare KANEA dataset folders and optional demo data.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root containing data/ and scripts/.",
    )
    parser.add_argument(
        "--generate-forensic-demo",
        action="store_true",
        help="Generate a simulated forensic CSV for Module 3.",
    )
    parser.add_argument(
        "--forensic-rows",
        type=int,
        default=500,
        help="Number of rows to generate for the forensic demo dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    ensure_dirs(root)
    print(f"Dataset folders prepared under: {root / 'data'}")

    if args.generate_forensic_demo:
        generate_forensic_demo(
            csv_path=root / "data" / "forensic" / "forensic_dataset.csv",
            rows=args.forensic_rows,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
