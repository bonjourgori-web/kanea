from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def classify_status(waz: float, haz: float, whz: float) -> str:
    if min(waz, haz, whz) < -3:
        return "severe_undernutrition"
    if min(waz, haz, whz) < -2:
        return "moderate_undernutrition"
    if whz > 2:
        return "obesity"
    if whz > 1:
        return "overweight"
    return "normal"


def generate_dataset(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age_months = rng.integers(6, 61, size=rows)
    sex = rng.choice(["M", "F"], size=rows)

    height_cm = np.round(60 + age_months * 0.9 + rng.normal(0, 4.5, rows), 1)
    weight_kg = np.round(5 + age_months * 0.22 + rng.normal(0, 1.8, rows), 1)
    muac_cm = np.round(11 + age_months * 0.06 + rng.normal(0, 0.8, rows), 1)

    waz = np.round(rng.normal(-0.4, 1.2, rows), 2)
    haz = np.round(rng.normal(-0.5, 1.1, rows), 2)
    whz = np.round(rng.normal(-0.1, 1.0, rows), 2)

    statuses = [classify_status(a, b, c) for a, b, c in zip(waz, haz, whz)]

    dataframe = pd.DataFrame(
        {
            "age_months": age_months,
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "sex": sex,
            "muac_cm": muac_cm,
            "waz": waz,
            "haz": haz,
            "whz": whz,
            "nutrition_status": statuses,
        }
    )

    return dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a demo biometry dataset for KANEA.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/biometry/biometry_dataset.csv"),
    )
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = generate_dataset(rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(args.output, index=False)
    print(f"Biometry demo dataset written to: {args.output}")


if __name__ == "__main__":
    main()
