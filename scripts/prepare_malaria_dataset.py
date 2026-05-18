from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import pandas as pd


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
CLASSES = ["Parasitised", "Uninfected"]


def collect_images(source_dir: Path) -> dict[str, list[Path]]:
    images_by_class: dict[str, list[Path]] = {}
    for label in CLASSES:
        class_dir = source_dir / label
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        images = [
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        if not images:
            raise ValueError(f"No valid images found in: {class_dir}")
        images_by_class[label] = images
    return images_by_class


def build_split(
    images_by_class: dict[str, list[Path]],
    valid_pct: float,
    seed: int,
) -> tuple[list[tuple[Path, str, str]], list[tuple[Path, str, str]]]:
    train_records: list[tuple[Path, str, str]] = []
    valid_records: list[tuple[Path, str, str]] = []

    rng = random.Random(seed)
    for label, images in images_by_class.items():
        class_images = list(images)
        rng.shuffle(class_images)
        valid_size = max(1, int(len(class_images) * valid_pct))
        valid_set = class_images[:valid_size]
        train_set = class_images[valid_size:]

        train_records.extend((path, label, "train") for path in train_set)
        valid_records.extend((path, label, "valid") for path in valid_set)

    return train_records, valid_records


def copy_split(records: list[tuple[Path, str, str]], destination_dir: Path) -> None:
    for source_path, label, split in records:
        target_dir = destination_dir / split / label
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_dir / source_path.name)


def write_manifest(
    train_records: list[tuple[Path, str, str]],
    valid_records: list[tuple[Path, str, str]],
    destination_dir: Path,
    source_name: str,
) -> None:
    rows = []
    for source_path, label, split in [*train_records, *valid_records]:
        rows.append(
            {
                "relative_path": f"{split}/{label}/{source_path.name}",
                "label": label,
                "split": split,
                "source": source_name,
                "notes": "",
            }
        )
    manifest_path = destination_dir / "dataset_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"Manifest written to: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a real malaria dataset for KANEA training.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing Parasitised/ and Uninfected/ folders.",
    )
    parser.add_argument(
        "--destination-dir",
        type=Path,
        default=Path("data/malaria"),
        help="Target directory for the prepared train/valid structure.",
    )
    parser.add_argument(
        "--valid-pct",
        type=float,
        default=0.1,
        help="Validation fraction per class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split reproducibility.",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default="NIH_Malaria_Dataset",
        help="Dataset source label for the manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    destination_dir = args.destination_dir.resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    images_by_class = collect_images(source_dir)
    train_records, valid_records = build_split(
        images_by_class=images_by_class,
        valid_pct=args.valid_pct,
        seed=args.seed,
    )

    copy_split(train_records, destination_dir)
    copy_split(valid_records, destination_dir)
    write_manifest(train_records, valid_records, destination_dir, args.source_name)

    print("Malaria dataset prepared successfully.")
    for label in CLASSES:
        train_count = sum(1 for _, lbl, split in train_records if lbl == label and split == "train")
        valid_count = sum(1 for _, lbl, split in valid_records if lbl == label and split == "valid")
        print(f"{label}: train={train_count}, valid={valid_count}")


if __name__ == "__main__":
    main()
