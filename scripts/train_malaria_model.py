from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastai.vision.all import (
    CategoryBlock,
    DataBlock,
    GrandparentSplitter,
    ImageBlock,
    Normalize,
    RandomSplitter,
    Resize,
    ResizeMethod,
    aug_transforms,
    accuracy,
    cnn_learner,
    get_image_files,
    parent_label,
    resnet34,
)


def build_dls(dataset_dir: Path, image_size: int, batch_size: int, valid_pct: float):
    """Create fast.ai dataloaders for the NIH malaria dataset."""
    train_dir = dataset_dir / "train"
    valid_dir = dataset_dir / "valid"

    if train_dir.exists() and valid_dir.exists():
        splitter = GrandparentSplitter(train_name="train", valid_name="valid")
    else:
        splitter = RandomSplitter(valid_pct=valid_pct, seed=42)

    datablock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=splitter,
        get_y=parent_label,
        item_tfms=Resize(image_size, method=ResizeMethod.Squish),
        batch_tfms=[
            *aug_transforms(
                do_flip=True,
                flip_vert=True,
                max_rotate=25.0,
                max_lighting=0.2,
            ),
            Normalize.from_stats(*imagenet_stats()),
        ],
    )
    return datablock.dataloaders(dataset_dir, bs=batch_size)


def imagenet_stats():
    return ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def train_model(
    dataset_dir: Path,
    export_path: Path,
    image_size: int,
    batch_size: int,
    freeze_epochs: int,
    finetune_epochs: int,
    valid_pct: float,
) -> None:
    dls = build_dls(
        dataset_dir=dataset_dir,
        image_size=image_size,
        batch_size=batch_size,
        valid_pct=valid_pct,
    )

    learner = cnn_learner(dls, resnet34, metrics=[accuracy])
    learner.freeze()
    learner.fit_one_cycle(freeze_epochs, 1e-3)
    learner.unfreeze()
    learner.fit_one_cycle(finetune_epochs, slice(1e-5, 1e-4))

    export_path.parent.mkdir(parents=True, exist_ok=True)
    learner.export(export_path)
    print(f"Model exported to: {export_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the KANEA malaria deep learning model with fast.ai.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/malaria"),
        help="Path to the malaria dataset root directory.",
    )
    parser.add_argument(
        "--export-path",
        type=Path,
        default=Path("models/deep_learning/malaria_model.pkl"),
        help="Path where the exported fast.ai learner will be written.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--valid-pct", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        dataset_dir=args.dataset_dir,
        export_path=args.export_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        freeze_epochs=args.freeze_epochs,
        finetune_epochs=args.finetune_epochs,
        valid_pct=args.valid_pct,
    )


if __name__ == "__main__":
    main()
