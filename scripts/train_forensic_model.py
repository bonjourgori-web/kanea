from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from modules.bioid_ml.features import AIMS_KEYS, CRANIAL_KEYS, POSTCRANIAL_KEYS

FEATURE_COLUMNS = CRANIAL_KEYS + POSTCRANIAL_KEYS + AIMS_KEYS


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("pca", PCA(n_components=0.95)),
                    ]
                ),
                FEATURE_COLUMNS,
            )
        ]
    )


def build_classifier(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_regressor(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=10,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_model(
    csv_path: Path,
    export_path: Path,
    random_state: int,
) -> None:
    dataframe = pd.read_csv(csv_path)
    required_targets = ["biological_sex", "age_at_death", "ancestry", "stature_cm"]
    missing = [target for target in required_targets if target not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")

    X = dataframe[FEATURE_COLUMNS]

    sex_encoder = LabelEncoder()
    ancestry_encoder = LabelEncoder()
    y_sex = sex_encoder.fit_transform(dataframe["biological_sex"].astype(str))
    y_ancestry = ancestry_encoder.fit_transform(dataframe["ancestry"].astype(str))
    y_age = dataframe["age_at_death"].astype(float)
    y_stature = dataframe["stature_cm"].astype(float)

    X_train, X_test, y_sex_train, y_sex_test = train_test_split(
        X,
        y_sex,
        test_size=0.2,
        random_state=random_state,
        stratify=y_sex,
    )
    _, _, y_ancestry_train, y_ancestry_test = train_test_split(
        X,
        y_ancestry,
        test_size=0.2,
        random_state=random_state,
        stratify=y_ancestry,
    )
    _, _, y_age_train, y_age_test = train_test_split(
        X,
        y_age,
        test_size=0.2,
        random_state=random_state,
    )
    _, _, y_stature_train, y_stature_test = train_test_split(
        X,
        y_stature,
        test_size=0.2,
        random_state=random_state,
    )

    sex_model = build_classifier(random_state=random_state)
    age_model = build_regressor(random_state=random_state)
    ancestry_model = build_classifier(random_state=random_state)
    stature_model = build_regressor(random_state=random_state)

    sex_model.fit(X_train, y_sex_train)
    age_model.fit(X_train, y_age_train)
    ancestry_model.fit(X_train, y_ancestry_train)
    stature_model.fit(X_train, y_stature_train)

    sex_pred = sex_model.predict(X_test)
    age_pred = age_model.predict(X_test)
    ancestry_pred = ancestry_model.predict(X_test)
    stature_pred = stature_model.predict(X_test)

    print("Sex accuracy:", round(accuracy_score(y_sex_test, sex_pred) * 100, 2), "%")
    print("Ancestry accuracy:", round(accuracy_score(y_ancestry_test, ancestry_pred) * 100, 2), "%")
    print("Age MAE:", round(mean_absolute_error(y_age_test, age_pred), 2))
    print("Stature MAE:", round(mean_absolute_error(y_stature_test, stature_pred), 2))

    bundle = {
        "feature_columns": FEATURE_COLUMNS,
        "sex_encoder": sex_encoder,
        "ancestry_encoder": ancestry_encoder,
        "sex_model": sex_model,
        "age_model": age_model,
        "ancestry_model": ancestry_model,
        "stature_model": stature_model,
    }

    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("wb") as fh:
        pickle.dump(bundle, fh)
    print(f"Model exported to: {export_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the KANEA forensic profile models.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("data/forensic/forensic_dataset.csv"),
    )
    parser.add_argument(
        "--export-path",
        type=Path,
        default=Path("models/machine_learning/forensic_model.pkl"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        csv_path=args.csv_path,
        export_path=args.export_path,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
