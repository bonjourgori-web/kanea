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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

from modules.nutrition_ml.features import build_feature_payload


FEATURE_COLUMNS = [
    "age_months",
    "weight_kg",
    "height_cm",
    "sex",
    "muac_cm",
    "waz",
    "haz",
    "whz",
    "bmi",
]


def prepare_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in dataframe.iterrows():
        rows.append(build_feature_payload(row.to_dict()))
    return pd.DataFrame(rows)


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "age_months",
        "weight_kg",
        "height_cm",
        "muac_cm",
        "waz",
        "haz",
        "whz",
        "bmi",
    ]
    categorical_features = ["sex"]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_models(random_state: int) -> tuple[Pipeline, Pipeline]:
    preprocessor = build_preprocessor()

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
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

    xgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=random_state,
                ),
            ),
        ]
    )

    return rf_pipeline, xgb_pipeline


def majority_vote(rf_pred, xgb_pred):
    return rf_pred


def train_model(
    csv_path: Path,
    target_column: str,
    export_path: Path,
    random_state: int,
) -> None:
    dataframe = pd.read_csv(csv_path)
    if target_column not in dataframe.columns:
        raise ValueError(f"Missing target column: {target_column}")

    features = prepare_frame(dataframe)
    labels = dataframe[target_column].astype(str)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        features[FEATURE_COLUMNS],
        encoded_labels,
        test_size=0.2,
        random_state=random_state,
        stratify=encoded_labels,
    )

    rf_model, xgb_model = build_models(random_state=random_state)
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    final_pred = [majority_vote(rf, xgb) for rf, xgb in zip(rf_pred, xgb_pred)]

    print("Accuracy:", round(accuracy_score(y_test, final_pred) * 100, 2), "%")
    print("Recall:", round(recall_score(y_test, final_pred, average="weighted") * 100, 2), "%")
    print("F1:", round(f1_score(y_test, final_pred, average="weighted") * 100, 2), "%")
    print(classification_report(y_test, final_pred, target_names=label_encoder.classes_))

    bundle = {
        "feature_columns": FEATURE_COLUMNS,
        "label_encoder": label_encoder,
        "random_forest": rf_model,
        "xgboost": xgb_model,
        "target_column": target_column,
    }

    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("wb") as fh:
        pickle.dump(bundle, fh)
    print(f"Model exported to: {export_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the KANEA biometry nutrition ensemble model.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("data/biometry/biometry_dataset.csv"),
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="nutrition_status",
    )
    parser.add_argument(
        "--export-path",
        type=Path,
        default=Path("models/machine_learning/nutrition_model.pkl"),
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        csv_path=args.csv_path,
        target_column=args.target_column,
        export_path=args.export_path,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
