"""Train and persist the CivicLens machine-learning models.

This module is intentionally safe to import. Training only happens when
``train_models`` is called or when this file is executed as a script. The API
can therefore import the feature definitions without accidentally retraining
the models or consuming command-line arguments intended for the API process.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_DIR = Path(__file__).resolve().parent

VIOLATION_LABELS = [
    "far_violation",
    "height_violation",
    "setback_violation",
    "coverage_violation",
    "parking_violation",
]

FEATURES_A_NUM = [
    "plot_size_sqm",
    "road_width_m",
    "far_proposed",
    "far_permissible",
    "height_proposed_m",
    "height_permissible_m",
    "setback_front_m",
    "setback_rear_m",
    "setback_side_m",
    "ground_coverage_pct",
    "footprint_sqm",
    "floors",
    "built_up_area_sqm",
    "parking_required_ecs",
    "parking_provided_ecs",
]
FEATURES_A_CAT = ["city", "zone_type"]

FEATURES_B_NUM = [
    "plot_size_sqm",
    "road_width_m",
    "foundation_depth_m",
    "basement_proposed",
    "gis_record_confidence",
    "floors",
    "built_up_area_sqm",
]
FEATURES_B_CAT = ["city", "zone_type"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train smart-city compliance models.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_DIR / "data" / "india_smart_city_compliance_dataset.csv",
        help="Input dataset CSV path.",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=PROJECT_DIR / "artifacts",
        help="Directory for trained models and feature metadata.",
    )
    return parser.parse_args()


def _validate_dataset(df):
    required = set(
        FEATURES_A_NUM
        + FEATURES_A_CAT
        + FEATURES_B_NUM
        + VIOLATION_LABELS
        + ["overall_violation", "utility_conflict"]
    )
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )


def _make_preprocessor(categorical_features):
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            )
        ],
        remainder="passthrough",
    )


def _feature_importances(pipeline, numeric_features, categorical_features, multioutput):
    one_hot = pipeline.named_steps["prep"].named_transformers_["cat"]
    categorical_names = list(one_hot.get_feature_names_out(categorical_features))
    feature_names = categorical_names + numeric_features
    classifier = pipeline.named_steps["clf"]
    if multioutput:
        importances = np.mean(
            [estimator.feature_importances_ for estimator in classifier.estimators_],
            axis=0,
        )
    else:
        importances = classifier.feature_importances_
    return sorted(
        zip(feature_names, importances), key=lambda item: -item[1]
    )[:8]


def train_models(data_path=None, artifacts_dir=None, verbose=True):
    """Train both models, save artifacts, and return models plus metrics."""
    data_path = Path(
        data_path or PROJECT_DIR / "data" / "india_smart_city_compliance_dataset.csv"
    ).resolve()
    artifacts_dir = Path(artifacts_dir or PROJECT_DIR / "artifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    _validate_dataset(df)

    x_a = df[FEATURES_A_NUM + FEATURES_A_CAT]
    y_a = df[VIOLATION_LABELS]
    xa_train, xa_test, ya_train, ya_test = train_test_split(
        x_a,
        y_a,
        test_size=0.2,
        random_state=42,
        stratify=df["overall_violation"],
    )

    model_a = Pipeline(
        [
            ("prep", _make_preprocessor(FEATURES_A_CAT)),
            (
                "clf",
                MultiOutputClassifier(
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=12,
                        random_state=42,
                        n_jobs=-1,
                    )
                ),
            ),
        ]
    )
    model_a.fit(xa_train, ya_train)
    pred_a = model_a.predict(xa_test)

    model_a_metrics = {}
    for index, label in enumerate(VIOLATION_LABELS):
        model_a_metrics[label] = {
            "accuracy": float(accuracy_score(ya_test.iloc[:, index], pred_a[:, index])),
            "f1": float(f1_score(ya_test.iloc[:, index], pred_a[:, index])),
        }

    x_b = df[FEATURES_B_NUM + FEATURES_B_CAT]
    y_b = df["utility_conflict"]
    xb_train, xb_test, yb_train, yb_test = train_test_split(
        x_b,
        y_b,
        test_size=0.2,
        random_state=42,
        stratify=y_b,
    )

    model_b = Pipeline(
        [
            ("prep", _make_preprocessor(FEATURES_B_CAT)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model_b.fit(xb_train, yb_train)
    proba_b = model_b.predict_proba(xb_test)[:, 1]
    pred_b = (proba_b >= 0.5).astype(int)

    model_b_metrics = {
        "accuracy": float(accuracy_score(yb_test, pred_b)),
        "roc_auc": float(roc_auc_score(yb_test, proba_b)),
    }

    top_a = _feature_importances(
        model_a, FEATURES_A_NUM, FEATURES_A_CAT, multioutput=True
    )
    top_b = _feature_importances(
        model_b, FEATURES_B_NUM, FEATURES_B_CAT, multioutput=False
    )

    joblib.dump(model_a, artifacts_dir / "model_a_compliance.joblib")
    joblib.dump(model_b, artifacts_dir / "model_b_utility_conflict.joblib")
    with (artifacts_dir / "feature_importance.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "model_a_compliance": [
                    {"feature": name, "importance": round(float(value), 4)}
                    for name, value in top_a
                ],
                "model_b_utility_conflict": [
                    {"feature": name, "importance": round(float(value), 4)}
                    for name, value in top_b
                ],
            },
            file,
            indent=2,
        )

    metrics = {
        "model_a_compliance": model_a_metrics,
        "model_b_utility_conflict": model_b_metrics,
    }
    if verbose:
        print("=" * 70)
        print("MODEL A: Regulatory Compliance Classifier (multi-label)")
        print("=" * 70)
        for label, values in model_a_metrics.items():
            print(
                f"  {label:22s} accuracy={values['accuracy']:.3f} "
                f"f1={values['f1']:.3f}"
            )

        print()
        print("=" * 70)
        print("MODEL B: Underground Utility Conflict Probability")
        print("=" * 70)
        print(f"  accuracy = {model_b_metrics['accuracy']:.3f}")
        print(f"  roc_auc  = {model_b_metrics['roc_auc']:.3f}")
        print(
            classification_report(
                yb_test, pred_b, target_names=["no_conflict", "conflict"]
            )
        )
        print("\nTop drivers - Model A (compliance):")
        for name, value in top_a:
            print(f"  {name:30s} {value:.3f}")
        print("\nTop drivers - Model B (utility conflict):")
        for name, value in top_b:
            print(f"  {name:30s} {value:.3f}")
        print(f"\nSaved model artifacts to {artifacts_dir}")

    return {
        "model_a": model_a,
        "model_b": model_b,
        "metrics": metrics,
        "artifacts_dir": str(artifacts_dir),
    }


if __name__ == "__main__":
    arguments = parse_args()
    train_models(arguments.data, arguments.artifacts)
