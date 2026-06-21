"""Train, compare, and save explainable driver matching models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.config import (
        METRICS_PATH,
        MODEL_FEATURES,
        MODEL_PATH,
        PROCESSED_PATH,
        RANDOM_SEED,
        REPORTS_DIR,
    )
    from src.evaluate_model import calculate_metrics, save_confusion_matrix
    from src.feature_engineering import (
        build_processed_dataset,
        rankable_candidate_mask,
    )
    from src.utils import ensure_project_directories, setup_logging
except ModuleNotFoundError:
    from config import (
        METRICS_PATH,
        MODEL_FEATURES,
        MODEL_PATH,
        PROCESSED_PATH,
        RANDOM_SEED,
        REPORTS_DIR,
    )
    from evaluate_model import calculate_metrics, save_confusion_matrix
    from feature_engineering import (
        build_processed_dataset,
        rankable_candidate_mask,
    )
    from utils import ensure_project_directories, setup_logging

LOGGER = logging.getLogger(__name__)


def split_by_shipment(
    data: pd.DataFrame,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = RANDOM_SEED,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[str],
]:
    """Split complete shipment groups into train, validation, and test sets."""
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1")
    shipment_ids = data["shipment_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(shipment_ids)
    test_count = max(1, int(len(shuffled) * test_fraction))
    validation_count = max(1, int(len(shuffled) * validation_fraction))
    test_ids = shuffled[:test_count].tolist()
    validation_ids = shuffled[
        test_count : test_count + validation_count
    ].tolist()
    is_test = data["shipment_id"].isin(test_ids)
    is_validation = data["shipment_id"].isin(validation_ids)
    train_data = data.loc[~is_test & ~is_validation].copy()
    validation_data = data.loc[is_validation].copy()
    test_data = data.loc[is_test].copy()
    return (
        train_data,
        validation_data,
        test_data,
        validation_ids,
        test_ids,
    )


def build_models() -> dict[str, Pipeline]:
    """Return the baseline and stronger non-linear model."""
    baseline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    random_forest = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return {
        "Logistic Regression": baseline,
        "Random Forest": random_forest,
    }


def _save_feature_importance(model: Pipeline, model_name: str) -> None:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importance = np.abs(classifier.coef_[0])
    else:
        return
    report = (
        pd.DataFrame({"feature": MODEL_FEATURES, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    report["model_name"] = model_name
    report.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)


def train_and_save() -> dict[str, Any]:
    """Select on validation data, evaluate once on test data, and save."""
    ensure_project_directories()
    if not PROCESSED_PATH.exists():
        LOGGER.info("Processed dataset not found; building it from SQLite")
        build_processed_dataset()
    full_data = pd.read_csv(PROCESSED_PATH)
    data = full_data.loc[rankable_candidate_mask(full_data)].copy()
    (
        train_data,
        validation_data,
        test_data,
        validation_ids,
        test_ids,
    ) = split_by_shipment(data)
    X_train = train_data[MODEL_FEATURES]
    y_train = train_data["match_success"]
    X_validation = validation_data[MODEL_FEATURES]
    y_validation = validation_data["match_success"]
    X_test = test_data[MODEL_FEATURES]
    y_test = test_data["match_success"]

    LOGGER.info(
        "Rankable rows: %d of %d. Train %d, validation %d, test %d rows",
        len(data),
        len(full_data),
        len(train_data),
        len(validation_data),
        len(test_data),
    )
    validation_comparison: dict[str, dict[str, float]] = {}

    for model_name, model in build_models().items():
        LOGGER.info("Training %s", model_name)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_validation)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = calculate_metrics(
            y_validation,
            predictions,
            probabilities,
            validation_data["shipment_id"],
        )
        validation_comparison[model_name] = metrics
        save_confusion_matrix(
            y_validation,
            predictions,
            REPORTS_DIR
            / (
                f"{model_name.lower().replace(' ', '_')}"
                "_validation_confusion_matrix.png"
            ),
            f"{model_name} validation confusion matrix",
        )
        LOGGER.info(
            "%s validation | ROC-AUC %.3f | F1 %.3f | NDCG@5 %.3f",
            model_name,
            metrics["roc_auc"],
            metrics["f1"],
            metrics["ndcg_at_5"],
        )

    def selection_score(name: str) -> float:
        metrics = validation_comparison[name]
        return 0.70 * metrics["roc_auc"] + 0.30 * metrics["ndcg_at_5"]

    best_name = max(validation_comparison, key=selection_score)
    train_and_validation = pd.concat(
        [train_data, validation_data],
        ignore_index=True,
    )
    best_model = clone(build_models()[best_name])
    best_model.fit(
        train_and_validation[MODEL_FEATURES],
        train_and_validation["match_success"],
    )
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(int)
    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
        test_probabilities,
        test_data["shipment_id"],
    )
    save_confusion_matrix(
        y_test,
        test_predictions,
        REPORTS_DIR / "best_model_test_confusion_matrix.png",
        f"{best_name} held-out test confusion matrix",
    )
    artifact = {
        "model": best_model,
        "model_name": best_name,
        "features": MODEL_FEATURES,
        "validation_shipment_ids": validation_ids,
        "test_shipment_ids": test_ids,
        "selection_rule": "0.70 * ROC-AUC + 0.30 * NDCG@5",
        "candidate_policy": (
            "Exact vehicle type, sufficient capacity, and not offline"
        ),
        "metrics": test_metrics,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    _save_feature_importance(best_model, best_name)

    report: dict[str, Any] = {
        "dataset": {
            "generated_rows": int(len(full_data)),
            "rankable_rows": int(len(data)),
            "screened_out_rows": int(len(full_data) - len(data)),
            "shipments": int(full_data["shipment_id"].nunique()),
            "drivers": int(full_data["driver_id"].nunique()),
            "positive_rate": float(data["match_success"].mean()),
            "train_rows": int(len(train_data)),
            "validation_rows": int(len(validation_data)),
            "test_rows": int(len(test_data)),
        },
        "selection_rule": artifact["selection_rule"],
        "best_model": best_name,
        "validation_models": validation_comparison,
        "held_out_test": test_metrics,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    pd.DataFrame(validation_comparison).T.to_csv(
        REPORTS_DIR / "model_comparison.csv",
        index_label="model",
    )
    LOGGER.info(
        "Saved %s | held-out ROC-AUC %.3f | F1 %.3f | NDCG@5 %.3f",
        best_name,
        test_metrics["roc_auc"],
        test_metrics["f1"],
        test_metrics["ndcg_at_5"],
    )
    return report


def main() -> None:
    setup_logging()
    report = train_and_save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
