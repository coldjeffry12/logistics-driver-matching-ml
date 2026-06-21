"""Classification and ranking evaluation utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    ndcg_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from src.config import (
        METRICS_PATH,
        MODEL_FEATURES,
        MODEL_PATH,
        PROCESSED_PATH,
        REPORTS_DIR,
    )
    from src.feature_engineering import rankable_candidate_mask
    from src.utils import existing_path, setup_logging
except ModuleNotFoundError:
    from config import (
        METRICS_PATH,
        MODEL_FEATURES,
        MODEL_PATH,
        PROCESSED_PATH,
        REPORTS_DIR,
    )
    from feature_engineering import rankable_candidate_mask
    from utils import existing_path, setup_logging

LOGGER = logging.getLogger(__name__)


def calculate_ranking_metrics(
    shipment_ids: pd.Series,
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    k: int = 5,
) -> dict[str, float]:
    """Calculate metrics over each shipment's ranked driver candidates."""
    ranking_frame = pd.DataFrame(
        {
            "shipment_id": shipment_ids.to_numpy(),
            "actual": np.asarray(y_true, dtype=int),
            "score": probabilities,
        }
    )
    precisions: list[float] = []
    ndcgs: list[float] = []
    top_k_successes: list[float] = []

    for _, group in ranking_frame.groupby("shipment_id"):
        ordered = group.sort_values("score", ascending=False)
        top = ordered.head(k)
        precisions.append(float(top["actual"].mean()))
        top_k_successes.append(float(top["actual"].max()))
        if group["actual"].sum() == 0:
            ndcgs.append(0.0)
        else:
            ndcgs.append(
                float(
                    ndcg_score(
                        [group["actual"].to_numpy()],
                        [group["score"].to_numpy()],
                        k=min(k, len(group)),
                    )
                )
            )

    return {
        f"precision_at_{k}": float(np.mean(precisions)),
        f"ndcg_at_{k}": float(np.mean(ndcgs)),
        f"top_{k}_success_rate": float(np.mean(top_k_successes)),
    }


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    shipment_ids: pd.Series,
    k: int = 5,
) -> dict[str, float]:
    """Return standard classification and per-shipment ranking metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }
    metrics.update(
        calculate_ranking_metrics(shipment_ids, y_true, probabilities, k=k)
    )
    return metrics


def save_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Save a readable confusion matrix image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["Not successful", "Successful"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title(title)
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=150)
    plt.close(display.figure_)


def evaluate_saved_model(
    model_path: Path = MODEL_PATH,
    processed_path: Path = PROCESSED_PATH,
) -> dict[str, Any]:
    """Evaluate the saved best model on its held-out shipment IDs."""
    existing_path(model_path, "Saved model")
    existing_path(processed_path, "Processed dataset")
    artifact = joblib.load(model_path)
    data = pd.read_csv(processed_path)
    test_shipments = set(artifact["test_shipment_ids"])
    test_data = data[
        data["shipment_id"].isin(test_shipments)
        & rankable_candidate_mask(data)
    ].copy()
    if test_data.empty:
        raise ValueError("The saved test shipment split is not present in processed data.")

    model = artifact["model"]
    features = artifact.get("features", MODEL_FEATURES)
    probabilities = model.predict_proba(test_data[features])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = calculate_metrics(
        test_data["match_success"],
        predictions,
        probabilities,
        test_data["shipment_id"],
    )
    result: dict[str, Any] = {
        "model_name": artifact["model_name"],
        "test_rows": int(len(test_data)),
        "test_shipments": int(test_data["shipment_id"].nunique()),
        "positive_rate": float(test_data["match_success"].mean()),
        "candidate_policy": artifact.get("candidate_policy", "Not recorded"),
        "metrics": metrics,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORTS_DIR / "best_model_evaluation.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    save_confusion_matrix(
        test_data["match_success"],
        predictions,
        REPORTS_DIR / "best_model_test_confusion_matrix.png",
        f"{artifact['model_name']} held-out test confusion matrix",
    )
    return result


def main() -> None:
    setup_logging()
    result = evaluate_saved_model()
    LOGGER.info("Evaluation complete for %s", result["model_name"])
    print(json.dumps(result, indent=2))
    if not METRICS_PATH.exists():
        LOGGER.warning("Training comparison report not found at %s", METRICS_PATH)


if __name__ == "__main__":
    main()
