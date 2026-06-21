"""Tests for ranking metric definitions."""

import numpy as np
import pandas as pd

from src.evaluate_model import calculate_ranking_metrics


def test_ranking_metrics_use_per_shipment_top_k() -> None:
    shipment_ids = pd.Series(["A", "A", "A", "B", "B", "B"])
    actual = np.array([1, 0, 0, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.8, 0.9, 0.1])

    metrics = calculate_ranking_metrics(
        shipment_ids,
        actual,
        scores,
        k=2,
    )

    assert metrics["precision_at_2"] == 0.5
    assert metrics["top_2_success_rate"] == 1.0
    assert metrics["ndcg_at_2"] == 1.0
