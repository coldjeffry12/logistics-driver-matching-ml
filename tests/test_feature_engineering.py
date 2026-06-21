"""Tests for engineered matching features."""

import pandas as pd

from src.config import ENGINEERED_FEATURES, MODEL_FEATURES
from src.data_generation import (
    generate_drivers,
    generate_historical_matches,
    generate_shipments,
)
from src.feature_engineering import engineer_features


def test_feature_engineering_produces_required_features() -> None:
    shipments = generate_shipments(num_shipments=15)
    drivers = generate_drivers(num_drivers=25)
    matches = generate_historical_matches(
        shipments,
        drivers,
        candidates_per_shipment=5,
    )
    joined = (
        matches.merge(shipments, on="shipment_id", how="inner")
        .merge(drivers, on="driver_id", how="inner")
    )
    engineered = engineer_features(joined)

    assert set(ENGINEERED_FEATURES).issubset(engineered.columns)
    assert engineered["distance_score"].between(0, 1).all()
    assert engineered["rating_score"].between(0, 1).all()
    assert set(engineered["vehicle_type_match"].unique()).issubset({0, 1})
    assert not engineered[MODEL_FEATURES].replace(
        [float("inf"), float("-inf")], pd.NA
    ).isna().all(axis=1).any()

    post_outcome_columns = {
        "accepted",
        "completed_successfully",
        "delivery_delay_minutes",
        "customer_rating",
        "final_profit_score",
        "match_success",
    }
    assert post_outcome_columns.isdisjoint(MODEL_FEATURES)
    assert "vehicle_type_match" not in MODEL_FEATURES
    assert "capacity_match" not in MODEL_FEATURES
    assert "estimated_margin_score" not in MODEL_FEATURES


def test_missing_values_are_left_for_training_pipeline_imputation() -> None:
    shipments = generate_shipments(num_shipments=2)
    drivers = generate_drivers(num_drivers=20)
    matches = generate_historical_matches(
        shipments,
        drivers,
        candidates_per_shipment=5,
    )
    joined = (
        matches.merge(shipments, on="shipment_id", how="inner")
        .merge(drivers, on="driver_id", how="inner")
    )
    joined.loc[joined.index[0], "average_response_time_minutes"] = pd.NA
    engineered = engineer_features(joined)

    assert pd.isna(engineered.loc[engineered.index[0], "response_time_score"])
