"""Feature engineering for model training and live recommendations."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    from src.config import (
        DATABASE_PATH,
        ENGINEERED_FEATURES,
        MODEL_FEATURES,
        PROCESSED_PATH,
    )
    from src.database import read_training_join
    from src.utils import require_columns
except ModuleNotFoundError:
    from config import (
        DATABASE_PATH,
        ENGINEERED_FEATURES,
        MODEL_FEATURES,
        PROCESSED_PATH,
    )
    from database import read_training_join
    from utils import require_columns

LOGGER = logging.getLogger(__name__)

INPUT_COLUMNS = [
    "candidate_distance_km",
    "estimated_driver_price",
    "distance_km",
    "shipment_weight_kg",
    "required_vehicle_type",
    "delivery_urgency",
    "offered_price",
    "vehicle_type",
    "vehicle_capacity_kg",
    "driver_rating",
    "completed_jobs",
    "cancellation_rate",
    "delay_rate",
    "availability_status",
    "average_response_time_minutes",
    "historical_success_rate",
]


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create explainable numerical matching features from joined candidate data."""
    require_columns(INPUT_COLUMNS, frame.columns, "Candidate dataset")
    result = frame.copy()

    result["distance_score"] = np.exp(
        -result["candidate_distance_km"].clip(lower=0) / 220
    )
    result["vehicle_type_match"] = (
        result["vehicle_type"] == result["required_vehicle_type"]
    ).astype(int)
    result["capacity_match"] = (
        result["vehicle_capacity_kg"] >= result["shipment_weight_kg"]
    ).astype(int)
    result["capacity_utilization"] = (
        result["shipment_weight_kg"]
        / result["vehicle_capacity_kg"].replace(0, np.nan)
    ).clip(0, 1.5)
    result["rating_score"] = (result["driver_rating"] / 5).clip(0, 1)
    result["availability_flag"] = (
        result["availability_status"] == "available"
    ).astype(int)
    result["price_efficiency"] = (
        result["offered_price"]
        / result["estimated_driver_price"].replace(0, np.nan)
    ).clip(0, 3)
    result["delay_risk_score"] = (1 - result["delay_rate"]).clip(0, 1)
    result["cancellation_risk_score"] = (
        1 - result["cancellation_rate"]
    ).clip(0, 1)
    result["response_time_score"] = (
        1
        / (
            1
            + result["average_response_time_minutes"].clip(lower=0)
            / 25
        )
    ).clip(0, 1)

    urgency_factor = result["delivery_urgency"].map(
        {"standard": 0.75, "express": 0.90, "same_day": 1.00}
    ).fillna(0.75)
    result["urgency_match_score"] = (
        result["availability_flag"]
        * result["response_time_score"]
        * urgency_factor
    )
    result["estimated_margin_score"] = (
        (result["offered_price"] - result["estimated_driver_price"])
        / result["offered_price"].replace(0, np.nan)
    ).clip(-1, 1)
    result["route_distance_log"] = np.log1p(
        result["distance_km"].clip(lower=0)
    )
    result["completed_jobs_log"] = np.log1p(
        result["completed_jobs"].clip(lower=0)
    )

    result[ENGINEERED_FEATURES] = result[ENGINEERED_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    return result


def rankable_candidate_mask(frame: pd.DataFrame) -> pd.Series:
    """Return candidates that satisfy the same hard rules used in inference."""
    require_columns(
        ["vehicle_type_match", "capacity_match", "availability_status"],
        frame.columns,
        "Candidate dataset",
    )
    return (
        (frame["vehicle_type_match"] == 1)
        & (frame["capacity_match"] == 1)
        & (frame["availability_status"] != "offline")
    )


def build_processed_dataset() -> pd.DataFrame:
    """Load the SQL join, engineer features, and save the processed CSV."""
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"SQLite database not found at {DATABASE_PATH}. "
            "Run `python src/data_generation.py` first."
        )
    LOGGER.info("Reading joined candidate data through SQLite")
    joined = read_training_join(DATABASE_PATH)
    processed = engineer_features(joined)
    require_columns(["match_success"], processed.columns, "Processed dataset")
    processed["match_success"] = processed["match_success"].astype(int)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_PATH, index=False)
    LOGGER.info("Saved processed training data to %s", PROCESSED_PATH)
    return processed


if __name__ == "__main__":
    built = build_processed_dataset()
    print(built[ENGINEERED_FEATURES + ["match_success"]].describe().round(3))
