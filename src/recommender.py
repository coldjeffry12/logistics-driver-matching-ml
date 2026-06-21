"""Candidate generation, scoring, ranking, and plain-language explanations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from src.config import (
        CITY_COORDINATES,
        DRIVERS_PATH,
        MODEL_FEATURES,
        MODEL_PATH,
    )
    from src.feature_engineering import engineer_features
    from src.utils import existing_path, road_distance_km
except ModuleNotFoundError:
    from config import CITY_COORDINATES, DRIVERS_PATH, MODEL_FEATURES, MODEL_PATH
    from feature_engineering import engineer_features
    from utils import existing_path, road_distance_km

LOGGER = logging.getLogger(__name__)


class DriverRecommender:
    """Rank feasible drivers for one new shipment."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        drivers_path: Path = DRIVERS_PATH,
        model_artifact: dict[str, Any] | None = None,
        drivers: pd.DataFrame | None = None,
    ) -> None:
        if model_artifact is None:
            existing_path(model_path, "Saved model")
            model_artifact = joblib.load(model_path)
        if drivers is None:
            existing_path(drivers_path, "Driver data")
            drivers = pd.read_csv(drivers_path)
        self.model = model_artifact["model"]
        self.features = model_artifact.get("features", MODEL_FEATURES)
        self.model_name = model_artifact.get("model_name", "Matching model")
        self.drivers = drivers.copy()

    @staticmethod
    def _validate_shipment(shipment: dict[str, Any]) -> dict[str, Any]:
        required = {
            "origin_city",
            "destination_city",
            "shipment_weight_kg",
            "required_vehicle_type",
            "delivery_urgency",
            "offered_price",
            "pickup_hour",
            "shipment_category",
        }
        missing = sorted(required - set(shipment))
        if missing:
            raise ValueError(f"Shipment input is missing: {missing}")
        for city_field in ("origin_city", "destination_city"):
            if shipment[city_field] not in CITY_COORDINATES:
                raise ValueError(
                    f"Unknown {city_field}: {shipment[city_field]}. "
                    f"Choose one of {sorted(CITY_COORDINATES)}."
                )
        prepared = dict(shipment)
        if not prepared.get("distance_km"):
            prepared["distance_km"] = round(
                road_distance_km(
                    prepared["origin_city"],
                    prepared["destination_city"],
                    CITY_COORDINATES,
                ),
                1,
            )
        prepared.setdefault("shipment_id", "NEW-SHIPMENT")
        return prepared

    def _build_candidates(self, shipment: dict[str, Any]) -> pd.DataFrame:
        prepared = self._validate_shipment(shipment)
        candidates = self.drivers.copy()
        for key, value in prepared.items():
            candidates[key] = value
        candidates["candidate_distance_km"] = candidates["current_city"].map(
            lambda city: road_distance_km(
                str(city), prepared["origin_city"], CITY_COORDINATES
            )
        )
        candidates["estimated_driver_price"] = (
            candidates["distance_km"]
            + candidates["candidate_distance_km"] * 0.22
        ) * candidates["price_per_km"]
        return engineer_features(candidates)

    @staticmethod
    def _explain(row: pd.Series) -> str:
        reasons: list[str] = []
        if row["vehicle_type_match"]:
            reasons.append("vehicle type matches")
        if row["capacity_match"]:
            reasons.append("capacity is sufficient")
        if row["driver_rating"] >= 4.5:
            reasons.append("driver rating is high")
        if row["candidate_distance_km"] <= 30:
            reasons.append("driver is close to pickup")
        if row["delay_rate"] <= 0.12:
            reasons.append("historical delay risk is low")
        if row["price_efficiency"] >= 0.9:
            reasons.append("estimated price fits the offer")
        if row["availability_status"] == "available":
            reasons.append("driver is currently available")
        if not reasons:
            reasons.append("overall model score is competitive")
        return "Recommended because " + ", ".join(reasons[:5]) + "."

    def recommend(
        self,
        shipment: dict[str, Any],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return top-k feasible candidates with model/business scores."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        candidates = self._build_candidates(shipment)
        feasible = candidates[
            (candidates["capacity_match"] == 1)
            & (candidates["vehicle_type_match"] == 1)
            & (candidates["availability_status"] != "offline")
        ].copy()
        if feasible.empty:
            LOGGER.warning(
                "No non-offline driver has the required vehicle and capacity"
            )
            return []

        probabilities = self.model.predict_proba(feasible[self.features])[:, 1]
        feasible["predicted_match_score"] = np.clip(
            probabilities,
            0,
            1,
        )
        ranked = feasible.sort_values(
            ["predicted_match_score", "driver_rating"],
            ascending=[False, False],
        ).head(top_k)

        results: list[dict[str, Any]] = []
        for _, row in ranked.iterrows():
            results.append(
                {
                    "driver_id": str(row["driver_id"]),
                    "predicted_match_score": round(
                        float(row["predicted_match_score"]), 4
                    ),
                    "vehicle_type": str(row["vehicle_type"]),
                    "vehicle_capacity_kg": round(
                        float(row["vehicle_capacity_kg"]), 1
                    ),
                    "driver_rating": round(float(row["driver_rating"]), 2),
                    "availability_status": str(row["availability_status"]),
                    "current_city": str(row["current_city"]),
                    "pickup_distance_km": round(
                        float(row["candidate_distance_km"]), 1
                    ),
                    "estimated_driver_price": round(
                        float(row["estimated_driver_price"]), 2
                    ),
                    "delay_rate": round(float(row["delay_rate"]), 4),
                    "reason": self._explain(row),
                }
            )
        return results


def recommend_drivers(
    shipment: dict[str, Any],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Convenience function for scripts and notebooks."""
    return DriverRecommender().recommend(shipment, top_k=top_k)
