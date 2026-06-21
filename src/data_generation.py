"""Generate realistic synthetic logistics data for the portfolio prototype."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.config import (
        CITY_COORDINATES,
        DRIVERS_PATH,
        MATCHES_PATH,
        RANDOM_SEED,
        SHIPMENTS_PATH,
        VEHICLE_CAPACITY_RANGES,
    )
    from src.database import write_dataframes_to_sqlite
    from src.utils import ensure_project_directories, road_distance_km, setup_logging, sigmoid
except ModuleNotFoundError:
    from config import (
        CITY_COORDINATES,
        DRIVERS_PATH,
        MATCHES_PATH,
        RANDOM_SEED,
        SHIPMENTS_PATH,
        VEHICLE_CAPACITY_RANGES,
    )
    from database import write_dataframes_to_sqlite
    from utils import ensure_project_directories, road_distance_km, setup_logging, sigmoid

LOGGER = logging.getLogger(__name__)

CITY_NAMES = list(CITY_COORDINATES)
VEHICLE_TYPES = list(VEHICLE_CAPACITY_RANGES)
URGENCIES = ["standard", "express", "same_day"]
SHIPMENT_CATEGORIES = [
    "documents",
    "electronics",
    "food",
    "retail",
    "industrial",
    "furniture",
]


def _required_vehicle(weight_kg: float) -> str:
    if weight_kg <= 30:
        return "Motorcycle"
    if weight_kg <= 900:
        return "Van"
    if weight_kg <= 2500:
        return "Small Truck"
    if weight_kg <= 7000:
        return "Medium Truck"
    return "Large Truck"


def generate_shipments(
    num_shipments: int = 1_000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create synthetic shipment orders with route and business attributes."""
    rng = np.random.default_rng(seed)
    city_probabilities = np.array(
        [0.24, 0.11, 0.10, 0.08, 0.08, 0.08, 0.09, 0.09, 0.07, 0.06]
    )
    rows: list[dict[str, object]] = []

    vehicle_weights = np.array([0.12, 0.32, 0.25, 0.20, 0.11])
    urgency_multipliers = {"standard": 1.0, "express": 1.22, "same_day": 1.42}
    vehicle_rates = {
        "Motorcycle": 0.65,
        "Van": 1.35,
        "Small Truck": 2.15,
        "Medium Truck": 3.15,
        "Large Truck": 4.60,
    }

    for index in range(num_shipments):
        origin = str(rng.choice(CITY_NAMES, p=city_probabilities))
        destinations = [city for city in CITY_NAMES if city != origin]
        destination = str(rng.choice(destinations))
        route_distance = road_distance_km(origin, destination, CITY_COORDINATES)
        route_distance *= float(rng.normal(1.0, 0.035))

        intended_vehicle = str(rng.choice(VEHICLE_TYPES, p=vehicle_weights))
        low, high = VEHICLE_CAPACITY_RANGES[intended_vehicle]
        weight = float(rng.uniform(max(2, low * 0.35), high * 0.82))
        required_vehicle = _required_vehicle(weight)
        urgency = str(rng.choice(URGENCIES, p=[0.58, 0.29, 0.13]))
        category = str(
            rng.choice(
                SHIPMENT_CATEGORIES,
                p=[0.10, 0.18, 0.13, 0.24, 0.20, 0.15],
            )
        )
        base_price = route_distance * vehicle_rates[required_vehicle]
        handling_fee = 8 + np.log1p(weight) * 3.5
        offered_price = (base_price + handling_fee) * urgency_multipliers[urgency]
        offered_price *= float(rng.normal(1.03, 0.10))

        rows.append(
            {
                "shipment_id": f"SHP-{index + 1:05d}",
                "origin_city": origin,
                "destination_city": destination,
                "distance_km": round(max(route_distance, 10), 1),
                "shipment_weight_kg": round(weight, 1),
                "required_vehicle_type": required_vehicle,
                "delivery_urgency": urgency,
                "offered_price": round(max(offered_price, 15), 2),
                "pickup_hour": int(rng.integers(6, 22)),
                "shipment_category": category,
            }
        )
    return pd.DataFrame(rows)


def generate_drivers(
    num_drivers: int = 300,
    seed: int = RANDOM_SEED + 1,
) -> pd.DataFrame:
    """Create a varied driver/carrier population."""
    rng = np.random.default_rng(seed)
    vehicle_probabilities = [0.14, 0.31, 0.24, 0.20, 0.11]
    base_rates = {
        "Motorcycle": 0.52,
        "Van": 1.12,
        "Small Truck": 1.82,
        "Medium Truck": 2.70,
        "Large Truck": 4.00,
    }
    rows: list[dict[str, object]] = []

    for index in range(num_drivers):
        vehicle = str(rng.choice(VEHICLE_TYPES, p=vehicle_probabilities))
        low, high = VEHICLE_CAPACITY_RANGES[vehicle]
        capacity = float(rng.uniform(low, high))
        rating = float(3.0 + rng.beta(5.5, 1.8) * 2.0)
        completed_jobs = int(max(3, rng.lognormal(mean=4.7, sigma=0.75)))
        cancellation_rate = float(np.clip(rng.beta(1.4, 18), 0.003, 0.35))
        delay_rate = float(np.clip(rng.beta(1.8, 11), 0.01, 0.48))
        response_time = float(np.clip(rng.gamma(2.2, 5.5), 1, 70))
        availability = str(
            rng.choice(
                ["available", "busy", "offline"],
                p=[0.69, 0.22, 0.09],
            )
        )
        quality = (
            0.42 * (rating / 5)
            + 0.25 * (1 - delay_rate)
            + 0.18 * (1 - cancellation_rate)
            + 0.15 * min(np.log1p(completed_jobs) / 7, 1)
        )
        historical_success = float(
            np.clip(quality + rng.normal(0, 0.035), 0.42, 0.99)
        )

        rows.append(
            {
                "driver_id": f"DRV-{index + 1:04d}",
                "current_city": str(rng.choice(CITY_NAMES)),
                "vehicle_type": vehicle,
                "vehicle_capacity_kg": round(capacity, 1),
                "driver_rating": round(rating, 2),
                "completed_jobs": completed_jobs,
                "cancellation_rate": round(cancellation_rate, 4),
                "delay_rate": round(delay_rate, 4),
                "price_per_km": round(
                    base_rates[vehicle] * rng.uniform(0.86, 1.22), 2
                ),
                "availability_status": availability,
                "average_response_time_minutes": round(response_time, 1),
                "historical_success_rate": round(historical_success, 4),
            }
        )

    drivers = pd.DataFrame(rows)
    missing_indices = rng.choice(
        drivers.index,
        size=max(1, int(num_drivers * 0.01)),
        replace=False,
    )
    drivers.loc[missing_indices, "average_response_time_minutes"] = np.nan
    return drivers


def generate_historical_matches(
    shipments: pd.DataFrame,
    drivers: pd.DataFrame,
    candidates_per_shipment: int = 12,
    seed: int = RANDOM_SEED + 2,
) -> pd.DataFrame:
    """Generate candidate outcomes with learnable matching patterns."""
    if candidates_per_shipment > len(drivers):
        raise ValueError("candidates_per_shipment cannot exceed the number of drivers")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    urgency_bonus = {"standard": 0.0, "express": 0.18, "same_day": 0.30}

    for shipment in shipments.itertuples(index=False):
        pickup_distances = drivers["current_city"].map(
            lambda city: road_distance_km(
                str(city), shipment.origin_city, CITY_COORDINATES
            )
        )
        vehicle_compatible = (
            drivers["vehicle_type"] == shipment.required_vehicle_type
        )
        capacity_compatible = (
            drivers["vehicle_capacity_kg"] >= shipment.shipment_weight_kg
        )
        rankable_mask = (
            vehicle_compatible
            & capacity_compatible
            & (drivers["availability_status"] != "offline")
        )
        rankable_indices = drivers.index[rankable_mask].to_numpy()
        screened_indices = drivers.index[~rankable_mask].to_numpy()

        def weighted_sample(indices: np.ndarray, size: int) -> list[int]:
            if size <= 0 or len(indices) == 0:
                return []
            weights = (
                np.exp(-pickup_distances.loc[indices].to_numpy() / 280.0)
                + 0.05
            )
            weights /= weights.sum()
            return rng.choice(
                indices,
                size=min(size, len(indices)),
                replace=False,
                p=weights,
            ).tolist()

        # Historical dispatch candidates are mostly rankable, with a few
        # screened-out examples retained to demonstrate the business rules.
        desired_rankable = max(1, candidates_per_shipment - 3)
        candidate_indices = weighted_sample(rankable_indices, desired_rankable)
        remaining = candidates_per_shipment - len(candidate_indices)
        candidate_indices.extend(weighted_sample(screened_indices, remaining))
        remaining = candidates_per_shipment - len(candidate_indices)
        if remaining:
            unused_rankable = np.setdiff1d(
                rankable_indices,
                np.asarray(candidate_indices),
            )
            candidate_indices.extend(weighted_sample(unused_rankable, remaining))

        for driver_index in candidate_indices:
            driver = drivers.loc[driver_index]
            pickup_distance = float(pickup_distances.loc[driver_index])
            distance_score = float(np.exp(-pickup_distance / 220))
            vehicle_match = int(
                driver["vehicle_type"] == shipment.required_vehicle_type
            )
            capacity_match = int(
                driver["vehicle_capacity_kg"] >= shipment.shipment_weight_kg
            )
            availability_flag = int(driver["availability_status"] == "available")
            candidate_eligible = int(
                vehicle_match
                and capacity_match
                and driver["availability_status"] != "offline"
            )
            rating_score = float(driver["driver_rating"] / 5)
            delay_quality = float(1 - driver["delay_rate"])
            cancellation_quality = float(1 - driver["cancellation_rate"])
            response_minutes = float(
                driver["average_response_time_minutes"]
                if pd.notna(driver["average_response_time_minutes"])
                else 15.0
            )
            response_score = float(1 / (1 + response_minutes / 25))

            estimated_price = (
                shipment.distance_km + pickup_distance * 0.22
            ) * float(driver["price_per_km"])
            price_ratio = shipment.offered_price / max(estimated_price, 1)
            price_attractiveness = float(
                np.clip(price_ratio - 0.75, -0.8, 1.2)
            )
            profit_margin = (
                shipment.offered_price - estimated_price
            ) / max(shipment.offered_price, 1)
            capacity_utilization = (
                shipment.shipment_weight_kg
                / max(float(driver["vehicle_capacity_kg"]), 1)
            )
            capacity_fit = float(
                np.exp(-abs(capacity_utilization - 0.65) / 0.35)
            )
            close_pickup = int(pickup_distance <= 80)
            attractive_offer = int(price_ratio >= 0.95)
            urgent_long_pickup = int(
                shipment.delivery_urgency == "same_day"
                and pickup_distance > 120
            )
            busy_urgent = int(
                driver["availability_status"] == "busy"
                and shipment.delivery_urgency != "standard"
            )

            acceptance_logit = (
                -4.00
                + 1.20 * availability_flag
                + 1.00 * distance_score
                + 0.80 * rating_score
                + 0.50 * response_score
                + 0.80 * price_attractiveness
                + 0.50 * capacity_fit
                + 0.60 * attractive_offer
                + 0.45 * close_pickup
                + urgency_bonus[shipment.delivery_urgency]
                - 0.80 * float(driver["delay_rate"])
                - 0.70 * float(driver["cancellation_rate"])
                - 0.90 * busy_urgent
                - 0.80 * urgent_long_pickup
                + float(rng.normal(0, 0.25))
            )
            acceptance_probability = float(sigmoid(acceptance_logit))
            accepted = int(
                candidate_eligible
                and rng.random() < acceptance_probability
            )

            completion_logit = (
                -3.20
                + 1.20 * rating_score
                + 1.60 * delay_quality
                + 0.90 * cancellation_quality
                + 0.80 * float(driver["historical_success_rate"])
                + 0.30 * distance_score
                + 0.70
                * int(
                    driver["driver_rating"] >= 4.5
                    and driver["delay_rate"] <= 0.12
                )
                - (0.60 if shipment.delivery_urgency == "same_day" else 0.0)
                + float(rng.normal(0, 0.25))
            )
            completion_probability = float(sigmoid(completion_logit))
            completed = int(accepted and rng.random() < completion_probability)

            if completed:
                delay_scale = 8 + 80 * float(driver["delay_rate"])
                delivery_delay = max(-15.0, float(rng.normal(delay_scale, 22)))
                customer_rating = float(
                    np.clip(
                        4.75
                        - max(delivery_delay, 0) / 95
                        - float(driver["cancellation_rate"]) * 1.4
                        + rng.normal(0, 0.28),
                        1,
                        5,
                    )
                )
            else:
                delivery_delay = (
                    float(rng.uniform(80, 240)) if accepted else np.nan
                )
                customer_rating = np.nan

            final_profit_score = float(
                np.clip(
                    0.55 * np.clip((profit_margin + 0.5) / 1.5, 0, 1)
                    + 0.45 * (customer_rating / 5 if completed else 0),
                    0,
                    1,
                )
            )
            match_success = int(
                candidate_eligible
                and accepted
                and completed
                and delivery_delay <= 60
                and customer_rating >= 3.5
            )

            rows.append(
                {
                    "shipment_id": shipment.shipment_id,
                    "driver_id": driver["driver_id"],
                    "candidate_distance_km": round(pickup_distance, 1),
                    "estimated_driver_price": round(estimated_price, 2),
                    "candidate_eligible": candidate_eligible,
                    "accepted": accepted,
                    "completed_successfully": completed,
                    "delivery_delay_minutes": (
                        round(delivery_delay, 1)
                        if pd.notna(delivery_delay)
                        else np.nan
                    ),
                    "customer_rating": (
                        round(customer_rating, 2)
                        if pd.notna(customer_rating)
                        else np.nan
                    ),
                    "final_profit_score": round(final_profit_score, 4),
                    "match_success": match_success,
                }
            )
    return pd.DataFrame(rows)


def generate_all_data(
    num_shipments: int = 1_000,
    num_drivers: int = 300,
    candidates_per_shipment: int = 12,
    output_directory: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate all datasets and optionally save them to a custom directory."""
    shipments = generate_shipments(num_shipments)
    drivers = generate_drivers(num_drivers)
    matches = generate_historical_matches(
        shipments,
        drivers,
        candidates_per_shipment=candidates_per_shipment,
    )

    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
        shipments.to_csv(output_directory / "shipments.csv", index=False)
        drivers.to_csv(output_directory / "drivers.csv", index=False)
        matches.to_csv(output_directory / "historical_matches.csv", index=False)
    return shipments, drivers, matches


def main() -> None:
    """Generate CSV data, SQLite tables, and the processed training dataset."""
    setup_logging()
    ensure_project_directories()
    LOGGER.info("Generating synthetic logistics datasets")
    shipments, drivers, matches = generate_all_data()
    shipments.to_csv(SHIPMENTS_PATH, index=False)
    drivers.to_csv(DRIVERS_PATH, index=False)
    matches.to_csv(MATCHES_PATH, index=False)
    LOGGER.info(
        "Saved %d shipments, %d drivers, and %d historical matches",
        len(shipments),
        len(drivers),
        len(matches),
    )
    write_dataframes_to_sqlite(shipments, drivers, matches)

    try:
        from src.feature_engineering import build_processed_dataset
    except ModuleNotFoundError:
        from feature_engineering import build_processed_dataset

    processed = build_processed_dataset()
    LOGGER.info(
        "Data generation complete. Processed dataset has %d rows and %.1f%% positives.",
        len(processed),
        processed["match_success"].mean() * 100,
    )


if __name__ == "__main__":
    main()
