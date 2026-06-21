"""Tests for synthetic dataset schemas and relationships."""

from src.data_generation import (
    generate_drivers,
    generate_historical_matches,
    generate_shipments,
)


def test_synthetic_data_generation_produces_expected_columns() -> None:
    shipments = generate_shipments(num_shipments=80)
    drivers = generate_drivers(num_drivers=100)
    matches = generate_historical_matches(
        shipments,
        drivers,
        candidates_per_shipment=10,
    )

    assert {
        "shipment_id",
        "origin_city",
        "destination_city",
        "distance_km",
        "shipment_weight_kg",
        "required_vehicle_type",
        "delivery_urgency",
        "offered_price",
    }.issubset(shipments.columns)
    assert {
        "driver_id",
        "current_city",
        "vehicle_type",
        "vehicle_capacity_kg",
        "driver_rating",
        "delay_rate",
        "availability_status",
    }.issubset(drivers.columns)
    assert {
        "shipment_id",
        "driver_id",
        "candidate_eligible",
        "accepted",
        "completed_successfully",
        "delivery_delay_minutes",
        "customer_rating",
        "final_profit_score",
        "match_success",
    }.issubset(matches.columns)
    assert len(matches) == 80 * 10
    assert set(matches["match_success"].unique()).issubset({0, 1})

    joined = (
        matches.merge(shipments, on="shipment_id", how="inner")
        .merge(drivers, on="driver_id", how="inner")
    )
    successful = joined[joined["match_success"] == 1]
    assert not successful.empty
    assert successful["candidate_eligible"].eq(1).all()
    assert successful["accepted"].eq(1).all()
    assert successful["completed_successfully"].eq(1).all()
    assert (
        successful["vehicle_type"]
        == successful["required_vehicle_type"]
    ).all()
    assert (
        successful["vehicle_capacity_kg"]
        >= successful["shipment_weight_kg"]
    ).all()
    assert successful["delivery_delay_minutes"].le(60).all()
    assert successful["customer_rating"].ge(3.5).all()
    assert matches.loc[
        matches["candidate_eligible"] == 0,
        "match_success",
    ].eq(0).all()
