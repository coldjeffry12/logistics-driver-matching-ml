"""Tests for ranking output and score bounds."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.config import MODEL_FEATURES
from src.data_generation import generate_drivers
from src.recommender import DriverRecommender


def test_recommender_returns_ranked_bounded_scores() -> None:
    dummy_model = DummyClassifier(strategy="prior")
    training_features = pd.DataFrame(
        np.zeros((4, len(MODEL_FEATURES))),
        columns=MODEL_FEATURES,
    )
    dummy_model.fit(training_features, [0, 1, 0, 1])
    artifact = {
        "model": dummy_model,
        "model_name": "Test model",
        "features": MODEL_FEATURES,
    }
    drivers = generate_drivers(num_drivers=80)
    recommender = DriverRecommender(
        model_artifact=artifact,
        drivers=drivers,
    )
    shipment = {
        "origin_city": "Kuala Lumpur",
        "destination_city": "Penang",
        "distance_km": 355.0,
        "shipment_weight_kg": 200.0,
        "required_vehicle_type": "Van",
        "delivery_urgency": "express",
        "offered_price": 620.0,
        "pickup_hour": 9,
        "shipment_category": "retail",
    }

    results = recommender.recommend(shipment, top_k=5)
    scores = [result["predicted_match_score"] for result in results]

    assert len(results) == 5
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= score <= 1 for score in scores)
    assert all("Recommended because" in result["reason"] for result in results)
    assert all(result["vehicle_type"] == "Van" for result in results)
    assert all(
        result["vehicle_capacity_kg"] >= shipment["shipment_weight_kg"]
        for result in results
    )
    assert all(
        result["availability_status"] != "offline"
        for result in results
    )


def test_recommender_returns_empty_for_impossible_match() -> None:
    dummy_model = DummyClassifier(strategy="prior")
    training_features = pd.DataFrame(
        np.zeros((4, len(MODEL_FEATURES))),
        columns=MODEL_FEATURES,
    )
    dummy_model.fit(training_features, [0, 1, 0, 1])
    artifact = {
        "model": dummy_model,
        "model_name": "Test model",
        "features": MODEL_FEATURES,
    }
    drivers = generate_drivers(num_drivers=20)
    drivers["vehicle_type"] = "Motorcycle"
    drivers["vehicle_capacity_kg"] = 20.0
    recommender = DriverRecommender(
        model_artifact=artifact,
        drivers=drivers,
    )
    shipment = {
        "origin_city": "Kuala Lumpur",
        "destination_city": "Penang",
        "distance_km": 355.0,
        "shipment_weight_kg": 2_000.0,
        "required_vehicle_type": "Small Truck",
        "delivery_urgency": "standard",
        "offered_price": 1_000.0,
        "pickup_hour": 10,
        "shipment_category": "industrial",
    }

    assert recommender.recommend(shipment, top_k=5) == []
