"""Integration checks for the generated model artifact and API."""

import joblib
import pytest
from fastapi.testclient import TestClient

from src.api import app, get_recommender
from src.config import MODEL_FEATURES, MODEL_PATH


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Run python src/train_model.py to generate the model artifact.",
)
def test_saved_model_artifact_loads_with_expected_metadata() -> None:
    artifact = joblib.load(MODEL_PATH)

    assert artifact["features"] == MODEL_FEATURES
    assert hasattr(artifact["model"], "predict_proba")
    assert set(artifact["test_shipment_ids"]).isdisjoint(
        artifact["validation_shipment_ids"]
    )


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Run data generation and training before the API integration test.",
)
def test_api_health_and_recommendation() -> None:
    get_recommender.cache_clear()
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = client.post(
        "/recommend",
        json={
            "origin_city": "Kuala Lumpur",
            "destination_city": "Penang",
            "distance_km": 355,
            "shipment_weight_kg": 500,
            "required_vehicle_type": "Van",
            "delivery_urgency": "express",
            "offered_price": 650,
            "pickup_hour": 9,
            "shipment_category": "retail",
            "top_k": 5,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["recommendation_count"] == 5
    assert all(
        0 <= item["predicted_match_score"] <= 1
        for item in payload["recommendations"]
    )

    invalid = client.post(
        "/recommend",
        json={
            "origin_city": "Kuala Lumpur",
            "destination_city": "Penang",
            "shipment_weight_kg": 500,
            "required_vehicle_type": "Bicycle",
            "delivery_urgency": "express",
            "offered_price": 650,
            "pickup_hour": 9,
            "shipment_category": "retail",
        },
    )
    assert invalid.status_code == 422
