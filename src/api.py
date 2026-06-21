"""FastAPI service for driver recommendations."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from src.config import (
        CITY_COORDINATES,
        DRIVERS_PATH,
        MODEL_PATH,
        SHIPMENTS_PATH,
        VEHICLE_CAPACITY_RANGES,
    )
    from src.recommender import DriverRecommender
except ModuleNotFoundError:
    from config import (
        CITY_COORDINATES,
        DRIVERS_PATH,
        MODEL_PATH,
        SHIPMENTS_PATH,
        VEHICLE_CAPACITY_RANGES,
    )
    from recommender import DriverRecommender

app = FastAPI(
    title="Logistics Driver Matching API",
    description="Portfolio prototype for ranking suitable drivers for shipments.",
    version="1.0.0",
)


class ShipmentRequest(BaseModel):
    origin_city: str
    destination_city: str
    distance_km: float | None = Field(default=None, gt=0)
    shipment_weight_kg: float = Field(gt=0)
    required_vehicle_type: str
    delivery_urgency: Literal["standard", "express", "same_day"]
    offered_price: float = Field(gt=0)
    pickup_hour: int = Field(ge=0, le=23)
    shipment_category: str
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("origin_city", "destination_city")
    @classmethod
    def validate_city(cls, value: str) -> str:
        if value not in CITY_COORDINATES:
            raise ValueError(f"Choose one of: {', '.join(CITY_COORDINATES)}")
        return value

    @field_validator("required_vehicle_type")
    @classmethod
    def validate_vehicle_type(cls, value: str) -> str:
        if value not in VEHICLE_CAPACITY_RANGES:
            raise ValueError(
                f"Choose one of: {', '.join(VEHICLE_CAPACITY_RANGES)}"
            )
        return value

    @model_validator(mode="after")
    def validate_route(self) -> "ShipmentRequest":
        if self.origin_city == self.destination_city:
            raise ValueError("Origin and destination must be different")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
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
            }
        }
    }


class RecommendationItem(BaseModel):
    driver_id: str
    predicted_match_score: float = Field(ge=0, le=1)
    vehicle_type: str
    vehicle_capacity_kg: float
    driver_rating: float
    availability_status: str
    current_city: str
    pickup_distance_km: float
    estimated_driver_price: float
    delay_rate: float
    reason: str


class RecommendationResponse(BaseModel):
    model: str
    shipment: dict[str, object]
    recommendation_count: int
    recommendations: list[RecommendationItem]


@lru_cache(maxsize=1)
def get_recommender() -> DriverRecommender:
    return DriverRecommender()


@app.get("/health")
def health() -> dict[str, object]:
    model_ready = MODEL_PATH.exists()
    data_ready = DRIVERS_PATH.exists() and SHIPMENTS_PATH.exists()
    return {
        "status": "ok" if model_ready and data_ready else "degraded",
        "model_ready": model_ready,
        "data_ready": data_ready,
        "prototype": True,
    }


@app.get("/drivers")
def list_drivers(
    limit: int = Query(default=20, ge=1, le=300),
) -> dict[str, object]:
    if not DRIVERS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Driver data is missing. Run python src/data_generation.py.",
        )
    drivers = pd.read_csv(DRIVERS_PATH).head(limit)
    json_safe_drivers = (
        drivers.astype(object)
        .where(pd.notna(drivers), None)
        .to_dict(orient="records")
    )
    return {"count": len(drivers), "drivers": json_safe_drivers}


@app.get("/shipments")
def list_shipments(
    limit: int = Query(default=20, ge=1, le=300),
) -> dict[str, object]:
    if not SHIPMENTS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Shipment data is missing. Run python src/data_generation.py.",
        )
    shipments = pd.read_csv(SHIPMENTS_PATH).head(limit)
    return {
        "count": len(shipments),
        "shipments": shipments.to_dict(orient="records"),
    }


@app.get("/metadata")
def metadata() -> dict[str, object]:
    return {
        "cities": list(CITY_COORDINATES),
        "vehicle_types": list(VEHICLE_CAPACITY_RANGES),
        "urgencies": ["standard", "express", "same_day"],
    }


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: ShipmentRequest) -> RecommendationResponse:
    shipment = request.model_dump(exclude={"top_k"})
    try:
        recommender = get_recommender()
        recommendations = recommender.recommend(
            shipment,
            top_k=request.top_k,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return RecommendationResponse(
        model=recommender.model_name,
        shipment=shipment,
        recommendation_count=len(recommendations),
        recommendations=recommendations,
    )
