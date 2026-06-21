"""Shared configuration for the logistics matching prototype."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"
DATABASE_PATH = DATA_DIR / "logistics.db"

SHIPMENTS_PATH = RAW_DATA_DIR / "shipments.csv"
DRIVERS_PATH = RAW_DATA_DIR / "drivers.csv"
MATCHES_PATH = RAW_DATA_DIR / "historical_matches.csv"
PROCESSED_PATH = PROCESSED_DATA_DIR / "training_dataset.csv"
MODEL_PATH = MODELS_DIR / "best_match_model.joblib"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"

RANDOM_SEED = 42

CITY_COORDINATES = {
    "Kuala Lumpur": (3.1390, 101.6869),
    "Shah Alam": (3.0738, 101.5183),
    "Klang": (3.0449, 101.4456),
    "Seremban": (2.7297, 101.9381),
    "Melaka": (2.1896, 102.2501),
    "Ipoh": (4.5975, 101.0901),
    "Penang": (5.4141, 100.3288),
    "Johor Bahru": (1.4927, 103.7414),
    "Kuantan": (3.8077, 103.3260),
    "Kota Bharu": (6.1254, 102.2381),
}

VEHICLE_CAPACITY_RANGES = {
    "Motorcycle": (5, 35),
    "Van": (250, 1200),
    "Small Truck": (1000, 3000),
    "Medium Truck": (3000, 8000),
    "Large Truck": (8000, 18000),
}

ENGINEERED_FEATURES = [
    "distance_score",
    "vehicle_type_match",
    "capacity_match",
    "capacity_utilization",
    "rating_score",
    "availability_flag",
    "price_efficiency",
    "delay_risk_score",
    "cancellation_risk_score",
    "response_time_score",
    "urgency_match_score",
    "historical_success_rate",
    "estimated_margin_score",
    "route_distance_log",
    "completed_jobs_log",
]

# Vehicle and capacity compatibility are hard business rules, not prediction
# tasks. Estimated margin is retained for reporting but excluded from the model
# because it is a near-duplicate of price_efficiency.
MODEL_FEATURES = [
    feature
    for feature in ENGINEERED_FEATURES
    if feature
    not in {
        "vehicle_type_match",
        "capacity_match",
        "estimated_margin_score",
    }
]
