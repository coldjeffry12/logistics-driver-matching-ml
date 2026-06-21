"""Small reusable utilities."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from src.config import (
        DATA_DIR,
        MODELS_DIR,
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
        REPORTS_DIR,
    )
except ModuleNotFoundError:
    from config import DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR


def setup_logging(level: int = logging.INFO) -> None:
    """Configure consistent console logging once."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_project_directories() -> None:
    """Create all generated-data and artifact directories."""
    for path in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Return great-circle distance between two coordinates in kilometres."""
    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def road_distance_km(
    city_a: str,
    city_b: str,
    coordinates: dict[str, tuple[float, float]],
) -> float:
    """Approximate road distance from city coordinates."""
    if city_a == city_b:
        return 12.0
    lat1, lon1 = coordinates[city_a]
    lat2, lon2 = coordinates[city_b]
    return haversine_km(lat1, lon1, lat2, lon2) * 1.18 + 5.0


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable logistic transform."""
    clipped = np.clip(value, -30, 30)
    return 1.0 / (1.0 + np.exp(-clipped))


def require_columns(columns: Iterable[str], available: Iterable[str], label: str) -> None:
    """Raise a useful error when a dataframe schema is incomplete."""
    missing = sorted(set(columns) - set(available))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def existing_path(path: Path, description: str) -> Path:
    """Return an existing path or raise a friendly setup error."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found at {path}. "
            "Run `python src/data_generation.py` first."
        )
    return path

