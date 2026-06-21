"""Create reproducible local demo artifacts when a deployment starts empty."""

from __future__ import annotations

from threading import Lock

try:
    from src.config import (
        DATABASE_PATH,
        DRIVERS_PATH,
        MATCHES_PATH,
        MODEL_PATH,
        PROCESSED_PATH,
        SHIPMENTS_PATH,
    )
except ModuleNotFoundError:
    from config import (
        DATABASE_PATH,
        DRIVERS_PATH,
        MATCHES_PATH,
        MODEL_PATH,
        PROCESSED_PATH,
        SHIPMENTS_PATH,
    )

_ARTIFACT_LOCK = Lock()


def _generate_data() -> None:
    try:
        from src.data_generation import main as generate_data
    except ModuleNotFoundError:
        from data_generation import main as generate_data

    generate_data()


def _train_model() -> None:
    try:
        from src.train_model import train_and_save
    except ModuleNotFoundError:
        from train_model import train_and_save

    train_and_save()


def runtime_artifacts_ready() -> bool:
    """Return whether the recommender has all required generated artifacts."""
    return MODEL_PATH.exists() and DRIVERS_PATH.exists()


def ensure_runtime_artifacts() -> bool:
    """Generate missing synthetic data and model artifacts once.

    Returns ``True`` when this call generated or trained an artifact. The
    operation is deterministic because the project uses fixed random seeds.
    """
    if runtime_artifacts_ready():
        return False

    created_artifacts = False
    with _ARTIFACT_LOCK:
        data_paths = (
            SHIPMENTS_PATH,
            DRIVERS_PATH,
            MATCHES_PATH,
            PROCESSED_PATH,
            DATABASE_PATH,
        )
        if not all(path.exists() for path in data_paths):
            _generate_data()
            created_artifacts = True

        if not MODEL_PATH.exists():
            _train_model()
            created_artifacts = True

    return created_artifacts
