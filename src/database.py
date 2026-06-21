"""SQLite persistence helpers used to demonstrate SQL/data handling."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

try:
    from src.config import DATABASE_PATH
except ModuleNotFoundError:
    from config import DATABASE_PATH

LOGGER = logging.getLogger(__name__)


def write_dataframes_to_sqlite(
    shipments: pd.DataFrame,
    drivers: pd.DataFrame,
    matches: pd.DataFrame,
    database_path: Path = DATABASE_PATH,
) -> Path:
    """Replace the prototype's SQLite tables with generated data."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        shipments.to_sql("shipments", connection, if_exists="replace", index=False)
        drivers.to_sql("drivers", connection, if_exists="replace", index=False)
        matches.to_sql("historical_matches", connection, if_exists="replace", index=False)
        connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_matches_shipment
                ON historical_matches (shipment_id);
            CREATE INDEX IF NOT EXISTS idx_matches_driver
                ON historical_matches (driver_id);
            CREATE INDEX IF NOT EXISTS idx_drivers_city
                ON drivers (current_city);
            """
        )
    LOGGER.info("Wrote synthetic tables to SQLite: %s", database_path)
    return database_path


def read_training_join(database_path: Path = DATABASE_PATH) -> pd.DataFrame:
    """Read the three-table training join using SQL."""
    query = """
        SELECT
            m.*,
            s.origin_city,
            s.destination_city,
            s.distance_km,
            s.shipment_weight_kg,
            s.required_vehicle_type,
            s.delivery_urgency,
            s.offered_price,
            s.pickup_hour,
            s.shipment_category,
            d.current_city,
            d.vehicle_type,
            d.vehicle_capacity_kg,
            d.driver_rating,
            d.completed_jobs,
            d.cancellation_rate,
            d.delay_rate,
            d.price_per_km,
            d.availability_status,
            d.average_response_time_minutes,
            d.historical_success_rate
        FROM historical_matches AS m
        INNER JOIN shipments AS s USING (shipment_id)
        INNER JOIN drivers AS d USING (driver_id)
    """
    with sqlite3.connect(database_path) as connection:
        return pd.read_sql_query(query, connection)


def table_row_counts(database_path: Path = DATABASE_PATH) -> dict[str, int]:
    """Return row counts for quick validation and documentation."""
    counts: dict[str, int] = {}
    with sqlite3.connect(database_path) as connection:
        for table in ("shipments", "drivers", "historical_matches"):
            result = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            counts[table] = int(result[0])
    return counts

