"""Input validation and normalization for solar-PV anomaly detection."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .config import AnomalyConfig


class SchemaError(ValueError):
    """Raised when telemetry cannot support a reliable anomaly assessment."""


def required_columns(config: AnomalyConfig) -> list[str]:
    """Return the minimum columns required for rules and novelty detection."""
    return [config.timestamp_column, config.power_column, config.irradiance_column]


def available_columns(frame: pd.DataFrame, candidates: Iterable[str | None]) -> list[str]:
    """Return configured candidate columns that are present in a frame."""
    return [column for column in candidates if column is not None and column in frame.columns]


def prepare_telemetry(frame: pd.DataFrame, config: AnomalyConfig) -> pd.DataFrame:
    """Validate, normalize, and chronologically order telemetry without dropping rows.

    Invalid timestamps and nonnumeric values remain in the returned data so they
    can be emitted as data-quality anomalies instead of disappearing silently.
    """
    missing = [column for column in required_columns(config) if column not in frame.columns]
    if missing:
        raise SchemaError(f"Missing required telemetry columns: {', '.join(missing)}")

    prepared = frame.copy()
    prepared[config.timestamp_column] = pd.to_datetime(
        prepared[config.timestamp_column], errors="coerce", utc=True
    )
    numeric_columns = available_columns(
        prepared,
        [
            config.power_column,
            config.irradiance_column,
            config.temperature_column,
            config.wind_speed_column,
            config.solar_zenith_column,
        ],
    )
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    return prepared.sort_values(config.timestamp_column, kind="stable").reset_index(drop=True)
