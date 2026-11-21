"""Configuration for the canonical solar-PV anomaly pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnomalyConfig:
    """Column mapping and conservative, site-adjustable operating thresholds."""

    timestamp_column: str = "Timestamp"
    power_column: str = "AC_Output"
    irradiance_column: str = "GHI"
    temperature_column: str = "Temperature"
    wind_speed_column: str = "Wind_Speed"
    solar_zenith_column: str | None = "SZ_Angle"

    daylight_irradiance_wm2: float = 50.0
    maximum_night_irradiance_wm2: float = 10.0
    daytime_zero_irradiance_wm2: float = 200.0
    daytime_zero_power_kw: float = 0.1
    maximum_irradiance_wm2: float = 1361.0
    minimum_temperature_c: float = -10.0
    maximum_temperature_c: float = 55.0
    maximum_wind_speed: float = 40.0
    maximum_solar_zenith_deg: float = 85.0
    use_solar_zenith_gate: bool = False
    nighttime_generation_kw: float = 2.0

    residual_mad_multiplier: float = 3.0
    residual_prediction_bins: int = 1
    residual_min_bin_samples: int = 20
    isolation_forest_min_contamination: float = 0.01
    isolation_forest_max_contamination: float = 0.05
    isolation_forest_estimators: int = 200
    random_state: int = 42
    event_gap_hours: float = 2.0
