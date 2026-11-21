"""Dependency-free unit tests for the canonical anomaly pipeline."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from anomaly_detection import AnomalyPipeline
from anomaly_detection.pipeline import (
    STATUS_DATA_QUALITY,
    STATUS_NIGHT,
    STATUS_OPERATIONAL,
    STATUS_RESIDUAL,
)


def synthetic_telemetry(hours: int = 96) -> pd.DataFrame:
    timestamp = pd.date_range("2026-01-01", periods=hours, freq="h", tz="UTC")
    hour = timestamp.hour
    irradiance = np.where((hour >= 7) & (hour <= 17), 400.0 + 20.0 * (hour - 7), 0.0)
    power = np.where(irradiance > 0, 45.0, 0.0)
    return pd.DataFrame({
        "Timestamp": timestamp,
        "GHI": irradiance,
        "AC_Output": power,
        "Temperature": 25.0,
        "Wind_Speed": 3.0,
        "SZ_Angle": np.where(irradiance > 0, 45.0, 95.0),
        "Forecast_kW": power,
    })


class AnomalyPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = synthetic_telemetry()
        self.pipeline = AnomalyPipeline().fit(self.reference, prediction_column="Forecast_kW")

    def test_nighttime_rows_are_not_operational_anomalies(self) -> None:
        nighttime = self.reference[self.reference["GHI"] == 0].copy()
        scored = self.pipeline.score(nighttime, prediction_column="Forecast_kW")
        self.assertTrue((scored["Anomaly_Status"] == STATUS_NIGHT).all())

    def test_daylight_zero_generation_is_operational_candidate(self) -> None:
        frame = self.reference.iloc[12:36].copy()
        row = frame.index[0]
        frame.loc[row, "AC_Output"] = 0.0
        scored = self.pipeline.score(frame, prediction_column="Forecast_kW")
        self.assertIn(STATUS_OPERATIONAL, scored["Anomaly_Status"].values)

    def test_negative_power_is_data_quality_anomaly(self) -> None:
        frame = self.reference.iloc[12:36].copy()
        row = frame.index[0]
        frame.loc[row, "AC_Output"] = -1.0
        scored = self.pipeline.score(frame, prediction_column="Forecast_kW")
        self.assertIn(STATUS_DATA_QUALITY, scored["Anomaly_Status"].values)

    def test_daylight_deficit_uses_residual_category(self) -> None:
        frame = self.reference.iloc[12:36].copy()
        row = frame.index[0]
        frame.loc[row, "AC_Output"] = 10.0
        scored = self.pipeline.score(frame, prediction_column="Forecast_kW")
        self.assertIn(STATUS_RESIDUAL, scored["Anomaly_Status"].values)

    def test_contiguous_events_are_aggregated(self) -> None:
        frame = self.reference.iloc[12:36].copy()
        frame.loc[frame.index[:2], "AC_Output"] = 0.0
        scored = self.pipeline.score(frame, prediction_column="Forecast_kW")
        events = self.pipeline.aggregate_events(scored)
        operational = events[events["Anomaly_Status"] == STATUS_OPERATIONAL]
        self.assertEqual(len(operational), 1)
        self.assertEqual(int(operational.iloc[0]["Observation_Count"]), 2)


if __name__ == "__main__":
    unittest.main()
