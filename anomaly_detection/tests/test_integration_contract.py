"""Contract tests for prediction/anomaly integration."""

from __future__ import annotations

import unittest

import numpy as np

from anomaly_detection.pipeline import STATUS_NOVELTY
from anomaly_detection.tests.test_pipeline import synthetic_telemetry
from integration.prediction_anomaly_integration.integrated_pipeline import (
    fuse_predictions_and_anomalies,
    run_canonical_detection,
)


class IntegrationContractTests(unittest.TestCase):
    def test_canonical_integration_returns_scored_rows_and_events(self) -> None:
        telemetry = synthetic_telemetry(120)
        scored, events = run_canonical_detection(
            telemetry.iloc[:96], telemetry.iloc[96:], prediction_column="Forecast_kW"
        )
        self.assertEqual(len(scored), 24)
        self.assertIn("Anomaly_Status", scored.columns)
        self.assertIn("Event_ID", events.columns)

    def test_legacy_adapter_honors_sklearn_negative_one_anomaly_code(self) -> None:
        telemetry = synthetic_telemetry(24).iloc[12:14].reset_index(drop=True)
        result = fuse_predictions_and_anomalies(
            telemetry,
            predictions=np.array([45.0, 0.0]),
            anomaly_flags=np.array([-1, 1]),
        )
        self.assertEqual(result.loc[0, "Classification"], STATUS_NOVELTY)


if __name__ == "__main__":
    unittest.main()
