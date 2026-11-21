"""Bridge forecasts with the canonical anomaly pipeline and event output."""

import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anomaly_detection import AnomalyConfig, AnomalyPipeline


def run_canonical_detection(
    reference_df: pd.DataFrame,
    telemetry_df: pd.DataFrame,
    prediction_column: str | None = None,
    config: AnomalyConfig | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on chronological reference telemetry and score a separate holdout frame.

    The returned row-level results use named statuses rather than raw Isolation
    Forest codes. The second return value is an event table suitable for
    maintenance triage.
    """
    pipeline = AnomalyPipeline(config).fit(
        reference_df, prediction_column=prediction_column
    )
    scored = pipeline.score(telemetry_df, prediction_column=prediction_column)
    return scored, pipeline.aggregate_events(scored)

def fuse_predictions_and_anomalies(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    anomaly_flags: np.ndarray,
) -> pd.DataFrame:
    """Legacy adapter for externally supplied Isolation Forest flags.

    Scikit-learn Isolation Forest uses ``-1`` for an anomaly and ``1`` for a
    normal observation. This adapter preserves that convention explicitly and
    labels those rows as novelty only; it does not infer an operational fault.
    New callers should use :func:`run_canonical_detection`.
    """
    if len(test_df) != len(predictions) or len(test_df) != len(anomaly_flags):
        raise ValueError("test_df, predictions, and anomaly_flags must have matching lengths.")
    df = test_df.copy()
    df["Predicted_kW"] = predictions
    df["IsolationForest_Flag"] = np.asarray(anomaly_flags) == -1
    
    if "AC_Output" in df.columns:
        df["Actual_kW"] = df["AC_Output"]
        df["Residual_kW"] = df["Actual_kW"] - df["Predicted_kW"]
        df["Power_Deficit_kW"] = np.maximum(0, df["Predicted_kW"] - df["Actual_kW"])
        
        # Categorize observations
        conditions = [
            (df["GHI"] <= 50.0),
            df["IsolationForest_Flag"],
        ]
        choices = [
            "NIGHTTIME_INACTIVE",
            "ENVIRONMENTAL_NOVELTY",
        ]
        df["Classification"] = np.select(conditions, choices, default="DAYLIGHT_NORMAL")
        
    return df

if __name__ == "__main__":
    print("SURYOKTI Prediction-Anomaly Integration Pipeline loaded successfully.")

