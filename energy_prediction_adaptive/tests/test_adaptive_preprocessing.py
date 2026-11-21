"""
Unit tests for AdaptivePreprocessor: tolerance to missing and extra columns during inference.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.adaptive_preprocessor import AdaptivePreprocessor


def test_missing_features_imputation_at_inference():
    """Verify that if an expected column is missing at inference time, it imputes default value instead of crashing."""
    train_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=20, freq="h"),
        "temp": np.array([20.0] * 20),
        "humidity": np.array([50.0] * 20),
        "inverter_type": ["TypeA"] * 10 + ["TypeB"] * 10,
    })

    preprocessor = AdaptivePreprocessor(
        numerical_features=["temp", "humidity"],
        categorical_features=["inverter_type"],
        timestamp_col="timestamp"
    )
    preprocessor.fit(train_df)

    # Inference data missing 'humidity' column entirely!
    infer_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-02-01", periods=5, freq="h"),
        "temp": [25.0, 26.0, 27.0, 28.0, 29.0],
        # 'humidity' is absent!
        "inverter_type": ["TypeA", "TypeA", "TypeB", "TypeB", "TypeA"],
    })

    # Should not raise KeyError!
    X_trans = preprocessor.transform(infer_df)
    assert X_trans.shape[0] == 5
    assert np.all(np.isfinite(X_trans))


def test_extra_columns_tolerance():
    """Verify that unexpected extra columns in inference data are ignored gracefully."""
    train_df = pd.DataFrame({
        "temp": [15.0, 20.0, 25.0, 30.0],
        "ghi": [100.0, 200.0, 300.0, 400.0],
    })

    preprocessor = AdaptivePreprocessor(
        numerical_features=["temp", "ghi"],
        categorical_features=[],
        timestamp_col=None
    )
    preprocessor.fit(train_df)

    # Inference data with extra garbage columns
    infer_df = pd.DataFrame({
        "temp": [18.0, 22.0],
        "ghi": [150.0, 250.0],
        "extra_sensor_battery_voltage": [12.4, 12.3],
        "operator_notes": ["OK", "Warning"],
    })

    X_trans = preprocessor.transform(infer_df)
    assert X_trans.shape == (2, 2)
    assert np.all(np.isfinite(X_trans))


if __name__ == "__main__":
    test_missing_features_imputation_at_inference()
    test_extra_columns_tolerance()
    print("All AdaptivePreprocessor tests passed successfully!")

