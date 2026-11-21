"""
Unit tests for validation strategy and temporal isolation.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.validation_strategy import ValidationManager


def test_chronological_split_integrity():
    """Verify that train, val, and test partitions do not leak future information."""
    dates = pd.date_range("2025-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "value": range(100),
    })

    val_mgr = ValidationManager(timestamp_col="timestamp", train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
    train_df, val_df, test_df, meta = val_mgr.partition_data(df)

    assert len(train_df) == 70
    assert len(val_df) == 10
    assert len(test_df) == 20

    # Ensure strictly monotonic temporal isolation
    train_max_time = pd.to_datetime(train_df["timestamp"]).max()
    val_min_time = pd.to_datetime(val_df["timestamp"]).min()
    val_max_time = pd.to_datetime(val_df["timestamp"]).max()
    test_min_time = pd.to_datetime(test_df["timestamp"]).min()

    assert train_max_time < val_min_time, "Temporal leak detected: Train overlaps with Validation!"
    assert val_max_time < test_min_time, "Temporal leak detected: Validation overlaps with Test!"


if __name__ == "__main__":
    test_chronological_split_integrity()
    print("All ValidationStrategy tests passed successfully!")

