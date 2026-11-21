"""
Unit tests for SchemaProfiler: target resolution, leakage detection, frequency detection.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.schema_profiler import SchemaProfiler


def test_target_ambiguity_handling():
    """Verify that multiple candidate targets raise an error unless explicitly specified."""
    df = pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=10, freq="h"),
        "capacity_factor": np.random.rand(10),
        "active_power_kw": np.random.rand(10) * 50.0,
        "temp_c": np.random.rand(10) * 30.0,
    })

    profiler = SchemaProfiler()  # No target specified
    try:
        profiler.resolve_target(df)
        assert False, "Should have raised ValueError on ambiguous targets!"
    except ValueError as e:
        assert "Multiple target candidates detected" in str(e)

    # When target is explicitly specified
    profiler_explicit = SchemaProfiler(target_col="active_power_kw")
    resolved = profiler_explicit.resolve_target(df)
    assert resolved == "active_power_kw"


def test_leakage_detection():
    """Verify that heuristic keywords and high correlation trigger leakage detection."""
    np.random.seed(42)
    target = np.random.rand(100)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=100, freq="h"),
        "target": target,
        "temp": np.random.rand(100),
        "daily_energy_yield": np.random.rand(100),            # Keyword 'yield'
        "dc_power_input": np.random.rand(100),                # Keyword 'dc_power'
        "leaky_clone": target * 0.999 + np.random.normal(0, 0.001, 100),  # >0.98 corr
    })

    profiler = SchemaProfiler(target_col="target", timestamp_col="timestamp")
    leakage = profiler.detect_leakage_columns(df, target_col="target", timestamp_col="timestamp")

    assert "daily_energy_yield" in leakage
    assert "dc_power_input" in leakage
    assert "leaky_clone" in leakage
    assert "temp" not in leakage


def test_timestamp_frequency_inference():
    """Verify detection of hourly and 15-minute grids."""
    profiler = SchemaProfiler()

    # Hourly series
    hourly_ts = pd.Series(pd.date_range("2025-01-01", periods=50, freq="h"))
    info_hourly = profiler.detect_timestamp_frequency(hourly_ts)
    assert info_hourly["inferred_freq"] == "hourly"
    assert info_hourly["is_regular"] is True

    # 15-minute series
    m15_ts = pd.Series(pd.date_range("2025-01-01", periods=50, freq="15min"))
    info_15m = profiler.detect_timestamp_frequency(m15_ts)
    assert info_15m["inferred_freq"] == "15-minute"


if __name__ == "__main__":
    test_target_ambiguity_handling()
    test_leakage_detection()
    test_timestamp_frequency_inference()
    print("All SchemaProfiler tests passed successfully!")

