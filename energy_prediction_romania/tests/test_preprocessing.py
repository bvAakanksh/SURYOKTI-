"""
Unit tests for data preprocessing and feature engineering integrity.
Verifies strict target leakage prevention and scaling correctness.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import (
    FEATURE_COLS,
    FORBIDDEN_LEAKAGE_COLS,
    PRIMARY_TARGET,
    SECONDARY_TARGET,
)
from src.preprocessing import PreprocessingPipeline


def test_target_leakage_prevention():
    """Verify that neither targets nor telemetry columns are present in features."""
    for col in FORBIDDEN_LEAKAGE_COLS:
        assert col not in FEATURE_COLS, f"Forbidden leakage column '{col}' found in FEATURE_COLS!"

    assert PRIMARY_TARGET not in FEATURE_COLS, "Primary target 'capacity_factor' leaked into features!"
    assert SECONDARY_TARGET not in FEATURE_COLS, "Secondary target 'active_power_kw' leaked into features!"


def test_feature_count():
    """Verify feature dimension matches 10 weather + 6 cyclical = 16."""
    assert len(FEATURE_COLS) == 16, f"Expected 16 feature columns, got {len(FEATURE_COLS)}"


def test_preprocessing_pipeline_scaling():
    """Verify that StandardScaler fits properly and handles test data."""
    # Create dummy dataframe with required features
    data = {col: np.random.randn(50) for col in FEATURE_COLS}
    df_train = pd.DataFrame(data)
    df_test = pd.DataFrame(data)

    pipeline = PreprocessingPipeline(feature_cols=FEATURE_COLS)
    X_train_scaled = pipeline.fit_transform(df_train)
    X_test_scaled = pipeline.transform(df_test)

    assert X_train_scaled.shape == (50, 16)
    assert X_test_scaled.shape == (50, 16)
    assert np.all(np.isfinite(X_train_scaled))
    assert np.all(np.isfinite(X_test_scaled))

    # Mean of train should be ~0 and std ~1
    np.testing.assert_allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(X_train_scaled.std(axis=0), 1.0, atol=1e-5)


if __name__ == "__main__":
    test_target_leakage_prevention()
    test_feature_count()
    test_preprocessing_pipeline_scaling()
    print("All preprocessing tests passed successfully!")
