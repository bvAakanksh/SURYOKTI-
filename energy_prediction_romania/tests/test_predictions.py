"""
Integration tests for inference pipeline and saved model artifacts.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import MODELS_SITE_A_DIR, DATA_PROCESSED_DIR, RATED_CAPACITY_KW
from src.predict import SolarEnergyPredictor


def test_predictor_loading():
    """Verify SolarEnergyPredictor loads saved artifacts for Site A."""
    predictor = SolarEnergyPredictor(site="site_a")
    assert predictor.model is not None
    assert predictor.pipeline is not None
    assert len(predictor.feature_cols) == 16


def test_inference_and_kw_conversion():
    """Verify that predictions match capacity factor and kW scaling."""
    predictor = SolarEnergyPredictor(site="site_a")
    sample_df = pd.read_csv(DATA_PROCESSED_DIR / "site_a_processed.csv").iloc[:10].copy()

    preds = predictor.predict(sample_df)

    assert "predicted_capacity_factor" in preds.columns
    assert "predicted_active_power_kw" in preds.columns

    # Verify kW = CF * 50.0 within floating point precision
    expected_kw = preds["predicted_capacity_factor"] * RATED_CAPACITY_KW
    diff = np.abs(preds["predicted_active_power_kw"] - expected_kw)
    assert np.all(diff < 0.05), f"Max discrepancy in kW conversion: {diff.max()}"

    # Verify bounds [0, 1] and [0, 50]
    assert np.all(preds["predicted_capacity_factor"] >= 0.0)
    assert np.all(preds["predicted_capacity_factor"] <= 1.0)
    assert np.all(preds["predicted_active_power_kw"] >= 0.0)
    assert np.all(preds["predicted_active_power_kw"] <= RATED_CAPACITY_KW + 1e-3)


if __name__ == "__main__":
    test_predictor_loading()
    test_inference_and_kw_conversion()
    print("All prediction tests passed successfully!")

