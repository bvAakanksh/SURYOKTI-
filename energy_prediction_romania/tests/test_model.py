"""
Unit tests for model instantiation and training logic.
Ensures models produce finite predictions and satisfy physical bounds.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config import RF_HYPERPARAMS, XGB_HYPERPARAMS, FEATURE_COLS
from src.train_stacking_model import build_stacking_model
from src.utils import apply_physical_constraints


def test_base_models_instantiation():
    """Verify base models initialize with specified parameters."""
    rf = RandomForestRegressor(**RF_HYPERPARAMS)
    xgb = XGBRegressor(**XGB_HYPERPARAMS)

    assert rf.n_estimators == 200
    assert rf.max_depth == 8
    assert xgb.n_estimators == 200
    assert xgb.max_depth == 4


def test_stacking_model_fit_and_predict():
    """Verify StackingRegressor trains and outputs valid capacity factor predictions."""
    np.random.seed(42)
    X_train = np.random.randn(80, len(FEATURE_COLS))
    y_train = np.random.uniform(0.0, 1.0, size=80)

    X_test = np.random.randn(20, len(FEATURE_COLS))

    # Fast test stack with small estimators
    stack = build_stacking_model()
    # Replace with fast params for test
    stack.named_estimators["random_forest"].set_params(n_estimators=10, max_depth=3)
    stack.named_estimators["xgboost"].set_params(n_estimators=10, max_depth=2)
    stack.final_estimator.named_steps["nn"].set_params(max_iter=30)

    stack.fit(X_train, y_train)
    raw_preds = stack.predict(X_test)

    assert len(raw_preds) == 20
    assert np.all(np.isfinite(raw_preds))

    # Physical constraints
    elev = np.array([10.0] * 15 + [-5.0] * 5)
    constrained_preds = apply_physical_constraints(raw_preds, elev, elevation_threshold=0.0)

    assert np.all(constrained_preds >= 0.0)
    assert np.all(constrained_preds <= 1.0)
    # Nighttime must be 0.0
    assert np.all(constrained_preds[15:] == 0.0)


if __name__ == "__main__":
    test_base_models_instantiation()
    test_stacking_model_fit_and_predict()
    print("All model tests passed successfully!")

