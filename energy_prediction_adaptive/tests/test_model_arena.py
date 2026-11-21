"""
Unit tests for ModelArena training, tuning, and validation selection.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import KFold
from src.model_arena import ModelArena


def test_model_arena_execution():
    """Verify that ModelArena trains all models and selects winner without failure."""
    np.random.seed(42)
    X_train = np.random.randn(60, 5)
    y_train = np.random.rand(60)

    X_val = np.random.randn(20, 5)
    y_val = np.random.rand(20)

    cv_splitter = KFold(n_splits=2, shuffle=True, random_state=42)

    arena = ModelArena(tune_models=False)
    results = arena.compete(X_train, y_train, X_val, y_val, cv_splitter)

    assert results["winner_name"] is not None
    assert arena.winner_model_ is not None
    assert len(results["all_validation_metrics"]) >= 4
    assert "Baseline (Mean)" in results["all_validation_metrics"]
    assert "Random Forest" in results["all_validation_metrics"]
    assert "XGBoost" in results["all_validation_metrics"]
    assert "Stacking Ensemble" in results["all_validation_metrics"]


if __name__ == "__main__":
    test_model_arena_execution()
    print("All ModelArena tests passed successfully!")

