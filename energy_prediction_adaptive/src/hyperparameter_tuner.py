"""
Hyperparameter Tuning Module for SURYOKTI Adaptive Pipeline.
Implements controlled, time-safe RandomizedSearchCV across model parameter distributions.
"""
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from config import RANDOM_STATE, TUNING_N_ITER, TUNING_CV_SPLITS
from src.utils import setup_logger

logger = setup_logger("hyperparameter_tuner")

# Param Distributions
RF_PARAM_DIST = {
    "n_estimators": [100, 150, 200, 250],
    "max_depth": [6, 8, 10, 12, None],
    "min_samples_split": [2, 4, 6, 8],
    "min_samples_leaf": [2, 3, 5],
    "max_features": [0.6, 0.8, 1.0],
}

XGB_PARAM_DIST = {
    "n_estimators": [100, 150, 200, 250],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_alpha": [0.0, 0.1, 0.5],
    "reg_lambda": [1.0, 1.5, 3.0],
}

RIDGE_PARAM_DIST = {
    "alpha": np.logspace(-2, 3, 20).tolist(),
}


def tune_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splitter,
    n_iter: int = TUNING_N_ITER
) -> Tuple[Any, Dict[str, Any]]:
    """
    Tune specified model architecture using RandomizedSearchCV with time-safe CV.
    """
    logger.info(f"Tuning {model_name} (n_iter={n_iter})...")

    if model_name == "random_forest":
        base_estimator = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        param_dist = RF_PARAM_DIST
    elif model_name == "xgboost":
        base_estimator = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        param_dist = XGB_PARAM_DIST
    elif model_name == "ridge":
        base_estimator = Ridge()
        param_dist = RIDGE_PARAM_DIST
    else:
        raise ValueError(f"Tuning not configured for model: {model_name}")

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_dist,
        n_iter=min(n_iter, 15),
        scoring="neg_mean_absolute_error",
        cv=cv_splitter,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train)

    best_score_mae = -search.best_score_
    logger.info(f"Tuned {model_name}: Best CV MAE = {best_score_mae:.4f} | Best Params = {search.best_params_}")

    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_cv_mae": round(float(best_score_mae), 4),
    }

