"""
Model Arena and Selection Module for SURYOKTI Adaptive Architecture.
Trains and evaluates a competitive suite:
1. Baseline Model (Mean Predictor)
2. Ridge Regression
3. Random Forest Regressor (Tuned)
4. XGBoost Regressor (Tuned)
5. Canonical Stacking Ensemble (RF + XGB -> MLPRegressor Meta-Model)
Selects the overall winner based strictly on Validation set performance.
"""
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config import RANDOM_STATE
from src.utils import setup_logger, calculate_comprehensive_metrics
from src.hyperparameter_tuner import tune_model

logger = setup_logger("model_arena")


def build_stacking_ensemble(rf_estimator, xgb_estimator, cv_splits: int = 5) -> StackingRegressor:
    """Build canonical StackingRegressor combining RF and XGBoost into an MLP meta-model."""
    nn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("nn", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.05,
            learning_rate_init=0.003,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            max_iter=500,
            random_state=RANDOM_STATE
        ))
    ])

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    return StackingRegressor(
        estimators=[
            ("random_forest", rf_estimator),
            ("xgboost", xgb_estimator)
        ],
        final_estimator=nn_pipeline,
        passthrough=True,
        cv=kf,
        n_jobs=-1
    )


class ModelArena:
    """Orchestrates candidate model training, tuning, and validation competition."""

    def __init__(self, tune_models: bool = True, n_iter_tuning: int = 10):
        self.tune_models = tune_models
        self.n_iter_tuning = n_iter_tuning
        self.models_: Dict[str, Any] = {}
        self.val_performances_: Dict[str, Dict[str, float]] = {}
        self.winner_name_: Optional[str] = None
        self.winner_model_: Optional[Any] = None

    def compete(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        cv_splitter
    ) -> Dict[str, Any]:
        """
        Train all models on X_train, evaluate on X_val, and select winning model.
        """
        logger.info("--- Starting Model Arena Competition ---")

        # 1. Baseline: Mean Predictor
        logger.info("Evaluating Baseline (Dummy Mean)...")
        baseline = DummyRegressor(strategy="mean")
        baseline.fit(X_train, y_train)
        self.models_["Baseline (Mean)"] = baseline

        # 2. Linear / Ridge Regression
        logger.info("Evaluating Ridge Regression...")
        if self.tune_models:
            ridge, _ = tune_model("ridge", X_train, y_train, cv_splitter, n_iter=self.n_iter_tuning)
        else:
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
        self.models_["Ridge"] = ridge

        # 3. Random Forest
        logger.info("Evaluating Random Forest...")
        if self.tune_models:
            rf, _ = tune_model("random_forest", X_train, y_train, cv_splitter, n_iter=self.n_iter_tuning)
        else:
            rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
            rf.fit(X_train, y_train)
        self.models_["Random Forest"] = rf

        # 4. XGBoost
        logger.info("Evaluating XGBoost...")
        if self.tune_models:
            xgb, _ = tune_model("xgboost", X_train, y_train, cv_splitter, n_iter=self.n_iter_tuning)
        else:
            xgb = XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=4, random_state=RANDOM_STATE, n_jobs=-1)
            xgb.fit(X_train, y_train)
        self.models_["XGBoost"] = xgb

        # 5. Stacking Ensemble
        logger.info("Training Canonical Stacking Ensemble (RF + XGB -> MLP Meta-Model)...")
        stacking = build_stacking_ensemble(rf, xgb, cv_splits=3)
        stacking.fit(X_train, y_train)
        self.models_["Stacking Ensemble"] = stacking

        # Validation Evaluation
        logger.info("--- Scoring All Models on Validation Set ---")
        best_val_mae = float("inf")
        best_model_name = None

        for name, model in self.models_.items():
            val_preds = model.predict(X_val)
            val_metrics = calculate_comprehensive_metrics(y_val, val_preds)
            self.val_performances_[name] = val_metrics
            logger.info(f"[{name}] Validation R² = {val_metrics['r2']} | MAE = {val_metrics['mae']} | RMSE = {val_metrics['rmse']}")

            # Selection criterion: lowest validation MAE (and break tie with R²)
            if val_metrics["mae"] < best_val_mae:
                best_val_mae = val_metrics["mae"]
                best_model_name = name

        self.winner_name_ = best_model_name
        self.winner_model_ = self.models_[best_model_name]
        logger.info(f"★ WINNING MODEL SELECTED: '{self.winner_name_}' (Val MAE: {best_val_mae}) ★")

        return {
            "winner_name": self.winner_name_,
            "winner_val_mae": best_val_mae,
            "all_validation_metrics": self.val_performances_,
        }

