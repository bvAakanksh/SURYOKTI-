"""
Base Model Training Script for SURYOKTI Romania PV Implementation.
Trains standalone Random Forest and XGBoost regressors on chronologically partitioned training data.
Evaluates their standalone performance on validation and test sets.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config import (
    DATA_PROCESSED_DIR,
    MODELS_SITE_A_DIR,
    MODELS_SITE_B_DIR,
    REPORTS_DIR,
    FEATURE_COLS,
    PRIMARY_TARGET,
    RF_HYPERPARAMS,
    XGB_HYPERPARAMS,
    RATED_CAPACITY_KW,
)
from src.utils import setup_logger, save_artifact, save_json, calculate_metrics, apply_physical_constraints
from src.data_split import split_chronological
from src.preprocessing import PreprocessingPipeline

logger = setup_logger("train_base_models", REPORTS_DIR / "train_base_models.log")


def train_base_models_for_site(site_name: str, processed_path: Path, model_dir: Path) -> dict:
    """Train standalone RF and XGBoost base models for a specific site."""
    logger.info(f"=== Training Base Models for {site_name} ===")
    df = pd.read_csv(processed_path)

    # 1. Chronological Partition
    train_df, val_df, test_df, split_info = split_chronological(df)

    # 2. Fit Preprocessing Pipeline
    pipeline = PreprocessingPipeline(feature_cols=FEATURE_COLS)
    pipeline.fit(train_df)
    pipeline.save(model_dir / "preprocessing_pipeline.joblib")

    X_train = pipeline.transform(train_df)
    y_train = train_df[PRIMARY_TARGET].to_numpy(dtype=float)

    X_val = pipeline.transform(val_df)
    y_val = val_df[PRIMARY_TARGET].to_numpy(dtype=float)

    X_test = pipeline.transform(test_df)
    y_test = test_df[PRIMARY_TARGET].to_numpy(dtype=float)

    # Solar elevation for test post-processing
    elev_test = test_df["solar_elevation_deg"].to_numpy(dtype=float)

    # 3. Train Random Forest
    logger.info("Training Random Forest Regressor...")
    rf = RandomForestRegressor(**RF_HYPERPARAMS)
    rf.fit(X_train, y_train)
    rf_pred_raw = rf.predict(X_test)
    rf_pred = apply_physical_constraints(rf_pred_raw, elev_test, elevation_threshold=0.0)
    rf_metrics = calculate_metrics(y_test, rf_pred, rated_capacity_kw=RATED_CAPACITY_KW, prefix="test")
    logger.info(f"RF Test R2: {rf_metrics['test_r2']} | MAE: {rf_metrics['test_mae_kw']} kW | RMSE: {rf_metrics['test_rmse_kw']} kW")
    save_artifact(rf, model_dir / "rf_model.joblib")

    # 4. Train XGBoost
    logger.info("Training XGBoost Regressor...")
    xgb = XGBRegressor(**XGB_HYPERPARAMS)
    xgb.fit(X_train, y_train)
    xgb_pred_raw = xgb.predict(X_test)
    xgb_pred = apply_physical_constraints(xgb_pred_raw, elev_test, elevation_threshold=0.0)
    xgb_metrics = calculate_metrics(y_test, xgb_pred, rated_capacity_kw=RATED_CAPACITY_KW, prefix="test")
    logger.info(f"XGB Test R2: {xgb_metrics['test_r2']} | MAE: {xgb_metrics['test_mae_kw']} kW | RMSE: {xgb_metrics['test_rmse_kw']} kW")
    save_artifact(xgb, model_dir / "xgb_model.joblib")

    results = {
        "site": site_name,
        "split_info": split_info,
        "random_forest": rf_metrics,
        "xgboost": xgb_metrics,
    }
    save_json(results, model_dir / "base_models_performance.json")
    return results


def run_train_base():
    """Train base models for both sites."""
    train_base_models_for_site("Site A", DATA_PROCESSED_DIR / "site_a_processed.csv", MODELS_SITE_A_DIR)
    train_base_models_for_site("Site B", DATA_PROCESSED_DIR / "site_b_processed.csv", MODELS_SITE_B_DIR)


if __name__ == "__main__":
    run_train_base()

