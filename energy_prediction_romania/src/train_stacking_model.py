"""
Stacking Ensemble Model Training Script for SURYOKTI Romania PV Implementation.
Implements the canonical SURYOKTI architecture:
- Base Model 1: Random Forest Regressor
- Base Model 2: XGBoost Regressor
- Meta-Learner: MLPRegressor Neural Network (StandardScaler Pipeline)
- Stacking: Scikit-learn StackingRegressor with Out-Of-Fold Cross-Validation & Passthrough
- Physical Constraint: Non-negative clipping and solar elevation nocturnal cutoff.
"""
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    NN_HYPERPARAMS,
    STACKING_CV_SPLITS,
    RANDOM_STATE,
    RATED_CAPACITY_KW,
)
from src.utils import (
    setup_logger,
    save_artifact,
    save_json,
    calculate_metrics,
    apply_physical_constraints,
)
from src.data_split import split_chronological
from src.preprocessing import PreprocessingPipeline

logger = setup_logger("train_stacking_model", REPORTS_DIR / "train_stacking_model.log")


def build_stacking_model() -> StackingRegressor:
    """Build the canonical 3-stack SURYOKTI model."""
    rf = RandomForestRegressor(**RF_HYPERPARAMS)
    xgb = XGBRegressor(**XGB_HYPERPARAMS)

    # Neural network meta-model pipeline with internal scaling
    nn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("nn", MLPRegressor(**NN_HYPERPARAMS)),
    ])

    kf = KFold(n_splits=STACKING_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    stacking_regressor = StackingRegressor(
        estimators=[("random_forest", rf), ("xgboost", xgb)],
        final_estimator=nn_pipeline,
        passthrough=True,
        cv=kf,
        n_jobs=-1,
    )
    return stacking_regressor


def train_stacking_for_site(site_name: str, processed_path: Path, model_dir: Path) -> Dict[str, Any]:
    """Train full stacking ensemble on specified site data."""
    logger.info(f"==================================================")
    logger.info(f"--- Training SURYOKTI Stacking Model for {site_name} ---")
    logger.info(f"==================================================")

    df = pd.read_csv(processed_path)

    # 1. Chronological Split (70% Train, 10% Val, 20% Test)
    train_df, val_df, test_df, split_info = split_chronological(df)

    # 2. Fit Preprocessing Pipeline strictly on train
    pipeline = PreprocessingPipeline(feature_cols=FEATURE_COLS)
    pipeline.fit(train_df)
    pipeline.save(model_dir / "preprocessing_pipeline.joblib")

    X_train = pipeline.transform(train_df)
    y_train = train_df[PRIMARY_TARGET].to_numpy(dtype=float)

    X_val = pipeline.transform(val_df)
    y_val = val_df[PRIMARY_TARGET].to_numpy(dtype=float)

    X_test = pipeline.transform(test_df)
    y_test = test_df[PRIMARY_TARGET].to_numpy(dtype=float)

    elev_val = val_df["solar_elevation_deg"].to_numpy(dtype=float)
    elev_test = test_df["solar_elevation_deg"].to_numpy(dtype=float)

    # 3. Build & Train Stacking Regressor
    logger.info("Initializing Stacking Ensemble (RF + XGB -> NN Meta-Model)...")
    stack_model = build_stacking_model()

    logger.info(f"Fitting StackingRegressor on {len(X_train)} training samples...")
    stack_model.fit(X_train, y_train)
    logger.info("Stacking model training complete.")

    # 4. Predict & Evaluate on Validation Set
    val_pred_raw = stack_model.predict(X_val)
    val_pred = apply_physical_constraints(val_pred_raw, elev_val, elevation_threshold=0.0)
    val_metrics = calculate_metrics(y_val, val_pred, rated_capacity_kw=RATED_CAPACITY_KW, prefix="val")

    # 5. Predict & Evaluate on Test Set
    test_pred_raw = stack_model.predict(X_test)
    test_pred = apply_physical_constraints(test_pred_raw, elev_test, elevation_threshold=0.0)
    test_metrics = calculate_metrics(y_test, test_pred, rated_capacity_kw=RATED_CAPACITY_KW, prefix="test")

    logger.info(f"{site_name} Validation R2: {val_metrics['val_r2']} | MAE: {val_metrics['val_mae_kw']} kW")
    logger.info(f"{site_name} Test R2: {test_metrics['test_r2']} | MAE: {test_metrics['test_mae_kw']} kW | RMSE: {test_metrics['test_rmse_kw']} kW")

    # 6. Save Model Artifacts
    logger.info(f"Serializing all model artifacts to {model_dir}...")
    save_artifact(stack_model, model_dir / "stacking_ensemble.joblib")
    save_artifact(stack_model.named_estimators_["random_forest"], model_dir / "rf_model.joblib")
    save_artifact(stack_model.named_estimators_["xgboost"], model_dir / "xgb_model.joblib")
    save_artifact(stack_model.final_estimator_, model_dir / "meta_model.joblib")
    save_json({"feature_columns": FEATURE_COLS}, model_dir / "feature_list.json")

    summary = {
        "site": site_name,
        "model_architecture": "StackingRegressor (Random Forest + XGBoost -> MLPRegressor)",
        "split_info": split_info,
        "feature_count": len(FEATURE_COLS),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(summary, model_dir / "training_summary.json")

    return summary


def run_training():
    """Execute training for both Romanian sites."""
    summary_a = train_stacking_for_site("Site A", DATA_PROCESSED_DIR / "site_a_processed.csv", MODELS_SITE_A_DIR)
    summary_b = train_stacking_for_site("Site B", DATA_PROCESSED_DIR / "site_b_processed.csv", MODELS_SITE_B_DIR)

    logger.info("All site models trained and serialized successfully.")
    return summary_a, summary_b


if __name__ == "__main__":
    run_training()

