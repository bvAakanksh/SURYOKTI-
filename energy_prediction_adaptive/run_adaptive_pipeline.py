"""
Master Adaptive Pipeline Orchestrator for SURYOKTI Solar Energy Prediction.
Automates schema profiling, leakage checks, adaptive preprocessing, time-safe validation,
candidate model arena (RF, XGBoost, Stacking meta-model), evaluation, and artifact bundling.
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd

from config import (
    MODELS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
)
from src.utils import setup_logger, save_json
from src.schema_profiler import SchemaProfiler
from src.adaptive_preprocessor import AdaptivePreprocessor
from src.validation_strategy import ValidationManager
from src.model_arena import ModelArena
from src.metrics_evaluator import MetricsEvaluator
from src.pipeline_builder import build_and_save_bundle

logger = setup_logger("run_adaptive_pipeline", REPORTS_DIR / "pipeline_execution.log")


def run_pipeline(
    data_path: Path,
    target_col: str = None,
    timestamp_col: str = None,
    rated_capacity: float = None,
    tune_models: bool = True,
    n_iter_tuning: int = 10,
    output_prefix: str = "adaptive"
):
    """Execute end-to-end adaptive solar forecasting workflow."""
    logger.info("=================================================================")
    logger.info("=== STARTING SURYOKTI ADAPTIVE SOLAR FORECASTING PIPELINE ===")
    logger.info("=================================================================")
    logger.info(f"Dataset: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Input dataset not found at: {data_path}")

    df = pd.read_csv(data_path)

    # 1. Schema Profiling & Leakage Detection
    logger.info("\n>>> [Step 1/6] Automated Schema Profiling & Leakage Scanning")
    profiler = SchemaProfiler(target_col=target_col, timestamp_col=timestamp_col)
    profile = profiler.profile(df)
    save_json(profile, REPORTS_DIR / f"{output_prefix}_schema_profile.json")

    actual_target = profile["target_column"]
    actual_timestamp = profile["timestamp_column"]
    leakage_cols = profile["leakage_columns_detected"]

    # Drop rows where target is missing (strict supervised integrity)
    initial_len = len(df)
    df_clean = df.dropna(subset=[actual_target]).copy().reset_index(drop=True)
    if len(df_clean) < initial_len:
        logger.info(f"Excluded {initial_len - len(df_clean)} rows missing target '{actual_target}'.")

    # 2. Validation Partitioning
    logger.info("\n>>> [Step 2/6] Validation Partitioning (Train 70% / Val 10% / Holdout Test 20%)")
    val_mgr = ValidationManager(timestamp_col=actual_timestamp)
    train_df, val_df, test_df, split_meta = val_mgr.partition_data(df_clean)

    # 3. Adaptive Preprocessing
    logger.info("\n>>> [Step 3/6] Fitting Adaptive ColumnTransformer on Training Partition")
    preprocessor = AdaptivePreprocessor(
        numerical_features=profile["numerical_features"],
        categorical_features=profile["categorical_features"],
        timestamp_col=actual_timestamp,
    )
    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)

    y_train = train_df[actual_target].to_numpy(dtype=float)
    y_val = val_df[actual_target].to_numpy(dtype=float)
    y_test = test_df[actual_target].to_numpy(dtype=float)

    cv_splitter = val_mgr.get_cv_splitter(n_splits=3)

    # 4. Model Arena Competition & Tuning
    logger.info("\n>>> [Step 4/6] Model Arena Competition (Baseline, Ridge, RF, XGBoost, Stacking Ensemble)")
    arena = ModelArena(tune_models=tune_models, n_iter_tuning=n_iter_tuning)
    arena_results = arena.compete(X_train, y_train, X_val, y_val, cv_splitter)
    winner_name = arena_results["winner_name"]
    winner_model = arena.winner_model_

    # 5. Holdout Test Set Evaluation & Plotting
    logger.info("\n>>> [Step 5/6] Final Holdout Test Evaluation & Diagnostic Visualization")
    evaluator = MetricsEvaluator(
        rated_capacity=rated_capacity,
        target_unit=actual_target,
        elevation_col=profile["semantic_features"].get("solar_elevation", "solar_elevation_deg")
    )
    test_metrics, test_preds = evaluator.evaluate_test_set(
        models=arena.models_,
        X_test=X_test,
        y_test=y_test,
        test_df=test_df
    )

    save_json(test_metrics, METRICS_DIR / f"{output_prefix}_test_metrics.json")
    test_preds.to_csv(PREDICTIONS_DIR / f"{output_prefix}_test_predictions.csv", index=False)

    evaluator.generate_diagnostic_plots(
        test_metrics=test_metrics,
        val_metrics=arena_results["all_validation_metrics"],
        predictions_df=test_preds,
        winner_name=winner_name,
        plots_dir=PLOTS_DIR
    )

    # 6. Deployable Artifact Bundling
    logger.info("\n>>> [Step 6/6] Packaging Deployable AdaptiveInferenceBundle")
    metadata = {
        "dataset_name": data_path.name,
        "target_column": actual_target,
        "timestamp_column": actual_timestamp,
        "elevation_column": profile["semantic_features"].get("solar_elevation"),
        "rated_capacity": rated_capacity,
        "winner_model": winner_name,
        "split_metadata": split_meta,
        "validation_metrics": arena_results["all_validation_metrics"][winner_name],
        "test_metrics": test_metrics["all_hours"][winner_name],
        "leakage_features_excluded": leakage_cols,
        "numerical_features": profile["numerical_features"],
        "categorical_features": profile["categorical_features"],
    }
    bundle_path = build_and_save_bundle(preprocessor, winner_model, metadata, MODELS_DIR)

    logger.info("=================================================================")
    logger.info("=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY ===")
    logger.info(f"★ Selected Winner: '{winner_name}'")
    logger.info(f"★ Test All-Hours R²: {test_metrics['all_hours'][winner_name]['r2']} | MAE: {test_metrics['all_hours'][winner_name]['mae']}")
    logger.info(f"★ Bundle Saved: {bundle_path}")
    logger.info("=================================================================\n")

    return {
        "winner": winner_name,
        "validation_metrics": arena_results["all_validation_metrics"],
        "test_metrics": test_metrics,
        "bundle_path": str(bundle_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SURYOKTI Adaptive Solar Prediction Pipeline.")
    parser.add_argument("--data", type=str, default="energy_prediction_romania/datasets/raw/site_a_hourly_weather.csv", help="Path to solar dataset CSV.")
    parser.add_argument("--target", type=str, default="capacity_factor", help="Target column name.")
    parser.add_argument("--time-col", type=str, default=None, help="Timestamp column name (auto-detected if omitted).")
    parser.add_argument("--rated-capacity", type=float, default=50.0, help="Rated capacity of installation (for normalized metrics).")
    parser.add_argument("--no-tune", action="store_true", help="Disable randomized hyperparameter tuning.")
    parser.add_argument("--n-iter", type=int, default=10, help="Number of tuning iterations.")
    parser.add_argument("--prefix", type=str, default="site_a_adaptive", help="Output artifact prefix.")
    args = parser.parse_args()

    run_pipeline(
        data_path=Path(args.data),
        target_col=args.target,
        timestamp_col=args.time_col,
        rated_capacity=args.rated_capacity,
        tune_models=not args.no_tune,
        n_iter_tuning=args.n_iter,
        output_prefix=args.prefix
    )

