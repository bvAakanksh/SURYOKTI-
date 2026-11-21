"""
Master Pipeline Orchestrator for SURYOKTI Romania PV Implementation.
Executes the complete workflow end-to-end:
1. Dataset Inspection & Diagnostic Analysis
2. Data Cleaning & Domain Validation
3. Feature Engineering & Cyclical Encodings
4. Base Model Training (RF + XGBoost)
5. Stacking Ensemble Training (RF + XGBoost -> MLPRegressor)
6. Model Evaluation & Publication Diagnostic Plots
7. Bidirectional Cross-Site Generalization
8. Verification Inference Test
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from config import (
    DATA_RAW_DIR,
    DATA_CLEANED_DIR,
    DATA_PROCESSED_DIR,
    REPORTS_DIR,
)
from src.utils import setup_logger
from src.inspect_dataset import run_inspection
from src.data_cleaning import run_cleaning
from src.feature_engineering import run_feature_engineering
from src.train_base_models import run_train_base
from src.train_stacking_model import run_training as run_stacking_training
from src.evaluate_model import run_evaluation
from src.cross_site_evaluation import run_cross_site
from src.predict import run_test_inference

logger = setup_logger("run_pipeline", REPORTS_DIR / "pipeline_execution.log")


def verify_raw_datasets():
    """Ensure raw dataset CSVs are present before execution."""
    raw_a = DATA_RAW_DIR / "site_a_hourly_weather.csv"
    raw_b = DATA_RAW_DIR / "site_b_hourly_weather.csv"

    if not raw_a.exists():
        raise FileNotFoundError(
            f"Missing required raw dataset for Site A: {raw_a}.\n"
            f"Please ensure site_a_hourly_weather.csv is placed in {DATA_RAW_DIR}."
        )
    if not raw_b.exists():
        raise FileNotFoundError(
            f"Missing required raw dataset for Site B: {raw_b}.\n"
            f"Please ensure site_b_hourly_weather.csv is placed in {DATA_RAW_DIR}."
        )
    logger.info("Raw datasets verified successfully.")


def run_full_pipeline():
    """Run all pipeline phases sequentially."""
    logger.info("****************************************************************")
    logger.info("=== STARTING SURYOKTI ROMANIA PV PIPELINE EXECUTION ===")
    logger.info("****************************************************************")

    # Step 0: Pre-flight check
    logger.info("\n>>> [Step 0/8] Pre-Flight Verification")
    verify_raw_datasets()

    # Step 1: Inspection
    logger.info("\n>>> [Step 1/8] Inspecting Datasets & Generating Summary Reports")
    run_inspection()

    # Step 2: Data Cleaning
    logger.info("\n>>> [Step 2/8] Executing Domain Data Cleaning")
    run_cleaning()

    # Step 3: Feature Engineering
    logger.info("\n>>> [Step 3/8] Engineering Cyclical & Meteorological Features")
    run_feature_engineering()

    # Step 4: Base Models
    logger.info("\n>>> [Step 4/8] Training Standalone RF and XGBoost Base Models")
    run_train_base()

    # Step 5: Stacking Ensemble
    logger.info("\n>>> [Step 5/8] Training Canonical Stacking Ensemble (RF + XGB -> MLPRegressor)")
    run_stacking_training()

    # Step 6: Model Evaluation
    logger.info("\n>>> [Step 6/8] Evaluating Models & Generating Diagnostic Plots")
    run_evaluation()

    # Step 7: Cross-Site Evaluation
    logger.info("\n>>> [Step 7/8] Executing Bidirectional Cross-Site Generalization")
    run_cross_site()

    # Step 8: Sample Inference
    logger.info("\n>>> [Step 8/8] Executing Sample Inference Verification")
    run_test_inference()

    logger.info("****************************************************************")
    logger.info("=== SURYOKTI ROMANIA PV PIPELINE COMPLETED SUCCESSFULLY ===")
    logger.info("****************************************************************")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SURYOKTI Romania PV Pipeline.")
    parser.add_argument("--step", type=str, default="all", choices=["all", "inspect", "clean", "features", "train_base", "train_stack", "eval", "cross_site", "infer"], help="Pipeline stage to execute.")
    args = parser.parse_args()

    if args.step == "all":
        run_full_pipeline()
    elif args.step == "inspect":
        verify_raw_datasets()
        run_inspection()
    elif args.step == "clean":
        run_cleaning()
    elif args.step == "features":
        run_feature_engineering()
    elif args.step == "train_base":
        run_train_base()
    elif args.step == "train_stack":
        run_stacking_training()
    elif args.step == "eval":
        run_evaluation()
    elif args.step == "cross_site":
        run_cross_site()
    elif args.step == "infer":
        run_test_inference()

