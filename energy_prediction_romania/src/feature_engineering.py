"""
Feature Engineering Script for SURYOKTI Romania PV Implementation.
Derives meteorological and cyclical solar-temporal features without target leakage.
Exports processed datasets and feature definitions.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import (
    DATA_CLEANED_DIR,
    DATA_PROCESSED_DIR,
    REPORTS_DIR,
    FEATURE_COLS,
    METEOROLOGICAL_FEATURES,
    FORBIDDEN_LEAKAGE_COLS,
)
from src.utils import setup_logger, save_json, encode_cyclical

logger = setup_logger("feature_engineering", REPORTS_DIR / "feature_engineering.log")


def engineer_features(df: pd.DataFrame, site_name: str) -> pd.DataFrame:
    """Generate cyclical temporal features and assemble forecasting dataset."""
    logger.info(f"Engineering features for {site_name} ({len(df)} rows)...")

    # Ensure UTC datetime
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)

    # 1. Temporal components
    hour = df["datetime_utc"].dt.hour
    day_of_year = df["datetime_utc"].dt.dayofyear
    month = df["datetime_utc"].dt.month

    # 2. Cyclical Encodings
    df["hour_sin"], df["hour_cos"] = encode_cyclical(hour, 24.0)
    df["day_of_year_sin"], df["day_of_year_cos"] = encode_cyclical(day_of_year - 1, 365.25)
    df["month_sin"], df["month_cos"] = encode_cyclical(month - 1, 12.0)

    # 3. Check for forbidden columns in FEATURE_COLS
    leakage_detected = [col for col in FEATURE_COLS if col in FORBIDDEN_LEAKAGE_COLS]
    if leakage_detected:
        raise ValueError(f"CRITICAL: Target leakage features detected in FEATURE_COLS: {leakage_detected}")

    # Verify all feature columns exist
    missing_feats = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_feats:
        raise KeyError(f"Missing required feature columns: {missing_feats}")

    logger.info(f"{site_name}: Generated {len(FEATURE_COLS)} features successfully.")
    return df


def run_feature_engineering():
    """Run feature engineering on both Site A and Site B."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_a = DATA_CLEANED_DIR / "site_a_cleaned.csv"
    cleaned_b = DATA_CLEANED_DIR / "site_b_cleaned.csv"

    df_a = pd.read_csv(cleaned_a)
    df_b = pd.read_csv(cleaned_b)

    df_a_proc = engineer_features(df_a, "Site A")
    df_b_proc = engineer_features(df_b, "Site B")

    out_a = DATA_PROCESSED_DIR / "site_a_processed.csv"
    out_b = DATA_PROCESSED_DIR / "site_b_processed.csv"

    df_a_proc.to_csv(out_a, index=False)
    df_b_proc.to_csv(out_b, index=False)
    logger.info(f"Processed datasets saved to:\n  - {out_a}\n  - {out_b}")

    # Save feature metadata
    feature_meta = {
        "feature_count": len(FEATURE_COLS),
        "meteorological_features": METEOROLOGICAL_FEATURES,
        "cyclical_time_features": [
            "hour_sin", "hour_cos",
            "day_of_year_sin", "day_of_year_cos",
            "month_sin", "month_cos"
        ],
        "all_feature_columns": FEATURE_COLS,
        "primary_target": "capacity_factor",
        "secondary_target": "active_power_kw",
        "forbidden_leakage_columns_excluded": FORBIDDEN_LEAKAGE_COLS,
    }
    save_json(feature_meta, REPORTS_DIR / "feature_columns.json")
    logger.info("Feature engineering completed successfully.")


if __name__ == "__main__":
    run_feature_engineering()

