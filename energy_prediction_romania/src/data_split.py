"""
Data Splitting Module for SURYOKTI Romania PV Implementation.
Executes chronological (temporal) train/validation/test partitioning
to completely prevent lookahead bias and data leakage.
"""
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import (
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    SOLAR_ELEVATION_DAYTIME_THRESHOLD
)
from src.utils import setup_logger

logger = setup_logger("data_split")


def split_chronological(
    df: pd.DataFrame,
    train_frac: float = TRAIN_RATIO,
    val_frac: float = VAL_RATIO,
    timestamp_col: str = "datetime_utc"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split DataFrame chronologically into train, validation, and test subsets.
    Guarantees no temporal leakage (Train < Val < Test in time).
    """
    df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
    n = len(df_sorted)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_df = df_sorted.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[n_train + n_val:].copy().reset_index(drop=True)

    split_info = {
        "total_rows": n,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_date_range": [str(train_df[timestamp_col].min()), str(train_df[timestamp_col].max())],
        "val_date_range": [str(val_df[timestamp_col].min()), str(val_df[timestamp_col].max())],
        "test_date_range": [str(test_df[timestamp_col].min()), str(test_df[timestamp_col].max())],
    }

    logger.info(
        f"Chronological Split: Total={n} | "
        f"Train={len(train_df)} ({split_info['train_date_range'][0]} to {split_info['train_date_range'][1]}) | "
        f"Val={len(val_df)} | "
        f"Test={len(test_df)} ({split_info['test_date_range'][0]} to {split_info['test_date_range'][1]})"
    )

    return train_df, val_df, test_df, split_info


def filter_daytime(
    df: pd.DataFrame,
    threshold_deg: float = SOLAR_ELEVATION_DAYTIME_THRESHOLD,
    elevation_col: str = "solar_elevation_deg"
) -> pd.DataFrame:
    """Filter dataset to include daytime hours with solar elevation > threshold_deg."""
    if elevation_col not in df.columns:
        raise KeyError(f"Elevation column '{elevation_col}' not found in dataframe.")
    day_df = df[df[elevation_col] > threshold_deg].copy().reset_index(drop=True)
    logger.info(f"Daytime filter (>{threshold_deg}°): {len(df)} -> {len(day_df)} rows ({len(day_df)/len(df)*100:.1f}%)")
    return day_df

