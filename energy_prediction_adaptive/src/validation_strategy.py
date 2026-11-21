"""
Validation Strategy and Split Partitioning Module for SURYOKTI Adaptive Pipeline.
Detects temporal vs group dependencies, provides chronological partitioning,
and builds TimeSeriesSplit / GroupKFold cross-validators.
"""
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, KFold, GroupKFold

from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE
from src.utils import setup_logger

logger = setup_logger("validation_strategy")


class ValidationManager:
    """Manages train/val/test splitting and cross-validation generation."""

    def __init__(
        self,
        timestamp_col: Optional[str] = None,
        group_col: Optional[str] = None,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO
    ):
        self.timestamp_col = timestamp_col
        self.group_col = group_col
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def partition_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Partition dataset into strictly isolated Train, Validation, and Holdout Test sets.
        Enforces chronological order if a timestamp column is present.
        """
        n = len(df)
        df_sorted = df.copy()

        if self.timestamp_col and self.timestamp_col in df.columns:
            logger.info(f"Sorting chronologically by '{self.timestamp_col}'...")
            df_sorted["_dt_temp"] = pd.to_datetime(df_sorted[self.timestamp_col], utc=True)
            df_sorted = df_sorted.sort_values("_dt_temp").drop(columns=["_dt_temp"]).reset_index(drop=True)
            strategy = "chronological"
        else:
            logger.info("No timestamp provided; partitioning without time ordering.")
            strategy = "standard_partition"

        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_df = df_sorted.iloc[:n_train].copy().reset_index(drop=True)
        val_df = df_sorted.iloc[n_train:n_train + n_val].copy().reset_index(drop=True)
        test_df = df_sorted.iloc[n_train + n_val:].copy().reset_index(drop=True)

        meta = {
            "strategy": strategy,
            "total_samples": n,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
        }

        if self.timestamp_col and self.timestamp_col in df.columns:
            meta["train_time_span"] = [str(train_df[self.timestamp_col].iloc[0]), str(train_df[self.timestamp_col].iloc[-1])]
            meta["val_time_span"] = [str(val_df[self.timestamp_col].iloc[0]), str(val_df[self.timestamp_col].iloc[-1])]
            meta["test_time_span"] = [str(test_df[self.timestamp_col].iloc[0]), str(test_df[self.timestamp_col].iloc[-1])]

        logger.info(
            f"Partition complete ({strategy}): Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )
        return train_df, val_df, test_df, meta

    def get_cv_splitter(self, n_splits: int = 3):
        """
        Return appropriate cross-validation splitter:
        - TimeSeriesSplit for temporal data.
        - GroupKFold if grouping column specified.
        - KFold for generic i.i.d. tabular data.
        """
        if self.group_col:
            logger.info(f"Using GroupKFold on '{self.group_col}' with {n_splits} splits.")
            return GroupKFold(n_splits=n_splits)
        elif self.timestamp_col:
            logger.info(f"Using TimeSeriesSplit with {n_splits} splits (no future lookahead).")
            return TimeSeriesSplit(n_splits=n_splits)
        else:
            logger.info(f"Using KFold with {n_splits} splits (shuffle=True).")
            return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

