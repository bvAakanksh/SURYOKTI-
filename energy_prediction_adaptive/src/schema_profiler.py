"""
Schema Profiler and Leakage Detector for SURYOKTI Adaptive Prediction Pipeline.
Inspects arbitrary datasets, identifies timestamp frequency, handles target resolution
without silent guessing, categorizes column types, and detects data leakage risks.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import (
    TIMESTAMP_CANDIDATES,
    TARGET_CANDIDATES,
    LEAKAGE_KEYWORDS,
    LEAKAGE_CORRELATION_THRESHOLD,
    SEMANTIC_ALIASES,
)
from src.utils import setup_logger, save_json

logger = setup_logger("schema_profiler")


class SchemaProfiler:
    """Automated schema, target, and leakage profiler for solar datasets."""

    def __init__(
        self,
        target_col: Optional[str] = None,
        timestamp_col: Optional[str] = None,
        leakage_corr_threshold: float = LEAKAGE_CORRELATION_THRESHOLD
    ):
        self.user_target_col = target_col
        self.user_timestamp_col = timestamp_col
        self.leakage_corr_threshold = leakage_corr_threshold

    def resolve_timestamp(self, df: pd.DataFrame) -> Optional[str]:
        """Detect or validate timestamp column."""
        if self.user_timestamp_col:
            if self.user_timestamp_col not in df.columns:
                raise KeyError(f"Specified timestamp column '{self.user_timestamp_col}' not found in dataframe.")
            return self.user_timestamp_col

        # Auto-detect timestamp column from candidate names
        for candidate in TIMESTAMP_CANDIDATES:
            if candidate in df.columns:
                logger.info(f"Auto-detected timestamp column: '{candidate}'")
                return candidate

        # Check for any datetime dtype
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                logger.info(f"Auto-detected datetime dtype column: '{col}'")
                return col

        logger.warning("No timestamp column detected. Dataset will be treated as non-temporal.")
        return None

    def resolve_target(self, df: pd.DataFrame) -> str:
        """
        Resolve the target column.
        Strict Rule: Do not silently guess if multiple candidates exist!
        """
        if self.user_target_col:
            if self.user_target_col not in df.columns:
                raise KeyError(f"Specified target column '{self.user_target_col}' not found in dataset columns: {df.columns.tolist()}")
            logger.info(f"Using explicitly specified target column: '{self.user_target_col}'")
            return self.user_target_col

        # Find all matching candidates in dataset
        present_candidates = [c for c in TARGET_CANDIDATES if c in df.columns]

        if len(present_candidates) == 1:
            logger.info(f"Single target candidate identified: '{present_candidates[0]}'")
            return present_candidates[0]

        if len(present_candidates) > 1:
            raise ValueError(
                f"Multiple target candidates detected in dataset: {present_candidates}.\n"
                f"To prevent silent assumptions, please specify the intended target explicitly "
                f"via `--target <column_name>` (e.g. `--target {present_candidates[0]}`)."
            )

        raise ValueError(
            f"No standard solar target candidate found in dataset columns: {df.columns.tolist()}.\n"
            f"Please specify the target column explicitly using `--target <column_name>`."
        )

    def detect_timestamp_frequency(self, series: pd.Series) -> Dict[str, Any]:
        """Infer sampling frequency and regularity from timestamp series."""
        try:
            ts_dt = pd.to_datetime(series, utc=True)
            sorted_ts = ts_dt.sort_values().drop_duplicates()
            diffs = sorted_ts.diff().dropna()
            if len(diffs) == 0:
                return {"inferred_freq": "unknown", "is_regular": False}

            median_sec = diffs.dt.total_seconds().median()
            freq_desc = "unknown"
            if 270 <= median_sec <= 330:
                freq_desc = "5-minute"
            elif 540 <= median_sec <= 660:
                freq_desc = "10-minute"
            elif 840 <= median_sec <= 960:
                freq_desc = "15-minute"
            elif 1700 <= median_sec <= 1900:
                freq_desc = "30-minute"
            elif 3500 <= median_sec <= 3700:
                freq_desc = "hourly"
            elif 85000 <= median_sec <= 88000:
                freq_desc = "daily"
            else:
                freq_desc = f"{int(median_sec)} seconds"

            is_regular = bool((diffs.dt.total_seconds() == median_sec).mean() > 0.90)
            return {
                "inferred_freq": freq_desc,
                "median_interval_seconds": float(median_sec),
                "is_regular": is_regular,
                "start_time": str(sorted_ts.min()),
                "end_time": str(sorted_ts.max()),
                "is_monotonic": bool(ts_dt.is_monotonic_increasing),
            }
        except Exception as e:
            logger.warning(f"Failed to infer timestamp frequency: {e}")
            return {"inferred_freq": "error", "error": str(e)}

    def detect_leakage_columns(self, df: pd.DataFrame, target_col: str, timestamp_col: Optional[str]) -> List[str]:
        """
        Identify features with high risk of target leakage:
        1. Heuristic name keywords (e.g. yield, dc_power, cumulative, status).
        2. Extremely high correlation (|r| >= threshold) with target.
        """
        leakage_cols = []
        candidate_features = [c for c in df.columns if c not in [target_col, timestamp_col]]

        # 1. Keyword check
        for col in candidate_features:
            col_lower = col.lower()
            for kw in LEAKAGE_KEYWORDS:
                if kw in col_lower:
                    leakage_cols.append(col)
                    logger.warning(f"Potential leakage feature identified by keyword '{kw}': '{col}'")
                    break

        # 2. Correlation check on numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if target_col in numeric_cols:
            corrs = df[numeric_cols].corrwith(df[target_col]).abs()
            high_corr_cols = corrs[corrs >= self.leakage_corr_threshold].index.tolist()
            for col in high_corr_cols:
                if col != target_col and col not in leakage_cols:
                    leakage_cols.append(col)
                    logger.warning(f"Potential leakage feature identified by correlation ({corrs[col]:.3f} >= {self.leakage_corr_threshold}): '{col}'")

        return sorted(list(set(leakage_cols)))

    def categorize_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        timestamp_col: Optional[str],
        leakage_cols: List[str]
    ) -> Dict[str, List[str]]:
        """Categorize remaining valid features into numerical, categorical, and semantic roles."""
        excluded = [target_col]
        if timestamp_col:
            excluded.append(timestamp_col)
        excluded.extend(leakage_cols)

        # Exclude auxiliary timestamp or ID columns
        for c in df.columns:
            c_low = c.lower()
            if c not in excluded:
                if any(ts_word in c_low for ts_word in ["datetime", "timestamp", "date_time", "local_time", "time_utc"]):
                    excluded.append(c)
                    logger.info(f"Excluding auxiliary temporal column from features: '{c}'")

        valid_cols = [c for c in df.columns if c not in excluded]

        numeric_cols = []
        categorical_cols = []

        for col in valid_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                # If numeric but only 2-5 unique integer values, treat as categorical
                if df[col].nunique() <= 5 and not pd.api.types.is_float_dtype(df[col]):
                    categorical_cols.append(col)
                else:
                    numeric_cols.append(col)
            elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                # Only include reasonable cardinality categoricals (<= 20 unique levels)
                cardinality = df[col].nunique()
                if cardinality <= 20:
                    categorical_cols.append(col)
                else:
                    logger.info(f"Excluding high-cardinality string/ID column '{col}' ({cardinality} unique values) from features.")

        # Detect solar semantic features
        semantic_map = {}
        for semantic_role, aliases in SEMANTIC_ALIASES.items():
            for col in numeric_cols:
                if any(alias in col.lower() for alias in aliases):
                    semantic_map[semantic_role] = col
                    break

        return {
            "numerical_features": numeric_cols,
            "categorical_features": categorical_cols,
            "semantic_features": semantic_map,
        }

    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive schema inspection."""
        logger.info(f"Profiling dataset ({len(df)} rows, {len(df.columns)} columns)...")

        # 1. Target and Timestamp resolution
        target_col = self.resolve_target(df)
        timestamp_col = self.resolve_timestamp(df)

        # 2. Frequency analysis
        freq_info = self.detect_timestamp_frequency(df[timestamp_col]) if timestamp_col else {}

        # 3. Leakage check
        leakage_cols = self.detect_leakage_columns(df, target_col, timestamp_col)

        # 4. Feature categorization
        feature_cats = self.categorize_features(df, target_col, timestamp_col, leakage_cols)

        # 5. Missing values and duplicates
        missing_counts = df.isnull().sum().to_dict()
        duplicate_rows = int(df.duplicated().sum())

        # 6. Target summary
        target_summary = {}
        if pd.api.types.is_numeric_dtype(df[target_col]):
            target_series = df[target_col].dropna()
            target_summary = {
                "min": float(target_series.min()) if len(target_series) > 0 else None,
                "max": float(target_series.max()) if len(target_series) > 0 else None,
                "mean": round(float(target_series.mean()), 4) if len(target_series) > 0 else None,
                "std": round(float(target_series.std()), 4) if len(target_series) > 0 else None,
                "zero_count": int((target_series == 0).sum()),
                "zero_pct": round(float((target_series == 0).mean() * 100.0), 2) if len(target_series) > 0 else 0.0,
            }

        profile_report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "target_column": target_col,
            "timestamp_column": timestamp_col,
            "timestamp_frequency": freq_info,
            "leakage_columns_detected": leakage_cols,
            "numerical_features": feature_cats["numerical_features"],
            "categorical_features": feature_cats["categorical_features"],
            "semantic_features": feature_cats["semantic_features"],
            "missing_values": missing_counts,
            "duplicate_rows": duplicate_rows,
            "target_summary": target_summary,
        }

        logger.info(
            f"Profile Complete: Target='{target_col}' | Time='{timestamp_col}' ({freq_info.get('inferred_freq', 'N/A')}) | "
            f"Num Features={len(feature_cats['numerical_features'])} | Cat Features={len(feature_cats['categorical_features'])} | "
            f"Leakage Excluded={len(leakage_cols)}"
        )

        return profile_report
