"""
Adaptive Preprocessing Pipeline for SURYOKTI Adaptive Architecture.
Uses Scikit-Learn ColumnTransformer with SimpleImputer, StandardScaler, OneHotEncoder,
and Cyclical Temporal Feature Expansion. Resilient to missing and extra columns during inference.
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils import setup_logger

logger = setup_logger("adaptive_preprocessor")


class CyclicalTemporalTransformer(BaseEstimator, TransformerMixin):
    """Derives periodic cyclical features (sine/cosine of hour, day of year, month) from timestamp."""

    def __init__(self, timestamp_col: Optional[str] = None):
        self.timestamp_col = timestamp_col
        self.derived_feature_names: List[str] = [
            "hour_sin", "hour_cos",
            "day_of_year_sin", "day_of_year_cos",
            "month_sin", "month_cos"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.timestamp_col is None or self.timestamp_col not in X.columns:
            # Return zeros if no timestamp available
            return np.zeros((len(X), len(self.derived_feature_names)))

        dt = pd.to_datetime(X[self.timestamp_col], utc=True)
        hour = dt.dt.hour
        day_of_year = dt.dt.dayofyear
        month = dt.dt.month

        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        doy_sin = np.sin(2 * np.pi * (day_of_year - 1) / 365.25)
        doy_cos = np.cos(2 * np.pi * (day_of_year - 1) / 365.25)
        month_sin = np.sin(2 * np.pi * (month - 1) / 12.0)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12.0)

        return np.column_stack([hour_sin, hour_cos, doy_sin, doy_cos, month_sin, month_cos])

    def get_feature_names_out(self, input_features=None):
        return np.array(self.derived_feature_names)


class AdaptivePreprocessor:
    """
    Robust, schema-flexible preprocessing transformer.
    Wraps ColumnTransformer and Cyclical encoder; guarantees safe handling
    of missing features (imputed) and extra columns (ignored) during inference.
    """

    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        timestamp_col: Optional[str] = None,
    ):
        self.numerical_features = list(numerical_features)
        self.categorical_features = list(categorical_features)
        self.timestamp_col = timestamp_col
        self.column_transformer: Optional[ColumnTransformer] = None
        self.cyclical_transformer: Optional[CyclicalTemporalTransformer] = None
        self.feature_names_out_: List[str] = []
        self.is_fitted = False
        self.feature_defaults_: Dict[str, Any] = {}

    def _build_pipeline(self) -> ColumnTransformer:
        transformers = []

        if self.numerical_features:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", num_pipe, self.numerical_features))

        if self.categorical_features:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", cat_pipe, self.categorical_features))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _align_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Harmonize input DataFrame with expected training schema:
        1. Ignore superfluous extra columns.
        2. Impute any missing expected columns with training medians/defaults.
        """
        df_aligned = df.copy()

        # Fill missing expected numerical features with default
        for col in self.numerical_features:
            if col not in df_aligned.columns:
                default_val = self.feature_defaults_.get(col, 0.0)
                logger.warning(f"Inference input missing expected column '{col}'. Imputing with default ({default_val}).")
                df_aligned[col] = default_val

        # Fill missing expected categorical features
        for col in self.categorical_features:
            if col not in df_aligned.columns:
                logger.warning(f"Inference input missing expected categorical column '{col}'. Imputing with 'missing'.")
                df_aligned[col] = "missing"

        return df_aligned

    def fit(self, X: pd.DataFrame, y=None) -> "AdaptivePreprocessor":
        """Fit ColumnTransformer and cyclical encoder strictly on training data."""
        logger.info(
            f"Fitting AdaptivePreprocessor: {len(self.numerical_features)} numerical, "
            f"{len(self.categorical_features)} categorical, timestamp='{self.timestamp_col}'."
        )

        # Store training defaults for resilient inference
        for col in self.numerical_features:
            if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                self.feature_defaults_[col] = float(X[col].median())
            else:
                self.feature_defaults_[col] = 0.0

        # Build & Fit ColumnTransformer
        self.column_transformer = self._build_pipeline()
        self.column_transformer.fit(X)

        # Cyclical encoder
        if self.timestamp_col and self.timestamp_col in X.columns:
            self.cyclical_transformer = CyclicalTemporalTransformer(timestamp_col=self.timestamp_col)
            self.cyclical_transformer.fit(X)

        # Extract output feature names
        num_cat_names = []
        if hasattr(self.column_transformer, "get_feature_names_out"):
            num_cat_names = list(self.column_transformer.get_feature_names_out())
        else:
            num_cat_names = self.numerical_features + self.categorical_features

        cyclical_names = self.cyclical_transformer.derived_feature_names if self.cyclical_transformer else []
        self.feature_names_out_ = num_cat_names + cyclical_names
        self.is_fitted = True
        logger.info(f"Fitted successfully. Total output transformed features: {len(self.feature_names_out_)}")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform dataframe into model-ready array."""
        if not self.is_fitted:
            raise RuntimeError("AdaptivePreprocessor must be fitted before transform.")

        X_aligned = self._align_dataframe(X)

        # Transform numerical & categorical
        X_core = self.column_transformer.transform(X_aligned)

        # Transform cyclical temporal if enabled
        if self.cyclical_transformer:
            X_cyc = self.cyclical_transformer.transform(X_aligned)
            return np.hstack([X_core, X_cyc])

        return X_core

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        return self.feature_names_out_

