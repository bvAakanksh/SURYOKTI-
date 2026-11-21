"""
Inference and Prediction Pipeline for SURYOKTI Romania PV Implementation.
Loads trained stacking model artifacts and preprocessing pipelines to generate
physically constrained solar capacity factor and active power (kW) forecasts.
"""
import sys
import argparse
from pathlib import Path
from typing import Optional, Union

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import (
    MODELS_SITE_A_DIR,
    MODELS_SITE_B_DIR,
    DATA_PROCESSED_DIR,
    RATED_CAPACITY_KW,
    FEATURE_COLS,
)
from src.utils import (
    setup_logger,
    load_artifact,
    load_json,
    encode_cyclical,
    apply_physical_constraints,
    cf_to_kw,
)

logger = setup_logger("predict")


class SolarEnergyPredictor:
    """End-to-end predictor for Romanian PV sites."""

    def __init__(self, site: str = "site_a"):
        self.site = site.lower()
        if self.site == "site_a":
            self.model_dir = MODELS_SITE_A_DIR
        elif self.site == "site_b":
            self.model_dir = MODELS_SITE_B_DIR
        else:
            raise ValueError(f"Unknown site '{site}'. Choose 'site_a' or 'site_b'.")

        self.model_path = self.model_dir / "stacking_ensemble.joblib"
        self.pipeline_path = self.model_dir / "preprocessing_pipeline.joblib"
        self.feature_meta_path = self.model_dir / "feature_list.json"

        logger.info(f"Loading predictor artifacts for {self.site} from {self.model_dir}...")
        self.model = load_artifact(self.model_path)
        self.pipeline = load_artifact(self.pipeline_path)
        self.feature_cols = load_json(self.feature_meta_path)["feature_columns"]

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure cyclical encodings and all required features are present."""
        df_feat = df.copy()

        # Check if cyclical features are missing; if so, derive from datetime_utc
        if "hour_sin" not in df_feat.columns:
            if "datetime_utc" not in df_feat.columns:
                raise KeyError("Input dataframe must contain 'datetime_utc' column to derive temporal features.")
            dt = pd.to_datetime(df_feat["datetime_utc"], utc=True)
            df_feat["hour_sin"], df_feat["hour_cos"] = encode_cyclical(dt.dt.hour, 24.0)
            df_feat["day_of_year_sin"], df_feat["day_of_year_cos"] = encode_cyclical(dt.dt.dayofyear - 1, 365.25)
            df_feat["month_sin"], df_feat["month_cos"] = encode_cyclical(dt.dt.month - 1, 12.0)

        # Validate feature presence
        missing = [c for c in self.feature_cols if c not in df_feat.columns]
        if missing:
            raise KeyError(f"Input dataframe missing required features: {missing}")

        return df_feat

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate physically constrained predictions for input weather data.
        Returns dataframe with predicted_capacity_factor and predicted_active_power_kw.
        """
        df_prepared = self._prepare_features(df)
        X_scaled = self.pipeline.transform(df_prepared)

        # Stacking prediction
        pred_raw = self.model.predict(X_scaled)

        # Apply physical solar constraints
        solar_elev = df_prepared["solar_elevation_deg"].to_numpy(dtype=float) if "solar_elevation_deg" in df_prepared.columns else None
        pred_cf = apply_physical_constraints(pred_raw, solar_elev, elevation_threshold=0.0)
        pred_kw = cf_to_kw(pred_cf, rated_capacity_kw=RATED_CAPACITY_KW)

        results = df.copy()
        results["predicted_capacity_factor"] = np.round(pred_cf, 5)
        results["predicted_active_power_kw"] = np.round(pred_kw, 3)

        return results


def run_test_inference():
    """Run verification inference on 5 test rows from Site A."""
    logger.info("Executing sample inference test on Site A...")
    test_path = DATA_PROCESSED_DIR / "site_a_processed.csv"
    df = pd.read_csv(test_path).iloc[-5:].copy().reset_index(drop=True)

    predictor = SolarEnergyPredictor(site="site_a")
    results = predictor.predict(df)

    cols_to_display = ["datetime_utc", "temp_c", "ghi_wm2", "solar_elevation_deg", "capacity_factor", "predicted_capacity_factor", "active_power_kw", "predicted_active_power_kw"]
    logger.info("Sample Inference Output:\n" + results[cols_to_display].to_string())
    print("\n=== SAMPLE INFERENCE RESULTS ===")
    print(results[cols_to_display].to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PV power predictions using trained models.")
    parser.add_argument("--site", type=str, default="site_a", choices=["site_a", "site_b"], help="PV site model to use.")
    parser.add_argument("--input", type=str, default=None, help="Path to input weather CSV.")
    parser.add_argument("--output", type=str, default=None, help="Path to save output prediction CSV.")
    args = parser.parse_args()

    if args.input:
        predictor = SolarEnergyPredictor(site=args.site)
        input_df = pd.read_csv(args.input)
        preds = predictor.predict(input_df)
        if args.output:
            preds.to_csv(args.output, index=False)
            logger.info(f"Predictions saved to {args.output}")
        else:
            print(preds.head())
    else:
        run_test_inference()

