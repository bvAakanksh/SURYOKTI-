"""
Adaptive Inference and Prediction Script for SURYOKTI Adaptive Architecture.
Loads an AdaptiveInferenceBundle, accepts arbitrary weather data, tolerates missing
and extraneous features, enforces solar physical bounds, and writes predictions.
"""
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import MODELS_DIR
from src.utils import setup_logger, load_artifact, load_json, apply_physical_bounds

logger = setup_logger("predict")


class AdaptivePredictor:
    """Production predictor wrapping an AdaptiveInferenceBundle."""

    def __init__(self, bundle_path: Optional[Path] = None):
        if bundle_path is None:
            bundle_path = MODELS_DIR / "adaptive_model_bundle.joblib"

        if not bundle_path.exists():
            raise FileNotFoundError(f"Adaptive model bundle not found at: {bundle_path}")

        logger.info(f"Loading AdaptiveInferenceBundle from {bundle_path}...")
        self.bundle = load_artifact(bundle_path)
        self.metadata = self.bundle.metadata
        self.target_name = self.metadata.get("target_column", "predicted_target")
        self.elevation_col = self.metadata.get("elevation_column", "solar_elevation_deg")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for input DataFrame.
        Resilient to extra columns, missing non-critical features, and enforces solar physics.
        """
        results_df = df.copy()

        # Execute prediction through bundle
        raw_preds = self.bundle.predict(df)

        # Solar elevation for nocturnal zeroing if available
        solar_elev = None
        if self.elevation_col and self.elevation_col in df.columns:
            solar_elev = df[self.elevation_col].to_numpy(dtype=float)

        # Apply physical solar bounds
        clean_preds = apply_physical_bounds(raw_preds, solar_elevation=solar_elev, elevation_threshold=0.0)

        pred_col_name = f"predicted_{self.target_name}"
        results_df[pred_col_name] = np.round(clean_preds, 5)

        logger.info(f"Generated {len(results_df)} predictions successfully.")
        return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict solar generation using trained adaptive bundle.")
    parser.add_argument("--model", type=str, default=None, help="Path to adaptive_model_bundle.joblib.")
    parser.add_argument("--input", type=str, required=True, help="Path to input dataset CSV.")
    parser.add_argument("--output", type=str, default=None, help="Path to save prediction CSV.")
    args = parser.parse_args()

    model_p = Path(args.model) if args.model else None
    predictor = AdaptivePredictor(bundle_path=model_p)

    input_df = pd.read_csv(args.input)
    preds = predictor.predict(input_df)

    if args.output:
        preds.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
    else:
        print(preds.head())

