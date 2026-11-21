"""
Comprehensive Metrics Evaluator and Visualization Module.
Evaluates competing models on holdout test data, supports dual all-hours/daytime metrics,
and generates diagnostic publication plots.
"""
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import DAYTIME_ELEVATION_THRESHOLD, PLOTS_DIR, METRICS_DIR
from src.utils import setup_logger, calculate_comprehensive_metrics, apply_physical_bounds, save_json

logger = setup_logger("metrics_evaluator")


class MetricsEvaluator:
    """Evaluates models on test partitions with dual daytime/all-hours metrics and diagnostic figures."""

    def __init__(
        self,
        rated_capacity: Optional[float] = None,
        target_unit: str = "capacity_factor",
        elevation_col: Optional[str] = "solar_elevation_deg"
    ):
        self.rated_capacity = rated_capacity
        self.target_unit = target_unit
        self.elevation_col = elevation_col

    def evaluate_test_set(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_df: pd.DataFrame
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Evaluate all candidate models on the holdout test set.
        """
        logger.info(f"--- Evaluating Models on Holdout Test Set ({len(y_test)} samples) ---")

        # Check daytime mask
        daytime_mask = np.ones(len(y_test), dtype=bool)
        solar_elev = None
        if self.elevation_col and self.elevation_col in test_df.columns:
            solar_elev = test_df[self.elevation_col].to_numpy(dtype=float)
            daytime_mask = solar_elev > DAYTIME_ELEVATION_THRESHOLD
            logger.info(f"Daytime filter (>{DAYTIME_ELEVATION_THRESHOLD}°): {int(daytime_mask.sum())} / {len(y_test)} rows.")

        results = {
            "all_hours": {},
            "daytime_only": {},
        }

        preds_dict = {
            "actual": y_test,
        }
        if self.elevation_col and self.elevation_col in test_df.columns:
            preds_dict[self.elevation_col] = solar_elev

        for name, model in models.items():
            raw_preds = model.predict(X_test)
            # Physical bounds: non-negative and nocturnal zeroing
            clean_preds = apply_physical_bounds(raw_preds, solar_elev, elevation_threshold=0.0)

            # Store in prediction table
            preds_dict[f"{name}_pred"] = clean_preds
            preds_dict[f"{name}_residual"] = clean_preds - y_test

            # 1. All-Hours Metrics
            all_m = calculate_comprehensive_metrics(y_test, clean_preds, rated_capacity=self.rated_capacity)
            results["all_hours"][name] = all_m

            # 2. Daytime Metrics
            day_m = calculate_comprehensive_metrics(y_test[daytime_mask], clean_preds[daytime_mask], rated_capacity=self.rated_capacity)
            results["daytime_only"][name] = day_m

            logger.info(
                f"[{name}] Test All-Hours R²: {all_m['r2']} (MAE: {all_m['mae']}) | "
                f"Daytime R²: {day_m['r2']} (MAE: {day_m['mae']})"
            )

        predictions_df = pd.DataFrame(preds_dict)
        return results, predictions_df

    def generate_diagnostic_plots(
        self,
        test_metrics: Dict[str, Any],
        val_metrics: Dict[str, Dict[str, float]],
        predictions_df: pd.DataFrame,
        winner_name: str,
        plots_dir: Optional[Path] = None
    ) -> None:
        """Generate model comparison and residual diagnostic figures."""
        out_dir = plots_dir or PLOTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

        # 1. Model Comparison Bar Chart (Validation vs Test R² and MAE)
        comp_rows = []
        for name in test_metrics["all_hours"].keys():
            v_r2 = val_metrics.get(name, {}).get("r2", np.nan)
            v_mae = val_metrics.get(name, {}).get("mae", np.nan)
            t_r2 = test_metrics["all_hours"][name]["r2"]
            t_mae = test_metrics["all_hours"][name]["mae"]
            comp_rows.append({"Model": name, "Split": "Validation", "R²": v_r2, "MAE": v_mae})
            comp_rows.append({"Model": name, "Split": "Test", "R²": t_r2, "MAE": t_mae})

        df_comp = pd.DataFrame(comp_rows)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.barplot(data=df_comp, x="Model", y="R²", hue="Split", ax=axes[0], palette="muted")
        axes[0].set_title("Model Comparison: R² Score (Validation vs Test)", fontsize=12, fontweight="bold")
        axes[0].tick_params(axis="x", rotation=15)

        sns.barplot(data=df_comp, x="Model", y="MAE", hue="Split", ax=axes[1], palette="muted")
        axes[1].set_title(f"Model Comparison: MAE ({self.target_unit})", fontsize=12, fontweight="bold")
        axes[1].tick_params(axis="x", rotation=15)
        plt.tight_layout()
        plt.savefig(out_dir / "adaptive_model_comparison.png", dpi=300)
        plt.close()

        # 2. Winner Actual vs Predicted Hexbin
        pred_col = f"{winner_name}_pred"
        if pred_col in predictions_df.columns:
            plt.figure(figsize=(7, 6))
            hb = plt.hexbin(predictions_df["actual"], predictions_df[pred_col], gridsize=35, cmap="YlGnBu", mincnt=1)
            cb = plt.colorbar(hb)
            cb.set_label("Sample Count")

            max_val = max(predictions_df["actual"].max(), predictions_df[pred_col].max())
            plt.plot([0, max_val], [0, max_val], color="red", linestyle="--", lw=2, label="1:1 Perfect Prediction")
            plt.title(f"Winner ({winner_name}): Actual vs Predicted\nTest R² = {test_metrics['all_hours'][winner_name]['r2']}", fontsize=12, fontweight="bold")
            plt.xlabel(f"Actual {self.target_unit}")
            plt.ylabel(f"Predicted {self.target_unit}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "winner_actual_vs_predicted.png", dpi=300)
            plt.close()

        # 3. Residual Error Distribution
        resid_col = f"{winner_name}_residual"
        if resid_col in predictions_df.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(predictions_df[resid_col], kde=True, bins=35, color="#1f77b4")
            plt.axvline(0, color="red", linestyle="--", lw=1.5)
            plt.title(f"Winner ({winner_name}): Residual Error Distribution (Test Set)", fontsize=12, fontweight="bold")
            plt.xlabel(f"Residual Error ({self.target_unit})")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(out_dir / "winner_residuals.png", dpi=300)
            plt.close()

        logger.info(f"Saved diagnostic plots to {out_dir}")

