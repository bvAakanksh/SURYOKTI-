"""
Model Evaluation and Diagnostic Plotting Script for SURYOKTI Romania PV Implementation.
Evaluates standalone RF, XGBoost, and Stacking Ensemble models on the test set.
Computes dual metrics (All-Hours and Daytime-Only) in both capacity factor and kW units.
Generates comprehensive visual diagnostic figures.
"""
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DATA_PROCESSED_DIR,
    MODELS_SITE_A_DIR,
    MODELS_SITE_B_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
    PLOTS_MODEL_EVAL_DIR,
    REPORTS_DIR,
    FEATURE_COLS,
    PRIMARY_TARGET,
    RATED_CAPACITY_KW,
    SOLAR_ELEVATION_DAYTIME_THRESHOLD,
)
from src.utils import (
    setup_logger,
    load_artifact,
    save_json,
    calculate_metrics,
    apply_physical_constraints,
    cf_to_kw,
)
from src.data_split import split_chronological

logger = setup_logger("evaluate_model", REPORTS_DIR / "evaluate_model.log")


def evaluate_site_models(site_name: str, processed_path: Path, model_dir: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluate RF, XGBoost, and Stacking models for a site on chronological test set."""
    logger.info(f"=== Evaluating Models for {site_name} ===")

    df = pd.read_csv(processed_path)
    _, _, test_df, _ = split_chronological(df)

    # Load artifacts
    pipeline = load_artifact(model_dir / "preprocessing_pipeline.joblib")
    rf_model = load_artifact(model_dir / "rf_model.joblib")
    xgb_model = load_artifact(model_dir / "xgb_model.joblib")
    stack_model = load_artifact(model_dir / "stacking_ensemble.joblib")

    X_test = pipeline.transform(test_df)
    y_test_cf = test_df[PRIMARY_TARGET].to_numpy(dtype=float)
    y_test_kw = cf_to_kw(y_test_cf, RATED_CAPACITY_KW)
    elev_test = test_df["solar_elevation_deg"].to_numpy(dtype=float)

    # 1. Predictions
    rf_pred_raw = rf_model.predict(X_test)
    xgb_pred_raw = xgb_model.predict(X_test)
    stack_pred_raw = stack_model.predict(X_test)

    # Apply physical constraints (clip to [0, 1] and zero out below solar elevation threshold)
    rf_pred_cf = apply_physical_constraints(rf_pred_raw, elev_test, elevation_threshold=0.0)
    xgb_pred_cf = apply_physical_constraints(xgb_pred_raw, elev_test, elevation_threshold=0.0)
    stack_pred_cf = apply_physical_constraints(stack_pred_raw, elev_test, elevation_threshold=0.0)

    rf_pred_kw = cf_to_kw(rf_pred_cf, RATED_CAPACITY_KW)
    xgb_pred_kw = cf_to_kw(xgb_pred_cf, RATED_CAPACITY_KW)
    stack_pred_kw = cf_to_kw(stack_pred_cf, RATED_CAPACITY_KW)

    # Daytime mask (solar elevation > 5°)
    daytime_mask = elev_test > SOLAR_ELEVATION_DAYTIME_THRESHOLD

    # Compute metrics for all three models
    models = {
        "Random Forest": (rf_pred_cf, rf_pred_kw),
        "XGBoost": (xgb_pred_cf, xgb_pred_kw),
        "Stacking Ensemble": (stack_pred_cf, stack_pred_kw),
    }

    site_metrics = {
        "site": site_name,
        "test_sample_count": len(y_test_cf),
        "daytime_sample_count": int(daytime_mask.sum()),
        "all_hours": {},
        "daytime_only": {},
    }

    for model_name, (pred_cf, pred_kw) in models.items():
        # All Hours
        all_metrics = calculate_metrics(y_test_cf, pred_cf, rated_capacity_kw=RATED_CAPACITY_KW, prefix="all_hours")
        site_metrics["all_hours"][model_name] = all_metrics

        # Daytime Only
        day_metrics = calculate_metrics(
            y_test_cf[daytime_mask],
            pred_cf[daytime_mask],
            rated_capacity_kw=RATED_CAPACITY_KW,
            prefix="daytime"
        )
        site_metrics["daytime_only"][model_name] = day_metrics

        logger.info(
            f"[{site_name}] {model_name} -> "
            f"All-Hours R2: {all_metrics['all_hours_r2']} (MAE: {all_metrics['all_hours_mae_kw']} kW) | "
            f"Daytime R2: {day_metrics['daytime_r2']} (MAE: {day_metrics['daytime_mae_kw']} kW)"
        )

    # Build predictions dataframe
    pred_df = pd.DataFrame({
        "datetime_utc": test_df["datetime_utc"],
        "solar_elevation_deg": elev_test,
        "ghi_wm2": test_df["ghi_wm2"],
        "temp_c": test_df["temp_c"],
        "actual_cf": y_test_cf,
        "actual_kw": y_test_kw,
        "rf_pred_cf": rf_pred_cf,
        "rf_pred_kw": rf_pred_kw,
        "xgb_pred_cf": xgb_pred_cf,
        "xgb_pred_kw": xgb_pred_kw,
        "stack_pred_cf": stack_pred_cf,
        "stack_pred_kw": stack_pred_kw,
        "stack_residual_kw": stack_pred_kw - y_test_kw,
        "is_daytime": daytime_mask,
    })

    return site_metrics, pred_df


def generate_evaluation_plots(
    pred_df_a: pd.DataFrame,
    metrics_a: Dict[str, Any],
    pred_df_b: pd.DataFrame,
    metrics_b: Dict[str, Any]
) -> None:
    """Generate extensive publication-grade diagnostic plots."""
    logger.info("Generating evaluation diagnostic plots...")
    sns.set_theme(style="whitegrid")

    # 1. Actual vs Predicted Power (Hexbin / Scatter with 1:1 line)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, pred_df, site_name, metrics in zip(axes, [pred_df_a, pred_df_b], ["Site A", "Site B"], [metrics_a, metrics_b]):
        day_df = pred_df[pred_df["is_daytime"]].copy()
        r2_val = metrics["daytime_only"]["Stacking Ensemble"]["daytime_r2"]
        mae_val = metrics["daytime_only"]["Stacking Ensemble"]["daytime_mae_kw"]

        hb = ax.hexbin(day_df["actual_kw"], day_df["stack_pred_kw"], gridsize=35, cmap="YlGnBu", mincnt=1)
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Count", fontsize=10)

        # 1:1 perfect prediction line
        ax.plot([0, 55], [0, 55], color="red", linestyle="--", lw=2, label="Perfect Forecast (1:1)")
        ax.set_title(f"{site_name}: Daytime Actual vs Predicted Power\nStacking R²: {r2_val:.4f} | MAE: {mae_val:.2f} kW", fontsize=12, fontweight="bold")
        ax.set_xlabel("Actual Active Power (kW)", fontsize=11)
        ax.set_xlim(-1, 55)
        ax.set_ylim(-1, 55)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Predicted Active Power (kW)", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_MODEL_EVAL_DIR / "actual_vs_predicted.png", dpi=300)
    plt.close()

    # 2. Residual Distribution and Residual vs Predicted
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for col_idx, (pred_df, site_name) in enumerate(zip([pred_df_a, pred_df_b], ["Site A", "Site B"])):
        day_df = pred_df[pred_df["is_daytime"]].copy()
        residuals = day_df["stack_residual_kw"]

        # Residuals vs Predicted
        axes[0, col_idx].scatter(day_df["stack_pred_kw"], residuals, alpha=0.35, s=15, color="#1f77b4" if col_idx == 0 else "#ff7f0e")
        axes[0, col_idx].axhline(0, color="red", linestyle="--", lw=1.5)
        axes[0, col_idx].set_title(f"{site_name}: Residuals vs Predicted Power", fontsize=12, fontweight="bold")
        axes[0, col_idx].set_xlabel("Predicted Active Power (kW)", fontsize=11)
        axes[0, col_idx].set_ylabel("Residual (Pred - Actual) [kW]", fontsize=11)

        # Residual Distribution (Histogram + KDE)
        sns.histplot(residuals, kde=True, ax=axes[1, col_idx], color="#1f77b4" if col_idx == 0 else "#ff7f0e", bins=35)
        axes[1, col_idx].axvline(0, color="red", linestyle="--", lw=1.5)
        axes[1, col_idx].set_title(f"{site_name}: Residual Error Distribution (Daytime)", fontsize=12, fontweight="bold")
        axes[1, col_idx].set_xlabel("Error (kW)", fontsize=11)
        axes[1, col_idx].set_ylabel("Frequency", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_MODEL_EVAL_DIR / "residual_analysis.png", dpi=300)
    plt.close()

    # 3. Multi-day Time-Series Overlay (7-Day Slice)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=False)
    for ax, pred_df, site_name in zip(axes, [pred_df_a, pred_df_b], ["Site A", "Site B"]):
        pred_df["dt"] = pd.to_datetime(pred_df["datetime_utc"])
        # Take a clean 7-day window (168 consecutive hours)
        sample_slice = pred_df.iloc[100:100+168].copy()

        ax.plot(sample_slice["dt"], sample_slice["actual_kw"], label="Actual Generation", color="black", lw=2.2)
        ax.plot(sample_slice["dt"], sample_slice["rf_pred_kw"], label="Random Forest", color="#1f77b4", linestyle="--", alpha=0.8)
        ax.plot(sample_slice["dt"], sample_slice["xgb_pred_kw"], label="XGBoost", color="#ff7f0e", linestyle=":", alpha=0.8)
        ax.plot(sample_slice["dt"], sample_slice["stack_pred_kw"], label="Stacking Ensemble", color="#2ca02c", lw=2)

        ax.set_title(f"{site_name}: 7-Day Continuous Forecast Overlay (Hourly)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Active Power (kW)", fontsize=11)
        ax.set_ylim(-1, 55)
        ax.legend(loc="upper right", frameon=True)
    axes[1].set_xlabel("Timestamp (UTC)", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_MODEL_EVAL_DIR / "time_series_overlay.png", dpi=300)
    plt.close()

    # 4. Model Comparison Bar Chart (RF vs XGBoost vs Stacking)
    comp_data = []
    for site_name, metrics in zip(["Site A", "Site B"], [metrics_a, metrics_b]):
        for model in ["Random Forest", "XGBoost", "Stacking Ensemble"]:
            all_r2 = metrics["all_hours"][model]["all_hours_r2"]
            all_mae = metrics["all_hours"][model]["all_hours_mae_kw"]
            all_rmse = metrics["all_hours"][model]["all_hours_rmse_kw"]
            day_r2 = metrics["daytime_only"][model]["daytime_r2"]
            day_mae = metrics["daytime_only"][model]["daytime_mae_kw"]
            day_rmse = metrics["daytime_only"][model]["daytime_rmse_kw"]
            comp_data.append({
                "Site": site_name,
                "Model": model,
                "All_Hours_R2": all_r2,
                "All_Hours_MAE_kW": all_mae,
                "All_Hours_RMSE_kW": all_rmse,
                "Daytime_R2": day_r2,
                "Daytime_MAE_kW": day_mae,
                "Daytime_RMSE_kW": day_rmse,
            })
    comp_df = pd.DataFrame(comp_data)
    comp_df.to_csv(METRICS_DIR / "model_comparison_table.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=comp_df, x="Site", y="All_Hours_R2", hue="Model", ax=axes[0], palette="muted")
    axes[0].set_title("All-Hours Model R² Comparison", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0.7, 1.0)
    axes[0].set_ylabel("R² Score", fontsize=11)

    sns.barplot(data=comp_df, x="Site", y="All_Hours_MAE_kW", hue="Model", ax=axes[1], palette="muted")
    axes[1].set_title("All-Hours Model MAE Comparison (kW)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Mean Absolute Error (kW)", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_MODEL_EVAL_DIR / "model_comparison_bar.png", dpi=300)
    plt.close()

    # 5. Daytime vs All-Hours Metric Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    day_vs_all = []
    for site_name, metrics in zip(["Site A", "Site B"], [metrics_a, metrics_b]):
        day_vs_all.append({"Site": site_name, "Scope": "All-Hours (24h)", "R2": metrics["all_hours"]["Stacking Ensemble"]["all_hours_r2"], "MAE_kW": metrics["all_hours"]["Stacking Ensemble"]["all_hours_mae_kw"]})
        day_vs_all.append({"Site": site_name, "Scope": "Daytime Only (>5°)", "R2": metrics["daytime_only"]["Stacking Ensemble"]["daytime_r2"], "MAE_kW": metrics["daytime_only"]["Stacking Ensemble"]["daytime_mae_kw"]})
    dva_df = pd.DataFrame(day_vs_all)

    sns.barplot(data=dva_df, x="Site", y="R2", hue="Scope", ax=axes[0], palette="Set2")
    axes[0].set_title("Stacking Model R²: All-Hours vs Daytime", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0.7, 1.0)

    sns.barplot(data=dva_df, x="Site", y="MAE_kW", hue="Scope", ax=axes[1], palette="Set2")
    axes[1].set_title("Stacking Model MAE (kW): All-Hours vs Daytime", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_MODEL_EVAL_DIR / "daytime_vs_all_hours.png", dpi=300)
    plt.close()

    # 6. Generation-Level Error Across Power Deciles
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, pred_df, site_name in zip(axes, [pred_df_a, pred_df_b], ["Site A", "Site B"]):
        day_df = pred_df[pred_df["is_daytime"]].copy()
        bins = [0, 10, 20, 30, 40, 55]
        labels = ["0-10 kW", "10-20 kW", "20-30 kW", "30-40 kW", "40-55 kW"]
        day_df["power_bin"] = pd.cut(day_df["actual_kw"], bins=bins, labels=labels, include_lowest=True)
        bin_mae = day_df.groupby("power_bin", observed=False)["stack_residual_kw"].apply(lambda x: np.mean(np.abs(x)))

        sns.barplot(x=bin_mae.index, y=bin_mae.values, ax=ax, palette="Blues_d")
        ax.set_title(f"{site_name}: MAE Across Generation Levels (Daytime)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Actual Power Tier", fontsize=11)
        ax.set_ylabel("MAE (kW)", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_MODEL_EVAL_DIR / "generation_level_error.png", dpi=300)
    plt.close()

    logger.info("All model evaluation diagnostic plots saved successfully.")


def run_evaluation():
    """Run full evaluation on both Site A and Site B."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_MODEL_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    metrics_a, pred_a = evaluate_site_models("Site A", DATA_PROCESSED_DIR / "site_a_processed.csv", MODELS_SITE_A_DIR)
    metrics_b, pred_b = evaluate_site_models("Site B", DATA_PROCESSED_DIR / "site_b_processed.csv", MODELS_SITE_B_DIR)

    # Save metrics JSON
    save_json(metrics_a, METRICS_DIR / "site_a_metrics.json")
    save_json(metrics_b, METRICS_DIR / "site_b_metrics.json")

    # Save test predictions CSV
    pred_a.to_csv(PREDICTIONS_DIR / "site_a_test_predictions.csv", index=False)
    pred_b.to_csv(PREDICTIONS_DIR / "site_b_test_predictions.csv", index=False)
    logger.info("Saved test prediction CSV files.")

    # Generate diagnostic plots
    generate_evaluation_plots(pred_a, metrics_a, pred_b, metrics_b)
    logger.info("Model evaluation completed successfully.")
    return metrics_a, metrics_b


if __name__ == "__main__":
    run_evaluation()

