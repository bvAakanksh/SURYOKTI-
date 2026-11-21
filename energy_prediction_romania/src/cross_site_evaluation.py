"""
Cross-Site Generalization Evaluation Script for SURYOKTI Romania PV Implementation.
Tests transferability of models between geographically distinct PV installations:
- Transfer Experiment 1: Model trained on Site A evaluated on Site B
- Transfer Experiment 2: Model trained on Site B evaluated on Site A
Measures domain shift, zero-shot transferability, and degradation.
"""
import sys
from pathlib import Path
from typing import Dict, Any

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
    PLOTS_CROSS_SITE_DIR,
    REPORTS_DIR,
    PRIMARY_TARGET,
    RATED_CAPACITY_KW,
    SOLAR_ELEVATION_DAYTIME_THRESHOLD,
)
from src.utils import (
    setup_logger,
    load_artifact,
    save_json,
    load_json,
    calculate_metrics,
    apply_physical_constraints,
    cf_to_kw,
)
from src.data_split import split_chronological

logger = setup_logger("cross_site_evaluation", REPORTS_DIR / "cross_site_evaluation.log")


def evaluate_transfer(
    source_site: str,
    target_site: str,
    source_model_dir: Path,
    target_processed_path: Path,
    use_target_test_only: bool = True
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluate model trained on source_site against target_site data."""
    logger.info(f"--- Evaluating Cross-Site Transfer: {source_site} -> {target_site} ---")

    target_df = pd.read_csv(target_processed_path)
    if use_target_test_only:
        _, _, eval_df, _ = split_chronological(target_df)
    else:
        eval_df = target_df.copy()

    # Load source model artifacts
    source_pipeline = load_artifact(source_model_dir / "preprocessing_pipeline.joblib")
    source_stack = load_artifact(source_model_dir / "stacking_ensemble.joblib")

    X_target = source_pipeline.transform(eval_df)
    y_target_cf = eval_df[PRIMARY_TARGET].to_numpy(dtype=float)
    y_target_kw = cf_to_kw(y_target_cf, RATED_CAPACITY_KW)
    elev_target = eval_df["solar_elevation_deg"].to_numpy(dtype=float)

    # Predict with physical constraints
    pred_raw = source_stack.predict(X_target)
    pred_cf = apply_physical_constraints(pred_raw, elev_target, elevation_threshold=0.0)
    pred_kw = cf_to_kw(pred_cf, RATED_CAPACITY_KW)

    # Daytime mask
    daytime_mask = elev_target > SOLAR_ELEVATION_DAYTIME_THRESHOLD

    all_metrics = calculate_metrics(y_target_cf, pred_cf, rated_capacity_kw=RATED_CAPACITY_KW, prefix="all_hours")
    day_metrics = calculate_metrics(y_target_cf[daytime_mask], pred_cf[daytime_mask], rated_capacity_kw=RATED_CAPACITY_KW, prefix="daytime")

    transfer_results = {
        "source_site": source_site,
        "target_site": target_site,
        "sample_count": len(y_target_cf),
        "daytime_sample_count": int(daytime_mask.sum()),
        "all_hours_metrics": all_metrics,
        "daytime_metrics": day_metrics,
    }

    logger.info(
        f"Transfer {source_site} -> {target_site} | "
        f"All-Hours R2: {all_metrics['all_hours_r2']} (MAE: {all_metrics['all_hours_mae_kw']} kW) | "
        f"Daytime R2: {day_metrics['daytime_r2']} (MAE: {day_metrics['daytime_mae_kw']} kW)"
    )

    pred_df = pd.DataFrame({
        "datetime_utc": eval_df["datetime_utc"],
        "solar_elevation_deg": elev_target,
        "ghi_wm2": eval_df["ghi_wm2"],
        "actual_cf": y_target_cf,
        "actual_kw": y_target_kw,
        "transfer_pred_cf": pred_cf,
        "transfer_pred_kw": pred_kw,
        "transfer_residual_kw": pred_kw - y_target_kw,
        "is_daytime": daytime_mask,
    })

    return transfer_results, pred_df


def generate_cross_site_plots(
    res_a_to_b: Dict[str, Any],
    pred_a_to_b: pd.DataFrame,
    res_b_to_a: Dict[str, Any],
    pred_b_to_a: pd.DataFrame,
    in_domain_a: Dict[str, Any],
    in_domain_b: Dict[str, Any]
) -> None:
    """Generate diagnostic visualization for cross-site transferability."""
    logger.info("Generating cross-site transfer diagnostic plots...")
    sns.set_theme(style="whitegrid")

    # 1. In-Domain vs Transfer Performance Bar Chart
    transfer_comp = [
        {"Case": "Site A In-Domain (A->A)", "R2": in_domain_a["all_hours"]["Stacking Ensemble"]["all_hours_r2"], "MAE_kW": in_domain_a["all_hours"]["Stacking Ensemble"]["all_hours_mae_kw"], "Type": "In-Domain"},
        {"Case": "Transfer B to A (B->A)", "R2": res_b_to_a["all_hours_metrics"]["all_hours_r2"], "MAE_kW": res_b_to_a["all_hours_metrics"]["all_hours_mae_kw"], "Type": "Cross-Site Transfer"},
        {"Case": "Site B In-Domain (B->B)", "R2": in_domain_b["all_hours"]["Stacking Ensemble"]["all_hours_r2"], "MAE_kW": in_domain_b["all_hours"]["Stacking Ensemble"]["all_hours_mae_kw"], "Type": "In-Domain"},
        {"Case": "Transfer A to B (A->B)", "R2": res_a_to_b["all_hours_metrics"]["all_hours_r2"], "MAE_kW": res_a_to_b["all_hours_metrics"]["all_hours_mae_kw"], "Type": "Cross-Site Transfer"},
    ]
    tc_df = pd.DataFrame(transfer_comp)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(data=tc_df, x="Case", y="R2", hue="Type", ax=axes[0], palette=["#1f77b4", "#ff7f0e"])
    axes[0].set_title("Cross-Site Generalization: R² Score", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0.6, 1.0)
    axes[0].tick_params(axis="x", rotation=15)

    sns.barplot(data=tc_df, x="Case", y="MAE_kW", hue="Type", ax=axes[1], palette=["#1f77b4", "#ff7f0e"])
    axes[1].set_title("Cross-Site Generalization: MAE (kW)", fontsize=12, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(PLOTS_CROSS_SITE_DIR / "cross_site_transfer_comparison.png", dpi=300)
    plt.close()

    # 2. Hexbin / Scatter of Cross-Site Predictions
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, pred_df, title, metrics in zip(
        axes,
        [pred_a_to_b, pred_b_to_a],
        ["Transfer: Model A -> Target Site B", "Transfer: Model B -> Target Site A"],
        [res_a_to_b, res_b_to_a]
    ):
        day_df = pred_df[pred_df["is_daytime"]].copy()
        hb = ax.hexbin(day_df["actual_kw"], day_df["transfer_pred_kw"], gridsize=35, cmap="magma", mincnt=1)
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)

        ax.plot([0, 55], [0, 55], color="cyan", linestyle="--", lw=2, label="1:1 Perfect")
        r2_val = metrics["daytime_metrics"]["daytime_r2"]
        mae_val = metrics["daytime_metrics"]["daytime_mae_kw"]
        ax.set_title(f"{title}\nDaytime R²: {r2_val:.4f} | MAE: {mae_val:.2f} kW", fontsize=12, fontweight="bold")
        ax.set_xlabel("Target Actual Active Power (kW)", fontsize=11)
        ax.set_xlim(-1, 55)
        ax.set_ylim(-1, 55)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Transfer Predicted Active Power (kW)", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_CROSS_SITE_DIR / "cross_site_scatter.png", dpi=300)
    plt.close()

    # 3. Residual Error Distributions
    plt.figure(figsize=(10, 5))
    sns.kdeplot(pred_a_to_b[pred_a_to_b["is_daytime"]]["transfer_residual_kw"], label="Model A on Site B", color="#1f77b4", lw=2)
    sns.kdeplot(pred_b_to_a[pred_b_to_a["is_daytime"]]["transfer_residual_kw"], label="Model B on Site A", color="#ff7f0e", lw=2)
    plt.axvline(0, color="red", linestyle="--", alpha=0.7)
    plt.title("Cross-Site Transfer Daytime Error Distribution (Residuals)", fontsize=13, fontweight="bold")
    plt.xlabel("Transfer Residual (Predicted - Actual) [kW]", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_CROSS_SITE_DIR / "cross_site_residuals.png", dpi=300)
    plt.close()

    logger.info("Cross-site transfer plots saved successfully.")


def run_cross_site():
    """Run bidirectional cross-site evaluation."""
    PLOTS_CROSS_SITE_DIR.mkdir(parents=True, exist_ok=True)

    # Load in-domain metrics for comparison
    in_domain_a = load_json(METRICS_DIR / "site_a_metrics.json")
    in_domain_b = load_json(METRICS_DIR / "site_b_metrics.json")

    # Transfer 1: Model A -> Site B
    res_a_to_b, pred_a_to_b = evaluate_transfer(
        source_site="Site A",
        target_site="Site B",
        source_model_dir=MODELS_SITE_A_DIR,
        target_processed_path=DATA_PROCESSED_DIR / "site_b_processed.csv",
        use_target_test_only=True
    )
    save_json(res_a_to_b, METRICS_DIR / "cross_site_A_to_B.json")
    pred_a_to_b.to_csv(PREDICTIONS_DIR / "cross_site_A_to_B_predictions.csv", index=False)

    # Transfer 2: Model B -> Site A
    res_b_to_a, pred_b_to_a = evaluate_transfer(
        source_site="Site B",
        target_site="Site A",
        source_model_dir=MODELS_SITE_B_DIR,
        target_processed_path=DATA_PROCESSED_DIR / "site_a_processed.csv",
        use_target_test_only=True
    )
    save_json(res_b_to_a, METRICS_DIR / "cross_site_B_to_A.json")
    pred_b_to_a.to_csv(PREDICTIONS_DIR / "cross_site_B_to_A_predictions.csv", index=False)

    # Save summary
    cross_summary = {
        "transfer_A_to_B": res_a_to_b,
        "transfer_B_to_A": res_b_to_a,
    }
    save_json(cross_summary, METRICS_DIR / "cross_site_summary.json")

    # Generate plots
    generate_cross_site_plots(res_a_to_b, pred_a_to_b, res_b_to_a, pred_b_to_a, in_domain_a, in_domain_b)
    logger.info("Cross-site evaluation finished successfully.")
    return cross_summary


if __name__ == "__main__":
    run_cross_site()
