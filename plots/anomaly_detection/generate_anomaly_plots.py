"""
SURYOKTI Anomaly Detection — Diagnostic Visualizations Suite
Generates calibrated anomaly timelines, residual distributions, and
operational status reconciliation charts.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── 1. Anomaly Reconciliation Pie Chart & Residual Distribution ──────────────
anom_candidates = [
    PROJECT_ROOT / "datasets" / "anomaly_detection" / "validation_datasets" / "maui_system_2105_clean_hourly_anomaly_results.csv",
    PROJECT_ROOT / "experiments" / "regional_adaptation" / "maui" / "data" / "maui_system_2105_clean_hourly_anomaly_results.csv",
    PROJECT_ROOT / "anomaly_detection" / "results" / "csv_outputs" / "isolation_forest_results.csv",
]
anom_file = next((p for p in anom_candidates if p.exists()), None)

if anom_file and anom_file.exists():
    df_anom = pd.read_csv(anom_file)
    if "Timestamp" in df_anom.columns:
        df_anom["Timestamp"] = pd.to_datetime(df_anom["Timestamp"])

    # Residual Distribution
    if "Residual_Error" in df_anom.columns and "GHI" in df_anom.columns:
        plt.figure(figsize=(10, 5))
        daylight_res = df_anom.loc[df_anom["GHI"] > 10.0, "Residual_Error"]
        plt.hist(daylight_res, bins=60, color="#34495e", alpha=0.7, edgecolor="black", density=True, label="Daylight Residuals (kW)")
        plt.axvline(x=-63.33, color="#e74c3c", linestyle="--", lw=2, label="Calibrated Threshold (-63.3 kW)")
        plt.title("Maui Ocean Center — Model Residual Error Distribution (Daylight Hours)", fontsize=13, fontweight="bold")
        plt.xlabel("Residual Error: Actual - Predicted (kW)", fontsize=11)
        plt.ylabel("Probability Density", fontsize=11)
        plt.legend(frameon=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        p_res = PLOT_DIR / "maui_residual_distribution.png"
        plt.savefig(p_res, dpi=300)
        plt.close()
        print(f"Saved: {p_res}")

    # Pie Chart
    if "Reconciled_Operational_Status" in df_anom.columns:
        counts = df_anom["Reconciled_Operational_Status"].value_counts()
        colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]
        plt.figure(figsize=(7, 7))
        plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140,
                colors=colors[:len(counts)], explode=[0.02]*len(counts))
        plt.title("Operational Anomaly Reconciliation Breakdown (Holdout Test Period)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        p_pie = PLOT_DIR / "maui_anomaly_reconciliation_pie.png"
        plt.savefig(p_pie, dpi=300)
        plt.close()
        print(f"Saved: {p_pie}")


# ── 2. Candidate Anomaly Timelines ──────────────────────────────────────────
def plot_candidate_timeline(cand_id, cand_csv):
    cand_path = PROJECT_ROOT / "datasets" / "energy_prediction" / "processed" / cand_csv
    if not cand_path.exists():
        cand_path = PROJECT_ROOT / "experiments" / "regional_adaptation" / "maui" / "target_reconstruction" / "data" / cand_csv
    if not cand_path.exists():
        print(f"Skipping {cand_id}: {cand_path} not found.")
        return

    df = pd.read_csv(cand_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    
    # Daylight observations
    daylight = df[df["GHI"] >= 20.0].copy()
    if len(daylight) == 0:
        return
        
    # Baseline expected AC output
    ghi_ratio = daylight["GHI"] / 1000.0
    expected = daylight["GHI"] * (daylight["AC_Output"].quantile(0.95) / daylight["GHI"].quantile(0.95))
    daylight["Residual"] = np.abs(daylight["AC_Output"] - expected)
    
    threshold = daylight["Residual"].mean() + 2.5 * daylight["Residual"].std()
    daylight["Is_Anomaly"] = daylight["Residual"] > threshold

    # Sample a 14-day window for visual clarity
    sample_sub = daylight.iloc[:336].copy()
    anomalies = sample_sub[sample_sub["Is_Anomaly"]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Top Panel: Actual vs Expected with Anomalies Marked
    ax1.plot(sample_sub["Timestamp"], sample_sub["AC_Output"], label="Observed AC Power (kW)", color="#2c3e50", linewidth=1.4)
    if len(anomalies) > 0:
        ax1.scatter(anomalies["Timestamp"], anomalies["AC_Output"], color="#e74c3c", s=45, zorder=5, label="Flagged Potential Anomaly")
    ax1.set_ylabel("AC Output (kW)", fontsize=10, fontweight="bold")
    ax1.set_title(f"SURYOKTI Anomaly Audit: {cand_id} Daylight Operational Diagnostics", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    # Bottom Panel: Residual error
    ax2.plot(sample_sub["Timestamp"], sample_sub["Residual"], label="Daylight Residual (kW)", color="#8e44ad", linewidth=1.2)
    ax2.axhline(threshold, color="#e74c3c", linestyle="--", label=f"2.5σ Threshold ({threshold:.1f} kW)")
    ax2.axhline(0, color="black", linestyle=":", alpha=0.5)
    ax2.set_ylabel("Residual (kW)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Timestamp (HST)", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    out_path = PLOT_DIR / f"{cand_id.lower()}_anomaly_timeline.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")

plot_candidate_timeline("Candidate_2", "candidate_2_hourly.csv")
plot_candidate_timeline("Candidate_5", "candidate_5_hourly.csv")

