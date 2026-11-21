"""
SURYOKTI Regional Adaptation — Maui Visualization Suite
======================================================
Generates high-resolution diagnostic plots for both Energy Prediction
and Anomaly Detection on the Maui Ocean Center (110 kW) dataset.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# 1. Prediction Comparison Plot
pred_file = os.path.join(DATA_DIR, "maui_system_2105_clean_hourly_test_predictions.csv")
df_pred = pd.read_csv(pred_file)
df_pred["Timestamp"] = pd.to_datetime(df_pred["Timestamp"])

# Sample week: June 1 to June 8, 2023
sample = df_pred[(df_pred["Timestamp"] >= "2023-06-01") & (df_pred["Timestamp"] < "2023-06-08")].copy()

plt.figure(figsize=(14, 5))
plt.plot(sample["Timestamp"], sample["Actual_AC_Output"], label="Actual AC Output (kW)", color="black", lw=1.8)
plt.plot(sample["Timestamp"], sample["Pred_XGBoost (Tuned)"], label="XGBoost Predicted (kW)", color="#e67e22", lw=1.5, ls="--")
plt.plot(sample["Timestamp"], sample["Pred_Hybrid Stacking (RF + XGB -> RidgeCV)"], label="Hybrid Stacking Predicted (kW)", color="#2980b9", lw=1.5, ls=":")
plt.title("Maui Ocean Center (110 kW) — Actual vs. Predicted AC Power (June 2023 Sample)", fontsize=13, fontweight="bold")
plt.xlabel("Date & Time (HST)", fontsize=11)
plt.ylabel("AC Power Output (kW)", fontsize=11)
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
p1 = os.path.join(PLOTS_DIR, "maui_actual_vs_predicted.png")
plt.savefig(p1, dpi=300)
plt.close()

# 2. Model Accuracy vs Normalized Error Bar Chart
models = ["Linear Reg", "Random Forest", "XGBoost", "Hybrid Stacking"]
test_r2 = [-0.2001, 0.8571, 0.8658, 0.8649]
nmae = [42.44, 11.29, 11.68, 11.18]

fig, ax1 = plt.subplots(figsize=(9, 5))
color = "tab:blue"
ax1.set_xlabel("Model Architecture", fontweight="bold", fontsize=11)
ax1.set_ylabel("Holdout Test R² Score", color=color, fontweight="bold", fontsize=11)
ax1.bar(np.arange(len(models)) - 0.18, [max(0, r) for r in test_r2], width=0.35, color=color, label="Test R²")
ax1.tick_params(axis="y", labelcolor=color)
ax1.set_ylim(0, 1.0)

ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Capacity-Normalized MAE (nMAE %)", color=color, fontweight="bold", fontsize=11)
ax2.bar(np.arange(len(models)) + 0.18, nmae, width=0.35, color=color, label="nMAE (%)")
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_ylim(0, 50)

plt.title("Maui Model Benchmark: Holdout Test Accuracy vs. Error Rate", fontsize=12, fontweight="bold")
ax1.set_xticks(np.arange(len(models)))
ax1.set_xticklabels(models, fontweight="bold")
plt.tight_layout()
p2 = os.path.join(PLOTS_DIR, "maui_model_comparison.png")
plt.savefig(p2, dpi=300)
plt.close()

# 3. Anomaly Distribution and Residual Histogram
anom_file = os.path.join(DATA_DIR, "maui_system_2105_clean_hourly_anomaly_results.csv")
df_anom = pd.read_csv(anom_file)
df_anom["Timestamp"] = pd.to_datetime(df_anom["Timestamp"])

# Residual Distribution Plot
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
p3 = os.path.join(PLOTS_DIR, "maui_residual_distribution.png")
plt.savefig(p3, dpi=300)
plt.close()

# 4. Anomaly Operational Categories Pie Chart
counts = df_anom["Reconciled_Operational_Status"].value_counts()
colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

plt.figure(figsize=(7, 7))
plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140, colors=colors[:len(counts)], explode=[0.02]*len(counts))
plt.title("Operational Anomaly Reconciliation Breakdown (Holdout Test Period)", fontsize=12, fontweight="bold")
plt.tight_layout()
p4 = os.path.join(PLOTS_DIR, "maui_anomaly_reconciliation_pie.png")
plt.savefig(p4, dpi=300)
plt.close()

print(f"Generated all 4 visualization plots in {PLOTS_DIR}:")
print(f"  1. {p1}")
print(f"  2. {p2}")
print(f"  3. {p3}")
print(f"  4. {p4}")

