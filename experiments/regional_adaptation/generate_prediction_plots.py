"""
Generates visualization plots for Maui Ocean Center prediction models.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

pred_file = os.path.join(DATA_DIR, "maui_system_2105_clean_hourly_test_predictions.csv")
df = pd.read_csv(pred_file)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# 1. Actual vs Predicted on a 1-week sample
sample = df[(df["Timestamp"] >= "2023-06-01") & (df["Timestamp"] < "2023-06-08")].copy()

plt.figure(figsize=(14, 5))
plt.plot(sample["Timestamp"], sample["Actual_AC_Output"], label="Actual AC Output (kW)", color="black", lw=1.8)
plt.plot(sample["Timestamp"], sample["Pred_XGBoost (Tuned)"], label="XGBoost Predicted (kW)", color="#e67e22", lw=1.5, ls="--")
plt.plot(sample["Timestamp"], sample["Pred_Hybrid Stacking (RF + XGB -> RidgeCV)"], label="Hybrid Stacking Predicted (kW)", color="#2980b9", lw=1.5, ls=":")
plt.title("Maui Ocean Center (110 kW) — Actual vs. Predicted AC Power (1-Week Sample)", fontsize=13, fontweight="bold")
plt.xlabel("Date & Time (HST)", fontsize=11)
plt.ylabel("AC Power Output (kW)", fontsize=11)
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_path_1 = os.path.join(PLOTS_DIR, "maui_actual_vs_predicted.png")
plt.savefig(plot_path_1, dpi=300)
plt.close()

# 2. Model Comparison Bar Chart
models = ["Linear Reg", "Random Forest", "XGBoost", "Hybrid Stacking"]
test_r2 = [-0.2001, 0.8571, 0.8658, 0.8649]
nmae = [42.44, 11.29, 11.68, 11.18]

fig, ax1 = plt.subplots(figsize=(9, 5))

color = "tab:blue"
ax1.set_xlabel("Model Architecture", fontweight="bold", fontsize=11)
ax1.set_ylabel("Holdout Test R² Score", color=color, fontweight="bold", fontsize=11)
bars1 = ax1.bar(np.arange(len(models)) - 0.18, [max(0, r) for r in test_r2], width=0.35, color=color, label="Test R²")
ax1.tick_params(axis="y", labelcolor=color)
ax1.set_ylim(0, 1.0)

ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Capacity-Normalized MAE (nMAE %)", color=color, fontweight="bold", fontsize=11)
bars2 = ax2.bar(np.arange(len(models)) + 0.18, nmae, width=0.35, color=color, label="nMAE (%)")
ax2.tick_params(axis="y", labelcolor=color)
ax2.set_ylim(0, 50)

plt.title("Maui Model Benchmark: Holdout Test Accuracy vs. Error Rate", fontsize=12, fontweight="bold")
ax1.set_xticks(np.arange(len(models)))
ax1.set_xticklabels(models, fontweight="semibold")
plt.tight_layout()
plot_path_2 = os.path.join(PLOTS_DIR, "maui_model_comparison.png")
plt.savefig(plot_path_2, dpi=300)
plt.close()

print(f"Plots saved to:\n  * {plot_path_1}\n  * {plot_path_2}")

