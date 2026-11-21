# SURYOKTI: Publication Figures & Plots Catalog

This directory serves as the centralized repository for all publication-grade figures, diagnostic plots, and visual assets generated across the SURYOKTI project. All visual assets are strictly organized into dedicated domain folders, each paired with its particular generator script and descriptive manifest.

---

## 1. Directory Tree Architecture

```text
plots/
├── README.md                                    # Master visual assets index and reproduction guide (this file)
│
├── anomaly_detection/                           # Dedicated operational anomaly detection diagnostics
│   ├── README.md                                # Anomaly figures manifest and audit terminology
│   ├── generate_anomaly_plots.py                # Reproducible generation script for all anomaly plots
│   ├── candidate_2_anomaly_timeline.png         # Candidate 2 (De-dup Full Array) hourly anomaly timeline
│   ├── candidate_5_anomaly_timeline.png         # Candidate 5 (Twin Combiners) hourly anomaly timeline
│   ├── maui_anomaly_reconciliation_pie.png      # Daylight normal vs anomaly category distribution pie
│   └── maui_residual_distribution.png           # Daylight residual error probability density vs 3σ threshold
│
├── energy_prediction/                           # Dedicated AC output forecasting & model benchmarking
│   ├── README.md                                # Energy prediction figures index and metrics
│   ├── hybrid_model_plots.py                    # 8-figure evaluation suite generator for hybrid stacking
│   ├── plots.py                                 # Baseline exploratory data analysis & scatter generator
│   ├── generate_energy_plots.py                 # Regional adaptation prediction & error bar generator
│   │
│   │   # Baseline & Exploratory Plots
│   ├── actual_vs_predicted.png                  # Baseline XGBoost actual vs predicted scatter plot
│   ├── feature_importance.png                   # Feature importance ranking from tree models
│   ├── model_comparison.png                     # Holdout MAE, RMSE, and R² metric comparison bar chart
│   ├── parity_and_residuals.png                 # Dual panel showing parity scatter and residual histogram
│   │
│   │   # Hybrid Stacking Model Plots
│   ├── hybrid_actual_vs_predicted.png           # Hybrid stacking model actual vs predicted plot
│   ├── hybrid_parity_all_models.png             # Multi-panel parity comparisons (LR, RF, XGB, Stacking)
│   ├── hybrid_residuals.png                     # Residual error distributions and QQ-plots
│   ├── hybrid_timeseries_sample.png             # 7-day continuous time-series forecast overlay
│   ├── hybrid_feature_importance.png            # Feature attribution across engineered physical features
│   ├── hybrid_model_comparison.png              # Multi-metric comparative evaluation chart
│   ├── hybrid_kfold_r2_per_fold.png             # Stability of R² across all 5 cross-validation folds
│   ├── hybrid_overfitting_check.png             # Train vs Holdout R² score delta check
│   │
│   │   # Regional Adaptation (Maui 110 kW) Plots
│   ├── maui_actual_vs_predicted.png             # Maui 1-week continuous actual vs predicted overlay
│   └── maui_model_comparison.png                # Maui holdout R² accuracy vs capacity-normalized MAE %
│
└── project_overview/                            # Dedicated plant overview & inverter target reconstruction
    ├── README.md                                # Candidate inverter descriptions and channel provenance
    ├── generate_overview_plots.py               # Diurnal curve and capacity comparison generator
    ├── candidate_capacity_comparison.png        # Bar chart comparing peak AC power across Candidates 1-5
    ├── candidate_comparison_clear_summer.png    # Inverter profile comparison on a clear summer day
    ├── candidate_comparison_clear_winter.png    # Inverter profile comparison on a clear winter day
    ├── candidate_comparison_cloudy_summer.png   # Inverter profile comparison on a cloudy summer day
    └── candidate_comparison_cloudy_winter.png   # Inverter profile comparison on a cloudy winter day
```

---

## 2. Complete Visual Asset Mapping

### A. Anomaly Detection Figures (`plots/anomaly_detection/`)

| Figure File | Particular Generator Script | Source Dataset | Visual Diagnostic Purpose |
|---|---|---|---|
| [`maui_anomaly_reconciliation_pie.png`](file:///c:/VITB/suryokti%20draft%201/plots/anomaly_detection/maui_anomaly_reconciliation_pie.png) | [`generate_anomaly_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/anomaly_detection/generate_anomaly_plots.py) | `maui_system_2105_clean_hourly_anomaly_results.csv` | Categorizes the 8,760 annual hours into Nighttime Inactive (~50%), Daylight Normal (~44%), and Potential Operational Anomalies (~6%), preventing nighttime false alarms. |
| [`maui_residual_distribution.png`](file:///c:/VITB/suryokti%20draft%201/plots/anomaly_detection/maui_residual_distribution.png) | [`generate_anomaly_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/anomaly_detection/generate_anomaly_plots.py) | `maui_system_2105_clean_hourly_anomaly_results.csv` | Daylight model residual error distribution ($GHI > 10\text{ W/m}^2$) alongside calibrated statistical threshold ($-63.3\text{ kW}$). |
| [`candidate_2_anomaly_timeline.png`](file:///c:/VITB/suryokti%20draft%201/plots/anomaly_detection/candidate_2_anomaly_timeline.png) | [`generate_anomaly_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/anomaly_detection/generate_anomaly_plots.py) | `candidate_2_hourly.csv` | Two-panel timeline showing Candidate 2 (De-duplicated Full Array, 185 kW peak) actual vs expected output, with flagged daylight anomalies and $2.5\sigma$ residual exceedances. |
| [`candidate_5_anomaly_timeline.png`](file:///c:/VITB/suryokti%20draft%201/plots/anomaly_detection/candidate_5_anomaly_timeline.png) | [`generate_anomaly_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/anomaly_detection/generate_anomaly_plots.py) | `candidate_5_hourly.csv` | Two-panel timeline showing Candidate 5 (Twin Combiners, 86.9 kW peak, 1.25 ILR) actual vs expected output, with flagged daylight anomalies and $2.5\sigma$ residual exceedances. |

---

### B. Energy Prediction Figures (`plots/energy_prediction/`)

| Figure File | Particular Generator Script | Source Dataset | Visual Diagnostic Purpose |
|---|---|---|---|
| [`actual_vs_predicted.png`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/actual_vs_predicted.png) | [`plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/plots.py) | `Solar_Weather_Data_Processed.csv` | Baseline XGBoost actual vs predicted parity scatter on test holdout. |
| [`feature_importance.png`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/feature_importance.png) | [`plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/plots.py) | `Solar_Weather_Data_Processed.csv` | Feature importance ranking showing GHI, Solar Zenith Angle, and Temperature dominance. |
| [`model_comparison.png`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/model_comparison.png) | [`plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/plots.py) | `Solar_Weather_Data_Processed.csv` | Holdout evaluation metrics (MAE, RMSE, and $R^2$) comparing Linear Regression, Random Forest, and XGBoost. |
| [`parity_and_residuals.png`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/parity_and_residuals.png) | [`plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/plots.py) | `Solar_Weather_Data_Processed.csv` | Two-panel diagnostic combining parity scatter and residual histogram. |
| [`hybrid_actual_vs_predicted.png`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/hybrid_actual_vs_predicted.png) | [`hybrid_model_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/hybrid_model_plots.py) | `Solar_Weather_Data_Processed.csv` | Scatter parity with $R^2=0.9200$, MAE $54.14\text{ kW}$, and RMSE $103.19\text{ kW}$. |
| [`hybrid_parity_all_models.png`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/hybrid_parity_all_models.png) | [`hybrid_model_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/hybrid_model_plots.py) | `Solar_Weather_Data_Processed.csv` | Four-panel parity comparison visualizing all model prediction distributions side-by-side. |
| [`hybrid_residuals.png`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/hybrid_residuals.png) | [`hybrid_model_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/hybrid_model_plots.py) | `Solar_Weather_Data_Processed.csv` | Dual-panel showing residual scatter across predicted kW and error distribution histogram. |
| [`hybrid_timeseries_sample.png`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/hybrid_timeseries_sample.png) | [`hybrid_model_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/hybrid_model_plots.py) | `Solar_Weather_Data_Processed.csv` | 150 consecutive test samples comparing actual output vs Hybrid forecast with error shading. |
| [`hybrid_feature_importance.png`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/hybrid_feature_importance.png) | [`hybrid_model_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/hybrid_model_plots.py) | `Solar_Weather_Data_Processed.csv` | Side-by-side feature attribution for Random Forest and XGBoost base learners. |
| [`hybrid_model_comparison.png`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/hybrid_model_comparison.png) | [`hybrid_model_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/hybrid_model_plots.py) | `Solar_Weather_Data_Processed.csv` | Tri-panel comparison of MAE, RMSE, and $R^2$ comparing LR, RF, XGB, and Hybrid Stacking. |
| [`hybrid_kfold_r2_per_fold.png`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/hybrid_kfold_r2_per_fold.png) | [`hybrid_model_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/hybrid_model_plots.py) | `Solar_Weather_Data_Processed.csv` | Stability of $R^2$ across all 5 cross-validation folds for all 4 models. |
| [`hybrid_overfitting_check.png`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/hybrid_overfitting_check.png) | [`hybrid_model_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/hybrid_model_plots.py) | `Solar_Weather_Data_Processed.csv` | Training $R^2$ vs Test $R^2$ delta, highlighting Random Forest memorization vs Hybrid balance. |
| [`maui_actual_vs_predicted.png`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/maui_actual_vs_predicted.png) | [`generate_energy_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/generate_energy_plots.py) | `maui_system_2105_clean_hourly_test_predictions.csv` | 1-week continuous time-series overlay of actual vs XGBoost & Hybrid Stacking predictions. |
| [`maui_model_comparison.png`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/maui_model_comparison.png) | [`generate_energy_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/energy_prediction/generate_energy_plots.py) | `maui_system_2105_clean_hourly_test_predictions.csv` | Holdout test $R^2$ accuracy vs capacity-normalized MAE (nMAE %) across model families. |

---

### C. Project Overview & Plant Capacity Figures (`plots/project_overview/`)

| Figure File | Particular Generator Script | Source Dataset | Visual Diagnostic Purpose |
|---|---|---|---|
| [`candidate_capacity_comparison.png`](file:///c:/VITB/suryokti%20draft%201/plots/project_overview/candidate_capacity_comparison.png) | [`generate_overview_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/project_overview/generate_overview_plots.py) | `all_candidates_hourly.csv` | Juxtaposes Candidates 1-5 against the 110 kW DC nameplate capacity. Clearly demonstrates why Candidate 1 (234 kW) involves channel double-counting, while Candidate 5 (86.9 kW) conforms to standard 1.25 DC/AC loading. |
| [`candidate_comparison_clear_summer.png`](file:///c:/VITB/suryokti%20draft%201/plots/project_overview/candidate_comparison_clear_summer.png) | [`generate_overview_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/project_overview/generate_overview_plots.py) | `all_candidates_hourly.csv` | Inverter profile comparison on a clear summer day (2020-07-15) alongside solar irradiance (GHI). |
| [`candidate_comparison_clear_winter.png`](file:///c:/VITB/suryokti%20draft%201/plots/project_overview/candidate_comparison_clear_winter.png) | [`generate_overview_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/project_overview/generate_overview_plots.py) | `all_candidates_hourly.csv` | Inverter profile comparison on a clear winter day (2020-12-10) alongside solar irradiance (GHI). |
| [`candidate_comparison_cloudy_summer.png`](file:///c:/VITB/suryokti%20draft%201/plots/project_overview/candidate_comparison_cloudy_summer.png) | [`generate_overview_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/project_overview/generate_overview_plots.py) | `all_candidates_hourly.csv` | Inverter profile comparison on an intermittent cloudy summer day (2020-08-05). |
| [`candidate_comparison_cloudy_winter.png`](file:///c:/VITB/suryokti%20draft%201/plots/project_overview/candidate_comparison_cloudy_winter.png) | [`generate_overview_plots.py`](file:///c:/VITB/suryokti%20draft%201/plots/project_overview/generate_overview_plots.py) | `all_candidates_hourly.csv` | Inverter profile comparison on an overcast winter day (2021-01-20). |

---

## 3. Reproduction Commands

To reproduce or regenerate any figure suite:

```powershell
# 1. Anomaly detection visual diagnostics
python plots/anomaly_detection/generate_anomaly_plots.py

# 2. Energy prediction hybrid stacking plots
python plots/energy_prediction/hybrid_model_plots.py

# 3. Energy prediction regional adaptation plots
python plots/energy_prediction/generate_energy_plots.py

# 4. Project overview & inverter candidate comparisons
python plots/project_overview/generate_overview_plots.py
```
