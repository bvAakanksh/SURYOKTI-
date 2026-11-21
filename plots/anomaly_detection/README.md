# SURYOKTI: Anomaly Detection Visual Diagnostics

This directory contains high-resolution diagnostic plots and generator scripts dedicated to solar PV sensor, operational, and inverter fault detection under the SURYOKTI framework.

---

## 1. Inventory of Figures

| Figure File | Particular Source Script | Input Dataset | Visual Diagnostic Purpose |
|---|---|---|---|
| [`maui_anomaly_reconciliation_pie.png`](file:///c:/VITB/suryokti%20draft%201/plots/anomaly_detection/maui_anomaly_reconciliation_pie.png) | `generate_anomaly_plots.py` | `datasets/anomaly_detection/validation_datasets/maui_system_2105_clean_hourly_anomaly_results.csv` | Reconciles the 8,760 annual hours into Nighttime Inactive (~50%), Daylight Normal (~44%), and Potential Operational Anomalies (~6%). |
| [`maui_residual_distribution.png`](file:///c:/VITB/suryokti%20draft%201/plots/anomaly_detection/maui_residual_distribution.png) | `generate_anomaly_plots.py` | `datasets/anomaly_detection/validation_datasets/maui_system_2105_clean_hourly_anomaly_results.csv` | Probability density of model residual error during daylight ($GHI > 10 \text{ W/m}^2$) alongside calibrated statistical threshold ($-63.3\text{ kW}$). |
| [`candidate_2_anomaly_timeline.png`](file:///c:/VITB/suryokti%20draft%201/plots/anomaly_detection/candidate_2_anomaly_timeline.png) | `generate_anomaly_plots.py` | `datasets/energy_prediction/processed/candidate_2_hourly.csv` | Dual-panel timeline showing Candidate 2 (De-duplicated Full Array) actual AC output vs expected, with flagged daylight anomalies and residual exceedances. |
| [`candidate_5_anomaly_timeline.png`](file:///c:/VITB/suryokti%20draft%201/plots/anomaly_detection/candidate_5_anomaly_timeline.png) | `generate_anomaly_plots.py` | `datasets/energy_prediction/processed/candidate_5_hourly.csv` | Dual-panel timeline showing Candidate 5 (Twin Combiners, 1.25 ILR) actual AC output vs expected, with flagged daylight anomalies and residual exceedances. |

---

## 2. Reproduction Command

To re-execute detection logic and regenerate all figures in this folder:

```powershell
python plots/anomaly_detection/generate_anomaly_plots.py
```

