# SURYOKTI: Romania PV Solar Energy Prediction Implementation

Independent implementation and cross-site evaluation of the **SURYOKTI 3-Stack Solar Energy Prediction Model** on the **Two-Site Solar PV Inverter & Weather Dataset (Romania)**.

Dataset Source: [UPB NLP PV Generation Dataset](https://github.com/upb-nlp/PV-Generation-Dataset)

---

## 1. Project Context & Complete Isolation Confirmation

This project implements the SURYOKTI stacking architecture (**Random Forest + XGBoost → Neural Network MLPRegressor**) on two independent 50 kW AC solar PV installations located in southwestern Romania (Site A and Site B).

> [!IMPORTANT]
> **Complete Workspace Isolation:**
> This entire implementation is self-contained inside `energy_prediction_romania/`.
> Zero files in the original SURYOKTI codebase (`energy_prediction/`, `anomaly_detection/`, `datasets/`, `models/`, `plots/`, etc.) were edited, moved, overwritten, or modified in any manner.

---

## 2. Original Model Architecture & Clarification of the 92.30% Result

During inspection of the canonical model ([`train_meta_model.py`](file:///c:/VITB/suryokti%20draft%201/energy_prediction/training/train_meta_model.py)) and benchmark presentation scripts ([`generate_presentation.py`](file:///c:/VITB/suryokti%20draft%201/presentations/scripts/generate_presentation.py)):

- **Meaning of 92.30%:** The reported 92.30% score represents the **Coefficient of Determination ($R^2 = 0.9230$)** on a 20% holdout split. It is **not** a classification accuracy.
- **Original Scale:** The original target was plant power generation in kW with holdout $\text{MAE} = 52.26\text{ kW}$ and $\text{RMSE} = 101.18\text{ kW}$.
- **Original Model Pipeline:**
  1. Base Regressor 1: `RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_split=6, min_samples_leaf=3, max_features=0.8)`
  2. Base Regressor 2: `XGBRegressor(n_estimators=200, lr=0.03, max_depth=4, min_child_weight=3, subsample=0.8, colsample=0.8)`
  3. Meta-Model: `Pipeline([("scaler", StandardScaler()), ("nn", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.05, lr_init=0.003))])`
  4. Stacking: `StackingRegressor(passthrough=True, cv=KFold(5))`
  5. Post-Processing: Nocturnal gating via solar irradiance cutoff.

---

## 3. Directory Structure

```text
energy_prediction_romania/
│
├── README.md                              # Comprehensive project documentation
├── requirements.txt                       # Explicit pinned dependencies
├── config.py                              # Central configuration & hyperparameters
├── run_pipeline.py                        # Master CLI pipeline orchestrator
│
├── datasets/
│   ├── raw/                               # Unmodified raw CSVs from UPB repo
│   │   ├── site_a_hourly_weather.csv      # Site A (13,031 rows, Jan 2025 – Jun 2026)
│   │   └── site_b_hourly_weather.csv      # Site B (9,504 rows, Apr 2025 – May 2026)
│   ├── cleaned/                           # Cleaned, sorted, verified data
│   │   ├── site_a_cleaned.csv             # 12,771 valid rows
│   │   └── site_b_cleaned.csv             # 9,334 valid rows
│   └── processed/                         # Feature-engineered forecasting datasets
│       ├── site_a_processed.csv           # 16 features + normalized target
│       └── site_b_processed.csv
│
├── docs/
│   └── original_model_inspection.md       # Audit notes of original SURYOKTI model
│
├── src/
│   ├── inspect_dataset.py                 # Exploratory data profiling & checks
│   ├── data_cleaning.py                   # Domain sanitization & missing value handling
│   ├── feature_engineering.py             # Cyclical time & meteorological feature builder
│   ├── preprocessing.py                   # Leakage-free StandardScaler pipeline
│   ├── data_split.py                      # Chronological 70/10/20 & daytime partitioning
│   ├── train_base_models.py               # Standalone RF & XGBoost training
│   ├── train_stacking_model.py            # Canonical StackingRegressor ensemble
│   ├── evaluate_model.py                  # Dual evaluation & diagnostic figure generator
│   ├── cross_site_evaluation.py           # Bidirectional transfer testing (A->B, B->A)
│   ├── predict.py                         # Production inference class & CLI
│   └── utils.py                           # Metrics, physical bounds, and logging helpers
│
├── models/
│   ├── site_a/                            # Trained artifacts for Site A
│   │   ├── stacking_ensemble.joblib
│   │   ├── rf_model.joblib
│   │   ├── xgb_model.joblib
│   │   ├── meta_model.joblib
│   │   ├── preprocessing_pipeline.joblib
│   │   └── feature_list.json
│   └── site_b/                            # Trained artifacts for Site B
│
├── results/
│   ├── metrics/                           # JSON metrics & comparison CSVs
│   │   ├── site_a_metrics.json
│   │   ├── site_b_metrics.json
│   │   ├── cross_site_A_to_B.json
│   │   ├── cross_site_B_to_A.json
│   │   ├── cross_site_summary.json
│   │   └── model_comparison_table.csv
│   ├── predictions/                       # Prediction CSVs with residual columns
│   │   ├── site_a_test_predictions.csv
│   │   ├── site_b_test_predictions.csv
│   │   ├── cross_site_A_to_B_predictions.csv
│   │   └── cross_site_B_to_A_predictions.csv
│   └── reports/                           # Profiling summaries & logs
│       ├── dataset_inspection_report.md
│       ├── inspection_report.json
│       ├── feature_columns.json
│       └── data_cleaning_report.json
│
├── plots/
│   ├── data_analysis/                     # Exploratory analysis figures
│   │   ├── diurnal_power_profile.png
│   │   ├── ghi_vs_active_power.png
│   │   ├── capacity_factor_distribution.png
│   │   ├── monthly_seasonality.png
│   │   └── feature_correlation_heatmap.png
│   ├── model_evaluation/                  # Test set evaluation diagnostic figures
│   │   ├── actual_vs_predicted.png
│   │   ├── residual_analysis.png
│   │   ├── time_series_overlay.png
│   │   ├── model_comparison_bar.png
│   │   ├── daytime_vs_all_hours.png
│   │   └── generation_level_error.png
│   └── cross_site/                        # Transferability diagnostics
│       ├── cross_site_transfer_comparison.png
│       ├── cross_site_scatter.png
│       └── cross_site_residuals.png
│
└── tests/
    ├── test_preprocessing.py              # Zero-leakage & scaling verification
    ├── test_model.py                      # Model fitting & physical bounds verification
    └── test_predictions.py                # Artifact loading & kW conversion tests
```

---

## 4. Dataset Inspection & Domain Insights

Dataset inspection ([`dataset_inspection_report.md`](file:///c:/VITB/suryokti%20draft%201/energy_prediction_romania/results/reports/dataset_inspection_report.md)) revealed key physical characteristics:

1. **Inverter Overload Margin:**
   The installation is rated at 50 kW AC. However, raw `active_power_kw` reaches up to **55.0 kW** (Site A) and **53.07 kW** (Site B) during peak cold-spring irradiance spikes (10% overload margin). The dataset creators capped `capacity_factor` to `1.0` whenever power exceeded 50 kW.
2. **Missing Sensor Rows:**
   Site A has 260 missing target rows; Site B has 170. These coincide with DST shift gaps and sensor outages where both target and weather are missing.
   - **Handling Rule:** We **never** interpolate or fabricate missing target values. These rows are excluded from supervised training.
3. **Valid Nighttime Zeros:**
   43.46% (Site A) and 43.94% (Site B) of all hours are genuine nighttime hours ($\text{solar elevation} \le 0^\circ$). These valid zero values are preserved to maintain natural diurnal cycle training.

---

## 5. Strict Target Leakage Prevention & Feature Engineering

To ensure real-world forecasting validity, all real-time inverter telemetry was strictly excluded:

- **Forbidden Leakage Columns:** `active_power_kw`, `capacity_factor`, `daily_energy_kwh`, `cumulative_energy_kwh`, `dc_input_power_kw`, `efficiency_pct`, `is_generating`, `inverter_status`, `inverter_temp_c`, `grid_freq_hz`, `power_factor`.
- **Approved Feature Set (16 features):**
  - **10 Meteorological Variables:** `temp_c`, `humidity_pct`, `cloud_cover_pct`, `wind_speed_kmh`, `ghi_wm2`, `dni_wm2`, `dhi_wm2`, `precip_mm`, `solar_elevation_deg`, `solar_azimuth_deg`.
  - **6 Cyclical Solar-Temporal Variables:** `hour_sin`, `hour_cos`, `day_of_year_sin`, `day_of_year_cos`, `month_sin`, `month_cos`.

---

## 6. Comprehensive Performance Comparison Table

All models were evaluated on strictly chronological 20% holdout test sets (temporal validation with zero forward lookahead bias). Dual metrics are reported for both full 24-hour horizons and daytime-only ($\text{solar elevation} > 5^\circ$) operation.

| Dataset / Experiment | Target Scale | Temporal Split Method | All-Hours $R^2$ | All-Hours MAE | All-Hours RMSE | Daytime $R^2$ | Daytime MAE | Daytime RMSE |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original SURYOKTI Model** | Plant kW (0–1200 kW) | Random 80/20 Holdout | **0.9230** ($92.30\%$) | 52.26 kW | 101.18 kW | N/A | N/A | N/A |
| **Romania Site A (Random Forest)** | Inverter CF & kW (50 kW) | Chronological 70/10/20 | 0.9107 | 2.429 kW | 4.869 kW | 0.8339 | 4.390 kW | 6.564 kW |
| **Romania Site A (XGBoost)** | Inverter CF & kW (50 kW) | Chronological 70/10/20 | 0.9129 | 2.423 kW | 4.810 kW | 0.8379 | 4.376 kW | 6.484 kW |
| **Romania Site A (Stacking Ensemble)** | Inverter CF & kW (50 kW) | Chronological 70/10/20 | **0.9157** ($91.57\%$) | **2.291 kW** | **4.733 kW** | **0.8431** ($84.31\%$) | **4.134 kW** | **6.379 kW** |
| **Romania Site B (Random Forest)** | Inverter CF & kW (50 kW) | Chronological 70/10/20 | 0.8784 | 2.792 kW | 5.454 kW | 0.7655 | 5.310 kW | 7.541 kW |
| **Romania Site B (XGBoost)** | Inverter CF & kW (50 kW) | Chronological 70/10/20 | 0.8719 | 2.866 kW | 5.597 kW | 0.7531 | 5.446 kW | 7.738 kW |
| **Romania Site B (Stacking Ensemble)** | Inverter CF & kW (50 kW) | Chronological 70/10/20 | **0.8771** ($87.71\%$) | **2.883 kW** | **5.482 kW** | **0.7632** ($76.32\%$) | **5.474 kW** | **7.579 kW** |
| **Cross-Site: Model A on Site B** | Inverter CF & kW (50 kW) | Zero-Shot Transfer | **0.9086** ($90.86\%$) | **2.367 kW** | **4.728 kW** | **0.8239** ($82.39\%$) | **4.490 kW** | **6.536 kW** |
| **Cross-Site: Model B on Site A** | Inverter CF & kW (50 kW) | Zero-Shot Transfer | **0.8821** ($88.21\%$) | **2.899 kW** | **5.595 kW** | **0.7807** ($78.07\%$) | **5.239 kW** | **7.542 kW** |

> [!NOTE]
> **Why results differ between original and Romanian experiments:**
> 1. **Target & Capacity:** Original SURYOKTI models large commercial plants with power ranging up to megawatts, whereas Romania models 50 kW modular string inverters with normalized capacity factor $\in [0, 1]$.
> 2. **Split Rigor:** The original 92.30% $R^2$ was achieved via random train/test shuffling (where hours from the same days appear across splits). The Romanian benchmark enforces a strictly chronological holdout test set (March 2026 – June 2026), reflecting real forward operational forecasting.
> 3. **Daytime vs All-Hours:** All-hours $R^2$ includes nocturnal zero periods, which improves overall variance explanation. Daytime-only $R^2$ tests strictly when solar generation is active, providing the most rigorous engineering metric.
> 4. **Transferability:** Model A generalizes exceptionally well to Site B ($R^2 = 0.9086$), outperforming Site B's own locally trained base models due to Site A's longer historical training window (18 months vs 13 months).

---

## 7. How to Execute

### Run Complete End-to-End Pipeline
```powershell
.\.venv\Scripts\python.exe energy_prediction_romania/run_pipeline.py
```

### Run Individual Pipeline Stages
```powershell
# Inspect dataset
.\.venv\Scripts\python.exe energy_prediction_romania/src/inspect_dataset.py

# Clean dataset
.\.venv\Scripts\python.exe energy_prediction_romania/src/data_cleaning.py

# Feature engineering
.\.venv\Scripts\python.exe energy_prediction_romania/src/feature_engineering.py

# Train stacking models
.\.venv\Scripts\python.exe energy_prediction_romania/src/train_stacking_model.py

# Evaluate and generate diagnostic plots
.\.venv\Scripts\python.exe energy_prediction_romania/src/evaluate_model.py

# Cross-site transfer evaluation
.\.venv\Scripts\python.exe energy_prediction_romania/src/cross_site_evaluation.py
```

### Run Unit and Integration Tests
```powershell
.\.venv\Scripts\python.exe energy_prediction_romania/tests/test_preprocessing.py
.\.venv\Scripts\python.exe energy_prediction_romania/tests/test_model.py
.\.venv\Scripts\python.exe energy_prediction_romania/tests/test_predictions.py
```

### Execute Inference
```powershell
# CLI prediction
.\.venv\Scripts\python.exe energy_prediction_romania/src/predict.py --site site_a --input <path_to_weather.csv> --output <path_to_predictions.csv>

# Interactive test inference
.\.venv\Scripts\python.exe energy_prediction_romania/src/predict.py
```

### Run Canonical Anomaly Detection on Forecast Holdouts
```powershell
.\.venv\Scripts\python.exe energy_prediction_romania/run_anomaly_detection.py --site all
```
This assesses the existing chronological stacking-model holdouts using the shared canonical anomaly pipeline. Outputs are written to `results/anomaly_detection/site_a/` and `results/anomaly_detection/site_b/`. They are inspection candidates only because no maintenance/fault ground-truth labels are supplied with the dataset.

### Run Synthetic Fault-Detection Regression Validation
```powershell
.\.venv\Scripts\python.exe energy_prediction_romania/run_synthetic_fault_validation.py --site all
```
This creates a separate copy of each future assessment holdout, injects deterministic labeled daylight outages, partial derates, and negative-power sensor faults, and measures exact-status recall. Results are stored in `results/synthetic_fault_validation/site_a/` and `site_b/`. This is a repeatable software test only: synthetic recall is not real-world fault-detection precision or recall.

### Run Causally Tuned Non-CNN Forecasts
```powershell
.\.venv\Scripts\python.exe energy_prediction_romania/run_tuned_prediction_pipeline.py
```
This searches a small RF/XGBoost candidate set on the validation period only, selects a non-negative blend by daytime validation MAE, retrains on train+validation, and evaluates once on the untouched test period. Artifacts are written separately under `models/tuned/` and `results/tuned_predictions/`; baseline models are preserved.

To assess those forecasts with forecast-power-specific residual thresholds:
```powershell
.\.venv\Scripts\python.exe energy_prediction_romania/run_anomaly_detection.py --prediction-dir energy_prediction_romania/results/tuned_predictions --output-root energy_prediction_romania/results/tuned_anomaly_detection_recommended --residual-mad-multiplier 2.5
```

