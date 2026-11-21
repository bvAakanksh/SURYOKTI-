# SURYOKTI: Anomaly Detection Module

The **Anomaly Detection Module** identifies sensor malfunctions, communication freezes, data quality irregularities, statistical residual exceedances, and potential solar PV operational anomalies. The supported implementation is the canonical pipeline (`run_pipeline.py`), which reconciles deterministic rules, daylight-only Isolation Forest novelty detection, optional forecast residuals, and event aggregation.

---

## 1. Directory Structure

```text
anomaly_detection/
├── README.md                                    # Module architecture and methodology (this file)
├── CHANGELOG.md                                  # Change-by-change implementation checkpoints
├── config.py                                     # Central anomaly thresholds and column mapping
├── schema.py                                     # Non-destructive input validation and normalization
├── pipeline.py                                   # Rules, novelty, residuals, reconciliation, events
├── run_pipeline.py                               # Supported command-line entrypoint
├── tests/                                        # Dependency-free unit and integration tests
│
├── data/                                        # Data cleaning and inspection utilities
│   ├── data_clean.py                            # Column normalization and duplicate check
│   ├── insepct_data.py                          # Statistical profile and range audit
│   └── zero_power_investigate.py                # Daytime zero-power incidence investigation
│
├── detectors/                                   # Anomaly detector implementations
│   └── isolation_forest_detector.py             # Legacy prototype retained for comparison only
│
├── training/                                    # Model calibration and contamination tuning
├── evaluation/                                  # Metric scoring and precision/recall audit
├── testing/                                     # Edge case testing and unit verification
├── tracing/                                     # Irradiance vs Power tracking
│   └── time_irradiance.py                       # Synchronous solar irradiance vs power tracking
│
├── models/                                      # Serialized anomaly models
│   └── maui_isolation_forest.pkl                # Trained Isolation Forest model artifact
│
├── plots/                                       # Visualization figures
│   ├── anomaly_distribution/                    # Anomaly timeline and temporal scatter plots
│   ├── sensor_analysis/                         # GHI vs AC Power scatter diagnostics
│   └── detector_comparison/                     # Reconciliation pie charts and category distributions
│
├── results/                                     # Exported detection outputs
│   ├── csv_outputs/
│   │   └── isolation_forest_results.csv         # Enriched anomaly scores and residual flags
│   └── reports/                                 # Summary audit tables
│
├── logs/                                        # Execution and training logs
└── scripts/                                     # Detection runners and batch processors
```

---

## 2. Rigorous Anomaly Terminology Standards

To prevent false alarms and scientifically unjustified claims, the SURYOKTI framework enforces strict nomenclature:

| Terminology Classification | Formal Definition | Cause / Operational Meaning |
|---|---|---|
| `DAYLIGHT_NORMAL` | Daytime observations conforming to physics curves | Expected generation under prevailing irradiance and temperature |
| `NIGHTTIME_INACTIVE` | Zero power when $\text{GHI} \le 50 \text{ W/m}^2$ | Expected physical shutdown of inverter when solar flux is absent |
| `POTENTIAL_OPERATIONAL_ANOMALY` | Significant output deficit during clear daylight | Inverter trip, string disconnection, severe soiling, or shading |
| `STATISTICAL_RESIDUAL_EXCEEDANCE` | Observation exceeding $3 \cdot \sigma_{\text{robust}}$ error threshold | Model underperformance or unusual localized weather volatility |
| `DATA_QUALITY_ANOMALY` | Frozen sensor readings, out-of-range values, drops | Pyranometer clipping, communication bus dropouts, logging latency |

> [!WARNING]
> Without independently verified plant maintenance logs (such as SCADA work orders or operator incident tickets), detections are classified as **POTENTIAL_OPERATIONAL_ANOMALIES** or **STATISTICAL_RESIDUAL_EXCEEDANCES**. They are never labelled as confirmed hardware failures.

---

## 3. Daylight Gating and MAD Calibration

### The Nighttime Zero-Power Trap
In solar datasets, roughly 50% of all recorded timestamps occur during darkness when both irradiance and AC output are zero. Uncalibrated clustering or anomaly detection models (e.g. unconstrained Isolation Forest) frequently flag these zero-power rows as abnormal simply due to statistical clustering density.

### Daylight Robust Calibration
1. **Solar Gate Filter:** All anomaly calibration is partitioned strictly into daytime periods ($\text{GHI} > 50 \text{ W/m}^2$).
2. **Median Absolute Deviation (MAD):**
   $$\text{MAD} = \text{median}\left(\left| r_i - \text{median}(r) \right|\right)$$
   $$\sigma_{\text{robust}} = 1.4826 \cdot \text{MAD}$$
3. **Dynamic Thresholding:** A daytime observation is flagged as an operational anomaly only when:
   $$|P_{\text{actual}} - P_{\text{expected}}| > 3 \cdot \sigma_{\text{robust}} \quad \text{and} \quad \text{GHI} > 50 \text{ W/m}^2$$

---

## 4. Outputs and validation limits

The legacy prototype's values below are historical reference values only. The canonical pipeline exports its own `run_metadata.json`, `row_level_anomalies.csv`, and `anomaly_events.csv` for every run; its threshold and status distribution are dataset-specific.

| Observation Partition | Sample Count | $R^2$ Score | MAE (kW) | RMSE (kW) | Pearson $r$ |
|---|---:|---:|---:|---:|---:|
| **All Test Observations** | 834 | 0.8885 | 83.67 | 126.68 | 0.9457 |
| **Daylight Normal + Nighttime Inactive** | 726 | 0.8808 | 91.65 | 134.42 | 0.9414 |
| **Daylight Potential Anomalies** | 108 | -2.4205 | 30.01 | 49.79 | 0.5896 |

---

## 5. Execution Instructions

### A. Run Canonical Detection Without a Forecast
```powershell
.venv\Scripts\python.exe anomaly_detection/run_pipeline.py --input Solar_Weather_Data_Cleaned.csv --output-dir anomaly_detection/results/canonical_demo
```
This creates a fitted `anomaly_pipeline.joblib`, row-level status CSV, event CSV, and run metadata. Without a forecast column, only data-quality rules and multivariate novelty assessment are available.

### B. Run Canonical Detection With Forecast Residuals
```powershell
.venv\Scripts\python.exe anomaly_detection/run_pipeline.py --input <telemetry_with_forecast.csv> --prediction-column <forecast_column> --output-dir <output_directory>
```
The forecast must be generated without using the observed target being scored. The pipeline calibrates a daylight-only MAD threshold on the chronological training portion and evaluates the untouched holdout portion.

### C. Run the Test Suite
```powershell
.venv\Scripts\python.exe -m unittest discover -s anomaly_detection/tests -v
```

### D. Investigate Zero-Power Incidents
```powershell
.venv\Scripts\python.exe anomaly_detection/data/zero_power_investigate.py
```
Audits daytime hours where irradiance was positive but power was zero.

