# SURYOKTI Datasets Directory

This directory contains all data assets utilized across the SURYOKTI energy forecasting, anomaly detection, and regional adaptation pipelines.

---

## 1. Directory Structure

```text
datasets/
├── README.md                                    # Provenance, schemas, and usage guide (this file)
├── data_dictionary.csv                          # Comprehensive column descriptions & units
│
├── energy_prediction/
│   ├── demo_datasets/                           # Benchmark curated demonstration dataset
│   │   └── Solar_Weather_Data_Processed.csv     # 2,777 hourly records (Gandikota + HI-SEAS)
│   ├── processed/                               # Cleaned, hourly-aggregated, and candidate CSVs
│   │   ├── Solar_Weather_Data_Cleaned.csv       # Standardized feature column names
│   │   ├── maui_system_2105_clean_hourly.csv    # 8,760 hourly telemetry records (Maui, HI)
│   │   ├── all_candidates_hourly.csv            # Merged dataset comparing Candidates 1-5
│   │   ├── candidate_1_hourly.csv               # Candidate 1: Direct 11-inverter sum (234 kW peak)
│   │   ├── candidate_2_hourly.csv               # Candidate 2: Inverters 01,02,03,05,07,08 (62.5 kW peak)
│   │   ├── candidate_3_hourly.csv               # Candidate 3: Inverters 04,06,09,10,11 (171.6 kW peak)
│   │   ├── candidate_4_hourly.csv               # Candidate 4: Primary Combiner inv_04 (43.4 kW peak)
│   │   └── candidate_5_hourly.csv               # Candidate 5: Twin Combiners inv_04+10 (86.9 kW peak)
│   ├── raw/
│   │   └── maui_system_2105_clean_15min.csv     # 35,040 raw 15-minute telemetry intervals
│   └── final_datasets/                          # Production certified regional datasets
│
├── anomaly_detection/
│   ├── raw/                                     # Raw unprocessed anomaly sensor logs
│   ├── refined/                                 # Daytime filtered subset (GHI > 50 W/m²)
│   ├── processed/                               # Scaled feature tensors for detector inference
│   └── validation_datasets/
│       ├── isolation_forest_results.csv         # Baseline Isolation Forest anomaly scores (Demo)
│       └── maui_system_2105_clean_hourly_anomaly_results.csv # Scored hourly anomalies (Maui)
│
├── cnn_fault_detection/
│   ├── README.md                                # External image data directory pointers
│   ├── raw_images/                              # Links to raw RGB panel imagery (C:\VITB\suryokti vision)
│   ├── processed_images/                        # Resized 224x224 RGB image arrays
│   └── training_datasets/                       # Training, validation, and test split manifests
│
├── external_sources/
│   ├── kaggle/                                  # Primary Kaggle source archives
│   │   ├── Plant_1_Generation_Data.csv          # Gandikota PV Plant 1 AC power telemetry
│   │   ├── SolarPrediction.csv                  # NASA/HI-SEAS weather observations
│   │   └── Pasion et al dataset.csv             # Reference comparative weather dataset
│   └── nrel_pvdaq/
│       └── systems_20250624.csv                 # NREL PVDAQ System 2105 registry & metadata
│
└── unclassified/                                # Temporary or exploratory scratch data files
```

---

## 2. Dataset Provenance and Scientific Context

### A. Demo Curated Dataset (`Solar_Weather_Data_Processed.csv`)
* **Role:** Benchmark dataset for ML algorithm comparison, hyperparameter optimization, and regression baseline establishment.
* **Origin:** Synthetic integration of two Kaggle sources:
  1. `Plant_1_Generation_Data.csv`: AC power output from a utility-scale solar PV farm in Gandikota, Andhra Pradesh, India.
  2. `SolarPrediction.csv`: Meteorological observations collected near the HI-SEAS weather station in Mauna Loa, Hawaii.
* **Characteristics:** 2,777 hourly records spanning September 1, 2016 through December 26, 2016. Missing values: 0. Duplicate rows: 0.
* **Scientific Note:** While ideal for benchmarking statistical learning, weather and generation variables were not co-located at the same physical plant.

### B. Maui System 2105 Telemetry (`maui_system_2105_clean_hourly.csv`)
* **Role:** Real-world physical regional adaptation testbed from NREL PVDAQ System 2105 located on Maui, Hawaii (Latitude: 20.89° N, Longitude: -156.43° W).
* **Characteristics:** 8,760 continuous hourly intervals across calendar year 2022.
* **Telemetry Channels:** Synchronous meteorological variables (Global Horizontal Irradiance [W/m²], Ambient Temperature [°C], Wind Speed [m/s]) and co-located inverter output telemetry.
* **Timezone Alignment:** Source telemetry recorded in UTC with a verified -13-hour shift aligned to local solar noon and Hawaii Standard Time (HST, UTC-10).

### C. Maui Inverter Target Candidates (`candidate_1` to `candidate_5`)
Due to multi-channel data logger configurations across 11 individual inverter channels (`inv_01` to `inv_11`), 5 target candidate hypotheses were reconstructed:
* **Candidate 1 (Naive All-Channel Sum):** Sum of all 11 channels. Peak: 234.09 kW (exceeds 110 kW DC nameplate by 2.13× due to sub-combiner double counting).
* **Candidate 2 (Six Disjoint Strings):** Sum of `inv_01, inv_02, inv_03, inv_05, inv_07, inv_08`. Peak: 62.49 kW.
* **Candidate 3 (Five Sub-Combiners):** Sum of `inv_04, inv_06, inv_09, inv_10, inv_11`. Peak: 171.60 kW.
* **Candidate 4 (Single Sub-Array):** Meter `inv_04` only. Peak: 43.43 kW.
* **Candidate 5 (Plausible Plant Configuration):** Sum of `inv_04 + inv_10`. Peak: 86.85 kW AC. Yields an Inverter Loading Ratio (ILR) of 1.267 relative to the 110 kW DC nameplate rating.
* **Validation Status:** Plant AC-output reconstruction remains formally **UNVERIFIED** pending site electrical single-line diagrams.

---

## 3. Data Dictionary Summary

| Variable Name | Standardized Column | Physical Unit | Range / Values | Description |
|---|---|---|---|---|
| Global Irradiance | `GHI` | $\text{W/m}^2$ | $0.0 - 1300.0$ | Global Horizontal Irradiance measured by pyranometer |
| Ambient Temperature | `Temperature` | $^{\circ}\text{C}$ | $-10.0 - 55.0$ | Ambient air dry-bulb temperature |
| Relative Humidity | `Relative_Humidity` | $\%$ | $0.0 - 100.0$ | Ambient atmospheric relative humidity |
| Wind Speed | `Wind_Speed` | $\text{m/s}$ | $0.0 - 45.0$ | Anemometer horizontal wind velocity |
| Wind Direction | `Wind_Direction` | $^{\circ}$ (degrees) | $0.0 - 360.0$ | Compass wind direction (0° = North) |
| Solar Zenith Angle | `SZ_Angle` | $^{\circ}$ (degrees) | $0.0 - 180.0$ | Angle between sun position and local vertical |
| Solar Azimuth Angle | `SA_Angle` | $^{\circ}$ (degrees) | $0.0 - 360.0$ | Sun horizontal compass angle from true North |
| AC Power Output | `AC_Output` | $\text{kW}$ | $0.0 - 1250.0$ (Demo)<br>$0.0 - 250.0$ (Maui) | Measured alternating-current active power |

---

## 4. Usage Instructions

To read the standard demo dataset in any Python script:
```python
import pandas as pd
from pathlib import Path

data_path = Path("datasets/energy_prediction/demo_datasets/Solar_Weather_Data_Processed.csv")
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip().str.replace(" ", "_")
print(f"Loaded {len(df)} samples across {len(df.columns)} columns.")
```

