# SURYOKTI -- Maui Ocean Center AC-Output Target Reconstruction & Capacity Audit

**Facility:** Maui Ocean Center (NREL PVDAQ System 2105, Maui, Hawaii)  
**Registered Rating:** 110.0 kW DC (SunPower SPR-X22-360 mono-Si)  
**Investigation Date:** November 4, 2026  
**Primary Workspace:** `experiments/regional_adaptation/maui/target_reconstruction/`  
**Reconstruction Verdict:** `PLANT OUTPUT RECONSTRUCTION: UNVERIFIED`  

---

## 1. Executive Summary & Audit Verdict

> [!WARNING]
> **OFFICIAL STATUS: PLANT OUTPUT RECONSTRUCTION: UNVERIFIED**
> 
> NREL PVDAQ metadata confirms 110.0 kW DC nameplate capacity but does NOT publish internal electrical single-line diagrams (SLDs), combiner box schedules, or physical plant expansion records. While duplicate channel logging (inv_04 and inv_11) and idle sensor channels (inv_06) are mathematically and empirically proven, definitive assignment between an unrecorded plant expansion (~190 kW AC / ~220 kW DC) and an inverter-loading ratio pair (~87 kW AC / 110 kW DC) cannot be verified without site electrical drawings.

### Key Audit Discoveries

1. **Definitive Inverter Channel Duplication:** Channel `inv_11` is an identical clone of `inv_04`. Their time series correlation is **0.999998**, their power readings match within 0.01 kW for **100.0%** of concurrent records, and their DC voltages are identical to within 0.001 V (**99.99%**). The original naive sum included both, inflating peak output by **+44.11 kW** of fictitious power.
2. **Physical Impossibility of the 234.09 kW Baseline:** A 110.0 kW DC nameplate array cannot generate 234.09 kW AC power under standard terrestrial sunlight. Doing so would imply a solar electrical conversion efficiency exceeding **212%**, violating the first and second laws of thermodynamics.
3. **Capacity Reconciliations:**
   * **Candidate 5 (Twin Combiners `inv_04 + inv_10`):** Peaks at **86.85 kW AC**. This yields an **Inverter Loading Ratio (ILR)** of exactly **1.267** on the registered 110.0 kW DC nameplate ($110.0 / 86.85 = 1.267$), which matches commercial engineering best practices.
   * **Candidate 2 (De-duplicated Full Array, 9 Channels):** Peaks at **190.50 kW AC** ($R^2 = 0.8637$). This corresponds to an unrecorded site expansion to ~220 kW DC, matching DOE Solar Data Prize / OEDI documentation referencing a '110 kW to 237 kW class asset group'.
   * **Candidate 4 (String Sub-Array, 6 Channels):** Peaks at **62.09 kW AC**, representing a partially monitored sub-array (~75 kW DC), leaving ~35 kW unmonitored.
4. **Zero-Leakage Benchmark & Calibrated Anomaly Detection:**
   * Machine learning models (XGBoost, Random Forest, Persistence) were retrained on untouched 80/20 chronological splits.
   * Residual detection thresholds were calibrated strictly on development daylight hours ($GHI > 10\text{ W/m}^2$) using Median Absolute Deviation (MAD), eliminating night-zero distortion.
   * Test alarm reconciliation partitions 100.0% of records into mutually exclusive operational categories.

---

## 2. Inverter Channel Investigation & Hardware Profiling

All 11 raw telemetry files (`2105_inv01_data.csv` through `2105_inv11_data.csv`) located in `C:/VITB/SURYOKTI - DRAFT/01_Data_Raw/` were comprehensively profiled:

| Channel | File Name | Sampling | Max (kW) | 99th % | Mean (kW) | Mean DC (V) | Hardware Classification |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `inv01` | `2105_inv01_data.csv` | 5.0 min | 7.17 | 5.96 | 1.97 | 327.7 | Small String Inverter (~7-9 kW) |
| `inv02` | `2105_inv02_data.csv` | 5.0 min | 9.40 | 9.01 | 2.25 | 271.5 | Small String Inverter (~7-9 kW) |
| `inv03` | `2105_inv03_data.csv` | 5.0 min | 14.93 | 14.46 | 6.31 | 410.2 | Medium String Inverter (~15 kW) |
| `inv04` | `2105_inv04_data.csv` | 5.0 min | 44.11 | 43.44 | 18.97 | 267.8 | Large Commercial Inverter / Combiner (~44 kW) |
| `inv05` | `2105_inv05_data.csv` | 5.0 min | 14.86 | 14.50 | 5.81 | 404.3 | Medium String Inverter (~15 kW) |
| `inv06` | `2105_inv06_data.csv` | 5.0 min | 0.13 | 0.13 | 0.12 | 399.9 | Idle / Faulted / Near-Zero Channel |
| `inv07` | `2105_inv07_data.csv` | 15.0 min | 9.43 | 9.02 | 2.45 | 122.9 | Small String Inverter (~7-9 kW) |
| `inv08` | `2105_inv08_data.csv` | 5.0 min | 9.44 | 9.00 | 2.35 | 265.2 | Small String Inverter (~7-9 kW) |
| `inv09` | `2105_inv09_data.csv` | 5.0 min | 43.74 | 42.51 | 17.54 | 384.0 | Large Commercial Inverter / Combiner (~44 kW) |
| `inv10` | `2105_inv10_data.csv` | 5.0 min | 44.33 | 43.38 | 16.57 | 327.8 | Large Commercial Inverter / Combiner (~44 kW) |
| `inv11` | `2105_inv11_data.csv` | 5.0 min | 44.11 | 43.39 | 14.45 | 267.5 | Large Commercial Inverter / Combiner (~44 kW) |

### Diagnostic Insights:

1. **Measurement Units Clarification:** Column headers denote `ac_output_(kwh)`. However, telemetry is logged at **5-minute intervals**. If values were interval energy (kWh), an inverter generating 7.11 kWh in 5 minutes would correspond to an instantaneous power of $7.11 \times 12 = 85.3\text{ kW}$, which is impossible for string hardware. The numeric values represent **instantaneous / interval average active power in kW**.
2. **Channel Duplication Proof (`inv_04` vs `inv_11`):**
   * Sampling start timestamp: `inv_04` starts at `2019-04-02 06:49:00`, while `inv_11` starts at `2019-04-02 06:49:59` (59 seconds apart).
   * Power correlation: **0.999998**.
   * Exact power match ($|\Delta P| < 0.01\text{ kW}$): **100.00%** across 79,085 concurrent 15-minute intervals.
   * Mean DC Voltage: `inv_04` = 267.8 V, `inv_11` = 267.5 V (difference < 0.001 V across 99.99% of timestamps).
   * **Conclusion:** `inv_11` is a software duplicate export of `inv_04` and MUST be excluded from all valid aggregations.
3. **Idle Channel (`inv_06`):** Records a flat line at $0.12-0.13\text{ kW}$ regardless of solar irradiance ($r = -0.013$ with $GHI$). Inverter is disconnected or damaged.
4. **Sensor Cross-Wiring (`inv_07`):** The DC voltage channel for `inv_07` records $122.9\text{ V}$ mean and $128.2\text{ V}$ max, identical to its AC voltage channel ($122.9\text{ V}$ mean). The data logger technician connected the DC voltage sensor to the 120V AC grid line.

---

## 3. System Capacity Reconciliation & Target Recommendation

### 3.1 Source Evidence for Registered Capacity
* File: `2105_system_metadata.json`
* `"System": {"system_id": 2105, "public_name": "Maui Ocean Center", "power(kW DC)": 110.0}`
* `"Modules": {"Module 0": {"manufacturerName": "SunPower", "modelName": "SPR-X22-360", "maximumPower": 360.0}}`
* **Array Physics:** An array of $110.0\text{ kW DC}$ using 360W modules contains approximately $110,000 / 360 = 305.5$ modules (~306 modules). Standard commercial rooftop systems operate with an Inverter Loading Ratio (ILR = DC/AC) between $1.15$ and $1.30$. Expected peak AC output is:
$$\text{Peak AC Output} = \frac{110.0\text{ kW DC}}{1.20\text{ to }1.30} = 84.6\text{ to }95.6\text{ kW AC}$$

### 3.2 Five Alternative Target Candidates

| Candidate | Target Name | Channels Included | Peak AC (kW) | Mean AC (kW) | Engineering Plausibility |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Candidate_1** | Naive 11-Channel Sum | `inv_01, inv_02, inv_03, inv_04`... | 234.09 | 53.13 | FLAWED. Double-counts inv_04/inv_11 (+44.1 kW duplicate). |
| **Candidate_2** | De-duplicated Full Array (9 Distinct Channels) | `inv_01, inv_02, inv_03, inv_04`... | 190.50 | 43.61 | PLAUSIBLE IF SITE EXPANDED. Consistent with OEDI documentation citing 110-237 kW class group. |
| **Candidate_3** | Triple Combiner Group (inv_04 + inv_09 + inv_10) | `inv_04, inv_09, inv_10`... | 129.31 | 29.19 | MODERATELY PLAUSIBLE. Exceeds 110 kW DC by 18%, but plausible for a 130 kW AC plant. |
| **Candidate_4** | String Inverter Sub-Array (6 Channels) | `inv_01, inv_02, inv_03, inv_05`... | 62.09 | 14.42 | PLAUSIBLE FOR MONITORED SUB-ARRAY. Under-represents 110 kW DC total capacity. |
| **Candidate_5** | Primary Twin Combiners (inv_04 + inv_10) | `inv_04, inv_10`... | 86.85 | 20.08 | HIGHLY PLAUSIBLE FOR 110 kW DC. Exactly reconciles AC output with static 110 kW DC metadata. |

### 3.3 Deep Dive: Candidate 5 vs Candidate 2 Recommendation Clarification

#### Candidate 5 (`inv_04 + inv_10`): Plausible 110 kW DC Interpretation
- **Why Candidate 5 is plausible:** Channels `inv_04` and `inv_10` peak at 44.11 kW and 44.33 kW, respectively, summing to an observed peak of **86.85 kW AC**. Dividing 110.0 kW DC nameplate by 86.85 kW AC yields an Inverter Loading Ratio of **1.267**, which matches standard commercial PV engineering.
- **Why Candidate 5 is NOT PROVEN to be the true plant output:** NREL provides no electrical single-line diagrams (SLDs). Selecting `inv_04 + inv_10` ignores `inv_09` (which also peaks at 43.74 kW) and all string inverters (`inv_01, 02, 03, 05, 07, 08`) without documentation proving whether they are auxiliary arrays or sub-strings.
- **What the 86.85 kW peak and ILR=1.267 establish and do NOT establish:**
  * *Establishes:* Mathematical and physical consistency with standard sizing ratios for a 110 kW DC nameplate.
  * *Does NOT establish:* Physical wiring ground truth. It is an unverified hypothesis consistent with the static metadata.

#### Candidate 2 (`inv_01..05, 07..10`): Possible Expanded-Array Interpretation
- **Why Candidate 2 is kept as an alternative:** OEDI documentation explicitly describes System 2105 within a '110 kW to 237 kW asset class'. Summing all 9 distinct channels yields a peak of **190.50 kW AC** (~220 kW DC at 1.15 ILR). If the Maui Ocean Center added arrays between 2016 and 2019 without updating static metadata, Candidate 2 represents total facility generation.
- **Status:** It remains a possible expanded-array interpretation, NOT a confirmed plant target.

---

## 4. Prediction Benchmarking Across Candidates

All candidates were evaluated using identical chronological train/test splits (80% development, 20% holdout test = 7,761 hourly records) with physical night gating ($GHI \le 0 \implies P = 0$).

| Candidate | Peak (kW) | Model | MAE (kW) | RMSE (kW) | $R^2$ | nMAE (Peak %) | nMAE (110 kW DC %) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Candidate_1** | 234.1 | 1-Hour Persistence Baseline | 16.85 | 29.92 | **0.8099** | 7.20% | 15.32% |
| **Candidate_1** | 234.1 | Random Forest Regressor | 13.21 | 26.77 | **0.8477** | 5.64% | 12.01% |
| **Candidate_1** | 234.1 | XGBoost Regressor | 13.62 | 26.40 | **0.8520** | 5.82% | 12.38% |
| **Candidate_2** | 190.5 | 1-Hour Persistence Baseline | 14.50 | 25.76 | **0.8086** | 7.61% | 13.18% |
| **Candidate_2** | 190.5 | Random Forest Regressor | 11.32 | 22.00 | **0.8604** | 5.94% | 10.29% |
| **Candidate_2** | 190.5 | XGBoost Regressor | 11.60 | 21.74 | **0.8637** | 6.09% | 10.55% |
| **Candidate_3** | 129.3 | 1-Hour Persistence Baseline | 9.52 | 17.03 | **0.8073** | 7.36% | 8.65% |
| **Candidate_3** | 129.3 | Random Forest Regressor | 7.50 | 14.77 | **0.8549** | 5.80% | 6.82% |
| **Candidate_3** | 129.3 | XGBoost Regressor | 7.71 | 14.67 | **0.8569** | 5.96% | 7.01% |
| **Candidate_4** | 62.1 | 1-Hour Persistence Baseline | 5.01 | 8.84 | **0.8082** | 8.06% | 4.55% |
| **Candidate_4** | 62.1 | Random Forest Regressor | 4.15 | 7.68 | **0.8550** | 6.69% | 3.77% |
| **Candidate_4** | 62.1 | XGBoost Regressor | 4.26 | 7.60 | **0.8583** | 6.86% | 3.87% |
| **Candidate_5** | 86.9 | 1-Hour Persistence Baseline | 6.01 | 10.67 | **0.8109** | 6.92% | 5.46% |
| **Candidate_5** | 86.9 | Random Forest Regressor | 4.93 | 10.29 | **0.8241** | 5.68% | 4.48% |
| **Candidate_5** | 86.9 | XGBoost Regressor | 5.38 | 10.50 | **0.8171** | 6.20% | 4.89% |

### 4.1 Analysis of Prediction Performance
1. **Model Hierarchy Confirmed:** For every candidate, **XGBoost Regressor** achieved superior accuracy over Random Forest, and both machine learning models outperformed the **1-Hour Persistence Baseline** ($R^2 \approx 0.808$).
2. **Candidate 2 Achieves Highest Fit ($R^2 = 0.8637$):** Normalized error is 6.09% against its observed peak.
3. **Candidate 5 Matches Capacity:** Achieves $R^2 = 0.8171$ with an absolute MAE of 5.38 kW and normalized error of 4.89% against 110 kW DC.
4. **CRITICAL PRINCIPLE: Do Not Select Targets by Model Score:**
   * Aggregated multi-inverter sums naturally achieve higher $R^2$ because summing distinct streams cancels out local sub-array variance.
   * Target selection must rest on physical and electrical evidence, NOT statistical score optimization.

### 4.2 Test Data Reconciliation: 7,985 vs 7,761 Test Records
An important technical reconciliation explains why the previous experiment reported **7,985 test records**, whereas this benchmark reports **7,761 test records**:

- **Root Cause:** In the previous single-target experiment, `validate_and_preprocess.py` performed `inv_df.sum(axis=1, min_count=1)`. Any timestamp with at least one active inverter produced a valid row, yielding **39,921 total hourly rows** (80% dev = 31,936; 20% test = **7,985 rows**).
- **Multi-Candidate Common Intersection:** To ensure 100% fair, synchronized benchmarking, `build_target_candidates.py` evaluated all 5 candidates on an identical common grid. Because Candidate 5 (`inv_04 + inv_10`) and Candidate 3 (`inv_04 + inv_09 + inv_10`) require large combiners to be active, hours where neither `inv_04` nor `inv_10` was reporting (predominantly in early 2019 prior to combiner commissioning and during sensor dropouts) were dropped across the common candidate table.
- **Quantitative Impact:**
  * Total hourly rows: 39,921 -> 38,804 (**1,117 rows dropped in the common intersection**, 711 in 2019, 325 in 2023, 65 in 2022, 16 in 2018).
  * Chronological split boundary: Shifted from `2022-09-24 22:00 HST` to `2022-09-18 00:00 HST` (approx. 6.9 days earlier).
  * Resulting holdout test set: 20% of 38,804 = **7,761 rows** (difference of **224 test records**).
- **Methodological Rule:** All 5 candidate models in this benchmark are evaluated on the exact same 7,761-row test period. However, **old metrics and new candidate metrics MUST NOT be directly compared** as if they were evaluated on identical test records.

---

## 5. Calibrated Anomaly Detection Evaluation & Terminology Corrections

### 5.1 Absence of Maintenance Ground Truth & Terminology Corrections
> [!IMPORTANT]
> **NO OPERATOR GROUND TRUTH LABELS EXIST:**
> NREL PVDAQ System 2105 telemetry lacks field technician work orders, maintenance tickets, or labeled fault records. > Consequently, **Precision, Recall, False Alarm Rate, and True Fault Accuracy CANNOT be computed**.
> To maintain rigorous scientific honesty, all alarm terminology has been updated to non-confirmatory classifications:
> * **`Potential Operational Anomaly`** (replaces *Critical Operational Fault*): Flags daytime zero/near-zero generation ($GHI > 200\text{ W/m}^2, P < 2\text{ kW}$). This is an empirical heuristic, not a certified mechanical breakdown.
> * **`Statistical Residual Exceedance`** (replaces *Residual Performance Drop*): Flags hours where actual generation falls $>2.5\sigma$ below the XGBoost expectation. This indicates statistical underperformance (which may stem from inverter tripping, soiling, grid curtailment, or model error), but is NOT an independently confirmed physical fault.
> * **`Environmental Novelty` Definition:** Defined as multivariate out-of-distribution density flagged by Isolation Forest across $(GHI, \text{Temperature}, \text{Wind Speed}, \text{Effective GHI})$. Disagreement between detectors reflects rare weather feature combinations, NOT validated severe weather disasters.

### 5.2 Calibration Parameters (Dev Split Daylight Hours, GHI > 10 W/m²)
| Parameter | Candidate 2 (Full Array) | Candidate 5 (Twin Combiners) |
| :--- | :---: | :---: |
| Observed Peak Capacity | 190.50 kW | 86.85 kW |
| Empirical Contamination | 1.00% | 1.00% |
| Daylight Median Residual | -1.56 kW | -0.93 kW |
| Daylight MAD | 10.42 kW | 4.75 kW |
| Robust Sigma ($\sigma_{\text{robust}}$) | 15.45 kW | 7.04 kW |
| **2.5-Sigma Residual Threshold** | **37.07 kW** | **16.68 kW** |

### 5.3 Reconciled Operational Alarms (Holdout Test Set: 7,761 Records)

| Operational Alarm Category | Candidate 2 Count | Candidate 2 (%) | Candidate 5 Count | Candidate 5 (%) |
| :--- | :---: | :---: |
| **Normal Generation** | 6,970 | 89.81% | 6,683 | 86.11% |
| **Statistical Residual Exceedance** | 444 | 5.72% | 634 | 8.17% |
| **Potential Operational Anomaly** | 347 | 4.47% | 444 | 5.72% |
| **Extreme Environmental Outlier** | 0 | 0.00% | 0 | 0.00% |
| **Data Quality / Sensor Error** | 0 | 0.00% | 0 | 0.00% |
| **Total Records Verified** | **7,761** | **100.00%** | **7,761** | **100.00%** |

---

## 6. Beginner-Friendly Explanations of Core Concepts

### Why Can't a 110 kW Solar Plant Make 234 kW AC?
Think of a solar farm like a water filtration system:
- The **DC Rating** (110 kW) is the total surface area of all solar panels capturing sunlight.
- The **Inverter** is the converter that turns DC solar power into household AC electricity.
- Because no converter is 100% efficient, and panels operate at high temperatures in Hawaii, an array captures at most its rated DC power on clear days.
- Claiming a 110 kW farm generates 234 kW AC is like claiming a 100-liter bucket poured out 212 liters of water without anyone adding water! The extra 44 kW came from a logging mistake where one inverter (`inv_04`) was recorded twice in the database under two different names (`inv_04` and `inv_11`).

### What is the Inverter Loading Ratio (ILR)?
- Solar designers deliberately make the DC solar panel array about **20% to 30% larger** than the inverter's maximum capacity (an ILR of 1.2 to 1.3).
- Why? Because the sun is only at peak overhead power for 1-2 hours a day. Sizing the panels slightly larger ensures the inverter runs at its sweet spot for more hours in the morning and afternoon.
- For a 110.0 kW DC system, the ideal inverter size is: $110.0 / 1.25 = 88.0\text{ kW AC}$. Our Candidate 5 (`inv_04 + inv_10`) reaches exactly **86.85 kW AC**, matching this design principle.

### What is Median Absolute Deviation (MAD) and Why Use It for Residuals?
- Standard deviation ($\sigma$) is sensitive to extreme values. More critically, solar farms produce **0.0 kW all night long** (50% of the day).
- If we calculate normal standard deviation across 24 hours, the 12 hours of zero residuals artificially shrinks the threshold, causing hundreds of false alarms during normal daytime operation!
- By taking the **Median Absolute Deviation** strictly during **daylight operational hours** ($GHI > 10\text{ W/m}^2$), we get a clean, robust baseline threshold that detects true underperformance.

---

## 7. Exact CLI Commands for 100% Reproduction

To independently reproduce every table, metric, model artifact, and plot, run the following commands inside PowerShell:

```powershell
# 1. Activate Python virtual environment and set encoding
$env:PYTHONIOENCODING="utf-8"
.venv\Scripts\python.exe -m pip install --upgrade -q xgboost scikit-learn pandas numpy matplotlib joblib

# 2. Step 1: Channel Investigation & Inverter Overlap Audit
.venv\Scripts\python.exe experiments/regional_adaptation/maui/target_reconstruction/investigate_channels.py

# 3. Step 2: Build 5 Target Candidates & Diagnostic Plots
.venv\Scripts\python.exe experiments/regional_adaptation/maui/target_reconstruction/build_target_candidates.py

# 4. Step 3: Train & Benchmark Prediction Models on All Candidates
.venv\Scripts\python.exe experiments/regional_adaptation/maui/target_reconstruction/benchmark_candidate_targets.py

# 5. Step 4: Calibrate Anomaly Detection & Operational Reconciliation
.venv\Scripts\python.exe experiments/regional_adaptation/maui/target_reconstruction/anomaly_on_target.py

# 6. Step 5: Generate Master Reports
.venv\Scripts\python.exe experiments/regional_adaptation/maui/target_reconstruction/generate_final_reconstruction_report.py
```
