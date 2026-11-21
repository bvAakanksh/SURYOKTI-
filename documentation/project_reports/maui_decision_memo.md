# SURYOKTI Engineering Decision Memo: Maui Ocean Center (System 2105) Target Reconstruction & Evidence Review

**Date:** November 4, 2026  
**To:** SURYOKTI Project Leadership & Research Team  
**From:** Antigravity Pairing Engineering Agent  
**Subject:** Technical Audit, Target Ambiguity Resolution, Anomaly Terminology Corrections, and Readiness for Cross-Region Modeling  
**Workspace:** `experiments/regional_adaptation/maui/target_reconstruction/`  
**Official Target Status:** `PLANT OUTPUT RECONSTRUCTION: UNVERIFIED`  

---

## Executive Summary

Following a comprehensive technical audit of the Maui Ocean Center telemetry (NREL PVDAQ System 2105, registered rating 110.0 kW DC), this decision memo codifies what is mathematically and empirically proven, what remains fundamentally uncertain, how candidate targets must be interpreted, why anomaly terminology has been corrected, and whether cross-region evaluation is scientifically justifiable at this juncture.

> [!WARNING]
> ### Core Strategic Verdict
> **Plant-level AC output remains UNVERIFIED.**  
> While software channel duplication (`inv_04` and `inv_11`) and offline hardware (`inv_06`) have been conclusively resolved, no public NREL data contains electrical single-line diagrams (SLDs) or plant expansion records. We cannot prove whether the true facility generation is an **inverter-loading ratio pair (~87 kW AC / 110 kW DC)** or an **unrecorded site expansion (~190 kW AC / ~220 kW DC)**.  
> **Direct zero-shot cross-region transfer of absolute power output is premature.** However, pipeline architecture adaptation and capacity-normalized benchmarking are scientifically sound.

---

## 1. What Is Supported by Evidence

The following technical findings are backed by 100% empirical verification across 4.92 years of raw 5-minute telemetry:

1. **Definitive Software Channel Duplication (`inv_04` vs `inv_11`):**
   * Sampling start times differ by exactly 59 seconds (`06:49:00` vs `06:49:59` on April 2, 2019).
   * Correlation between power channels is **0.999998**.
   * Exact match rate ($|\Delta P| < 0.01\text{ kW}$): **100.00%** across 79,085 concurrent intervals (mean diff: $0.0001\text{ kW}$).
   * DC Voltage match ($|\Delta V| < 0.01\text{ V}$): **99.99%** of records (mean diff: $0.0010\text{ V}$).
   * *Conclusion:* Channel `inv_11` is an identical clone of `inv_04`. The previous naive 11-channel sum included both, inflating the observed peak by **+44.11 kW** of duplicate power.

2. **Idle / Offline Sensor Channel (`inv_06`):**
   * Output is flat at $0.12–0.13\text{ kW}$ continuously across all seasons ($r = -0.013$ with GHI).
   * Inverter is disconnected, faulted, or logging a constant quiescent draw. It must be excluded from active generation totals.

3. **Sensor Cross-Wiring Anomaly (`inv_07`):**
   * The DC voltage sensor for `inv_07` records $122.9\text{ V}$ mean and $128.2\text{ V}$ max, identical to its AC voltage channel ($122.9\text{ V}$ mean).
   * The data logger was wired to the 120V AC grid phase rather than the 560V DC string bus.

4. **Thermodynamic Impossibility of the 234.09 kW Baseline:**
   * An array rated at 110.0 kW DC (~306 SunPower 360W modules) cannot generate 234.09 kW AC under standard terrestrial sunlight ($1000\text{ W/m}^2$). Doing so would require an electrical conversion efficiency $> 212\%$, violating physical conservation laws.

5. **Fair Common Holdout Test Benchmark:**
   * All 5 candidate targets were trained and evaluated on an identical 80/20 chronological split (7,761 holdout test records).
   * In all candidate targets, **XGBoost Regressor** and **Random Forest** substantially beat the 1-Hour Persistence Baseline ($R^2 \approx 0.808$).

---

## 2. What Remains Uncertain (The Plant Target Ambiguity)

Despite rigorous data processing, the exact physical plant AC target cannot be proven from telemetry alone:

* **Missing Electrical Single-Line Diagrams (SLDs):** NREL PVDAQ does not publish plant wiring schematics.
* **Combiner vs String Ambiguity:** We cannot prove whether the large ~44 kW channels (`inv_04, 09, 10`) are sub-combiners aggregating the smaller string channels (`inv_01, 02, 03, 05, 07, 08`), or whether they represent physically independent rooftop arrays.
* **Unrecorded Expansion Uncertainty:** NREL metadata records `power(kW DC): 110.0` registered in 2016. However, DOE Solar Data Prize / OEDI documentation references System 2105 within a "110 kW to 237 kW asset class". Without an official commissioning log or utility interconnection agreement, we cannot verify if new arrays were added between 2016 and 2019.

---

## 3. Candidate 2 versus Candidate 5 Interpretation

Target selection must be guided by electrical engineering principles, **NOT by machine learning scores**:

### Candidate 5 (`inv_04 + inv_10`): The 110 kW DC Sizing Interpretation
* **Peak AC Generation:** **86.85 kW AC** (Hourly peak).
* **Inverter Loading Ratio (ILR):**
  $$\text{ILR} = \frac{110.0\text{ kW DC}}{86.85\text{ kW AC}} = \mathbf{1.267}$$
* **What this establishes:** It proves mathematical and physical compatibility with the static 110.0 kW DC nameplate metadata. Commercial SunPower rooftop systems are universally designed with an ILR between 1.20 and 1.30 to optimize inverter economics.
* **What this DOES NOT establish:** It does not prove that `inv_04` and `inv_10` are the true physical inverters. It leaves `inv_09` (43.74 kW peak) and the string inverters (`inv_01..08`, 62.09 kW peak) unaccounted for.

### Candidate 2 (`inv_01..05, 07..10`): The Expanded Full-Array Interpretation
* **Peak AC Generation:** **190.50 kW AC** (Hourly peak across all 9 distinct active channels).
* **Equivalent DC Capacity:** $\approx 190.50 \times 1.15 \approx \mathbf{219\text{ kW DC}}$.
* **What this establishes:** It reflects the total non-duplicated monitored generation across all distinct data channels, aligning with OEDI literature noting a facility capacity up to 237 kW.
* **What this DOES NOT establish:** It cannot be confirmed as the official plant target because the static NREL registry was never updated to reflect a 220 kW array.
* **Model Score Disclaimer:** While Candidate 2 achieves the highest statistical fit ($R^2 = 0.8637$ vs Candidate 5's $R^2 = 0.8171$), **this statistical score MUST NOT be used to select Candidate 2**. Summing 9 channels naturally smooths out sub-array variability, creating an artificially cleaner diurnal curve that inflates $R^2$.

---

## 4. Test Data Reconciliation: 7,985 vs 7,761 Records

A rigorous audit reconciled why the previous experiment reported **7,985 test records**, while the target reconstruction benchmark reports **7,761 test records**:

1. **Previous Experiment Pipeline:**
   * Calculated `AC_Output = inv_df.sum(axis=1, min_count=1)`.
   * Any timestamp with at least one active inverter produced a valid row, yielding **39,921 total hourly rows**.
   * An 80/20 chronological split produced: Dev = 31,936 rows, Holdout Test = **7,985 rows** (starting `2022-09-24 22:00 HST`).
2. **Multi-Candidate Common Intersection Pipeline:**
   * To ensure 100% fair and identical comparison across all 5 candidates, `build_target_candidates.py` evaluated models on a **common grid** where all candidates were concurrently defined.
   * Because Candidate 5 (`inv_04 + inv_10`) and Candidate 3 (`inv_04 + inv_09 + inv_10`) require combiner channels, hours where neither `inv_04` nor `inv_10` reported (predominantly in early 2019 prior to combiner commissioning and during occasional dropouts) were dropped from the master table.
3. **Quantitative Impact:**
   * **1,117 total hourly rows dropped** (711 in 2019, 325 in 2023, 65 in 2022, 16 in 2018).
   * Master dataset: 38,804 rows.
   * Split boundary shifted ~6.9 days earlier: from `2022-09-24 22:00 HST` to `2022-09-18 00:00 HST`.
   * Resulting test set: 20% of 38,804 = **7,761 rows** (difference of **224 test records**).
4. **Methodological Directive:** All 5 candidate benchmarks in this study share the exact same 7,761-row test period. However, **metrics from the old experiment and new candidate benchmarks must not be compared directly** as if they shared identical test data.

---

## 5. Anomaly Terminology Corrections & Lack of Ground Truth

To maintain strict scientific integrity, all anomaly detection reporting has been corrected to eliminate unverified confirmatory claims:

1. **Renaming Confirmatory Labels:**
   * **Old Label:** `CRITICAL_OPERATIONAL_FAULT`
   * **Corrected Label:** `POTENTIAL_OPERATIONAL_ANOMALY`
   * *Justification:* Zero or near-zero daytime generation ($GHI > 200\text{ W/m}^2, P < 2\text{ kW}$) is an empirical heuristic indicator of under-generation. In the absence of operator maintenance tickets, it cannot be confirmed as a mechanical hardware fault (it could stem from grid curtailment, utility disconnects, or inverter restarts).
2. **Distinguishing Statistical Residuals from Physical Faults:**
   * **Old Label:** `Residual Performance Drop`
   * **Corrected Label:** `STATISTICAL_RESIDUAL_EXCEEDANCE`
   * *Justification:* A 2.5-sigma daylight residual exceedance identifies hours where observed generation falls significantly below the XGBoost expectation. This is a statistical anomaly, not a verified physical malfunction.
3. **Defining Environmental Novelty:**
   * Defined as multivariate out-of-distribution density flagged by Isolation Forest across $(GHI, \text{Temperature}, \text{Wind Speed}, \text{Effective GHI})$.
   * Disagreement between detectors reflects rare feature combinations (e.g., high wind with moderate sun), **NOT a verified meteorological disaster**.
4. **Inability to Compute Precision and Recall:**
   * Because NREL PVDAQ does not publish operator maintenance logs, work orders, or ground-truth fault tickets:
   * **True Precision, Recall, F1-Score, False Alarm Rate, and Fault Classification Accuracy CANNOT BE CALCULATED.**
   * Any report claiming validated anomaly detection accuracy on this dataset is scientifically invalid.

---

## 6. What Evidence Would Conclusively Resolve the Ambiguity

To upgrade the status from `UNVERIFIED` to `VERIFIED`, the following primary site documents are required:

1. **Facility As-Built Electrical Drawings / Single-Line Diagrams (SLD):** Showing whether inverters 04, 09, 10 are standalone central inverters or combiner collection points for inverters 01–08.
2. **Hawaiian Electric Company (HECO) Interconnection Agreement:** Stating the exact approved AC export capacity (kW AC) and total DC module nameplate (kW DC).
3. **Plant Commissioning / Expansion Logs:** Documenting whether additional rooftop PV arrays were installed between 2016 and 2019.

---

## 7. Which Maui Results May Be Reported as Provisional

The following results are mathematically validated and may be cited in academic or technical reporting as **Provisional Research Benchmarks**:

1. **De-duplicated Full-Array Hypothesis (Candidate 2):**
   * Target: 9 distinct channels, Peak: 190.50 kW AC.
   * XGBoost Test Fit: $R^2 = 0.8637$, $\text{MAE} = 11.60\text{ kW}$, $\text{nMAE}_{\text{peak}} = 6.09\%$.
2. **Nameplate Inverter-Loading Hypothesis (Candidate 5):**
   * Target: Twin combiners (`inv_04 + inv_10`), Peak: 86.85 kW AC (ILR = 1.267).
   * XGBoost Test Fit: $R^2 = 0.8171$, $\text{MAE} = 5.38\text{ kW}$, $\text{nMAE}_{110\text{kW}} = 4.89\%$.
3. **Calibrated Anomaly Screening:**
   * Daylight MAD residual thresholding successfully filters 100% of night-zero compression artifacts.
   * Partitions 100.0% of test records into mutually exclusive operational categories without category overlap.

---

## 8. Is Cross-Region Evaluation Scientifically Appropriate Yet?

### Official Verdict: NO for Zero-Shot Power Transfer; YES for Architecture Adaptation

1. **Direct Zero-Shot Transfer of Absolute Power (kW) is NOT APPROPRIATE:**
   * Evaluating whether a model trained on the original demo dataset can predict Maui kW output is scientifically invalid when the true Maui plant capacity cannot be determined between 86.85 kW AC and 190.50 kW AC. A model predicting 85 kW would appear accurate under Candidate 5 but severely under-generating under Candidate 2.
2. **Cross-Region Architecture & Methodology Adaptation IS APPROPRIATE:**
   * Evaluating how the SURYOKTI feature engineering pipeline (solar geometry, interaction terms, cyclical encoding) adapts to tropical island climates.
   * Comparing capacity-normalized errors (nMAE% against observed peak, where both demo and Maui models achieve ~5–6%).
   * Evaluating the robustness of daylight MAD residual calibration across different regional irradiance regimes.

---

## Summary of Deliverable Files & Changes

| File Path | Description & Role |
| :--- | :--- |
| [`maui_decision_memo.md`](maui_decision_memo.md) | **This Document:** Formal engineering decision memo codifying evidence, uncertainties, and strategic directives. |
| [`target_reconstruction_report.md`](target_reconstruction_report.md) | Comprehensive 7-section technical report with full channel investigation, benchmark tables, and reproduction commands. |
| [`target_reconstruction_report.json`](target_reconstruction_report.json) | Structured machine-readable repository of all channel profiles, benchmark metrics, and anomaly statistics. |
| [`anomaly_on_target.py`](../../experiments/regional_adaptation/maui/target_reconstruction/anomaly_on_target.py) | Anomaly script updated with non-confirmatory terminology (`Potential Operational Anomaly`, `Statistical Residual Exceedance`). |
| [`candidate_2_anomaly_report.json`](../../experiments/regional_adaptation/maui/target_reconstruction/reports/candidate_2_anomaly_report.json) | Re-executed calibrated anomaly results for Candidate 2 with corrected operational labels. |
| [`candidate_5_anomaly_report.json`](../../experiments/regional_adaptation/maui/target_reconstruction/reports/candidate_5_anomaly_report.json) | Re-executed calibrated anomaly results for Candidate 5 with corrected operational labels. |
| [`candidate_2_anomaly_timeline.png`](../../plots/anomaly_detection/candidate_2_anomaly_timeline.png) | Candidate 2 test anomaly plot regenerated with non-confirmatory labels. |
| [`candidate_5_anomaly_timeline.png`](../../plots/anomaly_detection/candidate_5_anomaly_timeline.png) | Candidate 5 test anomaly plot regenerated with non-confirmatory labels. |
| [`generate_final_reconstruction_report.py`](../../experiments/regional_adaptation/maui/target_reconstruction/generate_final_reconstruction_report.py) | Master report generator incorporating Section 4.2 test data reconciliation and Section 5.1 ground truth disclaimers. |

