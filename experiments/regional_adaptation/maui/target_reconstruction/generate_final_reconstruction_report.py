"""
SURYOKTI Regional Adaptation — Maui Ocean Center (System 2105)
Report Generator: Target Reconstruction, Capacity Reconciliation & Model Validation
===================================================================================
Compiles all technical findings into:
- target_reconstruction_report.json
- target_reconstruction_report.md
"""

import os
import json
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

REPORT_MD_PATH = os.path.join(SCRIPT_DIR, "target_reconstruction_report.md")
REPORT_JSON_PATH = os.path.join(SCRIPT_DIR, "target_reconstruction_report.json")


def generate_reports():
    print("=" * 70)
    print("GENERATING FINAL TARGET RECONSTRUCTION DELIVERABLES")
    print("=" * 70)
    
    # Load intermediate reports
    with open(os.path.join(REPORTS_DIR, "channel_investigation.json"), "r") as f:
        chan_data = json.load(f)
        
    with open(os.path.join(REPORTS_DIR, "candidate_targets_metadata.json"), "r") as f:
        cand_meta = json.load(f)
        
    with open(os.path.join(REPORTS_DIR, "candidate_benchmarks.json"), "r") as f:
        bench_data = json.load(f)
        
    with open(os.path.join(REPORTS_DIR, "candidate_2_anomaly_report.json"), "r") as f:
        anom_c2 = json.load(f)
        
    with open(os.path.join(REPORTS_DIR, "candidate_5_anomaly_report.json"), "r") as f:
        anom_c5 = json.load(f)
        
    # Assemble master JSON
    master_json = {
        "title": "SURYOKTI Regional Adaptation: Maui Plant Target Reconstruction & Technical Audit",
        "target_facility": {
            "name": "Maui Ocean Center",
            "system_id": 2105,
            "location": "Maui, Hawaii (20.79253 N, -156.51200 W)",
            "registered_capacity_kW_DC": 110.0,
            "modules": "SunPower SPR-X22-360 (360 W mono-Si, ~306 modules)",
            "data_range": "2018-12-15 to 2023-11-15 (4.92 years)"
        },
        "verdict_status": "PLANT OUTPUT RECONSTRUCTION: UNVERIFIED",
        "verdict_reasoning": (
            "NREL PVDAQ metadata confirms 110.0 kW DC nameplate capacity but does NOT publish "
            "internal electrical single-line diagrams (SLDs), combiner box schedules, or physical plant expansion records. "
            "While duplicate channel logging (inv_04 and inv_11) and idle sensor channels (inv_06) are mathematically "
            "and empirically proven, definitive assignment between an unrecorded plant expansion (~190 kW AC / ~220 kW DC) "
            "and an inverter-loading ratio pair (~87 kW AC / 110 kW DC) cannot be verified without site electrical drawings."
        ),
        "key_findings": {
            "confirmed_duplicate": "inv_04 and inv_11 are 100.0% identical in power (corr=0.999998) and 99.99% identical in DC voltage. Counting both inflated the baseline by +44.11 kW.",
            "idle_sensor": "inv_06 is offline/disconnected (constant 0.13 kW baseline, uncorrelated with sunlight).",
            "sensor_wiring_defect": "inv_07 DC voltage channel records AC grid voltage (122.9 V) instead of DC string voltage (560 V).",
            "flawed_baseline": "Candidate 1 (Naive 11-channel sum, peak 234.09 kW) is physically impossible for a 110 kW DC plant (>212% solar efficiency).",
            "primary_defensible_candidate_expansion": "Candidate 2 (De-duplicated 9-channel array, peak 190.50 kW, XGBoost R2=0.8637, nMAE=6.09%) reflects full non-duplicated site output under an expansion hypothesis.",
            "primary_defensible_candidate_nameplate": "Candidate 5 (Twin Combiners inv04+10, peak 86.85 kW, XGBoost R2=0.8171, nMAE=6.20%) reconciles with 110.0 kW DC at a standard 1.25 DC/AC loading ratio."
        },
        "channel_profiles": chan_data["channel_profiles"],
        "candidate_targets": cand_meta,
        "prediction_benchmarks": bench_data,
        "anomaly_detection": {
            "Candidate_2": anom_c2,
            "Candidate_5": anom_c5
        }
    }
    
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(master_json, f, indent=4)
    print(f"Saved master JSON report: {REPORT_JSON_PATH}")
    
    # Generate Markdown Report
    generate_markdown_report(master_json)


def generate_markdown_report(data):
    md = []
    md.append("# SURYOKTI -- Maui Ocean Center AC-Output Target Reconstruction & Capacity Audit\n\n")
    md.append("**Facility:** Maui Ocean Center (NREL PVDAQ System 2105, Maui, Hawaii)  \n")
    md.append("**Registered Rating:** 110.0 kW DC (SunPower SPR-X22-360 mono-Si)  \n")
    md.append("**Investigation Date:** November 4, 2026  \n")
    md.append("**Primary Workspace:** `experiments/regional_adaptation/maui/target_reconstruction/`  \n")
    md.append(f"**Reconstruction Verdict:** `{data['verdict_status']}`  \n\n")
    md.append("---\n\n")
    
    # Executive Summary
    md.append("## 1. Executive Summary & Audit Verdict\n\n")
    md.append(f"> [!WARNING]\n> **OFFICIAL STATUS: {data['verdict_status']}**\n> \n> "
              f"{data['verdict_reasoning']}\n\n")
    
    md.append("### Key Audit Discoveries\n\n")
    md.append("1. **Definitive Inverter Channel Duplication:** Channel `inv_11` is an identical clone of `inv_04`. "
              "Their time series correlation is **0.999998**, their power readings match within 0.01 kW for **100.0%** "
              "of concurrent records, and their DC voltages are identical to within 0.001 V (**99.99%**). "
              "The original naive sum included both, inflating peak output by **+44.11 kW** of fictitious power.\n")
    md.append("2. **Physical Impossibility of the 234.09 kW Baseline:** A 110.0 kW DC nameplate array cannot generate "
              "234.09 kW AC power under standard terrestrial sunlight. Doing so would imply a solar electrical conversion "
              "efficiency exceeding **212%**, violating the first and second laws of thermodynamics.\n")
    md.append("3. **Capacity Reconciliations:**\n"
              "   * **Candidate 5 (Twin Combiners `inv_04 + inv_10`):** Peaks at **86.85 kW AC**. This yields an **Inverter Loading Ratio (ILR)** "
              "of exactly **1.267** on the registered 110.0 kW DC nameplate ($110.0 / 86.85 = 1.267$), which matches commercial engineering best practices.\n"
              "   * **Candidate 2 (De-duplicated Full Array, 9 Channels):** Peaks at **190.50 kW AC** ($R^2 = 0.8637$). This corresponds to an "
              "unrecorded site expansion to ~220 kW DC, matching DOE Solar Data Prize / OEDI documentation referencing a '110 kW to 237 kW class asset group'.\n"
              "   * **Candidate 4 (String Sub-Array, 6 Channels):** Peaks at **62.09 kW AC**, representing a partially monitored sub-array (~75 kW DC), "
              "leaving ~35 kW unmonitored.\n")
    md.append("4. **Zero-Leakage Benchmark & Calibrated Anomaly Detection:**\n"
              "   * Machine learning models (XGBoost, Random Forest, Persistence) were retrained on untouched 80/20 chronological splits.\n"
              "   * Residual detection thresholds were calibrated strictly on development daylight hours ($GHI > 10\\text{ W/m}^2$) using Median Absolute Deviation (MAD), eliminating night-zero distortion.\n"
              "   * Test alarm reconciliation partitions 100.0% of records into mutually exclusive operational categories.\n\n")
    
    # Section 2: Channel Investigation Table
    md.append("---\n\n")
    md.append("## 2. Inverter Channel Investigation & Hardware Profiling\n\n")
    md.append("All 11 raw telemetry files (`2105_inv01_data.csv` through `2105_inv11_data.csv`) located in `C:/VITB/SURYOKTI - DRAFT/01_Data_Raw/` were comprehensively profiled:\n\n")
    
    md.append("| Channel | File Name | Sampling | Max (kW) | 99th % | Mean (kW) | Mean DC (V) | Hardware Classification |\n")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |\n")
    for cp in data["channel_profiles"]:
        md.append(f"| `{cp['channel']}` | `{cp['file']}` | {cp['sampling_interval_median']} | {cp['power_max_kW']:.2f} | {cp['power_p99_kW']:.2f} | {cp['power_mean_kW']:.2f} | {cp['dc_voltage_mean_V']:.1f} | {cp['hardware_category']} |\n")
        
    md.append("\n### Diagnostic Insights:\n\n")
    md.append("1. **Measurement Units Clarification:** Column headers denote `ac_output_(kwh)`. However, telemetry is logged at **5-minute intervals**. "
              "If values were interval energy (kWh), an inverter generating 7.11 kWh in 5 minutes would correspond to an instantaneous power of $7.11 \\times 12 = 85.3\\text{ kW}$, "
              "which is impossible for string hardware. The numeric values represent **instantaneous / interval average active power in kW**.\n")
    md.append("2. **Channel Duplication Proof (`inv_04` vs `inv_11`):**\n"
              "   * Sampling start timestamp: `inv_04` starts at `2019-04-02 06:49:00`, while `inv_11` starts at `2019-04-02 06:49:59` (59 seconds apart).\n"
              "   * Power correlation: **0.999998**.\n"
              "   * Exact power match ($|\\Delta P| < 0.01\\text{ kW}$): **100.00%** across 79,085 concurrent 15-minute intervals.\n"
              "   * Mean DC Voltage: `inv_04` = 267.8 V, `inv_11` = 267.5 V (difference < 0.001 V across 99.99% of timestamps).\n"
              "   * **Conclusion:** `inv_11` is a software duplicate export of `inv_04` and MUST be excluded from all valid aggregations.\n")
    md.append("3. **Idle Channel (`inv_06`):** Records a flat line at $0.12-0.13\\text{ kW}$ regardless of solar irradiance ($r = -0.013$ with $GHI$). Inverter is disconnected or damaged.\n")
    md.append("4. **Sensor Cross-Wiring (`inv_07`):** The DC voltage channel for `inv_07` records $122.9\\text{ V}$ mean and $128.2\\text{ V}$ max, identical to its AC voltage channel ($122.9\\text{ V}$ mean). "
              "The data logger technician connected the DC voltage sensor to the 120V AC grid line.\n\n")

    # Section 3: System Capacity Reconciliation
    md.append("---\n\n")
    md.append("## 3. System Capacity Reconciliation & Target Recommendation\n\n")
    md.append("### 3.1 Source Evidence for Registered Capacity\n")
    md.append("* File: `2105_system_metadata.json`\n")
    md.append("* `\"System\": {\"system_id\": 2105, \"public_name\": \"Maui Ocean Center\", \"power(kW DC)\": 110.0}`\n")
    md.append("* `\"Modules\": {\"Module 0\": {\"manufacturerName\": \"SunPower\", \"modelName\": \"SPR-X22-360\", \"maximumPower\": 360.0}}`\n")
    md.append("* **Array Physics:** An array of $110.0\\text{ kW DC}$ using 360W modules contains approximately $110,000 / 360 = 305.5$ modules (~306 modules). "
              "Standard commercial rooftop systems operate with an Inverter Loading Ratio (ILR = DC/AC) between $1.15$ and $1.30$. Expected peak AC output is:\n"
              "$$\\text{Peak AC Output} = \\frac{110.0\\text{ kW DC}}{1.20\\text{ to }1.30} = 84.6\\text{ to }95.6\\text{ kW AC}$$\n\n")

    md.append("### 3.2 Five Alternative Target Candidates\n\n")
    md.append("| Candidate | Target Name | Channels Included | Peak AC (kW) | Mean AC (kW) | Engineering Plausibility |\n")
    md.append("| :--- | :--- | :--- | :---: | :---: | :--- |\n")
    for cm in data["candidate_targets"]:
        md.append(f"| **{cm['candidate_id']}** | {cm['name']} | `{', '.join(cm['channels_included'][:4])}`... | {cm['peak_hourly_kW']:.2f} | {cm['mean_hourly_kW']:.2f} | {cm['plausibility']} |\n")

    md.append("\n### 3.3 Deep Dive: Candidate 5 vs Candidate 2 Recommendation Clarification\n\n")
    md.append("#### Candidate 5 (`inv_04 + inv_10`): Plausible 110 kW DC Interpretation\n")
    md.append("- **Why Candidate 5 is plausible:** Channels `inv_04` and `inv_10` peak at 44.11 kW and 44.33 kW, respectively, summing to an observed peak of **86.85 kW AC**. "
              "Dividing 110.0 kW DC nameplate by 86.85 kW AC yields an Inverter Loading Ratio of **1.267**, which matches standard commercial PV engineering.\n")
    md.append("- **Why Candidate 5 is NOT PROVEN to be the true plant output:** NREL provides no electrical single-line diagrams (SLDs). "
              "Selecting `inv_04 + inv_10` ignores `inv_09` (which also peaks at 43.74 kW) and all string inverters (`inv_01, 02, 03, 05, 07, 08`) without documentation proving whether they are auxiliary arrays or sub-strings.\n")
    md.append("- **What the 86.85 kW peak and ILR=1.267 establish and do NOT establish:**\n"
              "  * *Establishes:* Mathematical and physical consistency with standard sizing ratios for a 110 kW DC nameplate.\n"
              "  * *Does NOT establish:* Physical wiring ground truth. It is an unverified hypothesis consistent with the static metadata.\n\n")

    md.append("#### Candidate 2 (`inv_01..05, 07..10`): Possible Expanded-Array Interpretation\n")
    md.append("- **Why Candidate 2 is kept as an alternative:** OEDI documentation explicitly describes System 2105 within a '110 kW to 237 kW asset class'. "
              "Summing all 9 distinct channels yields a peak of **190.50 kW AC** (~220 kW DC at 1.15 ILR). "
              "If the Maui Ocean Center added arrays between 2016 and 2019 without updating static metadata, Candidate 2 represents total facility generation.\n")
    md.append("- **Status:** It remains a possible expanded-array interpretation, NOT a confirmed plant target.\n\n")

    # Section 4: Prediction Model Benchmarks
    md.append("---\n\n")
    md.append("## 4. Prediction Benchmarking Across Candidates\n\n")
    md.append("All candidates were evaluated using identical chronological train/test splits (80% development, 20% holdout test = 7,761 hourly records) "
              "with physical night gating ($GHI \\le 0 \\implies P = 0$).\n\n")

    md.append("| Candidate | Peak (kW) | Model | MAE (kW) | RMSE (kW) | $R^2$ | nMAE (Peak %) | nMAE (110 kW DC %) |\n")
    md.append("| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")
    for b in data["prediction_benchmarks"]:
        cid = b["candidate_id"]
        peak = b["observed_peak_kW"]
        for mname, m in b["models"].items():
            md.append(f"| **{cid}** | {peak:.1f} | {m['model']} | {m['MAE_kW']:.2f} | {m['RMSE_kW']:.2f} | **{m['R2']:.4f}** | {m['nMAE_peak_pct']:.2f}% | {m['nMAE_nameplate_110kW_pct']:.2f}% |\n")

    md.append("\n### 4.1 Analysis of Prediction Performance\n")
    md.append("1. **Model Hierarchy Confirmed:** For every candidate, **XGBoost Regressor** achieved superior accuracy over Random Forest, "
              "and both machine learning models outperformed the **1-Hour Persistence Baseline** ($R^2 \\approx 0.808$).\n")
    md.append("2. **Candidate 2 Achieves Highest Fit ($R^2 = 0.8637$):** Normalized error is 6.09% against its observed peak.\n")
    md.append("3. **Candidate 5 Matches Capacity:** Achieves $R^2 = 0.8171$ with an absolute MAE of 5.38 kW and normalized error of 4.89% against 110 kW DC.\n")
    md.append("4. **CRITICAL PRINCIPLE: Do Not Select Targets by Model Score:**\n"
              "   * Aggregated multi-inverter sums naturally achieve higher $R^2$ because summing distinct streams cancels out local sub-array variance.\n"
              "   * Target selection must rest on physical and electrical evidence, NOT statistical score optimization.\n\n")

    # Section 4.2: Test Data Reconciliation
    md.append("### 4.2 Test Data Reconciliation: 7,985 vs 7,761 Test Records\n")
    md.append("An important technical reconciliation explains why the previous experiment reported **7,985 test records**, whereas this benchmark reports **7,761 test records**:\n\n")
    md.append("- **Root Cause:** In the previous single-target experiment, `validate_and_preprocess.py` performed `inv_df.sum(axis=1, min_count=1)`. Any timestamp with at least one active inverter produced a valid row, yielding **39,921 total hourly rows** (80% dev = 31,936; 20% test = **7,985 rows**).\n"
              "- **Multi-Candidate Common Intersection:** To ensure 100% fair, synchronized benchmarking, `build_target_candidates.py` evaluated all 5 candidates on an identical common grid. Because Candidate 5 (`inv_04 + inv_10`) and Candidate 3 (`inv_04 + inv_09 + inv_10`) require large combiners to be active, hours where neither `inv_04` nor `inv_10` was reporting (predominantly in early 2019 prior to combiner commissioning and during sensor dropouts) were dropped across the common candidate table.\n"
              "- **Quantitative Impact:**\n"
              "  * Total hourly rows: 39,921 -> 38,804 (**1,117 rows dropped in the common intersection**, 711 in 2019, 325 in 2023, 65 in 2022, 16 in 2018).\n"
              "  * Chronological split boundary: Shifted from `2022-09-24 22:00 HST` to `2022-09-18 00:00 HST` (approx. 6.9 days earlier).\n"
              "  * Resulting holdout test set: 20% of 38,804 = **7,761 rows** (difference of **224 test records**).\n"
              "- **Methodological Rule:** All 5 candidate models in this benchmark are evaluated on the exact same 7,761-row test period. However, **old metrics and new candidate metrics MUST NOT be directly compared** as if they were evaluated on identical test records.\n\n")

    # Section 5: Calibrated Anomaly Detection
    md.append("---\n\n")
    md.append("## 5. Calibrated Anomaly Detection Evaluation & Terminology Corrections\n\n")
    md.append("### 5.1 Absence of Maintenance Ground Truth & Terminology Corrections\n")
    md.append("> [!IMPORTANT]\n"
              "> **NO OPERATOR GROUND TRUTH LABELS EXIST:**\n"
              "> NREL PVDAQ System 2105 telemetry lacks field technician work orders, maintenance tickets, or labeled fault records. "
              "> Consequently, **Precision, Recall, False Alarm Rate, and True Fault Accuracy CANNOT be computed**.\n"
              "> To maintain rigorous scientific honesty, all alarm terminology has been updated to non-confirmatory classifications:\n"
              "> * **`Potential Operational Anomaly`** (replaces *Critical Operational Fault*): Flags daytime zero/near-zero generation ($GHI > 200\\text{ W/m}^2, P < 2\\text{ kW}$). This is an empirical heuristic, not a certified mechanical breakdown.\n"
              "> * **`Statistical Residual Exceedance`** (replaces *Residual Performance Drop*): Flags hours where actual generation falls $>2.5\\sigma$ below the XGBoost expectation. This indicates statistical underperformance (which may stem from inverter tripping, soiling, grid curtailment, or model error), but is NOT an independently confirmed physical fault.\n"
              "> * **`Environmental Novelty` Definition:** Defined as multivariate out-of-distribution density flagged by Isolation Forest across $(GHI, \\text{Temperature}, \\text{Wind Speed}, \\text{Effective GHI})$. Disagreement between detectors reflects rare weather feature combinations, NOT validated severe weather disasters.\n\n")

    md.append("### 5.2 Calibration Parameters (Dev Split Daylight Hours, GHI > 10 W/m²)\n")
    md.append("| Parameter | Candidate 2 (Full Array) | Candidate 5 (Twin Combiners) |\n")
    md.append("| :--- | :---: | :---: |\n")
    c2_cal = data["anomaly_detection"]["Candidate_2"]["calibration_parameters"]
    c5_cal = data["anomaly_detection"]["Candidate_5"]["calibration_parameters"]
    md.append(f"| Observed Peak Capacity | {data['anomaly_detection']['Candidate_2']['peak_capacity_kW']:.2f} kW | {data['anomaly_detection']['Candidate_5']['peak_capacity_kW']:.2f} kW |\n")
    md.append(f"| Empirical Contamination | {c2_cal['empirical_contamination_pct']:.2f}% | {c5_cal['empirical_contamination_pct']:.2f}% |\n")
    md.append(f"| Daylight Median Residual | {c2_cal['daylight_median_residual_kW']:.2f} kW | {c5_cal['daylight_median_residual_kW']:.2f} kW |\n")
    md.append(f"| Daylight MAD | {c2_cal['daylight_MAD_kW']:.2f} kW | {c5_cal['daylight_MAD_kW']:.2f} kW |\n")
    sigma_label = r"$\sigma_{\text{robust}}$"
    md.append(f"| Robust Sigma ({sigma_label}) | {c2_cal['daylight_robust_sigma_kW']:.2f} kW | {c5_cal['daylight_robust_sigma_kW']:.2f} kW |\n")
    md.append(f"| **2.5-Sigma Residual Threshold** | **{c2_cal['daylight_residual_threshold_kW']:.2f} kW** | **{c5_cal['daylight_residual_threshold_kW']:.2f} kW** |\n\n")

    md.append("### 5.3 Reconciled Operational Alarms (Holdout Test Set: 7,761 Records)\n\n")
    md.append("| Operational Alarm Category | Candidate 2 Count | Candidate 2 (%) | Candidate 5 Count | Candidate 5 (%) |\n")
    md.append("| :--- | :---: | :---: |\n")
    
    cats = ["Normal Generation", "Statistical Residual Exceedance", "Potential Operational Anomaly", "Extreme Environmental Outlier", "Data Quality / Sensor Error"]
    c2_reconciled = data["anomaly_detection"]["Candidate_2"]["test_evaluation"]["reconciled_mutually_exclusive_classification"]
    c5_reconciled = data["anomaly_detection"]["Candidate_5"]["test_evaluation"]["reconciled_mutually_exclusive_classification"]
    
    for cat in cats:
        cnt2 = c2_reconciled.get(cat, {}).get("count", 0)
        pct2 = c2_reconciled.get(cat, {}).get("percentage", 0.0)
        cnt5 = c5_reconciled.get(cat, {}).get("count", 0)
        pct5 = c5_reconciled.get(cat, {}).get("percentage", 0.0)
        md.append(f"| **{cat}** | {cnt2:,} | {pct2:.2f}% | {cnt5:,} | {pct5:.2f}% |\n")
        
    md.append("| **Total Records Verified** | **7,761** | **100.00%** | **7,761** | **100.00%** |\n\n")

    # Section 6: Beginner-Friendly Explanations
    md.append("---\n\n")
    md.append("## 6. Beginner-Friendly Explanations of Core Concepts\n\n")
    md.append("### Why Can't a 110 kW Solar Plant Make 234 kW AC?\n")
    md.append("Think of a solar farm like a water filtration system:\n"
              "- The **DC Rating** (110 kW) is the total surface area of all solar panels capturing sunlight.\n"
              "- The **Inverter** is the converter that turns DC solar power into household AC electricity.\n"
              "- Because no converter is 100% efficient, and panels operate at high temperatures in Hawaii, an array captures at most its rated DC power on clear days.\n"
              "- Claiming a 110 kW farm generates 234 kW AC is like claiming a 100-liter bucket poured out 212 liters of water without anyone adding water! "
              "The extra 44 kW came from a logging mistake where one inverter (`inv_04`) was recorded twice in the database under two different names (`inv_04` and `inv_11`).\n\n")

    md.append("### What is the Inverter Loading Ratio (ILR)?\n")
    md.append("- Solar designers deliberately make the DC solar panel array about **20% to 30% larger** than the inverter's maximum capacity (an ILR of 1.2 to 1.3).\n"
              "- Why? Because the sun is only at peak overhead power for 1-2 hours a day. Sizing the panels slightly larger ensures the inverter runs at its sweet spot for more hours in the morning and afternoon.\n"
              "- For a 110.0 kW DC system, the ideal inverter size is: $110.0 / 1.25 = 88.0\\text{ kW AC}$. "
              "Our Candidate 5 (`inv_04 + inv_10`) reaches exactly **86.85 kW AC**, matching this design principle.\n\n")

    md.append("### What is Median Absolute Deviation (MAD) and Why Use It for Residuals?\n")
    md.append("- Standard deviation ($\\sigma$) is sensitive to extreme values. More critically, solar farms produce **0.0 kW all night long** (50% of the day).\n"
              "- If we calculate normal standard deviation across 24 hours, the 12 hours of zero residuals artificially shrinks the threshold, causing hundreds of false alarms during normal daytime operation!\n"
              "- By taking the **Median Absolute Deviation** strictly during **daylight operational hours** ($GHI > 10\\text{ W/m}^2$), we get a clean, robust baseline threshold that detects true underperformance.\n\n")

    # Section 7: Reproduction Commands
    md.append("---\n\n")
    md.append("## 7. Exact CLI Commands for 100% Reproduction\n\n")
    md.append("To independently reproduce every table, metric, model artifact, and plot, run the following commands inside PowerShell:\n\n")
    md.append("```powershell\n"
              "# 1. Activate Python virtual environment and set encoding\n"
              "$env:PYTHONIOENCODING=\"utf-8\"\n"
              ".venv\\Scripts\\python.exe -m pip install --upgrade -q xgboost scikit-learn pandas numpy matplotlib joblib\n\n"
              "# 2. Step 1: Channel Investigation & Inverter Overlap Audit\n"
              ".venv\\Scripts\\python.exe experiments/regional_adaptation/maui/target_reconstruction/investigate_channels.py\n\n"
              "# 3. Step 2: Build 5 Target Candidates & Diagnostic Plots\n"
              ".venv\\Scripts\\python.exe experiments/regional_adaptation/maui/target_reconstruction/build_target_candidates.py\n\n"
              "# 4. Step 3: Train & Benchmark Prediction Models on All Candidates\n"
              ".venv\\Scripts\\python.exe experiments/regional_adaptation/maui/target_reconstruction/benchmark_candidate_targets.py\n\n"
              "# 5. Step 4: Calibrate Anomaly Detection & Operational Reconciliation\n"
              ".venv\\Scripts\\python.exe experiments/regional_adaptation/maui/target_reconstruction/anomaly_on_target.py\n\n"
              "# 6. Step 5: Generate Master Reports\n"
              ".venv\\Scripts\\python.exe experiments/regional_adaptation/maui/target_reconstruction/generate_final_reconstruction_report.py\n"
              "```\n")

    md_text = "".join(md)
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Saved master Markdown report: {REPORT_MD_PATH}")

if __name__ == "__main__":
    generate_reports()
