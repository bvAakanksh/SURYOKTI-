# SURYOKTI: Research Experiments & Regional Adaptation

This directory isolates specialized research experiments, regional adaptation studies, and plant-level target reconstruction investigations to ensure the core SURYOKTI codebase remains clean and reproducible.

---

## 1. Directory Structure

```text
experiments/
├── README.md                                    # Experiments guide and status summary (this file)
│
├── regional_adaptation/                         # Regional adaptation to NREL PVDAQ System 2105 (Maui, HI)
│   ├── data/                                    # Maui 15-minute and hourly cleaned telemetry
│   ├── models/                                  # Regional models trained on Maui data
│   ├── plots/                                   # Parity and anomaly reconciliation figures
│   ├── reports/                                 # Initial regional adaptation report
│   ├── validate_and_preprocess_maui.py          # Data validation and UTC to HST conversion
│   ├── train_predict_maui.py                    # Regional model training & evaluation
│   └── generate_prediction_plots.py             # Prediction plot generator
│
├── target_reconstruction/                       # Plant AC-Output Target Reconstruction & Audit
│   ├── data/                                    # Hourly candidate datasets (Candidates 1 to 5)
│   ├── plots/                                   # Inverter capacity and diurnal candidate comparisons
│   ├── reports/                                 # Executive decision memos
│   │   └── maui_decision_memo.md                # Decision memo on evidence, certainty, and status
│   ├── investigate_channels.py                  # Channel-by-channel electrical and statistical audit
│   ├── build_target_candidates.py               # Generates candidate datasets from raw inverter files
│   ├── benchmark_candidate_targets.py           # Trains & evaluates models across all 5 candidate targets
│   ├── anomaly_on_target.py                     # Daylight anomaly calibration on Candidates 2 & 5
│   ├── generate_final_reconstruction_report.py  # Generates comprehensive Markdown and JSON reports
│   ├── target_reconstruction_report.md          # Full target reconstruction technical report
│   └── target_reconstruction_report.json        # Machine-readable candidate statistics
│
└── archived/                                    # Deprecated and historical exploratory runs
```

---

## 2. The Maui System 2105 Experiment Summary

* **Location:** Maui, Hawaii (Latitude: 20.89° N, Longitude: -156.43° W).
* **Registry Nameplate:** 110 kW DC.
* **Key Challenge Discovered in Technical Audit:**
  - Raw logger contains 11 inverter telemetry feeds (`inv_01` to `inv_11`).
  - A naive summation of all 11 channels peaks at 234.09 kW, exceeding the 110 kW DC nameplate by 2.13× due to internal sub-combiner overlap.
  - Channels `inv_04` (43.4 kW peak) and `inv_10` (43.4 kW peak) exhibit sub-combiner characteristics aggregating the individual string inverters.
* **Target Candidate Benchmarks:**
  - **Candidate 1 (All 11 Sum):** 234.09 kW peak — Rejected (Double counting).
  - **Candidate 2 (6 Disjoint Strings):** 62.49 kW peak — Explains individual strings, but under-represents plant capacity.
  - **Candidate 5 (Twin Combiners `inv_04 + inv_10`):** 86.85 kW peak — Conforms to standard 1.25 DC/AC loading ratio for a 110 kW DC plant.
* **Scientific Verdict:** Plant-level reconstruction is maintained as **UNVERIFIED** pending site electrical single-line drawings, preserving complete scientific transparency before cross-region claims.

