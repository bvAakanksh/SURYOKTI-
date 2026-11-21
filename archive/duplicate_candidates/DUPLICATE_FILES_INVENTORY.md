# SURYOKTI: Duplicate Files & Canonical Mapping Inventory

This document tracks all identical files (verified by MD5 hash) across legacy, experimental, and reorganized directories. To maintain complete backwards compatibility and prevent data loss, legacy files remain preserved while the canonical active files are organized into their modular homes.

---

## 1. Summary of Duplicate Groups

* **Total Duplicate Groups:** 89 groups across models, datasets, reports, and plots.
* **Preservation Policy:** No legacy files were deleted or overwritten. All active code points dynamically to canonical paths with legacy fallbacks.

---

## 2. Key Canonical File Mappings

### A. Core Datasets

| Dataset Description | MD5 Hash | Canonical Modular Location | Preserved Legacy Location(s) |
|---|---|---|---|
| **Maui 15-Minute Raw Telemetry** | `f3c8040796b77686152142b46c682649` | `datasets/energy_prediction/raw/maui_system_2105_clean_15min.csv` | `All_Datasets/`, `experiments/regional_adaptation/data/` |
| **Maui Hourly Processed Telemetry** | `965a46f6448811a58d99953228e56ce5` | `datasets/energy_prediction/processed/maui_system_2105_clean_hourly.csv` | `All_Datasets/`, `experiments/regional_adaptation/data/` |
| **All Inverter Candidates Hourly** | `5e2d5ff04b3653ccac1a98224d87e896` | `datasets/energy_prediction/processed/all_candidates_hourly.csv` | `All_Datasets/`, `experiments/target_reconstruction/data/` |
| **Candidate 1 Hourly (All 11 Sum)** | `82fa9edbb6ec0680496d682e0420fc78` | `datasets/energy_prediction/processed/candidate_1_hourly.csv` | `All_Datasets/`, `experiments/target_reconstruction/data/` |
| **Candidate 2 Hourly (6 Strings)** | `3a7ed3e38ebac080ce434ea004bc5160` | `datasets/energy_prediction/processed/candidate_2_hourly.csv` | `All_Datasets/`, `experiments/target_reconstruction/data/` |
| **Candidate 3 Hourly (5 Combiners)**| `e0550fa6f4935ac2763f51b8a357cc17` | `datasets/energy_prediction/processed/candidate_3_hourly.csv` | `All_Datasets/`, `experiments/target_reconstruction/data/` |
| **Candidate 4 Hourly (inv_04)** | `e864954d747dd70ec009d03c00ada70d` | `datasets/energy_prediction/processed/candidate_4_hourly.csv` | `All_Datasets/`, `experiments/target_reconstruction/data/` |
| **Candidate 5 Hourly (inv_04+10)** | `a2c2e5861e9f0bb1fa517e6919096d8a` | `datasets/energy_prediction/processed/candidate_5_hourly.csv` | `All_Datasets/`, `experiments/target_reconstruction/data/` |
| **Kaggle Plant 1 Generation Data** | `97fd3db0b78b1218acfb6fac42a3ea17` | `datasets/external_sources/kaggle/Plant_1_Generation_Data.csv` | `All_Datasets/`, `Original Kaggle Source Files/` |
| **Kaggle SolarPrediction Data** | `cf2caa35e3aa34a8e9a1e2e359d864ee` | `datasets/external_sources/kaggle/SolarPrediction.csv` | `All_Datasets/`, `Original Kaggle Source Files/` |
| **Data Dictionary** | `7b460f5a774a621180921e4220bcb2c3` | `datasets/data_dictionary.csv` | `All_Datasets/`, `Data Dictionary/` |

---

### B. Machine Learning Models & Packages

| Model Description | MD5 Hash | Canonical Modular Location | Preserved Legacy Location(s) |
|---|---|---|---|
| **Production Stacking Model** | `01c80ee13cd96f2b1520003e3477509d` | `models/energy_prediction/fine_tuned_final_stack.pkl` | `models/`, `energy_prediction/models/` |
| **Hybrid Model Package** | `77d33d937000bf1b4dd1b3539828453d` | `models/energy_prediction/suryokti_hybrid.suryokti` | `suryokti_hybrid.suryokti` (root), `energy_prediction/models/` |
| **Best Model XGBoost** | `7d100c82f93c66f7f3a8bdfa73fa132f` | `models/energy_prediction/best_model.pkl` | `models/`, `energy_prediction/models/` |
| **StandardScaler Object** | `be15bd29024f34e6c541280b0528e9d1` | `models/energy_prediction/scaler.pkl` | `models/`, `energy_prediction/models/` |
| **Feature Schema (Ordered)** | `07cfc035400809dc5c5ff9c1f00651ab` | `models/energy_prediction/feature_cols.pkl` | `models/`, `energy_prediction/models/` |
| **Maui Isolation Forest** | `c8a7eedd121e4c11d83f239d5ac9ab6e` | `models/anomaly_detection/maui_isolation_forest.pkl` | `experiments/regional_adaptation/maui/models/` |
| **Candidate 5 RF Model** | `b5242873b62c23da161a23bad38ef147` | `experiments/target_reconstruction/models/candidate_5_rf_model.joblib` | `experiments/regional_adaptation/maui/target_reconstruction/models/` |

---

### C. Technical Reports & Audit Documentation

| Report Title | MD5 Hash | Canonical Modular Location | Preserved Legacy Location(s) |
|---|---|---|---|
| **Maui Decision Memo** | `5101159b7b9aa37a110b7858c75d59fc` | `documentation/project_reports/maui_decision_memo.md` | `experiments/regional_adaptation/maui/target_reconstruction/reports/` |
| **Maui Technical Audit Report** | `b019ee6cece7fc9b084e3bb07c126ec0` | `documentation/technical_audits/maui_technical_audit.md` | `reports/` |
| **Maui Technical Audit JSON** | `867f87a6ab53fca0e76144a5dbaf3e21` | `documentation/technical_audits/maui_technical_audit.json` | `reports/` |
| **License and Provenance Note**| `e34444a30538a1428cd2b59228efe71d` | `documentation/license/license_and_provenance_note.txt` | `License and Provenance Note/` |

