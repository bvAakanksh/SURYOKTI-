# SURYOKTI: Energy Prediction — Full Code Repository

This directory contains an unpartitioned, flat collection of all Python (`.py`) scripts across every subfolder of the **Energy Prediction Module** (`data/`, `training/`, `evaluation/`, `testing/`, `inference/`, and `plots/`).

---

## 1. Inventory of Scripts & Original Subdirectories

| Copied Script File | Source Subdirectory | Execution Role & Description |
|---|---|---|
| [`data_cleaning.py`](file:///c:/VITB/suryokti%20draft%201/energy_prediction/full%20code/data_cleaning.py) | `energy_prediction/data/` | Inspects data types, removes missing values, normalizes column names, and checks correlation against `AC_Output`. |
| [`preprocessing.py`](file:///c:/VITB/suryokti%20draft%201/energy_prediction/full%20code/preprocessing.py) | `energy_prediction/data/` | Merges solar generation and weather datasets, synchronizes timestamps, and creates baseline feature sets. |
| [`train_linear_regression.py`](file:///c:/VITB/suryokti%20draft%201/energy_prediction/full%20code/train_linear_regression.py) | `energy_prediction/training/` | Trains baseline Linear Regression model, computing standard train/test splits and baseline error metrics. |
| [`train_random_forest.py`](file:///c:/VITB/suryokti%20draft%201/energy_prediction/full%20code/train_random_forest.py) | `energy_prediction/training/` | Trains baseline Random Forest Regressor and evaluates holdout accuracy and feature importances. |
| [`train_meta_model.py`](file:///c:/VITB/suryokti%20draft%201/energy_prediction/full%20code/train_meta_model.py) | `energy_prediction/training/` | Trains multi-layer stacking meta-models exploring RF + XGBoost combinations. |
| [`fine_tune_pipeline.py`](file:///c:/VITB/suryokti%20draft%201/energy_prediction/full%20code/fine_tune_pipeline.py) | `energy_prediction/training/` | Full 8-stage hyperparameter tuning and cross-validation search for Random Forest, XGBoost, and Stacking. |
| [`fine_tune_pipeline_corrected.py`](file:///c:/VITB/suryokti%20draft%201/energy_prediction/full%20code/fine_tune_pipeline_corrected.py) | `energy_prediction/training/` | Production fine-tuning pipeline with strict TimeSeriesSplit temporal splits and physical constraints. |
| [`compare_metrics.py`](file:///c:/VITB/suryokti%20draft%201/energy_prediction/full%20code/compare_metrics.py) | `energy_prediction/evaluation/` | Multi-model evaluation comparing Linear Regression, Random Forest, XGBoost, and Stacking across KFold/TimeSeries CV. |
| [`test_suryokti_model.py`](file:///c:/VITB/suryokti%20draft%201/energy_prediction/full%20code/test_suryokti_model.py) | `energy_prediction/testing/` | Validates serialized production hybrid model package (`suryokti_hybrid.suryokti`) on holdout datasets. |
| [`predict_energy.py`](file:///c:/VITB/suryokti%20draft%201/energy_prediction/full%20code/predict_energy.py) | `energy_prediction/inference/` | Standalone CLI and programmatic inference engine predicting AC output with night gating and clipping. |
| [`generate_plots.py`](file:///c:/VITB/suryokti%20draft%201/energy_prediction/full%20code/generate_plots.py) | `energy_prediction/plots/` | Generates baseline exploratory distributions and scatter relationships. |
| [`generate_hybrid_plots.py`](file:///c:/VITB/suryokti%20draft%201/energy_prediction/full%20code/generate_hybrid_plots.py) | `energy_prediction/plots/` | Generates the comprehensive 8-figure publication plot suite for the Hybrid Stacking ensemble. |

---

## 2. Standard Execution Sequence

To execute the complete end-to-end energy prediction workflow from this folder:

```powershell
# Step 1: Preprocess and clean data
python "energy_prediction/full code/data_cleaning.py"

# Step 2: Run baseline model benchmarking
python "energy_prediction/full code/compare_metrics.py"

# Step 3: Run fine-tuned stacking pipeline
python "energy_prediction/full code/fine_tune_pipeline_corrected.py"

# Step 4: Verify production hybrid artifact
python "energy_prediction/full code/test_suryokti_model.py"

# Step 5: Run inference demonstration
python "energy_prediction/full code/predict_energy.py"
```

