# SURYOKTI: Models Inventory and Artifact Store

This directory stores all serialized machine learning models, regularized stacking ensembles, preprocessors, and feature schemas.

---

## 1. Directory Structure

```text
models/
├── README.md                                    # Model inventory, metadata, and provenance (this file)
│
├── best_model.pkl                               # Canonical XGBoost forecasting model
├── feature_cols.pkl                             # Exact ordered feature schema
├── scaler.pkl                                   # StandardScaler used by linear-model paths
├── fine_tuned_final_stack.pkl                   # StackingRegressor experiment artifact
├── fine_tuned_features.pkl                      # Stacking feature schema
├── fine_tuned_metadata.json                     # Training provenance and metrics
├── energy_prediction/                           # Compatibility copy of the forecasting artifacts
│   ├── best_model.pkl                           # Compatibility copy; identical to the root artifact
│   ├── feature_cols.pkl                         # Exact 12-feature ordering required by best_model
│   ├── scaler.pkl                               # StandardScaler instance fitted on training split
│   ├── fine_tuned_final_stack.pkl               # Production StackingRegressor (XGB + NN meta-learner)
│   ├── fine_tuned_features.pkl                  # Exact 10-feature ordering for stacking model
│   ├── fine_tuned_metadata.json                 # JSON metadata containing parameters & validation metrics
│   └── suryokti_hybrid.suryokti                 # Self-contained zip package containing manifest & XGBoost JSON
│
├── anomaly_detection/                           # Anomaly detector artifacts
│   └── maui_isolation_forest.pkl                # Trained Isolation Forest model calibrated for Maui System 2105
│
└── archived_experiments/                        # Regional adaptation & historical prototypes
    ├── maui_best_model.pkl                      # Archived best XGBoost regressor trained on Maui hourly telemetry
    └── maui_features.pkl                        # Archived feature list for Maui regional model
```

---

## 2. Model Specifications & Lineage

### A. `suryokti_hybrid.suryokti`
* **Format:** Custom zip bundle containing `manifest.json` and `correction.json`.
* **Model Type:** XGBoost Regressor (240 estimators, max depth 3).
* **Architecture:** Hybrid physical-statistical model mapping normalized ratios (`physics_ratio`, `poa_ratio`, NOCT cell temperature) to actual AC power output.
* **Testing:** Validated by [`energy_prediction/testing/test_suryokti_model.py`](../energy_prediction/testing/test_suryokti_model.py).

### B. `best_model.pkl`
* **Format:** Scikit-learn / XGBoost serialized via Joblib.
* **Trained On:** 80% train split of `Solar_Weather_Data_Processed.csv`.
* **Holdout Score:** $R^2 \approx 0.9496$, $\text{MAE} \approx 41.83 \text{ kW}$.
* **Inference:** Loaded automatically by [`predict_energy.py`](../energy_prediction/inference/predict_energy.py). The canonical copy is at `models/best_model.pkl`; `models/energy_prediction/` retains a compatibility duplicate.

### C. `fine_tuned_final_stack.pkl`
* **Format:** Scikit-learn `StackingRegressor`.
* **Base Estimators:** XGBoost Regressor (`n_estimators=100`) + MLP Regressor (`hidden_layer_sizes=(64, 32)`).
* **Final Meta-Estimator:** RidgeCV with cross-validated $L_2$ regularization.
* **Validation:** Achieved lowest generalization gap (0.0138) and highest TimeSeries CV $R^2$ (0.9049).

---

## 3. Loading Artifacts in Code

```python
import joblib
from pathlib import Path

models_dir = Path("models")
model = joblib.load(models_dir / "best_model.pkl")
scaler = joblib.load(models_dir / "scaler.pkl")
features = joblib.load(models_dir / "feature_cols.pkl")

print(f"Loaded {type(model).__name__} expecting {len(features)} features.")
```

