# SURYOKTI: Energy Prediction Module

The **Energy Prediction Module** provides solar photovoltaic (PV) AC active power output forecasting using physics-informed feature engineering coupled with high-performance gradient boosting and regularized stacking ensembles.

---

## 1. Directory Structure

```text
energy_prediction/
├── README.md                                    # Module architecture and execution guide (this file)
├── requirements.txt                             # Module dependencies
│
├── data/                                        # Data preparation and cleaning
│   ├── preprocessing.py                         # Kaggle raw source merge reproducing demo dataset
│   └── data_cleaning.py                         # Missing value verification and column stripping
│
├── training/                                    # Model training scripts
│   ├── train_linear_regression.py               # Baseline Linear Regression & RidgeCV
│   ├── train_random_forest.py                   # Random Forest Regressor with 5-Fold CV
│   ├── fine_tune_pipeline.py                    # Standard stacking pipeline trainer
│   ├── fine_tune_pipeline_corrected.py          # Chronological split stacking pipeline with physical gating
│   └── train_meta_model.py                      # 3-Stack meta-learner exploration
│
├── evaluation/                                  # Cross-validation and model benchmarking
│   └── compare_metrics.py                       # Rigorous K-Fold vs TimeSeriesSplit comparison
│
├── testing/                                     # Exported package testing
│   └── test_suryokti_model.py                   # Tests suryokti_hybrid.suryokti package
│
├── inference/                                   # Production forecasting inference
│   └── predict_energy.py                        # Standalone AC output prediction with physical gates
│
├── models/                                      # Compatibility package metadata and hybrid bundle
│   ├── fine_tuned_metadata.json                 # Training provenance and metadata
│   └── suryokti_hybrid.suryokti                 # Standalone deployable hybrid package
│                                                   # Canonical .pkl artifacts are in ../models/
│
├── plots/                                       # Generated evaluation figures
│   ├── actual_vs_predicted/                     # Parity and time-series overlay plots
│   ├── residual_plots/                          # Residual error distributions
│   ├── model_comparison/                        # K-Fold R² bar charts and metrics
│   └── feature_importance/                      # XGBoost feature importance rankings
│
├── results/                                     # Exported prediction CSVs and performance summaries
│   ├── metrics/                                 # Tabular evaluation tables
│   └── predictions/                             # Test set prediction CSVs
│
├── notebooks/                                   # Jupyter exploration
│   └── merge_preprocessing_code.ipynb           # Interactive preprocessing notebook
│
├── full code/                                   # Consolidated repository of all module Python scripts
│   ├── README.md                                # Script index, origins, and execution order
│   ├── data_cleaning.py                         # From data/
│   ├── preprocessing.py                         # From data/
│   ├── compare_metrics.py                       # From evaluation/
│   ├── predict_energy.py                        # From inference/
│   ├── generate_plots.py                        # From plots/
│   ├── generate_hybrid_plots.py                 # From plots/
│   ├── test_suryokti_model.py                   # From testing/
│   ├── fine_tune_pipeline.py                    # From training/
│   ├── fine_tune_pipeline_corrected.py          # From training/
│   ├── train_linear_regression.py               # From training/
│   ├── train_random_forest.py                   # From training/
│   └── train_meta_model.py                      # From training/
│
└── scripts/                                     # Pipeline runner helpers
```

---

## 2. Feature Engineering & Solar Physics

To maximize predictive accuracy across varying atmospheric conditions, the module derives physical and cyclical transformations from solar position angles and meteorological inputs:

| Engineered Feature | Mathematical Formulation | Physical Significance |
|---|---|---|
| `Hour_sin`, `Hour_cos` | $\sin(2\pi \cdot h / 24)$, $\cos(2\pi \cdot h / 24)$ | Encodes daily solar diurnal cycle continuously across midnight |
| `Month_sin`, `Month_cos` | $\sin(2\pi \cdot m / 12)$, $\cos(2\pi \cdot m / 12)$ | Captures annual seasonal solar declination trajectory |
| `Wind_Dir_sin`, `Wind_Dir_cos` | $\sin(\theta_{\text{wind}})$, $\cos(\theta_{\text{wind}})$ | Eliminates angular discontinuity at 0° / 360° North |
| `Effective_GHI` | $\text{GHI} \cdot \max(0, \cos(\theta_{\text{zenith}}))$ | Direct horizontal solar flux projected onto panel aperture |
| `Thermal_Stress` | $\frac{\text{GHI} \cdot T_{\text{ambient}}}{v_{\text{wind}} + 1.0}$ | Proxy for PV cell temperature rise and convective cooling losses |

---

## 3. Physical Gating Constraints

Machine learning models can occasionally produce non-physical predictions (such as negative power or non-zero generation during night). The inference engine enforces strict solar physical boundary gates:
1. **Non-Negativity Constraint:** $\hat{P}_{\text{AC}} = \max(0, \hat{P}_{\text{raw}})$
2. **Nighttime Zero-Irradiance Gate:** $\hat{P}_{\text{AC}} = 0 \quad \forall \; \text{GHI} \le 0.1 \text{ W/m}^2$

---

## 4. Benchmark Performance Comparison

Results evaluated on the 2,777-sample benchmark dataset across 5-Fold K-Fold Cross Validation and 5-Fold TimeSeriesSplit (chronological):

| Model Architecture | Holdout MAE (kW) | Holdout RMSE (kW) | Holdout $R^2$ | K-Fold CV $R^2$ | TimeSeries CV $R^2$ | Generalization Gap |
|---|---:|---:|---:|---:|---:|---:|
| **Hybrid: XGB + NN Stacking** | **54.39** | **104.03** | **0.9186** | **0.9182** | **0.9049** | **0.0138** (Lowest Overfit) |
| **XGBoost Regressor** | 55.30 | 103.18 | 0.9199 | 0.9162 | 0.8929 | 0.0245 |
| **Hybrid: RF + XGB Stacking** | 54.14 | 103.19 | 0.9200 | 0.9159 | 0.8950 | 0.0487 |
| **Random Forest Regressor** | 53.98 | 105.74 | 0.9159 | 0.9104 | 0.8910 | 0.0711 |
| **Linear Regression** | 158.56 | 189.61 | 0.7297 | 0.7391 | 0.7065 | 0.0149 |

---

## 5. Execution Instructions

### A. Run Inference
```powershell
.venv\Scripts\python.exe energy_prediction/inference/predict_energy.py
```
Outputs model type, test sample predictions, and overall test set evaluation metrics ($R^2$, MAE, RMSE).

### B. Run Full Benchmark Comparison
```powershell
.venv\Scripts\python.exe energy_prediction/evaluation/compare_metrics.py
```
Executes cross-validation and produces the long-term stability table.

### C. Train Production Stacking Pipeline
```powershell
.venv\Scripts\python.exe energy_prediction/training/fine_tune_pipeline_corrected.py
```
Trains the stacking regressor using chronological splits. Canonical exported artifacts are stored in the repository-level `models/` directory; `models/energy_prediction/` is retained as a compatibility copy where present.

