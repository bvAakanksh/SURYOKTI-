# SURYOKTI Original Model Inspection & Architectural Blueprint

This document records the architectural inspection of the original SURYOKTI Energy Prediction model implementation, clarifying the origin and meaning of the reported **92.30%** performance figure, its hyperparameters, feature set, and adaptation strategy for the Romania two-site PV dataset.

---

## 1. Inspection of Original Architecture & 92.30% Score

### A. Source Implementation
- **Canonical Script:** [`energy_prediction/training/train_meta_model.py`](file:///c:/VITB/suryokti%20draft%201/energy_prediction/training/train_meta_model.py)
- **Benchmarked Name:** `"Unified 3-Stack (RF+XGB->NN)"` in [`presentations/scripts/generate_presentation.py`](file:///c:/VITB/suryokti%20draft%201/presentations/scripts/generate_presentation.py)
- **Reported Score:** $R^2 = 0.9230$ ($92.30\%$) on 20% holdout split.
- **Clarification:** The reported 92.30% is specifically the **Coefficient of Determination ($R^2$) score**, not a classification accuracy percentage. It was accompanied by Holdout $\text{MAE} = 52.26\text{ kW}$ and $\text{RMSE} = 101.18\text{ kW}$.

### B. Original Hyperparameters
1. **Random Forest Base Regressor:**
   ```python
   RandomForestRegressor(
       n_estimators=200,
       max_depth=8,              # Constrained depth prevents tree memorization
       min_samples_split=6,
       min_samples_leaf=3,
       max_features=0.8,
       random_state=42,
       n_jobs=-1
   )
   ```
2. **XGBoost Base Regressor:**
   ```python
   XGBRegressor(
       n_estimators=200,
       learning_rate=0.03,
       max_depth=4,
       min_child_weight=3,
       subsample=0.8,
       colsample_bytree=0.8,
       reg_alpha=0.1,
       reg_lambda=1.5,
       random_state=42,
       n_jobs=-1
   )
   ```
3. **Neural Network Meta-Learner Pipeline:**
   ```python
   Pipeline([
       ("scaler", StandardScaler()),
       ("nn", MLPRegressor(
           hidden_layer_sizes=(64, 32),
           activation="relu",
           solver="adam",
           alpha=0.05,           # L2 weight regularization
           learning_rate_init=0.003,
           early_stopping=True,
           validation_fraction=0.15,
           n_iter_no_change=20,
           max_iter=1000,
           random_state=42
       ))
   ])
   ```
4. **Stacking Ensemble:**
   ```python
   StackingRegressor(
       estimators=[("random_forest", rf), ("xgboost", xgb)],
       final_estimator=nn_meta_model,
       passthrough=True,
       cv=kf,
       n_jobs=-1
   )
   ```
5. **Physical Post-Processing:**
   ```python
   y_pred = np.where(GHI <= 0, 0.0, np.clip(y_pred, 0.0, None))
   ```

---

## 2. Adaptation Strategy for Romania Two-Site Dataset

1. **Target Normalization:**
   - The Romania dataset provides 50 kW AC installations.
   - Primary target: `capacity_factor` $\in [0, 1]$, defined as $\text{active\_power\_kw} / 50.0$.
   - Supported conversion: $\text{predicted\_active\_power\_kw} = \text{predicted\_capacity\_factor} \times 50.0$.
2. **Strict Leakage Prevention:**
   - Telemetry columns like `dc_input_power_kw`, `daily_energy_kwh`, `cumulative_energy_kwh`, `efficiency_pct`, `is_generating`, and `inverter_status` are strictly omitted from features.
   - Only meteorological and solar-position features are used.
3. **Temporal Integrity & Stacking CV:**
   - Replacing random KFold with chronological out-of-fold splitting (e.g. `TimeSeriesSplit` / expanding window) so that the meta-learner is trained on out-of-fold predictions without future lookahead bias.
4. **Daytime vs All-Hours Dual Reporting:**
   - Both daytime ($\text{solar\_elevation\_deg} > 5^\circ$) and full 24-hour benchmarks are evaluated and documented separately.

