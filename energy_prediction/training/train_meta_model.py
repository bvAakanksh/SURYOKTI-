# ============================================================
# SURYOKTI — UNIFIED 3-MODEL HYBRID STACK (TESTING)
# BASE MODELS : RANDOM FOREST + XGBOOST
# META MODEL  : NEURAL NETWORK (MLPRegressor + StandardScaler)
# ============================================================

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, cross_validate
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# 1. Load & Clean Dataset
data_path = os.path.join("Solar Weather Processed Data", "Solar_Weather_Data_Processed.csv")
if not os.path.exists(data_path):
    data_path = os.path.join("Data Preprocessing", "Solar_Weather_Data_Cleaned.csv")

df = pd.read_csv(data_path)
df.columns = df.columns.str.strip().str.replace(" ", "_")

# 2. Solar Physics & Cyclical Feature Engineering
def add_solar_features(data):
    d = data.copy()
    d["Hour_sin"] = np.sin(2 * np.pi * d["Hour"] / 24.0)
    d["Hour_cos"] = np.cos(2 * np.pi * d["Hour"] / 24.0)
    d["Month_sin"] = np.sin(2 * np.pi * d["Month"] / 12.0)
    d["Month_cos"] = np.cos(2 * np.pi * d["Month"] / 12.0)
    d["Wind_Dir_sin"] = np.sin(np.radians(d["Wind_Direction"]))
    d["Wind_Dir_cos"] = np.cos(np.radians(d["Wind_Direction"]))
    d["Effective_GHI"] = d["GHI"] * np.clip(np.cos(np.radians(d["SZ_Angle"])), 0, None)
    d["Thermal_Stress"] = d["GHI"] * d["Temperature"] / (d["Wind_Speed"] + 1.0)
    return d

df = add_solar_features(df)

features = [
    "GHI", "Temperature", "Relative_Humidity", "Wind_Speed",
    "Wind_Direction", "Hour", "Day", "Month", "SZ_Angle", "SA_Angle",
    "Hour_sin", "Hour_cos", "Month_sin", "Month_cos",
    "Wind_Dir_sin", "Wind_Dir_cos", "Effective_GHI", "Thermal_Stress"
]
target = "AC_Output"

X = df[features]
y = df[target]

# 3. Train / Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Define 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 5. Define Level-2 Neural Network Meta-Model Pipeline
nn_meta_model = Pipeline([
    ("scaler", StandardScaler()),
    ("nn", MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.05,                # L2 regularization to prevent overfitting
        learning_rate_init=0.003,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        max_iter=1000,
        random_state=42
    ))
])

# 6. Build Combined Hybrid: (Random Forest + XGBoost) -> Neural Network Meta-Model
model_rf_xgb_nn = StackingRegressor(
    estimators=[
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=8,           # Constrained depth prevents RF memorization
                min_samples_split=6,
                min_samples_leaf=3,
                max_features=0.8,
                random_state=42,
                n_jobs=-1
            )
        ),
        (
            "xgboost",
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
        )
    ],
    final_estimator=nn_meta_model,     # Neural Network combines RF + XGBoost + Weather context
    passthrough=True,
    cv=kf,
    n_jobs=-1
)

# 7. Fit and Evaluate on Holdout Test Set
print("Training Combined Hybrid (RF + XGBoost -> Neural Network)...")
model_rf_xgb_nn.fit(X_train, y_train)

y_pred_test = model_rf_xgb_nn.predict(X_test)
y_pred_test = np.where(X_test["GHI"] <= 0, 0.0, np.clip(y_pred_test, 0.0, None))

y_pred_train = model_rf_xgb_nn.predict(X_train)
y_pred_train = np.where(X_train["GHI"] <= 0, 0.0, np.clip(y_pred_train, 0.0, None))

holdout_mae = mean_absolute_error(y_test, y_pred_test)
holdout_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
holdout_r2 = r2_score(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
r2_gap = train_r2 - holdout_r2

# 8. 5-Fold Cross-Validation (on Training Partition)
print("Running 5-Fold Cross-Validation...")
cv_scores = cross_validate(
    model_rf_xgb_nn, X_train, y_train, cv=kf,
    scoring={"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
    n_jobs=1
)

# 9. Time Series Cross-Validation (Chronological Order)
print("Running Time Series Cross-Validation...")
df_sorted = df.sort_values(by=["Month", "Day", "Hour"]).reset_index(drop=True)
tscv = TimeSeriesSplit(n_splits=5)
ts_scores = cross_validate(
    model_rf_xgb_nn, df_sorted[features], df_sorted[target], cv=tscv,
    scoring={"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
    n_jobs=1
)

print("\n============================================================")
print("   RESULTS: RF + XGBOOST + NEURAL NETWORK META-MODEL")
print("============================================================")
print(f"Holdout MAE        : {holdout_mae:.4f} kW")
print(f"Holdout RMSE       : {holdout_rmse:.4f} kW")
print(f"Holdout R2 (Acc.)  : {holdout_r2:.4f} ({holdout_r2*100:.2f}%)")
print(f"Train R2           : {train_r2:.4f} ({train_r2*100:.2f}%)")
print(f"Overfitting R2 Gap : {r2_gap:.4f} ({r2_gap*100:.2f}%)")
print("------------------------------------------------------------")
print(f"5-Fold CV MAE      : {-cv_scores['test_MAE'].mean():.4f} kW (std: {cv_scores['test_MAE'].std():.4f})")
print(f"5-Fold CV RMSE     : {-cv_scores['test_RMSE'].mean():.4f} kW (std: {cv_scores['test_RMSE'].std():.4f})")
print(f"5-Fold CV R2       : {cv_scores['test_R2'].mean():.4f} (+/-{cv_scores['test_R2'].std():.4f})")
print("------------------------------------------------------------")
print(f"Time Series CV MAE : {-ts_scores['test_MAE'].mean():.4f} kW (std: {ts_scores['test_MAE'].std():.4f})")
print(f"Time Series CV RMSE: {-ts_scores['test_RMSE'].mean():.4f} kW (std: {ts_scores['test_RMSE'].std():.4f})")
print(f"Time Series CV R2  : {ts_scores['test_R2'].mean():.4f} (+/-{ts_scores['test_R2'].std():.4f})")
print("============================================================")

