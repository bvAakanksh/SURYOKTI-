# ============================================================
# SURYOKTI
# RANDOM FOREST + XGBOOST + META MODEL
# WITH K-FOLD CROSS VALIDATION
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV  # ADDED: Regularized meta-learner
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# ============================================================
# 2. LOAD DATASET
# ============================================================
from pathlib import Path

def _find_dataset():
    candidates = [
        Path(__file__).resolve().parents[2] / "datasets" / "energy_prediction" / "demo_datasets" / "Solar_Weather_Data_Processed.csv",
        Path("datasets/energy_prediction/demo_datasets/Solar_Weather_Data_Processed.csv"),
        Path(__file__).resolve().parents[2] / "Solar Weather Processed Data" / "Solar_Weather_Data_Processed.csv",
        Path("Solar Weather Processed Data/Solar_Weather_Data_Processed.csv"),
        Path(r"C:\VITB\Curated Solar Weather and PV AC Power Output Datas\Solar Weather Processed Data\Solar_Weather_Data_Processed.csv"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return r"Solar Weather Processed Data\Solar_Weather_Data_Processed.csv"

df = pd.read_csv(_find_dataset())

# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
)

# ============================================================
# ADDED: SOLAR PHYSICS & CYCLICAL FEATURE ENGINEERING
# ============================================================
def add_solar_features(data):
    d = data.copy()
    # Cyclical time and angle encodings
    d["Hour_sin"] = np.sin(2 * np.pi * d["Hour"] / 24.0)
    d["Hour_cos"] = np.cos(2 * np.pi * d["Hour"] / 24.0)
    d["Month_sin"] = np.sin(2 * np.pi * d["Month"] / 12.0)
    d["Month_cos"] = np.cos(2 * np.pi * d["Month"] / 12.0)
    d["Wind_Dir_sin"] = np.sin(np.radians(d["Wind_Direction"]))
    d["Wind_Dir_cos"] = np.cos(np.radians(d["Wind_Direction"]))
    # Effective vertical irradiance & thermal cooling proxy
    d["Effective_GHI"] = d["GHI"] * np.clip(np.cos(np.radians(d["SZ_Angle"])), 0, None)
    d["Thermal_Stress"] = d["GHI"] * d["Temperature"] / (d["Wind_Speed"] + 1.0)
    return d

df = add_solar_features(df)

# ============================================================
# 4. FEATURES AND TARGET
# ============================================================
features = [
    "GHI",
    "Temperature",
    "Relative_Humidity",
    "Wind_Speed",
    "Wind_Direction",
    "Hour",
    "Day",
    "Month",
    "SZ_Angle",
    "SA_Angle",
    # ADDED ENGINEERED FEATURES:
    "Hour_sin",
    "Hour_cos",
    "Month_sin",
    "Month_cos",
    "Wind_Dir_sin",
    "Wind_Dir_cos",
    "Effective_GHI",
    "Thermal_Stress"
]

target = "AC_Output"

X = df[features]
y = df[target]

print("\nDATA INFORMATION")
print("==============================")
print("Feature Shape:", X.shape)
print("Target Shape :", y.shape)

# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAIN / TEST SPLIT")
print("==============================")
print("Training Data:", X_train.shape)
print("Testing Data :", X_test.shape)

# ============================================================
# 6. LINEAR REGRESSION
# ============================================================
model_lr = LinearRegression()
model_lr.fit(
    X_train,
    y_train
)

# Test prediction
y_pred_lr = model_lr.predict(
    X_test
)

# Training prediction
train_pred_lr = model_lr.predict(
    X_train
)

# Metrics
mae_lr = mean_absolute_error(
    y_test,
    y_pred_lr
)
rmse_lr = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_lr
    )
)
r2_lr = r2_score(
    y_test,
    y_pred_lr
)

train_r2_lr = r2_score(
    y_train,
    train_pred_lr
)
test_r2_lr = r2_score(
    y_test,
    y_pred_lr
)

print("\nLINEAR REGRESSION")
print("==============================")
print("MAE :", mae_lr)
print("RMSE:", rmse_lr)
print("R²  :", r2_lr)

# ============================================================
# 7. RANDOM FOREST
# ============================================================
model_rf = RandomForestRegressor(
    n_estimators=200,          # ADDED: Tuned from 100
    max_depth=8,               # ADDED: Prevents deep tree memorization (fixes 0.071 R2 gap)
    min_samples_split=6,       # ADDED: Regularizes node splits
    min_samples_leaf=3,        # ADDED: Forces generalized leaves
    max_features=0.8,          # ADDED: Feature subsampling per tree
    random_state=42,
    n_jobs=-1
)

model_rf.fit(
    X_train,
    y_train
)

# Test prediction
y_pred_rf = model_rf.predict(
    X_test
)

# Training prediction
train_pred_rf = model_rf.predict(
    X_train
)

# Metrics
mae_rf = mean_absolute_error(
    y_test,
    y_pred_rf
)
rmse_rf = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_rf
    )
)
r2_rf = r2_score(
    y_test,
    y_pred_rf
)

train_r2_rf = r2_score(
    y_train,
    train_pred_rf
)
test_r2_rf = r2_score(
    y_test,
    y_pred_rf
)

print("\nRANDOM FOREST")
print("==============================")
print("MAE :", mae_rf)
print("RMSE:", rmse_rf)
print("R²  :", r2_rf)

# ============================================================
# 8. XGBOOST
# ============================================================
model_xgb = XGBRegressor(
    n_estimators=200,          # ADDED: Tuned from 100
    learning_rate=0.03,        # ADDED: Slower learning rate for better generalization
    max_depth=4,
    min_child_weight=3,        # ADDED: Prevents splitting on outlier noise
    subsample=0.8,             # ADDED: 80% row subsampling
    colsample_bytree=0.8,      # ADDED: 80% feature subsampling
    reg_alpha=0.1,             # ADDED: L1 regularization
    reg_lambda=1.5,            # ADDED: L2 regularization
    random_state=42,
    n_jobs=-1
)

model_xgb.fit(
    X_train,
    y_train
)

# Test prediction
y_pred_xgb = model_xgb.predict(
    X_test
)

# Training prediction
train_pred_xgb = model_xgb.predict(
    X_train
)

# Metrics
mae_xgb = mean_absolute_error(
    y_test,
    y_pred_xgb
)
rmse_xgb = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_xgb
    )
)
r2_xgb = r2_score(
    y_test,
    y_pred_xgb
)

train_r2_xgb = r2_score(
    y_train,
    train_pred_xgb
)
test_r2_xgb = r2_score(
    y_test,
    y_pred_xgb
)

print("\nXGBOOST")
print("==============================")
print("MAE :", mae_xgb)
print("RMSE:", rmse_xgb)
print("R²  :", r2_xgb)

# ============================================================
# 9. K-FOLD CROSS VALIDATION
# ============================================================
# We use 5 folds.
# Dataset:
# Fold 1 → Validation
# Fold 2 → Validation
# Fold 3 → Validation
# Fold 4 → Validation
# Fold 5 → Validation
# Every sample gets used for validation once.
# ============================================================
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ============================================================
# 10. CREATE META MODEL
# ============================================================
# Base models:
# Random Forest
# XGBoost
# Meta model:
# RidgeCV (Upgraded from Linear Regression for L2 regularization)
# cv=5 means the meta model receives
# out-of-fold predictions from the base models.
# ============================================================
model_meta = StackingRegressor(
    estimators=[
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=8,              # ADDED: Stops RF overfitting inside stack
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
    final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),  # ADDED: L2 Regularized Meta-Learner
    cv=kf,
    n_jobs=-1
)

# ============================================================
# 11. TRAIN META MODEL
# ============================================================
model_meta.fit(
    X_train,
    y_train
)

# ============================================================
# 12. META MODEL PREDICTION
# ============================================================
y_pred_meta = model_meta.predict(
    X_test
)
# ADDED: Physical constraint post-processing (no negative power at night)
y_pred_meta = np.where(X_test["GHI"] <= 0, 0.0, np.clip(y_pred_meta, 0.0, None))

# Training prediction
train_pred_meta = model_meta.predict(
    X_train
)
train_pred_meta = np.where(X_train["GHI"] <= 0, 0.0, np.clip(train_pred_meta, 0.0, None))

# ============================================================
# 13. META MODEL EVALUATION
# ============================================================
mae_meta = mean_absolute_error(
    y_test,
    y_pred_meta
)
rmse_meta = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_meta
    )
)
r2_meta = r2_score(
    y_test,
    y_pred_meta
)

train_r2_meta = r2_score(
    y_train,
    train_pred_meta
)
test_r2_meta = r2_score(
    y_test,
    y_pred_meta
)

print("\nMETA MODEL - RF + XGBOOST")
print("==============================")
print("Base Models : Random Forest + XGBoost")
print("Meta Model  : RidgeCV (L2 Regularized Linear Model)")
print("MAE :", mae_meta)
print("RMSE:", rmse_meta)
print("R²  :", r2_meta)

# ============================================================
# 14. K-FOLD VALIDATION FOR ALL MODELS
# ============================================================
models = {
    "Linear Regression": model_lr,
    "Random Forest": model_rf,
    "XGBoost": model_xgb,
    "Meta Model (RF + XGBoost)": model_meta
}

# ============================================================
# 15. RUN 5-FOLD CROSS VALIDATION
# ============================================================
cv_results = []

for name, model in models.items():
    print(
        "\nRunning K-Fold Validation:",
        name
    )

    scores = cross_validate(
        model,
        X_train,  # ADDED FIX: Run CV strictly on X_train so X_test stays 100% unseen
        y_train,
        cv=kf,
        scoring={
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2"
        },
        n_jobs=-1
    )

    # Convert negative errors to positive
    mae = -scores["test_MAE"].mean()
    rmse = -scores["test_RMSE"].mean()
    r2 = scores["test_R2"].mean()

    cv_results.append({
        "Model": name,
        "CV_MAE": mae,
        "CV_RMSE": rmse,
        "CV_R2": r2
    })

# Convert to DataFrame
cv_results = pd.DataFrame(
    cv_results
)

# ============================================================
# 16. PRINT K-FOLD RESULTS
# ============================================================
print("\n\n5-FOLD CROSS VALIDATION RESULTS")
print("==========================================")
print(
    cv_results.round(4)
)

# ============================================================
# 17. K-FOLD MODEL RANKING
# ============================================================
cv_results["MAE_Rank"] = (
    cv_results["CV_MAE"]
    .rank(ascending=True)
)
cv_results["RMSE_Rank"] = (
    cv_results["CV_RMSE"]
    .rank(ascending=True)
)
cv_results["R2_Rank"] = (
    cv_results["CV_R2"]
    .rank(ascending=False)
)

cv_results["Average_Rank"] = (
    cv_results["MAE_Rank"]
    + cv_results["RMSE_Rank"]
    + cv_results["R2_Rank"]
) / 3

cv_results = cv_results.sort_values(
    "Average_Rank"
)

# ============================================================
# 18. PRINT FINAL K-FOLD RANKING
# ============================================================
print("\n\nK-FOLD MODEL RANKING")
print("==========================================")
print(
    cv_results.round(4)
)

# ============================================================
# 19. HOLDOUT TEST SET COMPARISON
# ============================================================
results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost",
        "Meta Model (RF + XGBoost)"
    ],
    "MAE": [
        mae_lr,
        mae_rf,
        mae_xgb,
        mae_meta
    ],
    "RMSE": [
        rmse_lr,
        rmse_rf,
        rmse_xgb,
        rmse_meta
    ],
    "R2": [
        r2_lr,
        r2_rf,
        r2_xgb,
        r2_meta
    ]
})

# ============================================================
# 20. HOLDOUT MODEL RANKING
# ============================================================
results["MAE_Rank"] = (
    results["MAE"]
    .rank(ascending=True)
)
results["RMSE_Rank"] = (
    results["RMSE"]
    .rank(ascending=True)
)
results["R2_Rank"] = (
    results["R2"]
    .rank(ascending=False)
)

results["Average_Rank"] = (
    results["MAE_Rank"]
    + results["RMSE_Rank"]
    + results["R2_Rank"]
) / 3

results = results.sort_values(
    "Average_Rank"
)

print("\n\nHOLDOUT TEST SET MODEL COMPARISON")
print("==========================================")
print(
    results.round(4)
)

# ============================================================
# 21. OVERFITTING CHECK
# ============================================================
overfitting_check = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost",
        "Meta Model (RF + XGBoost)"
    ],
    "Training_R2": [
        train_r2_lr,
        train_r2_rf,
        train_r2_xgb,
        train_r2_meta
    ],
    "Testing_R2": [
        test_r2_lr,
        test_r2_rf,
        test_r2_xgb,
        test_r2_meta
    ]
})

# Calculate R² gap
overfitting_check["R2_Gap"] = (
    overfitting_check["Training_R2"]
    - overfitting_check["Testing_R2"]
)

print("\n\nOVERFITTING CHECK")
print("==============================")
print(
    overfitting_check.round(4)
)

# ============================================================
# 22. ACTUAL VS PREDICTED
# ============================================================
prediction_comparison = pd.DataFrame({
    "Actual_AC_Output": y_test.values,
    "Random_Forest": y_pred_rf,
    "XGBoost": y_pred_xgb,
    "Meta_Model": y_pred_meta
})

print("\n\nACTUAL VS PREDICTED")
print("==============================")
print(
    prediction_comparison.head(10)
)

# ============================================================
# 23. EXISTING INPUT TEST
# ============================================================
sample = X_test.iloc[[0]]
actual_output = y_test.iloc[0]

# Predictions
sample_rf = model_rf.predict(sample)[0]
sample_xgb = model_xgb.predict(sample)[0]
sample_meta = model_meta.predict(sample)[0]

print("\n\nEXISTING TEST SAMPLE")
print("==============================")
print("Actual AC Output        :", actual_output)
print("Random Forest           :", sample_rf)
print("XGBoost                 :", sample_xgb)
print("Meta Model              :", sample_meta)

# ============================================================
# 24. NEW INPUT DATA
# ============================================================
new_data = pd.DataFrame({
    "GHI": [800],
    "Temperature": [25],
    "Relative_Humidity": [60],
    "Wind_Speed": [3],
    "Wind_Direction": [180],
    "Hour": [12],
    "Day": [15],
    "Month": [10],
    "SZ_Angle": [30],
    "SA_Angle": [0]
})

# ADDED: Apply engineered features to new_data so columns match X_train
new_data = add_solar_features(new_data)[features]

# ============================================================
# 25. NEW INPUT PREDICTIONS
# ============================================================
new_rf_prediction = model_rf.predict(new_data)[0]
new_xgb_prediction = model_xgb.predict(new_data)[0]
new_meta_prediction = model_meta.predict(new_data)[0]

print("\n\nNEW INPUT PREDICTION")
print("==============================")
print("Random Forest Prediction :", new_rf_prediction)
print("XGBoost Prediction       :", new_xgb_prediction)
print("Meta Model Prediction    :", new_meta_prediction)

# ============================================================
# 26. META MODEL INFORMATION
# ============================================================
# print("\n\nMETA MODEL INFORMATION")
# print("==============================")
# print(
# "Meta Model Type: RidgeCV"
# )
# print(
# "Base Model 1   : Random Forest"
# )
# print(
# "Base Model 2   : XGBoost"
# )
# print(
# "Cross Validation: 5-Fold K-Fold"
# )