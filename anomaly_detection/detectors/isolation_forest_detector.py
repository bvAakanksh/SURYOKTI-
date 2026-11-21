import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    max_error,
    explained_variance_score
)
from scipy.stats import pearsonr
from xgboost import XGBRegressor


# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

import os
from pathlib import Path
candidates = [
    Path(__file__).resolve().parents[2] / "datasets" / "energy_prediction" / "demo_datasets" / "Solar_Weather_Data_Processed.csv",
    Path("datasets/energy_prediction/demo_datasets/Solar_Weather_Data_Processed.csv"),
    Path(__file__).resolve().parents[2] / "Solar Weather Processed Data" / "Solar_Weather_Data_Processed.csv",
    Path("Solar Weather Processed Data/Solar_Weather_Data_Processed.csv"),
]
file_path = next((str(c) for c in candidates if c.exists()), str(candidates[0]))

df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Convert timestamp
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

# Sort data chronologically
df = df.sort_values("Timestamp").reset_index(drop=True)

print("Dataset Loaded")
print("Shape:", df.shape)


# ==========================================
# STEP 2: CREATE TIME FEATURES
# ==========================================

# Use Timestamp as the reliable source of time
df["Timestamp_Hour"] = df["Timestamp"].dt.hour

# Cyclical encoding of hour
df["Hour_Sin"] = np.sin(
    2 * np.pi * df["Timestamp_Hour"] / 24
)

df["Hour_Cos"] = np.cos(
    2 * np.pi * df["Timestamp_Hour"] / 24
)

# Convert Wind Direction into cyclical features
df["Wind_Direction_Sin"] = np.sin(
    np.deg2rad(df["Wind_Direction"])
)

df["Wind_Direction_Cos"] = np.cos(
    np.deg2rad(df["Wind_Direction"])
)

# Convert Solar Azimuth Angle into cyclical features
df["SA_Angle_Sin"] = np.sin(
    np.deg2rad(df["SA_Angle"])
)

df["SA_Angle_Cos"] = np.cos(
    np.deg2rad(df["SA_Angle"])
)


# ==========================================
# STEP 3: SELECT FEATURES
# ==========================================

features = [
    "GHI",
    "Temperature",
    "Relative_Humidity",
    "Wind_Speed",
    "AC_Output",
    "SZ_Angle",
    "Hour_Sin",
    "Hour_Cos",
    "Wind_Direction_Sin",
    "Wind_Direction_Cos",
    "SA_Angle_Sin",
    "SA_Angle_Cos"
]

# Keep only required columns
data = df[["Timestamp"] + features].copy()

# Remove rows with missing values in required columns
data = data.dropna(subset=features)

print("\nFeatures Used:")
print(features)

print("\nClean Dataset Shape:", data.shape)


# ==========================================
# STEP 4: CHRONOLOGICAL TRAIN-TEST SPLIT
# ==========================================

split_index = int(len(data) * 0.70)

train_data = data.iloc[:split_index].copy()
test_data = data.iloc[split_index:].copy()

X_train = train_data[features]
X_test = test_data[features]

print("\nTraining Rows:", len(X_train))
print("Testing Rows:", len(X_test))


# ==========================================
# STEP 5: TRAIN DAYLIGHT ISOLATION FOREST
# ==========================================

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

DAYLIGHT_GHI_THRESHOLD = 50.0
daylight_train_mask = train_data["GHI"] > DAYLIGHT_GHI_THRESHOLD
daylight_test_mask = test_data["GHI"] > DAYLIGHT_GHI_THRESHOLD

if not daylight_train_mask.any():
    raise ValueError(
        f"No daylight training observations found (GHI > {DAYLIGHT_GHI_THRESHOLD})."
    )

model.fit(X_train.loc[daylight_train_mask])

print(f"\nIsolation Forest Training Completed on {daylight_train_mask.sum()} daylight observations")


# ==========================================
# STEP 6: DETECT ANOMALIES
# ==========================================

# Prediction:
#  1  = Normal
# -1  = Anomaly

test_data["Anomaly_Prediction"] = 1
test_data["Anomaly_Score"] = np.nan
if daylight_test_mask.any():
    test_data.loc[daylight_test_mask, "Anomaly_Prediction"] = model.predict(
        X_test.loc[daylight_test_mask]
    )
    test_data.loc[daylight_test_mask, "Anomaly_Score"] = model.decision_function(
        X_test.loc[daylight_test_mask]
    )
test_data["Anomaly_Status"] = np.select(
    [
        ~daylight_test_mask,
        test_data["Anomaly_Prediction"] == -1,
    ],
    ["NIGHTTIME_INACTIVE", "POTENTIAL_OPERATIONAL_ANOMALY"],
    default="DAYLIGHT_NORMAL",
)


# ==========================================
# STEP 7: DISPLAY RESULTS
# ==========================================

total_anomalies = (
    test_data["Anomaly_Prediction"] == -1
).sum()

total_normal = (
    test_data["Anomaly_Prediction"] == 1
).sum()

print("\n========== ANOMALY DETECTION RESULTS ==========")

print("Total Test Observations:", len(test_data))
print("Normal Observations    :", total_normal)
print("Anomalies Detected     :", total_anomalies)

print(
    "Anomaly Percentage     :",
    round(total_anomalies / len(test_data) * 100, 2),
    "%"
)


# ==========================================
# STEP 8: DISPLAY ANOMALY OBSERVATIONS
# ==========================================

anomalies = test_data[test_data["Anomaly_Prediction"] == -1]

print("\nFirst 20 Detected Anomalies:")

print(
    anomalies[
        [
            "Timestamp",
            "GHI",
            "Temperature",
            "Relative_Humidity",
            "AC_Output",
            "Anomaly_Score"
        ]
    ].head(20).to_string(index=False)
)


# ==========================================
# STEP 9: SAVE ANOMALY DETECTION RESULTS
# ==========================================

output_path = str(Path(__file__).resolve().parents[1] / "results" / "csv_outputs" / "isolation_forest_results.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
test_data.to_csv(output_path, index=False)
print("\nInitial Results Saved To:", output_path)


# ==========================================
# STEP 10: REGRESSION PERFORMANCE EVALUATION
# (R², MAE, RMSE, MSE, MAPE, Max Error, Pearson r, nRMSE)
# ==========================================

print("\n============================================================")
print("STEP 10: REGRESSION METRICS & PARAMETER EVALUATION")
print("============================================================")

# Predictor features (exclude target AC_Output to prevent data leakage)
reg_features = [f for f in features if f != "AC_Output"]
target_col = "AC_Output"

print(f"Regression Predictor Features ({len(reg_features)}):")
print(reg_features)

# Train a regression model on the training set
print("\nTraining Regression Model (XGBoost Regressor) on Training Split...")
reg_model = XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1
)

reg_model.fit(train_data[reg_features], train_data[target_col])

# Generate predictions on test data with physical clipping constraints
raw_pred_test = reg_model.predict(test_data[reg_features])
pred_test = np.clip(raw_pred_test, 0.0, None)
# Physical constraint: night / zero GHI must yield zero AC power
pred_test = np.where(test_data["GHI"] <= 0, 0.0, pred_test)

# Attach predictions and residual errors to test_data
test_data["Predicted_AC_Output"] = pred_test
test_data["Residual_Error"] = test_data[target_col] - pred_test
test_data["Abs_Error"] = np.abs(test_data["Residual_Error"])


def compute_comprehensive_metrics(y_true, y_pred):
    """
    Computes all standard regression parameters and solar forecasting metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    if n == 0:
        return {}

    # Core Metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    max_err = max_error(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred)

    # Pearson Correlation Coefficient (r)
    if np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        pearson_r, _ = pearsonr(y_true, y_pred)
    else:
        pearson_r = 0.0

    # Mean Bias Error (MBE) -> positive means overprediction, negative means underprediction
    mbe = np.mean(y_pred - y_true)

    # Normalized RMSE (nRMSE in %) relative to mean of actual output
    mean_actual = np.mean(y_true)
    nrmse_mean = (rmse / mean_actual * 100.0) if mean_actual > 0 else 0.0

    # Normalized RMSE relative to capacity/range (max - min)
    range_actual = np.ptp(y_true)
    nrmse_range = (rmse / range_actual * 100.0) if range_actual > 0 else 0.0

    # MAPE (calculated only for daytime/positive generation > 10 kW to prevent division by zero)
    positive_mask = y_true > 10.0
    if np.any(positive_mask):
        mape = np.mean(np.abs((y_true[positive_mask] - y_pred[positive_mask]) / y_true[positive_mask])) * 100.0
    else:
        mape = np.nan

    return {
        "Count": n,
        "R2_Score": r2,
        "MAE_kW": mae,
        "RMSE_kW": rmse,
        "MSE_kW2": mse,
        "Max_Error_kW": max_err,
        "Explained_Variance": ev,
        "Pearson_r": pearson_r,
        "MBE_kW": mbe,
        "nRMSE_Mean_%": nrmse_mean,
        "nRMSE_Range_%": nrmse_range,
        "MAPE_%": mape
    }


# Evaluate across 3 subsets:
# 1. All Test Observations
# 2. Normal Observations Only
# 3. Anomalous Observations Only
metrics_all = compute_comprehensive_metrics(
    test_data[target_col],
    test_data["Predicted_AC_Output"]
)

normal_mask = test_data["Anomaly_Prediction"] == 1
metrics_normal = compute_comprehensive_metrics(
    test_data.loc[normal_mask, target_col],
    test_data.loc[normal_mask, "Predicted_AC_Output"]
)

anomaly_mask = test_data["Anomaly_Prediction"] == -1
metrics_anomaly = compute_comprehensive_metrics(
    test_data.loc[anomaly_mask, target_col],
    test_data.loc[anomaly_mask, "Predicted_AC_Output"]
)

# Format into a clean comparison table
metrics_comparison = pd.DataFrame([
    {"Subset": "All Test Observations", **metrics_all},
    {"Subset": "Normal Data Only (Clean)", **metrics_normal},
    {"Subset": "Anomalies Only (Flagged)", **metrics_anomaly}
])

print("\n--- COMPREHENSIVE PARAMETER EVALUATION TABLE ---")
pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(metrics_comparison.to_string(index=False))


# ==========================================
# STEP 11: IMPACT OF REMOVING TRAINING ANOMALIES
# (Retraining Regressor on Anomaly-Filtered Training Data)
# ==========================================

print("\n============================================================")
print("STEP 11: EFFECT OF ANOMALY FILTERING ON MODEL RETRAINING")
print("============================================================")

# Detect anomalies in training set using the fitted Isolation Forest
train_anom_preds = model.predict(X_train.loc[daylight_train_mask])
train_normal_mask = np.ones(len(train_data), dtype=bool)
train_normal_mask[daylight_train_mask.to_numpy()] = train_anom_preds == 1
train_data_clean = train_data.loc[train_normal_mask].copy()

print(f"Total Training Samples: {len(train_data)}")
print(f"Clean Training Samples: {len(train_data_clean)} ({len(train_data) - len(train_data_clean)} training anomalies removed)")

# Train clean regressor
reg_model_clean = XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1
)
reg_model_clean.fit(train_data_clean[reg_features], train_data_clean[target_col])

# Predict with clean model on normal test set
pred_test_clean = np.clip(reg_model_clean.predict(test_data[reg_features]), 0.0, None)
pred_test_clean = np.where(test_data["GHI"] <= 0, 0.0, pred_test_clean)

metrics_clean_trained_on_normal = compute_comprehensive_metrics(
    test_data.loc[normal_mask, target_col],
    pred_test_clean[normal_mask]
)

training_impact_df = pd.DataFrame([
    {
        "Training Strategy": "Trained on Raw Train Data",
        "Evaluated On": "Normal Test Data",
        "R2_Score": metrics_normal["R2_Score"],
        "MAE_kW": metrics_normal["MAE_kW"],
        "RMSE_kW": metrics_normal["RMSE_kW"],
        "Pearson_r": metrics_normal["Pearson_r"]
    },
    {
        "Training Strategy": "Trained on Clean Train Data (Anomalies Filtered)",
        "Evaluated On": "Normal Test Data",
        "R2_Score": metrics_clean_trained_on_normal["R2_Score"],
        "MAE_kW": metrics_clean_trained_on_normal["MAE_kW"],
        "RMSE_kW": metrics_clean_trained_on_normal["RMSE_kW"],
        "Pearson_r": metrics_clean_trained_on_normal["Pearson_r"]
    }
])

print("\n--- MODEL RETRAINING COMPARISON ---")
print(training_impact_df.to_string(index=False))


# ==========================================
# STEP 12: SAVE FINAL ENRICHED RESULTS
# ==========================================

test_data.to_csv(output_path, index=False)
print(f"\nFinal Enriched Results (with Predictions, Residuals & Scores) Saved To: {output_path}")
print("============================================================\n")
