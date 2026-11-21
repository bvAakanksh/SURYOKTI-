import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.model_selection import TimeSeriesSplit

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
# print(df.shape)
# print(df.head())

df.columns = (df.columns.str.strip().str.replace(" ", "_"))
# print(df.columns)

# -------- ADDED: SOLAR PHYSICS & CYCLICAL FEATURE ENGINEERING --------
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

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# BASELINE MODEL - Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)
train_pred_lr = model_lr.predict(X_train)

train_r2_lr = r2_score(y_train, train_pred_lr)
test_r2_lr = r2_score(y_test, y_pred_lr)

# MAE
mae_lr = mean_absolute_error(y_test, y_pred_lr)
# RMSE
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
# R-Squared
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression")
print("------------------------")
print("MAE :", mae_lr)
print("RMSE:", rmse_lr)
print("R²  :", r2_lr)

# RANDOM FOREST REGRESSION
model_rf = RandomForestRegressor(
    n_estimators=200,          # ADDED: Tuned parameters to prevent overfitting
    max_depth=8,
    min_samples_split=6,
    min_samples_leaf=3,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)
model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)

# EVALUATION METRICS
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

train_pred_rf = model_rf.predict(X_train)
train_r2_rf = r2_score(y_train, train_pred_rf)
test_r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest")
print("------------------------")
print("MAE :", mae_rf)
print("RMSE:", rmse_rf)
print("R²  :", r2_rf)

# ---- XBoost MODEL ----
model_xgb = XGBRegressor(
    n_estimators=200,          # ADDED: Tuned parameters with subsampling & regularization
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
model_xgb.fit(X_train, y_train)

y_pred_xgb = model_xgb.predict(X_test)

train_pred_xgb = model_xgb.predict(X_train)
train_r2_xgb = r2_score(y_train, train_pred_xgb)
test_r2_xgb = r2_score(y_test, y_pred_xgb)

# EVALUATION METRICS
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print("XBoost")
print("------------------------")
print("MAE :", mae_xgb)
print("RMSE:", rmse_xgb)
print("R²  :", r2_xgb)

# -------- HYBRID MODEL: XGBOOST + NEURAL NETWORK --------
# Explicit shuffled KFold for inner stacking loop (fixes unshuffled cv=5 bug)
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Neural Network pipeline (Upgraded with L2 alpha regularization & Early Stopping)
model_nn = Pipeline([
    ("scaler", StandardScaler()),
    ("nn", MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.05,               # ADDED: L2 regularization to keep R2 gap minimal
        learning_rate_init=0.003, # ADDED: Stable initial learning rate
        early_stopping=True,      # ADDED: Stops automatically when validation plateaus
        validation_fraction=0.15,
        n_iter_no_change=20,
        max_iter=1000,
        random_state=42
    ))
])

# Hybrid stacking model (Upgraded: XGBoost Base -> Neural Network Level-2 Meta-Model)
model_hybrid_xgb_nn = StackingRegressor(
    estimators=[
        (
            "xgb",
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
    final_estimator=model_nn,     # UPGRADED: Neural Network acts as the true Meta-Model
    passthrough=True,             # ADDED: Passes scaled weather features + XGB OOF pred to NN
    cv=cv,                        # FIXED: Uses shuffled KFold instead of unshuffled cv=5
    n_jobs=-1
)

# -------- CROSS VALIDATION --------
models = {
    "Linear Regression": model_lr,
    "Random Forest": model_rf,
    "XGBoost": model_xgb,
    "Hybrid (XGBoost + Neural Network)": model_hybrid_xgb_nn
}

cv_results = []

for name, model in models.items():
    scores = cross_validate(
        model,
        X_train,  # ADDED FIX: Run CV on X_train so X_test remains unseen holdout data
        y_train,
        cv=cv,
        scoring={
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2"
        },
        n_jobs=-1
    )

    mae = -scores["test_MAE"].mean()
    rmse = -scores["test_RMSE"].mean()
    r2 = scores["test_R2"].mean()

    cv_results.append({
        "Model": name,
        "CV_MAE": mae,
        "CV_RMSE": rmse,
        "CV_R2": r2
    })

cv_results = pd.DataFrame(cv_results)

print("\n5-FOLD CROSS VALIDATION RESULTS")
print("======================================")
print(cv_results.round(4))

cv_results["MAE_Rank"] = cv_results["CV_MAE"].rank(ascending=True)
cv_results["RMSE_Rank"] = cv_results["CV_RMSE"].rank(ascending=True)
cv_results["R2_Rank"] = cv_results["CV_R2"].rank(ascending=False)

cv_results["Average_Rank"] = (
    cv_results["MAE_Rank"] +
    cv_results["RMSE_Rank"] +
    cv_results["R2_Rank"]
) / 3

cv_results = cv_results.sort_values("Average_Rank")

print("\nCROSS-VALIDATION MODEL RANKING")
print("======================================")
print(cv_results.round(4))

# Train
model_hybrid_xgb_nn.fit(X_train, y_train)

# Test predictions
y_pred_hybrid_xgb_nn = model_hybrid_xgb_nn.predict(X_test)
# ADDED: Physical constraint post-processing (clips negative predictions at night to 0)
y_pred_hybrid_xgb_nn = np.where(X_test["GHI"] <= 0, 0.0, np.clip(y_pred_hybrid_xgb_nn, 0.0, None))

# Training predictions
train_pred_hybrid_xgb_nn = model_hybrid_xgb_nn.predict(X_train)
train_pred_hybrid_xgb_nn = np.where(X_train["GHI"] <= 0, 0.0, np.clip(train_pred_hybrid_xgb_nn, 0.0, None))

# Evaluation
mae_hybrid_xgb_nn = mean_absolute_error(
    y_test, y_pred_hybrid_xgb_nn
)
rmse_hybrid_xgb_nn = np.sqrt(
    mean_squared_error(
        y_test, y_pred_hybrid_xgb_nn
    )
)
r2_hybrid_xgb_nn = r2_score(
    y_test, y_pred_hybrid_xgb_nn
)

train_r2_hybrid_xgb_nn = r2_score(
    y_train, train_pred_hybrid_xgb_nn
)
test_r2_hybrid_xgb_nn = r2_score(
    y_test, y_pred_hybrid_xgb_nn
)

print("\nHybrid Model (XGBoost + Neural Network)")
print("---------------------------------------")
print("MAE :", mae_hybrid_xgb_nn)
print("RMSE:", rmse_hybrid_xgb_nn)
print("R²  :", r2_hybrid_xgb_nn)

# Comparison
results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost",
        "Hybrid (XGBoost + Neural Network)"
    ],
    "MAE": [
        mae_lr,
        mae_rf,
        mae_xgb,
        mae_hybrid_xgb_nn
    ],
    "RMSE": [
        rmse_lr,
        rmse_rf,
        rmse_xgb,
        rmse_hybrid_xgb_nn
    ],
    "R2": [
        r2_lr,
        r2_rf,
        r2_xgb,
        r2_hybrid_xgb_nn
    ]
})

# Rank the models
results["MAE_Rank"] = results["MAE"].rank(ascending=True)
results["RMSE_Rank"] = results["RMSE"].rank(ascending=True)
results["R2_Rank"] = results["R2"].rank(ascending=False)

# ==========================================
# TIME SERIES CROSS VALIDATION
# ==========================================
# IMPORTANT:
# Make sure data is in chronological order
df_sorted = df.sort_values(
    by=["Month", "Day", "Hour"]
).reset_index(drop=True)

# Re-create X and y after sorting
X_ts = df_sorted[features]
y_ts = df_sorted[target]

# Time Series Split
tscv = TimeSeriesSplit(
    n_splits=5
)

# Models to compare
models_ts = {
    "Linear Regression": model_lr,
    "Random Forest": model_rf,
    "XGBoost": model_xgb,
    "Hybrid (XGBoost + Neural Network)": model_hybrid_xgb_nn
}

ts_results = []

for name, model in models_ts.items():
    scores = cross_validate(
        model,
        X_ts,
        y_ts,
        cv=tscv,
        scoring={
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2"
        },
        n_jobs=-1
    )

    # Convert negative sklearn errors back to positive
    mae = -scores["test_MAE"].mean()
    rmse = -scores["test_RMSE"].mean()
    r2 = scores["test_R2"].mean()

    ts_results.append({
        "Model": name,
        "TS_MAE": mae,
        "TS_RMSE": rmse,
        "TS_R2": r2
    })

# Convert to dataframe
ts_results = pd.DataFrame(ts_results)

print("\nTIME SERIES CROSS VALIDATION RESULTS")
print("==========================================")
print(ts_results.round(4))

# ==========================================
# TIME SERIES MODEL RANKING
# ==========================================
ts_results["MAE_Rank"] = ts_results["TS_MAE"].rank(ascending=True)
ts_results["RMSE_Rank"] = ts_results["TS_RMSE"].rank(ascending=True)
ts_results["R2_Rank"] = ts_results["TS_R2"].rank(ascending=False)

ts_results["Average_Rank"] = (
    ts_results["MAE_Rank"] +
    ts_results["RMSE_Rank"] +
    ts_results["R2_Rank"]
) / 3

ts_results = ts_results.sort_values("Average_Rank")

print("\nTIME SERIES MODEL RANKING")
print("==========================================")
print(ts_results.round(4))

# Calculate average rank
results["Average_Rank"] = (
    results["MAE_Rank"] +
    results["RMSE_Rank"] +
    results["R2_Rank"]
) / 3

# Sort by overall rank
results = results.sort_values("Average_Rank")

print("\nMODEL COMPARISON")
print("==============================")
print(results.round(4))

# -------EXISTING INPUT FEATURES--------
# Take one sample from the test dataset
sample = X_test.iloc[[0]]

# Get the actual AC output
actual_output = y_test.iloc[0]

# Predict using our trained XGBoost model
predicted_output = model_xgb.predict(sample)[0]

# print("Actual AC Output    :", actual_output)
# print("Predicted AC Output :", predicted_output)
# print("Prediction Error    :", actual_output - predicted_output)
# print("Input conditions:")
# print(sample)
# print("\nActual AC Output:")
# print(actual_output)
# print("\nPredicted AC Output:")
# print(predicted_output)

# -----NEW INPUT FEATURES------
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

# ADDED: Apply engineered features to new_data so columns match training data
new_data = add_solar_features(new_data)[features]

new_prediction = model_xgb.predict(new_data)
# print("Predicted AC Output:", new_prediction[0])

# ---Overfitting Check---
overfitting_check = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost",
        "Hybrid (XGBoost + Neural Network)"
    ],
    "Training_R2": [
        train_r2_lr,
        train_r2_rf,
        train_r2_xgb,
        train_r2_hybrid_xgb_nn
    ],
    "Testing_R2": [
        test_r2_lr,
        test_r2_rf,
        test_r2_xgb,
        test_r2_hybrid_xgb_nn
    ]
})

overfitting_check["R2_Gap"] = (
    overfitting_check["Training_R2"] -
    overfitting_check["Testing_R2"]
)

print("\nOVERFITTING CHECK")
print("==============================")
print(overfitting_check.round(4))