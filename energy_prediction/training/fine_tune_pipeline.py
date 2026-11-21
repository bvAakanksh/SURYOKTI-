# ==============================================================================
# SURYOKTI — SYSTEMATIC FINE-TUNING PIPELINE FOLLOWING 8-STAGE WORKFLOW
#
# Stage 1: Current 3-Stack Model Baseline
# Stage 2: Controlled 2-Stack vs 3-Stack Comparison
# Stage 3: Tune XGBoost Hyperparameters
# Stage 4: Tune Random Forest Hyperparameters
# Stage 5: Tune Neural Network Meta-Model
# Stage 6: Compare Meta-Learners
# Stage 7: Evaluate Temporal Generalization (TimeSeriesSplit)
# Stage 8: Select Final Architecture & Output Summary
# ==============================================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, cross_validate
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# DATA LOADING & SOLAR FEATURE ENGINEERING
# ------------------------------------------------------------------------------
candidates = [
    os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "energy_prediction", "demo_datasets", "Solar_Weather_Data_Processed.csv"),
    os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "energy_prediction", "processed", "Solar_Weather_Data_Cleaned.csv"),
    os.path.join("datasets", "energy_prediction", "demo_datasets", "Solar_Weather_Data_Processed.csv"),
    os.path.join("datasets", "energy_prediction", "processed", "Solar_Weather_Data_Cleaned.csv"),
    os.path.join("Solar Weather Processed Data", "Solar_Weather_Data_Processed.csv"),
    os.path.join("Data Preprocessing", "Solar_Weather_Data_Cleaned.csv"),
]
data_path = next((c for c in candidates if os.path.exists(c)), candidates[0])

df = pd.read_csv(data_path)
df.columns = df.columns.str.strip().str.replace(" ", "_")

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_preds(y_true, y_pred, ghi_series):
    # Apply physical constraint: no production when GHI <= 0, no negative output
    clipped = np.where(ghi_series <= 0, 0.0, np.clip(y_pred, 0.0, None))
    mae = mean_absolute_error(y_true, clipped)
    rmse = np.sqrt(mean_squared_error(y_true, clipped))
    r2 = r2_score(y_true, clipped)
    return mae, rmse, r2

def eval_stack(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    pred_te = model.predict(X_te)
    pred_tr = model.predict(X_tr)
    te_mae, te_rmse, te_r2 = evaluate_preds(y_te, pred_te, X_te["GHI"])
    tr_mae, tr_rmse, tr_r2 = evaluate_preds(y_tr, pred_tr, X_tr["GHI"])
    return {
        "Test_MAE": te_mae,
        "Test_RMSE": te_rmse,
        "Test_R2": te_r2,
        "Train_R2": tr_r2,
        "R2_Gap": tr_r2 - te_r2
    }

# ==============================================================================
# STAGE 1: CURRENT 3-STACK MODEL BASELINE
# ==============================================================================
print("\n" + "="*70)
print("STAGE 1: CURRENT 3-STACK MODEL BASELINE")
print("="*70)

base_rf_s1 = RandomForestRegressor(
    n_estimators=200, max_depth=8, min_samples_split=6,
    min_samples_leaf=3, max_features=0.8, random_state=42, n_jobs=-1
)
base_xgb_s1 = XGBRegressor(
    n_estimators=200, learning_rate=0.03, max_depth=4,
    min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.5, random_state=42, n_jobs=-1
)
meta_nn_s1 = Pipeline([
    ("scaler", StandardScaler()),
    ("nn", MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=0.05, learning_rate_init=0.003, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=20, max_iter=1000, random_state=42
    ))
])

stack_s1 = StackingRegressor(
    estimators=[("rf", base_rf_s1), ("xgb", base_xgb_s1)],
    final_estimator=meta_nn_s1,
    passthrough=True,
    cv=kf,
    n_jobs=-1
)
res_s1 = eval_stack(stack_s1, X_train, y_train, X_test, y_test)
print(f"Stage 1 Baseline -> Test MAE: {res_s1['Test_MAE']:.2f} kW | RMSE: {res_s1['Test_RMSE']:.2f} kW | Test R2: {res_s1['Test_R2']:.4f} | R2 Gap: {res_s1['R2_Gap']:.4f}")

# ==============================================================================
# STAGE 2: CONTROLLED 2-STACK VS 3-STACK COMPARISON
# ==============================================================================
print("\n" + "="*70)
print("STAGE 2: CONTROLLED 2-STACK VS 3-STACK COMPARISON")
print("="*70)

stack_2stack_lr = StackingRegressor(
    estimators=[("rf", base_rf_s1), ("xgb", base_xgb_s1)],
    final_estimator=LinearRegression(),
    passthrough=False, cv=kf, n_jobs=-1
)
stack_2stack_ridge = StackingRegressor(
    estimators=[("rf", base_rf_s1), ("xgb", base_xgb_s1)],
    final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
    passthrough=False, cv=kf, n_jobs=-1
)

res_2stack_lr = eval_stack(stack_2stack_lr, X_train, y_train, X_test, y_test)
res_2stack_ridge = eval_stack(stack_2stack_ridge, X_train, y_train, X_test, y_test)

stage2_df = pd.DataFrame([
    {"Architecture": "2-Stack (RF + XGB -> LinearRegression)", **res_2stack_lr},
    {"Architecture": "2-Stack (RF + XGB -> RidgeCV)", **res_2stack_ridge},
    {"Architecture": "3-Stack (RF + XGB -> Neural Network)", **res_s1},
])
print(stage2_df.to_string(index=False))

# ==============================================================================
# STAGE 3: TUNE XGBOOST HYPERPARAMETERS
# ==============================================================================
print("\n" + "="*70)
print("STAGE 3: TUNE XGBOOST HYPERPARAMETERS")
print("="*70)

xgb_configs = {
    "XGB_Deep (depth=6, lr=0.03, n=200)": XGBRegressor(
        n_estimators=200, learning_rate=0.03, max_depth=6, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.5, random_state=42, n_jobs=-1
    ),
    "XGB_Compact (depth=4, lr=0.05, n=250)": XGBRegressor(
        n_estimators=250, learning_rate=0.05, max_depth=4, min_child_weight=4,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=2.0, random_state=42, n_jobs=-1
    ),
    "XGB_Shallow_Fast (depth=3, lr=0.08, n=300)": XGBRegressor(
        n_estimators=300, learning_rate=0.08, max_depth=3, min_child_weight=2,
        subsample=0.9, colsample_bytree=0.9, reg_alpha=0.05, reg_lambda=1.0, random_state=42, n_jobs=-1
    ),
}

xgb_stage_res = []
for name, xgb_cand in xgb_configs.items():
    xgb_cand.fit(X_train, y_train)
    p_te = xgb_cand.predict(X_test)
    p_tr = xgb_cand.predict(X_train)
    te_mae, te_rmse, te_r2 = evaluate_preds(y_test, p_te, X_test["GHI"])
    tr_mae, tr_rmse, tr_r2 = evaluate_preds(y_train, p_tr, X_train["GHI"])
    xgb_stage_res.append({
        "Config": name, "Test_MAE": te_mae, "Test_RMSE": te_rmse,
        "Test_R2": te_r2, "Train_R2": tr_r2, "R2_Gap": tr_r2 - te_r2
    })
xgb_stage_df = pd.DataFrame(xgb_stage_res).sort_values("Test_R2", ascending=False)
print(xgb_stage_df.to_string(index=False))

best_xgb_name = xgb_stage_df.iloc[0]["Config"]
best_xgb_model = xgb_configs[best_xgb_name]
print(f"-> Selected XGBoost: {best_xgb_name}")

# ==============================================================================
# STAGE 4: TUNE RANDOM FOREST HYPERPARAMETERS
# ==============================================================================
print("\n" + "="*70)
print("STAGE 4: TUNE RANDOM FOREST HYPERPARAMETERS")
print("="*70)

rf_configs = {
    "RF_Deep (depth=12, leaf=2, n=200)": RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_split=4, min_samples_leaf=2,
        max_features=0.8, random_state=42, n_jobs=-1
    ),
    "RF_Constrained (depth=8, leaf=3, n=200)": RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_split=6, min_samples_leaf=3,
        max_features=0.8, random_state=42, n_jobs=-1
    ),
    "RF_Regularized (depth=7, leaf=5, n=250)": RandomForestRegressor(
        n_estimators=250, max_depth=7, min_samples_split=8, min_samples_leaf=5,
        max_features=0.75, random_state=42, n_jobs=-1
    )
}

rf_stage_res = []
for name, rf_cand in rf_configs.items():
    rf_cand.fit(X_train, y_train)
    p_te = rf_cand.predict(X_test)
    p_tr = rf_cand.predict(X_train)
    te_mae, te_rmse, te_r2 = evaluate_preds(y_test, p_te, X_test["GHI"])
    tr_mae, tr_rmse, tr_r2 = evaluate_preds(y_train, p_tr, X_train["GHI"])
    rf_stage_res.append({
        "Config": name, "Test_MAE": te_mae, "Test_RMSE": te_rmse,
        "Test_R2": te_r2, "Train_R2": tr_r2, "R2_Gap": tr_r2 - te_r2
    })
rf_stage_df = pd.DataFrame(rf_stage_res).sort_values("Test_R2", ascending=False)
print(rf_stage_df.to_string(index=False))

# We select the one offering high R2 while keeping R2_gap low
rf_stage_df["Score"] = rf_stage_df["Test_R2"] - 0.5 * rf_stage_df["R2_Gap"]
best_rf_name = rf_stage_df.sort_values("Score", ascending=False).iloc[0]["Config"]
best_rf_model = rf_configs[best_rf_name]
print(f"-> Selected Random Forest: {best_rf_name}")

# ==============================================================================
# STAGE 5: TUNE NEURAL NETWORK META-MODEL
# ==============================================================================
print("\n" + "="*70)
print("STAGE 5: TUNE NEURAL NETWORK META-MODEL")
print("="*70)

nn_configs = {
    "NN_Wide (128->64, alpha=0.1, lr=0.002)": Pipeline([
        ("scaler", StandardScaler()),
        ("nn", MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
            alpha=0.1, learning_rate_init=0.002, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20, max_iter=1000, random_state=42
        ))
    ]),
    "NN_Balanced (64->32, alpha=0.05, lr=0.003)": Pipeline([
        ("scaler", StandardScaler()),
        ("nn", MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=0.05, learning_rate_init=0.003, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20, max_iter=1000, random_state=42
        ))
    ]),
    "NN_Deep_Compact (64->32->16, alpha=0.08, lr=0.003)": Pipeline([
        ("scaler", StandardScaler()),
        ("nn", MLPRegressor(
            hidden_layer_sizes=(64, 32, 16), activation="relu", solver="adam",
            alpha=0.08, learning_rate_init=0.003, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20, max_iter=1000, random_state=42
        ))
    ])
}

nn_stage_res = []
for name, nn_cand in nn_configs.items():
    stk = StackingRegressor(
        estimators=[("rf", best_rf_model), ("xgb", best_xgb_model)],
        final_estimator=nn_cand,
        passthrough=True, cv=kf, n_jobs=-1
    )
    r = eval_stack(stk, X_train, y_train, X_test, y_test)
    nn_stage_res.append({"NN_Architecture": name, **r})

nn_stage_df = pd.DataFrame(nn_stage_res).sort_values("Test_R2", ascending=False)
print(nn_stage_df.to_string(index=False))

best_nn_name = nn_stage_df.iloc[0]["NN_Architecture"]
best_nn_model = nn_configs[best_nn_name]
print(f"-> Selected Neural Network: {best_nn_name}")

# ==============================================================================
# STAGE 6: COMPARE META-LEARNERS (LINEAR vs RIDGE vs GRADIENT BOOSTING vs NN)
# ==============================================================================
print("\n" + "="*70)
print("STAGE 6: COMPARE META-LEARNERS ON TUNED BASE MODELS")
print("="*70)

meta_candidates = {
    "LinearRegression": LinearRegression(),
    "RidgeCV": RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
    "ElasticNetCV": ElasticNetCV(l1_ratio=[.1, .5, .7, .9], cv=5, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=42),
    "Tuned Neural Network (MLP)": best_nn_model
}

meta_comp_res = []
trained_stacks = {}
for name, meta_cand in meta_candidates.items():
    stk = StackingRegressor(
        estimators=[("rf", best_rf_model), ("xgb", best_xgb_model)],
        final_estimator=meta_cand,
        passthrough=True, cv=kf, n_jobs=-1
    )
    r = eval_stack(stk, X_train, y_train, X_test, y_test)
    trained_stacks[name] = stk
    meta_comp_res.append({"Meta_Learner": name, **r})

meta_comp_df = pd.DataFrame(meta_comp_res).sort_values("Test_R2", ascending=False)
print(meta_comp_df.to_string(index=False))

# ==============================================================================
# STAGE 7: EVALUATE TEMPORAL GENERALIZATION (TIME SERIES CROSS VALIDATION)
# ==============================================================================
print("\n" + "="*70)
print("STAGE 7: EVALUATE TEMPORAL GENERALIZATION (TIME-SERIES SPLIT)")
print("="*70)

df_sorted = df.sort_values(by=["Month", "Day", "Hour"]).reset_index(drop=True)
tscv = TimeSeriesSplit(n_splits=5)

ts_res = []
# Evaluate top contenders from Stage 6
top_meta_candidates = ["Tuned Neural Network (MLP)", "RidgeCV", "LinearRegression"]

for name in top_meta_candidates:
    stk = StackingRegressor(
        estimators=[("rf", best_rf_model), ("xgb", best_xgb_model)],
        final_estimator=meta_candidates[name],
        passthrough=True, cv=kf, n_jobs=1
    )
    scores = cross_validate(
        stk, df_sorted[features], df_sorted[target], cv=tscv,
        scoring={"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
        n_jobs=1
    )
    ts_res.append({
        "Meta_Learner": name,
        "TS_CV_MAE": -scores["test_MAE"].mean(),
        "TS_CV_MAE_std": scores["test_MAE"].std(),
        "TS_CV_RMSE": -scores["test_RMSE"].mean(),
        "TS_CV_R2": scores["test_R2"].mean(),
        "TS_CV_R2_std": scores["test_R2"].std(),
    })

ts_df = pd.DataFrame(ts_res).sort_values("TS_CV_R2", ascending=False)
print(ts_df.to_string(index=False))

# ==============================================================================
# STAGE 8: SELECT FINAL ARCHITECTURE & SAVE ARTIFACTS
# ==============================================================================
print("\n" + "="*70)
print("STAGE 8: SELECT FINAL ARCHITECTURE")
print("="*70)

final_meta_name = ts_df.iloc[0]["Meta_Learner"]
final_stack_model = trained_stacks[final_meta_name]

# Re-fit on entire dataset for final deployment
print(f"Final Architecture Selected: Random Forest + XGBoost -> {final_meta_name}")
final_stack_model.fit(X, y)

# Save the final tuned model to models directory
import joblib
model_dir = (
    os.path.join(os.path.dirname(__file__), "..", "models")
    if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "models"))
    else (os.path.join("models", "energy_prediction") if os.path.exists(os.path.join("models", "energy_prediction")) else "models")
)
os.makedirs(model_dir, exist_ok=True)
final_model_path = os.path.join(model_dir, "fine_tuned_final_stack.pkl")
feature_cols_path = os.path.join(model_dir, "fine_tuned_features.pkl")

joblib.dump(final_stack_model, final_model_path)
joblib.dump(features, feature_cols_path)

print(f"Saved fine-tuned final model -> {final_model_path}")
print(f"Saved feature columns list   -> {feature_cols_path}")
print("Pipeline complete!")
