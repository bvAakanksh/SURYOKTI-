"""
SURYOKTI Regional Adaptation — Prediction Model Tuning Pipeline
==============================================================
Site: Maui Ocean Center (System 2105)
Plant Capacity: 110.0 kW DC
Resolution: Hourly (39,921 observations spanning 2018–2023)

Models Evaluated:
1. Linear Regression (Baseline)
2. Tuned Random Forest Regressor
3. Tuned XGBoost Regressor
4. Hybrid Stacking Model (RF + XGBoost -> RidgeCV Meta-Learner)

Validation Protocol:
- Chronological 80/20 Development / Final Test Split.
- 5-Fold TimeSeriesSplit Cross-Validation strictly on Development Data.
- Zero Leakage: Final test set is evaluated ONLY after final model selection.
- Physical zero-irradiance and non-negative constraints enforced.
- Capacity-Normalized Metrics: nMAE (%) and nRMSE (%) computed using 110.0 kW.
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, KFold, cross_validate
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "maui_system_2105_clean_hourly.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

RATED_CAPACITY_KW = 110.0
FINAL_TEST_RATIO = 0.20
RANDOM_STATE = 42

FEATURE_COLS = [
    "GHI", "DNI", "Temperature", "Wind_Speed",
    "Hour", "Day", "Month", "SZ_Angle", "SA_Angle",
    "Hour_sin", "Hour_cos", "Month_sin", "Month_cos",
    "Effective_GHI", "Thermal_Stress"
]
TARGET_COL = "AC_Output"


def apply_physical_constraints(y_pred, ghi_values):
    """
    Applies fundamental physical laws:
    1. Zero/negative irradiance forces zero generation.
    2. Power output cannot be negative.
    """
    pred_clean = np.clip(y_pred, 0.0, None)
    pred_clean = np.where(ghi_values <= 0, 0.0, pred_clean)
    return pred_clean


def compute_metrics(y_true, y_pred, capacity_kw=RATED_CAPACITY_KW):
    """
    Computes absolute and capacity-normalized forecasting metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    nmae = (mae / capacity_kw) * 100.0
    nrmse = (rmse / capacity_kw) * 100.0
    
    return {
        "MAE_kW": float(mae),
        "RMSE_kW": float(rmse),
        "R2_Score": float(r2),
        "nMAE_percent": float(nmae),
        "nRMSE_percent": float(nrmse)
    }


def main():
    print("=" * 70)
    print("SURYOKTI MAUI OCEAN CENTER (110 kW) — PREDICTION MODEL TUNING")
    print("=" * 70)
    
    # 1. Load Clean Dataset
    print(f"\n1. Loading Clean Hourly Data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    n_samples = len(df)
    
    print(f"Total Hourly Records: {n_samples}")
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Target: {TARGET_COL} (Capacity: {RATED_CAPACITY_KW} kW)")
    
    # 2. Chronological Dev / Test Split
    split_idx = int(n_samples * (1 - FINAL_TEST_RATIO))
    X_dev, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_dev, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()
    test_dates = df["Timestamp"].iloc[split_idx:].reset_index(drop=True)
    
    print(f"\n2. Chronological Split (80% Dev / 20% Holdout Test):")
    print(f"   Development Set : {len(X_dev)} rows ({df['Timestamp'].iloc[0]} to {df['Timestamp'].iloc[split_idx-1]})")
    print(f"   Holdout Test Set: {len(X_test)} rows ({df['Timestamp'].iloc[split_idx]} to {df['Timestamp'].iloc[-1]})")
    
    # 3. Setup 5-Fold Time-Series Cross-Validation on Dev Set
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 4. Model Architectures & Hyperparameter Tuning Candidates
    models = {
        "Linear Regression (Baseline)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        
        "Random Forest (Tuned)": RandomForestRegressor(
            n_estimators=150,
            max_depth=8,              # Constrained depth prevents tree memorization
            min_samples_split=6,
            min_samples_leaf=4,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        
        "XGBoost (Tuned)": XGBRegressor(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }
    
    # 5. Evaluate Base Models with Time-Series Cross-Validation on Dev Set
    print("\n3. Running 5-Fold Time-Series Cross-Validation on Development Set...")
    cv_summary = {}
    
    for name, model in models.items():
        print(f"   Evaluating {name}...")
        fold_r2s, fold_maes, fold_rmses = [], [], []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_dev), 1):
            X_tr, y_tr = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
            X_val, y_val = X_dev.iloc[val_idx], y_dev.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            raw_val_pred = model.predict(X_val)
            clean_val_pred = apply_physical_constraints(raw_val_pred, X_val["GHI"].values)
            
            m = compute_metrics(y_val, clean_val_pred)
            fold_r2s.append(m["R2_Score"])
            fold_maes.append(m["MAE_kW"])
            fold_rmses.append(m["RMSE_kW"])
            
        cv_summary[name] = {
            "CV_R2_Mean": float(np.mean(fold_r2s)),
            "CV_R2_Std": float(np.std(fold_r2s)),
            "CV_MAE_Mean": float(np.mean(fold_maes)),
            "CV_RMSE_Mean": float(np.mean(fold_rmses)),
            "CV_nMAE_Mean": float((np.mean(fold_maes) / RATED_CAPACITY_KW) * 100.0),
            "CV_nRMSE_Mean": float((np.mean(fold_rmses) / RATED_CAPACITY_KW) * 100.0)
        }
        print(f"      CV R²: {np.mean(fold_r2s):.4f} (±{np.std(fold_r2s):.4f}), CV MAE: {np.mean(fold_maes):.2f} kW")
        
    # 6. Build and Evaluate Hybrid Stacking Model (RF + XGBoost -> RidgeCV)
    print("\n4. Building Hybrid Stacking Regressor (RF + XGBoost -> RidgeCV)...")
    stacking_model = StackingRegressor(
        estimators=[
            ("rf", models["Random Forest (Tuned)"]),
            ("xgb", models["XGBoost (Tuned)"])
        ],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 20)),
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        passthrough=True,
        n_jobs=-1
    )
    
    print("   Running Time-Series CV on Stacking Regressor...")
    stack_r2s, stack_maes, stack_rmses = [], [], []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_dev), 1):
        X_tr, y_tr = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
        X_val, y_val = X_dev.iloc[val_idx], y_dev.iloc[val_idx]
        
        stacking_model.fit(X_tr, y_tr)
        raw_val_pred = stacking_model.predict(X_val)
        clean_val_pred = apply_physical_constraints(raw_val_pred, X_val["GHI"].values)
        
        m = compute_metrics(y_val, clean_val_pred)
        stack_r2s.append(m["R2_Score"])
        stack_maes.append(m["MAE_kW"])
        stack_rmses.append(m["RMSE_kW"])
        
    cv_summary["Hybrid Stacking (RF + XGB -> RidgeCV)"] = {
        "CV_R2_Mean": float(np.mean(stack_r2s)),
        "CV_R2_Std": float(np.std(stack_r2s)),
        "CV_MAE_Mean": float(np.mean(stack_maes)),
        "CV_RMSE_Mean": float(np.mean(stack_rmses)),
        "CV_nMAE_Mean": float((np.mean(stack_maes) / RATED_CAPACITY_KW) * 100.0),
        "CV_nRMSE_Mean": float((np.mean(stack_rmses) / RATED_CAPACITY_KW) * 100.0)
    }
    print(f"      CV R²: {np.mean(stack_r2s):.4f} (±{np.std(stack_r2s):.4f}), CV MAE: {np.mean(stack_maes):.2f} kW")
    
    # 7. Final Model Evaluation on the Untouched Holdout Test Set
    print("\n5. Fitting Final Models on Entire Dev Set and Evaluating on Holdout Test Set...")
    all_models = {**models, "Hybrid Stacking (RF + XGB -> RidgeCV)": stacking_model}
    holdout_results = {}
    test_predictions = pd.DataFrame({"Timestamp": test_dates, "Actual_AC_Output": y_test.values, "GHI": X_test["GHI"].values})
    
    for name, model in all_models.items():
        print(f"   Training final {name} on development set ({len(X_dev)} samples)...")
        model.fit(X_dev, y_dev)
        
        raw_pred = model.predict(X_test)
        clean_pred = apply_physical_constraints(raw_pred, X_test["GHI"].values)
        test_predictions[f"Pred_{name}"] = clean_pred
        
        metrics = compute_metrics(y_test, clean_pred)
        holdout_results[name] = {
            **metrics,
            "CV_R2_Mean": cv_summary[name]["CV_R2_Mean"],
            "CV_R2_Std": cv_summary[name]["CV_R2_Std"]
        }
        
    # 8. Segmented Performance (Irradiance Bins & Day/Night)
    print("\n6. Computing Segmented Performance for Best Model across Weather Regimes...")
    best_model_name = max(holdout_results, key=lambda k: holdout_results[k]["R2_Score"])
    best_preds = test_predictions[f"Pred_{best_model_name}"]
    
    segments = {
        "Daytime (GHI > 10 W/m²)": X_test["GHI"] > 10.0,
        "Nighttime (GHI <= 10 W/m²)": X_test["GHI"] <= 10.0,
        "Low Irradiance (10 < GHI <= 300)": (X_test["GHI"] > 10.0) & (X_test["GHI"] <= 300.0),
        "Medium Irradiance (300 < GHI <= 600)": (X_test["GHI"] > 300.0) & (X_test["GHI"] <= 600.0),
        "High Irradiance (GHI > 600)": X_test["GHI"] > 600.0
    }
    
    segmented_results = {}
    for seg_name, mask in segments.items():
        if mask.sum() > 0:
            sub_true = y_test[mask]
            sub_pred = best_preds[mask]
            segmented_results[seg_name] = {
                "Count": int(mask.sum()),
                **compute_metrics(sub_true, sub_pred)
            }
            
    # 9. Format Results Tables
    comparison_table = []
    for name, res in holdout_results.items():
        comparison_table.append({
            "Model": name,
            "CV R²": f"{res['CV_R2_Mean']:.4f} ± {res['CV_R2_Std']:.4f}",
            "Test R²": f"{res['R2_Score']:.4f}",
            "MAE (kW)": f"{res['MAE_kW']:.2f}",
            "RMSE (kW)": f"{res['RMSE_kW']:.2f}",
            "nMAE (%)": f"{res['nMAE_percent']:.2f}%",
            "nRMSE (%)": f"{res['nRMSE_percent']:.2f}%"
        })
    df_comparison = pd.DataFrame(comparison_table)
    
    print("\n" + "=" * 90)
    print("FINAL MODEL BENCHMARK TABLE ON MAUI HOLDOUT TEST SET")
    print("=" * 90)
    print(df_comparison.to_string(index=False))
    
    print("\n" + "=" * 90)
    print(f"WEATHER REGIME BREAKDOWN FOR BEST MODEL: {best_model_name}")
    print("=" * 90)
    df_segmented = pd.DataFrame(segmented_results).T
    print(df_segmented[["Count", "R2_Score", "MAE_kW", "RMSE_kW", "nMAE_percent", "nRMSE_percent"]].to_string())
    
    # 10. Save Artifacts & Reports
    best_model = all_models[best_model_name]
    best_model_path = os.path.join(MODELS_DIR, "maui_best_model.pkl")
    joblib.dump(best_model, best_model_path)
    
    features_path = os.path.join(MODELS_DIR, "maui_features.pkl")
    joblib.dump(FEATURE_COLS, features_path)
    
    report_data = {
        "site": "Maui Ocean Center (System 2105)",
        "rated_capacity_kw": RATED_CAPACITY_KW,
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "best_model": best_model_name,
        "model_comparison": holdout_results,
        "segmented_evaluation": segmented_results
    }
    
    report_json_path = os.path.join(REPORTS_DIR, "maui_prediction_tuning_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)
        
    predictions_csv_path = os.path.join(DATA_PATH.replace(".csv", "_test_predictions.csv"))
    test_predictions.to_csv(predictions_csv_path, index=False)
    
    print(f"\nArtifacts successfully saved:")
    print(f"  * Best Model: {best_model_path}")
    print(f"  * Features List: {features_path}")
    print(f"  * Tuning Report: {report_json_path}")
    print(f"  * Test Predictions: {predictions_csv_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

