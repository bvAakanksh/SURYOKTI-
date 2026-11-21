"""
SURYOKTI Regional Adaptation — Maui Ocean Center (System 2105)
Multi-Candidate Prediction Benchmarking
==============================================================
Compares prediction performance across all 5 target candidates:
1. Candidate 1: Naive 11-Channel Baseline (Peak ~234.1 kW)
2. Candidate 2: De-duplicated Full Array (9 Channels, Peak ~190.5 kW)
3. Candidate 3: Triple Combiner Group (inv04+09+10, Peak ~129.3 kW)
4. Candidate 4: String Sub-Array (6 Channels, Peak ~62.1 kW)
5. Candidate 5: Primary Twin Combiners (inv04+10, Peak ~86.9 kW)

Models evaluated on untouched chronological 20% holdout test set:
- 1-Hour Persistence Baseline
- Tuned Random Forest Regressor
- Tuned XGBoost Regressor

Metrics: MAE, RMSE, R2, nMAE_peak%, nRMSE_peak%, nMAE_dc%, nRMSE_dc%
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")

RATED_CAPACITY_KW = 110.0
SPLIT_RATIO = 0.80

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

FEATURE_COLS = [
    "GHI", "DNI", "Temperature", "Wind_Speed",
    "Hour", "Day", "Month", "SZ_Angle", "SA_Angle",
    "Hour_sin", "Hour_cos", "Month_sin", "Month_cos",
    "Effective_GHI", "Thermal_Stress"
]

CANDIDATES = [
    {"id": "Candidate_1", "file": "candidate_1_hourly.csv", "name": "Naive 11-Channel (Flawed Baseline)", "desc": "Sums all 11 files with duplicate inv11"},
    {"id": "Candidate_2", "file": "candidate_2_hourly.csv", "name": "De-duplicated Full Array (9 Channels)", "desc": "All distinct channels, site expansion hypothesis (~190 kW AC)"},
    {"id": "Candidate_3", "file": "candidate_3_hourly.csv", "name": "Triple Combiners (inv04+09+10)", "desc": "Three large commercial building inverters (~129 kW AC)"},
    {"id": "Candidate_4", "file": "candidate_4_hourly.csv", "name": "String Sub-Array (6 Channels)", "desc": "Only small string inverters (~62 kW AC, leaving ~35 kW unmonitored)"},
    {"id": "Candidate_5", "file": "candidate_5_hourly.csv", "name": "Twin Combiners (inv04+10, 1.25 ILR)", "desc": "Reconciles with 110 kW DC nameplate via 1.25 DC/AC loading ratio (~87 kW AC)"}
]


def evaluate_candidate(candidate_info):
    cid = candidate_info["id"]
    fname = candidate_info["file"]
    cname = candidate_info["name"]
    
    print("\n" + "=" * 70)
    print(f"BENCHMARKING: {cid} — {cname}")
    print("=" * 70)
    
    csv_path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(csv_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    
    n_total = len(df)
    split_idx = int(n_total * SPLIT_RATIO)
    
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    obs_peak = float(df["AC_Output"].max())
    test_peak = float(test_df["AC_Output"].max())
    
    print(f"Total Rows: {n_total:,} | Dev (80%): {len(train_df):,} rows | Test (20%): {len(test_df):,} rows")
    print(f"Full Dataset Peak: {obs_peak:.2f} kW | Test Set Peak: {test_peak:.2f} kW")
    
    X_train = train_df[FEATURE_COLS]
    y_train = train_df["AC_Output"]
    
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["AC_Output"]
    
    # 1. Persistence Baseline (1-Hour Lagged)
    # y_pred[t] = y[t-1]; for first test point use last dev point
    last_dev_val = train_df["AC_Output"].iloc[-1]
    y_pers = np.empty(len(test_df))
    y_pers[0] = last_dev_val
    y_pers[1:] = test_df["AC_Output"].iloc[:-1].values
    
    # Apply night gate to persistence: if GHI <= 0, generation must be 0
    night_mask_test = test_df["GHI"].values <= 0
    y_pers[night_mask_test] = 0.0
    y_pers = np.clip(y_pers, 0.0, None)
    
    mae_pers = mean_absolute_error(y_test, y_pers)
    rmse_pers = np.sqrt(mean_squared_error(y_test, y_pers))
    r2_pers = r2_score(y_test, y_pers)
    
    print(f"\n[Persistence Baseline] MAE: {mae_pers:.2f} kW | RMSE: {rmse_pers:.2f} kW | R2: {r2_pers:.4f}")
    
    # 2. Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_pred_rf[night_mask_test] = 0.0
    y_pred_rf = np.clip(y_pred_rf, 0.0, None)
    
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"[Random Forest]        MAE: {mae_rf:.2f} kW | RMSE: {rmse_rf:.2f} kW | R2: {r2_rf:.4f}")
    
    # 3. XGBoost Regressor
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_pred_xgb[night_mask_test] = 0.0
    y_pred_xgb = np.clip(y_pred_xgb, 0.0, None)
    
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)
    print(f"[XGBoost Regressor]    MAE: {mae_xgb:.2f} kW | RMSE: {rmse_xgb:.2f} kW | R2: {r2_xgb:.4f}")
    
    # Save models
    rf_path = os.path.join(MODELS_DIR, f"{cid.lower()}_rf_model.joblib")
    xgb_path = os.path.join(MODELS_DIR, f"{cid.lower()}_xgb_model.joblib")
    joblib.dump(rf, rf_path)
    joblib.dump(xgb, xgb_path)
    
    # Calculate metrics helper
    def get_metrics_dict(model_name, mae, rmse, r2):
        return {
            "model": model_name,
            "MAE_kW": float(mae),
            "RMSE_kW": float(rmse),
            "R2": float(r2),
            "nMAE_peak_pct": float(mae / obs_peak * 100.0),
            "nRMSE_peak_pct": float(rmse / obs_peak * 100.0),
            "nMAE_nameplate_110kW_pct": float(mae / RATED_CAPACITY_KW * 100.0),
            "nRMSE_nameplate_110kW_pct": float(rmse / RATED_CAPACITY_KW * 100.0)
        }
        
    cand_results = {
        "candidate_id": cid,
        "candidate_name": cname,
        "description": candidate_info["desc"],
        "observed_peak_kW": obs_peak,
        "test_peak_kW": test_peak,
        "registered_nameplate_kW_DC": RATED_CAPACITY_KW,
        "models": {
            "Persistence": get_metrics_dict("1-Hour Persistence Baseline", mae_pers, rmse_pers, r2_pers),
            "RandomForest": get_metrics_dict("Random Forest Regressor", mae_rf, rmse_rf, r2_rf),
            "XGBoost": get_metrics_dict("XGBoost Regressor", mae_xgb, rmse_xgb, r2_xgb)
        }
    }
    
    return cand_results


def run_all_benchmarks():
    all_benchmarks = []
    
    for cinfo in CANDIDATES:
        res = evaluate_candidate(cinfo)
        all_benchmarks.append(res)
        
    out_json = os.path.join(REPORTS_DIR, "candidate_benchmarks.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_benchmarks, f, indent=4)
        
    print("\n" + "=" * 70)
    print("MASTER SUMMARY TABLE: CANDIDATE TARGET BENCHMARK RESULTS")
    print("=" * 70)
    
    summary_rows = []
    for b in all_benchmarks:
        cid = b["candidate_id"]
        cname = b["candidate_name"]
        peak = b["observed_peak_kW"]
        
        xgb_m = b["models"]["XGBoost"]
        rf_m = b["models"]["RandomForest"]
        pers_m = b["models"]["Persistence"]
        
        summary_rows.append({
            "Candidate": cid,
            "Peak (kW)": f"{peak:.1f}",
            "XGB R2": f"{xgb_m['R2']:.4f}",
            "XGB MAE (kW)": f"{xgb_m['MAE_kW']:.2f}",
            "XGB nMAE (Peak %)": f"{xgb_m['nMAE_peak_pct']:.2f}%",
            "XGB nMAE (110kW %)": f"{xgb_m['nMAE_nameplate_110kW_pct']:.2f}%",
            "RF R2": f"{rf_m['R2']:.4f}",
            "Pers R2": f"{pers_m['R2']:.4f}"
        })
        
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print(f"\nSaved benchmark metrics to: {out_json}")

if __name__ == "__main__":
    run_all_benchmarks()

