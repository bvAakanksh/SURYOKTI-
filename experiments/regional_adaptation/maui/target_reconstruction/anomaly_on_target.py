"""
SURYOKTI Regional Adaptation — Maui Ocean Center (System 2105)
Calibrated Anomaly Detection on Defensible Targets
==============================================================
Runs calibrated anomaly detection pipeline on:
1. Candidate 2: De-duplicated Full Array (9 Channels, Peak 190.5 kW)
2. Candidate 5: Twin Combiners (inv04+10, Peak 86.9 kW, 1.25 ILR)

Key Methodological Guarantees:
- Strict zero-leakage calibration on development split (first 80%).
- Daylight-restricted residual threshold calibration (GHI > 10 W/m²) using MAD.
- Contamination rate derived empirically from development data physical violations.
- Mutually exclusive reconciliation of operational alarms (100% partition).
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

SPLIT_RATIO = 0.80

FEATURE_COLS = [
    "GHI", "DNI", "Temperature", "Wind_Speed",
    "Hour", "Day", "Month", "SZ_Angle", "SA_Angle",
    "Hour_sin", "Hour_cos", "Month_sin", "Month_cos",
    "Effective_GHI", "Thermal_Stress"
]


def run_calibrated_anomaly_detection(candidate_id="Candidate_2", fname="candidate_2_hourly.csv"):
    print("\n" + "=" * 70)
    print(f"ANOMALY DETECTION PIPELINE: {candidate_id}")
    print("=" * 70)
    
    csv_path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(csv_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    
    n_total = len(df)
    split_idx = int(n_total * SPLIT_RATIO)
    
    dev_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    peak_kw = float(df["AC_Output"].max())
    print(f"Total Rows: {n_total:,} | Dev (80%): {len(dev_df):,} | Test (20%): {len(test_df):,}")
    print(f"Candidate Peak AC Output: {peak_kw:.2f} kW")
    
    # 1. Calibrate on Dev Set
    # Physical Rule Checks on Dev
    dev_env_outliers = (
        (dev_df["GHI"] > 1250) |
        (dev_df["Temperature"] > 45) |
        (dev_df["Temperature"] < 0) |
        (dev_df["Wind_Speed"] > 25)
    )
    dev_sensor_err = (
        (dev_df["AC_Output"] < -0.01) |
        ((dev_df["GHI"] <= 0) & (dev_df["AC_Output"] > 1.0))
    )
    dev_zero_gen = (
        (dev_df["GHI"] > 200) &
        (dev_df["AC_Output"] < 2.0)
    )
    
    total_dev_violations = (dev_env_outliers | dev_sensor_err | dev_zero_gen).sum()
    empirical_contamination = float(np.clip(total_dev_violations / len(dev_df), 0.01, 0.08))
    print(f"Empirical Contamination from Dev Rules: {empirical_contamination * 100:.2f}% (Violations: {total_dev_violations:,})")
    
    # Load trained XGBoost model for residuals
    xgb_path = os.path.join(MODELS_DIR, f"{candidate_id.lower()}_xgb_model.joblib")
    xgb = joblib.load(xgb_path)
    
    # Dev Predictions & Daylight Residual Calibration
    dev_preds = xgb.predict(dev_df[FEATURE_COLS])
    dev_preds[dev_df["GHI"] <= 0] = 0.0
    dev_preds = np.clip(dev_preds, 0.0, None)
    
    dev_residuals = dev_preds - dev_df["AC_Output"].values
    
    # Restrict to daylight hours (GHI > 10 W/m²) to prevent 50% night zero bias
    daylight_dev_mask = dev_df["GHI"].values > 10
    daylight_residuals = dev_residuals[daylight_dev_mask]
    
    median_res = float(np.median(daylight_residuals))
    mad_res = float(np.median(np.abs(daylight_residuals - median_res)))
    robust_sigma = mad_res * 1.4826
    residual_threshold = median_res + 2.5 * robust_sigma
    
    print(f"Daylight Residual Calibration (N={daylight_dev_mask.sum():,}):")
    print(f"  * Median Residual: {median_res:.2f} kW")
    print(f"  * MAD: {mad_res:.2f} kW | Robust Sigma: {robust_sigma:.2f} kW")
    print(f"  * 2.5-Sigma Residual Threshold: {residual_threshold:.2f} kW")
    
    # Fit Isolation Forest on Dev
    iso_features = ["GHI", "Temperature", "Wind_Speed", "Effective_GHI", "AC_Output"]
    iso = IsolationForest(
        n_estimators=150,
        contamination=empirical_contamination,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(dev_df[iso_features])
    iso_path = os.path.join(MODELS_DIR, f"{candidate_id.lower()}_iso_forest.joblib")
    joblib.dump(iso, iso_path)
    
    # 2. Evaluate on Untouched Holdout Test Set
    test_preds = xgb.predict(test_df[FEATURE_COLS])
    test_preds[test_df["GHI"] <= 0] = 0.0
    test_preds = np.clip(test_preds, 0.0, None)
    
    test_residuals = test_preds - test_df["AC_Output"].values
    test_df["Residual"] = test_residuals
    test_df["y_pred"] = test_preds
    
    # Anomaly Flags on Test
    test_env_outlier = (
        (test_df["GHI"] > 1250) |
        (test_df["Temperature"] > 45) |
        (test_df["Temperature"] < 0) |
        (test_df["Wind_Speed"] > 25)
    ).values
    
    test_sensor_err = (
        (test_df["AC_Output"] < -0.01) |
        ((test_df["GHI"] <= 0) & (test_df["AC_Output"] > 1.0))
    ).values
    
    test_zero_gen = (
        (test_df["GHI"] > 200) &
        (test_df["AC_Output"] < 2.0)
    ).values
    
    test_iso_anomaly = (iso.predict(test_df[iso_features]) == -1)
    
    # Daylight residual underperformance
    test_residual_alarm = (
        (test_df["GHI"].values > 10) &
        (test_residuals > residual_threshold)
    )
    
    # Mutually Exclusive Reconciliation Hierarchy:
    # Priority 1: Data Quality / Sensor Error
    # Priority 2: Potential Operational Anomaly (Daylight Zero/Near-Zero Generation)
    # Priority 3: Extreme Environmental Outlier
    # Priority 4: Statistical Residual Exceedance (Model Underperformance)
    # Priority 5: Normal Generation
    
    reconciled_status = np.full(len(test_df), "Normal Generation", dtype=object)
    
    # Statistical Residual Exceedance (Priority 4)
    reconciled_status[test_residual_alarm] = "Statistical Residual Exceedance"
    # Environmental Outlier (Priority 3)
    reconciled_status[test_env_outlier] = "Extreme Environmental Outlier"
    # Potential Operational Anomaly (Priority 2)
    reconciled_status[test_zero_gen] = "Potential Operational Anomaly"
    # Data Quality / Sensor Error (Priority 1)
    reconciled_status[test_sensor_err] = "Data Quality / Sensor Error"
    
    test_df["Anomaly_Category"] = reconciled_status
    
    # Output breakdown
    category_counts = pd.Series(reconciled_status).value_counts().to_dict()
    total_test_records = len(test_df)
    
    print("\n" + "-" * 50)
    print("RECONCILED OPERATIONAL ANOMALY CLASSIFICATION (TEST SET)")
    print("-" * 50)
    for cat, cnt in category_counts.items():
        pct = cnt / total_test_records * 100.0
        print(f"  * {cat:<34}: {cnt:5,} ({pct:5.2f}%)")
    print("-" * 50)
    print(f"  Total Test Records: {total_test_records:,} (100.00% Verified)")
    
    # Save Report
    report_dict = {
        "candidate_id": candidate_id,
        "telemetry_file": fname,
        "peak_capacity_kW": peak_kw,
        "ground_truth_status": "UNLABELED / NO MAINTENANCE GROUND TRUTH",
        "ground_truth_disclaimer": (
            "NREL PVDAQ System 2105 telemetry lacks operator maintenance logs, work orders, "
            "or labeled fault logs. True physical Precision, Recall, False Alarm Rate, and "
            "fault-detection accuracy CANNOT be calculated. All alarms represent unconfirmed "
            "statistical deviations or heuristic rule flags."
        ),
        "environmental_novelty_definition": (
            "Environmental novelty is defined as multivariate out-of-distribution density "
            "identified by Isolation Forest across (GHI, Temperature, Wind_Speed, Effective_GHI). "
            "Disagreements between detectors represent statistical feature novelty, NOT "
            "independently validated severe meteorological events."
        ),
        "statistical_residual_vs_physical_fault": (
            "Statistical residual exceedances mark timestamps where actual output falls >2.5 sigma "
            "below the XGBoost expectation. While consistent with operational degradation, soiling, "
            "grid curtailment, or sub-string trips, they remain unconfirmed without field verification."
        ),
        "calibration_parameters": {
            "dev_records": len(dev_df),
            "empirical_contamination_pct": empirical_contamination * 100.0,
            "daylight_records_calibrated": int(daylight_dev_mask.sum()),
            "daylight_median_residual_kW": median_res,
            "daylight_MAD_kW": mad_res,
            "daylight_robust_sigma_kW": robust_sigma,
            "daylight_residual_threshold_kW": residual_threshold
        },
        "test_evaluation": {
            "test_records": total_test_records,
            "test_date_start": str(test_df["Timestamp"].min()),
            "test_date_end": str(test_df["Timestamp"].max()),
            "raw_detector_triggers": {
                "environmental_outliers": int(test_env_outlier.sum()),
                "sensor_data_quality_errors": int(test_sensor_err.sum()),
                "potential_operational_anomalies": int(test_zero_gen.sum()),
                "isolation_forest_triggers": int(test_iso_anomaly.sum()),
                "statistical_residual_exceedances": int(test_residual_alarm.sum())
            },
            "reconciled_mutually_exclusive_classification": {
                cat: {"count": int(cnt), "percentage": float(cnt / total_test_records * 100.0)}
                for cat, cnt in category_counts.items()
            }
        }
    }
    
    out_json = os.path.join(REPORTS_DIR, f"{candidate_id.lower()}_anomaly_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)
    print(f"Saved anomaly report to: {out_json}")
    
    # Plot Anomaly Timeline
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    
    sample_sub = test_df.iloc[1000:1500].copy()
    ax1.plot(sample_sub["Timestamp"], sample_sub["AC_Output"], label="Actual AC Generation", color="#2c3e50", linewidth=1.5)
    ax1.plot(sample_sub["Timestamp"], sample_sub["y_pred"], label="XGBoost Predicted AC Generation", color="#2980b9", linestyle="--", alpha=0.8)
    
    # Overlay anomalies
    fault_mask = sample_sub["Anomaly_Category"] == "Potential Operational Anomaly"
    res_mask = sample_sub["Anomaly_Category"] == "Statistical Residual Exceedance"
    
    if fault_mask.sum() > 0:
        ax1.scatter(sample_sub.loc[fault_mask, "Timestamp"], sample_sub.loc[fault_mask, "AC_Output"], color="#e74c3c", s=50, label="Potential Operational Anomaly (Daylight Under-Gen)", zorder=5)
    if res_mask.sum() > 0:
        ax1.scatter(sample_sub.loc[res_mask, "Timestamp"], sample_sub.loc[res_mask, "AC_Output"], color="#f39c12", s=30, label="Statistical Residual Exceedance", zorder=4)
        
    ax1.set_ylabel("AC Power Output (kW)", fontsize=11, fontweight='bold')
    ax1.set_title(f"SURYOKTI Calibrated Operational Anomaly Detection -- {candidate_id} Holdout Sample", fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)
    
    ax2.plot(sample_sub["Timestamp"], sample_sub["Residual"], label="Daylight Residual (kW)", color="#8e44ad", linewidth=1.2)
    ax2.axhline(residual_threshold, color="#e74c3c", linestyle="--", label=f"2.5σ Threshold ({residual_threshold:.1f} kW)")
    ax2.axhline(0, color="black", linestyle=":", alpha=0.5)
    ax2.set_ylabel("Residual (kW)", fontsize=10, fontweight='bold')
    ax2.set_xlabel("Timestamp (HST)", fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)
    
    fig.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"{candidate_id.lower()}_anomaly_timeline.png")
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved anomaly plot to: {plot_path}")
    
    return report_dict

if __name__ == "__main__":
    # Run on Candidate 2 (De-duplicated Full Array)
    run_calibrated_anomaly_detection("Candidate_2", "candidate_2_hourly.csv")
    # Run on Candidate 5 (Twin Combiners)
    run_calibrated_anomaly_detection("Candidate_5", "candidate_5_hourly.csv")

