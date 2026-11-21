"""
SURYOKTI: Master End-to-End Plant Orchestration Demo
Executes the runnable pipeline:
Telemetry -> Forecast -> Daylight residual screening -> Prioritized dispatch.

Vision-based classification is deliberately not fabricated here. A future CNN
integration can supply a verified ``Fault_Type`` before maintenance scoring.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is importable
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from energy_prediction.inference.predict_energy import load_test_data, predict_ac_power
from integration.prediction_anomaly_integration.integrated_pipeline import run_canonical_detection
from maintenance.prioritization.fault_priority_matrix import compute_priority
from maintenance.recommendation_logic.actionable_recommendations import generate_work_orders

def run_end_to_end():
    print("=" * 70)
    print("SURYOKTI: END-TO-END SOLAR PV INTELLIGENCE PIPELINE")
    print("=" * 70)
    
    # 1. Telemetry Ingestion
    print("\n[STEP 1] Ingesting solar weather & generation telemetry...")
    df = load_test_data(test_ratio=0.4, random_state=None)
    print(f"Ingested {len(df)} telemetry samples.")
    
    # 2. Energy Prediction
    print("\n[STEP 2] Running physics-constrained PV AC power forecasting...")
    df["Predicted_AC_kW"] = predict_ac_power(df, verbose=False)
    print(f"Mean Forecasted Power : {df['Predicted_AC_kW'].mean():.2f} kW")
    print(f"Mean Actual Power     : {df['AC_Output'].mean():.2f} kW")
    
    # 3. Canonical anomaly assessment on a chronological holdout period.
    print("\n[STEP 3] Running canonical daylight anomaly assessment and event aggregation...")
    split_idx = int(len(df) * 0.70)
    scored, events = run_canonical_detection(
        df.iloc[:split_idx],
        df.iloc[split_idx:],
        prediction_column="Predicted_AC_kW",
    )
    print("Holdout status counts:")
    print(scored["Anomaly_Status"].value_counts().to_string())
    print(f"Aggregated events             : {len(events)}")

    dispatchable = events[
        events["Anomaly_Status"].isin(
            ["POTENTIAL_OPERATIONAL_ANOMALY", "STATISTICAL_RESIDUAL_EXCEEDANCE"]
        )
    ].copy()
    if dispatchable.empty:
        print("Plant operating nominally; no anomalies requiring dispatch.")
        return
        
    # 4. Preserve uncertainty until an independent inspection supplies a fault type.
    print("\n[STEP 4] Preparing top canonical events for field verification...")
    top_events = dispatchable.nlargest(3, "Peak_Deficit_kW").copy().reset_index(drop=True)
    top_events["Fault_Type"] = "Unclassified"
    top_events["Inverter_ID"] = "ARRAY_GENERIC"
    top_events["Power_Deficit_kW"] = top_events["Peak_Deficit_kW"]
    top_events["Persistence_Hours"] = top_events["Duration_Hours"].clip(lower=1.0)
    
    # 5. Maintenance Prioritization
    print("\n[STEP 5] Computing Maintenance Priority Index & Financial Loss...")
    ranked = compute_priority(top_events)
    cols = ["Inverter_ID", "Fault_Type", "Power_Deficit_kW", "Daily_Revenue_Loss_USD", "Priority_Score", "Action_Level"]
    print(ranked[cols].to_string(index=False))
    
    # 6. Actionable Work Order Generation
    print("\n[STEP 6] Generating Dispatch Work Orders for O&M Field Technicians...")
    work_orders = generate_work_orders(ranked)
    df_wo = pd.DataFrame(work_orders)
    wo_cols = ["Work_Order_ID", "Asset", "Fault_Detected", "Action_Code", "Recommended_Procedure", "Required_Tools"]
    print(df_wo[wo_cols].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE: All modules synchronized successfully.")
    print("=" * 70)

if __name__ == "__main__":
    run_end_to_end()

