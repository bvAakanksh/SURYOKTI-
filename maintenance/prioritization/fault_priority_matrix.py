"""
SURYOKTI: Maintenance Prioritization Matrix
Ranks detected solar PV faults by financial loss, hardware risk, and persistence.
"""

import pandas as pd
import numpy as np

TARIFF_PER_KWH = 0.12  # Standard commercial solar feed-in tariff ($/kWh)
AVERAGE_SUN_HOURS = 5.5 # Average effective daily peak sun hours

SEVERITY_WEIGHTS = {
    "Physical-Damage": 1.0,
    "Electrical-damage": 0.9,
    "Bird-drop": 0.7,
    "Snow-Covered": 0.5,
    "Dusty": 0.3,
    "Sensor-Drift": 0.1,
    "Clean": 0.0
}

def compute_priority(incidents_df: pd.DataFrame) -> pd.DataFrame:
    df = incidents_df.copy()
    
    # 1. Financial daily loss ($/day)
    df["Daily_Energy_Loss_kWh"] = df["Power_Deficit_kW"] * AVERAGE_SUN_HOURS
    df["Daily_Revenue_Loss_USD"] = df["Daily_Energy_Loss_kWh"] * TARIFF_PER_KWH
    
    # 2. Risk score from fault classification
    df["Risk_Score"] = df["Fault_Type"].map(SEVERITY_WEIGHTS).fillna(0.2)
    
    # 3. Composite Priority Score (0 to 100)
    # Normalize revenue loss (capping at $150/day max scale)
    loss_norm = np.clip(df["Daily_Revenue_Loss_USD"] / 150.0, 0, 1.0)
    persist_norm = np.clip(df["Persistence_Hours"] / 48.0, 0, 1.0)
    
    df["Priority_Score"] = (
        0.45 * loss_norm +
        0.35 * df["Risk_Score"] +
        0.20 * persist_norm
    ) * 100.0
    
    # Assign Action Levels
    def assign_level(score):
        if score >= 75:
            return "P1 - CRITICAL (Dispatch < 4h)"
        elif score >= 50:
            return "P2 - HIGH (Dispatch < 24h)"
        elif score >= 25:
            return "P3 - MEDIUM (Next Scheduled Service)"
        else:
            return "P4 - LOW (Routine Observation)"
            
    df["Action_Level"] = df["Priority_Score"].apply(assign_level)
    return df.sort_values("Priority_Score", ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    sample_incidents = pd.DataFrame([
        {"Inverter_ID": "INV_04", "Fault_Type": "Electrical-damage", "Power_Deficit_kW": 38.5, "Persistence_Hours": 6},
        {"Inverter_ID": "INV_02", "Fault_Type": "Physical-Damage", "Power_Deficit_kW": 12.0, "Persistence_Hours": 24},
        {"Inverter_ID": "INV_07", "Fault_Type": "Bird-drop", "Power_Deficit_kW": 8.5, "Persistence_Hours": 72},
        {"Inverter_ID": "INV_01", "Fault_Type": "Dusty", "Power_Deficit_kW": 15.0, "Persistence_Hours": 120},
        {"Inverter_ID": "INV_09", "Fault_Type": "Sensor-Drift", "Power_Deficit_kW": 4.0, "Persistence_Hours": 12},
    ])
    
    print("=== SURYOKTI Maintenance Fault Priority Matrix ===")
    ranked = compute_priority(sample_incidents)
    cols = ["Inverter_ID", "Fault_Type", "Power_Deficit_kW", "Daily_Revenue_Loss_USD", "Priority_Score", "Action_Level"]
    print(ranked[cols].to_string(index=False))

