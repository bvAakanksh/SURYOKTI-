"""
SURYOKTI: Maintenance Scheduling Engine
Schedules panel cleaning, technician visits, and inspections considering weather forecasts.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def optimize_schedule(
    ranked_incidents: pd.DataFrame,
    weather_forecast_rain_days: list,
    reference_time: datetime | None = None,
) -> pd.DataFrame:
    """
    Optimizes schedule considering upcoming weather.
    If rain is predicted within 48h, routine cleaning is postponed to conserve water and cost.
    Physical and electrical repairs proceed immediately regardless of rain.
    """
    schedule = []
    base_time = reference_time or datetime.now().astimezone()
    
    for _, row in ranked_incidents.iterrows():
        inv = row["Inverter_ID"]
        fault = row["Fault_Type"]
        level = row["Action_Level"]
        
        if "CRITICAL" in level:
            sched_time = base_time + timedelta(hours=4)
            note = "Immediate electrical/safety dispatch required."
        elif "HIGH" in level:
            sched_time = base_time + timedelta(hours=20)
            note = "Inspect for localized reverse-bias hotspots."
        elif fault == "Dusty":
            if any(day in [0, 1, 2] for day in weather_forecast_rain_days):
                sched_time = base_time + timedelta(days=4)
                note = "Rain forecasted in next 48h. Panel wash postponed to save cleaning costs."
            else:
                sched_time = base_time + timedelta(days=2)
                note = "Automated wash or field team cleaning scheduled."
        else:
            sched_time = base_time + timedelta(days=7)
            note = "Batch with routine monthly plant maintenance cycle."
            
        schedule.append({
            "Inverter_ID": inv,
            "Fault_Type": fault,
            "Scheduled_Time": sched_time.strftime("%Y-%m-%d %H:%M"),
            "Dispatcher_Note": note
        })
        
    return pd.DataFrame(schedule)

if __name__ == "__main__":
    from maintenance.prioritization.fault_priority_matrix import compute_priority
    sample_incidents = pd.DataFrame([
        {"Inverter_ID": "INV_04", "Fault_Type": "Electrical-damage", "Power_Deficit_kW": 38.5, "Persistence_Hours": 6},
        {"Inverter_ID": "INV_01", "Fault_Type": "Dusty", "Power_Deficit_kW": 15.0, "Persistence_Hours": 120},
        {"Inverter_ID": "INV_07", "Fault_Type": "Bird-drop", "Power_Deficit_kW": 8.5, "Persistence_Hours": 72},
    ])
    ranked = compute_priority(sample_incidents)
    # Simulate rain forecast on day 1 (tomorrow)
    scheduled = optimize_schedule(ranked, weather_forecast_rain_days=[1])
    print("=== SURYOKTI Maintenance Work Order Schedule ===")
    print(scheduled.to_string(index=False))

