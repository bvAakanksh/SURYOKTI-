"""
SURYOKTI: Actionable Recommendation Engine
Maps anomalies, deficit values, and vision classes into formal O&M work orders.
"""

from datetime import date, datetime

import pandas as pd

ACTION_CATALOG = {
    "Physical-Damage": {
        "Action_Code": "ACT_REP_MOD",
        "Action_Name": "Module Replacement & Safety Isolation",
        "Equipment_Needed": ["Replacement PV Module", "Insulated Toolset", "Multimeter"],
        "Estimated_Hours": 2.5
    },
    "Electrical-damage": {
        "Action_Code": "ACT_ELEC_INSP",
        "Action_Name": "String Diode & Inverter Terminal Audit",
        "Equipment_Needed": ["Thermal IR Camera", "I-V Curve Tracer", "Safety PPE"],
        "Estimated_Hours": 3.0
    },
    "Bird-drop": {
        "Action_Code": "ACT_SPOT_WASH",
        "Action_Name": "Targeted Panel Decontamination",
        "Equipment_Needed": ["Deionized Water Wand", "Soft Sponge", "Safety Harness"],
        "Estimated_Hours": 0.5
    },
    "Dusty": {
        "Action_Code": "ACT_ARRAY_CLEAN",
        "Action_Name": "Automated / Manual Array Soiling Wash",
        "Equipment_Needed": ["Water Truck / Sprinkler Grid", "Squeegee Robot"],
        "Estimated_Hours": 4.0
    },
    "Sensor-Drift": {
        "Action_Code": "ACT_CALIB_SENS",
        "Action_Name": "Pyranometer & RTD Sensor Recalibration",
        "Equipment_Needed": ["Reference Secondary Pyranometer", "Calibration Rig"],
        "Estimated_Hours": 1.5
    },
    "Unclassified": {
        "Action_Code": "ACT_FIELD_VERIFY",
        "Action_Name": "Field Inspection and Evidence Collection",
        "Equipment_Needed": ["Thermal IR Camera", "I-V Curve Tracer", "Safety PPE"],
        "Estimated_Hours": 2.0
    }
}

def generate_work_orders(
    incidents_df: pd.DataFrame,
    work_order_date: date | datetime | None = None,
) -> list:
    """Create work orders without asserting an unverified physical fault type."""
    if work_order_date is None:
        work_order_date = date.today()
    if isinstance(work_order_date, datetime):
        work_order_date = work_order_date.date()
    date_code = work_order_date.strftime("%Y%m%d")

    work_orders = []
    for idx, row in incidents_df.iterrows():
        fault = row.get("Fault_Type", "Sensor-Drift")
        catalog_entry = ACTION_CATALOG.get(fault, ACTION_CATALOG["Sensor-Drift"])
        
        wo = {
            "Work_Order_ID": f"WO-{date_code}-{idx+1:04d}",
            "Asset": row.get("Inverter_ID", "ARRAY_GENERIC"),
            "Fault_Detected": fault,
            "Deficit_kW": row.get("Power_Deficit_kW", 0.0),
            "Action_Code": catalog_entry["Action_Code"],
            "Recommended_Procedure": catalog_entry["Action_Name"],
            "Required_Tools": ", ".join(catalog_entry["Equipment_Needed"]),
            "Est_Duration_Hours": catalog_entry["Estimated_Hours"]
        }
        work_orders.append(wo)
    return work_orders

if __name__ == "__main__":
    sample = pd.DataFrame([
        {"Inverter_ID": "INV_04", "Fault_Type": "Electrical-damage", "Power_Deficit_kW": 38.5},
        {"Inverter_ID": "INV_01", "Fault_Type": "Dusty", "Power_Deficit_kW": 15.0},
        {"Inverter_ID": "INV_07", "Fault_Type": "Bird-drop", "Power_Deficit_kW": 8.5},
    ])
    wos = generate_work_orders(sample)
    df_wo = pd.DataFrame(wos)
    print("=== SURYOKTI O&M Actionable Work Orders ===")
    print(df_wo[["Work_Order_ID", "Asset", "Fault_Detected", "Action_Code", "Recommended_Procedure"]].to_string(index=False))

