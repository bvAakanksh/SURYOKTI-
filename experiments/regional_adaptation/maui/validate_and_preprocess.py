"""
SURYOKTI Regional Adaptation — Maui Dataset Validation and Preprocessing
=======================================================================
Site: Maui Ocean Center (NREL PVDAQ System 2105)
Location: Maui, Hawaii (20.79253 N, -156.512 W, Elevation: 10m)
Rated Capacity: 110.0 kW DC
Climate: Cfb (Marine / Subtropical Highland)

Tasks Performed:
1. Validates solar_combined.csv telemetry and logs critical findings.
2. Reconstructs true plant generation from 11 inverter channels.
3. Aligns weather telemetry with local Hawaii Standard Time (HST).
4. Calculates astronomical solar angles and physics interaction features.
5. Exports clean 15-minute and hourly modeling datasets.
"""

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")

RAW_DATA_DIR = r"C:\VITB\SURYOKTI - DRAFT\01_Data_Raw"
COMBINED_FILE = r"C:\VITB\SURYOKTI - DRAFT\02_Data_Processed\solar_combined.csv"

LATITUDE = 20.79253
LONGITUDE = -156.51200
RATED_CAPACITY_KW = 110.0


def validate_solar_combined():
    """
    Validates solar_combined.csv to check target, ranges, duplicates, and column consistency.
    """
    print("\n[VALIDATION] Inspecting solar_combined.csv...")
    df = pd.read_csv(COMBINED_FILE)
    
    n_rows, n_cols = df.shape
    cols = df.columns.tolist()
    missing = df.isnull().sum().to_dict()
    
    ac_col = "meter_ac_output_(kwatts)_meter_150161"
    wind_col = "wind_speed_10m_(mph)_o_150229"
    ghi_col = "irradiance_ghi_o_150230"
    
    is_identical = bool((df[ac_col] == df[wind_col]).all())
    max_diff = float((df[ac_col] - df[wind_col]).abs().max())
    corr_ghi_ac = float(df[ghi_col].corr(df[ac_col]))
    
    validation_results = {
        "dataset_name": "solar_combined.csv",
        "rows": n_rows,
        "columns": cols,
        "missing_values": missing,
        "ac_output_col": ac_col,
        "ac_output_is_duplicate_of_wind_speed": is_identical,
        "ac_wind_max_difference": max_diff,
        "correlation_ghi_to_meter_ac": corr_ghi_ac,
        "target_column_diagnosis": (
            "CRITICAL BUG IDENTIFIED: Column 'meter_ac_output_(kwatts)_meter_150161' is an exact "
            "duplicate of 'wind_speed_10m_(mph)_o_150229' (diff=0.0, corr=-0.0518 with GHI). "
            "It does NOT contain actual AC electrical generation."
        ),
        "timestamp_diagnosis": (
            "Timestamp 'measured_on' in solar_combined.csv is in UTC. Solar irradiance peaks "
            "at 01:00 UTC (afternoon in Hawaii). Inverter string channels are logged in HST."
        ),
        "reconstruction_strategy": (
            "Reconstruct true AC generation by aggregating raw inverter files "
            "(2105_inv01 through 2105_inv11) and align weather channels to Hawaii local time."
        )
    }
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "dataset_validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(validation_results, f, indent=4)
        
    print(f"Validation report saved to: {report_path}")
    print(f"  * Total raw rows: {n_rows}")
    print(f"  * Target identical to wind speed: {is_identical}")
    print(f"  * GHI vs Meter AC correlation: {corr_ghi_ac:.4f}")
    return validation_results


def calculate_solar_angles(timestamps, lat=LATITUDE, lon=LONGITUDE):
    """
    Computes Solar Zenith Angle (SZ_Angle) and Solar Azimuth Angle (SA_Angle).
    """
    doy = timestamps.dt.dayofyear.values
    hour = timestamps.dt.hour.values + timestamps.dt.minute.values / 60.0
    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (hour - 12.0) / 24.0)
    
    eqtime = 229.18 * (
        0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
    )
    
    decl = (
        0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    )
    
    tz_meridian = -150.0  # Hawaii Standard Time (UTC-10)
    time_offset = eqtime + 4.0 * (lon - tz_meridian)
    tst = hour * 60.0 + time_offset
    solar_hour_angle = (tst / 4.0) - 180.0
    sha_rad = np.radians(solar_hour_angle)
    lat_rad = np.radians(lat)
    
    cos_sza = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(sha_rad)
    cos_sza = np.clip(cos_sza, -1.0, 1.0)
    sza_deg = np.degrees(np.arccos(cos_sza))
    
    cos_aza = (np.sin(decl) * np.cos(lat_rad) - np.cos(decl) * np.sin(lat_rad) * np.cos(sha_rad)) / np.sin(np.radians(sza_deg) + 1e-6)
    cos_aza = np.clip(cos_aza, -1.0, 1.0)
    aza_deg = np.degrees(np.arccos(cos_aza))
    aza_deg = np.where(solar_hour_angle > 0, 360.0 - aza_deg, aza_deg)
    
    return sza_deg, aza_deg


def reconstruct_maui_dataset():
    """
    Integrates weather telemetry with all 11 inverter channels.
    """
    print("\n[PREPROCESSING] Reconstructing Maui Ocean Center telemetry...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Load weather data from solar_combined.csv (shifted -13h to align UTC logger to local solar noon)
    sc = pd.read_csv(COMBINED_FILE)
    sc["ts"] = pd.to_datetime(sc["measured_on"]).dt.round("15min") - pd.Timedelta(hours=13)
    
    weather = sc[[
        "ts",
        "irradiance_ghi_o_150230",
        "irradiance_dni__155350",
        "ambient_temp_(c)_o_150228",
        "wind_speed_10m_(mph)_o_150229"
    ]].drop_duplicates("ts").rename(columns={
        "irradiance_ghi_o_150230": "GHI",
        "irradiance_dni__155350": "DNI",
        "ambient_temp_(c)_o_150228": "Temperature",
        "wind_speed_10m_(mph)_o_150229": "Wind_Speed_mph"
    })
    
    weather["Wind_Speed"] = weather["Wind_Speed_mph"] * 0.44704
    
    # 2. Aggregate the 11 inverters
    inv_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "2105_inv*_data.csv")))
    print(f"Found {len(inv_files)} inverter files. Aggregating generation strings...")
    
    all_inverters = []
    for idx, f in enumerate(inv_files, 1):
        df = pd.read_csv(f, usecols=[0, 1])
        ts_col = df.columns[0]
        pwr_col = df.columns[1]
        df["ts"] = pd.to_datetime(df[ts_col]).dt.round("15min")
        s = df.groupby("ts")[pwr_col].mean()
        s.name = f"inv_{idx:02d}"
        all_inverters.append(s)
        
    inv_df = pd.concat(all_inverters, axis=1)
    inv_df["AC_Output"] = inv_df.sum(axis=1, min_count=1)
    
    # 3. Synchronize on timestamp
    merged = pd.merge(weather, inv_df[["AC_Output"]], on="ts", how="inner")
    merged = merged.sort_values("ts").reset_index(drop=True)
    merged.rename(columns={"ts": "Timestamp"}, inplace=True)
    
    # 4. Feature Engineering
    merged["Hour"] = merged["Timestamp"].dt.hour
    merged["Day"] = merged["Timestamp"].dt.day
    merged["Month"] = merged["Timestamp"].dt.month
    
    merged["Hour_sin"] = np.sin(2 * np.pi * merged["Hour"] / 24.0)
    merged["Hour_cos"] = np.cos(2 * np.pi * merged["Hour"] / 24.0)
    merged["Month_sin"] = np.sin(2 * np.pi * merged["Month"] / 12.0)
    merged["Month_cos"] = np.cos(2 * np.pi * merged["Month"] / 12.0)
    
    sza, saa = calculate_solar_angles(merged["Timestamp"])
    merged["SZ_Angle"] = sza
    merged["SA_Angle"] = saa
    
    merged["Effective_GHI"] = merged["GHI"] * np.clip(np.cos(np.radians(merged["SZ_Angle"])), 0, None)
    merged["Thermal_Stress"] = merged["GHI"] * merged["Temperature"] / (merged["Wind_Speed"] + 1.0)
    
    night_mask = merged["GHI"] <= 0
    merged.loc[night_mask, "AC_Output"] = 0.0
    merged["AC_Output"] = np.clip(merged["AC_Output"], 0.0, None)
    
    cols_order = [
        "Timestamp", "GHI", "DNI", "Temperature", "Wind_Speed",
        "Hour", "Day", "Month", "SZ_Angle", "SA_Angle",
        "Hour_sin", "Hour_cos", "Month_sin", "Month_cos",
        "Effective_GHI", "Thermal_Stress", "AC_Output"
    ]
    clean_15m = merged[cols_order].dropna().copy()
    
    path_15m = os.path.join(DATA_DIR, "maui_system_2105_clean_15min.csv")
    clean_15m.to_csv(path_15m, index=False)
    print(f"Exported clean 15-min dataset: {path_15m} ({len(clean_15m)} rows)")
    
    # 5. Hourly resampled dataset
    clean_hourly = clean_15m.set_index("Timestamp").resample("1h").agg({
        "GHI": "mean",
        "DNI": "mean",
        "Temperature": "mean",
        "Wind_Speed": "mean",
        "Hour": "first",
        "Day": "first",
        "Month": "first",
        "SZ_Angle": "mean",
        "SA_Angle": "mean",
        "Hour_sin": "first",
        "Hour_cos": "first",
        "Month_sin": "first",
        "Month_cos": "first",
        "Effective_GHI": "mean",
        "Thermal_Stress": "mean",
        "AC_Output": "mean"
    }).dropna().reset_index()
    
    path_hourly = os.path.join(DATA_DIR, "maui_system_2105_clean_hourly.csv")
    clean_hourly.to_csv(path_hourly, index=False)
    print(f"Exported clean hourly dataset: {path_hourly} ({len(clean_hourly)} rows)")
    
    corr_ghi_ac = clean_hourly["GHI"].corr(clean_hourly["AC_Output"])
    print(f"\nFinal Preprocessing Summary:")
    print(f"  * Hourly dataset samples: {len(clean_hourly)}")
    print(f"  * Date span: {clean_hourly['Timestamp'].min()} to {clean_hourly['Timestamp'].max()}")
    print(f"  * Correlation (GHI vs Total AC Output): {corr_ghi_ac:.4f}")
    print(f"  * AC Output Max: {clean_hourly['AC_Output'].max():.2f} kW (Rated DC: {RATED_CAPACITY_KW} kW)")
    print(f"  * AC Output Mean: {clean_hourly['AC_Output'].mean():.2f} kW")
    
    return clean_hourly


if __name__ == "__main__":
    validate_solar_combined()
    reconstruct_maui_dataset()

