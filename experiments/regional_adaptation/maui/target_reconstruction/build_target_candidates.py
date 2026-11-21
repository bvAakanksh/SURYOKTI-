"""
SURYOKTI Regional Adaptation — Maui Ocean Center (System 2105)
Target Reconstruction & Candidate Formulation
==============================================================
Site: Maui Ocean Center (NREL PVDAQ System 2105)
Registered Rating: 110.0 kW DC (SunPower SPR-X22-360, mono-Si)

Tasks:
1. Reconstructs all 5 target candidates with documented aggregation logic.
2. Synchronizes with local Hawaii Standard Time (HST) aligned weather telemetry.
3. Computes astronomical solar features and physical interaction terms.
4. Exports hourly candidate datasets for rigorous comparative benchmarking.
5. Generates diagnostic time-series comparison plots across seasons and sky conditions.
"""

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")

RAW_DATA_DIR = r"C:\VITB\SURYOKTI - DRAFT\01_Data_Raw"
COMBINED_FILE = r"C:\VITB\SURYOKTI - DRAFT\02_Data_Processed\solar_combined.csv"

LATITUDE = 20.79253
LONGITUDE = -156.51200
RATED_CAPACITY_KW = 110.0

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


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


def build_candidates():
    print("=" * 70)
    print("STEP 1: LOADING & SYNCHRONIZING TELEMETRY")
    print("=" * 70)
    
    # 1. Weather Telemetry (shifted -13h to align UTC logger to local solar noon)
    print("Loading weather telemetry from solar_combined.csv (HST shift: -13h)...")
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
    
    # 2. Inverter Telemetry (All 11 raw files)
    print("Loading and aggregating all 11 inverter channels to 15-min grid...")
    inv_series = {}
    for idx in range(1, 12):
        fpath = os.path.join(RAW_DATA_DIR, f"2105_inv{idx:02d}_data.csv")
        df = pd.read_csv(fpath, usecols=[0, 1])
        ts_col = df.columns[0]
        pwr_col = df.columns[1]
        df["ts"] = pd.to_datetime(df[ts_col]).dt.round("15min")
        s = df.groupby("ts")[pwr_col].mean()
        s.name = f"inv_{idx:02d}"
        inv_series[f"inv_{idx:02d}"] = s
        
    inv_df = pd.DataFrame(inv_series)
    print(f"Inverter dataframe 15-min shape: {inv_df.shape}")
    
    # 3. Formulate Candidate Targets
    print("\n" + "=" * 70)
    print("STEP 2: FORMULATING ALTERNATIVE TARGET CANDIDATES")
    print("=" * 70)
    
    # Candidate 1: Naive 11-Channel Sum (Pre-Audit Reference)
    inv_df["target_naive_11"] = inv_df[[f"inv_{i:02d}" for i in range(1, 12)]].sum(axis=1, min_count=1)
    
    # Candidate 2: De-duplicated Full Array (9 Distinct Channels)
    # Omits duplicate inv_11 and idle inv_06
    cols_c2 = ["inv_01", "inv_02", "inv_03", "inv_04", "inv_05", "inv_07", "inv_08", "inv_09", "inv_10"]
    inv_df["target_dedup_9"] = inv_df[cols_c2].sum(axis=1, min_count=1)
    
    # Candidate 3: Triple Combiner Group (3 Large Inverters)
    cols_c3 = ["inv_04", "inv_09", "inv_10"]
    inv_df["target_combiners_3"] = inv_df[cols_c3].sum(axis=1, min_count=1)
    
    # Candidate 4: String Inverter Sub-Array (6 Small/Medium Inverters)
    cols_c4 = ["inv_01", "inv_02", "inv_03", "inv_05", "inv_07", "inv_08"]
    inv_df["target_strings_6"] = inv_df[cols_c4].sum(axis=1, min_count=1)
    
    # Candidate 5: Primary Twin Combiners (inv_04 + inv_10)
    # Peaks at ~88 kW AC, matching 110 kW DC nameplate with standard 1.25 DC/AC loading ratio
    cols_c5 = ["inv_04", "inv_10"]
    inv_df["target_pair_04_10"] = inv_df[cols_c5].sum(axis=1, min_count=1)
    
    # Print candidate summary stats
    candidate_meta = [
        {
            "candidate_id": "Candidate_1",
            "name": "Naive 11-Channel Sum",
            "column": "target_naive_11",
            "channels_included": [f"inv_{i:02d}" for i in range(1, 12)],
            "channels_excluded": [],
            "hypothesis": "Pre-audit baseline. Sums all 11 raw files indiscriminately.",
            "plausibility": "FLAWED. Double-counts inv_04/inv_11 (+44.1 kW duplicate).",
            "peak_15m_kW": float(inv_df["target_naive_11"].max()),
            "mean_15m_kW": float(inv_df["target_naive_11"].mean())
        },
        {
            "candidate_id": "Candidate_2",
            "name": "De-duplicated Full Array (9 Distinct Channels)",
            "column": "target_dedup_9",
            "channels_included": cols_c2,
            "channels_excluded": ["inv_06 (idle 0.13 kW)", "inv_11 (identical clone of inv_04)"],
            "hypothesis": "Full site generation assuming all 9 distinct channels are separate physical arrays (site expansion to ~220 kW DC).",
            "plausibility": "PLAUSIBLE IF SITE EXPANDED. Consistent with OEDI documentation citing 110-237 kW class group.",
            "peak_15m_kW": float(inv_df["target_dedup_9"].max()),
            "mean_15m_kW": float(inv_df["target_dedup_9"].mean())
        },
        {
            "candidate_id": "Candidate_3",
            "name": "Triple Combiner Group (inv_04 + inv_09 + inv_10)",
            "column": "target_combiners_3",
            "channels_included": cols_c3,
            "channels_excluded": ["inv_01..inv_03, inv_05..inv_08 (assumed sub-strings)", "inv_11 (clone)"],
            "hypothesis": "Channels 04, 09, 10 represent three ~44 kW commercial main building inverters (~130 kW AC peak, ~150 kW DC).",
            "plausibility": "MODERATELY PLAUSIBLE. Exceeds 110 kW DC by 18%, but plausible for a 130 kW AC plant.",
            "peak_15m_kW": float(inv_df["target_combiners_3"].max()),
            "mean_15m_kW": float(inv_df["target_combiners_3"].mean())
        },
        {
            "candidate_id": "Candidate_4",
            "name": "String Inverter Sub-Array (6 Channels)",
            "column": "target_strings_6",
            "channels_included": cols_c4,
            "channels_excluded": ["inv_04, 09, 10, 11 (large units)", "inv_06 (idle)"],
            "hypothesis": "Only small string inverters (7-15 kW) are genuine array inverters. Covers ~63 kW AC peak (~75 kW DC), leaving ~35 kW unmonitored.",
            "plausibility": "PLAUSIBLE FOR MONITORED SUB-ARRAY. Under-represents 110 kW DC total capacity.",
            "peak_15m_kW": float(inv_df["target_strings_6"].max()),
            "mean_15m_kW": float(inv_df["target_strings_6"].mean())
        },
        {
            "candidate_id": "Candidate_5",
            "name": "Primary Twin Combiners (inv_04 + inv_10)",
            "column": "target_pair_04_10",
            "channels_included": cols_c5,
            "channels_excluded": ["inv_01..03, 05..08, 09, 11"],
            "hypothesis": "Two ~44 kW inverters summing to 88 kW AC peak. Corresponds to exact 1.25 DC/AC loading ratio on 110.0 kW DC nameplate (110 / 1.25 = 88 kW).",
            "plausibility": "HIGHLY PLAUSIBLE FOR 110 kW DC. Exactly reconciles AC output with static 110 kW DC metadata.",
            "peak_15m_kW": float(inv_df["target_pair_04_10"].max()),
            "mean_15m_kW": float(inv_df["target_pair_04_10"].mean())
        }
    ]
    
    for c in candidate_meta:
        print(f"[{c['candidate_id']}] {c['name']:<45} Peak: {c['peak_15m_kW']:6.2f} kW | Mean: {c['mean_15m_kW']:5.2f} kW")
        
    # 4. Merge Weather & Inverter Candidates
    print("\nSynchronizing weather and inverter candidates on timestamp...")
    target_cols = [c["column"] for c in candidate_meta]
    merged = pd.merge(weather, inv_df[target_cols], on="ts", how="inner").sort_values("ts").reset_index(drop=True)
    merged.rename(columns={"ts": "Timestamp"}, inplace=True)
    
    # 5. Feature Engineering
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
    
    # Apply night physical gate to all targets
    night_mask = merged["GHI"] <= 0
    for col in target_cols:
        merged.loc[night_mask, col] = 0.0
        merged[col] = np.clip(merged[col], 0.0, None)
        
    feature_cols = [
        "Timestamp", "GHI", "DNI", "Temperature", "Wind_Speed",
        "Hour", "Day", "Month", "SZ_Angle", "SA_Angle",
        "Hour_sin", "Hour_cos", "Month_sin", "Month_cos",
        "Effective_GHI", "Thermal_Stress"
    ]
    
    # 6. Hourly Resampling & Export
    print("\nResampling to hourly resolution and exporting candidate datasets...")
    resample_dict = {
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
        "Thermal_Stress": "mean"
    }
    for col in target_cols:
        resample_dict[col] = "mean"
        
    clean_hourly = merged[feature_cols + target_cols].set_index("Timestamp").resample("1h").agg(resample_dict).dropna().reset_index()
    
    # Update candidate meta with hourly peaks
    for c in candidate_meta:
        c["peak_hourly_kW"] = float(clean_hourly[c["column"]].max())
        c["mean_hourly_kW"] = float(clean_hourly[c["column"]].mean())
        
        # Save individual candidate dataset
        cand_df = clean_hourly[feature_cols + [c["column"]]].rename(columns={c["column"]: "AC_Output"})
        cand_path = os.path.join(DATA_DIR, f"{c['candidate_id'].lower()}_hourly.csv")
        cand_df.to_csv(cand_path, index=False)
        print(f"Exported {c['candidate_id']}: {cand_path} ({len(cand_df):,} rows, Peak={c['peak_hourly_kW']:.2f} kW)")
        
    # Also save the master multi-candidate file for direct comparison
    master_path = os.path.join(DATA_DIR, "all_candidates_hourly.csv")
    clean_hourly.to_csv(master_path, index=False)
    print(f"Exported master candidate file: {master_path}")
    
    # Save candidate metadata json
    meta_path = os.path.join(REPORTS_DIR, "candidate_targets_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(candidate_meta, f, indent=4)
    print(f"Saved candidate metadata to: {meta_path}")
    
    # 7. Generate Diagnostic Plots
    generate_comparison_plots(clean_hourly, candidate_meta)
    
    return clean_hourly, candidate_meta


def generate_comparison_plots(df, candidate_meta):
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING DIAGNOSTIC COMPARISON PLOTS")
    print("=" * 70)
    
    df_plot = df.set_index("Timestamp")
    
    test_days = [
        {"date": "2020-07-15", "title": "Clear-Sky Summer Day (2020-07-15)", "fname": "candidate_comparison_clear_summer.png"},
        {"date": "2020-08-05", "title": "Intermittent / Cloud-Covered Summer Day (2020-08-05)", "fname": "candidate_comparison_cloudy_summer.png"},
        {"date": "2020-12-10", "title": "Clear-Sky Winter Day (2020-12-10)", "fname": "candidate_comparison_clear_winter.png"},
        {"date": "2021-01-20", "title": "Dynamic / Overcast Winter Day (2021-01-20)", "fname": "candidate_comparison_cloudy_winter.png"}
    ]
    
    colors = {
        "target_naive_11": "#7f7f7f",     # gray dashed (flawed reference)
        "target_dedup_9": "#1f77b4",      # blue
        "target_combiners_3": "#ff7f0e",  # orange
        "target_strings_6": "#2ca02c",    # green
        "target_pair_04_10": "#d62728"    # red
    }
    
    labels = {
        "target_naive_11": "Candidate 1: Naive 11-Channel (Flawed Baseline, Peak ~234 kW)",
        "target_dedup_9": "Candidate 2: De-duplicated Full Array (9 Channels, Peak ~190 kW)",
        "target_combiners_3": "Candidate 3: Triple Combiners (inv04+09+10, Peak ~130 kW)",
        "target_strings_6": "Candidate 4: String Sub-Array (6 Channels, Peak ~63 kW)",
        "target_pair_04_10": "Candidate 5: Twin Combiners (inv04+10, Peak ~88 kW, 1.25 ILR)"
    }
    
    for tday in test_days:
        dstr = tday["date"]
        sub = df_plot.loc[f"{dstr} 05:00":f"{dstr} 19:00"]
        if len(sub) == 0:
            continue
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
        
        # Upper subplot: AC Power Output candidates
        for col in ["target_naive_11", "target_dedup_9", "target_combiners_3", "target_pair_04_10", "target_strings_6"]:
            linestyle = "--" if col == "target_naive_11" else "-"
            lw = 2.5 if col in ["target_dedup_9", "target_pair_04_10"] else 1.8
            ax1.plot(sub.index, sub[col], label=labels[col], color=colors[col], linestyle=linestyle, linewidth=lw)
            
        ax1.axhline(RATED_CAPACITY_KW, color="black", linestyle=":", linewidth=1.5, label="Registered Nameplate: 110.0 kW DC")
        ax1.set_ylabel("AC Power Output (kW)", fontsize=11, fontweight='bold')
        ax1.set_title(f"SURYOKTI Target Reconstruction — {tday['title']}", fontsize=13, fontweight='bold', pad=10)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
        
        # Lower subplot: Solar Irradiance (GHI)
        ax2.fill_between(sub.index, sub["GHI"], color="#f39c12", alpha=0.3, label="GHI (W/m²)")
        ax2.plot(sub.index, sub["GHI"], color="#d35400", linewidth=1.5)
        ax2.set_ylabel("GHI (W/m²)", fontsize=10, fontweight='bold')
        ax2.set_xlabel("Local Hawaiian Standard Time (HST)", fontsize=10, fontweight='bold')
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper left", fontsize=9)
        
        fig.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, tday["fname"])
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"Saved diagnostic comparison plot: {plot_path}")
        
    # Bar Chart: Capacity and Peak Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    c_names = [
        "Candidate 1\nNaive 11-Channel\n(Flawed Ref)",
        "Candidate 2\nDe-dup 9-Channel\n(Full Array)",
        "Candidate 3\nTriple Combiners\n(inv04+09+10)",
        "Candidate 5\nTwin Combiners\n(inv04+10, 1.25 ILR)",
        "Candidate 4\nString Sub-Array\n(6 Strings)"
    ]
    peaks = [
        candidate_meta[0]["peak_hourly_kW"],
        candidate_meta[1]["peak_hourly_kW"],
        candidate_meta[2]["peak_hourly_kW"],
        candidate_meta[4]["peak_hourly_kW"],
        candidate_meta[3]["peak_hourly_kW"]
    ]
    bar_colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]
    
    bars = ax.bar(c_names, peaks, color=bar_colors, width=0.55, edgecolor='black', linewidth=1.2)
    ax.axhline(RATED_CAPACITY_KW, color="black", linestyle="--", linewidth=2.0, label="Static Registry Nameplate: 110.0 kW DC")
    ax.axhline(88.0, color="#d62728", linestyle=":", linewidth=1.8, label="Expected AC Peak at 1.25 ILR: 88.0 kW AC")
    
    for bar, val in zip(bars, peaks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3.0, f"{val:.1f} kW", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    ax.set_ylabel("Peak Hourly AC Power Output (kW)", fontsize=11, fontweight='bold')
    ax.set_title("SURYOKTI Target Reconstruction: Observed Peaks vs Registered Capacity", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(0, 270)
    ax.grid(True, linestyle="--", alpha=0.5, axis='y')
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    
    fig.tight_layout()
    bar_path = os.path.join(PLOTS_DIR, "candidate_capacity_comparison.png")
    plt.savefig(bar_path, dpi=300)
    plt.close(fig)
    print(f"Saved capacity comparison chart: {bar_path}")

if __name__ == "__main__":
    build_candidates()

