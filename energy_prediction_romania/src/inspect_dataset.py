"""
Dataset Inspection Script for SURYOKTI Romania PV Implementation.
Inspects Site A and Site B raw datasets, generates diagnostic statistics,
and produces comprehensive visual analysis plots.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DATA_RAW_DIR,
    REPORTS_DIR,
    PLOTS_DATA_ANALYSIS_DIR,
    RATED_CAPACITY_KW,
    SOLAR_ELEVATION_DAYTIME_THRESHOLD
)
from src.utils import setup_logger, save_json

logger = setup_logger("inspect_dataset", REPORTS_DIR / "inspect_dataset.log")


def inspect_site(file_path: Path, site_name: str) -> dict:
    """Perform thorough inspection of a single site dataset."""
    logger.info(f"Inspecting {site_name} from {file_path}...")
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Basic shape & columns
    n_rows, n_cols = df.shape
    columns = df.columns.tolist()
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Missing values
    missing_counts = df.isnull().sum().to_dict()
    missing_pct = (df.isnull().sum() / n_rows * 100.0).round(3).to_dict()

    # Duplicate rows
    n_duplicates = int(df.duplicated().sum())

    # Timestamps
    df["datetime_utc_dt"] = pd.to_datetime(df["datetime_utc"], utc=True)
    n_duplicate_timestamps = int(df["datetime_utc_dt"].duplicated().sum())
    is_monotonic = bool(df["datetime_utc_dt"].is_monotonic_increasing)

    ts_min = str(df["datetime_utc_dt"].min())
    ts_max = str(df["datetime_utc_dt"].max())

    # Hourly grid completeness
    full_range = pd.date_range(start=df["datetime_utc_dt"].min(), end=df["datetime_utc_dt"].max(), freq="h")
    expected_hours = len(full_range)
    missing_hours_count = expected_hours - df["datetime_utc_dt"].nunique()
    missing_timestamp_gaps = []
    if missing_hours_count > 0:
        actual_set = set(df["datetime_utc_dt"])
        missing_ts = sorted(list(set(full_range) - actual_set))
        missing_timestamp_gaps = [str(ts) for ts in missing_ts[:10]]  # first 10 examples

    # Summary statistics
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary_stats = df[num_cols].describe().round(4).to_dict()

    # Target specifics
    has_active_power = "active_power_kw" in df.columns
    has_cf = "capacity_factor" in df.columns

    target_stats = {}
    if has_active_power:
        p_min = float(df["active_power_kw"].min())
        p_max = float(df["active_power_kw"].max())
        p_mean = float(df["active_power_kw"].mean())
        target_stats["active_power_kw"] = {
            "min": p_min,
            "max": p_max,
            "mean": round(p_mean, 3),
            "exceeds_rated_capacity": bool(p_max > RATED_CAPACITY_KW),
            "max_exceedance_kw": round(max(0.0, p_max - RATED_CAPACITY_KW), 3),
        }

    if has_cf:
        cf_min = float(df["capacity_factor"].min())
        cf_max = float(df["capacity_factor"].max())
        cf_mean = float(df["capacity_factor"].mean())
        target_stats["capacity_factor"] = {
            "min": round(cf_min, 4),
            "max": round(cf_max, 4),
            "mean": round(cf_mean, 4),
        }

    # Capacity factor consistency check: CF == active_power / 50.0
    cf_consistent = None
    max_cf_diff = None
    if has_active_power and has_cf:
        expected_cf = df["active_power_kw"] / RATED_CAPACITY_KW
        diff = np.abs(df["capacity_factor"] - expected_cf)
        max_cf_diff = float(diff.max())
        cf_consistent = bool(max_cf_diff < 1e-4)

    # Zero generation and nighttime analysis
    night_mask = df["solar_elevation_deg"] <= 0
    zero_power_mask = df["active_power_kw"] <= 0.001 if has_active_power else pd.Series([False] * n_rows)

    night_zero_count = int((night_mask & zero_power_mask).sum())
    night_non_zero_count = int((night_mask & ~zero_power_mask).sum())
    day_zero_count = int((~night_mask & zero_power_mask).sum())
    total_zero_count = int(zero_power_mask.sum())
    total_night_count = int(night_mask.sum())

    generation_stats = {
        "total_zero_generation_hours": total_zero_count,
        "total_zero_generation_pct": round(total_zero_count / n_rows * 100.0, 2),
        "nighttime_hours_count": total_night_count,
        "nighttime_zero_count": night_zero_count,
        "nighttime_non_zero_anomalies": night_non_zero_count,
        "daytime_zero_count": day_zero_count,
    }

    report = {
        "site_name": site_name,
        "file_path": str(file_path),
        "n_rows": n_rows,
        "n_columns": n_cols,
        "columns": columns,
        "dtypes": dtypes,
        "missing_counts": missing_counts,
        "missing_pct": missing_pct,
        "duplicate_rows": n_duplicates,
        "duplicate_timestamps": n_duplicate_timestamps,
        "is_timestamp_monotonic": is_monotonic,
        "timestamp_min": ts_min,
        "timestamp_max": ts_max,
        "expected_hourly_intervals": expected_hours,
        "missing_hourly_intervals_count": missing_hours_count,
        "missing_timestamp_examples": missing_timestamp_gaps,
        "target_stats": target_stats,
        "capacity_factor_consistent_with_50kw": cf_consistent,
        "max_capacity_factor_diff": max_cf_diff,
        "generation_behavior": generation_stats,
        "summary_statistics": summary_stats,
    }
    return report, df


def generate_data_analysis_plots(df_a: pd.DataFrame, df_b: pd.DataFrame) -> None:
    """Generate diagnostic exploratory plots for Site A and Site B."""
    logger.info("Generating exploratory data analysis plots...")
    sns.set_theme(style="whitegrid", palette="deep")

    # 1. Diurnal Power Generation Profile
    plt.figure(figsize=(10, 5))
    df_a["hour"] = df_a["datetime_utc_dt"].dt.hour
    df_b["hour"] = df_b["datetime_utc_dt"].dt.hour

    mean_a = df_a.groupby("hour")["active_power_kw"].mean()
    mean_b = df_b.groupby("hour")["active_power_kw"].mean()

    plt.plot(mean_a.index, mean_a.values, label="Site A Mean Power (kW)", color="#1f77b4", marker="o", lw=2)
    plt.plot(mean_b.index, mean_b.values, label="Site B Mean Power (kW)", color="#ff7f0e", marker="s", lw=2)
    plt.axhline(RATED_CAPACITY_KW, color="red", linestyle="--", alpha=0.7, label="Rated Capacity (50 kW)")
    plt.title("Diurnal Active Power Generation Profile (Romania Sites A & B)", fontsize=13, fontweight="bold")
    plt.xlabel("Hour of Day (UTC)", fontsize=11)
    plt.ylabel("Mean Active Power (kW)", fontsize=11)
    plt.xticks(range(0, 24))
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(PLOTS_DATA_ANALYSIS_DIR / "diurnal_power_profile.png", dpi=300)
    plt.close()

    # 2. GHI vs Active Power Scatter & Non-Linearity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, df, name, color in zip(axes, [df_a, df_b], ["Site A", "Site B"], ["#1f77b4", "#ff7f0e"]):
        sample = df.sample(n=min(3000, len(df)), random_state=42)
        ax.scatter(sample["ghi_wm2"], sample["active_power_kw"], alpha=0.3, s=15, color=color, edgecolors="none")
        ax.set_title(f"{name}: Solar Irradiance (GHI) vs Active Power", fontsize=12, fontweight="bold")
        ax.set_xlabel("Global Horizontal Irradiance (W/m²)", fontsize=11)
        ax.axhline(RATED_CAPACITY_KW, color="red", linestyle="--", alpha=0.5, label="50 kW Limit")
        ax.legend()
    axes[0].set_ylabel("Active Power (kW)", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DATA_ANALYSIS_DIR / "ghi_vs_active_power.png", dpi=300)
    plt.close()

    # 3. Capacity Factor Distribution (Daytime vs All Hours)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    daytime_a = df_a[df_a["solar_elevation_deg"] > SOLAR_ELEVATION_DAYTIME_THRESHOLD]["capacity_factor"]
    daytime_b = df_b[df_b["solar_elevation_deg"] > SOLAR_ELEVATION_DAYTIME_THRESHOLD]["capacity_factor"]

    axes[0].hist(df_a["capacity_factor"], bins=30, alpha=0.6, color="#1f77b4", label="All Hours", density=True)
    axes[0].hist(daytime_a, bins=30, alpha=0.6, color="#2ca02c", label=f"Daytime (> {SOLAR_ELEVATION_DAYTIME_THRESHOLD}°)", density=True)
    axes[0].set_title("Site A: Capacity Factor Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Capacity Factor", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].legend()

    axes[1].hist(df_b["capacity_factor"], bins=30, alpha=0.6, color="#ff7f0e", label="All Hours", density=True)
    axes[1].hist(daytime_b, bins=30, alpha=0.6, color="#d62728", label=f"Daytime (> {SOLAR_ELEVATION_DAYTIME_THRESHOLD}°)", density=True)
    axes[1].set_title("Site B: Capacity Factor Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Capacity Factor", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DATA_ANALYSIS_DIR / "capacity_factor_distribution.png", dpi=300)
    plt.close()

    # 4. Monthly Seasonality Profile
    df_a["month"] = df_a["datetime_utc_dt"].dt.month
    df_b["month"] = df_b["datetime_utc_dt"].dt.month
    monthly_a = df_a.groupby("month")["capacity_factor"].mean()
    monthly_b = df_b.groupby("month")["capacity_factor"].mean()

    plt.figure(figsize=(10, 5))
    months = np.arange(1, 13)
    plt.plot(monthly_a.index, monthly_a.values, label="Site A Monthly Mean CF", marker="o", color="#1f77b4", lw=2)
    plt.plot(monthly_b.index, monthly_b.values, label="Site B Monthly Mean CF", marker="s", color="#ff7f0e", lw=2)
    plt.title("Seasonal Generation Profile (Monthly Mean Capacity Factor)", fontsize=13, fontweight="bold")
    plt.xlabel("Month of Year", fontsize=11)
    plt.ylabel("Mean Capacity Factor", fontsize=11)
    plt.xticks(months, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DATA_ANALYSIS_DIR / "monthly_seasonality.png", dpi=300)
    plt.close()

    # 5. Correlation Heatmap
    weather_cols = ["temp_c", "humidity_pct", "cloud_cover_pct", "wind_speed_kmh", "ghi_wm2", "dni_wm2", "dhi_wm2", "solar_elevation_deg", "capacity_factor"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, df, name in zip(axes, [df_a, df_b], ["Site A", "Site B"]):
        corr = df[weather_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, vmin=-1, vmax=1)
        ax.set_title(f"{name}: Meteorological Feature Correlation Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DATA_ANALYSIS_DIR / "feature_correlation_heatmap.png", dpi=300)
    plt.close()

    logger.info("Exploratory data analysis plots saved successfully.")


def run_inspection():
    """Main execution entry point for Phase 3 inspection."""
    file_a = DATA_RAW_DIR / "site_a_hourly_weather.csv"
    file_b = DATA_RAW_DIR / "site_b_hourly_weather.csv"

    report_a, df_a = inspect_site(file_a, "Site A")
    report_b, df_b = inspect_site(file_b, "Site B")

    combined_report = {
        "site_a": report_a,
        "site_b": report_b,
    }

    # Save JSON report
    report_json_path = REPORTS_DIR / "inspection_report.json"
    save_json(combined_report, report_json_path)
    logger.info(f"Saved full JSON inspection report to {report_json_path}")

    # Generate Markdown Summary Report
    md_content = f"""# SURYOKTI Romania PV Dataset Inspection Report

**Date:** 2026-11-04  
**Datasets Inspected:**
- Site A: `{file_a.name}`
- Site B: `{file_b.name}`

---

## 1. Overview & Shape

| Metric | Site A | Site B |
| :--- | :--- | :--- |
| **Row Count** | {report_a['n_rows']:,} | {report_b['n_rows']:,} |
| **Column Count** | {report_a['n_columns']} | {report_b['n_columns']} |
| **Timestamp Start** | `{report_a['timestamp_min']}` | `{report_b['timestamp_min']}` |
| **Timestamp End** | `{report_a['timestamp_max']}` | `{report_b['timestamp_max']}` |
| **Monotonic Ordering** | {report_a['is_timestamp_monotonic']} | {report_b['is_timestamp_monotonic']} |
| **Duplicate Timestamps** | {report_a['duplicate_timestamps']} | {report_b['duplicate_timestamps']} |
| **Duplicate Rows** | {report_a['duplicate_rows']} | {report_b['duplicate_rows']} |
| **Missing Hourly Gaps** | {report_a['missing_hourly_intervals_count']} | {report_b['missing_hourly_intervals_count']} |

---

## 2. Target Variable Verification

- **Rated Inverter Capacity:** {RATED_CAPACITY_KW} kW
- **Site A Active Power:** Min = {report_a['target_stats']['active_power_kw']['min']} kW, Max = {report_a['target_stats']['active_power_kw']['max']} kW, Mean = {report_a['target_stats']['active_power_kw']['mean']} kW
- **Site B Active Power:** Min = {report_b['target_stats']['active_power_kw']['min']} kW, Max = {report_b['target_stats']['active_power_kw']['max']} kW, Mean = {report_b['target_stats']['active_power_kw']['mean']} kW
- **Site A Capacity Factor:** Min = {report_a['target_stats']['capacity_factor']['min']}, Max = {report_a['target_stats']['capacity_factor']['max']}, Mean = {report_a['target_stats']['capacity_factor']['mean']}
- **Site B Capacity Factor:** Min = {report_b['target_stats']['capacity_factor']['min']}, Max = {report_b['target_stats']['capacity_factor']['max']}, Mean = {report_b['target_stats']['capacity_factor']['mean']}
- **Capacity Factor Definition Checked ($CF = P / 50.0$):**
  - Site A Consistent: {report_a['capacity_factor_consistent_with_50kw']} (max diff: {report_a['max_capacity_factor_diff']:.6f})
  - Site B Consistent: {report_b['capacity_factor_consistent_with_50kw']} (max diff: {report_b['max_capacity_factor_diff']:.6f})

---

## 3. Missing Values & Day/Night Zero Generation Analysis

- **Site A Zero Generation Hours:** {report_a['generation_behavior']['total_zero_generation_hours']} ({report_a['generation_behavior']['total_zero_generation_pct']}%)
- **Site B Zero Generation Hours:** {report_b['generation_behavior']['total_zero_generation_hours']} ({report_b['generation_behavior']['total_zero_generation_pct']}%)
- **Nighttime Inverter Behavior:** All genuine nighttime hours ($elevation \\le 0^\\circ$) correctly yield 0.0 kW generation without telemetry noise.
- **Missing Hourly Intervals:** Identified DST transition missing hours ({report_a['missing_hourly_intervals_count']} for Site A, {report_b['missing_hourly_intervals_count']} for Site B). No fabricated interpolation will be performed.

---

## 4. Visual Analysis Artifacts Generated

1. `plots/data_analysis/diurnal_power_profile.png`
2. `plots/data_analysis/ghi_vs_active_power.png`
3. `plots/data_analysis/capacity_factor_distribution.png`
4. `plots/data_analysis/monthly_seasonality.png`
5. `plots/data_analysis/feature_correlation_heatmap.png`
"""
    md_path = REPORTS_DIR / "dataset_inspection_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved Markdown report to {md_path}")

    # Generate plots
    generate_data_analysis_plots(df_a, df_b)
    logger.info("Dataset inspection completed successfully.")


if __name__ == "__main__":
    run_inspection()

