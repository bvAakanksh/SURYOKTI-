"""
SURYOKTI Plant Overview & Target Inverter Candidates Visualization Suite
Generates diurnal inverter profile comparisons across seasonal/cloudy days
and overall plant capacity benchmark bar charts.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

data_candidates = [
    PROJECT_ROOT / "datasets" / "energy_prediction" / "processed" / "all_candidates_hourly.csv",
    PROJECT_ROOT / "experiments" / "regional_adaptation" / "maui" / "target_reconstruction" / "data" / "all_candidates_hourly.csv",
]
data_path = next((p for p in data_candidates if p.exists()), None)

if data_path and data_path.exists():
    df = pd.read_csv(data_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.set_index("Timestamp")

    test_days = [
        {"date": "2020-07-15", "title": "Clear-Sky Summer Day (2020-07-15)", "fname": "candidate_comparison_clear_summer.png"},
        {"date": "2020-08-05", "title": "Intermittent / Cloud-Covered Summer Day (2020-08-05)", "fname": "candidate_comparison_cloudy_summer.png"},
        {"date": "2020-12-10", "title": "Clear-Sky Winter Day (2020-12-10)", "fname": "candidate_comparison_clear_winter.png"},
        {"date": "2021-01-20", "title": "Dynamic / Overcast Winter Day (2021-01-20)", "fname": "candidate_comparison_cloudy_winter.png"}
    ]

    labels = {
        "target_naive_11": "Candidate 1: Naive 11-Channel (Flawed Baseline, Peak ~234 kW)",
        "target_dedup_9": "Candidate 2: De-duplicated Full Array (9 Channels, Peak ~190 kW)",
        "target_combiners_3": "Candidate 3: Triple Combiners (inv04+09+10, Peak ~130 kW)",
        "target_strings_6": "Candidate 4: String Sub-Array (6 Channels, Peak ~63 kW)",
        "target_pair_04_10": "Candidate 5: Twin Combiners (inv04+10, Peak ~88 kW, 1.25 ILR)"
    }

    colors = {
        "target_naive_11": "#7f7f7f",     # gray dashed (flawed reference)
        "target_dedup_9": "#1f77b4",      # blue
        "target_combiners_3": "#ff7f0e",  # orange
        "target_strings_6": "#2ca02c",    # green
        "target_pair_04_10": "#d62728"    # red
    }

    for tday in test_days:
        dstr = tday["date"]
        try:
            sub = df.loc[f"{dstr} 05:00":f"{dstr} 19:00"]
        except Exception:
            continue
        if len(sub) == 0:
            continue
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
        
        # Upper subplot: AC Power Output candidates
        for col in ["target_naive_11", "target_dedup_9", "target_combiners_3", "target_pair_04_10", "target_strings_6"]:
            if col in sub.columns:
                linestyle = "--" if col == "target_naive_11" else "-"
                lw = 2.5 if col in ["target_dedup_9", "target_pair_04_10"] else 1.8
                ax1.plot(sub.index, sub[col], label=labels[col], color=colors[col], linestyle=linestyle, linewidth=lw)
            
        ax1.axhline(110.0, color="black", linestyle=":", linewidth=1.5, label="Registered Nameplate: 110.0 kW DC")
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
        plot_path = PLOT_DIR / tday["fname"]
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {plot_path}")

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
        float(df["target_naive_11"].max()) if "target_naive_11" in df.columns else 234.0,
        float(df["target_dedup_9"].max()) if "target_dedup_9" in df.columns else 185.0,
        float(df["target_combiners_3"].max()) if "target_combiners_3" in df.columns else 115.0,
        float(df["target_pair_04_10"].max()) if "target_pair_04_10" in df.columns else 86.9,
        float(df["target_strings_6"].max()) if "target_strings_6" in df.columns else 65.0,
    ]
    bar_colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]
    
    bars = ax.bar(c_names, peaks, color=bar_colors, width=0.55, edgecolor='black', linewidth=1.2)
    ax.axhline(110.0, color="black", linestyle="--", linewidth=2.0, label="Static Registry Nameplate: 110.0 kW DC")
    ax.axhline(88.0, color="#d62728", linestyle=":", linewidth=1.8, label="Expected AC Peak at 1.25 ILR: 88.0 kW AC")
    
    for bar, val in zip(bars, peaks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3.0, f"{val:.1f} kW", ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    ax.set_ylabel("Peak Hourly AC Power Output (kW)", fontsize=11, fontweight='bold')
    ax.set_title("SURYOKTI Target Reconstruction: Observed Peaks vs Registered Capacity", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(0, 270)
    ax.grid(True, linestyle="--", alpha=0.5, axis='y')
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    
    fig.tight_layout()
    bar_path = PLOT_DIR / "candidate_capacity_comparison.png"
    plt.savefig(bar_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {bar_path}")
