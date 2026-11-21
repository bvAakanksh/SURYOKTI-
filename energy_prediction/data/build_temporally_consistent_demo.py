"""Build a temporally consistent version of the cross-source demo dataset.

This does not claim to make the Gandikota/Hawaii-style benchmark physically
site-accurate. It corrects only deterministic timestamp-derived fields and
preserves the original solar-angle columns under explicit legacy names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build(input_path: Path, output_path: Path, report_path: Path) -> dict:
    frame = pd.read_csv(input_path)
    frame.columns = frame.columns.str.strip().str.replace(" ", "_", regex=False)
    required = {"Timestamp", "GHI", "AC_Output"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(sorted(missing))}")

    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    invalid_timestamp_count = int(frame["Timestamp"].isna().sum())
    source_hour = pd.to_numeric(frame.get("Hour"), errors="coerce")
    timestamp_hour = frame["Timestamp"].dt.hour
    hour_mismatch_count = int((source_hour != timestamp_hour).sum())

    legacy_geometry = [column for column in ("SZ_Angle", "SA_Angle") if column in frame.columns]
    geometry_conflicts = {
        "high_ghi_with_legacy_night_zenith": int(
            ((frame["GHI"] > 50) & (frame.get("SZ_Angle", pd.Series(index=frame.index)) >= 85)).sum()
        ),
        "low_ghi_with_legacy_day_zenith": int(
            ((frame["GHI"] <= 10) & (frame.get("SZ_Angle", pd.Series(index=frame.index)) < 85)).sum()
        ),
    }

    for column in legacy_geometry:
        frame.rename(columns={column: f"Legacy_{column}"}, inplace=True)
    frame["Hour"] = timestamp_hour
    frame["Day"] = frame["Timestamp"].dt.day
    frame["Month"] = frame["Timestamp"].dt.month
    frame = frame.sort_values("Timestamp", kind="stable").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(frame),
        "invalid_timestamp_count": invalid_timestamp_count,
        "timestamp_hour_mismatch_corrected": hour_mismatch_count,
        "legacy_geometry_columns_preserved": legacy_geometry,
        "legacy_geometry_conflicts": geometry_conflicts,
        "usage_status": "SOFTWARE_BENCHMARK_ONLY",
        "limitation": (
            "This dataset combines unrelated source periods and has no verified site coordinates. "
            "It is unsuitable for site-physics claims, hardware-fault validation, or production deployment."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correct deterministic time features in the demo dataset.")
    parser.add_argument("--input", default="Solar_Weather_Data_Cleaned.csv")
    parser.add_argument(
        "--output",
        default="datasets/energy_prediction/processed/Solar_Weather_Data_Temporal_Consistent.csv",
    )
    parser.add_argument(
        "--report",
        default="datasets/energy_prediction/processed/Solar_Weather_Data_Temporal_Consistent_report.json",
    )
    args = parser.parse_args()
    print(json.dumps(build(Path(args.input), Path(args.output), Path(args.report)), indent=2))
