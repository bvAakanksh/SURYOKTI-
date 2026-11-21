"""CLI for the canonical solar-PV anomaly detection pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anomaly_detection import AnomalyConfig, AnomalyPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and evaluate daylight-aware PV anomaly detection.")
    parser.add_argument("--input", required=True, help="Telemetry CSV containing timestamp, power, and irradiance.")
    parser.add_argument("--output-dir", required=True, help="Directory for model, row-level scores, events, and metadata.")
    parser.add_argument("--prediction-column", help="Optional existing forecast column for MAD residual detection.")
    parser.add_argument("--timestamp-column", default="Timestamp")
    parser.add_argument("--power-column", default="AC_Output")
    parser.add_argument("--irradiance-column", default="GHI")
    parser.add_argument("--temperature-column", default="Temperature")
    parser.add_argument("--wind-speed-column", default="Wind_Speed")
    parser.add_argument("--solar-zenith-column", default="SZ_Angle")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    args = parser.parse_args()

    if not 0.1 <= args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be at least 0.1 and less than 1.0.")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input telemetry was not found: {input_path}")
    frame = pd.read_csv(input_path)
    split_at = int(len(frame) * args.train_ratio)
    if split_at < 20 or len(frame) - split_at < 1:
        raise ValueError("Input must provide at least 20 calibration rows and one holdout row.")

    config = AnomalyConfig(
        timestamp_column=args.timestamp_column,
        power_column=args.power_column,
        irradiance_column=args.irradiance_column,
        temperature_column=args.temperature_column,
        wind_speed_column=args.wind_speed_column,
        solar_zenith_column=args.solar_zenith_column or None,
    )
    pipeline = AnomalyPipeline(config).fit(
        frame.iloc[:split_at], prediction_column=args.prediction_column
    )
    scored = pipeline.score(frame.iloc[split_at:], prediction_column=args.prediction_column)
    events = pipeline.aggregate_events(scored)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.save(output_dir / "anomaly_pipeline.joblib")
    scored.to_csv(output_dir / "row_level_anomalies.csv", index=False)
    events.to_csv(output_dir / "anomaly_events.csv", index=False)
    metadata = {
        "input": str(input_path),
        "train_rows": split_at,
        "holdout_rows": len(scored),
        "prediction_column": args.prediction_column,
        "config": asdict(config),
        "calibration": {
            "contamination": pipeline.contamination_,
            "residual_median_kw": pipeline.residual_median_kw_,
            "residual_sigma_kw": pipeline.residual_sigma_kw_,
            "residual_underperformance_threshold_kw": pipeline.residual_underperformance_threshold_kw_,
        },
        "status_counts": scored["Anomaly_Status"].value_counts().to_dict(),
        "event_count": len(events),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
