"""Run canonical anomaly detection on Romania's leakage-safe forecast holdouts."""

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


def prepare_holdout(predictions: pd.DataFrame) -> pd.DataFrame:
    """Map Romania forecast output to the canonical anomaly schema."""
    required = {
        "datetime_utc", "actual_kw", "ghi_wm2", "temp_c",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction input is missing required columns: {', '.join(sorted(missing))}")
    frame = predictions.copy()
    frame["Timestamp"] = pd.to_datetime(frame["datetime_utc"], errors="coerce", utc=True)
    frame["AC_Output"] = pd.to_numeric(frame["actual_kw"], errors="coerce")
    prediction_column = next((
        column for column in ("tuned_pred_kw", "stack_pred_kw") if column in frame.columns
    ), None)
    if prediction_column is None:
        raise ValueError("Prediction input requires either 'stack_pred_kw' or 'tuned_pred_kw'.")
    frame["Predicted_AC_Output"] = pd.to_numeric(frame[prediction_column], errors="coerce")
    frame["GHI"] = pd.to_numeric(frame["ghi_wm2"], errors="coerce")
    frame["Temperature"] = pd.to_numeric(frame["temp_c"], errors="coerce")
    return frame.sort_values("Timestamp", kind="stable").reset_index(drop=True)


def run(
    site: str,
    train_ratio: float = 0.70,
    prediction_dir: Path | None = None,
    output_root: Path | None = None,
    residual_prediction_bins: int = 1,
    residual_mad_multiplier: float = 3.0,
) -> dict:
    if site not in {"site_a", "site_b"}:
        raise ValueError("site must be 'site_a' or 'site_b'")
    if not 0.1 <= train_ratio < 1.0:
        raise ValueError("train_ratio must be at least 0.1 and less than 1.0")

    base_dir = Path(__file__).resolve().parent
    prediction_dir = prediction_dir or base_dir / "results" / "predictions"
    output_root = output_root or base_dir / "results" / "anomaly_detection"
    prediction_path = prediction_dir / f"{site}_test_predictions.csv"
    output_dir = output_root / site
    frame = prepare_holdout(pd.read_csv(prediction_path))
    split_at = int(len(frame) * train_ratio)
    calibration, assessment = frame.iloc[:split_at], frame.iloc[split_at:]

    config = AnomalyConfig(
        timestamp_column="Timestamp",
        power_column="AC_Output",
        irradiance_column="GHI",
        temperature_column="Temperature",
        wind_speed_column=None,
        solar_zenith_column=None,
        daylight_irradiance_wm2=50.0,
        residual_prediction_bins=residual_prediction_bins,
        residual_mad_multiplier=residual_mad_multiplier,
    )
    pipeline = AnomalyPipeline(config).fit(calibration, prediction_column="Predicted_AC_Output")
    scored = pipeline.score(assessment, prediction_column="Predicted_AC_Output")
    events = pipeline.aggregate_events(scored)

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.save(output_dir / "anomaly_pipeline.joblib")
    scored.to_csv(output_dir / "row_level_anomalies.csv", index=False)
    events.to_csv(output_dir / "anomaly_events.csv", index=False)
    metadata = {
        "site": site,
        "input_predictions": str(prediction_path),
        "usage_status": "RESEARCH_BENCHMARK_NO_FAULT_GROUND_TRUTH",
        "calibration_rows": len(calibration),
        "assessment_rows": len(assessment),
        "prediction_column": "Predicted_AC_Output",
        "residual_prediction_bins": residual_prediction_bins,
        "residual_mad_multiplier": residual_mad_multiplier,
        "config": asdict(config),
        "calibration": {
            "contamination": pipeline.contamination_,
            "residual_median_kw": pipeline.residual_median_kw_,
            "residual_sigma_kw": pipeline.residual_sigma_kw_,
            "residual_underperformance_threshold_kw": pipeline.residual_underperformance_threshold_kw_,
        },
        "status_counts": scored["Anomaly_Status"].value_counts().to_dict(),
        "event_count": len(events),
        "limitation": (
            "Forecasts are chronological holdout predictions, but no maintenance or fault labels are available. "
            "Detected events are candidates for inspection, not confirmed equipment failures."
        ),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assess Romania forecast residuals with canonical anomaly detection.")
    parser.add_argument("--site", choices=["site_a", "site_b", "all"], default="all")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--prediction-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--residual-prediction-bins", type=int, default=1)
    parser.add_argument("--residual-mad-multiplier", type=float, default=3.0)
    args = parser.parse_args()
    sites = ["site_a", "site_b"] if args.site == "all" else [args.site]
    print(json.dumps({
        site: run(
            site, args.train_ratio, args.prediction_dir, args.output_root,
            args.residual_prediction_bins, args.residual_mad_multiplier,
        )
        for site in sites
    }, indent=2))
