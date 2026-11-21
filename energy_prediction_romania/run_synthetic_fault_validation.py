"""Validate canonical detection with labeled synthetic faults on forecast holdouts.

This is a software regression benchmark, not evidence of field performance.  It
adds faults only to a copy of an already future-only assessment period, while
the anomaly model is calibrated exclusively on the earlier untouched period.
"""

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
from anomaly_detection.pipeline import (
    STATUS_DATA_QUALITY,
    STATUS_OPERATIONAL,
    STATUS_RESIDUAL,
)
from energy_prediction_romania.run_anomaly_detection import prepare_holdout


FAULT_SPECS = (
    ("SIMULATED_DAYLIGHT_OUTAGE", STATUS_OPERATIONAL),
    ("SIMULATED_PARTIAL_DERATE", STATUS_RESIDUAL),
    ("SIMULATED_NEGATIVE_POWER_SENSOR", STATUS_DATA_QUALITY),
)


def anomaly_config(residual_prediction_bins: int = 1, residual_mad_multiplier: float = 3.0) -> AnomalyConfig:
    """Return the site mapping shared with the Romania holdout assessment."""
    return AnomalyConfig(
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


def inject_faults(
    assessment: pd.DataFrame,
    events_per_type: int = 3,
    event_rows: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a labeled copy with deterministic, non-overlapping daylight events.

    An outage has zero generation under high irradiance, a derate reduces output
    by 65%, and a sensor fault supplies impossible negative power.  Events are
    selected only where both actual and forecast generation are positive.
    """
    if events_per_type < 1 or event_rows < 1:
        raise ValueError("events_per_type and event_rows must both be positive.")
    frame = assessment.copy().reset_index(drop=True)
    frame["Synthetic_Fault_Label"] = "NONE"
    frame["Synthetic_Fault_Event_ID"] = pd.NA
    eligible = (
        (frame["GHI"] >= 200.0)
        & (frame["AC_Output"] > 1.0)
        & (frame["Predicted_AC_Output"] > 1.0)
    )
    selected: set[int] = set()
    manifest: list[dict] = []
    event_number = 0

    for fault_label, expected_status in FAULT_SPECS:
        added = 0
        for start in range(0, len(frame) - event_rows + 1):
            rows = list(range(start, start + event_rows))
            timestamps = frame.loc[rows, "Timestamp"]
            contiguous = timestamps.diff().dropna().dt.total_seconds().le(2 * 3600).all()
            if (
                added >= events_per_type
                or not eligible.loc[rows].all()
                or selected.intersection(rows)
                or not contiguous
            ):
                continue
            event_number += 1
            event_id = f"SYN-{event_number:03d}"
            if fault_label == "SIMULATED_DAYLIGHT_OUTAGE":
                frame.loc[rows, "AC_Output"] = 0.0
            elif fault_label == "SIMULATED_PARTIAL_DERATE":
                frame.loc[rows, "AC_Output"] *= 0.35
            else:
                frame.loc[rows, "AC_Output"] = -1.0
            frame.loc[rows, "Synthetic_Fault_Label"] = fault_label
            frame.loc[rows, "Synthetic_Fault_Event_ID"] = event_id
            selected.update(rows)
            manifest.append({
                "Synthetic_Fault_Event_ID": event_id,
                "Synthetic_Fault_Label": fault_label,
                "Expected_Anomaly_Status": expected_status,
                "Start_Time": timestamps.iloc[0],
                "End_Time": timestamps.iloc[-1],
                "Injected_Row_Count": event_rows,
            })
            added += 1
        if added != events_per_type:
            raise ValueError(
                f"Could not place {events_per_type} non-overlapping {fault_label} events "
                f"of {event_rows} rows in this assessment period."
            )
    return frame, pd.DataFrame(manifest)


def summarize_detection(scored: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    """Calculate transparent synthetic-fault detection metrics."""
    injected = scored["Synthetic_Fault_Label"].ne("NONE")
    critical = {STATUS_OPERATIONAL, STATUS_RESIDUAL, STATUS_DATA_QUALITY}
    summary: dict[str, object] = {
        "injected_row_count": int(injected.sum()),
        "non_injected_daylight_critical_alert_rate": float(
            (
                (~injected)
                & scored["Is_Daylight"]
                & scored["Anomaly_Status"].isin(critical)
            ).sum()
            / max(int(((~injected) & scored["Is_Daylight"]).sum()), 1)
        ),
        "per_fault_type": {},
    }
    for fault_label, expected_status in FAULT_SPECS:
        rows = scored["Synthetic_Fault_Label"].eq(fault_label)
        events = manifest[manifest["Synthetic_Fault_Label"].eq(fault_label)]
        exact = scored.loc[rows, "Anomaly_Status"].eq(expected_status)
        detected_event_count = sum(
            scored.loc[
                scored["Synthetic_Fault_Event_ID"].eq(event_id), "Anomaly_Status"
            ].eq(expected_status).any()
            for event_id in events["Synthetic_Fault_Event_ID"]
        )
        summary["per_fault_type"][fault_label] = {
            "expected_status": expected_status,
            "injected_rows": int(rows.sum()),
            "exact_status_row_recall": float(exact.mean()),
            "injected_events": int(len(events)),
            "exact_status_event_recall": float(detected_event_count / max(len(events), 1)),
        }
    return summary


def run(
    site: str,
    train_ratio: float = 0.70,
    events_per_type: int = 3,
    event_rows: int = 3,
    prediction_dir: Path | None = None,
    output_root: Path | None = None,
    residual_prediction_bins: int = 1,
    residual_mad_multiplier: float = 3.0,
) -> dict:
    """Run the synthetic regression benchmark for one Romania site."""
    if site not in {"site_a", "site_b"}:
        raise ValueError("site must be 'site_a' or 'site_b'")
    if not 0.1 <= train_ratio < 1.0:
        raise ValueError("train_ratio must be at least 0.1 and less than 1.0")
    base_dir = Path(__file__).resolve().parent
    prediction_dir = prediction_dir or base_dir / "results" / "predictions"
    output_root = output_root or base_dir / "results" / "synthetic_fault_validation"
    source = prediction_dir / f"{site}_test_predictions.csv"
    output_dir = output_root / site
    frame = prepare_holdout(pd.read_csv(source))
    split_at = int(len(frame) * train_ratio)
    calibration = frame.iloc[:split_at]
    injected, manifest = inject_faults(frame.iloc[split_at:], events_per_type, event_rows)
    pipeline = AnomalyPipeline(anomaly_config(residual_prediction_bins, residual_mad_multiplier)).fit(
        calibration, prediction_column="Predicted_AC_Output"
    )
    scored = pipeline.score(injected, prediction_column="Predicted_AC_Output")
    events = pipeline.aggregate_events(scored)
    metrics = summarize_detection(scored, manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "injection_manifest.csv", index=False)
    scored.to_csv(output_dir / "row_level_synthetic_assessment.csv", index=False)
    events.to_csv(output_dir / "anomaly_events.csv", index=False)
    metadata = {
        "site": site,
        "usage_status": "SYNTHETIC_SOFTWARE_REGRESSION_BENCHMARK_ONLY",
        "input_predictions": str(source),
        "calibration_rows": len(calibration),
        "assessment_rows": len(injected),
        "fault_injections": {"events_per_type": events_per_type, "event_rows": event_rows},
        "config": asdict(anomaly_config(residual_prediction_bins, residual_mad_multiplier)),
        "metrics": metrics,
        "limitation": (
            "These recall values measure only deterministic synthetic faults injected into a copy of "
            "unlabeled holdout telemetry. They are not field precision, real fault recall, or proof "
            "of equipment-failure detection. Confirmed maintenance and inspection labels remain required."
        ),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run labeled synthetic-fault validation on Romania forecast holdouts.")
    parser.add_argument("--site", choices=["site_a", "site_b", "all"], default="all")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--events-per-type", type=int, default=3)
    parser.add_argument("--event-rows", type=int, default=3)
    parser.add_argument("--prediction-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--residual-prediction-bins", type=int, default=1)
    parser.add_argument("--residual-mad-multiplier", type=float, default=3.0)
    args = parser.parse_args()
    sites = ["site_a", "site_b"] if args.site == "all" else [args.site]
    print(json.dumps({
        site: run(
            site, args.train_ratio, args.events_per_type, args.event_rows,
            args.prediction_dir, args.output_root, args.residual_prediction_bins,
            args.residual_mad_multiplier,
        ) for site in sites
    }, indent=2))
