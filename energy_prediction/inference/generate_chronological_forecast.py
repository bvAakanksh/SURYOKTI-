"""Generate leakage-safe chronological holdout forecasts for anomaly assessment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_COLUMNS = [
    "GHI", "Temperature", "Relative_Humidity", "Wind_Speed", "Wind_Direction", "Hour", "Day", "Month",
]


def generate(input_path: Path, output_path: Path, model_path: Path, metadata_path: Path, train_ratio: float) -> dict:
    if not 0.1 <= train_ratio < 1.0:
        raise ValueError("train_ratio must be at least 0.1 and less than 1.0")
    frame = pd.read_csv(input_path)
    required = set(FEATURE_COLUMNS + ["Timestamp", "AC_Output"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(sorted(missing))}")
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    frame = frame.sort_values("Timestamp", kind="stable").reset_index(drop=True)
    split_at = int(len(frame) * train_ratio)
    train, holdout = frame.iloc[:split_at].copy(), frame.iloc[split_at:].copy()
    if train.empty or holdout.empty:
        raise ValueError("Chronological split produced an empty training or holdout partition.")

    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42)
    model.fit(train[FEATURE_COLUMNS], train["AC_Output"])
    predicted = np.clip(model.predict(holdout[FEATURE_COLUMNS]), 0.0, None)
    predicted[holdout["GHI"].to_numpy() <= 0.1] = 0.0
    holdout["Predicted_AC_Output"] = predicted
    holdout["Residual_kW"] = holdout["AC_Output"] - predicted
    holdout["Power_Deficit_kW"] = np.maximum(0.0, predicted - holdout["AC_Output"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(output_path, index=False)
    joblib.dump(model, model_path)
    r2 = float(r2_score(holdout["AC_Output"], predicted))
    metadata = {
        "input": str(input_path),
        "output": str(output_path),
        "model": str(model_path),
        "usage_status": "SOFTWARE_BENCHMARK_ONLY",
        "forecast_suitability": (
            "NOT_SUITABLE_FOR_RESIDUAL_ANOMALY_CLAIMS"
            if r2 <= 0 else "EXPERIMENTAL_BENCHMARK_ONLY"
        ),
        "feature_columns": FEATURE_COLUMNS,
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "metrics": {
            "r2": r2,
            "mae_kw": float(mean_absolute_error(holdout["AC_Output"], predicted)),
            "rmse_kw": float(np.sqrt(mean_squared_error(holdout["AC_Output"], predicted))),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train chronologically and export leakage-safe holdout forecasts.")
    parser.add_argument("--input", default="datasets/energy_prediction/processed/Solar_Weather_Data_Temporal_Consistent.csv")
    parser.add_argument("--output", default="datasets/energy_prediction/processed/forecast/Solar_Weather_Holdout_Forecast.csv")
    parser.add_argument("--model", default="models/energy_prediction/temporal_consistent_forecast.joblib")
    parser.add_argument("--metadata", default="datasets/energy_prediction/processed/forecast/Solar_Weather_Holdout_Forecast_metadata.json")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    args = parser.parse_args()
    print(json.dumps(generate(Path(args.input), Path(args.output), Path(args.model), Path(args.metadata), args.train_ratio), indent=2))
