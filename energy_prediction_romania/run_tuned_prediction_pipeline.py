"""Causally tune non-CNN PV forecasts on existing Romania datasets.

Candidate models are chosen on the chronological validation partition only.
After selection, the winner is retrained on train+validation and evaluated once
on the untouched test partition.  Baseline artifacts are never overwritten.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energy_prediction_romania.config import (
    DATA_PROCESSED_DIR, FEATURE_COLS, PRIMARY_TARGET, RANDOM_STATE,
    RATED_CAPACITY_KW, SOLAR_ELEVATION_DAYTIME_THRESHOLD,
)
from energy_prediction_romania.src.data_split import split_chronological
from energy_prediction_romania.src.preprocessing import PreprocessingPipeline
from energy_prediction_romania.src.utils import apply_physical_constraints, calculate_metrics, save_artifact


RF_CANDIDATES = (
    {"name": "rf_baseline", "n_estimators": 300, "max_depth": 8, "min_samples_split": 6, "min_samples_leaf": 3, "max_features": 0.8},
    {"name": "rf_deeper", "n_estimators": 400, "max_depth": 12, "min_samples_split": 4, "min_samples_leaf": 2, "max_features": 0.9},
    {"name": "rf_regularized", "n_estimators": 400, "max_depth": 10, "min_samples_split": 8, "min_samples_leaf": 4, "max_features": 0.8},
)
XGB_CANDIDATES = (
    {"name": "xgb_baseline", "n_estimators": 300, "learning_rate": 0.03, "max_depth": 4, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.5},
    {"name": "xgb_shallow", "n_estimators": 500, "learning_rate": 0.025, "max_depth": 3, "min_child_weight": 4, "subsample": 0.85, "colsample_bytree": 0.9, "reg_alpha": 0.1, "reg_lambda": 2.0},
    {"name": "xgb_deeper", "n_estimators": 350, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 4, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.2, "reg_lambda": 2.0},
)


def _daytime_mae(y_true: np.ndarray, pred: np.ndarray, elevation: np.ndarray) -> float:
    mask = elevation > SOLAR_ELEVATION_DAYTIME_THRESHOLD
    return float(np.mean(np.abs(y_true[mask] - pred[mask])))


def _fit_rf(candidate: dict, x: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    params = {key: value for key, value in candidate.items() if key != "name"}
    return RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1).fit(x, y)


def _fit_xgb(candidate: dict, x: np.ndarray, y: np.ndarray) -> XGBRegressor:
    params = {key: value for key, value in candidate.items() if key != "name"}
    return XGBRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1, objective="reg:squarederror").fit(x, y)


def _constrain(prediction: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    return apply_physical_constraints(prediction, elevation, elevation_threshold=0.0)


def tune_site(site: str) -> dict:
    """Select RF/XGBoost candidates and a validation-selected convex blend."""
    processed_path = DATA_PROCESSED_DIR / f"{site}_processed.csv"
    model_dir = PROJECT_ROOT / "energy_prediction_romania" / "models" / "tuned" / site
    output_dir = PROJECT_ROOT / "energy_prediction_romania" / "results" / "tuned_predictions"
    frame = pd.read_csv(processed_path)
    train, validation, test, split_info = split_chronological(frame)
    preprocessor = PreprocessingPipeline(feature_cols=FEATURE_COLS).fit(train)
    x_train, x_val = preprocessor.transform(train), preprocessor.transform(validation)
    y_train = train[PRIMARY_TARGET].to_numpy(float)
    y_val = validation[PRIMARY_TARGET].to_numpy(float)
    val_elevation = validation["solar_elevation_deg"].to_numpy(float)

    candidates: list[dict] = []
    fitted: dict[str, object] = {}
    for family, specs, factory in (("rf", RF_CANDIDATES, _fit_rf), ("xgb", XGB_CANDIDATES, _fit_xgb)):
        for spec in specs:
            model = factory(spec, x_train, y_train)
            prediction = _constrain(model.predict(x_val), val_elevation)
            score = _daytime_mae(y_val, prediction, val_elevation)
            candidates.append({"family": family, "name": spec["name"], "validation_daytime_mae_cf": score, "params": spec})
            fitted[spec["name"]] = model
    best_rf = min((row for row in candidates if row["family"] == "rf"), key=lambda row: row["validation_daytime_mae_cf"])
    best_xgb = min((row for row in candidates if row["family"] == "xgb"), key=lambda row: row["validation_daytime_mae_cf"])
    rf_val = _constrain(fitted[best_rf["name"]].predict(x_val), val_elevation)
    xgb_val = _constrain(fitted[best_xgb["name"]].predict(x_val), val_elevation)
    blend_candidates = []
    for rf_weight in np.linspace(0.0, 1.0, 21):
        blend = rf_weight * rf_val + (1 - rf_weight) * xgb_val
        blend_candidates.append((float(rf_weight), _daytime_mae(y_val, blend, val_elevation)))
    rf_weight, validation_daytime_mae = min(blend_candidates, key=lambda row: row[1])

    development = pd.concat([train, validation], ignore_index=True)
    final_preprocessor = PreprocessingPipeline(feature_cols=FEATURE_COLS).fit(development)
    x_development = final_preprocessor.transform(development)
    y_development = development[PRIMARY_TARGET].to_numpy(float)
    final_rf = _fit_rf(best_rf["params"], x_development, y_development)
    final_xgb = _fit_xgb(best_xgb["params"], x_development, y_development)
    x_test = final_preprocessor.transform(test)
    test_elevation = test["solar_elevation_deg"].to_numpy(float)
    rf_pred = _constrain(final_rf.predict(x_test), test_elevation)
    xgb_pred = _constrain(final_xgb.predict(x_test), test_elevation)
    blend_pred = rf_weight * rf_pred + (1 - rf_weight) * xgb_pred
    y_test = test[PRIMARY_TARGET].to_numpy(float)
    metrics = calculate_metrics(y_test, blend_pred, rated_capacity_kw=RATED_CAPACITY_KW, prefix="test")
    daytime = test_elevation > SOLAR_ELEVATION_DAYTIME_THRESHOLD
    metrics.update(calculate_metrics(y_test[daytime], blend_pred[daytime], rated_capacity_kw=RATED_CAPACITY_KW, prefix="daytime"))

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_preprocessor.save(model_dir / "preprocessing_pipeline.joblib")
    save_artifact(final_rf, model_dir / "rf_model.joblib")
    save_artifact(final_xgb, model_dir / "xgb_model.joblib")
    prediction_frame = pd.DataFrame({
        "datetime_utc": test["datetime_utc"], "solar_elevation_deg": test_elevation,
        "ghi_wm2": test["ghi_wm2"], "temp_c": test["temp_c"],
        "actual_cf": y_test, "actual_kw": y_test * RATED_CAPACITY_KW,
        "rf_pred_cf": rf_pred, "xgb_pred_cf": xgb_pred,
        "tuned_pred_cf": blend_pred, "tuned_pred_kw": blend_pred * RATED_CAPACITY_KW,
        "tuned_residual_kw": (blend_pred - y_test) * RATED_CAPACITY_KW,
        "is_daytime": daytime,
    })
    prediction_frame.to_csv(output_dir / f"{site}_test_predictions.csv", index=False)
    result = {
        "site": site, "usage_status": "CHRONOLOGICALLY_TUNED_NON_CNN_BENCHMARK",
        "selection_metric": "validation_daytime_mae_cf", "split_info": split_info,
        "candidate_results": candidates, "selected_rf": best_rf, "selected_xgb": best_xgb,
        "selected_rf_weight": rf_weight, "validation_daytime_mae_cf": validation_daytime_mae,
        "untouched_test_metrics": metrics,
        "limitation": "Hyperparameters and blend weight used only the validation period; test metrics are a single final estimate, not a fault-label evaluation.",
    }
    (model_dir / "tuning_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps({site: tune_site(site) for site in ("site_a", "site_b")}, indent=2))
