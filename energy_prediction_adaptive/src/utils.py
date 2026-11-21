"""
Utility functions for logging, metrics calculation, artifact serialization,
and physical solar boundary enforcement.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


def setup_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Set up and return a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def calculate_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
    rated_capacity: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute R², MAE, RMSE, MBE (Mean Bias Error), and Normalized RMSE (nRMSE).
    Handles non-finite values safely.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.all(mask):
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {}

    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))

    # Mean Bias Error (MBE = mean(y_pred - y_true)): positive means over-predicting
    mbe = float(np.mean(y_pred - y_true))

    # Normalized RMSE (%)
    mean_val = float(np.mean(y_true))
    nrmse_mean = float((rmse / mean_val) * 100.0) if mean_val > 1e-4 else 0.0

    metrics = {
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mbe": round(mbe, 4),
        "nrmse_mean_pct": round(nrmse_mean, 2),
        "sample_count": int(len(y_true)),
    }

    if rated_capacity and rated_capacity > 0:
        nrmse_capacity = float((rmse / rated_capacity) * 100.0)
        metrics["nrmse_rated_pct"] = round(nrmse_capacity, 2)
        metrics["rated_capacity"] = rated_capacity

    if prefix:
        return {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def apply_physical_bounds(
    predictions: np.ndarray,
    solar_elevation: Optional[np.ndarray] = None,
    elevation_threshold: float = 0.0,
    max_bound: Optional[float] = None,
    min_bound: float = 0.0
) -> np.ndarray:
    """
    Enforce solar physical boundaries:
    1. Clip below min_bound (solar power cannot be negative).
    2. Clip above max_bound if specified.
    3. Zero out power when solar elevation is at or below elevation_threshold.
    """
    clipped = np.clip(predictions, a_min=min_bound, a_max=max_bound)
    if solar_elevation is not None:
        clipped = np.where(solar_elevation <= elevation_threshold, min_bound, clipped)
    return clipped


def save_json(data: Any, filepath: Path) -> None:
    """Save data to JSON with indentation."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(filepath: Path) -> Any:
    """Load data from JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_artifact(obj: Any, filepath: Path) -> None:
    """Serialize model/transformer via joblib."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, filepath)


def load_artifact(filepath: Path) -> Any:
    """Load model/transformer via joblib."""
    return joblib.load(filepath)

