"""
Utility functions for SURYOKTI Romania PV Implementation.
Includes logging, metrics, cyclical transformation, and physical constraints.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


def setup_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Set up and return a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def encode_cyclical(series: pd.Series, max_val: float) -> Tuple[pd.Series, pd.Series]:
    """Encode a periodic series into sine and cosine components."""
    sin_comp = np.sin(2 * np.pi * series / max_val)
    cos_comp = np.cos(2 * np.pi * series / max_val)
    return sin_comp, cos_comp


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rated_capacity_kw: float = 50.0,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute R2, MAE, RMSE in capacity factor units and convert to kW equivalents.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Ensure finite values
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.all(valid_mask):
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return {}

    # Capacity factor metrics
    r2 = float(r2_score(y_true, y_pred))
    mae_cf = float(mean_absolute_error(y_true, y_pred))
    rmse_cf = float(root_mean_squared_error(y_true, y_pred))

    # kW metrics
    mae_kw = float(mae_cf * rated_capacity_kw)
    rmse_kw = float(rmse_cf * rated_capacity_kw)

    # Normalized RMSE (%) relative to rated capacity
    nrmse_rated_pct = float((rmse_kw / rated_capacity_kw) * 100.0)

    # Normalized RMSE relative to mean generation (daytime/non-zero)
    mean_val = float(np.mean(y_true))
    nrmse_mean_pct = float((rmse_cf / mean_val * 100.0)) if mean_val > 1e-4 else 0.0

    pfx = f"{prefix}_" if prefix else ""
    return {
        f"{pfx}r2": round(r2, 4),
        f"{pfx}mae_cf": round(mae_cf, 5),
        f"{pfx}rmse_cf": round(rmse_cf, 5),
        f"{pfx}mae_kw": round(mae_kw, 3),
        f"{pfx}rmse_kw": round(rmse_kw, 3),
        f"{pfx}nrmse_rated_pct": round(nrmse_rated_pct, 3),
        f"{pfx}nrmse_mean_pct": round(nrmse_mean_pct, 3),
        f"{pfx}sample_count": int(len(y_true)),
    }


def apply_physical_constraints(
    predictions_cf: np.ndarray,
    solar_elevation_deg: Optional[np.ndarray] = None,
    elevation_threshold: float = 0.0
) -> np.ndarray:
    """
    Clip capacity factor predictions to physical bounds [0.0, 1.0].
    Optionally set to 0.0 when solar elevation is below threshold.
    """
    clipped = np.clip(predictions_cf, 0.0, 1.0)
    if solar_elevation_deg is not None:
        clipped = np.where(solar_elevation_deg <= elevation_threshold, 0.0, clipped)
    return clipped


def cf_to_kw(capacity_factor: np.ndarray, rated_capacity_kw: float = 50.0) -> np.ndarray:
    """Convert capacity factor to active power in kW."""
    return np.asarray(capacity_factor, dtype=float) * rated_capacity_kw


def save_json(data: Any, filepath: Path) -> None:
    """Save data structure to JSON with formatting."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(filepath: Path) -> Any:
    """Load JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_artifact(obj: Any, filepath: Path) -> None:
    """Save model or transformer via joblib."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, filepath)


def load_artifact(filepath: Path) -> Any:
    """Load model or transformer via joblib."""
    return joblib.load(filepath)

