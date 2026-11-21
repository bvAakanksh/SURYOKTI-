"""
Configuration and Hyperparameter Settings for SURYOKTI Romania PV Implementation.
All paths are relative to this project directory, strictly isolating it from the original codebase.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent

# Data Directories
DATA_DIR = BASE_DIR / "datasets"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_CLEANED_DIR = DATA_DIR / "cleaned"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Model Artifacts
MODELS_DIR = BASE_DIR / "models"
MODELS_SITE_A_DIR = MODELS_DIR / "site_a"
MODELS_SITE_B_DIR = MODELS_DIR / "site_b"

# Results & Artifacts
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
LOGS_DIR = RESULTS_DIR / "logs"
REPORTS_DIR = RESULTS_DIR / "reports"

# Plots Directories
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DATA_ANALYSIS_DIR = PLOTS_DIR / "data_analysis"
PLOTS_MODEL_EVAL_DIR = PLOTS_DIR / "model_evaluation"
PLOTS_CROSS_SITE_DIR = PLOTS_DIR / "cross_site"

# System / Target Specifications
RATED_CAPACITY_KW = 50.0
PRIMARY_TARGET = "capacity_factor"
SECONDARY_TARGET = "active_power_kw"

# Meteorological Features (available at forecast time)
METEOROLOGICAL_FEATURES = [
    "temp_c",
    "humidity_pct",
    "cloud_cover_pct",
    "wind_speed_kmh",
    "ghi_wm2",
    "dni_wm2",
    "dhi_wm2",
    "precip_mm",
    "solar_elevation_deg",
    "solar_azimuth_deg",
]

# Cyclical Time Features
TIME_CYCLICAL_FEATURES = [
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "month_sin",
    "month_cos",
]

# Complete Model Feature Column Set
FEATURE_COLS = METEOROLOGICAL_FEATURES + TIME_CYCLICAL_FEATURES

# Strict Target Leakage Exclusion List
FORBIDDEN_LEAKAGE_COLS = [
    "active_power_kw",
    "capacity_factor",
    "daily_energy_kwh",
    "cumulative_energy_kwh",
    "dc_input_power_kw",
    "efficiency_pct",
    "is_generating",
    "inverter_status",
    "inverter_temp_c",
    "grid_freq_hz",
    "power_factor",
]

# Day/Night Solar Filtering
SOLAR_ELEVATION_DAYTIME_THRESHOLD = 5.0  # degrees above horizon

# Chronological Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Canonical Hyperparameters (from inspected SURYOKTI architecture)
RANDOM_STATE = 42

RF_HYPERPARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_split": 6,
    "min_samples_leaf": 3,
    "max_features": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGB_HYPERPARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

NN_HYPERPARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.05,
    "learning_rate_init": 0.003,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 20,
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

STACKING_CV_SPLITS = 5

