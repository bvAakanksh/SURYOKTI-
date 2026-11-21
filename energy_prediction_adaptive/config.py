"""
Configuration and Dynamic Schema Defaults for SURYOKTI Adaptive Prediction Pipeline.
Provides relative paths, semantic alias catalogs, leakage patterns, and hyperparameter grids.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
REPORTS_DIR = RESULTS_DIR / "reports"
PLOTS_DIR = BASE_DIR / "plots"

# Common Timestamp Column Candidate Names
TIMESTAMP_CANDIDATES = [
    "datetime_utc",
    "datetime_local",
    "datetime",
    "timestamp",
    "date_time",
    "time",
    "date",
]

# Common Solar Target Candidate Names (in order of priority if unspecified)
TARGET_CANDIDATES = [
    "capacity_factor",
    "active_power_kw",
    "ac_power",
    "power_kw",
    "generation_kw",
    "power",
    "target",
]

# Leakage Keyword Patterns (telemetry / contemporaneous inverter measurements)
LEAKAGE_KEYWORDS = [
    "dc_power",
    "dc_input",
    "daily_energy",
    "cumulative_energy",
    "yield",
    "efficiency",
    "is_generating",
    "inverter_status",
    "inverter_temp",
    "grid_freq",
    "power_factor",
    "pac",
    "pdc",
]

# Correlation threshold triggering potential leakage warning
LEAKAGE_CORRELATION_THRESHOLD = 0.95

# Semantic Aliases for Solar Forecasting Features
SEMANTIC_ALIASES = {
    "ghi": ["ghi", "ghi_wm2", "global_horizontal", "solar_radiation", "radiation", "irradiance", "poa_global"],
    "dni": ["dni", "dni_wm2", "direct_normal", "direct_radiation"],
    "dhi": ["dhi", "dhi_wm2", "diffuse_horizontal", "diffuse_radiation"],
    "temp": ["temp_c", "temp", "temperature", "ambient_temp", "tamb", "air_temperature"],
    "humidity": ["humidity_pct", "humidity", "rh", "relative_humidity"],
    "wind_speed": ["wind_speed_kmh", "wind_speed_ms", "wind_speed", "ws"],
    "cloud_cover": ["cloud_cover_pct", "cloud_cover", "clouds", "cloudiness"],
    "solar_elevation": ["solar_elevation_deg", "solar_elevation", "elevation", "sun_elevation", "altitude"],
    "solar_azimuth": ["solar_azimuth_deg", "solar_azimuth", "azimuth", "sun_azimuth"],
}

# Daylight Solar Elevation Cutoff for dual evaluation
DAYTIME_ELEVATION_THRESHOLD = 5.0  # degrees

# Chronological Partitioning Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Tuning Settings
RANDOM_STATE = 42
TUNING_N_ITER = 15
TUNING_CV_SPLITS = 3

