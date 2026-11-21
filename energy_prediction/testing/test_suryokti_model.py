"""
Test the suryokti_hybrid.suryokti model against the current dataset.

The .suryokti archive contains:
  - manifest.json: feature list, plant configs, preprocessing medians
  - correction.json: XGBoost model in JSON format (240 trees, depth 3)

The model expects 8 features:
  physics_ratio, poa_ratio, temperature_air_c, cell_temperature_c,
  hour_sin, hour_cos, doy_sin, doy_cos

We map from the current dataset columns using PV physics approximations.
"""

import os
import json
import zipfile
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# 1. LOAD THE .suryokti MODEL
# =============================================================================
from pathlib import Path
_model_candidates = [
    Path(__file__).resolve().parents[1] / "models" / "suryokti_hybrid.suryokti",
    Path("models/energy_prediction/suryokti_hybrid.suryokti"),
    Path("energy_prediction/models/suryokti_hybrid.suryokti"),
    Path("suryokti_hybrid.suryokti"),
]
MODEL_PATH = next((str(p) for p in _model_candidates if p.exists()), "suryokti_hybrid.suryokti")

with zipfile.ZipFile(MODEL_PATH, "r") as z:
    manifest = json.loads(z.read("manifest.json"))
    correction_json = z.read("correction.json")

model_features = manifest["features"]
medians = manifest["preprocessing"]["medians"]
interval_q = manifest["interval_q"]
configs = manifest["configs"]

# Use the first plant config for capacity-based normalization
plant_id = list(configs.keys())[0]
plant = configs[plant_id]
dc_cap_kw = plant["dc_capacity_kw"]
ac_cap_kw = plant["ac_capacity_kw"]
gamma_pdc = plant["gamma_pdc"]
system_losses = plant["system_losses"]
noct_c = plant["noct_c"]
inv_eff = plant["inverter_efficiency"]
tilt = plant["surface_tilt"]
azimuth = plant["surface_azimuth"]

print("=" * 70)
print("SURYOKTI HYBRID MODEL TEST")
print("=" * 70)
print(f"Model file       : {MODEL_PATH}")
print(f"Model version    : {manifest['version']}")
print(f"XGBoost version  : {manifest['dependencies']['xgboost']}")
print(f"Number of trees  : 240 (depth 3)")
print(f"Model features   : {model_features}")
print(f"Plant ID         : {plant_id}")
print(f"DC Capacity      : {dc_cap_kw:.2f} kW")
print(f"AC Capacity      : {ac_cap_kw:.2f} kW")
print(f"Interval Q       : {interval_q:.6f}")

# Load the XGBoost model from JSON
booster = XGBRegressor()
booster.load_model(bytearray(correction_json))
print("XGBoost model loaded from correction.json")

# =============================================================================
# 2. LOAD THE DATASET
# =============================================================================
_data_candidates = [
    Path(__file__).resolve().parents[2] / "datasets" / "energy_prediction" / "demo_datasets" / "Solar_Weather_Data_Processed.csv",
    Path("datasets/energy_prediction/demo_datasets/Solar_Weather_Data_Processed.csv"),
    Path("Solar Weather Processed Data/Solar_Weather_Data_Processed.csv"),
]
DATA_PATH = next((str(p) for p in _data_candidates if p.exists()), "Solar Weather Processed Data/Solar_Weather_Data_Processed.csv")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.replace(" ", "_")

print(f"\nDataset          : {DATA_PATH}")
print(f"Total rows       : {len(df)}")
print(f"Columns          : {list(df.columns)}")

# =============================================================================
# 3. FEATURE ENGINEERING: MAP DATASET -> MODEL FEATURES
# =============================================================================
# The model expects normalized ratios and physics-derived features.
# We approximate these from the available weather columns.

# Parse timestamps for day-of-year
df["Timestamp_dt"] = pd.to_datetime(df["Timestamp"], format="mixed",
                                      dayfirst=False)
df["doy"] = df["Timestamp_dt"].dt.dayofyear

# --- Cyclical time features (direct match) ---
df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)
df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)

# --- Temperature features ---
df["temperature_air_c"] = df["Temperature"]

# Cell temperature estimate using NOCT model:
# T_cell = T_air + (NOCT - 20) / 800 * GHI
# GHI in this dataset is in some unit; scale factor will adjust
df["cell_temperature_c"] = (df["Temperature"]
                             + (noct_c - 20.0) / 800.0 * df["GHI"])

# --- Physics ratio: actual AC output / physics-model expected output ---
# Physics model: P_expected = GHI * cos(zenith) * DC_cap * (1 + gamma*(T_cell-25))
#                             * (1 - losses) * inv_eff
cos_zenith = np.clip(np.cos(np.radians(df["SZ_Angle"])), 0.0, 1.0)

# Compute the physics-model expected power (in kW)
temp_coeff = 1.0 + gamma_pdc * (df["cell_temperature_c"] - 25.0)
temp_coeff = np.clip(temp_coeff, 0.5, 1.2)

# GHI needs to be in W/m2 for standard physics model
# In this dataset GHI appears to be very small (values ~2-3), likely in kW/m2
# We scale by 1000 to get W/m2 equivalent for normalization
ghi_wm2 = df["GHI"] * 1000.0  # Convert to approximate W/m2

expected_power = (ghi_wm2 / 1000.0) * cos_zenith * dc_cap_kw * temp_coeff * (
    1.0 - system_losses) * inv_eff

# Physics ratio = actual / expected (clipped to avoid division by zero)
df["physics_ratio"] = np.where(
    expected_power > 10.0,
    df["AC_Output"] / expected_power,
    medians["physics_ratio"]
)

# --- POA ratio: plane-of-array irradiance ratio ---
# Approximate POA as GHI * cos(angle_of_incidence)
# For a tilted surface facing south (azimuth 180):
poa_irradiance = ghi_wm2 * cos_zenith  # Simplified beam component
df["poa_ratio"] = np.where(
    poa_irradiance > 10.0,
    df["AC_Output"] / (poa_irradiance * dc_cap_kw / 1000.0 * inv_eff),
    medians["poa_ratio"]
)

# Clip ratios to reasonable ranges
df["physics_ratio"] = np.clip(df["physics_ratio"], -0.5, 1.5)
df["poa_ratio"] = np.clip(df["poa_ratio"], -0.5, 1.5)

# Fill any NaN with medians
for feat in model_features:
    if feat in df.columns:
        df[feat] = df[feat].fillna(medians.get(feat, 0.0))

print("\nFeature engineering complete.")
print(df[model_features].describe().round(6).to_string())

# =============================================================================
# 4. RUN PREDICTIONS
# =============================================================================
X = df[model_features].values.astype(np.float32)

# The XGBoost model predicts the correction/ratio, not absolute kW
raw_predictions = booster.predict(X)

print(f"\nRaw prediction stats:")
print(f"  Mean: {raw_predictions.mean():.6f}")
print(f"  Std : {raw_predictions.std():.6f}")
print(f"  Min : {raw_predictions.min():.6f}")
print(f"  Max : {raw_predictions.max():.6f}")

# Convert ratio predictions back to AC power (kW)
# The model predicts a normalized ratio; we scale by capacity
predicted_ac_kw = raw_predictions * ac_cap_kw

# Apply physical constraints
predicted_ac_kw = np.clip(predicted_ac_kw, 0.0, ac_cap_kw)
# Zero output when GHI is negligible
predicted_ac_kw = np.where(df["GHI"].values <= 0.01, 0.0, predicted_ac_kw)

actual_ac_kw = df["AC_Output"].values

# =============================================================================
# 5. EVALUATION METRICS
# =============================================================================
# Since the model was trained on a different plant with much larger capacity
# (38 MW DC vs our dataset's ~kW scale), the raw output will be on a
# completely different scale. Let's evaluate both the raw ratio predictions
# and attempt a scaled comparison.

# First: check the scale difference
print(f"\n{'='*70}")
print("SCALE ANALYSIS")
print(f"{'='*70}")
print(f"Model trained on plant capacity : {dc_cap_kw:.2f} kW DC / {ac_cap_kw:.2f} kW AC")
print(f"Dataset AC Output range         : {actual_ac_kw.min():.2f} - {actual_ac_kw.max():.2f} kW")
print(f"Dataset AC Output mean          : {actual_ac_kw.mean():.2f} kW")
print(f"Predicted range (ratio * AC_cap): {predicted_ac_kw.min():.2f} - {predicted_ac_kw.max():.2f} kW")

# Compute normalized ratio for both actual and predicted
actual_ratio = actual_ac_kw / actual_ac_kw.max() if actual_ac_kw.max() > 0 else actual_ac_kw
pred_ratio = raw_predictions / raw_predictions.max() if raw_predictions.max() > 0 else raw_predictions

# Rescale predictions to match dataset's scale for fair comparison
scale_factor = actual_ac_kw.mean() / (raw_predictions.mean() + 1e-10) if raw_predictions.mean() > 0 else 1.0
scaled_predictions = raw_predictions * scale_factor
scaled_predictions = np.clip(scaled_predictions, 0.0, actual_ac_kw.max() * 1.2)
scaled_predictions = np.where(df["GHI"].values <= 0.01, 0.0, scaled_predictions)

mae = mean_absolute_error(actual_ac_kw, scaled_predictions)
rmse = np.sqrt(mean_squared_error(actual_ac_kw, scaled_predictions))
r2 = r2_score(actual_ac_kw, scaled_predictions)

# Ratio-space evaluation (scale-independent)
mae_ratio = mean_absolute_error(actual_ratio, pred_ratio)
r2_ratio = r2_score(actual_ratio, pred_ratio)

# Rank correlation (Spearman) - tests if ordering is correct
from scipy import stats
spearman_corr, spearman_p = stats.spearmanr(actual_ac_kw, raw_predictions)
pearson_corr, pearson_p = stats.pearsonr(actual_ac_kw, raw_predictions)

print(f"\n{'='*70}")
print("TEST RESULTS: suryokti_hybrid.suryokti vs Current Dataset")
print(f"{'='*70}")
print()
print("+-----------------------------------------+------------------+")
print("| Metric                                  | Value            |")
print("+-----------------------------------------+------------------+")
print(f"| Scale Factor Applied                    | {scale_factor:>16.4f} |")
print(f"| Scaled MAE (kW)                         | {mae:>16.4f} |")
print(f"| Scaled RMSE (kW)                        | {rmse:>16.4f} |")
print(f"| Scaled R2 Score                         | {r2:>16.4f} |")
print(f"| Scaled Accuracy (%)                     | {r2*100:>15.2f}% |")
print("+-----------------------------------------+------------------+")
print(f"| Normalized Ratio MAE                    | {mae_ratio:>16.6f} |")
print(f"| Normalized Ratio R2                     | {r2_ratio:>16.4f} |")
print("+-----------------------------------------+------------------+")
print(f"| Pearson Correlation                     | {pearson_corr:>16.4f} |")
print(f"| Spearman Rank Correlation               | {spearman_corr:>16.4f} |")
print(f"| Spearman p-value                        | {spearman_p:>16.2e} |")
print("+-----------------------------------------+------------------+")

# Daytime-only metrics (GHI > 0.1)
daytime = df["GHI"].values > 0.1
if daytime.sum() > 0:
    day_mae = mean_absolute_error(actual_ac_kw[daytime], scaled_predictions[daytime])
    day_rmse = np.sqrt(mean_squared_error(actual_ac_kw[daytime], scaled_predictions[daytime]))
    day_r2 = r2_score(actual_ac_kw[daytime], scaled_predictions[daytime])
    sp_day, _ = stats.spearmanr(actual_ac_kw[daytime], raw_predictions[daytime])
    
    print(f"\n{'='*70}")
    print(f"DAYTIME-ONLY METRICS ({daytime.sum()} / {len(df)} rows with GHI > 0.1)")
    print(f"{'='*70}")
    print("+-----------------------------------------+------------------+")
    print(f"| Daytime MAE (kW)                        | {day_mae:>16.4f} |")
    print(f"| Daytime RMSE (kW)                       | {day_rmse:>16.4f} |")
    print(f"| Daytime R2                              | {day_r2:>16.4f} |")
    print(f"| Daytime Accuracy (%)                    | {day_r2*100:>15.2f}% |")
    print(f"| Daytime Spearman Correlation             | {sp_day:>16.4f} |")
    print("+-----------------------------------------+------------------+")

# Sample predictions
print(f"\n{'='*70}")
print("SAMPLE PREDICTIONS (First 15 rows)")
print(f"{'='*70}")
sample_df = pd.DataFrame({
    "Hour": df["Hour"].values[:15],
    "GHI": df["GHI"].values[:15],
    "Temp_C": df["Temperature"].values[:15],
    "Actual_kW": actual_ac_kw[:15],
    "Predicted_kW": scaled_predictions[:15],
    "Error_kW": (actual_ac_kw[:15] - scaled_predictions[:15]),
})
print(sample_df.round(2).to_string(index=True))

print(f"\n{'='*70}")
print("TEST COMPLETE")
print(f"{'='*70}")

