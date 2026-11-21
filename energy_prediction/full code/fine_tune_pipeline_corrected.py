"""
SURYOKTI — CORRECTED PREDICTION MODEL FINE-TUNING PIPELINE
---------------------------------------------------------
Purpose:
  * Tune prediction models using chronological cross-validation on development data.
  * Keep the latest chronological portion as an untouched final test set.
  * Apply solar output constraints consistently during every evaluation and prediction.
  * Compare Random Forest + XGBoost stacking with different meta-learners.
  * Save a model trained on development data (not on the final test data).

IMPORTANT:
  1. If your CSV contains a timestamp column, set TIMESTAMP_COLUMN below.
  2. If it does not, the script uses the existing row order. Make sure rows are
     already in chronological order before running it.
  3. This script expects the columns used by the current SURYOKTI demo dataset.
     For a new dataset, update BASE_FEATURES and the feature engineering function.
"""

import os
import warnings
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import TimeSeriesSplit, cross_validate, KFold
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "energy_prediction", "demo_datasets", "Solar_Weather_Data_Processed.csv"),
    os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "energy_prediction", "processed", "Solar_Weather_Data_Cleaned.csv"),
    os.path.join("datasets", "energy_prediction", "demo_datasets", "Solar_Weather_Data_Processed.csv"),
    os.path.join("datasets", "energy_prediction", "processed", "Solar_Weather_Data_Cleaned.csv"),
    os.path.join("Solar Weather Processed Data", "Solar_Weather_Data_Processed.csv"),
    os.path.join("Data Preprocessing", "Solar_Weather_Data_Cleaned.csv"),
]

# If you have a timestamp column, enter its exact cleaned name here, e.g. "Timestamp".
# If None, the script will retain the existing row order (which must be chronological).
TIMESTAMP_COLUMN = None

FINAL_TEST_FRACTION = 0.20
N_SPLITS = 5
RANDOM_STATE = 42
MODEL_DIR = (
    os.path.join(os.path.dirname(__file__), "..", "models")
    if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "models"))
    else (os.path.join("models", "energy_prediction") if os.path.exists(os.path.join("models", "energy_prediction")) else "models")
)

TARGET = "AC_Output"

BASE_FEATURES = [
    "GHI", "Temperature", "Relative_Humidity", "Wind_Speed",
    "Wind_Direction", "Hour", "Day", "Month", "SZ_Angle", "SA_Angle",
]


# =============================================================================
# DATA & FEATURE ENGINEERING
# =============================================================================

def load_data():
    data_path = next((p for p in DATA_PATH_CANDIDATES if os.path.exists(p)), None)
    if data_path is None:
        raise FileNotFoundError(
            "Could not find the input CSV. Checked:\n" + "\n".join(DATA_PATH_CANDIDATES)
        )

    data = pd.read_csv(data_path)
    data.columns = (
        data.columns.astype(str).str.strip().str.replace(" ", "_", regex=False)
    )

    if TARGET not in data.columns:
        raise ValueError(f"Target column '{TARGET}' is missing from the CSV.")

    missing = [c for c in BASE_FEATURES if c not in data.columns]
    if missing:
        raise ValueError(
            f"Required feature columns are missing: {missing}\n"
            "Update BASE_FEATURES and feature engineering for this dataset."
        )

    if TIMESTAMP_COLUMN:
        if TIMESTAMP_COLUMN not in data.columns:
            raise ValueError(f"Timestamp column '{TIMESTAMP_COLUMN}' not found.")
        data[TIMESTAMP_COLUMN] = pd.to_datetime(data[TIMESTAMP_COLUMN], errors="coerce")
        data = data.dropna(subset=[TIMESTAMP_COLUMN]).sort_values(TIMESTAMP_COLUMN)
    else:
        print(
            "NOTE: TIMESTAMP_COLUMN is None. Using CSV row order. "
            "Confirm that the rows are already chronological."
        )

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=BASE_FEATURES + [TARGET]).reset_index(drop=True)

    if len(data) < 50:
        raise ValueError(f"Only {len(data)} valid rows remain; too few for this workflow.")

    return data, data_path


def add_solar_features(data):
    d = data.copy()
    d["Hour_sin"] = np.sin(2 * np.pi * d["Hour"] / 24.0)
    d["Hour_cos"] = np.cos(2 * np.pi * d["Hour"] / 24.0)
    d["Month_sin"] = np.sin(2 * np.pi * d["Month"] / 12.0)
    d["Month_cos"] = np.cos(2 * np.pi * d["Month"] / 12.0)
    d["Wind_Dir_sin"] = np.sin(np.radians(d["Wind_Direction"]))
    d["Wind_Dir_cos"] = np.cos(np.radians(d["Wind_Direction"]))
    d["Effective_GHI"] = d["GHI"] * np.clip(
        np.cos(np.radians(d["SZ_Angle"])), 0, None
    )
    d["Thermal_Stress"] = d["GHI"] * d["Temperature"] / (d["Wind_Speed"] + 1.0)
    return d


FEATURES = BASE_FEATURES + [
    "Hour_sin", "Hour_cos", "Month_sin", "Month_cos",
    "Wind_Dir_sin", "Wind_Dir_cos", "Effective_GHI", "Thermal_Stress",
]


# =============================================================================
# CONSTRAINT-AWARE MODEL WRAPPER
# =============================================================================

class SolarOutputRegressor(BaseEstimator, RegressorMixin):
    """Wrap a regressor and enforce nonnegative, zero-nighttime predictions."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        prediction = np.asarray(self.estimator_.predict(X), dtype=float)
        ghi = np.asarray(X["GHI"], dtype=float)
        return np.where(ghi <= 0, 0.0, np.clip(prediction, 0.0, None))


def metrics(y_true, y_pred):
    return {
        "MAE_kW": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_kW": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def solar_mae_scorer(estimator, X, y):
    return -mean_absolute_error(y, estimator.predict(X))


def solar_rmse_scorer(estimator, X, y):
    return -np.sqrt(mean_squared_error(y, estimator.predict(X)))


def solar_r2_scorer(estimator, X, y):
    return r2_score(y, estimator.predict(X))


def cv_evaluate(model, X, y, cv):
    """All CV scores use SolarOutputRegressor.predict(), including constraints."""
    scoring = {
        "MAE": solar_mae_scorer,
        "RMSE": solar_rmse_scorer,
        "R2": solar_r2_scorer,
    }
    scores = cross_validate(
        model, X, y, cv=cv, scoring=scoring, n_jobs=1,
        return_train_score=False, error_score="raise"
    )
    return {
        "CV_MAE_kW": -scores["test_MAE"].mean(),
        "CV_MAE_std": scores["test_MAE"].std(),
        "CV_RMSE_kW": -scores["test_RMSE"].mean(),
        "CV_RMSE_std": scores["test_RMSE"].std(),
        "CV_R2": scores["test_R2"].mean(),
        "CV_R2_std": scores["test_R2"].std(),
    }


def make_base_models():
    rf = RandomForestRegressor(
        n_estimators=250, max_depth=7, min_samples_split=8,
        min_samples_leaf=5, max_features=0.75,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb = XGBRegressor(
        n_estimators=200, learning_rate=0.03, max_depth=6,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.5,
        objective="reg:squarederror", eval_metric="rmse",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    return rf, xgb


def make_nn(hidden_layers=(64, 32), alpha=0.05, learning_rate=0.003):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("nn", MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu", solver="adam",
            alpha=alpha, learning_rate_init=learning_rate,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])


def make_stack(rf, xgb, meta_learner, inner_cv):
    # StackingRegressor creates out-of-fold base-model predictions for meta fitting.
    return SolarOutputRegressor(
        StackingRegressor(
            estimators=[("rf", clone(rf)), ("xgb", clone(xgb))],
            final_estimator=clone(meta_learner),
            passthrough=True,
            cv=inner_cv,
            n_jobs=1,
        )
    )


# =============================================================================
# MAIN WORKFLOW — 8 STAGES
# =============================================================================

def main():
    df, data_path = load_data()
    df = add_solar_features(df)

    X = df[FEATURES].copy()
    y = df[TARGET].astype(float).copy()

    # Final test is the latest 20% of rows; do not use it to choose configurations.
    split_at = int(len(df) * (1.0 - FINAL_TEST_FRACTION))
    X_dev, X_test = X.iloc[:split_at].copy(), X.iloc[split_at:].copy()
    y_dev, y_test = y.iloc[:split_at].copy(), y.iloc[split_at:].copy()

    if len(X_dev) <= N_SPLITS + 5:
        raise ValueError("Development set is too small for the selected CV split count.")

    outer_tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    # Inner stack CV: standard KFold creates OOF predictions over all training rows.
    # Outer TimeSeriesSplit is used for the future-oriented evaluation.
    inner_kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n" + "=" * 78)
    print("SURYOKTI — PREDICTION MODEL FINE-TUNING")
    print("=" * 78)
    print(f"Input file: {data_path}")
    print(f"Valid rows: {len(df)} | Development rows: {len(X_dev)} | Final test rows: {len(X_test)}")
    print("Selection: chronological CV on development data")
    print("Final test: latest chronological segment; reserved until model selection")

    rf_default, xgb_default = make_base_models()

    # Stage 1 — baseline stack
    print("\nSTAGE 1 — BASELINE STACK (RF + XGBoost -> MLP)")
    baseline_meta = make_nn()
    baseline_stack = make_stack(rf_default, xgb_default, baseline_meta, inner_kfold)
    baseline_cv = cv_evaluate(baseline_stack, X_dev, y_dev, outer_tscv)
    print(pd.DataFrame([{"Architecture": "RF + XGB -> MLP", **baseline_cv}]).to_string(index=False))

    # Stage 2 — controlled meta learner comparison; same base models and folds
    print("\nSTAGE 2 — CONTROLLED STACKING COMPARISON")
    stage2_candidates = {
        "RF + XGB -> LinearRegression": LinearRegression(),
        "RF + XGB -> RidgeCV": RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
        "RF + XGB -> MLP": make_nn(),
    }
    stage2_results = []
    for name, meta in stage2_candidates.items():
        model = make_stack(rf_default, xgb_default, meta, inner_kfold)
        stage2_results.append({"Architecture": name, **cv_evaluate(model, X_dev, y_dev, outer_tscv)})
    stage2_df = pd.DataFrame(stage2_results).sort_values("CV_R2", ascending=False)
    print(stage2_df.to_string(index=False))

    # Stage 3 — XGBoost configurations; evaluate on development CV only
    print("\nSTAGE 3 — XGBOOST TUNING")
    xgb_candidates = {
        "XGB_Deep": XGBRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=6, min_child_weight=3,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.5,
            objective="reg:squarederror", eval_metric="rmse", random_state=RANDOM_STATE, n_jobs=-1),
        "XGB_Compact": XGBRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=4, min_child_weight=4,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=2.0,
            objective="reg:squarederror", eval_metric="rmse", random_state=RANDOM_STATE, n_jobs=-1),
        "XGB_Shallow": XGBRegressor(
            n_estimators=300, learning_rate=0.08, max_depth=3, min_child_weight=2,
            subsample=0.9, colsample_bytree=0.9, reg_alpha=0.05, reg_lambda=1.0,
            objective="reg:squarederror", eval_metric="rmse", random_state=RANDOM_STATE, n_jobs=-1),
    }
    xgb_results = []
    for name, candidate in xgb_candidates.items():
        wrapped = SolarOutputRegressor(clone(candidate))
        result = cv_evaluate(wrapped, X_dev, y_dev, outer_tscv)
        xgb_results.append({"Configuration": name, **result})
    xgb_df = pd.DataFrame(xgb_results).sort_values("CV_R2", ascending=False)
    print(xgb_df.to_string(index=False))
    best_xgb_name = xgb_df.iloc[0]["Configuration"]
    best_xgb = clone(xgb_candidates[best_xgb_name])
    print(f"Selected by development CV: {best_xgb_name}")

    # Stage 4 — Random Forest configurations
    print("\nSTAGE 4 — RANDOM FOREST TUNING")
    rf_candidates = {
        "RF_Deep": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=4, min_samples_leaf=2,
            max_features=0.8, random_state=RANDOM_STATE, n_jobs=-1),
        "RF_Constrained": RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_split=6, min_samples_leaf=3,
            max_features=0.8, random_state=RANDOM_STATE, n_jobs=-1),
        "RF_Regularized": RandomForestRegressor(
            n_estimators=250, max_depth=7, min_samples_split=8, min_samples_leaf=5,
            max_features=0.75, random_state=RANDOM_STATE, n_jobs=-1),
    }
    rf_results = []
    for name, candidate in rf_candidates.items():
        result = cv_evaluate(SolarOutputRegressor(clone(candidate)), X_dev, y_dev, outer_tscv)
        rf_results.append({"Configuration": name, **result})
    rf_df = pd.DataFrame(rf_results).sort_values("CV_R2", ascending=False)
    print(rf_df.to_string(index=False))
    best_rf_name = rf_df.iloc[0]["Configuration"]
    best_rf = clone(rf_candidates[best_rf_name])
    print(f"Selected by development CV: {best_rf_name}")

    # Stage 5 — MLP meta-learner tuning using the selected RF/XGB base estimators
    print("\nSTAGE 5 — NEURAL NETWORK META-LEARNER TUNING")
    nn_candidates = {
        "NN_Balanced_64_32": make_nn((64, 32), alpha=0.05, learning_rate=0.003),
        "NN_Wide_128_64": make_nn((128, 64), alpha=0.1, learning_rate=0.002),
        "NN_Deep_64_32_16": make_nn((64, 32, 16), alpha=0.08, learning_rate=0.003),
    }
    nn_results = []
    for name, meta in nn_candidates.items():
        model = make_stack(best_rf, best_xgb, meta, inner_kfold)
        nn_results.append({"Meta_Architecture": name, **cv_evaluate(model, X_dev, y_dev, outer_tscv)})
    nn_df = pd.DataFrame(nn_results).sort_values("CV_R2", ascending=False)
    print(nn_df.to_string(index=False))
    best_nn_name = nn_df.iloc[0]["Meta_Architecture"]
    best_nn = clone(nn_candidates[best_nn_name])
    print(f"Selected by development CV: {best_nn_name}")

    # Stage 6 — compare meta learners using the same tuned base models and folds
    print("\nSTAGE 6 — META-LEARNER COMPARISON")
    meta_candidates = {
        "LinearRegression": LinearRegression(),
        "RidgeCV": RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
        "ElasticNetCV": ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9], cv=5, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=80, max_depth=3, random_state=RANDOM_STATE),
        f"Tuned MLP ({best_nn_name})": best_nn,
    }
    meta_results = []
    for name, meta in meta_candidates.items():
        model = make_stack(best_rf, best_xgb, meta, inner_kfold)
        meta_results.append({"Meta_Learner": name, **cv_evaluate(model, X_dev, y_dev, outer_tscv)})
    meta_df = pd.DataFrame(meta_results).sort_values("CV_R2", ascending=False)
    print(meta_df.to_string(index=False))

    # Stage 7 — temporal generalization table of contenders.
    # This is the selection CV, not the final test set.
    print("\nSTAGE 7 — TEMPORAL GENERALIZATION (DEVELOPMENT DATA ONLY)")
    contenders = {
        "RidgeCV": meta_candidates["RidgeCV"],
        "LinearRegression": meta_candidates["LinearRegression"],
        f"Tuned MLP ({best_nn_name})": best_nn,
    }
    temporal_results = []
    for name, meta in contenders.items():
        model = make_stack(best_rf, best_xgb, meta, inner_kfold)
        temporal_results.append({"Meta_Learner": name, **cv_evaluate(model, X_dev, y_dev, outer_tscv)})
    temporal_df = pd.DataFrame(temporal_results).sort_values("CV_R2", ascending=False)
    print(temporal_df.to_string(index=False))

    # Stage 8 — select by chronological CV R2, then fit on development rows only.
    # The final test is evaluated exactly once after model selection.
    print("\nSTAGE 8 — SELECT, FIT, AND FINAL TEST")
    best_meta_name = meta_df.iloc[0]["Meta_Learner"]
    selected_meta = clone(meta_candidates[best_meta_name])
    selected_stack = make_stack(best_rf, best_xgb, selected_meta, inner_kfold)

    selected_stack.fit(X_dev, y_dev)
    final_predictions = selected_stack.predict(X_test)
    final_metrics = metrics(y_test, final_predictions)

    print(f"Selected architecture: RF ({best_rf_name}) + XGBoost ({best_xgb_name}) -> {best_meta_name}")
    print("Untouched chronological final-test metrics:")
    print(pd.DataFrame([final_metrics]).to_string(index=False))

    # Save the fitted model. It remains trained on development data, preserving
    # the independent final-test evaluation. Retrain on all data only after
    # reporting this evaluation and when preparing a deployment version.
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "fine_tuned_final_stack.pkl")
    features_path = os.path.join(MODEL_DIR, "fine_tuned_features.pkl")
    metadata_path = os.path.join(MODEL_DIR, "fine_tuned_metadata.json")

    joblib.dump(selected_stack, model_path)
    joblib.dump(FEATURES, features_path)

    metadata = {
        "data_file": data_path,
        "target": TARGET,
        "feature_columns": FEATURES,
        "development_rows": int(len(X_dev)),
        "final_test_rows": int(len(X_test)),
        "selected_rf": best_rf_name,
        "selected_xgb": best_xgb_name,
        "selected_meta_learner": best_meta_name,
        "final_test_metrics": final_metrics,
        "timestamp_column": TIMESTAMP_COLUMN,
        "note": "Model fit on development data; final chronological test kept independent.",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model: {model_path}")
    print(f"Saved features: {features_path}")
    print(f"Saved metadata: {metadata_path}")
    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
