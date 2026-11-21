"""
Standalone Inference Script for Solar PV AC Power Output Forecasting
Loads trained model artifacts from models/ and predicts AC Output.
"""

import os
import joblib
import numpy as np
import pandas as pd


def load_artifacts(model_dir=None):
    if model_dir is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "models"),
            "models/energy_prediction",
            "models",
        ]
        for c in candidates:
            if os.path.exists(os.path.join(c, "best_model.pkl")):
                model_dir = c
                break
        if model_dir is None:
            model_dir = "models"
    model_path = os.path.join(model_dir, "best_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    cols_path = os.path.join(model_dir, "feature_cols.pkl")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, cols_path]):
        raise FileNotFoundError("Model artifacts missing. Run train_test.py first to train and export the model.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(cols_path)
    return model, scaler, feature_cols


def predict_ac_power(input_df: pd.DataFrame, model_dir="models", verbose: bool = True) -> np.ndarray:
    """
    Given a DataFrame with weather variables, returns predicted PV AC Power Output in kW.
    
    Parameters:
        input_df: DataFrame containing input weather features.
        model_dir: Directory containing trained model artifacts.
        verbose: If True, previews the exact feature columns and test rows passed into the model.
    """
    model, scaler, feature_cols = load_artifacts(model_dir)

    df = input_df.copy()

    # Engineer diurnal cyclical features if not present
    if "Hour_sin" not in df.columns or "Hour_cos" not in df.columns:
        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)

    # Ensure feature alignment
    X = df[feature_cols]

    if verbose:
        print("\n================ PREVIEW: INPUT ROWS & COLUMNS USED BY MODEL ================")
        print(f"Loaded Model Type      : {type(model).__name__}")
        print(f"Total Rows to Predict  : {len(X)}")
        print(f"Total Feature Columns  : {len(feature_cols)}")
        print(f"Feature Column List    : {feature_cols}\n")
        print("Feature Matrix Preview (First 5 Rows taken by model):")
        print(X.head(5).to_string())
        print("=============================================================================\n")

    # Predict
    if type(model).__name__ in ["LinearRegression", "Ridge"]:
        X_scaled = scaler.transform(X)
        raw_pred = model.predict(X_scaled)
    else:
        raw_pred = model.predict(X)

    # Apply physical constraints (no negative predictions, zero output when GHI is 0)
    pred_constrained = np.clip(raw_pred, 0, None)
    if "GHI" in df.columns:
        pred_constrained[df["GHI"].values == 0] = 0.0

    return pred_constrained


def load_test_data(csv_path=None, test_ratio=0.2, random_state=42) -> pd.DataFrame:
    """
    Loads solar weather dataset and returns the 20% test data split matching train_test.py.
    """
    if csv_path is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "energy_prediction", "demo_datasets", "Solar_Weather_Data_Processed.csv"),
            os.path.join("datasets", "energy_prediction", "demo_datasets", "Solar_Weather_Data_Processed.csv"),
            os.path.join("Solar Weather Processed Data", "Solar_Weather_Data_Processed.csv"),
            "Solar_Weather_Data_Cleaned.csv",
        ]
        csv_path = next((c for c in candidates if os.path.exists(c)), candidates[0])

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    if random_state is not None:
        from sklearn.model_selection import train_test_split
        _, test_df = train_test_split(df, test_size=test_ratio, random_state=random_state)
        test_df = test_df.reset_index(drop=True)
    else:
        # Chronological split (last test_ratio portion of dataset)
        split_idx = int(len(df) * (1 - test_ratio))
        test_df = df.iloc[split_idx:].reset_index(drop=True)

    return test_df


if __name__ == "__main__":
    print("--- Solar PV AC Output Prediction on 20% Test Data ---")

    # 1. Load 20% test dataset
    test_df = load_test_data(test_ratio=0.2, random_state=42)
    print(f"Loaded 20% Test Dataset: {len(test_df)} samples")

    # 2. Predict on 20% test data with verbose feature & row preview enabled
    test_df["Predicted_AC_Output"] = predict_ac_power(test_df, verbose=True)

    # 3. Evaluate performance on 20% test set if target column exists
    if "AC_Output" in test_df.columns:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        y_true = test_df["AC_Output"]
        y_pred = test_df["Predicted_AC_Output"]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        print("=== Test Data Evaluation Metrics (20% Split) ===")
        print(f"Mean Absolute Error (MAE) : {mae:.4f} kW")
        print(f"Root Mean Squared Error   : {rmse:.4f} kW")
        print(f"R² Score                  : {r2:.4f}\n")

    # 4. Display sample test set predictions with row index and features
    print("=== Sample Test Set Predictions (First 10 Rows) ===")
    display_cols = ["GHI", "Temperature", "Hour"]
    if "AC_Output" in test_df.columns:
        display_cols.append("AC_Output")
    display_cols.append("Predicted_AC_Output")

    available_cols = [c for c in display_cols if c in test_df.columns]
    print(test_df[available_cols].head(10).to_string(index=True))

    # 5. Single sample prediction demonstration
    print("\n=== Single Sample Prediction Demonstration ===")
    sample_weather = pd.DataFrame([{
        "GHI": 800.0,
        "Temperature": 25.0,
        "Relative_Humidity": 60.0,
        "Wind_Speed": 3.0,
        "Wind_Direction": 180.0,
        "Hour": 12,
        "Day": 15,
        "Month": 10,
        "SZ_Angle": 30.0,
        "SA_Angle": 180.0
    }])

    single_pred = predict_ac_power(sample_weather)
    print(f"Input Irradiance (GHI): {sample_weather['GHI'].iloc[0]} W/m²")
    print(f"Input Temperature:     {sample_weather['Temperature'].iloc[0]} °C")
    print(f"Predicted AC Output:   {single_pred[0]:.2f} kW")


