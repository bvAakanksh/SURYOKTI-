
import pandas as pd

# Step 1: Load dataset
from pathlib import Path
candidates = [
    Path(__file__).resolve().parents[2] / "datasets" / "energy_prediction" / "demo_datasets" / "Solar_Weather_Data_Processed.csv",
    Path("datasets/energy_prediction/demo_datasets/Solar_Weather_Data_Processed.csv"),
    Path("Solar Weather Processed Data/Solar_Weather_Data_Processed.csv"),
]
file_path = next((str(c) for c in candidates if c.exists()), str(candidates[0]))

df = pd.read_csv(file_path)

# Step 2: Clean column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")

print("Cleaned Columns:")
print(df.columns.tolist())

# Step 3: Convert Timestamp into datetime format
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

# Step 4: Sort data chronologically
df = df.sort_values("Timestamp")

# Step 5: Check duplicate timestamps
print("\nDuplicate Timestamps:")
print(df["Timestamp"].duplicated().sum())

# Step 6: Check invalid timestamps
print("\nInvalid Timestamps:")
print(df["Timestamp"].isnull().sum())

# Step 7: Check important feature ranges
features = [
    "GHI",
    "Temperature",
    "Relative_Humidity",
    "Wind_Speed",
    "Wind_Direction",
    "AC_Output"
]

print("\nFeature Ranges:")
print(df[features].agg(["min", "max", "mean"]))

# Step 8: Check zero power generation
print("\nZero AC Output Rows:")
print((df["AC_Output"] == 0).sum())

# Step 9: Check power output when sunlight is available
print("\nRows with GHI > 100 and AC Output = 0:")
print(
    ((df["GHI"] > 100) & (df["AC_Output"] == 0)).sum()
)

# Step 10: Check whether all rows are unique
print("\nDuplicate Rows:")
print(df.duplicated().sum())
