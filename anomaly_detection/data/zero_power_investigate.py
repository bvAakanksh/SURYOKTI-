
import pandas as pd

from pathlib import Path
candidates = [
    Path(__file__).resolve().parents[2] / "datasets" / "energy_prediction" / "demo_datasets" / "Solar_Weather_Data_Processed.csv",
    Path("datasets/energy_prediction/demo_datasets/Solar_Weather_Data_Processed.csv"),
    Path("Solar Weather Processed Data/Solar_Weather_Data_Processed.csv"),
]
file_path = next((str(c) for c in candidates if c.exists()), str(candidates[0]))

df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Convert timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Sort by time
df = df.sort_values("Timestamp")

# 1. Count zero-output rows by GHI range
df["GHI_Range"] = pd.cut(
    df["GHI"],
    bins=[-1, 10, 100, 300, 600, 1200],
    labels=[
        "Very Low",
        "Low",
        "Moderate",
        "High",
        "Very High"
    ]
)

print("\n--- Zero Output by GHI Range ---")

zero_output = df[df["AC_Output"] == 0]

print(
    zero_output["GHI_Range"].value_counts()
    .sort_index()
)

# 2. Count zero-output rows by hour
print("\n--- Zero Output by Hour ---")

print(
    zero_output["Hour"].value_counts()
    .sort_index()
)

# 3. Show suspicious observations
print("\n--- Zero Output with GHI > 100 ---")

suspicious = df[
    (df["GHI"] > 100) &
    (df["AC_Output"] == 0)
]

print(
    suspicious[
        ["Timestamp", "GHI", "Temperature", "AC_Output"]
    ].head(20)
)

# 4. Show count
print("\nTotal suspicious observations:", len(suspicious))
