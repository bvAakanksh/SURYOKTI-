
import pandas as pd

# Step 1: Load the dataset
from pathlib import Path
candidates = [
    Path(__file__).resolve().parents[2] / "datasets" / "energy_prediction" / "demo_datasets" / "Solar_Weather_Data_Processed.csv",
    Path("datasets/energy_prediction/demo_datasets/Solar_Weather_Data_Processed.csv"),
    Path("Solar Weather Processed Data/Solar_Weather_Data_Processed.csv"),
]
file_path = next((str(c) for c in candidates if c.exists()), str(candidates[0]))

df = pd.read_csv(file_path)

# Step 2: Display the first 5 rows
print("\n--- First 5 Rows ---")
print(df.head())

# Step 3: Display dataset dimensions
print("\n--- Dataset Shape ---")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Step 4: Display all column names
print("\n--- Column Names ---")
print(df.columns.tolist())

# Step 5: Check data types
print("\n--- Data Types ---")
print(df.dtypes)

# Step 6: Check missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Step 7: Statistical summary
print("\n--- Statistical Summary ---")
print(df.describe())
