
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

# Sort chronologically
df = df.sort_values("Timestamp")

# 1. Compare Timestamp hour with the existing Hour feature
df["Timestamp_Hour"] = df["Timestamp"].dt.hour

print("\n--- Timestamp Hour vs Dataset Hour ---")

print(
    (df["Timestamp_Hour"] != df["Hour"]).sum(),
    "rows have mismatched hours"
)

# 2. Display the first 30 observations
print("\n--- First 30 Observations ---")

print(
    df[
        ["Timestamp", "Timestamp_Hour", "Hour", "GHI", "AC_Output"]
    ].head(30).to_string(index=False)
)

# 3. Average GHI and AC output for every hour
print("\n--- Hourly GHI and AC Output Summary ---")

hourly_summary = df.groupby("Hour").agg(
    GHI_Mean=("GHI", "mean"),
    GHI_Max=("GHI", "max"),
    AC_Output_Mean=("AC_Output", "mean"),
    Zero_Output_Count=("AC_Output", lambda x: (x == 0).sum()),
    Total_Observations=("AC_Output", "count")
)

print(hourly_summary)

# 4. Display observations where GHI is high but output is zero
print("\n--- High GHI with Zero Output ---")

print(
    df[
        (df["GHI"] > 100) &
        (df["AC_Output"] == 0)
    ][
        ["Timestamp", "Hour", "GHI", "AC_Output"]
    ].head(30).to_string(index=False)
)



#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#


import pandas as pd

from pathlib import Path
candidates = [
    Path(__file__).resolve().parents[2] / "datasets" / "energy_prediction" / "demo_datasets" / "Solar_Weather_Data_Processed.csv",
    Path("datasets/energy_prediction/demo_datasets/Solar_Weather_Data_Processed.csv"),
    Path("Solar Weather Processed Data/Solar_Weather_Data_Processed.csv"),
]
file_path = next((str(c) for c in candidates if c.exists()), str(candidates[0]))

df = pd.read_csv(file_path)

df.columns = df.columns.str.strip().str.replace(" ", "_")

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df["Timestamp_Hour"] = df["Timestamp"].dt.hour

print("\n--- Hour Column Data Type ---")
print(df["Hour"].dtype)

print("\n--- Timestamp_Hour Data Type ---")
print(df["Timestamp_Hour"].dtype)

print("\n--- First 10 Values of Hour ---")
print(df["Hour"].head(10).tolist())

print("\n--- First 10 Values of Timestamp_Hour ---")
print(df["Timestamp_Hour"].head(10).tolist())

# Convert Hour to numeric only for comparison
hour_numeric = pd.to_numeric(df["Hour"], errors="coerce")

mismatch = df["Timestamp_Hour"] != hour_numeric

print("\n--- Number of Mismatches ---")
print(mismatch.sum())

print("\n--- First 20 Mismatched Rows ---")

print(
    df.loc[
        mismatch,
        ["Timestamp", "Timestamp_Hour", "Hour", "GHI", "AC_Output"]
    ].head(20).to_string(index=False)
)

print("\n--- Unique Hour Values ---")
print(sorted(hour_numeric.dropna().unique()))

#------------------------------------------------------------------------#
##analyze_hour_mismatch##


import pandas as pd

from pathlib import Path
candidates = [
    Path(__file__).resolve().parents[2] / "datasets" / "energy_prediction" / "demo_datasets" / "Solar_Weather_Data_Processed.csv",
    Path("datasets/energy_prediction/demo_datasets/Solar_Weather_Data_Processed.csv"),
    Path("Solar Weather Processed Data/Solar_Weather_Data_Processed.csv"),
]
file_path = next((str(c) for c in candidates if c.exists()), str(candidates[0]))

df = pd.read_csv(file_path)

df.columns = df.columns.str.strip().str.replace(" ", "_")

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Extract hour directly from Timestamp
df["Timestamp_Hour"] = df["Timestamp"].dt.hour

# Calculate the difference between the two hour features
df["Hour_Difference"] = df["Hour"] - df["Timestamp_Hour"]

# Check how frequently each difference occurs
print("\n--- Hour Difference Counts ---")

print(
    df["Hour_Difference"].value_counts()
    .sort_index()
)

# Display the first 50 rows where the values do not match
mismatch = df[df["Hour"] != df["Timestamp_Hour"]]

print("\n--- First 50 Mismatched Rows ---")

print(
    mismatch[
        [
            "Timestamp",
            "Timestamp_Hour",
            "Hour",
            "Hour_Difference",
            "GHI",
            "AC_Output"
        ]
    ].head(50).to_string(index=False)
)

# Check whether the mismatch changes over time
print("\n--- Mismatch by Date ---")

df["Date"] = df["Timestamp"].dt.date

daily_mismatch = df.groupby("Date").agg(
    Total_Rows=("Hour", "count"),
    Mismatched_Rows=(
        "Hour_Difference",
        lambda x: (x != 0).sum()
    )
)

print(daily_mismatch.head(15))


#----------------------------------------------------------------------#
#Plot


import pandas as pd
import matplotlib.pyplot as plt

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

# Sort data chronologically
df = df.sort_values("Timestamp")

# Select the first 7 days
start_date = df["Timestamp"].min()
end_date = start_date + pd.Timedelta(days=7)

sample = df[
    (df["Timestamp"] >= start_date) &
    (df["Timestamp"] < end_date)
]

# Plot GHI over time
plt.figure(figsize=(14, 5))

plt.plot(
    sample["Timestamp"],
    sample["GHI"],
    label="GHI"
)

plt.title("GHI over Time")
plt.xlabel("Timestamp")
plt.ylabel("GHI")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot AC Output over time
plt.figure(figsize=(14, 5))

plt.plot(
    sample["Timestamp"],
    sample["AC_Output"],
    label="AC Output"
)

plt.title("AC Output over Time")
plt.xlabel("Timestamp")
plt.ylabel("AC Output")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()





#--------------------------------------------------------------------------------#


import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
candidates = [
    Path(__file__).resolve().parents[2] / "datasets" / "energy_prediction" / "demo_datasets" / "Solar_Weather_Data_Processed.csv",
    Path("datasets/energy_prediction/demo_datasets/Solar_Weather_Data_Processed.csv"),
    Path("Solar Weather Processed Data/Solar_Weather_Data_Processed.csv"),
]
file_path = next((str(c) for c in candidates if c.exists()), str(candidates[0]))

df = pd.read_csv(file_path)

df.columns = df.columns.str.strip().str.replace(" ", "_")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df = df.sort_values("Timestamp")

# Inspect the period where Hour mismatch begins
start = "2016-09-07"
end = "2016-09-11"

sample = df[
    (df["Timestamp"] >= start) &
    (df["Timestamp"] < end)
]

# Plot GHI and AC Output together
plt.figure(figsize=(14, 6))

plt.plot(
    sample["Timestamp"],
    sample["GHI"],
    label="GHI"
)

plt.plot(
    sample["Timestamp"],
    sample["AC_Output"],
    label="AC Output"
)

plt.title("GHI and AC Output: September 7–10")
plt.xlabel("Timestamp")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Check mismatch in this period
sample["Timestamp_Hour"] = sample["Timestamp"].dt.hour

print("\n--- Hour Mismatch in Selected Period ---")

print(
    sample[
        sample["Timestamp_Hour"] != sample["Hour"]
    ][
        ["Timestamp", "Timestamp_Hour", "Hour", "GHI", "AC_Output"]
    ].head(30).to_string(index=False)
)
  

