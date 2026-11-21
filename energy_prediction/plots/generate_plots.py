import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"Solar Weather Processed Data\Solar_Weather_Data_Processed.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_")

plt.figure(figsize=(8, 5))

plt.hist(df["AC_Output"], bins=40)


plt.xlabel("AC Output")
plt.ylabel("Frequency")
plt.title("Distribution of AC Output")

plt.figure(figsize=(8, 5))

plt.hist(df["GHI"], bins=40)

plt.xlabel("GHI")
plt.ylabel("Frequency")
plt.title("Distribution of GHI")


plt.figure(figsize=(12, 5))

plt.plot(
    df["Timestamp"],
    df["AC_Output"]
)

plt.xlabel("Timestamp")
plt.ylabel("AC Output")
plt.title("AC Output Over Time")
plt.xticks(rotation=45)
plt.tight_layout()

plt.figure(figsize=(8, 5))

plt.scatter(
    df["GHI"],
    df["AC_Output"],
    alpha=0.5
)

plt.xlabel("GHI")
plt.ylabel("AC Output")
plt.title("GHI vs AC Output")

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Temperature"],
    df["AC_Output"],
    alpha=0.5
)

plt.xlabel("Temperature")
plt.ylabel("AC Output")
plt.title("Temperature vs AC Output")

plt.show()

hourly_summary = df.groupby("Hour").agg(
    Average_GHI=("GHI", "mean"),
    Average_AC_Output=("AC_Output", "mean"),
    Zero_Output_Count=("AC_Output", lambda x: (x == 0).sum())
)

print(hourly_summary)