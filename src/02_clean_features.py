import os
import pandas as pd

CLEAN_IN = r"D:\Download-Office\gwent_dashboard\data_clean\gwent_street_2022-07_to_2025-06.csv"
OUT_DIR = r"D:\Download-Office\gwent_dashboard\data_clean"

df = pd.read_csv(CLEAN_IN, low_memory=False)
df["Month"] = pd.to_datetime(df["Month"], errors="coerce")

keep = ["Crime ID","Month","Longitude","Latitude","Location",
        "LSOA code","LSOA name","Crime type","Last outcome category","source_file"]
df = df[keep].copy()

for c in ["Location","LSOA name","Crime type","Last outcome category"]:
    df[c] = df[c].astype(str).str.strip()

df["Year"] = df["Month"].dt.year
df["MonthNum"] = df["Month"].dt.month
df["YearMonth"] = df["Month"].dt.to_period("M").astype(str)

df = df[(df["Longitude"].between(-7.8, 1.8, inclusive="both")) &
        (df["Latitude"].between(49.5, 59.5, inclusive="both") ) | df["Longitude"].isna()]

OUT = os.path.join(OUT_DIR, "gwent_street_cleaned.csv")
df.to_csv(OUT, index=False)
print("Saved:", OUT, df.shape)

print("Date range:", df["Month"].min(), "→", df["Month"].max())
print("Crime type counts (top 10):")
print(df["Crime type"].value_counts().head(10))
