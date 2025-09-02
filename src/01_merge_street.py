import os, glob
import pandas as pd

ROOT = r"D:\Download-Office\gwent_dashboard\data_raw"

out_dir = os.path.join(os.path.dirname(ROOT), "data_clean")
os.makedirs(out_dir, exist_ok=True)

street_files = sorted(glob.glob(os.path.join(ROOT, "*", "*-gwent-street.csv")))
print(f"Found {len(street_files)} monthly street files")

frames = []
for f in street_files:
    df = pd.read_csv(f)
    df["source_file"] = os.path.basename(f)
    frames.append(df)

street_all = pd.concat(frames, ignore_index=True)

street_all["Month"] = pd.to_datetime(street_all["Month"], errors="coerce")
street_all.rename(columns=lambda x: x.strip(), inplace=True)

clean_csv = os.path.join(out_dir, "gwent_street_2022-07_to_2025-06.csv")
street_all.to_csv(clean_csv, index=False)
print("Merged street data:", street_all.shape, "->", clean_csv)

monthly_counts = (street_all
                  .groupby(street_all["Month"].dt.to_period("M"))
                  .size()
                  .rename("count")
                  .reset_index())
monthly_counts["Month"] = monthly_counts["Month"].astype(str)
mc_csv = os.path.join(out_dir, "monthly_counts.csv")
monthly_counts.to_csv(mc_csv, index=False)
print("Monthly counts:", monthly_counts.shape, "->", mc_csv)

schemas = {}
for f in street_files:
    cols = tuple(pd.read_csv(f, nrows=0).columns)
    schemas[cols] = schemas.get(cols, 0) + 1
print("Unique column schemas across months:", len(schemas))
