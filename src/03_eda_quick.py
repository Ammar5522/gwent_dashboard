import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CLEAN = r"D:\Download-Office\gwent_dashboard\data_clean\gwent_street_cleaned.csv"
FIG_DIR = r"D:\Download-Office\gwent_dashboard\data_clean\figs"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(CLEAN, parse_dates=["Month"])
df = df.dropna(subset=["Month"]).copy()
df["YearMonth"] = df["Month"].dt.to_period("M").dt.to_timestamp()

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    print("Saved:", path)

topN = 12
type_counts = df["Crime type"].value_counts().head(topN)
total = len(df)
plt.figure(figsize=(8, 5))
(type_counts.sort_values()
 .plot(kind="barh"))
for i, (k, v) in enumerate(type_counts.sort_values().items()):
    plt.text(v, i, f"  {v:,} ({v/total:,.1%})", va="center")
plt.title(f"Top {topN} Crime Types (Jul-2022 to Jun-2025)")
plt.xlabel("Incidents")
plt.ylabel("")
plt.tight_layout()
savefig("01_top_types.png")
plt.show()

monthly = df.groupby("YearMonth").size().rename("count")
ma3 = monthly.rolling(3).mean()

plt.figure(figsize=(10, 4.5))
plt.plot(monthly.index, monthly.values, label="Monthly")
plt.plot(ma3.index, ma3.values, label="3-month MA", linewidth=2)

end = monthly.index.max()
start = (end - pd.DateOffset(months=5)).replace(day=1)
plt.axvspan(start, end + pd.offsets.MonthEnd(1), color="gray", alpha=0.15, label="Last 6 months")

plt.title("Monthly Total Crimes (with 3-month MA)")
plt.ylabel("Incidents")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
savefig("02_monthly_trend_ma.png")
plt.show()

yoy = monthly.pct_change(12) * 100
plt.figure(figsize=(10, 3.8))
plt.axhline(0, color="black", linewidth=0.8)
plt.plot(yoy.index, yoy.values)
plt.title("Year-over-Year Change in Monthly Crimes (%)")
plt.ylabel("YoY %")
plt.xticks(rotation=45)
plt.tight_layout()
savefig("03_monthly_yoy.png")
plt.show()

top_types = df["Crime type"].value_counts().head(12).index.tolist()
pivot = (df[df["Crime type"].isin(top_types)]
         .groupby(["YearMonth", "Crime type"]).size()
         .unstack(fill_value=0))

pivot = pivot.sort_index()

USE_LOG = True
vals = np.log1p(pivot.values) if USE_LOG else pivot.values

plt.figure(figsize=(12, 5))
im = plt.imshow(vals.T, aspect="auto", interpolation="nearest")
plt.yticks(range(len(pivot.columns)), pivot.columns)
xticks = np.arange(0, len(pivot.index), max(1, len(pivot.index)//12))
plt.xticks(xticks, [d.strftime("%Y-%m") for d in pivot.index[xticks]], rotation=90)
cbar = plt.colorbar(im)
cbar.set_label("Incidents (log1p)" if USE_LOG else "Incidents")
plt.title("Heatmap: Crime Type by Month (raw counts)")
plt.tight_layout()
savefig("04_heatmap_raw.png")
plt.show()

row_norm = pivot.div(pivot.max(axis=0).replace(0, np.nan), axis=1).fillna(0)

plt.figure(figsize=(12, 5))
im = plt.imshow(row_norm.T.values, aspect="auto", interpolation="nearest")
plt.yticks(range(len(row_norm.columns)), row_norm.columns)
xticks = np.arange(0, len(row_norm.index), max(1, len(row_norm.index)//12))
plt.xticks(xticks, [d.strftime("%Y-%m") for d in row_norm.index[xticks]], rotation=90)
cbar = plt.colorbar(im)
cbar.set_label("Relative intensity (0–1)")
plt.title("Heatmap: Crime Type by Month (row-normalised)")
plt.tight_layout()
savefig("05_heatmap_rownorm.png")
plt.show()

top6 = df["Crime type"].value_counts().head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
axes = axes.ravel()
for i, ct in enumerate(top6):
    s = (df[df["Crime type"] == ct]
         .groupby("YearMonth").size())
    axes[i].plot(s.index, s.values, label=ct)
    axes[i].plot(s.index, s.rolling(3).mean().values, linewidth=2, label="MA(3)")
    axes[i].set_title(ct)
    axes[i].tick_params(axis="x", rotation=45)
    if i % 3 == 0:
        axes[i].set_ylabel("Incidents")
axes[0].legend(loc="upper left", fontsize=8)
plt.tight_layout()
savefig("06_small_multiples_top6.png")
plt.show()

HAS_COORDS = df["Longitude"].notna().any() and df["Latitude"].notna().any()
if HAS_COORDS:
    end_m = df["YearMonth"].max()
    start_m = (end_m - pd.DateOffset(months=11)).to_period("M").to_timestamp()
    dsel = df[(df["YearMonth"] >= start_m) & (df["YearMonth"] <= end_m)].dropna(subset=["Longitude","Latitude"])

    if len(dsel) > 0:
        plt.figure(figsize=(6.8, 6))
        plt.hexbin(dsel["Longitude"], dsel["Latitude"], gridsize=40, bins="log")
        plt.colorbar(label="Incidents (log count)")
        plt.title(f"Spatial Density (hexbin): last 12 months\n{start_m.strftime('%Y-%m')} to {end_m.strftime('%Y-%m')}")
        plt.xlabel("Longitude"); plt.ylabel("Latitude")
        plt.tight_layout()
        savefig("07_spatial_hexbin_last12m.png")
        plt.show()
