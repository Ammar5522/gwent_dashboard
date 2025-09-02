# --- 05_backtest.py ---
import os, math
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ========= PATHS =========
DATA = r"D:\Download-Office\gwent_dashboard\data_clean\monthly_counts.csv"
OUT_DIR = r"D:\Download-Office\gwent_dashboard\models"
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ========= CONFIG =========
INITIAL_TRAIN_END = pd.Timestamp("2023-12-01")  # first origin (must leave >=12 months history)
HORIZON = 6                                     # forecast steps per fold
STEP = "MS"                                     # slide origin monthly
RANDOM_STATE = 42

# ========= METRICS =========
def smape(y_true, y_pred):
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(yt) + np.abs(yp)) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom == 0, np.nan, np.abs(yt - yp) / denom)
    return float(np.nanmean(out) * 100)

def metric_row(y_true, y_pred):
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    mae  = mean_absolute_error(yt, yp)
    rmse = sqrt(mean_squared_error(yt, yp))
    mape = np.nanmean(np.abs((yt - yp) / np.where(yt == 0, np.nan, yt))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape, "sMAPE%": smape(yt, yp)}

# ========= LOAD =========
counts = pd.read_csv(DATA)
counts["Month"] = pd.to_datetime(counts["Month"])
counts = counts.sort_values("Month").reset_index(drop=True)

# ========= FEATURE HELPERS =========
MONTH_CATS = list(range(1, 13))

def build_feature_table(df):
    """Builds lag/roll/month features for ALL rows; dummies are stable across folds."""
    g = df.copy()
    g["lag1"]  = g["count"].shift(1)
    g["lag12"] = g["count"].shift(12)
    g["roll3"] = g["count"].rolling(3).mean()
    g["month"] = pd.Categorical(g["Month"].dt.month, categories=MONTH_CATS)
    X = pd.get_dummies(g[["lag1","lag12","roll3","month"]], columns=["month"], drop_first=True)
    y = g["count"]
    return g[["Month","count"]], X, y

def rf_fit(X_train, y_train):
    rf = RandomForestRegressor(
        n_estimators=700, max_depth=None, min_samples_leaf=1,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf

def rf_recursive_forecast(rf, history_series, last_month, h, feature_cols):
    """
    Recursive multi-step forecast: predict month t+1, append, then predict t+2, ...
    Uses lag1, lag12, roll3, and month dummies aligned to feature_cols.
    """
    preds, months = [], []
    s = history_series.copy()  # pandas Series indexed by Month

    for k in range(1, h+1):
        m = last_month + pd.offsets.MonthBegin(k)
        # build one row of features
        lag1  = s.iloc[-1]
        lag12 = s.iloc[-12] if len(s) >= 12 else lag1
        last3 = s.iloc[-3:] if len(s) >= 3 else s
        roll3 = float(last3.mean())
        row = {"lag1": lag1, "lag12": lag12, "roll3": roll3}

        # month dummies (drop_first=True -> months 2..12)
        for j in range(2, 13):
            row[f"month_{j}"] = 1.0 if m.month == j else 0.0

        # align to training columns
        x = (pd.DataFrame([row])
             .reindex(columns=feature_cols, fill_value=0.0))
        yhat = float(rf.predict(x)[0])
        preds.append(yhat); months.append(m)
        # append for next step
        s = pd.concat([s, pd.Series([yhat], index=[m])])

    return months, preds

# ========= PRECOMPUTE GLOBAL FEATURES (stable dummy columns) =========
base, X_all, y_all = build_feature_table(counts)

# we will slice X_all/y_all per fold, but keep columns stable
FEATURE_COLS = X_all.columns.tolist()

# ========= GENERATE ORIGINS =========
last_possible_origin = counts["Month"].max() - pd.offsets.MonthBegin(HORIZON)
origins = pd.date_range(INITIAL_TRAIN_END, last_possible_origin, freq=STEP)

rows = []  # predictions for all folds/models

for origin in origins:
    # training mask up to origin, and y history as a Series for recursive features
    train_mask = base["Month"] <= origin
    # need to drop NaNs created by lags/rolls
    train_idx = X_all[train_mask].dropna().index
    X_train = X_all.loc[train_idx, FEATURE_COLS]
    y_train = y_all.loc[train_idx]

    # Only proceed if we have enough training rows
    if len(train_idx) == 0:
        continue

    # history series for recursive forecasting
    hist_series = (
        base.loc[base["Month"] <= origin, ["Month","count"]]
            .set_index("Month")["count"]
    )

    # RANDOM FOREST
    rf = rf_fit(X_train, y_train)
    months_rf, preds_rf = rf_recursive_forecast(rf, hist_series, origin, HORIZON, FEATURE_COLS)
    # collect truth for available months
    truth_rf = (
        base.set_index("Month").reindex(months_rf)["count"].values
    )
    for m, yhat, ytrue, k in zip(months_rf, preds_rf, truth_rf, range(1, HORIZON+1)):
        if not math.isnan(ytrue):  # only keep if we actually have ground truth
            rows.append({"origin": origin, "model": "random_forest",
                         "Month": m, "h": k, "y_true": float(ytrue), "y_pred": float(yhat)})

    # SEASONAL NAÏVE baseline for the same months
    # --- Seasonal Naïve baseline (fixed) ---
    s_all = base.set_index("Month")["count"]  # make it a Series
    for k, m in enumerate(months_rf, start=1):
        ytrue = s_all.reindex([m]).iloc[0] if m in s_all.index else np.nan
        yhat  = s_all.reindex([m - pd.DateOffset(months=12)]).iloc[0] if (m - pd.DateOffset(months=12)) in s_all.index else np.nan
        if not (np.isnan(ytrue) or np.isnan(yhat)):
            rows.append({
                "origin": origin, "model": "seasonal_naive",
                "Month": m, "h": k, "y_true": float(ytrue), "y_pred": float(yhat)
            })


# ========= COLLATE & SCORE =========
bt = pd.DataFrame(rows).sort_values(["model","origin","Month"]).reset_index(drop=True)
bt_path = os.path.join(OUT_DIR, "backtest_predictions.csv")
bt.to_csv(bt_path, index=False)
print("Saved:", bt_path, bt.shape)

def score(df):
    return pd.Series(metric_row(df["y_true"], df["y_pred"]))

overall = bt.groupby("model", as_index=False).apply(score)
overall_path = os.path.join(OUT_DIR, "backtest_metrics_overall.csv")
overall.to_csv(overall_path, index=False)
print("Saved:", overall_path); print(overall)

by_h = bt.groupby(["model","h"], as_index=False).apply(score)
by_h_path = os.path.join(OUT_DIR, "backtest_metrics_by_h.csv")
by_h.to_csv(by_h_path, index=False)
print("Saved:", by_h_path); print(by_h)

# --- Plots + champion selection ---
import matplotlib.pyplot as plt

bm = pd.read_csv(os.path.join(OUT_DIR, "backtest_metrics_by_h.csv"))

# MAE by horizon (lines per model)
pivot_mae = bm.pivot(index="h", columns="model", values="MAE")
plt.figure(figsize=(7.2, 4.2))
pivot_mae.plot(marker="o", ax=plt.gca())
plt.ylabel("MAE"); plt.title("Backtest MAE by horizon")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "backtest_mae_by_h.png"), dpi=160)

# Relative RF vs Seasonal-Naïve (negative = RF better)
if {"random_forest","seasonal_naive"}.issubset(pivot_mae.columns):
    rel = 100*(pivot_mae["random_forest"]/pivot_mae["seasonal_naive"] - 1)
    plt.figure(figsize=(7.2, 4.2))
    plt.axhline(0, color="k", lw=1)
    rel.plot(kind="bar")
    plt.ylabel("% worse (+) / better (-) than seasonal_naive")
    plt.title("RF vs Seasonal-Naïve by horizon")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "backtest_rel_rf_vs_sn.png"), dpi=160)

# Champion model per horizon (lowest MAE)
champ = (bm.sort_values(["h","MAE"])
           .groupby("h", as_index=False)
           .first()[["h","model"]])
champ.to_csv(os.path.join(OUT_DIR, "backtest_champion_by_h.csv"), index=False)
print("Saved:", os.path.join(OUT_DIR, "backtest_champion_by_h.csv"))
print(champ)

