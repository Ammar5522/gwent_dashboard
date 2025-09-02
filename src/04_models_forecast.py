import os, json, warnings
from math import sqrt

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = r"D:\Download-Office\gwent_dashboard\data_clean\monthly_counts.csv"
OUT_DIR = r"D:\Download-Office\gwent_dashboard\models"
os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name: str):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    print("Saved figure:", path)

def to_csv(df: pd.DataFrame, name: str):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print("Saved table:", path)

counts = pd.read_csv(DATA)
counts["Month"] = pd.to_datetime(counts["Month"])
counts = counts.sort_values("Month").reset_index(drop=True)

train = counts[counts["Month"] <= "2024-12-01"].copy()
test  = counts[counts["Month"] >= "2025-01-01"].copy()

y_train = train["count"].values
y_test  = test["count"].values
test_months = counts.loc[test.index, "Month"].values

def smape(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(yt) + np.abs(yp)) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom == 0, np.nan, np.abs(yt - yp) / denom)
    return float(np.nanmean(out) * 100)

def metrics(y_true, y_pred) -> dict:
    yt, yp = np.array(y_true, float), np.array(y_pred, float)
    mae  = mean_absolute_error(yt, yp)
    rmse = sqrt(mean_squared_error(yt, yp))
    mape = np.nanmean(np.abs((yt - yp) / np.where(yt == 0, np.nan, yt))) * 100
    r2   = r2_score(yt, yp)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE%": float(mape),
        "sMAPE%": smape(yt, yp),
        "R2": float(r2),
    }

counts["sn"] = counts["count"].shift(12)
sn_pred = counts.loc[test.index, "sn"].values
scores_sn = metrics(y_test, sn_pred)

warnings.filterwarnings("ignore")
y_ts = train["count"].astype(float)
sar_specs = [
    ((0, 1, 1), (0, 1, 1, 12)),
    ((1, 1, 1), (0, 1, 1, 12)),
    ((0, 1, 1), (1, 1, 0, 12)),
    ((1, 1, 0), (1, 1, 0, 12)),
]
best_aic, best_sar, best_spec = 1e18, None, None
for order, sorder in sar_specs:
    try:
        m = sm.tsa.statespace.SARIMAX(
            y_ts,
            order=order,
            seasonal_order=sorder,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        r = m.fit(disp=False)
        if r.aic < best_aic:
            best_aic, best_sar, best_spec = r.aic, r, (order, sorder)
    except Exception:
        pass

if best_sar is None:
    sar_mean, sar_ci = counts.loc[test.index, "count"].shift(12).values, None
else:
    fc = best_sar.get_forecast(steps=len(test))
    sar_mean = fc.predicted_mean.values
    sar_ci = fc.conf_int(alpha=0.05)
    joblib.dump(best_sar, os.path.join(OUT_DIR, "sarimax.pkl"))

scores_sar = metrics(y_test, sar_mean)
if sar_ci is not None:
    lower, upper = sar_ci.iloc[:, 0].values, sar_ci.iloc[:, 1].values
    scores_sar["PI_95%_coverage"] = float(np.mean((y_test >= lower) & (y_test <= upper)) * 100)
    scores_sar["PI_95%_avg_width"] = float(np.mean(upper - lower))

ets_specs = [
    dict(trend="add", seasonal="add", damped_trend=True),
    dict(trend="add", seasonal="mul", damped_trend=True),
    dict(trend="mul", seasonal="mul", damped_trend=True),
    dict(trend="add", seasonal="add", damped_trend=False),
]
best_ets, best_ets_aic, best_ets_spec = None, 1e18, None
for spec in ets_specs:
    try:
        mod = ExponentialSmoothing(
            train["count"],
            trend=spec["trend"],
            seasonal=spec["seasonal"],
            seasonal_periods=12,
            damped_trend=spec["damped_trend"],
        )
        res = mod.fit(optimized=True, use_brute=True)
        aic = getattr(res, "aic", np.inf)
        if aic < best_ets_aic:
            best_ets_aic, best_ets, best_ets_spec = aic, res, spec
    except Exception:
        pass

if best_ets is None:
    ets_pred = counts.loc[test.index, "count"].shift(12).values
else:
    ets_pred = best_ets.forecast(len(test)).values
    joblib.dump(best_ets, os.path.join(OUT_DIR, "ets.pkl"))

scores_ets = metrics(y_test, ets_pred)

df = counts.copy()
df["lag1"]  = df["count"].shift(1)
df["lag12"] = df["count"].shift(12)
df["roll3"] = df["count"].rolling(3).mean()  
df["month"] = df["Month"].dt.month
df = df.dropna().reset_index(drop=True)

X = pd.get_dummies(df[["lag1", "lag12", "roll3", "month"]], columns=["month"], drop_first=True)
y_tab = df["count"]

X_train = X[df["Month"] <= "2024-12-01"]
y_train_tab = y_tab[df["Month"] <= "2024-12-01"]
X_test  = X[df["Month"] >= "2025-01-01"]
y_test_tab  = y_tab[df["Month"] >= "2025-01-01"]

rf = RandomForestRegressor(
    n_estimators=700,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train_tab)
rf_pred = rf.predict(X_test)
scores_rf = metrics(y_test_tab.values, rf_pred)
joblib.dump(rf, os.path.join(OUT_DIR, "rf_monthly.pkl"))

results = {
    "seasonal_naive": scores_sn,
    "sarimax": scores_sar | {"spec": str(best_spec) if best_spec else "fallback seasonal naive"},
    "ets":     scores_ets | {"spec": str(best_ets_spec) if best_ets_spec else "fallback seasonal naive"},
    "random_forest": scores_rf,
}
print(json.dumps(results, indent=2))
with open(os.path.join(OUT_DIR, "forecast_metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

metrics_full_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
to_csv(metrics_full_df, "forecast_metrics_table.csv")

pred_table = pd.DataFrame({
    "Month": test_months,
    "Actual": y_test,
    "SeasonalNaive": sn_pred,
    "SARIMAX_mean": sar_mean,
    "ETS": ets_pred,
    "RF": rf_pred,
})
if best_sar is not None and sar_ci is not None:
    pred_table["SARIMAX_lower"] = lower
    pred_table["SARIMAX_upper"] = upper
to_csv(pred_table, "test_predictions.csv")

plt.figure(figsize=(9.2, 4.8))
plt.plot(pred_table["Month"], pred_table["Actual"], label="Actual", linewidth=2)
plt.plot(pred_table["Month"], pred_table["SeasonalNaive"], "--", label="Seasonal-Naïve")
plt.plot(pred_table["Month"], pred_table["SARIMAX_mean"], "-.", label="SARIMAX")
plt.plot(pred_table["Month"], pred_table["ETS"], label="ETS (Holt-Winters)")
plt.plot(pred_table["Month"], pred_table["RF"], ":", label="Random Forest")
plt.legend()
plt.title("Actual vs Predicted (Jan–Jun 2025)")
plt.xticks(rotation=45)
plt.tight_layout()
savefig("actual_vs_pred.png")
plt.show()

if best_sar is not None and sar_ci is not None:
    plt.figure(figsize=(9.2, 4.8))
    plt.plot(pred_table["Month"], pred_table["Actual"], label="Actual", linewidth=2)
    plt.plot(pred_table["Month"], pred_table["SARIMAX_mean"], label="SARIMAX mean")
    plt.fill_between(
        pred_table["Month"], pred_table["SARIMAX_lower"], pred_table["SARIMAX_upper"],
        alpha=0.2, label="95% PI"
    )
    cov = results["sarimax"].get("PI_95%_coverage", np.nan)
    plt.title(f"SARIMAX Forecast with 95% Prediction Interval (coverage: {cov:.1f}%)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    savefig("sarimax_pi.png")
    plt.show()

rf_resid = y_test_tab.values - rf_pred
plt.figure(figsize=(6.5, 4))
plt.hist(rf_resid, bins=10, edgecolor="k")
plt.axvline(0, color="k", lw=1)
plt.title("Random Forest Residuals (Test)")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.tight_layout()
savefig("rf_residuals_hist.png")
plt.show()

imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values()
imp_df = imp.reset_index().rename(columns={"index": "Feature", 0: "Importance"})
to_csv(imp_df.sort_values("Importance", ascending=False), "rf_feature_importance.csv")
plt.figure(figsize=(8, 5))
imp.plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
savefig("rf_feature_importance.png")
plt.show()

METRIC_KEYS = ["MAE", "RMSE", "MAPE%", "sMAPE%"]

metrics_public = (
    pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
)
keep_cols = ["Model"] + [c for c in METRIC_KEYS if c in metrics_public.columns]
metrics_public = metrics_public[keep_cols]
to_csv(metrics_public, "forecast_metrics_table_clean.csv")

plot_df = metrics_public.set_index("Model")
plot_cols = [c for c in METRIC_KEYS if c in plot_df.columns]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.ravel()
for i, m in enumerate(plot_cols):
    plot_df[m].plot(kind="bar", ax=axes[i])
    axes[i].set_title(m)
    axes[i].set_ylabel(m)
    axes[i].grid(axis="y", alpha=0.3)
    axes[i].set_xlabel("")
for j in range(len(plot_cols), 4):
    fig.delaxes(axes[j])
plt.suptitle("Model Metrics Comparison (Jan–Jun 2025)", y=1.02, fontsize=12)
plt.tight_layout()
savefig("metrics_comparison.png")
plt.show()

if "seasonal_naive" in plot_df.index:
    baseline = plot_df.loc["seasonal_naive"][plot_cols]
    rel = (1.0 - plot_df[plot_cols].div(baseline)) * 100.0
    rel = rel.drop(index="seasonal_naive", errors="ignore")
    plt.figure(figsize=(9.5, 4.8))
    rel.plot(kind="bar")
    plt.axhline(0, color="k", lw=1)
    plt.ylabel("Improvement vs Seasonal-Naïve (%)  ↑ better")
    plt.title("Improvement over Seasonal-Naïve (Jan–Jun 2025)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    savefig("metrics_vs_baseline.png")
    plt.show()
