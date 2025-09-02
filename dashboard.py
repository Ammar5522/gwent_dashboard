# --- dashboard.py ---
import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------- App config -----------------------
st.set_page_config(page_title="Gwent Crime Analytics", layout="wide")
APP_TITLE = "Gwent Police Crime Analytics (Jul 2022 – Jun 2025)"
st.title(APP_TITLE)

DATA_CLEAN = os.path.join("data_clean", "gwent_street_cleaned.csv")
MONTHLY    = os.path.join("data_clean", "monthly_counts.csv")
MODELS_DIR = "models"
FIG_DIR    = os.path.join(MODELS_DIR, "figs")

BACKTEST_PRED   = os.path.join(MODELS_DIR, "backtest_predictions.csv")
BACKTEST_OVER   = os.path.join(MODELS_DIR, "backtest_metrics_overall.csv")
BACKTEST_BY_H   = os.path.join(MODELS_DIR, "backtest_metrics_by_h.csv")
BACKTEST_CHAMP  = os.path.join(MODELS_DIR, "backtest_champion_by_h.csv")

# ----------------------- Helpers -----------------------
def fig_bytes(fig) -> bytes:
    """Return a matplotlib figure as PNG bytes (for downloads)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    return buf.getvalue()

def file_bytes(path) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def show_png_if_exists(path, caption=None):
    if os.path.exists(path):
        st.image(path, use_container_width=True, caption=caption)

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_CLEAN, parse_dates=["Month"])
    monthly = pd.read_csv(MONTHLY, parse_dates=["Month"]).sort_values("Month")
    df["YearMonth"] = df["Month"].dt.to_period("M").dt.to_timestamp()
    return df, monthly

@st.cache_resource(show_spinner=False)
def load_model(path):
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_data(show_spinner=False)
def load_csv_if_exists(path, **kwargs):
    if os.path.exists(path):
        return pd.read_csv(path, **kwargs)
    return None

# ----------------------- Guard rails -----------------------
if not (os.path.exists(DATA_CLEAN) and os.path.exists(MONTHLY)):
    st.error("Clean data not found. Run your 01–03 scripts first.")
    st.stop()

df, monthly = load_data()

# ----------------------- Sidebar -----------------------
st.sidebar.title("Filters")
min_d, max_d = df["Month"].min().date(), df["Month"].max().date()
date_range = st.sidebar.date_input("Date range", (min_d, max_d))

all_types = sorted(df["Crime type"].dropna().unique().tolist())
types = st.sidebar.multiselect("Crime types", all_types)

mask = (df["Month"].dt.date >= date_range[0]) & (df["Month"].dt.date <= date_range[-1])
df_f = df.loc[mask].copy()
if types:
    df_f = df_f[df_f["Crime type"].isin(types)]

section = st.sidebar.radio("Go to:", ["Home", "Data", "EDA", "Forecast", "About"])

# =======================================================
# Home
# =======================================================
if section == "Home":
    st.subheader("Key indicators (filtered)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Incidents", f"{df_f.shape[0]:,}")
    c2.metric("Unique crime types", df_f["Crime type"].nunique())

    m_counts = df_f.groupby(df_f["Month"].dt.to_period("M")).size()
    yoy_text = "n/a"
    if len(m_counts) >= 13:
        yoy = ((m_counts.iloc[-1] - m_counts.iloc[-13]) / m_counts.iloc[-13]) * 100
        yoy_text = f"{yoy:+.1f}%"
    c3.metric("YoY change (last month)", yoy_text)

    if len(m_counts) >= 18:
        last6 = int(m_counts.iloc[-6:].sum())
        prev6 = int(m_counts.iloc[-12:-6].sum())
        delta = ((last6 - prev6) / prev6) * 100 if prev6 else np.nan
        c4.metric("6m vs prior 6m", f"{last6:,}", f"{delta:+.1f}%")
    else:
        c4.metric("6m vs prior 6m", "n/a")

    with st.expander("Download data"):
        d1, d2 = st.columns(2)
        d1.download_button(
            "Download filtered rows (CSV)",
            data=df_f.to_csv(index=False).encode("utf-8"),
            file_name="filtered_rows.csv",
            mime="text/csv",
        )
        d2.download_button(
            "Download monthly counts (CSV)",
            data=monthly.to_csv(index=False).encode("utf-8"),
            file_name="monthly_counts.csv",
            mime="text/csv",
        )

# =======================================================
# Data
# =======================================================
elif section == "Data":
    st.subheader("Filtered rows")
    st.dataframe(df_f, use_container_width=True, height=520)

# =======================================================
# EDA
# =======================================================
elif section == "EDA":
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Crime types", "Monthly trend", "Heatmap", "Top-6 small multiples"]
    )

    with tab1:
        topN = st.slider("Top N types", 5, 20, 12)
        type_counts = df_f["Crime type"].value_counts().head(topN).sort_values()
        total = max(1, len(df_f))
        fig, ax = plt.subplots(figsize=(8, 5))
        type_counts.plot(kind="barh", ax=ax)
        for i, (k, v) in enumerate(type_counts.items()):
            ax.text(v, i, f"  {v:,} ({v/total:.1%})", va="center")
        ax.set_title(f"Top {topN} Crime Types")
        ax.set_xlabel("Incidents"); ax.set_ylabel("")
        st.pyplot(fig)
        st.download_button("Download figure (PNG)", data=fig_bytes(fig),
                           file_name="top_types.png", mime="image/png")

    with tab2:
        s = df_f.groupby(df_f["Month"].dt.to_period("M")).size()
        s.index = s.index.to_timestamp()
        ma3 = s.rolling(3).mean()

        fig1, ax1 = plt.subplots(figsize=(10, 4.2))
        ax1.plot(s.index, s.values, label="Monthly")
        ax1.plot(ma3.index, ma3.values, label="MA(3)", linewidth=2)
        ax1.set_title("Monthly total crimes")
        ax1.legend(); ax1.set_ylabel("Incidents"); ax1.tick_params(axis="x", rotation=45)
        st.pyplot(fig1)

        yoy = s.pct_change(12) * 100
        fig2, ax2 = plt.subplots(figsize=(10, 3.6))
        ax2.axhline(0, color="k", lw=0.8)
        ax2.plot(yoy.index, yoy.values)
        ax2.set_title("Year-over-Year change (%)")
        ax2.set_ylabel("YoY %"); ax2.tick_params(axis="x", rotation=45)
        st.pyplot(fig2)

    with tab3:
        st.caption("Tip: Use log scale for raw counts; use row-normalise (0–1) to compare seasonality across types.")
        topN = st.slider("Top N for heatmap", 6, 20, 12, key="hmn")
        use_log = st.checkbox("Log-scale color", value=True)
        row_norm = st.checkbox("Row-normalise by type (0–1)", value=False)

        top_types = df_f["Crime type"].value_counts().head(topN).index
        pivot = (df_f[df_f["Crime type"].isin(top_types)]
                 .groupby([df_f["YearMonth"], "Crime type"]).size()
                 .unstack(fill_value=0).sort_index())

        if row_norm:
            data = pivot.div(pivot.max(axis=0).replace(0, np.nan), axis=1).fillna(0).T.values
            cbar_lab = "Relative intensity (0–1)"
        else:
            vals = pivot.values
            data = np.log1p(vals).T if use_log else vals.T
            cbar_lab = "Incidents (log1p)" if use_log else "Incidents"

        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(data, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(pivot.columns))); ax.set_yticklabels(pivot.columns)
        step = max(1, len(pivot.index)//12)
        ax.set_xticks(np.arange(0, len(pivot.index), step))
        ax.set_xticklabels([d.strftime("%Y-%m") for d in pivot.index[::step]], rotation=90)
        fig.colorbar(im, ax=ax, label=cbar_lab)
        ax.set_title("Crime type × month heatmap")
        st.pyplot(fig)

    with tab4:
        top6 = df_f["Crime type"].value_counts().head(6).index.tolist()
        fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
        axes = axes.ravel()
        for i, ct in enumerate(top6):
            sct = df_f[df_f["Crime type"] == ct].groupby("YearMonth").size()
            axes[i].plot(sct.index, sct.values, label=ct)
            axes[i].plot(sct.index, sct.rolling(3).mean().values, linewidth=2, label="MA(3)")
            axes[i].set_title(ct); axes[i].tick_params(axis="x", rotation=45)
            if i % 3 == 0:
                axes[i].set_ylabel("Incidents")
        axes[0].legend(fontsize=8, loc="upper left")
        st.pyplot(fig)

# =======================================================
# Forecast
# =======================================================
elif section == "Forecast":
    st.subheader("Time-series forecasting (monthly totals)")
    st.line_chart(monthly.set_index("Month")["count"])

    tabA, tabB, tabC = st.tabs(["Test set", "Backtest (rolling origin)", "Make a forecast"])

    # ---------- Tab A: Test-set results ----------
    with tabA:
        tbl_path = os.path.join(MODELS_DIR, "forecast_metrics_table_clean.csv")
        st.markdown("**Test metrics (Jan–Jun 2025)**")
        if os.path.exists(tbl_path):
            dfm = pd.read_csv(tbl_path)
            st.dataframe(dfm, use_container_width=True)
            st.download_button("Download metrics CSV", data=file_bytes(tbl_path),
                               file_name="forecast_metrics_table_clean.csv")
        else:
            st.info("Metrics file not found. Run `src/04_models_forecast.py` first.")

        with st.expander("Diagnostics & comparisons (saved figures)"):
            show_png_if_exists(os.path.join(FIG_DIR, "metrics_comparison.png"), "Model metrics comparison")
            show_png_if_exists(os.path.join(FIG_DIR, "metrics_vs_baseline.png"), "Improvement over Seasonal-Naïve")
            c1, c2 = st.columns(2)
            with c1:
                show_png_if_exists(os.path.join(FIG_DIR, "actual_vs_pred.png"), "Actual vs Predicted (test)")
                show_png_if_exists(os.path.join(FIG_DIR, "sarimax_pi.png"), "SARIMAX with 95% PI")
            with c2:
                show_png_if_exists(os.path.join(FIG_DIR, "rf_feature_importance.png"), "RF Feature importance")
                show_png_if_exists(os.path.join(FIG_DIR, "rf_residuals_hist.png"), "RF residuals")

    # ---------- Tab B: Backtest ----------
    with tabB:
        b_over  = load_csv_if_exists(BACKTEST_OVER)
        b_by_h  = load_csv_if_exists(BACKTEST_BY_H)
        b_champ = load_csv_if_exists(BACKTEST_CHAMP)

        if (b_over is None) or (b_by_h is None):
            st.info("No backtest outputs found. Run `src/05_backtest.py` to generate them.")
        else:
            st.markdown("**Overall backtest metrics (expanding-window CV)**")
            st.dataframe(b_over, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("Download metrics_overall.csv", data=file_bytes(BACKTEST_OVER),
                                   file_name="backtest_metrics_overall.csv")
            with c2:
                st.download_button("Download metrics_by_h.csv", data=file_bytes(BACKTEST_BY_H),
                                   file_name="backtest_metrics_by_h.csv")

            # MAE by horizon (lines per model)
            pivot_mae = b_by_h.pivot(index="h", columns="model", values="MAE")
            fig1, ax1 = plt.subplots(figsize=(7.8, 4.2))
            pivot_mae.plot(marker="o", ax=ax1)
            ax1.set_ylabel("MAE"); ax1.set_title("Backtest MAE by horizon")
            ax1.grid(axis="y", alpha=0.3)
            st.pyplot(fig1)
            st.caption("Lower is better. RF wins at short horizons; seasonal-naïve becomes sturdier beyond ~2 months.")

            # Relative RF vs seasonal-naïve (negative = RF better)
            if {"random_forest","seasonal_naive"}.issubset(pivot_mae.columns):
                rel = 100*(pivot_mae["random_forest"]/pivot_mae["seasonal_naive"] - 1.0)
                fig2, ax2 = plt.subplots(figsize=(7.8, 4.2))
                ax2.axhline(0, color="k", lw=1)
                rel.plot(kind="bar", ax=ax2)
                ax2.set_ylabel("% worse (+) / better (–) than seasonal_naïve")
                ax2.set_title("RF vs Seasonal-Naïve by horizon")
                st.pyplot(fig2)
                st.caption("Negative bars = RF **improves** on the baseline; positive = baseline is better.")

            if b_champ is not None:
                st.markdown("**Champion model by forecast horizon (lowest MAE)**")
                st.dataframe(b_champ, use_container_width=True)
                # Simple guidance string
                champions = dict(zip(b_champ["h"], b_champ["model"]))
                short = [h for h, m in champions.items() if m == "random_forest"]
                mid   = [h for h, m in champions.items() if m == "seasonal_naive"]
                msg = "Recommendation: "
                if short:
                    msg += f"Use **Random Forest** for {min(short)}–{max(short)}-month ahead. "
                if mid:
                    msg += f"Use **Seasonal-Naïve** for {min(mid)}–{max(mid)}-month ahead."
                st.info(msg)

    # ---------- Tab C: Make a forecast ----------
    with tabC:
        st.subheader("Make a new forecast")

        sar_model = load_model(os.path.join(MODELS_DIR, "sarimax.pkl"))
        rf_model  = load_model(os.path.join(MODELS_DIR, "rf_monthly.pkl"))

        model_opts = []
        if sar_model is not None: model_opts.append("SARIMAX (saved)")
        if rf_model  is not None: model_opts.append("Random Forest (saved)")
        if not model_opts:
            st.info("No saved models found. Run `src/04_models_forecast.py` first.")
            st.stop()

        model_choice = st.selectbox("Model", model_opts)
        horizon = st.slider("Forecast horizon (months)", 1, 12, 6)

        if model_choice.startswith("SARIMAX"):
            fc = sar_model.get_forecast(steps=horizon)
            pred_mean = fc.predicted_mean
            conf      = fc.conf_int(alpha=0.05)
            future_idx = pd.date_range(monthly["Month"].max() + pd.offsets.MonthBegin(1),
                                       periods=horizon, freq="MS")
            fdf = pd.DataFrame({
                "Month": future_idx,
                "forecast": pred_mean.values,
                "lower": conf.iloc[:, 0].values,
                "upper": conf.iloc[:, 1].values
            })

            st.dataframe(fdf, use_container_width=True)
            fig, ax = plt.subplots(figsize=(9.5, 4.6))
            tail = monthly.tail(24)
            ax.plot(tail["Month"], tail["count"], label="History")
            ax.plot(fdf["Month"], fdf["forecast"], label="Forecast")
            ax.fill_between(fdf["Month"], fdf["lower"], fdf["upper"], alpha=0.2, label="95% PI")
            ax.legend(); ax.set_title("SARIMAX forecast"); ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)

            st.download_button("Download forecast (CSV)",
                               data=fdf.to_csv(index=False).encode("utf-8"),
                               file_name="sarimax_forecast.csv", mime="text/csv")

        else:
            # Build recursive RF forecast with safe column alignment
            hist = monthly.set_index("Month")["count"].copy()
            if len(hist) < 13:
                st.warning("Not enough history to produce RF lag12 features.")
                st.stop()

            preds = []
            last_date = hist.index.max()
            series = hist.copy()
            # Column schema from the trained model:
            trained_cols = list(getattr(rf_model, "feature_names_in_", []))

            for i in range(1, horizon + 1):
                cur_month = (last_date + pd.offsets.MonthBegin(i))
                lag1  = float(series.iloc[-1])
                lag12 = float(series.iloc[-12])
                roll3 = float(series.iloc[-3:].mean())
                month_num = int(cur_month.month)

                row = {"lag1": lag1, "lag12": lag12, "roll3": roll3}
                for m in range(2, 13):
                    row[f"month_{m}"] = 1.0 if month_num == m else 0.0

                # SAFE alignment to training schema
                Xrow = (pd.DataFrame([row])
                          .reindex(columns=trained_cols, fill_value=0.0))
                yhat = float(rf_model.predict(Xrow)[0])
                preds.append((cur_month, yhat))
                series = pd.concat([series, pd.Series([yhat], index=[cur_month])])

            fdf = pd.DataFrame(preds, columns=["Month", "forecast"])
            st.dataframe(fdf, use_container_width=True)

            fig, ax = plt.subplots(figsize=(9.5, 4.6))
            tail = monthly.tail(24)
            ax.plot(tail["Month"], tail["count"], label="History")
            ax.plot(fdf["Month"], fdf["forecast"], "o--", label="RF forecast")
            ax.legend(); ax.set_title("Random Forest forecast"); ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)

            st.download_button("Download forecast (CSV)",
                               data=fdf.to_csv(index=False).encode("utf-8"),
                               file_name="rf_forecast.csv", mime="text/csv")

# =======================================================
# About
# =======================================================
elif section == "About":
    st.markdown("""
### What this app does
- **Scope:** UK Police (Gwent) street-level crime, **Jul-2022 → Jun-2025**.
- **Purpose:** Provide **descriptive** (EDA) and **predictive** (time-series) analytics to support **planning and resource allocation**.
- **Audience:** Analysts and decision-makers who need quick situational awareness and short-term (1–6 month) forecasts.

### Data & wrangling
- Source: monthly “street” CSVs from data.police.uk merged (script **01**), cleaned (**02**).
- Aggregations: monthly totals and type-level pivots; out-of-UK coordinates filtered.
- Reproducibility: all derived tables/figures are created by scripts **01–04**; backtesting by **05**.

### Modelling choices
- **Baselines:** Seasonal-Naïve (count[t] = count[t-12]).
- **Classical:** SARIMAX (small grid) and ETS (Holt-Winters).
- **ML:** Random Forest on **lag1, lag12, roll3 + month dummies**.
- **Evaluation:**  
  - **Hold-out (Jan–Jun 2025)** with MAE/RMSE/MAPE/sMAPE (no R² shown—misleading for time-series).  
  - **Rolling-origin backtest:** expanding-window CV with horizons **h=1..6** for robustness.

### Key takeaways (from your results)
- **Short horizon (h=1–2):** **Random Forest** is strongest.  
- **Medium horizon (h=3–6):** **Seasonal-Naïve** is surprisingly competitive/stronger.  
- Use the **Backtest tab** for the evidence; negative bars mean the model beats the baseline.

### Ethics & limitations
- Coordinates are **approximate/anonymised** by the provider.
- Forecasts are **aggregate** and **uncertainty-bearing**—do **not** use for individual-level “predictive policing”.
- Small samples for the test window (6 months) → rely on **backtests** for stable conclusions.

### How to reproduce
1. Run **01_merge_monthlies.py**, **02_clean.py**, **03_eda_figs.py**.  
2. Train/evaluate with **04_models_forecast.py** (creates metrics, saved models & figures).  
3. Run **05_backtest.py** for rolling-origin CV.  
4. Launch: `streamlit run dashboard.py`.
""")
