from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Sri Lanka Gold Rate — Ensemble Learning Dashboard",
    page_icon="🪙",
    layout="wide",
)

BUNDLE_PATH = Path(__file__).parent / "models" / "gold_ensemble_bundle.pkl"


@st.cache_resource
def load_bundle():
    with open(BUNDLE_PATH, "rb") as f:
        return pickle.load(f)


def predict_next(bundle: dict, X: pd.DataFrame) -> dict[str, float]:
    """Compute next-trading-day LKR/Troy oz forecasts across all trained models."""
    lp = float(bundle["last_gold_lkr"])
    models = bundle["models"]
    preds: dict[str, float] = {}

    for m_name in [
        "CatBoost (Boosting)",
        "Random Forest (Bagging)",
        "XGBoost (Tuned)",
        "LightGBM (Boosting)",
        "Stacking Ensemble (RidgeCV Meta)",
    ]:
        if m_name in models:
            ret = float(models[m_name].predict(X)[0])
            preds[m_name] = lp * (1.0 + ret)

    arima_log = float(bundle["arima_next_log_forecast"])
    preds["Hybrid: RF + ARIMA"] = float(
        np.exp(arima_log + float(models["_rf_arima_resid"].predict(X)[0]))
    )
    if "_cat_arima_resid" in models:
        preds["Hybrid: CatBoost + ARIMA"] = float(
            np.exp(arima_log + float(models["_cat_arima_resid"].predict(X)[0]))
        )
    preds["ARIMA(1,1,1) Standalone"] = float(np.exp(arima_log))
    preds["Naive Persistence (y[t-1])"] = lp
    return preds


bundle = load_bundle()
lb: pd.DataFrame = bundle["leaderboard"].copy()
paradigm_map: dict[str, str] = dict(zip(lb["Model"], lb["Paradigm"]))
oz_to_pw = float(bundle["troy_oz_to_pawan"])
top_features: list[str] = bundle["top_features"][:4]
last_date_str = pd.Timestamp(bundle["last_date"]).strftime("%Y-%m-%d")
next_date_str = pd.Timestamp(bundle["next_date"]).strftime("%Y-%m-%d")
last_oz = float(bundle["last_gold_lkr"])

# Sidebar Controls
st.sidebar.title("Controls")
unit_choice = st.sidebar.radio(
    "Gold Price Unit",
    ["LKR / Troy oz", "LKR / Pawan (8g, 24K)"],
    index=0,
)
unit_scale = 1.0 if unit_choice == "LKR / Troy oz" else oz_to_pw
unit_short = "LKR/oz" if unit_choice == "LKR / Troy oz" else "LKR/Pawan"

all_models = lb["Model"].tolist()
default_models = all_models[:4]
selected_models = st.sidebar.multiselect(
    "Models to Compare",
    options=all_models,
    default=default_models,
)
if not selected_models:
    selected_models = default_models

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Data Source:** Central Bank of Sri Lanka (CBSL) — Daily Buying & Selling Exchange Rates "
    "([cbsl.gov.lk](https://www.cbsl.gov.lk/en/rates-and-indicators/exchange-rates)) & Daily Gold Price."
)

# Header
st.title("Sri Lanka Gold Rate Forecasting — Ensemble Learning Dashboard")
st.caption(
    "Bagging (`Random Forest`), Boosting (`CatBoost`, `XGBoost`, `LightGBM`), `Stacking`, and "
    "Time-Series Hybrids (`RF + ARIMA`, `CatBoost + ARIMA`) trained on official CBSL daily data (2010–2026)."
)

# 4 Focused Tabs (designed for crisp walkthrough & <=3-min presentation)
tab_forecast, tab_leaderboard, tab_backtest, tab_about = st.tabs([
    "🔮 Forecast & Overview",
    "🏆 Model Comparison",
    "📈 Backtest & Drivers",
    "ℹ️ About & AI Use",
])

# ---------------------------------------------------------------------------
# Tab 1: Forecast & Overview
# ---------------------------------------------------------------------------
with tab_forecast:
    X_base = bundle["next_day_features"].copy()
    base_preds = predict_next(bundle, X_base)

    best_row = lb.iloc[0]
    best_name = str(best_row["Model"])
    best_rmse_disp = float(best_row["RMSE (LKR/oz)"]) * unit_scale
    best_next_disp = base_preds[best_name] * unit_scale
    best_delta_pct = (base_preds[best_name] / last_oz - 1.0) * 100.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Latest CBSL Gold ({last_date_str})", f"{last_oz * unit_scale:,.2f} {unit_short}")
    c2.metric("Best Out-of-Sample Model", best_name, f"RMSE: {best_rmse_disp:,.2f} {unit_short}", delta_color="off")
    c3.metric(f"Next-Day Forecast ({next_date_str})", f"{best_next_disp:,.2f} {unit_short}", f"{best_delta_pct:+.3f}% vs. last")
    c4.metric("Random Forest OOB R²", f"{bundle['rf_oob_r2']:.4f}", "Bagging (300 trees)", delta_color="off")

    st.subheader(f"Next-Trading-Day Forecasts ({next_date_str})")

    def _reset_whatif():
        for feat in top_features:
            st.session_state[f"whatif_{feat}"] = round(float(X_base[feat].iloc[0]) * 100.0, 2)

    X_sim = X_base.copy()
    with st.expander("What-If Shock Simulator (Adjust Top 4 Lagged FX/Bullion Drivers)", expanded=False):
        st.button("Reset to Observed CBSL Values", on_click=_reset_whatif)
        cols = st.columns(2)
        for idx, feat in enumerate(top_features):
            cur_pct = round(float(X_base[feat].iloc[0]) * 100.0, 2)
            lo_pct, hi_pct = [round(float(v) * 100.0, 2) for v in bundle["feature_bounds"][feat]]
            lo_pct = min(lo_pct, cur_pct)
            hi_pct = max(hi_pct, cur_pct)
            key = f"whatif_{feat}"
            if key not in st.session_state:
                st.session_state[key] = cur_pct
            with cols[idx % 2]:
                sim_pct = st.slider(
                    f"{feat} (observed: {cur_pct:+.2f}%)",
                    min_value=float(lo_pct),
                    max_value=float(hi_pct),
                    step=0.10,
                    format="%.2f%%",
                    key=key,
                )
                X_sim.iloc[0, X_sim.columns.get_loc(feat)] = sim_pct / 100.0

    sim_preds = predict_next(bundle, X_sim)
    rows = []
    for m_name in selected_models:
        if m_name in sim_preds:
            b_oz = base_preds[m_name]
            s_oz = sim_preds[m_name]
            rows.append({
                "Model": m_name,
                "Paradigm": paradigm_map.get(m_name, "—"),
                "Forecast (LKR/oz)": s_oz,
                "Forecast (LKR/Pawan)": s_oz * oz_to_pw,
                "Change vs. Last (%)": (s_oz / last_oz - 1.0) * 100.0,
                f"What-If Shift ({unit_short})": (s_oz - b_oz) * unit_scale,
            })

    fc_df = pd.DataFrame(rows)
    st.dataframe(
        fc_df.style.format({
            "Forecast (LKR/oz)": "{:,.2f}",
            "Forecast (LKR/Pawan)": "{:,.2f}",
            "Change vs. Last (%)": "{:+.3f}%",
            f"What-If Shift ({unit_short})": "{:+,.2f}",
        }),
        width="stretch",
        hide_index=True,
    )

    chart_col = "Forecast (LKR/oz)" if unit_choice == "LKR / Troy oz" else "Forecast (LKR/Pawan)"
    fig_fc = px.bar(
        fc_df.sort_values(chart_col),
        x=chart_col,
        y="Model",
        color="Paradigm",
        orientation="h",
        title=f"Predicted Sri Lanka Gold Rate for {next_date_str} ({unit_choice})",
        text_auto=",.0f",
    )
    fig_fc.add_shape(
        type="line",
        x0=last_oz * unit_scale,
        x1=last_oz * unit_scale,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#0f172a", dash="dash", width=2),
    )
    fig_fc.update_layout(
        xaxis_range=[
            min(fc_df[chart_col].min(), last_oz * unit_scale) * 0.996,
            max(fc_df[chart_col].max(), last_oz * unit_scale) * 1.004,
        ],
        height=380,
    )
    st.plotly_chart(fig_fc, width="stretch")

# ---------------------------------------------------------------------------
# Tab 2: Model Comparison (Leaderboard + Interpretation)
# ---------------------------------------------------------------------------
with tab_leaderboard:
    st.subheader("Chronological Out-of-Sample Leaderboard (2024–2026, 685 Trading Days)")
    paradigms = sorted(lb["Paradigm"].unique().tolist())
    chosen_paradigms = st.multiselect("Filter by Paradigm", options=paradigms, default=paradigms)
    lb_view = lb[lb["Paradigm"].isin(chosen_paradigms)].copy()

    fmt_lb = lb_view.copy()
    fmt_lb["Dir. Accuracy (%)"] = fmt_lb["Dir. Accuracy (%)"].apply(
        lambda v: "—" if pd.isna(v) else f"{v:.2f}%"
    )
    st.dataframe(
        fmt_lb.style.format({
            "RMSE (LKR/oz)": "{:,.2f}",
            "RMSE (LKR/Pawan)": "{:,.2f}",
            "MAE (LKR/oz)": "{:,.2f}",
            "MAPE (%)": "{:.4f}%",
            "R2 (Level)": "{:.4f}",
            "R2 (Return)": "{:.4f}",
        }),
        width="stretch",
        hide_index=True,
    )

    lb_plot = lb.copy()
    lb_plot["RMSE_Display"] = lb_plot["RMSE (LKR/oz)"] * unit_scale
    naive_rmse = float(lb_plot.loc[lb_plot["Model"] == "Naive Persistence (y[t-1])", "RMSE_Display"].iloc[0])

    paradigm_colors = {
        "Boosting": "#2563eb",
        "Bagging": "#16a34a",
        "Stacking": "#d97706",
        "Hybrid (Time-Series)": "#7c3aed",
        "Baseline": "#64748b",
        "Time-Series Baseline": "#dc2626",
    }

    c_left, c_right = st.columns(2)
    with c_left:
        rmse_sorted = lb_plot.sort_values("RMSE_Display", ascending=True)
        fig_rmse = px.bar(
            rmse_sorted,
            x="RMSE_Display",
            y="Model",
            color="Paradigm",
            color_discrete_map=paradigm_colors,
            category_orders={"Model": rmse_sorted["Model"].tolist()},
            orientation="h",
            title=f"Test RMSE ({unit_short}) — Lower is Better",
            labels={"RMSE_Display": f"Test RMSE ({unit_short})"},
        )
        fig_rmse.add_shape(
            type="line",
            x0=naive_rmse,
            x1=naive_rmse,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#0f172a", dash="dash", width=2),
        )
        fig_rmse.update_layout(
            xaxis_range=[lb_plot["RMSE_Display"].min() * 0.985, lb_plot["RMSE_Display"].max() * 1.01],
            height=420,
        )
        st.plotly_chart(fig_rmse, width="stretch")

    with c_right:
        dir_df = lb_plot.dropna(subset=["Dir. Accuracy (%)"]).sort_values("Dir. Accuracy (%)", ascending=False)
        fig_dir = px.bar(
            dir_df,
            x="Dir. Accuracy (%)",
            y="Model",
            color="Paradigm",
            color_discrete_map=paradigm_colors,
            category_orders={"Model": dir_df["Model"].tolist()},
            orientation="h",
            title="Directional Accuracy (%) — Higher is Better",
        )
        fig_dir.add_shape(
            type="line",
            x0=50.0,
            x1=50.0,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#dc2626", dash="dash", width=2),
        )
        fig_dir.update_layout(xaxis_range=[48, 60], height=420)
        st.plotly_chart(fig_dir, width="stretch")

    st.info(
        "**Why Models Ranked This Way:**\n"
        "- **CatBoost (#1 RMSE: `17,429 LKR/oz`)** won on daily returns thanks to symmetric trees + ordered boosting resisting financial noise.\n"
        "- **Random Forest (#2: `17,539 LKR/oz`, OOB R² `+0.0420`)** & **Tuned XGBoost (#3: `17,540 LKR/oz`)** both beat Naive Persistence (`17,752 LKR/oz`) via strong regularization.\n"
        "- **Stacking (#1 Dir. Accuracy: `57.66%`)** blends tree ensembles with linear `Ridge` via a `RidgeCV` meta-learner.\n"
        "- **Hybrids (`RF + ARIMA` `-201 LKR/oz`, `CatBoost + ARIMA` `-195 LKR/oz` vs. standalone ARIMA):** ARIMA extrapolates the trend past the `707k` training ceiling while the Stage-2 tree learns nonlinear FX residuals—with Bagging (`RF`) edging out Boosting (`CatBoost`) on the noisier leftover residuals."
    )

# ---------------------------------------------------------------------------
# Tab 3: Backtest & Feature Drivers
# ---------------------------------------------------------------------------
with tab_backtest:
    st.subheader("Out-of-Sample Backtest (2024–2026) & Tree Extrapolation Proof")
    preds_df: pd.DataFrame = bundle["test_predictions"].copy()
    preds_df["Date"] = pd.to_datetime(preds_df["Date"])
    min_d, max_d = preds_df["Date"].min().date(), preds_df["Date"].max().date()

    col_d, col_chk = st.columns([3, 1.3])
    with col_d:
        date_range = st.slider("Backtest Window", min_value=min_d, max_value=max_d, value=(min_d, max_d))
    with col_chk:
        show_ceiling = st.checkbox("Show Raw-Price RF ceiling demo", value=False)

    win_df = preds_df[(preds_df["Date"].dt.date >= date_range[0]) & (preds_df["Date"].dt.date <= date_range[1])]

    fig_bt = go.Figure()
    fig_bt.add_trace(
        go.Scatter(
            x=win_df["Date"],
            y=win_df["Actual_Gold_LKR"] * unit_scale,
            name="Actual CBSL Gold",
            line=dict(color="#0f172a", width=2.4),
        )
    )
    for m_name in selected_models:
        if m_name in win_df.columns:
            fig_bt.add_trace(
                go.Scatter(
                    x=win_df["Date"],
                    y=win_df[m_name] * unit_scale,
                    name=m_name,
                    line=dict(width=1.6),
                    opacity=0.88,
                )
            )
    if show_ceiling and "RF (Raw Price — Ceiling Demo)" in win_df.columns:
        fig_bt.add_trace(
            go.Scatter(
                x=win_df["Date"],
                y=win_df["RF (Raw Price — Ceiling Demo)"] * unit_scale,
                name="RF (Raw Price — Ceiling Demo)",
                line=dict(color="#dc2626", width=2.0, dash="dot"),
            )
        )
    train_ceil = float(bundle["train_max_gold"]) * unit_scale
    fig_bt.add_shape(
        type="line",
        x0=0,
        x1=1,
        xref="paper",
        y0=train_ceil,
        y1=train_ceil,
        line=dict(color="#dc2626", dash="dash", width=1.4),
    )
    fig_bt.add_annotation(
        x=0.01,
        xref="paper",
        y=train_ceil,
        text=f"2011–2023 Train Max ({train_ceil:,.0f} {unit_short})",
        showarrow=False,
        yshift=10,
        font=dict(color="#dc2626", size=11),
    )
    fig_bt.update_layout(
        title=f"Actual vs. Predicted Sri Lanka Gold Price ({unit_choice})",
        xaxis_title="Trading Date",
        yaxis_title=unit_choice,
        height=440,
        hovermode="x unified",
    )
    st.plotly_chart(fig_bt, width="stretch")

    st.subheader("Top 10 Predictive Drivers (Normalized Feature Importance %)")
    fi_df: pd.DataFrame = bundle["feature_importances"].copy()
    fi_model = st.selectbox(
        "Inspect Importance For",
        ["Mean_Importance", "CatBoost", "Random Forest", "XGBoost (Tuned)", "LightGBM"],
        index=0,
    )
    fi_top = fi_df.sort_values(fi_model, ascending=False).head(10)
    fig_fi = px.bar(
        fi_top.iloc[::-1],
        x=fi_model,
        y="Feature",
        orientation="h",
        text_auto=".2f",
        color=fi_model,
        color_continuous_scale="YlOrBr",
    )
    fig_fi.update_layout(height=360, coloraxis_showscale=False, xaxis_title="Importance (%)")
    st.plotly_chart(fig_fi, width="stretch")

# ---------------------------------------------------------------------------
# Tab 4: About, Research Reflection & GenAI Use
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("1. Local CBSL Pipeline & Anti-Leakage Design")
    st.markdown(
        """
- **Local Sri Lankan Dataset:** Official CBSL Daily Commercial Bank TT Exchange Rates (`8` currencies after dropping `CNY` which starts in 2017) merged with CBSL Daily Indicative Gold Price (`3,518` clean trading days, `2010–2026`, `0` nulls).
- **Data Hygiene:** Removed 526 non-date summary rows, restored 3 year-stamp typos, and interpolated 19 isolated CBSL data-entry errors while preserving genuine macro shocks (March 2022 LKR float).
- **Zero-Leakage Evaluation:** All `50` predictors are strictly lagged (`t-1` or earlier), trained on `2011–2023` (`2,802` days), and evaluated chronologically out-of-sample on `2024–2026` (`685` days).
- **Target Reformulation:** Predicting stationary daily returns (or `ARIMA(1,1,1)` residuals) overcomes raw-price tree non-extrapolation when 86% of test prices sit above the training max (`707k LKR/oz`).
        """
    )

    st.subheader("2. Research Reflection: Tabular vs. Image/Text Data")
    st.markdown(
        """
| Dimension | Tree Ensembles (`RF`, `XGBoost`, `LightGBM`, `CatBoost`) | Deep Learning (`CNNs`, `Transformers`, `LLMs`) |
| :--- | :--- | :--- |
| **Best Modality** | Structured tabular data (FX/gold rates, banking, telecom churn) | Unstructured images, audio, and raw text |
| **Sample Efficiency** | Excels on `1k–100k` rows (our `3,487` CBSL days) in seconds on CPU | Requires large datasets (`100k+`) and GPUs |
| **Key Trade-off** | Cannot extrapolate raw trends without return/ARIMA differencing | Opaque and prone to overfitting small noisy financial tables |
        """
    )

    st.subheader("3. AI Usage")
    st.markdown(
        """
1. **Google AI Studio** was used in a complete lockdown mode (with no internet access) to guide the initial layout by discussing the type of dataset needed for such a study, while I found the datasets on my own from the Central Bank of Sri Lanka website and decided on merging the two datasets to create this project idea.
2. **Antigravity** was used as a coding assistant during this assignment for Python and Streamlit dashboard development, while I thoroughly reviewed and reworked the notebook to keep it focused and authored/reworked the markdown explanations myself.
3. **Presentation and README** were prepared on Canva and were not connected to the agent or any AI tool.
        """
    )

st.markdown("---")
st.caption("Student Index: `224112A` | Data Source: Central Bank of Sri Lanka (CBSL) | Academic ML Project — Not Financial Advice")
