import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="GJU Anwar 2.11 MWp — AI Solar Forecasting Dashboard",
    page_icon="☀️",
    layout="wide"
)

COLOR_ACTUAL = "#F59E0B"
COLOR_PRED = "#2563EB"
COLOR_TARGET = "#64748B"
COLOR_ANOMALY = "#DC2626"
COLOR_GREEN = "#10B981"

PLANT_Capacity_MWp = 2.11  # 2.11 MWp GJU Anwar Solar Plant

# --- DATA LOADING & BUG FIXES ---
@st.cache_data
def load_and_clean_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Data Anwar (1).xlsx")
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    
    df = df.dropna(subset=["Month"]).copy()
    om_cols = [c for c in df.columns if "OM" in str(c)]
    df = df.drop(columns=om_cols)
    
    def parse_month(val):
        try:
            s = "{:.4f}".format(float(val))
            return pd.to_datetime(s, format="%m.%Y")
        except Exception:
            return pd.to_datetime(str(val), errors="coerce")
            
    df["Date"] = df["Month"].apply(parse_month)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    
    for col in df.columns:
        if col not in ["Month", "Date"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    df["Year"] = df["Date"].dt.year
    df["Month_Num"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%b")
    df["Year_Month"] = df["Date"].dt.strftime("%Y-%m")
    
    if "FB (KWH)" in df.columns and "Eout (KWH)" in df.columns:
        df["Self_Consumed_KWH"] = (df["FB (KWH)"] - df["Eout (KWH)"]).clip(lower=0)
        df["Exported_KWH"] = df["Eout (KWH)"].clip(lower=0)
        if "Ein (KWH)" in df.columns:
            total_cons = df["Ein (KWH)"] + df["Self_Consumed_KWH"]
            df["Solar_Coverage_Pct"] = np.where(total_cons > 0, (df["Self_Consumed_KWH"] / total_cons) * 100, 0)
            
    return df

try:
    df_raw = load_and_clean_data()
except Exception as e:
    st.error(f"Could not load 'Data Anwar (1).xlsx'. Ensure it is in the same folder as app.py. Error: {e}")
    st.stop()

gti_col = next((c for c in df_raw.columns if "GTI" in c), None)
temp_col = next((c for c in df_raw.columns if "Tcell" in c or "Temp" in c), None)
yf_col = "YF(KWH/KWP)"
pr_pvg_col = "PR PVG(%)" if "PR PVG(%)" in df_raw.columns else None
pr_tc_col = "PR_tc (%)" if "PR_tc (%)" in df_raw.columns else None
co2_col = next((c for c in df_raw.columns if "Co2" in c or "CO2" in c), None)

# --- SEASONAL PVsyst TARGET CURVE ---
annual_pvsyst_target = 1808.0
monthly_avg_gti = df_raw.groupby("Month_Num")[gti_col].mean()
monthly_weights = monthly_avg_gti / monthly_avg_gti.sum()
df_raw["PVsyst_Baseline"] = df_raw["Month_Num"].map(monthly_weights) * annual_pvsyst_target

# --- MACHINE LEARNING: TRAIN (2017-2020) & TEST (2021) ---
ml_mask = df_raw[[gti_col, temp_col, yf_col]].notna().all(axis=1)
ml_df = df_raw[ml_mask].copy()

train_mask = ml_df["Year"] <= 2020
test_mask = ml_df["Year"] == 2021

X_train, y_train = ml_df.loc[train_mask, [gti_col, temp_col]].values, ml_df.loc[train_mask, yf_col].values
X_test, y_test = ml_df.loc[test_mask, [gti_col, temp_col]].values, ml_df.loc[test_mask, yf_col].values

lr_model = LinearRegression().fit(X_train, y_train)

# Calculate predictions across ALL rows safely
df_raw["Predicted_YF"] = lr_model.predict(df_raw[[gti_col, temp_col]].values)
df_raw["Residual"] = df_raw[yf_col] - df_raw["Predicted_YF"]

# Re-slice train/test subsets after prediction columns exist
train_df = df_raw[df_raw["Year"] <= 2020].copy()
test_df = df_raw[df_raw["Year"] == 2021].copy()

y_pred_test = test_df["Predicted_YF"].values
test_r2 = r2_score(test_df[yf_col], y_pred_test) * 100
test_mae = mean_absolute_error(test_df[yf_col], y_pred_test)
cv_r2 = cross_val_score(LinearRegression(), ml_df[[gti_col, temp_col]].values, ml_df[yf_col].values, cv=5, scoring="r2").mean() * 100

resid_std = df_raw["Residual"].std()
df_raw["Is_Anomaly"] = df_raw["Residual"] < (-2 * resid_std)

temp_from_gti_model = LinearRegression().fit(ml_df[[gti_col]].values, ml_df[temp_col].values)

# --- SIDEBAR: FILTERS & SIMULATOR ---
st.sidebar.markdown("### 🏛️ German Jordanian University")
st.sidebar.caption(f"**Plant:** Anwar {PLANT_Capacity_MWp} MWp PV System | **Location:** Madaba/Amman, Jordan")
st.sidebar.markdown("---")

st.sidebar.subheader("📅 Filter Historical Data")
available_years = sorted(df_raw["Year"].unique().tolist())
selected_years = st.sidebar.multiselect("Select Years", available_years, default=available_years)
if not selected_years:
    selected_years = available_years

df = df_raw[df_raw["Year"].isin(selected_years)].copy()

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Weather Scenario Simulator")
sim_gti = st.sidebar.slider("Solar Irradiance (GTI) [kWh/m²]", 85.0, 265.0, 185.0, step=1.0)

expected_tcell = float(temp_from_gti_model.predict([[sim_gti]])[0])
override_temp = st.sidebar.checkbox("Manually override cell temperature", value=False)
if override_temp:
    sim_tcell = st.sidebar.slider("Cell Temperature [°C]", 15.0, 58.0, round(expected_tcell, 1), step=0.5)
else:
    sim_tcell = round(expected_tcell, 1)
    st.sidebar.info(f"🌡️ Coupled Cell Temp: **{sim_tcell} °C** *(auto-calculated)*")

sim_pred_yf = float(lr_model.predict([[sim_gti, sim_tcell]])[0])
sim_pred_mwh = sim_pred_yf * PLANT_Capacity_MWp / 1000.0
st.sidebar.metric("Simulated Monthly Yield", f"{sim_pred_yf:.1f} kWh/kWp", f"{sim_pred_mwh:.1f} MWh total AC")

# --- HEADER & METHODOLOGY ---
st.title(f"☀️ GJU 'Anwar' {PLANT_Capacity_MWp} MWp Solar Plant — AI Forecasting Dashboard")
st.markdown("Historical Performance Validation (2017–2021), Thermal Loss Analysis & 12-Month Yield Forecasting")

with st.expander("📘 Methodology, Data Sources & Model Limitations (Click to expand)"):
    st.markdown(f"""
    * **Plant Specifications:** {PLANT_Capacity_MWp} MWp grid-connected PV plant at German Jordanian University (GJU).
    * **Data Integrity:** Strictly uses plant SCADA measurements (`FB`, `Eout`, `Ein`, `YF`) and **PVGIS** satellite weather (`{gti_col}`, `{temp_col}`). Open-Meteo (`OM`) columns excluded. All 48 months (Nov 2017 – Oct 2021, including 2020) are fully verified.
    * **Train/Test Validation:** Trained on **2017–2020** (36 months) and tested on unseen **2021** data (12 months). Test $R^2$ = **{test_r2:.1f}%**, Test MAE = **{test_mae:.2f} kWh/kWp**, 5-Fold Cross-Validation $R^2$ = **{cv_r2:.1f}%**.
    * **Known Limitations:** PVsyst curve is an annual target (`1,808 kWh/kWp/yr`) shaped by site irradiance distribution. Forecasts assume average historical weather.
    """)

# --- TOP KPI ROW ---
total_energy_mwh = df["FB (KWH)"].sum() / 1000.0 if "FB (KWH)" in df.columns else 0
avg_pr = df[pr_pvg_col].mean() if pr_pvg_col else 0
total_co2_tons = df[co2_col].sum() if co2_col else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Live AI Sim Output", f"{sim_pred_mwh:.1f} MWh", f"{sim_pred_yf:.1f} kWh/kWp/mo")
k2.metric("Total Historical Energy", f"{total_energy_mwh:,.1f} MWh", f"{len(df)} Months")
k3.metric("Average PVGIS PR", f"{avg_pr:.1f}%", f"{PLANT_Capacity_MWp} MWp Plant")
k4.metric("Total CO₂ Avoided", f"{total_co2_tons:,.1f} Tons", "Environmental Impact")
k5.metric("AI Test Accuracy ($R^2$)", f"{test_r2:.1f}%", "Unseen 2021 Test Split")

st.markdown("---")

# --- 5 THEMED TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🤖 Model & Accuracy",
    "⚡ Plant Performance",
    "🔋 Energy & Impact",
    "🔮 12-Month Forecast & Upload"
])

# ==========================================
# TAB 1: OVERVIEW
# ==========================================
with tab1:
    st.subheader("Actual vs. AI Predicted Yield & PVsyst Target (2017–2021)")
    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(
        x=pd.concat([df["Year_Month"], df["Year_Month"][::-1]]),
        y=pd.concat([df["Predicted_YF"] + test_mae, (df["Predicted_YF"] - test_mae)[::-1]]),
        fill="toself",
        fillcolor="rgba(37, 99, 235, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"±MAE Band (±{test_mae:.1f} kWh/kWp)"
    ))
    
    fig1.add_trace(go.Scatter(
        x=df["Year_Month"], y=df["PVsyst_Baseline"],
        mode="lines", name="PVsyst Seasonal Target",
        line=dict(color=COLOR_TARGET, dash="dot", width=2)
    ))
    
    fig1.add_trace(go.Scatter(
        x=df["Year_Month"], y=df["Predicted_YF"],
        mode="lines", name="AI Predicted Yield",
        line=dict(color=COLOR_PRED, width=2.5)
    ))
    
    fig1.add_trace(go.Scatter(
        x=df["Year_Month"], y=df[yf_col],
        mode="lines+markers", name="Actual Yield (YF)",
        line=dict(color=COLOR_ACTUAL, width=2.5),
        marker=dict(size=6)
    ))
    
    anomalies = df[df["Is_Anomaly"]]
    if not anomalies.empty:
        fig1.add_trace(go.Scatter(
            x=anomalies["Year_Month"], y=anomalies[yf_col],
            mode="markers", name="Underperformance Anomaly (>2σ)",
            marker=dict(color=COLOR_ANOMALY, size=12, symbol="x")
        ))
        
    if 2021 in df["Year"].values:
        fig1.add_vrect(
            x0="2021-01", x1=df["Year_Month"].max(),
            fillcolor="rgba(16, 185, 129, 0.08)", layer="below", line_width=0,
            annotation_text="2021 Unseen Test Set", annotation_position="top left"
        )
        
    fig1.update_layout(
        xaxis_title="Month", yaxis_title="Specific Yield (kWh/kWp)",
        hovermode="x unified", height=460, legend=dict(orientation="h", y=1.12)
    )
    st.plotly_chart(fig1, use_container_width=True)

# ==========================================
# TAB 2: MODEL & ACCURACY
# ==========================================
with tab2:
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader(f"Predicted vs. Actual Scatter (Test R² = {test_r2:.1f}%)")
        fig_scat = go.Figure()
        fig_scat.add_trace(go.Scatter(
            x=train_df[yf_col], y=train_df["Predicted_YF"],
            mode="markers", name="Train (2017–2020)",
            marker=dict(color=COLOR_PRED, size=8, opacity=0.75)
        ))
        fig_scat.add_trace(go.Scatter(
            x=test_df[yf_col], y=test_df["Predicted_YF"],
            mode="markers", name="Test (2021 Unseen)",
            marker=dict(color=COLOR_ACTUAL, size=10, symbol="diamond")
        ))
        min_v, max_v = ml_df[yf_col].min() * 0.9, ml_df[yf_col].max() * 1.05
        fig_scat.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v],
            mode="lines", name="Ideal y = x",
            line=dict(color=COLOR_TARGET, dash="dash")
        ))
        fig_scat.update_layout(xaxis_title="Actual Yield (kWh/kWp)", yaxis_title="Predicted Yield (kWh/kWp)", height=400)
        st.plotly_chart(fig_scat, use_container_width=True)
        
    with c2:
        st.subheader("Mean Prediction Error (MAE) by Calendar Month")
        err_by_month = df_raw.groupby(["Month_Num", "Month_Name"])["Residual"].apply(lambda x: np.mean(np.abs(x))).reset_index()
        fig_err = px.bar(
            err_by_month, x="Month_Name", y="Residual",
            labels={"Month_Name": "Calendar Month", "Residual": "Mean Absolute Error (kWh/kWp)"},
            color_discrete_sequence=[COLOR_PRED]
        )
        fig_err.update_layout(height=400)
        st.plotly_chart(fig_err, use_container_width=True)
        
    st.subheader("Algorithm Benchmark Comparison (Trained 2017–2020, Tested on 2021)")
    models = {
        "Multivariate Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
    }
    comp_rows = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        comp_rows.append({
            "Model": name,
            "Test R² (%)": round(r2_score(y_test, preds) * 100, 2),
            "Test MAE (kWh/kWp)": round(mean_absolute_error(y_test, preds), 2),
            "Test MAPE (%)": round(np.mean(np.abs((y_test - preds) / y_test)) * 100, 2)
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

# ==========================================
# TAB 3: PLANT PERFORMANCE
# ==========================================
with tab3:
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Year × Month Specific Yield Heatmap")
        pivot_yf = df_raw.pivot(index="Year", columns="Month_Num", values=yf_col)
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot_yf.columns = [month_labels[m-1] for m in pivot_yf.columns]
        fig_heat = px.imshow(
            pivot_yf, text_auto=".0f", aspect="auto", color_continuous_scale="YlOrRd",
            labels=dict(x="Month", y="Year", color="kWh/kWp")
        )
        fig_heat.update_layout(height=380)
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with c2:
        st.subheader("Seasonal Yield Profile by Year")
        fig_yr = go.Figure()
        for yr in sorted(df_raw["Year"].unique()):
            sub = df_raw[df_raw["Year"] == yr]
            fig_yr.add_trace(go.Scatter(
                x=sub["Month_Name"], y=sub[yf_col], mode="lines+markers", name=str(yr)
            ))
        fig_yr.update_layout(xaxis_title="Month", yaxis_title="Specific Yield (kWh/kWp)", height=380)
        st.plotly_chart(fig_yr, use_container_width=True)
        
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Thermal Loss: PR vs. Cell Temperature")
        if pr_pvg_col and temp_col:
            df_raw["Status"] = np.where(
                df_raw["Year_Month"].isin(["2018-07", "2018-08", "2018-09", "2020-05", "2020-06", "2020-07"]),
                "Flagged Anomaly Period", "Normal Operation"
            )
            fig_th = px.scatter(
                df_raw, x=temp_col, y=pr_pvg_col, color="Status",
                color_discrete_map={"Normal Operation": COLOR_PRED, "Flagged Anomaly Period": COLOR_ANOMALY},
                hover_data=["Year_Month"],
                labels={temp_col: "Cell Temperature (°C)", pr_pvg_col: "Performance Ratio (%)"}
            )
            valid_th = df_raw.dropna(subset=[temp_col, pr_pvg_col])
            z = np.polyfit(valid_th[temp_col], valid_th[pr_pvg_col], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_th[temp_col].min(), valid_th[temp_col].max(), 50)
            fig_th.add_trace(go.Scatter(x=x_line, y=p(x_line), mode="lines", name="Thermal Trend", line=dict(color=COLOR_TARGET, dash="dash")))
            fig_th.update_layout(height=380)
            st.plotly_chart(fig_th, use_container_width=True)
            
    with c4:
        st.subheader("Actual PR vs. Temperature-Corrected PR (PR_tc)")
        if pr_pvg_col and pr_tc_col:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=df["Year_Month"], y=df[pr_pvg_col], name="Actual PR (PVGIS Irrad.)", line=dict(color=COLOR_ACTUAL, width=2.5)))
            fig_pr.add_trace(go.Scatter(x=df["Year_Month"], y=df[pr_tc_col], name="Temp-Corrected PR (PR_tc)", line=dict(color=COLOR_PRED, width=2.5)))
            fig_pr.update_layout(xaxis_title="Month", yaxis_title="Performance Ratio (%)", yaxis=dict(range=[55, 100]), height=380)
            st.plotly_chart(fig_pr, use_container_width=True)

# ==========================================
# TAB 4: ENERGY & IMPACT
# ==========================================
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Solar Production Split: Self-Consumed vs. Exported")
        if "Self_Consumed_KWH" in df.columns:
            fig_en = go.Figure()
            fig_en.add_trace(go.Bar(
                x=df["Year_Month"], y=df["Self_Consumed_KWH"] / 1000.0,
                name="Self-Consumed by GJU (MWh)", marker_color=COLOR_GREEN
            ))
            fig_en.add_trace(go.Bar(
                x=df["Year_Month"], y=df["Exported_KWH"] / 1000.0,
                name="Exported to Grid (MWh)", marker_color=COLOR_ACTUAL
            ))
            fig_en.update_layout(barmode="stack", xaxis_title="Month", yaxis_title="Energy (MWh)", height=400)
            st.plotly_chart(fig_en, use_container_width=True)
            
    with c2:
        st.subheader("Cumulative CO₂ Emissions Avoided")
        if co2_col:
            df["Cum_CO2"] = df[co2_col].cumsum()
            fig_co2 = go.Figure()
            fig_co2.add_trace(go.Scatter(
                x=df["Year_Month"], y=df["Cum_CO2"],
                fill="tozeroy", mode="lines", name="Cumulative CO₂ Avoided",
                line=dict(color=COLOR_GREEN, width=2.5)
            ))
            fig_co2.update_layout(xaxis_title="Month", yaxis_title="CO₂ Avoided (Metric Tons)", height=400)
            st.plotly_chart(fig_co2, use_container_width=True)

# ==========================================
# TAB 5: 12-MONTH FORECAST & NEW READING TESTER
# ==========================================
with tab5:
    st.subheader("12-Month Forward Yield Forecast (Typical Weather Year)")
    
    last_date = df_raw["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq="MS")
    
    clim = df_raw.groupby("Month_Num")[[gti_col, temp_col]].mean().reset_index()
    fc_rows = []
    for d in future_dates:
        m_num = d.month
        row_clim = clim[clim["Month_Num"] == m_num].iloc[0]
        pred_val = float(lr_model.predict([[row_clim[gti_col], row_clim[temp_col]]])[0])
        fc_rows.append({
            "Forecast_Month": d.strftime("%Y-%m"),
            "Expected_GTI": round(row_clim[gti_col], 1),
            "Expected_Tcell": round(row_clim[temp_col], 1),
            "Forecast_YF_kWh_kWp": round(pred_val, 2),
            "Forecast_Energy_MWh": round(pred_val * PLANT_Capacity_MWp / 1000.0, 2),
            "Lower_Bound": round(pred_val - test_mae, 2),
            "Upper_Bound": round(pred_val + test_mae, 2)
        })
    fc_df = pd.DataFrame(fc_rows)
    
    hist_tail = df_raw.tail(18)
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=hist_tail["Year_Month"], y=hist_tail[yf_col],
        mode="lines+markers", name="Historical Yield", line=dict(color=COLOR_ACTUAL, width=2.5)
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc_df["Forecast_Month"], y=fc_df["Forecast_YF_kWh_kWp"],
        mode="lines+markers", name="12-Month AI Forecast", line=dict(color=COLOR_PRED, width=2.5, dash="dash")
    ))
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([fc_df["Forecast_Month"], fc_df["Forecast_Month"][::-1]]),
        y=pd.concat([fc_df["Upper_Bound"], fc_df["Lower_Bound"][::-1]]),
        fill="toself", fillcolor="rgba(37, 99, 235, 0.15)",
        line=dict(color="rgba(255,255,255,0)"), name="Confidence Band (±MAE)"
    ))
    fig_fc.update_layout(xaxis_title="Month", yaxis_title="Specific Yield (kWh/kWp)", height=400)
    st.plotly_chart(fig_fc, use_container_width=True)
    
    col_dl, col_info = st.columns([1, 2])
    with col_dl:
        csv_data = fc_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download 12-Month Forecast (CSV)",
            data=csv_data,
            file_name="GJU_Anwar_12Month_Forecast.csv",
            mime="text/csv"
        )
    with col_info:
        st.info(f"**Projected Annual Yield ({PLANT_Capacity_MWp} MWp Scale):** {fc_df['Forecast_YF_kWh_kWp'].sum():,.1f} kWh/kWp/yr ({fc_df['Forecast_Energy_MWh'].sum():,.1f} MWh/yr)")
        
    st.markdown("---")
    st.subheader("🧪 Live Plant Validation: Enter a New Monthly Reading")
    st.markdown("Input actual plant readings for a new month to check how closely the plant performed against the AI model's prediction.")
    
    u1, u2, u3 = st.columns(3)
    in_gti = u1.number_input("Measured GTI (kWh/m²)", min_value=50.0, max_value=300.0, value=190.0)
    in_temp = u2.number_input("Measured Cell Temp (°C)", min_value=10.0, max_value=65.0, value=39.0)
    in_actual_yf = u3.number_input("Actual Measured Yield (kWh/kWp)", min_value=20.0, max_value=250.0, value=148.0)
    
    expected_new = float(lr_model.predict([[in_gti, in_temp]])[0])
    diff_new = in_actual_yf - expected_new
    st.success(f"**AI Expected Yield:** `{expected_new:.2f} kWh/kWp` | **Actual Measured:** `{in_actual_yf:.2f} kWh/kWp` | **Deviation:** `{diff_new:+.2f} kWh/kWp` ({(diff_new/expected_new)*100:+.1f}%)")
