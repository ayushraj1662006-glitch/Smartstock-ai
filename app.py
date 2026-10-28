"""
SmartStock AI — FMCG Demand Forecasting & Inventory Optimization System
Enterprise Streamlit Application tailored for ITC Limited Logistics & Warehousing Case.

Authors: Undergraduate Engineering Project
Tech Stack: Python 3.11+, Streamlit, Pandas, NumPy, Statsmodels, Plotly, Openpyxl
"""

import os
import io
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Internal modules
from src.config import (
    COLORS,
    SERVICE_LEVELS,
    DEFAULT_SERVICE_LEVEL,
    DEFAULT_ANNUAL_HOLDING_RATE,
    DEFAULT_REVIEW_PERIOD_DAYS,
    REQUIRED_COLUMNS,
)
from src.data_generator import save_synthetic_data, generate_fmcg_dataset
from src.data_validation import (
    validate_and_clean_data,
    suggest_column_mapping,
    parse_uploaded_file,
)
from src.forecasting import (
    prepare_product_series,
    calculate_metrics,
    forecast_moving_average,
    forecast_exponential_smoothing,
    evaluate_models_chronological,
    run_future_forecast,
)
from src.inventory import (
    compute_portfolio_inventory_metrics,
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_inventory_position,
    calculate_replenishment_quantity,
)
from src.reporting import generate_executive_excel_report
from src.ui_components import (
    inject_custom_css,
    render_header,
    render_kpi_card,
    get_status_badge_html,
    create_plotly_theme,
)


# ==============================================================================
# Streamlit App Configuration
# ==============================================================================
st.set_page_config(
    page_title="SmartStock AI | FMCG Supply Chain Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Enterprise CSS
inject_custom_css()
plotly_theme = create_plotly_theme()


# ==============================================================================
# Session State Initialization & Data Loader
# ==============================================================================
@st.cache_data
def load_default_data() -> pd.DataFrame:
    """Loads default synthetic FMCG dataset from disk or generates if missing."""
    csv_path = os.path.join(os.path.dirname(__file__), "data", "fmcg_inventory_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = save_synthetic_data(csv_path=csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


if "raw_data" not in st.session_state:
    st.session_state.raw_data = load_default_data()

if "service_level" not in st.session_state:
    st.session_state.service_level = DEFAULT_SERVICE_LEVEL

if "holding_rate" not in st.session_state:
    st.session_state.holding_rate = DEFAULT_ANNUAL_HOLDING_RATE

if "review_period" not in st.session_state:
    st.session_state.review_period = DEFAULT_REVIEW_PERIOD_DAYS

if "include_lt_var" not in st.session_state:
    st.session_state.include_lt_var = True

# Recompute inventory table when data or parameters change
@st.cache_data
def get_inventory_table(
    df: pd.DataFrame,
    service_level: str,
    holding_rate: float,
    review_period: int,
    include_lt_var: bool,
) -> pd.DataFrame:
    return compute_portfolio_inventory_metrics(
        df=df,
        service_level_str=service_level,
        annual_holding_rate=holding_rate,
        review_period_days=review_period,
        include_lead_time_variance=include_lt_var,
    )


# ==============================================================================
# Sidebar Navigation & Global Controls
# ==============================================================================
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #1E293B; margin-bottom: 16px;">
            <div style="font-size: 20px; font-weight: 700; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                <span>📦</span> SmartStock AI
            </div>
            <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">
                FMCG Demand & Inventory Optimization
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "📊 Executive Dashboard",
            "📈 Demand Forecasting",
            "⚙️ Inventory Optimization",
            "🔍 Product Explorer",
            "📁 Excel & Data Management",
            "📑 Reports & Exports",
            "📚 Methodology & Guide",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🎛️ Supply Chain Parameters")

    selected_sl = st.selectbox(
        "Service Level (Target Fill Rate)",
        options=list(SERVICE_LEVELS.keys()),
        index=list(SERVICE_LEVELS.keys()).index(st.session_state.service_level),
        help="Probability of fulfilling demand from stock during the replenishment cycle.",
    )
    st.session_state.service_level = selected_sl

    selected_holding_rate = st.slider(
        "Annual Holding Cost Rate (%)",
        min_value=10.0,
        max_value=35.0,
        value=float(st.session_state.holding_rate * 100.0),
        step=1.0,
        help="Annual capital cost, warehousing, storage, insurance, and obsolescence.",
    ) / 100.0
    st.session_state.holding_rate = selected_holding_rate

    selected_review_period = st.slider(
        "Review Period (Days)",
        min_value=3,
        max_value=21,
        value=int(st.session_state.review_period),
        step=1,
        help="Days between periodic inventory reviews and purchase order placements.",
    )
    st.session_state.review_period = selected_review_period

    include_lt_var = st.checkbox(
        "Include Lead Time Variability (σ_L)",
        value=st.session_state.include_lt_var,
        help="When checked, accounts for supplier delivery delays in safety stock formula.",
    )
    st.session_state.include_lt_var = include_lt_var

    st.markdown("---")
    # Active Dataset info badge
    if not st.session_state.raw_data.empty and pd.notna(st.session_state.raw_data["Date"].min()):
        min_d = st.session_state.raw_data["Date"].min().strftime("%b %d, %Y")
        max_d = st.session_state.raw_data["Date"].max().strftime("%b %d, %Y")
    else:
        min_d = "N/A"
        max_d = "N/A"
    total_skus = st.session_state.raw_data["Product_ID"].nunique() if not st.session_state.raw_data.empty else 0

    st.markdown(
        f"""
        <div style="background: #1E293B; border-radius: 8px; padding: 12px; font-size: 11.5px; color: #CBD5E1; line-height: 1.5;">
            <div style="font-weight: 600; color: #38BDF8; margin-bottom: 4px;">📂 Active Dataset</div>
            <div><strong>SKUs:</strong> {total_skus} FMCG Products</div>
            <div><strong>Span:</strong> {min_d} – {max_d}</div>
            <div><strong>Records:</strong> {len(st.session_state.raw_data):,} rows</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔄 Reset to Default Dataset", use_container_width=True):
        st.session_state.raw_data = load_default_data()
        st.rerun()


# Active inventory metrics (computed synchronously with the latest parameters)
inv_df = get_inventory_table(
    st.session_state.raw_data,
    st.session_state.service_level,
    st.session_state.holding_rate,
    st.session_state.review_period,
    st.session_state.include_lt_var,
)


# ==============================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# ==============================================================================
if page == "📊 Executive Dashboard":
    render_header(
        "Executive Dashboard",
        "Multi-Echelon Inventory Health, Stockout Risk & Demand Overview",
        "ITC FMCG Distribution Hub",
    )

    # 1. Global Filters Bar
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        col_f1, col_f2, col_f3 = st.columns([1.5, 1.5, 1])

        with col_f1:
            all_categories = sorted(st.session_state.raw_data["Category"].unique().tolist())
            selected_categories = st.multiselect(
                "Filter by Vertical / Category",
                options=all_categories,
                default=all_categories,
            )

        with col_f2:
            min_date_available = st.session_state.raw_data["Date"].min().date()
            max_date_available = st.session_state.raw_data["Date"].max().date()
            date_range = st.date_input(
                "Historical Date Window",
                value=(min_date_available, max_date_available),
                min_value=min_date_available,
                max_value=max_date_available,
            )

        with col_f3:
            available_skus = st.session_state.raw_data[
                st.session_state.raw_data["Category"].isin(selected_categories)
            ]["Product_Name"].unique()
            selected_skus = st.multiselect(
                "Filter by SKU (Optional)",
                options=available_skus,
                default=[],
                placeholder="All SKUs in category",
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Apply filters to raw and inventory frames
    filtered_raw = st.session_state.raw_data[
        st.session_state.raw_data["Category"].isin(selected_categories)
    ].copy()

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_raw = filtered_raw[
            (filtered_raw["Date"] >= start_d) & (filtered_raw["Date"] <= end_d)
        ]

    filtered_inv = inv_df[inv_df["Category"].isin(selected_categories)].copy()
    if selected_skus:
        filtered_raw = filtered_raw[filtered_raw["Product_Name"].isin(selected_skus)]
        filtered_inv = filtered_inv[filtered_inv["Product_Name"].isin(selected_skus)]

    if filtered_inv.empty:
        st.warning("No records match the selected filter criteria. Please broaden your selection.")
        st.stop()

    # 2. Executive KPI Cards
    total_products = len(filtered_inv)
    total_units_on_hand = int(filtered_inv["On_Hand_Stock"].sum())
    total_inventory_val = float(filtered_inv["Inventory_Value"].sum())
    reorder_required_count = int((filtered_inv["Inventory_Status"] == "Reorder Required").sum())
    stockout_risk_count = int((filtered_inv["Inventory_Status"] == "Potential Stockout Risk").sum())
    excess_count = int((filtered_inv["Inventory_Status"] == "Excess Inventory").sum())

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    with kpi1:
        render_kpi_card("Total Products", f"{total_products}", "Active Portfolio SKUs", "secondary", "🏷️")
    with kpi2:
        render_kpi_card("Units On Hand", f"{total_units_on_hand:,}", "Physical Depots Stock", "secondary", "📦")
    with kpi3:
        render_kpi_card("Inventory Value", f"₹{total_inventory_val/1e5:,.2f} L", "Current Working Capital", "secondary", "💰")
    with kpi4:
        render_kpi_card("Reorder Needed", f"{reorder_required_count}", f"{(reorder_required_count/total_products)*100:.0f}% of portfolio", "warning", "🔄")
    with kpi5:
        render_kpi_card("Stockout Risk", f"{stockout_risk_count}", "Critically low buffer", "danger", "⚠️")
    with kpi6:
        render_kpi_card("Excess Stock", f"{excess_count}", "Working capital tied up", "info", "📈")

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Interactive Charts - Row 1
    c1, c2 = st.columns([1.5, 1])

    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📅 Historical Monthly Demand Trajectory</div>', unsafe_allow_html=True)
        
        if not filtered_raw.empty:
            monthly_demand = filtered_raw.copy()
            monthly_demand["YearMonth"] = monthly_demand["Date"].dt.to_period("M").astype(str)
            monthly_agg = monthly_demand.groupby(["YearMonth", "Category"])["Units_Sold"].sum().reset_index()

            fig_month = px.bar(
                monthly_agg,
                x="YearMonth",
                y="Units_Sold",
                color="Category",
                barmode="stack",
                color_discrete_sequence=[COLORS["primary"], COLORS["secondary"], COLORS["accent"], "#F59E0B", "#10B981"],
            )
            fig_month.update_layout(plotly_theme)
            fig_month.update_layout(
                height=320,
                xaxis_title="Month",
                yaxis_title="Total Units Sold",
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_month, use_container_width=True)
        else:
            st.info("No transaction data available for the selected filters.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Capital Distribution by Category</div>', unsafe_allow_html=True)
        
        cat_val = filtered_inv.groupby("Category")["Inventory_Value"].sum().reset_index()
        fig_pie = px.pie(
            cat_val,
            values="Inventory_Value",
            names="Category",
            hole=0.48,
            color_discrete_sequence=[COLORS["primary"], COLORS["secondary"], COLORS["accent"], "#F59E0B", "#10B981"],
        )
        fig_pie.update_layout(plotly_theme)
        fig_pie.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="v", x=1.0, y=0.5),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. Interactive Charts - Row 2
    c3, c4 = st.columns([1.2, 1.2])

    with c3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔥 Highest-Demand Products (Top 8 SKUs)</div>', unsafe_allow_html=True)
        
        if not filtered_raw.empty:
            top_demand = filtered_raw.groupby("Product_Name")["Units_Sold"].sum().reset_index()
            top_demand = top_demand.sort_values(by="Units_Sold", ascending=True).tail(8)

            fig_top = px.bar(
                top_demand,
                x="Units_Sold",
                y="Product_Name",
                orientation="h",
                color_discrete_sequence=[COLORS["secondary"]],
            )
            fig_top.update_layout(plotly_theme)
            fig_top.update_layout(
                height=320,
                xaxis_title="Total Units Sold (Period)",
                yaxis_title="",
                margin=dict(l=10, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("No transaction data available.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🚨 Inventory Position vs Reorder Point (Deficit Analysis)</div>', unsafe_allow_html=True)
        
        reorder_skus = filtered_inv[filtered_inv["Inventory_Position"] <= filtered_inv["Reorder_Point"]].copy()
        if not reorder_skus.empty:
            reorder_skus = reorder_skus.sort_values(by="Inventory_Position").head(7)
            fig_rop = go.Figure()
            fig_rop.add_trace(go.Bar(
                name="Inventory Position",
                x=reorder_skus["Product_Name"],
                y=reorder_skus["Inventory_Position"],
                marker_color="#DC2626",
            ))
            fig_rop.add_trace(go.Bar(
                name="Reorder Point (ROP)",
                x=reorder_skus["Product_Name"],
                y=reorder_skus["Reorder_Point"],
                marker_color="#CBD5E1",
            ))
            fig_rop.update_layout(barmode="group")
            fig_rop.update_layout(plotly_theme)
            fig_rop.update_layout(
                height=320,
                xaxis_title="",
                yaxis_title="Units",
                xaxis_tickangle=-25,
            )
            st.plotly_chart(fig_rop, use_container_width=True)
        else:
            st.success("🎉 All SKUs in selected category are currently above their Reorder Point!")
        st.markdown('</div>', unsafe_allow_html=True)

    # 5. Critical Priority Action Items
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚡ Immediate Warehousing Action Items</div>', unsafe_allow_html=True)
    action_items = filtered_inv[
        filtered_inv["Inventory_Status"].isin(["Potential Stockout Risk", "Reorder Required"])
    ].copy()

    if not action_items.empty:
        action_table = action_items[[
            "Product_ID", "Product_Name", "Category", "On_Hand_Stock", "On_Order_Stock",
            "Inventory_Position", "Reorder_Point", "Recommended_Order_Qty", "Unit_Cost", "Inventory_Status"
        ]].copy()
        action_table["Action_Required"] = np.where(
            action_table["Recommended_Order_Qty"] > 0,
            "Place Purchase Order",
            "Expedite In-Transit PO"
        )
        action_table["Est_PO_Value"] = action_table["Recommended_Order_Qty"] * action_table["Unit_Cost"]

        st.dataframe(
            action_table[[
                "Product_ID", "Product_Name", "Category", "On_Hand_Stock", "On_Order_Stock",
                "Inventory_Position", "Reorder_Point", "Action_Required", "Recommended_Order_Qty",
                "Unit_Cost", "Est_PO_Value", "Inventory_Status"
            ]].style.format({
                "On_Hand_Stock": "{:,}",
                "On_Order_Stock": "{:,}",
                "Inventory_Position": "{:,}",
                "Reorder_Point": "{:,}",
                "Recommended_Order_Qty": "{:,}",
                "Unit_Cost": "₹{:,.2f}",
                "Est_PO_Value": "₹{:,.2f}",
            }),
            use_container_width=True,
            height=240,
        )
    else:
        st.info("No immediate replenishment actions required for the filtered products.")
    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: DEMAND FORECASTING
# ==============================================================================
elif page == "📈 Demand Forecasting":
    render_header(
        "Demand Forecasting Engine",
        "Chronological Time-Series Modeling, Holt-Winters Exponential Smoothing & Evaluation",
        "Statistical Forecasting Suite",
    )

    if inv_df.empty or st.session_state.raw_data.empty:
        st.warning("No product data available for forecasting. Please check or reload data.")
        st.stop()

    # 1. Product & Horizon Selection
    col_fc1, col_fc2, col_fc3 = st.columns([2, 1, 1])

    product_options = {
        f"{row['Product_ID']} — {row['Product_Name']} ({row['Category']})": row["Product_ID"]
        for _, row in inv_df.iterrows()
    }

    with col_fc1:
        selected_label = st.selectbox(
            "Select FMCG Product for Forecasting",
            options=list(product_options.keys()),
            index=0,
        )
        selected_sku = product_options[selected_label]

    with col_fc2:
        horizon_days = st.radio(
            "Forecast Horizon",
            options=[7, 14, 30],
            index=1,
            horizontal=True,
            format_func=lambda x: f"{x} Days",
        )

    with col_fc3:
        forecast_method = st.selectbox(
            "Forecasting Model",
            options=["Exponential Smoothing (Holt-Winters)", "Moving Average Baseline"],
            index=0,
        )

    # Prepare daily series
    daily_series = prepare_product_series(st.session_state.raw_data, selected_sku)
    sku_meta_rows = inv_df[inv_df["Product_ID"] == selected_sku]
    if sku_meta_rows.empty:
        st.error("Product metadata not found in the current inventory catalog.")
        st.stop()
    sku_meta = sku_meta_rows.iloc[0]

    if len(daily_series) < 14:
        st.error("Insufficient historical records for this SKU to perform time-series modeling.")
        st.stop()

    # 2. Chronological Backtest Evaluation (Zero Data Leakage)
    backtest_res = evaluate_models_chronological(daily_series, test_horizon=horizon_days, ma_window=14)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Out-of-Sample Chronological Model Evaluation</div>', unsafe_allow_html=True)
    st.caption(
        f"Evaluated strictly on the most recent {horizon_days} calendar days (Unseen Test Split). "
        "Training data strictly precedes test data — zero future data leakage. "
        "WAPE is preferred over MAPE as it eliminates zero-demand division distortion."
    )

    if backtest_res.get("has_data"):
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        exp_m = backtest_res["exp_metrics"]
        ma_m = backtest_res["ma_metrics"]

        with m_col1:
            render_kpi_card(
                "Holt-Winters WAPE",
                f"{exp_m['WAPE']:.1f}%",
                f"MAE: {exp_m['MAE']} units | Bias: {exp_m['Bias']:+.1f}%",
                "success" if exp_m["WAPE"] <= ma_m["WAPE"] else "warning",
                "🎯",
            )
        with m_col2:
            render_kpi_card(
                "Moving Avg WAPE",
                f"{ma_m['WAPE']:.1f}%",
                f"MAE: {ma_m['MAE']} units | Bias: {ma_m['Bias']:+.1f}%",
                "secondary",
                "📏",
            )
        with m_col3:
            winner = "Holt-Winters" if exp_m["WAPE"] <= ma_m["WAPE"] else "Moving Average"
            render_kpi_card("Champion Model", winner, f"Selected: {backtest_res['exp_model_name']}", "secondary", "🏆")
        with m_col4:
            render_kpi_card("Train / Test Split", f"{backtest_res['train_size']}d / {backtest_res['test_size']}d", "Chronological Holdout", "secondary", "✂️")
    else:
        st.info(backtest_res.get("error", "Insufficient historical records for chronological train/test split at this horizon."))

    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Future Horizon Forecast Generation
    method_key = "Moving Average" if "Moving Average" in forecast_method else "Exponential Smoothing"
    future_fc, active_model_name, forecast_table = run_future_forecast(
        daily_series, method=method_key, horizon=horizon_days, ma_window=14
    )

    # 4. Interactive Plotly Forecast Chart
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">📈 Demand Forecast & Backtest Trajectory — {sku_meta["Product_Name"]}</div>', unsafe_allow_html=True)

    fig_fc = go.Figure()

    # Historical demand (last 90 days for clean chart visual)
    display_history = daily_series.iloc[-90:]
    fig_fc.add_trace(go.Scatter(
        x=display_history.index,
        y=display_history.values,
        name="Historical Actual Demand",
        mode="lines",
        line=dict(color="#64748B", width=1.5),
    ))

    # Backtest Test Actuals
    if backtest_res.get("has_data"):
        test_s = backtest_res["test_series"]
        fig_fc.add_trace(go.Scatter(
            x=test_s.index,
            y=test_s.values,
            name=f"Holdout Actuals (Last {horizon_days}d)",
            mode="lines+markers",
            marker=dict(size=6, color=COLORS["primary"]),
            line=dict(color=COLORS["primary"], width=2),
        ))
        # Backtest Prediction
        fig_fc.add_trace(go.Scatter(
            x=test_s.index,
            y=backtest_res["exp_preds"].values,
            name="Backtest Fit",
            mode="lines",
            line=dict(color="#F59E0B", width=2, dash="dot"),
        ))

    # Future Forecast
    fig_fc.add_trace(go.Scatter(
        x=future_fc.index,
        y=future_fc.values,
        name=f"Future Forecast ({horizon_days}d)",
        mode="lines+markers",
        marker=dict(size=6, color=COLORS["accent"]),
        line=dict(color=COLORS["accent"], width=2.5),
    ))

    # 95% Confidence Interval Band
    upper_bounds = forecast_table["Upper_Bound_95%"].values
    lower_bounds = forecast_table["Lower_Bound_95%"].values
    fig_fc.add_trace(go.Scatter(
        x=list(future_fc.index) + list(future_fc.index[::-1]),
        y=list(upper_bounds) + list(lower_bounds[::-1]),
        fill="toself",
        fillcolor="rgba(6, 182, 212, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="95% Confidence Interval",
    ))

    fig_fc.update_layout(plotly_theme)
    fig_fc.update_layout(
        height=380,
        xaxis_title="Calendar Date",
        yaxis_title="Units Demanded",
        hovermode="x unified",
    )
    st.plotly_chart(fig_fc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 5. Forecast Values Table & Method Explanation
    col_t1, col_t2 = st.columns([1.2, 1])

    with col_t1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">📋 Daily Projected Demand (Next {horizon_days} Days)</div>', unsafe_allow_html=True)
        st.dataframe(
            forecast_table.style.format({
                "Forecast_Units": "{:.1f}",
                "Lower_Bound_95%": "{:.1f}",
                "Upper_Bound_95%": "{:.1f}",
                "Cumulative_Forecast": "{:.1f}",
            }),
            use_container_width=True,
            height=280,
        )
        csv_fc = forecast_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Daily Forecast CSV",
            data=csv_fc,
            file_name=f"Forecast_{selected_sku}_{horizon_days}d.csv",
            mime="text/csv",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_t2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💡 Statistical Method Rationale</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            **Active Model:** `{active_model_name}`

            - **Holt-Winters Exponential Smoothing:** Decomposes the time series into level, trend, and periodic weekly seasonality:
              $$\\hat{y}_{{t+h}} = \\ell_t + h b_t + s_{{t+h-m}}$$
              In consumer goods (FMCG), consumer purchasing pulses heavily on weekends, creating strong 7-day cyclicality.
            - **Handling Intermittent & Zero Demand:** Missing calendar dates are filled with zero demand. The model handles zero-sales days gracefully.
            - **Why WAPE over MAPE?**
              $$\\text{{WAPE}} = \\frac{{\\sum |y_t - \\hat{{y}}_t|}}{{\\sum y_t}} \\times 100\\%$$
              When an FMCG product sells 0 units on a given day, standard MAPE ($\\frac{{|y_t - \\hat{{y}}_t|}}{{y_t}}$) produces an undefined division by zero ($0/0$). WAPE weights errors by volume, making it the industry standard in FMCG.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: INVENTORY OPTIMIZATION
# ==============================================================================
elif page == "⚙️ Inventory Optimization":
    render_header(
        "Inventory Optimization Engine",
        "Safety Stock Modeling, Reorder Points & Net Replenishment Recommendations",
        "Stochastic Supply Chain Modeling",
    )

    # 1. Summary Metrics & Assumptions Banner
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)

    total_po_units = int(inv_df["Recommended_Order_Qty"].sum())
    total_po_val = float((inv_df["Recommended_Order_Qty"] * inv_df["Unit_Cost"]).sum())
    avg_dos = float(inv_df["Days_of_Supply"].replace(999.0, np.nan).mean())
    skus_to_order = int((inv_df["Recommended_Order_Qty"] > 0).sum())

    with col_opt1:
        render_kpi_card("Replenishment Volume", f"{total_po_units:,} units", f"{skus_to_order} SKUs requiring PO", "warning", "📦")
    with col_opt2:
        render_kpi_card("Recommended PO Capital", f"₹{total_po_val/1e5:,.2f} L", "Working capital required", "secondary", "💳")
    with col_opt3:
        render_kpi_card("Average Days of Supply", f"{avg_dos:.1f} Days", "Overall portfolio coverage", "secondary", "⏳")
    with col_opt4:
        render_kpi_card("Active Service Level", f"{st.session_state.service_level.split()[0]}", f"Holding Rate: {st.session_state.holding_rate*100:.0f}% p.a.", "secondary", "🛡️")
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Status Filter Tabs
    st.markdown("### 📋 Portfolio Stock Level Analysis")
    status_filter = st.radio(
        "Filter Portfolio by Status",
        options=["All Statuses", "Reorder Required", "Potential Stockout Risk", "Excess Inventory", "Healthy Stock"],
        index=0,
        horizontal=True,
    )

    display_inv = inv_df.copy()
    if status_filter != "All Statuses":
        display_inv = display_inv[display_inv["Inventory_Status"] == status_filter]

    # Display Table
    table_cols = [
        "Product_ID", "Product_Name", "Category", "On_Hand_Stock", "On_Order_Stock",
        "Inventory_Position", "Lead_Time_Demand", "Safety_Stock", "Reorder_Point",
        "Recommended_Order_Qty", "Days_of_Supply", "Unit_Cost", "Inventory_Status"
    ]
    st.dataframe(
        display_inv[table_cols].style.format({
            "On_Hand_Stock": "{:,}",
            "On_Order_Stock": "{:,}",
            "Inventory_Position": "{:,}",
            "Lead_Time_Demand": "{:.1f}",
            "Safety_Stock": "{:,}",
            "Reorder_Point": "{:,}",
            "Recommended_Order_Qty": "{:,}",
            "Days_of_Supply": "{:.1f}",
            "Unit_Cost": "₹{:,.2f}",
        }),
        use_container_width=True,
        height=320,
    )

    # 3. Mathematical SKU Drill-Down
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Analytical Recommendation Audit (SKU Deep-Dive)</div>', unsafe_allow_html=True)

    sku_drill_options = {
        f"{row['Product_ID']} — {row['Product_Name']} [{row['Inventory_Status']}]": row["Product_ID"]
        for _, row in inv_df.iterrows()
    }
    drill_sku_label = st.selectbox("Select Product to inspect calculation breakdown", options=list(sku_drill_options.keys()))
    drill_sku_id = sku_drill_options[drill_sku_label]
    sku_row = inv_df[inv_df["Product_ID"] == drill_sku_id].iloc[0]

    # Recompute safety stock breakdown details
    ss_val, z_val, ss_bk = calculate_safety_stock(
        avg_daily_demand=sku_row["Avg_Daily_Demand"],
        std_daily_demand=sku_row["Std_Daily_Demand"],
        lead_time_days=sku_row["Lead_Time_Days"],
        std_lead_time_days=sku_row["Lead_Time_Std"],
        service_level_str=st.session_state.service_level,
        include_lead_time_variance=st.session_state.include_lt_var,
    )

    c_d1, c_d2 = st.columns([1.2, 1])

    with c_d1:
        st.markdown(
            f"""
            #### 🧮 Step-by-Step Supply Chain Formulas
            
            1. **Inventory Position ($IP$):**
               $$IP = \\text{{On-Hand}} + \\text{{On-Order}} - \\text{{Backorders}}$$
               $$IP = {sku_row['On_Hand_Stock']} + {sku_row['On_Order_Stock']} - {sku_row['Backorders']} = \\mathbf{{{sku_row['Inventory_Position']}}} \\text{{ units}}$$
               *Note: Including On-Order prevents double-ordering shipments already in transit.*

            2. **Expected Lead-Time Demand ($LTD$):**
               $$LTD = \\bar{{d}} \\times L = {sku_row['Avg_Daily_Demand']} \\times {sku_row['Lead_Time_Days']} = \\mathbf{{{sku_row['Lead_Time_Demand']}}} \\text{{ units}}$$

            3. **Safety Stock ($SS$):**
               $$SS = Z \\times \\sqrt{{L \\cdot \\sigma_d^2 + \\bar{{d}}^2 \\cdot \\sigma_L^2}}$$
               $$SS = {z_val} \\times \\sqrt{{{sku_row['Lead_Time_Days']} \\cdot ({sku_row['Std_Daily_Demand']})^2 + ({sku_row['Avg_Daily_Demand']})^2 \\cdot ({sku_row['Lead_Time_Std']})^2}} = \\mathbf{{{sku_row['Safety_Stock']}}} \\text{{ units}}$$
               - Demand Variability Contribution: `{ss_bk['demand_variance_contrib']:.1f}`
               - Supplier Lead-Time Delay Contribution: `{ss_bk['lead_time_variance_contrib']:.1f}`

            4. **Reorder Point ($ROP$):**
               $$ROP = LTD + SS = {sku_row['Lead_Time_Demand']} + {sku_row['Safety_Stock']} = \\mathbf{{{sku_row['Reorder_Point']}}} \\text{{ units}}$$
            """
        )

    with c_d2:
        status_badge = get_status_badge_html(sku_row["Inventory_Status"])
        st.markdown(
            f"""
            #### 📦 Decision Recommendation
            <div style="margin-bottom: 12px;">{status_badge}</div>
            
            - **Inventory Position vs ROP:** ${sku_row['Inventory_Position']} \\le {sku_row['Reorder_Point']}$ $\\rightarrow$ **{'YES (Order Triggered)' if sku_row['Inventory_Position'] <= sku_row['Reorder_Point'] else 'NO (Sufficient Stock)'}**
            - **Recommended Order Quantity:** **`{sku_row['Recommended_Order_Qty']}` units**
            - **Estimated PO Value:** **₹`{sku_row['Recommended_Order_Qty'] * sku_row['Unit_Cost']:,.2f}`**
            - **Current Days of Supply:** `{sku_row['Days_of_Supply']}` days

            **Auditor Explanation:**
            _{sku_row['Recommendation_Reason']}_
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: PRODUCT EXPLORER
# ==============================================================================
elif page == "🔍 Product Explorer":
    render_header(
        "Product Explorer & SKU Deep-Dive",
        "Granular Inventory Profiles, Demand Patterns & Buffer Waterline Gauges",
        "SKU Intelligence Center",
    )

    # 1. Search and Filters
    col_s1, col_s2, col_s3 = st.columns([1.5, 1, 1])
    with col_s1:
        search_query = st.text_input("🔍 Search SKU or Product Name", placeholder="e.g. Atta, Dark Fantasy, Savlon...")
    with col_s2:
        exp_cat = st.selectbox("Category", options=["All"] + sorted(inv_df["Category"].unique().tolist()))
    with col_s3:
        exp_status = st.selectbox("Inventory Status", options=["All"] + sorted(inv_df["Inventory_Status"].unique().tolist()))

    filtered_skus = inv_df.copy()
    if search_query:
        filtered_skus = filtered_skus[
            filtered_skus["Product_Name"].str.contains(search_query, case=False)
            | filtered_skus["Product_ID"].str.contains(search_query, case=False)
        ]
    if exp_cat != "All":
        filtered_skus = filtered_skus[filtered_skus["Category"] == exp_cat]
    if exp_status != "All":
        filtered_skus = filtered_skus[filtered_skus["Inventory_Status"] == exp_status]

    # Grid Display
    st.dataframe(
        filtered_skus[[
            "Product_ID", "Product_Name", "Category", "On_Hand_Stock", "Inventory_Position",
            "Avg_Daily_Demand", "Safety_Stock", "Reorder_Point", "Recommended_Order_Qty",
            "Days_of_Supply", "Unit_Cost", "Inventory_Status"
        ]].style.format({
            "On_Hand_Stock": "{:,}",
            "Inventory_Position": "{:,}",
            "Avg_Daily_Demand": "{:.1f}",
            "Safety_Stock": "{:,}",
            "Reorder_Point": "{:,}",
            "Recommended_Order_Qty": "{:,}",
            "Days_of_Supply": "{:.1f}",
            "Unit_Cost": "₹{:,.2f}",
        }),
        use_container_width=True,
        height=260,
    )

    if filtered_skus.empty:
        st.info("No products match your search filters.")
        st.stop()

    # 2. Detailed Single SKU Inspector
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Granular SKU Diagnostic Card</div>', unsafe_allow_html=True)

    sku_select = st.selectbox(
        "Choose SKU to display demand trajectory and buffer waterline",
        options=filtered_skus["Product_ID"].tolist(),
        format_func=lambda x: f"{x} — {inv_df.loc[inv_df['Product_ID'] == x, 'Product_Name'].values[0]}",
    )

    sku_info = inv_df[inv_df["Product_ID"] == sku_select].iloc[0]
    sku_series = prepare_product_series(st.session_state.raw_data, sku_select)

    col_g1, col_g2 = st.columns([1.4, 1])

    with col_g1:
        # Demand trajectory
        fig_trend = px.line(
            x=sku_series.index[-120:],
            y=sku_series.values[-120:],
            labels={"x": "Date", "y": "Daily Units Sold"},
            title=f"120-Day Demand History — {sku_info['Product_Name']}",
        )
        fig_trend.update_traces(line=dict(color=COLORS["secondary"], width=2))
        # Add 30-day moving average overlay (with min_periods=1 to support shorter series)
        window_sz = min(30, max(1, len(sku_series)))
        ma30 = sku_series.rolling(window_sz, min_periods=1).mean().iloc[-120:]
        fig_trend.add_trace(go.Scatter(
            x=ma30.index,
            y=ma30.values,
            name="30-Day Moving Avg",
            line=dict(color="#F59E0B", width=2, dash="dash"),
        ))
        fig_trend.update_layout(plotly_theme)
        fig_trend.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=30))
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_g2:
        # Buffer Waterline Chart (Bullet / Stacked comparison)
        fig_waterline = go.Figure()
        fig_waterline.add_trace(go.Bar(
            name="On-Hand Physical Stock",
            x=["Inventory Levels"],
            y=[sku_info["On_Hand_Stock"]],
            marker_color=COLORS["primary"],
        ))
        fig_waterline.add_trace(go.Bar(
            name="Pipeline On-Order",
            x=["Inventory Levels"],
            y=[sku_info["On_Order_Stock"]],
            marker_color=COLORS["accent"],
        ))
        # ROP Line
        fig_waterline.add_hline(
            y=sku_info["Reorder_Point"],
            line_dash="dot",
            line_color="#DC2626",
            annotation_text=f"ROP ({sku_info['Reorder_Point']})",
            annotation_position="bottom right",
        )
        # Safety Stock Line
        fig_waterline.add_hline(
            y=sku_info["Safety_Stock"],
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text=f"Safety Stock ({sku_info['Safety_Stock']})",
            annotation_position="top right",
        )
        fig_waterline.update_layout(barmode="stack")
        fig_waterline.update_layout(plotly_theme)
        fig_waterline.update_layout(
            title=f"Buffer Waterline Analysis",
            height=300,
            margin=dict(l=10, r=10, t=30, b=30),
        )
        st.plotly_chart(fig_waterline, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 5: EXCEL AND DATA MANAGEMENT
# ==============================================================================
elif page == "📁 Excel & Data Management":
    render_header(
        "Excel & Data Management Hub",
        "Upload Custom Datasets, Schema Validation, Column Mapping & Sample Downloads",
        "Data Pipeline & Integration",
    )

    tab_upload, tab_template, tab_schema = st.tabs(["📤 Upload & Validate Dataset", "📥 Download Sample Workbook", "📖 Schema Specification"])

    with tab_upload:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Upload Custom Inventory Data (.csv or .xlsx)</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Select an Excel (.xlsx) or CSV file containing daily sales & inventory records",
            type=["csv", "xlsx", "xls"],
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            raw_parsed = parse_uploaded_file(file_bytes, uploaded_file.name)

            if raw_parsed is not None and not raw_parsed.empty:
                st.success(f"File parsed successfully: **{uploaded_file.name}** ({len(raw_parsed):,} rows detected)")

                # Column mapping helper
                suggested_map = suggest_column_mapping(raw_parsed.columns.tolist())
                unmapped_reqs = [col for col, mapped in suggested_map.items() if mapped is None]

                if unmapped_reqs:
                    st.warning(f"Could not automatically match columns for: {', '.join(unmapped_reqs)}. Please configure mapping below.")

                with st.expander("🛠️ Column Mapping Configuration", expanded=bool(unmapped_reqs)):
                    col_map_inputs = {}
                    map_cols = st.columns(2)
                    for i, req_col in enumerate(REQUIRED_COLUMNS):
                        with map_cols[i % 2]:
                            default_idx = 0
                            opts = ["-- Select Column --"] + raw_parsed.columns.tolist()
                            if suggested_map.get(req_col) and suggested_map[req_col] in raw_parsed.columns:
                                default_idx = opts.index(suggested_map[req_col])
                            selected_col = st.selectbox(
                                f"Map to '{req_col}'",
                                options=opts,
                                index=default_idx,
                                key=f"map_{req_col}",
                            )
                            if selected_col != "-- Select Column --":
                                col_map_inputs[req_col] = selected_col

                # Run validation
                is_valid, errors, warnings, clean_df, stats = validate_and_clean_data(raw_parsed, col_map_inputs)

                if warnings:
                    for w in warnings:
                        st.info(f"ℹ️ {w}")

                if not is_valid:
                    st.error("Validation failed with the following issues:")
                    for err in errors:
                        st.markdown(f"- ❌ {err}")
                else:
                    st.success("✅ Dataset schema, constraints, and data types validated successfully!")
                    st.markdown(
                        f"""
                        - **Total Valid Records:** `{stats['total_records']:,}`
                        - **Products Detected:** `{stats['total_products']}` SKUs
                        - **Date Range:** `{stats['min_date']}` to `{stats['max_date']}` ({stats['total_days_span']} days)
                        - **Categories:** `{', '.join(stats['categories'])}`
                        """
                    )

                    if st.button("🚀 Apply and Use This Dataset in SmartStock AI"):
                        st.session_state.raw_data = clean_df
                        st.success("Dataset applied successfully! Recomputing inventory metrics...")
                        st.rerun()

            else:
                st.error("Could not parse file. Ensure it is a valid CSV or Excel workbook.")

        st.markdown('</div>', unsafe_allow_html=True)

    with tab_template:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Download Official Sample Excel Workbook</div>', unsafe_allow_html=True)
        st.write(
            "Download the realistic 24-SKU, 365-day synthetic FMCG dataset formatted with a dedicated Data Dictionary tab. "
            "You can use this template to prepare external supply-chain datasets."
        )

        template_path = os.path.join(os.path.dirname(__file__), "data", "sample_template.xlsx")
        if os.path.exists(template_path):
            with open(template_path, "rb") as f:
                template_bytes = f.read()
            st.download_button(
                label="📥 Download FMCG_Inventory_Template.xlsx",
                data=template_bytes,
                file_name="SmartStock_FMCG_Sample_Template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.warning("Template file is being generated. Please reset data or click below.")
            if st.button("Generate Template Now"):
                save_synthetic_data()
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with tab_schema:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Expected Dataset Schema Specification</div>', unsafe_allow_html=True)
        schema_df = pd.DataFrame([
            {"Column": "Date", "Required Type": "YYYY-MM-DD", "Nullable": "No", "Description": "Calendar transaction date"},
            {"Column": "Product_ID", "Required Type": "String (e.g. SKU-001)", "Nullable": "No", "Description": "Unique SKU identifier"},
            {"Column": "Product_Name", "Required Type": "String", "Nullable": "No", "Description": "Commercial product brand name"},
            {"Column": "Category", "Required Type": "String", "Nullable": "No", "Description": "Product division or category"},
            {"Column": "Units_Sold", "Required Type": "Integer (>= 0)", "Nullable": "No", "Description": "Daily units sold to retail"},
            {"Column": "On_Hand_Stock", "Required Type": "Integer (>= 0)", "Nullable": "No", "Description": "Physical depot inventory"},
            {"Column": "On_Order_Stock", "Required Type": "Integer (>= 0)", "Nullable": "No", "Description": "In-transit PO quantities"},
            {"Column": "Backorders", "Required Type": "Integer (>= 0)", "Nullable": "No", "Description": "Unfulfilled retail orders"},
            {"Column": "Lead_Time_Days", "Required Type": "Integer (> 0)", "Nullable": "No", "Description": "Replenishment lead time"},
            {"Column": "Unit_Cost", "Required Type": "Float (> 0)", "Nullable": "No", "Description": "Acquisition cost in INR"},
        ])
        st.table(schema_df)
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 6: REPORTS & EXPORTS
# ==============================================================================
elif page == "📑 Reports & Exports":
    render_header(
        "Executive Supply Chain Reports",
        "Generate & Export Formatted Multi-Sheet Excel Workbooks with Openpyxl",
        "Audit & Management Reporting",
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Multi-Sheet Excel Workbook Generator</div>', unsafe_allow_html=True)
    st.write(
        "Generate an executive-grade Excel report containing 7 distinct sheets: "
        "**Inventory Summary**, **Demand Forecasts**, **Forecast Accuracy**, "
        "**Reorder Recommendations**, **Stockout Risk**, **Excess Inventory**, and **Methodology & Assumptions**. "
        "All sheets are styled with frozen headers, professional currency/integer number formatting, and auto-fit column widths."
    )

    with st.spinner("Generating styled multi-sheet Excel report..."):
        excel_buffer = generate_executive_excel_report(
            inventory_df=inv_df,
            raw_df=st.session_state.raw_data,
            service_level_str=st.session_state.service_level,
            holding_rate=st.session_state.holding_rate,
        )
        report_bytes = excel_buffer.getvalue()

    st.download_button(
        label="📥 Download Executive Inventory Report (.xlsx)",
        data=report_bytes,
        file_name=f"SmartStock_Executive_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Live Previews of the 7 sheets
    st.markdown("### 🔍 Report Sheet Previews")
    preview_tab1, preview_tab2, preview_tab3, preview_tab4, preview_tab5, preview_tab6, preview_tab7 = st.tabs([
        "1. Inventory Summary",
        "2. Demand Forecasts",
        "3. Forecast Accuracy",
        "4. Reorder Recommendations",
        "5. Stockout Risk",
        "6. Excess Inventory",
        "7. Methodology",
    ])

    with preview_tab1:
        st.dataframe(inv_df.head(10), use_container_width=True)

    with preview_tab2:
        if not inv_df.empty:
            sample_fc_sku = inv_df["Product_ID"].iloc[0]
            sample_fc_name = inv_df["Product_Name"].iloc[0]
            st.write(f"14-Day Forward Projections for **{sample_fc_name}** ({sample_fc_sku}):")
            s_ser = prepare_product_series(st.session_state.raw_data, sample_fc_sku)
            _, _, fc_prev_df = run_future_forecast(s_ser, horizon=14)
            st.dataframe(fc_prev_df, use_container_width=True)
        else:
            st.info("No active inventory records available.")

    with preview_tab3:
        st.write("Out-of-sample chronological backtest evaluation metrics (MAE, WAPE, Bias)")
        acc_samples = []
        if not inv_df.empty:
            for sku_id in inv_df["Product_ID"].iloc[:8]:
                ser = prepare_product_series(st.session_state.raw_data, sku_id)
                eval_res = evaluate_models_chronological(ser, test_horizon=14)
                if eval_res.get("has_data"):
                    acc_samples.append({
                        "Product_ID": sku_id,
                        "Model": eval_res["exp_model_name"],
                        "MAE": eval_res["exp_metrics"]["MAE"],
                        "WAPE_%": eval_res["exp_metrics"]["WAPE"],
                        "Bias_%": eval_res["exp_metrics"]["Bias"],
                    })
        if acc_samples:
            st.dataframe(pd.DataFrame(acc_samples), use_container_width=True)
        else:
            st.info("Insufficient historical records for backtest evaluation.")

    with preview_tab4:
        reorder_prev = inv_df[inv_df["Recommended_Order_Qty"] > 0][[
            "Product_ID", "Product_Name", "Category", "On_Hand_Stock", "Inventory_Position",
            "Reorder_Point", "Recommended_Order_Qty", "Unit_Cost"
        ]]
        st.dataframe(reorder_prev, use_container_width=True)

    with preview_tab5:
        stockout_prev = inv_df[inv_df["Inventory_Status"] == "Potential Stockout Risk"][[
            "Product_ID", "Product_Name", "Category", "On_Hand_Stock", "Lead_Time_Demand", "Days_of_Supply", "Recommendation_Reason"
        ]]
        st.dataframe(stockout_prev, use_container_width=True)

    with preview_tab6:
        excess_prev = inv_df[inv_df["Inventory_Status"] == "Excess Inventory"][[
            "Product_ID", "Product_Name", "Category", "On_Hand_Stock", "Reorder_Point", "Days_of_Supply", "Recommendation_Reason"
        ]]
        st.dataframe(excess_prev, use_container_width=True)

    with preview_tab7:
        st.markdown(
            """
            - **Target Service Level:** Standard normal distribution Z-scores applied to lead-time demand variability.
            - **Inventory Position Policy:** (R, s, S) Periodic review replenishing up to base stock target.
            - **Holding Cost Calculation:** 20% annual cost applied to physical on-hand valuation.
            - **Error Metric Standard:** WAPE applied universally to eliminate zero-demand distortion.
            """
        )


# ==============================================================================
# PAGE 7: METHODOLOGY & INTERVIEW GUIDE
# ==============================================================================
elif page == "📚 Methodology & Guide":
    render_header(
        "Methodology & Supply Chain Guide",
        "Mathematical Formulations, FMCG Distribution Mechanics & Interview Prep",
        "ITC Internship Interview Guide",
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📘 Supply Chain Theory & Mathematical Formulations</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### 1. The Core FMCG Distribution Architecture
        In fast-moving consumer goods (such as ITC's Foods, Personal Care, and Stationery businesses), goods flow across multiple echelons:
        $$\\text{Factory / Plant} \\longrightarrow \\text{Central Distribution Center (CDC)} \\longrightarrow \\text{Regional Distribution Center (RDC)} \\longrightarrow \\text{Wholesaler / Distributor} \\longrightarrow \\text{Retail Kirana Stores}$$

        Each echelon faces distinct lead times and demand variability. Maintaining high on-shelf availability while minimizing inventory carrying costs requires accurate demand forecasting and stochastic inventory modeling.
        
        ---

        ### 2. Demand vs. Sales vs. Replenishment (Crucial Distinction)
        - **Demand:** The true volume of customer requests arriving in a given period.
        - **Sales:** The volume actually fulfilled. If stock runs out ($On\\text{-}Hand = 0$), sales drop to 0 even if demand is high. This is called **stockout censoring**.
        - **Replenishment Orders:** Quantities ordered from upstream plants to replenish depots, which arrive after a **supplier lead time ($L$)**.

        ---

        ### 3. Forecasting Models Implemented
        #### A. Moving Average Baseline
        $$\\hat{y}_{t+h} = \\frac{1}{N} \\sum_{i=0}^{N-1} y_{t-i}$$
        Smooths out random day-to-day fluctuations but lags behind seasonal changes and shifts.

        #### B. Holt-Winters Exponential Smoothing (Statsmodels)
        Uses triple exponential smoothing for level ($\\ell$), trend ($b$), and additive 7-day seasonality ($s$):
        $$\\ell_t = \\alpha (y_t - s_{t-m}) + (1 - \\alpha)(\\ell_{t-1} + b_{t-1})$$
        $$b_t = \\beta (\\ell_t - \\ell_{t-1}) + (1 - \\beta) b_{t-1}$$
        $$s_t = \\gamma (y_t - \\ell_{t-1} - b_{t-1}) + (1 - \\gamma) s_{t-m}$$
        $$\\hat{y}_{t+h} = \\ell_t + h b_t + s_{t+h-m}$$
        This is critical for FMCG products where Friday and Saturday grocery purchases create regular weekly demand spikes.

        ---

        ### 4. Why WAPE Instead of MAPE?
        Standard Mean Absolute Percentage Error (MAPE) is calculated as:
        $$\\text{MAPE} = \\frac{1}{n} \\sum \\left| \\frac{y_t - \\hat{y}_t}{y_t} \\right| \\times 100\\%$$
        In FMCG depots, slow-moving SKUs frequently have zero sales on certain days ($y_t = 0$), causing MAPE to divide by zero ($0/0$) or explode to infinity.
        
        **Weighted Absolute Percentage Error (WAPE)** resolves this:
        $$\\text{WAPE} = \\frac{\\sum |y_t - \\hat{y}_t|}{\\sum y_t} \\times 100\\%$$
        WAPE weights errors by volume, ensuring that larger products and higher-volume days have a proportional impact.

        ---

        ### 5. Stochastic Safety Stock Mathematics
        When both daily demand ($d$) and supplier lead time ($L$) are subject to stochastic variation:
        $$\\sigma_{LTD}^2 = L \\cdot \\sigma_d^2 + \\bar{d}^2 \\cdot \\sigma_L^2$$
        $$\\mathbf{SS = Z \\times \\sqrt{L \\cdot \\sigma_d^2 + \\bar{d}^2 \\cdot \\sigma_L^2}}$$
        
        Where:
        - $Z$: Service factor corresponding to the target non-stockout probability (e.g., $Z=1.65$ for 95%, $Z=2.33$ for 99%).
        - $L \\cdot \\sigma_d^2$: Uncertainty stemming from fluctuating daily consumer sales.
        - $\\bar{d}^2 \\cdot \\sigma_L^2$: Uncertainty stemming from delayed factory truck arrivals.

        ---

        ### 6. Reorder Point ($ROP$) & Inventory Position ($IP$)
        $$\\mathbf{ROP = (\\bar{d} \\times L) + SS}$$
        $$\\mathbf{IP = On\\text{-}Hand + On\\text{-}Order - Backorders}$$

        **Key Principle:** A purchase order is triggered when $IP \\le ROP$, **not** when $On\\text{-}Hand \\le ROP$. If 500 units are already in transit from the factory, double-ordering would lead to severe depot congestion and working-capital waste.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Interview Q&A Section
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💼 Top 5 ITC Internship Interview Questions & Model Answers</div>', unsafe_allow_html=True)

    with st.expander("Q1: If an RDC increases target customer service level from 95% to 99%, how does inventory cost change?"):
        st.markdown(
            """
            **Model Answer:**
            Increasing service level from 95% ($Z = 1.645$) to 99% ($Z = 2.326$) causes a **41.4% increase in safety stock**:
            $$\\frac{2.326 - 1.645}{1.645} \\approx +41.4\\%$$
            Because the normal distribution tails are asymptotic, the inventory holding cost curve rises exponentially past 95%. In FMCG logistics, companies often segment SKUs using ABC/XYZ analysis, assigning 98-99% service levels to high-margin Class A staples (like Aashirvaad Atta) and 90-95% to slower-moving Class C items.
            """
        )

    with st.expander("Q2: Why must you use Inventory Position rather than Physical On-Hand stock to trigger reorders?"):
        st.markdown(
            """
            **Model Answer:**
            If we only evaluated physical on-hand stock, any ongoing replenishment order in transit would be ignored. During the multi-day factory lead time, the system would repeatedly trigger purchase orders every day, resulting in severe overstocking when all shipments arrive simultaneously. $IP = On\\text{-}Hand + On\\text{-}Order - Backorders$ guarantees true pipeline visibility.
            """
        )

    with st.expander("Q3: How do you address the Bullwhip Effect in an FMCG distribution network?"):
        st.markdown(
            """
            **Model Answer:**
            The Bullwhip Effect occurs when small demand variations at the retail kirana level become amplified as orders move upstream to distributors, RDCs, and manufacturing plants. We mitigate it through:
            1. **Point-of-Sale (POS) data sharing:** Forecasting based on actual consumer sell-through rather than distributor order spikes.
            2. **Smaller, more frequent deliveries:** Reducing batching and order review periods ($R$).
            3. **Everyday low pricing (EDLP):** Smoothing artificial demand surges caused by trade promotions.
            4. **Vendor-Managed Inventory (VMI):** Upstream suppliers managing inventory levels at regional depots.
            """
        )

    with st.expander("Q4: How do you handle stockout-censored demand data when training forecasting models?"):
        st.markdown(
            """
            **Model Answer:**
            Historical sales data records what was sold, not what was demanded. If a depot ran out of stock for 3 days, recorded sales are zero, which artificially pulls down future forecasts.
            In production supply chain systems:
            - Days where $On\\text{-}Hand = 0$ are flagged as stockout-censored.
            - Demand on those days is imputed using either the average sales of in-stock days, unfulfilled backorders, or Tobit regression/censored likelihood models.
            """
        )

    with st.expander("Q5: What are the primary drivers of inventory holding cost in an FMCG warehouse?"):
        st.markdown(
            """
            **Model Answer:**
            The typical annual inventory holding rate in FMCG is 18% to 25% of product value, comprising:
            1. **Cost of Capital (WACC):** 10-14% (opportunity cost of working capital tied up in inventory).
            2. **Storage & Warehousing:** 3-5% (warehouse rent, pallet racking, material handling equipment, electricity for cold storage).
            3. **Obsolescence & Expiry Risk:** 2-4% (FMCG shelf-life constraints, packaging redesigns, damage during transit).
            4. **Insurance & Taxes:** 1-2% (coverage against warehouse fire, transit theft, pilferage).
            """
        )
    st.markdown('</div>', unsafe_allow_html=True)

