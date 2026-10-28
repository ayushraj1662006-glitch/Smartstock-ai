"""
Unit tests for SmartStock AI core calculations and workflows.
Tests forecasting formulas, inventory mathematics, safety stock models,
error metrics, and data validation rules.
"""

import numpy as np
import pandas as pd
import pytest

from src.forecasting import (
    calculate_metrics,
    forecast_moving_average,
    forecast_exponential_smoothing,
    evaluate_models_chronological,
)
from src.inventory import (
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_inventory_position,
    calculate_replenishment_quantity,
    classify_inventory_status,
)
from src.data_validation import validate_and_clean_data, suggest_column_mapping


# ==============================================================================
# Forecasting & Metrics Tests
# ==============================================================================
def test_calculate_metrics_standard():
    actuals = np.array([100, 110, 90, 120])
    preds = np.array([105, 100, 95, 110])
    # abs errors: 5, 10, 5, 10 -> MAE = 7.5
    # sum actual = 420, sum abs err = 30 -> WAPE = (30/420)*100 = 7.14%
    # sum pred = 410 -> Bias = ((410 - 420)/420)*100 = -2.38%
    res = calculate_metrics(actuals, preds)
    assert res["MAE"] == 7.5
    assert abs(res["WAPE"] - 7.14) < 0.05
    assert abs(res["Bias"] - (-2.38)) < 0.05


def test_calculate_metrics_zero_actuals():
    # Test zero-division resilience (crucial for intermittent FMCG demand)
    actuals = np.array([0, 0, 0])
    preds = np.array([5, 0, 0])
    res = calculate_metrics(actuals, preds)
    assert res["MAE"] == round(5 / 3, 2)
    assert res["WAPE"] == 100.0  # Safe handling, does not crash with ZeroDivisionError


def test_forecast_moving_average_values():
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    series = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], index=dates)
    # 5-day window: mean of [60, 70, 80, 90, 100] = 80.0
    fc = forecast_moving_average(series, window=5, horizon=3)
    assert len(fc) == 3
    assert (fc == 80.0).all()
    assert fc.index[0] == pd.Timestamp("2026-01-11")


def test_forecast_exponential_smoothing_runs():
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    # Synthetic weekly pattern
    vals = [50 + 10 * np.sin(i * 2 * np.pi / 7) for i in range(60)]
    series = pd.Series(vals, index=dates)
    fc, model_name = forecast_exponential_smoothing(series, horizon=14)
    assert len(fc) == 14
    assert (fc >= 0).all()
    assert isinstance(model_name, str)


def test_chronological_evaluation_no_leakage():
    dates = pd.date_range("2026-01-01", periods=45, freq="D")
    vals = np.linspace(50, 100, 45)
    series = pd.Series(vals, index=dates)
    res = evaluate_models_chronological(series, test_horizon=14, ma_window=7)
    assert res["has_data"] is True
    assert res["train_size"] == 31
    assert res["test_size"] == 14
    # Train index should strictly precede test index
    assert res["train_series"].index.max() < res["test_series"].index.min()
    assert "MAE" in res["exp_metrics"]
    assert "WAPE" in res["exp_metrics"]


# ==============================================================================
# Inventory Mathematics Tests
# ==============================================================================
def test_safety_stock_constant_lead_time():
    # d_bar = 100, sigma_d = 20, L = 4, sigma_L = 0, Z = 1.645 (95%)
    # SS = 1.645 * 20 * sqrt(4) = 1.645 * 20 * 2 = 65.8 -> ceil = 66
    ss, z, breakdown = calculate_safety_stock(
        avg_daily_demand=100,
        std_daily_demand=20,
        lead_time_days=4,
        std_lead_time_days=0.0,
        service_level_str="95% (Z = 1.65)",
        include_lead_time_variance=False,
    )
    assert z == 1.645
    assert ss == 66.0
    assert breakdown["lead_time_variance_contrib"] == 0.0


def test_safety_stock_variable_lead_time():
    # With lead time variability, SS must be strictly greater than constant lead time
    ss_const, _, _ = calculate_safety_stock(
        avg_daily_demand=100,
        std_daily_demand=20,
        lead_time_days=4,
        std_lead_time_days=0.0,
        include_lead_time_variance=False,
    )
    ss_var, _, breakdown = calculate_safety_stock(
        avg_daily_demand=100,
        std_daily_demand=20,
        lead_time_days=4,
        std_lead_time_days=1.5,
        include_lead_time_variance=True,
    )
    assert ss_var > ss_const
    assert breakdown["lead_time_variance_contrib"] > 0


def test_safety_stock_monotonic_service_level():
    # Higher service level = strictly higher safety stock
    ss_90, _, _ = calculate_safety_stock(100, 20, 5, service_level_str="90% (Z = 1.28)")
    ss_95, _, _ = calculate_safety_stock(100, 20, 5, service_level_str="95% (Z = 1.65)")
    ss_98, _, _ = calculate_safety_stock(100, 20, 5, service_level_str="98% (Z = 2.05)")
    ss_99, _, _ = calculate_safety_stock(100, 20, 5, service_level_str="99% (Z = 2.33)")
    assert ss_90 < ss_95 < ss_98 < ss_99


def test_reorder_point():
    # LTD = 200, SS = 50 -> ROP = 250
    rop = calculate_reorder_point(lead_time_demand=200, safety_stock=50)
    assert rop == 250.0


def test_inventory_position():
    # IP = OH + OO - BO
    # 500 + 200 - 50 = 650
    ip = calculate_inventory_position(on_hand=500, on_order=200, backorders=50)
    assert ip == 650.0


def test_replenishment_quantity_logic():
    # Scenario 1: IP <= ROP -> Should recommend replenishment
    # IP = 100, ROP = 150, avg_demand = 10, review_period = 7
    # Target stock = 150 + (10 * 7) = 220
    # Deficit = 220 - 100 = 120
    qty1 = calculate_replenishment_quantity(
        inventory_position=100,
        reorder_point=150,
        avg_daily_demand=10,
        review_period_days=7,
    )
    assert qty1 == 120.0

    # Scenario 2: IP > ROP -> Sufficient stock, Order Qty should be 0
    qty2 = calculate_replenishment_quantity(
        inventory_position=180,
        reorder_point=150,
        avg_daily_demand=10,
        review_period_days=7,
    )
    assert qty2 == 0.0

    # Scenario 3: Large on-order pipeline prevents double ordering!
    # Even if on-hand is low (say 20), if on-order is 200, IP = 220 > ROP (150) -> Qty = 0
    ip_with_pipeline = calculate_inventory_position(on_hand=20, on_order=200, backorders=0)
    qty3 = calculate_replenishment_quantity(
        inventory_position=ip_with_pipeline,
        reorder_point=150,
        avg_daily_demand=10,
    )
    assert qty3 == 0.0


def test_classify_inventory_status():
    # 1. Critical stockout risk
    status, _ = classify_inventory_status(
        on_hand=5, inventory_position=5, reorder_point=100, lead_time_demand=50, days_of_supply=1.0, backorders=0
    )
    assert status == "Potential Stockout Risk"

    # 2. Reorder Required
    status, _ = classify_inventory_status(
        on_hand=80, inventory_position=80, reorder_point=100, lead_time_demand=50, days_of_supply=12.0, backorders=0
    )
    assert status == "Reorder Required"

    # 3. Excess Inventory
    status, _ = classify_inventory_status(
        on_hand=600, inventory_position=600, reorder_point=100, lead_time_demand=50, days_of_supply=65.0, backorders=0
    )
    assert status == "Excess Inventory"

    # 4. Healthy Stock
    status, _ = classify_inventory_status(
        on_hand=150, inventory_position=150, reorder_point=100, lead_time_demand=50, days_of_supply=20.0, backorders=0
    )
    assert status == "Healthy Stock"


# ==============================================================================
# Data Validation Tests
# ==============================================================================
def test_validate_and_clean_data_valid():
    df = pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-02"],
        "Product_ID": ["SKU-001", "SKU-001"],
        "Product_Name": ["Test Product", "Test Product"],
        "Category": ["Snacks", "Snacks"],
        "Units_Sold": [10, 15],
        "On_Hand_Stock": [50, 40],
        "On_Order_Stock": [0, 20],
        "Backorders": [0, 0],
        "Lead_Time_Days": [3, 3],
        "Unit_Cost": [45.0, 45.0],
    })
    is_valid, errors, warnings, clean_df, stats = validate_and_clean_data(df)
    assert is_valid is True
    assert len(errors) == 0
    assert clean_df is not None
    assert stats["total_records"] == 2


def test_validate_and_clean_data_missing_column():
    df = pd.DataFrame({"Date": ["2026-01-01"], "Product_ID": ["SKU-001"]})
    is_valid, errors, _, _, _ = validate_and_clean_data(df)
    assert is_valid is False
    assert any("Missing required columns" in err for err in errors)


def test_validate_and_clean_data_negative_values():
    df = pd.DataFrame({
        "Date": ["2026-01-01"],
        "Product_ID": ["SKU-001"],
        "Product_Name": ["Test"],
        "Category": ["Snacks"],
        "Units_Sold": [-5],  # Invalid negative
        "On_Hand_Stock": [50],
        "On_Order_Stock": [0],
        "Backorders": [0],
        "Lead_Time_Days": [3],
        "Unit_Cost": [45.0],
    })
    is_valid, errors, _, _, _ = validate_and_clean_data(df)
    assert is_valid is False
    assert any("negative values" in err for err in errors)


def test_column_mapping_synonyms():
    uploaded_cols = ["date", "sku", "item_name", "category", "sales", "stock", "in_transit", "backorder", "lead_time", "cost"]
    mapping = suggest_column_mapping(uploaded_cols)
    assert mapping["Date"] == "date"
    assert mapping["Product_ID"] == "sku"
    assert mapping["Units_Sold"] == "sales"
    assert mapping["On_Hand_Stock"] == "stock"
    assert mapping["Unit_Cost"] == "cost"


def test_portfolio_inventory_metrics():
    from src.data_generator import generate_fmcg_dataset
    from src.inventory import compute_portfolio_inventory_metrics

    df = generate_fmcg_dataset(days=30)
    inv = compute_portfolio_inventory_metrics(df)
    assert len(inv) == 24
    assert "Inventory_Status" in inv.columns
    assert "Reorder_Point" in inv.columns
    assert "Recommended_Order_Qty" in inv.columns
    assert (inv["Recommended_Order_Qty"] >= 0).all()


def test_excel_report_generation():
    from src.data_generator import generate_fmcg_dataset
    from src.inventory import compute_portfolio_inventory_metrics
    from src.reporting import generate_executive_excel_report
    import openpyxl

    df = generate_fmcg_dataset(days=30)
    inv = compute_portfolio_inventory_metrics(df)
    buf = generate_executive_excel_report(inv, df)
    assert buf is not None
    wb = openpyxl.load_workbook(buf)
    sheet_names = wb.sheetnames
    expected_sheets = [
        "Inventory_Summary",
        "Demand_Forecasts",
        "Forecast_Accuracy",
        "Reorder_Recommendations",
        "Stockout_Risk",
        "Excess_Inventory",
        "Methodology_Assumptions",
    ]
    for sheet in expected_sheets:
        assert sheet in sheet_names


def test_forecast_constant_demand():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    series = pd.Series([50.0] * 30, index=dates)
    fc, model_name = forecast_exponential_smoothing(series, horizon=7)
    assert len(fc) == 7
    assert (fc == 50.0).all()
    assert "Constant" in model_name


def test_forecast_zero_demand():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    series = pd.Series([0.0] * 30, index=dates)
    fc, model_name = forecast_exponential_smoothing(series, horizon=7)
    assert len(fc) == 7
    assert (fc == 0.0).all()


def test_forecast_insufficient_history():
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    series = pd.Series([10.0, 20.0, 30.0, 40.0], index=dates)
    fc, model_name = forecast_exponential_smoothing(series, horizon=5)
    assert len(fc) == 5
    assert (fc == 25.0).all()
    assert "Insufficient History" in model_name


def test_safety_stock_fuzzy_lookup():
    ss1, z1, _ = calculate_safety_stock(50, 10, 4, service_level_str="95%")
    ss2, z2, _ = calculate_safety_stock(50, 10, 4, service_level_str="95% (Z = 1.65)")
    assert z1 == z2 == 1.645
    assert ss1 == ss2


def test_inventory_latest_snapshot_multi_date():
    from src.inventory import compute_portfolio_inventory_metrics
    df = pd.DataFrame([
        {"Date": "2026-01-01", "Product_ID": "SKU-A", "Product_Name": "Item A", "Category": "Cat", "Units_Sold": 5, "On_Hand_Stock": 10, "On_Order_Stock": 0, "Backorders": 0, "Lead_Time_Days": 3, "Unit_Cost": 10.0},
        {"Date": "2026-01-02", "Product_ID": "SKU-A", "Product_Name": "Item A", "Category": "Cat", "Units_Sold": 6, "On_Hand_Stock": 20, "On_Order_Stock": 0, "Backorders": 0, "Lead_Time_Days": 3, "Unit_Cost": 10.0},
        {"Date": "2026-01-01", "Product_ID": "SKU-B", "Product_Name": "Item B", "Category": "Cat", "Units_Sold": 7, "On_Hand_Stock": 30, "On_Order_Stock": 5, "Backorders": 0, "Lead_Time_Days": 4, "Unit_Cost": 20.0},
    ])
    inv = compute_portfolio_inventory_metrics(df)
    assert len(inv) == 2
    assert set(inv["Product_ID"]) == {"SKU-A", "SKU-B"}
    # SKU-A latest is 20, SKU-B is 30
    assert inv.loc[inv["Product_ID"] == "SKU-A", "On_Hand_Stock"].values[0] == 20
    assert inv.loc[inv["Product_ID"] == "SKU-B", "On_Hand_Stock"].values[0] == 30


def test_inventory_excess_status_populated():
    from src.data_generator import generate_fmcg_dataset
    from src.inventory import compute_portfolio_inventory_metrics
    df = generate_fmcg_dataset()
    inv = compute_portfolio_inventory_metrics(df)
    statuses = set(inv["Inventory_Status"].unique())
    assert "Excess Inventory" in statuses
    assert "Potential Stockout Risk" in statuses
    assert "Reorder Required" in statuses
    assert "Healthy Stock" in statuses


def test_parse_uploaded_file_multi_sheet_excel():
    import io
    from src.data_validation import parse_uploaded_file
    
    # Create workbook with cover sheet first, then data
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame([{"Title": "Project Cover Sheet"}]).to_excel(writer, sheet_name="Cover", index=False)
        pd.DataFrame([{
            "Date": "2026-01-01", "Product_ID": "SKU-001", "Product_Name": "Item",
            "Category": "Cat", "Units_Sold": 10, "On_Hand_Stock": 50,
            "On_Order_Stock": 0, "Backorders": 0, "Lead_Time_Days": 3, "Unit_Cost": 20.0
        }]).to_excel(writer, sheet_name="Data", index=False)
    
    parsed = parse_uploaded_file(buf.getvalue(), "multi_sheet.xlsx")
    assert parsed is not None
    assert "Units_Sold" in parsed.columns
    assert len(parsed) == 1

