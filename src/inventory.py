"""
Inventory optimization module for SmartStock AI.
Calculates Lead-Time Demand, Safety Stock, Reorder Point, Inventory Position,
Replenishment Recommendations, Holding Costs, and Status Classification.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import SERVICE_LEVELS, DEFAULT_SERVICE_LEVEL, DEFAULT_ANNUAL_HOLDING_RATE, DEFAULT_REVIEW_PERIOD_DAYS


def calculate_safety_stock(
    avg_daily_demand: float,
    std_daily_demand: float,
    lead_time_days: float,
    std_lead_time_days: float = 0.0,
    service_level_str: str = DEFAULT_SERVICE_LEVEL,
    include_lead_time_variance: bool = True,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Computes Safety Stock using analytical normal distribution formula:
    
    If lead time is constant (std_lead_time = 0 or flag False):
        SS = Z * std_daily_demand * sqrt(lead_time_days)
    
    If lead time is also variable:
        SS = Z * sqrt( lead_time_days * (std_daily_demand ** 2) + (avg_daily_demand ** 2) * (std_lead_time_days ** 2) )
        
    Returns:
        (safety_stock, z_score, breakdown_dict)
    """
    # Flexible service level key lookup
    z_score = SERVICE_LEVELS.get(service_level_str)
    if z_score is None:
        # Match by prefix (e.g. "95%")
        for k, v in SERVICE_LEVELS.items():
            if str(service_level_str).strip() in k:
                z_score = v
                break
        if z_score is None:
            z_score = 1.645

    # Sanitize inputs to non-negative floats
    avg_daily_demand = max(0.0, float(np.nan_to_num(avg_daily_demand, nan=0.0)))
    std_daily_demand = max(0.0, float(np.nan_to_num(std_daily_demand, nan=0.0)))
    lead_time_days = max(0.0, float(np.nan_to_num(lead_time_days, nan=0.0)))
    std_lead_time_days = max(0.0, float(np.nan_to_num(std_lead_time_days, nan=0.0)))

    var_demand_term = lead_time_days * (std_daily_demand ** 2)

    if include_lead_time_variance and std_lead_time_days > 0:
        var_lt_term = (avg_daily_demand ** 2) * (std_lead_time_days ** 2)
        total_variance = max(0.0, var_demand_term + var_lt_term)
        sigma_ltd = np.sqrt(total_variance)
    else:
        var_lt_term = 0.0
        total_variance = max(0.0, var_demand_term)
        sigma_ltd = std_daily_demand * np.sqrt(lead_time_days)

    safety_stock = z_score * sigma_ltd

    breakdown = {
        "z_score": z_score,
        "sigma_ltd": round(float(sigma_ltd), 2),
        "demand_variance_contrib": round(float(var_demand_term), 2),
        "lead_time_variance_contrib": round(float(var_lt_term), 2),
        "include_lt_var": include_lead_time_variance,
    }

    return float(np.ceil(max(0.0, float(np.nan_to_num(safety_stock, nan=0.0))))), z_score, breakdown


def calculate_reorder_point(
    lead_time_demand: float,
    safety_stock: float,
) -> float:
    """
    Reorder Point (ROP) = Expected Lead-Time Demand + Safety Stock
    """
    return float(np.ceil(max(0.0, lead_time_demand + safety_stock)))


def calculate_inventory_position(
    on_hand: float,
    on_order: float,
    backorders: float,
) -> float:
    """
    Inventory Position (IP) = On-Hand Stock + On-Order Stock - Backorders
    """
    return float(on_hand + on_order - backorders)


def calculate_replenishment_quantity(
    inventory_position: float,
    reorder_point: float,
    avg_daily_demand: float,
    review_period_days: int = DEFAULT_REVIEW_PERIOD_DAYS,
    min_order_qty: int = 1,
) -> float:
    """
    Calculates recommended replenishment order quantity using periodic (R, s, S) logic:
    If Inventory Position <= Reorder Point:
        Target Max Stock = Reorder Point + (avg_daily_demand * review_period_days)
        Order Quantity = Target Max Stock - Inventory Position
    Else:
        Order Quantity = 0
        
    Guarantees:
    - Never negative.
    - Prevents double-counting existing pipeline stock (since IP includes on-order).
    """
    if inventory_position <= reorder_point:
        cycle_stock = avg_daily_demand * review_period_days
        target_stock = reorder_point + cycle_stock
        deficit = target_stock - inventory_position
        order_qty = max(float(min_order_qty), deficit)
        return float(np.ceil(order_qty))
    return 0.0


def classify_inventory_status(
    on_hand: float,
    inventory_position: float,
    reorder_point: float,
    lead_time_demand: float,
    days_of_supply: float,
    backorders: float,
) -> Tuple[str, str]:
    """
    Classifies SKU into one of 4 standardized supply chain statuses:
    1. Potential Stockout Risk (Critical)
    2. Reorder Required (Actionable PO)
    3. Excess Inventory (Working Capital Alert)
    4. Healthy Stock (Balanced)
    """
    if backorders > 0 or on_hand <= max(1.0, 0.4 * lead_time_demand) or days_of_supply < 3.0:
        status = "Potential Stockout Risk"
        explanation = (
            f"Physical on-hand ({int(on_hand)} units) is critically below 40% of lead-time demand "
            f"({round(0.4 * lead_time_demand, 1)} units) or days of supply ({round(days_of_supply, 1)}d) is under 3 days. "
            f"High risk of lost sales."
        )
    elif inventory_position <= reorder_point:
        status = "Reorder Required"
        explanation = (
            f"Inventory position ({int(inventory_position)} units) has fallen to or below the Reorder Point "
            f"({int(reorder_point)} units). Order must be placed now to avoid stockout during replenishment lead time."
        )
    elif (reorder_point > 0 and inventory_position > (2.5 * reorder_point)) or days_of_supply > 45.0:
        status = "Excess Inventory"
        explanation = (
            f"Inventory position ({int(inventory_position)} units) exceeds 2.5x the Reorder Point "
            f"({int(reorder_point)} units) with {round(days_of_supply, 1)} days of supply. Capital is tied up."
        )
    else:
        status = "Healthy Stock"
        explanation = (
            f"Stock levels are optimal. Inventory position ({int(inventory_position)} units) is above ROP "
            f"({int(reorder_point)} units) with comfortable buffer ({round(days_of_supply, 1)} days of supply)."
        )

    return status, explanation


def compute_portfolio_inventory_metrics(
    df: pd.DataFrame,
    service_level_str: str = DEFAULT_SERVICE_LEVEL,
    annual_holding_rate: float = DEFAULT_ANNUAL_HOLDING_RATE,
    review_period_days: int = DEFAULT_REVIEW_PERIOD_DAYS,
    include_lead_time_variance: bool = True,
    demand_lookback_days: int = 90,
) -> pd.DataFrame:
    """
    Computes complete, audit-ready inventory optimization table across all SKUs in the active dataset.
    Uses the latest available snapshot date in `df` for current on-hand, on-order, and backorders.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    latest_date = df["Date"].max()
    cutoff_date = latest_date - pd.Timedelta(days=demand_lookback_days)

    # 1. Latest inventory snapshot per product (ensures every SKU is included even if ending dates differ)
    latest_snapshot = df.sort_values("Date").groupby("Product_ID", as_index=False).last()

    # 2. Historical demand lookback slice
    history_slice = df[(df["Date"] >= cutoff_date) & (df["Date"] <= latest_date)]

    demand_stats = history_slice.groupby("Product_ID").agg(
        avg_daily_demand=("Units_Sold", "mean"),
        std_daily_demand=("Units_Sold", "std"),
    ).reset_index()
    demand_stats["std_daily_demand"] = demand_stats["std_daily_demand"].fillna(0.0)

    # Merge snapshot with demand stats
    merged = pd.merge(latest_snapshot, demand_stats, on="Product_ID", how="left")
    merged["avg_daily_demand"] = merged["avg_daily_demand"].fillna(1.0)
    merged["std_daily_demand"] = merged["std_daily_demand"].fillna(0.2 * merged["avg_daily_demand"])

    results = []

    for _, row in merged.iterrows():
        sku_id = row["Product_ID"]
        sku_name = row["Product_Name"]
        category = row["Category"]
        on_hand = float(row["On_Hand_Stock"])
        on_order = float(row["On_Order_Stock"])
        backorders = float(row["Backorders"])
        lead_time = float(row["Lead_Time_Days"])
        unit_cost = float(row["Unit_Cost"])
        avg_demand = float(row["avg_daily_demand"])
        std_demand = float(row["std_daily_demand"])

        # Realistic lead time std dev (approx 20% of lead time)
        std_lead_time = round(lead_time * 0.20, 1)

        # Expected demand during lead time (LTD)
        ltd = avg_demand * lead_time

        # Safety Stock (SS)
        safety_stock, z_score, ss_breakdown = calculate_safety_stock(
            avg_daily_demand=avg_demand,
            std_daily_demand=std_demand,
            lead_time_days=lead_time,
            std_lead_time_days=std_lead_time,
            service_level_str=service_level_str,
            include_lead_time_variance=include_lead_time_variance,
        )

        # Reorder Point (ROP)
        rop = calculate_reorder_point(ltd, safety_stock)

        # Inventory Position (IP)
        ip = calculate_inventory_position(on_hand, on_order, backorders)

        # Days of Supply (DOS)
        dos = (on_hand / avg_demand) if avg_demand > 0 else 999.0

        # Replenishment Order Recommendation
        rec_order_qty = calculate_replenishment_quantity(
            inventory_position=ip,
            reorder_point=rop,
            avg_daily_demand=avg_demand,
            review_period_days=review_period_days,
        )

        # Status & Reason
        status, explanation = classify_inventory_status(
            on_hand=on_hand,
            inventory_position=ip,
            reorder_point=rop,
            lead_time_demand=ltd,
            days_of_supply=dos,
            backorders=backorders,
        )

        # Financial valuations
        inventory_value = on_hand * unit_cost
        pipeline_value = on_order * unit_cost
        annual_holding_cost = inventory_value * annual_holding_rate

        results.append({
            "Product_ID": sku_id,
            "Product_Name": sku_name,
            "Category": category,
            "On_Hand_Stock": int(on_hand),
            "On_Order_Stock": int(on_order),
            "Backorders": int(backorders),
            "Inventory_Position": int(ip),
            "Lead_Time_Days": int(lead_time),
            "Avg_Daily_Demand": round(avg_demand, 1),
            "Std_Daily_Demand": round(std_demand, 1),
            "Lead_Time_Demand": round(ltd, 1),
            "Safety_Stock": int(safety_stock),
            "Reorder_Point": int(rop),
            "Recommended_Order_Qty": int(rec_order_qty),
            "Days_of_Supply": round(dos, 1),
            "Unit_Cost": round(unit_cost, 2),
            "Inventory_Value": round(inventory_value, 2),
            "Pipeline_Value": round(pipeline_value, 2),
            "Annual_Holding_Cost": round(annual_holding_cost, 2),
            "Inventory_Status": status,
            "Recommendation_Reason": explanation,
            "Z_Score": z_score,
            "Lead_Time_Std": std_lead_time,
        })

    return pd.DataFrame(results).sort_values(by=["Category", "Product_Name"]).reset_index(drop=True)
