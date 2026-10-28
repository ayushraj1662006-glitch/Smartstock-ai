"""
Data generator module for SmartStock AI.
Generates realistic, internally consistent synthetic FMCG dataset spanning 365 days
across 24 diverse products simulating an enterprise consumer goods supply chain.

DISCLAIMER: This dataset is entirely synthetic and simulated for academic/demonstration
purposes. It does not represent actual commercial sales or proprietary data of ITC Limited.
"""

from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd

from src.config import FMCG_PRODUCT_CATALOG


def generate_fmcg_dataset(
    start_date: str = "2025-11-01",
    days: int = 365,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a 365-day realistic time-series dataset of FMCG sales and inventory.

    Simulates:
    - Daily demand with day-of-week patterns (weekend retail replenishment surges)
    - Macro seasonality (summer, festive Q4, back-to-school, monsoon, winter)
    - Realistic inventory replenishment dynamics with lead-time delivery lags
    - A controlled snapshot on the final day containing diverse stock statuses
      (Healthy, Reorder Required, Potential Stockout Risk, Excess Inventory).
    """
    rng = np.random.default_rng(random_seed)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    date_range = [start_dt + timedelta(days=i) for i in range(days)]

    records = []

    for product in FMCG_PRODUCT_CATALOG:
        sku_id = product["id"]
        sku_name = product["name"]
        cat = product["category"]
        base_demand = product["base_daily_demand"]
        std_pct = product["demand_std_pct"]
        seasonality_type = product["seasonality"]
        lead_time = product["lead_time"]
        unit_cost = product["unit_cost"]
        pack_size = product["min_pack_size"]

        # Calculate rough Reorder Point for simulation replenishment logic
        avg_lead_time_demand = base_demand * lead_time
        safety_stock_est = 1.65 * (base_demand * std_pct) * np.sqrt(lead_time)
        rop_est = avg_lead_time_demand + safety_stock_est
        eoq_est = max(pack_size * 2, int(np.round(avg_lead_time_demand * 1.5 / pack_size) * pack_size))

        # Initial inventory state
        # Spread inventory around healthy, excess, or reorder levels
        if "Dark Fantasy" in sku_name or "Mango Pulp" in sku_name:
            # Stressed inventory profile
            current_on_hand = int(rop_est * 0.4)
            current_on_order = 0
        elif "Pulse Spiral" in sku_name or "Chana Dal" in sku_name:
            # Slightly overstocked profile
            current_on_hand = int(rop_est * 2.8)
            current_on_order = 0
        elif "Pure Cow Ghee" in sku_name or "Tedhe Medhe" in sku_name:
            # Reorder required profile
            current_on_hand = int(rop_est * 0.85)
            current_on_order = 0
        else:
            # Normal healthy profile
            current_on_hand = int(rop_est * 1.3)
            current_on_order = 0

        pending_orders = []  # List of tuples: (arrival_day_index, qty)
        current_backorders = 0

        for t, cur_date in enumerate(date_range):
            # 1. Receive incoming shipments scheduled for today
            arrived_qty = sum(qty for arr_day, qty in pending_orders if arr_day == t)
            pending_orders = [(arr_day, qty) for arr_day, qty in pending_orders if arr_day > t]
            current_on_hand += arrived_qty

            # Fulfill backorders first
            if current_backorders > 0:
                fulfilled = min(current_on_hand, current_backorders)
                current_on_hand -= fulfilled
                current_backorders -= fulfilled

            # 2. Compute seasonal demand multiplier
            month = cur_date.month
            day_of_week = cur_date.weekday()  # 0=Monday, 6=Sunday

            # Day of week variation (weekend peak for snacks & staples)
            dow_mult = 1.0
            if day_of_week in (4, 5):  # Friday, Saturday
                dow_mult = 1.15
            elif day_of_week == 6:  # Sunday
                dow_mult = 1.08
            elif day_of_week == 0:  # Monday depot restocking
                dow_mult = 1.05

            # Annual Seasonality multiplier
            season_mult = 1.0
            if seasonality_type == "summer":
                # Peak in Apr (4), May (5), Jun (6)
                if month in (4, 5, 6):
                    season_mult = 1.45 + rng.uniform(-0.1, 0.1)
                elif month in (11, 12, 1):
                    season_mult = 0.65
            elif seasonality_type == "festive_q4":
                # Peak in Oct (10), Nov (11), Dec (12)
                if month in (10, 11, 12):
                    season_mult = 1.55 + rng.uniform(-0.1, 0.1)
                elif month in (2, 3):
                    season_mult = 0.85
            elif seasonality_type == "back_to_school":
                # Peak in Jun (6), Jul (7)
                if month in (6, 7):
                    season_mult = 1.70 + rng.uniform(-0.1, 0.1)
                elif month in (12, 1):
                    season_mult = 0.75
            elif seasonality_type == "monsoon":
                # Peak in Jul (7), Aug (8), Sep (9)
                if month in (7, 8, 9):
                    season_mult = 1.35 + rng.uniform(-0.08, 0.08)
            elif seasonality_type == "winter":
                # Peak in Nov (11), Dec (12), Jan (1)
                if month in (11, 12, 1):
                    season_mult = 1.40 + rng.uniform(-0.1, 0.1)
                elif month in (5, 6):
                    season_mult = 0.70

            # 3. Compute realized demand
            expected_demand = base_demand * dow_mult * season_mult
            demand_std = expected_demand * std_pct
            realized_demand = rng.normal(expected_demand, demand_std)
            realized_demand = max(0, int(round(realized_demand)))

            # Introduce realistic intermittent zeros for slow/niche products
            if base_demand < 45 and rng.random() < 0.10:
                realized_demand = 0

            # 4. Fulfill demand from on-hand stock
            if current_on_hand >= realized_demand:
                units_sold = realized_demand
                current_on_hand -= units_sold
            else:
                # Stockout occurred
                units_sold = current_on_hand
                unmet_demand = realized_demand - current_on_hand
                current_backorders += unmet_demand
                current_on_hand = 0

            # 5. Inventory position and replenishment decision
            current_on_order = sum(qty for _, qty in pending_orders)
            inventory_position = current_on_hand + current_on_order - current_backorders

            # Periodic check or ROP trigger (every 3 days or when IP <= ROP)
            # Avoid placing orders on the very last 5 days for certain SKUs to keep realistic stress states
            allow_order = True
            if t >= days - 7:
                if sku_id in ["SKU-FS-004", "SKU-BS-005", "SKU-PC-014"]:
                    # Keep these as critical stockout risk for the dashboard
                    allow_order = False

            # Simulate scheduled bulk factory replenishment arriving near the end of the year
            # for 2 SKUs to represent realistic overstocked/excess inventory profiles
            if t == days - 2 and sku_id in ["SKU-ST-022", "SKU-FS-003"]:
                current_on_hand += int(rop_est * 2.8)

            if allow_order and inventory_position <= rop_est:
                order_qty = int(np.ceil((rop_est + eoq_est - inventory_position) / pack_size) * pack_size)
                if order_qty > 0:
                    lead_time_noise = max(1, int(round(lead_time + rng.normal(0, 0.8))))
                    delivery_day = t + lead_time_noise
                    pending_orders.append((delivery_day, order_qty))
                    current_on_order += order_qty

            # Record daily state
            records.append({
                "Date": cur_date.strftime("%Y-%m-%d"),
                "Product_ID": sku_id,
                "Product_Name": sku_name,
                "Category": cat,
                "Units_Sold": units_sold,
                "On_Hand_Stock": current_on_hand,
                "On_Order_Stock": current_on_order,
                "Backorders": current_backorders,
                "Lead_Time_Days": lead_time,
                "Unit_Cost": unit_cost,
            })

    df = pd.DataFrame(records)
    # Ensure correct data types
    df["Date"] = pd.to_datetime(df["Date"])
    df["Units_Sold"] = df["Units_Sold"].astype(int)
    df["On_Hand_Stock"] = df["On_Hand_Stock"].astype(int)
    df["On_Order_Stock"] = df["On_Order_Stock"].astype(int)
    df["Backorders"] = df["Backorders"].astype(int)
    df["Lead_Time_Days"] = df["Lead_Time_Days"].astype(int)
    df["Unit_Cost"] = df["Unit_Cost"].astype(float)

    return df


def save_synthetic_data(
    csv_path: str = "data/fmcg_inventory_data.csv",
    excel_path: Optional[str] = "data/sample_template.xlsx",
) -> pd.DataFrame:
    """Generates and writes the synthetic FMCG dataset to CSV and Excel."""
    df = generate_fmcg_dataset()
    df.to_csv(csv_path, index=False)

    if excel_path:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name="FMCG_Inventory_Data", index=False)
            
            # Data dictionary sheet
            data_dict = pd.DataFrame([
                {"Column Name": "Date", "Data Type": "YYYY-MM-DD", "Description": "Calendar observation date"},
                {"Column Name": "Product_ID", "Data Type": "String (SKU-XX-000)", "Description": "Unique stock keeping unit identifier"},
                {"Column Name": "Product_Name", "Data Type": "String", "Description": "Commercial product brand and description"},
                {"Column Name": "Category", "Data Type": "String", "Description": "FMCG business vertical / product category"},
                {"Column Name": "Units_Sold", "Data Type": "Integer (>=0)", "Description": "Daily sales volume fulfilled from stock"},
                {"Column Name": "On_Hand_Stock", "Data Type": "Integer (>=0)", "Description": "Physical usable stock inside warehouse/depot at end of day"},
                {"Column Name": "On_Order_Stock", "Data Type": "Integer (>=0)", "Description": "Purchase orders placed with plant/vendor in transit"},
                {"Column Name": "Backorders", "Data Type": "Integer (>=0)", "Description": "Unfulfilled customer orders pending stock arrival"},
                {"Column Name": "Lead_Time_Days", "Data Type": "Integer (>0)", "Description": "Vendor replenishment lead time in days"},
                {"Column Name": "Unit_Cost", "Data Type": "Float (>0)", "Description": "Unit acquisition / manufacturing cost in INR"},
            ])
            data_dict.to_excel(writer, sheet_name="Data_Dictionary", index=False)

    return df


if __name__ == "__main__":
    df_gen = save_synthetic_data()
    print(f"Generated {len(df_gen)} records across {df_gen['Product_ID'].nunique()} products.")
    print("Sample:\n", df_gen.head())

