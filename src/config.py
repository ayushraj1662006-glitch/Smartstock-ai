"""
Configuration module for SmartStock AI.
Defines UI styling, theme palettes, standard calculation constants,
and FMCG product definitions.
"""

from typing import Dict, Any

# ==============================================================================
# Enterprise Color Palette & Styling
# ==============================================================================
COLORS = {
    "primary": "#0F172A",       # Deep Navy
    "primary_light": "#1E293B", # Slate Navy
    "secondary": "#0E7490",     # Deep Teal
    "accent": "#06B6D4",        # Cyan Accent
    "accent_hover": "#0891B2",  # Darker Cyan
    "bg_light": "#F8FAFC",      # Clean Slate Off-White
    "card_bg": "#FFFFFF",       # Card White
    "card_border": "#E2E8F0",   # Soft Border Grey
    "text_dark": "#0F172A",     # Primary Text
    "text_muted": "#64748B",    # Muted Gray Text
    "success": "#059669",       # Emerald Green (Healthy)
    "success_bg": "#ECFDF5",    # Emerald Light Tint
    "warning": "#D97706",       # Amber (Reorder Required)
    "warning_bg": "#FFFBEB",    # Amber Light Tint
    "danger": "#DC2626",        # Crimson Red (Stockout Risk)
    "danger_bg": "#FEF2F2",     # Crimson Light Tint
    "info": "#2563EB",          # Royal Blue (Excess)
    "info_bg": "#EFF6FF",       # Blue Light Tint
}

# Standard Z-Scores for Normal Distribution Service Levels
SERVICE_LEVELS = {
    "90% (Z = 1.28)": 1.282,
    "95% (Z = 1.65)": 1.645,
    "98% (Z = 2.05)": 2.054,
    "99% (Z = 2.33)": 2.326,
}

# Default assumptions
DEFAULT_SERVICE_LEVEL = "95% (Z = 1.65)"
DEFAULT_ANNUAL_HOLDING_RATE = 0.20  # 20% annual holding cost rate
DEFAULT_REVIEW_PERIOD_DAYS = 7     # Periodic review cycle (days)

# ==============================================================================
# Realistic FMCG Portfolio Metadata (Simulated ITC-style SKU Catalog)
# ==============================================================================
FMCG_PRODUCT_CATALOG = [
    # Foods & Staples
    {
        "id": "SKU-FS-001",
        "name": "Aashirvaad Shudh Chakki Atta 5kg",
        "category": "Foods & Staples",
        "base_daily_demand": 140,
        "demand_std_pct": 0.15,
        "seasonality": "steady",
        "lead_time": 4,
        "lead_time_std": 1.0,
        "unit_cost": 210.0,
        "min_pack_size": 10,
    },
    {
        "id": "SKU-FS-002",
        "name": "Aashirvaad Svasti Pure Cow Ghee 1L",
        "category": "Foods & Staples",
        "base_daily_demand": 55,
        "demand_std_pct": 0.20,
        "seasonality": "festive_q4",
        "lead_time": 6,
        "lead_time_std": 1.5,
        "unit_cost": 540.0,
        "min_pack_size": 6,
    },
    {
        "id": "SKU-FS-003",
        "name": "Aashirvaad Organic Chana Dal 1kg",
        "category": "Foods & Staples",
        "base_daily_demand": 75,
        "demand_std_pct": 0.18,
        "seasonality": "steady",
        "lead_time": 5,
        "lead_time_std": 1.0,
        "unit_cost": 115.0,
        "min_pack_size": 12,
    },
    {
        "id": "SKU-FS-004",
        "name": "ITC Master Chef Alphonso Mango Pulp 850g",
        "category": "Foods & Staples",
        "base_daily_demand": 25,
        "demand_std_pct": 0.40,
        "seasonality": "summer",
        "lead_time": 8,
        "lead_time_std": 2.0,
        "unit_cost": 180.0,
        "min_pack_size": 8,
    },
    # Biscuits & Snacks
    {
        "id": "SKU-BS-005",
        "name": "Sunfeast Dark Fantasy Choco Fills 300g",
        "category": "Biscuits & Snacks",
        "base_daily_demand": 180,
        "demand_std_pct": 0.22,
        "seasonality": "festive_q4",
        "lead_time": 3,
        "lead_time_std": 0.8,
        "unit_cost": 105.0,
        "min_pack_size": 24,
    },
    {
        "id": "SKU-BS-006",
        "name": "Sunfeast Mom's Magic Rich Cashew 200g",
        "category": "Biscuits & Snacks",
        "base_daily_demand": 130,
        "demand_std_pct": 0.18,
        "seasonality": "steady",
        "lead_time": 4,
        "lead_time_std": 1.0,
        "unit_cost": 38.0,
        "min_pack_size": 30,
    },
    {
        "id": "SKU-BS-007",
        "name": "Bingo! Mad Angles Masala Madness 80g",
        "category": "Biscuits & Snacks",
        "base_daily_demand": 160,
        "demand_std_pct": 0.25,
        "seasonality": "steady",
        "lead_time": 3,
        "lead_time_std": 0.5,
        "unit_cost": 16.0,
        "min_pack_size": 40,
    },
    {
        "id": "SKU-BS-008",
        "name": "Bingo! Tedhe Medhe Masala Tadka 90g",
        "category": "Biscuits & Snacks",
        "base_daily_demand": 150,
        "demand_std_pct": 0.24,
        "seasonality": "steady",
        "lead_time": 3,
        "lead_time_std": 0.5,
        "unit_cost": 15.5,
        "min_pack_size": 40,
    },
    {
        "id": "SKU-BS-009",
        "name": "Sunfeast YiPPee! Magic Masala Noodles 280g",
        "category": "Biscuits & Snacks",
        "base_daily_demand": 210,
        "demand_std_pct": 0.20,
        "seasonality": "monsoon",
        "lead_time": 4,
        "lead_time_std": 1.0,
        "unit_cost": 42.0,
        "min_pack_size": 24,
    },
    # Personal Care & Hygiene
    {
        "id": "SKU-PC-010",
        "name": "Fiama Gel Bar Blackcurrant & Bearberry 125g",
        "category": "Personal Care",
        "base_daily_demand": 85,
        "demand_std_pct": 0.20,
        "seasonality": "summer",
        "lead_time": 5,
        "lead_time_std": 1.2,
        "unit_cost": 55.0,
        "min_pack_size": 16,
    },
    {
        "id": "SKU-PC-011",
        "name": "Savlon Moisture Shield Liquid Handwash 750ml",
        "category": "Personal Care",
        "base_daily_demand": 110,
        "demand_std_pct": 0.18,
        "seasonality": "monsoon",
        "lead_time": 5,
        "lead_time_std": 1.0,
        "unit_cost": 88.0,
        "min_pack_size": 12,
    },
    {
        "id": "SKU-PC-012",
        "name": "Savlon Antiseptic Liquid 500ml",
        "category": "Personal Care",
        "base_daily_demand": 70,
        "demand_std_pct": 0.16,
        "seasonality": "steady",
        "lead_time": 6,
        "lead_time_std": 1.2,
        "unit_cost": 95.0,
        "min_pack_size": 12,
    },
    {
        "id": "SKU-PC-013",
        "name": "Engage W1 Pocket Perfume Spray 120ml",
        "category": "Personal Care",
        "base_daily_demand": 45,
        "demand_std_pct": 0.30,
        "seasonality": "festive_q4",
        "lead_time": 7,
        "lead_time_std": 1.5,
        "unit_cost": 160.0,
        "min_pack_size": 12,
    },
    {
        "id": "SKU-PC-014",
        "name": "Charmis Deep Radiance Face Serum 30ml",
        "category": "Personal Care",
        "base_daily_demand": 30,
        "demand_std_pct": 0.35,
        "seasonality": "winter",
        "lead_time": 8,
        "lead_time_std": 2.0,
        "unit_cost": 230.0,
        "min_pack_size": 10,
    },
    # Beverages & Confectionery
    {
        "id": "SKU-BC-015",
        "name": "B Natural Mixed Fruit Beverage 1L",
        "category": "Beverages",
        "base_daily_demand": 95,
        "demand_std_pct": 0.28,
        "seasonality": "summer",
        "lead_time": 4,
        "lead_time_std": 1.0,
        "unit_cost": 82.0,
        "min_pack_size": 12,
    },
    {
        "id": "SKU-BC-016",
        "name": "B Natural Himalayan Mixed Berries 1L",
        "category": "Beverages",
        "base_daily_demand": 60,
        "demand_std_pct": 0.30,
        "seasonality": "summer",
        "lead_time": 5,
        "lead_time_std": 1.0,
        "unit_cost": 92.0,
        "min_pack_size": 12,
    },
    {
        "id": "SKU-BC-017",
        "name": "Sunbean Gourmet Beaten Caffe 200g",
        "category": "Beverages",
        "base_daily_demand": 40,
        "demand_std_pct": 0.25,
        "seasonality": "winter",
        "lead_time": 6,
        "lead_time_std": 1.5,
        "unit_cost": 190.0,
        "min_pack_size": 10,
    },
    {
        "id": "SKU-BC-018",
        "name": "Candyman Fantastik Choco Stick 150g",
        "category": "Beverages",
        "base_daily_demand": 125,
        "demand_std_pct": 0.22,
        "seasonality": "festive_q4",
        "lead_time": 4,
        "lead_time_std": 0.8,
        "unit_cost": 45.0,
        "min_pack_size": 24,
    },
    {
        "id": "SKU-BC-019",
        "name": "Jelimals Space Adventure Gummy Bears 100g",
        "category": "Beverages",
        "base_daily_demand": 80,
        "demand_std_pct": 0.25,
        "seasonality": "summer",
        "lead_time": 5,
        "lead_time_std": 1.0,
        "unit_cost": 32.0,
        "min_pack_size": 20,
    },
    # Paper & Stationery
    {
        "id": "SKU-ST-020",
        "name": "Classmate Pulse Spiral Notebook 180 Pages",
        "category": "Paper & Stationery",
        "base_daily_demand": 115,
        "demand_std_pct": 0.35,
        "seasonality": "back_to_school",
        "lead_time": 7,
        "lead_time_std": 1.5,
        "unit_cost": 65.0,
        "min_pack_size": 20,
    },
    {
        "id": "SKU-ST-021",
        "name": "Classmate Interleaved Practical Notebook 120p",
        "category": "Paper & Stationery",
        "base_daily_demand": 85,
        "demand_std_pct": 0.32,
        "seasonality": "back_to_school",
        "lead_time": 7,
        "lead_time_std": 1.5,
        "unit_cost": 48.0,
        "min_pack_size": 20,
    },
    {
        "id": "SKU-ST-022",
        "name": "Paperkraft Callista Premium Pen Pack of 2",
        "category": "Paper & Stationery",
        "base_daily_demand": 35,
        "demand_std_pct": 0.25,
        "seasonality": "festive_q4",
        "lead_time": 9,
        "lead_time_std": 2.0,
        "unit_cost": 220.0,
        "min_pack_size": 10,
    },
    {
        "id": "SKU-ST-023",
        "name": "Classmate Carbonless Graph Pad 50 Sheets",
        "category": "Paper & Stationery",
        "base_daily_demand": 40,
        "demand_std_pct": 0.28,
        "seasonality": "back_to_school",
        "lead_time": 6,
        "lead_time_std": 1.2,
        "unit_cost": 35.0,
        "min_pack_size": 25,
    },
    {
        "id": "SKU-ST-024",
        "name": "Classmate Colour Crew Wax Crayons 24 Shades",
        "category": "Paper & Stationery",
        "base_daily_demand": 65,
        "demand_std_pct": 0.26,
        "seasonality": "back_to_school",
        "lead_time": 5,
        "lead_time_std": 1.0,
        "unit_cost": 50.0,
        "min_pack_size": 24,
    },
]

# Required columns for data validation
REQUIRED_COLUMNS = [
    "Date",
    "Product_ID",
    "Product_Name",
    "Category",
    "Units_Sold",
    "On_Hand_Stock",
    "On_Order_Stock",
    "Backorders",
    "Lead_Time_Days",
    "Unit_Cost",
]

COLUMN_DATA_TYPES = {
    "Date": "datetime",
    "Product_ID": "string",
    "Product_Name": "string",
    "Category": "string",
    "Units_Sold": "numeric",
    "On_Hand_Stock": "numeric",
    "On_Order_Stock": "numeric",
    "Backorders": "numeric",
    "Lead_Time_Days": "numeric",
    "Unit_Cost": "numeric",
}

