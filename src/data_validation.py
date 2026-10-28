"""
Data validation module for SmartStock AI.
Performs schema checking, type coercion, missing value analysis,
outlier/negative value detection, and flexible column mapping.
"""

from typing import Dict, List, Tuple, Any, Optional
import io
import pandas as pd
import numpy as np

from src.config import REQUIRED_COLUMNS


# Common synonyms for standard column mapping
SYNONYM_MAP = {
    "Date": ["date", "day", "txn_date", "timestamp", "order_date", "record_date"],
    "Product_ID": ["product_id", "sku", "sku_id", "item_code", "product_code", "material_id"],
    "Product_Name": ["product_name", "item_name", "description", "sku_name", "product", "item"],
    "Category": ["category", "dept", "vertical", "product_line", "group", "family"],
    "Units_Sold": ["units_sold", "sales", "quantity_sold", "demand", "qty", "sold_qty", "volume"],
    "On_Hand_Stock": ["on_hand_stock", "on_hand", "stock", "inventory", "current_stock", "physical_stock", "oh_stock"],
    "On_Order_Stock": ["on_order_stock", "on_order", "incoming_stock", "in_transit", "po_qty", "pipeline_stock"],
    "Backorders": ["backorders", "backorder", "unmet_orders", "pending_orders", "bo_qty", "shortage"],
    "Lead_Time_Days": ["lead_time_days", "lead_time", "lt_days", "supplier_lead_time", "delivery_days"],
    "Unit_Cost": ["unit_cost", "cost", "purchase_cost", "cogs", "cost_price", "unit_price"],
}


def suggest_column_mapping(df_columns: List[str]) -> Dict[str, Optional[str]]:
    """
    Intelligently attempts to match uploaded column names to standard schema.
    Returns a dict {standard_col: matched_df_col_or_None}.
    """
    mapping = {}
    lower_cols = {col.lower().strip().replace(" ", "_"): col for col in df_columns}

    for std_col, synonyms in SYNONYM_MAP.items():
        matched = None
        # Exact match (case insensitive)
        std_key = std_col.lower().strip().replace(" ", "_")
        if std_key in lower_cols:
            matched = lower_cols[std_key]
        else:
            # Synonym match
            for syn in synonyms:
                if syn in lower_cols:
                    matched = lower_cols[syn]
                    break
        mapping[std_col] = matched

    return mapping


def validate_and_clean_data(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[bool, List[str], List[str], Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Validates and cleans the input dataset.

    Returns:
        (is_valid, error_list, warning_list, cleaned_dataframe, stats_dict)
    """
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {}

    if df is None or df.empty:
        return False, ["Uploaded file is empty or could not be read."], [], None, {}

    df_working = df.copy()

    # Apply column mapping if provided
    if column_mapping:
        # Invert or apply direct rename: {mapped_name: standard_name}
        rename_dict = {orig: std for std, orig in column_mapping.items() if orig in df_working.columns}
        df_working = df_working.rename(columns=rename_dict)

    # 1. Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_working.columns]
    if missing_cols:
        errors.append(
            f"Missing required columns: {', '.join(missing_cols)}. "
            f"Please map your file's columns or use the standard schema."
        )
        return False, errors, warnings, None, stats

    total_rows_initial = len(df_working)

    # 2. Date parsing & validation
    try:
        df_working["Date"] = pd.to_datetime(df_working["Date"], errors="coerce")
        null_dates = df_working["Date"].isna().sum()
        if null_dates > 0:
            errors.append(f"Found {null_dates} rows with unparseable or missing Date values.")
    except Exception as e:
        errors.append(f"Error parsing 'Date' column: {str(e)}")

    # 3. String columns validation
    for str_col in ["Product_ID", "Product_Name", "Category"]:
        null_count = df_working[str_col].isna().sum()
        if null_count > 0:
            errors.append(f"Column '{str_col}' contains {null_count} missing values.")
        df_working[str_col] = df_working[str_col].astype(str).str.strip()

    # 4. Numeric columns validation & coercion
    numeric_cols = [
        "Units_Sold",
        "On_Hand_Stock",
        "On_Order_Stock",
        "Backorders",
        "Lead_Time_Days",
        "Unit_Cost",
    ]

    for num_col in numeric_cols:
        # Convert to numeric
        df_working[num_col] = pd.to_numeric(df_working[num_col], errors="coerce")
        null_count = df_working[num_col].isna().sum()
        if null_count > 0:
            errors.append(f"Column '{num_col}' contains {null_count} non-numeric or missing values.")
        
        # Check non-negative constraints
        if num_col in ["Units_Sold", "On_Hand_Stock", "On_Order_Stock", "Backorders"]:
            neg_count = (df_working[num_col] < 0).sum()
            if neg_count > 0:
                errors.append(f"Column '{num_col}' contains {neg_count} negative values (must be >= 0).")
        
        # Lead time and unit cost must be strictly positive
        if num_col == "Lead_Time_Days":
            non_pos = (df_working[num_col] <= 0).sum()
            if non_pos > 0:
                errors.append(f"Column 'Lead_Time_Days' contains {non_pos} non-positive values (must be > 0 days).")
        
        if num_col == "Unit_Cost":
            non_pos = (df_working[num_col] <= 0).sum()
            if non_pos > 0:
                errors.append(f"Column 'Unit_Cost' contains {non_pos} non-positive values (must be > ₹0.00).")

    # If critical structural errors exist, stop here
    if errors:
        return False, errors, warnings, None, stats

    # 5. Check duplicate keys: (Date, Product_ID)
    duplicates = df_working.duplicated(subset=["Date", "Product_ID"], keep=False).sum()
    if duplicates > 0:
        warnings.append(
            f"Found {duplicates} duplicate (Date, Product_ID) pairs. "
            f"Keeping the last recorded observation."
        )
        df_working = df_working.drop_duplicates(subset=["Date", "Product_ID"], keep="last")

    # 6. Check time series continuity & summary stats
    min_date = df_working["Date"].min()
    max_date = df_working["Date"].max()
    unique_skus = df_working["Product_ID"].nunique()
    total_days = (max_date - min_date).days + 1 if pd.notna(min_date) and pd.notna(max_date) else 0

    if unique_skus < 20:
        warnings.append(f"Dataset has {unique_skus} products. ITC enterprise standard recommends >= 20 products for portfolio analysis.")

    if total_days < 60:
        warnings.append(f"Dataset spans only {total_days} days. Time-series forecasting reliability improves with at least 90-365 days of history.")

    # Cast integer columns cleanly
    int_cols = ["Units_Sold", "On_Hand_Stock", "On_Order_Stock", "Backorders", "Lead_Time_Days"]
    for col in int_cols:
        df_working[col] = df_working[col].round().astype(int)

    df_working["Unit_Cost"] = df_working["Unit_Cost"].astype(float).round(2)

    # Sort deterministically
    df_working = df_working.sort_values(by=["Product_ID", "Date"]).reset_index(drop=True)

    stats = {
        "total_records": len(df_working),
        "total_products": unique_skus,
        "categories": sorted(df_working["Category"].unique().tolist()),
        "min_date": min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else "N/A",
        "max_date": max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else "N/A",
        "total_days_span": total_days,
        "total_units_sold": int(df_working["Units_Sold"].sum()),
        "avg_daily_sales_overall": float(round(df_working["Units_Sold"].mean(), 2)),
    }

    return True, [], warnings, df_working, stats


def parse_uploaded_file(file_bytes: bytes, filename: str) -> Optional[pd.DataFrame]:
    """Parses an uploaded CSV or Excel file into a pandas DataFrame."""
    try:
        if filename.endswith(".csv"):
            return pd.read_csv(io.BytesIO(file_bytes))
        elif filename.endswith((".xlsx", ".xls")):
            xl = pd.ExcelFile(io.BytesIO(file_bytes))
            req_keywords = {"date", "product_id", "units_sold", "on_hand_stock", "sku", "sales", "stock", "category"}
            best_df = None
            max_matches = 0
            for s_name in xl.sheet_names:
                df_candidate = xl.parse(s_name)
                candidate_cols = {str(c).lower().strip().replace(" ", "_") for c in df_candidate.columns}
                matches = len(candidate_cols.intersection(req_keywords))
                if matches > max_matches:
                    max_matches = matches
                    best_df = df_candidate
            return best_df if best_df is not None else xl.parse(0)
        else:
            return None
    except Exception:
        return None

