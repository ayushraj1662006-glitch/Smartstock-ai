"""
Demand forecasting module for SmartStock AI.
Implements Moving Average and Holt-Winters / Exponential Smoothing using statsmodels,
chronological backtesting, MAE, WAPE, bias evaluation, and robust edge-case handling.
"""

from typing import Dict, Tuple, Any, Optional
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing


def prepare_product_series(
    df: pd.DataFrame,
    product_id: str,
) -> pd.Series:
    """
    Extracts, sorts, and reindexes the daily demand series for a single product.
    Fills any missing calendar dates with 0 units (standard retail demand treatment).
    """
    prod_df = df[df["Product_ID"] == product_id].copy()
    if prod_df.empty:
        return pd.Series(dtype=float)

    prod_df["Date"] = pd.to_datetime(prod_df["Date"])
    daily_demand = prod_df.groupby("Date")["Units_Sold"].sum().sort_index()

    # Reindex to full continuous daily calendar range
    full_idx = pd.date_range(start=daily_demand.index.min(), end=daily_demand.index.max(), freq="D")
    daily_demand = daily_demand.reindex(full_idx, fill_value=0)
    daily_demand.name = "Units_Sold"
    return daily_demand


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Computes MAE and WAPE on actual vs predicted values.
    WAPE = sum(|actual - predicted|) / sum(actual) * 100.
    Handles zero sums gracefully.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    min_len = min(len(actual), len(predicted))
    if min_len == 0:
        return {"MAE": 0.0, "WAPE": 0.0, "Bias": 0.0}
    actual = actual[:min_len]
    predicted = predicted[:min_len]

    # Clip predictions to non-negative
    predicted = np.clip(predicted, a_min=0, a_max=None)

    abs_errors = np.abs(actual - predicted)
    mae = float(np.mean(abs_errors)) if len(abs_errors) > 0 else 0.0

    sum_actual = float(np.sum(actual))
    sum_abs_err = float(np.sum(abs_errors))

    if sum_actual > 0:
        wape = float((sum_abs_err / sum_actual) * 100.0)
        # Bias: (sum(predicted) - sum(actual)) / sum(actual)
        bias = float(((np.sum(predicted) - sum_actual) / sum_actual) * 100.0)
    else:
        wape = 0.0 if sum_abs_err == 0 else 100.0
        bias = 0.0

    return {
        "MAE": round(mae, 2),
        "WAPE": round(wape, 2),
        "Bias": round(bias, 2),
    }


def forecast_moving_average(
    series: pd.Series,
    window: int = 14,
    horizon: int = 14,
) -> pd.Series:
    """
    Generates a baseline moving average forecast for the future horizon.
    """
    if len(series) == 0:
        return pd.Series(dtype=float)

    effective_window = min(window, len(series))
    recent_mean = float(series.iloc[-effective_window:].mean())
    recent_mean = float(np.nan_to_num(recent_mean, nan=0.0))
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    
    forecast_values = np.full(horizon, max(0.0, recent_mean))
    return pd.Series(forecast_values, index=future_dates, name="MA_Forecast")


def forecast_exponential_smoothing(
    series: pd.Series,
    horizon: int = 14,
    seasonal_periods: int = 7,
) -> Tuple[pd.Series, Optional[str]]:
    """
    Fits an Exponential Smoothing model via statsmodels.
    Falls back gracefully from Holt-Winters Additive Seasonality -> Holt's Linear -> Simple Exp Smoothing.
    """
    if len(series) < 7:
        # Insufficient history for smoothing
        fallback_val = float(series.mean()) if len(series) > 0 else 0.0
        fallback_val = float(np.nan_to_num(fallback_val, nan=0.0))
        last_date = series.index[-1] if len(series) > 0 else pd.Timestamp.today()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.Series(np.full(horizon, max(0.0, fallback_val)), index=future_dates), "Mean Baseline (Insufficient History)"

    # Series must have float dtype for statsmodels
    y = series.astype(float)
    last_date = y.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    # Fast path: Constant or zero demand
    if y.nunique() <= 1 or np.isclose(float(y.std()), 0.0):
        val = float(y.iloc[0]) if len(y) > 0 else 0.0
        return pd.Series(np.full(horizon, max(0.0, val)), index=future_dates), "Constant Demand Baseline"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Strategy 1: Holt-Winters with additive weekly seasonality (requires >= 2 full seasons, e.g. >= 14 days)
        if len(y) >= seasonal_periods * 2:
            try:
                model = ExponentialSmoothing(
                    y,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=seasonal_periods,
                    initialization_method="estimated",
                ).fit(optimized=True)
                forecast = model.forecast(horizon)
                return pd.Series(np.clip(forecast.values, 0, None), index=future_dates), "Holt-Winters (Trend + 7-Day Seasonality)"
            except Exception:
                pass

        # Strategy 2: Holt's Linear Trend (no seasonality)
        try:
            model = ExponentialSmoothing(
                y,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
            forecast = model.forecast(horizon)
            return pd.Series(np.clip(forecast.values, 0, None), index=future_dates), "Holt's Linear Trend"
        except Exception:
            pass

        # Strategy 3: Simple Exponential Smoothing (level only)
        try:
            model = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)
            forecast = model.forecast(horizon)
            return pd.Series(np.clip(forecast.values, 0, None), index=future_dates), "Simple Exponential Smoothing (SES)"
        except Exception:
            # Final fallback
            recent_mean = float(y.iloc[-14:].mean())
            recent_mean = float(np.nan_to_num(recent_mean, nan=0.0))
            return pd.Series(np.full(horizon, max(0.0, recent_mean)), index=future_dates), "Moving Average Fallback"


def evaluate_models_chronological(
    series: pd.Series,
    test_horizon: int = 14,
    ma_window: int = 14,
) -> Dict[str, Any]:
    """
    Performs chronological backtest:
    - Train set: all data up to the last `test_horizon` days
    - Test set: the last `test_horizon` days
    - Evaluates Moving Average and Exponential Smoothing on the unseen test set
    - Prevents future-data leakage
    """
    n = len(series)
    if n < test_horizon + 7:
        # Not enough data for meaningful split
        return {
            "error": "Insufficient historical data for chronological train/test split. At least 21 days required.",
            "has_data": False,
        }

    train_series = series.iloc[:-test_horizon]
    test_series = series.iloc[-test_horizon:]
    actuals = test_series.values

    # 1. Backtest Moving Average
    ma_preds = []
    window = min(ma_window, len(train_series))
    current_history = list(train_series.values)
    for _ in range(test_horizon):
        pred = np.mean(current_history[-window:])
        ma_preds.append(max(0.0, pred))
        # In multi-step ahead without feedback, we use recursive or flat prediction
    ma_preds = np.array(ma_preds)
    ma_metrics = calculate_metrics(actuals, ma_preds)

    # 2. Backtest Exponential Smoothing
    exp_preds_series, exp_model_name = forecast_exponential_smoothing(train_series, horizon=test_horizon)
    exp_preds = exp_preds_series.values
    exp_metrics = calculate_metrics(actuals, exp_preds)

    # Create evaluation comparison dataframe
    eval_df = pd.DataFrame({
        "Date": test_series.index.strftime("%Y-%m-%d"),
        "Actual_Demand": test_series.values,
        "MA_Predicted": np.round(ma_preds, 1),
        "ExpSmooth_Predicted": np.round(exp_preds, 1),
    })

    return {
        "has_data": True,
        "test_horizon": test_horizon,
        "train_size": len(train_series),
        "test_size": len(test_series),
        "train_series": train_series,
        "test_series": test_series,
        "ma_metrics": ma_metrics,
        "ma_preds": pd.Series(ma_preds, index=test_series.index),
        "exp_metrics": exp_metrics,
        "exp_preds": exp_preds_series,
        "exp_model_name": exp_model_name,
        "eval_df": eval_df,
    }


def run_future_forecast(
    series: pd.Series,
    method: str = "Exponential Smoothing",
    horizon: int = 14,
    ma_window: int = 14,
) -> Tuple[pd.Series, str, pd.DataFrame]:
    """
    Generates forward-looking demand forecast for the requested horizon (7, 14, or 30 days)
    using the full available historical series.
    Returns: (forecast_series, model_name, forecast_table_df)
    """
    if method == "Moving Average":
        fc = forecast_moving_average(series, window=ma_window, horizon=horizon)
        model_name = f"Moving Average ({ma_window}-day window)"
    else:
        fc, model_name = forecast_exponential_smoothing(series, horizon=horizon)

    # Calculate standard error / confidence bounds based on residual variance
    residuals = series.iloc[-30:] - series.iloc[-30:].mean()
    residual_std = float(np.std(residuals)) if len(residuals) > 1 else float(series.mean() * 0.15)
    
    # 95% interval ~ +/- 1.96 * residual_std * sqrt(h)
    table_rows = []
    cum_forecast = 0.0
    for step, (dt, val) in enumerate(fc.items(), 1):
        bound_margin = 1.96 * residual_std * np.sqrt(step / 7.0)
        lower_bound = max(0.0, round(val - bound_margin, 1))
        upper_bound = round(val + bound_margin, 1)
        cum_forecast += val
        table_rows.append({
            "Date": dt.strftime("%Y-%m-%d"),
            "Day": f"Day +{step}",
            "Forecast_Units": round(val, 1),
            "Lower_Bound_95%": lower_bound,
            "Upper_Bound_95%": upper_bound,
            "Cumulative_Forecast": round(cum_forecast, 1),
        })

    table_df = pd.DataFrame(table_rows)
    return fc, model_name, table_df

