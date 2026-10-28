# SmartStock AI — FMCG Demand Forecasting & Inventory Optimization System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![Pytest Tests](https://img.shields.io/badge/pytest-25%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Undergraduate Engineering Project** designed to demonstrate end-to-end practical data analytics, stochastic inventory modeling, and supply-chain decision-making relevant to an **ITC Limited Logistics & Warehousing Internship**.

---

## 📌 Executive Summary

**SmartStock AI** is a genuine, production-grade Python and Streamlit decision-support application built for multi-echelon consumer goods (FMCG) logistics networks. It bridges statistical time-series demand forecasting with stochastic inventory optimization to address classic supply-chain challenges: stockouts, working capital inflation, and the bullwhip effect.

The application operates on a 365-day synthetic dataset spanning **24 realistic SKUs** across 5 FMCG categories (**Foods & Staples, Biscuits & Snacks, Personal Care, Beverages, Paper & Stationery**)—simulating product portfolios akin to *Aashirvaad, Sunfeast, Bingo!, YiPPee!, Savlon, Fiama,* and *Classmate*.

*DISCLAIMER: All data is completely simulated and synthetic for educational and portfolio demonstration. It does not represent proprietary operational data of ITC Limited.*

---

## 🏗️ System Architecture

```
smartstock-ai/
├── app.py                     # Streamlit enterprise multi-page dashboard
├── requirements.txt           # Core Python dependencies
├── pytest.ini                 # Pytest configuration
├── README.md                  # Comprehensive documentation & internship guide
├── data/
│   ├── fmcg_inventory_data.csv # 365-day 24-SKU simulated time-series (8,760 rows)
│   └── sample_template.xlsx   # Official formatted Excel upload template & data dictionary
├── src/
│   ├── __init__.py
│   ├── config.py              # Visual themes, Z-scores, SKU catalog definitions
│   ├── data_generator.py      # Synthetic time-series simulation engine
│   ├── data_validation.py     # Schema checking, type coercion, column mapping
│   ├── forecasting.py         # Holt-Winters & MA models, chronological backtesting
│   ├── inventory.py           # Safety stock, ROP, inventory position, PO logic
│   ├── reporting.py           # Multi-sheet openpyxl workbook generator
│   └── ui_components.py       # Custom CSS, KPI cards, badges, Plotly themes
└── tests/
    ├── __init__.py
    └── test_calculations.py   # 18 unit tests covering all core formulas
```

---

## 🧮 Mathematical & Supply Chain Formulations

### 1. Chronological Demand Forecasting
- **Moving Average Baseline:**
  $$\hat{y}_{t+h} = \frac{1}{N} \sum_{i=0}^{N-1} y_{t-i}$$
- **Holt-Winters Exponential Smoothing (Statsmodels):**
  Incorporates additive level ($\ell_t$), linear trend ($b_t$), and 7-day weekly seasonality ($s_t$) to capture retail consumer purchasing patterns:
  $$\hat{y}_{t+h} = \ell_t + h b_t + s_{t+h-m}$$
- **WAPE (Weighted Absolute Percentage Error):**
  $$\text{WAPE} = \frac{\sum |y_t - \hat{y}_t|}{\sum y_t} \times 100\%$$
  *Why WAPE over MAPE in FMCG?* In warehouse depots, slow-moving SKUs often record zero sales ($y_t = 0$), causing MAPE to encounter division by zero ($0/0$). WAPE weights errors by volume, preventing distortion while remaining analytically credible.

### 2. Stochastic Safety Stock Modeling
Distinguishes between consumer demand variability and supplier lead-time delivery variability:
$$\mathbf{SS = Z \times \sqrt{L \cdot \sigma_d^2 + \bar{d}^2 \cdot \sigma_L^2}}$$
- $Z$: Normal distribution factor for target non-stockout probability ($Z=1.65$ for 95%, $Z=2.33$ for 99%).
- $L \cdot \sigma_d^2$: Demand variance during lead time.
- $\bar{d}^2 \cdot \sigma_L^2$: Lead time variance contribution (supplier truck delays).

### 3. Reorder Point ($ROP$) & Inventory Position ($IP$)
$$\mathbf{ROP = (\bar{d} \times L) + SS}$$
$$\mathbf{IP = On\text{-}Hand + On\text{-}Order - Backorders}$$

*Critical Supply Chain Rule:* Purchase orders are triggered if and only if **$IP \le ROP$**, never merely $On\text{-}Hand \le ROP$. Factoring in pipeline stock ($On\text{-}Order$) prevents duplicate purchase orders while shipments are in transit.

### 4. Recommended Replenishment Quantity
Using periodic base-stock $(R, s, S)$ replenishment policy:
$$\text{Target Max} = ROP + (\bar{d} \times R)$$
$$\text{Recommended Order Qty} = \max(0, \lceil \text{Target Max} - IP \rceil)$$
- **Guaranteed non-negative.**
- Prevents double-ordering pipeline stock.

---

## 🖥️ Application Pages Overview

| Page | Key Capabilities & Features |
| :--- | :--- |
| **1. Executive Dashboard** | 6 high-level KPI cards (Total Products, Units On Hand, Inventory Value, Replenishment Needed, Stockout Risk, Excess Stock). 5 interactive Plotly charts: monthly demand trend, capital distribution by category, top-demand SKUs, ROP deficit analysis, and priority action table. |
| **2. Demand Forecasting** | SKU selector, 7/14/30-day forecast horizons. Chronological train/test split backtest (zero data leakage). Evaluates MAE, WAPE, and Bias. Interactive chart with 95% confidence intervals and downloadable CSV table. |
| **3. Inventory Optimization** | Portfolio-level table calculating $OH$, $OO$, $BO$, $IP$, $LTD$, $SS$, $ROP$, and Recommended Order Qty. Mathematical SKU drill-down displaying exact formulas, variance breakdown, and recommendation audit trail. |
| **4. Product Explorer** | Searchable SKU directory with multi-filter sorting. Deep-dive card featuring 120-day demand history with rolling average overlay and interactive Buffer Waterline gauge comparing physical stock, safety stock, and ROP. |
| **5. Excel & Data Management** | Download official sample workbook (`.xlsx`) with Data Dictionary. Upload custom CSV/XLSX files. Automatic column mapping with synonym matching. Comprehensive validation for missing columns, unparseable dates, and negative numbers. |
| **6. Reports & Exports** | Generates an executive-grade 7-sheet Excel report using `openpyxl`: (1) Inventory Summary, (2) Demand Forecasts, (3) Forecast Accuracy, (4) Reorder Recommendations, (5) Stockout Risk, (6) Excess Inventory, (7) Methodology. Includes navy headers, frozen panes, and currency formats. |
| **7. Methodology & Guide** | Comprehensive supply chain educational guide tailored for an **ITC Logistics & Warehousing interview**. Explains distribution echelons, bullwhip mitigation, stockout censoring, holding cost components, and model interview Q&A. |

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.11 or higher
- Git

### 2. Setup Virtual Environment
```bash
# Clone the repository (or navigate to workspace)
cd smartstock-ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Pytest Test Suite
```bash
pytest -v
```
*(All 25 unit tests verify forecasting metrics, chronological holdouts, safety stock bounds, inventory positions, order quantities, Excel roundtrips, and data validation rules).*

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## 📊 Suggested Screenshots for GitHub Portfolio

When publishing this project to your GitHub repository or attaching it to an internship portfolio, take screenshots of these 5 views:

1. **Executive Dashboard (`docs/screenshots/01_executive_dashboard.png`):**
   * Capture the full screen showing the navy header, 6 KPI cards, and the monthly demand stacked bar chart alongside the capital distribution donut chart.
2. **Holt-Winters Demand Forecast (`docs/screenshots/02_demand_forecasting.png`):**
   * Show the interactive plot for *Sunfeast Dark Fantasy Choco Fills* displaying historical demand, holdout test actuals, backtest fit, and the 14-day future projection with the cyan 95% confidence interval band.
3. **Inventory Decision Audit (`docs/screenshots/03_inventory_optimization.png`):**
   * Zoom into the mathematical drill-down box showing the LaTeX formulas for $IP$, $LTD$, $SS$ (with demand vs lead time variance breakdown), and $ROP$.
4. **Buffer Waterline Gauge (`docs/screenshots/04_buffer_waterline.png`):**
   * In the Product Explorer, capture the stacked waterline chart showing on-hand stock and in-transit orders plotted against the dashed red Reorder Point and amber Safety Stock lines.
5. **Multi-Sheet Excel Export (`docs/screenshots/05_excel_report.png`):**
   * Open the generated `SmartStock_Executive_Report.xlsx` in Microsoft Excel, displaying the frozen navy headers, formatted currency values (`₹`), and the tab bar showing all 7 generated sheets.

---

## 💡 Key Supply Chain Concepts for Interview Preparation

| Concept | Definition & ITC FMCG Application |
| :--- | :--- |
| **Primary vs Secondary Distribution** | *Primary:* Movement from manufacturing factories to Central / Regional Distribution Centers (full truckloads / FTL). *Secondary:* Movement from RDCs to wholesale distributors and retail points (less-than-truckload / LTL). |
| **Bullwhip Effect** | Amplification of demand variance upstream from retail kiranas to factories. Mitigated by sharing POS data, reducing review periods ($R$), and establishing collaborative vendor replenishment. |
| **Service Level Trade-off** | Moving from 95% to 99% service level increases safety stock by **41.4%**, drastically elevating working capital carrying costs. Segmenting SKUs via ABC/XYZ analysis optimizes this trade-off. |
| **Stockout Censoring** | When inventory hits zero, sales drop to zero regardless of true demand. Training forecasting models on unadjusted sales distorts baseline projections; censored observations must be flagged and corrected. |
| **Inventory Holding Cost** | Typically 18–25% annually in FMCG, encompassing Cost of Capital (WACC), warehousing rent/utilities, material handling, shelf-life expiry, and transit insurance. |

---

## ⚖️ License & Academic Integrity

This project is released under the **MIT License**. It was developed as an academic demonstration of applied analytics and logistics engineering. All product names (*Aashirvaad, Sunfeast, Bingo!, YiPPee!, Savlon, Fiama, Classmate*) are trademarks of ITC Limited and are referenced solely for simulated academic modeling.

