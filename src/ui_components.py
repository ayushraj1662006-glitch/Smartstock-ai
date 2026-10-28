"""
UI components and styling module for SmartStock AI.
Provides custom CSS, reusable KPI cards, status badges, and styled Plotly charts
adhering to a polished navy, white, and teal enterprise design system.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.config import COLORS


def inject_custom_css():
    """Injects high-end, responsive enterprise CSS into the Streamlit app."""
    custom_css = f"""
    <style>
        /* Import clean modern typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: {COLORS['text_dark']};
        }}

        /* App Background */
        .stApp {{
            background-color: {COLORS['bg_light']};
        }}

        /* Top Header Styling */
        .main-header {{
            background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
            padding: 22px 28px;
            border-radius: 12px;
            color: #FFFFFF;
            margin-bottom: 24px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
        }}
        .main-header h1 {{
            color: #FFFFFF !important;
            font-size: 26px;
            font-weight: 700;
            margin: 0;
            padding: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .main-header p {{
            color: #E2E8F0;
            font-size: 13.5px;
            margin: 6px 0 0 0;
            font-weight: 400;
        }}

        /* KPI Card Container */
        .kpi-card {{
            background: {COLORS['card_bg']};
            border: 1px solid {COLORS['card_border']};
            border-radius: 10px;
            padding: 18px 20px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
            transition: all 0.2s ease-in-out;
            height: 100%;
        }}
        .kpi-card:hover {{
            box-shadow: 0 6px 16px rgba(14, 116, 144, 0.12);
            border-color: {COLORS['accent']};
            transform: translateY(-2px);
        }}
        .kpi-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {COLORS['text_muted']};
            font-weight: 600;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 26px;
            font-weight: 700;
            color: {COLORS['primary']};
            line-height: 1.2;
            margin-bottom: 6px;
        }}
        .kpi-subtext {{
            font-size: 12px;
            color: {COLORS['text_muted']};
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        /* Status Badge Pills */
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 11.5px;
            font-weight: 600;
            text-align: center;
        }}
        .badge-healthy {{
            background-color: {COLORS['success_bg']};
            color: {COLORS['success']};
            border: 1px solid rgba(5, 150, 105, 0.25);
        }}
        .badge-reorder {{
            background-color: {COLORS['warning_bg']};
            color: {COLORS['warning']};
            border: 1px solid rgba(217, 119, 6, 0.25);
        }}
        .badge-stockout {{
            background-color: {COLORS['danger_bg']};
            color: {COLORS['danger']};
            border: 1px solid rgba(220, 38, 38, 0.25);
        }}
        .badge-excess {{
            background-color: {COLORS['info_bg']};
            color: {COLORS['info']};
            border: 1px solid rgba(37, 99, 235, 0.25);
        }}

        /* Sidebar Customization */
        [data-testid="stSidebar"] {{
            background-color: #0B1120;
            border-right: 1px solid #1E293B;
        }}
        [data-testid="stSidebar"] * {{
            color: #E2E8F0 !important;
        }}
        [data-testid="stSidebar"] .stSelectbox label, 
        [data-testid="stSidebar"] .stMultiSelect label,
        [data-testid="stSidebar"] .stSlider label {{
            color: #94A3B8 !important;
            font-size: 12.5px;
            font-weight: 600;
        }}

        /* Card panels */
        .section-card {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.03);
        }}
        .section-title {{
            font-size: 16px;
            font-weight: 700;
            color: {COLORS['primary']};
            margin-bottom: 14px;
            border-bottom: 2px solid #F1F5F9;
            padding-bottom: 8px;
        }}

        /* Table styling tweak */
        div[data-testid="stDataFrame"] {{
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            overflow: hidden;
        }}

        /* Streamlit Button Tweaks */
        .stButton>button {{
            background: linear-gradient(135deg, {COLORS['secondary']} 0%, {COLORS['primary']} 100%);
            color: #FFFFFF !important;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            padding: 8px 18px;
            transition: all 0.2s ease;
        }}
        .stButton>button:hover {{
            background: linear-gradient(135deg, {COLORS['accent']} 0%, {COLORS['secondary']} 100%);
            color: #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3);
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, badge_text: str = "ITC Supply Chain Case"):
    """Renders the top enterprise header banner."""
    header_html = f"""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
            <div>
                <h1>📦 SmartStock AI <span style="font-size: 14px; font-weight: 500; background: rgba(255,255,255,0.18); padding: 3px 10px; border-radius: 6px; margin-left: 8px;">v2.4 Enterprise</span></h1>
                <p><strong>{title}</strong> — {subtitle}</p>
            </div>
            <div style="text-align: right;">
                <span class="badge" style="background: rgba(6, 182, 212, 0.2); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.4); font-size: 12px; padding: 5px 12px;">
                    🏭 {badge_text}
                </span>
            </div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_kpi_card(
    label: str,
    value: str,
    subtext: str = "",
    status_type: str = "default",
    icon: str = "📊",
):
    """Renders a styled enterprise KPI card."""
    accent_bar_color = COLORS.get(status_type, COLORS["secondary"])
    card_html = f"""
    <div class="kpi-card" style="border-top: 4px solid {accent_bar_color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="kpi-label">{label}</div>
            <span style="font-size: 18px;">{icon}</span>
        </div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtext">{subtext}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def get_status_badge_html(status: str) -> str:
    """Returns styled HTML badge for an inventory status string."""
    status_lower = status.lower()
    if "stockout" in status_lower:
        return f'<span class="badge badge-stockout">⚠️ {status}</span>'
    elif "reorder" in status_lower:
        return f'<span class="badge badge-reorder">🔄 {status}</span>'
    elif "excess" in status_lower:
        return f'<span class="badge badge-excess">📦 {status}</span>'
    else:
        return f'<span class="badge badge-healthy">✅ {status}</span>'


def create_plotly_theme() -> Dict[str, Any]:
    """Returns standard Plotly layout configuration for consistent styling."""
    return dict(
        font=dict(family="Inter, sans-serif", size=11, color=COLORS["text_dark"]),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=40, r=30, t=40, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            tickfont=dict(size=10, color=COLORS["text_muted"]),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
            tickfont=dict(size=10, color=COLORS["text_muted"]),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor=COLORS["primary"],
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#FFFFFF",
        ),
    )

