"""Dark theme helpers for Plotly and Streamlit."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

DARK_BG = "#0e1117"
PANEL_BG = "#1a1f2e"
GRID_COLOR = "#2d3446"
TEXT_COLOR = "#e6e6e6"
ACCENT = "#4fc3f7"
ACCENT_2 = "#81c784"
ACCENT_3 = "#ffb74d"
BENCHMARK_COLOR = "#ef5350"

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR, family="Inter, Segoe UI, sans-serif"),
        colorway=[ACCENT, ACCENT_2, ACCENT_3, BENCHMARK_COLOR, "#ba68c8", "#4db6ac"],
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        margin=dict(l=48, r=24, t=48, b=48),
    )
)

pio.templates["nifty_dark"] = PLOTLY_TEMPLATE
pio.templates.default = "nifty_dark"

STREAMLIT_CSS = """
<style>
    .stApp { background-color: #0e1117; }
    [data-testid="stMetricValue"] { color: #4fc3f7; }
    [data-testid="stMetricLabel"] { color: #b0b8c4; }
    .dashboard-caption { color: #8892a0; font-size: 0.9rem; }
</style>
"""


def apply_streamlit_theme() -> None:
    import streamlit as st

    st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)


def apply_figure_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(template="nifty_dark", hovermode="x unified")
    return fig
