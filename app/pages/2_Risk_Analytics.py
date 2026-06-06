"""Page 2 — Risk Analytics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go
import streamlit as st

from app.utils.cache import cached_backtest
from app.utils.dashboard_data import missing_artifacts_message, rolling_annualized_volatility
from app.utils.theme import ACCENT, ACCENT_3, apply_figure_theme, apply_streamlit_theme
from src.risk import drawdown_series, equity_curve_from_returns


def render() -> None:
    apply_streamlit_theme()
    st.header("Risk Analytics")

    profile = st.selectbox(
        "Portfolio profile",
        options=["Conservative", "Balanced", "Aggressive"],
        index=1,
        key="risk_profile",
    )

    result = cached_backtest(profile)
    if result is None:
        st.error(missing_artifacts_message() or "Backtest results unavailable.")
        return

    m = result.portfolio_metrics
    returns = result.portfolio_returns

    col1, col2, col3 = st.columns(3)
    col1.metric("VaR (95%)", f"{m.var_95 * 100:.2f}%")
    col2.metric("CVaR (95%)", f"{m.cvar_95 * 100:.2f}%")
    col3.metric("Annualized Volatility", f"{m.annualized_volatility * 100:.2f}%")

    equity = equity_curve_from_returns(returns)
    dd = drawdown_series(equity) * 100.0

    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line=dict(color=ACCENT_3, width=1.5),
        )
    )
    fig_dd.update_layout(
        title="Drawdown Profile",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=400,
    )
    apply_figure_theme(fig_dd)
    st.plotly_chart(fig_dd, use_container_width=True)

    roll_vol = rolling_annualized_volatility(returns) * 100.0
    fig_vol = go.Figure()
    fig_vol.add_trace(
        go.Scatter(
            x=roll_vol.index,
            y=roll_vol.values,
            mode="lines",
            name="63-day rolling vol",
            line=dict(color=ACCENT, width=2),
        )
    )
    fig_vol.update_layout(
        title="Rolling Annualized Volatility (63-day)",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        height=400,
    )
    apply_figure_theme(fig_vol)
    st.plotly_chart(fig_vol, use_container_width=True)


render()
