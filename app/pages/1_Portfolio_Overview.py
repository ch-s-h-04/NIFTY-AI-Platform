"""Page 1 — Portfolio Overview."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go
import streamlit as st

from app.utils.cache import cached_backtest
from app.utils.dashboard_data import missing_artifacts_message, portfolio_cagr
from app.utils.theme import ACCENT, apply_figure_theme, apply_streamlit_theme


def render() -> None:
    apply_streamlit_theme()
    st.header("Portfolio Overview")

    profile = st.selectbox(
        "Portfolio profile",
        options=["Conservative", "Balanced", "Aggressive"],
        index=1,
    )

    result = cached_backtest(profile)
    if result is None:
        st.error(missing_artifacts_message() or "Backtest results unavailable.")
        return

    m = result.portfolio_metrics
    cagr = portfolio_cagr(result)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CAGR", f"{cagr * 100:.2f}%")
    col2.metric("Sharpe Ratio", f"{m.sharpe:.2f}")
    col3.metric("Sortino Ratio", f"{m.sortino:.2f}")
    col4.metric("Max Drawdown", f"{m.max_drawdown * 100:.2f}%")

    st.caption(
        f"Profile: **{profile}** · {m.n_observations} trading days · "
        f"{result.config.start_date} → {result.config.end_date}"
    )

    equity = result.portfolio_equity
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="Portfolio",
            line=dict(color=ACCENT, width=2),
        )
    )
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        height=480,
    )
    apply_figure_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


render()
