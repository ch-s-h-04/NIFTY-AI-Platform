"""Page 3 — Benchmark Comparison."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.utils.cache import cached_backtest
from app.utils.dashboard_data import benchmark_cagr, missing_artifacts_message, portfolio_cagr
from app.utils.theme import ACCENT, BENCHMARK_COLOR, apply_figure_theme, apply_streamlit_theme
from src.risk import drawdown_series, equity_curve_from_returns


def render() -> None:
    apply_streamlit_theme()
    st.header("Benchmark Comparison")

    profile = st.selectbox(
        "Portfolio profile",
        options=["Conservative", "Balanced", "Aggressive"],
        index=1,
        key="bench_profile",
    )

    result = cached_backtest(profile)
    if result is None:
        st.error(missing_artifacts_message() or "Backtest results unavailable.")
        return

    port_eq = result.portfolio_equity
    bench_eq = result.benchmark_equity
    port_dd = drawdown_series(port_eq)
    bench_dd = drawdown_series(bench_eq)

    rel = (port_eq / bench_eq) * (bench_eq.iloc[0] / port_eq.iloc[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=port_eq.index,
            y=port_eq.values,
            mode="lines",
            name="Portfolio",
            line=dict(color=ACCENT, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bench_eq.index,
            y=bench_eq.values,
            mode="lines",
            name="NIFTY 50",
            line=dict(color=BENCHMARK_COLOR, width=2, dash="dot"),
        )
    )
    fig.update_layout(
        title="Portfolio vs NIFTY 50 — Equity Curves",
        xaxis_title="Date",
        yaxis_title="Growth of $1",
        height=440,
    )
    apply_figure_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    fig_rel = go.Figure()
    fig_rel.add_trace(
        go.Scatter(
            x=rel.index,
            y=rel.values,
            mode="lines",
            name="Relative performance",
            line=dict(color=ACCENT, width=2),
        )
    )
    fig_rel.add_hline(y=1.0, line_dash="dash", line_color="#8892a0", opacity=0.6)
    fig_rel.update_layout(
        title="Relative Performance (Portfolio / Benchmark)",
        xaxis_title="Date",
        yaxis_title="Relative index",
        height=360,
    )
    apply_figure_theme(fig_rel)
    st.plotly_chart(fig_rel, use_container_width=True)

    pm = result.portfolio_metrics
    bm = result.benchmark_metrics
    comparison = pd.DataFrame(
        {
            "Metric": ["CAGR", "Sharpe", "Sortino", "Max Drawdown", "Ann. Volatility"],
            "Portfolio": [
                f"{portfolio_cagr(result) * 100:.2f}%",
                f"{pm.sharpe:.2f}",
                f"{pm.sortino:.2f}",
                f"{pm.max_drawdown * 100:.2f}%",
                f"{pm.annualized_volatility * 100:.2f}%",
            ],
            "NIFTY 50": [
                f"{benchmark_cagr(result) * 100:.2f}%",
                f"{bm.sharpe:.2f}",
                f"{bm.sortino:.2f}",
                f"{bm.max_drawdown * 100:.2f}%",
                f"{bm.annualized_volatility * 100:.2f}%",
            ],
        }
    )

    st.subheader("CAGR & Risk Comparison")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    dd_table = pd.DataFrame(
        {
            "Series": ["Portfolio", "NIFTY 50"],
            "Max Drawdown": [f"{port_dd.min() * 100:.2f}%", f"{bench_dd.min() * 100:.2f}%"],
        }
    )
    st.subheader("Drawdown Comparison")
    st.dataframe(dd_table, use_container_width=True, hide_index=True)


render()
