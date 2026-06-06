"""NIFTY-50 AI Platform — Streamlit dashboard entry point."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.utils.dashboard_data import artifact_status, missing_artifacts_message
from app.utils.theme import apply_streamlit_theme

st.set_page_config(
    page_title="NIFTY-50 AI Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_streamlit_theme()

st.title("NIFTY-50 AI Investment Intelligence")
st.markdown(
    """
    Phase 4A dashboard for portfolio backtests, risk analytics, benchmark comparison,
    and model insights. All views read **precomputed Phase 3 outputs** — models are
    not retrained from this UI.
    """
)

status = artifact_status()
if status.any_missing:
    st.warning(missing_artifacts_message(status))
else:
    st.success("All dashboard artifacts found in `outputs/`.")

st.markdown("---")
st.subheader("Navigation")
st.markdown(
    """
    Use the sidebar to open:

    1. **Portfolio Overview** — CAGR, Sharpe, Sortino, max drawdown, equity curve
    2. **Risk Analytics** — drawdown, VaR/CVaR, rolling volatility
    3. **Benchmark Comparison** — portfolio vs NIFTY 50
    4. **Model Insights** — walk-forward metrics and feature importance
    """
)

with st.expander("Execution contract"):
    st.markdown(
        """
        - Signal at market **close** on day *t*
        - Fill at market **open** on day *t+1*
        - Monthly rebalance; horizon-21 predictions for portfolio construction
        - 10 bps transaction costs; CASH residual handling
        """
    )
