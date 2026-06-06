"""Cached dashboard loaders (no model training)."""

from __future__ import annotations

import streamlit as st

from app.utils.dashboard_data import (
    PROFILE_OPTIONS,
    load_feature_importance,
    load_fold_metrics,
    load_shap_feature_importance,
    load_shap_summary,
    load_summary_metrics,
    run_profile_backtest,
)
from src.backtest import BacktestResult
from src.portfolio import PortfolioProfile


@st.cache_data(show_spinner="Loading backtest simulation...")
def cached_backtest(profile_label: str) -> BacktestResult | None:
    profile = PROFILE_OPTIONS[profile_label]
    return run_profile_backtest(profile)


@st.cache_data(show_spinner="Loading model metrics...")
def cached_summary_metrics():
    return load_summary_metrics()


@st.cache_data(show_spinner="Loading fold metrics...")
def cached_fold_metrics():
    return load_fold_metrics()


@st.cache_data(show_spinner="Loading feature importance...")
def cached_feature_importance():
    return load_feature_importance()


@st.cache_data(show_spinner="Loading SHAP summary...")
def cached_shap_summary():
    return load_shap_summary()


@st.cache_data(show_spinner="Loading SHAP feature importance...")
def cached_shap_feature_importance():
    return load_shap_feature_importance()
