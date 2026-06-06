"""Page 4 — Model Insights."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.express as px
import streamlit as st

from app.utils.cache import cached_feature_importance, cached_summary_metrics
from app.utils.dashboard_data import missing_artifacts_message
from app.utils.theme import apply_figure_theme, apply_streamlit_theme


def render() -> None:
    apply_streamlit_theme()
    st.header("Model Insights")

    summary = cached_summary_metrics()
    if summary is None or summary.empty:
        st.warning(missing_artifacts_message() or "Model summary metrics not found.")
    else:
        st.subheader("Walk-Forward CV Metrics (Horizon 21)")
        display = summary.copy()
        if "horizon" in display.columns:
            display = display.loc[display["horizon"] == 21].reset_index(drop=True)

        for _, row in display.iterrows():
            model_name = str(row.get("model", "Model")).title()
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{model_name} — MAE", f"{row['mae']:.4f}")
            c2.metric(f"{model_name} — RMSE", f"{row['rmse']:.4f}")
            c3.metric(
                f"{model_name} — Directional Accuracy",
                f"{row['directional_accuracy'] * 100:.1f}%",
            )

        st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Feature Importance (LightGBM)")
    importance = cached_feature_importance()
    if importance is None or importance.empty:
        st.info(
            "Feature importance file not found. Export artifacts with "
            "`python -m app.utils.export_artifacts`."
        )
    else:
        top_n = st.slider("Top N features", min_value=5, max_value=30, value=15, step=1)
        top = importance.nlargest(top_n, "importance")
        fig = px.bar(
            top,
            x="importance",
            y="feature",
            orientation="h",
            title=f"Top {top_n} LightGBM Features (21-day horizon)",
        )
        fig.update_layout(yaxis=dict(categoryorder="total ascending"), height=520)
        apply_figure_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("SHAP Explainability")
    st.info(
        "**Phase 4B placeholder** — SHAP summary and waterfall plots will be integrated "
        "here after approval. No SHAP computation runs in Phase 4A."
    )


render()
