"""Page 4 — Model Insights."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.express as px
import streamlit as st

from app.utils.cache import (
    cached_feature_importance,
    cached_shap_feature_importance,
    cached_shap_summary,
    cached_summary_metrics,
)
from app.utils.dashboard_data import missing_artifacts_message, missing_shap_artifacts_message
from app.utils.shap_viz import (
    plot_shap_importance_bar,
    plot_shap_summary,
    plot_signed_feature_bar,
    top_negative_features,
    top_positive_features,
)
from app.utils.theme import ACCENT, ACCENT_3, apply_figure_theme, apply_streamlit_theme


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
    shap_summary = cached_shap_summary()
    shap_importance = cached_shap_feature_importance()

    if shap_summary is None or shap_importance is None or shap_summary.empty or shap_importance.empty:
        st.warning(
            missing_shap_artifacts_message()
            or "SHAP artifacts not found. Run `python -m app.utils.export_shap`."
        )
    else:
        shap_top_n = st.slider(
            "SHAP top N features",
            min_value=5,
            max_value=25,
            value=15,
            step=1,
            key="shap_top_n",
        )

        st.plotly_chart(
            plot_shap_summary(shap_summary, shap_importance, top_n=shap_top_n),
            use_container_width=True,
        )
        st.plotly_chart(
            plot_shap_importance_bar(shap_importance, top_n=shap_top_n),
            use_container_width=True,
        )

        pos = top_positive_features(shap_importance, n=10)
        neg = top_negative_features(shap_importance, n=10)

        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.markdown("**Top Positive Features**")
            st.caption("Features that increase predicted 21-day returns on average.")
            if pos.empty:
                st.info("No features with positive mean SHAP in this sample.")
            else:
                st.dataframe(
                    pos.assign(mean_shap=pos["mean_shap"].map(lambda x: f"{x:.6f}")),
                    use_container_width=True,
                    hide_index=True,
                )
                st.plotly_chart(
                    plot_signed_feature_bar(
                        pos,
                        value_col="mean_shap",
                        title="Top Positive Mean SHAP",
                        color=ACCENT,
                    ),
                    use_container_width=True,
                )

        with col_neg:
            st.markdown("**Top Negative Features**")
            st.caption("Features that decrease predicted 21-day returns on average.")
            if neg.empty:
                st.info("No features with negative mean SHAP in this sample.")
            else:
                st.dataframe(
                    neg.assign(mean_shap=neg["mean_shap"].map(lambda x: f"{x:.6f}")),
                    use_container_width=True,
                    hide_index=True,
                )
                st.plotly_chart(
                    plot_signed_feature_bar(
                        neg,
                        value_col="mean_shap",
                        title="Top Negative Mean SHAP",
                        color=ACCENT_3,
                    ),
                    use_container_width=True,
                )


render()
