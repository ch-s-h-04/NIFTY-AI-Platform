"""Plotly helpers for SHAP dashboard visualizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.utils.theme import ACCENT, ACCENT_3, apply_figure_theme


def plot_shap_summary(
    summary: pd.DataFrame,
    importance: pd.DataFrame,
    top_n: int = 15,
) -> go.Figure:
    """Beeswarm-style SHAP summary for the top features by mean |SHAP|."""
    top_feats = importance.nlargest(top_n, "mean_abs_shap")["feature"].tolist()
    plot_df = summary.loc[summary["feature"].isin(top_feats)].copy()

    rng = np.random.default_rng(0)
    y_map = {feat: i for i, feat in enumerate(reversed(top_feats))}
    plot_df["y_jitter"] = (
        plot_df["feature"].astype(str).map(y_map).to_numpy()
        + rng.uniform(-0.25, 0.25, len(plot_df))
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["shap_value"],
            y=plot_df["y_jitter"],
            mode="markers",
            marker=dict(
                color=plot_df["feature_value"],
                colorscale="RdBu",
                size=6,
                opacity=0.65,
                colorbar=dict(title="Feature value"),
            ),
            text=plot_df["feature"],
            hovertemplate=(
                "Feature: %{text}<br>"
                "SHAP: %{x:.4f}<br>"
                "Value: %{marker.color:.4f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=f"SHAP Summary — Top {top_n} Features",
        xaxis_title="SHAP value (impact on 21-day return prediction)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(top_feats))),
            ticktext=top_feats,
            title="",
        ),
        height=max(420, top_n * 28),
    )
    return apply_figure_theme(fig)


def plot_shap_importance_bar(importance: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top = importance.nlargest(top_n, "mean_abs_shap").sort_values(
        "mean_abs_shap", ascending=True
    )
    fig = go.Figure(
        go.Bar(
            x=top["mean_abs_shap"],
            y=top["feature"],
            orientation="h",
            marker_color=ACCENT,
            name="Mean |SHAP|",
        )
    )
    fig.update_layout(
        title=f"SHAP Feature Importance — Top {top_n}",
        xaxis_title="Mean |SHAP|",
        height=max(420, top_n * 28),
    )
    return apply_figure_theme(fig)


def plot_signed_feature_bar(
    features: pd.DataFrame,
    value_col: str,
    title: str,
    color: str,
) -> go.Figure:
    ordered = features.sort_values(value_col, ascending=True)
    fig = go.Figure(
        go.Bar(
            x=ordered[value_col],
            y=ordered["feature"],
            orientation="h",
            marker_color=color,
        )
    )
    fig.update_layout(title=title, xaxis_title="Mean SHAP", height=max(320, len(ordered) * 28))
    return apply_figure_theme(fig)


def top_positive_features(importance: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    positive = importance.loc[importance["mean_shap"] > 0].copy()
    return positive.nlargest(n, "mean_shap")[["feature", "mean_shap", "mean_abs_shap"]]


def top_negative_features(importance: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    negative = importance.loc[importance["mean_shap"] < 0].copy()
    return negative.nsmallest(n, "mean_shap")[["feature", "mean_shap", "mean_abs_shap"]]
