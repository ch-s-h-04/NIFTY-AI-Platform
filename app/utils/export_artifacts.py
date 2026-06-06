"""
One-time export of Phase 3 artifacts for the Streamlit dashboard.

Run from the project root (not from the dashboard):

    python -m app.utils.export_artifacts
"""

from __future__ import annotations

import logging

import pandas as pd

from app.utils.paths import (
    FEATURE_IMPORTANCE_FILE,
    FOLD_METRICS_FILE,
    OOS_PREDICTIONS_FILE,
    SUMMARY_METRICS_FILE,
    ensure_project_on_path,
    outputs_dir,
)

ensure_project_on_path()

from src.models import (  # noqa: E402
    LGBM_PARAMS,
    TARGET_COLUMNS,
    build_modeling_dataset,
    prepare_model_matrix,
    run_walk_forward_cv,
)
import lightgbm as lgb  # noqa: E402

logger = logging.getLogger("NIFTY_AI_DashboardExport")

DASHBOARD_SYMBOLS = [
    "INFY",
    "TCS",
    "RELIANCE",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "ITC",
    "LT",
    "WIPRO",
    "BHARTIARTL",
    "AXISBANK",
    "KOTAKBANK",
    "HINDUNILVR",
    "SUNPHARMA",
    "MARUTI",
    "TATAMOTORS",
    "BAJFINANCE",
    "ONGC",
    "ADANIPORTS",
    "NTPC",
]


def _export_feature_importance(dataset: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Fit a single LightGBM on the last training window for importance display."""
    horizon = 21
    target_col = TARGET_COLUMNS[horizon]
    train_end = pd.Timestamp("2017-12-31")
    train_df = dataset.loc[dataset["Date"] <= train_end].dropna(subset=[target_col])
    if train_df.empty:
        return pd.DataFrame(columns=["feature", "importance"])

    X_train, names = prepare_model_matrix(train_df, feature_cols)
    valid = X_train.notna().all(axis=1)
    X_train = X_train.loc[valid]
    y_train = train_df.loc[valid, target_col].to_numpy()

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    imp = pd.DataFrame(
        {"feature": names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return imp


def export_all() -> None:
    out = outputs_dir()
    out.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger.info("Building modeling dataset for dashboard artifacts...")
    dataset, feature_cols = build_modeling_dataset(
        symbols=DASHBOARD_SYMBOLS,
        start_date="2010-01-01",
        end_date="2018-12-31",
        warmup_days=60,
    )

    logger.info("Running walk-forward CV (saved for dashboard; not run from UI)...")
    wf = run_walk_forward_cv(
        dataset=dataset,
        feature_cols=feature_cols,
        horizons=[21],
    )

    wf.oos_predictions.to_parquet(out / OOS_PREDICTIONS_FILE, index=False)
    wf.summary_metrics.to_parquet(out / SUMMARY_METRICS_FILE, index=False)
    wf.fold_metrics.to_parquet(out / FOLD_METRICS_FILE, index=False)

    logger.info("Exporting LightGBM feature importance...")
    importance = _export_feature_importance(dataset, feature_cols)
    importance.to_parquet(out / FEATURE_IMPORTANCE_FILE, index=False)

    logger.info("Dashboard artifacts written to %s", out)


if __name__ == "__main__":
    export_all()
