"""
One-time SHAP artifact export for the LightGBM dashboard model.

Run from the project root (not from Streamlit):

    python -m app.utils.export_shap
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import shap

from app.utils.export_artifacts import DASHBOARD_SYMBOLS
from app.utils.paths import (
    SHAP_FEATURE_IMPORTANCE_FILE,
    SHAP_SUMMARY_FILE,
    ensure_project_on_path,
    outputs_dir,
)

ensure_project_on_path()

import lightgbm as lgb  # noqa: E402

from src.models import (  # noqa: E402
    LGBM_PARAMS,
    TARGET_COLUMNS,
    build_modeling_dataset,
    prepare_model_matrix,
)

logger = logging.getLogger("NIFTY_AI_SHAPExport")

HORIZON = 21
TRAIN_END = "2017-12-31"
SHAP_SAMPLE_SIZE = 500
SHAP_RANDOM_STATE = 42


@dataclass
class ShapExportResult:
    n_samples: int
    n_features: int
    elapsed_seconds: float
    summary_rows: int
    top_features: pd.DataFrame


def _fit_lgbm_snapshot(
    dataset: pd.DataFrame, feature_cols: list
) -> Tuple[lgb.LGBMRegressor, list[str], pd.DataFrame]:
    """Fit the same snapshot LightGBM used for dashboard feature importance."""
    target_col = TARGET_COLUMNS[HORIZON]
    train_end = pd.Timestamp(TRAIN_END)
    train_df = dataset.loc[dataset["Date"] <= train_end].dropna(subset=[target_col])
    if train_df.empty:
        raise RuntimeError("No training rows available for SHAP export.")

    X_train, names = prepare_model_matrix(train_df, feature_cols)
    valid = X_train.notna().all(axis=1)
    X_train = X_train.loc[valid]
    y_train = train_df.loc[valid, target_col].to_numpy()

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    return model, names, X_train


def _sample_explain_matrix(
    dataset: pd.DataFrame, feature_cols: list, names: list[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Draw a representative post-training sample for SHAP explanation."""
    target_col = TARGET_COLUMNS[HORIZON]
    train_end = pd.Timestamp(TRAIN_END)
    explain_df = dataset.loc[dataset["Date"] > train_end].dropna(subset=[target_col])
    if explain_df.empty:
        explain_df = dataset.dropna(subset=[target_col])

    X_all, _ = prepare_model_matrix(explain_df, feature_cols)
    X_all = X_all.reindex(columns=names, fill_value=0.0)
    valid = X_all.notna().all(axis=1)
    X_all = X_all.loc[valid]
    meta = explain_df.loc[valid, ["Date", "Symbol"]].reset_index(drop=True)
    X_all = X_all.reset_index(drop=True)

    n = min(SHAP_SAMPLE_SIZE, len(X_all))
    if n == 0:
        raise RuntimeError("No valid rows available for SHAP sampling.")

    rng = np.random.default_rng(SHAP_RANDOM_STATE)
    idx = rng.choice(len(X_all), size=n, replace=False)
    return X_all.iloc[idx].reset_index(drop=True), meta.iloc[idx].reset_index(drop=True)


def export_shap() -> ShapExportResult:
    """Compute and persist SHAP artifacts for the dashboard."""
    out = outputs_dir()
    out.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    t0 = time.perf_counter()

    logger.info("Building modeling dataset for SHAP export...")
    dataset, feature_cols = build_modeling_dataset(
        symbols=DASHBOARD_SYMBOLS,
        start_date="2010-01-01",
        end_date="2018-12-31",
        warmup_days=60,
    )

    model, names, _ = _fit_lgbm_snapshot(dataset, feature_cols)
    X_sample, meta = _sample_explain_matrix(dataset, feature_cols, names)

    logger.info("Computing SHAP values for %d samples...", len(X_sample))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)

    summary_rows = []
    for i in range(len(X_sample)):
        for j, feat in enumerate(names):
            summary_rows.append(
                {
                    "sample_index": i,
                    "feature": feat,
                    "shap_value": float(shap_values[i, j]),
                    "feature_value": float(X_sample.iloc[i, j]),
                    "Date": meta.iloc[i]["Date"],
                    "Symbol": meta.iloc[i]["Symbol"],
                }
            )
    shap_summary = pd.DataFrame(summary_rows)

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    shap_importance = (
        pd.DataFrame(
            {
                "feature": names,
                "mean_abs_shap": mean_abs,
                "mean_shap": mean_shap,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    shap_summary.to_parquet(out / SHAP_SUMMARY_FILE, index=False)
    shap_importance.to_parquet(out / SHAP_FEATURE_IMPORTANCE_FILE, index=False)

    elapsed = time.perf_counter() - t0
    top = shap_importance.head(10)
    logger.info("SHAP artifacts written to %s in %.2fs", out, elapsed)

    return ShapExportResult(
        n_samples=len(X_sample),
        n_features=len(names),
        elapsed_seconds=elapsed,
        summary_rows=len(shap_summary),
        top_features=top,
    )


if __name__ == "__main__":
    result = export_shap()
    print(f"n_samples={result.n_samples}")
    print(f"n_features={result.n_features}")
    print(f"elapsed_seconds={result.elapsed_seconds:.2f}")
    print(f"summary_rows={result.summary_rows}")
    print("top_10_features:")
    print(result.top_features.to_string(index=False))
