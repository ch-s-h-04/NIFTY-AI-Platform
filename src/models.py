# src/models.py
"""
Prediction module for the NIFTY-50 AI Platform (Phase 3A).

Trains Ridge and LightGBM regressors on a long-format modeling dataset with
purged walk-forward cross-validation and out-of-sample prediction export.

Execution timing contract (for Phase 3B backtesting engine)
------------------------------------------------------------
Signal time:    market close on day t  — features at row t use all information
                available at the close of day t (no feature shifting).
Execution time: market open on day t+1 — simulated fills in the backtest engine
                must occur at the next session open, not at close t.

Model targets are close-to-close forward returns from P_t to P_{t+k}; the
execution lag is enforced only during portfolio simulation (Phase 3B).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.config import SECTOR_MAP, SYMBOLS
from src.features import build_feature_matrix

logger = logging.getLogger("NIFTY_AI_Models")

# ---------------------------------------------------------------------------
# Execution timing — canonical contract for the Phase 3B backtesting engine
# ---------------------------------------------------------------------------
SIGNAL_TIME: str = "market_close_day_t"
EXECUTION_TIME: str = "market_open_day_t_plus_1"

EXECUTION_TIMING: Dict[str, str] = {
    "signal_time": SIGNAL_TIME,
    "execution_time": EXECUTION_TIME,
    "feature_availability": "close_of_day_t",
    "feature_shift_applied": "false",
    "target_definition": "close_to_close_forward_return",
    "backtest_fill_price": "open_day_t_plus_1",
}

# ---------------------------------------------------------------------------
# Modeling configuration
# ---------------------------------------------------------------------------
HORIZONS: List[int] = [5, 21]
TRAIN_DAYS: int = 1260
TEST_DAYS: int = 252
PURGE_DAYS: int = 21
STEP_DAYS: int = 252
WARMUP_DAYS: int = 200

TARGET_COLUMNS: Dict[int, str] = {h: f"fwd_ret_{h}" for h in HORIZONS}

NON_FEATURE_COLUMNS: frozenset = frozenset(
    {
        "Date",
        "Symbol",
        "Series",
        "Sector",
        "fwd_ret_5",
        "fwd_ret_21",
    }
)

RIDGE_ALPHA: float = 1.0
LGBM_PARAMS: Dict = {
    "objective": "regression",
    "metric": "mae",
    "max_depth": 4,
    "min_data_in_leaf": 50,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": -1,
}


@dataclass
class FoldMetrics:
    """Per-fold evaluation metrics for one horizon and model."""

    fold: int
    horizon: int
    model: str
    mae: float
    rmse: float
    directional_accuracy: float
    n_train: int
    n_test: int


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward CV output."""

    oos_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def wide_to_long(wide_df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """
    Convert a wide feature matrix (Date index, {SYMBOL}_{feature} columns)
    into long format with one row per (Date, Symbol).
    """
    frames: List[pd.DataFrame] = []
    for symbol in symbols:
        prefix = f"{symbol}_"
        symbol_cols = [c for c in wide_df.columns if c.startswith(prefix)]
        if not symbol_cols:
            logger.warning("No wide columns found for symbol '%s'; skipping.", symbol)
            continue

        stock_df = wide_df[symbol_cols].copy()
        stock_df.columns = [c[len(prefix) :] for c in symbol_cols]
        stock_df["Symbol"] = symbol
        stock_df["Date"] = wide_df.index
        stock_df["Sector"] = SECTOR_MAP.get(symbol, "Unknown")
        frames.append(stock_df)

    if not frames:
        return pd.DataFrame()

    long_df = pd.concat(frames, ignore_index=True)
    long_df["Date"] = pd.to_datetime(long_df["Date"])
    return long_df.sort_values(["Symbol", "Date"]).reset_index(drop=True)


def add_forward_return_targets(
    long_df: pd.DataFrame, horizons: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Add close-to-close forward return targets for each horizon.

    Target at day t: (Close_{t+k} - Close_t) / Close_t
    Valid under signal-at-close-t assumption (no feature shift).
    """
    horizons = horizons or HORIZONS
    df = long_df.copy()
    if "Close" not in df.columns:
        raise ValueError("Long dataset must contain a 'Close' column for target generation.")

    for h in horizons:
        col = TARGET_COLUMNS[h]
        df[col] = df.groupby("Symbol", group_keys=False)["Close"].transform(
            lambda s: s.shift(-h) / s - 1.0
        )
    return df


def get_feature_columns(long_df: pd.DataFrame) -> List[str]:
    """Return numeric feature column names suitable for model training."""
    feature_cols: List[str] = []
    for col in long_df.columns:
        if col in NON_FEATURE_COLUMNS:
            continue
        if col.startswith("fwd_ret_"):
            continue
        if pd.api.types.is_numeric_dtype(long_df[col]):
            feature_cols.append(col)
    return sorted(feature_cols)


def prepare_model_matrix(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the feature matrix with one-hot encoded sector dummies.
    Returns (X, expanded_column_names).
    """
    X = df[feature_cols].copy()
    sector_dummies = pd.get_dummies(df["Sector"], prefix="sector", dtype=float)
    X = pd.concat([X, sector_dummies], axis=1)
    return X, list(X.columns)


def build_modeling_dataset(
    symbols: Optional[List[str]] = None,
    start_date: str = "2010-01-01",
    end_date: str = "2021-04-30",
    warmup_days: int = WARMUP_DAYS,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a long-format modeling dataset with features and forward-return targets.

    Args:
        symbols: Stock tickers to include (defaults to all discovered SYMBOLS).
        start_date: Feature window start (inclusive).
        end_date: Feature window end (inclusive).
        warmup_days: Rows dropped per symbol to allow indicator warm-up.

    Returns:
        (long_df, feature_cols) where long_df has Date, Symbol, Sector,
        numeric features, and fwd_ret_5 / fwd_ret_21 targets.
    """
    symbols = symbols or SYMBOLS
    logger.info(
        "Building modeling dataset for %d symbols (%s to %s)...",
        len(symbols),
        start_date,
        end_date,
    )

    wide_df = build_feature_matrix(symbols, start_date=start_date, end_date=end_date)
    long_df = wide_to_long(wide_df, symbols)
    if long_df.empty:
        raise ValueError("Long-format dataset is empty after wide-to-long conversion.")

    long_df = add_forward_return_targets(long_df)
    feature_cols = get_feature_columns(long_df)

    # Drop warm-up rows per symbol (rolling indicators need history)
    if warmup_days > 0:
        trimmed = [
            group.iloc[warmup_days:]
            for _, group in long_df.groupby("Symbol", sort=False)
        ]
        long_df = pd.concat(trimmed, ignore_index=True)

    target_cols = list(TARGET_COLUMNS.values())
    long_df = long_df.dropna(subset=target_cols)

    # Require core price features to be present
    required = [c for c in ("Close", "RSI_14") if c in feature_cols]
    if required:
        long_df = long_df.dropna(subset=required)

    logger.info(
        "Modeling dataset ready: %d rows, %d base features.",
        len(long_df),
        len(feature_cols),
    )
    return long_df, feature_cols


# ---------------------------------------------------------------------------
# Purged walk-forward cross-validation
# ---------------------------------------------------------------------------


def purged_walk_forward_splits(
    dates: pd.Series,
    train_days: int = TRAIN_DAYS,
    test_days: int = TEST_DAYS,
    purge_days: int = PURGE_DAYS,
    step_days: int = STEP_DAYS,
) -> Generator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex], None, None]:
    """
    Yield (train_dates, test_dates) index pairs with a purge gap between them.

    Purge gap length equals purge_days calendar trading days to prevent
    overlapping forward-return labels between train and test sets.
    """
    unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    n = len(unique_dates)
    start = 0

    while True:
        train_end_idx = start + train_days - 1
        test_start_idx = train_end_idx + purge_days + 1
        test_end_idx = test_start_idx + test_days - 1

        if test_end_idx >= n:
            break

        train_dates = unique_dates[start : train_end_idx + 1]
        test_dates = unique_dates[test_start_idx : test_end_idx + 1]
        yield train_dates, test_dates
        start += step_days


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of observations where predicted and actual returns share the same sign."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute MAE, RMSE, and directional accuracy."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_t = y_true[mask]
    y_p = y_pred[mask]
    if len(y_t) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "directional_accuracy": float("nan")}
    return {
        "mae": float(mean_absolute_error(y_t, y_p)),
        "rmse": float(np.sqrt(mean_squared_error(y_t, y_p))),
        "directional_accuracy": directional_accuracy(y_t, y_p),
    }


# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------


def _fit_ridge(
    X_train: pd.DataFrame, y_train: np.ndarray
) -> Tuple[Ridge, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X_scaled, y_train)
    return model, scaler


def _predict_ridge(
    model: Ridge, scaler: StandardScaler, X: pd.DataFrame
) -> np.ndarray:
    return model.predict(scaler.transform(X))


def _fit_lightgbm(X_train: pd.DataFrame, y_train: np.ndarray) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    return model


def _align_columns(X: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    """Ensure X has the same columns (order and names) as training data."""
    return X.reindex(columns=expected_cols, fill_value=0.0)


# ---------------------------------------------------------------------------
# Walk-forward training pipeline
# ---------------------------------------------------------------------------


def run_walk_forward_cv(
    dataset: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    symbols: Optional[List[str]] = None,
    start_date: str = "2010-01-01",
    end_date: str = "2021-04-30",
    horizons: Optional[List[int]] = None,
) -> WalkForwardResult:
    """
    Run purged walk-forward CV with Ridge and LightGBM for each horizon.

    Returns OOS predictions and per-fold / summary metrics.
    """
    horizons = horizons or HORIZONS

    if dataset is None or feature_cols is None:
        dataset, feature_cols = build_modeling_dataset(
            symbols=symbols, start_date=start_date, end_date=end_date
        )

    oos_frames: List[pd.DataFrame] = []
    fold_metric_rows: List[Dict] = []

    for fold_idx, (train_dates, test_dates) in enumerate(
        purged_walk_forward_splits(dataset["Date"])
    ):
        train_mask = dataset["Date"].isin(train_dates)
        test_mask = dataset["Date"].isin(test_dates)
        train_df = dataset.loc[train_mask]
        test_df = dataset.loc[test_mask]

        X_train, train_feature_names = prepare_model_matrix(train_df, feature_cols)
        X_test, _ = prepare_model_matrix(test_df, feature_cols)
        X_test = _align_columns(X_test, train_feature_names)

        # Drop rows with NaN features
        train_valid = X_train.notna().all(axis=1)
        test_valid = X_test.notna().all(axis=1)
        X_train = X_train.loc[train_valid]
        X_test = X_test.loc[test_valid]
        train_df = train_df.loc[train_valid]
        test_df = test_df.loc[test_valid]

        if len(X_train) == 0 or len(X_test) == 0:
            logger.warning("Fold %d skipped: empty train or test after NaN drop.", fold_idx)
            continue

        for horizon in horizons:
            target_col = TARGET_COLUMNS[horizon]
            y_train = train_df[target_col].to_numpy()
            y_test = test_df[target_col].to_numpy()

            # --- Ridge ---
            ridge_model, ridge_scaler = _fit_ridge(X_train, y_train)
            ridge_pred = _predict_ridge(ridge_model, ridge_scaler, X_test)
            ridge_metrics = compute_metrics(y_test, ridge_pred)

            fold_metric_rows.append(
                {
                    "fold": fold_idx,
                    "horizon": horizon,
                    "model": "ridge",
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    **ridge_metrics,
                }
            )

            # --- LightGBM ---
            lgbm_model = _fit_lightgbm(X_train, y_train)
            lgbm_pred = lgbm_model.predict(X_test)
            lgbm_metrics = compute_metrics(y_test, lgbm_pred)

            fold_metric_rows.append(
                {
                    "fold": fold_idx,
                    "horizon": horizon,
                    "model": "lightgbm",
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    **lgbm_metrics,
                }
            )

            fold_oos = test_df[["Date", "Symbol", "Sector", target_col]].copy()
            fold_oos = fold_oos.rename(columns={target_col: "actual"})
            fold_oos["horizon"] = horizon
            fold_oos["fold"] = fold_idx
            fold_oos["ridge_pred"] = ridge_pred
            fold_oos["lgbm_pred"] = lgbm_pred
            oos_frames.append(fold_oos)

        logger.info(
            "Fold %d complete: train=%d rows, test=%d rows.",
            fold_idx,
            len(X_train),
            len(X_test),
        )

    if not oos_frames:
        raise RuntimeError("Walk-forward CV produced no OOS predictions. Check date range and split parameters.")

    oos_predictions = pd.concat(oos_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)

    summary_metrics = (
        fold_metrics.groupby(["horizon", "model"])[["mae", "rmse", "directional_accuracy"]]
        .mean()
        .reset_index()
    )

    return WalkForwardResult(
        oos_predictions=oos_predictions,
        fold_metrics=fold_metrics,
        summary_metrics=summary_metrics,
    )


def get_backtest_execution_policy() -> Dict[str, str]:
    """
    Return the execution timing policy for the Phase 3B backtesting engine.

    Signal at close t; fills at open t+1. Features are not shifted.
    """
    return dict(EXECUTION_TIMING)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_phase3a(
    symbols: Optional[List[str]] = None,
    start_date: str = "2010-01-01",
    end_date: str = "2018-12-31",
) -> Dict[str, object]:
    """
    Run a lightweight end-to-end verification of Phase 3A components.

    Uses a reduced symbol set and shorter date range for speed.
    """
    symbols = symbols or ["INFY", "TCS", "RELIANCE"]
    results: Dict[str, object] = {"passed": True, "checks": []}

    def record(name: str, ok: bool, detail: str = "") -> None:
        results["checks"].append({"check": name, "passed": ok, "detail": detail})
        if not ok:
            results["passed"] = False

    # 1. Execution timing contract
    policy = get_backtest_execution_policy()
    record(
        "execution_timing_contract",
        policy["signal_time"] == SIGNAL_TIME
        and policy["execution_time"] == EXECUTION_TIME,
        str(policy),
    )

    # 2. Dataset builder
    try:
        dataset, feature_cols = build_modeling_dataset(
            symbols=symbols, start_date=start_date, end_date=end_date, warmup_days=60
        )
        record(
            "build_modeling_dataset",
            len(dataset) > 0 and len(feature_cols) > 0,
            f"rows={len(dataset)}, features={len(feature_cols)}",
        )
    except Exception as exc:
        record("build_modeling_dataset", False, str(exc))
        return results

    # 3. Targets
    for h in HORIZONS:
        col = TARGET_COLUMNS[h]
        has_target = col in dataset.columns and dataset[col].notna().any()
        record(f"target_{col}", has_target, f"non_null={dataset[col].notna().sum()}")

    # 4. Purged splits
    splits = list(
        purged_walk_forward_splits(
            dataset["Date"], train_days=504, test_days=126, purge_days=21, step_days=126
        )
    )
    record("purged_walk_forward_splits", len(splits) >= 1, f"n_folds={len(splits)}")

    if splits:
        train_d, test_d = splits[0]
        gap = (test_d.min() - train_d.max()).days
        record("purge_gap_positive", gap > 0, f"gap_days={gap}")

    # 5. Walk-forward CV (small)
    try:
        wf_result = run_walk_forward_cv(
            dataset=dataset,
            feature_cols=feature_cols,
            horizons=[5],
        )
        oos = wf_result.oos_predictions
        record(
            "walk_forward_cv",
            len(oos) > 0 and {"ridge_pred", "lgbm_pred", "actual"}.issubset(oos.columns),
            f"oos_rows={len(oos)}",
        )
        record(
            "summary_metrics",
            len(wf_result.summary_metrics) > 0,
            wf_result.summary_metrics.to_string(),
        )
    except Exception as exc:
        record("walk_forward_cv", False, str(exc))

    return results


def run_verification() -> bool:
    """CLI entry point for Phase 3A verification. Returns True if all checks pass."""
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("Phase 3A Verification")
    print("=" * 60)
    print(f"Execution policy: {get_backtest_execution_policy()}")
    print()

    results = verify_phase3a()
    for check in results["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] {check['check']}: {check['detail']}")

    print()
    if results["passed"]:
        print("ALL CHECKS PASSED")
    else:
        print("VERIFICATION FAILED")
    return bool(results["passed"])


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_verification() else 1)
