"""Load precomputed Phase 3 artifacts and derive backtest views for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

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

from src.backtest import BacktestConfig, BacktestResult, run_backtest  # noqa: E402
from src.portfolio import PortfolioProfile  # noqa: E402
from src.risk import TRADING_DAYS_PER_YEAR, annualized_return  # noqa: E402

DEFAULT_START = "2010-01-01"
DEFAULT_END = "2018-12-31"

PROFILE_OPTIONS: Dict[str, PortfolioProfile] = {
    "Conservative": PortfolioProfile.CONSERVATIVE,
    "Balanced": PortfolioProfile.BALANCED,
    "Aggressive": PortfolioProfile.AGGRESSIVE,
}


@dataclass
class ArtifactStatus:
    """Which dashboard artifacts are present on disk."""

    oos_predictions: bool
    summary_metrics: bool
    fold_metrics: bool
    feature_importance: bool

    @property
    def any_missing(self) -> bool:
        return not all(
            [
                self.oos_predictions,
                self.summary_metrics,
                self.feature_importance,
            ]
        )


def artifact_status() -> ArtifactStatus:
    out = outputs_dir()
    return ArtifactStatus(
        oos_predictions=(out / OOS_PREDICTIONS_FILE).exists(),
        summary_metrics=(out / SUMMARY_METRICS_FILE).exists(),
        fold_metrics=(out / FOLD_METRICS_FILE).exists(),
        feature_importance=(out / FEATURE_IMPORTANCE_FILE).exists(),
    )


def missing_artifacts_message(status: Optional[ArtifactStatus] = None) -> str:
    status = status or artifact_status()
    missing: List[str] = []
    if not status.oos_predictions:
        missing.append(OOS_PREDICTIONS_FILE)
    if not status.summary_metrics:
        missing.append(SUMMARY_METRICS_FILE)
    if not status.feature_importance:
        missing.append(FEATURE_IMPORTANCE_FILE)
    if not missing:
        return ""
    return (
        "Missing dashboard artifacts in `outputs/`: "
        + ", ".join(missing)
        + ". Run `python -m app.utils.export_artifacts` from the project root "
        "to generate them (one-time; not triggered from the dashboard)."
    )


def _read_parquet(name: str) -> Optional[pd.DataFrame]:
    path = outputs_dir() / name
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def load_oos_predictions() -> Optional[pd.DataFrame]:
    return _read_parquet(OOS_PREDICTIONS_FILE)


def load_summary_metrics() -> Optional[pd.DataFrame]:
    return _read_parquet(SUMMARY_METRICS_FILE)


def load_fold_metrics() -> Optional[pd.DataFrame]:
    return _read_parquet(FOLD_METRICS_FILE)


def load_feature_importance() -> Optional[pd.DataFrame]:
    df = _read_parquet(FEATURE_IMPORTANCE_FILE)
    if df is None:
        return None
    if "importance" not in df.columns and "gain" in df.columns:
        df = df.rename(columns={"gain": "importance"})
    return df


def run_profile_backtest(
    profile: PortfolioProfile,
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
) -> Optional[BacktestResult]:
    """Simulate portfolio using saved OOS predictions (no model retraining)."""
    oos = load_oos_predictions()
    if oos is None or oos.empty:
        return None
    config = BacktestConfig(
        profile=profile,
        start_date=start_date,
        end_date=end_date,
    )
    return run_backtest(oos_predictions=oos, config=config)


def portfolio_cagr(result: BacktestResult) -> float:
    return annualized_return(result.portfolio_returns)


def benchmark_cagr(result: BacktestResult) -> float:
    return annualized_return(result.benchmark_returns.dropna())


def rolling_annualized_volatility(
    returns: pd.Series, window: int = 63
) -> pd.Series:
    r = returns.dropna()
    return r.rolling(window, min_periods=max(10, window // 3)).std(
        ddof=1
    ) * (TRADING_DAYS_PER_YEAR**0.5)
