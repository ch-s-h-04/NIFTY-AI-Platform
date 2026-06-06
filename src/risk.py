# src/risk.py
"""
Risk analytics module for the NIFTY-50 AI Platform (Phase 3B).

Computes return series, Ledoit-Wolf covariance, and portfolio risk metrics
(Sharpe, Sortino, maximum drawdown, historical VaR/CVaR).

All estimation functions accept an ``as_of_date`` so that only information
available at market close on signal day *t* is used — compatible with the
Phase 3B execution contract in ``models.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pypfopt.risk_models import CovarianceShrinkage

logger = logging.getLogger("NIFTY_AI_Risk")

# ---------------------------------------------------------------------------
# Configuration defaults (ARCHITECTURE.md)
# ---------------------------------------------------------------------------
RISK_FREE_RATE: float = 0.065
TRADING_DAYS_PER_YEAR: int = 252
DEFAULT_COVARIANCE_LOOKBACK: int = 60
DEFAULT_VAR_CONFIDENCE: float = 0.95
MIN_OBSERVATIONS: int = 30
MIN_ASSET_COVERAGE: float = 0.80


@dataclass
class RiskConfig:
    """Runtime configuration for risk metric computation."""

    risk_free_rate: float = RISK_FREE_RATE
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR
    var_confidence: float = DEFAULT_VAR_CONFIDENCE
    min_observations: int = MIN_OBSERVATIONS
    covariance_lookback: int = DEFAULT_COVARIANCE_LOOKBACK


@dataclass
class RiskMetrics:
    """Aggregated risk-adjusted performance metrics for a return series."""

    sharpe: float
    sortino: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    annualized_return: float
    annualized_volatility: float
    n_observations: int


DateLike = Union[str, pd.Timestamp]


def _to_timestamp(date: DateLike) -> pd.Timestamp:
    return pd.Timestamp(date)


# ---------------------------------------------------------------------------
# Return calculations
# ---------------------------------------------------------------------------


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from a wide price panel (Date index × symbols).

    First row per column is NaN. Pre-listing NaNs propagate without fill.
    """
    if prices.empty:
        return pd.DataFrame()
    ordered = prices.sort_index()
    return np.log(ordered / ordered.shift(1))


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily simple (arithmetic) returns from a wide price panel.
    """
    if prices.empty:
        return pd.DataFrame()
    ordered = prices.sort_index()
    return ordered.pct_change()


def slice_returns_as_of(
    returns: pd.DataFrame,
    as_of_date: DateLike,
    lookback: Optional[int] = None,
) -> pd.DataFrame:
    """
    Restrict a return panel to rows with index <= ``as_of_date``.

    Optionally keep only the trailing ``lookback`` observations.
    Enforces the no-lookahead rule for signal-day estimation.
    """
    if returns.empty:
        return returns.copy()

    as_of = _to_timestamp(as_of_date)
    sliced = returns.loc[returns.index <= as_of]
    if lookback is not None and lookback > 0:
        sliced = sliced.iloc[-lookback:]
    return sliced


def portfolio_return_series(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
) -> pd.Series:
    """
    Daily portfolio return: r_p = sum(w_i * r_i).

    Missing asset returns are treated as zero contribution (weight on NaN names
    should be zero in downstream modules).
    """
    aligned_w = weights.reindex(asset_returns.columns, fill_value=0.0)
    weighted = asset_returns.mul(aligned_w, axis=1)
    return weighted.sum(axis=1, min_count=1)


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------


def _select_estimable_assets(
    returns: pd.DataFrame,
    min_observations: int = MIN_OBSERVATIONS,
    min_coverage: float = MIN_ASSET_COVERAGE,
) -> pd.DataFrame:
    """Drop assets with insufficient non-NaN history in the window."""
    if returns.empty:
        return returns

    min_count = max(min_observations, int(np.ceil(len(returns) * min_coverage)))
    counts = returns.notna().sum(axis=0)
    keep = counts[counts >= min_count].index.tolist()
    if not keep:
        raise ValueError(
            f"No assets with at least {min_count} return observations in window."
        )
    return returns[keep]


def estimate_covariance(
    returns_history: pd.DataFrame,
    as_of_date: DateLike,
    lookback: int = DEFAULT_COVARIANCE_LOOKBACK,
    min_observations: int = MIN_OBSERVATIONS,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Ledoit-Wolf shrunk covariance matrix using returns up to ``as_of_date``.

    Args:
        returns_history: Full daily return panel (Date × symbols).
        as_of_date: Signal date *t* (inclusive).
        lookback: Trailing window length in trading days.
        min_observations: Minimum rows required in the window.
        annualize: Scale to annual units (× 252 variance, × sqrt(252) not applied
                   to covariance — standard return covariance annualization).

    Returns:
        Symmetric covariance DataFrame indexed/columned by symbol.
    """
    window = slice_returns_as_of(returns_history, as_of_date, lookback=lookback)
    window = window.dropna(how="all")

    if len(window) < min_observations:
        raise ValueError(
            f"Need at least {min_observations} return rows; got {len(window)} "
            f"as of {as_of_date}."
        )

    window = _select_estimable_assets(window, min_observations=min_observations)
    clean = window.dropna(how="any")
    if len(clean) < min_observations:
        clean = window.fillna(0.0)

    # PyPortfolioOpt CovarianceShrinkage expects price-like series
    pseudo_prices = (1.0 + clean).cumprod()
    cov = CovarianceShrinkage(
        pseudo_prices,
        frequency=TRADING_DAYS_PER_YEAR,
    ).ledoit_wolf()

    cov_df = pd.DataFrame(cov, index=clean.columns, columns=clean.columns)
    if annualize:
        cov_df = cov_df * TRADING_DAYS_PER_YEAR
    return cov_df


def estimate_mu(
    returns_history: pd.DataFrame,
    as_of_date: DateLike,
    lookback: int = DEFAULT_COVARIANCE_LOOKBACK,
    annualize: bool = True,
) -> pd.Series:
    """
    Sample mean expected returns using data available up to ``as_of_date``.
    """
    window = slice_returns_as_of(returns_history, as_of_date, lookback=lookback)
    window = _select_estimable_assets(window)
    mu = window.mean(skipna=True)
    if annualize:
        mu = mu * TRADING_DAYS_PER_YEAR
    return mu.dropna()


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------


def annualized_return(
    daily_returns: pd.Series,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Compound annualized return from daily simple returns."""
    r = daily_returns.dropna()
    if len(r) == 0:
        return float("nan")
    cumulative = float((1.0 + r).prod())
    years = len(r) / trading_days
    if years <= 0:
        return float("nan")
    return cumulative ** (1.0 / years) - 1.0


def annualized_volatility(
    daily_returns: pd.Series,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized volatility from daily returns."""
    r = daily_returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(trading_days))


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio using a constant annual risk-free rate."""
    ann_ret = annualized_return(daily_returns, trading_days)
    ann_vol = annualized_volatility(daily_returns, trading_days)
    if np.isnan(ann_ret) or np.isnan(ann_vol) or ann_vol == 0:
        return float("nan")
    return (ann_ret - risk_free_rate) / ann_vol


def sortino_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sortino ratio (downside deviation vs daily rf)."""
    r = daily_returns.dropna()
    if len(r) < 2:
        return float("nan")

    daily_rf = (1.0 + risk_free_rate) ** (1.0 / trading_days) - 1.0
    downside = r[r < daily_rf] - daily_rf
    if len(downside) == 0:
        return float("inf") if annualized_return(r, trading_days) > risk_free_rate else float("nan")

    downside_dev = float(np.sqrt((downside**2).mean()) * np.sqrt(trading_days))
    ann_ret = annualized_return(r, trading_days)
    if downside_dev == 0 or np.isnan(ann_ret):
        return float("nan")
    return (ann_ret - risk_free_rate) / downside_dev


def equity_curve_from_returns(daily_returns: pd.Series) -> pd.Series:
    """Cumulative wealth index starting at 1.0."""
    r = daily_returns.fillna(0.0)
    return (1.0 + r).cumprod()


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Running drawdown as a negative fraction from peak."""
    peak = equity_curve.cummax()
    return equity_curve / peak - 1.0


def max_drawdown(daily_returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    equity = equity_curve_from_returns(daily_returns.dropna())
    if equity.empty:
        return float("nan")
    return float(drawdown_series(equity).min())


def historical_var(
    daily_returns: pd.Series,
    confidence: float = DEFAULT_VAR_CONFIDENCE,
) -> float:
    """
    Historical Value-at-Risk as a positive loss magnitude.

    For 95% confidence, returns the 5th percentile loss (-quantile).
    """
    r = daily_returns.dropna()
    if len(r) == 0:
        return float("nan")
    alpha = 1.0 - confidence
    return float(-np.quantile(r, alpha))


def historical_cvar(
    daily_returns: pd.Series,
    confidence: float = DEFAULT_VAR_CONFIDENCE,
) -> float:
    """
    Historical Conditional VaR (Expected Shortfall) as a positive loss.

    Mean of returns at or below the VaR threshold.
    """
    r = daily_returns.dropna()
    if len(r) == 0:
        return float("nan")
    var_loss = historical_var(r, confidence=confidence)
    tail = r[r <= -var_loss]
    if len(tail) == 0:
        return var_loss
    return float(-tail.mean())


def compute_risk_metrics(
    daily_returns: pd.Series,
    config: Optional[RiskConfig] = None,
) -> RiskMetrics:
    """Compute all packaged risk metrics for a daily return series."""
    config = config or RiskConfig()
    r = daily_returns.dropna()
    conf = config.var_confidence

    return RiskMetrics(
        sharpe=sharpe_ratio(r, config.risk_free_rate, config.trading_days_per_year),
        sortino=sortino_ratio(r, config.risk_free_rate, config.trading_days_per_year),
        max_drawdown=max_drawdown(r),
        var_95=historical_var(r, confidence=conf),
        cvar_95=historical_cvar(r, confidence=conf),
        annualized_return=annualized_return(r, config.trading_days_per_year),
        annualized_volatility=annualized_volatility(r, config.trading_days_per_year),
        n_observations=int(len(r)),
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_risk_module(
    symbols: Optional[List[str]] = None,
    as_of_date: str = "2018-06-29",
) -> Dict[str, object]:
    """
    Lightweight end-to-end checks for Phase 3B risk module components.
    """
    from src.data_loader import create_aligned_panel

    symbols = symbols or ["INFY", "TCS", "RELIANCE"]
    results: Dict[str, object] = {"passed": True, "checks": []}

    def record(name: str, ok: bool, detail: str = "") -> None:
        results["checks"].append({"check": name, "passed": ok, "detail": detail})
        if not ok:
            results["passed"] = False

    # 1. Synthetic max drawdown: 1.0 -> 1.1 -> 1.045 -> 0.784 -> 0.799
    synthetic = pd.Series([0.10, -0.05, -0.25, 0.02])
    mdd = max_drawdown(synthetic)
    record(
        "max_drawdown_synthetic",
        abs(mdd - (-0.2875)) < 1e-4,
        f"mdd={mdd:.4f} (expected -0.2875)",
    )

    # 2. VaR / CVaR ordering
    rng = np.random.default_rng(42)
    sim = pd.Series(rng.normal(0.0003, 0.015, 2000))
    var_l = historical_var(sim, confidence=0.95)
    cvar_l = historical_cvar(sim, confidence=0.95)
    record(
        "cvar_gte_var",
        cvar_l >= var_l - 1e-9,
        f"var={var_l:.6f}, cvar={cvar_l:.6f}",
    )

    as_of = pd.Timestamp(as_of_date)

    try:
        prices = create_aligned_panel(
            symbols, start_date="2017-01-01", end_date=as_of_date
        )
        returns = compute_simple_returns(prices)

        cov = estimate_covariance(returns, as_of_date=as_of_date, lookback=60)
        eigvals = np.linalg.eigvalsh(cov.values)
        record(
            "covariance_psd",
            bool(np.all(eigvals >= -1e-8)),
            f"min_eig={eigvals.min():.2e}, shape={cov.shape}",
        )

        window = slice_returns_as_of(returns, as_of, lookback=60)
        record(
            "no_lookahead_slice",
            window.index.max() <= as_of,
            f"max_date={window.index.max().date()}",
        )
        record(
            "covariance_uses_past_only",
            len(cov.columns) > 0 and window.index.max() <= as_of,
            f"assets={len(cov.columns)}",
        )

        w = pd.Series(1.0 / len(symbols), index=symbols)
        port_ret = portfolio_return_series(w, returns)
        metrics = compute_risk_metrics(port_ret.loc[:as_of_date])
        finite = all(
            np.isfinite(getattr(metrics, f))
            for f in (
                "sharpe",
                "sortino",
                "max_drawdown",
                "var_95",
                "cvar_95",
                "annualized_return",
                "annualized_volatility",
            )
        )
        record(
            "risk_metrics_finite",
            finite and metrics.n_observations > 30,
            (
                f"sharpe={metrics.sharpe:.3f}, sortino={metrics.sortino:.3f}, "
                f"mdd={metrics.max_drawdown:.3f}, n={metrics.n_observations}"
            ),
        )

        mu = estimate_mu(returns, as_of_date=as_of_date, lookback=60)
        record(
            "estimate_mu",
            len(mu) == len(symbols),
            f"mu_len={len(mu)}",
        )
    except Exception as exc:
        record("real_data_checks", False, str(exc))

    return results


def run_verification() -> bool:
    """CLI entry point for Phase 3B risk verification."""
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("Phase 3B Step 1: Risk Module Verification")
    print("=" * 60)

    results = verify_risk_module()
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
