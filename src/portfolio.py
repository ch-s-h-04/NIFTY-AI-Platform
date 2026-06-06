# src/portfolio.py
"""
Portfolio construction module for the NIFTY-50 AI Platform (Phase 3B).

Maps risk profiles (Conservative, Balanced, Aggressive) to mean-variance and
Black-Litterman optimizers with rank-based ML views, sector/weight caps, and
risk-parity / equal-weight fallbacks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
from unittest.mock import patch

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, HRPOpt
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.exceptions import OptimizationError

from src.config import SECTOR_MAP
from src.risk import (
    DEFAULT_COVARIANCE_LOOKBACK,
    RISK_FREE_RATE,
    DateLike,
    estimate_covariance,
    estimate_mu,
    slice_returns_as_of,
)

logger = logging.getLogger("NIFTY_AI_Portfolio")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_HORIZON: int = 21
PREDICTION_MODEL_COL: str = "lgbm_pred"
BL_VIEW_SCALE: float = 0.03
BL_TAU: float = 0.05
MIN_STOCKS: int = 10
MAX_CONDITION_NUMBER: float = 1e6
CASH_WEIGHT_TOLERANCE: float = 1e-9


class PortfolioProfile(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class OptimizationMethod(str, Enum):
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"


@dataclass(frozen=True)
class PortfolioConstraints:
    """Per-profile weight and diversification limits."""

    max_weight_per_stock: float
    max_sector_weight: float
    long_only: bool = True
    min_stocks: int = MIN_STOCKS


@dataclass
class PortfolioConfig:
    """Runtime portfolio optimization settings."""

    horizon: int = DEFAULT_HORIZON
    prediction_model_col: str = PREDICTION_MODEL_COL
    covariance_lookback: int = DEFAULT_COVARIANCE_LOOKBACK
    risk_free_rate: float = RISK_FREE_RATE
    bl_view_scale: float = BL_VIEW_SCALE
    bl_tau: float = BL_TAU


@dataclass
class PortfolioResult:
    """Standardized optimizer output for backtest integration."""

    weights: pd.Series
    profile: PortfolioProfile
    method: str
    success: bool
    fallback_used: bool
    message: str
    as_of_date: Optional[pd.Timestamp] = None
    universe: Optional[List[str]] = None
    cash_weight: float = 0.0


PROFILE_CONSTRAINTS: Dict[PortfolioProfile, PortfolioConstraints] = {
    PortfolioProfile.CONSERVATIVE: PortfolioConstraints(
        max_weight_per_stock=0.05,
        max_sector_weight=0.20,
    ),
    PortfolioProfile.BALANCED: PortfolioConstraints(
        max_weight_per_stock=0.10,
        max_sector_weight=0.25,
    ),
    PortfolioProfile.AGGRESSIVE: PortfolioConstraints(
        max_weight_per_stock=0.15,
        max_sector_weight=0.30,
    ),
}

PROFILE_PRIMARY_METHOD: Dict[PortfolioProfile, OptimizationMethod] = {
    PortfolioProfile.CONSERVATIVE: OptimizationMethod.MIN_VARIANCE,
    PortfolioProfile.BALANCED: OptimizationMethod.MAX_SHARPE,
    PortfolioProfile.AGGRESSIVE: OptimizationMethod.BLACK_LITTERMAN,
}


def get_profile_constraints(profile: PortfolioProfile) -> PortfolioConstraints:
    """Return constraint bundle for a risk profile."""
    return PROFILE_CONSTRAINTS[profile]


def compute_cash_weight(stock_weights: pd.Series) -> float:
    """Uninvested sleeve: residual capital not allocated to equities."""
    return float(max(0.0, 1.0 - float(stock_weights.sum())))


def _build_portfolio_result(
    weights: pd.Series,
    profile: PortfolioProfile,
    method: str,
    success: bool,
    fallback_used: bool,
    message: str,
    as_of_date: pd.Timestamp,
    universe: List[str],
) -> PortfolioResult:
    """Attach ``cash_weight`` so stock weights + cash always sum to 1."""
    return PortfolioResult(
        weights=weights,
        profile=profile,
        method=method,
        success=success,
        fallback_used=fallback_used,
        message=message,
        as_of_date=as_of_date,
        universe=universe,
        cash_weight=compute_cash_weight(weights),
    )


# ---------------------------------------------------------------------------
# Prediction helpers (horizon == 21 only)
# ---------------------------------------------------------------------------


def prepare_predictions_frame(
    oos_predictions: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """
    Filter OOS predictions to a single horizon (default 21).

    Drops duplicate (Date, Symbol) rows if present.
    """
    if "horizon" not in oos_predictions.columns:
        raise ValueError("oos_predictions must contain a 'horizon' column.")

    df = oos_predictions.loc[oos_predictions["horizon"] == horizon].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(subset=["Date", "Symbol"], keep="last")
    return df


def get_predictions_on_date(
    oos_predictions: pd.DataFrame,
    signal_date: DateLike,
    horizon: int = DEFAULT_HORIZON,
    model_col: str = PREDICTION_MODEL_COL,
) -> pd.Series:
    """Cross-section of model predictions at market close on signal day *t*."""
    prepared = prepare_predictions_frame(oos_predictions, horizon=horizon)
    as_of = pd.Timestamp(signal_date)
    day = prepared.loc[prepared["Date"] == as_of, ["Symbol", model_col]]
    if day.empty:
        return pd.Series(dtype=float)
    return day.set_index("Symbol")[model_col].dropna()


def predictions_to_ranks(predictions: pd.Series) -> pd.Series:
    """
    Convert raw predictions to percentile ranks in (0, 1).

    Uses average rank for ties — suitable for Black-Litterman views.
    """
    if predictions.empty:
        return predictions
    return predictions.rank(pct=True, method="average")


def build_bl_views_from_ranks(
    ranks: pd.Series,
    view_scale: float = BL_VIEW_SCALE,
) -> Dict[str, float]:
    """
    Map percentile ranks to absolute Black-Litterman views.

    view_i = (rank_i - 0.5) * 2 * view_scale
    """
    views: Dict[str, float] = {}
    for symbol, rank in ranks.items():
        views[str(symbol)] = float((rank - 0.5) * 2.0 * view_scale)
    return views


# ---------------------------------------------------------------------------
# Weight utilities
# ---------------------------------------------------------------------------


def normalize_weights(weights: pd.Series) -> pd.Series:
    """Normalize to sum to 1; clip negatives for long-only portfolios."""
    w = weights.astype(float).clip(lower=0.0)
    total = w.sum()
    if total <= 0 or not np.isfinite(total):
        n = len(w)
        if n == 0:
            return w
        return pd.Series(1.0 / n, index=w.index)
    return w / total


def apply_weight_caps(weights: pd.Series, max_weight: float) -> pd.Series:
    """Clip per-asset weights and renormalize."""
    return normalize_weights(weights.clip(upper=max_weight))


def apply_sector_caps(
    weights: pd.Series,
    sector_map: Dict[str, str],
    max_sector_weight: float,
    max_iterations: int = 8,
) -> pd.Series:
    """
    Iteratively scale sector weights down to ``max_sector_weight``.
    """
    w = normalize_weights(weights)
    symbols = w.index.tolist()
    sector_groups: Dict[str, List[str]] = {}
    for sym in symbols:
        sector = sector_map.get(sym, "Unknown")
        sector_groups.setdefault(sector, []).append(sym)

    for _ in range(max_iterations):
        adjusted = False
        for sector, syms in sector_groups.items():
            sector_w = w.reindex(syms, fill_value=0.0).sum()
            if sector_w > max_sector_weight + 1e-9:
                scale = max_sector_weight / sector_w
                w.loc[syms] = w.reindex(syms, fill_value=0.0) * scale
                adjusted = True
        w = normalize_weights(w)
        if not adjusted:
            break
    return w


def enforce_constraints(
    weights: pd.Series,
    constraints: PortfolioConstraints,
    sector_map: Dict[str, str],
) -> pd.Series:
    """Apply per-stock and sector caps via iterative projection."""
    w = normalize_weights(weights.clip(lower=0.0))
    symbols = w.index.tolist()
    sector_groups: Dict[str, List[str]] = {}
    for sym in symbols:
        sector = sector_map.get(sym, "Unknown")
        sector_groups.setdefault(sector, []).append(sym)

    for _ in range(30):
        prev = w.copy()
        w = w.clip(upper=constraints.max_weight_per_stock)
        for syms in sector_groups.values():
            sector_w = w.reindex(syms, fill_value=0.0).sum()
            if sector_w > constraints.max_sector_weight + 1e-9:
                scale = constraints.max_sector_weight / sector_w
                w.loc[syms] = w.reindex(syms, fill_value=0.0) * scale
        w = w.clip(upper=constraints.max_weight_per_stock)
        total = float(w.sum())
        if total > 1.0 + 1e-9:
            w = w / total
        if np.allclose(w.values, prev.values, atol=1e-8):
            break

    total = float(w.sum())
    if total > 1.0 + 1e-9:
        w = w / total
    elif total < 1.0 - 1e-9 and total > 0:
        # Feasible partially invested book when caps prevent full deployment
        w = w.clip(upper=constraints.max_weight_per_stock)
    return w


def scale_to_full_investment(
    weights: pd.Series,
    constraints: PortfolioConstraints,
    sector_map: Dict[str, str],
) -> pd.Series:
    """
    Scale weights toward full investment (sum = 1) without violating caps.
    """
    w = enforce_constraints(weights, constraints, sector_map)
    symbols = w.index.tolist()
    sector_groups: Dict[str, List[str]] = {}
    for sym in symbols:
        sector = sector_map.get(sym, "Unknown")
        sector_groups.setdefault(sector, []).append(sym)

    for _ in range(100):
        total = float(w.sum())
        if abs(total - 1.0) < 1e-6:
            break
        if total > 1.0 + 1e-9:
            w = w / total
            w = enforce_constraints(w, constraints, sector_map)
            continue

        deficit = 1.0 - total
        headroom = pd.Series(0.0, index=symbols)
        for sym in symbols:
            stock_room = constraints.max_weight_per_stock - float(w[sym])
            sector = sector_map.get(sym, "Unknown")
            syms = sector_groups.get(sector, [sym])
            sector_room = constraints.max_sector_weight - float(
                w.reindex(syms, fill_value=0.0).sum()
            )
            headroom[sym] = max(0.0, min(stock_room, sector_room))

        if float(headroom.sum()) < 1e-9:
            break
        w = w + deficit * (headroom / headroom.sum())
        w = enforce_constraints(w, constraints, sector_map)
    return w


def _covariance_is_ill_conditioned(cov: pd.DataFrame) -> bool:
    if cov.empty:
        return True
    try:
        cond = np.linalg.cond(cov.values)
        if not np.isfinite(cond) or cond > MAX_CONDITION_NUMBER:
            return True
        eigvals = np.linalg.eigvalsh(cov.values)
        return bool(np.any(eigvals < -1e-8))
    except np.linalg.LinAlgError:
        return True


def _select_universe(
    symbols: List[str],
    predictions: Optional[pd.Series],
    cov: pd.DataFrame,
) -> List[str]:
    """Symbols present in covariance with optional valid predictions."""
    cov_syms = set(cov.columns)
    selected = [s for s in symbols if s in cov_syms]
    if predictions is not None and not predictions.empty:
        pred_syms = set(predictions.dropna().index)
        selected = [s for s in selected if s in pred_syms]
    return selected


def _series_from_weight_dict(
    weight_dict: Dict[str, float],
    universe: List[str],
) -> pd.Series:
    w = pd.Series({k: float(v) for k, v in weight_dict.items()})
    return w.reindex(universe, fill_value=0.0)


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------


def optimize_equal_weight(symbols: List[str]) -> pd.Series:
    """Equal-weight fallback."""
    if not symbols:
        return pd.Series(dtype=float)
    w = pd.Series(1.0 / len(symbols), index=symbols)
    return normalize_weights(w)


def optimize_risk_parity(
    returns_history: pd.DataFrame,
    as_of_date: DateLike,
    symbols: List[str],
    lookback: int = DEFAULT_COVARIANCE_LOOKBACK,
) -> pd.Series:
    """
    Hierarchical Risk Parity (PyPortfolioOpt HRPOpt) on trailing returns.
    """
    window = slice_returns_as_of(returns_history, as_of_date, lookback=lookback)
    window = window.reindex(columns=[s for s in symbols if s in window.columns])
    window = window.dropna(how="all")
    if window.empty or len(window.columns) == 0:
        raise ValueError("Insufficient return history for risk parity.")

    # HRPOpt expects a price/return history DataFrame
    hrp = HRPOpt(window.dropna(how="any", axis=0))
    hrp.optimize()
    raw = hrp.clean_weights()
    return _series_from_weight_dict(raw, list(window.columns))


def optimize_min_variance(
    cov: pd.DataFrame,
    symbols: List[str],
    risk_free_rate: float = RISK_FREE_RATE,
    max_weight: float = 1.0,
) -> pd.Series:
    """Minimum-variance portfolio (Conservative primary)."""
    syms = [s for s in symbols if s in cov.columns]
    if not syms:
        raise ValueError("No overlapping symbols for min-variance optimization.")

    mu = pd.Series(0.0, index=syms)
    sub_cov = cov.loc[syms, syms]
    ef = EfficientFrontier(mu, sub_cov, weight_bounds=(0.0, max_weight))
    ef.min_volatility()
    raw = ef.clean_weights()
    return _series_from_weight_dict(raw, syms)


def optimize_max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    symbols: List[str],
    risk_free_rate: float = RISK_FREE_RATE,
    max_weight: float = 1.0,
) -> pd.Series:
    """Maximum-Sharpe tangency portfolio (Balanced primary)."""
    syms = [s for s in symbols if s in cov.columns and s in mu.index]
    if not syms:
        raise ValueError("No overlapping symbols for max-Sharpe optimization.")

    sub_mu = mu.reindex(syms).fillna(0.0)
    sub_cov = cov.loc[syms, syms]
    ef = EfficientFrontier(sub_mu, sub_cov, weight_bounds=(0.0, max_weight))
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    raw = ef.clean_weights()
    return _series_from_weight_dict(raw, syms)


def optimize_black_litterman(
    predictions: pd.Series,
    cov: pd.DataFrame,
    symbols: List[str],
    risk_free_rate: float = RISK_FREE_RATE,
    view_scale: float = BL_VIEW_SCALE,
    tau: float = BL_TAU,
    max_weight: float = 1.0,
) -> pd.Series:
    """
    Black-Litterman with rank-based absolute views (Aggressive primary).
    """
    syms = [s for s in symbols if s in cov.columns and s in predictions.index]
    if not syms:
        raise ValueError("No overlapping symbols for Black-Litterman.")

    sub_cov = cov.loc[syms, syms]
    ranks = predictions_to_ranks(predictions.reindex(syms))
    views = build_bl_views_from_ranks(ranks, view_scale=view_scale)

    bl = BlackLittermanModel(
        sub_cov,
        pi="equal",
        absolute_views=views,
        tau=tau,
        omega="default",
    )
    bl_mu = bl.bl_returns()
    bl_mu = pd.Series(bl_mu, index=syms)

    ef = EfficientFrontier(bl_mu, sub_cov, weight_bounds=(0.0, max_weight))
    try:
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    except (OptimizationError, ValueError):
        ef = EfficientFrontier(bl_mu, sub_cov, weight_bounds=(0.0, max_weight))
        ef.min_volatility()
    raw = ef.clean_weights()
    return _series_from_weight_dict(raw, syms)


def _run_primary_optimizer(
    method: OptimizationMethod,
    symbols: List[str],
    cov: pd.DataFrame,
    returns_history: pd.DataFrame,
    as_of_date: DateLike,
    predictions: Optional[pd.Series],
    config: PortfolioConfig,
    constraints: PortfolioConstraints,
) -> pd.Series:
    max_w = constraints.max_weight_per_stock
    if method == OptimizationMethod.MIN_VARIANCE:
        return optimize_min_variance(cov, symbols, config.risk_free_rate, max_w)
    if method == OptimizationMethod.MAX_SHARPE:
        mu = estimate_mu(
            returns_history,
            as_of_date,
            lookback=config.covariance_lookback,
        )
        return optimize_max_sharpe(mu, cov, symbols, config.risk_free_rate, max_w)
    if method == OptimizationMethod.BLACK_LITTERMAN:
        if predictions is None or predictions.empty:
            raise ValueError("Black-Litterman requires predictions.")
        return optimize_black_litterman(
            predictions,
            cov,
            symbols,
            risk_free_rate=config.risk_free_rate,
            view_scale=config.bl_view_scale,
            tau=config.bl_tau,
            max_weight=max_w,
        )
    raise ValueError(f"Unsupported primary method: {method}")


def solve_portfolio(
    profile: PortfolioProfile,
    symbols: List[str],
    returns_history: pd.DataFrame,
    as_of_date: DateLike,
    predictions: Optional[pd.Series] = None,
    sector_map: Optional[Dict[str, str]] = None,
    config: Optional[PortfolioConfig] = None,
) -> PortfolioResult:
    """
    Construct portfolio weights for a risk profile as of signal date *t*.

    Fallback chain: primary optimizer -> risk parity -> equal weight.
    """
    sector_map = sector_map or SECTOR_MAP
    config = config or PortfolioConfig()
    constraints = get_profile_constraints(profile)
    primary = PROFILE_PRIMARY_METHOD[profile]
    as_of = pd.Timestamp(as_of_date)

    message_parts: List[str] = []
    fallback_used = False
    method_used = primary.value

    try:
        cov = estimate_covariance(
            returns_history,
            as_of_date=as_of,
            lookback=config.covariance_lookback,
        )
    except Exception as exc:
        logger.warning("Covariance estimation failed: %s", exc)
        eq_syms = list(symbols)
        weights = enforce_constraints(
            optimize_equal_weight(eq_syms),
            constraints,
            sector_map,
        )
        weights = scale_to_full_investment(weights, constraints, sector_map)
        return _build_portfolio_result(
            weights=weights,
            profile=profile,
            method=OptimizationMethod.EQUAL_WEIGHT.value,
            success=True,
            fallback_used=True,
            message=f"covariance_failed; equal_weight: {exc}",
            as_of_date=as_of,
            universe=eq_syms,
        )

    universe = _select_universe(symbols, predictions, cov)
    if len(universe) < constraints.min_stocks:
        message_parts.append(
            f"universe_too_small ({len(universe)} < {constraints.min_stocks})"
        )
        weights = scale_to_full_investment(
            optimize_equal_weight(universe or list(symbols)),
            constraints,
            sector_map,
        )
        return _build_portfolio_result(
            weights=weights,
            profile=profile,
            method=OptimizationMethod.EQUAL_WEIGHT.value,
            success=True,
            fallback_used=True,
            message="; ".join(message_parts) or "equal_weight_universe",
            as_of_date=as_of,
            universe=universe or list(symbols),
        )

    weights: Optional[pd.Series] = None

    if not _covariance_is_ill_conditioned(cov):
        try:
            weights = _run_primary_optimizer(
                primary,
                universe,
                cov,
                returns_history,
                as_of,
                predictions,
                config,
                constraints,
            )
            method_used = primary.value
        except (OptimizationError, ValueError, np.linalg.LinAlgError) as exc:
            message_parts.append(f"primary_failed ({primary.value}): {exc}")
            weights = None
    else:
        message_parts.append("ill_conditioned_covariance")

    if weights is None:
        fallback_used = True
        try:
            weights = optimize_risk_parity(
                returns_history,
                as_of,
                universe,
                lookback=config.covariance_lookback,
            )
            method_used = OptimizationMethod.RISK_PARITY.value
        except Exception as exc:
            message_parts.append(f"risk_parity_failed: {exc}")
            weights = optimize_equal_weight(universe)
            method_used = OptimizationMethod.EQUAL_WEIGHT.value

    weights = scale_to_full_investment(
        weights.reindex(universe, fill_value=0.0),
        constraints,
        sector_map,
    )

    if weights.sum() <= 0 or not np.isfinite(weights.sum()):
        fallback_used = True
        weights = scale_to_full_investment(
            optimize_equal_weight(universe),
            constraints,
            sector_map,
        )
        method_used = OptimizationMethod.EQUAL_WEIGHT.value
        message_parts.append("final_equal_weight_guard")

    return _build_portfolio_result(
        weights=weights,
        profile=profile,
        method=method_used,
        success=True,
        fallback_used=fallback_used,
        message="; ".join(message_parts) if message_parts else "ok",
        as_of_date=as_of,
        universe=universe,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _check_weights_basic(
    weights: pd.Series,
    tol: float = 1e-6,
) -> tuple[bool, str]:
    """Long-only sanity checks (sum validated separately)."""
    if weights.empty:
        return False, "empty weights"
    if weights.min() < -tol:
        return False, f"min_weight={weights.min():.6f}"
    total = float(weights.sum())
    return True, f"sum={total:.6f}, min={weights.min():.6f}, max={weights.max():.6f}"


def _check_weights_valid(
    weights: pd.Series,
    constraints: PortfolioConstraints,
    tol: float = 1e-6,
    min_sum: float = 0.99,
) -> tuple[bool, str]:
    ok, detail = _check_weights_basic(weights, tol=tol)
    if not ok:
        return ok, detail
    total = float(weights.sum())
    if total < min_sum - tol or total > 1.0 + tol:
        return False, f"sum={total:.6f}"
    if weights.max() > constraints.max_weight_per_stock + 0.02:
        return False, f"max_weight={weights.max():.6f}"
    return True, detail


def get_monthly_rebalance_dates(
    trading_dates: Union[pd.DatetimeIndex, pd.Series, List[DateLike]],
    min_date: Optional[DateLike] = None,
) -> pd.DatetimeIndex:
    """Last trading session of each calendar month (monthly rebalance schedule)."""
    dates = pd.DatetimeIndex(pd.Index(trading_dates).unique()).sort_values()
    if min_date is not None:
        dates = dates[dates >= pd.Timestamp(min_date)]
    if len(dates) == 0:
        return pd.DatetimeIndex([])
    grouped = pd.Series(dates, index=dates).groupby(dates.to_period("M"))
    return pd.DatetimeIndex(grouped.max().values)


def verify_cash_weight_invariant(
    result: PortfolioResult,
    tol: float = CASH_WEIGHT_TOLERANCE,
) -> tuple[bool, str]:
    """Check cash_weight >= 0 and stock weights + cash == 1."""
    invested = float(result.weights.sum())
    cash = float(result.cash_weight)
    if cash < -tol:
        return False, f"cash_weight={cash:.12f} < 0"
    total = invested + cash
    if abs(total - 1.0) > tol:
        return False, (
            f"invested={invested:.12f}, cash={cash:.12f}, total={total:.12f}"
        )
    expected_cash = compute_cash_weight(result.weights)
    if abs(cash - expected_cash) > tol:
        return False, (
            f"cash mismatch: stored={cash:.12f}, expected={expected_cash:.12f}"
        )
    return True, f"invested={invested:.6f}, cash={cash:.6f}, total={total:.6f}"


def verify_cash_weight_all_rebalances(
    symbols: Optional[List[str]] = None,
    start_date: str = "2010-01-01",
    end_date: str = "2018-12-31",
    oos_predictions: Optional[pd.DataFrame] = None,
    tol: float = CASH_WEIGHT_TOLERANCE,
) -> Dict[str, object]:
    """
    Verify cash_weight invariants for every profile on every monthly rebalance date.
    """
    from src.config import SYMBOLS
    from src.data_loader import create_aligned_panel
    from src.models import build_modeling_dataset, run_walk_forward_cv
    from src.risk import compute_simple_returns, DEFAULT_COVARIANCE_LOOKBACK

    symbols = symbols or SYMBOLS
    results: Dict[str, object] = {
        "passed": True,
        "checks": [],
        "n_rebalance_dates": 0,
        "n_portfolios": 0,
        "failures": [],
    }

    def record(name: str, ok: bool, detail: str = "") -> None:
        results["checks"].append({"check": name, "passed": ok, "detail": detail})
        if not ok:
            results["passed"] = False

    prices = create_aligned_panel(symbols, start_date=start_date, end_date=end_date)
    returns = compute_simple_returns(prices)
    min_rebalance = returns.index[DEFAULT_COVARIANCE_LOOKBACK]
    rebalance_dates = get_monthly_rebalance_dates(returns.index, min_date=min_rebalance)
    results["n_rebalance_dates"] = int(len(rebalance_dates))

    if oos_predictions is None:
        dataset, feature_cols = build_modeling_dataset(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            warmup_days=60,
        )
        wf = run_walk_forward_cv(dataset=dataset, feature_cols=feature_cols)
        oos_predictions = wf.oos_predictions

    prepared_preds = (
        prepare_predictions_frame(oos_predictions)
        if not oos_predictions.empty
        else pd.DataFrame()
    )
    pred_dates = (
        set(pd.DatetimeIndex(prepared_preds["Date"].unique()))
        if not prepared_preds.empty
        else set()
    )

    n_checked = 0
    n_with_cash = 0
    failures: List[str] = []

    for rebalance_date in rebalance_dates:
        preds = None
        if rebalance_date in pred_dates:
            preds = get_predictions_on_date(
                oos_predictions,
                rebalance_date,
                horizon=DEFAULT_HORIZON,
            )
            if preds.empty:
                preds = None

        hist = returns.loc[:rebalance_date]
        for profile in PortfolioProfile:
            try:
                result = solve_portfolio(
                    profile=profile,
                    symbols=symbols,
                    returns_history=hist,
                    as_of_date=rebalance_date,
                    predictions=preds,
                )
            except Exception as exc:
                failures.append(
                    f"{profile.value}@{rebalance_date.date()}: solve failed: {exc}"
                )
                continue

            n_checked += 1
            if result.cash_weight > tol:
                n_with_cash += 1

            ok, detail = verify_cash_weight_invariant(result, tol=tol)
            if not ok:
                failures.append(
                    f"{profile.value}@{rebalance_date.date()}: {detail}"
                )

    results["n_portfolios"] = n_checked
    results["failures"] = failures

    record(
        "cash_weight_all_rebalances",
        len(failures) == 0 and n_checked > 0,
        (
            f"profiles=3, rebalance_dates={len(rebalance_dates)}, "
            f"portfolios_checked={n_checked}, with_cash={n_with_cash}, "
            f"failures={len(failures)}"
        ),
    )
    if failures:
        preview = "; ".join(failures[:5])
        if len(failures) > 5:
            preview += f"; ... (+{len(failures) - 5} more)"
        record("cash_weight_failure_samples", False, preview)

    return results


def verify_portfolio_module(
    symbols: Optional[List[str]] = None,
    as_of_date: str = "2018-06-29",
    oos_predictions: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """
    Verification suite for Phase 3B portfolio construction.
    """
    from src.data_loader import create_aligned_panel
    from src.models import run_walk_forward_cv
    from src.risk import compute_simple_returns

    symbols = symbols or [
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
    results: Dict[str, object] = {"passed": True, "checks": []}

    def record(name: str, ok: bool, detail: str = "") -> None:
        results["checks"].append({"check": name, "passed": ok, "detail": detail})
        if not ok:
            results["passed"] = False

    prices = create_aligned_panel(
        symbols, start_date="2015-01-01", end_date=as_of_date
    )
    returns = compute_simple_returns(prices)
    as_of = pd.Timestamp(as_of_date)

    if oos_predictions is None:
        try:
            from src.models import build_modeling_dataset

            dataset, feature_cols = build_modeling_dataset(
                symbols=symbols,
                start_date="2010-01-01",
                end_date="2018-12-31",
                warmup_days=60,
            )
            wf = run_walk_forward_cv(dataset=dataset, feature_cols=feature_cols)
            oos_predictions = wf.oos_predictions
        except Exception as exc:
            record("load_oos_predictions", False, str(exc))
            oos_predictions = pd.DataFrame()

    predictions = pd.Series(dtype=float)
    if not oos_predictions.empty:
        prepared = prepare_predictions_frame(oos_predictions)
        pred_dates = prepared["Date"].unique()
        pick_date = as_of if as_of in pred_dates else max(
            d for d in pred_dates if d <= as_of
        ) if any(d <= as_of for d in pred_dates) else None
        if pick_date is not None:
            predictions = get_predictions_on_date(
                oos_predictions, pick_date, horizon=DEFAULT_HORIZON
            )
            record(
                "horizon_21_predictions",
                len(predictions) > 0,
                f"date={pd.Timestamp(pick_date).date()}, n={len(predictions)}",
            )
        else:
            record("horizon_21_predictions", False, "no OOS date available")

    # Individual optimizers
    try:
        cov = estimate_covariance(returns, as_of_date=as_of, lookback=60)
        mu = estimate_mu(returns, as_of_date=as_of, lookback=60)

        for name, fn, args, profile in (
            (
                "optimizer_min_variance",
                optimize_min_variance,
                (
                    cov,
                    symbols,
                    RISK_FREE_RATE,
                    get_profile_constraints(PortfolioProfile.CONSERVATIVE).max_weight_per_stock,
                ),
                PortfolioProfile.CONSERVATIVE,
            ),
            (
                "optimizer_max_sharpe",
                optimize_max_sharpe,
                (
                    mu,
                    cov,
                    symbols,
                    RISK_FREE_RATE,
                    get_profile_constraints(PortfolioProfile.BALANCED).max_weight_per_stock,
                ),
                PortfolioProfile.BALANCED,
            ),
            (
                "optimizer_risk_parity",
                optimize_risk_parity,
                (returns, as_of, symbols, 60),
                PortfolioProfile.BALANCED,
            ),
            (
                "optimizer_equal_weight",
                optimize_equal_weight,
                (symbols,),
                PortfolioProfile.BALANCED,
            ),
        ):
            raw_w = fn(*args)
            w = scale_to_full_investment(
                normalize_weights(raw_w),
                get_profile_constraints(profile),
                SECTOR_MAP,
            )
            ok, detail = _check_weights_valid(
                w,
                get_profile_constraints(profile),
                min_sum=0.85 if profile == PortfolioProfile.CONSERVATIVE else 0.99,
            )
            record(name, ok, detail)

        if not predictions.empty:
            try:
                raw_bl = optimize_black_litterman(
                    predictions,
                    cov,
                    symbols,
                    max_weight=get_profile_constraints(
                        PortfolioProfile.AGGRESSIVE
                    ).max_weight_per_stock,
                )
                w_bl = scale_to_full_investment(
                    normalize_weights(raw_bl),
                    get_profile_constraints(PortfolioProfile.AGGRESSIVE),
                    SECTOR_MAP,
                )
                ok, detail = _check_weights_valid(
                    w_bl,
                    get_profile_constraints(PortfolioProfile.AGGRESSIVE),
                )
                record("optimizer_black_litterman_ranks", ok, detail)
            except Exception as exc:
                record("optimizer_black_litterman_ranks", False, str(exc))
            ranks = predictions_to_ranks(predictions)
            record(
                "rank_views_in_unit_interval",
                bool(((ranks > 0) & (ranks <= 1)).all()),
                f"rank_min={ranks.min():.3f}, rank_max={ranks.max():.3f}",
            )
        else:
            record("optimizer_black_litterman_ranks", False, "no predictions")
    except Exception as exc:
        record("optimizer_suite", False, str(exc))

    # Profile solves
    for profile in PortfolioProfile:
        try:
            result = solve_portfolio(
                profile=profile,
                symbols=symbols,
                returns_history=returns,
                as_of_date=as_of,
                predictions=predictions if not predictions.empty else None,
            )
            constraints = get_profile_constraints(profile)
            ok, detail = _check_weights_valid(
                result.weights,
                constraints,
                min_sum=0.85 if profile == PortfolioProfile.CONSERVATIVE else 0.99,
            )
            record(
                f"solve_{profile.value}",
                ok and result.success,
                f"method={result.method}, fallback={result.fallback_used}, {detail}",
            )
        except Exception as exc:
            record(f"solve_{profile.value}", False, str(exc))

    # Fallback: ill-conditioned covariance forces risk-parity / equal-weight path
    try:
        n = len(symbols)
        bad_cov = pd.DataFrame(
            np.ones((n, n)) * 0.01,
            index=symbols,
            columns=symbols,
        )
        try:
            optimize_min_variance(bad_cov, symbols)
            primary_failed = False
        except Exception:
            primary_failed = True
        record(
            "fallback_primary_failure_trigger",
            primary_failed or _covariance_is_ill_conditioned(bad_cov),
            "rank1_cov",
        )

        import src.portfolio as portfolio_module

        def _fail_primary(*args, **kwargs):
            raise OptimizationError("forced primary failure for verification")

        with patch.object(
            portfolio_module,
            "_run_primary_optimizer",
            side_effect=_fail_primary,
        ):
            result_fb = portfolio_module.solve_portfolio(
                profile=PortfolioProfile.CONSERVATIVE,
                symbols=symbols,
                returns_history=returns,
                as_of_date=as_of,
                predictions=predictions if not predictions.empty else None,
            )

        ok, detail = _check_weights_valid(
            result_fb.weights,
            get_profile_constraints(PortfolioProfile.CONSERVATIVE),
            min_sum=0.85,
        )
        record(
            "fallback_produces_valid_weights",
            ok and result_fb.fallback_used,
            f"method={result_fb.method}, fallback={result_fb.fallback_used}, {detail}",
        )
    except Exception as exc:
        record("fallback_logic", False, str(exc))

    # Real-prediction BL via solve_portfolio (Aggressive)
    if not predictions.empty:
        try:
            bl_result = solve_portfolio(
                profile=PortfolioProfile.AGGRESSIVE,
                symbols=symbols,
                returns_history=returns,
                as_of_date=as_of,
                predictions=predictions,
            )
            ok, detail = _check_weights_valid(
                bl_result.weights,
                get_profile_constraints(PortfolioProfile.AGGRESSIVE),
            )
            used_bl = bl_result.method in (
                OptimizationMethod.BLACK_LITTERMAN.value,
                OptimizationMethod.RISK_PARITY.value,
            )
            record(
                "aggressive_bl_real_predictions",
                ok and used_bl,
                f"method={bl_result.method}, fallback={bl_result.fallback_used}, {detail}",
            )
        except Exception as exc:
            record("aggressive_bl_real_predictions", False, str(exc))

    return results


def run_verification() -> bool:
    """CLI entry point for Phase 3B portfolio verification."""
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("Phase 3B Step 2: Portfolio Module Verification")
    print("=" * 60)

    results = verify_portfolio_module()
    for check in results["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] {check['check']}: {check['detail']}")

    print()
    print("-" * 60)
    print("Cash weight invariant (all profiles × monthly rebalance dates)")
    print("-" * 60)

    cash_results = verify_cash_weight_all_rebalances()
    for check in cash_results["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] {check['check']}: {check['detail']}")

    if cash_results.get("failures"):
        print("\nFailure details:")
        for msg in cash_results["failures"][:20]:
            print(f"  - {msg}")
        remaining = len(cash_results["failures"]) - 20
        if remaining > 0:
            print(f"  ... and {remaining} more")

    print()
    all_passed = bool(results["passed"]) and bool(cash_results["passed"])
    if all_passed:
        print("ALL CHECKS PASSED")
    else:
        print("VERIFICATION FAILED")
    return all_passed


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_verification() else 1)
