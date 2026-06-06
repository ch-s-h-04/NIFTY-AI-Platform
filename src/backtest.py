# src/backtest.py
"""
Phase 3B backtesting engine for the NIFTY-50 AI Platform.

Monthly rebalance at market close on day *t*; fills at open on day *t+1*.
Supports CASH residual handling, transaction costs, and NIFTY-50 benchmark comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.config import SYMBOLS
from src.data_loader import load_index_data, load_stock_data
from src.models import get_backtest_execution_policy
from src.portfolio import (
    CASH_WEIGHT_TOLERANCE,
    DEFAULT_HORIZON,
    PortfolioConfig,
    PortfolioProfile,
    PortfolioResult,
    compute_cash_weight,
    get_monthly_rebalance_dates,
    get_predictions_on_date,
    prepare_predictions_frame,
    solve_portfolio,
    verify_cash_weight_invariant,
)
from src.risk import (
    DEFAULT_COVARIANCE_LOOKBACK,
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    DateLike,
    RiskConfig,
    RiskMetrics,
    annualized_return,
    compute_risk_metrics,
    compute_simple_returns,
    equity_curve_from_returns,
)

logger = logging.getLogger("NIFTY_AI_Backtest")

CASH_SYMBOL: str = "CASH"
TRANSACTION_COST_BPS: float = 10.0
TRANSACTION_COST_RATE: float = TRANSACTION_COST_BPS / 10_000.0
NIFTY50_INDEX_NAME: str = "NIFTY 50"


@dataclass
class BacktestConfig:
    """Runtime settings for portfolio simulation."""

    profile: PortfolioProfile = PortfolioProfile.BALANCED
    start_date: str = "2010-01-01"
    end_date: str = "2018-12-31"
    horizon: int = DEFAULT_HORIZON
    transaction_cost_bps: float = TRANSACTION_COST_BPS
    risk_free_rate: float = RISK_FREE_RATE
    covariance_lookback: int = DEFAULT_COVARIANCE_LOOKBACK


@dataclass
class BacktestResult:
    """Output of a full walk-forward portfolio simulation."""

    profile: PortfolioProfile
    config: BacktestConfig
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series
    portfolio_equity: pd.Series
    benchmark_equity: pd.Series
    portfolio_metrics: RiskMetrics
    benchmark_metrics: RiskMetrics
    rebalance_log: pd.DataFrame
    signal_dates: pd.DatetimeIndex
    execution_dates: pd.DatetimeIndex
    trading_calendar: pd.DatetimeIndex = field(repr=False)


def _to_timestamp(date: DateLike) -> pd.Timestamp:
    return pd.Timestamp(date)


# ---------------------------------------------------------------------------
# Trading calendar
# ---------------------------------------------------------------------------


def get_nifty50_trading_calendar(
    start_date: DateLike = "2000-01-03",
    end_date: DateLike = "2021-04-30",
) -> pd.DatetimeIndex:
    """
    NSE trading sessions from the NIFTY 50 index file (actual exchange dates).
    """
    start = _to_timestamp(start_date)
    end = _to_timestamp(end_date)
    index_df = load_index_data(NIFTY50_INDEX_NAME)
    if index_df.empty:
        raise ValueError(
            f"Cannot build trading calendar: '{NIFTY50_INDEX_NAME}' index data missing."
        )
    dates = pd.DatetimeIndex(index_df["Date"].unique()).sort_values()
    return dates[(dates >= start) & (dates <= end)]


def get_monthly_rebalance_schedule(
    calendar: pd.DatetimeIndex,
    min_date: Optional[DateLike] = None,
) -> pd.DatetimeIndex:
    """Month-end signal dates on the NIFTY 50 trading calendar."""
    return get_monthly_rebalance_dates(calendar, min_date=min_date)


def signal_to_execution_date(
    signal_date: DateLike,
    calendar: pd.DatetimeIndex,
) -> Optional[pd.Timestamp]:
    """
    Map signal close on day *t* to execution open on the next NIFTY session *t+1*.
    """
    signal = _to_timestamp(signal_date)
    calendar = pd.DatetimeIndex(calendar).sort_values()
    loc = calendar.get_indexer([signal], method=None)
    if loc[0] < 0:
        return None
    idx = int(loc[0])
    if idx + 1 >= len(calendar):
        return None
    return pd.Timestamp(calendar[idx + 1])


def build_signal_execution_map(
    signal_dates: pd.DatetimeIndex,
    calendar: pd.DatetimeIndex,
    end_date: DateLike,
) -> Dict[pd.Timestamp, pd.Timestamp]:
    """Execution date -> signal date (only pairs with a valid next session)."""
    end = _to_timestamp(end_date)
    mapping: Dict[pd.Timestamp, pd.Timestamp] = {}
    for signal in signal_dates:
        exec_date = signal_to_execution_date(signal, calendar)
        if exec_date is not None and exec_date <= end:
            mapping[exec_date] = pd.Timestamp(signal)
    return mapping


# ---------------------------------------------------------------------------
# Price panels & returns
# ---------------------------------------------------------------------------


def create_aligned_open_panel(
    symbols: List[str],
    start_date: str,
    end_date: str,
    calendar: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Wide open-price panel aligned to the NIFTY 50 trading calendar."""
    calendar = (
        get_nifty50_trading_calendar(start_date, end_date)
        if calendar is None
        else calendar
    )
    aligned = pd.DataFrame(index=calendar)
    aligned.index.name = "Date"

    for symbol in symbols:
        stock_df = load_stock_data(symbol)
        if stock_df.empty or "Open" not in stock_df.columns:
            aligned[symbol] = np.nan
            continue
        series = (
            stock_df[["Date", "Open"]]
            .drop_duplicates(subset=["Date"])
            .set_index("Date")["Open"]
        )
        aligned[symbol] = series.reindex(calendar)

    return aligned


def create_aligned_close_panel(
    symbols: List[str],
    start_date: str,
    end_date: str,
    calendar: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Wide close-price panel aligned to the NIFTY 50 trading calendar."""
    from src.data_loader import create_aligned_panel

    calendar = (
        get_nifty50_trading_calendar(start_date, end_date)
        if calendar is None
        else calendar
    )
    panel = create_aligned_panel(symbols, start_date=start_date, end_date=end_date)
    return panel.reindex(calendar)


def compute_open_to_close_returns(
    open_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Same-day open-to-close simple returns (execution-day convention)."""
    aligned_open = open_prices.reindex(close_prices.index)
    return close_prices / aligned_open - 1.0


def load_benchmark_close_series(
    start_date: str,
    end_date: str,
    calendar: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """NIFTY 50 benchmark close prices on the trading calendar."""
    calendar = (
        get_nifty50_trading_calendar(start_date, end_date)
        if calendar is None
        else calendar
    )
    index_df = load_index_data(NIFTY50_INDEX_NAME)
    if index_df.empty:
        raise ValueError("NIFTY 50 benchmark data unavailable.")
    closes = (
        index_df[["Date", "Close"]]
        .drop_duplicates(subset=["Date"])
        .set_index("Date")["Close"]
        .reindex(calendar)
    )
    return closes


# ---------------------------------------------------------------------------
# CASH, turnover, costs
# ---------------------------------------------------------------------------


def daily_cash_return(
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Constant daily risk-free return for the cash sleeve."""
    return float((1.0 + risk_free_rate) ** (1.0 / trading_days) - 1.0)


def to_effective_weights(result: PortfolioResult) -> pd.Series:
    """Expand ``PortfolioResult`` to a full weight vector including CASH."""
    weights = result.weights.astype(float).clip(lower=0.0)
    cash = float(result.cash_weight)
    if cash <= 0.0:
        cash = compute_cash_weight(weights)
    effective = weights.copy()
    effective[CASH_SYMBOL] = cash
    return effective


def extract_stock_weights(effective: pd.Series) -> pd.Series:
    """Stock sleeve only (exclude CASH key)."""
    return effective.drop(CASH_SYMBOL, errors="ignore")


def compute_turnover(
    weights_old: pd.Series,
    weights_new: pd.Series,
) -> float:
    """One-way turnover: 0.5 * sum(|Δw|) over all legs including CASH."""
    all_idx = weights_old.index.union(weights_new.index)
    old = weights_old.reindex(all_idx, fill_value=0.0)
    new = weights_new.reindex(all_idx, fill_value=0.0)
    return float(0.5 * np.abs(new - old).sum())


def compute_stock_transaction_cost(
    old_stock: pd.Series,
    new_stock: pd.Series,
    cost_rate: float = TRANSACTION_COST_RATE,
) -> float:
    """
    One-way transaction cost on equity trades only (10 bps default).

    cost = rate * sum(|Δw_stock|)
    """
    all_idx = old_stock.index.union(new_stock.index)
    old = old_stock.reindex(all_idx, fill_value=0.0)
    new = new_stock.reindex(all_idx, fill_value=0.0)
    return float(cost_rate * np.abs(new - old).sum())


def portfolio_daily_return(
    stock_weights: pd.Series,
    cash_weight: float,
    stock_returns: pd.Series,
    cash_return: float,
) -> float:
    """Single-day portfolio return with explicit cash leg."""
    aligned = stock_weights.reindex(stock_returns.index, fill_value=0.0)
    stock_contrib = float((aligned * stock_returns.fillna(0.0)).sum())
    return stock_contrib + float(cash_weight) * cash_return


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


def _resolve_predictions(
    oos_predictions: pd.DataFrame,
    signal_date: pd.Timestamp,
    horizon: int,
) -> Optional[pd.Series]:
    prepared = prepare_predictions_frame(oos_predictions, horizon=horizon)
    if prepared.empty:
        return None
    preds = get_predictions_on_date(oos_predictions, signal_date, horizon=horizon)
    return preds if not preds.empty else None


def run_backtest(
    symbols: Optional[List[str]] = None,
    config: Optional[BacktestConfig] = None,
    oos_predictions: Optional[pd.DataFrame] = None,
    portfolio_config: Optional[PortfolioConfig] = None,
) -> BacktestResult:
    """
    Run a monthly-rebalance backtest with open execution and CASH handling.

    Signal at close *t* (month-end); trades execute at open *t+1*.
    """
    symbols = symbols or SYMBOLS
    config = config or BacktestConfig()
    portfolio_config = portfolio_config or PortfolioConfig(horizon=config.horizon)
    cost_rate = config.transaction_cost_bps / 10_000.0
    cash_ret = daily_cash_return(config.risk_free_rate)

    calendar = get_nifty50_trading_calendar(config.start_date, config.end_date)
    min_signal = calendar[config.covariance_lookback]
    signal_dates = get_monthly_rebalance_schedule(calendar, min_date=min_signal)
    exec_map = build_signal_execution_map(signal_dates, calendar, config.end_date)
    execution_dates = pd.DatetimeIndex(sorted(exec_map.keys()))

    if oos_predictions is not None and not oos_predictions.empty:
        oos_predictions = prepare_predictions_frame(
            oos_predictions, horizon=config.horizon
        )

    close_prices = create_aligned_close_panel(
        symbols, config.start_date, config.end_date, calendar=calendar
    )
    open_prices = create_aligned_open_panel(
        symbols, config.start_date, config.end_date, calendar=calendar
    )
    close_to_close = compute_simple_returns(close_prices)
    open_to_close = compute_open_to_close_returns(open_prices, close_prices)
    benchmark_closes = load_benchmark_close_series(
        config.start_date, config.end_date, calendar=calendar
    )
    benchmark_returns = benchmark_closes.pct_change()

    if not execution_dates.size:
        raise ValueError("No valid rebalance execution dates in backtest window.")

    sim_start = execution_dates[0]
    sim_end = _to_timestamp(config.end_date)
    sim_days = calendar[(calendar >= sim_start) & (calendar <= sim_end)]

    current_effective = pd.Series({CASH_SYMBOL: 1.0})
    portfolio_returns: List[float] = []
    return_index: List[pd.Timestamp] = []
    rebalance_rows: List[Dict[str, object]] = []

    for day in sim_days:
        txn_cost = 0.0
        turnover = 0.0

        if day in exec_map:
            signal_date = exec_map[day]
            preds = (
                _resolve_predictions(oos_predictions, signal_date, config.horizon)
                if oos_predictions is not None and not oos_predictions.empty
                else None
            )
            hist = close_to_close.loc[:signal_date]
            result = solve_portfolio(
                profile=config.profile,
                symbols=symbols,
                returns_history=hist,
                as_of_date=signal_date,
                predictions=preds,
                config=portfolio_config,
            )
            ok, _ = verify_cash_weight_invariant(result, tol=CASH_WEIGHT_TOLERANCE)
            if not ok:
                raise RuntimeError(
                    f"Cash invariant failed at signal {signal_date.date()} "
                    f"for profile {config.profile.value}."
                )

            new_effective = to_effective_weights(result)
            old_stock = extract_stock_weights(current_effective)
            new_stock = result.weights
            turnover = compute_turnover(current_effective, new_effective)
            txn_cost = compute_stock_transaction_cost(
                old_stock, new_stock, cost_rate=cost_rate
            )
            current_effective = new_effective

            rebalance_rows.append(
                {
                    "signal_date": signal_date,
                    "execution_date": day,
                    "stock_weight_sum": float(result.weights.sum()),
                    "cash_weight": float(result.cash_weight),
                    "turnover": turnover,
                    "transaction_cost": txn_cost,
                    "method": result.method,
                    "fallback_used": result.fallback_used,
                }
            )

        stock_w = extract_stock_weights(current_effective)
        cash_w = float(current_effective.get(CASH_SYMBOL, 0.0))

        if day in exec_map:
            day_stock_returns = open_to_close.loc[day]
        else:
            day_stock_returns = close_to_close.loc[day]

        gross = portfolio_daily_return(stock_w, cash_w, day_stock_returns, cash_ret)
        net = gross - txn_cost
        portfolio_returns.append(net)
        return_index.append(day)

    port_ret = pd.Series(portfolio_returns, index=pd.DatetimeIndex(return_index))
    bench_ret = benchmark_returns.reindex(port_ret.index)

    risk_cfg = RiskConfig(risk_free_rate=config.risk_free_rate)
    port_metrics = compute_risk_metrics(port_ret, config=risk_cfg)
    bench_metrics = compute_risk_metrics(bench_ret.dropna(), config=risk_cfg)

    return BacktestResult(
        profile=config.profile,
        config=config,
        portfolio_returns=port_ret,
        benchmark_returns=bench_ret,
        portfolio_equity=equity_curve_from_returns(port_ret),
        benchmark_equity=equity_curve_from_returns(bench_ret.fillna(0.0)),
        portfolio_metrics=port_metrics,
        benchmark_metrics=bench_metrics,
        rebalance_log=pd.DataFrame(rebalance_rows),
        signal_dates=signal_dates,
        execution_dates=execution_dates,
        trading_calendar=calendar,
    )


def performance_summary(result: BacktestResult) -> pd.DataFrame:
    """Side-by-side portfolio vs benchmark statistics."""
    rows = [
        {
            "series": "portfolio",
            "cagr": annualized_return(result.portfolio_returns),
            "sharpe": result.portfolio_metrics.sharpe,
            "sortino": result.portfolio_metrics.sortino,
            "max_drawdown": result.portfolio_metrics.max_drawdown,
            "ann_vol": result.portfolio_metrics.annualized_volatility,
            "n_days": result.portfolio_metrics.n_observations,
        },
        {
            "series": "benchmark",
            "cagr": annualized_return(result.benchmark_returns.dropna()),
            "sharpe": result.benchmark_metrics.sharpe,
            "sortino": result.benchmark_metrics.sortino,
            "max_drawdown": result.benchmark_metrics.max_drawdown,
            "ann_vol": result.benchmark_metrics.annualized_volatility,
            "n_days": result.benchmark_metrics.n_observations,
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_phase3b(
    symbols: Optional[List[str]] = None,
    start_date: str = "2015-01-01",
    end_date: str = "2018-06-29",
    oos_predictions: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """
    Verification suite for Phase 3B backtest engine components.
    """
    from src.models import build_modeling_dataset, run_walk_forward_cv

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

    policy = get_backtest_execution_policy()
    record(
        "execution_policy_contract",
        policy["signal_time"] == "market_close_day_t"
        and policy["execution_time"] == "market_open_day_t_plus_1"
        and policy["backtest_fill_price"] == "open_day_t_plus_1",
        str(policy),
    )

    calendar = get_nifty50_trading_calendar(start_date, end_date)
    record(
        "nifty50_calendar_nonempty",
        len(calendar) > 0,
        f"n_days={len(calendar)}, first={calendar[0].date()}, last={calendar[-1].date()}",
    )

    min_signal = calendar[DEFAULT_COVARIANCE_LOOKBACK]
    signal_dates = get_monthly_rebalance_schedule(calendar, min_date=min_signal)
    month_periods = signal_dates.to_period("M")
    is_month_end = all(
        signal == calendar[calendar.to_period("M") == period][-1]
        for signal, period in zip(signal_dates, month_periods)
    )
    record(
        "rebalance_dates_month_end",
        is_month_end and len(signal_dates) > 0,
        f"n_rebalances={len(signal_dates)}",
    )

    exec_map = build_signal_execution_map(signal_dates, calendar, end_date)
    exec_ok = True
    exec_details: List[str] = []
    for exec_date, signal in exec_map.items():
        expected = signal_to_execution_date(signal, calendar)
        if expected is None or pd.Timestamp(exec_date) != expected:
            exec_ok = False
            exec_details.append(f"{signal.date()}->{exec_date}")
        loc_sig = calendar.get_loc(signal)
        if loc_sig + 1 >= len(calendar) or calendar[loc_sig + 1] != exec_date:
            exec_ok = False
    record(
        "execution_mapping_t_plus_1",
        exec_ok and len(exec_map) > 0,
        f"n_pairs={len(exec_map)}",
    )

    # Cash accounting unit test
    dummy = PortfolioResult(
        weights=pd.Series({"A": 0.4, "B": 0.5}),
        profile=PortfolioProfile.CONSERVATIVE,
        method="test",
        success=True,
        fallback_used=False,
        message="test",
        cash_weight=0.1,
    )
    eff = to_effective_weights(dummy)
    record(
        "cash_effective_weights",
        abs(float(eff.sum()) - 1.0) < CASH_WEIGHT_TOLERANCE
        and float(eff[CASH_SYMBOL]) == 0.1,
        f"sum={eff.sum():.6f}, cash={eff[CASH_SYMBOL]:.6f}",
    )

    r_cash = daily_cash_return()
    record(
        "cash_accrues_risk_free",
        0.0 < r_cash < 0.001,
        f"daily_rf={r_cash:.8f}, annual={RISK_FREE_RATE}",
    )

    # Turnover & transaction cost synthetic
    old = pd.Series({"A": 0.5, CASH_SYMBOL: 0.5})
    new = pd.Series({"A": 0.6, "B": 0.3, CASH_SYMBOL: 0.1})
    turnover = compute_turnover(old, new)
    expected_turnover = 0.5 * (0.1 + 0.3 + 0.4)  # |ΔA|+|ΔB|+|ΔCASH|
    record(
        "turnover_calculation",
        abs(turnover - expected_turnover) < 1e-9,
        f"turnover={turnover:.4f}, expected={expected_turnover:.4f}",
    )
    txn = compute_stock_transaction_cost(
        extract_stock_weights(old), extract_stock_weights(new)
    )
    expected_txn = TRANSACTION_COST_RATE * (0.1 + 0.3)
    record(
        "transaction_cost_stock_only",
        abs(txn - expected_txn) < 1e-12,
        f"cost={txn:.8f}, expected={expected_txn:.8f}",
    )

    if oos_predictions is None:
        try:
            dataset, feature_cols = build_modeling_dataset(
                symbols=symbols,
                start_date="2010-01-01",
                end_date=end_date,
                warmup_days=60,
            )
            wf = run_walk_forward_cv(dataset=dataset, feature_cols=feature_cols)
            oos_predictions = wf.oos_predictions
        except Exception as exc:
            record("load_oos_predictions", False, str(exc))
            oos_predictions = pd.DataFrame()

    if not oos_predictions.empty:
        oos_predictions = prepare_predictions_frame(oos_predictions)
        record(
            "horizon_21_only",
            set(oos_predictions["horizon"].unique()) == {DEFAULT_HORIZON},
            f"horizons={sorted(oos_predictions['horizon'].unique())}",
        )

    try:
        bt = run_backtest(
            symbols=symbols,
            config=BacktestConfig(
                profile=PortfolioProfile.BALANCED,
                start_date=start_date,
                end_date=end_date,
            ),
            oos_predictions=oos_predictions,
        )
    except Exception as exc:
        record("run_backtest", False, str(exc))
        return results

    record(
        "benchmark_alignment",
        bt.portfolio_returns.index.equals(bt.benchmark_returns.index)
        and len(bt.portfolio_returns) > 0,
        f"n_days={len(bt.portfolio_returns)}",
    )

    finite = all(
        np.isfinite(
            [
                bt.portfolio_metrics.sharpe,
                bt.portfolio_metrics.max_drawdown,
                bt.benchmark_metrics.sharpe,
                bt.benchmark_metrics.max_drawdown,
            ]
        )
    )
    record(
        "risk_metrics_finite",
        finite,
        (
            f"port_sharpe={bt.portfolio_metrics.sharpe:.3f}, "
            f"bench_sharpe={bt.benchmark_metrics.sharpe:.3f}"
        ),
    )

    cash_inv_ok = True
    cash_inv_detail = ""
    for _, row in bt.rebalance_log.iterrows():
        total = float(row["stock_weight_sum"]) + float(row["cash_weight"])
        if abs(total - 1.0) > CASH_WEIGHT_TOLERANCE:
            cash_inv_ok = False
            cash_inv_detail = (
                f"signal={row['signal_date']}: total={total:.12f}"
            )
            break
    record(
        "cash_weight_invariant_rebalances",
        cash_inv_ok and len(bt.rebalance_log) > 0,
        cash_inv_detail or f"n_rebalances={len(bt.rebalance_log)}",
    )

    if len(bt.rebalance_log) > 0:
        max_cost = float(bt.rebalance_log["transaction_cost"].max())
        record(
            "transaction_costs_applied",
            max_cost > 0.0,
            f"max_cost={max_cost:.6f}, mean_turnover={bt.rebalance_log['turnover'].mean():.4f}",
        )

    # Conservative partial-investment path on same window
    try:
        bt_cons = run_backtest(
            symbols=symbols,
            config=BacktestConfig(
                profile=PortfolioProfile.CONSERVATIVE,
                start_date=start_date,
                end_date=end_date,
            ),
            oos_predictions=oos_predictions,
        )
        has_cash = bool((bt_cons.rebalance_log["cash_weight"] > CASH_WEIGHT_TOLERANCE).any())
        record(
            "conservative_cash_accounting",
            has_cash,
            f"max_cash={bt_cons.rebalance_log['cash_weight'].max():.4f}",
        )
    except Exception as exc:
        record("conservative_cash_accounting", False, str(exc))

    results["backtest_result"] = bt
    results["performance_summary"] = performance_summary(bt)
    return results


def run_verification() -> bool:
    """CLI entry point for Phase 3B backtest verification."""
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("Phase 3B Step 3: Backtest Module Verification")
    print("=" * 60)

    results = verify_phase3b()
    for check in results["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] {check['check']}: {check['detail']}")

    print()
    if "performance_summary" in results:
        print("Performance summary (Balanced profile, verification window):")
        print(results["performance_summary"].to_string(index=False))

    print()
    if results["passed"]:
        print("ALL CHECKS PASSED")
    else:
        print("VERIFICATION FAILED")
    return bool(results["passed"])


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_verification() else 1)
