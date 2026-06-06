"""
Dashboard verification (no model training).

Run from project root:

    python -m app.verify_dashboard
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import dashboard_data, paths, theme  # noqa: E402
from src.backtest import BacktestConfig, BacktestResult  # noqa: E402
from src.portfolio import PortfolioProfile  # noqa: E402
from src.risk import compute_risk_metrics, drawdown_series, equity_curve_from_returns  # noqa: E402


def _compile_pages() -> None:
    page_dir = ROOT / "app" / "pages"
    for page in page_dir.glob("*.py"):
        source = page.read_text(encoding="utf-8")
        compile(source, str(page), "exec")


def _import_pages() -> None:
    import importlib

    for page in sorted((ROOT / "app" / "pages").glob("*.py")):
        module_name = f"app.pages.{page.stem}"
        importlib.import_module(module_name)


def _synthetic_backtest() -> BacktestResult:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2015-01-01", periods=500)
    port = pd.Series(rng.normal(0.0004, 0.012, len(idx)), index=idx)
    bench = pd.Series(rng.normal(0.0003, 0.011, len(idx)), index=idx)
    cfg = BacktestConfig(profile=PortfolioProfile.BALANCED)
    return BacktestResult(
        profile=PortfolioProfile.BALANCED,
        config=cfg,
        portfolio_returns=port,
        benchmark_returns=bench,
        portfolio_equity=equity_curve_from_returns(port),
        benchmark_equity=equity_curve_from_returns(bench),
        portfolio_metrics=compute_risk_metrics(port),
        benchmark_metrics=compute_risk_metrics(bench),
        rebalance_log=pd.DataFrame(),
        signal_dates=pd.DatetimeIndex([]),
        execution_dates=pd.DatetimeIndex([]),
        trading_calendar=idx,
    )


def _test_plotly_charts() -> None:
    result = _synthetic_backtest()
    theme.apply_figure_theme(
        go.Figure(
            go.Scatter(x=result.portfolio_equity.index, y=result.portfolio_equity.values)
        )
    )
    dd = drawdown_series(result.portfolio_equity)
    theme.apply_figure_theme(go.Figure(go.Scatter(x=dd.index, y=dd.values)))
    roll = dashboard_data.rolling_annualized_volatility(result.portfolio_returns)
    theme.apply_figure_theme(go.Figure(go.Scatter(x=roll.index, y=roll.values)))


def _test_artifact_io() -> None:
    status = dashboard_data.artifact_status()
    msg = dashboard_data.missing_artifacts_message(status)
    if status.any_missing:
        assert msg
    if status.oos_predictions:
        bt = dashboard_data.run_profile_backtest(PortfolioProfile.BALANCED)
        assert bt is not None
        assert len(bt.portfolio_equity) > 0


def _test_shap_artifact_io() -> None:
    status = dashboard_data.artifact_status()
    if status.shap_missing:
        assert dashboard_data.missing_shap_artifacts_message(status)
        return
    summary = dashboard_data.load_shap_summary()
    importance = dashboard_data.load_shap_feature_importance()
    assert summary is not None and not summary.empty
    assert importance is not None and not importance.empty
    assert dashboard_data.shap_sample_count(summary) > 0
    from app.utils.shap_viz import plot_shap_importance_bar, plot_shap_summary

    theme.apply_figure_theme(plot_shap_summary(summary, importance, top_n=10))
    theme.apply_figure_theme(plot_shap_importance_bar(importance, top_n=10))


def main() -> int:
    checks = [
        ("compile_pages", _compile_pages),
        ("import_pages", _import_pages),
        ("plotly_charts", _test_plotly_charts),
        ("artifact_io", _test_artifact_io),
        ("shap_artifact_io", _test_shap_artifact_io),
    ]
    print("=" * 60)
    print("Phase 4 Dashboard Verification")
    print("=" * 60)
    passed = True
    for name, fn in checks:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:
            passed = False
            print(f"[FAIL] {name}: {exc}")
    status = dashboard_data.artifact_status()
    print()
    print("Artifact status:")
    print(f"  oos_predictions:     {status.oos_predictions}")
    print(f"  summary_metrics:     {status.summary_metrics}")
    print(f"  feature_importance:  {status.feature_importance}")
    print(f"  shap_summary:        {status.shap_summary}")
    print(f"  shap_importance:     {status.shap_feature_importance}")
    if status.any_missing:
        print()
        print(dashboard_data.missing_artifacts_message(status))
    if status.shap_missing:
        print()
        print(dashboard_data.missing_shap_artifacts_message(status))
    elif dashboard_data.load_shap_feature_importance() is not None:
        top = dashboard_data.load_shap_feature_importance().head(10)
        print()
        print(f"SHAP samples: {dashboard_data.shap_sample_count()}")
        print("Top 10 SHAP features (mean |SHAP|):")
        print(top[["feature", "mean_abs_shap", "mean_shap"]].to_string(index=False))
    print()
    print("ALL CHECKS PASSED" if passed else "VERIFICATION FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
