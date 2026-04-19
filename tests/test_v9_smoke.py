"""Smoke tests covering the v9 public API surface.

These are intentionally broad, not exhaustive: the goal is to confirm that
the v10 changes (purge/embargo, new Kelly, PSR, PBO) did **not** break any
of the core v9 entry points. More detailed correctness tests live in the
dedicated ``test_v10_upgrades.py`` file and in future dedicated v9 test
files.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from chan_trading.backtest.engine import run_backtest
from chan_trading.backtest.event_engine import run_event_backtest
from chan_trading.config import (
    CrossSectionalMeanReversionConfig,
    CrossSectionalMomentumConfig,
    EventBacktestConfig,
    JohansenConfig,
    MeanReversionConfig,
    MomentumConfig,
    WalkForwardConfig,
)
from chan_trading.data.loaders import load_prices_csv
from chan_trading.features.cointegration import johansen_basket_from_prices
from chan_trading.features.statistics import (
    adf_stationarity_test,
    engle_granger_cointegration_test,
    estimate_half_life,
    estimate_spread_diagnostics_object,
    estimate_static_hedge_model,
    estimate_variance_ratio,
    rolling_zscore,
)
from chan_trading.portfolio.kelly import (
    apply_kelly_scaling,
    kelly_fraction,
    rolling_kelly_fraction,
)
from chan_trading.portfolio.sizing import (
    apply_cppi_throttle,
    apply_drawdown_throttle,
    apply_vol_target,
    cppi_scale,
    lag_positions,
    turnover,
)
from chan_trading.risk.metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
    summarize_risk,
)
from chan_trading.risk.monte_carlo import simulate_max_drawdown
from chan_trading.strategies.basket_mean_reversion import BasketMeanReversionStrategy
from chan_trading.strategies.cross_sectional_mean_reversion import (
    CrossSectionalMeanReversionStrategy,
)
from chan_trading.strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from chan_trading.strategies.mean_reversion import PairMeanReversionStrategy
from chan_trading.strategies.momentum import TimeSeriesMomentumStrategy
from chan_trading.validation.walkforward import (
    generate_walkforward_windows,
    run_walkforward_pair_mean_reversion,
)
from chan_trading.validation.walkforward_basket import (
    run_walkforward_johansen_basket_event_backtest,
)
from chan_trading.validation.walkforward_event import (
    run_pair_mean_reversion_event_backtest,
    run_walkforward_pair_mean_reversion_event_backtest,
)


# ---------------------------------------------------------------------------
# Features: statistics & cointegration
# ---------------------------------------------------------------------------


def test_rolling_zscore_shape_and_center() -> None:
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 1, 300))
    z = rolling_zscore(s, window=20)
    assert z.shape == s.shape
    # After warm-up the mean should be near 0
    assert abs(z.iloc[100:].mean()) < 0.5


def test_static_hedge_model_recovers_beta() -> None:
    rng = np.random.default_rng(2)
    x = pd.Series(rng.normal(100, 1, 300))
    true_beta = 0.75
    y = 5 + true_beta * x + pd.Series(rng.normal(0, 0.1, 300))
    model = estimate_static_hedge_model(y, x)
    assert abs(model.beta - true_beta) < 0.05
    assert abs(model.alpha - 5) < 0.5


def test_adf_and_half_life_on_stationary_series() -> None:
    rng = np.random.default_rng(3)
    # AR(1) with phi=0.5 -> very stationary
    eps = rng.normal(0, 1, 500)
    x = np.zeros(500)
    for i in range(1, 500):
        x[i] = 0.5 * x[i - 1] + eps[i]
    s = pd.Series(x)
    stat, pvalue = adf_stationarity_test(s)
    assert pvalue < 0.05
    hl = estimate_half_life(s)
    assert np.isfinite(hl) and 0.5 < hl < 5.0


def test_variance_ratio_and_engle_granger_on_sample_data() -> None:
    prices = load_prices_csv("data/sample_prices.csv")
    vr = estimate_variance_ratio(prices["SPY"])
    assert np.isfinite(vr)
    _stat, pvalue = engle_granger_cointegration_test(prices["SPY"], prices["IVV"])
    assert 0.0 <= pvalue <= 1.0


def test_spread_diagnostics_object_populated() -> None:
    prices = load_prices_csv("data/sample_prices.csv")
    diag = estimate_spread_diagnostics_object(
        prices["SPY"], prices["IVV"], use_log_prices=True, signal_mode="residual"
    )
    assert np.isfinite(diag.beta)
    assert 0.0 <= diag.adf_pvalue <= 1.0
    assert diag.spread_std is not None


def test_johansen_basket_from_sample_data() -> None:
    prices = load_prices_csv("data/sample_basket_prices.csv")
    weights, spread, report = johansen_basket_from_prices(
        prices,
        config=JohansenConfig(significance=0.10),
        use_log_prices=True,
    )
    assert report.rank >= 1
    assert np.isfinite(report.adf_pvalue)
    assert abs(weights.abs().sum() - 1.0) < 1e-9
    assert spread.shape[0] == prices.dropna().shape[0]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def test_pair_mean_reversion_strategy_produces_positions() -> None:
    prices = load_prices_csv("data/sample_prices.csv")
    diag = estimate_spread_diagnostics_object(prices["SPY"], prices["IVV"])
    strat = PairMeanReversionStrategy(
        y_col="SPY",
        x_col="IVV",
        hedge_ratio=diag.beta,
        intercept=diag.alpha,
        config=MeanReversionConfig(lookback=20, entry_z=1.5, exit_z=0.5),
    )
    pos = strat.generate_positions(prices)
    assert list(pos.columns) == ["SPY", "IVV"]
    assert pos.shape[0] == prices.shape[0]
    # Signal should produce at least one non-zero bar
    assert (pos.abs().sum(axis=1) > 0).any()


def test_time_series_momentum_and_stop_loss() -> None:
    rng = np.random.default_rng(10)
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    prices = pd.DataFrame(
        100.0 + rng.normal(0, 1, (300, 3)).cumsum(axis=0),
        index=idx,
        columns=list("ABC"),
    )
    strat = TimeSeriesMomentumStrategy(
        config=MomentumConfig(lookback=20, max_leverage=1.0, stop_loss=0.05, stop_loss_lookback=5)
    )
    pos = strat.generate_positions(prices)
    # Gross never exceeds max_leverage
    assert (pos.abs().sum(axis=1) <= 1.0 + 1e-9).all()


def test_cross_sectional_momentum_and_mean_reversion() -> None:
    rng = np.random.default_rng(11)
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    prices = pd.DataFrame(
        100.0 + rng.normal(0, 1, (300, 5)).cumsum(axis=0),
        index=idx,
        columns=list("ABCDE"),
    )
    mom = CrossSectionalMomentumStrategy(
        CrossSectionalMomentumConfig(lookback=20, top_fraction=0.4, bottom_fraction=0.4, max_leverage=1.0)
    )
    mr = CrossSectionalMeanReversionStrategy(
        CrossSectionalMeanReversionConfig(lookback=1, top_fraction=0.4, bottom_fraction=0.4, max_leverage=1.0)
    )
    pos_mom = mom.generate_positions(prices)
    pos_mr = mr.generate_positions(prices)
    # Both should dollar-neutralize on active rows (long-short)
    active = pos_mom.abs().sum(axis=1) > 0
    if active.any():
        # Net exposure should be near zero for long-short momentum
        assert pos_mom.loc[active].sum(axis=1).abs().max() < 0.1
    assert pos_mr.shape == pos_mom.shape


def test_basket_mean_reversion_strategy() -> None:
    prices = load_prices_csv("data/sample_basket_prices.csv")
    weights, _spread, _report = johansen_basket_from_prices(
        prices, config=JohansenConfig(significance=0.10), use_log_prices=True
    )
    strat = BasketMeanReversionStrategy(
        weights=weights, config=MeanReversionConfig(lookback=20, entry_z=1.5, exit_z=0.5)
    )
    pos = strat.generate_positions(prices)
    assert set(pos.columns) == set(weights.index)


# ---------------------------------------------------------------------------
# Backtest engines
# ---------------------------------------------------------------------------


def test_vectorized_backtest_produces_equity() -> None:
    prices = load_prices_csv("data/sample_prices.csv")
    diag = estimate_spread_diagnostics_object(prices["SPY"], prices["IVV"])
    strat = PairMeanReversionStrategy(
        y_col="SPY",
        x_col="IVV",
        hedge_ratio=diag.beta,
        intercept=diag.alpha,
        config=MeanReversionConfig(lookback=20, entry_z=1.5, exit_z=0.5),
    )
    pos = strat.generate_positions(prices)
    result = run_backtest(prices=prices[["SPY", "IVV"]], target_positions=pos, transaction_cost_bps=0.5)
    assert result.equity_curve.shape[0] == prices.shape[0]
    assert math.isfinite(result.equity_curve.iloc[-1])


def test_event_backtest_with_slippage_and_borrow_cost() -> None:
    prices = load_prices_csv("data/sample_prices.csv")
    diag = estimate_spread_diagnostics_object(prices["SPY"], prices["IVV"])
    strat = PairMeanReversionStrategy(
        y_col="SPY",
        x_col="IVV",
        hedge_ratio=diag.beta,
        intercept=diag.alpha,
        config=MeanReversionConfig(lookback=20, entry_z=1.5, exit_z=0.5),
    )
    target = strat.generate_positions(prices)
    event_cfg = EventBacktestConfig(
        initial_cash=1_000_000.0,
        commission_bps=1.0,
        slippage_bps=0.5,
        borrow_bps_annual=50.0,
    )
    result = run_event_backtest(
        prices=prices[["SPY", "IVV"]], target_weights=target, config=event_cfg
    )
    assert result.equity_curve.iloc[-1] > 0


# ---------------------------------------------------------------------------
# Sizing helpers
# ---------------------------------------------------------------------------


def test_lag_positions_turnover_and_vol_target() -> None:
    rng = np.random.default_rng(4)
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    prices = pd.DataFrame(
        100.0 + rng.normal(0, 1, (200, 2)).cumsum(axis=0),
        index=idx,
        columns=["A", "B"],
    )
    pos = pd.DataFrame(
        {
            "A": np.where(np.arange(200) % 10 < 5, 0.5, -0.5),
            "B": np.where(np.arange(200) % 10 < 5, -0.5, 0.5),
        },
        index=idx,
    )
    lagged = lag_positions(pos)
    assert lagged.iloc[0].sum() == 0.0
    t = turnover(pos)
    assert (t >= 0).all()
    scaled = apply_vol_target(prices=prices, positions=pos, target_vol=0.1, lookback=20, max_leverage=2.0)
    assert scaled.shape == pos.shape


def test_cppi_and_drawdown_throttle() -> None:
    eq = pd.Series(np.linspace(1.0, 1.2, 100)).append(pd.Series(np.linspace(1.2, 0.9, 50))) if False else (
        pd.concat([pd.Series(np.linspace(1.0, 1.2, 100)), pd.Series(np.linspace(1.2, 0.9, 50))]).reset_index(drop=True)
    )
    scale = cppi_scale(eq, floor_fraction=0.8, multiplier=3.0, max_scale=1.0)
    assert scale.min() >= 0
    assert scale.max() <= 1.0
    # When equity hits the floor, scale should drop toward 0
    assert scale.iloc[-1] < 1.0

    pos = pd.DataFrame({"A": np.ones(len(eq))}, index=eq.index)
    th1 = apply_drawdown_throttle(pos, eq, soft_limit=0.1, soft_scale=0.5, hard_limit=0.2)
    th2 = apply_cppi_throttle(pos, eq, floor_fraction=0.8)
    assert th1.shape == pos.shape
    assert th2.shape == pos.shape


# ---------------------------------------------------------------------------
# Kelly & Monte Carlo (v9 API)
# ---------------------------------------------------------------------------


def test_scalar_kelly_and_rolling_kelly() -> None:
    rng = np.random.default_rng(6)
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    k = kelly_fraction(r)
    assert np.isfinite(k)
    rk = rolling_kelly_fraction(r, lookback=60)
    assert rk.shape == r.shape


def test_apply_kelly_scaling_preserves_shape() -> None:
    rng = np.random.default_rng(8)
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    prices = pd.DataFrame(
        100.0 + rng.normal(0, 1, (200, 2)).cumsum(axis=0), index=idx, columns=["A", "B"]
    )
    pos = pd.DataFrame(
        {"A": np.where(np.arange(200) % 20 < 10, 0.5, -0.5),
         "B": np.where(np.arange(200) % 20 < 10, -0.5, 0.5)},
        index=idx,
    )
    scaled = apply_kelly_scaling(prices, pos, kelly_fraction_multiplier=0.5, lookback=50, max_leverage=2.0)
    assert scaled.shape == pos.shape


def test_monte_carlo_drawdown_report() -> None:
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0005, 0.01, 400))
    rep = simulate_max_drawdown(r, horizon=250, n_paths=200, method="block", block_size=20, seed=1)
    assert rep.n_paths == 200
    assert rep.horizon == 250
    assert rep.max_drawdown_median <= 0.0  # drawdowns are non-positive


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------


def test_core_risk_metrics() -> None:
    rng = np.random.default_rng(9)
    r = pd.Series(rng.normal(0.0005, 0.01, 500))
    eq = (1.0 + r).cumprod()
    assert np.isfinite(max_drawdown(eq))
    assert np.isfinite(annualized_return(r))
    assert np.isfinite(annualized_volatility(r))
    assert np.isfinite(sharpe_ratio(r))
    rep = summarize_risk(eq, r)
    assert np.isfinite(rep.sharpe)


# ---------------------------------------------------------------------------
# Walkforward end-to-end pipelines
# ---------------------------------------------------------------------------


def test_walkforward_basic_windows_no_purge_no_embargo() -> None:
    idx = pd.RangeIndex(300)
    wins = generate_walkforward_windows(idx, train_size=100, test_size=20)
    assert len(wins) > 0
    for tr, te in wins:
        assert te.start == tr.stop  # no purge by default


def test_walkforward_pair_mean_reversion_pipeline() -> None:
    prices = load_prices_csv("data/sample_prices.csv")
    rep = run_walkforward_pair_mean_reversion(
        prices,
        "SPY",
        "IVV",
        MeanReversionConfig(lookback=20, entry_z=2.0, exit_z=0.5),
        WalkForwardConfig(
            train_size=120, test_size=30, step_size=30, adf_alpha=0.99, eg_alpha=0.99
        ),
    )
    assert len(rep.windows) > 0
    assert rep.backtest.equity_curve.shape[0] == prices.shape[0]


def test_walkforward_pair_event_backtest_pipeline() -> None:
    prices = load_prices_csv("data/sample_prices.csv")
    rep = run_walkforward_pair_mean_reversion_event_backtest(
        prices,
        "SPY",
        "IVV",
        MeanReversionConfig(lookback=20, entry_z=2.0, exit_z=0.5),
        WalkForwardConfig(
            train_size=120, test_size=30, step_size=30, adf_alpha=0.99, eg_alpha=0.99
        ),
        EventBacktestConfig(initial_cash=1_000_000.0, commission_bps=1.0),
    )
    assert len(rep.windows) > 0
    assert rep.backtest.equity_curve.shape[0] == prices.shape[0]


def test_pair_mean_reversion_event_backtest_single_window() -> None:
    prices = load_prices_csv("data/sample_prices.csv")
    diag, weights, result = run_pair_mean_reversion_event_backtest(
        prices,
        "SPY",
        "IVV",
        MeanReversionConfig(lookback=20, entry_z=2.0, exit_z=0.5),
        EventBacktestConfig(initial_cash=1_000_000.0),
    )
    assert np.isfinite(diag.beta)
    assert weights.shape == (prices.shape[0], 2)
    assert result.equity_curve.iloc[-1] > 0


def test_walkforward_basket_pipeline() -> None:
    prices = load_prices_csv("data/sample_basket_prices.csv")
    rep = run_walkforward_johansen_basket_event_backtest(
        prices,
        list(prices.columns),
        MeanReversionConfig(lookback=20, entry_z=1.5, exit_z=0.5),
        WalkForwardConfig(
            train_size=120, test_size=30, step_size=30, adf_alpha=0.99
        ),
        JohansenConfig(significance=0.10),
        EventBacktestConfig(initial_cash=1_000_000.0),
    )
    assert len(rep.windows) > 0
    assert rep.backtest.equity_curve.shape[0] == prices.shape[0]
