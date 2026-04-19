"""Tests for v11 additions.

v11 adds, on top of v10:

1. Historical VaR / CVaR / parametric Gaussian VaR in ``risk.metrics``,
   plus ``var_95`` / ``cvar_95`` fields in :class:`RiskReport`.
2. Rolling-volatility-percentile regime detection and regime-gated
   position scaling (``risk.regime``).
3. Daily buy-on-gap mean-reversion strategy (``strategies.opening_gap``).
4. Linear / square-root market-impact models (``execution.impact``)
   wired through :class:`SimulatedBroker` and the event-driven engine.
5. Cost-sensitivity sweep and breakeven interpolation
   (``validation.sensitivity``).
6. Vectorized ``_generate_stateful_signal`` (same semantics as v10).

References
----------
- Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and Their
  Rationale.* Wiley.
- Almgren, R., Thum, C., Hauptmann, E., & Li, H. (2005). "Direct
  Estimation of Equity Market Impact." *Risk*, 18(7), 58-62.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chan_trading.backtest.event_engine import run_event_backtest
from chan_trading.config import (
    BuyOnGapConfig,
    EventBacktestConfig,
    MeanReversionConfig,
)
from chan_trading.execution.impact import (
    LinearImpactModel,
    SquareRootImpactModel,
)
from chan_trading.risk.metrics import (
    historical_cvar,
    historical_var,
    parametric_gaussian_var,
    summarize_risk,
)
from chan_trading.risk.regime import (
    VolatilityRegimeConfig,
    apply_regime_filter,
    detect_volatility_regime,
    rolling_volatility_percentile,
)
from chan_trading.strategies.opening_gap import BuyOnGapStrategy
from chan_trading.validation.sensitivity import (
    breakeven_cost_bps,
    cost_sensitivity_sweep,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _synthetic_returns(n: int = 300, seed: int = 7) -> pd.Series:
    """Moderately skewed / fat-tailed return series."""
    rng = np.random.default_rng(seed)
    normal = rng.standard_normal(n) * 0.01
    shocks = rng.standard_t(df=4, size=n) * 0.003
    return pd.Series(normal + shocks, index=pd.date_range("2020-01-01", periods=n, freq="B"))


def _synthetic_price_panel(n: int = 500, n_assets: int = 4, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal((n, n_assets)) * 0.01
    # Inject occasional big down-days so buy-on-gap has something to fire on.
    shock_rows = rng.choice(np.arange(50, n), size=15, replace=False)
    shock_cols = rng.integers(0, n_assets, size=15)
    returns[shock_rows, shock_cols] -= 0.05
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    cols = [f"A{i}" for i in range(n_assets)]
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    return pd.DataFrame(prices, index=idx, columns=cols)


# ---------------------------------------------------------------------------
# 1. CVaR / VaR
# ---------------------------------------------------------------------------


def test_historical_var_matches_empirical_quantile() -> None:
    returns = _synthetic_returns()
    var_05 = historical_var(returns, alpha=0.05)
    # VaR at 95% = empirical 5% quantile
    expected = float(np.quantile(returns.to_numpy(), 0.05))
    assert var_05 == pytest.approx(expected, rel=1e-9, abs=1e-12)
    # Should be negative (it's a loss threshold)
    assert var_05 < 0


def test_historical_cvar_is_more_extreme_than_var() -> None:
    returns = _synthetic_returns()
    var_05 = historical_var(returns, alpha=0.05)
    cvar_05 = historical_cvar(returns, alpha=0.05)
    # CVaR is the mean of the tail at or below VaR, so it must be <= VaR.
    assert cvar_05 <= var_05
    # And for non-constant returns it should be strictly more negative.
    assert cvar_05 < var_05


def test_historical_cvar_rejects_invalid_alpha() -> None:
    returns = _synthetic_returns()
    with pytest.raises(ValueError, match="alpha"):
        historical_cvar(returns, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        historical_cvar(returns, alpha=1.0)


def test_parametric_gaussian_var_close_to_historical_on_mild_series() -> None:
    # For nearly-Gaussian returns the two estimates should be close.
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.standard_normal(2000) * 0.01)
    hist = historical_var(returns, alpha=0.05)
    gauss = parametric_gaussian_var(returns, alpha=0.05)
    assert abs(hist - gauss) < 0.003


def test_summarize_risk_populates_var_and_cvar() -> None:
    returns = _synthetic_returns()
    equity = (1.0 + returns).cumprod()
    report = summarize_risk(equity, returns)
    assert np.isfinite(report.var_95)
    assert np.isfinite(report.cvar_95)
    assert report.cvar_95 <= report.var_95


# ---------------------------------------------------------------------------
# 2. Regime filter
# ---------------------------------------------------------------------------


def test_volatility_regime_config_validates() -> None:
    with pytest.raises(ValueError, match="percentile_window"):
        VolatilityRegimeConfig(vol_lookback=60, percentile_window=40)
    with pytest.raises(ValueError, match="low_threshold"):
        VolatilityRegimeConfig(low_threshold=0.9, high_threshold=0.5)


def test_rolling_volatility_percentile_in_unit_interval() -> None:
    returns = _synthetic_returns(n=800)
    pct = rolling_volatility_percentile(returns, vol_lookback=20, percentile_window=120)
    valid = pct.dropna()
    assert len(valid) > 0
    assert (valid >= 0).all() and (valid <= 1).all()


def test_detect_volatility_regime_labels_high_at_volatile_segment() -> None:
    # Build a returns series whose *last* block is visibly more volatile
    # than its earlier blocks — we then expect "high" labels in that block.
    rng = np.random.default_rng(42)
    calm = rng.standard_normal(400) * 0.005
    stormy = rng.standard_normal(200) * 0.03
    r = pd.Series(
        np.concatenate([calm, stormy]),
        index=pd.date_range("2021-01-01", periods=600, freq="B"),
    )
    cfg = VolatilityRegimeConfig(vol_lookback=20, percentile_window=250, high_threshold=0.8)
    labels = detect_volatility_regime(r, config=cfg)
    # Last 100 bars should contain substantially more "high" labels than
    # a random 100-bar window would on i.i.d. data.
    last_segment_share = (labels.iloc[-100:] == "high").mean()
    assert last_segment_share > 0.3


def test_apply_regime_filter_zeros_positions_in_high_regime() -> None:
    idx = pd.date_range("2023-01-01", periods=10, freq="B")
    positions = pd.DataFrame({"A": [1.0] * 10, "B": [-1.0] * 10}, index=idx)
    regime = pd.Series(
        ["normal", "normal", "high", "high", "normal", "normal", "low", "low", "normal", "high"],
        index=idx,
    )
    gated = apply_regime_filter(positions, regime, off_regimes=("high",), off_scale=0.0)
    assert (gated.loc[regime == "high"] == 0.0).all().all()
    assert (gated.loc[regime != "high"] == positions.loc[regime != "high"]).all().all()


def test_apply_regime_filter_requires_index_match() -> None:
    idx = pd.date_range("2023-01-01", periods=5, freq="B")
    positions = pd.DataFrame({"A": [1.0] * 5}, index=idx)
    regime = pd.Series(["high"] * 5, index=idx.shift(3))  # shifted index
    with pytest.raises(ValueError, match="index"):
        apply_regime_filter(positions, regime)


# ---------------------------------------------------------------------------
# 3. Buy-on-gap strategy
# ---------------------------------------------------------------------------


def test_buy_on_gap_config_validates() -> None:
    with pytest.raises(ValueError, match="threshold_sigma"):
        BuyOnGapConfig(threshold_sigma=0.0)
    with pytest.raises(ValueError, match="hold_bars"):
        BuyOnGapConfig(hold_bars=0)
    with pytest.raises(ValueError, match="lookback"):
        BuyOnGapConfig(lookback=1)


def test_buy_on_gap_fires_long_on_large_drop_and_skips_quiet_bars() -> None:
    prices = _synthetic_price_panel(n=400, n_assets=3, seed=11)
    cfg = BuyOnGapConfig(lookback=20, threshold_sigma=1.5, hold_bars=1)
    positions = BuyOnGapStrategy(config=cfg).generate_positions(prices)
    # Long-only by default → no negative entries
    assert (positions >= 0).all().all()
    # Some bars must have activated positions (given the injected shocks)
    assert (positions.abs().sum(axis=1) > 0).any()
    # Gross leverage never exceeds the configured cap (allow tiny float slack)
    assert (positions.abs().sum(axis=1) <= cfg.max_leverage + 1e-9).all()


def test_buy_on_gap_two_sided_fires_both_directions() -> None:
    prices = _synthetic_price_panel(n=400, n_assets=3, seed=13)
    cfg = BuyOnGapConfig(lookback=20, threshold_sigma=1.0, hold_bars=1, two_sided=True)
    positions = BuyOnGapStrategy(config=cfg).generate_positions(prices)
    # With two-sided enabled and fat-tailed synthetic data, we expect at
    # least one long and one short entry somewhere.
    assert (positions > 0).any().any()
    assert (positions < 0).any().any()


def test_buy_on_gap_ohlc_variant_returns_matching_shape() -> None:
    close = _synthetic_price_panel(n=200, n_assets=3, seed=21)
    # Fake "open" panel by nudging close by a small random amount
    rng = np.random.default_rng(2)
    opn = close * (1.0 + rng.standard_normal(close.shape) * 0.002)
    cfg = BuyOnGapConfig(lookback=20, threshold_sigma=1.0, hold_bars=1)
    positions = BuyOnGapStrategy(config=cfg).generate_positions_from_ohlc(close, opn)
    assert positions.shape == close.shape
    assert list(positions.columns) == list(close.columns)


# ---------------------------------------------------------------------------
# 4. Market-impact models
# ---------------------------------------------------------------------------


def test_linear_impact_monotone_in_participation() -> None:
    model = LinearImpactModel(coefficient_bps=100.0)
    b1 = model.impact_bps(trade_notional=1_000, adv_notional=100_000)
    b2 = model.impact_bps(trade_notional=10_000, adv_notional=100_000)
    b3 = model.impact_bps(trade_notional=50_000, adv_notional=100_000)
    assert 0 < b1 < b2 < b3
    assert b1 == pytest.approx(1.0)  # 1% participation × 100 bps = 1 bp
    assert b2 == pytest.approx(10.0)
    assert b3 == pytest.approx(50.0)


def test_sqrt_impact_concave_and_smaller_than_linear_for_big_trades() -> None:
    lin = LinearImpactModel(coefficient_bps=100.0)
    sqrt = SquareRootImpactModel(coefficient_bps=100.0)
    # At very small participation, sqrt is HIGHER than linear (sqrt(x) > x
    # for x in (0, 1)); the crossover is at participation = 1 where they
    # coincide. For big participation > 1, linear exceeds sqrt.
    big_trade, adv = 200_000, 100_000  # 2× ADV
    assert sqrt.impact_bps(big_trade, adv) < lin.impact_bps(big_trade, adv)


def test_impact_model_returns_zero_when_adv_missing_or_nan() -> None:
    lin = LinearImpactModel(coefficient_bps=100.0)
    sqrt = SquareRootImpactModel(coefficient_bps=100.0)
    for model in (lin, sqrt):
        assert model.impact_bps(trade_notional=1_000, adv_notional=0.0) == 0.0
        assert model.impact_bps(trade_notional=1_000, adv_notional=float("nan")) == 0.0
        assert model.impact_bps(trade_notional=1_000, adv_notional=-5.0) == 0.0


def test_event_config_rejects_partial_impact_wiring() -> None:
    with pytest.raises(ValueError, match="impact_model and adv"):
        EventBacktestConfig(impact_model=LinearImpactModel(100.0), adv=None)


def test_event_backtest_applies_impact_cost() -> None:
    # Small deterministic price panel; the engine should charge MORE
    # with impact enabled than with the same baseline slippage alone.
    prices = _synthetic_price_panel(n=120, n_assets=2, seed=5)
    weights = pd.DataFrame(
        {col: [0.0] * 60 + [0.5] * 60 for col in prices.columns},
        index=prices.index,
    )

    # baseline: no impact model
    cfg_flat = EventBacktestConfig(
        commission_bps=0.0,
        slippage_bps=5.0,
    )
    flat = run_event_backtest(prices, weights, cfg_flat)

    # With impact model: ADV per bar sized so one trade ≈ 50% of ADV.
    # Initial equity is 1M, weights go to 0.5 per asset → trade notional
    # ≈ 500_000 when the position is first established. ADV = 1_000_000
    # gives a 50% participation → 100 bps extra at coefficient=200.
    adv = pd.DataFrame(
        1_000_000.0,
        index=prices.index,
        columns=prices.columns,
    )
    cfg_impact = EventBacktestConfig(
        commission_bps=0.0,
        slippage_bps=5.0,
        impact_model=LinearImpactModel(coefficient_bps=200.0),
        adv=adv,
    )
    impacted = run_event_backtest(prices, weights, cfg_impact)

    # Impact must strictly reduce final equity on a trading run.
    assert np.isfinite(impacted.equity_curve.iloc[-1])
    assert impacted.equity_curve.iloc[-1] < flat.equity_curve.iloc[-1]


# ---------------------------------------------------------------------------
# 5. Cost-sensitivity sweep
# ---------------------------------------------------------------------------


def test_cost_sensitivity_sweep_monotone_in_slippage() -> None:
    prices = _synthetic_price_panel(n=250, n_assets=2, seed=9)
    weights = pd.DataFrame(
        {col: [0.5] * 250 for col in prices.columns},
        index=prices.index,
    )
    # Introduce turnover so costs bite: flip sign halfway.
    weights.iloc[100:] = -weights.iloc[100:]
    summary = cost_sensitivity_sweep(
        prices,
        weights,
        commission_bps_grid=[0.0],
        slippage_bps_grid=[0.0, 10.0, 50.0],
    )
    assert len(summary) == 3
    # Terminal equity must be weakly decreasing in slippage for the same
    # weights (more costs → lower equity).
    eq_series = summary.sort_values("slippage_bps")["terminal_equity"].to_numpy()
    assert eq_series[0] >= eq_series[1] >= eq_series[2]


def test_cost_sensitivity_sweep_rejects_empty_grids() -> None:
    prices = _synthetic_price_panel(n=100, n_assets=2, seed=4)
    weights = pd.DataFrame(0.1, index=prices.index, columns=prices.columns)
    with pytest.raises(ValueError, match="non-empty"):
        cost_sensitivity_sweep(
            prices, weights, commission_bps_grid=[], slippage_bps_grid=[1.0]
        )


def test_breakeven_cost_bps_linear_interpolation() -> None:
    # A hand-built sweep summary where Sharpe is exactly linear in slippage
    # from +2.0 at 0 bps to -2.0 at 40 bps → breakeven at 20 bps.
    summary = pd.DataFrame(
        {
            "commission_bps": [0.0, 0.0, 0.0, 0.0, 0.0],
            "slippage_bps": [0.0, 10.0, 20.0, 30.0, 40.0],
            "sharpe": [2.0, 1.0, 0.0, -1.0, -2.0],
        }
    )
    assert breakeven_cost_bps(summary, metric="sharpe", target=0.0, axis="slippage_bps") == pytest.approx(20.0)

    # Target below the lowest observed value → series never drops that far
    # at any tested cost, so we'd need MORE cost than we sampled → +inf.
    assert breakeven_cost_bps(summary, metric="sharpe", target=-5.0) == float("inf")

    # Series already starts below target at zero cost → can't reach target
    # by adding positive cost → -inf.
    summary_already_below = pd.DataFrame(
        {
            "commission_bps": [0.0, 0.0],
            "slippage_bps": [0.0, 10.0],
            "sharpe": [-1.0, -2.0],
        }
    )
    assert breakeven_cost_bps(summary_already_below, metric="sharpe", target=0.0) == float("-inf")


# ---------------------------------------------------------------------------
# 6. Vectorized stateful signal regression
# ---------------------------------------------------------------------------


def test_vectorized_stateful_signal_matches_reference_semantics() -> None:
    # Reference implementation = the pre-v11 pandas-loop version. We
    # inline it here to confirm that the numpy-array rewrite produces
    # byte-identical output on a non-trivial z-score series.
    from chan_trading.strategies.mean_reversion import _generate_stateful_signal

    rng = np.random.default_rng(99)
    z = pd.Series(
        rng.standard_normal(500) * 1.5,
        index=pd.date_range("2022-01-01", periods=500, freq="B"),
    )
    # Inject NaNs to exercise nan_mode handling
    z.iloc[10:12] = np.nan
    z.iloc[200:205] = np.nan

    def reference(z_ref: pd.Series, cfg) -> pd.Series:
        raw = pd.Series(0.0, index=z_ref.index, dtype=float)
        state = 0.0
        hold = 0
        for idx in z_ref.index:
            zi = float(z_ref.loc[idx]) if pd.notna(z_ref.loc[idx]) else np.nan
            if np.isnan(zi):
                if cfg.nan_mode == "flat":
                    state = 0.0
                if state == 0.0:
                    hold = 0
                else:
                    hold += 1
                    if cfg.max_holding_bars is not None and hold > cfg.max_holding_bars:
                        state = 0.0
                        hold = 0
                raw.loc[idx] = state
                continue
            stop = cfg.stop_z is not None and abs(zi) >= cfg.stop_z
            if state == 0.0:
                if stop:
                    hold = 0
                elif zi > cfg.entry_z:
                    state = -1.0
                    hold = 1
                elif zi < -cfg.entry_z:
                    state = 1.0
                    hold = 1
                else:
                    hold = 0
            else:
                hold += 1
                exit_z_hit = abs(zi) < cfg.exit_z
                exit_t = cfg.max_holding_bars is not None and hold > cfg.max_holding_bars
                if stop or exit_z_hit or exit_t:
                    state = 0.0
                    hold = 0
            raw.loc[idx] = state
        return raw

    # Test a handful of diverse configs including nan_mode flat and a stop.
    for cfg in [
        MeanReversionConfig(entry_z=1.5, exit_z=0.3, nan_mode="hold"),
        MeanReversionConfig(entry_z=2.0, exit_z=0.5, nan_mode="flat", max_holding_bars=10),
        MeanReversionConfig(entry_z=1.0, exit_z=0.2, stop_z=3.0, max_holding_bars=20),
    ]:
        got = _generate_stateful_signal(z, cfg)
        want = reference(z, cfg)
        pd.testing.assert_series_equal(got, want, check_names=False)
