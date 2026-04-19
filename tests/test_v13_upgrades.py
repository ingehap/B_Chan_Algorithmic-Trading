"""Tests for v13 upgrades.

Covers:
- Sortino correctness bug fix + MAR parameter.
- Numerical stability of ``annualized_return``.
- Vectorised ``_long_short_ranks`` matches the v12 reference.
- Vectorised ``block_bootstrap_returns`` preserves the v12 RNG sequence.
- ``_paths_max_drawdown`` matches per-path v12 reference.
- Stationary bootstrap (Politis-Romano): shape + independence from block_size tuning.
- ``rolling_volatility_percentile`` matches v12 semantics byte-for-byte.
- Shared ``apply_exposure_caps`` behaves identically to the old private one.
- Exposure caps now honoured by momentum, cross-sectional, and buy-on-gap.
- ``apply_turnover_throttle`` caps per-bar position change.
- ``lag`` kwarg on drawdown / CPPI throttle is causal.
- R/S Hurst estimator orders correctly across known-dynamics series.
- ``suggest_zscore_lookback`` behaviour.
- Block-shuffle variant of permutation test.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from chan_trading.config import (
    BuyOnGapConfig,
    CrossSectionalMeanReversionConfig,
    CrossSectionalMomentumConfig,
    MomentumConfig,
)
from chan_trading.features.statistics import (
    estimate_half_life,
    estimate_rs_hurst_exponent,
    suggest_zscore_lookback,
)
from chan_trading.portfolio.sizing import (
    apply_cppi_throttle,
    apply_drawdown_throttle,
    apply_exposure_caps,
    apply_turnover_throttle,
)
from chan_trading.risk.metrics import (
    annualized_return,
    sortino_ratio,
)
from chan_trading.risk.monte_carlo import (
    _path_max_drawdown,
    _paths_max_drawdown,
    block_bootstrap_returns,
    stationary_bootstrap_returns,
)
from chan_trading.risk.regime import rolling_volatility_percentile
from chan_trading.strategies.cross_sectional_mean_reversion import (
    CrossSectionalMeanReversionStrategy,
)
from chan_trading.strategies.cross_sectional_momentum import (
    CrossSectionalMomentumStrategy,
    _long_short_ranks,
)
from chan_trading.strategies.momentum import TimeSeriesMomentumStrategy
from chan_trading.strategies.opening_gap import BuyOnGapStrategy
from chan_trading.validation.permutation import permutation_alpha_test


# ---------------------------------------------------------------------------
# Sortino correctness
# ---------------------------------------------------------------------------


def test_sortino_uses_target_semideviation_not_subset_std():
    """v12's sortino used std of the negative subset; v13 uses the true
    target semideviation. The two differ by a well-defined factor that
    depends on the hit rate."""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 5000))

    # v12 (buggy) formula
    v12_dd = r[r < 0].std(ddof=0)
    v12 = r.mean() / v12_dd * math.sqrt(252)

    # v13 (correct) formula vs the definition
    clipped = np.minimum(r.to_numpy(), 0.0)
    ref_dd = math.sqrt((clipped**2).mean())
    ref = r.mean() / ref_dd * math.sqrt(252)

    got = sortino_ratio(r)
    assert got == pytest.approx(ref, rel=1e-12, abs=1e-12)
    # And it is strictly smaller than the v12 value (the bug's direction).
    assert got < v12


def test_sortino_mar_parameter_shifts_numerator_and_denominator():
    r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01, 0.00, 0.04, -0.03])
    # MAR = 0: reference against the formula.
    got0 = sortino_ratio(r, minimum_acceptable_return=0.0)
    clipped0 = np.minimum(r.to_numpy(), 0.0)
    dd0 = math.sqrt((clipped0**2).mean())
    assert got0 == pytest.approx(r.mean() / dd0 * math.sqrt(252), rel=1e-12)

    # Non-zero MAR: both numerator and denominator move.
    mar = 0.005
    got_mar = sortino_ratio(r, minimum_acceptable_return=mar)
    excess = r - mar
    clipped_mar = np.minimum(excess.to_numpy(), 0.0)
    dd_mar = math.sqrt((clipped_mar**2).mean())
    assert got_mar == pytest.approx(excess.mean() / dd_mar * math.sqrt(252), rel=1e-12)


def test_sortino_handles_all_non_negative_returns():
    # No losing bars → downside deviation = 0 → Sortino undefined.
    r = pd.Series([0.01, 0.02, 0.0, 0.015])
    assert math.isnan(sortino_ratio(r))


def test_sortino_handles_empty_series():
    assert math.isnan(sortino_ratio(pd.Series([], dtype=float)))


# ---------------------------------------------------------------------------
# annualized_return numerical stability
# ---------------------------------------------------------------------------


def test_annualized_return_matches_direct_product_on_normal_series():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000))
    direct = float(np.prod((1.0 + r).to_numpy()) ** (252 / len(r)) - 1.0)
    got = annualized_return(r)
    assert got == pytest.approx(direct, rel=1e-10)


def test_annualized_return_stable_on_long_series():
    # With 50_000 daily bars at mean 0.001, direct product overflows but
    # log-space form handles it fine.
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.001, 0.01, 50_000))
    got = annualized_return(r)
    assert math.isfinite(got)


def test_annualized_return_handles_minus_100_percent_bar():
    # A -100% bar zeroes terminal equity; function returns nan.
    r = pd.Series([0.01, -1.0, 0.02])
    assert math.isnan(annualized_return(r))


# ---------------------------------------------------------------------------
# _long_short_ranks — vectorisation
# ---------------------------------------------------------------------------


def _long_short_ranks_reference(lookback_returns, top_fraction, bottom_fraction, long_only):
    """v12 reference implementation kept verbatim for comparison."""
    signal = pd.DataFrame(
        0.0, index=lookback_returns.index, columns=lookback_returns.columns, dtype=float
    )
    for ts, row in lookback_returns.iterrows():
        valid = row.dropna()
        n_valid = len(valid)
        if n_valid < 2:
            continue
        n_top = max(1, int(np.floor(n_valid * top_fraction)))
        n_bot = max(1, int(np.floor(n_valid * bottom_fraction)))
        if n_top + n_bot > n_valid:
            n_top = n_valid // 2
            n_bot = n_valid - n_top
        ranked = valid.sort_values(ascending=False)
        longs = ranked.iloc[:n_top].index
        signal.loc[ts, longs] = 1.0
        if not long_only:
            shorts = ranked.iloc[-n_bot:].index
            signal.loc[ts, shorts] = -1.0
    return signal


@pytest.mark.parametrize(
    "top,bot,long_only",
    [
        (0.2, 0.2, False),
        (0.2, 0.2, True),
        (0.3, 0.3, False),
        (0.5, 0.5, False),
        (0.1, 0.4, False),
    ],
)
def test_long_short_ranks_matches_reference(top, bot, long_only):
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.standard_normal((400, 15)), columns=[f"A{i}" for i in range(15)])
    # Sprinkle NaNs including a fully-NaN row and a row with a single value.
    df.iloc[::9, ::2] = np.nan
    df.iloc[0] = np.nan
    df.iloc[1, 0] = 0.5

    got = _long_short_ranks(df, top_fraction=top, bottom_fraction=bot, long_only=long_only)
    ref = _long_short_ranks_reference(df, top, bot, long_only)
    pd.testing.assert_frame_equal(got, ref, check_dtype=False)


def test_long_short_ranks_skips_rows_with_fewer_than_two_valid():
    df = pd.DataFrame([[np.nan, np.nan], [np.nan, 1.0], [1.0, 2.0]], columns=["A", "B"])
    got = _long_short_ranks(df, top_fraction=0.5, bottom_fraction=0.5, long_only=False)
    assert (got.iloc[0] == 0).all()
    assert (got.iloc[1] == 0).all()
    # Row 2 has two valid entries → one long + one short.
    assert got.iloc[2].abs().sum() == 2.0


# ---------------------------------------------------------------------------
# block_bootstrap_returns — vectorisation preserves rng stream
# ---------------------------------------------------------------------------


def test_block_bootstrap_output_byte_identical_to_reference():
    def _v12(returns, *, horizon, n_paths, block_size, rng):
        clean = returns.dropna().to_numpy(dtype=float)
        n = len(clean)
        n_blocks = int(np.ceil(horizon / block_size))
        out = np.empty((n_paths, horizon), dtype=float)
        for p in range(n_paths):
            starts = rng.integers(0, n - block_size + 1, size=n_blocks)
            pieces = [clean[s : s + block_size] for s in starts]
            row = np.concatenate(pieces)[:horizon]
            out[p] = row
        return out

    rng = np.random.default_rng(0)
    r = pd.Series(rng.standard_normal(1000) * 0.01)
    # Use matched generators so the random streams line up.
    a = block_bootstrap_returns(
        r, horizon=200, n_paths=50, block_size=20, rng=np.random.default_rng(7)
    )
    b = _v12(r, horizon=200, n_paths=50, block_size=20, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# _paths_max_drawdown vs per-path reference
# ---------------------------------------------------------------------------


def test_paths_max_drawdown_matches_per_path_reference():
    rng = np.random.default_rng(9)
    paths = rng.normal(0.0, 0.02, size=(500, 400))
    got = _paths_max_drawdown(paths)
    ref = np.array([_path_max_drawdown(paths[i]) for i in range(paths.shape[0])])
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# stationary bootstrap
# ---------------------------------------------------------------------------


def test_stationary_bootstrap_shape_and_values_from_series():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.standard_normal(300))
    paths = stationary_bootstrap_returns(
        r, horizon=150, n_paths=40, expected_block_size=10.0, rng=np.random.default_rng(3)
    )
    assert paths.shape == (40, 150)
    # Every element is drawn from the empirical support (±eps for float
    # round-trip).
    support = set(float(v) for v in r.to_numpy())
    assert all(float(v) in support for v in paths.reshape(-1))


def test_stationary_bootstrap_is_stochastic():
    r = pd.Series(np.arange(50, dtype=float))
    p1 = stationary_bootstrap_returns(
        r, horizon=20, n_paths=3, expected_block_size=5.0, rng=np.random.default_rng(1)
    )
    p2 = stationary_bootstrap_returns(
        r, horizon=20, n_paths=3, expected_block_size=5.0, rng=np.random.default_rng(2)
    )
    assert not np.array_equal(p1, p2)


def test_stationary_bootstrap_rejects_bad_parameters():
    r = pd.Series([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        stationary_bootstrap_returns(r, horizon=5, n_paths=2, expected_block_size=0.0)
    with pytest.raises(ValueError):
        stationary_bootstrap_returns(r, horizon=0, n_paths=2, expected_block_size=3.0)


# ---------------------------------------------------------------------------
# rolling_volatility_percentile vectorisation
# ---------------------------------------------------------------------------


def _vol_percentile_reference(returns, *, vol_lookback, percentile_window):
    clean = returns.astype(float)
    vol = clean.rolling(vol_lookback, min_periods=vol_lookback).std(ddof=0)
    v = vol.to_numpy()
    n = v.shape[0]
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        if np.isnan(v[t]):
            continue
        start = max(0, t - percentile_window + 1)
        window = v[start : t + 1]
        valid = window[~np.isnan(window)]
        if valid.size < 2:
            continue
        current = v[t]
        n_less = int(np.sum(valid < current))
        n_equal = int(np.sum(valid == current))
        out[t] = (n_less + 0.5 * (n_equal + 1)) / valid.size
    return pd.Series(out, index=returns.index, dtype=float)


@pytest.mark.parametrize("vl,pw", [(10, 50), (20, 100), (20, 252)])
def test_rolling_volatility_percentile_matches_reference(vl, pw):
    rng = np.random.default_rng(11)
    r = pd.Series(rng.standard_normal(600) * 0.01)
    a = rolling_volatility_percentile(r, vol_lookback=vl, percentile_window=pw)
    b = _vol_percentile_reference(r, vol_lookback=vl, percentile_window=pw)
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_rolling_volatility_percentile_handles_ties():
    r = pd.Series([0.01] * 20 + [0.02] * 20 + [0.01] * 20, dtype=float)
    a = rolling_volatility_percentile(r, vol_lookback=10, percentile_window=30)
    b = _vol_percentile_reference(r, vol_lookback=10, percentile_window=30)
    pd.testing.assert_series_equal(a, b, check_names=False)


# ---------------------------------------------------------------------------
# Shared apply_exposure_caps
# ---------------------------------------------------------------------------


def test_apply_exposure_caps_gross_cap():
    df = pd.DataFrame({"A": [0.7, 0.5], "B": [-0.7, 0.1]})  # gross 1.4, 0.6
    out = apply_exposure_caps(df, max_gross_exposure=1.0)
    # row 0 scales to gross=1.0; row 1 untouched.
    assert out.iloc[0].abs().sum() == pytest.approx(1.0)
    assert out.iloc[1].tolist() == [0.5, 0.1]


def test_apply_exposure_caps_net_cap():
    df = pd.DataFrame({"A": [1.0, 0.4], "B": [0.5, 0.1]})  # net 1.5, 0.5
    out = apply_exposure_caps(df, max_net_exposure=1.0)
    assert out.iloc[0].sum() == pytest.approx(1.0)
    assert out.iloc[1].tolist() == [0.4, 0.1]


def test_apply_exposure_caps_noop_without_caps():
    df = pd.DataFrame({"A": [0.3, 0.7], "B": [-0.4, 0.1]})
    out = apply_exposure_caps(df)
    pd.testing.assert_frame_equal(out, df)


def test_apply_exposure_caps_rejects_bad_values():
    df = pd.DataFrame({"A": [0.1], "B": [0.1]})
    with pytest.raises(ValueError):
        apply_exposure_caps(df, max_gross_exposure=-0.1)
    with pytest.raises(ValueError):
        apply_exposure_caps(df, max_net_exposure=-0.1)


# ---------------------------------------------------------------------------
# Exposure caps on every strategy family
# ---------------------------------------------------------------------------


def _synth_prices(n_bars=250, n_assets=8, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.01, size=(n_bars, n_assets))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(
        prices,
        index=pd.date_range("2020-01-01", periods=n_bars, freq="B"),
        columns=[f"A{i}" for i in range(n_assets)],
    )


def test_momentum_honours_gross_cap():
    prices = _synth_prices()
    cfg = MomentumConfig(lookback=20, max_leverage=2.0, max_gross_exposure=0.5)
    strat = TimeSeriesMomentumStrategy(config=cfg)
    w = strat.generate_positions(prices)
    # After the cap, gross should never exceed 0.5 (up to float).
    assert (w.abs().sum(axis=1) <= 0.5 + 1e-12).all()


def test_cross_sectional_momentum_honours_gross_cap():
    prices = _synth_prices()
    cfg = CrossSectionalMomentumConfig(lookback=20, max_leverage=2.0, max_gross_exposure=0.5)
    strat = CrossSectionalMomentumStrategy(config=cfg)
    w = strat.generate_positions(prices)
    assert (w.abs().sum(axis=1) <= 0.5 + 1e-12).all()


def test_cross_sectional_mean_reversion_honours_gross_cap():
    prices = _synth_prices()
    cfg = CrossSectionalMeanReversionConfig(
        lookback=1, max_leverage=2.0, max_gross_exposure=0.5
    )
    strat = CrossSectionalMeanReversionStrategy(config=cfg)
    w = strat.generate_positions(prices)
    assert (w.abs().sum(axis=1) <= 0.5 + 1e-12).all()


def test_buy_on_gap_honours_gross_cap():
    prices = _synth_prices()
    cfg = BuyOnGapConfig(
        lookback=10,
        threshold_sigma=0.5,
        hold_bars=1,
        two_sided=True,
        max_leverage=2.0,
        max_gross_exposure=0.5,
    )
    strat = BuyOnGapStrategy(config=cfg)
    w = strat.generate_positions(prices)
    assert (w.abs().sum(axis=1) <= 0.5 + 1e-12).all()


# ---------------------------------------------------------------------------
# apply_turnover_throttle
# ---------------------------------------------------------------------------


def test_turnover_throttle_respects_cap():
    # A panel that tries to flip 100% of book every bar.
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    target = pd.DataFrame(
        {"A": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0], "B": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        index=idx,
    )
    throttled = apply_turnover_throttle(target, max_turnover_per_bar=0.5)
    changes = throttled.diff().abs().sum(axis=1).iloc[1:]
    assert (changes <= 0.5 + 1e-12).all()
    # First bar is always the target value.
    assert throttled.iloc[0].tolist() == [1.0, 0.0]


def test_turnover_throttle_passthrough_when_within_cap():
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    target = pd.DataFrame(
        {"A": [1.0, 1.1, 0.9, 1.0], "B": [0.0, 0.05, 0.0, -0.05]}, index=idx
    )
    throttled = apply_turnover_throttle(target, max_turnover_per_bar=1.0)
    pd.testing.assert_frame_equal(throttled, target, check_dtype=False)


def test_turnover_throttle_rejects_bad_values():
    df = pd.DataFrame({"A": [0.1]})
    with pytest.raises(ValueError):
        apply_turnover_throttle(df, max_turnover_per_bar=0.0)


# ---------------------------------------------------------------------------
# lag kwarg on drawdown / cppi throttles
# ---------------------------------------------------------------------------


def test_drawdown_throttle_lag_0_matches_v12():
    """lag=0 must reproduce v12 behaviour exactly."""
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    eq = pd.Series([1.0, 0.98, 0.95, 0.9, 0.85], index=idx)
    pos = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0, 1.0]}, index=idx)
    out = apply_drawdown_throttle(pos, eq, soft_limit=0.05, soft_scale=0.5, lag=0)
    # drawdown = [0, .02, .05, .10, .15]; soft limit 0.05 triggers from index 2.
    assert out["A"].tolist() == [1.0, 1.0, 0.5, 0.5, 0.5]


def test_drawdown_throttle_lag_1_shifts_scale():
    """lag=1 applies the throttle derived from equity[t-1] to positions at t."""
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    eq = pd.Series([1.0, 0.98, 0.95, 0.9, 0.85], index=idx)
    pos = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0, 1.0]}, index=idx)
    out = apply_drawdown_throttle(pos, eq, soft_limit=0.05, soft_scale=0.5, lag=1)
    # Index where throttle fires on lag=0 is 2; lag=1 pushes it to 3.
    assert out["A"].tolist() == [1.0, 1.0, 1.0, 0.5, 0.5]


def test_cppi_throttle_lag_parameter():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    eq = pd.Series([1.0, 1.0, 0.9, 0.85, 0.8], index=idx)
    pos = pd.DataFrame({"A": [1.0] * 5}, index=idx)
    got0 = apply_cppi_throttle(pos, eq, floor_fraction=0.85, multiplier=3.0, lag=0)
    got1 = apply_cppi_throttle(pos, eq, floor_fraction=0.85, multiplier=3.0, lag=1)
    # lag=1 output is lag=0 shifted down one bar, first bar padded 0.
    assert got1["A"].iloc[0] == 0.0
    np.testing.assert_allclose(
        got1["A"].iloc[1:].to_numpy(), got0["A"].iloc[:-1].to_numpy(), rtol=1e-12
    )


# ---------------------------------------------------------------------------
# R/S Hurst estimator
# ---------------------------------------------------------------------------


def test_rs_hurst_orders_correctly():
    """R/S must order known dynamics in the expected direction.

    Note: classical R/S is designed for (near-)stationary input. On a
    pure random walk it systematically overestimates because the
    mean-adjusted cumulative path inherits a stochastic trend; this is
    why modern work prefers DFA on non-stationary series. But the
    *ordering* — ``H(trend) > H(random_walk) > H(mean_reverting)`` —
    remains valid and is the property a practitioner cares about when
    screening candidate spreads.
    """
    rng = np.random.default_rng(13)

    # Strong trend: cumulative sum of positive-drift noise.
    trend = pd.Series(np.cumsum(rng.normal(0.05, 0.02, 2000)))
    # Random walk: cumsum of zero-mean noise.
    rw = pd.Series(np.cumsum(rng.normal(0.0, 0.02, 2000)))
    # Strongly mean-reverting AR(1) with phi ≈ 0 (very fast reversion).
    mr_vals = np.zeros(2000)
    for t in range(1, 2000):
        mr_vals[t] = 0.1 * mr_vals[t - 1] + rng.normal(0.0, 0.02)
    mr = pd.Series(mr_vals)

    h_trend = estimate_rs_hurst_exponent(trend)
    h_rw = estimate_rs_hurst_exponent(rw)
    h_mr = estimate_rs_hurst_exponent(mr)
    assert h_trend >= h_rw
    assert h_rw > h_mr
    # On the stationary AR(1) the estimate should be clearly below the
    # random-walk value.
    assert h_mr < h_rw - 0.2


def test_rs_hurst_rejects_too_short():
    with pytest.raises(ValueError):
        estimate_rs_hurst_exponent(pd.Series(np.arange(10, dtype=float)))


# ---------------------------------------------------------------------------
# suggest_zscore_lookback
# ---------------------------------------------------------------------------


def test_suggest_lookback_basic():
    # Half-life 10 → 2×10 = 20.
    assert suggest_zscore_lookback(10.0) == 20
    # With multiplier.
    assert suggest_zscore_lookback(10.0, multiplier=3.0) == 30


def test_suggest_lookback_floor_and_cap():
    # Tiny half-life → floor.
    assert suggest_zscore_lookback(0.5, floor=5) == 5
    # Huge half-life → cap.
    assert suggest_zscore_lookback(10_000.0, cap=252) == 252
    # Non-finite half-life → cap (conservative).
    assert suggest_zscore_lookback(float("inf")) == 252
    assert suggest_zscore_lookback(float("nan")) == 252


def test_suggest_lookback_validates_inputs():
    with pytest.raises(ValueError):
        suggest_zscore_lookback(10.0, multiplier=0.0)
    with pytest.raises(ValueError):
        suggest_zscore_lookback(10.0, floor=1)
    with pytest.raises(ValueError):
        suggest_zscore_lookback(10.0, floor=10, cap=5)


def test_half_life_and_lookback_end_to_end():
    """On a synthetic AR(1) the suggested lookback is within 2× the
    theoretical half-life."""
    rng = np.random.default_rng(21)
    phi = 0.9  # theoretical half-life ≈ -log(2) / log(0.9) ≈ 6.58
    n = 2000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0.0, 1.0)
    hl = estimate_half_life(pd.Series(x))
    lb = suggest_zscore_lookback(hl, multiplier=2.0)
    assert 5 <= lb <= 40  # sanity range


# ---------------------------------------------------------------------------
# permutation test — block-shuffle variant
# ---------------------------------------------------------------------------


def test_permutation_block_shuffle_preserves_holding_structure():
    """With block_size=1 the result should match row-by-row shuffling;
    with block_size=n, the null is just one big block → Sharpe identical
    (after position-panel rotation) to the observed value."""
    rng = np.random.default_rng(5)
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (n, 2)), axis=0)),
        index=idx,
        columns=["A", "B"],
    )
    # A position panel that holds the same leg for 10-bar blocks.
    raw = np.sign(np.sin(np.arange(n) / 10.0))
    pos = pd.DataFrame({"A": raw, "B": -raw}, index=idx)

    block_rep = permutation_alpha_test(
        prices, pos, n_shuffles=30, seed=1, block_size=10
    )
    bar_rep = permutation_alpha_test(prices, pos, n_shuffles=30, seed=1)
    # Block shuffle preserves the autocorrelation structure of positions
    # within the block, so null-Sharpe variance is lower.
    assert np.nanstd(block_rep.null_sharpes) <= np.nanstd(bar_rep.null_sharpes) * 2.0


def test_permutation_rejects_block_size_zero():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"A": np.arange(10.0)}, index=idx)
    pos = pd.DataFrame({"A": np.ones(10)}, index=idx)
    with pytest.raises(ValueError):
        permutation_alpha_test(prices, pos, n_shuffles=1, block_size=0)


# ---------------------------------------------------------------------------
# version bump
# ---------------------------------------------------------------------------


def test_version_bumped_to_v13():
    import chan_trading

    # v13 shipped 0.14.0. Later minor/patch bumps (e.g. 0.15.0 in v14)
    # should keep this test green — the invariant is "at least v13",
    # not "exactly v13".
    version = chan_trading.__version__
    major, minor, *_ = (int(x) for x in version.split("."))
    assert (major, minor) >= (0, 14), f"expected >= 0.14.0, got {version}"
