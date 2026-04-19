"""Tests for v14 upgrades.

Covers:
1. ``newey_west_sharpe_variance`` / ``newey_west_sharpe_tstat``
2. ``rolling_sharpe_ratio``
3. ``information_ratio``
4. ``SimulatedBroker`` asymmetric ``half_spread_bps``
5. ``EventBacktestConfig.half_spread_bps`` wiring
6. Fully vectorised ``stationary_bootstrap_returns`` (bit-identity vs. v13)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chan_trading.backtest.broker import SimulatedBroker
from chan_trading.backtest.event_engine import run_event_backtest
from chan_trading.config import EventBacktestConfig
from chan_trading.risk.metrics import (
    information_ratio,
    newey_west_sharpe_tstat,
    newey_west_sharpe_variance,
    rolling_sharpe_ratio,
    sharpe_ratio,
)
from chan_trading.risk.monte_carlo import stationary_bootstrap_returns
from chan_trading.types import OrderEvent


# ---------------------------------------------------------------------------
# 1. Newey-West HAC variance of the Sharpe ratio
# ---------------------------------------------------------------------------


class TestNeweyWestSharpe:
    """HAC-adjusted Sharpe variance and t-stat."""

    def test_iid_series_matches_classical_variance(self):
        """For IID returns, HAC variance should collapse to ~sigma^2/T times annualisation."""
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.0008, 0.01, size=2000))
        var_hac = newey_west_sharpe_variance(r, lags=0, periods_per_year=252)
        # With lags=0 the HAC reduces exactly to sigma^2, so Var(SR_annual) = 252/n.
        # The true SR_annual asymptotic variance under IID is ~1 per unit time
        # but our definition scales by the full annualisation so we expect
        # var ≈ 252 / n. Compare to 1% tolerance.
        expected = 252.0 / len(r)
        assert abs(var_hac - expected) / expected < 0.01

    def test_positive_autocorr_inflates_hac_variance(self):
        """AR(1) with phi > 0 should give HAC variance larger than IID assumption."""
        rng = np.random.default_rng(1)
        # AR(1) with phi = 0.5 (strong positive serial correlation)
        n = 2000
        eps = rng.normal(0.0, 0.01, size=n)
        r = np.empty(n)
        r[0] = eps[0]
        for t in range(1, n):
            r[t] = 0.5 * r[t - 1] + eps[t]
        r_series = pd.Series(r + 0.0005)  # tiny positive drift
        # IID variance of Sharpe
        var_iid = newey_west_sharpe_variance(r_series, lags=0, periods_per_year=252)
        # HAC with many lags
        var_hac = newey_west_sharpe_variance(r_series, lags=20, periods_per_year=252)
        # For phi=0.5 the long-run variance multiplier is (1+phi)/(1-phi) = 3.
        # So HAC variance should be roughly 2-3x the IID variance.
        assert var_hac > var_iid * 1.5
        assert var_hac < var_iid * 5.0

    def test_negative_autocorr_reduces_hac_variance(self):
        """AR(1) with phi < 0 should give HAC variance SMALLER than IID assumption."""
        rng = np.random.default_rng(2)
        n = 2000
        eps = rng.normal(0.0, 0.01, size=n)
        r = np.empty(n)
        r[0] = eps[0]
        for t in range(1, n):
            r[t] = -0.4 * r[t - 1] + eps[t]
        r_series = pd.Series(r + 0.0005)
        var_iid = newey_west_sharpe_variance(r_series, lags=0, periods_per_year=252)
        var_hac = newey_west_sharpe_variance(r_series, lags=10, periods_per_year=252)
        # (1+phi)/(1-phi) with phi=-0.4 is 0.6/1.4 ≈ 0.43, so HAC should be
        # notably below IID. Demand at least a 20% reduction to leave
        # slack for finite-sample noise.
        assert var_hac < var_iid * 0.8

    def test_auto_lag_selection_finite_and_reasonable(self):
        """When lags=None the automatic Newey-West (1994) rule kicks in."""
        rng = np.random.default_rng(3)
        r = pd.Series(rng.normal(0.001, 0.01, size=1000))
        var_auto = newey_west_sharpe_variance(r, lags=None, periods_per_year=252)
        assert np.isfinite(var_auto)
        assert var_auto > 0

    def test_tstat_sign_follows_sharpe(self):
        """t-stat should be positive for positive-Sharpe series and vice versa."""
        rng = np.random.default_rng(4)
        pos = pd.Series(rng.normal(0.002, 0.01, size=1000))
        neg = pd.Series(rng.normal(-0.002, 0.01, size=1000))
        t_pos = newey_west_sharpe_tstat(pos, lags=5)
        t_neg = newey_west_sharpe_tstat(neg, lags=5)
        assert t_pos > 0
        assert t_neg < 0

    def test_tstat_rejects_inflated_sharpe_with_positive_autocorr(self):
        """A positive-autocorr series should have a smaller t-stat than the
        naive IID-equivalent would suggest.

        Specifically: sample Sharpe divided by HAC-SE should be smaller
        than sample Sharpe divided by naive-SE.
        """
        rng = np.random.default_rng(5)
        n = 2000
        eps = rng.normal(0.0005, 0.01, size=n)
        r = np.empty(n)
        r[0] = eps[0]
        for t in range(1, n):
            r[t] = 0.5 * r[t - 1] + eps[t]
        r_series = pd.Series(r)
        t_hac = newey_west_sharpe_tstat(r_series, lags=20)
        # Naive IID t-stat
        sr = sharpe_ratio(r_series)
        naive_var = 252.0 / n  # IID asymptotic
        t_iid = sr / np.sqrt(naive_var)
        # HAC t-stat should be clearly smaller in magnitude
        assert abs(t_hac) < abs(t_iid) * 0.9

    def test_raises_on_negative_lags(self):
        r = pd.Series(np.random.normal(0, 1, size=100))
        with pytest.raises(ValueError):
            newey_west_sharpe_variance(r, lags=-1)

    def test_raises_on_too_many_lags(self):
        r = pd.Series(np.random.normal(0, 1, size=50))
        with pytest.raises(ValueError):
            newey_west_sharpe_variance(r, lags=50)

    def test_returns_nan_on_short_series(self):
        r = pd.Series(np.random.normal(0, 1, size=5))
        assert np.isnan(newey_west_sharpe_variance(r, lags=2))


# ---------------------------------------------------------------------------
# 2. rolling_sharpe_ratio
# ---------------------------------------------------------------------------


class TestRollingSharpeRatio:
    def test_output_shape_and_index(self):
        rng = np.random.default_rng(10)
        r = pd.Series(rng.normal(0.001, 0.01, size=500), name="strategy")
        rs = rolling_sharpe_ratio(r, lookback=63)
        assert isinstance(rs, pd.Series)
        assert len(rs) == len(r)
        assert rs.index.equals(r.index)
        # First lookback-1 values should be NaN
        assert rs.iloc[:62].isna().all()
        assert rs.iloc[62:].notna().all()

    def test_rolling_equals_static_sharpe_at_end_of_window(self):
        """For a short tail after enough data, rolling SR at time t should match
        static SR computed on the last `lookback` observations."""
        rng = np.random.default_rng(11)
        r = pd.Series(rng.normal(0.001, 0.01, size=300))
        rs = rolling_sharpe_ratio(r, lookback=63)
        # Compare rolling SR at the last bar to static SR on the last 63 obs.
        last_window = r.iloc[-63:]
        static_sr = sharpe_ratio(last_window)
        assert abs(rs.iloc[-1] - static_sr) < 1e-10

    def test_raises_on_small_lookback(self):
        r = pd.Series(np.random.normal(0, 1, size=50))
        with pytest.raises(ValueError):
            rolling_sharpe_ratio(r, lookback=1)

    def test_risk_free_rate_affects_output(self):
        rng = np.random.default_rng(12)
        r = pd.Series(rng.normal(0.001, 0.01, size=500))
        rs0 = rolling_sharpe_ratio(r, lookback=63, risk_free_rate=0.0)
        rs5 = rolling_sharpe_ratio(r, lookback=63, risk_free_rate=0.05)
        # Non-zero rf should give different values (rf > 0 => SR smaller for
        # positive-mean series)
        assert not (rs0.dropna() == rs5.dropna()).all()


# ---------------------------------------------------------------------------
# 3. information_ratio
# ---------------------------------------------------------------------------


class TestInformationRatio:
    def test_zero_benchmark_equals_sharpe(self):
        """IR against zero-benchmark should equal Sharpe up to annualisation factor."""
        rng = np.random.default_rng(20)
        r = pd.Series(rng.normal(0.001, 0.01, size=500))
        b = pd.Series(0.0, index=r.index)
        ir = information_ratio(r, b)
        sr = sharpe_ratio(r)
        assert abs(ir - sr) < 1e-10

    def test_matching_benchmark_gives_zero_ir(self):
        """When r == b exactly, active returns are all zero => tracking error
        is zero => IR is NaN."""
        r = pd.Series([0.01, -0.01, 0.005, -0.005] * 100)
        ir = information_ratio(r, r)
        assert np.isnan(ir)

    def test_sign_of_ir_follows_alpha(self):
        """Positive alpha should give positive IR, negative alpha negative IR."""
        rng = np.random.default_rng(21)
        b = pd.Series(rng.normal(0.0005, 0.01, size=500))
        # Alpha series: constant positive shift + small noise
        r_pos = b + 0.0003 + rng.normal(0, 0.005, size=500)
        r_neg = b - 0.0003 + rng.normal(0, 0.005, size=500)
        assert information_ratio(r_pos, b) > 0
        assert information_ratio(r_neg, b) < 0

    def test_handles_misaligned_indices_via_inner_join(self):
        dates_r = pd.date_range("2020-01-01", periods=100)
        dates_b = pd.date_range("2020-01-15", periods=100)
        r = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates_r)
        b = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates_b)
        ir = information_ratio(r, b)
        # Overlap is 100 - 14 = 86 days; the function should not crash and
        # should return a finite value.
        assert np.isfinite(ir)

    def test_raises_on_bad_periods_per_year(self):
        r = pd.Series(np.random.normal(0, 1, 100))
        b = pd.Series(np.random.normal(0, 1, 100))
        with pytest.raises(ValueError):
            information_ratio(r, b, periods_per_year=0)


# ---------------------------------------------------------------------------
# 4. SimulatedBroker asymmetric half_spread
# ---------------------------------------------------------------------------


def _make_order(ts, asset, qty, price=100.0):
    return OrderEvent(
        timestamp=ts,
        asset=asset,
        quantity=qty,
        reference_price=price,
        target_weight=0.0,
    )


class TestAsymmetricHalfSpread:
    def test_zero_half_spread_preserves_v13(self):
        """Default half_spread_bps=0 should reproduce v13 fills exactly."""
        broker = SimulatedBroker(slippage_bps=5.0, half_spread_bps=0.0)
        ts = pd.Timestamp("2021-01-01")
        buy = broker.fill_order(_make_order(ts, "A", +10, 100.0), market_price=100.0)
        sell = broker.fill_order(_make_order(ts, "A", -10, 100.0), market_price=100.0)
        # With only slippage 5bps, buy fills at 100 * 1.0005 = 100.05,
        # sell fills at 100 * (1 - 0.0005) = 99.95.
        assert abs(buy.fill_price - 100.05) < 1e-10
        assert abs(sell.fill_price - 99.95) < 1e-10

    def test_half_spread_adds_to_slippage(self):
        """Buys cross the ask, sells cross the bid — both move against the trader."""
        broker = SimulatedBroker(slippage_bps=5.0, half_spread_bps=3.0)
        ts = pd.Timestamp("2021-01-01")
        buy = broker.fill_order(_make_order(ts, "A", +10, 100.0), market_price=100.0)
        sell = broker.fill_order(_make_order(ts, "A", -10, 100.0), market_price=100.0)
        # Total adverse rate = (5 + 3) / 10000 = 8bps
        # Buy:  100 * (1 + 0.0008) = 100.08
        # Sell: 100 * (1 - 0.0008) = 99.92
        assert abs(buy.fill_price - 100.08) < 1e-10
        assert abs(sell.fill_price - 99.92) < 1e-10

    def test_half_spread_only(self):
        """With slippage=0 and half_spread>0, only the spread is paid."""
        broker = SimulatedBroker(slippage_bps=0.0, half_spread_bps=10.0)
        ts = pd.Timestamp("2021-01-01")
        buy = broker.fill_order(_make_order(ts, "A", +10, 100.0), market_price=100.0)
        sell = broker.fill_order(_make_order(ts, "A", -10, 100.0), market_price=100.0)
        assert abs(buy.fill_price - 100.1) < 1e-10
        assert abs(sell.fill_price - 99.9) < 1e-10

    def test_negative_half_spread_rejected(self):
        with pytest.raises(ValueError):
            SimulatedBroker(half_spread_bps=-1.0)

    def test_event_engine_propagates_half_spread(self):
        """EventBacktestConfig.half_spread_bps should flow through to fills."""
        dates = pd.date_range("2020-01-01", periods=10)
        prices = pd.DataFrame({"X": np.linspace(100, 101, 10)}, index=dates)
        # Constant target weight of 0.5 every bar.
        weights = pd.DataFrame({"X": [0.5] * 10}, index=dates)

        cfg_plain = EventBacktestConfig(
            commission_bps=0.0, slippage_bps=5.0, half_spread_bps=0.0
        )
        cfg_spread = EventBacktestConfig(
            commission_bps=0.0, slippage_bps=5.0, half_spread_bps=10.0
        )
        res_plain = run_event_backtest(prices, weights, config=cfg_plain)
        res_spread = run_event_backtest(prices, weights, config=cfg_spread)
        # With a positive target weight, orders are buys; adding spread
        # should strictly increase the slippage cost of the first fill.
        sc_plain = res_plain.fill_log["slippage_cost"].sum()
        sc_spread = res_spread.fill_log["slippage_cost"].sum()
        assert sc_spread > sc_plain + 1e-10

    def test_event_engine_rejects_negative_half_spread_in_config(self):
        with pytest.raises(ValueError):
            EventBacktestConfig(half_spread_bps=-5.0)


# ---------------------------------------------------------------------------
# 5. stationary_bootstrap_returns — regression test pinning v13 semantics
# ---------------------------------------------------------------------------


class TestStationaryBootstrapRegression:
    """Regression test that pins the v13 implementation of
    ``stationary_bootstrap_returns``.

    *v14 context.* A fully vectorised variant (cumulative-max +
    anchor-lookup, eliminating the Python loop over ``horizon``) was
    prototyped, but benchmarking showed it was consistently **0.5-1.0×**
    the speed of the v13 loop on realistic sizes because the extra
    ``O(n_paths · horizon)`` intermediate arrays blow past L2 cache. The
    v13 implementation is therefore retained. This test preserves the
    old reference implementation inline so any future attempt to swap
    it out has to first prove bit-identity.
    """

    @staticmethod
    def _v13_reference(returns, *, horizon, n_paths, expected_block_size, rng):
        """Reference implementation that matches the v13 loop semantics."""
        clean = returns.dropna().to_numpy(dtype=float)
        n = len(clean)
        p_new = 1.0 / float(expected_block_size)
        new_block = rng.random(size=(n_paths, horizon)) < p_new
        new_block[:, 0] = True
        starts = rng.integers(0, n, size=(n_paths, horizon))
        indices = np.empty((n_paths, horizon), dtype=np.int64)
        indices[:, 0] = starts[:, 0] % n
        for t in range(1, horizon):
            cont = (indices[:, t - 1] + 1) % n
            indices[:, t] = np.where(new_block[:, t], starts[:, t] % n, cont)
        return clean[indices]

    def test_bit_identical_to_v13_reference(self):
        r = pd.Series(np.random.default_rng(99).normal(0.0005, 0.01, size=500))
        rng_v13 = np.random.default_rng(42)
        rng_v14 = np.random.default_rng(42)
        out_v13 = self._v13_reference(
            r, horizon=200, n_paths=50, expected_block_size=10.0, rng=rng_v13
        )
        out_v14 = stationary_bootstrap_returns(
            r, horizon=200, n_paths=50, expected_block_size=10.0, rng=rng_v14
        )
        np.testing.assert_array_equal(out_v14, out_v13)

    def test_bit_identical_on_edge_cases(self):
        """Horizon=1, n_paths=1, and very small expected_block_size."""
        r = pd.Series(np.random.default_rng(7).normal(0, 0.01, size=50))
        for horizon, n_paths, ebs in [
            (1, 1, 5.0),
            (10, 1, 2.0),
            (1, 100, 20.0),
            (500, 30, 1.5),
        ]:
            rng_a = np.random.default_rng(0)
            rng_b = np.random.default_rng(0)
            ref = self._v13_reference(
                r, horizon=horizon, n_paths=n_paths, expected_block_size=ebs, rng=rng_a
            )
            got = stationary_bootstrap_returns(
                r, horizon=horizon, n_paths=n_paths, expected_block_size=ebs, rng=rng_b
            )
            np.testing.assert_array_equal(got, ref)

    def test_output_values_come_from_input(self):
        """Every bootstrapped value should be drawn from the input series."""
        r = pd.Series([0.01, -0.02, 0.003, -0.015, 0.008, -0.004])
        out = stationary_bootstrap_returns(
            r, horizon=50, n_paths=20, expected_block_size=5.0,
            rng=np.random.default_rng(0),
        )
        assert set(np.unique(out.ravel())).issubset(set(r.values))


# ---------------------------------------------------------------------------
# Version bump
# ---------------------------------------------------------------------------


def test_version_bumped_to_v14():
    import chan_trading

    # v14 ships 0.15.0. Future minor/patch bumps keep this green.
    version = chan_trading.__version__
    major, minor, *_ = (int(x) for x in version.split("."))
    assert (major, minor) >= (0, 15), f"expected >= 0.15.0, got {version}"
