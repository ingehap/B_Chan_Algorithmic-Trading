"""Tests for v12 additions.

v12 adds, on top of v11:

1. Profit factor, Omega ratio, gain-to-pain ratio in ``risk.metrics`` and
   corresponding fields on :class:`RiskReport`.
2. Trend / drift regime detector (``risk.regime.detect_trend_regime``,
   ``apply_trend_filter``) as a complement to the v11 volatility regime.
3. Vectorised ``rolling_volatility_percentile`` — byte-identical output
   at ~10x throughput compared with the v11 ``.apply(lambda)`` version.
4. Borrow and cash-interest costs in the vectorised
   ``backtest.engine.run_backtest`` (parity with the event engine).
5. Multi-strategy ensemble combiner (``portfolio.ensemble``) with equal,
   inverse-vol, risk-parity and custom weight schemes.
6. Futures continuous-contract helpers (``features.futures``) — ratio
   and difference adjustments, roll-return decomposition, calendar
   spread.
7. Permutation / shuffle alpha significance test
   (``validation.permutation``).

References
----------
- Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and Their
  Rationale.* Wiley.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chan_trading.backtest.engine import run_backtest
from chan_trading.features.futures import (
    build_continuous_contract,
    calendar_spread,
    decompose_futures_returns,
)
from chan_trading.portfolio.ensemble import (
    EnsembleConfig,
    combine_strategies,
    strategy_correlation_matrix,
)
from chan_trading.risk.metrics import (
    gain_to_pain_ratio,
    omega_ratio,
    profit_factor,
    summarize_risk,
)
from chan_trading.risk.regime import (
    TrendRegimeConfig,
    apply_trend_filter,
    detect_trend_regime,
    rolling_volatility_percentile,
)
from chan_trading.validation.permutation import permutation_alpha_test


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _returns(n: int = 250, seed: int = 7, vol: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.standard_normal(n) * vol,
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def _price_panel(n: int = 400, k: int = 3, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((n, k)) * 0.01
    p = 100.0 * np.cumprod(1.0 + r, axis=0)
    return pd.DataFrame(
        p,
        index=pd.date_range("2019-01-01", periods=n, freq="B"),
        columns=[f"A{i}" for i in range(k)],
    )


# ---------------------------------------------------------------------------
# 1. Profit factor, omega, gain-to-pain
# ---------------------------------------------------------------------------


def test_profit_factor_hand_checked() -> None:
    r = pd.Series([0.10, -0.05, 0.20, -0.10, 0.05])
    # gains = 0.10 + 0.20 + 0.05 = 0.35
    # losses = 0.05 + 0.10 = 0.15
    # pf = 0.35 / 0.15
    assert profit_factor(r) == pytest.approx(0.35 / 0.15)


def test_profit_factor_infinite_when_no_losses() -> None:
    r = pd.Series([0.01, 0.02, 0.03])
    assert profit_factor(r) == float("inf")


def test_profit_factor_nan_on_empty() -> None:
    assert np.isnan(profit_factor(pd.Series([], dtype=float)))


def test_omega_ratio_zero_threshold_matches_profit_factor() -> None:
    r = pd.Series([0.10, -0.05, 0.20, -0.10, 0.05])
    assert omega_ratio(r, threshold=0.0) == pytest.approx(profit_factor(r))


def test_omega_ratio_nonzero_threshold() -> None:
    # Above threshold 0.02: excess = [0.08, -0.07, 0.18, -0.12, 0.03]
    # Up area = 0.08 + 0.18 + 0.03 = 0.29; down area = 0.07 + 0.12 = 0.19
    r = pd.Series([0.10, -0.05, 0.20, -0.10, 0.05])
    assert omega_ratio(r, threshold=0.02) == pytest.approx(0.29 / 0.19)


def test_gain_to_pain_hand_checked() -> None:
    r = pd.Series([0.10, -0.05, 0.20, -0.10, 0.05])
    # sum = 0.20, pain = 0.15
    assert gain_to_pain_ratio(r) == pytest.approx(0.20 / 0.15)


def test_gain_to_pain_infinite_when_no_losses() -> None:
    r = pd.Series([0.01, 0.02])
    assert gain_to_pain_ratio(r) == float("inf")


def test_summarize_risk_populates_new_fields() -> None:
    r = _returns(n=200, seed=3)
    equity = (1.0 + r).cumprod()
    report = summarize_risk(equity, r)
    assert np.isfinite(report.profit_factor)
    assert np.isfinite(report.omega_ratio)
    assert np.isfinite(report.gain_to_pain_ratio)
    # Profit factor and omega at threshold 0 must be identical.
    assert report.profit_factor == pytest.approx(report.omega_ratio)


# ---------------------------------------------------------------------------
# 2. Trend regime
# ---------------------------------------------------------------------------


def test_trend_regime_config_validates() -> None:
    with pytest.raises(ValueError, match="lookback"):
        TrendRegimeConfig(lookback=1)
    with pytest.raises(ValueError, match="threshold"):
        TrendRegimeConfig(threshold=-0.001)


def test_detect_trend_regime_labels_bull_and_bear() -> None:
    # Clear bull window followed by clear bear window
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    r = pd.Series(
        np.concatenate([np.full(100, 0.005), np.full(100, -0.005)]),
        index=idx,
    )
    labels = detect_trend_regime(r, config=TrendRegimeConfig(lookback=30, threshold=0.001))
    # With lookback=30 + min_periods=30, bars 0..28 have NaN rolling mean → neutral.
    assert (labels.iloc[:29] == "neutral").all()
    # Deep into the bull regime (bars 60-99) the label must be bull.
    assert (labels.iloc[60:100] == "bull").all()
    # Deep into the bear regime (bars 160-199) the label must be bear.
    assert (labels.iloc[160:200] == "bear").all()


def test_detect_trend_regime_neutral_on_flat_drift() -> None:
    r = _returns(n=300, seed=5)  # mean ~ 0
    labels = detect_trend_regime(r, config=TrendRegimeConfig(lookback=30, threshold=0.005))
    # A 0.5%-per-day drift hurdle on mean-zero noise should be almost
    # always neutral.
    share = (labels == "neutral").mean()
    assert share > 0.9


def test_apply_trend_filter_flattens_bear() -> None:
    idx = pd.date_range("2023-01-01", periods=6, freq="B")
    positions = pd.DataFrame({"A": [1.0] * 6, "B": [-1.0] * 6}, index=idx)
    regime = pd.Series(["bull", "neutral", "bear", "bear", "neutral", "bull"], index=idx)
    gated = apply_trend_filter(positions, regime, off_regimes=("bear",), off_scale=0.0)
    assert (gated.loc[regime == "bear"] == 0.0).all().all()
    assert (gated.loc[regime != "bear"] == positions.loc[regime != "bear"]).all().all()


def test_apply_trend_filter_rejects_unknown_regime() -> None:
    idx = pd.date_range("2023-01-01", periods=3, freq="B")
    positions = pd.DataFrame({"A": [1.0] * 3}, index=idx)
    regime = pd.Series(["bull"] * 3, index=idx)
    with pytest.raises(ValueError, match="off_regimes"):
        apply_trend_filter(positions, regime, off_regimes=("mystery",))


# ---------------------------------------------------------------------------
# 3. Vectorised vol-percentile regression
# ---------------------------------------------------------------------------


def test_vectorised_vol_percentile_matches_reference() -> None:
    def ref(returns: pd.Series, *, vol_lookback: int, percentile_window: int) -> pd.Series:
        vol = returns.rolling(vol_lookback, min_periods=vol_lookback).std(ddof=0)
        return vol.rolling(percentile_window, min_periods=vol_lookback).apply(
            lambda w: float(pd.Series(w).rank(pct=True).iloc[-1]),
            raw=False,
        )

    r = _returns(n=600, seed=13)
    for vl, pw in [(20, 120), (10, 60), (30, 252)]:
        got = rolling_volatility_percentile(r, vol_lookback=vl, percentile_window=pw)
        want = ref(r, vol_lookback=vl, percentile_window=pw)
        diff = (got - want).dropna().abs().max()
        assert diff == pytest.approx(0.0, abs=1e-12)


def test_vectorised_vol_percentile_handles_ties() -> None:
    # Force ties in the vol series by replacing a block of returns with
    # identical values; ensure the pandas-rank equivalence still holds.
    r = _returns(n=400, seed=21).copy()
    r.iloc[100:120] = r.iloc[50]
    got = rolling_volatility_percentile(r, vol_lookback=10, percentile_window=60)
    vol = r.rolling(10, min_periods=10).std(ddof=0)
    want = vol.rolling(60, min_periods=10).apply(
        lambda w: float(pd.Series(w).rank(pct=True).iloc[-1]), raw=False
    )
    diff = (got - want).dropna().abs().max()
    assert diff == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 4. Vectorised engine financing costs
# ---------------------------------------------------------------------------


def test_run_backtest_backward_compat_when_costs_zero() -> None:
    # With both new args zero, the v12 engine must match the v11 behaviour
    # byte-for-byte on a non-trivial example.
    prices = _price_panel(n=150, k=2, seed=4)
    positions = pd.DataFrame(
        {"A0": np.linspace(0.0, 0.5, 150), "A1": np.linspace(0.5, 0.0, 150)},
        index=prices.index,
    )
    bt_default = run_backtest(prices, positions, transaction_cost_bps=2.0)
    bt_zero_financing = run_backtest(
        prices,
        positions,
        transaction_cost_bps=2.0,
        borrow_bps_annual=0.0,
        cash_interest_bps_annual=0.0,
    )
    pd.testing.assert_series_equal(
        bt_default.equity_curve, bt_zero_financing.equity_curve
    )


def test_run_backtest_borrow_reduces_equity_on_shorts() -> None:
    prices = _price_panel(n=200, k=2, seed=6)
    # Persistent short in A1 for half the sample → borrow cost should bite.
    positions = pd.DataFrame(
        {"A0": [0.5] * 200, "A1": [-0.5] * 200},
        index=prices.index,
    )
    flat = run_backtest(prices, positions, borrow_bps_annual=0.0)
    charged = run_backtest(prices, positions, borrow_bps_annual=500.0)
    assert charged.equity_curve.iloc[-1] < flat.equity_curve.iloc[-1]


def test_run_backtest_cash_interest_increases_equity_on_cash() -> None:
    prices = _price_panel(n=200, k=2, seed=8)
    # Very small gross exposure → large idle-cash fraction.
    positions = pd.DataFrame(
        {"A0": [0.05] * 200, "A1": [0.05] * 200},
        index=prices.index,
    )
    flat = run_backtest(prices, positions, cash_interest_bps_annual=0.0)
    paid = run_backtest(prices, positions, cash_interest_bps_annual=500.0)
    assert paid.equity_curve.iloc[-1] > flat.equity_curve.iloc[-1]


# ---------------------------------------------------------------------------
# 5. Ensemble combiner
# ---------------------------------------------------------------------------


def test_ensemble_config_validates() -> None:
    with pytest.raises(ValueError, match="scheme"):
        EnsembleConfig(scheme="mystery")
    with pytest.raises(ValueError, match="vol_lookback"):
        EnsembleConfig(vol_lookback=1)
    with pytest.raises(ValueError, match="custom_weights must be provided"):
        EnsembleConfig(scheme="custom")
    with pytest.raises(ValueError, match="non-negative"):
        EnsembleConfig(scheme="custom", custom_weights={"a": -1.0, "b": 1.0})
    with pytest.raises(ValueError, match="sum to > 0"):
        EnsembleConfig(scheme="custom", custom_weights={"a": 0.0, "b": 0.0})


def test_combine_strategies_equal_weight() -> None:
    prices = _price_panel(n=100, k=3, seed=1)
    s1 = pd.DataFrame(0.3, index=prices.index, columns=["A0", "A1"])
    s2 = pd.DataFrame(0.6, index=prices.index, columns=["A1", "A2"])
    combined = combine_strategies(
        prices, {"s1": s1, "s2": s2}, config=EnsembleConfig(scheme="equal_weight")
    )
    # Each strategy gets weight 0.5. A0 only appears in s1 → 0.5 * 0.3 = 0.15.
    # A1 appears in both → 0.5*0.3 + 0.5*0.6 = 0.45. A2 only in s2 → 0.5*0.6 = 0.30.
    assert combined.loc[prices.index[-1], "A0"] == pytest.approx(0.15)
    assert combined.loc[prices.index[-1], "A1"] == pytest.approx(0.45)
    assert combined.loc[prices.index[-1], "A2"] == pytest.approx(0.30)


def test_combine_strategies_custom_weights_normalise() -> None:
    prices = _price_panel(n=50, k=2, seed=2)
    s1 = pd.DataFrame(1.0, index=prices.index, columns=["A0"])
    s2 = pd.DataFrame(1.0, index=prices.index, columns=["A1"])
    combined = combine_strategies(
        prices,
        {"s1": s1, "s2": s2},
        config=EnsembleConfig(scheme="custom", custom_weights={"s1": 3.0, "s2": 1.0}),
    )
    # Weights normalise to 0.75 / 0.25
    assert combined.loc[prices.index[0], "A0"] == pytest.approx(0.75)
    assert combined.loc[prices.index[0], "A1"] == pytest.approx(0.25)


def test_combine_strategies_inverse_vol_tilts_toward_low_vol() -> None:
    # Build two synthetic strategies with clearly different realised vol.
    prices = _price_panel(n=300, k=2, seed=17)
    # Low-vol strategy: tiny position in A0
    s_low = pd.DataFrame(0.1, index=prices.index, columns=["A0"])
    # High-vol strategy: flipping sign each day in A1
    flip = pd.Series(
        ((-1.0) ** np.arange(300)) * 1.0, index=prices.index
    )
    s_hi = pd.DataFrame({"A1": flip}, index=prices.index)
    combined = combine_strategies(
        prices,
        {"low": s_low, "hi": s_hi},
        config=EnsembleConfig(scheme="inverse_vol", vol_lookback=30),
    )
    # At the tail of the sample the low-vol strategy should have a much
    # larger combined weight than the high-vol strategy. Its contribution
    # to A0 should exceed the high-vol strategy's contribution (in
    # absolute value) to A1.
    last = combined.iloc[-1]
    assert abs(last["A0"]) > abs(last["A1"])


def test_combine_strategies_gross_leverage_cap() -> None:
    prices = _price_panel(n=80, k=2, seed=3)
    s1 = pd.DataFrame(1.0, index=prices.index, columns=["A0", "A1"])
    s2 = pd.DataFrame(1.0, index=prices.index, columns=["A0", "A1"])
    combined = combine_strategies(
        prices,
        {"s1": s1, "s2": s2},
        config=EnsembleConfig(scheme="equal_weight", max_gross_leverage=1.0),
    )
    # Without the cap, gross = 0.5 * 2 + 0.5 * 2 = 2.0 per row. With the
    # cap = 1.0, every row must be scaled to exactly 1.0.
    assert np.all(np.isclose(combined.abs().sum(axis=1).to_numpy(), 1.0))


def test_combine_strategies_rejects_misaligned_index() -> None:
    prices = _price_panel(n=40, k=2, seed=9)
    s1 = pd.DataFrame(0.3, index=prices.index, columns=["A0"])
    s2 = pd.DataFrame(0.3, index=prices.index.shift(1, freq="B"), columns=["A1"])
    with pytest.raises(ValueError, match="index must equal prices.index"):
        combine_strategies(prices, {"s1": s1, "s2": s2})


def test_strategy_correlation_matrix_is_symmetric_and_unit_diagonal() -> None:
    prices = _price_panel(n=200, k=3, seed=19)
    s1 = pd.DataFrame(0.3, index=prices.index, columns=["A0"])
    s2 = pd.DataFrame(-0.3, index=prices.index, columns=["A1"])
    s3 = pd.DataFrame(0.2, index=prices.index, columns=["A2"])
    corr = strategy_correlation_matrix(prices, {"s1": s1, "s2": s2, "s3": s3})
    assert corr.shape == (3, 3)
    np.testing.assert_array_almost_equal(np.diag(corr.to_numpy()), [1.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal(corr.to_numpy(), corr.to_numpy().T)


# ---------------------------------------------------------------------------
# 6. Futures continuous contracts & roll returns
# ---------------------------------------------------------------------------


def _two_contract_futures_panel() -> tuple[pd.DataFrame, pd.Series]:
    """Stylised two-contract example with a known roll gap.

    The second contract prices are the first contract's prices shifted
    up by exactly 10 in the overlap, and the roll happens on day 50.
    Under ratio adjustment the gap produces a 10/100 = 10% factor on
    the historical series.
    """
    idx = pd.date_range("2021-01-01", periods=100, freq="B")
    rng = np.random.default_rng(123)
    base = 100.0 + np.cumsum(rng.standard_normal(100) * 0.1)
    c1 = base.copy()
    c2 = base + 10.0
    # C1 is listed through day 50 inclusive (still has a price on the roll
    # bar itself, which is required by the ratio/difference adjustment).
    # After day 50 C1 has rolled off and its column is NaN.
    c1_full = np.where(np.arange(100) <= 50, c1, np.nan)
    prices = pd.DataFrame({"C1": c1_full, "C2": c2}, index=idx)
    # Roll from C1 to C2 on day 50
    schedule = pd.Series(["C1"] * 50 + ["C2"] * 50, index=idx, dtype=object)
    return prices, schedule


def test_continuous_contract_ratio_preserves_percent_returns() -> None:
    prices, sched = _two_contract_futures_panel()
    res = build_continuous_contract(prices, sched, adjustment="ratio")
    # The continuous series' percent returns must equal the raw
    # front-contract's percent returns on every non-roll bar. (Both on
    # C1 up to day 49 and on C2 from day 50, the *percent* return is
    # preserved — that's the whole point of ratio adjustment.)
    cont_ret = res.price.pct_change()
    # Build the "native" return — pct change of whichever raw series is
    # active that bar, recognising that the series itself changes.
    raw_ret = res.raw_front_price.pct_change()
    # On same-contract bars these must coincide.
    same_contract = res.front_symbol.eq(res.front_symbol.shift(1))
    same_contract.iloc[0] = False  # first bar has no previous
    np.testing.assert_array_almost_equal(
        cont_ret[same_contract].to_numpy(),
        raw_ret[same_contract].to_numpy(),
        decimal=12,
    )


def test_continuous_contract_difference_preserves_dollar_changes() -> None:
    prices, sched = _two_contract_futures_panel()
    res = build_continuous_contract(prices, sched, adjustment="difference")
    cont_diff = res.price.diff()
    raw_diff = res.raw_front_price.diff()
    same_contract = res.front_symbol.eq(res.front_symbol.shift(1))
    same_contract.iloc[0] = False
    # On same-contract bars the dollar change must match.
    np.testing.assert_array_almost_equal(
        cont_diff[same_contract].to_numpy(),
        raw_diff[same_contract].to_numpy(),
        decimal=12,
    )


def test_continuous_contract_none_adjustment_equals_raw_front() -> None:
    prices, sched = _two_contract_futures_panel()
    res = build_continuous_contract(prices, sched, adjustment="none")
    pd.testing.assert_series_equal(res.price, res.raw_front_price, check_names=False)


def test_continuous_contract_roll_dates_detected() -> None:
    prices, sched = _two_contract_futures_panel()
    res = build_continuous_contract(prices, sched, adjustment="ratio")
    assert len(res.roll_dates) == 1
    assert res.roll_dates[0] == prices.index[50]


def test_continuous_contract_rejects_missing_front_price() -> None:
    idx = pd.date_range("2021-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"C1": [100.0] * 10}, index=idx)
    sched = pd.Series(["C1"] * 5 + ["C2"] * 5, index=idx)
    with pytest.raises(ValueError, match="unknown contract"):
        build_continuous_contract(prices, sched)


def test_decompose_futures_returns_zero_roll_on_non_roll_bars() -> None:
    prices, sched = _two_contract_futures_panel()
    res = build_continuous_contract(prices, sched, adjustment="ratio")
    dec = decompose_futures_returns(res)
    # Off-roll bars: roll_return is exactly zero; is_roll is False.
    off_roll = dec.loc[~dec["is_roll"]].iloc[1:]  # drop initial NaN bar
    assert np.all(np.isclose(off_roll["roll_return"].fillna(0.0).to_numpy(), 0.0))
    assert not off_roll["is_roll"].any()


def test_decompose_futures_returns_roll_bar_captured() -> None:
    prices, sched = _two_contract_futures_panel()
    res = build_continuous_contract(prices, sched, adjustment="ratio")
    dec = decompose_futures_returns(res)
    # Exactly one roll bar expected
    assert dec["is_roll"].sum() == 1
    # On the roll bar, spot_return should be 0 (by convention) and
    # roll_return should equal the continuous log return.
    roll_row = dec.loc[dec["is_roll"]].iloc[0]
    assert roll_row["spot_return"] == pytest.approx(0.0, abs=1e-12)
    assert roll_row["roll_return"] == pytest.approx(roll_row["continuous_return"])


def test_calendar_spread_log_form() -> None:
    prices, _ = _two_contract_futures_panel()
    # On day 0 C1 is raw base price, C2 = base + 10
    spread = calendar_spread(prices, "C2", "C1", use_log_prices=True)
    # Non-null region is only days 0..49 (C1 is NaN after that)
    valid = spread.dropna()
    assert len(valid) == 51
    assert (valid > 0).all()  # C2 > C1 everywhere


def test_calendar_spread_rejects_nonpositive_for_log() -> None:
    idx = pd.date_range("2021-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"A": [1.0] * 5, "B": [0.0] * 5}, index=idx)
    with pytest.raises(ValueError, match="positive"):
        calendar_spread(prices, "A", "B", use_log_prices=True)


# ---------------------------------------------------------------------------
# 7. Permutation alpha test
# ---------------------------------------------------------------------------


def test_permutation_alpha_test_random_positions_give_uniform_pvalue() -> None:
    # Random positions on random prices → there is no edge, so under H₀ the
    # p-value is approximately uniform on [0, 1]. We don't assert anything
    # about a single draw (it can be anywhere); we run several independent
    # tests and confirm they are not all extreme in the same direction.
    prices = _price_panel(n=250, k=3, seed=41)
    n_tests = 6
    p_values = []
    for seed in range(n_tests):
        rng = np.random.default_rng(seed + 100)
        positions = pd.DataFrame(
            rng.standard_normal(prices.shape) * 0.3,
            index=prices.index,
            columns=prices.columns,
        )
        rep = permutation_alpha_test(prices, positions, n_shuffles=150, seed=seed)
        p_values.append(rep.p_value)
    # A genuine edge would give very small p-values across all seeds. Here
    # at least some must land above 0.1 — otherwise the test would be
    # rejecting H₀ every time, which can't be right under H₀.
    assert max(p_values) > 0.1, f"all p-values extreme: {p_values}"
    assert min(p_values) < 0.9, f"all p-values extreme: {p_values}"


def test_permutation_alpha_test_synthetic_signal_gives_low_pvalue() -> None:
    # Build a position stream that is literally the sign of next-bar
    # return — this is a perfect look-ahead edge. The permutation test
    # should detect it and return a very small p-value.
    rng = np.random.default_rng(2)
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    r = rng.standard_normal(n) * 0.01
    prices = pd.DataFrame({"A": 100.0 * np.cumprod(1.0 + r)}, index=idx)
    actual_r = prices["A"].pct_change().fillna(0.0)
    # Position at t = sign of return at t+1 (oracle), held one bar.
    positions = pd.DataFrame(
        {"A": np.sign(actual_r.shift(-1)).fillna(0.0).to_numpy()},
        index=idx,
    )
    rep = permutation_alpha_test(prices, positions, n_shuffles=300, seed=0)
    assert rep.p_value < 0.01
    assert rep.observed_sharpe > np.mean(rep.null_sharpes)


def test_permutation_alpha_test_rejects_misaligned_inputs() -> None:
    prices = _price_panel(n=50, k=2, seed=3)
    positions = pd.DataFrame(
        0.1, index=prices.index.shift(1, freq="B"), columns=prices.columns
    )
    with pytest.raises(ValueError, match="same index"):
        permutation_alpha_test(prices, positions)


def test_permutation_alpha_test_p_value_bounds() -> None:
    prices = _price_panel(n=80, k=2, seed=4)
    positions = pd.DataFrame(0.1, index=prices.index, columns=prices.columns)
    rep = permutation_alpha_test(prices, positions, n_shuffles=50, seed=0)
    assert 0.0 < rep.p_value <= 1.0
