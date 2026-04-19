"""Tests for v10 additions.

This file exercises the four features that v10 adds on top of v9:

1. Purge and embargo in walk-forward window generation and ``WalkForwardConfig``.
2. Probability of Backtest Overfitting via CSCV.
3. Multivariate Kelly weights and Kelly-from-drawdown-cap.
4. Probabilistic Sharpe Ratio as a standalone function.

References
----------
- Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, Wiley.
- Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier."
  *Journal of Risk*, 15(2), 3-44.
- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2016).
  "The Probability of Backtest Overfitting." *Journal of Computational Finance*, 20(4), 39-69.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chan_trading.config import MeanReversionConfig, WalkForwardConfig
from chan_trading.portfolio.kelly import (
    kelly_fraction_with_drawdown_cap,
    multivariate_kelly_weights,
)
from chan_trading.risk.combinatorial_cv import (
    PBOReport,
    probability_of_backtest_overfitting,
)
from chan_trading.risk.metrics import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from chan_trading.validation.walkforward import (
    generate_walkforward_windows,
    run_walkforward_pair_mean_reversion,
)


# ---------------------------------------------------------------------------
# 1. Purge and embargo
# ---------------------------------------------------------------------------


def test_walkforward_config_validates_purge_embargo() -> None:
    with pytest.raises(ValueError, match="purge"):
        WalkForwardConfig(train_size=100, test_size=20, purge=-1)
    with pytest.raises(ValueError, match="embargo"):
        WalkForwardConfig(train_size=100, test_size=20, embargo=-5)
    # Positive values are accepted
    wf = WalkForwardConfig(train_size=100, test_size=20, purge=3, embargo=2)
    assert wf.purge == 3
    assert wf.embargo == 2


def test_purge_in_walkforward_windows_creates_gap() -> None:
    idx = pd.RangeIndex(400)
    wins_no_purge = generate_walkforward_windows(idx, train_size=100, test_size=20, purge=0)
    wins_with_purge = generate_walkforward_windows(idx, train_size=100, test_size=20, purge=5)
    assert len(wins_no_purge) > 0
    assert len(wins_with_purge) > 0

    # Without purge, test starts exactly at end of train
    tr, te = wins_no_purge[0]
    assert te.start == tr.stop

    # With purge=5, there is a 5-bar gap
    tr, te = wins_with_purge[0]
    assert te.start == tr.stop + 5
    assert te.stop - te.start == 20


def test_embargo_increases_step_between_windows() -> None:
    idx = pd.RangeIndex(400)
    wins_no_emb = generate_walkforward_windows(idx, train_size=100, test_size=20, step_size=20, embargo=0)
    wins_emb = generate_walkforward_windows(idx, train_size=100, test_size=20, step_size=20, embargo=10)

    # Embargo must reduce the number of fitted windows (each step is longer)
    assert len(wins_emb) < len(wins_no_emb)

    # Verify the step between successive train starts
    if len(wins_emb) >= 2:
        step = wins_emb[1][0].start - wins_emb[0][0].start
        assert step == 30  # step_size + embargo


def test_purge_embargo_negative_raise_in_window_fn() -> None:
    idx = pd.RangeIndex(200)
    with pytest.raises(ValueError):
        generate_walkforward_windows(idx, train_size=50, test_size=10, purge=-1)
    with pytest.raises(ValueError):
        generate_walkforward_windows(idx, train_size=50, test_size=10, embargo=-2)


def test_walkforward_expanding_respects_purge_embargo() -> None:
    idx = pd.RangeIndex(500)
    wins = generate_walkforward_windows(
        idx,
        train_size=80,
        test_size=20,
        step_size=20,
        min_train_size=80,
        purge=3,
        embargo=2,
    )
    assert len(wins) > 0
    for tr, te in wins:
        # Expanding: train always starts at 0
        assert tr.start == 0
        # Purge gap between train end and test start
        assert te.start == tr.stop + 3
        assert te.stop - te.start == 20


def test_walkforward_pipeline_uses_purge_embargo() -> None:
    """The full mean-reversion walk-forward pipeline threads through purge/embargo."""
    from chan_trading.data.loaders import load_prices_csv

    prices = load_prices_csv("data/sample_prices.csv")
    mr_cfg = MeanReversionConfig(lookback=20, entry_z=2.0, exit_z=0.5)

    # Loose filters so windows actually trade
    wf_no_purge = WalkForwardConfig(
        train_size=120, test_size=30, step_size=30, purge=0, embargo=0,
        adf_alpha=0.99, eg_alpha=0.99,
    )
    wf_with_purge = WalkForwardConfig(
        train_size=120, test_size=30, step_size=30, purge=5, embargo=5,
        adf_alpha=0.99, eg_alpha=0.99,
    )

    rep_no_purge = run_walkforward_pair_mean_reversion(prices, "SPY", "IVV", mr_cfg, wf_no_purge)
    rep_with_purge = run_walkforward_pair_mean_reversion(prices, "SPY", "IVV", mr_cfg, wf_with_purge)

    # Embargo -> fewer windows (each step is longer)
    assert len(rep_with_purge.windows) <= len(rep_no_purge.windows)

    # All windows should have train start <= train end < test start <= test end
    for w in rep_with_purge.windows:
        assert w.train_start <= w.train_end
        assert w.train_end < w.test_start
        assert w.test_start <= w.test_end


# ---------------------------------------------------------------------------
# 2. Probabilistic Sharpe Ratio
# ---------------------------------------------------------------------------


def test_psr_high_for_strong_positive_signal() -> None:
    rng = np.random.default_rng(42)
    # Daily 0.1% mean, 1% vol -> SR ~ 0.1 per day ~ 1.6 annualized
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    psr = probabilistic_sharpe_ratio(r, benchmark_sharpe=0.0)
    assert 0.8 <= psr <= 1.0


def test_psr_low_for_negative_drift() -> None:
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(-0.001, 0.01, 500))
    psr = probabilistic_sharpe_ratio(r, benchmark_sharpe=0.0)
    assert 0.0 <= psr <= 0.2


def test_psr_near_half_for_zero_mean() -> None:
    rng = np.random.default_rng(11)
    # Large sample with truly zero mean -> PSR(bench=0) should be near 0.5
    r = pd.Series(rng.normal(0.0, 0.01, 2000))
    psr = probabilistic_sharpe_ratio(r, benchmark_sharpe=0.0)
    assert 0.3 <= psr <= 0.7


def test_psr_equals_dsr_n_trials_one() -> None:
    """PSR against benchmark=0 equals DSR with n_trials=1 (degenerate no-multiple-testing case)."""
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0008, 0.01, 500))
    psr = probabilistic_sharpe_ratio(r, benchmark_sharpe=0.0)
    dsr = deflated_sharpe_ratio(r, n_trials=1, benchmark_sharpe=0.0)
    assert psr == pytest.approx(dsr, rel=1e-9, abs=1e-9)


def test_psr_requires_minimum_samples() -> None:
    r = pd.Series([0.01, -0.01, 0.005])
    out = probabilistic_sharpe_ratio(r)
    # Tiny sample -> NaN
    assert np.isnan(out)


# ---------------------------------------------------------------------------
# 3. Multivariate Kelly
# ---------------------------------------------------------------------------


def test_multivariate_kelly_positive_drift_gives_positive_weights() -> None:
    rng = np.random.default_rng(0)
    # Three uncorrelated positive-drift assets. With a large-enough sample,
    # the signal dominates the sample noise and all weights turn positive.
    data = rng.normal(0.005, 0.01, (2000, 3))
    df = pd.DataFrame(data, columns=list("ABC"))
    w = multivariate_kelly_weights(df)
    assert (w > 0).all()
    assert w.index.tolist() == ["A", "B", "C"]


def test_multivariate_kelly_shrinkage_stabilizes() -> None:
    """With highly correlated columns, shrinkage should produce more reasonable weights."""
    rng = np.random.default_rng(12)
    base = rng.normal(0.001, 0.01, 300)
    # Two nearly identical columns -> near-singular covariance
    df = pd.DataFrame(
        {
            "A": base + rng.normal(0, 1e-6, 300),
            "B": base + rng.normal(0, 1e-6, 300),
            "C": rng.normal(0.0005, 0.01, 300),
        }
    )
    # Without shrinkage, weights should be extreme
    w_no = multivariate_kelly_weights(df, shrinkage=0.0)
    # With shrinkage, weights should be tamer
    w_shrink = multivariate_kelly_weights(df, shrinkage=0.5)
    assert w_no.abs().sum() > w_shrink.abs().sum()


def test_multivariate_kelly_input_validation() -> None:
    with pytest.raises(TypeError):
        multivariate_kelly_weights(pd.Series([0.01, 0.02]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shrinkage"):
        multivariate_kelly_weights(pd.DataFrame({"A": [0.01] * 50}), shrinkage=2.0)
    with pytest.raises(ValueError, match="Not enough rows"):
        multivariate_kelly_weights(pd.DataFrame({"A": [0.01, 0.02], "B": [0.0, 0.01]}))


def test_kelly_dd_cap_decreases_with_stricter_tolerance() -> None:
    rng = np.random.default_rng(5)
    # Modest positive drift with noticeable vol
    r = pd.Series(rng.normal(0.0005, 0.015, 500))
    k_loose = kelly_fraction_with_drawdown_cap(
        r, max_drawdown_tolerance=0.80, seed=1, n_paths=300, confidence=0.95
    )
    k_strict = kelly_fraction_with_drawdown_cap(
        r, max_drawdown_tolerance=0.05, seed=1, n_paths=300, confidence=0.95
    )
    assert k_strict <= k_loose


def test_kelly_dd_cap_returns_zero_when_infeasible() -> None:
    """If tolerance is impossibly tight, we get 0."""
    rng = np.random.default_rng(99)
    r = pd.Series(rng.normal(-0.002, 0.03, 500))  # negative drift, high vol
    k = kelly_fraction_with_drawdown_cap(
        r,
        max_drawdown_tolerance=0.001,  # tolerance well below any plausible DD
        seed=0,
        n_paths=200,
        confidence=0.95,
        candidate_multipliers=np.array([0.2, 0.5, 1.0]),
    )
    assert k == 0.0


def test_kelly_dd_cap_input_validation() -> None:
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    with pytest.raises(ValueError, match="max_drawdown_tolerance"):
        kelly_fraction_with_drawdown_cap(r, max_drawdown_tolerance=0.0)
    with pytest.raises(ValueError, match="confidence"):
        kelly_fraction_with_drawdown_cap(r, max_drawdown_tolerance=0.2, confidence=1.5)
    with pytest.raises(ValueError, match="method"):
        kelly_fraction_with_drawdown_cap(
            r, max_drawdown_tolerance=0.2, method="bogus"
        )


# ---------------------------------------------------------------------------
# 4. Probability of Backtest Overfitting via CSCV
# ---------------------------------------------------------------------------


def test_pbo_result_is_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0, 0.01, (400, 8)))
    rep = probability_of_backtest_overfitting(df, n_splits=8)
    assert isinstance(rep, PBOReport)
    assert 0.0 <= rep.pbo <= 1.0
    assert rep.n_strategies == 8
    assert rep.n_splits == 8
    # C(8, 4) = 70
    assert rep.n_combinations == 70
    assert len(rep.logit_values) == rep.n_combinations


def test_pbo_high_for_pure_noise_strategies() -> None:
    """With iid noise, PBO should be non-trivial and well away from the dominant-signal limit."""
    rng = np.random.default_rng(123)
    df = pd.DataFrame(rng.normal(0.0, 0.01, (500, 12)))
    rep = probability_of_backtest_overfitting(df, n_splits=10)
    # PBO for pure noise has wide sampling variance with ~12 strategies;
    # we only require it to be far from the "strong signal" regime (<0.3).
    assert rep.pbo >= 0.3


def test_pbo_low_when_one_strategy_dominates() -> None:
    """A genuinely superior strategy should produce low PBO."""
    rng = np.random.default_rng(7)
    n = 500
    # 9 noise strategies + 1 truly better one
    cols = {f"noise_{i}": rng.normal(0.0, 0.01, n) for i in range(9)}
    cols["winner"] = rng.normal(0.002, 0.008, n)  # much higher Sharpe
    df = pd.DataFrame(cols)
    rep = probability_of_backtest_overfitting(df, n_splits=10)
    assert rep.pbo < 0.3


def test_pbo_input_validation() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(0, 0.01, (400, 4)))
    with pytest.raises(TypeError):
        probability_of_backtest_overfitting(df.iloc[:, 0], n_splits=8)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="even integer"):
        probability_of_backtest_overfitting(df, n_splits=7)
    with pytest.raises(ValueError, match="even integer"):
        probability_of_backtest_overfitting(df, n_splits=0)
    with pytest.raises(ValueError, match="at least 2 strategy columns"):
        probability_of_backtest_overfitting(df.iloc[:, :1], n_splits=8)
