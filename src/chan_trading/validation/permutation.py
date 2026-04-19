"""Permutation (shuffle) test for signal significance (v12).

Chan (Ch. 1) is dominated by the warning that *any* sufficiently
flexible research pipeline will produce impressive-looking backtests by
chance. The package already ships the multi-testing-aware defences
— Probability of Backtest Overfitting (PBO via CSCV) and the Deflated
Sharpe Ratio — but neither answers the simpler and more direct question
a researcher usually wants first:

    "Given the prices I actually observed and the trading activity my
    strategy actually did, is the observed return any different from
    what I would have gotten by trading at *random times with the same
    frequency*?"

The permutation / shuffle test answers exactly that. The null hypothesis
is **H₀: the positions carry no information about future returns** —
under which shuffling position rows in time should not change the
expected Sharpe ratio. We therefore:

1. Compute the observed strategy return and its Sharpe.
2. Build ``n_shuffles`` alternative strategies by randomly permuting the
   *rows* of the position panel (each permutation preserves the empirical
   distribution of positions but destroys its time alignment with
   returns).
3. Report the fraction of shuffled Sharpes that meet or exceed the
   observed Sharpe — this is the permutation p-value.

A small p-value means the alignment between positions and prices is
unlikely to be coincidental. A p-value near 0.5 means the strategy is
statistically indistinguishable from randomly-timed trading at the same
gross exposure.

This is complementary, not redundant, to PBO and DSR:

- **PBO** asks: *across many strategy variants you considered, how
  often does the best-in-sample one fail OOS?*
- **DSR** asks: *after correcting for skew, kurtosis, and the number of
  variants tried, is the observed Sharpe distinguishable from zero?*
- **This test** asks: *for this single strategy on this single price
  history, is the observed Sharpe distinguishable from random timing?*

References
----------
- Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and
  Their Rationale.* Wiley, Ch. 1.
- White, H. (2000). "A Reality Check for Data Snooping." *Econometrica*,
  68(5), 1097-1126. (Classical reference for reality-check permutation
  tests against a universe of strategies; this module implements the
  single-strategy specialisation.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from chan_trading.portfolio.sizing import lag_positions
from chan_trading.risk.metrics import sharpe_ratio


@dataclass(slots=True)
class PermutationTestReport:
    """Result of a permutation significance test.

    Attributes
    ----------
    observed_sharpe
        Annualised Sharpe ratio of the actual strategy.
    null_sharpes
        Array of length ``n_shuffles`` with the Sharpe of each
        time-permuted alternative.
    p_value
        Probability, under H₀, of seeing a Sharpe at least as large as
        ``observed_sharpe``. Conventional rule of thumb: ``< 0.05`` is
        weak evidence, ``< 0.01`` is stronger, ``< 0.001`` is strong.
    n_shuffles
        Number of permutations used.
    """

    observed_sharpe: float
    null_sharpes: np.ndarray
    p_value: float
    n_shuffles: int


def permutation_alpha_test(
    prices: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    n_shuffles: int = 1000,
    transaction_cost_bps: float = 0.0,
    periods_per_year: int = 252,
    seed: int | None = None,
    block_size: int | None = None,
) -> PermutationTestReport:
    """Permutation (shuffle) test of a strategy's Sharpe against random timing.

    The test preserves the empirical distribution of positions (same
    rows, same gross-exposure profile) but destroys the temporal
    alignment with prices. A strategy whose edge genuinely comes from
    forecasting should see its Sharpe drop sharply on every shuffle; a
    strategy whose Sharpe was a product of return autocorrelation or
    risk premia capture should see shuffled Sharpes close to the
    observed one.

    Parameters
    ----------
    prices
        Price panel shared with the positions.
    positions
        Target position panel (any gross-exposure convention; the test
        is scale-invariant). Must share ``prices.index`` and its columns
        must be a subset of ``prices.columns``.
    n_shuffles
        Number of permutations to draw. 1000 is a reasonable default;
        increase to 5000-10000 if the p-value needs to be resolved to
        three decimals.
    transaction_cost_bps
        Flat per-turnover cost applied identically to both the observed
        run and each permuted run. Permutations generate much higher
        turnover than the original, so this argument makes the null
        distribution noticeably more adversarial — using a realistic
        value is recommended when the strategy is turnover-light.
    periods_per_year
        Annualisation factor for Sharpe. 252 for daily.
    seed
        Optional seed for reproducibility.
    block_size
        *New in v13.* When provided, shuffle the positions in
        contiguous blocks of this length rather than bar-by-bar. The
        single-bar shuffle (default, ``None``) destroys holding
        periods, which inflates permuted turnover for any strategy that
        actually holds positions. A block shuffle with ``block_size``
        set to roughly the strategy's typical holding horizon keeps the
        null's turnover profile realistic and makes the Sharpe
        comparison much fairer.

    Returns
    -------
    :class:`PermutationTestReport`
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if not isinstance(positions, pd.DataFrame):
        raise TypeError("positions must be a pandas DataFrame")
    if not prices.index.equals(positions.index):
        raise ValueError("prices and positions must share the same index")
    missing = set(positions.columns) - set(prices.columns)
    if missing:
        raise ValueError(f"positions has columns missing from prices: {sorted(missing)}")
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be >= 1")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be >= 0")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")
    if block_size is not None and block_size < 1:
        raise ValueError("block_size must be >= 1 when provided")

    rng = np.random.default_rng(seed)
    cols = list(positions.columns)
    asset_returns = prices[cols].pct_change().fillna(0.0).to_numpy(dtype=float)
    pos_array = positions.to_numpy(dtype=float)
    n = pos_array.shape[0]

    def _sharpe_from_positions(pa: np.ndarray) -> float:
        """Lag, compute gross return and subtract turnover cost."""
        lagged = np.vstack([np.zeros((1, pa.shape[1]), dtype=float), pa[:-1]])
        gross = (lagged * asset_returns).sum(axis=1)
        turnover = np.abs(np.diff(lagged, axis=0, prepend=0.0)).sum(axis=1)
        costs = turnover * (transaction_cost_bps / 10_000.0)
        net = gross - costs
        sigma = float(np.std(net, ddof=0))
        if sigma == 0.0 or not np.isfinite(sigma):
            return float("nan")
        return float(np.mean(net) / sigma * np.sqrt(periods_per_year))

    def _block_permutation(size: int, block: int) -> np.ndarray:
        """Permute ``range(size)`` in contiguous blocks of length ``block``.

        Constructs a list of ``ceil(size / block)`` blocks (the final
        block possibly shorter), shuffles those blocks, then concatenates
        them back. Preserves holding-period structure inside each block.
        """
        n_blocks_full = size // block
        tail = size - n_blocks_full * block
        block_starts = [i * block for i in range(n_blocks_full)]
        if tail > 0:
            block_starts.append(n_blocks_full * block)
        order = rng.permutation(len(block_starts))
        perm_parts: list[np.ndarray] = []
        for idx in order:
            s = block_starts[idx]
            length = block if s + block <= size else (size - s)
            perm_parts.append(np.arange(s, s + length, dtype=np.int64))
        return np.concatenate(perm_parts)

    observed = _sharpe_from_positions(pos_array)

    null = np.empty(n_shuffles, dtype=float)
    for k in range(n_shuffles):
        if block_size is None:
            perm = rng.permutation(n)
        else:
            perm = _block_permutation(n, int(block_size))
        null[k] = _sharpe_from_positions(pos_array[perm])

    # One-sided p-value: P(null >= observed)
    if np.isnan(observed):
        p = float("nan")
    else:
        valid = null[~np.isnan(null)]
        if valid.size == 0:
            p = float("nan")
        else:
            # +1 in numerator and denominator is the standard finite-
            # sample correction (Phipson & Smyth, 2010) so that p > 0
            # even when no permutation exceeds the observed.
            p = float((np.sum(valid >= observed) + 1) / (valid.size + 1))

    return PermutationTestReport(
        observed_sharpe=observed,
        null_sharpes=null,
        p_value=p,
        n_shuffles=n_shuffles,
    )
