"""Combinatorially Symmetric Cross-Validation (CSCV) and PBO.

Chan (Ch. 1) devotes significant attention to the threat of backtest
overfitting: when a researcher searches over many strategy variants, the
best one in-sample tends to look spuriously good out-of-sample. The
*Probability of Backtest Overfitting* (PBO) framework of
Bailey, Borwein, López de Prado & Zhu (2016) quantifies this directly.

The idea:

1. You collect a ``T × N`` matrix of per-period returns, one column per
   strategy variant (same universe, same period, different parameters).
2. Partition the ``T`` rows into ``S`` equal blocks. For each of the
   ``C(S, S/2)`` ways to pick ``S/2`` blocks as the in-sample set:

   a. Rank all ``N`` strategies by in-sample Sharpe. Pick the best, call it
      ``n*``.
   b. Compute the out-of-sample Sharpe of *every* strategy on the held-out
      ``S/2`` blocks, and compute the relative rank ``ω`` of ``n*`` among
      them (``ω = 1`` is best, ``ω = 0`` is worst).
   c. The logit ``log(ω / (1 - ω))`` is a signed measure of how well the
      IS winner holds up OOS; negative means OOS underperformance.

3. The PBO is the fraction of partitions where the logit is non-positive —
   i.e. the probability that the *best* IS configuration fails to even
   break even OOS. Values near 0 are good; values near or above 0.5 mean
   the research pipeline is essentially overfit.

Reference
---------
Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2016).
"The Probability of Backtest Overfitting."
*Journal of Computational Finance*, 20(4), 39–69.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PBOReport:
    """Result of a CSCV/PBO evaluation.

    Attributes
    ----------
    pbo
        Probability of backtest overfitting — the fraction of CSCV
        partitions whose IS-best strategy had non-positive OOS logit.
        Lower is better; 0.5 is the no-information baseline.
    logit_values
        Per-partition logit scores of the IS winner's OOS rank.
    n_combinations
        Number of partitions evaluated (``C(n_splits, n_splits/2)``).
    n_strategies
        Number of strategy variants in the input matrix.
    n_splits
        Number of blocks ``S`` the sample was divided into.
    """

    pbo: float
    logit_values: np.ndarray = field(repr=False)
    n_combinations: int
    n_strategies: int
    n_splits: int


def _sharpe_per_column(block_returns: np.ndarray) -> np.ndarray:
    """Per-period Sharpe for each column of a 2-D ``block_returns`` array.

    Not annualized — CSCV only needs the ranking, which is invariant to
    any shared positive multiplier.
    """
    mu = np.nanmean(block_returns, axis=0)
    sigma = np.nanstd(block_returns, axis=0, ddof=0)
    # Avoid division by zero: columns with zero vol get -inf Sharpe so
    # they never rank as the "best".
    sharpe = np.where(sigma > 0, mu / np.where(sigma > 0, sigma, 1.0), -np.inf)
    return sharpe


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame,
    *,
    n_splits: int = 16,
) -> PBOReport:
    """Compute the Probability of Backtest Overfitting via CSCV.

    Parameters
    ----------
    returns_matrix
        ``T × N`` DataFrame of per-period returns with one column per
        strategy variant. All columns must share the same time index.
    n_splits
        Number of equal-size blocks ``S`` the sample is divided into.
        Must be even and ``>= 2``. Typical values are 10–16. More splits
        give a finer PBO estimate but explode the number of combinations
        (``C(S, S/2)``). Default ``16`` gives ``12,870`` partitions.

    Returns
    -------
    :class:`PBOReport`

    Raises
    ------
    TypeError
        If ``returns_matrix`` is not a DataFrame.
    ValueError
        If ``n_splits`` is not a positive even integer, if there are
        fewer than 2 strategies, or if the sample cannot be split into
        ``n_splits`` non-empty blocks.

    References
    ----------
    Bailey, Borwein, López de Prado & Zhu (2016), JCF.
    """
    if not isinstance(returns_matrix, pd.DataFrame):
        raise TypeError("returns_matrix must be a pandas DataFrame")
    if returns_matrix.shape[1] < 2:
        raise ValueError("Need at least 2 strategy columns")
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValueError("n_splits must be an even integer >= 2")

    data = returns_matrix.dropna(how="any").to_numpy(dtype=float)
    n_rows, n_strats = data.shape
    if n_rows < n_splits:
        raise ValueError(
            f"Need at least n_splits={n_splits} rows, got {n_rows}"
        )

    # Build S blocks. We drop trailing rows so each block has equal size,
    # matching the original CSCV paper.
    block_size = n_rows // n_splits
    usable = block_size * n_splits
    trimmed = data[:usable]
    blocks = trimmed.reshape(n_splits, block_size, n_strats)

    half = n_splits // 2
    all_idx = set(range(n_splits))
    combos = list(combinations(range(n_splits), half))
    n_combos = len(combos)

    logits = np.empty(n_combos, dtype=float)
    for c_idx, is_idx in enumerate(combos):
        oos_idx = tuple(sorted(all_idx - set(is_idx)))
        is_block = blocks[list(is_idx)].reshape(-1, n_strats)
        oos_block = blocks[list(oos_idx)].reshape(-1, n_strats)

        is_sharpe = _sharpe_per_column(is_block)
        # Best IS strategy — break ties by lowest column index (stable).
        n_star = int(np.argmax(is_sharpe))

        oos_sharpe = _sharpe_per_column(oos_block)
        # Relative OOS rank in (0, 1). Using "average" ranks is closer to
        # the original formulation; rank 1 = worst, rank N = best.
        oos_ranks = pd.Series(oos_sharpe).rank(method="average").to_numpy()
        rel_rank = oos_ranks[n_star] / (n_strats + 1.0)
        # Clip to avoid infinities in the logit.
        rel_rank = min(max(rel_rank, 1e-6), 1.0 - 1e-6)
        logits[c_idx] = float(np.log(rel_rank / (1.0 - rel_rank)))

    pbo = float(np.mean(logits <= 0.0))
    return PBOReport(
        pbo=pbo,
        logit_values=logits,
        n_combinations=n_combos,
        n_strategies=n_strats,
        n_splits=n_splits,
    )
