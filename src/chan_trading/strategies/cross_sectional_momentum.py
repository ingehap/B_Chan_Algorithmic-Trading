"""Cross-sectional momentum on a universe of assets.

Chan (Ch. 6) describes cross-sectional momentum as longing assets that
recently outperformed the cross-section and shorting those that
underperformed, each day. This differs from time-series momentum in that
the signal is a *rank* across assets, not a sign on each asset
independently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from chan_trading.config import CrossSectionalMomentumConfig
from chan_trading.portfolio.sizing import apply_exposure_caps, apply_vol_target
from chan_trading.strategies.base import Strategy


def _long_short_ranks(
    lookback_returns: pd.DataFrame,
    *,
    top_fraction: float,
    bottom_fraction: float,
    long_only: bool,
) -> pd.DataFrame:
    """Build ±1 long/short signals from per-row cross-sectional ranks.

    *Vectorised in v13.* The v12 implementation used ``.iterrows()`` and
    per-cell ``.loc`` assignments inside a Python loop, which was the
    dominant bottleneck of both cross-sectional strategies on any non-
    trivial panel (~1.6 s for T=2000, N=50). The v13 rewrite uses pandas'
    C-level ``.rank(axis=1, method='first')`` over a fully vectorised
    mask, producing the same output roughly 50-100× faster.

    Semantics preserved verbatim from v12:

    - per-row valid count ``n_valid = count of non-NaN entries``;
    - ``n_top = max(1, floor(n_valid * top_fraction))``;
    - ``n_bot = max(1, floor(n_valid * bottom_fraction))``;
    - if ``n_top + n_bot > n_valid``, fall back to a half-half split;
    - highest-return assets get ``+1`` (long), lowest get ``-1`` (short);
    - in ``long_only`` mode, only the top slice is set.

    Ties are broken by column order (`method='first'`), matching the v12
    behaviour which picked the first-by-index-order column inside
    ``sort_values(ascending=False)``.
    """
    cols = lookback_returns.columns
    index = lookback_returns.index
    values = lookback_returns.to_numpy(dtype=float)

    # Non-NaN count per row and the target slice sizes.
    n_valid_per_row = np.sum(np.isfinite(values), axis=1)
    n_top_raw = np.maximum(1, (n_valid_per_row * top_fraction).astype(int))
    n_bot_raw = np.maximum(1, (n_valid_per_row * bottom_fraction).astype(int))
    # Apply the v12 fallback: when the requested slices overlap, split
    # exactly in half.
    overflow = (n_top_raw + n_bot_raw) > n_valid_per_row
    n_top_fallback = n_valid_per_row // 2
    n_bot_fallback = n_valid_per_row - n_top_fallback
    n_top = np.where(overflow, n_top_fallback, n_top_raw)
    n_bot = np.where(overflow, n_bot_fallback, n_bot_raw)
    # Rows with fewer than 2 valid assets are skipped.
    active_rows = n_valid_per_row >= 2

    # Descending rank among valid entries: highest-return = rank 1.
    # ``.rank(axis=1, method="first", ascending=False)`` ranks within each
    # row, assigns NaN to NaN inputs, and breaks ties by column order so
    # the output matches v12.
    desc_rank = lookback_returns.rank(axis=1, method="first", ascending=False).to_numpy()

    # Long mask: rank ∈ [1, n_top].
    rank_is_finite = np.isfinite(desc_rank)
    long_mask = rank_is_finite & (desc_rank <= n_top[:, None]) & active_rows[:, None]

    # Short mask: rank ∈ [n_valid - n_bot + 1, n_valid] — i.e. the bottom
    # ``n_bot`` by return. Equivalent to an ascending rank ≤ n_bot.
    ascending_rank = lookback_returns.rank(axis=1, method="first", ascending=True).to_numpy()
    short_mask = (
        rank_is_finite
        & (ascending_rank <= n_bot[:, None])
        & active_rows[:, None]
    )
    # Don't double-assign when top and bottom slices touch (degenerate
    # small-universe case). Long takes precedence, matching v12 ordering.
    short_mask = short_mask & ~long_mask

    signal = np.zeros_like(values, dtype=float)
    signal[long_mask] = 1.0
    if not long_only:
        signal[short_mask] = -1.0
    return pd.DataFrame(signal, index=index, columns=cols, dtype=float)


@dataclass(slots=True)
class CrossSectionalMomentumStrategy(Strategy):
    """Cross-sectional momentum strategy.

    Ranks assets by their trailing ``lookback``-bar return and longs the top
    fraction, shorts the bottom fraction. Weights are equal within each leg
    and scaled to ``max_leverage`` gross.
    """

    config: CrossSectionalMomentumConfig

    def generate_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty:
            raise ValueError("prices must not be empty")
        if prices.shape[1] < 2:
            raise ValueError("Need at least 2 assets for cross-sectional momentum")

        price_table = prices.astype(float)
        lookback_returns = price_table / price_table.shift(self.config.lookback) - 1.0

        signal = _long_short_ranks(
            lookback_returns,
            top_fraction=self.config.top_fraction,
            bottom_fraction=self.config.bottom_fraction,
            long_only=self.config.long_only,
        )

        gross = signal.abs().sum(axis=1).replace(0.0, np.nan)
        weights = signal.div(gross, axis=0).fillna(0.0) * self.config.max_leverage

        if self.config.vol_target is not None:
            weights = apply_vol_target(
                prices=price_table,
                positions=weights,
                target_vol=self.config.vol_target,
                lookback=self.config.vol_lookback,
                max_leverage=self.config.max_leverage,
            )
        # v13: honour shared exposure caps on every strategy family.
        weights = apply_exposure_caps(
            weights,
            max_gross_exposure=self.config.max_gross_exposure,
            max_net_exposure=self.config.max_net_exposure,
        )
        return weights.fillna(0.0)
