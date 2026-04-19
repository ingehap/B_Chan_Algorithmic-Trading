"""Daily buy-on-gap mean reversion across a basket.

Chan (Ch. 4 and 7) highlights buy-on-gap as a persistent short-horizon
reversal pattern: after a stock experiences a large adverse overnight
move, it tends to recover part of that move within the next trading day.
In the book the pattern is formulated with an open price:

    if today's open < yesterday's low − k · volatility(yesterday) → buy at open
    close at today's close

When only daily close prices are available (as in many research
workflows, including the sample data in this package) the same economic
idea can be expressed with close-to-close returns. That is what this
strategy does: a one-bar drop of at least ``threshold_sigma`` trailing
standard deviations triggers a long position for the next bar, which is
then closed. This preserves the "buy weakness, sell immediately"
structure without requiring intraday data.

When OHLC are available (by passing a separately-prepared ``open_prices``
DataFrame to :meth:`BuyOnGapStrategy.generate_positions_from_ohlc`),
the gap signal is computed from the open relative to yesterday's close
— a closer match to Chan's original.

Compared with :class:`~chan_trading.strategies.cross_sectional_mean_reversion.CrossSectionalMeanReversionStrategy`:

- Buy-on-gap is a **trigger / threshold** strategy (fires only on large
  moves); cross-sectional MR is a **rank** strategy (always fully
  invested across the top/bottom quantiles).
- Buy-on-gap holds for a fixed number of bars (typically 1);
  cross-sectional MR re-forms the portfolio each bar.

Both approaches capture short-horizon reversal, but the risk profiles
differ: buy-on-gap has variable gross exposure (zero when no gap fires)
while cross-sectional MR is always ~unit-gross.

References
----------
- Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and
  Their Rationale.* Wiley, Ch. 4 (short-term mean reversion in stocks) and
  Ch. 7 (opening-gap and breakout patterns).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from chan_trading.config import BuyOnGapConfig
from chan_trading.portfolio.sizing import apply_exposure_caps, apply_vol_target
from chan_trading.strategies.base import Strategy


@dataclass(slots=True)
class BuyOnGapStrategy(Strategy):
    """Daily buy-on-gap mean reversion over a basket of assets.

    Signal (close-only mode):

    1. For each asset, compute the one-bar return ``r_t = P_t / P_{t-1} − 1``.
    2. Compute rolling std of ``r`` over ``config.lookback`` bars.
    3. If ``r_t < −config.threshold_sigma · sigma_{t-1}`` the asset enters
       a long position on bar ``t+1`` and exits after ``config.hold_bars``.
    4. Active signals are equal-weighted across the basket and scaled to
       ``config.max_leverage`` gross.

    When ``config.two_sided=True`` a symmetric short signal is added on
    large up-moves (``r_t > +config.threshold_sigma · sigma_{t-1}``),
    which turns the strategy into a two-sided daily reversal.

    All indexing is strictly causal — the signal computed at bar ``t`` uses
    returns and sigma through ``t-1`` (``.shift(1)``), so it can be
    consumed by the standard backtest engine which applies an additional
    one-bar execution lag.
    """

    config: BuyOnGapConfig

    def generate_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty:
            raise ValueError("prices must not be empty")
        if prices.shape[1] < 1:
            raise ValueError("Need at least one asset")

        price_table = prices.astype(float)
        returns = price_table.pct_change()
        sigma = returns.rolling(
            self.config.lookback, min_periods=self.config.lookback
        ).std(ddof=0)

        # The gap signal must be evaluated on data available *before* the
        # position is entered. We compare r_t with sigma_{t-1} (shifted).
        threshold = sigma.shift(1) * self.config.threshold_sigma
        down_gap = returns < -threshold
        up_gap = returns > threshold

        entry = pd.DataFrame(
            0.0, index=price_table.index, columns=price_table.columns, dtype=float
        )
        entry = entry.mask(down_gap, 1.0)
        if self.config.two_sided:
            entry = entry.mask(up_gap, -1.0)

        # Hold each firing entry for `hold_bars` bars. This is a forward-
        # looking *target*; the execution lag handled by the backtest
        # engine takes care of avoiding same-bar look-ahead.
        hold = self.config.hold_bars
        if hold < 1:
            raise ValueError("hold_bars must be >= 1")
        if hold == 1:
            raw_target = entry
        else:
            # Sum overlapping holds (vectorized). Equivalent to a moving
            # sum of `hold` consecutive entry signals.
            raw_target = entry.rolling(hold, min_periods=1).sum()
            # Clip to [−1, +1] so overlapping gaps don't double-stack into
            # per-asset leverage above 1 before equal-weighting.
            raw_target = raw_target.clip(lower=-1.0, upper=1.0)

        active = raw_target.abs().sum(axis=1).replace(0.0, np.nan)
        weights = raw_target.div(active, axis=0).fillna(0.0) * self.config.max_leverage

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

    def generate_positions_from_ohlc(
        self,
        close_prices: pd.DataFrame,
        open_prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Variant of :meth:`generate_positions` using OHLC data.

        Here the gap is computed from ``today's open`` versus
        ``yesterday's close``, which is the form Chan uses in the book:

            gap_t = open_t / close_{t-1} − 1

        The trigger ``gap_t < −threshold_sigma · sigma_{t-1}`` still uses
        the trailing std of close-to-close returns. This matches the
        "sigma estimated on settle-to-settle returns" convention.

        Both DataFrames must share the same index and columns.
        """
        if not close_prices.index.equals(open_prices.index):
            raise ValueError("close and open prices must share the same index")
        if list(close_prices.columns) != list(open_prices.columns):
            raise ValueError("close and open prices must have identical columns")

        close = close_prices.astype(float)
        opn = open_prices.astype(float)

        settle_returns = close.pct_change()
        sigma = settle_returns.rolling(
            self.config.lookback, min_periods=self.config.lookback
        ).std(ddof=0)

        gap = opn / close.shift(1) - 1.0
        threshold = sigma.shift(1) * self.config.threshold_sigma
        down_gap = gap < -threshold
        up_gap = gap > threshold

        entry = pd.DataFrame(0.0, index=close.index, columns=close.columns, dtype=float)
        entry = entry.mask(down_gap, 1.0)
        if self.config.two_sided:
            entry = entry.mask(up_gap, -1.0)

        hold = self.config.hold_bars
        if hold < 1:
            raise ValueError("hold_bars must be >= 1")
        raw_target = (
            entry if hold == 1 else entry.rolling(hold, min_periods=1).sum().clip(-1.0, 1.0)
        )

        active = raw_target.abs().sum(axis=1).replace(0.0, np.nan)
        weights = raw_target.div(active, axis=0).fillna(0.0) * self.config.max_leverage

        if self.config.vol_target is not None:
            weights = apply_vol_target(
                prices=close,
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
