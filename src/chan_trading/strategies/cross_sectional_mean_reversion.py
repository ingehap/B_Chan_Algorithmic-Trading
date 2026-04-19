"""Cross-sectional mean reversion on a stock basket.

Chan (Ch. 4) describes cross-sectional mean reversion as the mirror of
cross-sectional momentum: long the recent cross-sectional losers, short the
recent cross-sectional winners, on the hypothesis that relative dispersions
mean-revert. A typical implementation uses a 1-bar lookback (previous-day
return) as the signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from chan_trading.config import CrossSectionalMeanReversionConfig
from chan_trading.portfolio.sizing import apply_exposure_caps, apply_vol_target
from chan_trading.strategies.base import Strategy
from chan_trading.strategies.cross_sectional_momentum import _long_short_ranks


@dataclass(slots=True)
class CrossSectionalMeanReversionStrategy(Strategy):
    """Cross-sectional mean-reversion strategy.

    Ranks assets by their trailing ``lookback``-bar return. Longs the bottom
    fraction (recent losers) and shorts the top fraction (recent winners).
    """

    config: CrossSectionalMeanReversionConfig

    def generate_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty:
            raise ValueError("prices must not be empty")
        if prices.shape[1] < 2:
            raise ValueError("Need at least 2 assets for cross-sectional mean reversion")

        price_table = prices.astype(float)
        lookback_returns = price_table / price_table.shift(self.config.lookback) - 1.0

        # Mean reversion = flip the momentum sign: losers long, winners short.
        momentum_signal = _long_short_ranks(
            lookback_returns,
            top_fraction=self.config.top_fraction,
            bottom_fraction=self.config.bottom_fraction,
            long_only=False,
        )
        if self.config.long_only:
            # long losers only
            signal = momentum_signal.clip(upper=0.0).abs()
        else:
            signal = -momentum_signal

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
