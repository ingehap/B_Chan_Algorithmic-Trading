from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from chan_trading.config import MomentumConfig
from chan_trading.portfolio.sizing import apply_exposure_caps, apply_vol_target
from chan_trading.strategies.base import Strategy


def _apply_momentum_stop_loss(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    stop_loss: float,
    lookback: int,
) -> pd.DataFrame:
    """Flatten positions after an adverse move exceeds ``stop_loss``.

    Chan (Ch. 8) argues that stop-loss is logically consistent with momentum
    (a sharp adverse move *is* the signal that the trend has broken) but
    dangerous for mean reversion (where an adverse move increases expected
    forward return). This helper is intentionally only exposed to the
    momentum family.

    The rule: for each asset, look at the realized return over the last
    ``lookback`` bars along the direction of the current position. If that
    realized return is worse than ``-stop_loss``, flatten that asset on the
    next bar.
    """
    if stop_loss <= 0 or not np.isfinite(stop_loss):
        return positions
    returns = prices[positions.columns].pct_change().fillna(0.0)
    trailing = returns.rolling(lookback, min_periods=1).sum()
    # Directional return = trailing return * sign(current position)
    sign = np.sign(positions)
    directional = trailing * sign
    hit = (directional < -stop_loss).fillna(False).astype(bool)
    # Apply on the next bar — shift to avoid same-bar look-ahead
    hit = hit.shift(1).fillna(False).astype(bool)
    return positions.mask(hit, 0.0)


@dataclass(slots=True)
class TimeSeriesMomentumStrategy(Strategy):
    """Simple interday time-series momentum strategy.

    The signal is the sign of the lookback return for each asset. Active signals
    are equal-weighted across the traded universe and scaled to the configured
    gross leverage cap.
    """

    config: MomentumConfig

    def generate_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        if prices.empty:
            raise ValueError("prices must not be empty")

        price_table = prices.astype(float)
        lookback_returns = price_table / price_table.shift(self.config.lookback) - 1.0

        signal = pd.DataFrame(0.0, index=price_table.index, columns=price_table.columns, dtype=float)
        signal = signal.mask(lookback_returns > self.config.return_threshold, 1.0)
        if self.config.long_only:
            signal = signal.mask(lookback_returns < -self.config.return_threshold, 0.0)
        else:
            signal = signal.mask(lookback_returns < -self.config.return_threshold, -1.0)

        active = signal.abs().sum(axis=1).replace(0.0, float("nan"))
        weights = signal.div(active, axis=0).fillna(0.0) * self.config.max_leverage

        if self.config.stop_loss is not None:
            weights = _apply_momentum_stop_loss(
                weights,
                price_table,
                stop_loss=self.config.stop_loss,
                lookback=self.config.stop_loss_lookback,
            )

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
