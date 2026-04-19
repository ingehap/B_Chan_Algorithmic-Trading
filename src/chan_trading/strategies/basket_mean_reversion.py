from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from chan_trading.config import MeanReversionConfig
from chan_trading.strategies.mean_reversion import _apply_exposure_caps, _compute_zscore, _generate_stateful_signal


@dataclass(slots=True)
class BasketMeanReversionStrategy:
    """Mean-reversion strategy for a Johansen basket of 3+ assets."""

    weights: pd.Series
    config: MeanReversionConfig

    def generate_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.weights.index) - set(prices.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        basket_prices = prices[self.weights.index].astype(float)
        if self.config.use_log_prices:
            if (basket_prices <= 0).any().any():
                raise ValueError("Log-price basket modeling requires strictly positive prices")
            transformed = pd.DataFrame(
                np.log(basket_prices.to_numpy(dtype=float)),
                index=basket_prices.index,
                columns=basket_prices.columns,
            )
        else:
            transformed = basket_prices
        spread = transformed @ self.weights.astype(float)
        z = _compute_zscore(spread, self.config)
        signal = _generate_stateful_signal(z, self.config)

        gross = float(np.abs(self.weights).sum())
        if gross <= 0:
            raise ValueError("weights must have non-zero gross exposure")

        normalized = self.weights / gross
        target = pd.DataFrame(index=prices.index, columns=self.weights.index, dtype=float)
        for asset, weight in normalized.items():
            target[asset] = signal * weight * self.config.max_leverage
        return _apply_exposure_caps(target.fillna(0.0), self.config)
