"""Market-impact models for the event-driven backtester.

Chan (Ch. 1) is explicit that backtests look good precisely because they
charge unrealistically low trading costs — and that intraday and
high-frequency strategies are *especially* sensitive to this assumption.
A flat ``slippage_bps`` approximates quoted-spread cost but misses the
dominant source of cost at non-trivial trade sizes: **market impact**,
which scales with participation rate (the fraction of average daily
volume the order consumes).

This module introduces two classic impact models:

- :class:`LinearImpactModel` — impact in bps is linear in participation.
  Simple, conservative for small trades, aggressive for large trades.
- :class:`SquareRootImpactModel` — impact in bps is proportional to the
  square root of participation (the "Almgren law"). The standard default
  in practitioner work; concave, so very small trades pay very little
  and very large trades pay less than the linear model would charge.

Both models plug into :class:`chan_trading.backtest.broker.SimulatedBroker`
via its ``impact_model`` attribute and the event engine via
``EventBacktestConfig.impact_model`` + ``EventBacktestConfig.adv``.

References
----------
- Almgren, R., Thum, C., Hauptmann, E., & Li, H. (2005). "Direct
  Estimation of Equity Market Impact." *Risk*, 18(7), 58-62.
- Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and
  Their Rationale.* Wiley, Ch. 1.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class ImpactModel(ABC):
    """Abstract base for participation-rate-based market-impact models.

    Implementations take a notional trade size (in currency) and the
    asset's average daily volume (in the same currency) and return an
    impact cost in basis points, which the broker adds to its baseline
    slippage. Impact is always a *cost* — i.e. always non-negative.
    """

    @abstractmethod
    def impact_bps(self, trade_notional: float, adv_notional: float) -> float:
        """Return impact in basis points for a given trade and ADV.

        Both arguments must be in the same currency units. ``ImpactModel``
        implementations should return ``0.0`` when ``adv_notional`` is
        zero, NaN, or non-positive — i.e. gracefully degrade when ADV is
        not available for an asset.
        """
        raise NotImplementedError


@dataclass(slots=True)
class LinearImpactModel(ImpactModel):
    """Impact (bps) = ``coefficient`` × participation.

    ``participation = trade_notional / adv_notional``. If you think of
    ``coefficient = 100`` bps, then consuming 10% of ADV costs 10 bps
    of extra slippage on the whole trade.
    """

    coefficient_bps: float = 100.0

    def __post_init__(self) -> None:
        if self.coefficient_bps < 0:
            raise ValueError("coefficient_bps must be >= 0")

    def impact_bps(self, trade_notional: float, adv_notional: float) -> float:
        if adv_notional is None or not np.isfinite(adv_notional) or adv_notional <= 0:
            return 0.0
        if trade_notional is None or not np.isfinite(trade_notional):
            return 0.0
        participation = abs(float(trade_notional)) / float(adv_notional)
        return float(self.coefficient_bps * participation)


@dataclass(slots=True)
class SquareRootImpactModel(ImpactModel):
    """Impact (bps) = ``coefficient`` × √participation (Almgren law).

    The concave square-root shape is the industry workhorse for
    mid-frequency US equity execution; empirical calibrations put
    ``coefficient`` in the 10-30 bps range for typical single names.
    """

    coefficient_bps: float = 10.0

    def __post_init__(self) -> None:
        if self.coefficient_bps < 0:
            raise ValueError("coefficient_bps must be >= 0")

    def impact_bps(self, trade_notional: float, adv_notional: float) -> float:
        if adv_notional is None or not np.isfinite(adv_notional) or adv_notional <= 0:
            return 0.0
        if trade_notional is None or not np.isfinite(trade_notional):
            return 0.0
        participation = abs(float(trade_notional)) / float(adv_notional)
        return float(self.coefficient_bps * np.sqrt(participation))
