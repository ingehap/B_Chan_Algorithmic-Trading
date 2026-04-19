from __future__ import annotations

from chan_trading.execution.impact import ImpactModel
from chan_trading.types import FillEvent, OrderEvent


class SimulatedBroker:
    """Simple broker with bps commissions, slippage, spread, and a ticket minimum.

    *Extended in v11.* The broker now accepts an optional
    :class:`~chan_trading.execution.impact.ImpactModel`. When supplied,
    the caller passes ``adv_notional`` to :meth:`fill_order` and the
    broker adds the model's impact (in bps) to its baseline slippage for
    that fill. When no impact model is configured, ``adv_notional`` is
    ignored, preserving v10 behaviour exactly.

    *Extended in v14.* Separates the **deterministic half-spread** from
    the **stochastic slippage**. Chan (Ch. 1) repeatedly stresses that
    these are two different cost components and that lumping them
    together obscures where an executing strategy is leaking money:

    - ``slippage_bps`` — symmetric price-impact-style slippage paid on
      both sides of every trade. Historically the only knob this class
      exposed; kept for backward compatibility.
    - ``half_spread_bps`` (*new in v14*) — half of the assumed quoted
      bid-ask spread. Buys cross the ask at ``mid · (1 + half_spread)``
      and sells cross the bid at ``mid · (1 − half_spread)``. This is a
      deterministic friction that always goes against the trader,
      matching the textbook "pay the spread" execution assumption.

    The total adverse price move per fill is therefore
    ``slippage_bps + half_spread_bps + impact_bps`` (the last only when
    an ``ImpactModel`` + ``adv_notional`` are supplied), all in bps.
    When ``half_spread_bps`` is 0 (default) the v13 output is reproduced
    byte-for-byte.
    """

    def __init__(
        self,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        min_commission: float = 0.0,
        impact_model: ImpactModel | None = None,
        *,
        half_spread_bps: float = 0.0,
    ) -> None:
        if commission_bps < 0:
            raise ValueError("commission_bps must be >= 0")
        if slippage_bps < 0:
            raise ValueError("slippage_bps must be >= 0")
        if min_commission < 0:
            raise ValueError("min_commission must be >= 0")
        if half_spread_bps < 0:
            raise ValueError("half_spread_bps must be >= 0")
        self.commission_rate = commission_bps / 10_000.0
        self.slippage_rate = slippage_bps / 10_000.0
        self.half_spread_rate = half_spread_bps / 10_000.0
        self.min_commission = float(min_commission)
        self.impact_model = impact_model

    def fill_order(
        self,
        order: OrderEvent,
        market_price: float,
        adv_notional: float | None = None,
    ) -> FillEvent:
        sign = 1.0 if order.quantity >= 0 else -1.0
        # Baseline slippage (bps) is augmented by any market-impact bps
        # when an impact model is configured and ADV is provided. Impact
        # degrades gracefully to zero when ADV is missing or non-positive.
        # v14: the deterministic half-spread is added on top of both.
        total_adverse_rate = self.slippage_rate + self.half_spread_rate
        if self.impact_model is not None and adv_notional is not None:
            trade_notional = abs(float(order.quantity) * float(market_price))
            impact_bps = float(self.impact_model.impact_bps(trade_notional, adv_notional))
            total_adverse_rate += impact_bps / 10_000.0

        fill_price = float(market_price) * (1.0 + sign * total_adverse_rate)
        gross_value = abs(order.quantity * fill_price)
        commission = max(gross_value * self.commission_rate, self.min_commission if gross_value > 0 else 0.0)
        slippage_cost = abs(order.quantity) * abs(fill_price - float(market_price))
        return FillEvent(
            timestamp=order.timestamp,
            asset=order.asset,
            quantity=float(order.quantity),
            fill_price=fill_price,
            gross_value=gross_value,
            commission=commission,
            slippage_cost=slippage_cost,
            reference_price=float(market_price),
        )
