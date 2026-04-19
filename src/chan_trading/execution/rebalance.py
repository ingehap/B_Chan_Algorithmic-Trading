from __future__ import annotations

from typing import Mapping
import math

import pandas as pd

from chan_trading.types import OrderEvent


def _round_quantity(quantity: float, *, allow_fractional: bool) -> float:
    if allow_fractional:
        return float(quantity)
    if quantity >= 0:
        return float(math.floor(quantity))
    return float(math.ceil(quantity))


def target_weights_to_orders(
    timestamp: pd.Timestamp,
    target_weights: pd.Series,
    current_positions: Mapping[str, float],
    prices: pd.Series,
    equity: float,
    min_trade_value: float = 0.0,
    allow_fractional: bool = True,
    min_weight_change: float = 0.0,
) -> list[OrderEvent]:
    """Convert target weights into concrete share orders."""
    orders: list[OrderEvent] = []
    for asset in target_weights.index:
        price = float(prices.loc[asset])
        if price <= 0 or not math.isfinite(price):
            continue

        target_weight = float(target_weights.loc[asset])
        target_value = target_weight * equity
        current_quantity = float(current_positions.get(asset, 0.0))
        current_value = current_quantity * price
        current_weight = current_value / equity if equity != 0 else 0.0
        delta_value = target_value - current_value

        if abs(target_weight - current_weight) < min_weight_change:
            continue
        if abs(delta_value) < min_trade_value:
            continue

        quantity = delta_value / price
        quantity = _round_quantity(quantity, allow_fractional=allow_fractional)
        if abs(quantity) < 1e-12:
            continue

        orders.append(
            OrderEvent(
                timestamp=timestamp,
                asset=asset,
                quantity=float(quantity),
                reference_price=price,
                target_weight=target_weight,
            )
        )
    return orders
