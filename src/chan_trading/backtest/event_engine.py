from __future__ import annotations

import pandas as pd

from chan_trading.backtest.broker import SimulatedBroker
from chan_trading.config import EventBacktestConfig
from chan_trading.execution.rebalance import target_weights_to_orders
from chan_trading.types import EventBacktestResult


def _positions_to_weights(
    positions: dict[str, float],
    prices: pd.Series,
    equity: float,
) -> dict[str, float]:
    if equity == 0:
        return {asset: 0.0 for asset in prices.index}
    return {
        asset: float(positions.get(asset, 0.0) * float(prices.loc[asset]) / equity)
        for asset in prices.index
    }


def _apply_daily_financing(
    *,
    cash: float,
    positions: dict[str, float],
    prices: pd.Series,
    borrow_bps_annual: float,
    cash_interest_bps_annual: float,
    periods_per_year: int = 252,
) -> tuple[float, float, float]:
    short_notional = 0.0
    for asset, quantity in positions.items():
        if quantity < 0:
            short_notional += abs(float(quantity) * float(prices.loc[asset]))
    borrow_cost = short_notional * (borrow_bps_annual / 10_000.0) / periods_per_year
    cash_interest = max(cash, 0.0) * (cash_interest_bps_annual / 10_000.0) / periods_per_year
    updated_cash = cash - borrow_cost + cash_interest
    return float(updated_cash), float(borrow_cost), float(cash_interest)




def _effective_target_shift(config: EventBacktestConfig) -> int:
    """Map fill mode to an effective target-weight lag."""
    if config.fill_mode == "next_bar":
        return config.lag_target_weights
    return max(config.lag_target_weights - 1, 0)


def run_event_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    config: EventBacktestConfig | None = None,
) -> EventBacktestResult:
    """Run a daily-bar event-driven backtest using target weights."""
    cfg = config or EventBacktestConfig()

    if not prices.index.equals(target_weights.index):
        raise ValueError("prices and target_weights must have matching indices")
    if list(prices.columns) != list(target_weights.columns):
        raise ValueError("prices and target_weights must have identical columns in the same order")

    broker = SimulatedBroker(
        commission_bps=cfg.commission_bps,
        slippage_bps=cfg.slippage_bps,
        min_commission=cfg.min_commission,
        impact_model=cfg.impact_model,
        half_spread_bps=cfg.half_spread_bps,
    )
    effective_shift = _effective_target_shift(cfg)
    live_targets = target_weights.shift(effective_shift).fillna(0.0)

    # If an ADV panel was supplied with the impact model, align it to the
    # price index. v11 new behaviour: impact_model and adv are both-or-
    # neither (see EventBacktestConfig.__post_init__).
    adv_panel = cfg.adv
    if adv_panel is not None:
        import pandas as _pd
        if not isinstance(adv_panel, _pd.DataFrame):
            raise TypeError("EventBacktestConfig.adv must be a pandas DataFrame")
        if not adv_panel.index.equals(prices.index):
            raise ValueError("ADV panel must share the same index as prices")
        missing_cols = set(prices.columns) - set(adv_panel.columns)
        if missing_cols:
            raise ValueError(f"ADV panel is missing columns: {sorted(missing_cols)}")

    cash = float(cfg.initial_cash)
    positions = {asset: 0.0 for asset in prices.columns}
    equity_history: list[float] = []
    position_rows: list[dict[str, float]] = []
    trade_rows: list[dict[str, float]] = []
    order_rows: list[dict[str, object]] = []
    fill_rows: list[dict[str, object]] = []

    for timestamp in prices.index:
        price_row = prices.loc[timestamp].astype(float)
        portfolio_value = sum(float(positions[a]) * float(price_row.loc[a]) for a in prices.columns)
        cash, borrow_cost, cash_interest = _apply_daily_financing(
            cash=cash,
            positions=positions,
            prices=price_row,
            borrow_bps_annual=cfg.borrow_bps_annual,
            cash_interest_bps_annual=cfg.cash_interest_bps_annual,
        )
        equity_before = cash + portfolio_value
        desired = live_targets.loc[timestamp].astype(float)

        orders = target_weights_to_orders(
            timestamp=timestamp,
            target_weights=desired,
            current_positions=positions,
            prices=price_row,
            equity=equity_before,
            min_trade_value=cfg.min_trade_value,
            allow_fractional=cfg.allow_fractional,
            min_weight_change=cfg.min_weight_change,
        )

        for order in orders:
            order_rows.append(
                {
                    "timestamp": order.timestamp,
                    "asset": order.asset,
                    "quantity": order.quantity,
                    "reference_price": order.reference_price,
                    "target_weight": order.target_weight,
                }
            )
            adv_for_fill: float | None = None
            if adv_panel is not None:
                adv_for_fill = float(adv_panel.loc[timestamp, order.asset])
            fill = broker.fill_order(
                order,
                market_price=float(price_row.loc[order.asset]),
                adv_notional=adv_for_fill,
            )
            positions[fill.asset] += fill.quantity
            cash -= fill.quantity * fill.fill_price
            cash -= fill.commission
            fill_rows.append(
                {
                    "timestamp": fill.timestamp,
                    "asset": fill.asset,
                    "quantity": fill.quantity,
                    "fill_price": fill.fill_price,
                    "gross_value": fill.gross_value,
                    "commission": fill.commission,
                    "slippage_cost": fill.slippage_cost,
                    "reference_price": fill.reference_price,
                }
            )

        portfolio_value_after = sum(float(positions[a]) * float(price_row.loc[a]) for a in prices.columns)
        equity_after = cash + portfolio_value_after
        equity_history.append(equity_after)
        position_rows.append(_positions_to_weights(positions, price_row, equity_after))
        trade_rows.append(
            {
                "timestamp": timestamp,
                "equity": equity_after,
                "cash": cash,
                "portfolio_value": portfolio_value_after,
                "n_orders": float(len(orders)),
                "borrow_cost": borrow_cost,
                "cash_interest": cash_interest,
            }
        )

    equity_curve = pd.Series(equity_history, index=prices.index, name="equity")
    returns = equity_curve.pct_change().fillna(0.0)
    positions_df = pd.DataFrame(position_rows, index=prices.index, columns=prices.columns).fillna(0.0)
    trades_df = pd.DataFrame(trade_rows).set_index("timestamp")

    if order_rows:
        order_log = pd.DataFrame(order_rows).set_index("timestamp")
    else:
        order_log = pd.DataFrame(columns=["asset", "quantity", "reference_price", "target_weight"])
        order_log.index.name = "timestamp"

    if fill_rows:
        fill_log = pd.DataFrame(fill_rows).set_index("timestamp")
    else:
        fill_log = pd.DataFrame(
            columns=["asset", "quantity", "fill_price", "gross_value", "commission", "slippage_cost", "reference_price"]
        )
        fill_log.index.name = "timestamp"

    return EventBacktestResult(
        equity_curve=equity_curve,
        returns=returns,
        positions=positions_df,
        trades=trades_df,
        order_log=order_log,
        fill_log=fill_log,
    )
