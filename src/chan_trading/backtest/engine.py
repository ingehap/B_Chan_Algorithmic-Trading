from __future__ import annotations

import pandas as pd

from chan_trading.portfolio.sizing import lag_positions, turnover
from chan_trading.types import BacktestResult


def run_backtest(
    prices: pd.DataFrame,
    target_positions: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
    *,
    borrow_bps_annual: float = 0.0,
    cash_interest_bps_annual: float = 0.0,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Run a simple daily close-to-close vectorized backtest.

    *Extended in v12.* Added optional ``borrow_bps_annual`` and
    ``cash_interest_bps_annual`` so the vectorized engine charges the
    same financing side effects the event-driven engine already models.
    Defaults remain ``0`` so the v11 output is reproduced byte-for-byte
    when neither new argument is set.

    Mechanics
    ---------
    - **Borrow cost** — per-bar charge on the dollar-weight of *short*
      positions: ``short_weight_t · borrow_bps_annual / 10_000 /
      periods_per_year``.
    - **Cash interest** — per-bar credit on the *unused* fraction of
      capital, approximated as ``max(1 − gross_leverage_t, 0)``.

    Both adjustments enter the net-return stream additively; they do not
    change the gross-return calculation.
    """
    if not prices.index.equals(target_positions.index):
        raise ValueError("prices and target_positions must have matching indices")
    if set(target_positions.columns) - set(prices.columns):
        raise ValueError("All target position columns must exist in prices")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be >= 0")
    if borrow_bps_annual < 0:
        raise ValueError("borrow_bps_annual must be >= 0")
    if cash_interest_bps_annual < 0:
        raise ValueError("cash_interest_bps_annual must be >= 0")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")

    returns = prices[target_positions.columns].pct_change().fillna(0.0)
    live_positions = lag_positions(target_positions, periods=1)
    gross_returns = (live_positions * returns).sum(axis=1)

    cost_rate = transaction_cost_bps / 10_000.0
    costs = turnover(live_positions) * cost_rate

    # v12 financing side effects (zero-cost when both annual bps == 0).
    borrow_rate = borrow_bps_annual / 10_000.0 / periods_per_year
    cash_rate = cash_interest_bps_annual / 10_000.0 / periods_per_year

    short_weight = live_positions.clip(upper=0.0).abs().sum(axis=1)
    gross_leverage = live_positions.abs().sum(axis=1)
    cash_weight = (1.0 - gross_leverage).clip(lower=0.0)

    borrow_cost_series = short_weight * borrow_rate
    cash_interest_series = cash_weight * cash_rate

    net_returns = gross_returns - costs - borrow_cost_series + cash_interest_series
    equity_curve = (1.0 + net_returns).cumprod()

    trades = pd.DataFrame(
        {
            "turnover": turnover(live_positions),
            "cost": costs,
            "borrow_cost": borrow_cost_series,
            "cash_interest": cash_interest_series,
            "gross_return": gross_returns,
            "net_return": net_returns,
        },
        index=prices.index,
    )
    return BacktestResult(
        equity_curve=equity_curve,
        returns=net_returns,
        positions=live_positions,
        trades=trades,
    )
