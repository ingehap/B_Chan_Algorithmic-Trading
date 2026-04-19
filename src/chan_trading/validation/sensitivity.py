"""Cost-sensitivity analysis for event-driven backtests.

Chan's core warning in Chapter 1 is that an edge observed in a backtest
can evaporate once realistic transaction costs are applied — and that
intraday and high-frequency strategies are *especially* fragile to
cost assumptions. The standard defence is not to pick a single
commission/slippage pair, but to **sweep the cost grid** and inspect how
key performance metrics (Sharpe, max drawdown, net return) behave as
costs increase.

This module provides :func:`cost_sensitivity_sweep`, which re-runs the
event-driven engine across a grid of ``(commission_bps, slippage_bps)``
combinations and returns a tidy ``pd.DataFrame`` indexed by that grid.
The output makes the strategy's cost elasticity immediately visible — a
shallow slope means the edge is robust, a steep slope means the live
fill assumptions need to be very carefully calibrated before deploying.

All sweeps use the same pre-computed ``target_weights``, so the *only*
thing varying across rows is the cost regime. This isolates cost
sensitivity from any look-ahead or re-estimation artefacts.

References
----------
- Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and
  Their Rationale.* Wiley, Ch. 1.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np
import pandas as pd

from chan_trading.backtest.event_engine import run_event_backtest
from chan_trading.config import EventBacktestConfig
from chan_trading.risk.metrics import (
    annualized_return,
    max_drawdown,
    sharpe_ratio,
)


def cost_sensitivity_sweep(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    commission_bps_grid: Iterable[float],
    slippage_bps_grid: Iterable[float],
    base_config: EventBacktestConfig | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Run the event-driven backtest across a (commission, slippage) grid.

    Parameters
    ----------
    prices
        Price panel to feed the engine.
    target_weights
        Pre-computed target weights aligned to ``prices``. They are held
        constant across the sweep so the only difference between runs is
        the cost regime.
    commission_bps_grid
        Iterable of commission-in-bps values to test.
    slippage_bps_grid
        Iterable of slippage-in-bps values to test.
    base_config
        Base :class:`~chan_trading.config.EventBacktestConfig`. The sweep
        *replaces* ``commission_bps`` and ``slippage_bps`` on this base;
        all other fields (fill mode, borrow cost, impact model, …) are
        preserved across runs. If ``None``, a default config is used.
    periods_per_year
        Annualisation factor for the reported Sharpe and return.

    Returns
    -------
    summary
        DataFrame with one row per ``(commission, slippage)`` pair. Columns:

        - ``commission_bps``
        - ``slippage_bps``
        - ``sharpe``
        - ``annual_return``
        - ``max_drawdown``
        - ``terminal_equity``
        - ``n_orders``

    Notes
    -----
    The sweep is ``O(len(commission_grid) × len(slippage_grid))`` full
    backtests. Keep the grid small (e.g. 3-5 points per axis) on large
    return panels.
    """
    base = base_config or EventBacktestConfig()
    commission_list = [float(c) for c in commission_bps_grid]
    slippage_list = [float(s) for s in slippage_bps_grid]
    if not commission_list or not slippage_list:
        raise ValueError("commission_bps_grid and slippage_bps_grid must be non-empty")
    if any(c < 0 for c in commission_list) or any(s < 0 for s in slippage_list):
        raise ValueError("grid values must be >= 0")

    rows: list[dict[str, float]] = []
    for commission in commission_list:
        for slippage in slippage_list:
            cfg = replace(base, commission_bps=commission, slippage_bps=slippage)
            result = run_event_backtest(prices, target_weights, cfg)
            returns = result.returns
            equity = result.equity_curve
            total_orders = float(result.order_log.shape[0])
            rows.append(
                {
                    "commission_bps": commission,
                    "slippage_bps": slippage,
                    "sharpe": sharpe_ratio(returns, periods_per_year=periods_per_year),
                    "annual_return": annualized_return(returns, periods_per_year=periods_per_year),
                    "max_drawdown": max_drawdown(equity),
                    "terminal_equity": float(equity.iloc[-1]) if len(equity) else float("nan"),
                    "n_orders": total_orders,
                }
            )

    summary = pd.DataFrame(rows)
    # Leave the index as a simple RangeIndex — the grid columns are
    # explicit, so callers can pivot or group as they wish.
    return summary


def breakeven_cost_bps(
    sweep_summary: pd.DataFrame,
    *,
    metric: str = "sharpe",
    target: float = 0.0,
    axis: str = "slippage_bps",
) -> float:
    """Linearly interpolate the breakeven cost that drives ``metric`` to ``target``.

    Useful follow-up to :func:`cost_sensitivity_sweep` when the user wants
    a single number to report: *"how many bps of slippage until the
    Sharpe ratio hits zero?"*. The interpolation is done on the two
    adjacent sweep points that bracket ``target``; if no bracket exists
    the function returns ``+inf`` (metric never crosses) or ``-inf``
    (already below target at zero).

    Parameters
    ----------
    sweep_summary
        Output of :func:`cost_sensitivity_sweep`, filtered so the other
        axis is held constant. For example, to find the slippage
        breakeven at commission = 0 bps, pass
        ``sweep[sweep.commission_bps == 0]``.
    metric
        Column name in ``sweep_summary`` to cross against ``target``.
    target
        Metric value defining breakeven.
    axis
        Column name in ``sweep_summary`` giving the varying cost axis.
    """
    if sweep_summary.empty:
        return float("nan")
    data = sweep_summary.sort_values(axis).reset_index(drop=True)
    xs = data[axis].to_numpy(dtype=float)
    ys = data[metric].to_numpy(dtype=float)
    if np.any(np.isnan(ys)):
        return float("nan")

    # If the starting point is already below target, report -inf (we can't
    # take negative costs to reach it).
    if ys[0] <= target:
        return float("-inf")
    # If the series never crosses target, report +inf.
    if (ys > target).all():
        return float("inf")

    # Find the first bracket where ys crosses from above to at/below target.
    for i in range(1, len(ys)):
        if ys[i] <= target:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            if y0 == y1:
                return float(x1)
            # Linear interpolation: solve y0 + (y1-y0)*(x-x0)/(x1-x0) = target
            return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
    return float("inf")
