"""Futures term-structure utilities (v12).

Chapter 5 of Chan (2013) is devoted to mean-reversion patterns in currencies
and futures — especially calendar spreads, intermarket spreads, and the
distinction between **spot** returns and **roll** returns in futures
markets. None of that machinery existed in v11. This module fills the gap
with the three primitives most commonly needed in research:

- :func:`build_continuous_contract` — stitch adjacent contracts into a
  single continuous price series using either the *ratio* or *difference*
  adjustment convention.
- :func:`decompose_futures_returns` — split a continuous-contract return
  into a spot-return component (change in the front contract's raw price)
  and a roll-return component (the "jump" observed on roll dates,
  reflecting carry / backwardation / contango).
- :func:`calendar_spread` — build the simple front-minus-next calendar
  spread Chan uses as the canonical example in Ch. 5.

Conventions
-----------
All helpers operate on a DataFrame whose columns are individual contract
prices (e.g. ``["ESH24", "ESM24", "ESU24", ...]``) indexed by date. The
caller supplies a ``roll_schedule`` — a Series indexed by date whose
value at each bar is the **current front-month contract symbol**. The
roll dates are the bars where this symbol changes.

References
----------
- Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and
  Their Rationale.* Wiley, Ch. 5 (mean reversion of futures).
- CME Group methodology notes for back-adjusted continuous contracts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


_VALID_ADJUSTMENTS = ("ratio", "difference", "none")


@dataclass(slots=True)
class ContinuousContractResult:
    """Output of :func:`build_continuous_contract`.

    Attributes
    ----------
    price
        Continuous-contract price series (back-adjusted unless
        ``adjustment='none'``).
    roll_dates
        Index of bars on which the front contract rolled.
    front_symbol
        Series of the active front-contract symbol at each bar.
    raw_front_price
        Series of the *unadjusted* front-contract price (useful for
        decomposing returns into spot and roll components).
    """

    price: pd.Series
    roll_dates: pd.Index
    front_symbol: pd.Series
    raw_front_price: pd.Series


def build_continuous_contract(
    contract_prices: pd.DataFrame,
    roll_schedule: pd.Series,
    *,
    adjustment: Literal["ratio", "difference", "none"] = "ratio",
) -> ContinuousContractResult:
    """Stitch individual contracts into a single continuous price.

    Parameters
    ----------
    contract_prices
        DataFrame with one column per contract symbol; each column is the
        daily close price for that contract. Missing values (before a
        contract listed / after it expired) are permitted and denoted by
        ``NaN``.
    roll_schedule
        Series sharing ``contract_prices.index`` whose value at each bar
        is the current front-month symbol. Rolls are detected as the
        bars where this symbol changes.
    adjustment
        Back-adjustment convention:

        - ``"ratio"`` (default) — all historical values are multiplied by
          the ratio of the new-front to old-front price on the roll
          date. Preserves percent returns (the standard CSI / CME
          convention and what Chan uses in Ch. 5).
        - ``"difference"`` — a constant offset is added so that the
          roll-day new-front price equals the old-front price
          immediately before the roll. Preserves *dollar* changes but
          not percent returns; occasionally creates negative prices for
          long histories.
        - ``"none"`` — no adjustment; the series jumps on every roll.
          Useful when the caller wants to model the roll yield
          explicitly.

    Returns
    -------
    :class:`ContinuousContractResult`

    Notes
    -----
    - Every ``roll_date`` must have non-null prices for both the old and
      new front contracts, otherwise the adjustment ratio / offset is
      undefined and the function raises ``ValueError``.
    - The output is causal: bar ``t``'s price uses only prices through
      ``t``.
    """
    if not isinstance(contract_prices, pd.DataFrame):
        raise TypeError("contract_prices must be a pandas DataFrame")
    if not isinstance(roll_schedule, pd.Series):
        raise TypeError("roll_schedule must be a pandas Series")
    if not contract_prices.index.equals(roll_schedule.index):
        raise ValueError("contract_prices and roll_schedule must share an index")
    if adjustment not in _VALID_ADJUSTMENTS:
        raise ValueError(f"adjustment must be one of {sorted(_VALID_ADJUSTMENTS)}")

    symbols = roll_schedule.astype(str)
    idx = contract_prices.index
    # Pull the front-contract raw price at each bar
    raw_front = pd.Series(np.nan, index=idx, dtype=float)
    for sym in symbols.unique():
        if sym not in contract_prices.columns:
            raise ValueError(f"roll_schedule references unknown contract '{sym}'")
        mask = symbols == sym
        raw_front.loc[mask] = contract_prices.loc[mask, sym].astype(float).values

    if raw_front.isna().any():
        missing = idx[raw_front.isna()]
        raise ValueError(
            f"Front-contract price is missing on {len(missing)} bar(s), "
            f"first at {missing[0]!s}"
        )

    # Detect roll dates (symbol changed)
    changed = symbols.ne(symbols.shift(1))
    # First bar is not a roll by definition
    changed.iloc[0] = False
    roll_dates = idx[changed]

    if adjustment == "none":
        return ContinuousContractResult(
            price=raw_front.astype(float).copy(),
            roll_dates=roll_dates,
            front_symbol=symbols,
            raw_front_price=raw_front,
        )

    # Walk through roll dates in reverse, applying cumulative adjustments
    # to the price series *before* the roll. This back-adjusts — the most
    # recent contract's price is preserved and earlier prices are shifted.
    adjusted = raw_front.astype(float).copy()
    arr = adjusted.to_numpy(dtype=float).copy()

    pos_by_date = {d: i for i, d in enumerate(idx)}
    for roll_date in reversed(list(roll_dates)):
        i = pos_by_date[roll_date]
        # At index i, the *new* front (symbols[i]) is active. The *old*
        # front was symbols[i-1]. We need both contracts' prices at bar
        # i-1 (the last bar of the old regime) or equivalently at bar i
        # (first bar of the new regime). We use bar i: new price is
        # already in arr[i]; old-contract price at bar i is
        # contract_prices[old_sym][i].
        old_sym = str(symbols.iloc[i - 1])
        new_sym = str(symbols.iloc[i])
        new_price_at_roll = float(contract_prices.loc[roll_date, new_sym])
        old_price_at_roll = float(contract_prices.loc[roll_date, old_sym])
        if not (np.isfinite(new_price_at_roll) and np.isfinite(old_price_at_roll)):
            raise ValueError(
                f"Both {old_sym} and {new_sym} prices required at roll date {roll_date!s}"
            )
        if adjustment == "ratio":
            if old_price_at_roll == 0.0:
                raise ValueError(
                    f"Cannot compute ratio adjustment at {roll_date!s}: "
                    f"old-contract price {old_sym} is zero"
                )
            factor = new_price_at_roll / old_price_at_roll
            # Scale everything strictly before the roll
            arr[:i] = arr[:i] * factor
        else:  # difference
            offset = new_price_at_roll - old_price_at_roll
            arr[:i] = arr[:i] + offset

    adjusted = pd.Series(arr, index=idx, dtype=float, name="continuous")
    return ContinuousContractResult(
        price=adjusted,
        roll_dates=roll_dates,
        front_symbol=symbols,
        raw_front_price=raw_front,
    )


def decompose_futures_returns(
    result: ContinuousContractResult,
) -> pd.DataFrame:
    """Split continuous-contract returns into **spot** and **roll** components.

    Chan (Ch. 5) defines:

    - **Spot return** — the change in the raw front-contract price. On
      non-roll days this equals the observed return; on roll days it is
      *only* the within-contract price change of the new front.
    - **Roll return** — the step that appears on roll dates due to the
      price gap between the old and new front. Under ratio back-
      adjustment this gap is what makes the continuous-contract return
      diverge from pure spot return; on a backwardated curve it is
      positive, on a contangoed curve it is negative. Summed over a
      holding period, roll return is the *carry* component of total
      futures P&L.

    The decomposition satisfies:
        continuous_return[t] ≈ spot_return[t] + roll_return[t]
    (exactly for difference adjustment; approximately for ratio when
    returns are small — the ratio-adjusted decomposition is derived from
    log returns so the identity holds for log ``r``.)

    Returns a DataFrame with columns ``["continuous_return",
    "spot_return", "roll_return", "is_roll"]`` aligned to the continuous
    series' index.
    """
    cont = result.price.astype(float)
    raw = result.raw_front_price.astype(float)
    front = result.front_symbol

    cont_log_return = np.log(cont).diff()
    # Spot return = pct change in raw front *within* the same contract;
    # undefined (NaN) on roll bars because the series just switched
    # contracts. We fill those roll-bar NaNs with 0 BEFORE subtracting
    # so the roll return absorbs the full continuous-return step on the
    # roll bar rather than inheriting NaN.
    same_contract = front.eq(front.shift(1))
    spot_log_return = np.log(raw).diff().where(same_contract).fillna(0.0)

    # Roll return = continuous − spot (by construction). On non-roll
    # bars this is identically zero up to float precision because spot
    # equals continuous log return; on roll bars it captures the full
    # term-structure jump.
    roll_log_return = cont_log_return - spot_log_return
    roll_log_return = roll_log_return.where(~same_contract, 0.0)

    is_roll = (~same_contract) & cont_log_return.notna()
    is_roll.iloc[0] = False  # first bar has no previous

    return pd.DataFrame(
        {
            "continuous_return": cont_log_return,
            "spot_return": spot_log_return,
            "roll_return": roll_log_return,
            "is_roll": is_roll,
        },
        index=cont.index,
    )


def calendar_spread(
    contract_prices: pd.DataFrame,
    near_symbol: str,
    far_symbol: str,
    *,
    use_log_prices: bool = True,
) -> pd.Series:
    """Near-minus-far calendar spread for two listed contracts.

    The most direct implementation of Chan's Ch. 5 motif: when two
    adjacent futures contracts on the same underlying exhibit a stable
    long-run relationship, their price difference (or log ratio)
    frequently mean-reverts. Pass the resulting spread into
    :func:`chan_trading.features.statistics.adf_stationarity_test` and
    the standard mean-reversion diagnostics.

    Parameters
    ----------
    contract_prices
        DataFrame with both ``near_symbol`` and ``far_symbol`` as columns.
    near_symbol, far_symbol
        Column names for the near (front) and far (back) contracts.
    use_log_prices
        If ``True`` (default) the spread is ``log(near) − log(far)``
        which is scale-invariant and typically more stationary; if
        ``False`` the spread is ``near − far`` in raw price units.
    """
    for sym in (near_symbol, far_symbol):
        if sym not in contract_prices.columns:
            raise ValueError(f"'{sym}' not in contract_prices columns")
    near = contract_prices[near_symbol].astype(float)
    far = contract_prices[far_symbol].astype(float)
    if use_log_prices:
        if (near <= 0).any() or (far <= 0).any():
            raise ValueError("Log-price spread requires strictly positive prices")
        return np.log(near) - np.log(far)
    return near - far
