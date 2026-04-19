"""Multi-strategy portfolio ensemble (v12).

Chan (Ch. 6 and Ch. 8) repeatedly argues that **diversification across
uncorrelated strategies is the single biggest free lunch available to a
systematic trader** — far more reliable than trying to further optimise
any individual strategy. A mean-reversion pair, a momentum sleeve, and a
buy-on-gap basket have fundamentally different return mechanics; combining
them smooths the equity curve in ways no single strategy can.

Prior to v12 the package produced per-strategy position panels
(``PairMeanReversionStrategy``, ``TimeSeriesMomentumStrategy``,
``BuyOnGapStrategy``, ``CrossSectionalMeanReversionStrategy``,
``BasketMeanReversionStrategy``, ...) but offered no way to *combine*
them into a single portfolio panel that could then be fed to the
event-driven or vectorized backtester. This module fills that gap.

Weighting schemes
-----------------

All schemes share the same two-step structure:

1. Compute a per-strategy scalar weight (equal, inverse vol, risk parity,
   or a user-supplied vector).
2. Multiply each strategy's position panel by its scalar weight and sum,
   then optionally cap gross leverage.

Inverse-vol and risk-parity weights are **strictly causal** — they use a
rolling estimate of each strategy's realised return volatility through
bar ``t`` only, never peeking. This matches the look-ahead hygiene
applied elsewhere in the package.

References
----------
- Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and
  Their Rationale.* Wiley, Ch. 6 (diversifying across momentum
  variants) and Ch. 8 (risk management through diversification).
- Maillard, S., Roncalli, T., & Teïletche, J. (2010). "The Properties
  of Equally Weighted Risk Contribution Portfolios." *Journal of
  Portfolio Management*, 36(4), 60-70. (Risk parity reference.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


_VALID_SCHEMES = ("equal_weight", "inverse_vol", "risk_parity", "custom")


@dataclass(slots=True)
class EnsembleConfig:
    """Configuration for :func:`combine_strategies`.

    Parameters
    ----------
    scheme
        One of:

        - ``"equal_weight"`` — every strategy gets the same scalar weight
          ``1 / n_strategies``.
        - ``"inverse_vol"`` — strategy ``i`` gets weight
          ``(1 / sigma_i) / sum_j (1 / sigma_j)`` where ``sigma_i`` is
          the rolling realised volatility of strategy ``i``'s return.
        - ``"risk_parity"`` — the naive risk-parity simplification that
          also uses ``1 / sigma`` (identical to ``inverse_vol`` for
          uncorrelated strategies; included for API symmetry and to
          signal intent).
        - ``"custom"`` — use ``custom_weights`` directly.
    vol_lookback
        Window used to estimate each strategy's rolling volatility for
        the inverse-vol / risk-parity schemes. Ignored under the other
        schemes.
    max_gross_leverage
        Optional cap on the gross leverage of the *combined* position
        panel, applied row-by-row after summation. ``None`` (default)
        means no cap.
    custom_weights
        Only used when ``scheme="custom"``. Must be provided, must be
        non-negative, and must sum to a positive value.
    """

    scheme: str = "equal_weight"
    vol_lookback: int = 63
    max_gross_leverage: float | None = None
    custom_weights: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.scheme not in _VALID_SCHEMES:
            raise ValueError(f"scheme must be one of {sorted(_VALID_SCHEMES)}")
        if self.vol_lookback < 2:
            raise ValueError("vol_lookback must be >= 2")
        if self.max_gross_leverage is not None and self.max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be > 0 when provided")
        if self.scheme == "custom":
            if self.custom_weights is None:
                raise ValueError("custom_weights must be provided when scheme='custom'")
            if any(w < 0 for w in self.custom_weights.values()):
                raise ValueError("custom_weights must be non-negative")
            total = float(sum(self.custom_weights.values()))
            if total <= 0:
                raise ValueError("custom_weights must sum to > 0")


def _strategy_returns(prices: pd.DataFrame, positions: pd.DataFrame) -> pd.Series:
    """Per-bar return of one strategy given shared ``prices`` (lagged positions).

    The strategy panel is lag-shifted by one bar so the return at ``t``
    reflects the position taken at the close of bar ``t-1``, matching
    the rest of the package's convention.
    """
    cols = [c for c in positions.columns if c in prices.columns]
    asset_returns = prices[cols].pct_change().fillna(0.0)
    lagged = positions[cols].shift(1).fillna(0.0)
    return (lagged * asset_returns).sum(axis=1)


def _rolling_inverse_vol_weights(
    strategy_returns: pd.DataFrame,
    *,
    lookback: int,
) -> pd.DataFrame:
    """Rolling ``1 / sigma`` weights normalised to sum to 1 per bar.

    Strictly causal: both ``sigma`` and the normalisation use only the
    window ending at bar ``t``.
    """
    if strategy_returns.shape[1] == 0:
        return strategy_returns.copy()

    vol = strategy_returns.rolling(lookback, min_periods=lookback).std(ddof=0)
    # Replace zero / NaN vol with NaN so they drop out of the sum below.
    vol = vol.replace(0.0, np.nan)
    inv = 1.0 / vol

    # Normalise row-wise: ignore NaNs, fall back to equal weights across
    # the *currently-valid* subset if no volatility estimate is available.
    row_sum = inv.sum(axis=1, skipna=True).replace(0.0, np.nan)
    weights = inv.div(row_sum, axis=0)

    # Warmup rows: equal weight across all strategies until the vol
    # estimate exists. This is a conservative default — it keeps the
    # combined panel non-trivial during the rolling-vol warmup instead of
    # producing zero positions.
    n = strategy_returns.shape[1]
    eq_weights = pd.Series(1.0 / n, index=strategy_returns.columns, dtype=float)
    warmup_mask = weights.isna().all(axis=1)
    weights.loc[warmup_mask] = eq_weights.values
    # Any remaining NaNs (some strategies warmed up, some not) get filled
    # with 0 and the row is re-normalised.
    weights = weights.fillna(0.0)
    row_sum2 = weights.sum(axis=1).replace(0.0, np.nan)
    weights = weights.div(row_sum2, axis=0).fillna(0.0)
    return weights


def combine_strategies(
    prices: pd.DataFrame,
    strategies: Mapping[str, pd.DataFrame],
    *,
    config: EnsembleConfig | None = None,
) -> pd.DataFrame:
    """Combine several strategy position panels into one ensemble panel.

    Parameters
    ----------
    prices
        Shared price panel, used only for strategy-return computation in
        the inverse-vol and risk-parity schemes. Must cover every asset
        that appears in any strategy's panel.
    strategies
        Mapping ``{strategy_name: position_panel}``. All panels must
        share the same index as ``prices``. Individual panels may have
        different column sets; the combined output is aligned to the
        *union* of asset columns, with missing entries filled with zero.
    config
        An :class:`EnsembleConfig`; defaults are used if omitted.

    Returns
    -------
    combined
        Combined position panel indexed by ``prices.index`` with columns
        equal to the union of all strategies' asset columns. Under
        ``equal_weight`` and ``custom`` schemes the weights are constant
        through time; under ``inverse_vol`` / ``risk_parity`` the
        per-strategy weight varies bar-by-bar.

    Notes
    -----
    Because each strategy's panel may already include its own gross-
    leverage cap, the raw sum can exceed 1.0 unit of gross exposure. Use
    ``config.max_gross_leverage`` to impose a combined cap; it is applied
    row-by-row after summation.

    Examples
    --------
    >>> combined = combine_strategies(
    ...     prices,
    ...     {"pair_mr": pair_positions, "momentum": mom_positions,
    ...      "buy_on_gap": gap_positions},
    ...     config=EnsembleConfig(scheme="inverse_vol", vol_lookback=63,
    ...                           max_gross_leverage=1.0),
    ... )
    """
    cfg = config or EnsembleConfig()
    if not strategies:
        raise ValueError("strategies mapping must not be empty")
    for name, panel in strategies.items():
        if not isinstance(panel, pd.DataFrame):
            raise TypeError(f"strategy '{name}' must be a pandas DataFrame")
        if not panel.index.equals(prices.index):
            raise ValueError(f"strategy '{name}' index must equal prices.index")

    names = list(strategies.keys())
    # Union of asset columns across all strategies
    asset_cols: list[str] = []
    seen: set[str] = set()
    for panel in strategies.values():
        for col in panel.columns:
            if col not in seen:
                seen.add(col)
                asset_cols.append(col)
    missing_from_prices = [c for c in asset_cols if c not in prices.columns]
    if missing_from_prices:
        raise ValueError(
            f"prices is missing columns required by strategies: {missing_from_prices}"
        )

    # Align every panel to the common index + column layout
    aligned: dict[str, pd.DataFrame] = {}
    for name, panel in strategies.items():
        aligned[name] = panel.reindex(columns=asset_cols).fillna(0.0)

    if cfg.scheme == "equal_weight":
        w = 1.0 / len(names)
        scalar_weights = pd.DataFrame(
            {name: [w] * len(prices.index) for name in names},
            index=prices.index,
        )
    elif cfg.scheme == "custom":
        assert cfg.custom_weights is not None  # validated in __post_init__
        missing = set(names) - set(cfg.custom_weights.keys())
        if missing:
            raise ValueError(f"custom_weights missing entries for strategies: {sorted(missing)}")
        total = float(sum(cfg.custom_weights[name] for name in names))
        scalar_weights = pd.DataFrame(
            {name: [cfg.custom_weights[name] / total] * len(prices.index) for name in names},
            index=prices.index,
        )
    else:  # inverse_vol or risk_parity — same math for this simplified form
        ret_panel = pd.DataFrame(
            {name: _strategy_returns(prices, aligned[name]) for name in names},
            index=prices.index,
        )
        scalar_weights = _rolling_inverse_vol_weights(ret_panel, lookback=cfg.vol_lookback)

    # Combine
    combined = pd.DataFrame(0.0, index=prices.index, columns=asset_cols, dtype=float)
    for name in names:
        combined = combined + aligned[name].mul(scalar_weights[name], axis=0)

    if cfg.max_gross_leverage is not None:
        gross = combined.abs().sum(axis=1)
        scale = pd.Series(1.0, index=prices.index, dtype=float)
        mask = gross > cfg.max_gross_leverage
        scale.loc[mask] = cfg.max_gross_leverage / gross.loc[mask]
        combined = combined.mul(scale, axis=0)
    return combined.fillna(0.0)


def strategy_correlation_matrix(
    prices: pd.DataFrame,
    strategies: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Pairwise Pearson correlation of strategies' per-bar returns.

    A diagnostic companion to :func:`combine_strategies`: it reveals
    whether the strategies are actually diversifying one another or
    simply stacking the same factor. Chan (Ch. 6) emphasises that the
    value of combining strategies is proportional to 1 − their average
    pairwise correlation; this helper surfaces that number directly.
    """
    if not strategies:
        raise ValueError("strategies mapping must not be empty")
    ret_panel = pd.DataFrame(
        {name: _strategy_returns(prices, panel) for name, panel in strategies.items()},
        index=prices.index,
    )
    return ret_panel.corr()
