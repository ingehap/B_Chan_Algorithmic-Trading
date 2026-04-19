"""Kelly and fractional-Kelly position sizing.

Chan (Ch. 8) treats Kelly as the reference point for sizing a systematic
strategy. Full Kelly assumes a correctly specified return distribution and
is numerically aggressive in practice, so Chan recommends **half-Kelly** (or
another fractional multiplier) as the robust default.

For a single return stream with mean ``mu`` and variance ``sigma^2``, the
Kelly fraction is ``f* = mu / sigma^2``. For a vector of assets with
return vector ``mu`` and covariance ``Sigma``, it is ``f* = Sigma^{-1} mu``.
Both expressions assume geometric growth maximization.

This module provides:

- ``kelly_fraction`` — scalar Kelly for a single return series.
- ``rolling_kelly_fraction`` — time-varying scalar Kelly fraction with a
  rolling estimation window.
- ``apply_kelly_scaling`` — rescale a DataFrame of positions by a single
  rolling Kelly fraction for the portfolio.
- ``multivariate_kelly_weights`` *(new in v10)* — full vector Kelly
  ``f* = Sigma^{-1} mu`` for a panel of asset returns, with optional
  diagonal shrinkage for numerical stability.
- ``kelly_fraction_with_drawdown_cap`` *(new in v10)* — largest
  fractional Kelly multiplier whose Monte Carlo drawdown respects a user
  tolerance at a given confidence level.

All implementations are strictly causal: the scale applied at time ``t`` uses
only data up to ``t - 1`` so there is no look-ahead bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from chan_trading.portfolio.sizing import lag_positions

_EPS = 1e-12


def kelly_fraction(
    returns: pd.Series,
    *,
    periods_per_year: int = 252,
    annualize: bool = False,
) -> float:
    """Compute the scalar Kelly fraction ``mu / sigma^2``.

    Parameters
    ----------
    returns
        Per-period returns of the strategy.
    periods_per_year
        Used only when ``annualize=True``.
    annualize
        If ``True``, return the annualized Kelly fraction. The default is
        ``False`` because the raw per-period value is typically what is used
        to scale per-period positions.
    """
    clean = returns.dropna().astype(float)
    if len(clean) < 2:
        return float("nan")
    mu = float(clean.mean())
    var = float(clean.var(ddof=0))
    if var <= _EPS or not np.isfinite(var):
        return float("nan")
    fraction = mu / var
    if annualize:
        fraction = fraction / periods_per_year
    return float(fraction)


def rolling_kelly_fraction(
    returns: pd.Series,
    *,
    lookback: int = 63,
    min_periods: int | None = None,
    clip_negative: bool = True,
) -> pd.Series:
    """Rolling per-period Kelly fraction from a return series.

    The output is strictly causal: the value at ``t`` uses the window ending at
    ``t - 1``. Callers should align positions to this series directly (no
    additional lag needed).

    Parameters
    ----------
    returns
        Per-period strategy return series.
    lookback
        Rolling window length used to estimate mean and variance.
    min_periods
        Minimum observations before a value is produced. Defaults to ``lookback``.
    clip_negative
        When ``True``, negative Kelly estimates are treated as zero. This is
        the default because Chan recommends pausing (not inverting) a strategy
        whose recent Kelly is negative.
    """
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    min_p = lookback if min_periods is None else min_periods
    if min_p < 2:
        raise ValueError("min_periods must be >= 2")

    shifted = returns.astype(float).shift(1)
    mean = shifted.rolling(lookback, min_periods=min_p).mean()
    var = shifted.rolling(lookback, min_periods=min_p).var(ddof=0)
    fraction = mean / var.replace(0.0, np.nan)
    fraction = fraction.replace([np.inf, -np.inf], np.nan)
    if clip_negative:
        fraction = fraction.clip(lower=0.0)
    return fraction


def apply_kelly_scaling(
    prices: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    kelly_fraction_multiplier: float = 0.5,
    lookback: int = 63,
    max_leverage: float = 2.0,
    clip_negative: bool = True,
) -> pd.DataFrame:
    """Scale positions by a rolling Kelly fraction computed on the strategy itself.

    The approach is:

    1. Compute the strategy's per-period return using the **unscaled**
       (already lagged) positions.
    2. Estimate a causal rolling Kelly fraction from that return series.
    3. Multiply by ``kelly_fraction_multiplier`` (``0.5`` for half-Kelly).
    4. Cap the resulting leverage at ``max_leverage``.

    Parameters
    ----------
    prices
        Price panel aligned to ``positions``.
    positions
        Target positions before Kelly scaling (weights).
    kelly_fraction_multiplier
        Multiplier on the raw Kelly fraction. ``0.5`` reproduces the
        half-Kelly rule of thumb. Must be in ``(0, 1]``.
    lookback
        Rolling window for the strategy-return-based Kelly estimate.
    max_leverage
        Hard cap on the resulting per-period Kelly scale.
    clip_negative
        Passed through to :func:`rolling_kelly_fraction`.
    """
    if not (0 < kelly_fraction_multiplier <= 1):
        raise ValueError("kelly_fraction_multiplier must be in (0, 1]")
    if max_leverage <= 0:
        raise ValueError("max_leverage must be > 0")

    asset_returns = prices[positions.columns].pct_change().fillna(0.0)
    strategy_returns = (lag_positions(positions) * asset_returns).sum(axis=1)
    fraction = rolling_kelly_fraction(
        strategy_returns,
        lookback=lookback,
        clip_negative=clip_negative,
    )
    scale = (kelly_fraction_multiplier * fraction).clip(upper=max_leverage)
    scale = scale.fillna(0.0)
    return positions.mul(scale, axis=0)


# ---------------------------------------------------------------------------
# v10 additions: multivariate Kelly and drawdown-capped Kelly
# ---------------------------------------------------------------------------


def multivariate_kelly_weights(
    returns: pd.DataFrame,
    *,
    shrinkage: float = 0.0,
) -> pd.Series:
    """Vector Kelly weights ``f* = Sigma^{-1} mu`` for a panel of assets.

    Chan (Ch. 8) notes that the scalar Kelly formula generalizes naturally
    when sizing a *portfolio* of assets: the geometric-growth-optimal
    weights are the product of the inverse return covariance and the mean
    return vector. This is strictly a one-period, full-investment result
    assuming returns are approximately normal — Chan recommends using it
    as an *upper bound* and then applying a fractional (typically half)
    multiplier in practice.

    Parameters
    ----------
    returns
        T × N panel of per-period asset returns. The index runs over time;
        each column is one asset.
    shrinkage
        Optional diagonal shrinkage intensity in ``[0, 1]``. The covariance
        used for inversion becomes
        ``(1 - shrinkage) * Sigma + shrinkage * diag(Sigma)``, which
        stabilizes ``Sigma^{-1}`` when columns are highly correlated or
        when ``T`` is small relative to ``N``. ``0.0`` (default) keeps the
        raw sample covariance.

    Returns
    -------
    weights
        ``pd.Series`` indexed by asset column name. Entries may be of
        either sign; callers typically multiply by a fractional Kelly
        multiplier and cap gross leverage.

    References
    ----------
    Chan (2013), Ch. 8; Thorp, E. (2006), "The Kelly Criterion in
    Blackjack, Sports Betting, and the Stock Market." *Handbook of Asset
    and Liability Management*, Vol. 1, Elsevier.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if returns.shape[1] < 1:
        raise ValueError("returns must have at least one column")
    if not (0.0 <= shrinkage <= 1.0):
        raise ValueError("shrinkage must be in [0, 1]")

    clean = returns.dropna(how="any").astype(float)
    if len(clean) < max(10, returns.shape[1] + 1):
        raise ValueError("Not enough rows to estimate covariance")

    mu = clean.mean().to_numpy(dtype=float)
    sigma = clean.cov(ddof=0).to_numpy(dtype=float)

    if shrinkage > 0.0:
        diag = np.diag(np.diag(sigma))
        sigma = (1.0 - shrinkage) * sigma + shrinkage * diag

    try:
        inv_sigma = np.linalg.inv(sigma)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Covariance matrix is singular; try increasing shrinkage") from exc

    weights = inv_sigma @ mu
    return pd.Series(weights, index=returns.columns, dtype=float)


def kelly_fraction_with_drawdown_cap(
    returns: pd.Series,
    *,
    max_drawdown_tolerance: float,
    confidence: float = 0.95,
    candidate_multipliers: np.ndarray | None = None,
    n_paths: int = 1000,
    horizon: int | None = None,
    method: str = "block",
    block_size: int = 20,
    seed: int | None = None,
) -> float:
    """Largest fractional-Kelly multiplier whose simulated drawdown respects a cap.

    Full Kelly is well known to produce very large transient drawdowns
    even for positive-Sharpe strategies — Chan (Ch. 8) is explicit that
    this is why he recommends half-Kelly. This helper is the
    drawdown-aware generalization: rather than picking a fixed multiplier
    like 0.5, it searches for the largest ``k`` in ``[0, 1]`` such that
    the ``confidence``-quantile of the Monte Carlo max drawdown of the
    scaled return stream ``k * returns`` is still above
    ``-max_drawdown_tolerance``.

    The search is done by evaluating the max-drawdown distribution of
    ``k * returns`` under the chosen bootstrap scheme for each candidate
    ``k``, and returning the largest ``k`` that keeps the tail drawdown
    within tolerance. Returns ``0.0`` if no candidate satisfies the
    constraint.

    Parameters
    ----------
    returns
        Per-period strategy returns (typically the return stream of a
        unit-Kelly sized strategy).
    max_drawdown_tolerance
        Maximum tolerated drawdown as a **positive** fraction
        (``0.2`` means "no worse than -20%").
    confidence
        Probability that the drawdown stays within tolerance. For example
        ``0.95`` means "the 5th percentile of the max-drawdown
        distribution must be above ``-max_drawdown_tolerance``".
    candidate_multipliers
        1-D array of candidate ``k`` values. Defaults to
        ``np.linspace(0.05, 1.0, 20)``.
    n_paths, horizon, method, block_size, seed
        Passed to the Monte Carlo engine; see
        :func:`chan_trading.risk.monte_carlo.simulate_max_drawdown`.

    Returns
    -------
    k
        Largest passing multiplier in ``[0, 1]`` (or ``0.0`` if none).

    References
    ----------
    Chan (2013), Ch. 8.
    """
    # Local import to keep the top-level module import cycle clean.
    from chan_trading.risk.monte_carlo import (
        _paths_max_drawdown,
        block_bootstrap_returns,
        bootstrap_returns,
        parametric_student_t_returns,
        stationary_bootstrap_returns,
    )

    if max_drawdown_tolerance <= 0 or max_drawdown_tolerance >= 1:
        raise ValueError("max_drawdown_tolerance must be in (0, 1)")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0, 1)")
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")

    clean = returns.dropna().astype(float)
    if len(clean) < 20:
        raise ValueError("Need at least 20 observations")

    if candidate_multipliers is None:
        candidate_multipliers = np.linspace(0.05, 1.0, 20)
    candidate_multipliers = np.asarray(candidate_multipliers, dtype=float)
    if (candidate_multipliers < 0).any() or (candidate_multipliers > 1).any():
        raise ValueError("candidate_multipliers must be within [0, 1]")

    horizon_val = horizon if horizon is not None else len(clean)
    rng = np.random.default_rng(seed)

    # Draw the bootstrapped paths ONCE using the unscaled return series,
    # then evaluate each candidate k by scaling the paths in-place. This
    # keeps the comparison across candidates apples-to-apples.
    if method == "iid":
        base_paths = bootstrap_returns(clean, horizon=horizon_val, n_paths=n_paths, rng=rng)
    elif method == "block":
        base_paths = block_bootstrap_returns(
            clean, horizon=horizon_val, n_paths=n_paths, block_size=block_size, rng=rng
        )
    elif method == "stationary":
        base_paths = stationary_bootstrap_returns(
            clean,
            horizon=horizon_val,
            n_paths=n_paths,
            expected_block_size=float(block_size),
            rng=rng,
        )
    elif method == "student_t":
        base_paths = parametric_student_t_returns(
            clean, horizon=horizon_val, n_paths=n_paths, rng=rng
        )
    else:
        raise ValueError("method must be one of 'iid', 'block', 'stationary', 'student_t'")

    # Sort candidates descending — the first passing one is the answer.
    sorted_candidates = np.sort(candidate_multipliers)[::-1]
    lower_q = 1.0 - confidence
    for k in sorted_candidates:
        if k <= 0:
            continue
        scaled = k * base_paths
        # v13: vectorised max-drawdown across all paths at once.
        dd_tail = _paths_max_drawdown(scaled)
        tail_quantile = float(np.quantile(dd_tail, lower_q))
        if tail_quantile >= -max_drawdown_tolerance:
            return float(k)
    return 0.0
