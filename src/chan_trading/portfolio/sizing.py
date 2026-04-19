from __future__ import annotations

import numpy as np
import pandas as pd


def lag_positions(positions: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Lag positions to avoid same-bar look-ahead bias."""
    if periods < 0:
        raise ValueError("periods must be >= 0")
    return positions.shift(periods).fillna(0.0)


def turnover(positions: pd.DataFrame) -> pd.Series:
    """Approximate turnover from absolute position changes."""
    return positions.diff().abs().sum(axis=1).fillna(0.0)


def _portfolio_realized_vol(
    prices: pd.DataFrame,
    positions: pd.DataFrame,
    lookback: int,
    periods_per_year: int = 252,
) -> pd.Series:
    asset_returns = prices[positions.columns].pct_change().fillna(0.0)
    port_returns = (lag_positions(positions) * asset_returns).sum(axis=1)
    rolling_vol = port_returns.rolling(lookback, min_periods=lookback).std(ddof=0)
    return pd.Series(rolling_vol * np.sqrt(periods_per_year), index=positions.index, dtype=float)


def apply_vol_target(
    prices: pd.DataFrame,
    positions: pd.DataFrame,
    target_vol: float,
    lookback: int = 20,
    max_leverage: float = 2.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Scale positions to target annualized portfolio volatility."""
    if target_vol <= 0:
        raise ValueError("target_vol must be > 0")
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    if max_leverage <= 0:
        raise ValueError("max_leverage must be > 0")

    realized_vol = _portfolio_realized_vol(
        prices=prices,
        positions=positions,
        lookback=lookback,
        periods_per_year=periods_per_year,
    ).replace(0.0, np.nan)

    scaler = (target_vol / realized_vol).clip(upper=max_leverage)
    scaler = scaler.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return positions.mul(scaler, axis=0)


def apply_exposure_caps(
    positions: pd.DataFrame,
    *,
    max_gross_exposure: float | None = None,
    max_net_exposure: float | None = None,
) -> pd.DataFrame:
    """Enforce row-wise gross- and net-exposure caps on a position panel.

    *Promoted to a shared utility in v13.* Previously this logic lived
    inside ``chan_trading.strategies.mean_reversion._apply_exposure_caps``
    and was only wired into the pair and basket mean-reversion
    strategies — momentum, cross-sectional, and buy-on-gap strategies
    silently ignored their config fields for gross / net caps. v13
    promotes the function to the shared ``portfolio.sizing`` namespace
    and makes it available to all strategies (see the ``apply_caps=...``
    kwarg on each ``generate_positions``) and to arbitrary
    post-processing pipelines.

    Chan (Ch. 8) repeatedly stresses that gross-leverage caps and
    net-exposure caps are a **part of the strategy**, not an
    afterthought — a strategy whose position-sizing math occasionally
    produces 3× gross leverage or a large net tilt is not the same
    strategy as its cap-respecting counterpart.

    Parameters
    ----------
    positions
        Position / target-weight panel indexed by time.
    max_gross_exposure
        Optional cap on ``positions.abs().sum(axis=1)`` per row.
        ``None`` disables the cap. When binding, every asset in that row
        is scaled by ``cap / gross``.
    max_net_exposure
        Optional cap on ``|positions.sum(axis=1)|`` per row. ``None``
        disables the cap. Applied **after** the gross cap so the
        interaction order is well-defined.

    Returns
    -------
    capped
        A new DataFrame, same shape and index as ``positions``. Rows
        below the caps are untouched.
    """
    if max_gross_exposure is None and max_net_exposure is None:
        return positions.copy()
    capped = positions.copy()
    if max_gross_exposure is not None:
        if max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be > 0 when provided")
        gross = capped.abs().sum(axis=1)
        gross_scale = pd.Series(1.0, index=capped.index)
        gross_mask = gross > max_gross_exposure
        gross_scale.loc[gross_mask] = max_gross_exposure / gross.loc[gross_mask]
        capped = capped.mul(gross_scale, axis=0)
    if max_net_exposure is not None:
        if max_net_exposure < 0:
            raise ValueError("max_net_exposure must be >= 0 when provided")
        net = capped.sum(axis=1).abs()
        net_scale = pd.Series(1.0, index=capped.index)
        net_mask = net > max_net_exposure
        net_scale.loc[net_mask] = max_net_exposure / net.loc[net_mask]
        capped = capped.mul(net_scale, axis=0)
    return capped.fillna(0.0)


def apply_turnover_throttle(
    positions: pd.DataFrame,
    *,
    max_turnover_per_bar: float,
) -> pd.DataFrame:
    """Cap per-bar turnover by blending toward the previous bar's positions.

    *New in v13.* Chan (Ch. 1) warns that backtests which look great on
    paper often do so because they implicitly assume frictionless,
    instant rebalancing — an assumption that breaks down fast when
    turnover gets large. A hard cap on ``Σᵢ |Δwᵢ|`` per bar is a cheap
    way to keep a strategy's trading footprint within what a realistic
    execution system can absorb, especially for intraday or
    cross-sectional strategies where the raw signal can demand nearly
    100% turnover every day.

    For each bar ``t`` let ``target_t`` be the desired position vector
    and ``live_{t-1}`` be the result after throttling through ``t-1``.
    If the raw turnover
    ``τ_t = Σᵢ |target_{t,i} − live_{t-1,i}| > max_turnover_per_bar``
    the realised position is a convex combination
    ``live_t = α · target_t + (1 − α) · live_{t-1}`` with
    ``α = max_turnover_per_bar / τ_t``; otherwise ``live_t = target_t``.
    This is the natural extension of "do at most X bps of turnover
    today" — it does not forbid a large target move, it simply smears
    it across days.

    Parameters
    ----------
    positions
        Target-weight panel.
    max_turnover_per_bar
        Cap on per-bar gross position change, in the same units as
        ``positions.abs().sum()``. Must be ``> 0``.

    Returns
    -------
    throttled
        A new panel the same shape as ``positions``. The first bar is
        always passed through unchanged (there is no "previous"
        position to anchor against).
    """
    if max_turnover_per_bar <= 0:
        raise ValueError("max_turnover_per_bar must be > 0")
    target = positions.astype(float).to_numpy(copy=True)
    if target.shape[0] == 0:
        return positions.copy()
    live = np.empty_like(target)
    live[0] = target[0]
    for t in range(1, target.shape[0]):
        delta = target[t] - live[t - 1]
        raw_turnover = float(np.abs(delta).sum())
        if raw_turnover <= max_turnover_per_bar or raw_turnover == 0.0:
            live[t] = target[t]
        else:
            alpha = max_turnover_per_bar / raw_turnover
            live[t] = live[t - 1] + alpha * delta
    return pd.DataFrame(live, index=positions.index, columns=positions.columns, dtype=float)


def apply_drawdown_throttle(
    positions: pd.DataFrame,
    equity_curve: pd.Series,
    *,
    soft_limit: float | None = None,
    soft_scale: float = 0.5,
    hard_limit: float | None = None,
    lag: int = 0,
) -> pd.DataFrame:
    """Scale positions down when the running drawdown breaches thresholds.

    *Extended in v13.* A new ``lag`` keyword (default ``0``, preserving
    v12 output byte-for-byte) lets callers enforce strict causality
    between equity and throttle: ``lag=1`` applies the throttle computed
    from equity at ``t-1`` to positions at ``t``. The v12 code used the
    equity at ``t`` to decide the throttle at ``t``, which is still
    causal **iff** the downstream engine lag-shifts positions by one
    bar (as every engine in this package does) but can be confusing in
    standalone use. Use ``lag=1`` when backtesting the throttled panel
    directly without any further lagging, or when porting to live code.
    """
    if soft_limit is None and hard_limit is None:
        return positions
    if not (0 < soft_scale <= 1):
        raise ValueError("soft_scale must be in (0, 1]")
    if lag < 0:
        raise ValueError("lag must be >= 0")

    running_max = equity_curve.cummax()
    drawdown = 1.0 - equity_curve / running_max.replace(0.0, np.nan)
    scale = pd.Series(1.0, index=positions.index, dtype=float)

    if soft_limit is not None:
        scale.loc[drawdown >= soft_limit] = soft_scale
    if hard_limit is not None:
        scale.loc[drawdown >= hard_limit] = 0.0
    if lag > 0:
        scale = scale.shift(lag).fillna(1.0)
    return positions.mul(scale.fillna(1.0), axis=0)


def cppi_scale(
    equity_curve: pd.Series,
    *,
    floor_fraction: float,
    multiplier: float = 3.0,
    max_scale: float = 1.0,
) -> pd.Series:
    """Compute a Constant-Proportion-Portfolio-Insurance scaling series.

    Chan (Ch. 8) recommends CPPI as a smoother, continuous drawdown control
    compared to a binary soft/hard throttle. The rule is:

        cushion_t = max(equity_t - floor_t, 0)
        scale_t   = clip(multiplier * cushion_t / equity_t, 0, max_scale)

    where ``floor_t = floor_fraction * running_max_t``. As equity falls toward
    the floor, the cushion shrinks and scale goes smoothly to zero. As equity
    rises, the floor ratchets up with the new high-water mark so that realized
    gains are partially locked in.

    Parameters
    ----------
    equity_curve
        Running equity series.
    floor_fraction
        Floor as a fraction of the running high-water mark. ``0.8`` means the
        strategy targets not giving back more than 20% from any equity peak.
        Must be in ``(0, 1)``.
    multiplier
        CPPI multiplier ``m``. Typical values are 2-5. Larger values are more
        aggressive (hold more risky exposure for the same cushion).
    max_scale
        Upper cap on the resulting scale. Default 1.0 means CPPI only scales
        down, never up.
    """
    if not (0 < floor_fraction < 1):
        raise ValueError("floor_fraction must be in (0, 1)")
    if multiplier <= 0:
        raise ValueError("multiplier must be > 0")
    if max_scale <= 0:
        raise ValueError("max_scale must be > 0")

    running_max = equity_curve.cummax()
    floor = floor_fraction * running_max
    cushion = (equity_curve - floor).clip(lower=0.0)
    equity_safe = equity_curve.replace(0.0, np.nan)
    scale = (multiplier * cushion / equity_safe).clip(lower=0.0, upper=max_scale)
    return scale.fillna(0.0)


def apply_cppi_throttle(
    positions: pd.DataFrame,
    equity_curve: pd.Series,
    *,
    floor_fraction: float,
    multiplier: float = 3.0,
    max_scale: float = 1.0,
    lag: int = 0,
) -> pd.DataFrame:
    """Scale ``positions`` by the CPPI schedule derived from ``equity_curve``.

    *Extended in v13:* new ``lag`` keyword mirrors
    :func:`apply_drawdown_throttle` — ``lag=1`` makes scale at ``t`` use
    equity at ``t-1``. Default ``0`` preserves v12 output.
    """
    if lag < 0:
        raise ValueError("lag must be >= 0")
    scale = cppi_scale(
        equity_curve,
        floor_fraction=floor_fraction,
        multiplier=multiplier,
        max_scale=max_scale,
    )
    if lag > 0:
        scale = scale.shift(lag).fillna(0.0)
    return positions.mul(scale, axis=0)
