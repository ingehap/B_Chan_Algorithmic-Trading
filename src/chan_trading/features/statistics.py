from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import CollinearityWarning
from statsmodels.tsa.stattools import adfuller, coint

from chan_trading.types import HedgeModel, SpreadDiagnostics


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score using rolling mean and standard deviation."""
    if window < 2:
        raise ValueError("window must be >= 2")

    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def ewm_zscore(series: pd.Series, span: int) -> pd.Series:
    """Compute exponentially weighted z-score."""
    if span < 2:
        raise ValueError("span must be >= 2")
    mean = series.ewm(span=span, adjust=False, min_periods=span).mean()
    std = series.ewm(span=span, adjust=False, min_periods=span).std(bias=False)
    return (series - mean) / std.replace(0.0, np.nan)


def transform_pair_prices(
    y: pd.Series,
    x: pd.Series,
    *,
    use_log_prices: bool,
) -> tuple[pd.Series, pd.Series]:
    """Optionally transform aligned pair prices into log-price space."""
    if not y.index.equals(x.index):
        raise ValueError("Series indices must match")

    y_out = y.astype(float)
    x_out = x.astype(float)
    if use_log_prices:
        if (y_out <= 0).any() or (x_out <= 0).any():
            raise ValueError("Log-price pair modeling requires strictly positive prices")
        y_out = pd.Series(np.log(y_out.to_numpy(dtype=float)), index=y_out.index, name=y_out.name, dtype=float)
        x_out = pd.Series(np.log(x_out.to_numpy(dtype=float)), index=x_out.index, name=x_out.name, dtype=float)
    return y_out, x_out


def spread_from_ratio(y: pd.Series, x: pd.Series, hedge_ratio: float | pd.Series) -> pd.Series:
    """Construct spread = y - hedge_ratio * x."""
    if not y.index.equals(x.index):
        raise ValueError("Series indices must match")
    return y - hedge_ratio * x


def spread_from_regression(
    y: pd.Series,
    x: pd.Series,
    alpha: float | pd.Series,
    beta: float | pd.Series,
) -> pd.Series:
    """Construct regression residual spread = y - (alpha + beta * x)."""
    if not y.index.equals(x.index):
        raise ValueError("Series indices must match")
    return y - (alpha + beta * x)


def estimate_static_hedge_model(y: pd.Series, x: pd.Series) -> HedgeModel:
    """Estimate a static linear hedge model via OLS with intercept."""
    if len(y) != len(x):
        raise ValueError("Series lengths must match")

    mask = y.notna() & x.notna()
    yv = y.loc[mask].to_numpy(dtype=float)
    xv = x.loc[mask].to_numpy(dtype=float)
    if len(yv) < 5:
        raise ValueError("Not enough valid samples to estimate hedge model")

    x_design = np.column_stack([np.ones_like(xv), xv])
    coeffs, *_ = np.linalg.lstsq(x_design, yv, rcond=None)
    return HedgeModel(alpha=float(coeffs[0]), beta=float(coeffs[1]))


def estimate_rolling_hedge_model(
    y: pd.Series,
    x: pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Estimate a rolling linear hedge model via rolling covariance and variance.

    The result contains time-aligned ``alpha`` and ``beta`` series that only use
    data available up to each timestamp.
    """
    if not y.index.equals(x.index):
        raise ValueError("Series indices must match")
    if window < 2:
        raise ValueError("window must be >= 2")

    min_periods = window if min_periods is None else min_periods
    if min_periods < 2:
        raise ValueError("min_periods must be >= 2")

    yv = y.astype(float)
    xv = x.astype(float)
    mean_y = yv.rolling(window=window, min_periods=min_periods).mean()
    mean_x = xv.rolling(window=window, min_periods=min_periods).mean()
    cov_xy = xv.rolling(window=window, min_periods=min_periods).cov(yv, ddof=0)
    var_x = xv.rolling(window=window, min_periods=min_periods).var(ddof=0).replace(0.0, np.nan)
    beta = cov_xy / var_x
    alpha = mean_y - beta * mean_x
    return pd.DataFrame({"alpha": alpha, "beta": beta}, index=y.index)


def estimate_kalman_hedge_model(
    y: pd.Series,
    x: pd.Series,
    *,
    delta: float = 1e-4,
    observation_var: float = 1e-3,
) -> pd.DataFrame:
    """Estimate a dynamic hedge model using a simple Kalman filter.

    The state vector is ``[alpha, beta]`` in ``y_t = alpha_t + beta_t x_t + e_t``.
    Both coefficients evolve as random walks, so the estimate adapts to changing
    relationships without peeking ahead.
    """
    if not y.index.equals(x.index):
        raise ValueError("Series indices must match")
    if delta <= 0:
        raise ValueError("delta must be > 0")
    if observation_var <= 0:
        raise ValueError("observation_var must be > 0")

    state = np.zeros(2, dtype=float)
    covariance = np.eye(2, dtype=float)
    transition_cov = delta / (1.0 - delta) * np.eye(2, dtype=float)
    alpha_vals: list[float] = []
    beta_vals: list[float] = []

    for yi, xi in zip(y.astype(float).to_numpy(dtype=float), x.astype(float).to_numpy(dtype=float), strict=False):
        covariance = covariance + transition_cov
        if not np.isfinite(yi) or not np.isfinite(xi):
            alpha_vals.append(float(state[0]))
            beta_vals.append(float(state[1]))
            continue

        design = np.array([1.0, xi], dtype=float)
        pred_var = float(design @ covariance @ design + observation_var)
        if pred_var <= 0 or not np.isfinite(pred_var):
            alpha_vals.append(float(state[0]))
            beta_vals.append(float(state[1]))
            continue

        gain = covariance @ design / pred_var
        forecast = float(design @ state)
        state = state + gain * (yi - forecast)
        covariance = covariance - np.outer(gain, design) @ covariance
        alpha_vals.append(float(state[0]))
        beta_vals.append(float(state[1]))

    return pd.DataFrame({"alpha": alpha_vals, "beta": beta_vals}, index=y.index, dtype=float)


def estimate_static_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """Estimate static hedge ratio via OLS slope with intercept."""
    return estimate_static_hedge_model(y, x).beta


def adf_stationarity_test(series: pd.Series) -> tuple[float, float]:
    """Return ADF statistic and p-value."""
    clean = series.dropna().astype(float)
    if len(clean) < 20:
        raise ValueError("Need at least 20 non-null samples for ADF test")

    statistic, pvalue, *_ = adfuller(clean.to_numpy(), autolag="AIC")
    return float(statistic), float(pvalue)


def engle_granger_cointegration_test_details(y: pd.Series, x: pd.Series) -> tuple[float, float, bool]:
    """Return Engle-Granger test statistic, p-value, and a collinearity flag."""
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 20:
        raise ValueError("Need at least 20 aligned samples for cointegration test")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", CollinearityWarning)
        statistic, pvalue, _ = coint(df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy())
    collinearity_warning = any(issubclass(w.category, CollinearityWarning) for w in caught)
    return float(statistic), float(pvalue), bool(collinearity_warning)


def engle_granger_cointegration_test(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """Return the standard Engle-Granger test statistic and p-value."""
    statistic, pvalue, _ = engle_granger_cointegration_test_details(y, x)
    return statistic, pvalue


def estimate_half_life(series: pd.Series) -> float:
    """Estimate mean-reversion half-life from an AR(1)-style regression.

    Returns ``math.inf`` if the estimate is non-mean-reverting or numerically unstable.
    """
    clean = series.dropna().astype(float)
    if len(clean) < 20:
        raise ValueError("Need at least 20 non-null samples for half-life estimation")

    lagged = clean.shift(1).dropna()
    delta = clean.diff().dropna()
    aligned = pd.concat([lagged, delta], axis=1).dropna()
    if len(aligned) < 10:
        return math.inf

    x = aligned.iloc[:, 0].to_numpy(dtype=float)
    y = aligned.iloc[:, 1].to_numpy(dtype=float)
    x_design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    phi = float(beta[1])
    if not np.isfinite(phi) or phi >= 0.0:
        return math.inf

    half_life = -np.log(2.0) / phi
    if not np.isfinite(half_life) or half_life <= 0.0:
        return math.inf
    return float(half_life)


def estimate_hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """Estimate the Hurst exponent using the log-log R/S slope approximation."""
    clean = series.dropna().astype(float)
    if len(clean) < max(32, max_lag + 2):
        raise ValueError("Need more data to estimate Hurst exponent")

    lags = np.arange(2, max_lag + 1)
    tau: list[float] = []
    for lag in lags:
        diffs = clean.to_numpy()[lag:] - clean.to_numpy()[:-lag]
        scale = np.std(diffs, ddof=0)
        tau.append(float(scale))
    tau_arr = np.asarray(tau, dtype=float)
    valid = np.isfinite(tau_arr) & (tau_arr > 0)
    if valid.sum() < 3:
        return math.nan
    slope, _ = np.polyfit(np.log(lags[valid]), np.log(tau_arr[valid]), 1)
    return float(slope)


def estimate_variance_ratio(series: pd.Series, lag: int = 2) -> float:
    """Estimate a simple Lo-MacKinlay-style variance ratio."""
    clean = series.dropna().astype(float)
    if len(clean) < max(20, lag + 2):
        raise ValueError("Need more data to estimate variance ratio")
    if lag < 2:
        raise ValueError("lag must be >= 2")

    increments = clean.diff().dropna().to_numpy(dtype=float)
    single_var = float(np.var(increments, ddof=0))
    if single_var == 0 or not np.isfinite(single_var):
        return math.nan

    k_step = clean.diff(lag).dropna().to_numpy(dtype=float)
    multi_var = float(np.var(k_step, ddof=0))
    return float(multi_var / (lag * single_var))


def estimate_rs_hurst_exponent(
    series: pd.Series,
    *,
    min_block: int = 8,
    max_block: int | None = None,
    n_blocks: int = 6,
) -> float:
    """Rescaled-range (R/S) Hurst exponent estimator (Mandelbrot/Hurst).

    *New in v13.* The default :func:`estimate_hurst_exponent` uses the
    log-log slope of the standard deviation of lagged differences. That
    estimator is consistent for fractional Brownian motion but can be
    noisy on finite financial spreads, where the classical rescaled-range
    estimator is often quoted side-by-side (Chan, Ch. 2, discusses both).

    The rescaled-range statistic for a window of length ``n`` is the
    range of the cumulative deviations from the mean, divided by the
    standard deviation of the original increments. Across many window
    sizes, ``log(R/S)`` grows approximately linearly in ``log(n)`` with
    slope ``H``. Values of ``H`` below ``0.5`` indicate mean reversion;
    ``H = 0.5`` is a random walk; ``H`` above ``0.5`` indicates
    persistence.

    Parameters
    ----------
    series
        Input series (typically the spread or log-price).
    min_block
        Smallest block size to include in the regression. Defaults to 8
        which is small enough to give many blocks on short samples but
        large enough that the R/S ratio is not dominated by noise.
    max_block
        Largest block size. Defaults to ``len(series) // 4`` so every
        chosen block length has at least four non-overlapping
        realisations to average over.
    n_blocks
        Number of log-spaced block sizes between ``min_block`` and
        ``max_block``. More points give a lower-variance slope estimate
        but also pull in larger, noisier blocks.

    Returns
    -------
    H
        R/S Hurst exponent estimate. Returns ``nan`` if there is too
        little data or all R/S values are zero.

    References
    ----------
    Hurst, H. E. (1951). "Long-term storage capacity of reservoirs."
    *Transactions of the American Society of Civil Engineers*, 116(1),
    770–799. Mandelbrot, B. B., & Wallis, J. R. (1969). "Robustness of
    the rescaled range R/S in the measurement of noncyclic long run
    statistical dependence." *Water Resources Research*, 5(5), 967–988.
    """
    clean = series.dropna().astype(float).to_numpy(dtype=float)
    n = clean.size
    if n < 32:
        raise ValueError("Need at least 32 non-null samples for R/S Hurst")
    if min_block < 4:
        raise ValueError("min_block must be >= 4")
    if n_blocks < 3:
        raise ValueError("n_blocks must be >= 3")
    hi = max_block if max_block is not None else n // 4
    if hi <= min_block:
        raise ValueError("max_block must exceed min_block")

    # Log-spaced unique block sizes
    sizes = np.unique(
        np.round(np.exp(np.linspace(np.log(min_block), np.log(hi), n_blocks))).astype(int)
    )
    sizes = sizes[sizes >= min_block]
    if sizes.size < 3:
        return math.nan

    log_sizes: list[float] = []
    log_rs: list[float] = []
    for block in sizes:
        n_segments = n // int(block)
        if n_segments < 2:
            continue
        trimmed = clean[: n_segments * int(block)].reshape(n_segments, int(block))
        # Per-segment rescaled range of cumulative mean-adjusted path.
        deviations = trimmed - trimmed.mean(axis=1, keepdims=True)
        cum = np.cumsum(deviations, axis=1)
        ranges = cum.max(axis=1) - cum.min(axis=1)
        stds = trimmed.std(axis=1, ddof=0)
        valid = stds > 0
        if valid.sum() == 0:
            continue
        rs_vals = ranges[valid] / stds[valid]
        mean_rs = float(np.mean(rs_vals))
        if mean_rs <= 0:
            continue
        log_sizes.append(float(np.log(block)))
        log_rs.append(float(np.log(mean_rs)))

    if len(log_sizes) < 3:
        return math.nan
    slope, _ = np.polyfit(np.asarray(log_sizes), np.asarray(log_rs), 1)
    return float(slope)


def suggest_zscore_lookback(
    half_life: float,
    *,
    multiplier: float = 2.0,
    floor: int = 5,
    cap: int = 252,
) -> int:
    """Pick a z-score lookback from an estimated mean-reversion half-life.

    *New in v13.* Chan (Ch. 2–3) argues that the right lookback for a
    mean-reversion signal is coupled to the estimated half-life of the
    spread: too short and the window estimates noise, too long and the
    mean drifts before you can trade it. A common heuristic is
    ``lookback = 2 × half_life``. This helper returns exactly that,
    clamped into a usable range.

    Parameters
    ----------
    half_life
        Estimated half-life in bars (e.g. from
        :func:`estimate_half_life`). ``inf`` / ``nan`` / non-positive
        values fall back to ``cap``.
    multiplier
        Lookback multiplier applied to the half-life. Default ``2.0``
        (Chan's rule of thumb); typical range ``[1.5, 3.0]``.
    floor
        Minimum lookback in bars. Default ``5``.
    cap
        Maximum lookback in bars. Default ``252`` (≈ 1 year of daily
        bars) so the suggestion stays sensible when the half-life
        estimate blows up.
    """
    if multiplier <= 0:
        raise ValueError("multiplier must be > 0")
    if floor < 2:
        raise ValueError("floor must be >= 2")
    if cap < floor:
        raise ValueError("cap must be >= floor")
    if not np.isfinite(half_life) or half_life <= 0:
        return int(cap)
    raw = int(round(multiplier * float(half_life)))
    return int(min(max(raw, floor), cap))


def estimate_spread_diagnostics_object(
    y: pd.Series,
    x: pd.Series,
    *,
    use_log_prices: bool = True,
    signal_mode: str = "residual",
) -> SpreadDiagnostics:
    """Estimate a rich pair-diagnostics object used by the workflow."""
    y_model, x_model = transform_pair_prices(y, x, use_log_prices=use_log_prices)
    hedge_model = estimate_static_hedge_model(y_model, x_model)
    if signal_mode == "ratio":
        spread = spread_from_ratio(y_model, x_model, hedge_model.beta)
    elif signal_mode == "residual":
        spread = spread_from_regression(y_model, x_model, hedge_model.alpha, hedge_model.beta)
    else:
        raise ValueError("signal_mode must be 'residual' or 'ratio'")

    adf_stat, adf_pvalue = adf_stationarity_test(spread)
    eg_stat, eg_pvalue, collinearity_warning = engle_granger_cointegration_test_details(y_model, x_model)
    half_life = estimate_half_life(spread)
    hurst = estimate_hurst_exponent(spread)
    variance_ratio = estimate_variance_ratio(spread)
    return SpreadDiagnostics(
        alpha=hedge_model.alpha,
        beta=hedge_model.beta,
        adf_statistic=adf_stat,
        adf_pvalue=adf_pvalue,
        eg_statistic=eg_stat,
        eg_pvalue=eg_pvalue,
        half_life=half_life,
        hurst_exponent=hurst,
        variance_ratio=variance_ratio,
        spread_mean=float(spread.mean()),
        spread_std=float(spread.std(ddof=0)),
        collinearity_warning=collinearity_warning,
    )


def estimate_spread_diagnostics(
    y: pd.Series,
    x: pd.Series,
    *,
    use_log_prices: bool = True,
    signal_mode: str = "residual",
) -> tuple[float, float, float, float, float]:
    """Backwards-compatible tuple view of pair diagnostics used by the workflow."""
    diag = estimate_spread_diagnostics_object(
        y,
        x,
        use_log_prices=use_log_prices,
        signal_mode=signal_mode,
    )
    return diag.beta, diag.adf_statistic, diag.adf_pvalue, diag.eg_statistic, diag.eg_pvalue
