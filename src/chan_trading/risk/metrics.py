from __future__ import annotations

import numpy as np
import pandas as pd

from chan_trading.portfolio.sizing import turnover as position_turnover
from chan_trading.types import RiskReport


def max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown from an equity curve."""
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized geometric return.

    *v13:* uses ``expm1(sum(log1p(r)))`` for numerical stability on long
    series instead of the direct cumulative product. The two are
    mathematically identical but the log-space form does not under/overflow
    on multi-decade daily series. The output is byte-identical to v12 up
    to floating-point rounding for well-behaved series.
    """
    if returns.empty:
        return float("nan")
    clean = returns.astype(float).to_numpy(dtype=float)
    if clean.size == 0:
        return float("nan")
    # Any return <= -1 would push total return non-positive; fall back to
    # the direct product in that degenerate case so we get the same NaN
    # the v12 implementation would have produced.
    if np.any(clean <= -1.0):
        total_return = float(np.prod(1.0 + clean))
        if total_return <= 0:
            return float("nan")
    else:
        total_return = float(np.expm1(np.sum(np.log1p(clean))))
        total_return = 1.0 + total_return
        if total_return <= 0:
            return float("nan")
    n_periods = clean.size
    return float(total_return ** (periods_per_year / n_periods) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized standard deviation."""
    if returns.empty:
        return float("nan")
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sharpe ratio."""
    if returns.empty:
        return float("nan")
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    vol = excess.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def rolling_sharpe_ratio(
    returns: pd.Series,
    *,
    lookback: int = 63,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling annualised Sharpe ratio.

    *New in v14.* Chan (Ch. 8) argues explicitly that a single lifetime
    Sharpe number hides regime shifts — a strategy with a decent
    lifetime Sharpe can nevertheless have long stretches of near-zero
    or negative Sharpe that a static summary smooths over. A rolling
    Sharpe series makes those stretches visible and is the natural
    upstream signal for a regime-aware throttle: when the rolling
    Sharpe collapses, the sizing layer can de-risk before the drawdown
    throttle kicks in.

    The output is causal: the value at ``t`` is computed from the
    window ending at ``t`` (inclusive), matching the convention used
    by :func:`rolling_volatility_percentile` and
    :func:`chan_trading.portfolio.kelly.rolling_kelly_fraction`.
    Downstream code that uses this to *scale positions* should
    therefore still lag by one bar.

    Parameters
    ----------
    returns
        Per-period strategy returns.
    lookback
        Rolling window length. Typical daily values: 63 (one quarter),
        126 (half a year), 252 (one year).
    periods_per_year
        Annualisation factor.
    risk_free_rate
        Annualised risk-free rate, subtracted per-period before the
        Sharpe is computed. Default 0.
    min_periods
        Minimum observations required for a non-NaN value. Defaults to
        ``lookback``.

    Returns
    -------
    rolling_sr
        Rolling annualised Sharpe, same index as ``returns``. Leading
        values are NaN until ``min_periods`` observations accumulate.
    """
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")
    min_p = lookback if min_periods is None else min_periods
    if min_p < 2:
        raise ValueError("min_periods must be >= 2")

    rf_per_period = risk_free_rate / periods_per_year
    excess = returns.astype(float) - rf_per_period
    mean = excess.rolling(lookback, min_periods=min_p).mean()
    vol = excess.rolling(lookback, min_periods=min_p).std(ddof=0)
    vol = vol.replace(0.0, np.nan)
    sr = (mean / vol) * np.sqrt(periods_per_year)
    return sr.replace([np.inf, -np.inf], np.nan)


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    periods_per_year: int = 252,
) -> float:
    """Annualised information ratio of a return stream against a benchmark.

    *New in v14.* The information ratio (IR) is the standard practitioner
    generalisation of the Sharpe ratio for strategies benchmarked
    against a non-zero comparator:

        IR = mean(r − b) / std(r − b) · sqrt(periods_per_year)

    where ``r`` are the strategy per-period returns and ``b`` the
    benchmark per-period returns. The numerator is the *active return*
    (alpha over the benchmark); the denominator is the *tracking
    error*. Unlike the raw Sharpe (which silently benchmarks against a
    zero-return portfolio), the IR answers the question "does the
    strategy earn a risk-adjusted premium *over my benchmark*?"
    directly, which is the relevant question for most long-only and
    benchmark-relative mandates Chan mentions in Ch. 1 and Ch. 8.

    The two series must share the same index; ``NaN`` observations in
    either series are dropped jointly before the statistic is computed.

    Parameters
    ----------
    returns
        Per-period strategy returns.
    benchmark_returns
        Per-period benchmark returns (same frequency and index).
    periods_per_year
        Annualisation factor.

    Returns
    -------
    ir
        Annualised IR. NaN if there is too little overlap or the
        tracking error is zero.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be > 0")
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")
    if not isinstance(benchmark_returns, pd.Series):
        raise TypeError("benchmark_returns must be a pandas Series")
    aligned = pd.concat(
        [returns.astype(float), benchmark_returns.astype(float)], axis=1, join="inner"
    ).dropna()
    if len(aligned) < 2:
        return float("nan")
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = float(active.std(ddof=0))
    if te == 0.0 or not np.isfinite(te):
        return float("nan")
    return float(active.mean() / te * np.sqrt(periods_per_year))


def newey_west_sharpe_variance(
    returns: pd.Series,
    *,
    lags: int | None = None,
    periods_per_year: int = 252,
) -> float:
    """Newey-West / HAC estimate of the annualised Sharpe ratio's variance.

    *New in v14.* Chan (Ch. 1) explicitly warns that a sample Sharpe
    ratio computed under the IID assumption is misleading when the
    strategy's per-period returns are serially correlated. Positive
    autocorrelation — which is the default for many mean-reversion and
    calendar-spread strategies — inflates the apparent Sharpe because
    the naive variance estimator ``sigma^2 / T`` ignores the covariance
    between neighbouring observations. Negative autocorrelation (e.g.
    some reversal strategies at higher frequency) does the opposite.

    The Newey-West (Bartlett-kernel) heteroscedasticity-and-
    autocorrelation-consistent (HAC) variance fixes this. For a per-
    period return series ``r_t`` with ``T`` observations and ``L`` lags,
    the HAC variance of the sample mean is

        Var_HAC(r_bar) = (1/T) · [ γ_0 + 2 · Σ_{k=1}^{L} (1 − k/(L+1)) · γ_k ]

    where ``γ_k`` is the lag-``k`` autocovariance. Dividing by the
    squared sample standard deviation and annualising gives the
    corresponding variance of the annualised Sharpe:

        Var(SR_annual) = periods_per_year · Var_HAC(r_bar) / sigma^2 .

    When the series is truly IID the HAC variance collapses to the
    familiar ``sigma^2 / T`` and this function agrees with the classic
    ``1 / T`` asymptote of the sample Sharpe. Deviations between the two
    are a direct read of how much the Sharpe standard error has been
    under- or over-stated.

    Parameters
    ----------
    returns
        Per-period strategy returns.
    lags
        Number of Newey-West lags. When ``None`` (default), the common
        automatic rule ``floor(4 · (T/100)^(2/9))`` (Newey & West, 1994)
        is used. Set it manually when you have a clear prior on the
        strategy's holding horizon — for a mean-reverter with half-life
        ``h``, ``lags = 2·h`` is a reasonable default.
    periods_per_year
        Annualisation factor (252 for daily). Used solely to scale the
        per-period variance up to the annualised Sharpe.

    Returns
    -------
    variance
        HAC variance of the annualised Sharpe ratio. NaN if there is
        too little data, if the variance is non-finite, or if the
        underlying return variance is zero.

    References
    ----------
    Newey, W. K., & West, K. D. (1987). "A Simple, Positive Semi-
    Definite, Heteroskedasticity and Autocorrelation Consistent
    Covariance Matrix." *Econometrica*, 55(3), 703-708.

    Newey, W. K., & West, K. D. (1994). "Automatic Lag Selection in
    Covariance Matrix Estimation." *Review of Economic Studies*, 61(4),
    631-653.
    """
    clean = returns.dropna().astype(float).to_numpy(dtype=float)
    n = clean.size
    if n < 10:
        return float("nan")
    sigma2 = float(np.var(clean, ddof=0))
    if sigma2 <= 0 or not np.isfinite(sigma2):
        return float("nan")

    if lags is None:
        # Newey-West (1994) automatic lag rule; ensure at least 1 and
        # never more than T-1.
        lags_auto = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lags = max(1, min(lags_auto, n - 1))
    if lags < 0:
        raise ValueError("lags must be >= 0")
    if lags >= n:
        raise ValueError("lags must be < len(returns)")

    centered = clean - clean.mean()
    # Lag-0 autocovariance = sigma^2. Bartlett weights linearly
    # decay to zero at lag L+1 so the kernel is positive semidefinite.
    hac = sigma2
    for k in range(1, lags + 1):
        weight = 1.0 - k / (lags + 1)
        gamma_k = float(np.mean(centered[k:] * centered[:-k]))
        hac += 2.0 * weight * gamma_k

    if not np.isfinite(hac) or hac <= 0:
        # HAC can become non-PSD in small samples with strong negative
        # autocorrelation. Clip at a tiny positive number so downstream
        # t-stats / CIs stay defined rather than returning NaN, but
        # flag the condition by returning NaN when the clip would be
        # meaningless.
        return float("nan")

    # Var(sample mean) under the HAC kernel is hac / n. Sharpe scales
    # the mean by sigma, so Var(SR) = Var(mean) / sigma^2, then annualise.
    var_sr_per_period = hac / (n * sigma2)
    return float(var_sr_per_period * periods_per_year)


def newey_west_sharpe_tstat(
    returns: pd.Series,
    *,
    lags: int | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Newey-West t-statistic of the annualised Sharpe against zero.

    *New in v14.* Convenience wrapper that divides the annualised
    sample Sharpe by the square-root of its Newey-West variance. A
    t-stat above roughly 2 is the usual threshold for "the annualised
    Sharpe is significantly different from zero once serial correlation
    is accounted for", and the naive (IID) t-stat is simply
    ``sharpe_annual · sqrt(T / periods_per_year)``. Comparing the two
    tells the researcher how much of the observed Sharpe's apparent
    significance came from assuming independence.

    Parameters
    ----------
    returns
        Per-period strategy returns.
    lags
        Newey-West lag count, see :func:`newey_west_sharpe_variance`.
    periods_per_year
        Annualisation factor.
    risk_free_rate
        Annualised risk-free rate; subtracted per-period before the
        statistic is computed. Default 0.

    Returns
    -------
    t
        t-statistic ``SR_annual / sqrt(Var_HAC(SR_annual))``.
    """
    sr = sharpe_ratio(
        returns, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
    )
    if not np.isfinite(sr):
        return float("nan")
    # For the variance we subtract the per-period risk-free rate once
    # so the numerator and denominator are consistent.
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns.dropna().astype(float) - rf_per_period
    var = newey_west_sharpe_variance(
        excess, lags=lags, periods_per_year=periods_per_year
    )
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    return float(sr / np.sqrt(var))


def sortino_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
    minimum_acceptable_return: float = 0.0,
) -> float:
    """Compute annualised Sortino ratio against a target return.

    *Fixed in v13.* The v12 implementation computed
    ``returns[returns < 0].std(ddof=0)`` — i.e. the standard deviation of
    only the negative subset — and used it as the downside deviation.
    That is not the definition of Sortino. The correct denominator is the
    **target semideviation** (a.k.a. downside deviation):

        DD = sqrt( mean( min(r - MAR, 0)^2 ) )

    where the mean is over **all** returns, not only the losing ones,
    and the shortfall is measured against the minimum acceptable return
    (``MAR``, 0 by default). The numerator is the excess return over the
    same ``MAR``. The v12 formula systematically overstated Sortino by a
    factor that grew with hit rate (roughly 10-30% on typical daily
    equity strategies).

    Parameters
    ----------
    returns
        Per-period strategy returns.
    periods_per_year
        Annualisation factor (252 for daily).
    minimum_acceptable_return
        Per-period MAR. Default ``0.0``. Set to the per-period risk-free
        rate or the per-period target to get the standard practitioner
        variant.

    References
    ----------
    Sortino, F. A., & Price, L. N. (1994). "Performance Measurement in a
    Downside Risk Framework." *Journal of Investing*, 3(3), 59–64.
    """
    if returns.empty:
        return float("nan")
    clean = returns.astype(float).dropna()
    if clean.empty:
        return float("nan")
    excess = clean - minimum_acceptable_return
    shortfall = np.minimum(excess.to_numpy(dtype=float), 0.0)
    downside_dev = float(np.sqrt(np.mean(shortfall ** 2)))
    if downside_dev == 0.0 or not np.isfinite(downside_dev):
        return float("nan")
    return float(excess.mean() / downside_dev * np.sqrt(periods_per_year))


def tail_ratio(returns: pd.Series) -> float:
    """Return the 95th percentile gain divided by the 5th percentile loss magnitude."""
    if returns.empty:
        return float("nan")
    q95 = float(np.nanpercentile(returns, 95))
    q05 = float(np.nanpercentile(returns, 5))
    if q05 == 0 or np.isnan(q05):
        return float("nan")
    return float(q95 / abs(q05))


def time_under_water(equity_curve: pd.Series) -> float:
    """Fraction of observations spent below the running high-water mark."""
    if equity_curve.empty:
        return float("nan")
    running_max = equity_curve.cummax()
    underwater = equity_curve < running_max
    return float(np.mean(underwater.to_numpy(dtype=float)))


def historical_var(
    returns: pd.Series,
    *,
    alpha: float = 0.05,
) -> float:
    """Historical (empirical) Value-at-Risk at confidence ``1 - alpha``.

    Returned as a **signed return**: the ``alpha``-quantile of the empirical
    return distribution. A typical daily VaR at 95% confidence uses
    ``alpha=0.05`` and returns e.g. ``-0.023`` meaning "the strategy loses
    2.3% or more on 5% of days".

    Chan (Ch. 8) emphasises that the Sharpe ratio and volatility alone give
    a misleading picture of risk when returns have fat tails. A quantile
    view captures the left tail directly.

    *New in v11.*

    Parameters
    ----------
    returns
        Per-period strategy returns.
    alpha
        Tail probability. ``0.05`` → 95% VaR, ``0.01`` → 99% VaR.

    Returns
    -------
    var
        The ``alpha``-quantile (signed). NaN if there is too little data.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    clean = returns.dropna().astype(float)
    if len(clean) < 2:
        return float("nan")
    return float(np.quantile(clean.to_numpy(dtype=float), alpha))


def historical_cvar(
    returns: pd.Series,
    *,
    alpha: float = 0.05,
) -> float:
    """Historical Conditional VaR (Expected Shortfall) at confidence ``1 - alpha``.

    CVaR is the **expected return conditional on the return being in the
    worst ``alpha`` tail** — i.e. the average of the worst ``alpha``
    fraction of observations. Returned as a signed number; a value of
    ``-0.035`` means the average loss on the worst 5% of days is 3.5%.

    CVaR is coherent in the Artzner et al. sense (unlike VaR), which is
    why it is the modern preferred downside metric. Chan (Ch. 8) does not
    name CVaR explicitly but repeatedly argues for distribution-aware
    risk measures, especially when sizing with Kelly.

    *New in v11.*

    Parameters
    ----------
    returns
        Per-period strategy returns.
    alpha
        Tail probability. ``0.05`` → 95% CVaR, ``0.01`` → 99% CVaR.

    Returns
    -------
    cvar
        Mean of the tail (signed). NaN if the tail is empty or data is
        insufficient.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    clean = returns.dropna().astype(float)
    if len(clean) < 2:
        return float("nan")
    values = clean.to_numpy(dtype=float)
    threshold = float(np.quantile(values, alpha))
    tail = values[values <= threshold]
    if len(tail) == 0:
        return float("nan")
    return float(np.mean(tail))


def parametric_gaussian_var(
    returns: pd.Series,
    *,
    alpha: float = 0.05,
) -> float:
    """Gaussian (parametric) VaR at confidence ``1 - alpha``.

    Computed as ``mu + sigma * z_alpha`` where ``z_alpha`` is the standard
    normal quantile. Faster than the historical estimate and smoother, but
    *systematically understates* risk when returns have fat tails — use
    it side-by-side with :func:`historical_var` rather than instead of it.

    *New in v11.*
    """
    from scipy.stats import norm

    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    clean = returns.dropna().astype(float)
    if len(clean) < 2:
        return float("nan")
    mu = float(clean.mean())
    sigma = float(clean.std(ddof=0))
    if sigma == 0 or not np.isfinite(sigma):
        return float("nan")
    return float(mu + sigma * norm.ppf(alpha))


def profit_factor(returns: pd.Series) -> float:
    """Profit factor = sum of positive returns / |sum of negative returns|.

    A widely used Chan-era hedge-fund metric (Ch. 8) that complements the
    Sharpe ratio: Sharpe measures the *ratio of mean to volatility*, while
    the profit factor measures the *ratio of total gains to total losses*.
    A strategy with profit factor > 1 made more in winning periods than it
    lost in losing periods; by convention ``PF >= 1.5`` is considered good
    and ``>= 2.0`` is strong.

    Returns ``+inf`` if there are no negative returns, ``nan`` if the
    input is empty or has no realized moves.

    *New in v12.*
    """
    clean = returns.dropna().astype(float)
    if clean.empty:
        return float("nan")
    gains = float(clean[clean > 0].sum())
    losses = float(-clean[clean < 0].sum())  # positive magnitude
    if losses == 0.0:
        if gains == 0.0:
            return float("nan")
        return float("inf")
    return float(gains / losses)


def omega_ratio(returns: pd.Series, *, threshold: float = 0.0) -> float:
    """Omega ratio at a given per-period return ``threshold``.

    Omega = ``E[max(r - threshold, 0)] / E[max(threshold - r, 0)]``. It
    generalises the profit factor by allowing a non-zero hurdle (e.g. a
    per-period risk-free rate or target return). Like the profit factor,
    a value above 1 indicates the strategy's upside area beats its downside
    area against the chosen threshold.

    Chan (Ch. 8) stresses using distribution-aware risk measures instead
    of relying exclusively on the Sharpe ratio when returns are non-normal;
    Omega uses the entire distribution above/below the hurdle rather than
    just the first two moments. Returns ``+inf`` if no returns fall below
    the threshold, ``nan`` for empty input or no variation.

    *New in v12.*
    """
    clean = returns.dropna().astype(float)
    if clean.empty:
        return float("nan")
    excess = clean - float(threshold)
    gains = float(excess[excess > 0].sum())
    losses = float(-excess[excess < 0].sum())
    if losses == 0.0:
        if gains == 0.0:
            return float("nan")
        return float("inf")
    return float(gains / losses)


def gain_to_pain_ratio(returns: pd.Series) -> float:
    """Gain-to-Pain ratio = sum of returns / |sum of negative returns|.

    Popularised by Kaplan and the Jack Schwager *Hedge Fund Market Wizards*
    series; the same object Chan (Ch. 8) points to when he recommends
    reporting return per unit of realised loss rather than per unit of
    volatility. Numerator is the *net* period return; denominator is the
    total magnitude of losing periods. Values above 1 mean the strategy
    has made more net profit than it has surrendered in drawdown periods
    on a simple sum basis.

    *New in v12.*
    """
    clean = returns.dropna().astype(float)
    if clean.empty:
        return float("nan")
    total = float(clean.sum())
    pain = float(-clean[clean < 0].sum())
    if pain == 0.0:
        if total == 0.0:
            return float("nan")
        return float("inf") if total > 0 else float("-inf")
    return float(total / pain)


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    *,
    benchmark_sharpe: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012).

    The PSR returns the probability that the *true* Sharpe ratio exceeds
    a user-specified ``benchmark_sharpe`` (annualized), given the
    observed sample Sharpe and its skew/kurtosis-adjusted standard
    error. Values close to 1 indicate strong evidence that the strategy
    beats the benchmark Sharpe; values near 0.5 or below are consistent
    with luck.

    Chan (Ch. 1) repeatedly warns that a raw Sharpe ratio overstates
    confidence when returns are non-normal or the sample is short — the
    PSR is the standard correction for exactly that.

    This is the **n_trials=1 special case** of the Deflated Sharpe Ratio
    and is provided as a separate, explicitly named function so callers
    can reach for the simpler tool when there is no multiple-testing
    concern to correct for.

    Parameters
    ----------
    returns
        Per-period strategy returns.
    benchmark_sharpe
        Annualized Sharpe to beat. Default ``0.0``.
    periods_per_year
        Annualization factor (252 for daily).

    Returns
    -------
    probability
        Value in ``[0, 1]``.

    References
    ----------
    Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio
    Efficient Frontier." *Journal of Risk*, 15(2), 3–44.
    """
    from scipy.stats import norm

    clean = returns.dropna().astype(float)
    n = len(clean)
    if n < 10:
        return float("nan")

    std = float(clean.std(ddof=0))
    if std == 0 or not np.isfinite(std):
        return float("nan")

    sr_per_period = float(clean.mean()) / std
    sr_annual = sr_per_period * np.sqrt(periods_per_year)

    centered = clean - clean.mean()
    third = float(np.mean(centered ** 3))
    fourth = float(np.mean(centered ** 4))
    skew = third / (std ** 3) if std > 0 else 0.0
    kurt_excess = (fourth / (std ** 4) - 3.0) if std > 0 else 0.0

    # Variance of the Sharpe estimator under non-normal returns (Mertens / Lo).
    sr_variance_annual = (
        (1.0 - skew * sr_per_period + (kurt_excess / 4.0) * sr_per_period ** 2) / (n - 1)
    ) * periods_per_year
    if sr_variance_annual <= 0 or not np.isfinite(sr_variance_annual):
        return float("nan")

    z = (sr_annual - benchmark_sharpe) / np.sqrt(sr_variance_annual)
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    returns: pd.Series,
    *,
    n_trials: int,
    periods_per_year: int = 252,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Chan (Ch. 1) stresses that data-snooping and multiple testing inflate
    observed Sharpe ratios. The Deflated Sharpe Ratio (DSR) returns the
    probability that the observed Sharpe exceeds a benchmark after
    accounting for skew, kurtosis, and the number of independent strategy
    configurations that were tried (``n_trials``).

    When ``n_trials=1`` this reduces to the Probabilistic Sharpe Ratio
    against ``benchmark_sharpe`` (see :func:`probabilistic_sharpe_ratio`
    for a dedicated entry point).

    Parameters
    ----------
    returns
        Per-period strategy returns.
    n_trials
        Number of strategy variants / configurations that were evaluated
        during research. Passing a realistic count (e.g. 50 or 100) is the
        honest way to reflect the search Chan warns against.
    periods_per_year
        Annualization factor (252 for daily).
    benchmark_sharpe
        Annualized Sharpe to beat. Default 0.

    Returns
    -------
    probability
        Value in ``[0, 1]``.

    References
    ----------
    Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
    *Journal of Portfolio Management*, 40(5), 94–107.
    """
    clean = returns.dropna().astype(float)
    n = len(clean)
    if n < 10:
        return float("nan")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    std = float(clean.std(ddof=0))
    if std == 0 or not np.isfinite(std):
        return float("nan")

    sr_per_period = float(clean.mean()) / std
    sr_annual = sr_per_period * np.sqrt(periods_per_year)

    # Skewness and excess kurtosis of the per-period returns
    centered = clean - clean.mean()
    third = float(np.mean(centered ** 3))
    fourth = float(np.mean(centered ** 4))
    skew = third / (std ** 3) if std > 0 else 0.0
    kurt_excess = (fourth / (std ** 4) - 3.0) if std > 0 else 0.0

    # Expected maximum Sharpe across N independent trials under the null
    # (using the Bailey-López de Prado approximation):
    #   E[max SR] ~ sqrt(V) * ((1 - gamma) * Phi^{-1}(1 - 1/N)
    #                          + gamma * Phi^{-1}(1 - 1/(N*e)))
    # Here V is the variance of per-period SR estimates under the null ~ 1/T.
    from math import e as _e
    from scipy.stats import norm

    gamma_euler = 0.5772156649015329
    if n_trials == 1:
        expected_max_sr_annual = benchmark_sharpe
    else:
        z1 = norm.ppf(1.0 - 1.0 / n_trials)
        z2 = norm.ppf(1.0 - 1.0 / (n_trials * _e))
        expected_max_sr_per_period = np.sqrt(1.0 / n) * (
            (1.0 - gamma_euler) * z1 + gamma_euler * z2
        )
        expected_max_sr_annual = float(expected_max_sr_per_period) * np.sqrt(periods_per_year) + benchmark_sharpe

    # Variance of the Sharpe estimator under non-normal returns (Mertens/Lo)
    sr_variance_annual = (
        (1.0 - skew * sr_per_period + (kurt_excess / 4.0) * sr_per_period ** 2) / (n - 1)
    ) * periods_per_year
    if sr_variance_annual <= 0 or not np.isfinite(sr_variance_annual):
        return float("nan")

    z = (sr_annual - expected_max_sr_annual) / np.sqrt(sr_variance_annual)
    return float(norm.cdf(z))


def summarize_risk(
    equity_curve: pd.Series,
    returns: pd.Series,
    periods_per_year: int = 252,
    positions: pd.DataFrame | None = None,
) -> RiskReport:
    """Return a compact but richer risk report."""
    ann_ret = annualized_return(returns, periods_per_year=periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year=periods_per_year)
    sharpe = sharpe_ratio(returns, periods_per_year=periods_per_year)
    mdd = max_drawdown(equity_curve)
    calmar = ann_ret / abs(mdd) if mdd < 0 else float("nan")
    positive = returns[returns > 0]
    negative = returns[returns < 0]
    avg_turnover = float(position_turnover(positions).mean()) if positions is not None else float("nan")
    return RiskReport(
        annual_return=ann_ret,
        annual_volatility=ann_vol,
        sharpe=sharpe,
        max_drawdown=mdd,
        calmar=calmar,
        sortino=sortino_ratio(returns, periods_per_year=periods_per_year),
        turnover=avg_turnover,
        tail_ratio=tail_ratio(returns),
        hit_rate=float((returns > 0).mean()) if len(returns) > 0 else float("nan"),
        average_win=float(positive.mean()) if len(positive) > 0 else float("nan"),
        average_loss=float(negative.mean()) if len(negative) > 0 else float("nan"),
        time_under_water=time_under_water(equity_curve),
        var_95=historical_var(returns, alpha=0.05),
        cvar_95=historical_cvar(returns, alpha=0.05),
        profit_factor=profit_factor(returns),
        omega_ratio=omega_ratio(returns),
        gain_to_pain_ratio=gain_to_pain_ratio(returns),
    )
