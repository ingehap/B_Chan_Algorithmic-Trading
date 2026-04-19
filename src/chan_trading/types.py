from __future__ import annotations

from dataclasses import dataclass, field
import math
import pandas as pd


@dataclass(slots=True)
class HedgeModel:
    """Static linear hedge model ``y ~= alpha + beta * x``."""

    alpha: float
    beta: float


@dataclass(slots=True)
class SpreadDiagnostics:
    """Summary diagnostics for a candidate mean-reverting spread."""

    alpha: float
    beta: float
    adf_statistic: float
    adf_pvalue: float
    eg_statistic: float
    eg_pvalue: float
    half_life: float
    hurst_exponent: float | None = None
    variance_ratio: float | None = None
    spread_mean: float | None = None
    spread_std: float | None = None
    collinearity_warning: bool = False


@dataclass(slots=True)
class BacktestResult:
    """Container for vectorized backtest outputs."""

    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame


@dataclass(slots=True)
class EventBacktestResult:
    """Container for event-driven backtest outputs."""

    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    order_log: pd.DataFrame
    fill_log: pd.DataFrame


@dataclass(slots=True)
class RiskReport:
    """Summary risk metrics.

    *Extended in v11:* added ``var_95`` and ``cvar_95`` (historical VaR and
    CVaR at 95% confidence). Both are signed tail returns — see
    :func:`chan_trading.risk.metrics.historical_var` and
    :func:`chan_trading.risk.metrics.historical_cvar`.

    *Extended in v12:* added ``profit_factor``, ``omega_ratio``, and
    ``gain_to_pain_ratio`` — the three gain-vs-loss ratios Chan (Ch. 8)
    treats as complementary to Sharpe. See
    :func:`chan_trading.risk.metrics.profit_factor`,
    :func:`chan_trading.risk.metrics.omega_ratio`, and
    :func:`chan_trading.risk.metrics.gain_to_pain_ratio`.
    """

    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    calmar: float
    sortino: float = math.nan
    turnover: float = math.nan
    tail_ratio: float = math.nan
    hit_rate: float = math.nan
    average_win: float = math.nan
    average_loss: float = math.nan
    time_under_water: float = math.nan
    var_95: float = math.nan
    cvar_95: float = math.nan
    profit_factor: float = math.nan
    omega_ratio: float = math.nan
    gain_to_pain_ratio: float = math.nan


@dataclass(slots=True)
class CointegrationReport:
    """Summary of spread diagnostics for pair trading."""

    adf_statistic: float
    adf_pvalue: float
    eg_statistic: float
    eg_pvalue: float
    hedge_ratio: float
    intercept: float = 0.0
    half_life: float = math.nan
    hurst_exponent: float = math.nan
    variance_ratio: float = math.nan


@dataclass(slots=True)
class JohansenBasketReport:
    """Summary diagnostics for Johansen basket estimation."""

    rank: int
    trace_statistics: list[float]
    critical_values: list[list[float]]
    eigenvector: list[float]
    adf_statistic: float
    adf_pvalue: float
    spread_mean: float
    spread_std: float
    half_life: float = math.nan
    hurst_exponent: float = math.nan
    variance_ratio: float = math.nan


@dataclass(slots=True)
class WalkForwardWindowResult:
    """Metadata for one pair-trading walk-forward window."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    hedge_ratio: float
    adf_pvalue: float
    eg_pvalue: float
    traded: bool
    skip_reason: str | None = None
    intercept: float = 0.0
    half_life: float = math.nan
    hurst_exponent: float = math.nan
    variance_ratio: float = math.nan


@dataclass(slots=True)
class WalkForwardReport:
    """Aggregated walk-forward outputs for pair trading."""

    backtest: BacktestResult
    windows: list[WalkForwardWindowResult] = field(default_factory=list)


@dataclass(slots=True)
class WalkForwardEventReport:
    """Aggregated event-driven walk-forward outputs for pair trading."""

    backtest: EventBacktestResult
    target_weights: pd.DataFrame
    windows: list[WalkForwardWindowResult] = field(default_factory=list)


@dataclass(slots=True)
class BasketWalkForwardWindowResult:
    """Metadata for one Johansen basket walk-forward window."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    rank: int
    adf_pvalue: float
    traded: bool
    weights: dict[str, float]
    skip_reason: str | None = None
    half_life: float = math.nan
    hurst_exponent: float = math.nan
    variance_ratio: float = math.nan


@dataclass(slots=True)
class BasketWalkForwardReport:
    """Aggregated outputs for Johansen basket walk-forward trading."""

    backtest: EventBacktestResult
    target_weights: pd.DataFrame
    windows: list[BasketWalkForwardWindowResult] = field(default_factory=list)


@dataclass(slots=True)
class OrderEvent:
    """Target order submitted to the simulated broker."""

    timestamp: pd.Timestamp
    asset: str
    quantity: float
    reference_price: float
    target_weight: float


@dataclass(slots=True)
class FillEvent:
    """Executed order returned by the simulated broker."""

    timestamp: pd.Timestamp
    asset: str
    quantity: float
    fill_price: float
    gross_value: float
    commission: float
    slippage_cost: float
    reference_price: float = math.nan
