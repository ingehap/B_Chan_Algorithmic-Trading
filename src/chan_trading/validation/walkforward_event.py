from __future__ import annotations

import math

import pandas as pd

from chan_trading.backtest.event_engine import run_event_backtest
from chan_trading.config import EventBacktestConfig, MeanReversionConfig, WalkForwardConfig
from chan_trading.features.statistics import estimate_spread_diagnostics_object
from chan_trading.portfolio.sizing import apply_drawdown_throttle, apply_vol_target
from chan_trading.strategies.mean_reversion import PairMeanReversionStrategy
from chan_trading.types import EventBacktestResult, SpreadDiagnostics, WalkForwardEventReport, WalkForwardWindowResult
from chan_trading.validation.walkforward import generate_walkforward_windows


def _target_weights_for_test_window(
    history_prices: pd.DataFrame,
    test_index: pd.Index,
    strategy: PairMeanReversionStrategy,
    strategy_config: MeanReversionConfig,
) -> pd.DataFrame:
    target_weights_full = strategy.generate_positions(history_prices)
    if strategy_config.vol_target is not None:
        target_weights_full = apply_vol_target(
            prices=history_prices,
            positions=target_weights_full,
            target_vol=strategy_config.vol_target,
            lookback=strategy_config.vol_lookback,
            max_leverage=strategy_config.max_leverage,
        )
    return target_weights_full.loc[test_index]


def _window_skip_reason(diagnostics: SpreadDiagnostics, wf_config: WalkForwardConfig) -> str | None:
    if diagnostics.adf_pvalue >= wf_config.adf_alpha:
        return "adf_pvalue"
    if diagnostics.eg_pvalue >= wf_config.eg_alpha:
        return "engle_granger_pvalue"
    if diagnostics.collinearity_warning:
        return "collinearity_warning"
    if wf_config.min_half_life is not None and (
        not math.isfinite(diagnostics.half_life) or diagnostics.half_life < wf_config.min_half_life
    ):
        return "half_life_too_short"
    if wf_config.max_half_life is not None and (
        not math.isfinite(diagnostics.half_life) or diagnostics.half_life > wf_config.max_half_life
    ):
        return "half_life_too_long"
    if (
        wf_config.max_hurst_exponent is not None
        and diagnostics.hurst_exponent is not None
        and math.isfinite(diagnostics.hurst_exponent)
        and diagnostics.hurst_exponent > wf_config.max_hurst_exponent
    ):
        return "hurst_too_high"
    if (
        wf_config.max_variance_ratio is not None
        and diagnostics.variance_ratio is not None
        and math.isfinite(diagnostics.variance_ratio)
        and diagnostics.variance_ratio > wf_config.max_variance_ratio
    ):
        return "variance_ratio_too_high"
    if (
        wf_config.min_spread_std is not None
        and diagnostics.spread_std is not None
        and math.isfinite(diagnostics.spread_std)
        and diagnostics.spread_std < wf_config.min_spread_std
    ):
        return "spread_std_too_small"
    return None


def run_pair_mean_reversion_event_backtest(
    prices: pd.DataFrame,
    y_col: str,
    x_col: str,
    strategy_config: MeanReversionConfig,
    event_config: EventBacktestConfig,
) -> tuple[SpreadDiagnostics, pd.DataFrame, EventBacktestResult]:
    """Run pair mean reversion with the daily-bar event engine."""
    diagnostics = estimate_spread_diagnostics_object(
        prices[y_col],
        prices[x_col],
        use_log_prices=strategy_config.use_log_prices,
        signal_mode=strategy_config.signal_mode,
    )
    strategy = PairMeanReversionStrategy(
        y_col=y_col,
        x_col=x_col,
        hedge_ratio=diagnostics.beta,
        intercept=diagnostics.alpha,
        config=strategy_config,
    )
    target_weights = strategy.generate_positions(prices[[y_col, x_col]])
    if strategy_config.vol_target is not None:
        target_weights = apply_vol_target(
            prices=prices[[y_col, x_col]],
            positions=target_weights,
            target_vol=strategy_config.vol_target,
            lookback=strategy_config.vol_lookback,
            max_leverage=strategy_config.max_leverage,
        )
    backtest = run_event_backtest(
        prices=prices[[y_col, x_col]],
        target_weights=target_weights[[y_col, x_col]],
        config=event_config,
    )
    return diagnostics, target_weights, backtest


def run_walkforward_pair_mean_reversion_event_backtest(
    prices: pd.DataFrame,
    y_col: str,
    x_col: str,
    strategy_config: MeanReversionConfig,
    wf_config: WalkForwardConfig,
    event_config: EventBacktestConfig,
) -> WalkForwardEventReport:
    """Run event-driven walk-forward pair mean reversion with train-set diagnostics."""
    windows = generate_walkforward_windows(
        index=prices.index,
        train_size=wf_config.train_size,
        test_size=wf_config.test_size,
        step_size=wf_config.step_size,
        min_train_size=wf_config.min_train_size,
        purge=wf_config.purge,
        embargo=wf_config.embargo,
    )
    if not windows:
        raise ValueError("No walk-forward windows could be created")

    full_target_weights = pd.DataFrame(0.0, index=prices.index, columns=[y_col, x_col])
    metadata: list[WalkForwardWindowResult] = []

    for train_slice, test_slice in windows:
        train = prices.iloc[train_slice]
        test = prices.iloc[test_slice]
        history = prices.iloc[train_slice.start : test_slice.stop]
        diagnostics = estimate_spread_diagnostics_object(
            train[y_col],
            train[x_col],
            use_log_prices=strategy_config.use_log_prices,
            signal_mode=strategy_config.signal_mode,
        )
        skip_reason = _window_skip_reason(diagnostics, wf_config)
        traded = skip_reason is None

        if traded:
            strategy = PairMeanReversionStrategy(
                y_col=y_col,
                x_col=x_col,
                hedge_ratio=diagnostics.beta,
                intercept=diagnostics.alpha,
                config=strategy_config,
            )
            target_weights = _target_weights_for_test_window(
                history_prices=history[[y_col, x_col]],
                test_index=test.index,
                strategy=strategy,
                strategy_config=strategy_config,
            )
            full_target_weights.loc[test.index, [y_col, x_col]] = target_weights

        metadata.append(
            WalkForwardWindowResult(
                train_start=train.index[0],
                train_end=train.index[-1],
                test_start=test.index[0],
                test_end=test.index[-1],
                hedge_ratio=diagnostics.beta,
                adf_pvalue=diagnostics.adf_pvalue,
                eg_pvalue=diagnostics.eg_pvalue,
                traded=traded,
                skip_reason=None if traded else skip_reason,
                intercept=diagnostics.alpha,
                half_life=diagnostics.half_life,
                hurst_exponent=float(diagnostics.hurst_exponent) if diagnostics.hurst_exponent is not None else math.nan,
                variance_ratio=float(diagnostics.variance_ratio) if diagnostics.variance_ratio is not None else math.nan,
            )
        )

    preliminary_backtest = run_event_backtest(
        prices=prices[[y_col, x_col]],
        target_weights=full_target_weights[[y_col, x_col]],
        config=event_config,
    )
    throttled_target_weights = apply_drawdown_throttle(
        full_target_weights[[y_col, x_col]],
        preliminary_backtest.equity_curve,
        soft_limit=strategy_config.drawdown_soft_limit,
        soft_scale=strategy_config.drawdown_soft_scale,
        hard_limit=strategy_config.drawdown_hard_limit,
    )
    backtest = run_event_backtest(
        prices=prices[[y_col, x_col]],
        target_weights=throttled_target_weights[[y_col, x_col]],
        config=event_config,
    )
    return WalkForwardEventReport(
        backtest=backtest,
        target_weights=throttled_target_weights,
        windows=metadata,
    )
