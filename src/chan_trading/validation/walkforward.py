from __future__ import annotations

import math

import pandas as pd

from chan_trading.backtest.engine import run_backtest
from chan_trading.config import MeanReversionConfig, WalkForwardConfig
from chan_trading.features.statistics import estimate_spread_diagnostics_object
from chan_trading.portfolio.sizing import apply_drawdown_throttle, apply_vol_target
from chan_trading.strategies.mean_reversion import PairMeanReversionStrategy
from chan_trading.types import SpreadDiagnostics, WalkForwardReport, WalkForwardWindowResult


def generate_walkforward_windows(
    index: pd.Index,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    min_train_size: int | None = None,
    purge: int = 0,
    embargo: int = 0,
) -> list[tuple[slice, slice]]:
    """Generate sequential walk-forward windows, optionally with purge/embargo.

    Chan (Ch. 1) stresses the importance of clean train/test separation when
    the strategy uses overlapping labels, lookbacks, or lagged signals. The
    purge/embargo scheme popularised by López de Prado (2018) addresses this:

    - **purge**: number of bars removed *between* the end of the train window
      and the start of the test window. This drops bars whose labels or
      signals depend on overlapping information.
    - **embargo**: number of bars added *after* the test window before the
      next walk-forward step begins. This prevents leakage of test-window
      information into the next training set via serial correlation.

    Both default to ``0`` so that v9 behaviour is preserved exactly.

    When ``min_train_size`` is provided, the train window becomes expanding and
    starts at ``max(train_size, min_train_size)`` observations. In that mode,
    the train window grows by ``step + embargo`` each iteration.
    """
    if purge < 0:
        raise ValueError("purge must be >= 0")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")

    step = step_size or test_size
    windows: list[tuple[slice, slice]] = []
    n = len(index)

    if min_train_size is None:
        start = 0
        while start + train_size + purge + test_size <= n:
            train_slice = slice(start, start + train_size)
            test_slice = slice(
                start + train_size + purge,
                start + train_size + purge + test_size,
            )
            windows.append((train_slice, test_slice))
            start += step + embargo
        return windows

    train_end = max(train_size, min_train_size)
    while train_end + purge + test_size <= n:
        train_slice = slice(0, train_end)
        test_slice = slice(train_end + purge, train_end + purge + test_size)
        windows.append((train_slice, test_slice))
        train_end += step + embargo
    return windows


def _positions_for_test_window(
    history_prices: pd.DataFrame,
    test_index: pd.Index,
    strategy: PairMeanReversionStrategy,
    strategy_config: MeanReversionConfig,
) -> pd.DataFrame:
    positions_full = strategy.generate_positions(history_prices)
    if strategy_config.vol_target is not None:
        positions_full = apply_vol_target(
            prices=history_prices,
            positions=positions_full,
            target_vol=strategy_config.vol_target,
            lookback=strategy_config.vol_lookback,
            max_leverage=strategy_config.max_leverage,
        )
    return positions_full.loc[test_index]


def _passes_additional_filters(
    half_life: float,
    hurst: float | None,
    variance_ratio: float | None,
    spread_std: float | None,
    wf_config: WalkForwardConfig,
) -> bool:
    if wf_config.min_half_life is not None and (not math.isfinite(half_life) or half_life < wf_config.min_half_life):
        return False
    if wf_config.max_half_life is not None and (not math.isfinite(half_life) or half_life > wf_config.max_half_life):
        return False
    if (
        wf_config.max_hurst_exponent is not None
        and hurst is not None
        and math.isfinite(hurst)
        and hurst > wf_config.max_hurst_exponent
    ):
        return False
    if (
        wf_config.max_variance_ratio is not None
        and variance_ratio is not None
        and math.isfinite(variance_ratio)
        and variance_ratio > wf_config.max_variance_ratio
    ):
        return False
    if (
        wf_config.min_spread_std is not None
        and spread_std is not None
        and math.isfinite(spread_std)
        and spread_std < wf_config.min_spread_std
    ):
        return False
    return True


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


def run_walkforward_pair_mean_reversion(
    prices: pd.DataFrame,
    y_col: str,
    x_col: str,
    strategy_config: MeanReversionConfig,
    wf_config: WalkForwardConfig,
) -> WalkForwardReport:
    """Run walk-forward mean-reversion with train-set diagnostics and test-set trading."""
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

    full_positions = pd.DataFrame(0.0, index=prices.index, columns=[y_col, x_col])
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
        traded = skip_reason is None and _passes_additional_filters(
            diagnostics.half_life,
            diagnostics.hurst_exponent,
            diagnostics.variance_ratio,
            diagnostics.spread_std,
            wf_config,
        )

        if traded:
            strategy = PairMeanReversionStrategy(
                y_col=y_col,
                x_col=x_col,
                hedge_ratio=diagnostics.beta,
                intercept=diagnostics.alpha,
                config=strategy_config,
            )
            positions = _positions_for_test_window(
                history_prices=history[[y_col, x_col]],
                test_index=test.index,
                strategy=strategy,
                strategy_config=strategy_config,
            )
            full_positions.loc[test.index, [y_col, x_col]] = positions

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

    preliminary_backtest = run_backtest(
        prices=prices[[y_col, x_col]],
        target_positions=full_positions[[y_col, x_col]],
        transaction_cost_bps=strategy_config.transaction_cost_bps,
    )
    throttled_positions = apply_drawdown_throttle(
        full_positions[[y_col, x_col]],
        preliminary_backtest.equity_curve,
        soft_limit=strategy_config.drawdown_soft_limit,
        soft_scale=strategy_config.drawdown_soft_scale,
        hard_limit=strategy_config.drawdown_hard_limit,
    )
    backtest = run_backtest(
        prices=prices[[y_col, x_col]],
        target_positions=throttled_positions,
        transaction_cost_bps=strategy_config.transaction_cost_bps,
    )
    return WalkForwardReport(backtest=backtest, windows=metadata)
