from __future__ import annotations

import math

import pandas as pd

from chan_trading.backtest.event_engine import run_event_backtest
from chan_trading.config import EventBacktestConfig, JohansenConfig, MeanReversionConfig, WalkForwardConfig
from chan_trading.features.cointegration import johansen_basket_from_prices
from chan_trading.portfolio.sizing import apply_vol_target
from chan_trading.strategies.basket_mean_reversion import BasketMeanReversionStrategy
from chan_trading.types import BasketWalkForwardReport, BasketWalkForwardWindowResult
from chan_trading.validation.walkforward import generate_walkforward_windows


def _target_weights_for_test_window(
    history_prices: pd.DataFrame,
    test_index: pd.Index,
    strategy: BasketMeanReversionStrategy,
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


def _basket_skip_reason(
    *,
    estimation_failed: bool,
    rank: int,
    adf_pvalue: float,
    half_life: float,
    hurst: float,
    variance_ratio: float,
    spread_std: float,
    wf_config: WalkForwardConfig,
) -> str | None:
    if estimation_failed:
        return "johansen_estimation_failed"
    if rank < 1:
        return "rank_zero"
    if adf_pvalue >= wf_config.adf_alpha:
        return "adf_pvalue"
    if wf_config.max_half_life is not None and (not math.isfinite(half_life) or half_life > wf_config.max_half_life):
        return "half_life_too_long"
    if wf_config.max_hurst_exponent is not None and math.isfinite(hurst) and hurst > wf_config.max_hurst_exponent:
        return "hurst_too_high"
    if wf_config.max_variance_ratio is not None and math.isfinite(variance_ratio) and variance_ratio > wf_config.max_variance_ratio:
        return "variance_ratio_too_high"
    if wf_config.min_spread_std is not None and math.isfinite(spread_std) and spread_std < wf_config.min_spread_std:
        return "spread_std_too_small"
    return None


def run_walkforward_johansen_basket_event_backtest(
    prices: pd.DataFrame,
    asset_cols: list[str],
    strategy_config: MeanReversionConfig,
    wf_config: WalkForwardConfig,
    johansen_config: JohansenConfig,
    event_config: EventBacktestConfig,
) -> BasketWalkForwardReport:
    """Run walk-forward Johansen basket mean-reversion with event-driven execution."""
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

    full_target_weights = pd.DataFrame(0.0, index=prices.index, columns=asset_cols)
    metadata: list[BasketWalkForwardWindowResult] = []

    for train_slice, test_slice in windows:
        train = prices.iloc[train_slice][asset_cols]
        test = prices.iloc[test_slice][asset_cols]
        history = prices.iloc[train_slice.start : test_slice.stop][asset_cols]

        weight_map = {asset: 0.0 for asset in asset_cols}
        rank = 0
        adf_pvalue = 1.0
        half_life = math.nan
        hurst = math.nan
        variance_ratio = math.nan
        spread_std = math.nan
        estimation_failed = False

        try:
            weights, _spread, report = johansen_basket_from_prices(
                train,
                config=johansen_config,
                use_log_prices=strategy_config.use_log_prices,
            )
            rank = report.rank
            adf_pvalue = report.adf_pvalue
            half_life = report.half_life
            hurst = report.hurst_exponent
            variance_ratio = report.variance_ratio
            spread_std = report.spread_std
            weight_map = {asset: float(weights.loc[asset]) for asset in asset_cols}
        except ValueError:
            estimation_failed = True

        skip_reason = _basket_skip_reason(
            estimation_failed=estimation_failed,
            rank=rank,
            adf_pvalue=adf_pvalue,
            half_life=half_life,
            hurst=hurst,
            variance_ratio=variance_ratio,
            spread_std=spread_std,
            wf_config=wf_config,
        )
        traded = skip_reason is None

        if traded:
            strategy = BasketMeanReversionStrategy(weights=weights, config=strategy_config)
            target_weights = _target_weights_for_test_window(
                history_prices=history,
                test_index=test.index,
                strategy=strategy,
                strategy_config=strategy_config,
            )
            full_target_weights.loc[test.index, asset_cols] = target_weights[asset_cols]

        metadata.append(
            BasketWalkForwardWindowResult(
                train_start=train.index[0],
                train_end=train.index[-1],
                test_start=test.index[0],
                test_end=test.index[-1],
                rank=rank,
                adf_pvalue=adf_pvalue,
                traded=traded,
                weights=weight_map,
                skip_reason=None if traded else skip_reason,
                half_life=half_life,
                hurst_exponent=hurst,
                variance_ratio=variance_ratio,
            )
        )

    backtest = run_event_backtest(
        prices=prices[asset_cols],
        target_weights=full_target_weights[asset_cols],
        config=event_config,
    )
    return BasketWalkForwardReport(
        backtest=backtest,
        target_weights=full_target_weights,
        windows=metadata,
    )
