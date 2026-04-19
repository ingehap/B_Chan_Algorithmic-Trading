from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from chan_trading.config import MeanReversionConfig
from chan_trading.features.statistics import (
    estimate_kalman_hedge_model,
    estimate_rolling_hedge_model,
    ewm_zscore,
    rolling_zscore,
    spread_from_ratio,
    spread_from_regression,
    transform_pair_prices,
)
from chan_trading.portfolio.sizing import apply_exposure_caps
from chan_trading.strategies.base import Strategy


def _compute_zscore(series: pd.Series, config: MeanReversionConfig) -> pd.Series:
    if config.band_mode == "rolling":
        return rolling_zscore(series, config.lookback)
    if config.band_mode == "ewm":
        return ewm_zscore(series, config.lookback)
    raise ValueError(f"Unsupported band_mode: {config.band_mode}")


def _state_on_nan(state: float, config: MeanReversionConfig) -> float:
    if config.nan_mode == "hold":
        return state
    if config.nan_mode == "flat":
        return 0.0
    raise ValueError(f"Unsupported nan_mode: {config.nan_mode}")


def _apply_exposure_caps(weights: pd.DataFrame, config: MeanReversionConfig) -> pd.DataFrame:
    """Thin shim over :func:`chan_trading.portfolio.sizing.apply_exposure_caps`.

    *v13:* the implementation was promoted to a shared ``portfolio.sizing``
    utility so every strategy family can use it (momentum, cross-sectional,
    buy-on-gap did not previously honour their config's exposure caps).
    Kept here as a private shim for import-path back-compatibility with
    anything that used to import ``_apply_exposure_caps`` from this
    module.
    """
    return apply_exposure_caps(
        weights,
        max_gross_exposure=config.max_gross_exposure,
        max_net_exposure=config.max_net_exposure,
    )


def _rolling_or_static_hedge(
    y_model: pd.Series,
    x_model: pd.Series,
    hedge_ratio: float,
    intercept: float,
    config: MeanReversionConfig,
) -> tuple[pd.Series, pd.Series]:
    if config.hedge_mode == "static":
        alpha = pd.Series(intercept, index=y_model.index, dtype=float)
        beta = pd.Series(hedge_ratio, index=y_model.index, dtype=float)
        return alpha, beta
    if config.hedge_mode == "rolling":
        window = config.hedge_lookback or config.lookback
        hedge = estimate_rolling_hedge_model(y_model, x_model, window=window)
        return hedge["alpha"].ffill(), hedge["beta"].ffill()
    if config.hedge_mode == "kalman":
        hedge = estimate_kalman_hedge_model(
            y_model,
            x_model,
            delta=config.kalman_delta,
            observation_var=config.kalman_observation_var,
        )
        return hedge["alpha"].ffill(), hedge["beta"].ffill()
    raise ValueError(f"Unsupported hedge_mode: {config.hedge_mode}")


def _generate_stateful_signal(z: pd.Series, config: MeanReversionConfig) -> pd.Series:
    """Path-dependent ±1/0 state machine from a z-score series.

    The state machine is inherently path-dependent (the signal at ``t``
    depends on the position at ``t-1``), so this function still walks
    the series bar-by-bar. *New in v11*: the walk is now over a
    preallocated ``numpy`` array rather than per-bar ``.loc`` assignments,
    which runs roughly 50-100× faster for typical series lengths without
    changing the transition semantics at all.

    Semantics (identical to v10):

    - flat → long (+1) when ``z < -entry_z``; flat → short (−1) when
      ``z > +entry_z``.
    - held position exits to flat when ``|z| < exit_z``, when
      ``max_holding_bars`` is exceeded, or when ``|z| >= stop_z``.
    - NaN bars are resolved via ``nan_mode`` (``hold`` or ``flat``) and
      still count toward ``max_holding_bars`` when a position is held.
    """
    values = z.to_numpy(dtype=float)
    out = np.zeros(values.shape[0], dtype=float)
    state = 0.0
    hold_bars = 0

    entry_z = float(config.entry_z)
    exit_z = float(config.exit_z)
    stop_z = float(config.stop_z) if config.stop_z is not None else None
    max_hold = int(config.max_holding_bars) if config.max_holding_bars is not None else None
    nan_mode_flat = config.nan_mode == "flat"

    for i in range(values.shape[0]):
        zi = values[i]

        if np.isnan(zi):
            if nan_mode_flat:
                state = 0.0
            if state == 0.0:
                hold_bars = 0
            else:
                hold_bars += 1
                if max_hold is not None and hold_bars > max_hold:
                    state = 0.0
                    hold_bars = 0
            out[i] = state
            continue

        stop_hit = stop_z is not None and abs(zi) >= stop_z
        if state == 0.0:
            if stop_hit:
                hold_bars = 0
            elif zi > entry_z:
                state = -1.0
                hold_bars = 1
            elif zi < -entry_z:
                state = 1.0
                hold_bars = 1
            else:
                hold_bars = 0
        else:
            hold_bars += 1
            exit_for_z = abs(zi) < exit_z
            exit_for_time = max_hold is not None and hold_bars > max_hold
            if stop_hit or exit_for_z or exit_for_time:
                state = 0.0
                hold_bars = 0

        out[i] = state

    return pd.Series(out, index=z.index, dtype=float)


@dataclass(slots=True)
class PairMeanReversionStrategy(Strategy):
    """Simple Chan-style pairs mean-reversion strategy."""

    y_col: str
    x_col: str
    hedge_ratio: float
    config: MeanReversionConfig
    intercept: float = 0.0

    def generate_positions(self, prices: pd.DataFrame) -> pd.DataFrame:
        required = {self.y_col, self.x_col}
        missing = required - set(prices.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        y_raw = prices[self.y_col].astype(float)
        x_raw = prices[self.x_col].astype(float)
        y_model, x_model = transform_pair_prices(
            y_raw,
            x_raw,
            use_log_prices=self.config.use_log_prices,
        )
        alpha_series, beta_series = _rolling_or_static_hedge(
            y_model=y_model,
            x_model=x_model,
            hedge_ratio=self.hedge_ratio,
            intercept=self.intercept,
            config=self.config,
        )

        if self.config.signal_mode == "residual":
            spread = spread_from_regression(y_model, x_model, alpha_series, beta_series)
        elif self.config.signal_mode == "ratio":
            spread = spread_from_ratio(y_model, x_model, beta_series)
        else:
            raise ValueError(f"Unsupported signal_mode: {self.config.signal_mode}")

        z = _compute_zscore(spread, self.config)
        raw_signal = _generate_stateful_signal(z, self.config)

        gross = 1.0 + beta_series.abs()
        y_w = raw_signal * (1.0 / gross) * self.config.max_leverage
        x_w = -raw_signal * (beta_series / gross) * self.config.max_leverage
        weights = pd.DataFrame({self.y_col: y_w, self.x_col: x_w}, index=prices.index).fillna(0.0)
        return _apply_exposure_caps(weights, self.config)
