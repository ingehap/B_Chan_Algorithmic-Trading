from __future__ import annotations

from dataclasses import dataclass


_ALLOWED_SIGNAL_MODES = {"residual", "ratio"}
_ALLOWED_BAND_MODES = {"rolling", "ewm"}
_ALLOWED_NAN_MODES = {"hold", "flat"}
_ALLOWED_FILL_MODES = {"same_bar", "next_bar"}
_ALLOWED_HEDGE_MODES = {"static", "rolling", "kalman"}


@dataclass(slots=True)
class MeanReversionConfig:
    """Configuration for z-score based mean-reversion strategies."""

    lookback: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.5
    max_leverage: float = 1.0
    transaction_cost_bps: float = 2.0
    vol_target: float | None = None
    vol_lookback: int = 20
    use_log_prices: bool = True
    signal_mode: str = "residual"
    band_mode: str = "rolling"
    nan_mode: str = "hold"
    hedge_mode: str = "static"
    hedge_lookback: int | None = None
    kalman_delta: float = 1e-4
    kalman_observation_var: float = 1e-3
    max_holding_bars: int | None = None
    stop_z: float | None = None
    max_gross_exposure: float | None = None
    max_net_exposure: float | None = None
    drawdown_soft_limit: float | None = None
    drawdown_soft_scale: float = 0.5
    drawdown_hard_limit: float | None = None

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be >= 2")
        if self.entry_z <= 0:
            raise ValueError("entry_z must be > 0")
        if self.exit_z < 0:
            raise ValueError("exit_z must be >= 0")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be > 0")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be >= 0")
        if self.vol_target is not None and self.vol_target <= 0:
            raise ValueError("vol_target must be > 0 when provided")
        if self.vol_lookback < 2:
            raise ValueError("vol_lookback must be >= 2")
        if self.signal_mode not in _ALLOWED_SIGNAL_MODES:
            raise ValueError(f"signal_mode must be one of {sorted(_ALLOWED_SIGNAL_MODES)}")
        if self.band_mode not in _ALLOWED_BAND_MODES:
            raise ValueError(f"band_mode must be one of {sorted(_ALLOWED_BAND_MODES)}")
        if self.nan_mode not in _ALLOWED_NAN_MODES:
            raise ValueError(f"nan_mode must be one of {sorted(_ALLOWED_NAN_MODES)}")
        if self.hedge_mode not in _ALLOWED_HEDGE_MODES:
            raise ValueError(f"hedge_mode must be one of {sorted(_ALLOWED_HEDGE_MODES)}")
        if self.hedge_lookback is not None and self.hedge_lookback < 2:
            raise ValueError("hedge_lookback must be >= 2 when provided")
        if self.kalman_delta <= 0:
            raise ValueError("kalman_delta must be > 0")
        if self.kalman_observation_var <= 0:
            raise ValueError("kalman_observation_var must be > 0")
        if self.max_holding_bars is not None and self.max_holding_bars < 1:
            raise ValueError("max_holding_bars must be >= 1 when provided")
        if self.stop_z is not None and self.stop_z <= self.entry_z:
            raise ValueError("stop_z must be greater than entry_z when provided")
        if self.max_gross_exposure is not None and self.max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be > 0 when provided")
        if self.max_net_exposure is not None and self.max_net_exposure < 0:
            raise ValueError("max_net_exposure must be >= 0 when provided")
        if self.drawdown_soft_limit is not None and not (0 < self.drawdown_soft_limit < 1):
            raise ValueError("drawdown_soft_limit must be between 0 and 1 when provided")
        if not (0 < self.drawdown_soft_scale <= 1):
            raise ValueError("drawdown_soft_scale must be in (0, 1]")
        if self.drawdown_hard_limit is not None and not (0 < self.drawdown_hard_limit < 1):
            raise ValueError("drawdown_hard_limit must be between 0 and 1 when provided")
        if (
            self.drawdown_soft_limit is not None
            and self.drawdown_hard_limit is not None
            and self.drawdown_soft_limit >= self.drawdown_hard_limit
        ):
            raise ValueError("drawdown_soft_limit must be smaller than drawdown_hard_limit")


@dataclass(slots=True)
class MomentumConfig:
    """Configuration for simple interday time-series momentum."""

    lookback: int = 63
    long_only: bool = False
    return_threshold: float = 0.0
    max_leverage: float = 1.0
    vol_target: float | None = None
    vol_lookback: int = 20
    stop_loss: float | None = None
    stop_loss_lookback: int = 5
    # v13: exposure caps are now honoured by every strategy family.
    max_gross_exposure: float | None = None
    max_net_exposure: float | None = None

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be >= 2")
        if self.return_threshold < 0:
            raise ValueError("return_threshold must be >= 0")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be > 0")
        if self.vol_target is not None and self.vol_target <= 0:
            raise ValueError("vol_target must be > 0 when provided")
        if self.vol_lookback < 2:
            raise ValueError("vol_lookback must be >= 2")
        if self.stop_loss is not None and not (0 < self.stop_loss < 1):
            raise ValueError("stop_loss must be in (0, 1) when provided")
        if self.stop_loss_lookback < 1:
            raise ValueError("stop_loss_lookback must be >= 1")
        if self.max_gross_exposure is not None and self.max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be > 0 when provided")
        if self.max_net_exposure is not None and self.max_net_exposure < 0:
            raise ValueError("max_net_exposure must be >= 0 when provided")


@dataclass(slots=True)
class CrossSectionalMomentumConfig:
    """Configuration for cross-sectional (rank-based) momentum."""

    lookback: int = 126
    top_fraction: float = 0.2
    bottom_fraction: float = 0.2
    long_only: bool = False
    max_leverage: float = 1.0
    vol_target: float | None = None
    vol_lookback: int = 20
    # v13: universally-supported exposure caps.
    max_gross_exposure: float | None = None
    max_net_exposure: float | None = None

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be >= 2")
        if not (0 < self.top_fraction < 1):
            raise ValueError("top_fraction must be in (0, 1)")
        if not (0 < self.bottom_fraction < 1):
            raise ValueError("bottom_fraction must be in (0, 1)")
        if self.top_fraction + self.bottom_fraction > 1:
            raise ValueError("top_fraction + bottom_fraction must be <= 1")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be > 0")
        if self.vol_target is not None and self.vol_target <= 0:
            raise ValueError("vol_target must be > 0 when provided")
        if self.vol_lookback < 2:
            raise ValueError("vol_lookback must be >= 2")
        if self.max_gross_exposure is not None and self.max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be > 0 when provided")
        if self.max_net_exposure is not None and self.max_net_exposure < 0:
            raise ValueError("max_net_exposure must be >= 0 when provided")


@dataclass(slots=True)
class CrossSectionalMeanReversionConfig:
    """Configuration for cross-sectional mean reversion on a stock basket."""

    lookback: int = 1
    top_fraction: float = 0.2
    bottom_fraction: float = 0.2
    long_only: bool = False
    max_leverage: float = 1.0
    vol_target: float | None = None
    vol_lookback: int = 20
    # v13: universally-supported exposure caps.
    max_gross_exposure: float | None = None
    max_net_exposure: float | None = None

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1")
        if not (0 < self.top_fraction < 1):
            raise ValueError("top_fraction must be in (0, 1)")
        if not (0 < self.bottom_fraction < 1):
            raise ValueError("bottom_fraction must be in (0, 1)")
        if self.top_fraction + self.bottom_fraction > 1:
            raise ValueError("top_fraction + bottom_fraction must be <= 1")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be > 0")
        if self.vol_target is not None and self.vol_target <= 0:
            raise ValueError("vol_target must be > 0 when provided")
        if self.vol_lookback < 2:
            raise ValueError("vol_lookback must be >= 2")
        if self.max_gross_exposure is not None and self.max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be > 0 when provided")
        if self.max_net_exposure is not None and self.max_net_exposure < 0:
            raise ValueError("max_net_exposure must be >= 0 when provided")


@dataclass(slots=True)
class BuyOnGapConfig:
    """Configuration for the daily buy-on-gap mean-reversion strategy.

    *New in v11.* Implements Chan's opening-gap / short-term reversal
    pattern (Ch. 4 & 7) adapted to daily close-to-close bars. See
    :class:`chan_trading.strategies.opening_gap.BuyOnGapStrategy` for
    semantics.

    Parameters
    ----------
    lookback
        Rolling window used to estimate per-asset return volatility. The
        trigger compares today's one-bar return to this trailing std.
    threshold_sigma
        Minimum adverse move, measured in trailing standard deviations,
        required to fire a long entry (and, if ``two_sided``, a symmetric
        short on up-moves). Typical values 1.0-2.0.
    hold_bars
        Number of bars to hold each fired entry.
    two_sided
        If ``True``, also short large up-gaps. Default ``False`` because
        Chan's empirical result is long-only on down-gaps.
    max_leverage
        Gross-exposure cap applied after equal-weighting active signals.
    vol_target, vol_lookback
        Optional portfolio-level vol target and its lookback, matching
        the convention used by the other strategy configs.
    """

    lookback: int = 20
    threshold_sigma: float = 1.0
    hold_bars: int = 1
    two_sided: bool = False
    max_leverage: float = 1.0
    vol_target: float | None = None
    vol_lookback: int = 20
    # v13: universally-supported exposure caps.
    max_gross_exposure: float | None = None
    max_net_exposure: float | None = None

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be >= 2")
        if self.threshold_sigma <= 0:
            raise ValueError("threshold_sigma must be > 0")
        if self.hold_bars < 1:
            raise ValueError("hold_bars must be >= 1")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be > 0")
        if self.vol_target is not None and self.vol_target <= 0:
            raise ValueError("vol_target must be > 0 when provided")
        if self.vol_lookback < 2:
            raise ValueError("vol_lookback must be >= 2")
        if self.max_gross_exposure is not None and self.max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be > 0 when provided")
        if self.max_net_exposure is not None and self.max_net_exposure < 0:
            raise ValueError("max_net_exposure must be >= 0 when provided")


@dataclass(slots=True)
class WalkForwardConfig:
    """Configuration for walk-forward evaluation.

    **New in v10** — ``purge`` and ``embargo`` implement the standard
    López de Prado hygiene around overlapping return windows:

    - ``purge`` — number of bars to *drop* between the end of train and
      the start of test. Set it to at least the holding horizon to prevent
      labels computed on train bars from overlapping with features on test
      bars.
    - ``embargo`` — number of bars to skip *after* the test window before
      the next walk step. This prevents information from the test window
      from leaking into the next training block via serial correlation.

    Both default to 0, which reproduces v9 behavior exactly.
    """

    train_size: int = 252
    test_size: int = 63
    step_size: int | None = None
    min_train_size: int | None = None
    adf_alpha: float = 0.10
    eg_alpha: float = 0.10
    min_half_life: float | None = None
    max_half_life: float | None = None
    max_hurst_exponent: float | None = None
    max_variance_ratio: float | None = None
    min_spread_std: float | None = None
    purge: int = 0
    embargo: int = 0

    def __post_init__(self) -> None:
        if self.train_size < 20:
            raise ValueError("train_size must be >= 20")
        if self.test_size < 1:
            raise ValueError("test_size must be >= 1")
        if self.step_size is not None and self.step_size < 1:
            raise ValueError("step_size must be >= 1 when provided")
        if self.min_train_size is not None and self.min_train_size < 20:
            raise ValueError("min_train_size must be >= 20 when provided")
        if not (0 < self.adf_alpha < 1):
            raise ValueError("adf_alpha must be between 0 and 1")
        if not (0 < self.eg_alpha < 1):
            raise ValueError("eg_alpha must be between 0 and 1")
        if self.min_half_life is not None and self.min_half_life <= 0:
            raise ValueError("min_half_life must be > 0 when provided")
        if self.max_half_life is not None and self.max_half_life <= 0:
            raise ValueError("max_half_life must be > 0 when provided")
        if (
            self.min_half_life is not None
            and self.max_half_life is not None
            and self.min_half_life > self.max_half_life
        ):
            raise ValueError("min_half_life cannot exceed max_half_life")
        if self.max_hurst_exponent is not None and not (0 < self.max_hurst_exponent < 1):
            raise ValueError("max_hurst_exponent must be between 0 and 1 when provided")
        if self.max_variance_ratio is not None and self.max_variance_ratio <= 0:
            raise ValueError("max_variance_ratio must be > 0 when provided")
        if self.min_spread_std is not None and self.min_spread_std <= 0:
            raise ValueError("min_spread_std must be > 0 when provided")
        if self.purge < 0:
            raise ValueError("purge must be >= 0")
        if self.embargo < 0:
            raise ValueError("embargo must be >= 0")


@dataclass(slots=True)
class JohansenConfig:
    """Configuration for Johansen cointegration estimation."""

    det_order: int = 0
    k_ar_diff: int = 1
    significance: float = 0.05

    def __post_init__(self) -> None:
        if self.det_order not in {-1, 0, 1}:
            raise ValueError("det_order must be one of -1, 0, 1")
        if self.k_ar_diff < 1:
            raise ValueError("k_ar_diff must be >= 1")
        if self.significance not in {0.10, 0.05, 0.01}:
            raise ValueError("significance must be one of 0.10, 0.05, 0.01")


@dataclass(slots=True)
class EventBacktestConfig:
    """Configuration for the daily-bar event-driven execution engine.

    *Extended in v11.* Added optional ``impact_model`` and ``adv`` fields
    so the simulator can charge a participation-rate-sensitive impact
    cost on top of the flat ``slippage_bps``. When both are ``None``
    (default) v10 behaviour is reproduced exactly.
    """

    initial_cash: float = 1_000_000.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    min_trade_value: float = 0.0
    lag_target_weights: int = 1
    allow_fractional: bool = True
    min_commission: float = 0.0
    borrow_bps_annual: float = 0.0
    cash_interest_bps_annual: float = 0.0
    min_weight_change: float = 0.0
    fill_mode: str = "next_bar"
    impact_model: object | None = None  # ImpactModel; typed as object to avoid import cycle
    adv: object | None = None  # pd.DataFrame aligned to prices (asset ADV in currency)
    # v14: deterministic half bid-ask spread in bps. Added on top of
    # slippage_bps and any market-impact bps inside the broker.
    half_spread_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be > 0")
        if self.commission_bps < 0:
            raise ValueError("commission_bps must be >= 0")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be >= 0")
        if self.min_trade_value < 0:
            raise ValueError("min_trade_value must be >= 0")
        if self.lag_target_weights < 0:
            raise ValueError("lag_target_weights must be >= 0")
        if self.min_commission < 0:
            raise ValueError("min_commission must be >= 0")
        if self.borrow_bps_annual < 0:
            raise ValueError("borrow_bps_annual must be >= 0")
        if self.cash_interest_bps_annual < 0:
            raise ValueError("cash_interest_bps_annual must be >= 0")
        if self.min_weight_change < 0:
            raise ValueError("min_weight_change must be >= 0")
        if self.fill_mode not in _ALLOWED_FILL_MODES:
            raise ValueError(f"fill_mode must be one of {sorted(_ALLOWED_FILL_MODES)}")
        if (self.impact_model is None) != (self.adv is None):
            raise ValueError(
                "impact_model and adv must both be provided together, or both be None"
            )
        if self.half_spread_bps < 0:
            raise ValueError("half_spread_bps must be >= 0")
