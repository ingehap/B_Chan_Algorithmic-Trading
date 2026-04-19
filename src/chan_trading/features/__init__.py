"""Statistical features and diagnostics."""

from chan_trading.features.futures import (
    ContinuousContractResult,
    build_continuous_contract,
    calendar_spread,
    decompose_futures_returns,
)
from chan_trading.features.statistics import (
    estimate_rs_hurst_exponent,
    suggest_zscore_lookback,
)

__all__ = [
    "ContinuousContractResult",
    "build_continuous_contract",
    "calendar_spread",
    "decompose_futures_returns",
    "estimate_rs_hurst_exponent",
    "suggest_zscore_lookback",
]
