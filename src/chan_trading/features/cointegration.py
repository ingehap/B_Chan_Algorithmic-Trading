from __future__ import annotations

import math

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from chan_trading.config import JohansenConfig
from chan_trading.features.statistics import (
    adf_stationarity_test,
    estimate_half_life,
    estimate_hurst_exponent,
    estimate_variance_ratio,
)
from chan_trading.types import JohansenBasketReport


_CRIT_INDEX = {0.10: 0, 0.05: 1, 0.01: 2}


def _prepare_price_matrix(prices: pd.DataFrame, use_log_prices: bool = True) -> pd.DataFrame:
    clean = prices.dropna().astype(float)
    if clean.shape[0] < 30:
        raise ValueError("Need at least 30 aligned rows for Johansen estimation")
    if clean.shape[1] < 2:
        raise ValueError("Need at least 2 assets for Johansen estimation")
    if use_log_prices:
        if (clean <= 0).any().any():
            raise ValueError("Log-price Johansen requires strictly positive prices")
        clean = pd.DataFrame(np.log(clean.to_numpy(dtype=float)), index=clean.index, columns=clean.columns)
    return clean


def johansen_rank(trace_statistics: np.ndarray, critical_values: np.ndarray, significance: float) -> int:
    """Select rank using the Johansen trace statistic."""
    crit_col = _CRIT_INDEX[significance]
    rank = 0
    max_rank = min(len(trace_statistics), critical_values.shape[0])
    for i in range(max_rank):
        if float(trace_statistics[i]) > float(critical_values[i, crit_col]):
            rank += 1
        else:
            break
    return rank


def normalize_eigenvector(vector: np.ndarray, columns: list[str]) -> pd.Series:
    """Normalize eigenvector to unit gross exposure and stable sign."""
    vec = np.asarray(vector, dtype=float).reshape(-1)
    gross = np.abs(vec).sum()
    if gross == 0 or np.isnan(gross):
        raise ValueError("Cannot normalize a zero Johansen eigenvector")
    vec = vec / gross

    for value in vec:
        if abs(value) > 1e-12:
            if value < 0:
                vec = -vec
            break

    return pd.Series(vec, index=columns, dtype=float)


def _candidate_score(spread: pd.Series) -> tuple[float, float, float, float]:
    adf_stat, adf_pvalue = adf_stationarity_test(spread)
    half_life = estimate_half_life(spread)
    finite_half_life = half_life if np.isfinite(half_life) else math.inf
    hurst = estimate_hurst_exponent(spread)
    finite_hurst = hurst if np.isfinite(hurst) else math.inf
    return float(adf_pvalue), float(finite_half_life), float(finite_hurst), float(adf_stat)


def _select_best_cointegrating_vector(
    clean: pd.DataFrame,
    raw_eigenvectors: np.ndarray,
    rank: int,
) -> tuple[pd.Series, pd.Series, float, float]:
    best_weights: pd.Series | None = None
    best_spread: pd.Series | None = None
    best_score: tuple[float, float, float, float] | None = None
    best_adf_stat = math.nan
    best_adf_pvalue = math.nan

    for idx in range(rank):
        weights = normalize_eigenvector(raw_eigenvectors[:, idx], list(clean.columns))
        spread = clean @ weights
        score = _candidate_score(spread)
        if best_score is None or score < best_score:
            best_weights = weights
            best_spread = spread
            best_score = score
            best_adf_stat = score[3]
            best_adf_pvalue = score[0]

    if best_weights is None or best_spread is None:
        raise ValueError("Failed to select a Johansen cointegration vector")
    return best_weights, best_spread, best_adf_stat, best_adf_pvalue


def johansen_basket_from_prices(
    prices: pd.DataFrame,
    config: JohansenConfig | None = None,
    use_log_prices: bool = True,
) -> tuple[pd.Series, pd.Series, JohansenBasketReport]:
    """Estimate basket weights and spread diagnostics from Johansen cointegration."""
    cfg = config or JohansenConfig()
    clean = _prepare_price_matrix(prices, use_log_prices=use_log_prices)
    result = coint_johansen(clean.to_numpy(), det_order=cfg.det_order, k_ar_diff=cfg.k_ar_diff)

    rank = johansen_rank(result.lr1, result.cvt, significance=cfg.significance)
    if rank < 1:
        raise ValueError("Johansen test did not detect a cointegration relation at the requested significance")

    weights, spread, adf_stat, adf_pvalue = _select_best_cointegrating_vector(clean, result.evec, rank)
    half_life = estimate_half_life(spread)
    hurst = estimate_hurst_exponent(spread)
    variance_ratio = estimate_variance_ratio(spread)

    report = JohansenBasketReport(
        rank=rank,
        trace_statistics=[float(x) for x in result.lr1.tolist()],
        critical_values=[[float(v) for v in row] for row in result.cvt.tolist()],
        eigenvector=[float(x) for x in weights.tolist()],
        adf_statistic=adf_stat,
        adf_pvalue=adf_pvalue,
        spread_mean=float(spread.mean()),
        spread_std=float(spread.std(ddof=0)),
        half_life=half_life,
        hurst_exponent=hurst,
        variance_ratio=variance_ratio,
    )
    return weights, spread, report
