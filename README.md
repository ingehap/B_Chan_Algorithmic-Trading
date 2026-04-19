# chan-trading

A Python research scaffold for the statistical-arbitrage and momentum
strategies described in Ernest P. Chan's *Algorithmic Trading: Winning
Strategies and Their Rationale* (Wiley, 2013). The library provides
cointegration-based pair and basket mean-reversion, time-series and
cross-sectional momentum, a daily-bar event-driven execution engine
with realistic frictions, Kelly and CPPI position sizing, Monte Carlo
drawdown tools, walk-forward validation, a multi-strategy ensemble
combiner, futures term-structure helpers, permutation-based data-
snooping defence, and — new in v14 — Newey-West HAC-adjusted Sharpe
variance/t-stat, a rolling Sharpe helper, an information ratio against
a benchmark, and an asymmetric bid-ask half-spread in the simulated
broker.

**Current version: 0.15.0 (v14).** v14 extends the robustness and
cost-realism surface: it adds Chan's Ch. 1 concern about serially
correlated returns inflating the naive Sharpe standard error
(Newey-West HAC variance), separates deterministic half-spread from
stochastic slippage in the broker, exposes a rolling Sharpe series for
regime-aware position scaling, and adds the standard benchmark-relative
information ratio. See `UPGRADE_REPORT_v14.md` for the full change
list.

## Installation

```bash
pip install -e .[dev]
pytest -q
```

Dependencies: Python ≥3.10, `numpy`, `pandas`, `scipy`, `statsmodels`.

## What's new in v14

### 1. Newey-West HAC-adjusted Sharpe variance and t-stat

Chan (Ch. 1) explicitly warns that a sample Sharpe ratio computed
under the IID assumption is misleading when the strategy's returns are
serially correlated — the default for most mean-reversion and
calendar-spread strategies. v14 adds

- `risk.metrics.newey_west_sharpe_variance(returns, *, lags, periods_per_year)`
- `risk.metrics.newey_west_sharpe_tstat(returns, *, lags, periods_per_year, risk_free_rate)`

For a per-period series with ``T`` observations and ``L`` Bartlett-kernel
lags,

```
Var_HAC(r_bar) = (1/T) · [ γ_0 + 2 · Σ_{k=1}^{L} (1 − k/(L+1)) · γ_k ]
Var(SR_annual) = periods_per_year · Var_HAC(r_bar) / sigma^2
```

When `lags=0` the HAC collapses exactly to `sigma^2 / T` and reproduces
the naive IID baseline, so a side-by-side comparison quantifies how
much the Sharpe SE has been under- or over-stated. The default lag
uses the Newey-West (1994) automatic rule.

### 2. Asymmetric bid-ask half-spread in `SimulatedBroker`

Chan (Ch. 1) treats spread (deterministic) and slippage (stochastic)
as conceptually distinct friction layers; v13 lumped them into a
single `slippage_bps` knob. v14 adds a new keyword to
`SimulatedBroker` (and `EventBacktestConfig`):

```python
broker = SimulatedBroker(
    commission_bps=1.0,
    slippage_bps=2.0,        # stochastic impact-style
    half_spread_bps=5.0,      # deterministic "pay the spread"
)
```

Buys cross the ask at `mid · (1 + half_spread_rate + slippage_rate)`
and sells cross the bid at `mid · (1 − half_spread_rate − slippage_rate)`,
both moving against the trader as they should. Default
`half_spread_bps=0.0` reproduces v13 fills byte-for-byte.

### 3. Rolling Sharpe ratio

```python
from chan_trading.risk.metrics import rolling_sharpe_ratio

rs = rolling_sharpe_ratio(strategy_returns, lookback=63)
```

Chan (Ch. 8) argues that a single lifetime Sharpe hides the regime
collapses a strategy is most vulnerable to. A rolling Sharpe series is
the natural upstream signal for a regime-aware sizing layer — feed it
into a `cppi_scale` or `apply_drawdown_throttle` pipeline to de-risk
before the drawdown throttle actually fires.

### 4. Information ratio

```python
from chan_trading.risk.metrics import information_ratio

ir = information_ratio(strategy_returns, benchmark_returns)
```

Standard benchmark-relative performance metric: active return divided
by tracking error, annualised. Matches Sharpe ratio exactly when the
benchmark is the zero-return portfolio.

## Quick start

```python
import pandas as pd
from chan_trading.config import MeanReversionConfig, WalkForwardConfig
from chan_trading.data.loaders import load_prices_csv
from chan_trading.validation.walkforward import run_walkforward_pair_mean_reversion

prices = load_prices_csv("data/sample_prices.csv")

mr = MeanReversionConfig(lookback=20, entry_z=2.0, exit_z=0.5)
wf = WalkForwardConfig(
    train_size=120, test_size=30, step_size=30,
    purge=3, embargo=2,              # v10: hygiene
    adf_alpha=0.05, eg_alpha=0.05,
)

report = run_walkforward_pair_mean_reversion(prices, "SPY", "IVV", mr, wf)
print(report.backtest.equity_curve.tail())
```

### Check whether a Sharpe is inflated by serial correlation (v14)

```python
from chan_trading.risk.metrics import (
    sharpe_ratio, newey_west_sharpe_tstat, newey_west_sharpe_variance,
)

sr = sharpe_ratio(strategy_returns)
t_iid = sr / (252.0 / len(strategy_returns)) ** 0.5   # naive IID t-stat
t_hac = newey_west_sharpe_tstat(strategy_returns, lags=20)

print(f"Sample Sharpe:           {sr:.3f}")
print(f"t-stat (IID assumption): {t_iid:.2f}")
print(f"t-stat (Newey-West HAC): {t_hac:.2f}")
# If t_hac is materially smaller than t_iid, positive autocorrelation
# was inflating the apparent significance of the Sharpe.
```

### Track rolling Sharpe to gate a throttle (v14)

```python
from chan_trading.risk.metrics import rolling_sharpe_ratio

rs = rolling_sharpe_ratio(strategy_returns, lookback=63)
# Use rs < 0 (or < 0.5, etc.) as a regime gate in combination with
# the drawdown throttle already in portfolio.sizing.
```

### Asymmetric execution costs (v14)

```python
from chan_trading.config import EventBacktestConfig

cfg = EventBacktestConfig(
    commission_bps=1.0,
    slippage_bps=2.0,
    half_spread_bps=5.0,   # v14
    lag_target_weights=1,
)
```

### Diversify across strategies (v12)

```python
from chan_trading.portfolio.ensemble import EnsembleConfig, combine_strategies

combined = combine_strategies(
    prices,
    {
        "pair_mr":  pair_positions,
        "momentum": momentum_positions,
        "buy_gap":  gap_positions,
    },
    config=EnsembleConfig(scheme="inverse_vol", vol_lookback=63,
                          max_gross_leverage=1.0),
)
```

### Test whether the edge is real (v12 + v13 block-shuffle)

```python
from chan_trading.validation.permutation import permutation_alpha_test

rep_bar = permutation_alpha_test(prices, positions, n_shuffles=2000, seed=0)
rep_block = permutation_alpha_test(
    prices, positions, n_shuffles=2000, seed=0, block_size=10,
)
print(f"bar-shuffle p = {rep_bar.p_value:.4f}")
print(f"block-shuffle p = {rep_block.p_value:.4f}")
```

### Stationary-bootstrap drawdown stress (v13)

```python
from chan_trading.risk.monte_carlo import simulate_max_drawdown

report = simulate_max_drawdown(
    strategy_returns, method="stationary", expected_block_size=15, n_paths=2000, seed=0,
)
print(report.max_drawdown_q05, report.max_drawdown_q95)
```

## Layout

```
src/chan_trading/
├── backtest/             # vectorised + event-driven engines
│   └── broker.py         # v14: asymmetric half_spread_bps
├── data/                 # CSV loader
├── execution/
│   ├── impact.py         # linear / sqrt market-impact models (v11)
│   └── rebalance.py
├── features/
│   ├── cointegration.py
│   ├── futures.py        # v12: continuous contracts, roll-return decomposition
│   └── statistics.py     # v13: estimate_rs_hurst_exponent, suggest_zscore_lookback
├── portfolio/
│   ├── ensemble.py       # v12: multi-strategy combiner
│   ├── kelly.py          # scalar + rolling + multivariate Kelly (v10)
│   └── sizing.py         # v13: apply_exposure_caps (shared), apply_turnover_throttle
├── risk/
│   ├── combinatorial_cv.py  # PBO via CSCV (v10)
│   ├── metrics.py           # v14: newey_west_sharpe_variance/_tstat,
│   │                        #      rolling_sharpe_ratio, information_ratio
│   ├── monte_carlo.py       # v13: stationary_bootstrap_returns
│   └── regime.py
├── strategies/
│   ├── basket_mean_reversion.py
│   ├── cross_sectional_mean_reversion.py
│   ├── cross_sectional_momentum.py
│   ├── mean_reversion.py
│   ├── momentum.py
│   └── opening_gap.py
├── validation/
│   ├── permutation.py       # v13: block-shuffle variant
│   ├── sensitivity.py       # cost sweep + breakeven (v11)
│   ├── walkforward.py       # purge + embargo (v10)
│   ├── walkforward_basket.py
│   └── walkforward_event.py
└── types.py
```

## References

1. Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and
   Their Rationale.* Wiley.
2. López de Prado, M. (2018). *Advances in Financial Machine Learning.*
   Wiley. (Ch. 7 for purge/embargo.)
3. Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio
   Efficient Frontier." *Journal of Risk*, 15(2), 3–44.
4. Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe
   Ratio: Correcting for Selection Bias, Backtest Overfitting and
   Non-Normality." *Journal of Portfolio Management*, 40(5), 94–107.
5. Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J.
   (2016). "The Probability of Backtest Overfitting." *Journal of
   Computational Finance*, 20(4), 39–69.
6. Thorp, E. O. (2006). "The Kelly Criterion in Blackjack, Sports
   Betting, and the Stock Market." In S. A. Zenios & W. T. Ziemba
   (eds.), *Handbook of Asset and Liability Management*, Vol. 1
   (pp. 385–428). Elsevier.
7. Almgren, R., Thum, C., Hauptmann, E., & Li, H. (2005). "Direct
   Estimation of Equity Market Impact." *Risk*, 18(7), 58–62.
8. Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999). "Coherent
   Measures of Risk." *Mathematical Finance*, 9(3), 203–228.
9. Maillard, S., Roncalli, T., & Teïletche, J. (2010). "The Properties
   of Equally Weighted Risk Contribution Portfolios." *Journal of
   Portfolio Management*, 36(4), 60–70.
10. White, H. (2000). "A Reality Check for Data Snooping." *Econometrica*,
    68(5), 1097–1126.
11. Phipson, B., & Smyth, G. K. (2010). "Permutation P-values Should
    Never Be Zero: Calculating Exact P-values When Permutations Are
    Randomly Drawn." *Statistical Applications in Genetics and
    Molecular Biology*, 9(1), Art. 39.
12. Sortino, F. A., & Price, L. N. (1994). "Performance Measurement in
    a Downside Risk Framework." *Journal of Investing*, 3(3), 59–64.
13. Politis, D. N., & Romano, J. P. (1994). "The Stationary Bootstrap."
    *Journal of the American Statistical Association*, 89(428),
    1303–1313.
14. Hurst, H. E. (1951). "Long-term storage capacity of reservoirs."
    *Transactions of the American Society of Civil Engineers*, 116(1),
    770–799.
15. Mandelbrot, B. B., & Wallis, J. R. (1969). "Robustness of the
    rescaled range R/S in the measurement of noncyclic long run
    statistical dependence." *Water Resources Research*, 5(5), 967–988.
16. Newey, W. K., & West, K. D. (1987). "A Simple, Positive Semi-
    Definite, Heteroskedasticity and Autocorrelation Consistent
    Covariance Matrix." *Econometrica*, 55(3), 703–708.
17. Newey, W. K., & West, K. D. (1994). "Automatic Lag Selection in
    Covariance Matrix Estimation." *Review of Economic Studies*, 61(4),
    631–653.
18. Lo, A. W. (2002). "The Statistics of Sharpe Ratios." *Financial
    Analysts Journal*, 58(4), 36–52.
19. Grinold, R. C., & Kahn, R. N. (1999). *Active Portfolio Management:
    A Quantitative Approach for Producing Superior Returns and
    Controlling Risk* (2nd ed.). McGraw-Hill. (Ch. 4 on the information
    ratio.)
