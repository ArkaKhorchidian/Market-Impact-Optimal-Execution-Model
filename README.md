# Market Impact Model: Almgren–Chriss Optimal Execution

A Python implementation of the Almgren–Chriss (2001) optimal liquidation framework, with extensions for volatility estimation, Monte Carlo execution simulation, and comparison against TWAP/VWAP benchmarks.

---

## Overview

Executing a large order without moving the market is one of the central problems of quantitative trading.  Selling too fast drives the price down (temporary impact); selling too slowly exposes you to adverse price moves (timing risk).  The AC2001 framework solves this trade-off analytically.

This repo provides:

- **Closed-form optimal trajectories** via the hyperbolic decay formula
- **Mean–variance efficient frontier** parameterised by risk aversion λ
- **Monte Carlo execution simulator** tracking implementation shortfall
- **Three volatility estimators**: close-to-close, GARCH(1,1), HAR-RV
- **Market impact models**: linear (AC2001) and square-root (Almgren et al. 2005)
- **Backtest notebook** applying the model to real equity data

---

## Mathematical Background

### Model Setup

Liquidate $X$ shares over horizon $T$, split into $N$ equal intervals of length $\tau = T/N$.

Define the **holdings trajectory**:

$$x_0 = X, \quad x_N = 0, \quad n_k = x_{k-1} - x_k \quad \text{(shares sold at step } k\text{)}$$

### Price Dynamics

$$S_k = S_{k-1} - g\!\left(\frac{n_k}{\tau}\right)\tau - \sigma\sqrt{\tau}\,\xi_k$$

where:
- $g(v) = \gamma v$ — **permanent impact**: persistent price depression proportional to trade rate
- $h(v) = \eta v$ — **temporary impact**: instantaneous execution cost, vanishes after each trade
- $\sigma$ — daily price volatility
- $\xi_k \sim \mathcal{N}(0,1)$ — i.i.d. price innovations

The execution price on trade $k$ is:

$$P_k = S_{k-1} - h(n_k/\tau)$$

### Cost Decomposition

$$E[C] = \underbrace{\frac{1}{2}\gamma X^2}_{\text{permanent (schedule-independent)}} + \underbrace{\eta \sum_{k=1}^N \frac{n_k^2}{\tau}}_{\text{temporary impact}}$$

$$\operatorname{Var}[C] = \sigma^2 \tau \sum_{k=1}^N x_k^2$$

The **mean–variance utility** to minimise:

$$U(\lambda) = E[C] + \lambda\,\operatorname{Var}[C]$$

### Optimal Trajectory Derivation

The discrete Euler–Lagrange conditions for minimising $U(\lambda)$ subject to $x_0=X$, $x_N=0$ reduce to the recurrence:

$$x_{j-1} - 2\cosh(\tilde{\kappa}\tau)\,x_j + x_{j+1} = 0$$

where $\tilde{\kappa}^2 = \lambda\sigma^2/\eta$.  The general solution is:

$$x_j^* = A\sinh(\tilde{\kappa}(N-j)\tau) + B\cosh(\tilde{\kappa}(N-j)\tau)$$

Applying boundary conditions $x_0=X$, $x_N=0$ gives the **closed-form optimal trajectory** (continuous limit):

$$\boxed{x^*(t) = X \cdot \frac{\sinh(\kappa(T-t))}{\sinh(\kappa T)}, \qquad \kappa^2 = \frac{\lambda\sigma^2}{\eta}}$$

**Limiting behaviour:**
- $\lambda \to 0$: $\kappa \to 0$, $x^* \to X(1-t/T)$ — uniform (TWAP)
- $\lambda \to \infty$: $\kappa \to \infty$ — immediate front-loading

### Efficient Frontier

As $\lambda$ varies over $[0,\infty)$, the pair $(E[C], \operatorname{Var}[C])$ traces the **Pareto-optimal frontier**: no feasible strategy can reduce expected cost without increasing variance.  TWAP lies at $\lambda=0$ — the risk-neutral extreme.

---

## Implementation

```
src/
├── almgren_chriss.py   # Closed-form x*(t), efficient frontier, cost formulas
├── market_impact.py    # Linear and √-impact models; Kyle's λ estimation
├── execution_sim.py    # Monte Carlo IS simulator; AC/TWAP/VWAP comparison
├── vol_estimator.py    # CC vol, GARCH(1,1), HAR-RV
└── utils.py            # Vol scaling, bps conversion, trajectory validation
```

### Quick Start

```python
from src.almgren_chriss import optimal_trajectory, efficient_frontier

# Optimal trajectory for λ = 1e-6
times, holdings = optimal_trajectory(
    X=1_000_000,   # shares
    T=1.0,         # 1 trading day
    N=390,         # 1-minute intervals
    sigma=0.02,    # 2% daily vol
    gamma=2.5e-7,  # permanent impact
    eta=2.5e-6,    # temporary impact
    lam=1e-6,      # risk aversion
)

# Efficient frontier
frontier = efficient_frontier(
    X=1_000_000, T=1.0, N=390, sigma=0.02,
    gamma=2.5e-7, eta=2.5e-6, n_points=100,
)
# frontier['expected_costs'], frontier['variances'], frontier['trajectories']
```

```python
from src.execution_sim import compare_strategies, slippage_summary

stats = compare_strategies(
    X=1_000_000, T=1.0, N=390, sigma=0.02,
    gamma=2.5e-7, eta=2.5e-6, lam=1e-6,
    S0=100.0, n_sims=1000, seed=42,
)
print(slippage_summary(stats))
# Strategy   Mean IS (bps)   Std IS (bps)   Sharpe IS
# AC2001             x.xx          x.xx         x.xxx
# TWAP               x.xx          x.xx         0.000
# VWAP               x.xx          x.xx         x.xxx
```

```python
from src.vol_estimator import GARCH11, HARRV, close_to_close_vol

# Fit GARCH(1,1) to price series
g = GARCH11().fit(prices)
print(f"GARCH daily vol: {g.daily_vol:.4f}")
print(f"GARCH uncond vol: {g.unconditional_vol:.4f}")

# Fit HAR-RV
import numpy as np
rv = np.diff(np.log(prices))**2
har = HARRV().fit(rv)
print(f"HAR-RV 1-day forecast vol: {har.daily_vol:.4f}")
```

---

## Results

### Efficient Frontier

The frontier curves downward-left as $\lambda$ increases from 0 (TWAP, low E[C] but high variance) toward infinity (immediate liquidation, high E[C] but zero variance).

Key observations:
- TWAP minimises E[C] but carries substantial timing risk
- The frontier is convex — diminishing returns to risk reduction
- For moderate risk aversion ($\lambda \sim 10^{-6}$), AC2001 reduces Var[C] by ~10–30% with a small E[C] penalty

### Backtest Summary

Run notebook 4 to generate results against real data:

```bash
jupyter nbconvert --to notebook --execute notebooks/04_real_data_backtest.ipynb --inplace
```

The notebook fetches 2 years of daily bars via `yfinance`, estimates σ with three estimators (CC, GARCH, HAR-RV), calibrates η and γ from the Almgren et al. (2005) scaling relation, and runs 500 paired Monte Carlo simulations per ticker. Results include mean IS (bps), std IS, and Sharpe of IS reduction vs TWAP for AAPL, MSFT, SPY, QQQ, and TSLA.

Run `jupyter nbconvert --to notebook --execute notebooks/04_real_data_backtest.ipynb` to populate results.

---

## Parameter Estimation

### Volatility

| Method | Description | Code |
|--------|-------------|------|
| Close-to-close | 21-day rolling std of log returns | `close_to_close_vol(prices)` |
| GARCH(1,1) | Conditional vol via MLE | `GARCH11().fit(prices).daily_vol` |
| HAR-RV | Long-memory model (Corsi 2009) | `HARRV().fit(rv_daily).daily_vol` |

### Market Impact

**Without execution data** (this repo's default): use the Almgren et al. (2005) cross-sectional scaling:

$$\eta \approx 0.142 \cdot \frac{\sigma}{P \cdot \text{ADV}}, \qquad \gamma \approx 0.314 \cdot \frac{\sigma}{P \cdot \text{ADV}}$$

```python
from src.market_impact import almgren_2005_params
p = almgren_2005_params(sigma_dollar=3.0, adv=5e6, price=150.0)
```

**With tick data**: regress mid-price changes on signed order flow (Kyle's λ):

```python
from src.market_impact import kyle_lambda_ols
result = kyle_lambda_ols(signed_volume, price_changes)
# result['lambda_'] is Kyle's λ in $/share
```

**Important**: these estimates are cross-sectional averages.  Actual impact varies by instrument, time of day, and broker.  Always state clearly in any report whether parameters are estimated or assumed.

---

## Usage

### Install dependencies

```bash
pip install numpy scipy pandas matplotlib seaborn statsmodels
pip install yfinance          # for real data backtest
pip install jupyter           # for notebooks
pip install pytest            # for tests
```

### Run tests

```bash
cd /path/to/Market-Impact-Optimal-Execution-Model
python -m pytest tests/ -v
```

### Run notebooks

```bash
jupyter notebook notebooks/
```

Notebooks are designed to run top-to-bottom with `Run All`.  Notebook 4 requires `yfinance`; if unavailable it falls back to synthetic GBM data.

---

## References

1. **Almgren, R. & Chriss, N. (2001).** "Optimal execution of portfolio transactions." *Journal of Risk*, 3(2), 5–39.
   — The core paper.  Derives the closed-form hyperbolic trajectory and the efficient frontier.

2. **Almgren, R. et al. (2005).** "Direct estimation of equity market impact." *Risk*, 18(7), 57–62.
   — Empirical cross-sectional estimates of η and γ for NYSE/AMEX stocks.  Source of the scaling relation used here.

3. **Corsi, F. (2009).** "A simple approximate long-memory model for realized volatility." *Journal of Financial Econometrics*, 7(2), 174–196.
   — HAR-RV model for volatility forecasting.

4. **Kyle, A.S. (1985).** "Continuous auctions and insider trading." *Econometrica*, 53(6), 1315–1335.
   — Theoretical foundation for linear price impact; motivates the permanent impact term.

5. **Engle, R.F. (1982).** "Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation." *Econometrica*, 50(4), 987–1007.
   — Original ARCH paper; GARCH(1,1) is the standard extension.
