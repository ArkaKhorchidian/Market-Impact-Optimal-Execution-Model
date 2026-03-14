# Data Directory

This directory stores data artifacts produced by the notebooks (figures, cached results) and documents how to obtain real tick and bar data for the backtest notebook.

Generated figures (created by `Run All` on the notebooks) are saved here as `.png` files and are gitignored by default.

---

## Suggested Data Sources

### 1. Yahoo Finance via `yfinance` (free, daily bars)

The simplest option — no API key required.

```bash
pip install yfinance
```

Notebook 4 (`04_real_data_backtest.ipynb`) fetches daily adjusted close prices automatically via `yfinance.download()`.  This gives ~2 years of daily OHLCV data, which is sufficient for estimating σ.

**Limitation**: daily bars only.  Cannot reconstruct intraday volume profiles or tick-level impact estimation.

---

### 2. Polygon.io (free tier, minute bars)

Provides 2 years of minute-bar data on the free tier, and tick data on paid plans.

1. Sign up at [polygon.io](https://polygon.io) and obtain an API key.
2. Store in environment: `export POLYGON_API_KEY=your_key_here`
3. Fetch minute bars:

```python
import requests, os

def polygon_minute_bars(ticker, date, api_key=None):
    key = api_key or os.environ['POLYGON_API_KEY']
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute"
        f"/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={key}"
    )
    resp = requests.get(url).json()
    return resp['results']  # list of {o, h, l, c, v, t}
```

Minute bars enable intraday volume profiling for VWAP and higher-frequency σ estimation.

---

### 3. Tardis.dev (institutional tick data, free sample datasets)

Tardis provides full L2 order book data (bid/ask ladders + trade tape) for crypto and US equity futures.  Free sample datasets are available.

- Website: [tardis.dev/datasets](https://tardis.dev/datasets)
- Python client: `pip install tardis-dev`

Useful for: estimating η and γ via Kyle's lambda regression on actual order flow.

---

### 4. LOBSTER (academic, L2 NASDAQ ITCH data)

Full limit-order-book reconstructions from NASDAQ ITCH for academic research.

- Website: [lobsterdata.com](https://lobsterdata.com)
- Free for academic users (requires institutional email)
- Data format: 10-level bid/ask ladders at millisecond resolution

Useful for: precise impact estimation, intraday volatility, and testing the linear vs square-root impact assumption.

---

## Parameter Estimation Notes

### Volatility (σ)

| Estimator | Data needed | Notes |
|-----------|-------------|-------|
| Close-to-close | Daily closes | Simple, unbiased, but noisy |
| GARCH(1,1) | Daily closes (≥ 60 obs) | Captures vol clustering |
| HAR-RV | Squared returns (≥ 30 obs) | Long-memory, Corsi (2009) |
| Parkinson | Daily high–low | More efficient than CC vol |

All estimators are in `src/vol_estimator.py`.

### Impact (η, γ)

**Without proprietary execution data** (this repo's default):
Use the Almgren et al. (2005) scaling relation via `src.market_impact.almgren_2005_params()`:

```python
from src.market_impact import almgren_2005_params
params = almgren_2005_params(sigma_dollar=3.0, adv=5e6, price=150.0)
# Returns {'eta': ..., 'gamma': ...}
```

**With tick data** (advanced):
Regress mid-price changes on signed order flow using `src.market_impact.kyle_lambda_ols()`.  Sign trades with the Lee–Ready algorithm (trade at ask = buy, at bid = sell).

Always document which approach was used and its limitations.

---

## Gitignore

The following patterns are excluded from version control (add to `.gitignore`):

```
data/*.csv
data/*.parquet
data/*.h5
data/*.pkl
data/*.png
data/*.pdf
!data/README.md
```
