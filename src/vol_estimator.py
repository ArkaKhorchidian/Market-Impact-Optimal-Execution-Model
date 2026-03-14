"""
vol_estimator.py — Volatility estimators for the Almgren–Chriss model.

Provides three σ estimators of increasing sophistication:

1. **Close-to-close realized volatility** — simple, unbiased baseline.
2. **GARCH(1,1)** — captures volatility clustering via MLE.
3. **HAR-RV** (Corsi 2009) — uses daily/weekly/monthly realized variance
   components to model long-memory behaviour of volatility.

All estimators return a daily σ in price-return units (not dollar units);
multiply by current price to get dollar volatility.

References
----------
- Corsi, F. (2009). "A simple approximate long-memory model for realized
  volatility." Journal of Financial Econometrics, 7(2), 174–196.
- Engle, R.F. (1982). "Autoregressive conditional heteroscedasticity with
  estimates of the variance of United Kingdom inflation."
  Econometrica, 50(4), 987–1007.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .utils import log_returns


# ---------------------------------------------------------------------------
# 1. Close-to-close realized volatility
# ---------------------------------------------------------------------------


def close_to_close_vol(
    prices: NDArray[np.float64],
    window: int | None = None,
    ddof: int = 1,
) -> float | NDArray[np.float64]:
    """
    Estimate daily volatility from close-to-close log returns.

    .. math::

        \\hat{\\sigma} = \\sqrt{\\frac{1}{T-1} \\sum_{t=1}^{T} (r_t - \\bar{r})^2}

    where  r_t = ln(S_t / S_{t-1}).

    Parameters
    ----------
    prices : NDArray[np.float64], shape (T,)
        Daily closing prices.
    window : int or None, default None
        Rolling window length.  If None, a single scalar is returned using
        all observations.  If an int, a rolling estimate is returned.
    ddof : int, default 1
        Delta degrees of freedom for the standard deviation.

    Returns
    -------
    float
        Scalar daily σ when ``window`` is None.
    NDArray[np.float64], shape (T - window,)
        Rolling σ estimates when ``window`` is provided.  The first valid
        estimate uses observations 0..window (inclusive).
    """
    r = log_returns(prices)  # shape (T-1,)

    if window is None:
        return float(np.std(r, ddof=ddof))

    if window < 2:
        raise ValueError(f"window must be ≥ 2, got {window}")

    n = len(r)
    if window > n:
        raise ValueError(f"window ({window}) exceeds number of returns ({n})")

    estimates = np.empty(n - window + 1)
    for i in range(len(estimates)):
        estimates[i] = np.std(r[i : i + window], ddof=ddof)
    return estimates


def parkinson_vol(
    highs: NDArray[np.float64],
    lows: NDArray[np.float64],
    window: int | None = None,
) -> float | NDArray[np.float64]:
    """
    Parkinson (1980) range-based volatility estimator.

    Uses the high-low range, which is more efficient than close-to-close::

        σ² ≈ (1 / (4 ln 2)) · E[(ln(H/L))²]

    .. math::

        \\hat{\\sigma}^2 = \\frac{1}{4 \\ln 2}
            \\cdot \\frac{1}{T} \\sum_{t=1}^{T} \\left(\\ln \\frac{H_t}{L_t}\\right)^2

    Parameters
    ----------
    highs, lows : NDArray[np.float64], shape (T,)
        Daily high and low prices.
    window : int or None
        Rolling window; None returns a single scalar.

    Returns
    -------
    float or NDArray[np.float64]
        Daily volatility estimate(s).
    """
    if np.any(lows <= 0):
        raise ValueError("lows must be strictly positive")
    hl = np.log(highs / lows) ** 2

    factor = 1.0 / (4.0 * np.log(2.0))

    if window is None:
        return float(np.sqrt(factor * np.mean(hl)))

    n = len(hl)
    estimates = np.empty(n - window + 1)
    for i in range(len(estimates)):
        estimates[i] = np.sqrt(factor * np.mean(hl[i : i + window]))
    return estimates


# ---------------------------------------------------------------------------
# 2. GARCH(1,1)
# ---------------------------------------------------------------------------


class GARCH11:
    """
    GARCH(1,1) volatility model fit by maximum likelihood.

    Conditional variance process::

        σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}

    where  ε_t = r_t − μ  are demeaned returns.

    Stationarity requires  α + β < 1.

    Attributes
    ----------
    omega_, alpha_, beta_, mu_ : float
        Fitted parameters (available after calling :meth:`fit`).
    conditional_variances_ : NDArray
        In-sample conditional variance series  σ²_1, …, σ²_T.
    log_likelihood_ : float
        Value of the log-likelihood at the optimum.
    """

    def __init__(self) -> None:
        self.omega_: float | None = None
        self.alpha_: float | None = None
        self.beta_: float | None = None
        self.mu_: float | None = None
        self.conditional_variances_: NDArray[np.float64] | None = None
        self.log_likelihood_: float | None = None

    # ------------------------------------------------------------------
    def _neg_log_likelihood(self, params: NDArray, returns: NDArray) -> float:
        """Negative Gaussian log-likelihood for GARCH(1,1)."""
        mu, omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10

        eps = returns - mu
        T = len(eps)
        sigma2 = np.empty(T)
        sigma2[0] = np.var(eps)  # initialise at unconditional variance

        for t in range(1, T):
            sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]

        # Guard against numerical issues
        sigma2 = np.maximum(sigma2, 1e-16)
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + eps**2 / sigma2)
        return -ll

    # ------------------------------------------------------------------
    def fit(
        self,
        prices: NDArray[np.float64],
        method: str = "L-BFGS-B",
    ) -> "GARCH11":
        """
        Fit GARCH(1,1) to a price series by MLE.

        Parameters
        ----------
        prices : NDArray[np.float64], shape (T,)
            Daily closing prices.
        method : str, default 'L-BFGS-B'
            Scipy optimisation method.

        Returns
        -------
        self
            Fitted estimator (allows chaining).
        """
        r = log_returns(prices)
        mu0 = float(np.mean(r))
        var0 = float(np.var(r))

        # Starting values: GARCH literature consensus heuristics
        x0 = np.array([mu0, var0 * 0.1, 0.1, 0.8])
        bounds = [
            (None, None),   # mu: unconstrained
            (1e-8, None),   # omega > 0
            (1e-6, 0.999),  # alpha in (0, 1)
            (1e-6, 0.999),  # beta  in (0, 1)
        ]

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(r,),
            method=method,
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12},
        )

        if not result.success:
            warnings.warn(
                f"GARCH optimisation did not converge: {result.message}",
                RuntimeWarning,
                stacklevel=2,
            )

        self.mu_, self.omega_, self.alpha_, self.beta_ = result.x
        self.log_likelihood_ = -result.fun

        # Compute conditional variances in-sample
        eps = r - self.mu_
        T = len(eps)
        sigma2 = np.empty(T)
        sigma2[0] = np.var(eps)
        for t in range(1, T):
            sigma2[t] = (
                self.omega_
                + self.alpha_ * eps[t - 1] ** 2
                + self.beta_ * sigma2[t - 1]
            )
        self.conditional_variances_ = sigma2
        return self

    # ------------------------------------------------------------------
    def forecast(self, h: int = 1) -> NDArray[np.float64]:
        """
        Multi-step-ahead variance forecast.

        For h > 1::

            E[σ²_{T+h}] = ω/(1−α−β) + (α+β)^{h-1} · (σ²_T − ω/(1−α−β))

        Parameters
        ----------
        h : int, default 1
            Forecast horizon in days.

        Returns
        -------
        NDArray[np.float64], shape (h,)
            Forecasted conditional variances for days T+1, …, T+h.
        """
        if self.conditional_variances_ is None:
            raise RuntimeError("Call fit() before forecast()")
        omega, alpha, beta = self.omega_, self.alpha_, self.beta_
        ab = alpha + beta
        uncond_var = omega / (1.0 - ab) if ab < 1.0 else float(np.mean(self.conditional_variances_))

        sigma2_T = self.conditional_variances_[-1]
        forecasts = np.empty(h)
        s2 = sigma2_T
        for i in range(h):
            if i == 0:
                s2 = omega + (alpha + beta) * sigma2_T
            else:
                s2 = uncond_var + ab ** i * (sigma2_T - uncond_var)
            forecasts[i] = s2
        return forecasts

    # ------------------------------------------------------------------
    @property
    def daily_vol(self) -> float:
        """Current (last in-sample) conditional daily volatility σ_T."""
        if self.conditional_variances_ is None:
            raise RuntimeError("Call fit() before accessing daily_vol")
        return float(np.sqrt(self.conditional_variances_[-1]))

    @property
    def unconditional_vol(self) -> float:
        """Unconditional (long-run) daily volatility √(ω/(1−α−β))."""
        if self.alpha_ is None:
            raise RuntimeError("Call fit() before accessing unconditional_vol")
        ab = self.alpha_ + self.beta_
        if ab >= 1.0:
            warnings.warn("GARCH process is non-stationary (α+β ≥ 1)", RuntimeWarning)
            return float(np.sqrt(np.mean(self.conditional_variances_)))
        return float(np.sqrt(self.omega_ / (1.0 - ab)))

    def __repr__(self) -> str:  # pragma: no cover
        if self.omega_ is None:
            return "GARCH11(unfitted)"
        return (
            f"GARCH11(ω={self.omega_:.3e}, α={self.alpha_:.4f}, "
            f"β={self.beta_:.4f}, μ={self.mu_:.4e})"
        )


# ---------------------------------------------------------------------------
# 3. HAR-RV (Corsi 2009)
# ---------------------------------------------------------------------------


class HARRV:
    """
    Heterogeneous Autoregressive model for Realized Variance (HAR-RV).

    Corsi (2009) shows that realized variance has approximate long-memory
    that can be captured parsimoniously by three aggregation levels::

        RV_d+1 = α + β_d·RV_d + β_w·RV_w + β_m·RV_m + ε_{d+1}

    where::

        RV_d = daily realized variance (1-day)
        RV_w = (1/5) · Σ_{i=0}^{4}  RV_{d-i}   (weekly average)
        RV_m = (1/22) · Σ_{i=0}^{21} RV_{d-i}  (monthly average)

    The model is estimated by OLS.

    Attributes
    ----------
    alpha_, beta_d_, beta_w_, beta_m_ : float
        Fitted coefficients after calling :meth:`fit`.
    r_squared_ : float
        In-sample R² of the OLS regression.
    """

    LAG_DAILY: int = 1
    LAG_WEEKLY: int = 5
    LAG_MONTHLY: int = 22

    def __init__(self) -> None:
        self.alpha_: float | None = None
        self.beta_d_: float | None = None
        self.beta_w_: float | None = None
        self.beta_m_: float | None = None
        self.r_squared_: float | None = None
        self._fitted_rv: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    @staticmethod
    def _rolling_mean(rv: NDArray, lag: int) -> NDArray:
        """Rolling backward mean of ``rv`` over ``lag`` periods."""
        out = np.empty(len(rv))
        out[:] = np.nan
        for t in range(lag - 1, len(rv)):
            out[t] = np.mean(rv[t - lag + 1 : t + 1])
        return out

    # ------------------------------------------------------------------
    def fit(self, realized_variances: NDArray[np.float64]) -> "HARRV":
        """
        Fit HAR-RV by OLS.

        Parameters
        ----------
        realized_variances : NDArray[np.float64], shape (T,)
            Daily realized variance series RV_1, …, RV_T.
            Must have T ≥ 23 (to form the monthly predictor).

        Returns
        -------
        self
        """
        rv = np.asarray(realized_variances, dtype=np.float64)
        if len(rv) < self.LAG_MONTHLY + 2:
            raise ValueError(
                f"Need at least {self.LAG_MONTHLY + 2} observations; got {len(rv)}"
            )

        rv_d = rv  # daily
        rv_w = self._rolling_mean(rv, self.LAG_WEEKLY)
        rv_m = self._rolling_mean(rv, self.LAG_MONTHLY)

        # First valid index where all three predictors are available
        start = self.LAG_MONTHLY - 1

        # Predictors at time t (predicting rv[t+1])
        y = rv[start + 1 :]
        X = np.column_stack([
            np.ones(len(y)),
            rv_d[start : start + len(y)],
            rv_w[start : start + len(y)],
            rv_m[start : start + len(y)],
        ])

        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self.alpha_, self.beta_d_, self.beta_w_, self.beta_m_ = coeffs

        y_hat = X @ coeffs
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        self.r_squared_ = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        self._fitted_rv = rv
        return self

    # ------------------------------------------------------------------
    def forecast(self, h: int = 1) -> NDArray[np.float64]:
        """
        Iterative multi-step-ahead forecast of realized variance.

        For each step the model is applied recursively using the updated
        rolling averages.

        Parameters
        ----------
        h : int, default 1
            Forecast horizon in days.

        Returns
        -------
        NDArray[np.float64], shape (h,)
            Forecasted daily realized variances.
        """
        if self._fitted_rv is None:
            raise RuntimeError("Call fit() before forecast()")

        rv_hist = list(self._fitted_rv.copy())
        forecasts = []

        for _ in range(h):
            rv_d = rv_hist[-1]
            rv_w = np.mean(rv_hist[-self.LAG_WEEKLY :])
            rv_m = np.mean(rv_hist[-self.LAG_MONTHLY :])
            rv_next = (
                self.alpha_
                + self.beta_d_ * rv_d
                + self.beta_w_ * rv_w
                + self.beta_m_ * rv_m
            )
            rv_next = max(rv_next, 0.0)  # variance must be non-negative
            forecasts.append(rv_next)
            rv_hist.append(rv_next)

        return np.array(forecasts)

    # ------------------------------------------------------------------
    @property
    def daily_vol(self) -> float:
        """1-day-ahead volatility forecast (√RV)."""
        return float(np.sqrt(self.forecast(h=1)[0]))

    def __repr__(self) -> str:  # pragma: no cover
        if self.alpha_ is None:
            return "HARRV(unfitted)"
        return (
            f"HARRV(α={self.alpha_:.3e}, β_d={self.beta_d_:.4f}, "
            f"β_w={self.beta_w_:.4f}, β_m={self.beta_m_:.4f}, "
            f"R²={self.r_squared_:.4f})"
        )


# ---------------------------------------------------------------------------
# Utility: compare all three estimators
# ---------------------------------------------------------------------------


def compare_vol_estimators(
    prices: NDArray[np.float64],
    window_cc: int = 21,
) -> dict[str, float]:
    """
    Fit all three volatility estimators and return a summary.

    Parameters
    ----------
    prices : NDArray[np.float64], shape (T,)
        Daily closing prices.  T should be ≥ 50 for reliable GARCH/HAR fits.
    window_cc : int, default 21
        Rolling window for the close-to-close estimator.

    Returns
    -------
    dict with keys:
        ``close_to_close`` : float — rolling-window std of log returns
        ``garch``          : float — GARCH(1,1) conditional daily vol (last in-sample)
        ``har_rv``         : float — HAR-RV 1-step-ahead forecast vol
        ``garch_unconditional`` : float — GARCH long-run vol
    """
    r = log_returns(prices)
    rv_daily = r**2  # simple proxy for realized variance (≡ squared returns)

    # 1. Close-to-close
    cc = float(np.std(r[-window_cc:], ddof=1))

    # 2. GARCH
    garch = GARCH11().fit(prices)

    # 3. HAR-RV
    har = HARRV().fit(rv_daily)

    return {
        "close_to_close": cc,
        "garch": garch.daily_vol,
        "garch_unconditional": garch.unconditional_vol,
        "har_rv": har.daily_vol,
    }
