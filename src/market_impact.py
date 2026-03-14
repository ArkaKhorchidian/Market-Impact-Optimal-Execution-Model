"""
market_impact.py — Market impact functions for the Almgren–Chriss framework.

Implements permanent and temporary impact models at two levels of complexity:

1. **Linear model** (AC2001) — analytically tractable, closed-form frontier.
2. **Square-root model** (Almgren et al. 2005) — empirically better calibrated
   for large trades; loses closed-form but retains convexity.

Also provides Kyle's λ estimation from trade-and-quote data when available.

References
----------
- Almgren, R. & Chriss, N. (2001). "Optimal execution of portfolio transactions."
  Journal of Risk, 3(2), 5–39.
- Almgren, R. et al. (2005). "Direct estimation of equity market impact."
  Risk, 18(7), 57–62.
- Kyle, A.S. (1985). "Continuous auctions and insider trading."
  Econometrica, 53(6), 1315–1335.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Empirical parameter ranges (Almgren et al. 2005, Table 1)
# ---------------------------------------------------------------------------

# These are rough order-of-magnitude benchmarks for liquid US large-cap equities.
# Units: price_impact_per_(shares/day) expressed as a fraction of price.
# η ≈ 0.142 · σ / (ADV · price)  (daily volatility / average daily volume)
# γ ≈ 0.314 · σ / (ADV · price)

ALMGREN_2005_ETA_COEFFICIENT: float = 0.142
"""Empirical coefficient for temporary impact from Almgren et al. (2005)."""

ALMGREN_2005_GAMMA_COEFFICIENT: float = 0.314
"""Empirical coefficient for permanent impact from Almgren et al. (2005)."""


def almgren_2005_params(
    sigma: float,
    adv: float,
    price: float,
) -> dict[str, float]:
    """
    Estimate η and γ from Almgren et al. (2005) empirical scaling relations.

    The paper provides cross-sectional regressions on NYSE/AMEX stocks::

        η ≈ c_η · σ / (ADV · price)
        γ ≈ c_γ · σ / (ADV · price)

    where c_η ≈ 0.142 and c_γ ≈ 0.314.

    **Important**: these are approximate, cross-sectional averages.  They
    should be treated as order-of-magnitude estimates, not precise calibrations,
    unless you have proprietary execution data for the specific instrument.

    Parameters
    ----------
    sigma : float
        Daily price volatility in dollars (i.e. price × fractional_vol).
    adv : float
        Average daily volume in shares.
    price : float
        Current mid-price in dollars per share.

    Returns
    -------
    dict with keys ``eta`` and ``gamma``.

    Examples
    --------
    >>> p = almgren_2005_params(sigma=0.02 * 150, adv=5e6, price=150.0)
    >>> round(p["eta"], 10)  # doctest: +SKIP
    """
    if adv <= 0:
        raise ValueError(f"adv must be positive, got {adv}")
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    base = sigma / (adv * price)
    return {
        "eta": ALMGREN_2005_ETA_COEFFICIENT * base,
        "gamma": ALMGREN_2005_GAMMA_COEFFICIENT * base,
    }


# ---------------------------------------------------------------------------
# Linear impact model (AC2001)
# ---------------------------------------------------------------------------


def permanent_impact_linear(trade_rate: float | NDArray, gamma: float) -> float | NDArray:
    """
    Permanent price impact under the linear model.

    .. math::

        g(v) = \\gamma \\cdot v

    where v = n_k / τ is the trade rate (shares per day).

    The *cumulative* price depression after trading at rate v for time dt is
    g(v) · dt = γ · n_k, independent of timing — a permanent shift in the
    fundamental price process.

    Parameters
    ----------
    trade_rate : float or NDArray
        Trade rate v = n_k / τ  (shares / day).  Positive = sell.
    gamma : float
        Permanent impact coefficient γ ($/share per share/day = $/day).

    Returns
    -------
    float or NDArray
        Price impact per unit time g(v)  (dollars / day).
    """
    return gamma * trade_rate


def temporary_impact_linear(trade_rate: float | NDArray, eta: float) -> float | NDArray:
    """
    Temporary price impact under the linear model.

    .. math::

        h(v) = \\eta \\cdot v

    The temporary impact h(v) is the additional cost (above mid) paid on
    a trade of size n_k executed over interval τ.  It vanishes at the end of
    the interval (hence "temporary").

    Parameters
    ----------
    trade_rate : float or NDArray
        Trade rate v = n_k / τ  (shares / day).
    eta : float
        Temporary impact coefficient η ($/share per share/day = $/day).

    Returns
    -------
    float or NDArray
        Execution cost per unit time h(v)  (dollars / day).
    """
    return eta * trade_rate


def execution_shortfall_linear(
    n_k: float | NDArray,
    tau: float,
    gamma: float,
    eta: float,
) -> float | NDArray:
    """
    Implementation shortfall cost for a single trade under the linear model.

    .. math::

        \\text{IS}_k = h(n_k/\\tau) \\cdot n_k = \\eta \\cdot n_k^2 / \\tau

    This is the dollar cost attributed to temporary impact for trade n_k.
    Summing over all trades recovers the temporary-impact component of E[C].

    Parameters
    ----------
    n_k : float or NDArray
        Trade size in shares (positive = sell).
    tau : float
        Interval length in days.
    gamma : float
        Permanent impact coefficient (not used in IS per trade but kept for
        interface consistency).
    eta : float
        Temporary impact coefficient.

    Returns
    -------
    float or NDArray
        Dollar implementation shortfall for this trade.
    """
    v = n_k / tau
    return eta * v * n_k  # = eta * n_k^2 / tau


# ---------------------------------------------------------------------------
# Square-root impact model (Almgren et al. 2005)
# ---------------------------------------------------------------------------


def temporary_impact_sqrt(
    trade_rate: float | NDArray,
    eta: float,
    sigma: float,
    adv: float,
) -> float | NDArray:
    """
    Temporary impact under the square-root model (Almgren et al. 2005).

    The empirical finding is that the *price* impact of a trade scales as the
    *square root* of participation rate, not linearly::

        h(v) = η · σ · √(v / ADV)

    This is sometimes written in the "three-fifths" form as well (Almgren 2005
    gives a 3/5 power for the instantaneous model), but the square-root version
    is the most commonly used approximation for block execution.

    .. math::

        h(v) = \\eta \\cdot \\sigma \\cdot \\sqrt{\\frac{v}{\\mathrm{ADV}}}

    Parameters
    ----------
    trade_rate : float or NDArray
        Trade rate v  (shares / day).  Must be ≥ 0.
    eta : float
        Dimensionless temporary impact coefficient (from regression).
    sigma : float
        Daily price volatility in dollars.
    adv : float
        Average daily volume in shares.

    Returns
    -------
    float or NDArray
        Temporary impact cost per share h(v)  (dollars / share).

    Notes
    -----
    The linear model is appropriate for  v ≪ ADV.  For larger participation
    rates the square-root model is preferred as it avoids infinite impact in
    the large-trade limit.
    """
    if adv <= 0:
        raise ValueError(f"adv must be positive, got {adv}")
    trade_rate = np.asarray(trade_rate, dtype=float)
    return eta * sigma * np.sqrt(np.abs(trade_rate) / adv)


def permanent_impact_sqrt(
    trade_rate: float | NDArray,
    gamma: float,
    sigma: float,
    adv: float,
) -> float | NDArray:
    """
    Permanent impact under the square-root model.

    .. math::

        g(v) = \\gamma \\cdot \\sigma \\cdot \\sqrt{|v| / \\mathrm{ADV}}

    Parameters
    ----------
    trade_rate : float or NDArray
        Trade rate v  (shares / day).
    gamma : float
        Permanent impact coefficient (dimensionless).
    sigma : float
        Daily price volatility in dollars.
    adv : float
        Average daily volume in shares.

    Returns
    -------
    float or NDArray
        Permanent impact per unit time  (dollars / share / day).
    """
    if adv <= 0:
        raise ValueError(f"adv must be positive, got {adv}")
    trade_rate = np.asarray(trade_rate, dtype=float)
    return gamma * sigma * np.sqrt(np.abs(trade_rate) / adv)


# ---------------------------------------------------------------------------
# Kyle's lambda (linear price impact from order flow)
# ---------------------------------------------------------------------------


def kyle_lambda_ols(
    signed_volume: NDArray[np.float64],
    price_changes: NDArray[np.float64],
) -> dict[str, float]:
    """
    Estimate Kyle's λ via OLS regression of price changes on signed order flow.

    Kyle (1985) shows that in a rational-expectations equilibrium the price
    change per unit of net order flow is::

        ΔS_t = λ · Q_t + ε_t

    where Q_t is signed order flow (positive = buy pressure) and λ is the
    *price impact per share*, also known as Kyle's lambda.

    This function fits the regression by OLS: λ̂ = Cov(ΔS, Q) / Var(Q).

    Parameters
    ----------
    signed_volume : NDArray[np.float64], shape (T,)
        Net signed order flow at each time step (shares).
        Positive = net buying; negative = net selling.
    price_changes : NDArray[np.float64], shape (T,)
        Mid-price changes ΔS_t (dollars).

    Returns
    -------
    dict with keys:
        ``lambda_`` : float
            Estimated Kyle's λ (dollars per share).
        ``r_squared`` : float
            OLS R² — indicates how much of price variation is explained by flow.
        ``intercept`` : float
            OLS intercept (should be near zero if market is informationally efficient).

    Notes
    -----
    This estimate requires reliable signed-volume data (e.g. Lee–Ready tick
    test applied to TAQ data, or direct broker-level flow).  Results will be
    noisy with inferred trade directions.
    """
    Q = np.asarray(signed_volume, dtype=np.float64)
    dS = np.asarray(price_changes, dtype=np.float64)

    if len(Q) != len(dS):
        raise ValueError("signed_volume and price_changes must have the same length")
    if len(Q) < 2:
        raise ValueError("Need at least 2 observations for OLS")

    # OLS: [intercept, lambda] via normal equations
    A = np.column_stack([np.ones(len(Q)), Q])
    coeffs, _, _, _ = np.linalg.lstsq(A, dS, rcond=None)
    intercept, lam = float(coeffs[0]), float(coeffs[1])

    # R²
    dS_hat = intercept + lam * Q
    ss_res = np.sum((dS - dS_hat) ** 2)
    ss_tot = np.sum((dS - dS.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"lambda_": lam, "r_squared": r2, "intercept": intercept}


# ---------------------------------------------------------------------------
# Impact model comparison utility
# ---------------------------------------------------------------------------


def compare_impact_models(
    trade_rates: NDArray[np.float64],
    gamma_linear: float,
    eta_linear: float,
    gamma_sqrt: float,
    eta_sqrt: float,
    sigma: float,
    adv: float,
) -> dict[str, NDArray[np.float64]]:
    """
    Evaluate both linear and square-root temporary impact across a range of
    trade rates, for visual comparison.

    Parameters
    ----------
    trade_rates : NDArray[np.float64]
        Grid of trade rates v = n/τ  (shares / day).
    gamma_linear, eta_linear : float
        Parameters for the linear model.
    gamma_sqrt, eta_sqrt : float
        Parameters for the square-root model.
    sigma : float
        Daily price volatility in dollars.
    adv : float
        Average daily volume in shares.

    Returns
    -------
    dict with keys:
        ``trade_rates`` : NDArray — input grid
        ``h_linear``    : NDArray — linear temporary impact
        ``h_sqrt``      : NDArray — square-root temporary impact
        ``g_linear``    : NDArray — linear permanent impact
        ``g_sqrt``      : NDArray — square-root permanent impact
    """
    return {
        "trade_rates": trade_rates,
        "h_linear": temporary_impact_linear(trade_rates, eta_linear),
        "h_sqrt": temporary_impact_sqrt(trade_rates, eta_sqrt, sigma, adv),
        "g_linear": permanent_impact_linear(trade_rates, gamma_linear),
        "g_sqrt": permanent_impact_sqrt(trade_rates, gamma_sqrt, sigma, adv),
    }
