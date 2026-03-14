"""
utils.py — Shared helpers for the Almgren–Chriss execution model.

Covers:
  - Annualisation / de-annualisation of volatility
  - Conversion between shares, notional, and basis-point costs
  - Trajectory validation
  - Logging setup for reproducible research output
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR: int = 252
"""Standard convention for number of trading days per year."""

MINUTES_PER_TRADING_DAY: int = 390
"""NYSE / NASDAQ regular session length in minutes."""


# ---------------------------------------------------------------------------
# Volatility scaling
# ---------------------------------------------------------------------------


def annualize_vol(daily_vol: float, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Scale daily volatility to an annualised figure.

    .. math::

        \\sigma_{\\text{annual}} = \\sigma_{\\text{daily}} \\cdot \\sqrt{T_{\\text{year}}}

    Parameters
    ----------
    daily_vol : float
        Volatility expressed per trading day (fractional, e.g. 0.02 for 2 %/day).
    periods_per_year : int, default 252
        Number of trading periods in a year.

    Returns
    -------
    float
        Annualised volatility.
    """
    return daily_vol * np.sqrt(periods_per_year)


def daily_vol_from_annual(annual_vol: float, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    De-annualise a volatility figure to a per-day basis.

    .. math::

        \\sigma_{\\text{daily}} = \\frac{\\sigma_{\\text{annual}}}{\\sqrt{T_{\\text{year}}}}

    Parameters
    ----------
    annual_vol : float
        Annualised volatility (fractional).
    periods_per_year : int, default 252
        Number of trading periods in a year.

    Returns
    -------
    float
        Daily volatility.
    """
    return annual_vol / np.sqrt(periods_per_year)


def scale_vol(vol: float, from_period: float, to_period: float) -> float:
    """
    Re-scale volatility between arbitrary period lengths using the √T rule.

    .. math::

        \\sigma_{t_2} = \\sigma_{t_1} \\cdot \\sqrt{t_2 / t_1}

    Parameters
    ----------
    vol : float
        Volatility expressed per ``from_period``.
    from_period : float
        Source period length (arbitrary units, e.g. days).
    to_period : float
        Target period length (same units).

    Returns
    -------
    float
        Volatility per ``to_period``.
    """
    if from_period <= 0 or to_period <= 0:
        raise ValueError("Period lengths must be positive.")
    return vol * np.sqrt(to_period / from_period)


# ---------------------------------------------------------------------------
# Cost / notional conversions
# ---------------------------------------------------------------------------


def shares_to_notional(shares: float, price: float) -> float:
    """Return the notional dollar value of a share position.

    Parameters
    ----------
    shares : float
        Number of shares.
    price : float
        Price per share (dollars).

    Returns
    -------
    float
        Notional value in dollars.
    """
    return shares * price


def cost_in_bps(cost: float, notional: float) -> float:
    """
    Express an execution cost as basis points of notional.

    .. math::

        \\text{bps} = \\frac{\\text{cost}}{\\text{notional}} \\times 10{,}000

    Parameters
    ----------
    cost : float
        Execution cost in dollars (or any currency).
    notional : float
        Trade notional in the same currency.

    Returns
    -------
    float
        Cost in basis points (bps).
    """
    if notional == 0.0:
        raise ValueError("notional must be non-zero")
    return (cost / notional) * 10_000.0


def bps_to_cost(bps: float, notional: float) -> float:
    """
    Convert a basis-point figure back to an absolute cost.

    Parameters
    ----------
    bps : float
        Cost in basis points.
    notional : float
        Trade notional.

    Returns
    -------
    float
        Absolute cost.
    """
    return bps * notional / 10_000.0


# ---------------------------------------------------------------------------
# Trajectory validation
# ---------------------------------------------------------------------------


def validate_trajectory(
    holdings: NDArray[np.float64],
    X: float,
    tol: float = 1e-6,
) -> None:
    """
    Assert that a holdings trajectory is well-formed.

    Checks:
      1. ``holdings[0] ≈ X`` (starts at full position)
      2. ``holdings[-1] ≈ 0`` (fully liquidated)
      3. All holdings are non-negative (no overselling)
      4. Holdings are weakly decreasing (no buying back)

    Parameters
    ----------
    holdings : NDArray[np.float64]
        Holdings path x_0, x_1, …, x_N.
    X : float
        Expected initial position size.
    tol : float, default 1e-6
        Absolute tolerance for boundary checks.

    Raises
    ------
    ValueError
        If any check fails.
    """
    if abs(holdings[0] - X) > tol * max(1.0, abs(X)):
        raise ValueError(
            f"holdings[0] = {holdings[0]:.6g} but expected X = {X:.6g}"
        )
    if abs(holdings[-1]) > tol * max(1.0, abs(X)):
        raise ValueError(
            f"holdings[-1] = {holdings[-1]:.6g} should be ≈ 0"
        )
    if np.any(holdings < -tol * abs(X)):
        raise ValueError("holdings contains negative values (overselling)")
    diffs = np.diff(holdings)
    if np.any(diffs > tol * abs(X)):
        raise ValueError("holdings is not weakly decreasing (buy-back detected)")


def compute_trades(holdings: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Derive signed trade sizes from a holdings path.

    Positive values mean *sell*.

    .. math::

        n_k = x_{k-1} - x_k, \\quad k = 1, \\ldots, N

    Parameters
    ----------
    holdings : NDArray[np.float64], shape (N+1,)

    Returns
    -------
    NDArray[np.float64], shape (N,)
    """
    return -np.diff(holdings)


# ---------------------------------------------------------------------------
# Returns / price series helpers
# ---------------------------------------------------------------------------


def log_returns(prices: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute log returns from a price series.

    .. math::

        r_t = \\ln(S_t / S_{t-1})

    Parameters
    ----------
    prices : NDArray[np.float64], shape (T,)

    Returns
    -------
    NDArray[np.float64], shape (T-1,)
    """
    return np.diff(np.log(prices))


def simple_returns(prices: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute simple (arithmetic) returns from a price series.

    .. math::

        r_t = S_t / S_{t-1} - 1

    Parameters
    ----------
    prices : NDArray[np.float64], shape (T,)

    Returns
    -------
    NDArray[np.float64], shape (T-1,)
    """
    return np.diff(prices) / prices[:-1]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a consistently-formatted logger for a module.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__``.
    level : int, default logging.INFO
        Log level.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
