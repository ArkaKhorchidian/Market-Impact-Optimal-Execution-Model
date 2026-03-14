"""
execution_sim.py — Monte Carlo execution simulator for the AC2001 model.

Provides:
  - :func:`simulate_execution`  — single-path P&L simulation given a holdings
    trajectory and the AC2001 price dynamics.
  - :func:`simulate_twap`       — execute a TWAP schedule.
  - :func:`simulate_vwap`       — execute a VWAP schedule given a volume profile.
  - :func:`compare_strategies`  — run N Monte Carlo paths for AC, TWAP, and VWAP
    and return cost statistics side-by-side.

Price dynamics (AC2001, eq. 2)::

    S_k = S_{k-1} - g(n_k/τ)·τ - σ·√τ·ξ_k

    Execution price on trade k:  P_k = S_{k-1} - h(n_k/τ)
    Cash proceeds:               revenue_k = n_k · P_k

Implementation shortfall (IS) measures deviation from the arrival price S_0::

    IS = X · S_0  -  Σ_k  n_k · P_k      (lower IS = better)

In basis points of initial notional:

    IS_bps = IS / (X · S_0) · 10,000
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .almgren_chriss import optimal_trajectory, twap_trajectory, trade_schedule
from .market_impact import permanent_impact_linear, temporary_impact_linear
from .utils import cost_in_bps


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """
    Full record of a single simulated execution.

    Attributes
    ----------
    times : NDArray, shape (N+1,)
        Time grid (days).
    holdings : NDArray, shape (N+1,)
        Remaining shares at each step.
    trades : NDArray, shape (N,)
        Shares sold at each step (positive = sell).
    mid_prices : NDArray, shape (N+1,)
        Mid-price process S_k after permanent impact and noise.
    exec_prices : NDArray, shape (N,)
        Execution price for each trade P_k = S_{k-1} − h(n_k/τ).
    revenue : float
        Total cash received Σ n_k · P_k.
    is_cost : float
        Implementation shortfall = X·S_0 − revenue  (dollars).
    is_bps : float
        IS expressed in basis points of arrival notional.
    arrival_price : float
        S_0, the price at the start of execution.
    """

    times: NDArray[np.float64]
    holdings: NDArray[np.float64]
    trades: NDArray[np.float64]
    mid_prices: NDArray[np.float64]
    exec_prices: NDArray[np.float64]
    revenue: float
    is_cost: float
    is_bps: float
    arrival_price: float


@dataclass
class StrategyStats:
    """
    Aggregate statistics for a strategy over many Monte Carlo simulations.

    Attributes
    ----------
    name : str
        Strategy identifier.
    mean_is_bps : float
        Mean IS cost in bps.
    std_is_bps : float
        Standard deviation of IS in bps.
    sharpe_is : float
        Sharpe ratio of IS reduction vs TWAP (computed externally if needed).
    is_bps_all : NDArray, shape (n_sims,)
        Full distribution of IS across simulations.
    mean_ec : float
        Mean expected cost E[C] (analytic).
    variance : float
        Variance of cost Var[C] (analytic).
    """

    name: str
    mean_is_bps: float
    std_is_bps: float
    sharpe_is: float = 0.0
    is_bps_all: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    mean_ec: float = 0.0
    variance: float = 0.0


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_execution(
    holdings: NDArray[np.float64],
    T: float,
    S0: float,
    sigma: float,
    gamma: float,
    eta: float,
    rng: np.random.Generator | None = None,
) -> ExecutionResult:
    """
    Simulate a single execution path under AC2001 price dynamics.

    Given a pre-computed holdings trajectory, this function:

    1. Generates a price path with permanent impact and Gaussian noise.
    2. Computes execution prices as mid minus temporary impact.
    3. Calculates implementation shortfall.

    Price dynamics::

        S_k = S_{k-1} − g(n_k/τ)·τ − σ·√τ·ξ_k
        P_k = S_{k-1} − h(n_k/τ)   (execution price, before noise settles)

    Parameters
    ----------
    holdings : NDArray[np.float64], shape (N+1,)
        Pre-computed holdings path (from :func:`~almgren_chriss.optimal_trajectory`
        or similar).  Must start at X and end at 0.
    T : float
        Liquidation horizon in days.
    S0 : float
        Arrival mid-price (dollars per share).
    sigma : float
        Daily price volatility (fractional, e.g. 0.02).  Converted to dollar
        volatility internally as ``sigma * S0``.
    gamma : float
        Permanent impact coefficient (same units as in AC2001, see
        :mod:`almgren_chriss`).
    eta : float
        Temporary impact coefficient.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.  If None, uses the
        default global generator.

    Returns
    -------
    ExecutionResult
        Full simulation record.

    Examples
    --------
    >>> import numpy as np
    >>> from almgren_chriss import optimal_trajectory
    >>> _, h = optimal_trajectory(1e6, 1.0, 390, 0.02, 2.5e-7, 2.5e-6, 1e-6)
    >>> res = simulate_execution(h, 1.0, 100.0, 0.02, 2.5e-7, 2.5e-6,
    ...                          rng=np.random.default_rng(42))
    >>> res.is_bps > 0
    True
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(holdings) - 1
    tau = T / N
    X = holdings[0]

    trades = trade_schedule(holdings)          # shape (N,)
    trade_rates = trades / tau                 # v_k = n_k / τ

    sigma_dollar = sigma * S0                  # dollar vol

    mid_prices = np.empty(N + 1)
    exec_prices = np.empty(N)
    mid_prices[0] = S0

    xi = rng.standard_normal(N)

    for k in range(N):
        perm = permanent_impact_linear(trade_rates[k], gamma) * tau
        noise = sigma_dollar * np.sqrt(tau) * xi[k]
        tmp = temporary_impact_linear(trade_rates[k], eta)

        exec_prices[k] = mid_prices[k] - tmp          # sell into temporary impact
        mid_prices[k + 1] = mid_prices[k] - perm - noise

    revenue = float(np.sum(trades * exec_prices))
    paper_value = X * S0                               # value at arrival price
    is_cost = paper_value - revenue
    is_bps = cost_in_bps(is_cost, paper_value)

    return ExecutionResult(
        times=np.linspace(0.0, T, N + 1),
        holdings=holdings.copy(),
        trades=trades,
        mid_prices=mid_prices,
        exec_prices=exec_prices,
        revenue=revenue,
        is_cost=is_cost,
        is_bps=is_bps,
        arrival_price=S0,
    )


# ---------------------------------------------------------------------------
# VWAP schedule construction
# ---------------------------------------------------------------------------


def vwap_holdings(
    X: float,
    volume_profile: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Build a VWAP holdings trajectory by participating proportionally to volume.

    The VWAP strategy sells a fraction of X at each interval proportional to
    that interval's share of total expected volume::

        n_k = X · V_k / Σ V_k,   x_j = X − Σ_{k=1}^{j} n_k

    Parameters
    ----------
    X : float
        Total position to liquidate (shares).
    volume_profile : NDArray[np.float64], shape (N,)
        Expected (or historical) volume at each interval.  Need not be
        normalised — only relative magnitudes matter.

    Returns
    -------
    NDArray[np.float64], shape (N+1,)
        Holdings path starting at X and ending at 0.
    """
    volume_profile = np.asarray(volume_profile, dtype=np.float64)
    if np.any(volume_profile < 0):
        raise ValueError("volume_profile must be non-negative")
    total = volume_profile.sum()
    if total == 0:
        raise ValueError("volume_profile sums to zero")

    fracs = volume_profile / total
    trades = X * fracs

    holdings = np.empty(len(trades) + 1)
    holdings[0] = X
    holdings[1:] = X - np.cumsum(trades)
    holdings[-1] = 0.0  # enforce exact boundary
    return holdings


def intraday_u_shape(N: int) -> NDArray[np.float64]:
    """
    Generate a stylised U-shaped intraday volume profile.

    Empirically, US equity volume follows a U-shape: high at open and close,
    low at midday.  This function returns a simple parametric approximation
    useful as a default VWAP profile.

    Parameters
    ----------
    N : int
        Number of intervals.

    Returns
    -------
    NDArray[np.float64], shape (N,)
        Non-negative volume weights, not normalised.
    """
    t = np.linspace(0.0, 1.0, N)
    # Quadratic U-shape: V(t) ∝ (t - 0.5)² + c
    profile = (t - 0.5) ** 2 + 0.04
    return profile / profile.sum()


# ---------------------------------------------------------------------------
# Multi-strategy Monte Carlo comparison
# ---------------------------------------------------------------------------


def compare_strategies(
    X: float,
    T: float,
    N: int,
    sigma: float,
    gamma: float,
    eta: float,
    lam: float,
    S0: float,
    n_sims: int = 500,
    volume_profile: NDArray[np.float64] | None = None,
    seed: int | None = None,
) -> dict[str, StrategyStats]:
    """
    Monte Carlo comparison of AC2001, TWAP, and VWAP execution strategies.

    Runs ``n_sims`` independent price-path simulations for each strategy and
    aggregates implementation shortfall statistics.

    Parameters
    ----------
    X : float
        Position to liquidate (shares).
    T : float
        Liquidation horizon (days).
    N : int
        Number of execution intervals.
    sigma : float
        Daily fractional volatility.
    gamma : float
        Permanent impact coefficient.
    eta : float
        Temporary impact coefficient.
    lam : float
        AC2001 risk-aversion parameter λ.
    S0 : float
        Arrival mid-price (dollars).
    n_sims : int, default 500
        Number of Monte Carlo simulations.
    volume_profile : NDArray, optional
        Volume profile for VWAP.  If None, uses a U-shaped profile via
        :func:`intraday_u_shape`.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict mapping strategy name → :class:`StrategyStats`
        Keys: ``"AC2001"``, ``"TWAP"``, ``"VWAP"``.

    Notes
    -----
    All three strategies face the *same* price paths in each simulation,
    isolating the effect of the execution schedule from noise realisations.
    This is the correct paired comparison methodology.
    """
    rng = np.random.default_rng(seed)

    # Pre-compute deterministic trajectories
    _, h_ac = optimal_trajectory(X, T, N, sigma, gamma, eta, lam)
    h_twap = twap_trajectory(X, N, T)

    if volume_profile is None:
        volume_profile = intraday_u_shape(N)
    h_vwap = vwap_holdings(X, volume_profile)

    strategies = {
        "AC2001": h_ac,
        "TWAP": h_twap,
        "VWAP": h_vwap,
    }

    results: dict[str, list[float]] = {name: [] for name in strategies}

    for _ in range(n_sims):
        # Each simulation shares the SAME random seed state so all strategies
        # see the same underlying price shocks (paired comparison).
        child_seed = int(rng.integers(0, 2**31))
        for name, holdings in strategies.items():
            r = simulate_execution(
                holdings, T, S0, sigma, gamma, eta,
                rng=np.random.default_rng(child_seed),
            )
            results[name].append(r.is_bps)

    stats: dict[str, StrategyStats] = {}
    for name, is_bps_list in results.items():
        arr = np.array(is_bps_list)
        stats[name] = StrategyStats(
            name=name,
            mean_is_bps=float(arr.mean()),
            std_is_bps=float(arr.std(ddof=1)),
            is_bps_all=arr,
        )

    # Sharpe of IS reduction relative to TWAP
    twap_is = results["TWAP"]
    for name in ("AC2001", "VWAP"):
        reduction = np.array(twap_is) - np.array(results[name])
        std_red = reduction.std(ddof=1)
        stats[name].sharpe_is = float(reduction.mean() / std_red) if std_red > 0 else 0.0

    return stats


# ---------------------------------------------------------------------------
# Slippage analysis helper
# ---------------------------------------------------------------------------


def slippage_summary(stats: dict[str, StrategyStats]) -> str:
    """
    Format a readable comparison table from :func:`compare_strategies` output.

    Parameters
    ----------
    stats : dict[str, StrategyStats]
        Output of :func:`compare_strategies`.

    Returns
    -------
    str
        Formatted table string.
    """
    lines = [
        f"{'Strategy':<10} {'Mean IS (bps)':>14} {'Std IS (bps)':>13} {'Sharpe IS':>10}",
        "-" * 52,
    ]
    for name, s in stats.items():
        lines.append(
            f"{name:<10} {s.mean_is_bps:>14.2f} {s.std_is_bps:>13.2f} {s.sharpe_is:>10.3f}"
        )
    return "\n".join(lines)
