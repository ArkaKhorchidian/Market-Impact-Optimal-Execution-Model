"""
almgren_chriss.py — Core Almgren–Chriss (2001) optimal execution model.

Implements the closed-form solution for optimal liquidation of a block position
under a linear market impact model, following:

    Almgren, R. & Chriss, N. (2001). "Optimal execution of portfolio transactions."
    Journal of Risk, 3(2), 5–39.

Model overview
--------------
Liquidate X shares over horizon T, split into N intervals of length τ = T/N.
Holdings trajectory:  x_0 = X,  x_N = 0,  trades  n_k = x_{k-1} − x_k.

Price dynamics::

    S_k = S_{k-1} − g(n_k/τ)·τ − σ·√τ·ξ_k

where:
  - g(v) = γ·v   (permanent impact, linear in trade rate)
  - h(v) = η·v   (temporary impact, linear in trade rate)
  - σ             daily volatility (price units per √day)
  - ξ_k           i.i.d. N(0,1) innovations

Cost decomposition::

    E[C]   = ½·γ·X² + η·Σ_k (n_k²/τ)
    Var[C] = σ²·τ·Σ_k x_k²
    U(λ)   = E[C] + λ·Var[C]

Optimal trajectory (closed form)::

    x*(t) = X · sinh(κ(T−t)) / sinh(κT)
    κ²    = λσ²/η

The frontier is parameterised by λ ∈ [0, ∞).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _kappa(lam: float, sigma: float, eta: float) -> float:
    """
    Compute the decay rate κ from risk-aversion and model parameters.

    .. math::

        \\kappa = \\sqrt{\\frac{\\lambda \\sigma^2}{\\eta}}

    Parameters
    ----------
    lam : float
        Risk-aversion coefficient λ (units: 1 / [cost²]).
    sigma : float
        Asset volatility (price units · day^{-1/2}).
    eta : float
        Temporary impact coefficient η (price units · days / shares).

    Returns
    -------
    float
        κ ≥ 0 (units: day^{-1}).
    """
    if eta <= 0:
        raise ValueError(f"eta must be positive, got {eta}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    return float(np.sqrt(lam * sigma**2 / eta))


def _expected_cost(
    trades: NDArray[np.float64],
    tau: float,
    gamma: float,
    eta: float,
    X: float,
) -> float:
    """
    Compute expected execution cost E[C] for a given trade schedule.

    .. math::

        E[C] = \\frac{1}{2}\\,\\gamma X^2
               + \\eta \\sum_{k=1}^{N} \\frac{n_k^2}{\\tau}

    The first term is the permanent-impact cost (independent of schedule);
    the second is the temporary-impact cost (convex in trade sizes).

    Parameters
    ----------
    trades : NDArray[np.float64], shape (N,)
        Trade sizes n_k = x_{k-1} − x_k  (shares, positive = sell).
    tau : float
        Length of each interval τ = T/N (days).
    gamma : float
        Permanent impact coefficient γ (price units · days / shares).
    eta : float
        Temporary impact coefficient η (price units · days / shares).
    X : float
        Total shares to liquidate (initial position size).

    Returns
    -------
    float
        E[C] in price units · shares (i.e. dollar cost if price in $/share).
    """
    permanent_cost = 0.5 * gamma * X**2
    temporary_cost = eta * np.sum(trades**2 / tau)
    return float(permanent_cost + temporary_cost)


def _variance_of_cost(
    holdings: NDArray[np.float64],
    tau: float,
    sigma: float,
) -> float:
    """
    Compute the variance of execution cost Var[C] for a holdings trajectory.

    .. math::

        \\operatorname{Var}[C] = \\sigma^2 \\tau \\sum_{k=1}^{N} x_k^2

    The sum runs over post-trade holdings x_1, …, x_N (not x_0 = X).

    Parameters
    ----------
    holdings : NDArray[np.float64], shape (N+1,)
        Holdings at each time step: x_0, x_1, …, x_N.
        Must satisfy x_0 = X and x_N = 0.
    tau : float
        Length of each interval τ = T/N (days).
    sigma : float
        Asset volatility (price units · day^{-1/2}).

    Returns
    -------
    float
        Var[C] in (price units · shares)².
    """
    # Sum over post-trade holdings x_1, …, x_N (exclude x_0)
    return float(sigma**2 * tau * np.sum(holdings[1:] ** 2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimal_trajectory(
    X: float,
    T: float,
    N: int,
    sigma: float,
    gamma: float,
    eta: float,
    lam: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the AC2001 optimal liquidation trajectory.

    Solves for the holdings path that minimises the mean–variance utility::

        U(λ) = E[C] + λ · Var[C]

    The closed-form solution is::

        x*(t) = X · sinh(κ(T−t)) / sinh(κT)
        κ²    = λσ²/η

    evaluated at the N+1 discrete time points  t_j = j·τ,  j = 0, 1, …, N.

    Boundary behaviour:
      - λ → 0  (risk-neutral): κ → 0, trajectory approaches uniform TWAP.
      - λ → ∞  (fully risk-averse): κ → ∞, position liquidated immediately.

    Parameters
    ----------
    X : float
        Initial position in shares (positive = long position to sell).
    T : float
        Liquidation horizon in days.
    N : int
        Number of execution intervals (i.e., number of trades = N).
    sigma : float
        Asset daily volatility in price units · day^{-1/2}.
    gamma : float
        Permanent market impact coefficient γ in price units · days / shares.
        Sets the long-run price depression per unit of trade rate.
    eta : float
        Temporary market impact coefficient η in price units · days / shares.
        Sets the instantaneous execution cost per unit of trade rate.
    lam : float
        Risk-aversion parameter λ ≥ 0.  Larger λ front-loads trading to
        reduce variance at the cost of higher expected impact.

    Returns
    -------
    times : NDArray[np.float64], shape (N+1,)
        Time grid  t_j = j·τ,  j = 0, …, N  (units: days).
    holdings : NDArray[np.float64], shape (N+1,)
        Remaining shares at each time step  x_j = x*(t_j).
        Satisfies  holdings[0] = X  and  holdings[N] ≈ 0.

    Raises
    ------
    ValueError
        If any parameter is out of its valid domain.

    Examples
    --------
    >>> times, holdings = optimal_trajectory(
    ...     X=1e6, T=1.0, N=10, sigma=0.02, gamma=2.5e-7, eta=2.5e-6, lam=1e-6
    ... )
    >>> holdings[0]
    1000000.0
    >>> abs(holdings[-1]) < 1e-6
    True
    """
    if X <= 0:
        raise ValueError(f"X must be positive, got {X}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if N < 1:
        raise ValueError(f"N must be at least 1, got {N}")
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")

    tau = T / N
    times = np.linspace(0.0, T, N + 1)  # shape (N+1,)

    kap = _kappa(lam, sigma, eta)

    if kap < 1e-12:
        # Risk-neutral limit (λ → 0): uniform liquidation (TWAP)
        holdings = X * (1.0 - times / T)
    else:
        sinh_kT = np.sinh(kap * T)
        if not np.isfinite(sinh_kT) or sinh_kT == 0.0:
            raise ValueError(
                f"sinh(κT) = {sinh_kT} is degenerate. "
                "Check that κT is not too large (reduce lam or T)."
            )
        holdings = X * np.sinh(kap * (T - times)) / sinh_kT

    # Enforce exact boundary conditions numerically
    holdings[0] = X
    holdings[-1] = 0.0

    return times, holdings


def trade_schedule(holdings: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Derive trade sizes from a holdings trajectory.

    .. math::

        n_k = x_{k-1} - x_k, \\quad k = 1, \\ldots, N

    All trades are positive for a sell programme (n_k > 0).

    Parameters
    ----------
    holdings : NDArray[np.float64], shape (N+1,)
        Holdings path from :func:`optimal_trajectory`.

    Returns
    -------
    trades : NDArray[np.float64], shape (N,)
        Trade sizes at each interval.  Positive = sell.
    """
    return np.diff(holdings) * -1.0  # n_k = x_{k-1} - x_k


def cost_of_trajectory(
    holdings: NDArray[np.float64],
    T: float,
    sigma: float,
    gamma: float,
    eta: float,
    X: float,
) -> dict[str, float]:
    """
    Compute expected cost and variance for an arbitrary holdings trajectory.

    Returns both components so callers can construct utility  U = E[C] + λ·Var[C]
    for any λ without re-running the trajectory.

    .. math::

        E[C]   = \\tfrac{1}{2}\\gamma X^2 + \\eta \\sum_k \\frac{n_k^2}{\\tau}

        \\operatorname{Var}[C] = \\sigma^2 \\tau \\sum_k x_k^2

    Parameters
    ----------
    holdings : NDArray[np.float64], shape (N+1,)
        Holdings at each time step.  Must start at X and end at 0.
    T : float
        Liquidation horizon in days.
    sigma : float
        Asset daily volatility (price units · day^{-1/2}).
    gamma : float
        Permanent impact coefficient γ (price units · days / shares).
    eta : float
        Temporary impact coefficient η (price units · days / shares).
    X : float
        Total shares to liquidate (= holdings[0]).

    Returns
    -------
    dict with keys:
        ``expected_cost`` : float
            E[C] in price-units · shares.
        ``variance`` : float
            Var[C] in (price-units · shares)².
        ``std_cost`` : float
            √Var[C] — standard deviation of cost.
    """
    N = len(holdings) - 1
    tau = T / N
    trades = trade_schedule(holdings)

    ec = _expected_cost(trades, tau, gamma, eta, X)
    vc = _variance_of_cost(holdings, tau, sigma)

    return {
        "expected_cost": ec,
        "variance": vc,
        "std_cost": float(np.sqrt(vc)),
    }


def efficient_frontier(
    X: float,
    T: float,
    N: int,
    sigma: float,
    gamma: float,
    eta: float,
    lambdas: NDArray[np.float64] | None = None,
    n_points: int = 100,
) -> dict[str, NDArray[np.float64]]:
    """
    Trace the mean–variance efficient frontier for the AC2001 model.

    For each risk-aversion level λ in ``lambdas``, computes the optimal
    trajectory and its (E[C], Var[C]) coordinates.  The resulting curve
    is the Pareto-optimal set of execution strategies — no feasible strategy
    can reduce expected cost without increasing variance, or vice versa.

    The frontier is parameterised implicitly by λ via::

        κ(λ) = √(λσ²/η)
        x*(t; λ) = X · sinh(κ(T−t)) / sinh(κT)

    At the extremes:
      - λ = 0   → risk-neutral TWAP (minimises E[C], ignores Var[C])
      - λ → ∞   → immediate liquidation (minimises Var[C], maximises E[C])

    Parameters
    ----------
    X : float
        Initial position in shares.
    T : float
        Liquidation horizon in days.
    N : int
        Number of execution intervals.
    sigma : float
        Asset daily volatility (price units · day^{-1/2}).
    gamma : float
        Permanent impact coefficient γ (price units · days / shares).
    eta : float
        Temporary impact coefficient η (price units · days / shares).
    lambdas : NDArray[np.float64], optional
        Array of risk-aversion levels to evaluate.  If None, a logarithmically
        spaced grid from 1e-8 to 1e-3 is used (n_points points).
    n_points : int, default 100
        Number of λ grid points when ``lambdas`` is None.

    Returns
    -------
    dict with keys:
        ``lambdas`` : NDArray[np.float64], shape (M,)
            Risk-aversion values used.
        ``expected_costs`` : NDArray[np.float64], shape (M,)
            E[C] for each λ.
        ``variances`` : NDArray[np.float64], shape (M,)
            Var[C] for each λ.
        ``std_costs`` : NDArray[np.float64], shape (M,)
            √Var[C] for each λ.
        ``utilities`` : NDArray[np.float64], shape (M,)
            U = E[C] + λ·Var[C] for each λ (the minimised objective).
        ``trajectories`` : list of NDArray[np.float64]
            Holdings arrays (shape N+1 each) for each λ.
        ``times`` : NDArray[np.float64], shape (N+1,)
            Shared time grid for all trajectories.

    Notes
    -----
    The permanent-impact term ½·γ·X² is constant across all strategies and
    therefore does not affect the *shape* of the frontier — only its vertical
    intercept.  It is included in E[C] for completeness.

    Examples
    --------
    >>> import numpy as np
    >>> frontier = efficient_frontier(
    ...     X=1e6, T=1.0, N=390, sigma=0.02,
    ...     gamma=2.5e-7, eta=2.5e-6, n_points=50
    ... )
    >>> frontier["expected_costs"].shape
    (50,)
    """
    if lambdas is None:
        lambdas = np.logspace(-8, -3, n_points)

    lambdas = np.asarray(lambdas, dtype=np.float64)

    expected_costs = np.empty(len(lambdas))
    variances = np.empty(len(lambdas))
    std_costs = np.empty(len(lambdas))
    utilities = np.empty(len(lambdas))
    trajectories: list[NDArray[np.float64]] = []
    times: NDArray[np.float64] | None = None

    for i, lam in enumerate(lambdas):
        t, h = optimal_trajectory(X, T, N, sigma, gamma, eta, lam)
        costs = cost_of_trajectory(h, T, sigma, gamma, eta, X)

        expected_costs[i] = costs["expected_cost"]
        variances[i] = costs["variance"]
        std_costs[i] = costs["std_cost"]
        utilities[i] = costs["expected_cost"] + lam * costs["variance"]
        trajectories.append(h)

        if times is None:
            times = t

    return {
        "lambdas": lambdas,
        "expected_costs": expected_costs,
        "variances": variances,
        "std_costs": std_costs,
        "utilities": utilities,
        "trajectories": trajectories,
        "times": times,
    }


def twap_trajectory(X: float, N: int, T: float) -> NDArray[np.float64]:
    """
    Compute a TWAP (Time-Weighted Average Price) holdings trajectory.

    TWAP sells equal-sized blocks each interval::

        n_k = X / N  for all k,
        x_j = X · (1 − j/N)

    This is the risk-neutral (λ = 0) optimal trajectory and serves as a
    natural benchmark: it minimises expected impact cost but ignores timing risk.

    Parameters
    ----------
    X : float
        Initial position in shares.
    N : int
        Number of execution intervals.
    T : float
        Liquidation horizon in days (used only to build the time grid via
        :func:`optimal_trajectory`; does not affect trade sizes).

    Returns
    -------
    holdings : NDArray[np.float64], shape (N+1,)
        TWAP holdings x_j = X · (1 − j/N).
    """
    j = np.arange(N + 1, dtype=np.float64)
    return X * (1.0 - j / N)


def utility(
    holdings: NDArray[np.float64],
    T: float,
    sigma: float,
    gamma: float,
    eta: float,
    X: float,
    lam: float,
) -> float:
    """
    Evaluate mean–variance utility U = E[C] + λ·Var[C] for a trajectory.

    .. math::

        U(\\lambda) = E[C] + \\lambda \\cdot \\operatorname{Var}[C]

    Parameters
    ----------
    holdings : NDArray[np.float64], shape (N+1,)
        Holdings path.
    T : float
        Liquidation horizon in days.
    sigma : float
        Asset daily volatility.
    gamma : float
        Permanent impact coefficient.
    eta : float
        Temporary impact coefficient.
    X : float
        Initial position size.
    lam : float
        Risk-aversion coefficient λ.

    Returns
    -------
    float
        Scalar utility value.
    """
    costs = cost_of_trajectory(holdings, T, sigma, gamma, eta, X)
    return costs["expected_cost"] + lam * costs["variance"]
