"""
Market Impact & Optimal Execution Model
========================================

A Python implementation of the Almgren–Chriss (2001) optimal liquidation
framework.

Submodules
----------
almgren_chriss
    Core AC2001 model: closed-form optimal trajectory and efficient frontier.
market_impact
    Permanent and temporary impact functions (linear and square-root models).
execution_sim
    Monte Carlo execution simulator: IS computation, strategy comparison.
vol_estimator
    Volatility estimators: close-to-close, GARCH(1,1), HAR-RV.
utils
    Shared helpers: vol scaling, bps conversion, trajectory validation.
"""

from .almgren_chriss import (
    optimal_trajectory,
    efficient_frontier,
    twap_trajectory,
    trade_schedule,
    cost_of_trajectory,
    utility,
)
from .execution_sim import (
    simulate_execution,
    vwap_holdings,
    compare_strategies,
    slippage_summary,
    intraday_u_shape,
)
from .market_impact import (
    permanent_impact_linear,
    temporary_impact_linear,
    temporary_impact_sqrt,
    almgren_2005_params,
    kyle_lambda_ols,
)
from .utils import (
    annualize_vol,
    daily_vol_from_annual,
    cost_in_bps,
    validate_trajectory,
    log_returns,
)
from .vol_estimator import (
    close_to_close_vol,
    parkinson_vol,
    GARCH11,
    HARRV,
    compare_vol_estimators,
)

__all__ = [
    # almgren_chriss
    "optimal_trajectory",
    "efficient_frontier",
    "twap_trajectory",
    "trade_schedule",
    "cost_of_trajectory",
    "utility",
    # execution_sim
    "simulate_execution",
    "vwap_holdings",
    "compare_strategies",
    "slippage_summary",
    "intraday_u_shape",
    # market_impact
    "permanent_impact_linear",
    "temporary_impact_linear",
    "temporary_impact_sqrt",
    "almgren_2005_params",
    "kyle_lambda_ols",
    # utils
    "annualize_vol",
    "daily_vol_from_annual",
    "cost_in_bps",
    "validate_trajectory",
    "log_returns",
    # vol_estimator
    "close_to_close_vol",
    "parkinson_vol",
    "GARCH11",
    "HARRV",
    "compare_vol_estimators",
]
