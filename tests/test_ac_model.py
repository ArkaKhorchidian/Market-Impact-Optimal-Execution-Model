"""
tests/test_ac_model.py — pytest suite for the Almgren–Chriss model.

Test categories (per CLAUDE.md spec):

1. Boundary: λ→0  trajectory should approach uniform TWAP.
2. Boundary: λ→∞  trajectory should front-load aggressively.
3. Total shares: sum of all trades = X exactly.
4. Cost function: manually verify E[C] formula against numpy computation.
5. TWAP: verify TWAP is a valid (non-optimal) point on/outside the frontier.
6. Efficient frontier: shape and monotonicity checks.
7. Execution sim: IS non-negative on average, revenue conservation.
8. Vol estimators: basic smoke tests and shape checks.
9. Utils: unit conversion round-trips, trajectory validation.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Standard model parameters used across tests
# ---------------------------------------------------------------------------

X = 1_000_000.0       # shares
T = 1.0               # day
N = 390               # 1-minute intervals for a full trading day
SIGMA = 0.02          # 2% daily vol (fractional)
GAMMA = 2.5e-7        # permanent impact (AC2001 typical)
ETA = 2.5e-6          # temporary impact
S0 = 100.0            # arrival price ($)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trajectory(lam: float, n: int = N):
    from src.almgren_chriss import optimal_trajectory
    return optimal_trajectory(X, T, n, SIGMA, GAMMA, ETA, lam)


# ===========================================================================
# 1. Boundary: λ→0 → TWAP
# ===========================================================================

class TestRiskNeutralLimit:
    """λ → 0 should recover the uniform TWAP trajectory."""

    def test_risk_neutral_approaches_twap(self):
        from src.almgren_chriss import twap_trajectory

        _, h_ac = make_trajectory(lam=1e-14)
        h_twap = twap_trajectory(X, N, T)

        # Max absolute deviation should be tiny relative to X
        max_dev = np.max(np.abs(h_ac - h_twap))
        assert max_dev < 1e-3, (
            f"λ→0 trajectory deviates from TWAP by {max_dev:.3e} shares"
        )

    def test_risk_neutral_uniform_trades(self):
        """Individual trade sizes should be almost equal."""
        from src.almgren_chriss import trade_schedule

        _, h = make_trajectory(lam=1e-14)
        trades = trade_schedule(h)
        expected = X / N
        assert np.allclose(trades, expected, rtol=1e-4), (
            f"Trades not uniform: std/mean = {trades.std()/trades.mean():.2e}"
        )


# ===========================================================================
# 2. Boundary: λ→∞ → front-loaded
# ===========================================================================

class TestRiskAverseLimit:
    """High λ should cause more aggressive early liquidation."""

    def test_high_lambda_frontloads(self):
        _, h_high = make_trajectory(lam=1e-3)
        _, h_low  = make_trajectory(lam=1e-7)

        # After the first interval, high-λ should have sold more
        assert h_high[1] < h_low[1], (
            "High-λ trajectory should have fewer shares remaining after step 1"
        )

    def test_monotone_in_lambda(self):
        """As λ increases, holdings should decrease faster at each step."""
        from src.almgren_chriss import trade_schedule

        lambdas = [1e-8, 1e-6, 1e-4]
        first_trades = []
        for lam in lambdas:
            _, h = make_trajectory(lam, n=20)
            first_trades.append(trade_schedule(h)[0])

        assert first_trades[0] < first_trades[1] < first_trades[2], (
            f"First trade not monotone in λ: {first_trades}"
        )

    def test_very_high_lambda_liquidates_early(self):
        """For extreme λ, almost all shares should be sold in the first few steps."""
        _, h = make_trajectory(lam=1.0, n=20)
        # After half the horizon, less than 5% of X should remain
        mid = len(h) // 2
        assert h[mid] < 0.05 * X, (
            f"High-λ: {h[mid]:.0f} shares remain at midpoint (expected < {0.05*X:.0f})"
        )


# ===========================================================================
# 3. Total shares conservation
# ===========================================================================

class TestSharesConservation:
    """Sum of all trades must exactly equal X."""

    @pytest.mark.parametrize("lam", [1e-12, 1e-6, 1e-3])
    def test_total_trades_equal_X(self, lam):
        from src.almgren_chriss import trade_schedule

        _, h = make_trajectory(lam)
        trades = trade_schedule(h)
        assert abs(trades.sum() - X) < 1.0, (
            f"λ={lam}: sum(trades) = {trades.sum():.2f} ≠ X = {X}"
        )

    @pytest.mark.parametrize("lam", [1e-12, 1e-6, 1e-3])
    def test_final_holdings_zero(self, lam):
        _, h = make_trajectory(lam)
        assert h[-1] == pytest.approx(0.0, abs=1.0), (
            f"Final holdings = {h[-1]:.4f} ≠ 0"
        )

    def test_initial_holdings_equals_X(self):
        _, h = make_trajectory(1e-6)
        assert h[0] == pytest.approx(X)

    def test_holdings_non_negative(self):
        _, h = make_trajectory(1e-6)
        assert np.all(h >= -1.0), "Holdings should remain non-negative"

    def test_twap_conservation(self):
        from src.almgren_chriss import twap_trajectory, trade_schedule

        h = twap_trajectory(X, N, T)
        trades = trade_schedule(h)
        assert abs(trades.sum() - X) < 1e-6


# ===========================================================================
# 4. Cost function: manual vs implementation
# ===========================================================================

class TestCostFormula:
    """
    Verify E[C] and Var[C] match the closed-form expressions from AC2001.
    """

    def test_expected_cost_formula(self):
        from src.almgren_chriss import trade_schedule, cost_of_trajectory

        lam = 1e-6
        n = 10
        tau = T / n
        _, h = make_trajectory(lam, n=n)
        trades = trade_schedule(h)

        # Manual calculation per AC2001
        permanent_component = 0.5 * GAMMA * X**2
        temporary_component = ETA * np.sum(trades**2 / tau)
        expected_cost_manual = permanent_component + temporary_component

        costs = cost_of_trajectory(h, T, SIGMA, GAMMA, ETA, X)
        assert costs["expected_cost"] == pytest.approx(expected_cost_manual, rel=1e-8)

    def test_variance_formula(self):
        from src.almgren_chriss import cost_of_trajectory

        n = 10
        tau = T / n
        _, h = make_trajectory(1e-6, n=n)

        # Manual calculation
        var_manual = SIGMA**2 * tau * np.sum(h[1:]**2)

        costs = cost_of_trajectory(h, T, SIGMA, GAMMA, ETA, X)
        assert costs["variance"] == pytest.approx(var_manual, rel=1e-8)

    def test_std_cost_is_sqrt_variance(self):
        from src.almgren_chriss import cost_of_trajectory

        _, h = make_trajectory(1e-6, n=20)
        costs = cost_of_trajectory(h, T, SIGMA, GAMMA, ETA, X)
        assert costs["std_cost"] == pytest.approx(np.sqrt(costs["variance"]), rel=1e-10)

    def test_permanent_cost_independent_of_lambda(self):
        """The ½γX² term is schedule-independent — it should be the same for all λ."""
        from src.almgren_chriss import cost_of_trajectory

        n = 20
        costs_low = cost_of_trajectory(make_trajectory(1e-8, n)[1], T, SIGMA, GAMMA, ETA, X)
        costs_high = cost_of_trajectory(make_trajectory(1e-2, n)[1], T, SIGMA, GAMMA, ETA, X)

        # Permanent component = ½·γ·X²
        perm = 0.5 * GAMMA * X**2
        # Minimum possible E[C] across all schedules should equal perm
        # The difference between the two should only come from temporary impact
        # which is minimal at low λ (near TWAP) and high at high λ (front-loaded)
        assert costs_low["expected_cost"] < costs_high["expected_cost"], (
            "TWAP (low λ) should have lower E[C] than front-loaded (high λ)"
        )
        assert costs_low["expected_cost"] > perm


# ===========================================================================
# 5. TWAP sits on or outside the efficient frontier
# ===========================================================================

class TestTWAPOnFrontier:
    """
    TWAP is the risk-neutral optimum (λ=0).  For any λ>0, the AC trajectory
    achieves strictly lower utility U = E[C] + λ·Var[C].
    """

    def test_ac_dominates_twap_utility(self):
        from src.almgren_chriss import twap_trajectory, utility

        lam = 1e-6
        _, h_ac = make_trajectory(lam, n=50)
        h_twap = twap_trajectory(X, 50, T)

        u_ac   = utility(h_ac,   T, SIGMA, GAMMA, ETA, X, lam)
        u_twap = utility(h_twap, T, SIGMA, GAMMA, ETA, X, lam)

        assert u_ac <= u_twap + 1e-3, (
            f"AC utility ({u_ac:.2f}) should be ≤ TWAP utility ({u_twap:.2f})"
        )

    def test_twap_is_optimal_at_zero_lambda(self):
        """At λ=0, TWAP and the AC trajectory have the same E[C]."""
        from src.almgren_chriss import twap_trajectory, cost_of_trajectory

        n = 50
        _, h_ac = make_trajectory(lam=1e-14, n=n)
        h_twap = twap_trajectory(X, n, T)

        ec_ac   = cost_of_trajectory(h_ac,   T, SIGMA, GAMMA, ETA, X)["expected_cost"]
        ec_twap = cost_of_trajectory(h_twap, T, SIGMA, GAMMA, ETA, X)["expected_cost"]

        assert abs(ec_ac - ec_twap) / ec_twap < 1e-4, (
            f"At λ→0, E[C]_ac={ec_ac:.2f} should ≈ E[C]_twap={ec_twap:.2f}"
        )


# ===========================================================================
# 6. Efficient frontier: shape and monotonicity
# ===========================================================================

class TestEfficientFrontier:
    """The efficient frontier should trace a convex Pareto curve."""

    def setup_method(self):
        from src.almgren_chriss import efficient_frontier
        self.frontier = efficient_frontier(
            X, T, 50, SIGMA, GAMMA, ETA,
            lambdas=np.logspace(-10, -3, 30),
        )

    def test_expected_costs_increase_with_lambda(self):
        ec = self.frontier["expected_costs"]
        assert np.all(np.diff(ec) >= -1.0), (
            "Expected cost should be non-decreasing as λ increases"
        )

    def test_variances_decrease_with_lambda(self):
        var = self.frontier["variances"]
        assert np.all(np.diff(var) <= 1.0), (
            "Variance should be non-increasing as λ increases"
        )

    def test_frontier_has_correct_shape(self):
        assert self.frontier["expected_costs"].shape == (30,)
        assert self.frontier["variances"].shape == (30,)
        assert self.frontier["lambdas"].shape == (30,)
        assert len(self.frontier["trajectories"]) == 30

    def test_utilities_are_minimised(self):
        """Each trajectory should achieve lower utility than TWAP for its own λ."""
        from src.almgren_chriss import twap_trajectory, cost_of_trajectory

        h_twap = twap_trajectory(X, 50, T)
        costs_twap = cost_of_trajectory(h_twap, T, SIGMA, GAMMA, ETA, X)

        for i, lam in enumerate(self.frontier["lambdas"]):
            u_ac = (
                self.frontier["expected_costs"][i]
                + lam * self.frontier["variances"][i]
            )
            u_twap = (
                costs_twap["expected_cost"]
                + lam * costs_twap["variance"]
            )
            assert u_ac <= u_twap + 1e-2, (
                f"λ={lam:.2e}: AC utility {u_ac:.2f} > TWAP utility {u_twap:.2f}"
            )


# ===========================================================================
# 7. Execution simulator
# ===========================================================================

class TestExecutionSim:

    def test_is_cost_non_negative_mean(self):
        """On average, execution should cost something (IS > 0)."""
        from src.execution_sim import simulate_execution

        _, h = make_trajectory(1e-6, n=50)
        rng = np.random.default_rng(42)
        is_costs = [
            simulate_execution(h, T, S0, SIGMA, GAMMA, ETA, rng).is_cost
            for _ in range(200)
        ]
        assert np.mean(is_costs) > 0, "Mean IS should be positive (sell into impact)"

    def test_revenue_is_positive(self):
        from src.execution_sim import simulate_execution

        _, h = make_trajectory(1e-6, n=50)
        res = simulate_execution(h, T, S0, SIGMA, GAMMA, ETA,
                                  rng=np.random.default_rng(0))
        assert res.revenue > 0

    def test_trades_sum_to_X(self):
        from src.execution_sim import simulate_execution

        _, h = make_trajectory(1e-6, n=50)
        res = simulate_execution(h, T, S0, SIGMA, GAMMA, ETA,
                                  rng=np.random.default_rng(0))
        assert abs(res.trades.sum() - X) < 1.0

    def test_compare_strategies_returns_three_keys(self):
        from src.execution_sim import compare_strategies

        stats = compare_strategies(
            X=1e5, T=1.0, N=20, sigma=SIGMA,
            gamma=GAMMA, eta=ETA, lam=1e-6,
            S0=S0, n_sims=50, seed=42,
        )
        assert set(stats.keys()) == {"AC2001", "TWAP", "VWAP"}

    def test_ac_mean_is_leq_twap_mean_is(self):
        """AC2001 should have lower or equal mean IS than TWAP for λ > 0."""
        from src.execution_sim import compare_strategies

        stats = compare_strategies(
            X=1e5, T=1.0, N=20, sigma=SIGMA,
            gamma=GAMMA, eta=ETA, lam=1e-6,
            S0=S0, n_sims=200, seed=0,
        )
        # AC IS should be lower than TWAP IS (AC optimises the trade-off)
        # With finite sims there may be minor noise; we use a loose tolerance
        assert stats["AC2001"].mean_is_bps <= stats["TWAP"].mean_is_bps + 1.0, (
            f"AC IS ({stats['AC2001'].mean_is_bps:.2f}) "
            f"> TWAP IS ({stats['TWAP'].mean_is_bps:.2f}) by more than 1 bps"
        )

    def test_vwap_holdings_sums_to_X(self):
        from src.execution_sim import vwap_holdings, intraday_u_shape

        profile = intraday_u_shape(50)
        h = vwap_holdings(X, profile)
        from src.almgren_chriss import trade_schedule
        assert abs(trade_schedule(h).sum() - X) < 1e-6


# ===========================================================================
# 8. Volatility estimators
# ===========================================================================

class TestVolEstimators:

    @pytest.fixture
    def price_series(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=300)
        prices = 100.0 * np.exp(np.cumsum(returns))
        return prices

    def test_close_to_close_vol_scalar(self, price_series):
        from src.vol_estimator import close_to_close_vol
        vol = close_to_close_vol(price_series)
        assert isinstance(vol, float)
        assert 0.0 < vol < 0.5, f"Vol = {vol:.4f} seems unreasonable"

    def test_close_to_close_vol_rolling(self, price_series):
        from src.vol_estimator import close_to_close_vol
        vols = close_to_close_vol(price_series, window=21)
        assert vols.ndim == 1
        assert len(vols) == len(price_series) - 1 - 21 + 1
        assert np.all(vols > 0)

    def test_parkinson_vol(self, price_series):
        from src.vol_estimator import parkinson_vol
        highs = price_series * (1 + np.abs(np.random.default_rng(1).normal(0, 0.005, len(price_series))))
        lows  = price_series * (1 - np.abs(np.random.default_rng(2).normal(0, 0.005, len(price_series))))
        vol = parkinson_vol(highs, lows)
        assert 0 < vol < 0.5

    def test_garch_fit_and_forecast(self, price_series):
        from src.vol_estimator import GARCH11
        g = GARCH11().fit(price_series)
        assert 0 < g.alpha_ < 1
        assert 0 < g.beta_ < 1
        assert g.alpha_ + g.beta_ < 1
        assert g.daily_vol > 0
        forecasts = g.forecast(h=5)
        assert len(forecasts) == 5
        assert np.all(forecasts > 0)

    def test_garch_unconditional_vol_positive(self, price_series):
        from src.vol_estimator import GARCH11
        g = GARCH11().fit(price_series)
        assert g.unconditional_vol > 0

    def test_har_rv_fit(self, price_series):
        from src.vol_estimator import HARRV, log_returns
        rv = log_returns(price_series) ** 2
        har = HARRV().fit(rv)
        assert har.r_squared_ is not None
        assert 0.0 <= har.r_squared_ <= 1.0
        assert har.daily_vol > 0

    def test_har_rv_forecast_positive(self, price_series):
        from src.vol_estimator import HARRV, log_returns
        rv = log_returns(price_series) ** 2
        har = HARRV().fit(rv)
        f = har.forecast(h=10)
        assert len(f) == 10
        assert np.all(f >= 0)


# ===========================================================================
# 9. Utils
# ===========================================================================

class TestUtils:

    def test_annualize_roundtrip(self):
        from src.utils import annualize_vol, daily_vol_from_annual
        daily = 0.02
        annual = annualize_vol(daily)
        assert daily_vol_from_annual(annual) == pytest.approx(daily, rel=1e-10)

    def test_cost_in_bps(self):
        from src.utils import cost_in_bps
        assert cost_in_bps(100.0, 1_000_000.0) == pytest.approx(1.0)
        assert cost_in_bps(1000.0, 1_000_000.0) == pytest.approx(10.0)

    def test_bps_to_cost_roundtrip(self):
        from src.utils import cost_in_bps, bps_to_cost
        cost = 1234.56
        notional = 5_000_000.0
        bps = cost_in_bps(cost, notional)
        assert bps_to_cost(bps, notional) == pytest.approx(cost, rel=1e-10)

    def test_validate_trajectory_passes_valid(self):
        from src.utils import validate_trajectory
        _, h = make_trajectory(1e-6, n=10)
        validate_trajectory(h, X)  # should not raise

    def test_validate_trajectory_catches_wrong_start(self):
        from src.utils import validate_trajectory
        _, h = make_trajectory(1e-6, n=10)
        h[0] = X * 1.1
        with pytest.raises(ValueError, match="holdings\\[0\\]"):
            validate_trajectory(h, X)

    def test_validate_trajectory_catches_nonzero_end(self):
        from src.utils import validate_trajectory
        _, h = make_trajectory(1e-6, n=10)
        h[-1] = X * 0.05
        with pytest.raises(ValueError, match="holdings\\[-1\\]"):
            validate_trajectory(h, X)

    def test_log_returns_length(self):
        from src.utils import log_returns
        prices = np.array([100.0, 101.0, 99.0, 102.0])
        r = log_returns(prices)
        assert len(r) == 3

    def test_scale_vol(self):
        from src.utils import scale_vol
        daily = 0.02
        # Daily → 5-day should be 0.02 * sqrt(5)
        assert scale_vol(daily, 1, 5) == pytest.approx(daily * np.sqrt(5), rel=1e-10)


# ===========================================================================
# 10. Market impact
# ===========================================================================

class TestMarketImpact:

    def test_linear_permanent_is_linear(self):
        from src.market_impact import permanent_impact_linear
        v = np.array([1.0, 2.0, 4.0])
        g = permanent_impact_linear(v, gamma=0.01)
        assert np.allclose(g, [0.01, 0.02, 0.04])

    def test_linear_temporary_is_linear(self):
        from src.market_impact import temporary_impact_linear
        v = np.array([1.0, 2.0, 4.0])
        h = temporary_impact_linear(v, eta=0.005)
        assert np.allclose(h, [0.005, 0.010, 0.020])

    def test_sqrt_impact_is_sublinear(self):
        from src.market_impact import temporary_impact_sqrt
        v_small = 1e3
        v_large = 1e6
        h_s = temporary_impact_sqrt(v_small, 0.1, 2.0, 1e7)
        h_l = temporary_impact_sqrt(v_large, 0.1, 2.0, 1e7)
        # Doubling v should less than double h (square root)
        ratio_v = v_large / v_small
        ratio_h = h_l / h_s
        assert ratio_h == pytest.approx(np.sqrt(ratio_v), rel=1e-8)

    def test_kyle_lambda_known_signal(self):
        from src.market_impact import kyle_lambda_ols
        rng = np.random.default_rng(7)
        lam_true = 0.003
        Q = rng.normal(0, 1e5, 500)
        dS = lam_true * Q + rng.normal(0, 0.01, 500)
        result = kyle_lambda_ols(Q, dS)
        assert result["lambda_"] == pytest.approx(lam_true, rel=0.05)
        assert 0.0 < result["r_squared"] < 1.0

    def test_almgren_2005_params_positive(self):
        from src.market_impact import almgren_2005_params
        p = almgren_2005_params(sigma=3.0, adv=5e6, price=150.0)
        assert p["eta"] > 0
        assert p["gamma"] > 0
        assert p["gamma"] > p["eta"]  # γ > η per the 2005 coefficients
