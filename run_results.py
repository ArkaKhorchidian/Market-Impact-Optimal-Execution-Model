"""Generate all results for the README."""
import sys
import warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 150

from src.almgren_chriss import (
    optimal_trajectory, efficient_frontier, twap_trajectory, cost_of_trajectory
)
from src.execution_sim import (
    compare_strategies, slippage_summary, vwap_holdings, intraday_u_shape
)
from src.vol_estimator import close_to_close_vol, GARCH11, HARRV, log_returns
from src.market_impact import almgren_2005_params

# ================================================================
# CANONICAL PARAMETERS (Almgren et al. 2005 order-of-magnitude)
# ================================================================
X, T, N = 1_000_000, 1.0, 390
sigma, gamma, eta = 0.02, 2.5e-7, 2.5e-6
S0 = 100.0


# ----------------------------------------------------------------
# FIG 1: OPTIMAL TRAJECTORIES
# ----------------------------------------------------------------
lambdas_plot = {
    "λ=1e-9 (near TWAP)":    1e-9,
    "λ=1e-6 (moderate)":     1e-6,
    "λ=1e-4 (risk-averse)":  1e-4,
    "λ=1e-2 (aggressive)":   1e-2,
}
fig, ax = plt.subplots(figsize=(9, 4.5))
for label, lam in lambdas_plot.items():
    t, h = optimal_trajectory(X, T, N, sigma, gamma, eta, lam)
    ax.plot(t, h / 1e6, label=label, linewidth=1.8)
h_twap = twap_trajectory(X, N, T)
ax.plot(np.linspace(0, T, N + 1), h_twap / 1e6, "k--", linewidth=1.5, label="TWAP")
ax.set_xlabel("Time (days)", fontsize=12)
ax.set_ylabel("Remaining position (M shares)", fontsize=12)
ax.set_title("AC2001 Optimal Holdings Trajectories", fontsize=13)
ax.legend(fontsize=9)
ax.set_xlim(0, T)
ax.set_ylim(0)
plt.tight_layout()
plt.savefig("data/fig_trajectories.png", dpi=150)
plt.close()
print("[1] fig_trajectories.png")


# ----------------------------------------------------------------
# FIG 2: EFFICIENT FRONTIER
# ----------------------------------------------------------------
lambdas_grid = np.logspace(-10, -2, 200)
fr = efficient_frontier(X, T, N, sigma, gamma, eta, lambdas=lambdas_grid)
h_twap = twap_trajectory(X, N, T)
twap_costs = cost_of_trajectory(h_twap, T, sigma, gamma, eta, X)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    fr["variances"] / 1e10,
    fr["expected_costs"] / 1e6,
    lw=2.0, color="steelblue", label="Efficient frontier",
)
for lam, marker in [(1e-9, "o"), (1e-4, "s"), (1e-2, "^")]:
    t_, h_ = optimal_trajectory(X, T, N, sigma, gamma, eta, lam)
    c_ = cost_of_trajectory(h_, T, sigma, gamma, eta, X)
    ax.scatter(
        c_["variance"] / 1e10, c_["expected_cost"] / 1e6,
        s=70, zorder=5, label=f"λ={lam:.0e}", marker=marker,
    )
ax.scatter(
    twap_costs["variance"] / 1e10, twap_costs["expected_cost"] / 1e6,
    s=90, color="tomato", zorder=6, marker="D", label="TWAP (λ=0)",
)
ax.set_xlabel("Var[C] (×10¹⁰)", fontsize=12)
ax.set_ylabel("E[C] ($M)", fontsize=12)
ax.set_title("AC2001 Mean–Variance Efficient Frontier", fontsize=13)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("data/fig_frontier.png", dpi=150)
plt.close()
print("[2] fig_frontier.png")

# Print frontier reference numbers
print("\n=== EFFICIENT FRONTIER NUMBERS ===")
twap_bps = twap_costs["expected_cost"] / (X * S0) * 10000
print(f"TWAP   E[C]=${twap_costs['expected_cost']:,.0f} ({twap_bps:.1f} bps)  "
      f"Var[C]={twap_costs['variance']:.4e}  std[C]=${twap_costs['std_cost']:,.0f}")
for lam in [1e-9, 1e-6, 1e-4, 1e-2]:
    t_, h_ = optimal_trajectory(X, T, N, sigma, gamma, eta, lam)
    c_ = cost_of_trajectory(h_, T, sigma, gamma, eta, X)
    bps = c_["expected_cost"] / (X * S0) * 10000
    var_reduction = (twap_costs["variance"] - c_["variance"]) / twap_costs["variance"] * 100
    print(
        f"λ={lam:.0e}  E[C]=${c_['expected_cost']:,.0f} ({bps:.1f} bps)  "
        f"Var={c_['variance']:.4e}  std=${c_['std_cost']:,.0f}  "
        f"var_reduction={var_reduction:.1f}%"
    )


# ----------------------------------------------------------------
# FIG 3: TRADE SCHEDULES (λ=1e-2 for visible difference)
# ----------------------------------------------------------------
lam_vis = 1e-2
t, h_ac  = optimal_trajectory(X, T, N, sigma, gamma, eta, lam_vis)
h_twap_s = twap_trajectory(X, N, T)
h_vwap   = vwap_holdings(X, intraday_u_shape(N))
n_ac     = -np.diff(h_ac)
n_twap_s = -np.diff(h_twap_s)
n_vwap   = -np.diff(h_vwap)
t_mid    = t[:-1] + (T / N) / 2

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(t, h_ac / 1e6,    label="AC2001 (λ=1e-2)", lw=2)
axes[0].plot(t, h_twap_s / 1e6, label="TWAP",            lw=2, ls="--")
axes[0].plot(t, h_vwap / 1e6,  label="VWAP (U-shape)",  lw=2, ls=":")
axes[0].set_xlabel("Time (days)")
axes[0].set_ylabel("Remaining position (M shares)")
axes[0].set_title("Holdings Trajectories")
axes[0].legend()

axes[1].plot(t_mid * 390, n_ac / 1e3,    label="AC2001 (λ=1e-2)", lw=1.5)
axes[1].plot(t_mid * 390, n_twap_s / 1e3, label="TWAP",            lw=1.5, ls="--")
axes[1].plot(t_mid * 390, n_vwap / 1e3,  label="VWAP",            lw=1.5, ls=":")
axes[1].set_xlabel("Minute from open")
axes[1].set_ylabel("Trade size (k shares)")
axes[1].set_title("Trade Schedules")
axes[1].legend()
plt.tight_layout()
plt.savefig("data/fig_schedules.png", dpi=150)
plt.close()
print("[3] fig_schedules.png")


# ----------------------------------------------------------------
# FIG 4: IS COMPARISON MONTE CARLO (λ=1e-2)
# ----------------------------------------------------------------
N_SIMS = 2000
lam_mc = 1e-2
print(f"\nRunning {N_SIMS} MC sims at λ={lam_mc}...")
stats = compare_strategies(
    X=X, T=T, N=N, sigma=sigma, gamma=gamma, eta=eta,
    lam=lam_mc, S0=S0, n_sims=N_SIMS, seed=42,
)
print(slippage_summary(stats))

rows = []
for name, s in stats.items():
    for val in s.is_bps_all:
        rows.append({"Strategy": name, "IS (bps)": val})
df = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(
    data=df, x="Strategy", y="IS (bps)", ax=axes[0],
    order=["AC2001", "TWAP", "VWAP"], width=0.5,
)
axes[0].set_title(f"IS Distribution ({N_SIMS:,} simulations, λ=1e-2)")
axes[0].set_ylabel("Implementation Shortfall (bps)")

for name, color in zip(["AC2001", "TWAP", "VWAP"], ["steelblue", "tomato", "seagreen"]):
    sns.kdeplot(
        stats[name].is_bps_all, ax=axes[1],
        label=name, color=color, fill=True, alpha=0.25,
    )
axes[1].set_xlabel("Implementation Shortfall (bps)")
axes[1].set_ylabel("Density")
axes[1].set_title("IS Density by Strategy")
axes[1].legend()
plt.tight_layout()
plt.savefig("data/fig_is_comparison.png", dpi=150)
plt.close()
print("[4] fig_is_comparison.png")

print("\n=== MC STATS (λ=1e-2) ===")
for name, s in stats.items():
    arr = s.is_bps_all
    print(
        f"{name:8s}: mean={s.mean_is_bps:.2f} bps, std={s.std_is_bps:.2f}, "
        f"p5={np.percentile(arr, 5):.1f}, p95={np.percentile(arr, 95):.1f}, "
        f"sharpe={s.sharpe_is:.4f}"
    )


# ----------------------------------------------------------------
# FIG 5 & 6: SENSITIVITY HEATMAPS
# ----------------------------------------------------------------
n_grid = 30
eta_vals = np.logspace(-7, -5, n_grid)
lam_vals = np.logspace(-8, -2, n_grid)
EC_grid  = np.empty((n_grid, n_grid))
for i, eta_v in enumerate(eta_vals):
    for j, lam_v in enumerate(lam_vals):
        _, h = optimal_trajectory(X, T, 50, sigma, gamma, eta_v, lam_v)
        EC_grid[i, j] = cost_of_trajectory(h, T, sigma, gamma, eta_v, X)["expected_cost"]

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.contourf(np.log10(lam_vals), np.log10(eta_vals), EC_grid / 1e6, levels=20, cmap="YlOrRd")
plt.colorbar(im, ax=ax, label="E[C] ($M)")
ax.set_xlabel("log₁₀(λ)", fontsize=12)
ax.set_ylabel("log₁₀(η)", fontsize=12)
ax.set_title("E[C] ($M) — heat map over (η, λ)", fontsize=13)
ax.scatter(
    [np.log10(1e-6)], [np.log10(2.5e-6)],
    s=100, color="white", edgecolors="black", zorder=5, label="Baseline",
)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("data/fig_heatmap_eta_lambda.png", dpi=150)
plt.close()
print("[5] fig_heatmap_eta_lambda.png")

sigma_vals = np.linspace(0.005, 0.06, n_grid)
lam_vals2  = np.logspace(-8, -2, n_grid)
VAR_grid   = np.empty((n_grid, n_grid))
for i, sig_v in enumerate(sigma_vals):
    for j, lam_v in enumerate(lam_vals2):
        _, h = optimal_trajectory(X, T, 50, sig_v, gamma, eta, lam_v)
        VAR_grid[i, j] = cost_of_trajectory(h, T, sig_v, gamma, eta, X)["variance"]

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.contourf(np.log10(lam_vals2), sigma_vals, VAR_grid / 1e8, levels=20, cmap="Blues")
plt.colorbar(im, ax=ax, label="Var[C] (×10⁸)")
ax.set_xlabel("log₁₀(λ)", fontsize=12)
ax.set_ylabel("σ (daily fractional vol)", fontsize=12)
ax.set_title("Var[C] — heat map over (σ, λ)", fontsize=13)
ax.scatter(
    [np.log10(1e-6)], [0.02],
    s=100, color="white", edgecolors="black", zorder=5, label="Baseline",
)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("data/fig_heatmap_sigma_lambda.png", dpi=150)
plt.close()
print("[6] fig_heatmap_sigma_lambda.png")


# ----------------------------------------------------------------
# FIG 7 & 8: BACKTEST WITH REAL DATA (or synthetic fallback)
# ----------------------------------------------------------------
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

TICKERS = ["AAPL", "MSFT", "SPY", "QQQ", "TSLA"]
N_SIMS_BT = 500
X_SHARES  = 500_000
T_DAYS    = 1.0
N_INT     = 390
LAM_BT    = 1e-6
SEED_BT   = 0
ADV = {"AAPL": 60e6, "MSFT": 25e6, "SPY": 100e6, "QQQ": 50e6, "TSLA": 80e6}


def fetch_prices(ticker):
    if HAS_YF:
        try:
            data = yf.download(ticker, period="2y", auto_adjust=True, progress=False, multi_level_index=False)
            return data["Close"].dropna().values.astype(float)
        except Exception:
            pass
    rng = np.random.default_rng(hash(ticker) % 2**32)
    daily_ret = rng.normal(0.0003, 0.02, size=504)
    return 100.0 * np.exp(np.cumsum(daily_ret))


price_data = {}
print("\n=== DATA FETCH ===")
for ticker in TICKERS:
    p = fetch_prices(ticker)
    price_data[ticker] = p
    src = "real" if HAS_YF else "synthetic"
    print(f"  {ticker}: {len(p)} bars ({src}), last=${float(p[-1]):.2f}")

# Vol estimation
print("\n=== VOLATILITY ESTIMATES (%/day) ===")
header = f"{'Ticker':6s} {'CC-21d':>8} {'GARCH':>8} {'GARCH-unc':>10} {'HAR-RV':>8}"
print(header)
vol_results = []
for ticker, p in price_data.items():
    r = log_returns(p)
    rv = r ** 2
    cc = float(np.std(r[-21:], ddof=1))
    try:
        g = GARCH11().fit(p)
        gv, gu = g.daily_vol, g.unconditional_vol
    except Exception:
        gv = gu = float("nan")
    try:
        har = HARRV().fit(rv)
        hv = har.daily_vol
    except Exception:
        hv = float("nan")
    vol_results.append(
        {"ticker": ticker, "cc": cc, "garch": gv, "garch_unc": gu, "har": hv, "price": p[-1]}
    )
    print(
        f"{ticker:6s} {cc*100:>8.3f} {gv*100:>8.3f} {gu*100:>10.3f} {hv*100:>8.3f}"
    )

# Vol bar chart
display_data = {
    "CC-21d": [r["cc"] * 100 for r in vol_results],
    "GARCH":  [r["garch"] * 100 for r in vol_results],
    "HAR-RV": [r["har"] * 100 for r in vol_results],
}
plot_df = pd.DataFrame(display_data, index=[r["ticker"] for r in vol_results])
fig, ax = plt.subplots(figsize=(10, 4.5))
plot_df.plot(kind="bar", ax=ax, colormap="tab10", rot=0)
ax.set_ylabel("Daily vol (%)", fontsize=12)
ax.set_title("Volatility Estimates: CC vs GARCH vs HAR-RV", fontsize=13)
ax.legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.savefig("data/fig_vol_comparison.png", dpi=150)
plt.close()
print("[7] fig_vol_comparison.png")

# Backtest
print(f"\n=== BACKTEST (X={X_SHARES:,} shares, λ={LAM_BT}, {N_SIMS_BT} sims) ===")
bt_rows = []
for vrow in vol_results:
    ticker = vrow["ticker"]
    price  = vrow["price"]
    sigma_bt = vrow["cc"]
    sigma_dollar = sigma_bt * price
    adv = ADV.get(ticker, 20e6)
    p_params = almgren_2005_params(sigma=sigma_dollar, adv=adv, price=price)
    eta_bt   = p_params["eta"]
    gamma_bt = p_params["gamma"]

    print(f"  {ticker} (σ={sigma_bt*100:.2f}%, η={eta_bt:.2e}, γ={gamma_bt:.2e})...", end=" ")
    bt_stats = compare_strategies(
        X=X_SHARES, T=T_DAYS, N=N_INT,
        sigma=sigma_bt, gamma=gamma_bt, eta=eta_bt, lam=LAM_BT,
        S0=price, n_sims=N_SIMS_BT, seed=SEED_BT,
    )
    print("done")
    for name, s in bt_stats.items():
        arr = s.is_bps_all
        bt_rows.append(
            {
                "Ticker": ticker,
                "Strategy": name,
                "Mean IS (bps)": round(s.mean_is_bps, 2),
                "Std IS (bps)":  round(s.std_is_bps, 2),
                "p5":  round(float(np.percentile(arr, 5)), 1),
                "p95": round(float(np.percentile(arr, 95)), 1),
                "Sharpe IS": round(s.sharpe_is, 3),
                "_is_all": arr,
            }
        )

bt_df = pd.DataFrame(bt_rows)
cols = ["Ticker", "Strategy", "Mean IS (bps)", "Std IS (bps)", "p5", "p95", "Sharpe IS"]
print(bt_df[cols].to_string(index=False))

# Boxplot
fig, axes = plt.subplots(1, len(TICKERS), figsize=(16, 5), sharey=False)
for ax, ticker in zip(axes, TICKERS):
    ticker_rows = [r for r in bt_rows if r["Ticker"] == ticker]
    data   = [r["_is_all"] for r in ticker_rows]
    labels = [r["Strategy"] for r in ticker_rows]
    ax.boxplot(
        data, labels=labels, widths=0.5, patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        medianprops=dict(color="black", lw=2),
    )
    ax.set_title(ticker, fontsize=12)
    if ax == axes[0]:
        ax.set_ylabel("IS (bps)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
fig.suptitle(
    f"IS Distribution — {N_SIMS_BT} simulations per ticker (λ={LAM_BT})",
    fontsize=13, y=1.01,
)
plt.tight_layout()
plt.savefig("data/fig_backtest_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("[8] fig_backtest_boxplots.png")

print("\nAll done.")
