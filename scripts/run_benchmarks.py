"""Reproduce every number and figure quoted in the README.

    python scripts/run_benchmarks.py            # ~1-2 minutes

Writes docs/results.json and docs/figures/*.png. Seeds are fixed, so the
output is identical on every run (timings excepted).
"""

from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mcquant import viz  # noqa: E402
from mcquant.american import lsm_price  # noqa: E402
from mcquant.analytic import binomial_price, bs_price, heston_price, implied_vol  # noqa: E402
from mcquant.data import estimate_parameters, log_returns, synthetic_market  # noqa: E402
from mcquant.models import GBM, Heston  # noqa: E402
from mcquant.payoffs import AsianArithmetic, European  # noqa: E402
from mcquant.pricer import geometric_asian_control, mc_price  # noqa: E402
from mcquant.risk import (  # noqa: E402
    Portfolio,
    Position,
    delta_normal_var,
    monte_carlo_var,
    rolling_var_backtest,
)
from mcquant.rng import mean_and_se  # noqa: E402
from mcquant.validation import run_benchmarks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
viz.use_matplotlib_style()
results: dict = {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine()}


def save(fig, name: str) -> None:
    fig.savefig(FIG / name)
    plt.close(fig)
    print("  wrote", name)


# --------------------------------------------------------------------------- #
print("1. Variance reduction on an arithmetic Asian call (52 weekly fixings)")
gbm = GBM(100, 0.05, 0.2)
asian = AsianArithmetic(100)
geo_cv = geometric_asian_control(gbm, 100)
configs = [("pseudo", None), ("antithetic", None), ("sobol", None), ("pseudo", geo_cv), ("sobol", geo_cv)]
sizes = [2**k for k in range(10, 18)]
rows = []
for method, cv in configs:
    label = method + ("+cv" if cv else "")
    for n in sizes:
        mc_price(gbm, asian, 1, 52, 2**10, method, 1, cv)  # warm-up
        reps = [mc_price(gbm, asian, 1, 52, n, method, seed, cv) for seed in (11, 12, 13)]
        rows.append(
            {
                "method": label,
                "paths": n,
                "std error": float(np.mean([r.std_error for r in reps])),
                "seconds": float(np.median([r.elapsed for r in reps])),
                "price": reps[0].price,
            }
        )
conv = pd.DataFrame(rows)
top = conv[conv.paths == sizes[-1]].set_index("method")
base = top.loc["pseudo"]
top["variance_reduction"] = (base["std error"] / top["std error"]) ** 2
top["efficiency_gain"] = top["variance_reduction"] * base["seconds"] / top["seconds"]
results["asian_variance_reduction"] = (
    top[["price", "std error", "variance_reduction", "efficiency_gain", "seconds"]].round(6).to_dict("index")
)
print(top[["std error", "variance_reduction", "efficiency_gain"]].round(2))

fig, ax = plt.subplots(figsize=(8, 4.8))
for label, d in conv.groupby("method", sort=False):
    ax.plot(d["paths"], d["std error"], marker="o", ms=5, label=label, color=viz.METHOD_COLORS[label])
ref = conv[conv.method == "pseudo"].iloc[0]
xs = np.array(sizes)
ax.plot(xs, ref["std error"] * np.sqrt(ref["paths"] / xs), ls="--", lw=1, color=viz.MUTED, label=r"$1/\sqrt{n}$")
ax.set(
    xscale="log",
    yscale="log",
    xlabel="Paths",
    ylabel="Standard error",
    title="Arithmetic Asian call: standard error by sampling method",
)
ax.set_xticks(sizes, [f"{n // 1024}k" for n in sizes])
ax.minorticks_off()
ax.legend(ncol=2, loc="upper right", fontsize=9)
save(fig, "variance_reduction.png")

# --------------------------------------------------------------------------- #
print("2. Validation panel")
val = run_benchmarks(n_paths=2**17, seed=2024)
results["validation"] = {
    "passed": int(val["pass"].sum()),
    "total": len(val),
    "max_abs_z": float(val["error / SE"].abs().max()),
    "rows": val.drop(columns=["seconds"]).round(6).to_dict("records"),
}
print(val[["benchmark", "error / SE", "pass"]])
fig, ax = plt.subplots(figsize=(8, 5))
y = np.arange(len(val))[::-1]
ax.axvspan(-2, 2, color=viz.GRID, alpha=0.6, lw=0, label="±2 SE (95% of runs)")
ax.axvline(0, color=viz.AXIS, lw=1)
for lim in (-4, 4):
    ax.axvline(lim, color=viz.MUTED, lw=1, ls="--")
ax.scatter(val["error / SE"], y, s=50, color=viz.BLUE, zorder=3, label="Monte Carlo vs exact")
ax.set_yticks(y, val["benchmark"])
ax.set(
    xlim=(-5, 5),
    xlabel="Error in standard errors  (pass band ±4)",
    title=f"Every Monte Carlo price vs an exact reference: {int(val['pass'].sum())}/{len(val)} pass",
)
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right")
save(fig, "validation.png")

# --------------------------------------------------------------------------- #
print("3. Heston implied-volatility smile: Monte Carlo vs semi-analytic")
hp = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.6, rho=-0.7)
hm = Heston(100, 0.03, **hp)
strikes = np.arange(70, 135, 5.0)
paths = hm.simulate(1.0, 200, 2**17, "antithetic", seed=7)
smile = []
for k in strikes:
    kind = "put" if k < 100 else "call"
    exact = heston_price(100, k, 1, 0.03, kind=kind, **hp)
    x = paths.discount * European(k, kind)(paths.spot)
    mc, se = mean_and_se(x, "antithetic")

    def iv(price, k=k, kind=kind):
        return implied_vol(price, 100, k, 1, 0.03, kind=kind)

    smile.append(
        {"K": k, "iv_exact": iv(exact), "iv_mc": iv(mc), "iv_lo": iv(mc - 1.96 * se), "iv_hi": iv(mc + 1.96 * se)}
    )
smile = pd.DataFrame(smile)
results["heston_smile_max_iv_error_bp"] = float((smile.iv_mc - smile.iv_exact).abs().max() * 1e4)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(smile.K, smile.iv_exact * 100, color=viz.BLUE, label="Semi-analytic (characteristic function)")
ax.errorbar(
    smile.K,
    smile.iv_mc * 100,
    yerr=[(smile.iv_mc - smile.iv_lo) * 100, (smile.iv_hi - smile.iv_mc) * 100],
    fmt="o",
    ms=6,
    color=viz.ORANGE,
    ecolor=viz.ORANGE,
    capsize=3,
    label="Monte Carlo, 95% CI",
)
ax.axhline(20, color=viz.MUTED, lw=1, ls="--")
ax.text(131, 20.25, "flat Black-Scholes vol", color=viz.MUTED, ha="right", fontsize=9)
ax.set(xlabel="Strike", ylabel="Implied volatility (%)", title="Heston volatility skew (rho = -0.7), one-year options")
ax.legend()
save(fig, "heston_smile.png")

# --------------------------------------------------------------------------- #
print("4. American put: Longstaff-Schwartz grid")
grid = []
for s0 in (36, 38, 40, 42, 44):
    for sig in (0.2, 0.4):
        for T in (1, 2):
            n_ex = 50 * T
            r = lsm_price(GBM(s0, 0.06, sig), 40, T, "put", n_ex, 100_000, seed=3)
            tree = binomial_price(s0, 40, T, 0.06, sig, n_steps=50 * n_ex, n_exercise=n_ex)
            grid.append(
                {
                    "S0": s0,
                    "sigma": sig,
                    "T": T,
                    "lsm": r.price,
                    "se": r.std_error,
                    "bermudan_tree": tree,
                    "european": bs_price(s0, 40, T, 0.06, sig, kind="put"),
                    "diff": r.price - tree,
                }
            )
grid = pd.DataFrame(grid)
results["lsm_grid"] = {
    "cases": len(grid),
    "max_abs_diff": float(grid["diff"].abs().max()),
    "mean_abs_diff": float(grid["diff"].abs().mean()),
    "rows": grid.round(4).to_dict("records"),
}
print(grid.round(4).to_string())
r = lsm_price(GBM(36, 0.06, 0.2), 40, 1, "put", 50, 200_000, seed=5)
fig, ax = plt.subplots(figsize=(8, 4.2))
ok = ~np.isnan(r.boundary_spots)
ax.plot(r.boundary_times[ok], r.boundary_spots[ok], marker="o", ms=4, color=viz.BLUE)
ax.fill_between(r.boundary_times[ok], 30, r.boundary_spots[ok], color=viz.BLUE, alpha=0.08)
ax.axhline(40, color=viz.INK_2, lw=1, ls="--")
ax.text(0.5, 40.2, "strike", color=viz.INK_2, fontsize=9, ha="center")
ax.text(0.5, 33.2, "exercise now (below the boundary)", color=viz.BLUE, ha="center")
ax.text(0.5, 38.6, "keep holding", color=viz.INK_2, ha="center")
tree_36 = binomial_price(36, 40, 1, 0.06, 0.2, n_steps=2500, n_exercise=50)
ax.set(
    ylim=(32, 41),
    xlabel="Time (years)",
    ylabel="Spot price",
    title=f"American put (S=36, K=40): LSM {r.price:.3f} vs tree {tree_36:.3f}",
)
save(fig, "lsm_boundary.png")

# --------------------------------------------------------------------------- #
print("5. Risk: option book, fat tails, backtest")
prices = synthetic_market(seed=42)
mu, cov = estimate_parameters(prices)
spots = prices.iloc[-1].to_numpy()
vols = np.sqrt(np.diag(cov))
# A short-volatility book: sold at-the-money straddles. Almost no delta, a lot of gamma.
straddle = Portfolio(
    spots,
    [
        Position(0, -2_000, "call", spots[0], 0.25, vols[0]),
        Position(0, -2_000, "put", spots[0], 0.25, vols[0]),
    ],
    r=0.03,
    names=list(prices.columns),
)
dn = delta_normal_var(straddle, mu, cov, 0.99, 10)
mcn = monte_carlo_var(straddle, mu, cov, 0.99, 10, 200_000, seed=1)
mct = monte_carlo_var(straddle, mu, cov, 0.99, 10, 200_000, "t", 4, seed=1)
results["short_vol_book"] = {
    "value": straddle.current_value,
    "net_dollar_delta": float(straddle.dollar_deltas().sum()),
    "delta_normal_var": dn.var,
    "mc_normal_var": mcn.var,
    "mc_t_var": mct.var,
    "mc_normal_cvar": mcn.cvar,
    "mc_t_cvar": mct.cvar,
    "understatement_x": mcn.var / dn.var,
}
print({k: round(v, 1) for k, v in results["short_vol_book"].items()})
fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(-80_000, mct.pnl.max(), 140) / 1e3
ax.hist(mcn.pnl / 1e3, bins=bins, color=viz.BLUE, alpha=0.55, label="Monte Carlo, normal shocks")
ax.hist(mct.pnl / 1e3, bins=bins, color=viz.ORANGE, alpha=0.55, label="Monte Carlo, Student-t (df=4)")
ax.set_yscale("log")
for v, c, txt, yy in (
    (dn.var, viz.INK_2, "delta-normal", 3e4),
    (mcn.var, viz.BLUE, "MC normal", 3e4),
    (mct.var, viz.ORANGE, "MC Student-t", 3e4),
):
    ax.axvline(-v / 1e3, color=c, lw=2, ls="--")
    ax.text(-v / 1e3 - 1, yy, f"{txt}\nVaR {v / 1e3:,.0f}k", color=viz.INK_2, fontsize=8.5, ha="right", va="top")
ax.set(
    xlim=(-80, mct.pnl.max() / 1e3 + 2),
    xlabel="10-day P&L (thousands)",
    ylabel="Scenarios (log scale)",
    title=f"Sold straddles: delta-normal VaR understates risk {mcn.var / dn.var:.1f}x",
)
ax.legend(loc="center left")
save(fig, "option_book_var.png")

w = np.array([0.3, 0.3, 0.25, 0.15])
port = pd.Series(np.expm1(log_returns(prices)).to_numpy() @ w, index=log_returns(prices).index)
bt = rolling_var_backtest(port, 0.99, 250)
results["backtest"] = bt.summary.round(4).reset_index().to_dict("records")
print(bt.summary[["exceptions", "expected", "p_pof", "p_ind", "basel_zone", "verdict"]])
fc = bt.forecasts
fig, ax = plt.subplots(figsize=(9, 4.3))
ax.scatter(fc.index, fc["return"] * 100, s=4, color=viz.AXIS, label="Daily return")
for m, c in (("historical", viz.ORANGE), ("fhs", viz.BLUE)):
    n_exc = int(bt.summary.loc[m, "exceptions"])
    ax.plot(fc.index, -fc[m] * 100, color=c, lw=1.6, label=f"-VaR {m} ({n_exc} breaches)")
exc = fc["return"] < -fc["historical"]
ax.scatter(
    fc.index[exc],
    fc["return"][exc] * 100,
    marker="x",
    s=30,
    color=viz.CRITICAL,
    label="Breach of historical VaR",
    zorder=3,
)
ax.set(ylabel="Return (%)", title=f"99% VaR backtest, {len(fc):,} days (expected breaches: {len(fc) * 0.01:.0f})")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=9)
save(fig, "backtest.png")

# --------------------------------------------------------------------------- #
print("6. Performance: vectorised NumPy vs a pure-Python loop (Asian call, 52 fixings)")


def python_loop(n, n_steps=52, s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0, seed=0):
    """The straightforward way to write it: one path at a time, one step at a time."""
    import random

    rnd = random.Random(seed)
    dt = T / n_steps
    drift, vol = (r - 0.5 * sigma**2) * dt, sigma * math.sqrt(dt)
    total = 0.0
    for _ in range(n):
        s, acc = s0, 0.0
        for _ in range(n_steps):
            s *= math.exp(drift + vol * rnd.gauss(0.0, 1.0))
            acc += s
        total += max(acc / n_steps - k, 0.0)
    return math.exp(-r * T) * total / n


n = 100_000
t0 = time.perf_counter()
loop_price = python_loop(n)
t_loop = time.perf_counter() - t0
t0 = time.perf_counter()
vec = mc_price(gbm, asian, 1, 52, n, "pseudo", seed=0)
t_vec = time.perf_counter() - t0
assert abs(loop_price - vec.price) < 5 * vec.std_error, "loop and vectorised engines disagree"
t0 = time.perf_counter()
GBM(100, 0.05, 0.2).simulate(1, 252, 100_000, seed=0)
t_paths = time.perf_counter() - t0
results["performance"] = {
    "benchmark": "arithmetic Asian call, 100k paths x 52 steps",
    "python_loop_s": t_loop,
    "numpy_s": t_vec,
    "speedup_x": t_loop / t_vec,
    "gbm_252x100k_s": t_paths,
    "path_steps_per_s": 252 * 100_000 / t_paths,
}
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in results["performance"].items()})

(ROOT / "docs" / "results.json").write_text(json.dumps(results, indent=2, default=float))
print("wrote docs/results.json")
