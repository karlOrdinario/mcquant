"""mcquant dashboard - interactive Monte Carlo pricing & risk.

Run locally:   streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow `streamlit run app/streamlit_app.py` from a fresh clone without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcquant import viz  # noqa: E402
from mcquant.american import lsm_price  # noqa: E402
from mcquant.analytic import (  # noqa: E402
    barrier_price,
    binomial_price,
    bs_digital_delta,
    bs_digital_price,
    bs_greeks,
    bs_price,
    geometric_asian_price,
    heston_price,
    merton_price,
)
from mcquant.data import estimate_parameters, load_csv, log_returns, synthetic_market  # noqa: E402
from mcquant.greeks import (  # noqa: E402
    fd_greeks,
    likelihood_ratio_delta,
    likelihood_ratio_gamma,
    pathwise_delta,
    pathwise_vega,
)
from mcquant.models import GBM, Heston, Merton  # noqa: E402
from mcquant.payoffs import AsianArithmetic, Barrier, Digital, European  # noqa: E402
from mcquant.pricer import geometric_asian_control, mc_price, stock_control  # noqa: E402
from mcquant.risk import (  # noqa: E402
    DEFAULT_STRESS,
    Portfolio,
    Position,
    delta_normal_var,
    historical_var,
    monte_carlo_var,
    rolling_var_backtest,
    stress_test,
)
from mcquant.validation import run_benchmarks  # noqa: E402

st.set_page_config(page_title="mcquant - Monte Carlo pricing & risk", layout="wide")
TEMPLATE = viz.plotly_template()
MAX_CELLS = 30_000_000  # paths x steps x factors, keeps peak memory well under 1 GB

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("mcquant")
st.sidebar.caption(
    "A validated Monte Carlo engine for derivatives pricing, portfolio risk and company valuation. "
    "The sidebar settings apply to the first six tabs."
)
st.sidebar.subheader("Simulation")
N_PATHS = st.sidebar.select_slider(
    "Paths", options=[2**12, 2**14, 2**15, 2**16, 2**17, 2**18], value=2**16, format_func=lambda n: f"{n:,}"
)
METHOD = st.sidebar.radio(
    "Sampling",
    ["pseudo", "antithetic", "sobol"],
    index=1,
    format_func={"pseudo": "Pseudo-random", "antithetic": "Antithetic", "sobol": "Sobol (randomised QMC)"}.get,
    help="Antithetic pairs every draw Z with -Z. Sobol uses 32 independently scrambled low-discrepancy "
    "sequences; the standard error is measured across them.",
)
SEED = int(st.sidebar.number_input("Seed", value=2024, step=1))
st.sidebar.caption("Every result shows its standard error. Same seed, same answer.")


def fig_layout(fig: go.Figure, title: str, x: str, y: str, height: int = 360) -> go.Figure:
    fig.update_layout(template=TEMPLATE, title=title, xaxis_title=x, yaxis_title=y, height=height)
    return fig


def guard_size(n_paths: int, n_steps: int, factors: int = 1) -> int:
    """Reduce paths if the request would use too much memory."""
    if n_paths * n_steps * factors <= MAX_CELLS:
        return n_paths
    capped = 2 ** int(math.log2(MAX_CELLS // (n_steps * factors)))
    st.info(f"Using {capped:,} paths for {n_steps} time steps to stay within memory limits.")
    return capped


def show(df: pd.DataFrame, **fmt) -> None:
    st.dataframe(df.style.format(fmt, na_rep="-"))


# --------------------------------------------------------------------------- #
# Cached computations (primitive args -> hashable)
# --------------------------------------------------------------------------- #
def build_model(name: str, p: dict):
    if name == "GBM (Black-Scholes)":
        return GBM(p["s0"], p["r"], p["sigma"], p["q"])
    if name == "Heston (stochastic vol)":
        return Heston(p["s0"], p["r"], p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"], p["q"])
    return Merton(p["s0"], p["r"], p["sigma"], p["lam"], p["mu_j"], p["sigma_j"], p["q"])


def build_payoff(o: dict):
    if o["type"] == "European":
        return European(o["K"], o["kind"])
    if o["type"] == "Asian (arithmetic)":
        return AsianArithmetic(o["K"], o["kind"])
    if o["type"] == "Barrier":
        return Barrier(o["K"], o["H"], o["btype"], o["kind"])
    return Digital(o["K"], o["kind"])


def reference_price(model_name: str, p: dict, o: dict, n_steps: int) -> tuple[float | None, str]:
    gbm = model_name.startswith("GBM")
    if o["type"] == "European":
        if gbm:
            return bs_price(p["s0"], o["K"], o["T"], p["r"], p["sigma"], p["q"], o["kind"]), "Black-Scholes"
        if model_name.startswith("Heston"):
            args = (p["s0"], o["K"], o["T"], p["r"], p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"], p["q"])
            return heston_price(*args, kind=o["kind"]), "Heston semi-analytic"
        args = (p["s0"], o["K"], o["T"], p["r"], p["sigma"], p["lam"], p["mu_j"], p["sigma_j"], p["q"])
        return merton_price(*args, kind=o["kind"]), "Merton series"
    if not gbm:
        return None, "no closed form"
    if o["type"] == "Digital":
        return bs_digital_price(p["s0"], o["K"], o["T"], p["r"], p["sigma"], p["q"], o["kind"]), "closed form"
    if o["type"] == "Barrier":
        ref = barrier_price(p["s0"], o["K"], o["H"], o["T"], p["r"], p["sigma"], o["btype"], o["kind"], p["q"], n_steps)
        return ref, "BGK-corrected formula"
    return None, "no closed form (geometric Asian is the control)"


@st.cache_data(show_spinner=False, max_entries=32)
def run_pricer(model_name, p_items, o_items, n_steps, n_paths, method, seed, use_cv):
    p, o = dict(p_items), dict(o_items)
    model, payoff = build_model(model_name, p), build_payoff(o)
    control = None
    if use_cv:
        if o["type"] == "Asian (arithmetic)" and model_name.startswith("GBM"):
            control = geometric_asian_control(model, o["K"], o["kind"])
        else:
            control = stock_control(model)
    res = mc_price(model, payoff, o["T"], n_steps, n_paths, method, seed, control)
    # For the pictures use at least ~100 steps so paths look like paths (pricing used n_steps).
    viz_steps = n_steps if n_steps >= 100 else n_steps * math.ceil(100 / n_steps)
    pic = model.simulate(o["T"], viz_steps, 4096, "pseudo", seed)
    q = np.quantile(pic.spot, [0.05, 0.5, 0.95], axis=0)
    terminal = model.simulate(o["T"], n_steps, min(n_paths, 2**15), "pseudo", seed).spot[:, -1]
    return res, pic.times, pic.spot[:60], q, terminal


@st.cache_data(show_spinner=False, max_entries=16)
def run_convergence(model_name, p_items, o_items, n_steps, max_paths, seed, use_cv):
    p, o = dict(p_items), dict(o_items)
    model, payoff = build_model(model_name, p), build_payoff(o)
    rows = []
    sizes = [2**k for k in range(10, int(math.log2(max_paths)) + 1)]
    for method in ("pseudo", "antithetic", "sobol"):
        for cv in (False, True) if use_cv else (False,):
            control = None
            if cv:
                is_asian_gbm = o["type"] == "Asian (arithmetic)" and model_name.startswith("GBM")
                control = geometric_asian_control(model, o["K"], o["kind"]) if is_asian_gbm else stock_control(model)
            for n in sizes:
                r = mc_price(model, payoff, o["T"], n_steps, n, method, seed, control)
                rows.append(
                    {
                        "method": method + ("+cv" if cv else ""),
                        "paths": n,
                        "std error": r.std_error,
                        "price": r.price,
                        "seconds": r.elapsed,
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
tabs = st.tabs(
    [
        "Option pricer",
        "American options",
        "Greeks",
        "Portfolio risk",
        "VaR backtest",
        "Validation",
        "Valuation (Monde Nissin)",
    ]
)

# =============================== Option pricer ============================== #
with tabs[0]:
    st.header("Option pricer")
    st.caption("Price an option by simulation and compare against an exact answer where one exists.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Market model")
        model_name = st.selectbox(
            "Model", ["GBM (Black-Scholes)", "Heston (stochastic vol)", "Merton (jump-diffusion)"]
        )
        p = {
            "s0": st.number_input("Spot S0", 1.0, 10_000.0, 100.0),
            "r": st.number_input("Risk-free rate r", -0.05, 0.3, 0.05, 0.005, format="%.3f"),
            "q": st.number_input("Dividend yield q", 0.0, 0.2, 0.0, 0.005, format="%.3f"),
        }
    with c2:
        st.subheader("Model parameters")
        if model_name.startswith("GBM"):
            p["sigma"] = st.slider("Volatility sigma", 0.05, 1.0, 0.20, 0.01)
        elif model_name.startswith("Heston"):
            p["v0"] = st.slider("Initial variance v0", 0.005, 0.25, 0.04, 0.005, format="%.3f")
            p["theta"] = st.slider("Long-run variance theta", 0.005, 0.25, 0.04, 0.005, format="%.3f")
            p["kappa"] = st.slider("Mean reversion kappa", 0.1, 10.0, 1.5, 0.1)
            p["xi"] = st.slider("Vol of vol xi", 0.05, 1.5, 0.5, 0.05)
            p["rho"] = st.slider("Spot-vol correlation rho", -0.95, 0.95, -0.7, 0.05)
            if 2 * p["kappa"] * p["theta"] < p["xi"] ** 2:
                st.caption("Feller condition violated: variance can touch zero. Use more time steps.")
        else:
            p["sigma"] = st.slider("Diffusion volatility sigma", 0.05, 1.0, 0.20, 0.01)
            p["lam"] = st.slider("Jumps per year lambda", 0.0, 5.0, 0.75, 0.05)
            p["mu_j"] = st.slider("Mean log jump", -0.5, 0.3, -0.10, 0.01)
            p["sigma_j"] = st.slider("Jump volatility", 0.0, 0.5, 0.15, 0.01)
    with c3:
        st.subheader("Contract")
        o = {"type": st.selectbox("Option type", ["European", "Asian (arithmetic)", "Barrier", "Digital"])}
        o["kind"] = st.radio("Call / put", ["call", "put"], horizontal=True)
        o["K"] = st.number_input("Strike K", 1.0, 10_000.0, 100.0)
        o["T"] = st.number_input("Maturity T (years)", 0.02, 10.0, 1.0, 0.25)
        if o["type"] == "Barrier":
            o["btype"] = st.selectbox("Barrier type", ["up-and-out", "up-and-in", "down-and-out", "down-and-in"])
            o["H"] = st.number_input("Barrier H", 1.0, 10_000.0, 125.0 if o["btype"].startswith("up") else 85.0)
    path_dep = o["type"] in ("Asian (arithmetic)", "Barrier") or model_name.startswith("Heston")
    default_steps = (
        252 if o["type"] == "Barrier" else (52 if o["type"].startswith("Asian") else (100 if path_dep else 1))
    )
    n_steps = st.slider(
        "Time steps (monitoring dates)",
        1,
        504,
        default_steps,
        help="European options under GBM/Merton are exact with 1 step.",
    )
    use_cv = st.toggle(
        "Control variate",
        value=True,
        help="Geometric Asian for Asian options under GBM, otherwise the discounted stock price.",
    )

    factors = 2 if model_name.startswith("Heston") else 1
    n_run = guard_size(N_PATHS, n_steps, factors)
    try:
        with st.spinner("Simulating..."):
            res, times, sample, bands, terminal = run_pricer(
                model_name, tuple(p.items()), tuple(o.items()), n_steps, n_run, METHOD, SEED, use_cv
            )
        ref, ref_name = reference_price(model_name, p, o, n_steps)
    except ValueError as e:
        st.error(f"Invalid input: {e}")
        st.stop()

    m = st.columns(4)
    m[0].metric("Monte Carlo price", f"{res.price:,.4f}")
    m[1].metric("Standard error", f"{res.std_error:,.4f}", help=f"95% CI {res.ci()[0]:,.4f} to {res.ci()[1]:,.4f}")
    if ref is not None:
        z = (res.price - ref) / res.std_error if res.std_error > 0 else 0.0
        m[2].metric(f"Reference ({ref_name})", f"{ref:,.4f}")
        m[3].metric("Error in standard errors", f"{z:+.2f}", help="Within about +/-2 is what chance alone produces.")
    else:
        m[2].metric("Reference", ref_name)
        if o["type"].startswith("Asian") and model_name.startswith("GBM"):
            geo = geometric_asian_price(p["s0"], o["K"], o["T"], p["r"], p["sigma"], n_steps, p["q"], o["kind"])
            m[3].metric("Geometric Asian (exact)", f"{geo:,.4f}")
    st.caption(f"{res.n_paths:,} paths, {n_steps} steps, {res.method}, {res.elapsed * 1000:,.0f} ms")

    g1, g2 = st.columns([3, 2])
    with g1:
        fig = go.Figure()
        for i, row in enumerate(sample):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=row,
                    mode="lines",
                    line=dict(color=viz.BLUE_LIGHT, width=1),
                    opacity=0.35,
                    hoverinfo="skip",
                    showlegend=i == 0,
                    name="Sample paths",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=bands[2],
                mode="lines",
                line=dict(color=viz.BLUE, width=2, dash="dot"),
                name="5th / 95th percentile",
            )
        )
        fig.add_trace(go.Scatter(x=times, y=bands[1], mode="lines", line=dict(color=viz.BLUE, width=2), name="Median"))
        fig.add_trace(
            go.Scatter(
                x=times,
                y=bands[0],
                mode="lines",
                line=dict(color=viz.BLUE, width=2, dash="dot"),
                showlegend=False,
                name="5th percentile",
            )
        )
        fig.add_hline(
            y=o["K"],
            line=dict(color=viz.INK_2, width=1, dash="dash"),
            annotation_text="Strike",
            annotation_position="bottom left",
            annotation_font_color=viz.INK_2,
        )
        if o["type"] == "Barrier":
            fig.add_hline(
                y=o["H"],
                line=dict(color=viz.ORANGE, width=2),
                annotation_text="Barrier",
                annotation_font_color=viz.INK_2,
            )
        st.plotly_chart(fig_layout(fig, "Simulated price paths", "Time (years)", "Price"))
    with g2:
        fig = go.Figure(go.Histogram(x=terminal, nbinsx=60, marker_color=viz.BLUE, name="S_T"))
        fig.add_vline(
            x=o["K"],
            line=dict(color=viz.INK_2, width=1, dash="dash"),
            annotation_text="Strike",
            annotation_font_color=viz.INK_2,
        )
        fig.update_layout(hovermode="closest", showlegend=False)
        st.plotly_chart(fig_layout(fig, "Distribution of the price at maturity", "S_T", "Paths"))

    st.subheader("Variance-reduction race")
    st.caption(
        "Standard error vs number of paths, every sampler on the same contract. Plain Monte Carlo falls as 1/sqrt(n); "
        "a 10x smaller error at the same n is worth 100x more paths."
    )
    if st.button("Run convergence study", type="primary"):
        with st.spinner("Racing samplers..."):
            df = run_convergence(
                model_name, tuple(p.items()), tuple(o.items()), n_steps, min(n_run, 2**16), SEED, use_cv
            )
        fig = go.Figure()
        for meth, d in df.groupby("method", sort=False):
            fig.add_trace(
                go.Scatter(
                    x=d["paths"],
                    y=d["std error"],
                    mode="lines+markers",
                    name=meth,
                    line=dict(color=viz.METHOD_COLORS[meth], width=2),
                    marker=dict(size=8),
                )
            )
        base = df[df.method == "pseudo"].iloc[0]
        xs = df["paths"].unique()
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=base["std error"] * np.sqrt(base["paths"] / xs),
                mode="lines",
                line=dict(color=viz.MUTED, width=1, dash="dash"),
                name="1/sqrt(n) reference",
            )
        )
        fig.update_xaxes(type="log", tickvals=list(xs), ticktext=[f"{n // 1024}k" for n in xs])
        fig.update_yaxes(type="log", dtick=1, exponentformat="power")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(
            fig_layout(fig, "Standard error by sampling method", "Paths (log scale)", "Standard error (log scale)", 420)
        )
        last = df[df.paths == df.paths.max()].set_index("method")
        pse = last.loc["pseudo"]
        last["variance reduction x"] = (pse["std error"] / last["std error"]) ** 2
        last["efficiency gain x"] = last["variance reduction x"] * pse["seconds"] / last["seconds"]
        show(
            last[["price", "std error", "variance reduction x", "efficiency gain x", "seconds"]],
            **{
                "price": "{:.4f}",
                "std error": "{:.5f}",
                "variance reduction x": "{:,.1f}",
                "efficiency gain x": "{:,.1f}",
                "seconds": "{:.3f}",
            },
        )
        st.caption(
            "Efficiency gain = variance reduction x time ratio: how many times faster you reach a target accuracy."
        )

# ============================== American options ============================ #
with tabs[1]:
    st.header("American options: Longstaff-Schwartz")
    st.caption(
        "At each exercise date, regress the discounted future cash flow on polynomials of the spot price. "
        "Exercise when today's payoff beats the fitted continuation value. Checked against a binomial tree "
        "restricted to the same exercise dates."
    )
    a1, a2 = st.columns([1, 2])
    with a1:
        s0 = st.number_input("Spot", 1.0, 1000.0, 36.0, key="a_s")
        k = st.number_input("Strike", 1.0, 1000.0, 40.0, key="a_k")
        r = st.number_input("Rate", 0.0, 0.3, 0.06, 0.01, key="a_r")
        sig = st.slider("Volatility", 0.05, 1.0, 0.2, 0.01, key="a_sig")
        t = st.number_input("Maturity (years)", 0.1, 5.0, 1.0, 0.25, key="a_t")
        kind = st.radio("Type", ["put", "call"], horizontal=True, key="a_kind")
        n_ex = st.slider("Exercise dates", 10, 200, 50, 10)
        degree = st.slider("Regression polynomial degree", 1, 6, 3)
    try:
        model_a = GBM(s0, r, sig)
        n_a = guard_size(N_PATHS, n_ex)
        with st.spinner("Running Longstaff-Schwartz..."):
            lsm = lsm_price(model_a, k, t, kind, n_ex, n_a, degree, METHOD, SEED)
        steps_tree = n_ex * max(1, 2000 // n_ex)
        berm = binomial_price(s0, k, t, r, sig, kind=kind, n_steps=steps_tree, n_exercise=n_ex)
        amer = binomial_price(s0, k, t, r, sig, kind=kind, n_steps=2000)
        euro = bs_price(s0, k, t, r, sig, kind=kind)
    except ValueError as e:
        st.error(f"Invalid input: {e}")
        st.stop()
    with a2:
        mm = st.columns(4)
        mm[0].metric("LSM price", f"{lsm.price:.4f}", help=f"+/- {lsm.std_error:.4f} standard error")
        mm[1].metric(
            "Bermudan tree",
            f"{berm:.4f}",
            delta=f"LSM {lsm.price - berm:+.4f}",
            delta_color="off",
            help="Binomial tree allowed to exercise only on the same dates as LSM",
        )
        mm[2].metric("American tree", f"{amer:.4f}")
        mm[3].metric("European (Black-Scholes)", f"{euro:.4f}")
        st.caption(
            f"Early-exercise premium: {lsm.price - euro:.4f}. Out-of-sample LSM is a lower bound: the regression rule "
            "is fitted on independent paths, so the error bar is honest."
        )
        b = pd.DataFrame({"t": lsm.boundary_times, "S*": lsm.boundary_spots}).dropna()
        fig = go.Figure(
            go.Scatter(
                x=b["t"],
                y=b["S*"],
                mode="lines+markers",
                line=dict(color=viz.BLUE, width=2),
                marker=dict(size=8),
                name="Exercise boundary",
            )
        )
        fig.add_hline(
            y=k,
            line=dict(color=viz.INK_2, dash="dash", width=1),
            annotation_text="Strike",
            annotation_font_color=viz.INK_2,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(
            fig_layout(
                fig,
                "Estimated early-exercise boundary",
                "Time (years)",
                "Exercise if spot is below" if kind == "put" else "Exercise if spot is above",
            )
        )

# ================================== Greeks ================================== #
with tabs[2]:
    st.header("Greeks: three Monte Carlo estimators")
    st.caption(
        "Pathwise differentiates the payoff along each path; likelihood ratio differentiates the probability "
        "density; finite differences bump the input and reuse the same random numbers (CRN)."
    )
    gc1, gc2 = st.columns([1, 2])
    with gc1:
        gs = st.number_input("Spot", 1.0, 1000.0, 100.0, key="g_s")
        gk = st.number_input("Strike", 1.0, 1000.0, 100.0, key="g_k")
        gt = st.number_input("Maturity", 0.05, 5.0, 1.0, 0.25, key="g_t")
        gr = st.number_input("Rate", 0.0, 0.3, 0.05, 0.01, key="g_r")
        gv = st.slider("Volatility", 0.05, 1.0, 0.2, 0.01, key="g_v")
        gkind = st.radio("Type", ["call", "put"], horizontal=True, key="g_kind")
        digital = st.toggle("Digital option (discontinuous payoff)", value=False)
    gm = GBM(gs, gr, gv)
    pay = Digital(gk, gkind) if digital else European(gk, gkind)
    exact = bs_greeks(gs, gk, gt, gr, gv, kind=gkind)
    rows = []
    fd = fd_greeks(gm, pay, gt, 1, N_PATHS, METHOD, SEED)
    lrd, lrg = (
        likelihood_ratio_delta(gm, pay, gt, N_PATHS, METHOD, SEED),
        likelihood_ratio_gamma(gm, pay, gt, N_PATHS, METHOD, SEED),
    )
    if digital:
        exact_delta = bs_digital_delta(gs, gk, gt, gr, gv, kind=gkind)
        rows.append(
            ["Delta", exact_delta, None, None, lrd.value, lrd.std_error, fd["delta"].value, fd["delta"].std_error]
        )
        note = "Pathwise is invalid here: the payoff is a step, so its derivative is zero on almost every path."
    else:
        pwd = pathwise_delta(gm, pay, gt, 1, N_PATHS, METHOD, SEED)
        pwv = pathwise_vega(gm, gk, gt, gkind, N_PATHS, METHOD, SEED)
        rows += [
            [
                "Delta",
                exact.delta,
                pwd.value,
                pwd.std_error,
                lrd.value,
                lrd.std_error,
                fd["delta"].value,
                fd["delta"].std_error,
            ],
            ["Gamma", exact.gamma, None, None, lrg.value, lrg.std_error, fd["gamma"].value, fd["gamma"].std_error],
            ["Vega", exact.vega, pwv.value, pwv.std_error, None, None, fd["vega"].value, fd["vega"].std_error],
        ]
        note = "Pathwise has the smallest error where it applies; likelihood ratio works for any payoff but is noisier."
    cols = [
        "Greek",
        "Exact",
        "Pathwise",
        "Pathwise s.e.",
        "Likelihood ratio",
        "LR s.e.",
        "Finite diff (CRN)",
        "FD s.e.",
    ]
    with gc2:
        greek_table = pd.DataFrame(rows, columns=cols).set_index("Greek")
        st.dataframe(greek_table.map(lambda v: "—" if v is None or pd.isna(v) else f"{v:.5f}"))
        st.caption(note)
        d = rows[0]
        est = [("Pathwise", d[3]), ("Likelihood ratio", d[5]), ("Finite diff (CRN)", d[7])]
        est = [(n, v) for n, v in est if v is not None]
        fig = go.Figure(
            go.Bar(
                x=[n for n, _ in est],
                y=[v for _, v in est],
                marker_color=[viz.BLUE, viz.ORANGE, viz.AQUA][: len(est)],
                text=[f"{v:.5f}" for _, v in est],
                textposition="outside",
                textfont=dict(color=viz.INK_2),
            )
        )
        fig.update_layout(showlegend=False, hovermode="closest")
        st.plotly_chart(
            fig_layout(fig, "Delta standard error by estimator (lower is better)", "", "Standard error", 320)
        )

# ============================== Portfolio risk ============================== #
DEFAULT_BOOK = pd.DataFrame(
    [
        {
            "asset": "TECH",
            "instrument": "stock",
            "quantity": 1_000,
            "strike % spot": None,
            "expiry (yrs)": None,
            "implied vol": None,
        },
        {
            "asset": "BANKS",
            "instrument": "stock",
            "quantity": 1_500,
            "strike % spot": None,
            "expiry (yrs)": None,
            "implied vol": None,
        },
        {
            "asset": "STAPLES",
            "instrument": "stock",
            "quantity": 2_000,
            "strike % spot": None,
            "expiry (yrs)": None,
            "implied vol": None,
        },
        {
            "asset": "ENERGY",
            "instrument": "stock",
            "quantity": 1_200,
            "strike % spot": None,
            "expiry (yrs)": None,
            "implied vol": None,
        },
        {
            "asset": "TECH",
            "instrument": "call",
            "quantity": -1_000,
            "strike % spot": 105.0,
            "expiry (yrs)": 0.25,
            "implied vol": None,
        },
        {
            "asset": "ENERGY",
            "instrument": "put",
            "quantity": 1_200,
            "strike % spot": 95.0,
            "expiry (yrs)": 0.25,
            "implied vol": None,
        },
    ]
)


@st.cache_data(show_spinner=False)
def get_prices(source: str, seed: int, upload: bytes | None) -> pd.DataFrame:
    if source == "Upload CSV" and upload is not None:
        import io

        return load_csv(io.BytesIO(upload))
    return synthetic_market(seed=seed)


def build_portfolio(prices: pd.DataFrame, book: pd.DataFrame, r: float) -> Portfolio:
    names = list(prices.columns)
    spots = prices.iloc[-1].to_numpy(float)
    _, cov = estimate_parameters(prices)
    vols = np.sqrt(np.diag(cov))
    positions = []
    for _, row in book.dropna(subset=["asset", "quantity"]).iterrows():
        if row["asset"] not in names:
            raise ValueError(f"unknown asset {row['asset']!r}; choose from {names}")
        i = names.index(row["asset"])
        if row["instrument"] == "stock":
            positions.append(Position(i, float(row["quantity"])))
        else:
            vol = row["implied vol"] if pd.notna(row["implied vol"]) else vols[i]
            positions.append(
                Position(
                    i,
                    float(row["quantity"]),
                    row["instrument"],
                    spots[i] * row["strike % spot"] / 100,
                    float(row["expiry (yrs)"]),
                    float(vol),
                )
            )
    if not positions:
        raise ValueError("the book is empty")
    return Portfolio(spots, positions, r, names)


with tabs[3]:
    st.header("Portfolio risk: VaR and expected shortfall")
    st.caption(
        "Monte Carlo simulates correlated moves, then fully reprices every position, options included. "
        "Delta-normal uses only first-order exposures, which is why it misses option risk."
    )
    d1, d2, d3 = st.columns(3)
    with d1:
        source = st.radio(
            "Data",
            ["Synthetic market", "Upload CSV"],
            horizontal=True,
            help="Synthetic: GARCH(1,1) volatility clustering with Student-t shocks. "
            "CSV: a Date column plus one price column per asset.",
        )
        upload = st.file_uploader("Prices CSV", type="csv") if source == "Upload CSV" else None
        data_seed = int(st.number_input("Data seed", value=42, step=1)) if source == "Synthetic market" else 0
    with d2:
        alpha = st.select_slider(
            "Confidence level", [0.95, 0.975, 0.99, 0.995], value=0.99, format_func=lambda a: f"{a:.1%}"
        )
        horizon = st.select_slider("Horizon (trading days)", [1, 5, 10, 20], value=10)
    with d3:
        df_t = st.slider(
            "Student-t degrees of freedom",
            3.0,
            30.0,
            5.0,
            0.5,
            help="Lower = fatter tails. Same covariance as the normal case.",
        )
        rate = st.number_input("Risk-free rate", 0.0, 0.2, 0.03, 0.005, format="%.3f")
    try:
        prices = get_prices(source, data_seed, upload.getvalue() if upload else None)
    except Exception as e:  # noqa: BLE001 - user files can fail in many ways
        st.error(f"Could not read prices: {e}")
        st.stop()
    names = list(prices.columns)
    book_default = (
        DEFAULT_BOOK
        if set(DEFAULT_BOOK["asset"]) <= set(names)
        else pd.DataFrame(
            [
                {
                    "asset": n,
                    "instrument": "stock",
                    "quantity": 100,
                    "strike % spot": None,
                    "expiry (yrs)": None,
                    "implied vol": None,
                }
                for n in names
            ]
        )
    )
    st.subheader("Positions")
    book = st.data_editor(
        book_default,
        num_rows="dynamic",
        hide_index=True,
        key=f"book_{source}",
        column_config={
            "asset": st.column_config.SelectboxColumn(options=names, required=True),
            "instrument": st.column_config.SelectboxColumn(options=["stock", "call", "put"], required=True),
            "quantity": st.column_config.NumberColumn(help="Negative = short"),
            "strike % spot": st.column_config.NumberColumn(help="Options only, e.g. 105 = 5% out of the money call"),
            "implied vol": st.column_config.NumberColumn(help="Blank = historical volatility"),
        },
    )
    try:
        pf = build_portfolio(prices, book, rate)
        mu, cov = estimate_parameters(prices)
        n_sc = min(N_PATHS, 2**17)
        with st.spinner("Simulating scenarios and repricing the book..."):
            reports = [
                delta_normal_var(pf, mu, cov, alpha, horizon),
                historical_var(pf, log_returns(prices), alpha, horizon),
                monte_carlo_var(pf, mu, cov, alpha, horizon, n_sc, "normal", seed=SEED),
                monte_carlo_var(pf, mu, cov, alpha, horizon, n_sc, "t", df_t, seed=SEED),
            ]
    except (ValueError, TypeError) as e:
        st.error(f"Check the positions table: {e}")
        st.stop()
    value = pf.current_value
    k1, k2, k3 = st.columns(3)
    k1.metric("Portfolio value", f"{value:,.0f}")
    k2.metric(
        f"{alpha:.1%} {horizon}-day VaR (MC, Student-t)",
        f"{reports[3].var:,.0f}",
        help=f"{reports[3].var / value:.2%} of value",
    )
    k3.metric("Expected shortfall (CVaR)", f"{reports[3].cvar:,.0f}", help="Average loss on the worst days beyond VaR")
    table = pd.DataFrame([r.as_dict() for r in reports]).set_index("method")
    table["VaR % value"] = table["VaR"] / value * 100
    table["CVaR % value"] = table["CVaR"] / value * 100
    r1, r2 = st.columns([1, 1])
    with r1:
        table = table.rename(columns={"VaR % value": "VaR %", "CVaR % value": "CVaR %"})
        show(
            table[["VaR", "VaR %", "CVaR", "CVaR %"]],
            **{"VaR": "{:,.0f}", "CVaR": "{:,.0f}", "VaR %": "{:.2f}%", "CVaR %": "{:.2f}%"},
        )
        st.caption(
            f"Positive numbers are losses, % of portfolio value. Monte Carlo VaR standard error: "
            f"about {reports[3].var_se:,.0f}."
        )
        st.subheader("Stress tests")
        stress = stress_test(pf, DEFAULT_STRESS)
        show(stress, **{"P&L": "{:+,.0f}", "P&L %": "{:+.2f}%"})
    with r2:
        fig = go.Figure()
        for rep, color in ((reports[2], viz.BLUE), (reports[3], viz.ORANGE)):
            fig.add_trace(go.Histogram(x=rep.pnl, nbinsx=120, name=rep.method, marker_color=color, opacity=0.55))
            fig.add_vline(x=-rep.var, line=dict(color=color, width=2, dash="dash"))
        fig.add_annotation(
            x=-reports[3].var,
            y=1,
            yref="paper",
            text=f"VaR (t) {reports[3].var:,.0f}",
            showarrow=False,
            xanchor="right",
            font=dict(color=viz.INK_2),
        )
        fig.update_layout(barmode="overlay", hovermode="closest")
        st.plotly_chart(
            fig_layout(fig, f"{horizon}-day P&L distribution: normal vs fat-tailed", "P&L", "Scenarios", 420)
        )
        corr = np.corrcoef(log_returns(prices).to_numpy(), rowvar=False)
        np.fill_diagonal(corr, np.nan)  # the diagonal is always 1: blank it so the off-diagonal stands out
        fig = go.Figure(
            go.Heatmap(
                z=corr,
                x=names,
                y=names,
                zmin=-1,
                zmax=1,
                colorscale=viz.DIVERGING,
                text=np.where(np.isnan(corr), "", np.char.mod("%.2f", np.nan_to_num(corr))),
                texttemplate="%{text}",
                showscale=False,
                textfont=dict(color=viz.INK),
                hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
                xgap=2,
                ygap=2,
            )
        )
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig_layout(fig, "Return correlation", "", "", 320))
    norm_px = prices / prices.iloc[0] * 100
    fig = go.Figure()
    for i, n in enumerate(names[:8]):
        fig.add_trace(go.Scatter(x=norm_px.index, y=norm_px[n], name=n, line=dict(color=viz.SERIES[i], width=2)))
    st.plotly_chart(fig_layout(fig, "Price history (rebased to 100)", "", "Index"))

# =============================== VaR backtest =============================== #
with tabs[4]:
    st.header("VaR backtest")
    st.caption(
        "Each day, forecast tomorrow's 1-day VaR from the trailing window only, then count the days the loss "
        "was worse. Kupiec tests the breach rate; Christoffersen tests whether breaches cluster; Basel assigns "
        "a traffic-light zone."
    )
    b1, b2, b3 = st.columns(3)
    with b1:
        bt_alpha = st.select_slider(
            "Confidence", [0.95, 0.975, 0.99], value=0.99, format_func=lambda a: f"{a:.1%}", key="bt_a"
        )
    with b2:
        window = st.select_slider("Estimation window (days)", [125, 250, 500], value=250)
    with b3:
        focus = st.selectbox(
            "Highlight method",
            ["normal", "student_t", "historical", "ewma", "fhs"],
            index=2,
            format_func={
                "normal": "Static normal",
                "student_t": "Static Student-t",
                "historical": "Historical simulation",
                "ewma": "EWMA (RiskMetrics)",
                "fhs": "Filtered historical simulation",
            }.get,
        )
    try:
        pf_bt = build_portfolio(prices, book, rate)
        stock_w = np.zeros(pf_bt.n_assets)
        for pos in pf_bt.positions:
            if pos.instrument == "stock":
                stock_w[pos.asset] += pos.quantity * pf_bt.spots[pos.asset]
        if not np.any(stock_w):
            stock_w[:] = 1.0
        stock_w = stock_w / stock_w.sum()
        port_ret = np.expm1(log_returns(prices)) @ stock_w
        bt = rolling_var_backtest(port_ret, bt_alpha, window)
    except ValueError as e:
        st.error(f"Backtest needs more data or a valid book: {e}")
        st.stop()
    st.caption(
        "Uses the stock positions' weights (options excluded) so the realised P&L is observable from prices alone."
    )
    fc = bt.forecasts
    exc = fc["return"] < -fc[focus]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fc.index, y=fc["return"] * 100, mode="markers", marker=dict(color=viz.AXIS, size=4), name="Daily return"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc.index, y=-fc[focus] * 100, mode="lines", line=dict(color=viz.BLUE, width=2), name=f"-VaR ({focus})"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc.index[exc],
            y=fc["return"][exc] * 100,
            mode="markers",
            marker=dict(color=viz.CRITICAL, size=9, symbol="x"),
            name=f"Breach ({int(exc.sum())})",
        )
    )
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig_layout(fig, "Realised returns against the VaR forecast", "", "Return (%)", 420))
    s = bt.summary.copy()
    s["Basel zone"] = s["basel_zone"].map({"green": "🟢 green", "yellow": "🟡 yellow", "red": "🔴 red"})
    s["result"] = s["verdict"].map({"pass": "✓ pass", "fail": "✗ fail"})
    show(
        s[["exceptions", "expected", "rate %", "p_pof", "p_ind", "p_cc", "Basel zone", "result"]],
        **{"expected": "{:.1f}", "rate %": "{:.2f}%", "p_pof": "{:.3f}", "p_ind": "{:.3f}", "p_cc": "{:.3f}"},
    )
    st.caption(
        "A method passes when both p-values exceed 5%: the right number of breaches, "
        "and breaches that don't bunch together."
    )

# ================================ Validation ================================ #
with tabs[5]:
    st.header("Self-validation")
    st.caption(
        "Prices a panel of contracts by simulation and checks each against an independent exact answer. "
        "A benchmark passes if the error is within 4 standard errors, plus a stated allowance for known "
        "method bias such as time discretisation."
    )
    if st.button("Run validation suite", type="primary"):
        with st.spinner("Running 13 benchmarks..."):
            v = run_benchmarks(n_paths=min(N_PATHS, 2**16), seed=SEED)
        n_pass = int(v["pass"].sum())
        st.metric("Benchmarks passed", f"{n_pass} / {len(v)}")
        v["result"] = v["pass"].map({True: "✓ pass", False: "✗ fail"})
        show(
            v.set_index("benchmark")[
                ["Monte Carlo", "std error", "reference", "reference method", "error / SE", "result", "seconds"]
            ],
            **{
                "Monte Carlo": "{:.5f}",
                "std error": "{:.5f}",
                "reference": "{:.5f}",
                "error / SE": "{:+.2f}",
                "seconds": "{:.2f}",
            },
        )


# ============================ Valuation (Monde Nissin) ============================ #
@st.cache_data(show_spinner=False, max_entries=16)
def run_company_valuation(
    margin_mode: float, g_mode: float, beta_mean: float, erp_mode: float, rf_mode_shift: float, n_sims: int
):
    import io

    from mcquant.valuation.company import build_valuation, implied_market_expectations, run_valuation
    from mcquant.valuation.excel_export import export_workbook
    from mcquant.valuation.montecarlo import Triangular, TruncatedNormal

    base = build_valuation()
    mc, w = base.mc_inputs, base.wacc_inputs

    def tri(d, mode):
        return Triangular(min(d.low, mode), mode, max(d.high, mode))

    overrides = {
        "ebitda_margin": tri(mc.ebitda_margin, margin_mode),
        "terminal_growth": tri(mc.terminal_growth, g_mode),
        "beta": TruncatedNormal(beta_mean, w.beta.sd, min(w.beta.low, beta_mean), max(w.beta.high, beta_mean)),
        "equity_risk_premium": tri(w.equity_risk_premium, erp_mode),
    }
    setup = build_valuation(driver_overrides=overrides)
    if rf_mode_shift:
        from dataclasses import replace as _replace

        rf = setup.wacc_inputs.risk_free
        new_rf = Triangular(rf.low + rf_mode_shift, rf.mode + rf_mode_shift, rf.high + rf_mode_shift)
        wi = _replace(setup.wacc_inputs, risk_free=new_rf)
        setup.wacc_inputs = wi
        setup.mc_inputs = _replace(setup.mc_inputs, wacc_inputs=wi)
    report = run_valuation(setup, n_sims=n_sims)
    implied = implied_market_expectations(setup)
    buf = io.BytesIO()
    export_workbook(report, buf)
    return report, implied, buf.getvalue()


with tabs[6]:
    st.header("Valuation: Monde Nissin (PSE: MONDE)")
    st.caption(
        "Discounted cash flow plus Monte Carlo simulation. Every company figure comes from a cited public source "
        "(see docs/ASSUMPTIONS.md); every forecast range has a written reason. Not investment advice."
    )
    with st.expander("Adjust the most likely value of the key assumptions", expanded=False):
        v1, v2, v3 = st.columns(3)
        with v1:
            margin_mode = st.slider(
                "EBITDA margin",
                0.150,
                0.215,
                0.190,
                0.005,
                format="%.3f",
                help="Core EBITDA margin was 16.9% (2023), 19.3% (2024), 18.4% (2025).",
            )
            g_mode = st.slider("Terminal growth g", 0.025, 0.055, 0.040, 0.005, format="%.3f")
        with v2:
            beta_mean = st.slider("Beta", 0.40, 1.00, 0.58, 0.01, help="Bottom-up 0.58; regression 0.62.")
            erp_mode = st.slider("Equity risk premium", 0.055, 0.085, 0.0669, 0.0005, format="%.4f")
        with v3:
            rf_shift = st.slider("Shift the risk-free range (percentage points)", -1.0, 2.0, 0.0, 0.25) / 100
            n_val = st.select_slider("Simulations", [10_000, 25_000, 50_000], value=50_000)
    try:
        with st.spinner("Running 50,000 discounted cash flows..."):
            rep, implied, xlsx = run_company_valuation(margin_mode, g_mode, beta_mean, erp_mode, rf_shift, n_val)
    except ValueError as e:
        st.error(f"Invalid combination of assumptions: {e}")
        st.stop()
    price = rep.price
    st.markdown(f"#### {rep.headline}")
    k = st.columns(5)
    k[0].metric("P5", f"₱{rep.mc.percentile(5):,.2f}")
    k[1].metric("Median", f"₱{rep.mc.percentile(50):,.2f}")
    k[2].metric("P95", f"₱{rep.mc.percentile(95):,.2f}")
    k[3].metric("Market price", f"₱{price:,.2f}", help="Close of 6 Nov 2026 (BusinessWorld)")
    p_above = rep.p_above_price
    p_text = "> 99%" if p_above >= 0.995 else "< 1%" if p_above <= 0.005 else f"{p_above:.0%}"
    k[4].metric("P(value > price)", p_text, help="Conditional on the stated assumptions, not a forecast")
    st.download_button(
        "Download the Excel model (live formulas)",
        xlsx,
        file_name="Monde_Nissin_DCF.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    c1, c2 = st.columns(2)
    with c1:
        v = rep.mc.value_per_share
        fig = go.Figure(go.Histogram(x=v, nbinsx=80, marker_color=viz.BLUE, name="Simulated value"))
        fig.add_vline(
            x=price,
            line=dict(color=viz.ORANGE, width=3),
            annotation_text=f"price ₱{price:.2f}",
            annotation_position="top left",
            annotation_font_color=viz.INK,
        )
        for q in (5, 50, 95):
            fig.add_vline(
                x=rep.mc.percentile(q),
                line=dict(color=viz.BLUE, width=1, dash="dot"),
                annotation_text=f"P{q}",
                annotation_position="top right",
                annotation_font_color=viz.INK_2,
            )
        fig.update_layout(showlegend=False, hovermode="closest")
        st.plotly_chart(fig_layout(fig, "Distribution of value per share", "₱ per share", "Simulations"))
    with c2:
        t = rep.tornado.iloc[::-1]
        base_v = float(t["base value"].iloc[0])
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=t["driver"],
                x=t["value at low input"] - base_v,
                base=base_v,
                orientation="h",
                marker_color=viz.ORANGE,
                name="input at 10th percentile",
            )
        )
        fig.add_trace(
            go.Bar(
                y=t["driver"],
                x=t["value at high input"] - base_v,
                base=base_v,
                orientation="h",
                marker_color=viz.BLUE,
                name="input at 90th percentile",
            )
        )
        fig.add_vline(x=base_v, line=dict(color=viz.INK, width=1))
        fig.update_layout(barmode="overlay", hovermode="closest")
        st.plotly_chart(fig_layout(fig, f"Tornado: value moves around ₱{base_v:.2f}", "₱ per share", ""))

    c3, c4 = st.columns(2)
    with c3:
        ff = rep.football[::-1]
        fig = go.Figure()
        for i, r in enumerate(ff):
            fig.add_trace(
                go.Bar(
                    y=[r["label"]],
                    x=[r["high"] - r["low"]],
                    base=[r["low"]],
                    orientation="h",
                    marker_color=viz.SERIES[(len(ff) - 1 - i) % 6],
                    showlegend=False,
                    hovertemplate=f"{r['label']}: ₱{r['low']:.2f} to ₱{r['high']:.2f}<extra></extra>",
                )
            )
        fig.add_vline(x=price, line=dict(color=viz.ORANGE, width=3))
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig_layout(fig, f"Football field (orange line: price ₱{price:.2f})", "₱ per share", "", 420))
    with c4:
        sens = rep.sensitivity
        z = sens.to_numpy(float)
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=[f"{g:.1%}" for g in sens.columns],
                y=[f"{w_:.1%}" for w_ in sens.index],
                colorscale=[[0, viz.CRITICAL], [0.5, "#f0efec"], [1, viz.BLUE]],
                zmid=price,
                showscale=False,
                text=np.where(np.isfinite(z), np.char.mod("₱%.2f", np.nan_to_num(z)), "n/m"),
                texttemplate="%{text}",
                hovertemplate="WACC %{y}, g %{x}: %{text}<extra></extra>",
                xgap=2,
                ygap=2,
            )
        )
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(type="category")
        fig.update_yaxes(type="category")
        st.plotly_chart(fig_layout(fig, "Sensitivity: WACC x terminal growth", "Terminal growth", "WACC", 420))

    st.subheader("What today's price implies (reverse DCF)")
    r1, r2, r3 = st.columns(3)
    if "implied_wacc" in implied:
        r1.metric("Implied WACC", f"{implied['implied_wacc']:.1%}", help=f"Base case {rep.base_wacc.wacc:.1%}")
    if "implied_ebitda_margin" in implied:
        r2.metric(
            "Implied permanent EBITDA margin",
            f"{implied['implied_ebitda_margin']:.1%}",
            help="2023 to 2025: 16.9% to 19.3%",
        )
    if "implied_terminal_growth" in implied:
        r3.metric("Implied perpetual growth", f"{implied['implied_terminal_growth']:.1%}")
    st.caption(
        "Each figure holds every other input at its most likely value and asks what this one input must be for "
        "the DCF to equal today's price."
    )

    fc = rep.base_dcf.forecast
    base_year = int(rep.setup.company.history.index[-1])
    tbl = fc[["revenue", "ebitda", "nopat", "capex", "change_nwc", "fcff", "pv_fcff"]].copy()
    tbl.index = [f"FY{base_year + int(t)}" for t in tbl.index]
    tbl.columns = ["Revenue", "EBITDA", "NOPAT", "Capex", "Change in NWC", "FCFF", "PV of FCFF"]
    st.subheader("Base-case forecast (₱ millions)")
    show(tbl, **{c: "{:,.0f}" for c in tbl.columns})
