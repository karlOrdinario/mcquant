"""Load a company's sourced facts and assumptions, and run the full valuation.

    setup  = build_valuation()          # reads config/monde.yaml + config/monde_assumptions.yaml
    report = run_valuation(setup)       # DCF, Monte Carlo, tornado, sensitivity, comps, football field
    print(report.headline)

Swap in another company by writing two YAML files with the same structure.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .comps import comps_valuation
from .dcf import DCFResult, roll_forward, run_dcf
from .inputs import DCFAssumptions, EquityBridge
from .montecarlo import (
    Distribution,
    Fixed,
    MCInputs,
    MCValuationResult,
    Triangular,
    TruncatedNormal,
    WACCInputs,
    simulate_valuation,
    tornado,
)
from .wacc import WACCResult, wacc

REPO = Path(__file__).resolve().parents[3]
DEFAULT_COMPANY = REPO / "config" / "monde.yaml"
DEFAULT_ASSUMPTIONS = REPO / "config" / "monde_assumptions.yaml"
TODO = "TODO_SOURCE"


# --------------------------------------------------------------------------- #
# Reading the YAML
# --------------------------------------------------------------------------- #
def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_sourced(node: Any, path: str = "") -> Iterator[tuple[str, dict]]:
    """Yield (path, item) for every dict that holds a ``value`` (a sourced number)."""
    if isinstance(node, dict):
        if "value" in node:
            yield path, node
            return
        for k, v in node.items():
            yield from iter_sourced(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from iter_sourced(v, f"{path}[{i}]")


def missing_sources(cfg: dict) -> list[str]:
    """Paths of numbers with no usable source or date, or still marked TODO_SOURCE."""
    bad = []
    for path, item in iter_sourced(cfg):
        src = str(item.get("source", "")).strip()
        if not src or TODO in src or item.get("value") in (TODO, None) or not item.get("as_of"):
            bad.append(path)
    return bad


def value_of(item: dict) -> float:
    """Number from a sourced item, converted to PHP millions where reported in thousands."""
    v = item["value"]
    if v == TODO or v is None:
        raise ValueError(f"unsourced number ({TODO}) cannot be used: {item}")
    v = float(v)
    unit = str(item.get("unit", ""))
    if unit == "PHP thousands":
        return v / 1_000
    if unit == "shares":
        return v / 1_000_000  # millions of shares, to match PHP millions
    return v


def _years_between(a: dt.date, b: dt.date) -> float:
    return (b - a).days / 365.0


def _dist(spec: dict) -> Distribution:
    if "triangular" in spec:
        t = spec["triangular"]
        return Triangular(t["low"], t["mode"], t["high"])
    if "truncated_normal" in spec:
        t = spec["truncated_normal"]
        return TruncatedNormal(t["mean"], t["sd"], t["low"], t["high"])
    if "fixed" in spec:
        return Fixed(spec["fixed"]["value"])
    raise ValueError(f"unknown distribution: {spec}")


@dataclass
class CompanyData:
    name: str
    ticker: str
    history: pd.DataFrame  # PHP millions by year
    bridge: EquityBridge  # PHP millions, millions of shares
    bridge_items: pd.DataFrame
    price: float
    shares_m: float
    week52: tuple[float, float]
    analyst_targets: tuple[float, float, float]  # low, mean, high
    core_eps: float
    dividends: list[tuple[dt.date, float]]
    rates: dict[str, float]
    peers: pd.DataFrame
    raw: dict

    @property
    def market_cap(self) -> float:
        return self.price * self.shares_m


def load_company(path: str | Path = DEFAULT_COMPANY) -> CompanyData:
    raw = load_yaml(path)
    h = raw["history"]
    years = sorted(h["net_sales"])
    hist = pd.DataFrame(
        {
            k: [value_of(h[k][y]) for y in years]
            for k in (
                "net_sales",
                "core_ebitda",
                "depreciation_amortization",
                "capex",
                "trade_receivables",
                "inventories",
                "payables",
            )
        },
        index=[int(y) for y in years],
    )
    hist["nwc"] = hist.trade_receivables + hist.inventories - hist.payables

    bs = raw["balance_sheet_2025"]
    item_rows = []
    totals = {}
    for group, sign in (
        ("debt", -1),
        ("debt_like", -1),
        ("cash", 1),
        ("non_operating_assets", 1),
        ("minority_interest", -1),
    ):
        totals[group] = 0.0
        for name, item in bs[group].items():
            v = value_of(item)
            totals[group] += v
            item_rows.append({"group": group, "item": name, "PHP m": v, "sign": sign})
    m = raw["market"]
    shares_m = value_of(m["shares_outstanding"])
    bridge = EquityBridge(
        debt=totals["debt"],
        cash=totals["cash"],
        diluted_shares=shares_m,
        minority_interest=totals["minority_interest"],
        debt_like_items=totals["debt_like"],
        non_operating_assets=totals["non_operating_assets"],
    )
    peers = pd.DataFrame(
        [
            {
                "name": p["name"],
                "ticker": p["ticker"],
                "country": p["country"],
                "ev_ebitda": value_of(p["ev_ebitda"]),
                "pe": value_of(p["pe"]),
                "as_of": p["ev_ebitda"]["as_of"],
            }
            for p in raw["peers"]
        ]
    )
    return CompanyData(
        name=raw["company"]["name"],
        ticker=raw["company"]["ticker"],
        history=hist,
        bridge=bridge,
        bridge_items=pd.DataFrame(item_rows),
        price=value_of(m["share_price"]),
        shares_m=shares_m,
        week52=(value_of(m["week52_low"]), value_of(m["week52_high"])),
        analyst_targets=(
            value_of(m["analyst_target_low"]),
            value_of(m["analyst_target_mean"]),
            value_of(m["analyst_target_high"]),
        ),
        core_eps=value_of(raw["recent_trading"]["core_eps_2025"]),
        dividends=[(d["date"], value_of(d["amount"])) for d in raw.get("dividends_after_valuation_date", [])],
        rates={k: value_of(v) for k, v in raw["rates"].items()},
        peers=peers,
        raw=raw,
    )


# --------------------------------------------------------------------------- #
# Building the model inputs
# --------------------------------------------------------------------------- #
def _check_risk_free(rf: Distribution, rates: dict[str, float]) -> None:
    """The risk-free distribution must be anchored on the sourced rates: its most
    likely value is the 10-year yield minus the default spread, and its high end
    is the unadjusted 10-year yield."""
    adjusted = rates["ph_10y_government_yield"] - rates["ph_moodys_rating_default_spread"]
    if abs(rf.mode - adjusted) > 1e-9:
        raise ValueError(f"risk-free mode {rf.mode} should equal 10y yield - default spread = {adjusted:.5f}")
    if isinstance(rf, Triangular) and abs(rf.high - rates["ph_10y_government_yield"]) > 1e-9:
        raise ValueError("risk-free high should equal the unadjusted 10-year yield")


@dataclass
class ValuationSetup:
    company: CompanyData
    assumptions_raw: dict
    mc_inputs: MCInputs
    wacc_inputs: WACCInputs
    roll_years: float
    dividends: list[tuple[float, float]]  # (years after valuation date, PHP per share)
    n_sims: int
    seed: int
    method: str

    @property
    def base_case(self) -> DCFAssumptions:
        return self.mc_inputs.mode_case()[0]

    @property
    def base_cost_of_equity(self) -> float:
        return self.mc_inputs.mode_case()[1]


def build_valuation(
    company_path: str | Path = DEFAULT_COMPANY,
    assumptions_path: str | Path = DEFAULT_ASSUMPTIONS,
    driver_overrides: dict[str, Distribution | float] | None = None,
) -> ValuationSetup:
    """Read both YAML files and assemble the Monte Carlo inputs.

    ``driver_overrides`` replaces a driver's distribution (used by the dashboard),
    e.g. ``{"ebitda_margin": Triangular(0.17, 0.19, 0.21)}``.
    """
    co = load_company(company_path)
    a = load_yaml(assumptions_path)
    d = a["drivers"]
    years = [str(y) for y in a["forecast_years"]]
    growth = [_dist(d["revenue_growth"][y]) for y in years]
    r = co.rates

    rf = _dist(a["wacc"]["risk_free"])
    _check_risk_free(rf, r)
    rd = r["ph_10y_government_yield"] + a["wacc"]["pre_tax_cost_of_debt"]["spread_over_government"]
    tax = _dist(d["tax_rate"]).mode
    debt_for_weights = co.bridge.debt
    w_in = WACCInputs(
        risk_free=rf,
        beta=_dist(a["wacc"]["beta"]),
        equity_risk_premium=_dist(a["wacc"]["equity_risk_premium"]),
        pre_tax_cost_of_debt=rd,
        tax_rate=tax,
        equity_weight=co.market_cap / (co.market_cap + debt_for_weights),
    )
    base_wacc = float(w_in.wacc(w_in.beta.mode, w_in.equity_risk_premium.mode))
    base = DCFAssumptions(
        base_revenue=float(co.history["net_sales"].iloc[-1]),
        revenue_growth=[g.mode for g in growth],
        ebitda_margin=_dist(d["ebitda_margin"]).mode,
        da_pct_revenue=_dist(d["da_pct_revenue"]).mode,
        capex_pct_revenue=_dist(d["capex_pct_revenue"]).mode,
        nwc_pct_revenue_change=_dist(d["nwc_pct_revenue_change"]).mode,
        tax_rate=tax,
        wacc=base_wacc,
        terminal_growth=_dist(d["terminal_growth"]).mode,
    )
    sim = a["simulation"]
    mc = MCInputs(
        base=base,
        wacc_inputs=w_in,
        revenue_growth=growth,
        ebitda_margin=_dist(d["ebitda_margin"]),
        capex_pct_revenue=_dist(d["capex_pct_revenue"]),
        terminal_growth=_dist(d["terminal_growth"]),
        growth_autocorrelation=sim["growth_autocorrelation"]["value"],
        growth_margin_correlation=sim["growth_margin_correlation"]["value"],
    )
    if driver_overrides:
        wacc_changes = {k: v for k, v in driver_overrides.items() if k in ("beta", "equity_risk_premium", "risk_free")}
        other = {k: v for k, v in driver_overrides.items() if k not in wacc_changes}
        if wacc_changes:
            w_in = replace(w_in, **wacc_changes)
        mc = replace(mc, wacc_inputs=w_in, **other)

    val_date, price_date = a["valuation_date"], a["price_date"]
    divs = [(_years_between(val_date, d_), amt) for d_, amt in co.dividends if val_date < d_ <= price_date]
    return ValuationSetup(
        co,
        a,
        mc,
        w_in,
        _years_between(val_date, price_date),
        divs,
        int(sim["n_sims"]),
        int(sim["seed"]),
        str(sim["method"]),
    )


# --------------------------------------------------------------------------- #
# Running everything
# --------------------------------------------------------------------------- #
@dataclass
class ValuationReport:
    setup: ValuationSetup
    base_dcf: DCFResult  # most-likely inputs, at the valuation date
    base_wacc: WACCResult
    base_value_at_price_date: float
    mc: MCValuationResult
    tornado: pd.DataFrame
    sensitivity: pd.DataFrame  # value per share at the price date
    comps: pd.DataFrame
    football: list[dict]

    @property
    def price(self) -> float:
        return self.setup.company.price

    @property
    def p_above_price(self) -> float:
        return self.mc.prob_above(self.price)

    @property
    def probability_text(self) -> str:
        p = self.p_above_price
        if p >= 0.995:
            return "a more than 99%"
        if p <= 0.005:
            return "a less than 1%"
        return f"a {p:.0%}"

    @property
    def headline(self) -> str:
        p5, p95 = self.mc.percentile(5), self.mc.percentile(95)
        return (
            f"Valued {self.setup.company.name} at ₱{p5:,.2f} to ₱{p95:,.2f} per share (90% range) "
            f"versus a market price of ₱{self.price:,.2f}, {self.probability_text} probability "
            f"the shares are worth more than today's price."
        )


def _roll(setup: ValuationSetup, value: float, cost_of_equity: float) -> float:
    return roll_forward(value, cost_of_equity, setup.roll_years, setup.dividends)


def sensitivity_at_price_date(setup: ValuationSetup, wacc_values, growth_values) -> pd.DataFrame:
    """WACC x g table, each value rolled forward to the price date. The cost of
    equity implied by each WACC (same weights and cost of debt) drives the roll-forward."""
    w = setup.wacc_inputs
    rd_after = w.pre_tax_cost_of_debt * (1 - w.tax_rate)
    rows = {}
    for wc in wacc_values:
        re = (wc - (1 - w.equity_weight) * rd_after) / w.equity_weight
        row = {}
        for g in growth_values:
            if wc - g < 0.01 - 1e-12:
                row[g] = float("nan")
            else:
                v = run_dcf(replace(setup.base_case, wacc=wc, terminal_growth=g), setup.company.bridge).value_per_share
                row[g] = _roll(setup, v, re)
        rows[wc] = row
    df = pd.DataFrame(rows).T
    df.index.name, df.columns.name = "WACC", "terminal growth"
    return df


def run_valuation(setup: ValuationSetup, n_sims: int | None = None, seed: int | None = None) -> ValuationReport:
    co = setup.company
    base = setup.base_case
    re = setup.base_cost_of_equity
    base_dcf = run_dcf(base, co.bridge)
    w = setup.wacc_inputs
    wres = wacc(co.market_cap, co.bridge.debt, re, w.pre_tax_cost_of_debt, w.tax_rate)
    mc = simulate_valuation(
        setup.mc_inputs,
        co.bridge,
        n_sims or setup.n_sims,
        seed if seed is not None else setup.seed,
        setup.method,
        setup.roll_years,
        setup.dividends,
    )
    tor = tornado(setup.mc_inputs, co.bridge, roll_forward_years=setup.roll_years, dividends=setup.dividends)
    step = 0.005
    waccs = [round(base.wacc + k * step, 6) for k in range(-4, 5)]
    gs = [round(base.terminal_growth + k * step, 6) for k in range(-2, 3)]
    sens = sensitivity_at_price_date(setup, waccs, gs)
    ebitda = float(co.history["core_ebitda"].iloc[-1])
    comps = comps_valuation(co.peers, ebitda, co.core_eps, co.bridge)
    football = [
        {
            "label": "DCF Monte Carlo (P5 to P95)",
            "low": mc.percentile(5),
            "high": mc.percentile(95),
            "mid": mc.percentile(50),
        },
        {
            "label": "DCF sensitivity (WACC +/-1pt, g +/-0.5pt)",
            "low": float(np.nanmin(sens.loc[waccs[2] : waccs[6], gs[1] : gs[3]].to_numpy())),
            "high": float(np.nanmax(sens.loc[waccs[2] : waccs[6], gs[1] : gs[3]].to_numpy())),
            "mid": _roll(setup, base_dcf.value_per_share, re),
        },
        {
            "label": "Trading comps: EV/EBITDA (peer IQR)",
            "low": comps.loc["EV/EBITDA", "value q1"],
            "high": comps.loc["EV/EBITDA", "value q3"],
            "mid": comps.loc["EV/EBITDA", "value median"],
        },
        {
            "label": "Trading comps: P/E (peer IQR)",
            "low": comps.loc["P/E", "value q1"],
            "high": comps.loc["P/E", "value q3"],
            "mid": comps.loc["P/E", "value median"],
        },
        {
            "label": "Analyst price targets",
            "low": co.analyst_targets[0],
            "high": co.analyst_targets[2],
            "mid": co.analyst_targets[1],
        },
        {"label": "52-week trading range", "low": co.week52[0], "high": co.week52[1], "mid": None},
    ]
    return ValuationReport(
        setup, base_dcf, wres, _roll(setup, base_dcf.value_per_share, re), mc, tor, sens, comps, football
    )


def implied_market_expectations(setup: ValuationSetup) -> dict[str, float]:
    """Reverse DCF: what terminal growth or EBITDA margin does today's price imply,
    holding everything else at the base case?"""
    from scipy.optimize import brentq

    co, base, re = setup.company, setup.base_case, setup.base_cost_of_equity

    def gap(**kw) -> float:
        v = run_dcf(replace(base, **kw), co.bridge).value_per_share
        return _roll(setup, v, re) - co.price

    out = {}
    lo_g = -0.05
    hi_g = base.wacc - 0.0101
    if gap(terminal_growth=lo_g) < 0 < gap(terminal_growth=hi_g):
        out["implied_terminal_growth"] = brentq(lambda g: gap(terminal_growth=g), lo_g, hi_g)
    if gap(ebitda_margin=0.05) < 0 < gap(ebitda_margin=0.6):
        out["implied_ebitda_margin"] = brentq(lambda m: gap(ebitda_margin=m), 0.05, 0.6)
    w = setup.wacc_inputs
    rd_after = w.pre_tax_cost_of_debt * (1 - w.tax_rate)

    def gap_w(wc: float) -> float:
        re_w = (wc - (1 - w.equity_weight) * rd_after) / w.equity_weight
        v = run_dcf(replace(base, wacc=wc), co.bridge).value_per_share
        return _roll(setup, v, re_w) - co.price

    lo_w, hi_w = base.terminal_growth + 0.0101, 0.40
    if gap_w(hi_w) < 0 < gap_w(lo_w):
        out["implied_wacc"] = brentq(gap_w, lo_w, hi_w)
    return out


__all__ = [
    "load_company",
    "build_valuation",
    "run_valuation",
    "iter_sourced",
    "value_of",
    "CompanyData",
    "ValuationSetup",
    "ValuationReport",
    "sensitivity_at_price_date",
    "implied_market_expectations",
    "math",
]
