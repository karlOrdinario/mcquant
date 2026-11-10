"""Monte Carlo DCF: turn uncertain inputs into a distribution of value per share.

Each simulation draws a full set of drivers (revenue growth for every year,
EBITDA margin, capex, terminal growth, beta and equity risk premium), runs the
whole DCF, and records the value per share. 50,000 simulations give a
distribution we can read percentiles and probabilities from.

Design choices:

* **Triangular** distributions for business drivers: a low, most-likely and high
  value is exactly how managers and analysts talk about forecasts.
* **Truncated normal** for beta: it is a statistical estimate with a standard
  error, but it must stay within sensible bounds.
* **Correlation** through a Gaussian copula: growth in one year is correlated
  with growth in the next, and growth is mildly correlated with margin
  (pricing power helps both).
* Samples with WACC - g below 1 percentage point are **discarded and counted**.
* Random numbers come from :mod:`mcquant.rng`, so pseudo, antithetic and Sobol
  sampling all work, with fixed seeds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.stats import norm, triang, truncnorm

from ..rng import Method, standard_normals
from .dcf import equity_bridge, run_dcf
from .inputs import MIN_WACC_SPREAD, DCFAssumptions, EquityBridge, _finite, spread_too_small


# --------------------------------------------------------------------------- #
# Distributions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Fixed:
    """A known value (no uncertainty)."""

    value: float

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(u, float), self.value)

    @property
    def mode(self) -> float:
        return self.value

    def quantile(self, q: float) -> float:
        return self.value


@dataclass(frozen=True)
class Triangular:
    """Low, most likely, high. Collapses to a constant when all three are equal."""

    low: float
    mode: float
    high: float

    def __post_init__(self) -> None:
        for n in ("low", "mode", "high"):
            _finite(n, getattr(self, n))
        if not self.low <= self.mode <= self.high:
            raise ValueError(f"need low <= mode <= high, got {self.low}, {self.mode}, {self.high}")

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, float)
        width = self.high - self.low
        if width == 0:
            return np.full_like(u, self.mode)
        c = (self.mode - self.low) / width
        return triang.ppf(u, c, loc=self.low, scale=width)

    def quantile(self, q: float) -> float:
        return float(self.ppf(np.array([q]))[0])


@dataclass(frozen=True)
class TruncatedNormal:
    """Normal(mean, sd) cut off at [low, high]."""

    mean: float
    sd: float
    low: float
    high: float

    def __post_init__(self) -> None:
        for n in ("mean", "sd", "low", "high"):
            _finite(n, getattr(self, n))
        if self.sd < 0 or not self.low <= self.mean <= self.high:
            raise ValueError("need sd >= 0 and low <= mean <= high")

    @property
    def mode(self) -> float:
        return self.mean

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, float)
        if self.sd == 0:
            return np.full_like(u, self.mean)
        a, b = (self.low - self.mean) / self.sd, (self.high - self.mean) / self.sd
        return truncnorm.ppf(u, a, b, loc=self.mean, scale=self.sd)

    def quantile(self, q: float) -> float:
        return float(self.ppf(np.array([q]))[0])


Distribution = Fixed | Triangular | TruncatedNormal


def _as_dist(x: float | Distribution) -> Distribution:
    return Fixed(float(x)) if isinstance(x, (int, float)) else x


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WACCInputs:
    """CAPM and WACC inputs. The risk-free rate, beta and ERP may be uncertain."""

    risk_free: float | Distribution
    beta: float | Distribution
    equity_risk_premium: float | Distribution
    pre_tax_cost_of_debt: float
    tax_rate: float
    equity_weight: float  # E / (D + E), at market values

    def __post_init__(self) -> None:
        if not 0 < self.equity_weight <= 1:
            raise ValueError("equity_weight must be in (0, 1]")

    def _rf(self, rf):
        return _as_dist(self.risk_free).mode if rf is None else rf

    def wacc(self, beta, erp, rf=None) -> np.ndarray:
        re = self.cost_of_equity(beta, erp, rf)
        rd_after = self.pre_tax_cost_of_debt * (1 - self.tax_rate)
        return self.equity_weight * re + (1 - self.equity_weight) * rd_after

    def cost_of_equity(self, beta, erp, rf=None) -> np.ndarray:
        return np.asarray(self._rf(rf)) + np.asarray(beta) * np.asarray(erp)


@dataclass(frozen=True)
class MCInputs:
    """Uncertain drivers around a deterministic base case.

    The base case (``base``) supplies every driver that is not listed here
    (D&A %, NWC %, tax rate, base revenue, mid-year setting). Its own ``wacc``
    is ignored: WACC is rebuilt from :class:`WACCInputs` in every simulation.
    """

    base: DCFAssumptions
    wacc_inputs: WACCInputs
    revenue_growth: Sequence[float | Distribution]
    ebitda_margin: float | Distribution
    capex_pct_revenue: float | Distribution
    terminal_growth: float | Distribution
    growth_autocorrelation: float = 0.6  # correlation of growth between forecast years
    growth_margin_correlation: float = 0.3
    n_years: int = field(init=False)

    def __post_init__(self) -> None:
        g = tuple(_as_dist(x) for x in self.revenue_growth)
        if len(g) != self.base.n_years:
            raise ValueError(f"need {self.base.n_years} growth distributions, got {len(g)}")
        object.__setattr__(self, "revenue_growth", g)
        object.__setattr__(self, "n_years", len(g))
        for name in ("ebitda_margin", "capex_pct_revenue", "terminal_growth"):
            object.__setattr__(self, name, _as_dist(getattr(self, name)))
        for name in ("growth_autocorrelation", "growth_margin_correlation"):
            if not -1 < getattr(self, name) < 1:
                raise ValueError(f"{name} must be strictly between -1 and 1")

    # The latent drivers, in the order of the correlation matrix.
    @property
    def driver_names(self) -> list[str]:
        return [f"growth_y{t + 1}" for t in range(self.n_years)] + [
            "ebitda_margin",
            "capex_pct_revenue",
            "terminal_growth",
            "beta",
            "equity_risk_premium",
            "risk_free",
        ]

    def correlation_matrix(self) -> np.ndarray:
        n = self.n_years
        k = n + 6
        c = np.eye(k)
        for i in range(n):
            for j in range(n):
                if i != j:
                    c[i, j] = self.growth_autocorrelation ** abs(i - j)  # AR(1)-style decay
            c[i, n] = c[n, i] = self.growth_margin_correlation
        if np.linalg.eigvalsh(c).min() <= 0:
            raise ValueError("correlation settings are inconsistent (matrix not positive definite)")
        return c

    def mode_case(self) -> tuple[DCFAssumptions, float]:
        """Deterministic case with every driver at its most-likely value; returns (assumptions, cost of equity)."""
        w = self.wacc_inputs
        beta, erp, rf = _as_dist(w.beta).mode, _as_dist(w.equity_risk_premium).mode, _as_dist(w.risk_free).mode
        a = replace(
            self.base,
            revenue_growth=[d.mode for d in self.revenue_growth],
            ebitda_margin=self.ebitda_margin.mode,
            capex_pct_revenue=self.capex_pct_revenue.mode,
            terminal_growth=self.terminal_growth.mode,
            wacc=float(w.wacc(beta, erp, rf)),
        )
        return a, float(w.cost_of_equity(beta, erp, rf))


# --------------------------------------------------------------------------- #
# Vectorised DCF
# --------------------------------------------------------------------------- #
def dcf_enterprise_values(
    base: DCFAssumptions,
    growth: np.ndarray,
    margin: np.ndarray,
    capex: np.ndarray,
    wacc: np.ndarray,
    terminal_growth: np.ndarray,
) -> np.ndarray:
    """The DCF of :func:`mcquant.valuation.dcf.run_dcf`, for many scenarios at once.

    ``growth`` is (n, years); the others are (n,). Returns EV per scenario.
    """
    n_years = base.n_years
    revenue = base.base_revenue * np.cumprod(1 + growth, axis=1)
    prior = np.concatenate([np.full((growth.shape[0], 1), base.base_revenue), revenue[:, :-1]], axis=1)
    da = revenue * np.asarray(base.da_pct_revenue)
    ebitda = revenue * margin[:, None]
    nopat = (ebitda - da) * (1 - base.tax_rate)
    fcff = nopat + da - revenue * capex[:, None] - (revenue - prior) * np.asarray(base.nwc_pct_revenue_change)
    years = np.arange(1, n_years + 1)
    periods = years - 0.5 if base.mid_year else years.astype(float)
    disc = (1 + wacc[:, None]) ** -periods
    tv = fcff[:, -1] * (1 + terminal_growth) / (wacc - terminal_growth)
    tv_period = n_years - 0.5 if base.mid_year else float(n_years)
    return (fcff * disc).sum(axis=1) + tv * (1 + wacc) ** -tv_period


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
@dataclass
class MCValuationResult:
    value_per_share: np.ndarray  # kept samples only
    enterprise_value: np.ndarray
    drivers: pd.DataFrame  # kept samples, one column per driver (+ wacc)
    n_requested: int
    n_discarded: int
    deterministic_value: float  # all drivers at their most likely values
    method: str

    @property
    def n_kept(self) -> int:
        return self.value_per_share.size

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.value_per_share, q))

    def prob_above(self, price: float) -> float:
        return float(np.mean(self.value_per_share > price))

    def summary(self, market_price: float | None = None) -> pd.Series:
        v = self.value_per_share
        out = {
            "simulations": self.n_requested,
            "discarded (WACC - g < 1%)": self.n_discarded,
            "mean": v.mean(),
            "std error of mean": v.std(ddof=1) / math.sqrt(v.size),
            "median (P50)": np.median(v),
            "P5": self.percentile(5),
            "P25": self.percentile(25),
            "P75": self.percentile(75),
            "P95": self.percentile(95),
            "deterministic (most likely inputs)": self.deterministic_value,
        }
        if market_price is not None:
            p = self.prob_above(market_price)
            out["market price"] = market_price
            out["P(value > price)"] = p
            out["std error of P"] = math.sqrt(p * (1 - p) / v.size)
        return pd.Series(out)


def simulate_valuation(
    inputs: MCInputs,
    bridge: EquityBridge,
    n_sims: int = 50_000,
    seed: int | None = 2026,
    method: Method = "pseudo",
    roll_forward_years: float = 0.0,
    dividends: Sequence[tuple[float, float]] = (),
) -> MCValuationResult:
    """Run the Monte Carlo DCF.

    If ``roll_forward_years`` > 0, every simulated value is rolled forward from
    the valuation date to the price date at its own cost of equity, minus
    dividends paid in between (see :func:`mcquant.valuation.dcf.roll_forward`).
    """
    if n_sims < 100:
        raise ValueError("n_sims must be at least 100")
    names = inputs.driver_names
    chol = np.linalg.cholesky(inputs.correlation_matrix())
    z = standard_normals(n_sims, len(names), method, seed) @ chol.T
    u = norm.cdf(z)
    n = inputs.n_years
    growth = np.column_stack([d.ppf(u[:, t]) for t, d in enumerate(inputs.revenue_growth)])
    margin = inputs.ebitda_margin.ppf(u[:, n])
    capex = inputs.capex_pct_revenue.ppf(u[:, n + 1])
    g = inputs.terminal_growth.ppf(u[:, n + 2])
    w = inputs.wacc_inputs
    beta = _as_dist(w.beta).ppf(u[:, n + 3])
    erp = _as_dist(w.equity_risk_premium).ppf(u[:, n + 4])
    rf = _as_dist(w.risk_free).ppf(u[:, n + 5])
    wacc = w.wacc(beta, erp, rf)

    keep = ~spread_too_small(wacc, g)
    ev = dcf_enterprise_values(inputs.base, growth[keep], margin[keep], capex[keep], wacc[keep], g[keep])
    per_share = (ev + bridge.net_adjustment) / bridge.diluted_shares
    if roll_forward_years:
        re = w.cost_of_equity(beta[keep], erp[keep], rf[keep])
        rolled = per_share * (1 + re) ** roll_forward_years
        for paid_at, amount in dividends:
            rolled -= amount * (1 + re) ** (roll_forward_years - paid_at)
        per_share = rolled

    drivers = pd.DataFrame(
        np.column_stack(
            [growth[keep], margin[keep], capex[keep], g[keep], beta[keep], erp[keep], rf[keep], wacc[keep]]
        ),
        columns=names + ["wacc"],
    )
    base_a, base_re = inputs.mode_case()
    det = run_dcf(base_a, bridge).value_per_share
    if roll_forward_years:
        det = det * (1 + base_re) ** roll_forward_years - sum(
            amt * (1 + base_re) ** (roll_forward_years - t) for t, amt in dividends
        )
    if not np.all(np.isfinite(per_share)):
        raise ValueError("simulation produced non-finite values; check the input ranges")
    return MCValuationResult(per_share, ev, drivers, n_sims, int((~keep).sum()), float(det), method)


# --------------------------------------------------------------------------- #
# Tornado and sensitivity
# --------------------------------------------------------------------------- #
def _value_with(inputs: MCInputs, bridge: EquityBridge, roll_years: float, dividends, **overrides: float) -> float:
    """Deterministic value with selected drivers moved away from their mode."""
    a, _ = inputs.mode_case()
    w = inputs.wacc_inputs
    beta = overrides.pop("beta", _as_dist(w.beta).mode)
    erp = overrides.pop("equity_risk_premium", _as_dist(w.equity_risk_premium).mode)
    rf = overrides.pop("risk_free", _as_dist(w.risk_free).mode)
    growth_shift = overrides.pop("growth_shift", None)
    changes = dict(overrides)
    changes["wacc"] = float(w.wacc(beta, erp, rf))
    if growth_shift is not None:
        changes["revenue_growth"] = [d.mode + growth_shift for d in inputs.revenue_growth]
    v = run_dcf(replace(a, **changes), bridge).value_per_share
    if roll_years:
        re = float(w.cost_of_equity(beta, erp, rf))
        v = v * (1 + re) ** roll_years - sum(amt * (1 + re) ** (roll_years - t) for t, amt in dividends)
    return v


def tornado(
    inputs: MCInputs,
    bridge: EquityBridge,
    low_q: float = 0.10,
    high_q: float = 0.90,
    roll_forward_years: float = 0.0,
    dividends: Sequence[tuple[float, float]] = (),
) -> pd.DataFrame:
    """One-at-a-time swings: move each driver to its 10th and 90th percentile,
    keep everything else at the most likely value, and record the value per
    share. Sorted with the biggest swing first."""
    w = inputs.wacc_inputs
    g_dists = inputs.revenue_growth
    # Revenue growth moves as one block: every year shifted by the average
    # distance of each year's percentile from its mode.
    shift_lo = float(np.mean([d.quantile(low_q) - d.mode for d in g_dists]))
    shift_hi = float(np.mean([d.quantile(high_q) - d.mode for d in g_dists]))
    specs = [
        ("Revenue growth (all years)", "growth_shift", shift_lo, shift_hi, "pp shift"),
        (
            "EBITDA margin",
            "ebitda_margin",
            inputs.ebitda_margin.quantile(low_q),
            inputs.ebitda_margin.quantile(high_q),
            "%",
        ),
        (
            "Capex % of revenue",
            "capex_pct_revenue",
            inputs.capex_pct_revenue.quantile(low_q),
            inputs.capex_pct_revenue.quantile(high_q),
            "%",
        ),
        (
            "Terminal growth g",
            "terminal_growth",
            inputs.terminal_growth.quantile(low_q),
            inputs.terminal_growth.quantile(high_q),
            "%",
        ),
        ("Beta", "beta", _as_dist(w.beta).quantile(low_q), _as_dist(w.beta).quantile(high_q), "x"),
        (
            "Equity risk premium",
            "equity_risk_premium",
            _as_dist(w.equity_risk_premium).quantile(low_q),
            _as_dist(w.equity_risk_premium).quantile(high_q),
            "%",
        ),
        (
            "Risk-free rate",
            "risk_free",
            _as_dist(w.risk_free).quantile(low_q),
            _as_dist(w.risk_free).quantile(high_q),
            "%",
        ),
    ]
    specs = [sp for sp in specs if sp[2] != sp[3]]  # drop inputs that are fixed
    base_value = _value_with(inputs, bridge, roll_forward_years, dividends)
    rows = []
    for label, key, lo, hi, unit in specs:
        v_lo = _value_with(inputs, bridge, roll_forward_years, dividends, **{key: lo})
        v_hi = _value_with(inputs, bridge, roll_forward_years, dividends, **{key: hi})
        rows.append(
            {
                "driver": label,
                "low input": lo,
                "high input": hi,
                "unit": unit,
                "value at low input": v_lo,
                "value at high input": v_hi,
                "swing": abs(v_hi - v_lo),
                "base value": base_value,
            }
        )
    return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)


def sensitivity_table(
    assumptions: DCFAssumptions,
    bridge: EquityBridge,
    wacc_values: Sequence[float],
    growth_values: Sequence[float],
) -> pd.DataFrame:
    """Value per share for each (WACC, g) pair. Cells where WACC - g is below
    1 percentage point are reported as NaN and must be shown as 'n/m' (not meaningful)."""
    rows = {}
    for w in wacc_values:
        row = {}
        for g in growth_values:
            if spread_too_small(w, g, MIN_WACC_SPREAD):
                row[g] = float("nan")
            else:
                row[g] = run_dcf(replace(assumptions, wacc=w, terminal_growth=g), bridge).value_per_share
        rows[w] = row
    df = pd.DataFrame(rows).T
    df.index.name, df.columns.name = "WACC", "terminal growth"
    return df


def zero_variance(inputs: MCInputs) -> MCInputs:
    """Same inputs with every distribution collapsed to its most likely value (for testing)."""
    w = inputs.wacc_inputs
    return replace(
        inputs,
        wacc_inputs=replace(
            w,
            beta=_as_dist(w.beta).mode,
            equity_risk_premium=_as_dist(w.equity_risk_premium).mode,
            risk_free=_as_dist(w.risk_free).mode,
        ),
        revenue_growth=[d.mode for d in inputs.revenue_growth],
        ebitda_margin=inputs.ebitda_margin.mode,
        capex_pct_revenue=inputs.capex_pct_revenue.mode,
        terminal_growth=inputs.terminal_growth.mode,
    )


__all__ = [
    "Fixed",
    "Triangular",
    "TruncatedNormal",
    "WACCInputs",
    "MCInputs",
    "MCValuationResult",
    "dcf_enterprise_values",
    "simulate_valuation",
    "tornado",
    "sensitivity_table",
    "zero_variance",
    "equity_bridge",
]
