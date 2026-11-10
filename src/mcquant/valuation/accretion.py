"""Merger accretion / dilution with uncertain synergies (stretch milestone M5).

A deal is **accretive** if the buyer's earnings per share (EPS) after the deal
are higher than they would have been without it, and **dilutive** if lower.

    Pro forma net income = buyer NI + target NI
                         + synergies x (1 - tax) - integration costs x (1 - tax)
                         - interest on new debt x (1 - tax)
                         - interest lost on cash used x (1 - tax)
    Pro forma shares     = buyer shares + new shares issued to pay for the target
    Accretion            = pro forma EPS / buyer standalone EPS - 1

Synergies and integration costs are uncertain, so we simulate them and report
the probability the deal is accretive in years 1 and 2.

**Hypothetical only.** The companies in the tests and notes (BuyerCo, TargetCo)
are invented. Nothing here describes or implies a real transaction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .inputs import _finite
from .montecarlo import Distribution, Fixed, Triangular, _as_dist


@dataclass(frozen=True)
class Company:
    name: str
    net_income: float  # year 1, PHP millions
    shares: float  # millions
    share_price: float
    net_income_growth: float = 0.0  # applied for year 2

    def __post_init__(self) -> None:
        for n in ("net_income", "shares", "share_price", "net_income_growth"):
            _finite(n, getattr(self, n))
        if self.shares <= 0 or self.share_price <= 0:
            raise ValueError("shares and share_price must be positive")

    @property
    def eps(self) -> float:
        return self.net_income / self.shares

    @property
    def pe(self) -> float:
        return self.share_price / self.eps


@dataclass(frozen=True)
class DealTerms:
    offer_premium: float  # e.g. 0.30 = 30% above the target's share price
    pct_cash: float
    pct_debt: float
    pct_stock: float
    debt_rate: float  # pre-tax interest on new debt
    cash_rate: float  # pre-tax interest the buyer earned on the cash it spends
    tax_rate: float
    synergies_run_rate: float | Distribution  # pre-tax, full-year
    synergy_phase_in: tuple[float, float] = (0.5, 1.0)  # share achieved in years 1 and 2
    integration_costs_y1: float | Distribution = 0.0  # pre-tax, one-off in year 1

    def __post_init__(self) -> None:
        mix = self.pct_cash + self.pct_debt + self.pct_stock
        if abs(mix - 1) > 1e-9 or min(self.pct_cash, self.pct_debt, self.pct_stock) < 0:
            raise ValueError("financing mix must be non-negative and sum to 100%")
        if not 0 <= self.tax_rate < 1:
            raise ValueError("tax_rate must be in [0, 1)")
        if self.offer_premium < -0.5:
            raise ValueError("offer_premium looks wrong")


def purchase_price(target: Company, terms: DealTerms) -> float:
    """Equity purchase price in PHP millions."""
    return target.share_price * (1 + terms.offer_premium) * target.shares


def pro_forma(
    buyer: Company,
    target: Company,
    terms: DealTerms,
    synergies: np.ndarray | float,
    integration: np.ndarray | float,
    year: int,
) -> dict[str, np.ndarray | float]:
    """Pro forma EPS and accretion for year 1 or 2 (vectorised over scenarios)."""
    if year not in (1, 2):
        raise ValueError("year must be 1 or 2")
    t = terms.tax_rate
    price = purchase_price(target, terms)
    new_debt, cash_used = terms.pct_debt * price, terms.pct_cash * price
    new_shares = terms.pct_stock * price / buyer.share_price
    buyer_ni = buyer.net_income * (1 + buyer.net_income_growth) ** (year - 1)
    target_ni = target.net_income * (1 + target.net_income_growth) ** (year - 1)
    syn = np.asarray(synergies) * terms.synergy_phase_in[year - 1]
    integ = np.asarray(integration) if year == 1 else 0.0
    pf_ni = (
        buyer_ni
        + target_ni
        + syn * (1 - t)
        - integ * (1 - t)
        - new_debt * terms.debt_rate * (1 - t)
        - cash_used * terms.cash_rate * (1 - t)
    )
    pf_shares = buyer.shares + new_shares
    standalone = buyer_ni / buyer.shares
    pf_eps = pf_ni / pf_shares
    return {
        "standalone_eps": standalone,
        "pro_forma_eps": pf_eps,
        "accretion": pf_eps / standalone - 1,
        "new_shares": new_shares,
        "pro_forma_net_income": pf_ni,
    }


def breakeven_synergies(
    buyer: Company, target: Company, terms: DealTerms, year: int = 1, integration: float | None = None
) -> float:
    """Pre-tax full-year (run-rate) synergies needed for zero accretion in ``year``."""
    integ = _as_dist(terms.integration_costs_y1).mode if integration is None else integration
    base = pro_forma(buyer, target, terms, 0.0, integ, year)
    shortfall = (base["standalone_eps"] - base["pro_forma_eps"]) * (buyer.shares + base["new_shares"])
    return float(shortfall / ((1 - terms.tax_rate) * terms.synergy_phase_in[year - 1]))


def simulate_accretion(
    buyer: Company, target: Company, terms: DealTerms, n_sims: int = 50_000, seed: int | None = 2026
) -> pd.DataFrame:
    """Accretion in years 1 and 2 for random synergies and integration costs."""
    if n_sims < 100:
        raise ValueError("n_sims must be at least 100")
    rng = np.random.default_rng(seed)
    syn = _as_dist(terms.synergies_run_rate).ppf(rng.uniform(size=n_sims))
    integ = _as_dist(terms.integration_costs_y1).ppf(rng.uniform(size=n_sims))
    y1 = pro_forma(buyer, target, terms, syn, integ, 1)["accretion"]
    y2 = pro_forma(buyer, target, terms, syn, integ, 2)["accretion"]
    return pd.DataFrame({"synergies": syn, "integration_costs": integ, "accretion_y1": y1, "accretion_y2": y2})


def summarise(sims: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "P(accretive, year 1)": float((sims.accretion_y1 > 0).mean()),
            "P(accretive, year 2)": float((sims.accretion_y2 > 0).mean()),
            "median accretion, year 1": float(sims.accretion_y1.median()),
            "median accretion, year 2": float(sims.accretion_y2.median()),
            "P5 accretion, year 1": float(sims.accretion_y1.quantile(0.05)),
            "P95 accretion, year 1": float(sims.accretion_y1.quantile(0.95)),
        }
    )


TOY_BUYER = Company(
    "BuyerCo (hypothetical)", net_income=1_000.0, shares=500.0, share_price=24.0, net_income_growth=0.05
)
TOY_TARGET = Company(
    "TargetCo (hypothetical)", net_income=200.0, shares=100.0, share_price=26.0, net_income_growth=0.05
)
TOY_TERMS = DealTerms(
    offer_premium=0.35,
    pct_cash=0.20,
    pct_debt=0.30,
    pct_stock=0.50,
    debt_rate=0.08,
    cash_rate=0.04,
    tax_rate=0.25,
    synergies_run_rate=Triangular(0.0, 100.0, 180.0),
    synergy_phase_in=(0.5, 1.0),
    integration_costs_y1=Triangular(20.0, 40.0, 80.0),
)

__all__ = [
    "Company",
    "DealTerms",
    "purchase_price",
    "pro_forma",
    "breakeven_synergies",
    "simulate_accretion",
    "summarise",
    "TOY_BUYER",
    "TOY_TARGET",
    "TOY_TERMS",
    "Fixed",
    "Triangular",
]
