"""Accretion / dilution (stretch milestone M5). Hypothetical companies only."""

from dataclasses import replace

import numpy as np
import pytest

from mcquant.valuation.accretion import (
    TOY_BUYER,
    TOY_TARGET,
    TOY_TERMS,
    Company,
    DealTerms,
    breakeven_synergies,
    pro_forma,
    purchase_price,
    simulate_accretion,
    summarise,
)
from mcquant.valuation.montecarlo import Fixed


def test_toy_deal_matches_hand_calculation():
    assert purchase_price(TOY_TARGET, TOY_TERMS) == pytest.approx(3_510.0)
    y1 = pro_forma(TOY_BUYER, TOY_TARGET, TOY_TERMS, 100.0, 40.0, 1)
    assert y1["new_shares"] == pytest.approx(73.125)
    assert y1["pro_forma_net_income"] == pytest.approx(1_123.26)
    assert y1["accretion"] == pytest.approx(-0.020056706652126555)
    y2 = pro_forma(TOY_BUYER, TOY_TARGET, TOY_TERMS, 100.0, 40.0, 2)
    assert y2["pro_forma_net_income"] == pytest.approx(1_250.76)
    assert y2["accretion"] == pytest.approx(0.039214830970556)


def test_breakeven_synergies_give_zero_accretion():
    for year in (1, 2):
        s = breakeven_synergies(TOY_BUYER, TOY_TARGET, TOY_TERMS, year)
        acc = pro_forma(TOY_BUYER, TOY_TARGET, TOY_TERMS, s, 40.0, year)["accretion"]
        assert acc == pytest.approx(0.0, abs=1e-12)


def test_all_stock_rule_of_thumb():
    """No synergies, all stock: accretive exactly when the buyer's P/E exceeds the offer P/E."""
    target = Company("T", 200.0, 100.0, 20.0)  # EPS 2.00
    for buyer_price, expect in ((40.0, True), (24.0, False)):  # buyer P/E 20x or 12x vs offer P/E 13x
        buyer = Company("B", 1_000.0, 500.0, buyer_price)
        terms = DealTerms(0.30, 0, 0, 1, 0.08, 0.04, 0.25, synergies_run_rate=0.0)
        acc = pro_forma(buyer, target, terms, 0.0, 0.0, 1)["accretion"]
        assert bool(acc > 0) == expect


def test_more_synergies_more_accretion():
    lo = pro_forma(TOY_BUYER, TOY_TARGET, TOY_TERMS, 50.0, 40.0, 2)["accretion"]
    hi = pro_forma(TOY_BUYER, TOY_TARGET, TOY_TERMS, 150.0, 40.0, 2)["accretion"]
    assert hi > lo


def test_simulation_probabilities():
    s = summarise(simulate_accretion(TOY_BUYER, TOY_TARGET, TOY_TERMS))
    assert 0 <= s["P(accretive, year 1)"] < s["P(accretive, year 2)"] <= 1
    assert s["P5 accretion, year 1"] < s["median accretion, year 1"] < s["P95 accretion, year 1"]


def test_zero_uncertainty_simulation_equals_deterministic():
    terms = replace(TOY_TERMS, synergies_run_rate=Fixed(100.0), integration_costs_y1=Fixed(40.0))
    sims = simulate_accretion(TOY_BUYER, TOY_TARGET, terms, n_sims=500)
    assert np.allclose(sims.accretion_y1, -0.020056706652126555)


@pytest.mark.parametrize(
    "bad",
    [
        lambda: DealTerms(0.3, 0.5, 0.5, 0.5, 0.08, 0.04, 0.25, 0.0),
        lambda: DealTerms(0.3, -0.1, 0.6, 0.5, 0.08, 0.04, 0.25, 0.0),
        lambda: DealTerms(0.3, 0.2, 0.3, 0.5, 0.08, 0.04, 1.2, 0.0),
        lambda: Company("X", 100.0, 0.0, 10.0),
        lambda: pro_forma(TOY_BUYER, TOY_TARGET, TOY_TERMS, 0.0, 0.0, 3),
        lambda: simulate_accretion(TOY_BUYER, TOY_TARGET, TOY_TERMS, n_sims=10),
    ],
)
def test_validation(bad):
    with pytest.raises(ValueError):
        bad()
