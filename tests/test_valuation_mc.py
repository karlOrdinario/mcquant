"""Monte Carlo valuation tests (plan section 6, items 3 to 6)."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import spearmanr

from mcquant.valuation import DCFAssumptions, EquityBridge, charts, run_dcf
from mcquant.valuation.dcf import roll_forward
from mcquant.valuation.montecarlo import (
    Fixed,
    MCInputs,
    Triangular,
    TruncatedNormal,
    WACCInputs,
    dcf_enterprise_values,
    sensitivity_table,
    simulate_valuation,
    tornado,
    zero_variance,
)

BASE = DCFAssumptions(1_000, [0.10, 0.08, 0.05], 0.20, 0.05, 0.06, 0.10, 0.25, 0.10, 0.03)
BRIDGE = EquityBridge(debt=300, cash=100, diluted_shares=100, minority_interest=20)
WACC_IN = WACCInputs(0.05, TruncatedNormal(1.0, 0.15, 0.6, 1.4), Triangular(0.05, 0.06, 0.075), 0.08, 0.25, 0.8)


def toy_inputs(**changes) -> MCInputs:
    params = dict(
        base=BASE,
        wacc_inputs=WACC_IN,
        revenue_growth=[Triangular(0.06, 0.10, 0.12), Triangular(0.05, 0.08, 0.10), Triangular(0.02, 0.05, 0.07)],
        ebitda_margin=Triangular(0.17, 0.20, 0.22),
        capex_pct_revenue=Triangular(0.05, 0.06, 0.07),
        terminal_growth=Triangular(0.02, 0.03, 0.04),
    )
    params.update(changes)
    return MCInputs(**params)


@pytest.mark.parametrize("method", ["pseudo", "antithetic", "sobol"])
def test_zero_variance_equals_deterministic_dcf(method):
    """Plan item 3: collapse every range to a point and Monte Carlo must equal the DCF."""
    r = simulate_valuation(zero_variance(toy_inputs()), BRIDGE, 2**12, seed=1, method=method)
    np.testing.assert_allclose(r.value_per_share, run_dcf(BASE, BRIDGE).value_per_share, rtol=1e-12)
    assert r.deterministic_value == pytest.approx(14.061883116883116, rel=1e-12)
    assert r.n_discarded == 0


def test_vectorised_engine_matches_run_dcf_scenario_by_scenario():
    rng = np.random.default_rng(0)
    n = 40
    growth = rng.uniform(-0.02, 0.15, (n, 3))
    margin, capex = rng.uniform(0.1, 0.3, n), rng.uniform(0.02, 0.09, n)
    wacc, g = rng.uniform(0.07, 0.14, n), rng.uniform(0.0, 0.05, n)
    ev = dcf_enterprise_values(BASE, growth, margin, capex, wacc, g)
    for i in range(n):
        a = replace(
            BASE,
            revenue_growth=list(growth[i]),
            ebitda_margin=margin[i],
            capex_pct_revenue=capex[i],
            wacc=wacc[i],
            terminal_growth=g[i],
        )
        assert ev[i] == pytest.approx(run_dcf(a, BRIDGE).enterprise_value, rel=1e-12)


def test_standard_error_shrinks_like_one_over_root_n():
    """Plan item 4: 4x the simulations should roughly halve the standard error."""
    se = {
        n: simulate_valuation(toy_inputs(), BRIDGE, n, seed=7).summary()["std error of mean"]
        for n in (5_000, 20_000, 80_000)
    }
    assert se[20_000] / se[5_000] == pytest.approx(0.5, abs=0.05)
    assert se[80_000] / se[20_000] == pytest.approx(0.5, abs=0.05)


def test_discarded_samples_are_counted_and_excluded():
    """Plan item 5: draws with WACC - g < 1% are dropped and reported."""
    wide_g = toy_inputs(terminal_growth=Triangular(0.02, 0.03, 0.12))
    r = simulate_valuation(wide_g, BRIDGE, 20_000, seed=3)
    assert r.n_discarded > 0
    assert r.n_kept + r.n_discarded == 20_000
    assert (r.drivers["wacc"] - r.drivers["terminal_growth"] >= 0.01 - 1e-12).all()
    assert r.summary()["discarded (WACC - g < 1%)"] == r.n_discarded


def test_sanity_bounds():
    """Plan item 6: probabilities in [0, 1]; percentiles ordered."""
    s = simulate_valuation(toy_inputs(), BRIDGE, 20_000, seed=4).summary(market_price=12.0)
    assert 0 <= s["P(value > price)"] <= 1
    assert s["P5"] <= s["P25"] <= s["median (P50)"] <= s["P75"] <= s["P95"]
    assert s["std error of P"] < 0.01


def test_reproducible_with_seed():
    a = simulate_valuation(toy_inputs(), BRIDGE, 5_000, seed=11).value_per_share
    b = simulate_valuation(toy_inputs(), BRIDGE, 5_000, seed=11).value_per_share
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("method", ["antithetic", "sobol"])
def test_variance_reduced_samplers_agree(method):
    ref = simulate_valuation(toy_inputs(), BRIDGE, 2**17, seed=5).summary()
    alt = simulate_valuation(toy_inputs(), BRIDGE, 2**15, seed=6, method=method).summary()
    assert alt["mean"] == pytest.approx(ref["mean"], abs=0.08)
    assert alt["P95"] == pytest.approx(ref["P95"], abs=0.2)


def test_correlations_are_applied():
    r = simulate_valuation(toy_inputs(), BRIDGE, 40_000, seed=8)
    d = r.drivers
    assert spearmanr(d["growth_y1"], d["growth_y2"])[0] == pytest.approx(0.6, abs=0.05)
    assert spearmanr(d["growth_y1"], d["growth_y3"])[0] == pytest.approx(0.36, abs=0.05)
    assert spearmanr(d["growth_y1"], d["ebitda_margin"])[0] == pytest.approx(0.3, abs=0.05)
    assert abs(spearmanr(d["growth_y1"], d["beta"])[0]) < 0.03


def test_distributions_respect_their_bounds():
    u = np.linspace(0.0001, 0.9999, 1001)
    t = Triangular(1, 2, 5).ppf(u)
    assert t.min() >= 1 and t.max() <= 5
    tn = TruncatedNormal(1.0, 0.5, 0.6, 1.2).ppf(u)
    assert tn.min() >= 0.6 and tn.max() <= 1.2
    assert Triangular(3, 3, 3).quantile(0.9) == 3
    assert Fixed(2.5).quantile(0.1) == 2.5
    with pytest.raises(ValueError):
        Triangular(2, 1, 3)
    with pytest.raises(ValueError):
        TruncatedNormal(0, -1, -1, 1)


def test_inconsistent_inputs_raise():
    with pytest.raises(ValueError):
        toy_inputs(revenue_growth=[0.1, 0.1])  # wrong number of years
    with pytest.raises(ValueError):
        toy_inputs(growth_autocorrelation=1.0)
    with pytest.raises(ValueError):
        simulate_valuation(toy_inputs(), BRIDGE, 10)
    with pytest.raises(ValueError):
        WACCInputs(0.05, 1.0, 0.06, 0.08, 0.25, 0.0)


def test_tornado():
    t = tornado(toy_inputs(), BRIDGE)
    assert list(t["swing"]) == sorted(t["swing"], reverse=True)
    assert t["base value"].iloc[0] == pytest.approx(14.061883116883116)
    beta = t.set_index("driver").loc["Beta"]
    assert beta["value at high input"] < beta["base value"] < beta["value at low input"]  # higher beta, lower value
    margin = t.set_index("driver").loc["EBITDA margin"]
    assert margin["value at high input"] > margin["base value"] > margin["value at low input"]


def test_sensitivity_table_matches_notes_and_flags_invalid_cells():
    s = sensitivity_table(BASE, BRIDGE, [0.08, 0.09, 0.10, 0.11], [0.02, 0.03, 0.075])
    assert s.loc[0.10, 0.03] == pytest.approx(14.061883116883116)
    assert s.loc[0.09, 0.03] == pytest.approx(16.796233482030132)  # note 04 table
    assert s.loc[0.11, 0.03] == pytest.approx(12.011539241944645)
    assert np.isnan(s.loc[0.08, 0.075])  # spread 0.5%: not meaningful
    assert np.isfinite(s.loc[0.09, 0.075])  # spread 1.5%: allowed


def test_roll_forward():
    assert roll_forward(10.0, 0.10, 1.0) == pytest.approx(11.0)
    # a 1.00 dividend paid half-way through the year leaves the company
    assert roll_forward(10.0, 0.10, 1.0, [(0.5, 1.0)]) == pytest.approx(11.0 - 1.1**0.5)
    with pytest.raises(ValueError):
        roll_forward(10.0, 0.10, 1.0, [(2.0, 1.0)])
    zv = zero_variance(toy_inputs())
    r = simulate_valuation(zv, BRIDGE, 1_000, seed=1, roll_forward_years=0.75, dividends=[(0.4, 0.5)])
    re = 0.05 + 1.0 * 0.06
    expected = roll_forward(14.061883116883116, re, 0.75, [(0.4, 0.5)])
    np.testing.assert_allclose(r.value_per_share, expected, rtol=1e-12)
    assert r.deterministic_value == pytest.approx(expected)


def test_charts_render():
    r = simulate_valuation(toy_inputs(), BRIDGE, 5_000, seed=1)
    figs = [
        charts.histogram(r.value_per_share, 12.0),
        charts.tornado(tornado(toy_inputs(), BRIDGE)),
        charts.sensitivity_heatmap(
            sensitivity_table(BASE, BRIDGE, [0.09, 0.10], [0.02, 0.03, 0.095]), 0.10, 0.03, 12.0
        ),
        charts.football_field([dict(label="DCF", low=10, high=18, mid=13), dict(label="52w", low=9, high=13)], 12.0),
    ]
    for f in figs:
        assert f.axes
    # A probability of 99.6% must not be printed as a rounded "100%"
    texts = [t.get_text() for t in charts.histogram(np.linspace(6.39, 20, 1000), 6.40).axes[0].texts]
    assert any("> 99%" in t for t in texts) and not any("100%" in t for t in texts)
