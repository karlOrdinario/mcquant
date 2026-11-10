"""DCF tests. The toy company is solved by hand and the code must reproduce it exactly."""

from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mcquant.valuation import (
    DCFAssumptions,
    EquityBridge,
    equity_bridge,
    exit_multiple_terminal_value,
    forecast_fcff,
    gordon_terminal_value,
    implied_exit_multiple,
    implied_perpetual_growth,
    implied_upside,
    run_dcf,
)


def toy(**overrides) -> DCFAssumptions:
    params = dict(
        base_revenue=1_000,
        revenue_growth=[0.10, 0.08, 0.05],
        ebitda_margin=0.20,
        da_pct_revenue=0.05,
        capex_pct_revenue=0.06,
        nwc_pct_revenue_change=0.10,
        tax_rate=0.25,
        wacc=0.10,
        terminal_growth=0.03,
    )
    params.update(overrides)
    return DCFAssumptions(**params)


TOY_BRIDGE = EquityBridge(debt=300, cash=100, diluted_shares=100, minority_interest=20)


# --------------------------------------------------------------------------- #
# The hand-checked toy company
# --------------------------------------------------------------------------- #
def test_toy_company_matches_hand_calculation():
    """Every number in the note 03 tables, to full precision."""
    f = forecast_fcff(toy())
    np.testing.assert_allclose(f["revenue"], [1_100, 1_188, 1_247.4], rtol=1e-14)
    np.testing.assert_allclose(f["ebitda"], [220, 237.6, 249.48], rtol=1e-14)
    np.testing.assert_allclose(f["d&a"], [55, 59.4, 62.37], rtol=1e-14)
    np.testing.assert_allclose(f["ebit"], [165, 178.2, 187.11], rtol=1e-14)
    np.testing.assert_allclose(f["nopat"], [123.75, 133.65, 140.3325], rtol=1e-14)
    np.testing.assert_allclose(f["capex"], [66, 71.28, 74.844], rtol=1e-14)
    np.testing.assert_allclose(f["change_nwc"], [10, 8.8, 5.94], rtol=1e-12)
    np.testing.assert_allclose(f["fcff"], [102.75, 112.97, 121.9185], rtol=1e-13)
    np.testing.assert_allclose(f["pv_fcff"].round(2), [93.41, 93.36, 91.60])

    r = run_dcf(toy(), TOY_BRIDGE)
    assert r.sum_pv_fcff == pytest.approx(278.3719008264463, rel=1e-12)
    assert r.terminal_value == pytest.approx(1_793.9436428571428, rel=1e-12)
    assert r.pv_terminal_value == pytest.approx(1_347.8164108618655, rel=1e-12)
    assert r.enterprise_value == pytest.approx(1_626.1883116883116, rel=1e-12)
    assert r.equity_value == pytest.approx(1_406.1883116883116, rel=1e-12)
    assert r.value_per_share == pytest.approx(14.061883116883116, rel=1e-12)
    assert round(r.terminal_share_of_ev, 3) == 0.829
    assert implied_upside(r.value_per_share, 12.0) == pytest.approx(0.17182, abs=1e-5)


def test_toy_company_with_exact_fractions():
    """An independent re-computation in exact rational arithmetic (no floats,
    no NumPy) must agree with the engine to 12 significant figures."""
    rev, growth = Fraction(1000), [Fraction(10, 100), Fraction(8, 100), Fraction(5, 100)]
    wacc, g, tax = Fraction(10, 100), Fraction(3, 100), Fraction(25, 100)
    pv_sum, prev = Fraction(0), rev
    for t, gr in enumerate(growth, start=1):
        r = prev * (1 + gr)
        ebitda, da = r * Fraction(20, 100), r * Fraction(5, 100)
        fcff = (ebitda - da) * (1 - tax) + da - r * Fraction(6, 100) - (r - prev) * Fraction(10, 100)
        pv_sum += fcff / (1 + wacc) ** t
        prev = r
    tv = fcff * (1 + g) / (wacc - g)
    ev = pv_sum + tv / (1 + wacc) ** 3
    per_share = (ev - 300 + 100 - 20) / 100
    assert run_dcf(toy(), TOY_BRIDGE).value_per_share == pytest.approx(float(per_share), rel=1e-12)


# --------------------------------------------------------------------------- #
# Terminal value and cross-checks
# --------------------------------------------------------------------------- #
def test_exit_multiple_method():
    r = run_dcf(toy(), TOY_BRIDGE, terminal_method="exit_multiple", exit_multiple=8.0)
    assert r.terminal_value == pytest.approx(1_995.84)
    assert r.pv_terminal_value == pytest.approx(1_499.504132231405)
    assert r.value_per_share == pytest.approx(15.578760330578513)
    assert r.implied_exit_multiple == pytest.approx(8.0)
    assert r.implied_growth == pytest.approx(0.03667344505995372)


def test_gordon_implies_a_multiple_and_growth_round_trips():
    r = run_dcf(toy(), TOY_BRIDGE)
    assert r.implied_exit_multiple == pytest.approx(7.190731292517007)
    assert r.implied_growth == pytest.approx(0.03, abs=1e-14)


@settings(max_examples=200, deadline=None)
@given(
    fcff=st.floats(1, 1e6),
    wacc=st.floats(0.03, 0.25),
    spread=st.floats(0.01, 0.10),
)
def test_implied_growth_inverts_gordon(fcff, wacc, spread):
    g = wacc - spread
    tv = gordon_terminal_value(fcff, wacc, g)
    assert implied_perpetual_growth(tv, fcff, wacc) == pytest.approx(g, abs=1e-9)


def test_zero_growth_company_equals_a_perpetuity():
    """Flat revenue, no growth anywhere: EV must equal FCFF / WACC exactly."""
    a = toy(revenue_growth=[0.0] * 5, terminal_growth=0.0)
    r = run_dcf(a, TOY_BRIDGE)
    fcff = r.forecast["fcff"].iloc[0]
    assert r.forecast["fcff"].nunique() == 1
    assert r.enterprise_value == pytest.approx(fcff / 0.10, rel=1e-12)


def test_wacc_spread_guard():
    with pytest.raises(ValueError, match="minimum"):
        toy(terminal_growth=0.095)
    with pytest.raises(ValueError, match="minimum"):
        gordon_terminal_value(100, 0.08, 0.075)
    assert toy(terminal_growth=0.09).wacc == 0.10  # exactly 1 point is allowed


def test_spread_of_exactly_one_point_survives_float_round_off():
    """Found by Hypothesis: 0.125 - 0.115 = 0.00999... in floating point and was
    wrongly rejected, although the plan allows WACC - g >= 1%."""
    assert 0.125 - 0.115 < 0.01  # the float trap itself
    assert gordon_terminal_value(100, 0.125, 0.115) == pytest.approx(100 * 1.115 / 0.01)
    assert toy(wacc=0.125, terminal_growth=0.115).terminal_growth == 0.115


def test_exit_multiple_needs_a_multiple():
    with pytest.raises(ValueError):
        run_dcf(toy(), TOY_BRIDGE, terminal_method="exit_multiple")
    with pytest.raises(ValueError):
        run_dcf(toy(), TOY_BRIDGE, terminal_method="multiples")
    with pytest.raises(ValueError):
        exit_multiple_terminal_value(100, 0)
    with pytest.raises(ValueError):
        implied_exit_multiple(1_000, 0)


# --------------------------------------------------------------------------- #
# Mid-year convention
# --------------------------------------------------------------------------- #
def test_mid_year_convention():
    r = run_dcf(toy(mid_year=True), TOY_BRIDGE)
    np.testing.assert_allclose(r.forecast["discount_period"], [0.5, 1.5, 2.5])
    assert r.terminal_discount_period == 2.5  # Gordon TV: N - 0.5
    assert r.value_per_share == pytest.approx(14.855606900895815, rel=1e-12)
    assert r.value_per_share > run_dcf(toy(), TOY_BRIDGE).value_per_share
    exit_mid = run_dcf(toy(mid_year=True), TOY_BRIDGE, "exit_multiple", 8.0)
    assert exit_mid.terminal_discount_period == 3.0  # exit TV: sale at year end


# --------------------------------------------------------------------------- #
# Equity bridge and economic sanity
# --------------------------------------------------------------------------- #
def test_equity_bridge_note_06_check_question():
    equity, per_share = equity_bridge(
        5_000, EquityBridge(debt=1_200, cash=400, diluted_shares=200, minority_interest=100)
    )
    assert equity == pytest.approx(4_100)
    assert per_share == pytest.approx(20.50)


@pytest.mark.parametrize(
    "change,direction",
    [
        (dict(wacc=0.11), -1),
        (dict(terminal_growth=0.04), +1),
        (dict(ebitda_margin=0.22), +1),
        (dict(capex_pct_revenue=0.08), -1),
        (dict(tax_rate=0.30), -1),
        (dict(nwc_pct_revenue_change=0.20), -1),
    ],
)
def test_value_moves_in_the_right_direction(change, direction):
    base = run_dcf(toy(), TOY_BRIDGE).value_per_share
    moved = run_dcf(toy(**change), TOY_BRIDGE).value_per_share
    assert np.sign(moved - base) == direction


def test_per_year_drivers():
    same = run_dcf(toy(ebitda_margin=[0.20, 0.20, 0.20]), TOY_BRIDGE).value_per_share
    assert same == pytest.approx(run_dcf(toy(), TOY_BRIDGE).value_per_share)
    expanding = run_dcf(toy(ebitda_margin=[0.20, 0.21, 0.22]), TOY_BRIDGE).value_per_share
    assert expanding > same


def test_more_cash_or_less_debt_raises_value_one_for_one():
    a = run_dcf(toy(), TOY_BRIDGE).value_per_share
    b = run_dcf(toy(), EquityBridge(debt=300, cash=200, diluted_shares=100, minority_interest=20)).value_per_share
    assert b - a == pytest.approx(1.0)  # +100 cash / 100 shares


@pytest.mark.parametrize(
    "bad",
    [
        lambda: toy(base_revenue=0),
        lambda: toy(revenue_growth=[]),
        lambda: toy(revenue_growth=[0.1, -1.0, 0.1]),
        lambda: toy(ebitda_margin=[0.2, 0.2]),  # wrong length
        lambda: toy(ebitda_margin=1.5),
        lambda: toy(capex_pct_revenue=-0.01),
        lambda: toy(tax_rate=1.0),
        lambda: toy(wacc=0.0),
        lambda: toy(wacc=10),  # percent as a whole number
        lambda: toy(revenue_growth=[0.1, float("nan"), 0.1]),
        lambda: toy(base_revenue="big"),
        lambda: EquityBridge(debt=-1, cash=0, diluted_shares=1),
        lambda: EquityBridge(debt=0, cash=0, diluted_shares=0),
        lambda: implied_upside(10, 0),
    ],
)
def test_bad_inputs_raise_instead_of_nan(bad):
    with pytest.raises(ValueError):
        bad()


def test_summary_table():
    s = run_dcf(toy(), TOY_BRIDGE).summary()
    assert s["Value per share"] == pytest.approx(14.061883116883116)
    assert not s.isna().any()
