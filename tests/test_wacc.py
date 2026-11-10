"""WACC tests: known inputs give known outputs."""

import pytest

from mcquant.valuation import after_tax_cost_of_debt, cost_of_equity_capm, wacc, wacc_from_capm


def test_toy_company_wacc_is_exactly_ten_percent():
    """Note 04 worked example: Rf 5%, beta 1.0, ERP 6%, Rd 8%, tax 25%, E 1,200, D 300."""
    res = wacc_from_capm(0.05, 1.0, 0.06, 0.08, 0.25, equity_value=1_200, debt_value=300)
    assert res.cost_of_equity == pytest.approx(0.11, abs=1e-15)
    assert res.after_tax_cost_of_debt == pytest.approx(0.06, abs=1e-15)
    assert res.equity_weight == pytest.approx(0.80, abs=1e-15)
    assert res.debt_weight == pytest.approx(0.20, abs=1e-15)
    assert res.wacc == pytest.approx(0.10, abs=1e-15)
    assert "WACC 10.00%" in str(res)


def test_note_04_check_questions():
    assert cost_of_equity_capm(0.06, 0.8, 0.07) == pytest.approx(0.116)
    assert after_tax_cost_of_debt(0.10, 0.30) == pytest.approx(0.07)


def test_no_debt_means_wacc_equals_cost_of_equity():
    res = wacc(1_000, 0, 0.12, 0.07, 0.25)
    assert res.wacc == pytest.approx(0.12)
    assert res.debt_weight == 0


def test_more_debt_lowers_wacc_when_debt_is_cheaper():
    low = wacc(900, 100, 0.12, 0.07, 0.25).wacc
    high = wacc(600, 400, 0.12, 0.07, 0.25).wacc
    assert high < low


def test_weights_sum_to_one():
    res = wacc(1_234, 567, 0.11, 0.06, 0.3)
    assert res.equity_weight + res.debt_weight == pytest.approx(1.0)


def test_defensive_beta_lowers_cost_of_equity():
    assert cost_of_equity_capm(0.06, 0.7, 0.06) < cost_of_equity_capm(0.06, 1.0, 0.06)


@pytest.mark.parametrize(
    "bad",
    [
        lambda: cost_of_equity_capm(6.0, 1.0, 0.06),  # percent passed as a whole number
        lambda: cost_of_equity_capm(0.06, 1.0, 6.0),
        lambda: cost_of_equity_capm(0.06, float("nan"), 0.06),
        lambda: after_tax_cost_of_debt(8.0, 0.25),
        lambda: after_tax_cost_of_debt(0.08, 1.0),
        lambda: wacc(0, 100, 0.1, 0.05, 0.25),
        lambda: wacc(100, -1, 0.1, 0.05, 0.25),
        lambda: wacc("a lot", 100, 0.1, 0.05, 0.25),
    ],
)
def test_bad_inputs_raise(bad):
    with pytest.raises(ValueError):
        bad()
