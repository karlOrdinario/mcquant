"""Milestone 3: real Monde Nissin inputs. Every number must have a public source."""

import math
from pathlib import Path

import pytest
import yaml

from mcquant.valuation.company import (
    DEFAULT_ASSUMPTIONS,
    DEFAULT_COMPANY,
    _check_risk_free,
    build_valuation,
    implied_market_expectations,
    iter_sourced,
    load_company,
    load_yaml,
    missing_sources,
    run_valuation,
    value_of,
)
from mcquant.valuation.montecarlo import Triangular

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report():
    return run_valuation(build_valuation(), n_sims=20_000)


def test_every_company_number_has_a_source():
    """Plan section 6 item 8: fails if any number in config/monde.yaml lacks a source."""
    cfg = load_yaml(DEFAULT_COMPANY)
    numbers = list(iter_sourced(cfg))
    assert len(numbers) > 60
    assert missing_sources(cfg) == []
    assert "TODO_SOURCE" not in (ROOT / "config" / "monde.yaml").read_text(encoding="utf-8").split("never guessed.")[1]


def test_source_check_catches_a_missing_source(tmp_path):
    cfg = load_yaml(DEFAULT_COMPANY)
    cfg["market"]["share_price"].pop("source")
    cfg["market"]["shares_outstanding"]["value"] = "TODO_SOURCE"
    assert set(missing_sources(cfg)) == {"market.share_price", "market.shares_outstanding"}
    with pytest.raises(ValueError):
        value_of({"value": "TODO_SOURCE", "unit": "PHP thousands"})


def test_every_assumption_has_a_reason():
    a = load_yaml(DEFAULT_ASSUMPTIONS)
    specs = list(a["drivers"]["revenue_growth"].values())
    specs += [v for k, v in a["drivers"].items() if k != "revenue_growth"]
    specs += list(a["wacc"].values()) + [
        a["simulation"]["growth_autocorrelation"],
        a["simulation"]["growth_margin_correlation"],
    ]
    for spec in specs:
        assert len(str(spec.get("why", ""))) > 40, spec


def test_assumptions_document_is_up_to_date():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_md", ROOT / "scripts" / "build_assumptions_md.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (ROOT / "docs" / "ASSUMPTIONS.md").read_text(encoding="utf-8") == mod.build(), (
        "run: python scripts/build_assumptions_md.py"
    )


def test_internal_cross_checks_of_the_source_data():
    co = load_company()
    # Dividends payable at year end = P0.16 x shares outstanding (declared Nov 2025, paid Jan 2026)
    dp = co.bridge_items.set_index("item").loc["dividends_payable", "PHP m"]
    assert dp == pytest.approx(0.16 * co.shares_m, rel=1e-3)
    # The 'why' text quotes the historical ratios correctly
    h = co.history
    margins = (h.core_ebitda / h.net_sales).round(3).tolist()
    assert margins == [0.169, 0.193, 0.184]
    assert (h.capex / h.net_sales).round(4).tolist() == [0.0454, 0.058, 0.0546]
    assert round(h.net_sales.iloc[-1] / h.net_sales.iloc[-2] - 1, 3) == 0.040
    # market cap close to the S&P figure of P115.18bn at P6.41 (stockanalysis, 18 Sep 2026)
    assert 6.41 * co.shares_m == pytest.approx(115_180, rel=0.001)


def test_units_are_converted_once():
    co = load_company()
    assert co.history.net_sales.iloc[-1] == pytest.approx(86_483.039)  # PHP millions
    assert co.shares_m == pytest.approx(17_968.611496)  # millions of shares
    assert co.price == 6.40
    assert co.bridge.cash == pytest.approx(15_398.256)


def test_wacc_build_up(report):
    w = report.base_wacc
    assert w.cost_of_equity == pytest.approx(0.06514 + 0.58 * 0.0669, abs=1e-12)
    assert w.equity_weight == pytest.approx(0.95645, abs=1e-4)
    assert w.wacc == pytest.approx(0.1024, abs=5e-5)


def test_risk_free_must_stay_anchored_to_sourced_rates():
    rates = {"ph_10y_government_yield": 0.08134, "ph_moodys_rating_default_spread": 0.0162}
    _check_risk_free(Triangular(0.06, 0.06514, 0.08134), rates)
    with pytest.raises(ValueError):
        _check_risk_free(Triangular(0.06, 0.05, 0.08134), rates)
    with pytest.raises(ValueError):
        _check_risk_free(Triangular(0.06, 0.06514, 0.09), rates)


def test_real_valuation_is_coherent(report):
    s = report.mc.summary(report.price)
    assert s["discarded (WACC - g < 1%)"] == 0
    assert s["P5"] < s["median (P50)"] < s["P95"]
    assert 0 <= s["P(value > price)"] <= 1
    assert report.base_dcf.terminal_share_of_ev == pytest.approx(0.75, abs=0.05)
    assert "per share (90% range) versus a market price of ₱6.40" in report.headline
    # roll-forward: base value grows ~ cost of equity for 0.85 years, less the P0.24 dividend
    re = report.setup.base_cost_of_equity
    rolled = report.base_dcf.value_per_share * (1 + re) ** report.setup.roll_years - 0.24 * (1 + re) ** (
        report.setup.roll_years - report.setup.dividends[0][0]
    )
    assert report.base_value_at_price_date == pytest.approx(rolled)


def test_football_field_rows(report):
    labels = [r["label"] for r in report.football]
    assert labels[0].startswith("DCF Monte Carlo")
    for r in report.football:
        assert r["low"] <= r["high"]
        assert all(math.isfinite(x) for x in (r["low"], r["high"]))


def test_reverse_dcf_backs_out_the_price():
    setup = build_valuation()
    imp = implied_market_expectations(setup)
    assert set(imp) == {"implied_terminal_growth", "implied_ebitda_margin", "implied_wacc"}
    from dataclasses import replace

    from mcquant.valuation.company import _roll
    from mcquant.valuation.dcf import run_dcf

    v = run_dcf(
        replace(setup.base_case, ebitda_margin=imp["implied_ebitda_margin"]), setup.company.bridge
    ).value_per_share
    assert _roll(setup, v, setup.base_cost_of_equity) == pytest.approx(6.40, abs=1e-6)


def test_driver_override_changes_the_answer():
    base = run_valuation(build_valuation(), n_sims=5_000)
    low_margin = run_valuation(
        build_valuation(driver_overrides={"ebitda_margin": Triangular(0.14, 0.15, 0.16)}), n_sims=5_000
    )
    assert low_margin.mc.percentile(50) < base.mc.percentile(50)


def test_yaml_is_plain_and_loadable():
    for p in (DEFAULT_COMPANY, DEFAULT_ASSUMPTIONS):
        assert isinstance(yaml.safe_load(Path(p).read_text(encoding="utf-8")), dict)


def test_published_results_match_a_fresh_run():
    """docs/monde_results.json must not drift from the code."""
    import json

    fresh = run_valuation(build_valuation())  # full run: config's simulation count and seed
    published = json.loads((ROOT / "docs" / "monde_results.json").read_text(encoding="utf-8"))
    s = fresh.mc.summary(fresh.price)
    for key in ("median (P50)", "P5", "P95", "P(value > price)"):
        assert published["monte_carlo"][key] == pytest.approx(s[key], rel=1e-12)
    assert published["base_case"]["wacc"] == pytest.approx(fresh.base_wacc.wacc, rel=1e-12)
