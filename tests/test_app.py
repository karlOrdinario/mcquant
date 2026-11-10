"""Headless dashboard tests using Streamlit's AppTest harness."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def fresh(paths: int = 2**12) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.select_slider[0].set_value(paths)
    at.run()
    return at


def assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_app_loads_all_tabs():
    at = fresh()
    assert_clean(at)
    assert len(at.tabs) == 7
    labels = [m.label for m in at.metric]
    assert "Monte Carlo price" in labels
    assert "LSM price" in labels
    assert "Portfolio value" in labels


@pytest.mark.parametrize("model", ["GBM (Black-Scholes)", "Heston (stochastic vol)", "Merton (jump-diffusion)"])
@pytest.mark.parametrize("option", ["European", "Asian (arithmetic)", "Barrier", "Digital"])
def test_pricer_every_model_and_contract(model, option):
    at = fresh()
    at.selectbox[0].set_value(model)
    at.selectbox[1].set_value(option)
    at.run()
    assert_clean(at)
    price = next(m for m in at.metric if m.label == "Monte Carlo price")
    assert float(price.value.replace(",", "")) >= 0


def test_reference_error_is_small_for_gbm_european():
    at = fresh(2**16)
    z = next(m for m in at.metric if m.label == "Error in standard errors")
    assert abs(float(z.value)) < 4


def test_convergence_and_validation_buttons():
    at = fresh(2**13)
    at.button[0].click()  # convergence study
    at.run()
    assert_clean(at)
    at.button[1].click()  # validation suite
    at.run()
    assert_clean(at)
    passed = next(m for m in at.metric if m.label == "Benchmarks passed")
    n_pass, n_total = (int(x) for x in passed.value.split("/"))
    assert n_pass == n_total


def test_sampling_methods_and_digital_greeks():
    for method in ("pseudo", "sobol"):
        at = fresh()
        at.sidebar.radio[0].set_value(method)
        at.toggle[1].set_value(True)  # digital greeks
        at.run()
        assert_clean(at)


def test_invalid_input_shows_error_not_crash():
    at = fresh()
    at.selectbox[1].set_value("Barrier")
    at.run()
    at.selectbox[2].set_value("up-and-out")
    at.run()
    assert_clean(at)


def test_valuation_tab_shows_the_real_result():
    at = fresh()
    assert_clean(at)
    labels = {m.label: m.value for m in at.metric}
    for key in ("P5", "Median", "P95", "Market price", "P(value > price)", "Implied WACC"):
        assert key in labels
    assert labels["Market price"] == "₱6.40"
    assert any("Download the Excel model" in b.label for b in at.get("download_button"))


def test_valuation_sliders_move_the_answer():
    at = fresh()
    median_before = float(next(m for m in at.metric if m.label == "Median").value.strip("₱"))
    margin = next(s for s in at.slider if s.label == "EBITDA margin")
    margin.set_value(0.165)
    at.run()
    assert_clean(at)
    median_after = float(next(m for m in at.metric if m.label == "Median").value.strip("₱"))
    assert median_after < median_before
