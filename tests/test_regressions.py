"""Regression tests for issues found in code review."""

import tracemalloc

import numpy as np
import pytest

from mcquant.analytic import bs_greeks, bs_price, merton_price
from mcquant.models import GBM, Heston, Merton
from mcquant.payoffs import European
from mcquant.pricer import mc_price, stock_control
from mcquant.risk import Portfolio, Position, simulate_log_returns, stress_test


@pytest.mark.parametrize("args", [(100, 100, 1, 0.05, 0.0), (100, 0, 1, 0.05, 0.2), (-5, 100, 1, 0.05, 0.2)])
def test_black_scholes_rejects_invalid_inputs_instead_of_nan(args):
    with pytest.raises(ValueError):
        bs_price(*args)
    with pytest.raises(ValueError):
        bs_greeks(*args)


def test_singular_covariance_is_accepted():
    """Perfectly correlated assets give a PSD (not PD) covariance; Cholesky fails, eigh works."""
    cov = np.array([[0.04, 0.06], [0.06, 0.09]])  # corr = 1
    r = simulate_log_returns(np.zeros(2), cov, 252, 200_000, seed=1)
    assert np.corrcoef(r, rowvar=False)[0, 1] == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(np.cov(r, rowvar=False), cov, rtol=0.02)


def test_stress_shock_below_minus_100pct_rejected():
    pf = Portfolio(np.array([100.0]), [Position(0, 1)])
    with pytest.raises(ValueError):
        stress_test(pf, {"impossible": {"spot_shock": -1.5}})


def test_merton_series_converges_for_frequent_jumps():
    """Was hard-capped at 200 terms; lam*T = 300 needs more."""
    m = Merton(100, 0.05, 0.15, 300.0, -0.002, 0.01)
    ref = merton_price(100, 100, 1, 0.05, 0.15, 300.0, -0.002, 0.01)
    res = mc_price(m, European(100), 1, 1, 2**17, "antithetic", seed=1, control=stock_control(m))
    assert res.within(ref, n_se=4)


@pytest.mark.parametrize(
    "model,budget",
    [
        (GBM(100, 0.05, 0.2), 3.2),
        (Merton(100, 0.05, 0.2, 1, -0.1, 0.1), 4.2),
        (Heston(100, 0.03, 0.04, 1.5, 0.04, 0.5, -0.7), 5.2),
    ],
)
def test_simulation_memory_budget(model, budget):
    """Peak memory in units of one (n_paths x n_steps) float64 matrix.
    Before the review fix Heston peaked at ~9x, GBM at ~5x."""
    n, m = 20_000, 252
    tracemalloc.start()
    model.simulate(1.0, m, n, seed=1)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak / (n * (m + 1) * 8) < budget


@pytest.mark.parametrize("strike", [90.0, 100.0, 120.0])
def test_antithetic_plus_control_variate_beats_each_alone(strike):
    """Found in QA: beta fitted on raw draws made antithetic+CV worse than CV alone."""
    m = GBM(100, 0.05, 0.2)
    pay = European(strike)
    cv = mc_price(m, pay, 1, 1, 2**16, "pseudo", seed=3, control=stock_control(m)).std_error
    anti = mc_price(m, pay, 1, 1, 2**16, "antithetic", seed=3).std_error
    both = mc_price(m, pay, 1, 1, 2**16, "antithetic", seed=3, control=stock_control(m)).std_error
    assert both < min(cv, anti) * 1.02


def test_rqmc_interval_uses_student_t_and_covers():
    """Found in QA: 16 batches + normal quantile covered only ~90%. Now 32 batches + t(31)."""
    from mcquant.analytic import bs_price

    m = GBM(100, 0.05, 0.2)
    ref = bs_price(100, 110, 1, 0.05, 0.2)
    rs = [mc_price(m, European(110), 1, 1, 2**14, "sobol", seed=s) for s in range(400)]
    assert rs[0].dof == 31
    lo, hi = rs[0].ci()
    assert hi - lo == pytest.approx(2 * 2.0395 * rs[0].std_error, rel=1e-3)  # t(31) quantile, not 1.96
    cover = np.mean([r.ci()[0] <= ref <= r.ci()[1] for r in rs])
    assert cover > 0.91
