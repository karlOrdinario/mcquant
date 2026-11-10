"""Every Monte Carlo price is checked against an independent closed form."""

import numpy as np
import pytest

from mcquant.analytic import (
    barrier_price,
    bs_digital_price,
    bs_price,
    geometric_asian_price,
    heston_price,
    merton_price,
)
from mcquant.models import GBM, Heston, Merton
from mcquant.payoffs import AsianArithmetic, AsianGeometric, Barrier, Digital, European
from mcquant.pricer import geometric_asian_control, mc_price, stock_control

GBM_M = GBM(100, 0.05, 0.2, q=0.01)
N = 2**16


@pytest.mark.parametrize("method", ["pseudo", "antithetic", "sobol"])
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_european_vs_black_scholes(method, kind, strike):
    res = mc_price(GBM_M, European(strike, kind), 1.0, 1, N, method, seed=7)
    assert res.within(bs_price(100, strike, 1.0, 0.05, 0.2, 0.01, kind), n_se=4)


def test_digital_vs_closed_form():
    res = mc_price(GBM_M, Digital(105), 1.0, 1, N, "sobol", seed=3)
    assert res.within(bs_digital_price(100, 105, 1.0, 0.05, 0.2, 0.01), n_se=4)


@pytest.mark.parametrize("method", ["pseudo", "antithetic", "sobol"])
def test_confidence_interval_coverage(method):
    """The reported standard error must be honest: ~95% of 95% CIs should
    contain the true price. This catches e.g. treating antithetic pairs as
    independent, which inflates the SE and over-covers."""
    truth = bs_price(100, 100, 1.0, 0.05, 0.2, 0.01)
    hits = [
        mc_price(GBM_M, European(100), 1.0, 1, 2048, method, seed=s).within(truth, n_se=1.959964) for s in range(300)
    ]
    assert 0.91 <= np.mean(hits) <= 0.985


def test_variance_reduction_ranking():
    se = {m: mc_price(GBM_M, European(100), 1.0, 1, N, m, seed=1).std_error for m in ("pseudo", "antithetic", "sobol")}
    cv = mc_price(GBM_M, European(100), 1.0, 1, N, "pseudo", seed=1, control=stock_control(GBM_M)).std_error
    assert se["antithetic"] < se["pseudo"]
    assert cv < se["pseudo"] / 2
    assert se["sobol"] < se["pseudo"] / 20


def test_geometric_asian_vs_closed_form():
    res = mc_price(GBM_M, AsianGeometric(100), 1.0, 12, N, "sobol", seed=5)
    assert res.within(geometric_asian_price(100, 100, 1.0, 0.05, 0.2, 12, 0.01), n_se=4)


def test_arithmetic_asian_control_variate():
    payoff = AsianArithmetic(100)
    plain = mc_price(GBM_M, payoff, 1.0, 52, N, "pseudo", seed=2)
    cv = mc_price(GBM_M, payoff, 1.0, 52, N, "pseudo", seed=2, control=geometric_asian_control(GBM_M, 100))
    assert cv.std_error < plain.std_error / 20  # > 400x variance reduction
    assert cv.control_beta == pytest.approx(1.0, abs=0.1)
    assert abs(cv.price - plain.price) < 4 * plain.std_error
    # Arithmetic >= geometric average (AM-GM), so the call is worth more
    assert cv.price > geometric_asian_price(100, 100, 1.0, 0.05, 0.2, 52, 0.01)


def test_geometric_control_requires_gbm():
    with pytest.raises(TypeError):
        geometric_asian_control(Merton(100, 0.05, 0.2, 1, 0, 0.1), 100)


@pytest.mark.parametrize(
    "btype,kind,K,H",
    [
        ("up-and-out", "call", 100, 125),
        ("down-and-out", "call", 100, 85),
        ("up-and-in", "put", 100, 115),
        ("down-and-in", "put", 105, 90),
        ("down-and-out", "put", 110, 85),
        ("up-and-out", "put", 95, 110),
    ],
)
def test_barrier_vs_bgk(btype, kind, K, H):
    """Daily monitored barrier vs continuous formula + Broadie-Glasserman-Kou shift.
    BGK itself is an approximation (error ~ 1e-3 relative), hence the small abs_tol."""
    m = GBM(100, 0.05, 0.25)
    res = mc_price(m, Barrier(K, H, btype, kind), 1.0, 252, 2**17, "pseudo", seed=8)
    ref = barrier_price(100, K, H, 1.0, 0.05, 0.25, btype, kind, n_monitor=252)
    assert res.within(ref, n_se=4, abs_tol=0.01)


def test_barrier_in_out_parity_pathwise():
    paths_out = mc_price(GBM_M, Barrier(100, 120, "up-and-out"), 1, 50, N, "pseudo", seed=4)
    paths_in = mc_price(GBM_M, Barrier(100, 120, "up-and-in"), 1, 50, N, "pseudo", seed=4)
    vanilla = mc_price(GBM_M, European(100), 1, 50, N, "pseudo", seed=4)
    assert paths_out.price + paths_in.price == pytest.approx(vanilla.price, abs=1e-10)


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_heston_vs_semi_analytic(K):
    """Full-truncation Euler has O(dt) bias; with 200 steps/yr it is < 1 cent."""
    m = Heston(100, 0.03, 0.04, 1.5, 0.04, 0.5, -0.7)
    res = mc_price(m, European(K), 1.0, 200, 2**16, "antithetic", seed=13, control=stock_control(m))
    ref = heston_price(100, K, 1.0, 0.03, 0.04, 1.5, 0.04, 0.5, -0.7)
    assert res.within(ref, n_se=4, abs_tol=0.02)


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_merton_vs_series(K):
    m = Merton(100, 0.05, 0.2, 0.75, -0.1, 0.15)
    res = mc_price(m, European(K), 1.0, 1, 2**17, "antithetic", seed=17, control=stock_control(m))
    assert res.within(merton_price(100, K, 1.0, 0.05, 0.2, 0.75, -0.1, 0.15), n_se=4)


def test_result_helpers():
    res = mc_price(GBM_M, European(100), 1.0, 1, 4096, seed=1)
    lo, hi = res.ci(0.95)
    assert lo < res.price < hi
    assert "95% CI" in str(res)
    assert res.elapsed > 0
