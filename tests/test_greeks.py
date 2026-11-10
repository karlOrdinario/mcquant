import pytest

from mcquant.analytic import bs_digital_delta, bs_greeks
from mcquant.greeks import fd_greeks, likelihood_ratio_delta, likelihood_ratio_gamma, pathwise_delta, pathwise_vega
from mcquant.models import GBM, Heston
from mcquant.payoffs import AsianArithmetic, Barrier, Digital, European

M = GBM(100, 0.05, 0.2, q=0.01)
N = 2**17


@pytest.mark.parametrize("kind", ["call", "put"])
def test_pathwise_delta_and_vega(kind):
    g = bs_greeks(100, 100, 1, 0.05, 0.2, 0.01, kind)
    d = pathwise_delta(M, European(100, kind), 1, n_paths=N, method="antithetic", seed=1)
    v = pathwise_vega(M, 100, 1, kind, N, "antithetic", seed=1)
    assert abs(d.value - g.delta) < 4 * d.std_error
    assert abs(v.value - g.vega) < 4 * v.std_error


def test_likelihood_ratio():
    g = bs_greeks(100, 100, 1, 0.05, 0.2, 0.01)
    d = likelihood_ratio_delta(M, European(100), 1, N, seed=2)
    gm = likelihood_ratio_gamma(M, European(100), 1, N, seed=2)
    assert abs(d.value - g.delta) < 4 * d.std_error
    assert abs(gm.value - g.gamma) < 4 * gm.std_error


def test_lr_handles_digital_where_pathwise_cannot():
    exact = bs_digital_delta(100, 100, 1, 0.05, 0.2, 0.01)
    d = likelihood_ratio_delta(M, Digital(100), 1, N, seed=3)
    assert abs(d.value - exact) < 4 * d.std_error
    with pytest.raises(ValueError):
        pathwise_delta(M, Digital(100), 1)
    with pytest.raises(ValueError):
        pathwise_delta(M, Barrier(100, 120), 1, n_steps=10)


def test_pathwise_has_lower_variance_than_lr():
    pw = pathwise_delta(M, European(100), 1, n_paths=N, seed=4)
    lr = likelihood_ratio_delta(M, European(100), 1, N, seed=4)
    assert pw.std_error < lr.std_error


def test_fd_with_common_random_numbers():
    g = bs_greeks(100, 100, 1, 0.05, 0.2, 0.01)
    fd = fd_greeks(M, European(100), 1, n_paths=N, method="antithetic", seed=5)
    assert abs(fd["delta"].value - g.delta) < 4 * fd["delta"].std_error + 1e-3
    assert abs(fd["gamma"].value - g.gamma) < 4 * fd["gamma"].std_error + 1e-3
    assert abs(fd["vega"].value - g.vega) < 4 * fd["vega"].std_error + 0.05


def test_fd_asian_matches_pathwise():
    a = AsianArithmetic(100)
    pw = pathwise_delta(M, a, 1, n_steps=12, n_paths=N, seed=6)
    fd = fd_greeks(M, a, 1, n_steps=12, n_paths=N, seed=6, vol_param=None)
    assert fd["delta"].value == pytest.approx(pw.value, abs=3e-3)
    assert "vega" not in fd


def test_fd_heston_v0_sensitivity_positive():
    h = Heston(100, 0.03, 0.04, 1.5, 0.04, 0.5, -0.7)
    out = fd_greeks(h, European(100), 1, n_steps=50, n_paths=2**15, seed=7, vol_param="v0", vol_bump=0.005)
    assert out["vega"].value > 0
    assert 0.4 < out["delta"].value < 0.8


def test_lr_type_checks():
    with pytest.raises(TypeError):
        likelihood_ratio_delta(M, AsianArithmetic(100), 1)
    with pytest.raises(TypeError):
        pathwise_vega(Heston(100, 0.03, 0.04, 1.5, 0.04, 0.5, -0.7), 100, 1)
