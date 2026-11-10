import math

import numpy as np
import pytest

from mcquant.models import GBM, Heston, Merton, warn_if_feller_violated
from mcquant.rng import mean_and_se

MODELS = {
    "gbm": GBM(100, 0.05, 0.2, q=0.02),
    "heston": Heston(100, 0.05, 0.04, 1.5, 0.04, 0.5, -0.7, q=0.02),
    "merton": Merton(100, 0.05, 0.2, 0.75, -0.1, 0.15, q=0.02),
}


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("method", ["pseudo", "antithetic", "sobol"])
def test_martingale(name, method):
    """Risk-neutral drift: E[S_T] = S_0 e^{(r-q)T} for every model and sampler."""
    m = MODELS[name]
    p = m.simulate(1.0, 50, 2**15, method, seed=21)
    est, se = mean_and_se(p.spot[:, -1], method)
    assert abs(est - 100 * math.exp(0.03)) < 4 * se + 1e-3
    assert p.spot.shape == (2**15, 51)
    assert np.all(p.spot[:, 0] == 100)
    assert np.all(p.spot > 0)


def test_gbm_log_variance():
    p = GBM(100, 0.05, 0.3).simulate(2.0, 1, 200_000, seed=1)
    assert np.log(p.spot[:, -1]).var() == pytest.approx(0.09 * 2.0, rel=0.01)


def test_gbm_exact_scheme_step_invariant():
    """The exact GBM scheme has no time-step bias: 1 step and 100 steps agree."""
    a = GBM(100, 0.05, 0.2).simulate(1, 1, 2**15, "sobol", seed=4).spot[:, -1]
    b = GBM(100, 0.05, 0.2).simulate(1, 100, 2**15, "sobol", seed=4).spot[:, -1]
    assert a.mean() == pytest.approx(b.mean(), rel=1e-3)
    assert np.quantile(a, 0.01) == pytest.approx(np.quantile(b, 0.01), rel=5e-3)


def test_heston_variance_nonnegative_and_mean_reverting():
    m = Heston(100, 0.0, 0.09, 3.0, 0.04, 0.6, -0.5)
    p = m.simulate(3.0, 300, 20_000, seed=2)
    assert np.all(p.variance >= 0)
    # E[v_t] = theta + (v0 - theta) e^{-kappa t}
    assert p.variance[:, -1].mean() == pytest.approx(0.04 + 0.05 * math.exp(-9.0), rel=0.03)


def test_heston_correlation():
    m = Heston(100, 0.0, 0.04, 2.0, 0.04, 0.3, -0.8)
    p = m.simulate(0.5, 100, 50_000, seed=3)
    dlog_s = np.diff(np.log(p.spot), axis=1).ravel()
    dv = np.diff(p.variance, axis=1).ravel()
    assert np.corrcoef(dlog_s, dv)[0, 1] == pytest.approx(-0.8, abs=0.05)


def test_feller_warning():
    with pytest.warns(UserWarning):
        warn_if_feller_violated(Heston(100, 0.0, 0.04, 0.5, 0.04, 1.0, -0.5))


def test_merton_jump_frequency():
    m = Merton(100, 0.0, 1e-4, 2.0, -0.2, 1e-4)
    p = m.simulate(1.0, 252, 20_000, seed=9)
    big_moves = (np.abs(np.diff(np.log(p.spot), axis=1)) > 0.1).sum(axis=1)
    assert big_moves.mean() == pytest.approx(2.0, rel=0.05)


@pytest.mark.parametrize(
    "bad",
    [
        lambda: GBM(-1, 0.05, 0.2),
        lambda: GBM(100, 0.05, 0.0),
        lambda: Heston(100, 0.0, 0.04, 1.0, 0.04, 0.3, -1.5),
        lambda: Merton(100, 0.0, 0.2, -1.0, 0.0, 0.1),
        lambda: GBM(100, 0.05, 0.2).simulate(0.0),
        lambda: GBM(100, 0.05, 0.2).simulate(1.0, 0),
    ],
)
def test_validation(bad):
    with pytest.raises(ValueError):
        bad()
