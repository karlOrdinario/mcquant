import pytest

from mcquant.american import lsm_price
from mcquant.analytic import binomial_price, bs_price
from mcquant.models import GBM, Heston


@pytest.mark.parametrize("s0,sigma,T", [(36, 0.2, 1.0), (40, 0.2, 1.0), (44, 0.4, 2.0), (38, 0.3, 1.0)])
def test_lsm_matches_bermudan_tree(s0, sigma, T):
    """Longstaff & Schwartz (2001) Table 1 grid. Benchmark is a 2,500-step tree
    restricted to the SAME 50 x T exercise dates, so the comparison is exact
    apples-to-apples (no Bermudan-vs-American gap)."""
    n_ex = int(50 * T)
    ref = binomial_price(s0, 40, T, 0.06, sigma, n_steps=50 * n_ex, n_exercise=n_ex)
    res = lsm_price(GBM(s0, 0.06, sigma), 40, T, "put", n_ex, 100_000, seed=3)
    # out-of-sample LSM is a lower bound; regression error costs a few cents at most
    assert ref - 0.04 - 4 * res.std_error < res.price < ref + 4 * res.std_error


def test_in_sample_vs_out_of_sample():
    m = GBM(36, 0.06, 0.2)
    oos = lsm_price(m, 40, 1, n_paths=50_000, seed=1, out_of_sample=True)
    ins = lsm_price(m, 40, 1, n_paths=50_000, seed=1, out_of_sample=False)
    assert abs(oos.price - ins.price) < 0.05


def test_american_call_without_dividends_is_european():
    m = GBM(100, 0.05, 0.2)
    res = lsm_price(m, 100, 1, "call", 50, 2**16, method="antithetic", seed=2)
    assert res.within(bs_price(100, 100, 1, 0.05, 0.2), n_se=4, abs_tol=0.02)
    assert res.early_exercise_premium == pytest.approx(0.0, abs=0.05)


def test_early_exercise_premium_positive_for_put():
    res = lsm_price(GBM(36, 0.06, 0.2), 40, 1, seed=5, n_paths=50_000)
    assert res.early_exercise_premium > 0.5
    # put boundary sits below the strike
    b = res.boundary_spots[1:-1]
    assert (b[~__import__("numpy").isnan(b)] < 40).all()


def test_deep_itm_exercises_now():
    res = lsm_price(GBM(10, 0.06, 0.2), 40, 1, n_paths=20_000, seed=1)
    assert res.price == pytest.approx(30.0)


@pytest.mark.parametrize("method", ["antithetic", "sobol"])
def test_variance_reduced_lsm(method):
    ref = binomial_price(36, 40, 1, 0.06, 0.2, n_steps=2500, n_exercise=50)
    res = lsm_price(GBM(36, 0.06, 0.2), 40, 1, "put", 50, 2**16, method=method, seed=9)
    assert ref - 0.04 - 4 * res.std_error < res.price < ref + 4 * res.std_error


def test_works_under_heston():
    res = lsm_price(Heston(36, 0.06, 0.04, 2.0, 0.04, 0.3, -0.5), 40, 1, n_paths=20_000, seed=1)
    assert res.price > res.european_price > 0


def test_validation():
    with pytest.raises(ValueError):
        lsm_price(GBM(36, 0.06, 0.2), 40, 1, kind="straddle")
    with pytest.raises(ValueError):
        lsm_price(GBM(36, 0.06, 0.2), 40, 1, degree=0)
