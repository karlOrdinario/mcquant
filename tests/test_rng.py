import numpy as np
import pytest

from mcquant.rng import brownian_increments, effective_samples, mean_and_se, standard_normals


@pytest.mark.parametrize("method", ["pseudo", "antithetic", "sobol"])
def test_shape_moments_and_reproducibility(method):
    z = standard_normals(2**14, 8, method, seed=11)
    assert z.shape == (2**14, 8)
    assert np.all(np.isfinite(z))
    assert abs(z.mean()) < 0.02
    assert z.std() == pytest.approx(1.0, abs=0.02)
    np.testing.assert_array_equal(z, standard_normals(2**14, 8, method, seed=11))
    assert not np.array_equal(z, standard_normals(2**14, 8, method, seed=12))


def test_antithetic_pairs_mirror():
    z = standard_normals(10, 3, "antithetic", seed=1)
    np.testing.assert_array_equal(z[:5], -z[5:])
    assert effective_samples(z[:, 0], "antithetic").tolist() == pytest.approx([0.0] * 5)


def test_sobol_is_more_uniform_than_pseudo():
    # Mean of a smooth function converges much faster with QMC.
    f = lambda z: np.exp(0.3 * z.sum(axis=1))  # noqa: E731
    exact = np.exp(0.5 * 0.09 * 4)
    err_qmc = abs(f(standard_normals(2**14, 4, "sobol", seed=3)).mean() - exact)
    err_mc = abs(f(standard_normals(2**14, 4, "pseudo", seed=3)).mean() - exact)
    assert err_qmc < err_mc / 5


def test_input_validation():
    with pytest.raises(ValueError):
        standard_normals(11, 2, "antithetic")
    with pytest.raises(ValueError):
        standard_normals(100, 2, "sobol", n_batches=16)
    with pytest.raises(ValueError):
        standard_normals(16, 2, "halton")
    with pytest.raises(ValueError):
        standard_normals(0, 2)


def test_mean_and_se_matches_textbook_formula():
    x = np.random.default_rng(0).normal(3.0, 2.0, 10_000)
    m, se = mean_and_se(x, "pseudo")
    assert m == pytest.approx(x.mean())
    assert se == pytest.approx(x.std(ddof=1) / 100)


@pytest.mark.parametrize("n_steps", [1, 2, 5, 12, 64])
def test_brownian_bridge_has_brownian_covariance(n_steps):
    times = np.linspace(0, 2.0, n_steps + 1)
    z = standard_normals(200_000, n_steps, "pseudo", seed=5)
    w = np.cumsum(brownian_increments(z, times, bridge=True), axis=1)
    cov = np.cov(w, rowvar=False).reshape(n_steps, n_steps)
    expected = np.minimum.outer(times[1:], times[1:])
    np.testing.assert_allclose(cov, expected, atol=0.03)


def test_bridge_terminal_value_uses_first_dimension():
    times = np.linspace(0, 1, 9)
    z = np.zeros((1, 8))
    z[0, 0] = 1.0
    w = np.cumsum(brownian_increments(z, times, bridge=True), axis=1)
    assert w[0, -1] == pytest.approx(1.0)  # W_T = sqrt(T) * z_0
    np.testing.assert_allclose(w[0], times[1:])  # all other points on the straight line
