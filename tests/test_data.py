import numpy as np
import pandas as pd
import pytest

from mcquant.data import estimate_parameters, load_csv, log_returns, synthetic_market


def test_synthetic_market_shape_and_stats():
    px = synthetic_market(n_days=5_000, seed=1)
    assert px.shape == (5_001, 4)
    assert (px.iloc[0] == 100).all()
    assert isinstance(px.index, pd.DatetimeIndex)
    mu, cov = estimate_parameters(px)
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), [0.28, 0.24, 0.14, 0.32], rtol=0.15)
    corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    assert corr[0, 1] == pytest.approx(0.55, abs=0.08)


def test_synthetic_market_has_stylised_facts():
    r = log_returns(synthetic_market(n_days=5_000, seed=2))["TECH"].to_numpy()
    excess_kurt = ((r - r.mean()) ** 4).mean() / r.var() ** 2 - 3
    assert excess_kurt > 1  # fat tails
    sq = r**2
    assert np.corrcoef(sq[:-1], sq[1:])[0, 1] > 0.05  # volatility clustering


def test_reproducible():
    pd.testing.assert_frame_equal(synthetic_market(seed=3), synthetic_market(seed=3))


def test_csv_roundtrip(tmp_path):
    px = synthetic_market(n_days=50, seed=4)
    path = tmp_path / "prices.csv"
    px.rename_axis("Date").to_csv(path)
    back = load_csv(path)
    np.testing.assert_allclose(back.to_numpy(), px.to_numpy())


def test_validation():
    with pytest.raises(ValueError):
        synthetic_market(garch_alpha=0.2, garch_beta=0.85)
    with pytest.raises(ValueError):
        synthetic_market(tickers=("A",))
