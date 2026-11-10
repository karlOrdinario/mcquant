import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from mcquant.risk import (
    Portfolio,
    Position,
    basel_zone,
    christoffersen_independence,
    cov_from_vols_corr,
    delta_normal_var,
    historical_var,
    kupiec_pof,
    monte_carlo_var,
    rolling_var_backtest,
    simulate_log_returns,
    stress_test,
    var_cvar,
)

CORR = np.array([[1.0, 0.6], [0.6, 1.0]])
COV = cov_from_vols_corr([0.2, 0.3], CORR)
MU = np.array([0.08, 0.10])


def stock_book():
    return Portfolio(np.array([100.0, 50.0]), [Position(0, 1_000), Position(1, 2_000)])


def test_var_cvar_on_known_normal():
    pnl = np.random.default_rng(0).normal(0, 1, 1_000_000)
    var, cvar = var_cvar(pnl, 0.99)
    assert var == pytest.approx(norm.ppf(0.99), abs=0.01)
    assert cvar == pytest.approx(norm.pdf(norm.ppf(0.99)) / 0.01, abs=0.02)
    assert cvar > var


def test_simulated_returns_have_target_covariance():
    for dist in ("normal", "t"):
        r = simulate_log_returns(MU, COV, 252, 400_000, dist, df=5, seed=1)
        np.testing.assert_allclose(np.cov(r, rowvar=False), COV, rtol=0.03, atol=1e-3)


def test_t_scenarios_have_fat_tails():
    r = simulate_log_returns(MU, COV, 1, 400_000, "t", df=4.5, seed=2)[:, 0]
    k = ((r - r.mean()) ** 4).mean() / r.var() ** 2 - 3
    assert k > 2  # normal would be 0


def test_monte_carlo_matches_delta_normal_for_linear_book():
    """Stocks only + short horizon: the P&L is nearly linear so MC ~ parametric."""
    pf = stock_book()
    mc = monte_carlo_var(pf, MU, COV, 0.99, 1, 400_000, seed=3)
    dn = delta_normal_var(pf, MU, COV, 0.99, 1)
    assert mc.var == pytest.approx(dn.var, rel=0.02)
    assert mc.cvar == pytest.approx(dn.cvar, rel=0.03)
    assert mc.var_se < 0.01 * mc.var


def test_fat_tails_raise_tail_risk():
    pf = stock_book()
    normal = monte_carlo_var(pf, MU, COV, 0.99, 1, 400_000, seed=4)
    fat = monte_carlo_var(pf, MU, COV, 0.99, 1, 400_000, dist="t", df=4, seed=4)
    assert fat.cvar > normal.cvar * 1.1


def test_option_book_nonlinearity():
    """A short-straddle book: delta-normal sees ~zero delta, MC sees the gamma risk."""
    s = np.array([100.0, 50.0])
    pf = Portfolio(s, [Position(0, -100, "call", 100, 0.25, 0.2), Position(0, -100, "put", 100, 0.25, 0.2)], r=0.02)
    dn = delta_normal_var(pf, MU, COV, 0.99, 10)
    mc = monte_carlo_var(pf, MU, COV, 0.99, 10, 200_000, seed=5)
    assert mc.var > 3 * dn.var


def test_protective_put_caps_loss():
    s = np.array([100.0, 50.0])
    pf = Portfolio(s, [Position(0, 100), Position(0, 100, "put", 95, 0.5, 0.2)], r=0.02)
    mc = monte_carlo_var(pf, MU, COV, 0.99, 20, 200_000, seed=6)
    max_loss = pf.current_value - 100 * 95  # floor at strike (+ residual time value)
    assert mc.cvar <= max_loss + 1e-6


def test_historical_var():
    rng = np.random.default_rng(7)
    rets = rng.multivariate_normal(np.zeros(2), COV / 252, size=5_000)
    pf = stock_book()
    hs = historical_var(pf, rets, 0.99, 1)
    dn = delta_normal_var(pf, np.zeros(2), COV, 0.99, 1)
    assert hs.var == pytest.approx(dn.var, rel=0.1)
    hs10 = historical_var(pf, rets, 0.99, 10)
    assert hs10.var == pytest.approx(hs.var * math.sqrt(10), rel=0.15)


def test_stress_test():
    pf = stock_book()
    table = stress_test(pf, {"down 10%": {"spot_shock": -0.1}, "split": {"spot_shock": [0.1, -0.2]}})
    assert table.loc["down 10%", "P&L"] == pytest.approx(-0.1 * pf.current_value)
    assert table.loc["split", "P&L"] == pytest.approx(0.1 * 100_000 - 0.2 * 100_000)
    assert len(stress_test(pf)) == 5


class TestBacktests:
    def test_kupiec(self):
        assert kupiec_pof(1000, 10, 0.99)["p_pof"] == pytest.approx(1.0)
        assert kupiec_pof(1000, 30, 0.99)["p_pof"] < 1e-5
        assert kupiec_pof(1000, 0, 0.99)["p_pof"] < 0.01  # too FEW exceptions also rejects
        # hand-computed LR for x=20, n=1000, p=0.01
        x, n, p = 20, 1000, 0.01
        lr = -2 * ((n - x) * math.log(1 - p) + x * math.log(p) - (n - x) * math.log(0.98) - x * math.log(0.02))
        assert kupiec_pof(n, x, 0.99)["LR_pof"] == pytest.approx(lr)

    def test_christoffersen_detects_clustering(self):
        spread = np.zeros(1000, bool)
        spread[::100] = True
        clustered = np.zeros(1000, bool)
        clustered[500:510] = True
        assert christoffersen_independence(spread)["p_ind"] > 0.5
        assert christoffersen_independence(clustered)["p_ind"] < 1e-4

    def test_basel_traffic_light(self):
        assert [basel_zone(250, x) for x in (0, 4, 5, 9, 10)] == ["green", "green", "yellow", "yellow", "red"]

    def test_correct_model_passes_and_wrong_model_fails(self):
        rng = np.random.default_rng(11)
        iid = pd.Series(rng.normal(0, 0.01, 3000))
        res = rolling_var_backtest(iid, 0.99, 250, methods=("normal", "historical"))
        assert res.summary.loc["normal", "verdict"] == "pass"
        assert res.summary.loc["normal", "rate %"] == pytest.approx(1.0, abs=0.5)
        # Regime shift: calm then turbulent -> static normal VaR fails, EWMA adapts
        shifted = pd.Series(np.concatenate([rng.normal(0, 0.005, 1500), rng.normal(0, 0.02, 1500)]))
        res2 = rolling_var_backtest(shifted, 0.99, 250, methods=("normal", "ewma"))
        assert res2.summary.loc["normal", "exceptions"] > res2.summary.loc["ewma", "exceptions"]

    def test_backtest_has_no_lookahead(self):
        """The forecast for day t must not change if day t's return changes."""
        r = np.random.default_rng(12).normal(0, 0.01, 600)
        a = rolling_var_backtest(r, 0.99, 250).forecasts
        r2 = r.copy()
        r2[400] = -0.5  # huge shock on day 400
        b = rolling_var_backtest(r2, 0.99, 250).forecasts
        pd.testing.assert_frame_equal(
            a.iloc[: 400 - 250].drop(columns="return"), b.iloc[: 400 - 250].drop(columns="return")
        )
        assert not a.iloc[400 - 250 + 1 :].drop(columns="return").equals(b.iloc[400 - 250 + 1 :].drop(columns="return"))


@pytest.mark.parametrize(
    "bad",
    [
        lambda: Position(0, 1, "call"),
        lambda: Position(0, 1, "future"),
        lambda: Portfolio(np.array([100.0]), [Position(3, 1)]),
        lambda: simulate_log_returns(MU, np.array([[1.0, 2.0], [2.0, 1.0]]), 1, 10),
        lambda: simulate_log_returns(MU, COV, 1, 10, "t", df=2),
        lambda: var_cvar(np.zeros(10), 0.3),
    ],
)
def test_validation(bad):
    with pytest.raises(ValueError):
        bad()
