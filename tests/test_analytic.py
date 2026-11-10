"""Closed-form references must themselves be right before we trust them as answer keys."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mcquant.analytic import (
    barrier_price,
    binomial_price,
    bs_delta,
    bs_digital_delta,
    bs_digital_price,
    bs_greeks,
    bs_price,
    geometric_asian_price,
    heston_price,
    implied_vol,
    merton_price,
)

S, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.2


class TestBlackScholes:
    def test_textbook_values(self):
        # Hull, Options Futures & Other Derivatives: S=K=100, T=1, r=5%, sigma=20%
        assert bs_price(S, K, T, R, SIG) == pytest.approx(10.4506, abs=1e-4)
        assert bs_price(S, K, T, R, SIG, kind="put") == pytest.approx(5.5735, abs=1e-4)

    @pytest.mark.parametrize("q", [0.0, 0.03])
    @pytest.mark.parametrize("k", [80.0, 100.0, 125.0])
    def test_put_call_parity(self, k, q):
        c = bs_price(S, k, T, R, SIG, q)
        p = bs_price(S, k, T, R, SIG, q, kind="put")
        assert c - p == pytest.approx(S * math.exp(-q * T) - k * math.exp(-R * T), abs=1e-10)

    def test_expired_returns_intrinsic(self):
        assert bs_price(110, 100, 0.0, R, SIG) == 10.0
        assert bs_price(90, 100, 0.0, R, SIG, kind="put") == 10.0

    def test_vectorised(self):
        s = np.array([90.0, 100.0, 110.0])
        out = bs_price(s, K, T, R, SIG)
        assert out.shape == (3,)
        assert out[1] == pytest.approx(bs_price(100.0, K, T, R, SIG))

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_greeks_match_finite_differences(self, kind):
        g = bs_greeks(S, K, T, R, SIG, 0.02, kind)
        f = lambda **kw: bs_price(**{"S": S, "K": K, "T": T, "r": R, "sigma": SIG, "q": 0.02, "kind": kind, **kw})  # noqa: E731
        h = 1e-3
        assert g.delta == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-6)
        assert g.gamma == pytest.approx((f(S=S + h) - 2 * f() + f(S=S - h)) / h**2, rel=1e-4)
        assert g.vega == pytest.approx((f(sigma=SIG + h) - f(sigma=SIG - h)) / (2 * h), rel=1e-6)
        assert g.theta == pytest.approx(-(f(T=T + h) - f(T=T - h)) / (2 * h), rel=1e-5)
        assert g.rho == pytest.approx((f(r=R + h) - f(r=R - h)) / (2 * h), rel=1e-5)
        assert bs_delta(S, K, T, R, SIG, 0.02, kind) == pytest.approx(g.delta)

    def test_digital(self):
        h = 1e-4
        fd = (bs_digital_price(S + h, K, T, R, SIG) - bs_digital_price(S - h, K, T, R, SIG)) / (2 * h)
        assert bs_digital_delta(S, K, T, R, SIG) == pytest.approx(fd, rel=1e-6)
        assert bs_digital_price(S, K, T, R, SIG) + bs_digital_price(S, K, T, R, SIG, kind="put") == pytest.approx(
            math.exp(-R * T)
        )

    def test_implied_vol_roundtrip(self):
        for vol in (0.05, 0.2, 0.8):
            assert implied_vol(bs_price(S, 110, T, R, vol), S, 110, T, R) == pytest.approx(vol, abs=1e-8)

    def test_bad_kind(self):
        with pytest.raises(ValueError):
            bs_price(S, K, T, R, SIG, kind="straddle")

    @settings(max_examples=200, deadline=None)
    @given(
        s=st.floats(1, 500),
        k=st.floats(1, 500),
        t=st.floats(0.01, 5),
        r=st.floats(0, 0.15),
        sig=st.floats(0.01, 1.5),
    )
    def test_no_arbitrage_bounds(self, s, k, t, r, sig):
        """Property-based: a call is worth at least its discounted intrinsic and at most the stock."""
        c = bs_price(s, k, t, r, sig)
        assert max(s - k * math.exp(-r * t), 0.0) - 1e-9 <= c <= s + 1e-9
        p = bs_price(s, k, t, r, sig, kind="put")
        assert max(k * math.exp(-r * t) - s, 0.0) - 1e-9 <= p <= k * math.exp(-r * t) + 1e-9


class TestGeometricAsian:
    def test_one_fixing_is_european(self):
        assert geometric_asian_price(S, K, T, R, SIG, 1) == pytest.approx(bs_price(S, K, T, R, SIG), abs=1e-12)

    def test_cheaper_than_european(self):
        assert geometric_asian_price(S, K, T, R, SIG, 52) < bs_price(S, K, T, R, SIG)

    def test_put_call_parity(self):
        n = 12
        c, p = geometric_asian_price(S, K, T, R, SIG, n), geometric_asian_price(S, K, T, R, SIG, n, kind="put")
        mu = math.log(S) + (R - 0.5 * SIG**2) * T * (n + 1) / (2 * n)
        var = SIG**2 * T * (n + 1) * (2 * n + 1) / (6 * n * n)
        assert c - p == pytest.approx(math.exp(-R * T) * (math.exp(mu + 0.5 * var) - K), abs=1e-10)


class TestBarrier:
    @pytest.mark.parametrize("kind", ["call", "put"])
    @pytest.mark.parametrize("h,k", [(120, 100), (130, 90), (105, 110), (80, 100), (95, 90), (70, 80)])
    def test_in_out_parity(self, kind, h, k):
        side = "up" if h > S else "down"
        out = barrier_price(S, k, h, T, R, 0.25, f"{side}-and-out", kind, q=0.01)
        inn = barrier_price(S, k, h, T, R, 0.25, f"{side}-and-in", kind, q=0.01)
        assert out >= -1e-12 and inn >= -1e-12
        assert out + inn == pytest.approx(bs_price(S, k, T, R, 0.25, 0.01, kind), abs=1e-10)

    def test_far_barrier_is_vanilla(self):
        assert barrier_price(S, K, 1e6, T, R, SIG, "up-and-out") == pytest.approx(bs_price(S, K, T, R, SIG), abs=1e-8)
        assert barrier_price(S, K, 1e-3, T, R, SIG, "down-and-out", "put") == pytest.approx(
            bs_price(S, K, T, R, SIG, kind="put"), abs=1e-8
        )

    def test_knocked_at_inception(self):
        assert barrier_price(S, K, 95, T, R, SIG, "up-and-out") == 0.0
        assert barrier_price(S, K, 95, T, R, SIG, "up-and-in") == pytest.approx(bs_price(S, K, T, R, SIG))

    def test_bgk_correction_direction(self):
        # Discrete monitoring knocks out less often -> worth more than continuous.
        cont = barrier_price(S, K, 120, T, R, SIG, "up-and-out")
        disc = barrier_price(S, K, 120, T, R, SIG, "up-and-out", n_monitor=12)
        assert disc > cont


class TestHeston:
    PARAMS = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)

    def test_reduces_to_black_scholes(self):
        # Almost no vol-of-vol and v0 = theta  =>  constant vol sqrt(theta)
        for k in (80, 100, 120):
            h = heston_price(S, k, T, R, 0.04, 2.0, 0.04, 1e-3, 0.0)
            assert h == pytest.approx(bs_price(S, k, T, R, 0.2), abs=2e-3)

    def test_put_call_parity(self):
        c = heston_price(S, 105, T, R, q=0.02, **self.PARAMS)
        p = heston_price(S, 105, T, R, q=0.02, kind="put", **self.PARAMS)
        assert c - p == pytest.approx(S * math.exp(-0.02) - 105 * math.exp(-R), abs=1e-8)

    def test_negative_rho_creates_skew(self):
        """rho < 0 => OTM puts richer than OTM calls in implied-vol terms."""
        iv_put = implied_vol(heston_price(S, 80, T, R, kind="put", **self.PARAMS), S, 80, T, R, kind="put")
        iv_call = implied_vol(heston_price(S, 120, T, R, **self.PARAMS), S, 120, T, R)
        assert iv_put > iv_call + 0.02

    def test_within_bounds(self):
        for k in (50, 100, 200):
            c = heston_price(S, k, T, R, **self.PARAMS)
            assert max(S - k * math.exp(-R), 0) - 1e-8 <= c <= S


class TestMerton:
    def test_no_jumps_is_black_scholes(self):
        assert merton_price(S, K, T, R, SIG, 0.0, -0.1, 0.15) == pytest.approx(bs_price(S, K, T, R, SIG), abs=1e-12)

    def test_zero_size_jumps_are_black_scholes(self):
        # mu_j = 0 and sigma_j ~ 0: every jump multiplies S by exp(0) = 1
        p = merton_price(S, K, T, R, SIG, 1.0, 0.0, 1e-8)
        assert p == pytest.approx(bs_price(S, K, T, R, SIG), abs=1e-6)

    def test_put_call_parity(self):
        c = merton_price(S, K, T, R, SIG, 0.75, -0.1, 0.15, q=0.01)
        p = merton_price(S, K, T, R, SIG, 0.75, -0.1, 0.15, q=0.01, kind="put")
        assert c - p == pytest.approx(S * math.exp(-0.01) - K * math.exp(-R), abs=1e-9)

    def test_jumps_add_value(self):
        assert merton_price(S, K, T, R, SIG, 0.75, -0.1, 0.15) > bs_price(S, K, T, R, SIG)


class TestBinomial:
    def test_european_converges_to_bs(self):
        assert binomial_price(S, K, T, R, SIG, kind="call", american=False, n_steps=4000) == pytest.approx(
            bs_price(S, K, T, R, SIG), abs=2e-3
        )

    def test_american_call_no_dividend_equals_european(self):
        a = binomial_price(S, K, T, R, SIG, kind="call", american=True)
        e = binomial_price(S, K, T, R, SIG, kind="call", american=False)
        assert a == pytest.approx(e, abs=1e-10)

    def test_american_put_premium(self):
        a = binomial_price(36, 40, 1, 0.06, 0.2)
        assert a == pytest.approx(4.4867, abs=1e-3)  # well-known reference value
        assert a > bs_price(36, 40, 1, 0.06, 0.2, kind="put")

    def test_bermudan_between_european_and_american(self):
        e = binomial_price(36, 40, 1, 0.06, 0.2, american=False, n_steps=2500)
        b = binomial_price(36, 40, 1, 0.06, 0.2, n_steps=2500, n_exercise=50)
        a = binomial_price(36, 40, 1, 0.06, 0.2, n_steps=2500)
        assert e < b < a

    def test_deep_itm_exercises_immediately(self):
        assert binomial_price(10, 40, 1, 0.06, 0.2) == pytest.approx(30.0, abs=1e-12)

    def test_bad_exercise_grid(self):
        with pytest.raises(ValueError):
            binomial_price(36, 40, 1, 0.06, 0.2, n_steps=1000, n_exercise=33)
