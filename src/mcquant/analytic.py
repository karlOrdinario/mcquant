"""Closed-form and lattice reference prices.

These are the "answer keys" the Monte Carlo engine is validated against:

* Black-Scholes-Merton price and Greeks (vectorised)
* Discretely monitored geometric-average Asian option (Kemna & Vorst style)
* Single-barrier options (Reiner & Rubinstein / Hull) with the
  Broadie-Glasserman-Kou continuity correction for discrete monitoring
* Heston (1993) semi-closed form via the characteristic function,
  using the "little trap" formulation of Albrecher et al. (2007)
* Merton (1976) jump-diffusion series
* Cox-Ross-Rubinstein binomial tree (European and American)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import norm

from .payoffs import BARRIER_TYPES, BarrierType, Kind

ArrayLike = float | np.ndarray

# Broadie, Glasserman & Kou (1997): beta = -zeta(1/2) / sqrt(2 pi)
BGK_BETA = 0.5825971579390106


def _check_kind(kind: str) -> None:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def _check_inputs(S, K, sigma) -> None:
    # Fail loudly instead of silently returning NaN.
    if np.any(np.asarray(S) <= 0) or np.any(np.asarray(K) <= 0):
        raise ValueError("spot and strike must be positive")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("volatility must be positive")


# --------------------------------------------------------------------------- #
# Black-Scholes-Merton
# --------------------------------------------------------------------------- #
def _d1_d2(S, K, T, r, sigma, q):
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    return d1, d1 - sigma * sqrt_t


def bs_price(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: float, sigma: ArrayLike, q: float = 0.0, kind: Kind = "call"
) -> ArrayLike:
    """Black-Scholes-Merton price. Broadcasts over array inputs.

    At or after expiry (``T <= 0``) the intrinsic value is returned.
    """
    _check_kind(kind)
    S, K, T, sigma = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (S, K, T, sigma)))
    _check_inputs(S, K, sigma)
    live = T > 0
    t = np.where(live, T, 1.0)  # placeholder avoids divide-by-zero; masked out below
    d1, d2 = _d1_d2(S, K, t, r, sigma, q)
    if kind == "call":
        val = S * np.exp(-q * t) * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
        intrinsic = np.maximum(S - K, 0.0)
    else:
        val = K * np.exp(-r * t) * norm.cdf(-d2) - S * np.exp(-q * t) * norm.cdf(-d1)
        intrinsic = np.maximum(K - S, 0.0)
    out = np.where(live, val, intrinsic)
    return float(out) if out.ndim == 0 else out


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float  # per year, dV/dt (calendar time passing)
    rho: float


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, kind: Kind = "call") -> Greeks:
    _check_kind(kind)
    _check_inputs(S, K, sigma)
    if T <= 0:
        raise ValueError("Greeks need T > 0")
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf, sqrt_t = norm.pdf(d1), math.sqrt(T)
    dq, dr = math.exp(-q * T), math.exp(-r * T)
    gamma = dq * pdf / (S * sigma * sqrt_t)
    vega = S * dq * pdf * sqrt_t
    decay = -S * dq * pdf * sigma / (2 * sqrt_t)
    if kind == "call":
        delta = dq * norm.cdf(d1)
        theta = decay - r * K * dr * norm.cdf(d2) + q * S * dq * norm.cdf(d1)
        rho = K * T * dr * norm.cdf(d2)
    else:
        delta = dq * (norm.cdf(d1) - 1.0)
        theta = decay + r * K * dr * norm.cdf(-d2) - q * S * dq * norm.cdf(-d1)
        rho = -K * T * dr * norm.cdf(-d2)
    return Greeks(float(delta), float(gamma), float(vega), float(theta), float(rho))


def bs_delta(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: float, sigma: ArrayLike, q: float = 0.0, kind: Kind = "call"
) -> ArrayLike:
    """Vectorised Black-Scholes delta (used for delta-normal VaR)."""
    _check_kind(kind)
    S, K, T, sigma = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (S, K, T, sigma)))
    _check_inputs(S, K, sigma)
    live = T > 0
    t = np.where(live, T, 1.0)
    d1, _ = _d1_d2(S, K, t, r, sigma, q)
    if kind == "call":
        val, expired = np.exp(-q * t) * norm.cdf(d1), (S > K).astype(float)
    else:
        val, expired = np.exp(-q * t) * (norm.cdf(d1) - 1.0), -(S < K).astype(float)
    out = np.where(live, val, expired)
    return float(out) if out.ndim == 0 else out


def bs_digital_price(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, kind: Kind = "call"
) -> float:
    """Cash-or-nothing digital: e^{-rT} N(+/- d2)."""
    _check_kind(kind)
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    return float(math.exp(-r * T) * norm.cdf(d2 if kind == "call" else -d2))


def bs_digital_delta(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, kind: Kind = "call"
) -> float:
    _check_kind(kind)
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    sign = 1.0 if kind == "call" else -1.0
    return float(sign * math.exp(-r * T) * norm.pdf(d2) / (S * sigma * math.sqrt(T)))


def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float = 0.0, kind: Kind = "call") -> float:
    """Black-Scholes implied volatility by Brent's method."""
    from scipy.optimize import brentq

    lo, hi = 1e-6, 5.0
    f = lambda s: bs_price(S, K, T, r, s, q, kind) - price  # noqa: E731
    if f(lo) * f(hi) > 0:
        raise ValueError("price is outside the no-arbitrage bounds for implied vol")
    return float(brentq(f, lo, hi, xtol=1e-12))


# --------------------------------------------------------------------------- #
# Geometric Asian (discrete monitoring at t_i = i T / n, i = 1..n)
# --------------------------------------------------------------------------- #
def geometric_asian_price(
    S: float, K: float, T: float, r: float, sigma: float, n_monitor: int, q: float = 0.0, kind: Kind = "call"
) -> float:
    """ln(G) is normal, so the geometric Asian is a Black-Scholes-type formula.

    mean  = ln S + (r - q - sigma^2/2) * T (n+1) / (2n)
    var   = sigma^2 T (n+1)(2n+1) / (6 n^2)
    """
    _check_kind(kind)
    n = n_monitor
    mu = math.log(S) + (r - q - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    s = sigma * math.sqrt(T * (n + 1) * (2 * n + 1) / (6 * n * n))
    d1 = (mu - math.log(K) + s * s) / s
    d2 = d1 - s
    fwd = math.exp(mu + 0.5 * s * s)
    if kind == "call":
        return math.exp(-r * T) * (fwd * norm.cdf(d1) - K * norm.cdf(d2))
    return math.exp(-r * T) * (K * norm.cdf(-d2) - fwd * norm.cdf(-d1))


# --------------------------------------------------------------------------- #
# Barrier options (continuous monitoring, optional BGK discrete correction)
# --------------------------------------------------------------------------- #
def barrier_price(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    barrier_type: BarrierType,
    kind: Kind = "call",
    q: float = 0.0,
    n_monitor: int | None = None,
) -> float:
    """Single-barrier option price under Black-Scholes (no rebate).

    If ``n_monitor`` is given the barrier is shifted by
    ``exp(+/- 0.5826 sigma sqrt(T / n_monitor))`` (Broadie-Glasserman-Kou), which
    turns the continuous-monitoring formula into an accurate approximation
    for a barrier checked only ``n_monitor`` times.
    """
    _check_kind(kind)
    if barrier_type not in BARRIER_TYPES:
        raise ValueError(f"barrier_type must be one of {BARRIER_TYPES}")
    up = barrier_type.startswith("up")
    if n_monitor:
        shift = math.exp(BGK_BETA * sigma * math.sqrt(T / n_monitor))
        H = H * shift if up else H / shift

    vanilla = bs_price(S, K, T, r, sigma, q, kind)
    # Already knocked at inception
    if (up and S >= H) or (not up and S <= H):
        return vanilla if barrier_type.endswith("in") else 0.0

    sqt = sigma * math.sqrt(T)
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    y = math.log(H * H / (S * K)) / sqt + lam * sqt
    x1 = math.log(S / H) / sqt + lam * sqt
    y1 = math.log(H / S) / sqt + lam * sqt
    dq, dr = math.exp(-q * T), math.exp(-r * T)
    hs2l = (H / S) ** (2 * lam)
    hs2l2 = (H / S) ** (2 * lam - 2)
    N = norm.cdf

    if kind == "call" and not up:
        if H <= K:
            knock_in = S * dq * hs2l * N(y) - K * dr * hs2l2 * N(y - sqt)
            knock_out = vanilla - knock_in
        else:
            knock_out = S * N(x1) * dq - K * dr * N(x1 - sqt) - S * dq * hs2l * N(y1) + K * dr * hs2l2 * N(y1 - sqt)
            knock_in = vanilla - knock_out
    elif kind == "call" and up:
        if H <= K:
            knock_out, knock_in = 0.0, vanilla
        else:
            knock_in = (
                S * N(x1) * dq
                - K * dr * N(x1 - sqt)
                - S * dq * hs2l * (N(-y) - N(-y1))
                + K * dr * hs2l2 * (N(-y + sqt) - N(-y1 + sqt))
            )
            knock_out = vanilla - knock_in
    elif kind == "put" and up:
        if H >= K:
            knock_in = -S * dq * hs2l * N(-y) + K * dr * hs2l2 * N(-y + sqt)
            knock_out = vanilla - knock_in
        else:
            knock_out = (
                -S * N(-x1) * dq + K * dr * N(-x1 + sqt) + S * dq * hs2l * N(-y1) - K * dr * hs2l2 * N(-y1 + sqt)
            )
            knock_in = vanilla - knock_out
    else:  # down put
        if H >= K:
            knock_out, knock_in = 0.0, vanilla
        else:
            knock_in = (
                -S * N(-x1) * dq
                + K * dr * N(-x1 + sqt)
                + S * dq * hs2l * (N(y) - N(y1))
                - K * dr * hs2l2 * (N(y - sqt) - N(y1 - sqt))
            )
            knock_out = vanilla - knock_in

    return float(knock_out if barrier_type.endswith("out") else knock_in)


# --------------------------------------------------------------------------- #
# Heston semi-closed form
# --------------------------------------------------------------------------- #
def heston_charfn(
    u: complex | np.ndarray,
    S: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    q: float = 0.0,
):
    """Characteristic function of ln(S_T), "little trap" form (numerically stable)."""
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b * b + xi * xi * (iu + u * u))
    g = (b - d) / (b + d)
    edt = np.exp(-d * T)
    C = (r - q) * iu * T + kappa * theta / xi**2 * ((b - d) * T - 2.0 * np.log((1.0 - g * edt) / (1.0 - g)))
    D = (b - d) / xi**2 * (1.0 - edt) / (1.0 - g * edt)
    return np.exp(C + D * v0 + iu * math.log(S))


def heston_price(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    q: float = 0.0,
    kind: Kind = "call",
) -> float:
    """European option under Heston via Gil-Pelaez inversion.

    C = S e^{-qT} P1 - K e^{-rT} P2 with
    P2 = 1/2 + 1/pi int_0^inf Re[e^{-iu lnK} phi(u) / (iu)] du
    P1 = 1/2 + 1/pi int_0^inf Re[e^{-iu lnK} phi(u - i) / (iu phi(-i))] du
    """
    _check_kind(kind)
    args = (S, T, r, v0, kappa, theta, xi, rho, q)
    ln_k = math.log(K)
    fwd = S * math.exp((r - q) * T)  # = phi(-i)

    def p_integrand(u: float, j: int) -> float:
        shift = -1j if j == 1 else 0.0
        val = np.exp(-1j * u * ln_k) * heston_charfn(u + shift, *args) / (1j * u)
        return float(np.real(val / fwd if j == 1 else val))

    opts = dict(limit=500, epsabs=1e-12, epsrel=1e-10)
    p1 = 0.5 + quad(p_integrand, 1e-10, np.inf, args=(1,), **opts)[0] / math.pi
    p2 = 0.5 + quad(p_integrand, 1e-10, np.inf, args=(2,), **opts)[0] / math.pi
    call = S * math.exp(-q * T) * p1 - K * math.exp(-r * T) * p2
    if kind == "call":
        return float(call)
    return float(call - S * math.exp(-q * T) + K * math.exp(-r * T))


# --------------------------------------------------------------------------- #
# Merton jump-diffusion series
# --------------------------------------------------------------------------- #
def merton_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    q: float = 0.0,
    kind: Kind = "call",
    tol: float = 1e-14,
) -> float:
    """Conditioning on n jumps gives a Black-Scholes price; weight by Poisson(lam' T).

    The number of terms adapts to the expected jump count (mean + 12 std devs
    of the Poisson distribution) so the truncation error is always below ``tol``.
    """
    _check_kind(kind)
    k = math.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    lam_p = lam * (1.0 + k)
    mean_jumps = lam_p * T
    max_terms = int(mean_jumps + 12 * math.sqrt(mean_jumps) + 50)
    total = 0.0
    for n in range(max_terms):
        log_w = -lam_p * T + n * math.log(lam_p * T) - gammaln(n + 1) if lam_p > 0 else (0.0 if n == 0 else -np.inf)
        w = math.exp(log_w)
        sig_n = math.sqrt(sigma**2 + n * sigma_j**2 / T)
        r_n = r - lam * k + n * math.log(1.0 + k) / T
        # Using lam' = lam (1 + k) in the Poisson weight already absorbs the
        # exp((r_n - r) T) factor from discounting at r instead of r_n.
        total += w * bs_price(S, K, T, r_n, sig_n, q, kind)
        if n > lam_p * T and w < tol:
            break
    return float(total)


# --------------------------------------------------------------------------- #
# Cox-Ross-Rubinstein binomial tree
# --------------------------------------------------------------------------- #
def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: Kind = "put",
    american: bool = True,
    n_steps: int = 2_000,
    n_exercise: int | None = None,
) -> float:
    """CRR tree. Error is O(1/n); 2,000 steps is accurate to roughly 1e-3.

    ``n_exercise`` turns the American option into a *Bermudan* one that can
    only be exercised on ``n_exercise`` equally spaced dates - the exact
    contract that Longstaff-Schwartz with ``n_exercise`` time steps prices.
    ``n_steps`` must then be a multiple of ``n_exercise``.
    """
    _check_kind(kind)
    if n_exercise is not None:
        if n_steps % n_exercise:
            raise ValueError("n_steps must be a multiple of n_exercise")
        every = n_steps // n_exercise
    else:
        every = 1
    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("risk-neutral probability outside (0,1); increase n_steps")
    disc = math.exp(-r * dt)
    j = np.arange(n_steps + 1)
    spot = S * u ** (2 * j - n_steps)
    sign = 1.0 if kind == "call" else -1.0
    value = np.maximum(sign * (spot - K), 0.0)
    for step in range(n_steps - 1, -1, -1):
        value = disc * (p * value[1:] + (1.0 - p) * value[:-1])
        if american and step % every == 0:  # step 0 = exercise immediately
            spot = S * u ** (2 * np.arange(step + 1) - step)
            value = np.maximum(value, sign * (spot - K))
    return float(value[0])
