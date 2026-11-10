"""Risk-neutral price-path models.

Every model exposes ``simulate(T, n_steps, n_paths, method, seed, n_batches)``
and returns a :class:`Paths` object. All dynamics are under the risk-neutral
measure, so ``E[S_T] = S_0 * exp((r - q) T)`` - the tests check this.

* :class:`GBM`    - Black-Scholes geometric Brownian motion (exact scheme).
* :class:`Heston` - stochastic variance, full-truncation Euler scheme.
* :class:`Merton` - GBM plus log-normal jumps (exact scheme).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from .rng import Method, brownian_increments, standard_normals


@dataclass(frozen=True)
class Paths:
    """Simulated paths plus the metadata needed to price on them."""

    times: np.ndarray  # (n_steps + 1,)
    spot: np.ndarray  # (n_paths, n_steps + 1)
    r: float
    method: Method
    n_batches: int
    variance: np.ndarray | None = field(default=None, repr=False)

    @property
    def T(self) -> float:
        return float(self.times[-1])

    @property
    def n_paths(self) -> int:
        return self.spot.shape[0]

    @property
    def discount(self) -> float:
        return math.exp(-self.r * self.T)


def _grid(T: float, n_steps: int) -> np.ndarray:
    if T <= 0:
        raise ValueError("maturity T must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    return np.linspace(0.0, T, n_steps + 1)


def _to_spot(log_s: np.ndarray, s0: float) -> np.ndarray:
    spot = np.exp(log_s, out=log_s)  # in place: the log array is not needed afterwards
    spot[:, 0] = s0  # exact, so barrier checks at t0 are not hit by exp(log(x)) round-off
    return spot


def _log_path(increments: np.ndarray, s0: float) -> np.ndarray:
    """(n_paths, n_steps) log increments -> (n_paths, n_steps + 1) log prices.

    Memory matters here: 100k paths x 252 steps is ~200 MB per float64 matrix,
    so everything is done with ``out=`` / in-place operations.
    """
    n_paths, n_steps = increments.shape
    log_s = np.empty((n_paths, n_steps + 1))
    log_s[:, 0] = 0.0
    np.cumsum(increments, axis=1, out=log_s[:, 1:])
    log_s += math.log(s0)
    return log_s


def _positive(**kwargs: float) -> None:
    for name, value in kwargs.items():
        if not value > 0:
            raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class GBM:
    """dS = (r - q) S dt + sigma S dW."""

    s0: float
    r: float
    sigma: float
    q: float = 0.0

    def __post_init__(self) -> None:
        _positive(s0=self.s0, sigma=self.sigma)

    def simulate(
        self,
        T: float,
        n_steps: int = 1,
        n_paths: int = 100_000,
        method: Method = "pseudo",
        seed: int | None = None,
        n_batches: int = 32,
    ) -> Paths:
        times = _grid(T, n_steps)
        z = standard_normals(n_paths, n_steps, method, seed, n_batches)
        inc = brownian_increments(z, times, bridge=(method == "sobol"))
        del z
        inc *= self.sigma
        inc += (self.r - self.q - 0.5 * self.sigma**2) * np.diff(times)
        log_s = _log_path(inc, self.s0)
        del inc
        return Paths(times, _to_spot(log_s, self.s0), self.r, method, n_batches)


@dataclass(frozen=True)
class Heston:
    """Heston (1993) stochastic-volatility model.

    dS = (r - q) S dt + sqrt(v) S dW_s
    dv = kappa (theta - v) dt + xi sqrt(v) dW_v,   d<W_s, W_v> = rho dt

    Discretised with the *full truncation* scheme (Lord, Koekkoek & van Dijk,
    2010): the variance may go negative between steps but only ``max(v, 0)``
    ever enters the drift and diffusion. It has the smallest bias among
    simple Euler fixes.
    """

    s0: float
    r: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    q: float = 0.0

    def __post_init__(self) -> None:
        _positive(s0=self.s0, v0=self.v0, kappa=self.kappa, theta=self.theta, xi=self.xi)
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [-1, 1]")

    @property
    def feller_satisfied(self) -> bool:
        """2 kappa theta >= xi^2 keeps the true variance strictly positive."""
        return 2 * self.kappa * self.theta >= self.xi**2

    def simulate(
        self,
        T: float,
        n_steps: int = 100,
        n_paths: int = 100_000,
        method: Method = "pseudo",
        seed: int | None = None,
        n_batches: int = 32,
    ) -> Paths:
        times = _grid(T, n_steps)
        dt = np.diff(times)
        z = standard_normals(n_paths, 2 * n_steps, method, seed, n_batches).reshape(n_paths, n_steps, 2)
        bridge = method == "sobol"
        dw_v = brownian_increments(z[:, :, 0], times, bridge)
        dw_2 = brownian_increments(z[:, :, 1], times, bridge)
        del z
        rho_c = math.sqrt(1.0 - self.rho**2)

        log_s = np.empty((n_paths, n_steps + 1))
        var = np.empty((n_paths, n_steps + 1))
        log_s[:, 0] = math.log(self.s0)
        var[:, 0] = self.v0
        for k in range(n_steps):
            v_pos = np.maximum(var[:, k], 0.0)
            sqrt_v = np.sqrt(v_pos)
            dw_s = self.rho * dw_v[:, k] + rho_c * dw_2[:, k]  # correlate one column at a time
            log_s[:, k + 1] = log_s[:, k] + (self.r - self.q - 0.5 * v_pos) * dt[k] + sqrt_v * dw_s
            var[:, k + 1] = var[:, k] + self.kappa * (self.theta - v_pos) * dt[k] + self.xi * sqrt_v * dw_v[:, k]
        del dw_v, dw_2
        np.maximum(var, 0.0, out=var)
        return Paths(times, _to_spot(log_s, self.s0), self.r, method, n_batches, variance=var)


@dataclass(frozen=True)
class Merton:
    """Merton (1976) jump-diffusion.

    Between jumps the stock follows GBM. Jumps arrive as a Poisson process
    with intensity ``lam`` per year; each jump multiplies the price by
    ``exp(J)`` with ``J ~ N(mu_j, sigma_j^2)``. The drift is compensated by
    ``lam * k`` with ``k = E[e^J] - 1`` so the discounted price is a
    martingale.

    Jump counts always come from a pseudo-random generator (they are discrete),
    while the diffusion uses the requested ``method``.
    """

    s0: float
    r: float
    sigma: float
    lam: float
    mu_j: float
    sigma_j: float
    q: float = 0.0

    def __post_init__(self) -> None:
        _positive(s0=self.s0, sigma=self.sigma)
        if self.lam < 0 or self.sigma_j < 0:
            raise ValueError("lam and sigma_j must be non-negative")

    @property
    def k(self) -> float:
        return math.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1.0

    def simulate(
        self,
        T: float,
        n_steps: int = 1,
        n_paths: int = 100_000,
        method: Method = "pseudo",
        seed: int | None = None,
        n_batches: int = 32,
    ) -> Paths:
        times = _grid(T, n_steps)
        dt = np.diff(times)
        z = standard_normals(n_paths, n_steps, method, seed, n_batches)
        inc = brownian_increments(z, times, bridge=(method == "sobol"))
        del z
        inc *= self.sigma
        inc += (self.r - self.q - 0.5 * self.sigma**2 - self.lam * self.k) * dt

        jump_rng = np.random.default_rng(None if seed is None else seed + 7_919)
        counts = jump_rng.poisson(self.lam * dt, size=(n_paths, n_steps))
        # Sum of N normal jumps is N(N mu, N sigma^2) - exact, no loop over jumps.
        jumps = jump_rng.standard_normal((n_paths, n_steps))
        jumps *= self.sigma_j
        jumps *= np.sqrt(counts)
        jumps += self.mu_j * counts
        del counts
        if method == "antithetic":
            # Mirror the jump part too, so each antithetic pair shares a jump history.
            h = n_paths // 2
            jumps[h:] = jumps[:h]
        inc += jumps
        del jumps
        log_s = _log_path(inc, self.s0)
        del inc
        return Paths(times, _to_spot(log_s, self.s0), self.r, method, n_batches)


Model = GBM | Heston | Merton


def warn_if_feller_violated(model: Heston) -> None:
    if not model.feller_satisfied:
        warnings.warn(
            "Feller condition 2*kappa*theta >= xi^2 is violated: variance can hit zero; "
            "use more time steps for an accurate price.",
            stacklevel=2,
        )
