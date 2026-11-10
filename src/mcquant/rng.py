"""Random number generation for Monte Carlo.

Three sampling schemes are supported:

* ``"pseudo"``     - plain i.i.d. standard normals from NumPy's PCG64 generator.
* ``"antithetic"`` - for every draw ``Z`` we also use ``-Z``. The first half of
  the returned rows are the originals, the second half their mirror images.
  Estimators must average each pair *before* computing a standard error,
  otherwise the reported error is wrong (see :func:`effective_samples`).
* ``"sobol"``      - randomized quasi-Monte Carlo (RQMC). ``n_batches`` (32)
  independently scrambled Sobol sequences are stacked batch-major. Each batch
  is an unbiased estimate, so the standard error is computed across batches
  and confidence intervals use a Student-t with ``n_batches - 1`` dof.

  Known caveat (measured in QA, see docs/QA_REPORT.md): batch means are
  slightly heavy-tailed because each scramble places one point in the extreme
  tail cell of the inverse normal CDF, so nominal 95% RQMC intervals cover
  about 93-94%. 32 batches is the measured sweet spot: 16 under-covers more
  (~92%), 64 doubles the error.

The Brownian-bridge helper re-orders how a path is built so that the first
(best-distributed) Sobol dimensions drive the largest-scale features of the
path - this is what makes QMC effective for path-dependent payoffs.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from scipy.stats import norm, qmc

Method = Literal["pseudo", "antithetic", "sobol"]
METHODS: tuple[str, ...] = ("pseudo", "antithetic", "sobol")

# Keep inverse-CDF output finite: Sobol can produce points extremely close to 0/1.
_EPS = 1e-12


def _check_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def standard_normals(
    n_paths: int,
    dim: int,
    method: Method = "pseudo",
    seed: int | None = None,
    n_batches: int = 32,
) -> np.ndarray:
    """Return an ``(n_paths, dim)`` array of standard normal draws.

    Parameters
    ----------
    n_paths:
        Number of rows (simulated paths).
    dim:
        Number of normals needed per path (time steps x risk factors).
    method:
        ``"pseudo"``, ``"antithetic"`` or ``"sobol"``.
    seed:
        Seed for reproducibility. The same seed always returns the same draws.
    n_batches:
        Number of independent scrambles for ``"sobol"``. ``n_paths`` must be a
        multiple of it.
    """
    _check_method(method)
    if n_paths <= 0 or dim <= 0:
        raise ValueError("n_paths and dim must be positive")

    if method == "pseudo":
        return np.random.default_rng(seed).standard_normal((n_paths, dim))

    if method == "antithetic":
        if n_paths % 2:
            raise ValueError("antithetic sampling needs an even n_paths")
        half = np.random.default_rng(seed).standard_normal((n_paths // 2, dim))
        return np.concatenate([half, -half], axis=0)

    # Randomized QMC
    if n_batches < 2:
        raise ValueError("sobol needs n_batches >= 2 to estimate a standard error")
    if n_paths % n_batches:
        raise ValueError(f"n_paths ({n_paths}) must be a multiple of n_batches ({n_batches})")
    per_batch = n_paths // n_batches
    children = np.random.SeedSequence(seed).spawn(n_batches)
    out = np.empty((n_paths, dim))
    power_of_two = per_batch & (per_batch - 1) == 0
    for b, child in enumerate(children):
        engine = qmc.Sobol(d=dim, scramble=True, seed=np.random.default_rng(child))
        with warnings.catch_warnings():
            # scipy warns when n is not a power of two (balance properties are
            # weaker but the estimator is still unbiased under scrambling).
            warnings.simplefilter("ignore", UserWarning)
            if power_of_two:
                u = engine.random_base2(int(np.log2(per_batch)))
            else:
                u = engine.random(per_batch)
        out[b * per_batch : (b + 1) * per_batch] = norm.ppf(np.clip(u, _EPS, 1 - _EPS))
    return out


def effective_samples(x: np.ndarray, method: Method, n_batches: int = 32) -> np.ndarray:
    """Collapse raw per-path samples into *independent* samples.

    * pseudo     -> unchanged
    * antithetic -> the average of each (Z, -Z) pair
    * sobol      -> the mean of each scrambled batch

    The standard error of the estimator is then simply
    ``std(eff, ddof=1) / sqrt(len(eff))``.
    """
    _check_method(method)
    x = np.asarray(x, dtype=float)
    if method == "pseudo":
        return x
    if method == "antithetic":
        h = x.shape[0] // 2
        return 0.5 * (x[:h] + x[h:])
    return x.reshape(n_batches, -1).mean(axis=1)


def mean_and_se(x: np.ndarray, method: Method, n_batches: int = 32) -> tuple[float, float]:
    """Monte Carlo mean and its standard error, respecting the sampling scheme."""
    eff = effective_samples(x, method, n_batches)
    if eff.size < 2:
        raise ValueError("need at least two effective samples for a standard error")
    return float(eff.mean()), float(eff.std(ddof=1) / np.sqrt(eff.size))


# --------------------------------------------------------------------------- #
# Brownian bridge
# --------------------------------------------------------------------------- #
def _bridge_schedule(n_steps: int) -> list[tuple[int, int, int]]:
    """Construction order (target, left, right) on grid indices 0..n_steps.

    The terminal point is built first, then midpoints breadth-first.
    """
    schedule: list[tuple[int, int, int]] = [(n_steps, 0, -1)]  # right=-1: no right anchor
    queue = [(0, n_steps)]
    while queue:
        nxt = []
        for left, right in queue:
            if right - left > 1:
                mid = (left + right) // 2
                schedule.append((mid, left, right))
                nxt.extend([(left, mid), (mid, right)])
        queue = nxt
    return schedule


def brownian_increments(z: np.ndarray, times: np.ndarray, bridge: bool = False) -> np.ndarray:
    """Turn standard normals into Brownian increments ``dW`` on a time grid.

    Parameters
    ----------
    z:
        ``(n_paths, n_steps)`` standard normals.
    times:
        ``(n_steps + 1,)`` increasing grid starting at 0.
    bridge:
        If ``True`` build the path with a Brownian bridge (column 0 of ``z``
        sets the terminal value, column 1 the midpoint, ...). The resulting
        increments have exactly the same distribution as the standard
        construction, only the mapping from ``z`` to the path changes.
    """
    z = np.asarray(z, dtype=float)
    dt = np.diff(times)
    n_steps = dt.size
    if z.shape[1] != n_steps:
        raise ValueError("z must have one column per time step")
    if not bridge:
        return z * np.sqrt(dt)

    w = np.zeros((z.shape[0], n_steps + 1))
    for k, (i, left, right) in enumerate(_bridge_schedule(n_steps)):
        if right < 0:  # terminal point
            w[:, i] = np.sqrt(times[i] - times[left]) * z[:, k]
            continue
        tl, ti, tr = times[left], times[i], times[right]
        mean = ((tr - ti) * w[:, left] + (ti - tl) * w[:, right]) / (tr - tl)
        std = np.sqrt((ti - tl) * (tr - ti) / (tr - tl))
        w[:, i] = mean + std * z[:, k]
    return np.diff(w, axis=1)
