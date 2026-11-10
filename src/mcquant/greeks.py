"""Monte Carlo Greeks - three estimators with different trade-offs.

=====================  =======================================  ==========================
Estimator              Idea                                     Works for
=====================  =======================================  ==========================
Pathwise (IPA)         differentiate the payoff along each path  continuous payoffs
Likelihood ratio (LR)  differentiate the *density* instead       any payoff, incl. digitals
Finite difference+CRN  bump the input, re-run with SAME randoms   anything (small bias)
=====================  =======================================  ==========================

Each function returns a :class:`GreekEstimate` with a standard error.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from .models import GBM, Model
from .payoffs import Barrier, Digital, European
from .pricer import Payoff
from .rng import Method, mean_and_se


@dataclass(frozen=True)
class GreekEstimate:
    value: float
    std_error: float
    method: str

    def __str__(self) -> str:
        return f"{self.value:.5f} +/- {self.std_error:.5f} ({self.method})"


def _seed(seed: int | None) -> int:
    # Common random numbers need a fixed seed shared by all bumped runs.
    return int(np.random.default_rng().integers(2**31)) if seed is None else seed


# --------------------------------------------------------------------------- #
# Pathwise
# --------------------------------------------------------------------------- #
def pathwise_delta(
    model: Model,
    payoff: Payoff,
    T: float,
    n_steps: int = 1,
    n_paths: int = 100_000,
    method: Method = "pseudo",
    seed: int | None = None,
    n_batches: int = 32,
) -> GreekEstimate:
    """dV/dS0 = E[ e^{-rT} * d payoff / d S0 ] computed path by path.

    In GBM, Heston and Merton every simulated price is proportional to S0
    (S_t = S0 * stuff-independent-of-S0), so dS_t/dS0 = S_t / S0 and the
    pathwise derivative is the directional derivative of the payoff along
    the path itself. That is computed here exactly for piecewise-linear
    payoffs (European, Asian) by a tiny symmetric scaling.
    """
    if isinstance(payoff, (Digital, Barrier)):
        raise ValueError(
            "pathwise derivative is invalid for discontinuous payoffs "
            "(digital, barrier): use likelihood_ratio_delta or fd_greeks"
        )
    paths = model.simulate(T, n_steps, n_paths, method, seed, n_batches)
    h = 1e-6
    up, dn = payoff(paths.spot * (1 + h)), payoff(paths.spot * (1 - h))
    samples = paths.discount * (up - dn) / (2 * h * model.s0)
    return GreekEstimate(*mean_and_se(samples, method, n_batches), "pathwise")


def pathwise_vega(
    model: GBM,
    strike: float,
    T: float,
    kind: str = "call",
    n_paths: int = 100_000,
    method: Method = "pseudo",
    seed: int | None = None,
    n_batches: int = 32,
) -> GreekEstimate:
    """European vega under GBM: dS_T/dsigma = S_T (W_T - sigma T)."""
    if not isinstance(model, GBM):
        raise TypeError("pathwise_vega is implemented for GBM")
    paths = model.simulate(T, 1, n_paths, method, seed, n_batches)
    st = paths.spot[:, -1]
    w_t = (np.log(st / model.s0) - (model.r - model.q - 0.5 * model.sigma**2) * T) / model.sigma
    ds_dsigma = st * (w_t - model.sigma * T)
    itm = st > strike if kind == "call" else st < strike
    sign = 1.0 if kind == "call" else -1.0
    samples = paths.discount * sign * itm * ds_dsigma
    return GreekEstimate(*mean_and_se(samples, method, n_batches), "pathwise")


# --------------------------------------------------------------------------- #
# Likelihood ratio
# --------------------------------------------------------------------------- #
def _lr_setup(model: GBM, payoff: Payoff, T, n_paths, method, seed, n_batches):
    if not isinstance(model, GBM):
        raise TypeError("likelihood-ratio Greeks are implemented for GBM")
    if not isinstance(payoff, (European, Digital)):
        raise TypeError("likelihood-ratio Greeks here need a payoff of S_T only (European, Digital)")
    paths = model.simulate(T, 1, n_paths, method, seed, n_batches)
    st = paths.spot[:, -1]
    sig_rt = model.sigma * math.sqrt(T)
    z = (np.log(st / model.s0) - (model.r - model.q - 0.5 * model.sigma**2) * T) / sig_rt
    return paths.discount * payoff(paths.spot), z, sig_rt


def likelihood_ratio_delta(
    model: GBM,
    payoff: Payoff,
    T: float,
    n_paths: int = 100_000,
    method: Method = "pseudo",
    seed: int | None = None,
    n_batches: int = 32,
) -> GreekEstimate:
    """Delta = E[ e^{-rT} f(S_T) * Z / (S0 sigma sqrt(T)) ]."""
    v, z, sig_rt = _lr_setup(model, payoff, T, n_paths, method, seed, n_batches)
    samples = v * z / (model.s0 * sig_rt)
    return GreekEstimate(*mean_and_se(samples, method, n_batches), "likelihood-ratio")


def likelihood_ratio_gamma(
    model: GBM,
    payoff: Payoff,
    T: float,
    n_paths: int = 100_000,
    method: Method = "pseudo",
    seed: int | None = None,
    n_batches: int = 32,
) -> GreekEstimate:
    """Gamma = E[ e^{-rT} f(S_T) * (Z^2 - 1 - Z sigma sqrt(T)) / (S0^2 sigma^2 T) ]."""
    v, z, sig_rt = _lr_setup(model, payoff, T, n_paths, method, seed, n_batches)
    samples = v * (z * z - 1.0 - z * sig_rt) / (model.s0**2 * sig_rt**2)
    return GreekEstimate(*mean_and_se(samples, method, n_batches), "likelihood-ratio")


# --------------------------------------------------------------------------- #
# Finite differences with common random numbers
# --------------------------------------------------------------------------- #
def _discounted_samples(model, payoff, T, n_steps, n_paths, method, seed, n_batches) -> np.ndarray:
    p = model.simulate(T, n_steps, n_paths, method, seed, n_batches)
    return p.discount * payoff(p.spot)


def fd_greeks(
    model: Model,
    payoff: Payoff,
    T: float,
    n_steps: int = 1,
    n_paths: int = 100_000,
    method: Method = "pseudo",
    seed: int | None = None,
    n_batches: int = 32,
    rel_bump: float = 0.01,
    vol_param: str | None = "auto",
    vol_bump: float = 0.001,
) -> dict[str, GreekEstimate]:
    """Central-difference delta, gamma and vega re-using the same random numbers.

    Re-using the random numbers (common random numbers, CRN) makes the bumped
    and base prices move together, so their *difference* has a tiny variance.
    Without CRN the noise would swamp a 1% bump.

    ``vol_param``: model attribute to bump for vega ("sigma" for GBM/Merton;
    for Heston pass "v0" to get sensitivity to initial variance). ``"auto"``
    picks "sigma" when the model has it, ``None`` skips vega.
    """
    seed = _seed(seed)
    args = (T, n_steps, n_paths, method, seed, n_batches)
    h = rel_bump * model.s0
    base = _discounted_samples(model, payoff, *args)
    up = _discounted_samples(dataclasses.replace(model, s0=model.s0 + h), payoff, *args)
    dn = _discounted_samples(dataclasses.replace(model, s0=model.s0 - h), payoff, *args)
    out = {
        "price": GreekEstimate(*mean_and_se(base, method, n_batches), "mc"),
        "delta": GreekEstimate(*mean_and_se((up - dn) / (2 * h), method, n_batches), "fd-crn"),
        "gamma": GreekEstimate(*mean_and_se((up - 2 * base + dn) / h**2, method, n_batches), "fd-crn"),
    }
    if vol_param == "auto":
        vol_param = "sigma" if hasattr(model, "sigma") else None
    if vol_param:
        v0 = getattr(model, vol_param)
        vu = _discounted_samples(dataclasses.replace(model, **{vol_param: v0 + vol_bump}), payoff, *args)
        vd = _discounted_samples(dataclasses.replace(model, **{vol_param: v0 - vol_bump}), payoff, *args)
        out["vega"] = GreekEstimate(*mean_and_se((vu - vd) / (2 * vol_bump), method, n_batches), "fd-crn")
    return out
