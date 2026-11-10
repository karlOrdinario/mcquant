"""American / Bermudan options by Longstaff-Schwartz least-squares Monte Carlo.

At each exercise date (working backwards from maturity) we regress the
discounted realised cash flow of in-the-money paths on polynomials of the
spot price. The fitted value is the *continuation value*: exercise only if
the payoff today is larger.

Two estimators:

* in-sample (classic LSM): fit and price on the same paths - slightly biased
  high because the rule "peeks" at the paths it is evaluated on;
* out-of-sample (default): fit the rule on an independent training set, then
  apply it to fresh pricing paths. Any exercise rule gives a *lower bound*, so
  this estimator is biased low by a provably small amount and has an honest
  standard error.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .models import Model
from .payoffs import Kind
from .pricer import MCResult
from .rng import Method, mean_and_se


@dataclass(frozen=True)
class LSMResult(MCResult):
    european_price: float = math.nan  # same paths, no early exercise
    boundary_times: np.ndarray | None = None
    boundary_spots: np.ndarray | None = None  # estimated early-exercise boundary

    @property
    def early_exercise_premium(self) -> float:
        return self.price - self.european_price


def _intrinsic(s: np.ndarray, strike: float, kind: Kind) -> np.ndarray:
    return np.maximum(s - strike, 0.0) if kind == "call" else np.maximum(strike - s, 0.0)


def _basis(x: np.ndarray, degree: int) -> np.ndarray:
    # Powers of moneyness S/K; scaling by K keeps the normal equations well conditioned.
    return np.vander(x, degree + 1, increasing=True)


def _backward_induction(
    spot: np.ndarray,
    strike: float,
    kind: Kind,
    disc: float,
    degree: int,
    coefs: list[np.ndarray | None] | None = None,
) -> tuple[np.ndarray, list[np.ndarray | None], np.ndarray]:
    n_paths, n_cols = spot.shape
    m = n_cols - 1
    cash = _intrinsic(spot[:, m], strike, kind)
    fitted: list[np.ndarray | None] = [None] * (m + 1)
    boundary = np.full(m + 1, np.nan)
    boundary[m] = strike
    for t in range(m - 1, 0, -1):
        cash *= disc
        ex = _intrinsic(spot[:, t], strike, kind)
        itm = np.flatnonzero(ex > 0)
        if coefs is None:
            if itm.size <= degree + 1:
                continue
            X = _basis(spot[itm, t] / strike, degree)
            beta, *_ = np.linalg.lstsq(X, cash[itm], rcond=None)
            fitted[t] = beta
        else:
            beta = coefs[t]
            if beta is None or itm.size == 0:
                continue
            X = _basis(spot[itm, t] / strike, degree)
        exercise = itm[ex[itm] > X @ beta]
        cash[exercise] = ex[exercise]
        if exercise.size:
            s_ex = spot[exercise, t]
            boundary[t] = s_ex.max() if kind == "put" else s_ex.min()
    cash *= disc
    return cash, fitted, boundary


def lsm_price(
    model: Model,
    strike: float,
    T: float,
    kind: Kind = "put",
    n_steps: int = 50,
    n_paths: int = 100_000,
    degree: int = 3,
    method: Method = "pseudo",
    seed: int | None = None,
    n_batches: int = 32,
    out_of_sample: bool = True,
) -> LSMResult:
    """Bermudan option exercisable on ``n_steps`` equally spaced dates.

    With ~50 dates per year this is a close approximation of the American price.
    """
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if degree < 1:
        raise ValueError("degree must be >= 1")
    start = time.perf_counter()
    disc = math.exp(-model.r * T / n_steps)
    paths = model.simulate(T, n_steps, n_paths, method, seed, n_batches)

    if out_of_sample:
        train_seed = None if seed is None else seed + 104_729
        train = model.simulate(T, n_steps, n_paths, "pseudo", train_seed)
        _, coefs, _ = _backward_induction(train.spot, strike, kind, disc, degree)
        cash, _, boundary = _backward_induction(paths.spot, strike, kind, disc, degree, coefs)
    else:
        cash, _, boundary = _backward_induction(paths.spot, strike, kind, disc, degree)

    est, se = mean_and_se(cash, method, n_batches)
    immediate = float(_intrinsic(np.array([model.s0]), strike, kind)[0])
    if immediate > est:  # exercising today beats waiting
        est, se = immediate, 0.0
    euro, _ = mean_and_se(paths.discount * _intrinsic(paths.spot[:, -1], strike, kind), method, n_batches)
    label = f"lsm-{'oos' if out_of_sample else 'is'}-{method}"
    return LSMResult(
        est,
        se,
        n_paths,
        label,
        time.perf_counter() - start,
        european_price=euro,
        boundary_times=paths.times,
        boundary_spots=boundary,
    )
