"""Monte Carlo pricer with variance reduction.

    price = e^{-rT} * mean(payoff(paths))

Variance reduction available here:

* sampling scheme (``method``): pseudo, antithetic, randomized Sobol
* control variates: any extra payoff whose exact price is known, e.g.
  the discounted terminal stock price or the geometric Asian option.

The control-variate estimator is ``X - b (Y - E[Y])`` with the variance-
minimising ``b = Cov(X, Y) / Var(Y)`` estimated from the same paths.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

from .analytic import geometric_asian_price
from .models import GBM, Model, Paths
from .payoffs import AsianGeometric, Kind
from .rng import Method, effective_samples, mean_and_se

Payoff = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    n_paths: int
    method: str
    elapsed: float
    control_beta: float | None = None
    dof: int | None = None  # degrees of freedom of the SE (set for RQMC: n_batches - 1)

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        """Normal interval, or Student-t when the SE rests on few replicates (RQMC)."""
        z = student_t.ppf(0.5 + level / 2, self.dof) if self.dof else norm.ppf(0.5 + level / 2)
        return self.price - z * self.std_error, self.price + z * self.std_error

    def within(self, reference: float, n_se: float = 4.0, abs_tol: float = 0.0) -> bool:
        """True if ``reference`` is within ``n_se`` standard errors (+ an absolute
        allowance for known discretisation bias)."""
        return abs(self.price - reference) <= n_se * self.std_error + abs_tol

    def __str__(self) -> str:
        lo, hi = self.ci()
        return (
            f"{self.price:.4f} +/- {self.std_error:.4f} (95% CI {lo:.4f}-{hi:.4f}, {self.method}, n={self.n_paths:,})"
        )


@dataclass(frozen=True)
class ControlVariate:
    """A second payoff evaluated on the same paths with a known expectation.

    ``fn(paths)`` returns *discounted* control values and
    ``expectation(T, n_steps)`` their exact mean.
    """

    fn: Callable[[Paths], np.ndarray]
    expectation: Callable[[float, int], float]
    name: str = "control"


def stock_control(model: Model) -> ControlVariate:
    """Discounted terminal price e^{-rT} S_T. Its mean is S_0 e^{-qT} under any
    risk-neutral model, so this works for GBM, Heston and Merton alike."""
    return ControlVariate(
        fn=lambda p: p.discount * p.spot[:, -1],
        expectation=lambda T, n_steps: model.s0 * math.exp(-model.q * T),
        name="stock",
    )


def geometric_asian_control(model: GBM, strike: float, kind: Kind = "call") -> ControlVariate:
    """The geometric Asian option, the textbook control for the arithmetic Asian
    (correlation between the two payoffs is typically > 0.99)."""
    if not isinstance(model, GBM):
        raise TypeError("the geometric-Asian control has a closed form only under GBM")
    geo = AsianGeometric(strike, kind)
    return ControlVariate(
        fn=lambda p: p.discount * geo(p.spot),
        expectation=lambda T, n_steps: geometric_asian_price(
            model.s0, strike, T, model.r, model.sigma, n_steps, model.q, kind
        ),
        name="geometric_asian",
    )


def mc_price(
    model: Model,
    payoff: Payoff,
    T: float,
    n_steps: int = 1,
    n_paths: int = 100_000,
    method: Method = "pseudo",
    seed: int | None = None,
    control: ControlVariate | None = None,
    n_batches: int = 32,
) -> MCResult:
    """Price ``payoff`` under ``model`` by Monte Carlo.

    Returns the estimate together with a correctly computed standard error
    for the chosen sampling scheme.
    """
    start = time.perf_counter()
    paths = model.simulate(T, n_steps, n_paths, method, seed, n_batches)
    x = paths.discount * payoff(paths.spot)
    beta = None
    if control is not None:
        y = control.fn(paths)
        ey = control.expectation(T, n_steps)
        # Fit beta on the samples the estimator actually averages. For antithetic
        # sampling that is the pair means: pairing already cancels the linear
        # part of the payoff, so a beta fitted on raw draws is the wrong one and
        # can make antithetic+CV *worse* than CV alone. (Sobol keeps raw draws:
        # 32 batch means are too few to fit beta reliably.)
        fx, fy = (effective_samples(x, method), effective_samples(y, method)) if method == "antithetic" else (x, y)
        var_y = np.var(fy, ddof=1)
        if var_y > 0:
            beta = float(np.cov(fx, fy, ddof=1)[0, 1] / var_y)
            x = x - beta * (y - ey)
    est, se = mean_and_se(x, method, n_batches)
    dof = n_batches - 1 if method == "sobol" else None
    return MCResult(est, se, n_paths, method + ("+cv" if control else ""), time.perf_counter() - start, beta, dof)
