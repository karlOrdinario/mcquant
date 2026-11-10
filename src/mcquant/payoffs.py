"""Option payoffs evaluated on simulated paths.

A payoff is any callable ``payoff(spot) -> np.ndarray`` where ``spot`` is the
``(n_paths, n_steps + 1)`` array from :class:`mcquant.models.Paths`. Column 0
is today's price; averages and barrier checks use the monitoring dates
``t_1 .. t_n`` (and the barrier check also includes ``t_0``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Kind = Literal["call", "put"]
BarrierType = Literal["up-and-out", "up-and-in", "down-and-out", "down-and-in"]
BARRIER_TYPES: tuple[str, ...] = ("up-and-out", "up-and-in", "down-and-out", "down-and-in")


def _intrinsic(x: np.ndarray, strike: float, kind: Kind) -> np.ndarray:
    if kind == "call":
        return np.maximum(x - strike, 0.0)
    if kind == "put":
        return np.maximum(strike - x, 0.0)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


@dataclass(frozen=True)
class European:
    strike: float
    kind: Kind = "call"

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        return _intrinsic(spot[:, -1], self.strike, self.kind)

    @property
    def label(self) -> str:
        return f"European {self.kind} K={self.strike:g}"


@dataclass(frozen=True)
class AsianArithmetic:
    """Pays on the arithmetic average of the monitoring prices S(t_1..t_n)."""

    strike: float
    kind: Kind = "call"

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        return _intrinsic(spot[:, 1:].mean(axis=1), self.strike, self.kind)

    @property
    def label(self) -> str:
        return f"Asian (arithmetic) {self.kind} K={self.strike:g}"


@dataclass(frozen=True)
class AsianGeometric:
    """Pays on the geometric average - has a closed form under GBM, so it is
    the ideal control variate for the arithmetic Asian."""

    strike: float
    kind: Kind = "call"

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        return _intrinsic(np.exp(np.log(spot[:, 1:]).mean(axis=1)), self.strike, self.kind)

    @property
    def label(self) -> str:
        return f"Asian (geometric) {self.kind} K={self.strike:g}"


@dataclass(frozen=True)
class Barrier:
    """Discretely monitored single-barrier option (no rebate)."""

    strike: float
    barrier: float
    barrier_type: BarrierType = "up-and-out"
    kind: Kind = "call"

    def __post_init__(self) -> None:
        if self.barrier_type not in BARRIER_TYPES:
            raise ValueError(f"barrier_type must be one of {BARRIER_TYPES}")

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        if self.barrier_type.startswith("up"):
            hit = (spot >= self.barrier).any(axis=1)
        else:
            hit = (spot <= self.barrier).any(axis=1)
        alive = ~hit if self.barrier_type.endswith("out") else hit
        return np.where(alive, _intrinsic(spot[:, -1], self.strike, self.kind), 0.0)

    @property
    def label(self) -> str:
        return f"{self.barrier_type} {self.kind} K={self.strike:g} H={self.barrier:g}"


@dataclass(frozen=True)
class Digital:
    """Cash-or-nothing digital paying 1 if S_T finishes in the money.

    Its payoff is a step function, so the pathwise Greek estimator is zero
    almost surely (wrong!) while the likelihood-ratio estimator still works -
    the classic illustration of when each method applies.
    """

    strike: float
    kind: Kind = "call"

    def __call__(self, spot: np.ndarray) -> np.ndarray:
        s = spot[:, -1]
        return (s > self.strike).astype(float) if self.kind == "call" else (s < self.strike).astype(float)

    @property
    def label(self) -> str:
        return f"Digital {self.kind} K={self.strike:g}"
