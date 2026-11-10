"""Self-validation: price a panel of benchmark contracts by Monte Carlo and
compare each with an independent closed-form or lattice answer.

A benchmark passes if the Monte Carlo estimate is within 4 standard errors of
the reference (plus a stated allowance for known *method* bias such as time
discretisation or the Broadie-Glasserman-Kou approximation).

    >>> from mcquant.validation import run_benchmarks
    >>> run_benchmarks(n_paths=2**16, seed=1)        # doctest: +SKIP
"""

from __future__ import annotations

import time

import pandas as pd

from .american import lsm_price
from .analytic import (
    barrier_price,
    binomial_price,
    bs_digital_price,
    bs_greeks,
    bs_price,
    geometric_asian_price,
    heston_price,
    merton_price,
)
from .greeks import likelihood_ratio_gamma, pathwise_delta, pathwise_vega
from .models import GBM, Heston, Merton
from .payoffs import AsianGeometric, Barrier, Digital, European
from .pricer import mc_price, stock_control


def run_benchmarks(n_paths: int = 2**16, seed: int = 2024, n_se: float = 4.0) -> pd.DataFrame:
    gbm = GBM(100, 0.05, 0.2, q=0.01)
    heston = Heston(100, 0.03, 0.04, 1.5, 0.04, 0.5, -0.7)
    merton = Merton(100, 0.05, 0.2, 0.75, -0.1, 0.15)
    g = bs_greeks(100, 100, 1, 0.05, 0.2, 0.01)

    cases = [
        (
            "European call (GBM)",
            "Black-Scholes",
            0.0,
            lambda: mc_price(gbm, European(100), 1, 1, n_paths, "pseudo", seed),
            lambda: bs_price(100, 100, 1, 0.05, 0.2, 0.01),
        ),
        (
            "European put (GBM, antithetic)",
            "Black-Scholes",
            0.0,
            lambda: mc_price(gbm, European(95, "put"), 1, 1, n_paths, "antithetic", seed),
            lambda: bs_price(100, 95, 1, 0.05, 0.2, 0.01, "put"),
        ),
        (
            "European call (GBM, Sobol RQMC)",
            "Black-Scholes",
            0.0,
            lambda: mc_price(gbm, European(110), 1, 1, n_paths, "sobol", seed),
            lambda: bs_price(100, 110, 1, 0.05, 0.2, 0.01),
        ),
        (
            "Digital call",
            "closed form",
            0.0,
            lambda: mc_price(gbm, Digital(105), 1, 1, n_paths, "antithetic", seed),
            lambda: bs_digital_price(100, 105, 1, 0.05, 0.2, 0.01),
        ),
        (
            "Geometric Asian call, 12 fixings",
            "closed form",
            0.0,
            lambda: mc_price(gbm, AsianGeometric(100), 1, 12, n_paths, "sobol", seed),
            lambda: geometric_asian_price(100, 100, 1, 0.05, 0.2, 12, 0.01),
        ),
        (
            "Up-and-out call, daily monitoring",
            "BGK-corrected formula",
            0.01,
            lambda: mc_price(gbm, Barrier(100, 125, "up-and-out"), 1, 252, n_paths, "antithetic", seed),
            lambda: barrier_price(100, 100, 125, 1, 0.05, 0.2, "up-and-out", q=0.01, n_monitor=252),
        ),
        (
            "Down-and-in put, daily monitoring",
            "BGK-corrected formula",
            0.01,
            lambda: mc_price(gbm, Barrier(100, 90, "down-and-in", "put"), 1, 252, n_paths, "antithetic", seed),
            lambda: barrier_price(100, 100, 90, 1, 0.05, 0.2, "down-and-in", "put", q=0.01, n_monitor=252),
        ),
        (
            "Heston call (rho = -0.7)",
            "Heston characteristic fn",
            0.02,
            lambda: mc_price(heston, European(100), 1, 200, n_paths, "antithetic", seed, stock_control(heston)),
            lambda: heston_price(100, 100, 1, 0.03, 0.04, 1.5, 0.04, 0.5, -0.7),
        ),
        (
            "Merton jump-diffusion call",
            "Merton series",
            0.0,
            lambda: mc_price(merton, European(100), 1, 1, n_paths, "antithetic", seed, stock_control(merton)),
            lambda: merton_price(100, 100, 1, 0.05, 0.2, 0.75, -0.1, 0.15),
        ),
        (
            "American put (LSM, 50 dates)",
            "Bermudan binomial tree",
            0.04,
            lambda: lsm_price(GBM(36, 0.06, 0.2), 40, 1, "put", 50, n_paths, seed=seed),
            lambda: binomial_price(36, 40, 1, 0.06, 0.2, n_steps=2500, n_exercise=50),
        ),
        (
            "Delta (pathwise)",
            "Black-Scholes delta",
            0.0,
            lambda: pathwise_delta(gbm, European(100), 1, n_paths=n_paths, method="antithetic", seed=seed),
            lambda: g.delta,
        ),
        (
            "Vega (pathwise)",
            "Black-Scholes vega",
            0.0,
            lambda: pathwise_vega(gbm, 100, 1, "call", n_paths, "antithetic", seed),
            lambda: g.vega,
        ),
        (
            "Gamma (likelihood ratio)",
            "Black-Scholes gamma",
            0.0,
            lambda: likelihood_ratio_gamma(gbm, European(100), 1, n_paths, seed=seed),
            lambda: g.gamma,
        ),
    ]

    rows = []
    for name, ref_name, bias_allowance, run, ref in cases:
        t0 = time.perf_counter()
        res = run()
        mc = getattr(res, "price", getattr(res, "value", None))
        se = res.std_error
        reference = ref()
        err = mc - reference
        ok = abs(err) <= n_se * se + bias_allowance
        rows.append(
            {
                "benchmark": name,
                "Monte Carlo": mc,
                "std error": se,
                "reference": reference,
                "reference method": ref_name,
                "error": err,
                "error / SE": err / se if se > 0 else 0.0,
                "pass": ok,
                "seconds": time.perf_counter() - t0,
            }
        )
    return pd.DataFrame(rows)
