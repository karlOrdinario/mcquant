"""mcquant - a validated Monte Carlo engine for derivatives pricing and portfolio risk."""

from .american import LSMResult, lsm_price
from .analytic import (
    barrier_price,
    binomial_price,
    bs_digital_delta,
    bs_digital_price,
    bs_greeks,
    bs_price,
    geometric_asian_price,
    heston_price,
    implied_vol,
    merton_price,
)
from .greeks import fd_greeks, likelihood_ratio_delta, likelihood_ratio_gamma, pathwise_delta, pathwise_vega
from .models import GBM, Heston, Merton, Paths
from .payoffs import AsianArithmetic, AsianGeometric, Barrier, Digital, European
from .pricer import ControlVariate, MCResult, geometric_asian_control, mc_price, stock_control
from .risk import (
    Portfolio,
    Position,
    delta_normal_var,
    historical_var,
    monte_carlo_var,
    rolling_var_backtest,
    stress_test,
)

__version__ = "1.1.0"

__all__ = [
    "GBM",
    "Heston",
    "Merton",
    "Paths",
    "European",
    "AsianArithmetic",
    "AsianGeometric",
    "Barrier",
    "Digital",
    "mc_price",
    "MCResult",
    "ControlVariate",
    "stock_control",
    "geometric_asian_control",
    "lsm_price",
    "LSMResult",
    "pathwise_delta",
    "pathwise_vega",
    "likelihood_ratio_delta",
    "likelihood_ratio_gamma",
    "fd_greeks",
    "bs_price",
    "bs_greeks",
    "bs_digital_price",
    "bs_digital_delta",
    "implied_vol",
    "geometric_asian_price",
    "barrier_price",
    "heston_price",
    "merton_price",
    "binomial_price",
    "Portfolio",
    "Position",
    "monte_carlo_var",
    "delta_normal_var",
    "historical_var",
    "stress_test",
    "rolling_var_backtest",
]
