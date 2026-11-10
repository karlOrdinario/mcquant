"""Portfolio risk: Monte Carlo VaR / CVaR with full revaluation, stress tests
and statistical VaR backtesting.

Conventions
-----------
* P&L is in currency units; a loss is a negative P&L.
* ``VaR_alpha`` is reported as a positive loss: the amount you lose no more
  than ``alpha`` of the time, ``VaR = -quantile(PnL, 1 - alpha)``.
* ``CVaR`` (expected shortfall) is the average loss *given* the loss exceeds
  VaR. It is the measure Basel's FRTB uses because it sees the tail shape.
* ``mu`` is the annual arithmetic drift and ``cov`` the annual covariance of
  log returns; horizons are in trading days (252 per year).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import xlogy
from scipy.stats import binom, chi2, kurtosis, norm
from scipy.stats import t as student_t

from .analytic import bs_delta, bs_price

TRADING_DAYS = 252
Instrument = Literal["stock", "call", "put"]
Distribution = Literal["normal", "t"]


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Position:
    asset: int
    quantity: float
    instrument: Instrument = "stock"
    strike: float | None = None
    expiry: float | None = None  # years from today
    vol: float | None = None  # implied vol used to revalue the option

    def __post_init__(self) -> None:
        if self.instrument not in ("stock", "call", "put"):
            raise ValueError("instrument must be 'stock', 'call' or 'put'")
        if self.instrument != "stock" and None in (self.strike, self.expiry, self.vol):
            raise ValueError("option positions need strike, expiry and vol")


@dataclass
class Portfolio:
    spots: np.ndarray
    positions: list[Position]
    r: float = 0.0
    names: list[str] | None = None

    def __post_init__(self) -> None:
        self.spots = np.asarray(self.spots, dtype=float)
        for p in self.positions:
            if not 0 <= p.asset < self.spots.size:
                raise ValueError(f"position refers to unknown asset {p.asset}")
        if self.names is None:
            self.names = [f"Asset {i}" for i in range(self.spots.size)]

    @property
    def n_assets(self) -> int:
        return self.spots.size

    def value(self, spots: np.ndarray | None = None, elapsed: float = 0.0, vol_shift: float = 0.0) -> np.ndarray:
        """Full revaluation. ``spots`` is ``(n_scenarios, n_assets)``; ``elapsed``
        is time passed in years (options lose time value); ``vol_shift`` is an
        additive shock to every implied vol."""
        s = np.atleast_2d(self.spots if spots is None else spots)
        total = np.zeros(s.shape[0])
        for p in self.positions:
            x = s[:, p.asset]
            if p.instrument == "stock":
                total += p.quantity * x
            else:
                ttm = max(p.expiry - elapsed, 0.0)
                vol = max(p.vol + vol_shift, 1e-4)
                total += p.quantity * np.asarray(bs_price(x, p.strike, ttm, self.r, vol, kind=p.instrument))
        return total

    @property
    def current_value(self) -> float:
        return float(self.value()[0])

    def dollar_deltas(self) -> np.ndarray:
        """Currency exposure per asset: d(value) / d(log S_i) ~= sum qty * delta * S."""
        exp_ = np.zeros(self.n_assets)
        for p in self.positions:
            s = self.spots[p.asset]
            d = (
                1.0
                if p.instrument == "stock"
                else float(bs_delta(s, p.strike, p.expiry, self.r, p.vol, kind=p.instrument))
            )
            exp_[p.asset] += p.quantity * d * s
        return exp_


# --------------------------------------------------------------------------- #
# Scenario generation
# --------------------------------------------------------------------------- #
def _matrix_sqrt(cov: np.ndarray) -> np.ndarray:
    """Cholesky factor, or an eigen-decomposition square root if ``cov`` is only
    positive *semi*-definite (e.g. two perfectly correlated assets)."""
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        if vals.min() < -1e-10 * max(vals.max(), 1.0):
            raise ValueError("cov must be positive semi-definite") from None
        return vecs * np.sqrt(np.clip(vals, 0.0, None))


def simulate_log_returns(
    mu: np.ndarray,
    cov: np.ndarray,
    horizon_days: int,
    n_scenarios: int,
    dist: Distribution = "normal",
    df: float = 5.0,
    seed: int | None = None,
) -> np.ndarray:
    """Correlated horizon log returns via Cholesky: R = m + L Z.

    ``dist="t"`` draws a multivariate Student-t with the *same covariance*
    (Z * sqrt((df - 2) / W), W ~ chi^2_df), so any difference in VaR is due to
    fatter tails only, not higher volatility.
    """
    mu, cov = np.asarray(mu, float), np.asarray(cov, float)
    if cov.shape != (mu.size, mu.size):
        raise ValueError("cov must be (n_assets, n_assets)")
    if not np.allclose(cov, cov.T):
        raise ValueError("cov must be symmetric")
    chol = _matrix_sqrt(cov)
    h = horizon_days / TRADING_DAYS
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_scenarios, mu.size)) @ chol.T
    if dist == "t":
        if df <= 2:
            raise ValueError("df must be > 2 for a finite variance")
        z *= np.sqrt((df - 2.0) / rng.chisquare(df, size=(n_scenarios, 1)))
    elif dist != "normal":
        raise ValueError("dist must be 'normal' or 't'")
    return (mu - 0.5 * np.diag(cov)) * h + z * math.sqrt(h)


def cov_from_vols_corr(vols: Sequence[float], corr: np.ndarray) -> np.ndarray:
    v = np.asarray(vols, float)
    return np.outer(v, v) * np.asarray(corr, float)


# --------------------------------------------------------------------------- #
# VaR / CVaR
# --------------------------------------------------------------------------- #
@dataclass
class RiskReport:
    method: str
    alpha: float
    horizon_days: int
    var: float
    cvar: float
    expected_pnl: float
    var_se: float = math.nan
    pnl: np.ndarray | None = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "VaR": self.var,
            "CVaR": self.cvar,
            "VaR s.e.": self.var_se,
            "E[P&L]": self.expected_pnl,
        }


def var_cvar(pnl: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    if not 0.5 < alpha < 1:
        raise ValueError("alpha must be in (0.5, 1)")
    pnl = np.asarray(pnl, float)
    q = np.quantile(pnl, 1.0 - alpha)
    return float(-q), float(-pnl[pnl <= q].mean())


def _batch_se(pnl: np.ndarray, alpha: float, n_batches: int = 20) -> float:
    vs = [var_cvar(b, alpha)[0] for b in np.array_split(pnl, n_batches)]
    return float(np.std(vs, ddof=1) / math.sqrt(n_batches))


def monte_carlo_var(
    portfolio: Portfolio,
    mu: np.ndarray,
    cov: np.ndarray,
    alpha: float = 0.99,
    horizon_days: int = 1,
    n_scenarios: int = 100_000,
    dist: Distribution = "normal",
    df: float = 5.0,
    seed: int | None = None,
) -> RiskReport:
    """Simulate horizon prices, fully reprice every position, read off the tail."""
    rets = simulate_log_returns(mu, cov, horizon_days, n_scenarios, dist, df, seed)
    future = portfolio.spots * np.exp(rets)
    pnl = portfolio.value(future, elapsed=horizon_days / TRADING_DAYS) - portfolio.current_value
    var, cvar = var_cvar(pnl, alpha)
    label = "Monte Carlo (normal)" if dist == "normal" else f"Monte Carlo (Student-t, df={df:g})"
    return RiskReport(label, alpha, horizon_days, var, cvar, float(pnl.mean()), _batch_se(pnl, alpha), pnl)


def delta_normal_var(
    portfolio: Portfolio, mu: np.ndarray, cov: np.ndarray, alpha: float = 0.99, horizon_days: int = 1
) -> RiskReport:
    """Parametric (variance-covariance) VaR from first-order exposures.

    Ignores gamma, so it misstates risk for option books - which the Monte
    Carlo full-revaluation number exposes.
    """
    e = portfolio.dollar_deltas()
    h = horizon_days / TRADING_DAYS
    m = float(e @ np.asarray(mu)) * h
    s = math.sqrt(float(e @ np.asarray(cov) @ e) * h)
    z = norm.ppf(alpha)
    cvar = s * norm.pdf(z) / (1 - alpha) - m
    return RiskReport("Delta-normal (parametric)", alpha, horizon_days, z * s - m, cvar, m)


def historical_var(
    portfolio: Portfolio, log_returns: np.ndarray | pd.DataFrame, alpha: float = 0.99, horizon_days: int = 1
) -> RiskReport:
    """Historical simulation: replay every past (overlapping) horizon move."""
    r = np.asarray(log_returns, float)
    if horizon_days > 1:
        r = sliding_window_view(r, horizon_days, axis=0).sum(axis=-1)
    future = portfolio.spots * np.exp(r)
    pnl = portfolio.value(future, elapsed=horizon_days / TRADING_DAYS) - portfolio.current_value
    var, cvar = var_cvar(pnl, alpha)
    return RiskReport("Historical simulation", alpha, horizon_days, var, cvar, float(pnl.mean()), pnl=pnl)


# --------------------------------------------------------------------------- #
# Stress testing
# --------------------------------------------------------------------------- #
DEFAULT_STRESS: dict[str, dict] = {
    "Equity crash (-20%, vol +15pts)": {"spot_shock": -0.20, "vol_shift": 0.15},
    "GFC-style (-40%, vol +30pts)": {"spot_shock": -0.40, "vol_shift": 0.30},
    "Correction (-10%, vol +5pts)": {"spot_shock": -0.10, "vol_shift": 0.05},
    "Vol spike only (+10pts)": {"spot_shock": 0.0, "vol_shift": 0.10},
    "Rally (+10%, vol -5pts)": {"spot_shock": 0.10, "vol_shift": -0.05},
}


def stress_test(portfolio: Portfolio, scenarios: dict[str, dict] | None = None) -> pd.DataFrame:
    """Instantaneous shocks. ``spot_shock`` is a simple return, scalar or per asset."""
    scenarios = scenarios or DEFAULT_STRESS
    base = portfolio.current_value
    rows = []
    for name, spec in scenarios.items():
        shock = np.broadcast_to(np.asarray(spec.get("spot_shock", 0.0), float), portfolio.spots.shape)
        if np.any(shock <= -1.0):
            raise ValueError(f"{name}: a spot shock of -100% or worse is not a valid price move")
        v = float(portfolio.value(portfolio.spots * (1 + shock), vol_shift=spec.get("vol_shift", 0.0))[0])
        rows.append({"scenario": name, "P&L": v - base, "P&L %": (v - base) / base * 100})
    return pd.DataFrame(rows).set_index("scenario")


# --------------------------------------------------------------------------- #
# Backtesting
# --------------------------------------------------------------------------- #
def kupiec_pof(n_obs: int, n_exceptions: int, alpha: float) -> dict:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test.

    H0: the true exception rate equals 1 - alpha. LR ~ chi^2(1).
    """
    p = 1.0 - alpha
    x, n = n_exceptions, n_obs
    phat = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - phat) + xlogy(x, phat)
    lr = max(-2.0 * (ll0 - ll1), 0.0)
    return {"LR_pof": float(lr), "p_pof": float(chi2.sf(lr, 1))}


def christoffersen_independence(exceptions: np.ndarray) -> dict:
    """Christoffersen (1998) test that exceptions do not cluster. LR ~ chi^2(1)."""
    e = np.asarray(exceptions, bool).astype(int)
    prev, curr = e[:-1], e[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = max(-2.0 * (ll0 - ll1), 0.0)
    return {"LR_ind": float(lr), "p_ind": float(chi2.sf(lr, 1))}


def basel_zone(n_obs: int, n_exceptions: int, alpha: float = 0.99) -> str:
    """Basel traffic light: green if the cumulative binomial probability of
    seeing this few exceptions is < 95%, yellow if < 99.99%, else red.
    (For 250 days at 99% that is green 0-4, yellow 5-9, red 10+.)"""
    c = binom.cdf(n_exceptions, n_obs, 1 - alpha)
    return "green" if c < 0.95 else ("yellow" if c < 0.9999 else "red")


BACKTEST_METHODS = ("normal", "student_t", "historical", "ewma", "fhs")


def _ewma_sigma(r: np.ndarray, lam: float, init_var: float) -> np.ndarray:
    """sigma[t] is the forecast for day t made with data up to t-1."""
    var = np.empty(r.size)
    var[0] = init_var
    for t in range(1, r.size):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return np.sqrt(var)


@dataclass
class BacktestResult:
    forecasts: pd.DataFrame  # realised return and each method's VaR (as positive loss fraction)
    summary: pd.DataFrame
    alpha: float


def rolling_var_backtest(
    returns: pd.Series | np.ndarray,
    alpha: float = 0.99,
    window: int = 250,
    methods: Sequence[str] = BACKTEST_METHODS,
    ewma_lambda: float = 0.94,
) -> BacktestResult:
    """Forecast 1-day VaR every day from the previous ``window`` returns and
    count how often the realised loss exceeds it.

    Methods: ``normal`` (static Gaussian), ``student_t`` (moment-matched t),
    ``historical``, ``ewma`` (RiskMetrics), ``fhs`` (filtered historical
    simulation: EWMA vol x empirical quantile of standardised returns).
    """
    s = returns if isinstance(returns, pd.Series) else pd.Series(np.asarray(returns, float))
    r = s.to_numpy(float)
    if r.size <= window + 10:
        raise ValueError("need more observations than the window")
    unknown = set(methods) - set(BACKTEST_METHODS)
    if unknown:
        raise ValueError(f"unknown methods {unknown}")
    p = 1.0 - alpha
    win = sliding_window_view(r, window)[:-1]  # win[i] -> forecast for r[window + i]
    realised = r[window:]
    mu, sd = win.mean(axis=1), win.std(axis=1, ddof=1)
    out = {"return": realised}

    if "normal" in methods:
        out["normal"] = -(mu + sd * norm.ppf(p))
    if "student_t" in methods:
        k = kurtosis(win, axis=1, fisher=True, bias=False)
        nu = np.clip(np.where(k > 0, 4.0 + 6.0 / np.maximum(k, 1e-9), 30.0), 4.1, 30.0)
        scale = sd * np.sqrt((nu - 2.0) / nu)
        out["student_t"] = -(mu + scale * student_t.ppf(p, nu))
    if "historical" in methods:
        out["historical"] = -np.quantile(win, p, axis=1)
    if "ewma" in methods or "fhs" in methods:
        sig = _ewma_sigma(r, ewma_lambda, init_var=float(np.var(r[:window])))
        if "ewma" in methods:
            out["ewma"] = -(sig[window:] * norm.ppf(p))
        if "fhs" in methods:
            z = r / sig
            zwin = sliding_window_view(z, window)[:-1]
            out["fhs"] = -(sig[window:] * np.quantile(zwin, p, axis=1))

    idx = s.index[window:]
    forecasts = pd.DataFrame(out, index=idx)
    rows = []
    n = realised.size
    for m in methods:
        exc = realised < -forecasts[m].to_numpy()
        x = int(exc.sum())
        k = kupiec_pof(n, x, alpha)
        c = christoffersen_independence(exc)
        lr_cc = k["LR_pof"] + c["LR_ind"]
        rows.append(
            {
                "method": m,
                "days": n,
                "exceptions": x,
                "expected": n * p,
                "rate %": 100 * x / n,
                **k,
                **c,
                "p_cc": float(chi2.sf(lr_cc, 2)),
                "basel_zone": basel_zone(n, x, alpha),
                "verdict": "pass" if (k["p_pof"] > 0.05 and c["p_ind"] > 0.05) else "fail",
            }
        )
    return BacktestResult(forecasts, pd.DataFrame(rows).set_index("method"), alpha)
