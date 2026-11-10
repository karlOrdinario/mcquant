"""Market data: a realistic synthetic generator plus real-data loaders.

The synthetic generator produces correlated multi-asset prices with
GARCH(1,1) volatility clustering and Student-t shocks, i.e. the two stylised
facts (fat tails, calm/turbulent regimes) that break naive Gaussian VaR.
That makes it a fair test bed for the backtesting module when no live data
is available.

Real data: :func:`load_csv` (any Date + price columns file) or
:func:`load_yfinance` (needs ``pip install yfinance`` and internet access).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_TICKERS = ("TECH", "BANKS", "STAPLES", "ENERGY")
DEFAULT_MU = (0.12, 0.08, 0.06, 0.07)
DEFAULT_VOL = (0.28, 0.24, 0.14, 0.32)
DEFAULT_CORR = np.array(
    [
        [1.00, 0.55, 0.35, 0.30],
        [0.55, 1.00, 0.40, 0.45],
        [0.35, 0.40, 1.00, 0.25],
        [0.30, 0.45, 0.25, 1.00],
    ]
)


def synthetic_market(
    n_days: int = 2_520,
    tickers: Sequence[str] = DEFAULT_TICKERS,
    annual_mu: Sequence[float] = DEFAULT_MU,
    annual_vol: Sequence[float] = DEFAULT_VOL,
    corr: np.ndarray = DEFAULT_CORR,
    df: float = 6.0,
    garch_alpha: float = 0.06,
    garch_beta: float = 0.92,
    start: str = "2015-01-02",
    s0: float = 100.0,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Daily prices with GARCH(1,1)-t dynamics and a fixed correlation matrix.

    Per asset: sigma2_t = omega + a * eps_{t-1}^2 + b * sigma2_{t-1}, with
    omega set so the long-run vol equals ``annual_vol``.

    Defaults keep the 4th moment finite (beta^2 + 2 alpha beta + alpha^2 E[z^4] < 1);
    otherwise sample volatilities are so skewed they rarely match the target.
    """
    k = len(tickers)
    mu, vol = np.asarray(annual_mu, float), np.asarray(annual_vol, float)
    corr = np.asarray(corr, float)
    if not (mu.size == vol.size == k and corr.shape == (k, k)):
        raise ValueError("tickers, annual_mu, annual_vol and corr must agree in size")
    if garch_alpha + garch_beta >= 1:
        raise ValueError("garch_alpha + garch_beta must be < 1 (stationary)")
    if df <= 2:
        raise ValueError("df must be > 2")
    if df > 4:
        kurt_z = 3 * (df - 2) / (df - 4)
        if garch_beta**2 + 2 * garch_alpha * garch_beta + garch_alpha**2 * kurt_z >= 1:
            import warnings

            warnings.warn("GARCH parameters imply infinite 4th moment of returns", stacklevel=2)

    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(corr)
    long_var = vol**2 / TRADING_DAYS
    omega = long_var * (1 - garch_alpha - garch_beta)
    # unit-variance correlated Student-t shocks
    shocks = rng.standard_normal((n_days, k)) @ chol.T
    shocks *= np.sqrt((df - 2) / rng.chisquare(df, size=(n_days, 1)))

    var = long_var.copy()
    rets = np.empty((n_days, k))
    for t in range(n_days):
        eps = np.sqrt(var) * shocks[t]
        rets[t] = mu / TRADING_DAYS - 0.5 * var + eps
        var = omega + garch_alpha * eps**2 + garch_beta * var

    dates = pd.bdate_range(start=start, periods=n_days + 1)
    prices = s0 * np.exp(np.vstack([np.zeros(k), np.cumsum(rets, axis=0)]))
    return pd.DataFrame(prices, index=dates, columns=list(tickers))


def load_csv(path: str | Path, date_col: str = "Date") -> pd.DataFrame:
    """Read a wide CSV: one date column, one price column per asset."""
    df = pd.read_csv(path, parse_dates=[date_col]).set_index(date_col).sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if df.empty:
        raise ValueError(f"no numeric price columns found in {path}")
    return df.ffill().dropna()


def load_yfinance(tickers: Sequence[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Adjusted close prices from Yahoo Finance (run locally; needs internet)."""
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("pip install yfinance to download real market data") from e
    data = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])
    return data.dropna(how="all").ffill().dropna()


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def estimate_parameters(prices: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Annualised arithmetic drift ``mu`` and log-return covariance ``cov``.

    mu_arith = 252 * mean(log r) + 0.5 * diag(cov), matching the convention
    in :func:`mcquant.risk.simulate_log_returns`.
    """
    r = log_returns(prices)
    cov = r.cov().to_numpy() * TRADING_DAYS
    mu = r.mean().to_numpy() * TRADING_DAYS + 0.5 * np.diag(cov)
    return mu, cov
