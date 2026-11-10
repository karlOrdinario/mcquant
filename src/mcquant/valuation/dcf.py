"""Deterministic FCFF discounted cash flow.

    Revenue_t  = Revenue_{t-1} x (1 + growth_t)
    EBITDA_t   = Revenue_t x margin_t
    EBIT_t     = EBITDA_t - D&A_t
    NOPAT_t    = EBIT_t x (1 - tax)
    FCFF_t     = NOPAT_t + D&A_t - Capex_t - change in NWC_t

    EV         = sum PV(FCFF_t) + PV(terminal value)
    Equity     = EV - debt + cash - minority interest
    Per share  = Equity / diluted shares

Worked example: the toy company in tests/test_dcf.py, solved by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .inputs import MIN_WACC_SPREAD, DCFAssumptions, EquityBridge, _finite, spread_too_small

TerminalMethod = Literal["gordon", "exit_multiple"]


# --------------------------------------------------------------------------- #
# Forecast
# --------------------------------------------------------------------------- #
def forecast_fcff(a: DCFAssumptions) -> pd.DataFrame:
    """Year-by-year forecast table, one row per forecast year (1..N)."""
    growth = np.asarray(a.revenue_growth)
    revenue = a.base_revenue * np.cumprod(1 + growth)
    prior_revenue = np.concatenate([[a.base_revenue], revenue[:-1]])
    ebitda = revenue * np.asarray(a.ebitda_margin)
    da = revenue * np.asarray(a.da_pct_revenue)
    ebit = ebitda - da
    nopat = ebit * (1 - a.tax_rate)
    capex = revenue * np.asarray(a.capex_pct_revenue)
    change_nwc = (revenue - prior_revenue) * np.asarray(a.nwc_pct_revenue_change)
    fcff = nopat + da - capex - change_nwc
    years = np.arange(1, a.n_years + 1)
    periods = years - 0.5 if a.mid_year else years.astype(float)
    discount_factor = 1 / (1 + a.wacc) ** periods
    return pd.DataFrame(
        {
            "revenue": revenue,
            "growth": growth,
            "ebitda": ebitda,
            "d&a": da,
            "ebit": ebit,
            "nopat": nopat,
            "capex": capex,
            "change_nwc": change_nwc,
            "fcff": fcff,
            "discount_period": periods,
            "discount_factor": discount_factor,
            "pv_fcff": fcff * discount_factor,
        },
        index=pd.Index(years, name="year"),
    )


# --------------------------------------------------------------------------- #
# Terminal value (note 05)
# --------------------------------------------------------------------------- #
def gordon_terminal_value(final_fcff: float, wacc: float, growth: float, min_spread: float = MIN_WACC_SPREAD) -> float:
    """Value at the end of year N of FCFF growing at ``growth`` forever.

    TV_N = FCFF_N x (1 + g) / (WACC - g). Raises if WACC - g < ``min_spread``.
    """
    f, w, g = _finite("final_fcff", final_fcff), _finite("wacc", wacc), _finite("growth", growth)
    if spread_too_small(w, g, min_spread):
        raise ValueError(f"WACC - g = {w - g:.4f} is below the {min_spread:.0%} minimum")
    return f * (1 + g) / (w - g)


def exit_multiple_terminal_value(final_ebitda: float, ev_ebitda_multiple: float) -> float:
    """TV_N = EBITDA_N x EV/EBITDA multiple (multiple from comparable companies)."""
    m = _finite("ev_ebitda_multiple", ev_ebitda_multiple)
    if m <= 0:
        raise ValueError("ev_ebitda_multiple must be positive")
    return _finite("final_ebitda", final_ebitda) * m


def implied_exit_multiple(terminal_value: float, final_ebitda: float) -> float:
    """EV/EBITDA multiple implied by a terminal value (cross-check for Gordon growth)."""
    e = _finite("final_ebitda", final_ebitda)
    if e <= 0:
        raise ValueError("final_ebitda must be positive for a multiple to mean anything")
    return _finite("terminal_value", terminal_value) / e


def implied_perpetual_growth(terminal_value: float, final_fcff: float, wacc: float) -> float:
    """Growth rate implied by a terminal value: g = (TV x WACC - FCFF_N) / (TV + FCFF_N)."""
    tv, f, w = _finite("terminal_value", terminal_value), _finite("final_fcff", final_fcff), _finite("wacc", wacc)
    if tv + f <= 0:
        raise ValueError("terminal_value + final_fcff must be positive")
    return (tv * w - f) / (tv + f)


# --------------------------------------------------------------------------- #
# Equity bridge (note 06)
# --------------------------------------------------------------------------- #
def equity_bridge(enterprise_value: float, bridge: EquityBridge) -> tuple[float, float]:
    """Return (equity value, value per share)."""
    ev = _finite("enterprise_value", enterprise_value)
    equity = ev + bridge.net_adjustment
    return equity, equity / bridge.diluted_shares


def roll_forward(
    value_per_share: float,
    cost_of_equity: float,
    years: float,
    dividends: Sequence[tuple[float, float]] = (),
) -> float:
    """Move a value per share from the valuation date to a later price date.

    An owner's stake grows at the cost of equity, but dividends paid in between
    leave the company, so they are subtracted (grown to the price date):

        V_later = V_0 x (1 + Re)^years - sum(D_i x (1 + Re)^(years - t_i))

    ``dividends`` holds (time paid in years after the valuation date, amount per share).
    """
    v, re, t = (
        _finite("value_per_share", value_per_share),
        _finite("cost_of_equity", cost_of_equity),
        _finite("years", years),
    )
    if t < 0:
        raise ValueError("years must be non-negative")
    out = v * (1 + re) ** t
    for paid_at, amount in dividends:
        paid_at, amount = _finite("dividend time", paid_at), _finite("dividend", amount)
        if not 0 <= paid_at <= t:
            raise ValueError("dividends must be paid between the valuation date and the price date")
        out -= amount * (1 + re) ** (t - paid_at)
    return out


def implied_upside(value_per_share: float, market_price: float) -> float:
    """Value per share / market price - 1. Positive means the model says undervalued."""
    p = _finite("market_price", market_price)
    if p <= 0:
        raise ValueError("market_price must be positive")
    return _finite("value_per_share", value_per_share) / p - 1


# --------------------------------------------------------------------------- #
# Full DCF
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DCFResult:
    forecast: pd.DataFrame
    terminal_method: TerminalMethod
    terminal_value: float  # at the end of year N
    terminal_discount_period: float
    pv_terminal_value: float
    sum_pv_fcff: float
    enterprise_value: float
    equity_value: float
    value_per_share: float
    implied_exit_multiple: float  # EV/EBITDA implied by the terminal value
    implied_growth: float  # perpetual growth implied by the terminal value

    @property
    def terminal_share_of_ev(self) -> float:
        """Share of enterprise value that comes from the terminal value."""
        return self.pv_terminal_value / self.enterprise_value

    def summary(self) -> pd.Series:
        return pd.Series(
            {
                "Sum of PV of FCFF": self.sum_pv_fcff,
                "Terminal value (year N)": self.terminal_value,
                "PV of terminal value": self.pv_terminal_value,
                "Enterprise value": self.enterprise_value,
                "Equity value": self.equity_value,
                "Value per share": self.value_per_share,
                "Terminal value % of EV": self.terminal_share_of_ev,
                "Implied EV/EBITDA exit multiple": self.implied_exit_multiple,
                "Implied perpetual growth": self.implied_growth,
            }
        )


def run_dcf(
    assumptions: DCFAssumptions,
    bridge: EquityBridge,
    terminal_method: TerminalMethod = "gordon",
    exit_multiple: float | None = None,
) -> DCFResult:
    """Run the full DCF and return every intermediate number.

    With the mid-year convention, a Gordon terminal value is discounted N - 0.5
    years (it is built from mid-year cash flows) and an exit-multiple terminal
    value N years (the sale happens at year end). Note 05 explains why and
    notes that practitioners differ.
    """
    a = assumptions
    table = forecast_fcff(a)
    last = table.iloc[-1]
    n = a.n_years

    if terminal_method == "gordon":
        tv = gordon_terminal_value(last["fcff"], a.wacc, a.terminal_growth)
        tv_period = n - 0.5 if a.mid_year else float(n)
    elif terminal_method == "exit_multiple":
        if exit_multiple is None:
            raise ValueError("terminal_method='exit_multiple' needs exit_multiple")
        tv = exit_multiple_terminal_value(last["ebitda"], exit_multiple)
        tv_period = float(n)
    else:
        raise ValueError("terminal_method must be 'gordon' or 'exit_multiple'")

    pv_tv = tv / (1 + a.wacc) ** tv_period
    sum_pv = float(table["pv_fcff"].sum())
    ev = sum_pv + pv_tv
    equity, per_share = equity_bridge(ev, bridge)
    # Cross-check numbers are undefined (reported as NaN, never used in value) when the
    # final year's EBITDA or FCFF is not positive.
    multiple = implied_exit_multiple(tv, last["ebitda"]) if last["ebitda"] > 0 else float("nan")
    growth = implied_perpetual_growth(tv, last["fcff"], a.wacc) if tv + last["fcff"] > 0 else float("nan")
    return DCFResult(
        forecast=table,
        terminal_method=terminal_method,
        terminal_value=float(tv),
        terminal_discount_period=tv_period,
        pv_terminal_value=float(pv_tv),
        sum_pv_fcff=sum_pv,
        enterprise_value=float(ev),
        equity_value=float(equity),
        value_per_share=float(per_share),
        implied_exit_multiple=float(multiple),
        implied_growth=float(growth),
    )
