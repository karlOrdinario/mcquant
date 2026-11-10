"""Validated inputs for the DCF.

Two groups of inputs, kept separate on purpose:

* :class:`DCFAssumptions` - the forecast drivers and discount rate (judgement calls).
* :class:`EquityBridge` - balance-sheet items that convert enterprise value to
  value per share (facts from the annual report).

Every field is checked on construction. Bad inputs raise ``ValueError`` instead
of quietly producing NaN. Loading these from cited YAML files arrives in
milestone 3.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

MIN_WACC_SPREAD = 0.01  # WACC - g must be at least 1 percentage point (plan section 5.3)
_SPREAD_TOL = 1e-12  # floating point: 0.125 - 0.115 = 0.00999...; exactly 1 point must still pass


def spread_too_small(wacc: float, growth: float, min_spread: float = MIN_WACC_SPREAD) -> bool:
    """True if WACC - g is below the minimum (with a tolerance for float round-off)."""
    return wacc - growth < min_spread - _SPREAD_TOL


def _finite(name: str, value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be a number, got {value!r}") from e
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _per_year(name: str, value: float | Sequence[float], n_years: int) -> tuple[float, ...]:
    """Accept one number (same every year) or one number per forecast year."""
    if isinstance(value, (int, float)):
        return tuple(_finite(name, value) for _ in range(n_years))
    values = tuple(_finite(name, v) for v in value)
    if len(values) != n_years:
        raise ValueError(f"{name} needs {n_years} values (one per forecast year), got {len(values)}")
    return values


@dataclass(frozen=True)
class DCFAssumptions:
    """Forecast drivers for an FCFF DCF.

    All rates are decimals (10% = 0.10). Per-year drivers accept a single
    number or a sequence with one value per forecast year.

    Attributes
    ----------
    base_revenue:
        Revenue in the last actual year (year 0).
    revenue_growth:
        Growth for each forecast year; its length sets the forecast horizon.
    ebitda_margin, da_pct_revenue, capex_pct_revenue:
        EBITDA, D&A and capex as a share of the same year's revenue.
    nwc_pct_revenue_change:
        Change in net working capital as a share of the change in revenue.
        Can be negative for businesses whose suppliers fund their inventory.
    tax_rate:
        Cash tax rate on EBIT.
    wacc:
        Discount rate.
    terminal_growth:
        Perpetual growth rate after the forecast (Gordon growth).
    mid_year:
        Discount cash flows at mid-year (0.5, 1.5, ...). Off by default.
    """

    base_revenue: float
    revenue_growth: Sequence[float]
    ebitda_margin: float | Sequence[float]
    da_pct_revenue: float | Sequence[float]
    capex_pct_revenue: float | Sequence[float]
    nwc_pct_revenue_change: float | Sequence[float]
    tax_rate: float
    wacc: float
    terminal_growth: float
    mid_year: bool = False
    n_years: int = field(init=False)

    def __post_init__(self) -> None:
        growth = tuple(_finite("revenue_growth", g) for g in self.revenue_growth)
        if not growth:
            raise ValueError("revenue_growth needs at least one forecast year")
        n = len(growth)
        set_ = object.__setattr__  # frozen dataclass: normalise fields once
        set_(self, "n_years", n)
        set_(self, "revenue_growth", growth)
        set_(self, "base_revenue", _finite("base_revenue", self.base_revenue))
        for name in ("ebitda_margin", "da_pct_revenue", "capex_pct_revenue", "nwc_pct_revenue_change"):
            set_(self, name, _per_year(name, getattr(self, name), n))
        for name in ("tax_rate", "wacc", "terminal_growth"):
            set_(self, name, _finite(name, getattr(self, name)))

        if self.base_revenue <= 0:
            raise ValueError("base_revenue must be positive")
        if any(g <= -1 for g in self.revenue_growth):
            raise ValueError("revenue_growth must be above -100%")
        if any(not -1 < m <= 1 for m in self.ebitda_margin):
            raise ValueError("ebitda_margin must be between -100% and 100%")
        for name in ("da_pct_revenue", "capex_pct_revenue"):
            if any(not 0 <= v < 1 for v in getattr(self, name)):
                raise ValueError(f"{name} must be between 0% and 100%")
        if any(not -1 < v < 1 for v in self.nwc_pct_revenue_change):
            raise ValueError("nwc_pct_revenue_change must be between -100% and 100%")
        if not 0 <= self.tax_rate < 1:
            raise ValueError("tax_rate must be in [0, 1)")
        if not 0 < self.wacc < 1:
            raise ValueError("wacc must be between 0% and 100%")
        if spread_too_small(self.wacc, self.terminal_growth):
            raise ValueError(
                f"WACC - g = {self.wacc - self.terminal_growth:.4f} is below the {MIN_WACC_SPREAD:.0%} minimum: "
                "the terminal value would explode"
            )


@dataclass(frozen=True)
class EquityBridge:
    """Items that turn enterprise value into value per share (note 06).

    ``debt_like_items`` are claims that behave like debt but are not loans,
    such as dividends declared but unpaid, or pension deficits.
    ``non_operating_assets`` are assets whose income is *not* in FCFF, such as
    short-term investments or stakes in associates, so the DCF has not valued
    them yet.
    """

    debt: float
    cash: float
    diluted_shares: float
    minority_interest: float = 0.0
    debt_like_items: float = 0.0
    non_operating_assets: float = 0.0

    def __post_init__(self) -> None:
        names = ("debt", "cash", "diluted_shares", "minority_interest", "debt_like_items", "non_operating_assets")
        for name in names:
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        for name in names:
            if name != "diluted_shares" and getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.diluted_shares <= 0:
            raise ValueError("diluted_shares must be positive")

    @property
    def net_adjustment(self) -> float:
        """Amount added to EV to reach equity value (negative means a net claim)."""
        return -self.debt - self.debt_like_items + self.cash + self.non_operating_assets - self.minority_interest
