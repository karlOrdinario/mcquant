"""Discount rate: CAPM cost of equity, after-tax cost of debt, and WACC.

    Re   = Rf + beta x ERP
    WACC = E/(D+E) x Re + D/(D+E) x Rd x (1 - tax)
"""

from __future__ import annotations

from dataclasses import dataclass

from .inputs import _finite


def cost_of_equity_capm(risk_free: float, beta: float, equity_risk_premium: float) -> float:
    """CAPM: the return shareholders require. All inputs are decimals."""
    rf = _finite("risk_free", risk_free)
    b = _finite("beta", beta)
    erp = _finite("equity_risk_premium", equity_risk_premium)
    if not -0.05 < rf < 0.5:
        raise ValueError(f"risk_free {rf} looks wrong: pass decimals, e.g. 0.06 for 6%")
    if not 0 < erp < 0.5:
        raise ValueError(f"equity_risk_premium {erp} looks wrong: pass decimals, e.g. 0.06 for 6%")
    return rf + b * erp


def after_tax_cost_of_debt(pre_tax_cost_of_debt: float, tax_rate: float) -> float:
    """Interest is tax-deductible, so debt costs Rd x (1 - tax)."""
    rd = _finite("pre_tax_cost_of_debt", pre_tax_cost_of_debt)
    t = _finite("tax_rate", tax_rate)
    if not 0 <= rd < 0.5:
        raise ValueError(f"pre_tax_cost_of_debt {rd} looks wrong: pass decimals, e.g. 0.07 for 7%")
    if not 0 <= t < 1:
        raise ValueError("tax_rate must be in [0, 1)")
    return rd * (1 - t)


@dataclass(frozen=True)
class WACCResult:
    wacc: float
    cost_of_equity: float
    after_tax_cost_of_debt: float
    equity_weight: float
    debt_weight: float

    def __str__(self) -> str:
        return (
            f"WACC {self.wacc:.2%} = {self.equity_weight:.1%} x {self.cost_of_equity:.2%} (equity) "
            f"+ {self.debt_weight:.1%} x {self.after_tax_cost_of_debt:.2%} (debt after tax)"
        )


def wacc(
    equity_value: float,
    debt_value: float,
    cost_of_equity: float,
    pre_tax_cost_of_debt: float,
    tax_rate: float,
) -> WACCResult:
    """Blend the cost of equity and after-tax cost of debt by market-value weights.

    ``equity_value`` should be the market capitalisation (price x diluted shares),
    not book equity. See note 04 for why.
    """
    e = _finite("equity_value", equity_value)
    d = _finite("debt_value", debt_value)
    re = _finite("cost_of_equity", cost_of_equity)
    if e <= 0:
        raise ValueError("equity_value must be positive")
    if d < 0:
        raise ValueError("debt_value cannot be negative")
    rd_after = after_tax_cost_of_debt(pre_tax_cost_of_debt, tax_rate)
    we, wd = e / (e + d), d / (e + d)
    return WACCResult(we * re + wd * rd_after, re, rd_after, we, wd)


def wacc_from_capm(
    risk_free: float,
    beta: float,
    equity_risk_premium: float,
    pre_tax_cost_of_debt: float,
    tax_rate: float,
    equity_value: float,
    debt_value: float,
) -> WACCResult:
    """Convenience wrapper: CAPM cost of equity, then WACC."""
    re = cost_of_equity_capm(risk_free, beta, equity_risk_premium)
    return wacc(equity_value, debt_value, re, pre_tax_cost_of_debt, tax_rate)
