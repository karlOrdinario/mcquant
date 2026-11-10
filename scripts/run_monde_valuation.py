"""Run the Monde Nissin valuation and write the report, figures and Excel model.

    python scripts/run_monde_valuation.py

Outputs:
    docs/MONDE_VALUATION.md            results report (numbers come only from this run)
    docs/monde_results.json            the same numbers, machine-readable
    docs/figures/valuation_*.png       histogram, tornado, sensitivity, football field, history
    output/Monde_Nissin_DCF.xlsx       Excel model with live formulas
"""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mcquant import viz  # noqa: E402
from mcquant.valuation import charts  # noqa: E402
from mcquant.valuation.company import build_valuation, implied_market_expectations, run_valuation  # noqa: E402

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def peso(x: float, d: int = 2) -> str:
    return f"₱{x:,.{d}f}"


def bn(x_m: float) -> str:
    return f"₱{x_m / 1_000:,.1f}bn"


def make_figures(report) -> None:
    co = report.setup.company
    charts.histogram(
        report.mc.value_per_share, co.price, f"Monde Nissin: {report.mc.n_requested:,} simulated values per share"
    ).savefig(FIG / "valuation_histogram.png")
    charts.tornado(report.tornado, "Monde Nissin: which input moves value most").savefig(FIG / "valuation_tornado.png")
    base = report.setup.base_case
    charts.sensitivity_heatmap(
        report.sensitivity,
        base.wacc,
        base.terminal_growth,
        co.price,
        "Monde Nissin value per share (₱): WACC x terminal growth",
    ).savefig(FIG / "valuation_sensitivity.png")
    charts.football_field(report.football, co.price, "Monde Nissin football field (₱ per share)").savefig(
        FIG / "valuation_football_field.png"
    )

    viz.use_matplotlib_style()
    h = co.history
    fc = report.base_dcf.forecast
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    years_h = list(h.index)
    years_f = [years_h[-1] + int(i) for i in fc.index]
    axes[0].bar(years_h, h.net_sales / 1e3, color=viz.BLUE, label="Actual")
    axes[0].bar(years_f, fc["revenue"] / 1e3, color=viz.BLUE_LIGHT, label="Forecast (base case)")
    axes[0].set(title="Revenue (₱bn)", xticks=years_h + years_f)
    axes[0].tick_params(axis="x", labelrotation=45)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="x", visible=False)
    margin_h = h.core_ebitda / h.net_sales * 100
    axes[1].plot(years_h, margin_h, marker="o", color=viz.BLUE, label="Actual core EBITDA margin")
    axes[1].plot(
        [years_h[-1]] + years_f,
        [margin_h.iloc[-1]] + [base.ebitda_margin[0] * 100] * len(years_f),
        ls="--",
        marker="o",
        color=viz.ORANGE,
        label="Forecast (most likely)",
    )
    m = report.setup.mc_inputs.ebitda_margin
    axes[1].fill_between(
        years_f,
        [m.low * 100] * len(years_f),
        [m.high * 100] * len(years_f),
        color=viz.ORANGE,
        alpha=0.12,
        label="Monte Carlo range",
    )
    axes[1].set(title="Core EBITDA margin (%)", xticks=years_h + years_f, ylim=(14, 23))
    axes[1].tick_params(axis="x", labelrotation=45)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "valuation_history.png")
    plt.close("all")


def results_dict(report, implied) -> dict:
    co = report.setup.company
    s = report.mc.summary(co.price)
    d = report.base_dcf
    return {
        "headline": report.headline,
        "price": co.price,
        "price_date": str(report.setup.assumptions_raw["price_date"]),
        "valuation_date": str(report.setup.assumptions_raw["valuation_date"]),
        "monte_carlo": {k: float(v) for k, v in s.items()},
        "base_case": {
            "wacc": report.base_wacc.wacc,
            "cost_of_equity": report.base_wacc.cost_of_equity,
            "enterprise_value_m": d.enterprise_value,
            "equity_value_m": d.equity_value,
            "value_per_share_valuation_date": d.value_per_share,
            "value_per_share_price_date": report.base_value_at_price_date,
            "terminal_share_of_ev": d.terminal_share_of_ev,
            "implied_exit_multiple": d.implied_exit_multiple,
        },
        "tornado": report.tornado.to_dict("records"),
        "comps": report.comps.reset_index().to_dict("records"),
        "football": report.football,
        "implied_by_price": implied,
        "market_ev_ebitda": (co.market_cap - co.bridge.net_adjustment) / co.history.core_ebitda.iloc[-1],
    }


def write_report(report, implied) -> str:
    co = report.setup.company
    s = report.mc.summary(co.price)
    d = report.base_dcf
    w = report.base_wacc
    wi = report.setup.wacc_inputs
    a = report.setup.base_case
    fc = d.forecast
    base_year = int(co.history.index[-1])
    market_ev = co.market_cap - co.bridge.net_adjustment
    lines = [
        "# Monde Nissin valuation: results",
        "",
        "Generated by `scripts/run_monde_valuation.py`. Every number below comes from that run. Inputs and sources: [ASSUMPTIONS.md](ASSUMPTIONS.md).",
        "",
        "> **Not investment advice.** This is a learning project built on public information and stated assumptions.",
        "",
        "## Headline",
        "",
        f"> {report.headline}",
        "",
        "## Monte Carlo summary",
        "",
        f"Value per share at the price date ({report.setup.assumptions_raw['price_date']}), {report.mc.n_requested:,} simulations, seed {report.setup.seed}.",
        "",
        "| Statistic | Value |",
        "|---|---|",
        f"| P5 | {peso(s['P5'])} |",
        f"| P25 | {peso(s['P25'])} |",
        f"| Median | {peso(s['median (P50)'])} |",
        f"| Mean | {peso(s['mean'])} (standard error {peso(s['std error of mean'], 3)}) |",
        f"| P75 | {peso(s['P75'])} |",
        f"| P95 | {peso(s['P95'])} |",
        f"| Most-likely-inputs DCF | {peso(report.base_value_at_price_date)} |",
        f"| Market price | {peso(co.price)} |",
        f"| P(value > price) | {s['P(value > price)']:.2%} |",
        f"| Discarded runs (WACC - g < 1%) | {int(s['discarded (WACC - g < 1%)'])} |",
        "",
        "![Histogram](figures/valuation_histogram.png)",
        "",
        "## Base case (every input at its most likely value)",
        "",
        "### WACC",
        "",
        "| Component | Value |",
        "|---|---|",
        f"| Risk-free rate (10-year yield 8.134% - default spread 1.62%) | {wi.risk_free.mode:.3%} |",
        f"| Beta (bottom-up) | {wi.beta.mode:.2f} |",
        f"| Equity risk premium (Damodaran, Philippines) | {wi.equity_risk_premium.mode:.2%} |",
        f"| Cost of equity | {w.cost_of_equity:.2%} |",
        f"| Pre-tax cost of debt | {wi.pre_tax_cost_of_debt:.3%} |",
        f"| After-tax cost of debt | {w.after_tax_cost_of_debt:.2%} |",
        f"| Equity weight | {w.equity_weight:.1%} |",
        f"| **WACC** | **{w.wacc:.2%}** |",
        "",
        "### Forecast (₱ millions)",
        "",
        "| Year | Revenue | Growth | EBITDA | EBIT | NOPAT | Capex | Change NWC | FCFF | PV of FCFF |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t, r in fc.iterrows():
        lines.append(
            f"| {base_year + int(t)} | {r.revenue:,.0f} | {r.growth:.1%} | {r.ebitda:,.0f} | {r.ebit:,.0f} | {r.nopat:,.0f} | "
            f"{r.capex:,.0f} | {r.change_nwc:,.0f} | {r.fcff:,.0f} | {r.pv_fcff:,.0f} |"
        )
    br = co.bridge
    lines += [
        "",
        "### From enterprise value to value per share (₱ millions)",
        "",
        "| Step | ₱ millions |",
        "|---|---|",
        f"| Sum of PV of FCFF ({base_year + 1} to {base_year + a.n_years}) | {d.sum_pv_fcff:,.0f} |",
        f"| PV of terminal value (g = {a.terminal_growth:.1%}) | {d.pv_terminal_value:,.0f} |",
        f"| **Enterprise value** | **{d.enterprise_value:,.0f}** |",
        f"| Less debt (loans, trust receipts, leases) | ({br.debt:,.0f}) |",
        f"| Less debt-like items (dividends payable, pension) | ({br.debt_like_items:,.0f}) |",
        f"| Plus cash | {br.cash:,.0f} |",
        f"| Plus non-operating assets (financial assets, guaranty asset, associates) | {br.non_operating_assets:,.0f} |",
        f"| Less minority interest | ({br.minority_interest:,.0f}) |",
        f"| **Equity value** | **{d.equity_value:,.0f}** |",
        f"| Diluted shares (millions) | {br.diluted_shares:,.1f} |",
        f"| Value per share at {report.setup.assumptions_raw['valuation_date']} | {peso(d.value_per_share)} |",
        f"| Rolled forward to {report.setup.assumptions_raw['price_date']} (cost of equity {w.cost_of_equity:.2%}, less ₱0.24 dividend) | **{peso(report.base_value_at_price_date)}** |",
        "",
        f"The terminal value is **{d.terminal_share_of_ev:.0%}** of enterprise value. The Gordon terminal value implies an exit multiple of **{d.implied_exit_multiple:.1f}x** EBITDA, against a peer median of {report.comps.loc['EV/EBITDA', 'multiple median']:.1f}x.",
        "",
        "## What drives the value",
        "",
        "![Tornado](figures/valuation_tornado.png)",
        "",
        "| Input | 10th to 90th percentile | Value at 10th | Value at 90th | Swing |",
        "|---|---|---|---|---|",
    ]

    def fmt_input(x: float, unit: str) -> str:
        if unit == "x":
            return f"{x:.2f}"
        return f"{x:+.1%}" if unit == "pp shift" else f"{x:.1%}"

    for _, r in report.tornado.iterrows():
        fmt = partial(fmt_input, unit=r.unit)
        lines.append(
            f"| {r.driver} | {fmt(r['low input'])} to {fmt(r['high input'])} | {peso(r['value at low input'])} | "
            f"{peso(r['value at high input'])} | {peso(r.swing)} |"
        )
    lines += [
        "",
        "![Sensitivity](figures/valuation_sensitivity.png)",
        "",
        "## What today's price implies (reverse DCF)",
        "",
        f"Holding every other input at its most likely value, the ₱{co.price:.2f} price is consistent with:",
        "",
    ]
    if "implied_wacc" in implied:
        lines.append(f"- a WACC of **{implied['implied_wacc']:.1%}** (base case {a.wacc:.1%}), or")
    if "implied_ebitda_margin" in implied:
        lines.append(
            f"- a permanent EBITDA margin of **{implied['implied_ebitda_margin']:.1%}** (2023 to 2025 range: 16.9% to 19.3%), or"
        )
    if "implied_terminal_growth" in implied:
        lines.append(
            f"- perpetual growth of **{implied['implied_terminal_growth']:.1%}** a year after 2030 (base case {a.terminal_growth:.1%})."
        )
    lines += [
        "",
        f"The market values Monde at about **{market_ev / co.history.core_ebitda.iloc[-1]:.1f}x** {base_year} core EBITDA "
        f"(market EV {bn(market_ev)}), below the peer median of {report.comps.loc['EV/EBITDA', 'multiple median']:.1f}x. "
        "Either the market expects much worse than our base case (a weaker margin, Quorn risk, higher Philippine rates), or it is too pessimistic. "
        "The Monte Carlo result says: **under our stated assumptions**, the second is likely. The reverse DCF says how wrong we would have to be.",
        "",
        "## Cross-checks: comparable companies and the football field",
        "",
        "| Method | Peer Q1 | Peer median | Peer Q3 | Value at Q1 | Value at median | Value at Q3 |",
        "|---|---|---|---|---|---|---|",
    ]
    for meth, r in report.comps.iterrows():
        lines.append(
            f"| {meth} | {r['multiple q1']:.1f}x | {r['multiple median']:.1f}x | {r['multiple q3']:.1f}x | "
            f"{peso(r['value q1'])} | {peso(r['value median'])} | {peso(r['value q3'])} |"
        )
    lines += [
        "",
        f"EV/EBITDA uses {base_year} core EBITDA ({bn(co.history.core_ebitda.iloc[-1])}) and the same equity bridge. P/E uses {base_year} core EPS (₱{co.core_eps:.2f}).",
        "",
        "![Football field](figures/valuation_football_field.png)",
        "",
        "| Method | Low | High |",
        "|---|---|---|",
    ]
    lines += [f"| {r['label']} | {peso(r['low'])} | {peso(r['high'])} |" for r in report.football]
    lines += [
        "",
        "## Limitations",
        "",
        "- **One consolidated DCF.** APAC branded food (about 84% of sales) and Quorn (meat alternatives, recently loss-making) have very different risks. A sum-of-the-parts valuation would value them separately.",
        "- **Constant tax rate and no explicit debt schedule.** Reasonable for a company with almost no debt.",
        "- **Data dates differ.** Financials are as of 31 December 2025, the price is 6 November 2026, and peer multiples are from September to November 2026.",
        "- **The guaranty asset** (₱0.43 per share) is included at its carrying value, which can change.",
        "- **The Monte Carlo probability is conditional on the input ranges.** It measures uncertainty inside the model, not the chance the model is wrong.",
        "- **Not investment advice.**",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    setup = build_valuation()
    report = run_valuation(setup)
    implied = implied_market_expectations(setup)
    make_figures(report)
    (ROOT / "docs" / "MONDE_VALUATION.md").write_text(write_report(report, implied), encoding="utf-8")
    (ROOT / "docs" / "monde_results.json").write_text(
        json.dumps(results_dict(report, implied), indent=2, default=float), encoding="utf-8"
    )
    try:
        from mcquant.valuation.excel_export import export_workbook

        out = ROOT / "output"
        out.mkdir(exist_ok=True)
        export_workbook(report, out / "Monde_Nissin_DCF.xlsx")
        print("wrote output/Monde_Nissin_DCF.xlsx")
    except ImportError:
        pass
    print(report.headline)
    print({k: round(v, 4) for k, v in implied.items()})
    print("P(value > price) exactly:", report.p_above_price, "| median", np.median(report.mc.value_per_share))


if __name__ == "__main__":
    main()
