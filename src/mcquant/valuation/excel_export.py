"""Export the valuation to an Excel workbook a banker can audit and change.

Sheets: Inputs, WACC, Forecast, DCF, Sensitivity, Monte Carlo, Sources.

* The deterministic DCF and the sensitivity table are **live formulas**: change
  an input on the Inputs sheet and everything recalculates.
* Monte Carlo results are pasted as **values**: Excel cannot re-run 50,000
  correlated simulations with plain formulas, and RAND()-based formulas would
  change on every recalculation. The sheet says so.
* Colour convention used by banks: **blue** font = typed-in input,
  **black** = formula.

tests/test_excel_export.py recalculates the workbook in LibreOffice and checks
it matches Python to within PHP 0.01 per share.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .company import iter_sourced

if TYPE_CHECKING:  # pragma: no cover
    from .company import ValuationReport

BLUE = Font(color="1F4FBF")
BOLD = Font(bold=True)
TITLE = Font(bold=True, size=14)
HEAD = Font(bold=True, color="FFFFFF")
HEAD_FILL = PatternFill("solid", fgColor="2A78D6")
INPUT_FILL = PatternFill("solid", fgColor="FFF8E1")
TOTAL = Border(top=Side(style="thin"))
PCT, PCT1, MULT, PESO, MILLIONS = "0.00%", "0.0%", '0.00"x"', '"₱"#,##0.00', "#,##0;(#,##0)"
YEARS = "BCDEF"  # forecast columns


def _hdr(ws, row: int, labels: list[str], start_col: int = 1) -> None:
    for i, label in enumerate(labels):
        c = ws.cell(row=row, column=start_col + i, value=label)
        c.font, c.fill = HEAD, HEAD_FILL
        c.alignment = Alignment(horizontal="center")


def _input(ws, ref: str, value, fmt: str | None = None) -> None:
    ws[ref] = value
    ws[ref].font, ws[ref].fill = BLUE, INPUT_FILL
    if fmt:
        ws[ref].number_format = fmt


def _f(ws, ref: str, formula: str, fmt: str | None = None, bold: bool = False) -> None:
    ws[ref] = formula
    if fmt:
        ws[ref].number_format = fmt
    if bold:
        ws[ref].font = BOLD


def _widths(ws, widths: dict[str, float]) -> None:
    for col, w in widths.items():
        ws.column_dimensions[col].width = w


def export_workbook(report: ValuationReport, path):
    """Write the workbook to ``path`` (a file path or a writable binary buffer)."""
    setup = report.setup
    co, a, w = setup.company, setup.base_case, setup.wacc_inputs
    base_year = int(co.history.index[-1])
    years = [base_year + t for t in range(1, a.n_years + 1)]
    if a.n_years != 5:
        raise ValueError("the Excel template is laid out for a 5-year forecast")
    wb = Workbook()

    # ------------------------------------------------------------------ Inputs
    ws = wb.active
    ws.title = "Inputs"
    ws["A1"], ws["A1"].font = f"{co.name}: DCF inputs (₱ millions unless stated)", TITLE
    ws["A2"] = "Blue cells are inputs: change them and every sheet recalculates. Sources are on the Sources sheet."
    ws["A3"] = f"Base-year revenue (FY{base_year})"
    _input(ws, "B3", round(a.base_revenue, 6), MILLIONS)
    _hdr(ws, 5, ["Forecast year"] + [f"FY{y}" for y in years])
    rows = [
        ("Revenue growth", a.revenue_growth),
        ("EBITDA margin", a.ebitda_margin),
        ("D&A % of revenue", a.da_pct_revenue),
        ("Capex % of revenue", a.capex_pct_revenue),
        ("NWC % of revenue change", a.nwc_pct_revenue_change),
    ]
    for r, (label, vals) in enumerate(rows, start=6):
        ws[f"A{r}"] = label
        for col, v in zip(YEARS, vals):
            _input(ws, f"{col}{r}", float(v), PCT1)
    singles = [
        (12, "Tax rate", a.tax_rate, PCT),
        (13, "Terminal growth g", a.terminal_growth, PCT),
        (14, "Mid-year convention (1 = on, 0 = off)", int(a.mid_year), "0"),
        (16, "Risk-free rate", float(w.risk_free.mode), "0.000%"),
        (17, "Beta", float(w.beta.mode), "0.00"),
        (18, "Equity risk premium", float(w.equity_risk_premium.mode), PCT),
        (19, "Pre-tax cost of debt", w.pre_tax_cost_of_debt, "0.000%"),
        (20, "Share price (₱)", co.price, PESO),
        (21, "Diluted shares (millions)", co.shares_m, "#,##0.0"),
        (23, "Debt (loans, trust receipts, leases)", co.bridge.debt, MILLIONS),
        (24, "Debt-like items (dividends payable, pension)", co.bridge.debt_like_items, MILLIONS),
        (25, "Cash and equivalents", co.bridge.cash, MILLIONS),
        (26, "Non-operating assets", co.bridge.non_operating_assets, MILLIONS),
        (27, "Minority interest", co.bridge.minority_interest, MILLIONS),
        (29, "Years from valuation date to price date", setup.roll_years, "0.0000"),
        (30, "Dividend per share paid in between (₱)", setup.dividends[0][1] if setup.dividends else 0.0, PESO),
        (31, "Paid at (years after valuation date)", setup.dividends[0][0] if setup.dividends else 0.0, "0.0000"),
    ]
    for r, label, v, fmt in singles:
        ws[f"A{r}"] = label
        _input(ws, f"B{r}", float(v), fmt)
    ws["A15"], ws["A15"].font = "Discount rate", BOLD
    ws["A22"], ws["A22"].font = "Equity bridge (31 Dec 2025)", BOLD
    ws["A28"], ws["A28"].font = "Roll-forward to the price date", BOLD
    _widths(ws, {"A": 46, "B": 14, "C": 12, "D": 12, "E": 12, "F": 12})

    # ------------------------------------------------------------------ WACC
    wc = wb.create_sheet("WACC")
    wc["A1"], wc["A1"].font = "WACC (CAPM)", TITLE
    lines = [
        (3, "Cost of equity = Rf + beta x ERP", "=Inputs!B16+Inputs!B17*Inputs!B18", PCT),
        (4, "After-tax cost of debt = Rd x (1 - tax)", "=Inputs!B19*(1-Inputs!B12)", PCT),
        (5, "Equity value E = price x shares (₱m)", "=Inputs!B20*Inputs!B21", MILLIONS),
        (6, "Debt D (₱m)", "=Inputs!B23", MILLIONS),
        (7, "Equity weight E / (D + E)", "=B5/(B5+B6)", PCT1),
        (8, "Debt weight D / (D + E)", "=B6/(B5+B6)", PCT1),
        (9, "WACC", "=B7*B3+B8*B4", PCT),
    ]
    for r, label, formula, fmt in lines:
        wc[f"A{r}"] = label
        _f(wc, f"B{r}", formula, fmt, bold=(r == 9))
    _widths(wc, {"A": 44, "B": 14})

    # ------------------------------------------------------------------ Forecast
    fc = wb.create_sheet("Forecast")
    fc["A1"], fc["A1"].font = "Free cash flow to the firm (₱ millions)", TITLE
    _hdr(fc, 3, ["", *[f"FY{y}" for y in years]])
    labels = [
        "Year number",
        "Revenue",
        "EBITDA",
        "D&A",
        "EBIT",
        "NOPAT",
        "Capex",
        "Change in NWC",
        "FCFF",
        "Discount period",
        "Discount factor",
        "PV of FCFF",
    ]
    for i, label in enumerate(labels, start=4):
        fc[f"A{i}"] = label
    for k, col in enumerate(YEARS):
        prev = "Inputs!$B$3" if k == 0 else f"{YEARS[k - 1]}5"
        fc[f"{col}4"] = k + 1
        _f(fc, f"{col}5", f"={prev}*(1+Inputs!{col}6)", MILLIONS)
        _f(fc, f"{col}6", f"={col}5*Inputs!{col}7", MILLIONS)
        _f(fc, f"{col}7", f"={col}5*Inputs!{col}8", MILLIONS)
        _f(fc, f"{col}8", f"={col}6-{col}7", MILLIONS)
        _f(fc, f"{col}9", f"={col}8*(1-Inputs!$B$12)", MILLIONS)
        _f(fc, f"{col}10", f"={col}5*Inputs!{col}9", MILLIONS)
        _f(fc, f"{col}11", f"=({col}5-{prev})*Inputs!{col}10", MILLIONS)
        _f(fc, f"{col}12", f"={col}9+{col}7-{col}10-{col}11", MILLIONS, bold=True)
        _f(fc, f"{col}13", f"={col}4-0.5*Inputs!$B$14", "0.0")
        _f(fc, f"{col}14", f"=1/(1+WACC!$B$9)^{col}13", "0.0000")
        _f(fc, f"{col}15", f"={col}12*{col}14", MILLIONS)
    _widths(fc, {"A": 22, **{c: 13 for c in YEARS}})

    # ------------------------------------------------------------------ DCF
    d = wb.create_sheet("DCF")
    d["A1"], d["A1"].font = "DCF valuation", TITLE
    dcf_lines = [
        (3, "Sum of PV of FCFF", "=SUM(Forecast!B15:F15)", MILLIONS),
        (
            4,
            f"Terminal value at end of FY{years[-1]} = FCFF x (1 + g) / (WACC - g)",
            "=IF(WACC!B9-Inputs!B13<0.01,NA(),Forecast!F12*(1+Inputs!B13)/(WACC!B9-Inputs!B13))",
            MILLIONS,
        ),
        (5, "Terminal value discount period", "=5-0.5*Inputs!B14", "0.0"),
        (6, "PV of terminal value", "=B4/(1+WACC!B9)^B5", MILLIONS),
        (7, "Enterprise value", "=B3+B6", MILLIONS),
        (8, "Less: debt", "=-Inputs!B23", MILLIONS),
        (9, "Less: debt-like items", "=-Inputs!B24", MILLIONS),
        (10, "Plus: cash", "=Inputs!B25", MILLIONS),
        (11, "Plus: non-operating assets", "=Inputs!B26", MILLIONS),
        (12, "Less: minority interest", "=-Inputs!B27", MILLIONS),
        (13, "Equity value", "=SUM(B7:B12)", MILLIONS),
        (14, "Value per share at valuation date (₱)", "=B13/Inputs!B21", PESO),
        (
            15,
            "Value per share at price date (₱): rolled forward at cost of equity, less dividend",
            "=B14*(1+WACC!B3)^Inputs!B29-Inputs!B30*(1+WACC!B3)^(Inputs!B29-Inputs!B31)",
            PESO,
        ),
        (16, "Market price (₱)", "=Inputs!B20", PESO),
        (17, "Upside / (downside) vs market price", "=B15/B16-1", PCT1),
        (19, "Terminal value as % of EV", "=B6/B7", PCT1),
        (20, "Implied exit multiple (TV / final-year EBITDA)", "=B4/Forecast!F6", MULT),
    ]
    for r, label, formula, fmt in dcf_lines:
        d[f"A{r}"] = label
        _f(d, f"B{r}", formula, fmt, bold=r in (7, 13, 15))
    for r in (7, 13):
        d[f"B{r}"].border = TOTAL
    _widths(d, {"A": 70, "B": 16})

    # ------------------------------------------------------------------ Sensitivity
    s = wb.create_sheet("Sensitivity")
    s["A1"], s["A1"].font = "Value per share at the price date (₱): WACC (rows) x terminal growth (columns)", TITLE
    s["A2"] = (
        "Every cell is a live formula: the full DCF re-run at that WACC and g. 'n/m' = not meaningful (WACC - g < 1%)."
    )
    sens = report.sensitivity
    s["A4"], s["A4"].font = "WACC \\ g", BOLD
    for j, g in enumerate(sens.columns, start=2):
        _input(s, f"{get_column_letter(j)}4", float(g), PCT1)
    for i, wv in enumerate(sens.index, start=5):
        _input(s, f"A{i}", float(wv), PCT)
        for j in range(2, 2 + len(sens.columns)):
            col = get_column_letter(j)
            wacc_ref, g_ref = f"$A{i}", f"{col}$4"
            ev = (
                f"SUMPRODUCT(Forecast!$B$12:$F$12,(1+{wacc_ref})^(-Forecast!$B$13:$F$13))"
                f"+Forecast!$F$12*(1+{g_ref})/({wacc_ref}-{g_ref})*(1+{wacc_ref})^(-DCF!$B$5)"
            )
            per_share = f"(({ev})+SUM(DCF!$B$8:$B$12))/Inputs!$B$21"
            re = f"(({wacc_ref}-WACC!$B$8*WACC!$B$4)/WACC!$B$7)"
            rolled = f"{per_share}*(1+{re})^Inputs!$B$29-Inputs!$B$30*(1+{re})^(Inputs!$B$29-Inputs!$B$31)"
            _f(s, f"{col}{i}", f'=IF({wacc_ref}-{g_ref}<0.01,"n/m",{rolled})', PESO)
    _widths(s, {"A": 12, **{get_column_letter(j): 12 for j in range(2, 2 + len(sens.columns))}})

    # ------------------------------------------------------------------ Monte Carlo (values)
    m = wb.create_sheet("Monte Carlo")
    m["A1"], m["A1"].font = "Monte Carlo summary (pasted as values)", TITLE
    m["A2"] = (
        "These are results from the Python model (mcquant.valuation.montecarlo). They are values, not formulas: "
        "Excel cannot re-run 50,000 correlated simulations with plain formulas, "
        "and RAND() would change them on every recalculation."
    )
    m["A2"].alignment = Alignment(wrap_text=True)
    summary = report.mc.summary(co.price)
    _hdr(m, 4, ["Statistic", "Value"])
    for r, (k, v) in enumerate(summary.items(), start=5):
        m[f"A{r}"] = k
        m[f"B{r}"] = float(v)
        m[f"B{r}"].number_format = "0.00%" if k.startswith("P(") or k.startswith("std error of P") else "#,##0.00"
    start = 5 + len(summary) + 2
    m[f"A{start}"], m[f"A{start}"].font = "Tornado (one input at a time, 10th to 90th percentile)", BOLD
    _hdr(m, start + 1, ["Driver", "Low input", "High input", "Value at low", "Value at high", "Swing"])
    for r, row in enumerate(report.tornado.itertuples(), start=start + 2):
        m[f"A{r}"] = row.driver
        for col, v, fmt in (
            ("B", row._2, "0.000"),
            ("C", row._3, "0.000"),
            ("D", row._5, PESO),
            ("E", row._6, PESO),
            ("F", row.swing, PESO),
        ):
            m[f"{col}{r}"] = float(v)
            m[f"{col}{r}"].number_format = fmt
    _widths(m, {"A": 40, "B": 14, "C": 14, "D": 14, "E": 14, "F": 12})
    m.row_dimensions[2].height = 45

    # ------------------------------------------------------------------ Sources
    src = wb.create_sheet("Sources")
    src["A1"], src["A1"].font = "Every input fact and its public source", TITLE
    _hdr(src, 3, ["Item", "Value as reported", "Unit", "As of", "Source"])
    raw = {k: v for k, v in co.raw.items() if k not in ("company", "sources")}
    for r, (p, item) in enumerate(iter_sourced(raw), start=4):
        src[f"A{r}"], src[f"B{r}"], src[f"C{r}"] = p, item["value"], item.get("unit", "")
        src[f"D{r}"], src[f"E{r}"] = str(item.get("as_of", "")), item["source"]
    _widths(src, {"A": 52, "B": 18, "C": 16, "D": 12, "E": 120})

    for sheet in wb.worksheets:
        sheet.sheet_view.showGridLines = sheet.title not in ("Inputs", "WACC", "DCF")
    if hasattr(path, "write"):  # an in-memory buffer, e.g. for a download button
        wb.save(path)
        return path
    out = Path(path)
    wb.save(out)
    return out
