"""Excel reconciliation (plan section 5.8): the workbook's live formulas,
recalculated by LibreOffice, must match Python to within PHP 0.01 per share."""

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from openpyxl import load_workbook

from mcquant.valuation.company import _roll, build_valuation, run_valuation, sensitivity_at_price_date
from mcquant.valuation.dcf import run_dcf
from mcquant.valuation.excel_export import export_workbook

SOFFICE = shutil.which("soffice") or shutil.which("libreoffice")
needs_lo = pytest.mark.skipif(SOFFICE is None, reason="LibreOffice is needed to recalculate formulas")


def recalc(path: Path, tmp: Path) -> Path:
    out = tmp / "recalc"
    subprocess.run(
        [
            SOFFICE,
            f"-env:UserInstallation=file://{tmp / 'lo_profile'}",
            "--headless",
            "--calc",
            "--convert-to",
            "xlsx:Calc MS Excel 2007 XML",
            "--outdir",
            str(out),
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
    return out / path.name


@pytest.fixture(scope="module")
def report():
    return run_valuation(build_valuation(), n_sims=5_000)


@pytest.fixture(scope="module")
def recalculated(report, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("xl")
    path = export_workbook(report, tmp / "model.xlsx")
    return load_workbook(recalc(path, tmp), data_only=True), tmp, path


def test_workbook_structure(report, tmp_path):
    wb = load_workbook(export_workbook(report, tmp_path / "m.xlsx"))
    assert wb.sheetnames == ["Inputs", "WACC", "Forecast", "DCF", "Sensitivity", "Monte Carlo", "Sources"]
    assert wb["DCF"]["B14"].value.startswith("=")  # a formula, not a number
    assert wb["Inputs"]["B7"].font.color.rgb.endswith("1F4FBF")  # inputs in blue
    assert isinstance(wb["Monte Carlo"]["B7"].value, float)  # Monte Carlo pasted as values
    assert wb["Sources"].max_row > 60


@needs_lo
def test_excel_dcf_matches_python_to_the_centavo(report, recalculated):
    wb, _, _ = recalculated
    d = wb["DCF"]
    assert d["B7"].value == pytest.approx(report.base_dcf.enterprise_value, rel=1e-9)
    assert abs(d["B14"].value - report.base_dcf.value_per_share) < 0.01
    assert abs(d["B15"].value - report.base_value_at_price_date) < 0.01
    assert wb["WACC"]["B9"].value == pytest.approx(report.base_wacc.wacc, rel=1e-9)
    fcff_xl = [wb["Forecast"][f"{c}12"].value for c in "BCDEF"]
    np.testing.assert_allclose(fcff_xl, report.base_dcf.forecast["fcff"], rtol=1e-9)


@needs_lo
def test_excel_sensitivity_matches_python(report, recalculated):
    wb, _, _ = recalculated
    s = wb["Sensitivity"]
    sens = report.sensitivity
    for i, w in enumerate(sens.index, start=5):
        for j, g in enumerate(sens.columns, start=2):
            xl, py = s.cell(i, j).value, sens.loc[w, g]
            if np.isnan(py):
                assert xl == "n/m"
            else:
                assert abs(xl - py) < 0.01, (w, g, xl, py)


@needs_lo
def test_changing_an_input_in_excel_recalculates(report, recalculated):
    """Proves the formulas are live: raise the 2026 EBITDA margin in the
    spreadsheet and compare with Python run on the same change."""
    _, tmp, path = recalculated
    wb = load_workbook(path)
    wb["Inputs"]["B7"] = 0.21
    edited = tmp / "edited.xlsx"
    wb.save(edited)
    out = load_workbook(recalc(edited, tmp), data_only=True)
    setup = report.setup
    margins = list(setup.base_case.ebitda_margin)
    margins[0] = 0.21
    v = run_dcf(replace(setup.base_case, ebitda_margin=margins), setup.company.bridge).value_per_share
    assert abs(out["DCF"]["B14"].value - v) < 0.01
    assert abs(out["DCF"]["B15"].value - _roll(setup, v, setup.base_cost_of_equity)) < 0.01


def test_sensitivity_helper_flags_invalid_cells(report):
    s = sensitivity_at_price_date(report.setup, [0.06, 0.10], [0.055, 0.04])
    assert np.isnan(s.loc[0.06, 0.055])
    assert np.isfinite(s.loc[0.10, 0.04])
