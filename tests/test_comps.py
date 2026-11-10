"""Trading comps: hand-checked statistics."""

import pandas as pd
import pytest

from mcquant.valuation.comps import (
    comps_valuation,
    implied_per_share_from_ev_ebitda,
    implied_per_share_from_pe,
    multiple_stats,
)
from mcquant.valuation.inputs import EquityBridge

# The six peers' EV/EBITDA multiples from config/monde.yaml.
EV_EBITDA = [6.44, 11.30, 6.22, 5.08, 10.02, 9.14]


def test_multiple_stats_by_hand():
    # Sorted: 5.08, 6.22, 6.44, 9.14, 10.02, 11.30
    # Median = (6.44 + 9.14) / 2 = 7.79
    # Q1 sits at position 0.25 x 5 = 1.25: 6.22 + 0.25 x (6.44 - 6.22) = 6.275
    # Q3 sits at position 0.75 x 5 = 3.75: 9.14 + 0.75 x (10.02 - 9.14) = 9.80
    s = multiple_stats(EV_EBITDA)
    assert s["median"] == pytest.approx(7.79)
    assert s["q1"] == pytest.approx(6.275)
    assert s["q3"] == pytest.approx(9.80)
    assert s["mean"] == pytest.approx(8.0333333, abs=1e-6)  # differs from the median
    assert s["n"] == 6


def test_median_resists_an_outlier_but_the_mean_does_not():
    s = multiple_stats([7.0, 8.0, 9.0, 60.0])
    assert s["median"] == pytest.approx(8.5)
    assert s["mean"] == pytest.approx(21.0)


def test_implied_values_by_hand():
    bridge = EquityBridge(debt=200, cash=100, diluted_shares=100)
    # EV = 8 x 50 = 400; equity = 400 - 200 + 100 = 300; per share = 3.00
    assert implied_per_share_from_ev_ebitda(8, 50, bridge) == pytest.approx(3.0)
    assert implied_per_share_from_pe(12, 0.5) == pytest.approx(6.0)


def test_comps_table_uses_quartiles_and_median():
    peers = pd.DataFrame({"ev_ebitda": EV_EBITDA, "pe": [11.53, 15.52, 10.49, 9.01, 18.67, 16.83]})
    bridge = EquityBridge(debt=0, cash=0, diluted_shares=10)
    t = comps_valuation(peers, ebitda=10, eps=1.0, bridge=bridge)
    assert t.loc["EV/EBITDA", "value median"] == pytest.approx(7.79)  # 7.79 x 10 / 10 shares
    assert t.loc["P/E", "multiple median"] == pytest.approx((11.53 + 15.52) / 2)
    assert t.loc["EV/EBITDA", "value q1"] < t.loc["EV/EBITDA", "value median"] < t.loc["EV/EBITDA", "value q3"]


@pytest.mark.parametrize("bad", [[7.0, 8.0], [7.0, -1.0, 9.0]])
def test_bad_peer_sets_are_rejected(bad):
    with pytest.raises(ValueError):
        multiple_stats(bad)


def test_negative_earnings_are_rejected():
    with pytest.raises(ValueError):
        implied_per_share_from_pe(12, -0.1)
    with pytest.raises(ValueError):
        implied_per_share_from_ev_ebitda(8, 0, EquityBridge(debt=0, cash=0, diluted_shares=1))
