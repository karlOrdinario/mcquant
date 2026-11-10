"""Trading comparables: value a company from what similar companies trade at.

    Implied EV           = peer EV/EBITDA x company EBITDA      -> bridge to equity -> per share
    Implied price        = peer P/E x company EPS

We use the peer **median** (robust to one odd peer) and the **interquartile
range** (25th to 75th percentile) as the range on the football field.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .dcf import equity_bridge
from .inputs import EquityBridge, _finite


def multiple_stats(values: Sequence[float]) -> dict[str, float]:
    v = np.asarray([_finite("multiple", x) for x in values], float)
    if v.size < 3:
        raise ValueError("need at least 3 peers for a meaningful median and range")
    if (v <= 0).any():
        raise ValueError("multiples must be positive (exclude loss-making peers from P/E)")
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {
        "min": float(v.min()),
        "q1": float(q1),
        "median": float(med),
        "mean": float(v.mean()),
        "q3": float(q3),
        "max": float(v.max()),
        "n": int(v.size),
    }


def implied_per_share_from_ev_ebitda(multiple: float, ebitda: float, bridge: EquityBridge) -> float:
    if ebitda <= 0:
        raise ValueError("EV/EBITDA needs positive EBITDA")
    return equity_bridge(multiple * ebitda, bridge)[1]


def implied_per_share_from_pe(multiple: float, eps: float) -> float:
    if eps <= 0:
        raise ValueError("P/E needs positive earnings per share")
    return multiple * eps


def comps_valuation(peers: pd.DataFrame, ebitda: float, eps: float, bridge: EquityBridge) -> pd.DataFrame:
    """Implied value per share at the peer Q1, median and Q3, for EV/EBITDA and P/E."""
    ev = multiple_stats(peers["ev_ebitda"])
    pe = multiple_stats(peers["pe"])
    rows = []
    for label, stats, fn in (
        ("EV/EBITDA", ev, lambda m: implied_per_share_from_ev_ebitda(m, ebitda, bridge)),
        ("P/E", pe, lambda m: implied_per_share_from_pe(m, eps)),
    ):
        rows.append(
            {
                "method": label,
                "peers": stats["n"],
                "multiple q1": stats["q1"],
                "multiple median": stats["median"],
                "multiple q3": stats["q3"],
                "value q1": fn(stats["q1"]),
                "value median": fn(stats["median"]),
                "value q3": fn(stats["q3"]),
            }
        )
    return pd.DataFrame(rows).set_index("method")
