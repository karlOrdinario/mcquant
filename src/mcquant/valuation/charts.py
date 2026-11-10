"""Banker-style valuation charts (Matplotlib, for reports).

* :func:`histogram`   - distribution of value per share with the market price marked
* :func:`tornado`     - which input moves value most
* :func:`sensitivity_heatmap` - WACC x terminal growth table
* :func:`football_field` - every valuation method side by side

The dashboard draws the same charts with Plotly. Colours come from
:mod:`mcquant.viz`, so both match.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .. import viz


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz.use_matplotlib_style()
    return plt


def histogram(values: np.ndarray, market_price: float | None = None, title: str = "", currency: str = "₱"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    v = np.asarray(values)
    lo, hi = np.percentile(v, [0.2, 99.8])
    ax.hist(v, bins=np.linspace(lo, hi, 90), color=viz.BLUE, alpha=0.85)
    p5, p50, p95 = np.percentile(v, [5, 50, 95])
    ymax = ax.get_ylim()[1]
    ax.axvspan(p5, p95, color=viz.BLUE, alpha=0.07, lw=0)
    for x, label in ((p5, "P5"), (p50, "median"), (p95, "P95")):
        ax.axvline(x, color=viz.BLUE, lw=1, ls=":")
        ax.text(x, ymax * 0.97, f" {label}\n {currency}{x:,.2f}", color=viz.INK_2, fontsize=8.5, va="top")
    if market_price is not None:
        ax.axvline(market_price, color=viz.ORANGE, lw=2.2)
        p = float(np.mean(v > market_price))
        p_text = "> 99%" if p >= 0.995 else "< 1%" if p <= 0.005 else f"{p:.0%}"  # never show a rounded 100%
        ax.text(
            market_price,
            ymax * 0.55,
            f" market price {currency}{market_price:,.2f}\n P(value > price) {p_text}",
            color=viz.INK,
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=viz.AXIS, alpha=0.9),
        )
    ax.set(xlabel=f"Value per share ({currency})", ylabel="Simulations", title=title or "Monte Carlo value per share")
    ax.grid(axis="x", visible=False)
    return fig


def tornado(table: pd.DataFrame, title: str = "", currency: str = "₱"):
    """``table`` from :func:`mcquant.valuation.montecarlo.tornado`."""
    plt = _plt()
    t = table.iloc[::-1]  # biggest at the top
    base = float(t["base value"].iloc[0])
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(t) + 1.6))
    y = np.arange(len(t))
    lows, highs = t["value at low input"].to_numpy(), t["value at high input"].to_numpy()
    left, right = np.minimum(lows, highs), np.maximum(lows, highs)
    # Orange = value when the input is at its 10th percentile, blue = at its 90th.
    for i, (lo_v, hi_v) in enumerate(zip(lows, highs)):
        ax.barh(
            y[i],
            lo_v - base,
            left=base,
            color=viz.ORANGE,
            height=0.6,
            label="input at its 10th percentile" if i == 0 else None,
        )
        ax.barh(
            y[i],
            hi_v - base,
            left=base,
            color=viz.BLUE,
            height=0.6,
            label="input at its 90th percentile" if i == 0 else None,
        )
    for i, (l_, r_) in enumerate(zip(left, right)):
        ax.text(l_, y[i], f"{currency}{l_:,.2f} ", va="center", ha="right", fontsize=8.5, color=viz.INK_2)
        ax.text(r_, y[i], f" {currency}{r_:,.2f}", va="center", ha="left", fontsize=8.5, color=viz.INK_2)
    ax.axvline(base, color=viz.INK, lw=1.2)
    ax.set_yticks(y, t["driver"])
    ax.set_xlim(left.min() - (right.max() - left.min()) * 0.18, right.max() + (right.max() - left.min()) * 0.18)
    ax.set(
        xlabel=f"Value per share ({currency}), base case {currency}{base:,.2f}",
        title=title or "Tornado: which input moves value most",
    )
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5)
    return fig


def sensitivity_heatmap(
    table: pd.DataFrame,
    base_wacc: float | None = None,
    base_g: float | None = None,
    market_price: float | None = None,
    title: str = "",
    currency: str = "₱",
):
    """``table`` from :func:`mcquant.valuation.montecarlo.sensitivity_table`."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    data = table.to_numpy(float)
    fig, ax = plt.subplots(figsize=(1.15 * data.shape[1] + 2.2, 0.55 * data.shape[0] + 1.6))
    finite = data[np.isfinite(data)]
    centre = market_price if market_price is not None else float(np.median(finite))
    # Robust colour range: an exploding corner cell (g close to WACC) must not wash out the rest.
    lo_c, hi_c = np.percentile(finite, [5, 90])
    vmin, vmax = min(lo_c, centre - 1e-9), max(hi_c, centre + 1e-9)
    cmap = LinearSegmentedColormap.from_list("div", [viz.CRITICAL, "#f0efec", viz.BLUE])
    masked = np.ma.masked_invalid(data)
    ax.imshow(masked, cmap=cmap, norm=TwoSlopeNorm(vcenter=centre, vmin=vmin, vmax=vmax), aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            txt = "n/m" if not np.isfinite(data[i, j]) else f"{data[i, j]:,.2f}"
            bold = (
                base_wacc is not None
                and np.isclose(table.index[i], base_wacc)
                and base_g is not None
                and np.isclose(table.columns[j], base_g)
            )
            ax.text(
                j, i, txt, ha="center", va="center", fontsize=9, color=viz.INK, fontweight="bold" if bold else "normal"
            )
    ax.set_xticks(range(data.shape[1]), [f"{g:.1%}" for g in table.columns])
    ax.set_yticks(range(data.shape[0]), [f"{w:.1%}" for w in table.index])
    ax.set(
        xlabel="Terminal growth g",
        ylabel="WACC",
        title=title or f"Value per share ({currency}): WACC x terminal growth",
    )
    ax.grid(False)
    if market_price is not None:
        ax.text(
            1.0,
            -0.13,
            f"Blue: above the market price of {currency}{market_price:,.2f}. Red: below.",
            transform=ax.transAxes,
            ha="right",
            fontsize=8.5,
            color=viz.INK_2,
        )
    return fig


def football_field(rows: Sequence[dict], market_price: float | None = None, title: str = "", currency: str = "₱"):
    """``rows``: dicts with ``label``, ``low``, ``high`` and optional ``mid`` (marker)."""
    plt = _plt()
    rows = list(rows)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(rows) + 1.7))
    colors = [viz.BLUE, viz.AQUA, viz.YELLOW, viz.MAGENTA, viz.VIOLET, viz.GREEN]
    for i, r in enumerate(rows):
        c = colors[(len(rows) - 1 - i) % len(colors)]
        ax.barh(i, r["high"] - r["low"], left=r["low"], height=0.55, color=c, alpha=0.9)
        ax.text(r["low"], i, f"{currency}{r['low']:,.2f} ", va="center", ha="right", fontsize=8.5, color=viz.INK_2)
        ax.text(r["high"], i, f" {currency}{r['high']:,.2f}", va="center", ha="left", fontsize=8.5, color=viz.INK_2)
        if r.get("mid") is not None:
            ax.plot([r["mid"]] * 2, [i - 0.27, i + 0.27], color=viz.INK, lw=2)
    if market_price is not None:
        ax.axvline(market_price, color=viz.ORANGE, lw=2.2)
        ax.text(
            market_price,
            -0.75,
            f" market price {currency}{market_price:,.2f}",
            color=viz.INK,
            fontsize=9,
            fontweight="bold",
            va="center",
        )
        ax.set_ylim(-1.0, len(rows) - 0.4)
    lo = min(r["low"] for r in rows)
    hi = max(r["high"] for r in rows)
    ax.set_xlim(lo - (hi - lo) * 0.15, hi + (hi - lo) * 0.15)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows])
    ax.set(xlabel=f"Value per share ({currency})", title=title or "Football field")
    ax.grid(axis="y", visible=False)
    return fig
