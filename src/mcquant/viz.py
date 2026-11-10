"""Shared chart styling for the dashboard (Plotly) and the README figures (Matplotlib).

One palette, defined once. Categorical colours are assigned in a fixed order
and always follow the entity (e.g. "Monte Carlo (Student-t)" is always the
same colour). Red is reserved for status (VaR breaches) and never used as a
series colour.
"""

from __future__ import annotations

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = SERIES[:7]
BLUE_LIGHT = "#86b6ef"  # sequential step 250
CRITICAL = "#d03b3b"  # status: breach / failure
GOOD = "#0ca30c"  # status: pass
DIVERGING = [[0.0, "#d03b3b"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]  # -1 red .. 0 gray .. +1 blue
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"

# Fixed colour per sampling method / estimator so every chart agrees.
METHOD_COLORS = {
    "pseudo": BLUE,
    "antithetic": ORANGE,
    "sobol": AQUA,
    "pseudo+cv": YELLOW,
    "antithetic+cv": MAGENTA,
    "sobol+cv": VIOLET,
}


def plotly_template():
    """A Plotly layout template matching the palette (import-light)."""
    import plotly.graph_objects as go

    axis = dict(
        gridcolor=GRID,
        gridwidth=1,
        zeroline=False,
        linecolor=AXIS,
        showline=True,
        tickfont=dict(color=MUTED, size=12),
        title=dict(font=dict(color=INK_2, size=13)),
    )
    return go.layout.Template(
        layout=dict(
            font=dict(family=FONT, color=INK, size=13),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=SURFACE,
            colorway=SERIES,
            xaxis=axis,
            yaxis=axis,
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(color=INK_2)),
            margin=dict(l=10, r=10, t=78, b=10),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(color=INK, family=FONT)),
            title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", y=0.98, yanchor="top", yref="container"),
            bargap=0.04,
        )
    )


def use_matplotlib_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "font.family": "sans-serif",
            "font.size": 11,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )
