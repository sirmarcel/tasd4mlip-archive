"""Plotting and table toolbox: the shared look of every figure in the write-ups.

Holds the style, the palette, and the save helpers; figure scripts live next to
the experiment they plot and import the names they use:

    from sadmof.tbx import fig_and_ax, model_colors, savefig

Any import from here applies `plots.mplstyle`, so a figure script does not have
to. `matplotlib` is an optional dependency (`uv sync --extra plots`); nothing
else in the package imports this one.
"""

import numpy as np  # noqa: F401 — re-exported for figure scripts

import os
from pathlib import Path

import matplotlib.pyplot as plt

from .literals import *  # noqa: F403
from .literals import __all__ as _literals_all

__all__ = [
    "plt",
    "np",
    "Path",
    "savefig",
    "savefile",
    "fig_and_ax",
    "major_ticks_every",
    "minor_ticks_every",
    "reversed_legend",
    "no_ticks",
    "STYLE",
    *_literals_all,
]

STYLE = Path(__file__).parent / "plots.mplstyle"

plt.style.use(STYLE)
if os.environ.get("SADMOF_NO_USETEX"):
    # Escape hatch for machines without a LaTeX install: mathtext renders the
    # `$...$` fragments, `\textsc{}` and friends come out verbatim.
    plt.rcParams["text.usetex"] = False


# ----
# Public API
# ----


def savefig(fig, path):
    """Write `fig` to `path` as both PDF (for TeX) and PNG (to look at).

    `path` carries no extension; parent directories are created.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(str(path) + suffix, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {path}.pdf, {path}.png")


def savefile(text, path):
    """Write a table (a string, or a list of lines) to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(text, str):
        text = "\n".join(text)
    path.write_text(text)
    print(f"wrote {path}")


def fig_and_ax(figsize=None):
    """A single-axes figure at the style default size unless told otherwise."""
    fig = plt.figure(figsize=figsize) if figsize else plt.figure()
    return fig, plt.axes()


def major_ticks_every(ax, spacing, direction="x"):
    _locator(ax, direction).set_major_locator(_multiple(spacing))


def minor_ticks_every(ax, spacing, direction="x"):
    _locator(ax, direction).set_minor_locator(_multiple(spacing))


def no_ticks(ax, direction="x"):
    _locator(ax, direction).set_major_locator(plt.NullLocator())


def reversed_legend(ax, **kwargs):
    """Legend in reverse plotting order — for stacked or ranked curves."""
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(list(reversed(handles)), list(reversed(labels)), **kwargs)


# ----
# Internals
# ----


def _locator(ax, direction):
    return ax.xaxis if direction == "x" else ax.yaxis


def _multiple(spacing):
    from matplotlib.ticker import MultipleLocator

    return MultipleLocator(spacing)
