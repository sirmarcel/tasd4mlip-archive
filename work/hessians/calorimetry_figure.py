"""fig:calorimetry — predicted C_V against experimental calorimetry.

    python calorimetry_figure.py

Reads the dense rungs from `results/ladder/` and the digitized experimental
heat-capacity curves, writes `figures/calorimetry.{pdf,png}`, and prints the
plotted values for transcription.
"""

import numpy as np

from pathlib import Path

import ladder_load

from sadmof.tbx import (
    darkgrey,
    model_colors,
    model_markers,
    model_names_short,
    model_order,
    plt,
    savefig,
    structure_names,
    textwidth,
)

EXPERIMENT = (
    Path(__file__).resolve().parents[2]
    / "sources"
    / "processed"
    / "big-mofs"
    / "experimental_curves.npz"
)
SOURCES = {"mof177": r"exp.\ (Kloutse 2015)", "mil101": r"exp.\ (Liu 2017)"}
STRUCTURES = ("mof177", "mil101")


def main():
    ladders = ladder_load.ladders()
    exp = np.load(EXPERIMENT)

    fig, axes = plt.subplots(
        1, 2, figsize=(textwidth, 2.2), sharey=True, constrained_layout=True
    )
    for ax, structure in zip(axes, STRUCTURES):
        panel(ax, structure, exp[structure], ladders)
    # Generic label: the grey curves are measured C_p, the markers computed C_V.
    axes[0].set_ylabel(r"Heat capacity (J\,g$^{-1}$\,K$^{-1}$)")

    savefig(fig, "figures/calorimetry")


def panel(ax, structure, curve, ladders):
    t_exp, cp_exp = curve
    ax.plot(t_exp, cp_exp, color=darkgrey, label=SOURCES[structure])

    for model in model_order:
        cv = ladders[(structure, model)]["dense"]["cv_J_per_gK_asr"]
        temperatures = sorted(cv, key=float)
        print(
            f"{structure} {model}: " + "  ".join(f"{t} K {cv[t]:.4f}" for t in temperatures)
        )
        ax.plot(
            [float(t) for t in temperatures],
            [cv[t] for t in temperatures],
            marker=model_markers[model],
            markersize=4,
            linestyle="none",
            color=model_colors[model],
            markeredgecolor="white",
            markeredgewidth=0.4,
            label=model_names_short[model],
        )

    lo = min(t_exp.min(), 250.0)
    hi = max(t_exp.max(), 400.0)
    pad = 0.03 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("Temperature (K)")
    ax.set_title(structure_names[structure], fontsize=8)
    ax.legend(loc="lower right", fontsize=6.5)


if __name__ == "__main__":
    main()
