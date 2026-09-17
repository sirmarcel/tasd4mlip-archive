"""fig:truncation — Hessian error, C_v error, and speedup versus hop count.

    python ladder_figure.py

Reads `results/ladder/`, writes `figures/truncation.{pdf,png}`, and prints the
per-rung medians and extremes the surrounding text quotes.

Left: relative Frobenius error of the truncated Hessian against dense. Center:
|C_v(300 K)| error against dense, ASR enforced. Right: end-to-end speedup over
dense, with the break-even line at 1. All three per model over the
complete-ladder structures — median dot, min-max whisker — at k = 1..4 and the
model's exact K (MACE's K is 4, so its ladder ends at the K position); the
same k sweep is read across all panels: error falls to the floor while the
speedup is largest at small k and roughly break-even at exact K.
"""

import numpy as np

import argparse

import ladder_load

from sadmof.tbx import (
    darkgrey,
    grey,
    model_colors,
    model_linestyles,
    model_markers,
    model_names_short,
    model_order,
    plt,
    savefig,
    textwidth,
)

TRUNCATED_HOPS = (1, 2, 3, 4)
X_EXACT = 5.4  # the exact-K position, set off from the truncated rungs
X_SPLIT = 0.5 * (TRUNCATED_HOPS[-1] + X_EXACT) - 0.1  # truncated | exact boundary
OFFSETS = {"mace": -0.15, "pet-xs": 0.0, "pet-s": 0.15}


def main():
    build_parser().parse_args()
    ladders = ladder_load.ladders()
    structures = ladder_load.complete(ladders, model_order, TRUNCATED_HOPS)

    fig, axes = plt.subplots(1, 3, figsize=(textwidth, 2.4), constrained_layout=True)
    error_panel(axes[0], ladders, structures, hessian_err)
    region_labels(axes[0], top=True)
    axes[0].set_ylabel(r"$\|H_k - H\|_\mathrm{F} \, / \, \|H\|_\mathrm{F}$")
    error_panel(axes[1], ladders, structures, cv_err)
    region_labels(axes[1], top=False)
    axes[1].set_ylabel(r"$|\delta C_V| \, / \, C_V$")
    axes[1].legend(loc="upper right", fontsize=6.5)
    error_panel(axes[2], ladders, structures, speedup)
    region_labels(axes[2], top=True)  # the bottom is the break-even line
    axes[2].set_ylabel("Speedup over dense")
    axes[2].axhline(1.0, color=darkgrey, linewidth=0.8, linestyle=(0, (4, 2)), zorder=0)

    savefig(fig, "figures/truncation")
    report(ladders, structures)


def hessian_err(row, dense):
    return ladder_load.hessian_rel_err(row)


def cv_err(row, dense):
    return abs(ladder_load.cv_rel_err_pct(row, dense)) / 100.0


def speedup(row, dense):
    # A rung re-derived from a cached Hessian has no fp32 timing.
    s = ladder_load.sparse_time_s(row)
    return np.nan if s is None else dense["hessian_cold_s"] / s


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def error_panel(ax, ladders, structures, metric):
    """One error metric against the hop count, per model over the structures.

    The medians carry the message: bold connected markers over the truncated
    rungs, a bare marker at `K`. The per-structure values stay visible as
    quiet dots behind them, so the spread is data rather than a whisker.
    """
    for model in model_order:
        color = model_colors[model]
        marker = model_markers[model]
        positions, medians = [], []
        for rung, x in rung_positions(ladders, model):
            values = np.array(
                [
                    metric(
                        ladders[(structure, model)][rung],
                        ladders[(structure, model)]["dense"],
                    )
                    for structure in structures
                ]
            )
            x += OFFSETS[model]
            ax.plot(
                np.full(len(values), x),
                values,
                marker=marker,
                markersize=2,
                linestyle="none",
                color=color,
                alpha=0.3,
                markeredgewidth=0,
            )
            positions.append(x)
            medians.append(np.nanmedian(values))
        # The line connects the truncated ladder only; `K` sits apart, exact
        # rather than one more rung, so it gets a bare marker.
        ax.plot(
            positions[:-1],
            medians[:-1],
            marker=marker,
            markersize=4.5,
            linewidth=1.4,
            linestyle=model_linestyles[model],
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=model_names_short[model],
            zorder=3,
        )
        ax.plot(
            positions[-1],
            medians[-1],
            marker=marker,
            markersize=4.5,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )
    ax.axvspan(
        X_SPLIT,
        X_EXACT + 1.0,
        color=grey,
        alpha=0.15,
        linewidth=0,
        zorder=0,
    )

    ax.set_yscale("log")
    ax.set_xlim(0.4, X_EXACT + 0.75)
    ax.set_xticks([*TRUNCATED_HOPS, X_EXACT])
    ax.set_xticklabels([*(str(h) for h in TRUNCATED_HOPS), "$K$"])
    ax.set_xlabel("Hop count $k$")
    ax.grid(axis="y", color=grey, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)


def region_labels(ax, top=True):
    """Name the regions the grey span separates, so each panel reads without
    the caption; at the top or the bottom, wherever the panel has room."""
    trans = ax.get_xaxis_transform()  # x in data, y in axes fraction
    y, va = (0.97, "top") if top else (0.03, "bottom")
    common = dict(y=y, va=va, fontsize=6, color=darkgrey, transform=trans)
    ax.text(x=X_SPLIT - 0.15, s=r"$\leftarrow$ truncated", ha="right", **common)
    ax.text(x=X_SPLIT + 0.15, s="exact", ha="left", **common)


def rung_positions(ladders, model):
    """`(rung, x)` pairs for one model: truncated rungs below `K`, then `K` at
    its own position. MACE's h4 is its exact rung, so it appears only at `K`."""
    exact = next(v for (s, m), v in ladders.items() if m == model)
    exact = next(iter(exact.values()))["exact_hops"]
    return [(h, float(h)) for h in TRUNCATED_HOPS if h < exact] + [(exact, X_EXACT)]


# ---------------------------------------------------------------------------
# Quotable numbers
# ---------------------------------------------------------------------------


def report(ladders, structures):
    print(f"{len(structures)} structures: {', '.join(structures)}")
    for model in model_order:
        print(f"\n{model}")
        for rung, x in rung_positions(ladders, model):
            h_values, cv_values = [], []
            for structure in structures:
                ladder = ladders[(structure, model)]
                h_values.append(hessian_err(ladder[rung], ladder["dense"]))
                cv_values.append(cv_err(ladder[rung], ladder["dense"]))
            name = "K" if x == X_EXACT else rung
            print(
                f"  k={name}: rel |dH| med {np.nanmedian(h_values):.3g} "
                f"max {max(h_values):.3g}   |dCv| med {100 * np.nanmedian(cv_values):.3g} % "
                f"max {100 * max(cv_values):.3g} %"
            )
    worst = max(
        (
            abs(ladder_load.cv_rel_err_pct(ladders[(s, m)][3], ladders[(s, m)]["dense"])),
            s,
            m,
        )
        for s in structures
        for m in model_order
    )
    print(f"\nworst case at k=3: {worst[1]}/{worst[2]} at {worst[0]:.2f} %")


def build_parser():
    return argparse.ArgumentParser(description=__doc__)


if __name__ == "__main__":
    main()
