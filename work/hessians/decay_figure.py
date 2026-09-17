"""fig:decay — force-constant block norms against interatomic distance.

    python decay_figure.py                     # every condition in results/decay
    python decay_figure.py --structures mil101

Reads `results/decay/`, writes `figures/decay.{pdf,png}`, and prints the exact
scalars the surrounding text quotes — the largest distance still carrying a
nonzero block, the model's reach `R_K`, and the count of nonzero blocks beyond
`K` hops, which the exact pattern requires to be zero.
"""

import numpy as np

import argparse
import csv

import common
import decay_load

from sadmof.tbx import (
    cyan,
    darkgrey,
    fig_and_ax,
    grey,
    magenta,
    major_ticks_every,
    model_colors,
    model_linestyles,
    model_names,
    model_names_short,
    model_order,
    plt,
    savefig,
    textwidth,
)

QUANTILES = (0.1, 0.5, 0.9)
MIN_COUNT = 32

# Chemistry classes, from the selection ranking; the giants are MOFs by
# construction and predate the ranking. Colors deliberately avoid the model
# palette — hue means model everywhere else in this experiment's figures.
CLASS_ORDER = ("MOF", "COF", "zeolite")
CLASS_COLORS = {"MOF": darkgrey, "COF": magenta, "zeolite": cyan}
# Grayscale safety, as with `model_markers`; deliberately not the model set,
# marker shape means class here.
CLASS_MARKERS = {"MOF": "o", "COF": "^", "zeolite": "s"}


def main():
    args = build_parser().parse_args()
    present = decay_load.available()
    models = [m for m in model_order if m in {m for _, m in present}]
    if args.models:
        models = [m for m in models if m in args.models]

    groups = []
    for model in models:
        structures = sorted({s for s, m in present if m == model})
        if args.structures:
            structures = [s for s in structures if s in args.structures]
        if structures:
            groups.append((model, structures))

    distance_figure(groups)
    shell_figure(groups)
    chemistry_figure(groups)


def distance_figure(groups, out="figures/decay"):
    """Block norm against coupling distance, hop reaches ticked along the top."""
    fig, ax = fig_and_ax(figsize=(textwidth, 2.9))

    for row, (model, structures) in enumerate(groups):
        hist, _hist_zero, dist_edges, lognorm_edges, metas = decay_load.aggregate(
            structures, model
        )
        c = decay_load.curves(hist, dist_edges, lognorm_edges, QUANTILES, MIN_COUNT)
        color = model_colors[model]

        ax.fill_between(
            c["distance"],
            c["quantiles"][0],
            c["quantiles"][2],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            c["distance"],
            c["quantiles"][1],
            color=color,
            linestyle=model_linestyles[model],
            label=model_names[model],
        )
        ax.plot(c["distance"], c["maximum"], color=color, linewidth=0.7, alpha=0.8)
        hop_marks(ax, model, metas, row)

        report(model, structures, metas, c)

    ax.set_yscale("log")
    log_minor_decades(ax)
    ax.set_xlim(left=0.0)
    ax.set_xlabel(r"Interatomic distance $d_{ij}$ (\r{A})")
    ax.set_ylabel(r"$\|\Phi_{ij}\|_\mathrm{F}$ (eV\,\r{A}$^{-2}$)")
    ax.legend(loc="lower left")
    ax.grid(axis="y", color=grey, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    savefig(fig, out)


def shell_figure(groups, out="figures/decay_shell"):
    """Block norm against hop shell — the axis truncation actually cuts.

    Shells are discrete, so no fills and no interpolation: per shell and model,
    a bar spanning the 10-90% quantiles with a dot at the median, and a whisker
    up to the exact per-shell maximum (`norm_max`, not a histogram readout).
    """
    fig, ax = fig_and_ax(figsize=(0.62 * textwidth, 2.9))
    offsets = {"mace": -0.22, "pet-xs": 0.0, "pet-s": 0.22}

    for model, structures in groups:
        hist, lognorm_edges, metas = decay_load.aggregate_shells(structures, model)
        exact_hops = metas[0]["exact_hops"]
        shell_edges = np.arange(-0.5, exact_hops + 1)
        c = decay_load.curves(hist, shell_edges, lognorm_edges, QUANTILES, MIN_COUNT)
        x = c["distance"] + offsets[model]
        color = model_colors[model]
        maximum = np.max([m["norm_max"] for m in metas], axis=0)

        ax.plot(
            x,
            c["quantiles"][1],
            color=color,
            linestyle=model_linestyles[model],
            linewidth=0.8,
            alpha=0.4,
        )
        for k in range(len(x)):
            ax.plot(
                [x[k], x[k]],
                [c["quantiles"][0][k], c["quantiles"][2][k]],
                color=color,
                linewidth=2.0,
                solid_capstyle="butt",
            )
            ax.plot(
                [x[k], x[k]],
                [c["quantiles"][2][k], maximum[k]],
                color=color,
                linewidth=0.6,
                alpha=0.55,
            )
        ax.plot(x, maximum, marker="_", markersize=4.5, linestyle="none", color=color)
        ax.plot(
            x,
            c["quantiles"][1],
            marker="o",
            markersize=3.5,
            linestyle="none",
            color=color,
            label=model_names_short[model],
        )

    ax.set_yscale("log")
    log_minor_decades(ax)
    ax.set_xlabel(r"Hop count $k$")
    ax.set_ylabel(r"$\|\Phi_{ij}\|_\mathrm{F}$ (eV\,\r{A}$^{-2}$)")
    major_ticks_every(ax, 1)
    ax.legend(loc="upper right")
    ax.grid(axis="y", color=grey, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    savefig(fig, out)


def chemistry_figure(groups, out="figures/decay_chemistry"):
    """Block norm against hop shell, split by chemistry class, one panel per model.

    The hop axis is what truncation cuts, so this is the decay figure the text
    argues from. Panel widths are proportional to each model's `K + 1` so a hop
    spans the same width everywhere. Quantiles are over coupled pairs, so a
    single-structure class still carries a distribution; the MOF band
    aggregates the rest.
    """
    classes = structure_classes()
    widths = [common.EXACT_HOPS[model] + 1 for model, _ in groups]
    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(textwidth, 2.3),
        sharey=True,
        width_ratios=widths,
        constrained_layout=True,
    )
    offsets = {"MOF": -0.22, "COF": 0.0, "zeolite": 0.22}

    for ax, (model, structures) in zip(np.atleast_1d(axes), groups):
        exact_hops = common.EXACT_HOPS[model]
        shell_edges = np.arange(-0.5, exact_hops + 1)
        for cls in CLASS_ORDER:
            members = [s for s in structures if classes.get(s) == cls]
            if not members:
                continue
            hist, lognorm_edges, _m = decay_load.aggregate_shells(members, model)
            c = decay_load.curves(hist, shell_edges, lognorm_edges, QUANTILES, MIN_COUNT)
            color = CLASS_COLORS[cls]
            x = c["distance"] + offsets[cls]

            ax.plot(x, c["quantiles"][1], color=color, linewidth=0.8, alpha=0.4)
            for k in range(len(x)):
                ax.plot(
                    [x[k], x[k]],
                    [c["quantiles"][0][k], c["quantiles"][2][k]],
                    color=color,
                    linewidth=2.0,
                    solid_capstyle="butt",
                )
            ax.plot(
                x,
                c["quantiles"][1],
                marker=CLASS_MARKERS[cls],
                markersize=3,
                linestyle="none",
                color=color,
                label=cls,
            )

        ax.set_yscale("log")
        log_minor_decades(ax)
        ax.set_xlim(-0.6, exact_hops + 0.6)
        major_ticks_every(ax, 1)
        ax.set_title(model_names_short[model], fontsize=8)
        ax.set_xlabel(r"Hop count $k$")
        ax.grid(axis="y", color=grey, linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)

    axes = np.atleast_1d(axes)
    axes[0].set_ylabel(r"$\|\Phi_{ij}\|_\mathrm{F}$ (eV\,\r{A}$^{-2}$)")
    axes[0].legend(loc="lower left", fontsize=6.5)
    savefig(fig, out)


def structure_classes():
    """`{structure: class}` from the selection ranking; giants are MOFs."""
    with common.RANKING.open() as f:
        classes = {row["structure"]: row["class"] for row in csv.DictReader(f)}
    return {**classes, **dict.fromkeys(common.giants(), "MOF")}


def log_minor_decades(ax):
    """Unlabeled minor ticks at every power of ten the major locator skips."""
    from matplotlib.ticker import LogLocator, NullFormatter

    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(1.0,), numticks=99))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="minor", length=2.5)


def hop_marks(ax, model, metas, row):
    """Per-hop reach radii `R_h` as a tick row at the top, one row per model.

    Ticks sit at the median over structures, the bar behind them spans the
    min-max; the label counts hops, `h = 1..K`.
    """
    radii = np.array([m["reach_radii_A"] for m in metas])
    med = np.median(radii, axis=0)
    y = 0.94 - 0.065 * row
    trans = ax.get_xaxis_transform()  # x in data, y in axes fraction
    color = model_colors[model]

    for h, x in enumerate(med, 1):
        ax.plot(
            [radii[:, h - 1].min(), radii[:, h - 1].max()],
            [y, y],
            color=color,
            linewidth=2.5,
            alpha=0.25,
            transform=trans,
            clip_on=False,
        )
        ax.plot(x, y, marker="|", markersize=5, color=color, transform=trans)
        ax.annotate(
            str(h),
            (x, y),
            xytext=(0, 3.5),
            textcoords="offset points",
            ha="center",
            fontsize=6,
            color=color,
            xycoords=trans,
        )


def report(model, structures, metas, c):
    """The exact scalars, printed for transcription into the write-up."""
    print(f"\n{model}  ({', '.join(structures)})")
    print(
        f"  K = {metas[0]['exact_hops']}   "
        f"R_K = {min(m['reach_radii_A'][-1] for m in metas):.2f}"
        f"-{max(m['reach_radii_A'][-1] for m in metas):.2f} A (over structures)"
    )
    beyond = sum(m["beyond_nonzero"] for m in metas)
    beyond_pairs = sum(m["beyond_pairs"] for m in metas)
    coupled = sum(sum(m["pair_count"]) for m in metas)
    print(f"  coupled pairs: {coupled}")
    print(f"  nonzero blocks beyond K hops: {beyond} of {beyond_pairs}")
    print(
        "  largest distance at a nonzero block: "
        f"{max(max(m['dist_max_nonzero_A']) for m in metas):.2f} A"
    )
    print(f"  max |Phi_ij|: {max(max(m['norm_max']) for m in metas):.4g} eV/A^2")

    valid = ~np.ma.getmaskarray(c["quantiles"][1])
    median = np.asarray(c["quantiles"][1][valid])
    distance = c["distance"][valid]
    # Against the running maximum, so a single dip in a thinly populated bin
    # does not read as the crossing; this is where the median stays below.
    envelope = np.maximum.accumulate(median[::-1])[::-1]
    for target in (1e-3, 1e-5):
        below = np.flatnonzero(envelope < target)
        where = f"{distance[below[0]]:.1f} A" if len(below) else "not reached"
        print(f"  median stays below {target:g} eV/A^2 from: {where}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structures", nargs="+", default=None)
    p.add_argument("--models", nargs="+", default=None)
    return p


if __name__ == "__main__":
    main()
