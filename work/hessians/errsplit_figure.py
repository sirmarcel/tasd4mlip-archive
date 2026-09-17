"""fig:errsplit — the two parts of the truncation error, over the ladder.

    python errsplit_figure.py

Reads `results/ladder/`, writes `figures/errsplit.{pdf,png}`, and prints the
ratio and noise-floor statistics the surrounding text quotes.

One marker per (structure, model, truncated rung): the contamination of the
retained entries against the discarded couplings, both relative to the dense
norm. Contamination tracks the discarded mass at a fixed ratio until it
bottoms out at the single-precision noise floor, which the exact-K rungs
(discarded ≡ 0) measure directly and the shaded band shows.
"""

import argparse

import common
import ladder_load

from sadmof.tbx import (
    fig_and_ax,
    model_colors,
    model_markers,
    model_names_short,
    model_order,
    np,
    savefig,
)

# The band drawn for the noise floor: the central range of exact-K errors. A
# few exact-K rungs sit well above the rest (structure-specific fp32 noise),
# so the extremes are not the floor.
FLOOR_QUANTILES = (0.1, 0.9)

# The ratio is quoted over every truncated rung whose discarded mass sits above
# the noise-floor band, i.e. wherever contamination is collision leakage rather
# than fp32 noise. That is the set the text describes ("tracks ... until it
# bottoms out at the floor"), and it spans about four decades. Do not quote a
# fixed cut such as discarded > 1e-3 instead: it keeps only the top two
# decades, where the ratio runs slightly higher (0.64 vs 0.60), so the median
# and the span would come from different point sets. The band-edge choice
# itself barely matters — all truncated rungs together also give 0.60.


def main():
    build_parser().parse_args()
    points, floor = collect()
    figure(points, floor)
    report(points, floor)


def collect():
    """Truncated rungs as `(discarded, contamination, structure, model, rung)`,
    and the exact-K total errors that set the noise floor."""
    points, floor = [], []
    for (structure, model), rungs in ladder_load.ladders().items():
        for rung, row in rungs.items():
            if rung == "dense" or not row.get("hessian_error"):
                continue
            if rung == common.EXACT_HOPS[model]:
                floor.append(ladder_load.hessian_rel_err(row))
                continue
            discarded = ladder_load.hessian_rel_err(row, part="frob_discarded")
            contamination = ladder_load.hessian_rel_err(row, part="frob_contamination")
            points.append((discarded, contamination, structure, model, rung))
    return points, np.array(floor)


def figure(points, floor, out="figures/errsplit"):
    fig, ax = fig_and_ax(figsize=(3.4, 2.6))

    lo, hi = np.quantile(floor, FLOOR_QUANTILES)
    ax.axhspan(lo, hi, color="0.88", lw=0, zorder=0)
    ax.text(
        0.6,
        lo * 1.6,
        "single-precision floor (exact $K$)",
        fontsize=6,
        color="0.4",
        ha="right",
    )

    for model in model_order:
        sel = [(d, c) for d, c, _, m, _ in points if m == model]
        if not sel:
            continue
        d, c = zip(*sel)
        ax.scatter(
            d,
            c,
            s=10,
            marker=model_markers[model],
            color=model_colors[model],
            label=model_names_short[model],
            zorder=2,
        )

    diagonal = np.array([1e-8, 1.0])
    ax.plot(diagonal, diagonal, ls="--", color="0.6", lw=0.8, zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-8, 1.0)
    ax.set_ylim(5e-8, 1.0)
    ax.set_xlabel(
        r"Discarded couplings $\lVert\Delta\mathbf{H}_\mathrm{disc}\rVert_F / \lVert\mathbf{H}\rVert_F$"
    )
    ax.set_ylabel(
        r"Contamination $\lVert\Delta\mathbf{H}_\mathrm{cont}\rVert_F / \lVert\mathbf{H}\rVert_F$"
    )
    ax.legend(loc="upper left", fontsize=6.5)
    savefig(fig, out)


def report(points, floor):
    """The quotable numbers: the contamination/discarded ratio where both are
    clear of the floor, and the floor itself."""
    lo, hi = np.quantile(floor, FLOOR_QUANTILES)
    above = [(d, c) for d, c, *_ in points if d > hi]
    disc = np.array([d for d, _ in above])
    ratio = np.array([c / d for d, c in above])
    print(
        f"noise floor (exact-K total error): median {np.median(floor):.1e}, "
        f"{FLOOR_QUANTILES[0]:.0%}-{FLOOR_QUANTILES[1]:.0%} band {lo:.1e}..{hi:.1e}, "
        f"max {floor.max():.1e}"
    )
    print(
        f"contamination / discarded over truncated rungs above the floor band "
        f"(n={len(ratio)}, discarded {disc.min():.1e}..{disc.max():.1e}, "
        f"{np.log10(disc.max() / disc.min()):.1f} decades): "
        f"median {np.median(ratio):.2f} / q25 {np.quantile(ratio, 0.25):.2f} "
        f"/ q75 {np.quantile(ratio, 0.75):.2f} / min {ratio.min():.2f} / max {ratio.max():.2f}"
    )
    print(
        f"truncated rungs whose discarded mass is inside the floor band: {len(points) - len(above)}"
    )


def build_parser():
    return argparse.ArgumentParser(description=__doc__)


if __name__ == "__main__":
    main()
