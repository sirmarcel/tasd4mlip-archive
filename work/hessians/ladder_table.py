"""tab:cost — dense vs sparse wall times and C_v errors across the roster.

    python ladder_table.py

Reads `results/ladder/`, writes the whole-page tabular `figures/ladder_table.tex`,
and prints the speedups and error extremes the surrounding text quotes.

One row per (structure, model), one block per chemistry class (MOF, COF,
zeolite), sorted by supercell size within a class. Sparse rungs are priced
end-to-end — pattern + coloring + Hessian — so a
rung whose coloring eats its speedup is visible as such; dense has no such
stages. Errors are C_v(300 K) against the same pair's dense rung, with the
acoustic sum rule enforced (without it, one leaked acoustic mode costs a k_B
and swamps the truncation error being measured).

Only structures with the full ladder — dense, h1-h4, exact K — for every model
appear: the rest of the roster was deliberately left partial, and a
dash that means "not run" would read like a result. MACE's exact K is 4, so its
h4 rung appears in the exact-K column and its h4 column stays empty.
"""

import argparse
from statistics import median

import common
import ladder_load
from decay_results import structure_classes

from sadmof.tbx import model_names_short, model_order, savefile, structure_names
from sadmof.tbx.tables import num_rounder, tabular, to_num

TRUNCATED_HOPS = (1, 2, 3, 4)
TEMPERATURE = "300"


def main():
    build_parser().parse_args()
    ladders = ladder_load.ladders()
    structures = ladder_load.complete(ladders, model_order, TRUNCATED_HOPS)

    table(ladders, structures)
    report(ladders, structures)


# ---------------------------------------------------------------------------
# The tabular
# ---------------------------------------------------------------------------


def table(ladders, structures, out="figures/ladder_table.tex"):
    seconds = num_rounder(0)

    def percent(value):
        if value is None:
            return "--"
        return to_num(f"{0.0 if abs(value) < 0.005 else value:.2f}")

    classes = structure_classes()
    blocks = []
    for cls in ("MOF", "COF", "zeolite"):
        members = [s for s in structures if classes[s] == cls]
        if members:
            blocks.append(class_block(ladders, members, cls, seconds, percent))

    heading = (
        r"& & & & \multicolumn{6}{c}{Wall time (\si{\second})}"
        r" & \multicolumn{5}{c}{$\delta C_V$ (\%)} \\"
        "\n"
        r"\cmidrule(lr){5-10} \cmidrule(lr){11-15}"
        "\n"
    )
    titles = [
        "Class",
        "Structure",
        "Model",
        "$N$",
        "dense",
        *(f"$k{{=}}{hops}$" for hops in TRUNCATED_HOPS),
        "$K$",
        *(f"$k{{=}}{hops}$" for hops in TRUNCATED_HOPS),
        "$K$",
    ]
    savefile(
        tabular(titles, None, layout="lllr rrrrrr rrrrr", heading=heading, blocks=blocks),
        out,
    )


def class_block(ladders, members, cls, seconds, percent):
    block = []
    for j, structure in enumerate(members):
        if j > 0:
            block.append(r"\addlinespace")
        for k, model in enumerate(model_order):
            ladder = ladders[(structure, model)]
            dense = ladder["dense"]
            exact = ladder[common.EXACT_HOPS[model]]

            times, errors = [], []
            for hops in TRUNCATED_HOPS:
                if model == "mace" and hops == common.EXACT_HOPS["mace"]:
                    times.append("--")
                    errors.append("--")
                    continue
                row = ladder[hops]
                times.append(seconds(ladder_load.sparse_time_s(row)))
                errors.append(percent(ladder_load.cv_rel_err_pct(row, dense)))

            block.append(
                [
                    cls if j == 0 and k == 0 else "",
                    structure_names.get(structure, structure) if k == 0 else "",
                    model_names_short[model],
                    to_num(dense["n_atoms"]),
                    seconds(dense["hessian_cold_s"]),
                    *times,
                    seconds(ladder_load.sparse_time_s(exact)),
                    *errors,
                    percent(ladder_load.cv_rel_err_pct(exact, dense)),
                ]
            )
    return block


# ---------------------------------------------------------------------------
# Quotable numbers
# ---------------------------------------------------------------------------


def report(ladders, structures):
    """Speedups against dense and error extremes, for transcription."""
    print(f"{len(structures)} structures")
    for model in model_order:
        exact_hops = common.EXACT_HOPS[model]
        print(f"\n{model}  (K = {exact_hops})")
        for rung, name in ((exact_hops, "exact K"), (3, "h3")):
            speedups, cv_errors, h_errors = [], [], []
            for structure in structures:
                ladder = ladders[(structure, model)]
                dense = ladder["dense"]
                row = ladder[rung]
                sparse_s = ladder_load.sparse_time_s(row)
                if sparse_s is None:  # re-derived from cache, no fp32 timing
                    print(f"  (no timing for {structure} {rung}, excluded)")
                else:
                    speedups.append((structure, dense["hessian_cold_s"] / sparse_s))
                cv_errors.append((structure, ladder_load.cv_rel_err_pct(row, dense)))
                h_errors.append((structure, ladder_load.hessian_rel_err(row)))

            ranked = sorted(s for _, s in speedups)
            losses = [s for s, v in speedups if v < 1.0]
            worst_cv = max(cv_errors, key=lambda item: abs(item[1]))
            worst_h = max(h_errors, key=lambda item: item[1])
            print(
                f"  {name}: end-to-end speedup "
                f"min {ranked[0]:.2f} / med {median(ranked):.2f}"
                f" / max {ranked[-1]:.2f}; "
                f"{len(losses)} losses ({', '.join(losses) or 'none'})"
            )
            print(
                f"    worst |dCv| {abs(worst_cv[1]):.2g} % ({worst_cv[0]}); "
                f"worst rel Hessian err {worst_h[1]:.3g} ({worst_h[0]})"
            )


def build_parser():
    return argparse.ArgumentParser(description=__doc__)


if __name__ == "__main__":
    main()
