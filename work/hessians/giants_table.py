"""tab:giants — dense, truncated, and exact-sparse Hessians for the giant MOFs.

    python giants_table.py

Reads `results/ladder/`, writes the main-text tabular `figures/giants_table.tex`,
and prints the speedups the surrounding text quotes.

The appendix cost table restricted to the giant MOFs and reported as speedups:
one row per (structure, model), one block per structure, largest unit cell
first, with speedup over dense and C_v error at every truncated rung and at
exact K. Sparse wall time is end-to-end (pattern + coloring + Hessian), the
C_v error is at 300 K with the acoustic sum rule enforced, against the same
pair's dense rung. The exact-K columns are shaded (`exactshade`, defined in
the preprint's snippets.tex) to match the truncation figure. MACE's exact K
is 4, so its k=4 column stays empty.
"""

import argparse

import common
import ladder_load

from sadmof.tbx import model_names_short, model_order, savefile, structure_names
from sadmof.tbx.tables import num_rounder, tabular, to_num

GIANTS = ("mil101", "mil100", "mof210", "mof177")
TRUNCATED_HOPS = (1, 2, 3, 4)
# the rung at which C_v is converged to 0.1 % for the vast majority of the
# benchmark, as quoted in the truncation paragraph
CONVERGED_HOPS = {"mace": 2, "pet-xs": 4, "pet-s": 3}


def main():
    build_parser().parse_args()
    ladders = ladder_load.ladders()
    table(ladders)
    report(ladders)


# ---------------------------------------------------------------------------
# The tabular
# ---------------------------------------------------------------------------


def table(ladders, out="figures/giants_table.tex"):
    seconds = num_rounder(0)
    ratio = num_rounder(1)

    blocks = []
    for structure in GIANTS:
        block = []
        for k, model in enumerate(model_order):
            ladder = ladders[(structure, model)]
            dense = ladder["dense"]
            exact = ladder[common.EXACT_HOPS[model]]
            speedups, errors = [], []
            for hops in TRUNCATED_HOPS:
                if hops == common.EXACT_HOPS[model]:
                    speedups.append("--")
                    errors.append("--")
                    continue
                row = ladder[hops]
                speedups.append(ratio(_speedup(dense, row)))
                errors.append(_permille(ladder_load.cv_rel_err_pct(row, dense)))
            block.append(
                [
                    structure_names[structure] if k == 0 else "",
                    model_names_short[model],
                    to_num(dense["n_atoms"]),
                    seconds(dense["hessian_cold_s"]),
                    *speedups,
                    ratio(_speedup(dense, exact)),
                    *errors,
                    _permille(ladder_load.cv_rel_err_pct(exact, dense)),
                ]
            )
        blocks.append(block)

    heading = (
        r"& & & Time & \multicolumn{5}{c}{Speedup over dense}"
        r" & \multicolumn{5}{c}{$\delta C_V$ (\textperthousand)} \\"
        "\n"
        r"\cmidrule(lr){5-9} \cmidrule(lr){10-14}"
        "\n"
        r"& & & dense & \multicolumn{4}{c}{truncated} & exact"
        r" & \multicolumn{4}{c}{truncated} & exact \\"
        "\n"
        r"\cmidrule(lr){5-8} \cmidrule(lr){9-9} \cmidrule(lr){10-13} \cmidrule(lr){14-14}"
        "\n"
    )
    titles = [
        "Structure",
        "Model",
        "$N$",
        r"(\si{\second})",
        *(f"$k{{=}}{hops}$" for hops in TRUNCATED_HOPS),
        "$K$",
        *(f"$k{{=}}{hops}$" for hops in TRUNCATED_HOPS),
        "$K$",
    ]
    exact_column = r">{\columncolor{exactshade}}r"
    layout = f"llrr rrrr{exact_column} rrrr{exact_column}"
    savefile(tabular(titles, None, layout=layout, heading=heading, blocks=blocks), out)


def _speedup(dense, row):
    return dense["hessian_cold_s"] / ladder_load.sparse_time_s(row)


def _permille(value):
    """Per mille with two decimals; below the rounding threshold the cell says
    so instead of showing a zero that is not one."""
    if value is None:
        return "--"
    value *= 10.0
    return to_num("<0.01" if abs(value) < 0.005 else f"{value:.2f}")


# ---------------------------------------------------------------------------
# Quotable numbers
# ---------------------------------------------------------------------------


def report(ladders):
    """Speedups over dense at exact K and at the converged rung, for transcription."""
    for model in model_order:
        exact_hops, conv_hops = common.EXACT_HOPS[model], CONVERGED_HOPS[model]
        print(f"\n{model}  (K = {exact_hops}, converged rung k = {conv_hops})")
        for structure in GIANTS:
            ladder = ladders[(structure, model)]
            dense, exact, trunc = ladder["dense"], ladder[exact_hops], ladder[conv_hops]
            print(
                f"  {structure_names[structure]:8s} N={dense['n_atoms']:5d}"
                f"  dense {dense['hessian_cold_s']:7.0f} s"
                f"  exact {_speedup(dense, exact):5.2f}x"
                f"  k={conv_hops} {_speedup(dense, trunc):5.2f}x"
                f"  dCv(k={conv_hops}) {ladder_load.cv_rel_err_pct(trunc, dense):+.3f} %"
            )


def build_parser():
    return argparse.ArgumentParser(description=__doc__)


if __name__ == "__main__":
    main()
