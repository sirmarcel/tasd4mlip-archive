"""tab:parity — deviations of every prediction path from the upstream fp64 model.

    uv run python parity_table.py

Reads `results/parity.json`, writes `figures/parity_table.tex`, and prints the
per-metric extremes the surrounding text quotes. Cells are `median (worst)`
over the roster.
"""

import argparse
from pathlib import Path

import parity_load

from sadmof.tbx import model_names_short, model_order, savefile
from sadmof.tbx.tables import rounder, tabular, to_num

HERE = Path(__file__).resolve().parent


def main():
    build_parser().parse_args()
    data = parity_load.parity()
    rows = parity_load.rows(data)

    table(rows)
    report(data, rows)


# ---------------------------------------------------------------------------
# The tabular
# ---------------------------------------------------------------------------


def table(rows, out="figures/parity_table.tex"):
    number = lambda value: to_num(rounder(1, "e")(value))

    def cell(median, worst, metric):
        return f"{number(median[metric])} ({number(worst[metric])})"

    paper = {key: row for key, row in rows.items() if key[1] in parity_load.COMPARISONS}
    counts = {len(row["structures"]) for row in paper.values()}
    if len(counts) > 1:
        raise SystemExit(
            f"comparisons cover different structure counts {sorted(counts)}; cells "
            "reduced over different rosters must not sit side by side. Complete the "
            "short conditions, or restrict the roster and re-extract."
        )

    blocks = []
    for model in model_order:
        block = []
        for comparison in parity_load.COMPARISONS:
            if (model, comparison) not in rows:
                continue
            arrays = rows[(model, comparison)]
            median = parity_load.reduce(arrays, "median")
            worst = parity_load.reduce(arrays, "max")
            stack, precision = parity_load.COMPARISON_NAMES[comparison]
            block.append(
                [
                    # Keyed off the block, not a loop index: a model missing its
                    # first comparison would otherwise lose its label.
                    model_names_short[model] if not block else "",
                    f"{stack} {precision}",
                    cell(median, worst, "de_per_atom"),
                    cell(median, worst, "df_max"),
                    cell(median, worst, "dstress_max"),
                ]
            )
        if block:
            blocks.append(block)

    titles = [
        "Model",
        "Prediction",
        r"$\Delta E$ (eV/atom)",
        r"$\Delta F$ (eV/\AA)",
        r"$\Delta \sigma$ (eV/\AA$^3$)",
    ]
    savefile(
        tabular(titles, None, layout="ll rrr", width=34, blocks=blocks),
        HERE / out,
    )


# ---------------------------------------------------------------------------
# The numbers the text quotes
# ---------------------------------------------------------------------------


def report(data, rows):
    counts = {len(row["structures"]) for row in rows.values()}
    print(f"\nreference {data['reference']}, roster {data['roster']}, {counts} structures")
    print("worst / median over the roster, per metric:\n")

    header = f"{'model':8s} {'prediction':26s}"
    for metric, unit in parity_load.METRICS.items():
        header += f" {metric + ' [' + unit + ']':>28s}"
    print(header)

    for model in model_order:
        for comparison in parity_load.all_comparisons(data):
            if (model, comparison) not in rows:
                continue
            arrays = rows[(model, comparison)]
            worst = parity_load.reduce(arrays, "max")
            median = parity_load.reduce(arrays, "median")
            line = f"{model:8s} {comparison:26s}"
            for metric in parity_load.METRICS:
                line += f" {worst[metric]:12.2e} /{median[metric]:12.2e}  "
            print(line)

    # fp32: ours against upstream's own deviation from upstream fp64.
    print("\nfp32: ours vs upstream's own deviation from upstream fp64")
    for model in model_order:
        ours = rows.get((model, "jax/float32"))
        theirs = rows.get((model, "torch/float32"))
        if ours is None or theirs is None:
            continue
        for metric in ("de_per_atom", "df_max"):
            a = parity_load.reduce(ours, "max")[metric]
            b = parity_load.reduce(theirs, "max")[metric]
            print(f"  {model:8s} {metric:14s} ratio {a / b:5.2f}  ({a:.2e} vs {b:.2e})")

    print("\nworst structure per model, forces:")
    for model in model_order:
        for comparison in parity_load.all_comparisons(data):
            if (model, comparison) not in rows:
                continue
            structure, value = parity_load.worst(rows[(model, comparison)], "df_max")
            print(f"  {model:8s} {comparison:26s} {structure} at {value:.2e} eV/Å")


def build_parser():
    return argparse.ArgumentParser(description=__doc__)


if __name__ == "__main__":
    main()
