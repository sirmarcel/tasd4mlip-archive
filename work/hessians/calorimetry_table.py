"""tab:calorimetry — predicted C_v(300 K) of the giants against calorimetry.

    python calorimetry_table.py

Writes the narrow tabular `figures/calorimetry_table.tex` and prints the
numbers it shows, for transcription. One row per giant: the measured C_p
(300 K, interpolated on the published curve) with a source marker, then each
model's dense-rung ASR C_v and its relative deviation. All heat capacities are
gravimetric, J/(g K); the caption carries the units and resolves the source
markers (a: Kloutse et al. 2015, b: Liu et al. 2017).

Every number comes through `calorimetry_results.results()`, so its guardrails
— pinned experimental values, overestimation, deviations below 100 % — gate
this table too.
"""

import argparse

import calorimetry_results

from sadmof.tbx import model_names_short, model_order, savefile, structure_names
from sadmof.tbx.tables import rounder, tabular

STRUCTURES = ("mof177", "mil101")
MARKERS = {"mof177": "a", "mil101": "b"}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    data = calorimetry_results.results()
    table(data)
    report(data)


# ---------------------------------------------------------------------------
# The tabular
# ---------------------------------------------------------------------------


def table(data, out="figures/calorimetry_table.tex"):
    value = rounder(3)

    def percent(deviation):
        return f"{deviation:+.1f}"

    body = []
    for structure in STRUCTURES:
        row = data[structure]
        cells = [
            structure_names[structure],
            value(row["exp_Cp_300K"]) + rf"\textsuperscript{{{MARKERS[structure]}}}",
        ]
        for model in model_order:
            entry = row["models"][model]
            cells += [value(entry["cv_300K"]), percent(entry["rel_dev_pct"])]
        body.append(cells)

    heading = (
        r"& $C_p^{\mathrm{exp}}$"
        + "".join(
            rf" & \multicolumn{{2}}{{c}}{{{model_names_short[m]}}}" for m in model_order
        )
        + " \\\\\n"
        + r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}"
        + "\n"
    )
    titles = [
        "Structure",
        "(300 K)",
        *(t for _ in model_order for t in ("$C_V$", r"$\delta$ (\%)")),
    ]
    savefile(tabular(titles, body, layout="l r rr rr rr", heading=heading), out)


# ---------------------------------------------------------------------------
# Quotable numbers
# ---------------------------------------------------------------------------


def report(data):
    """The table's numbers in plain text, for transcription."""
    for structure in STRUCTURES:
        row = data[structure]
        print(
            f"{structure}: exp C_p(300 K) = {row['exp_Cp_300K']:.4f} J/(g K) "
            f"({row['source']})"
        )
        for model in model_order:
            entry = row["models"][model]
            print(
                f"  {model}: C_v(300 K) = {entry['cv_300K']:.4f} J/(g K), "
                f"{entry['rel_dev_pct']:+.1f} %"
            )


if __name__ == "__main__":
    main()
