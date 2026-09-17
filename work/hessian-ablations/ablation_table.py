"""tab:ablations — the pipeline's departures from grm2025, summarized.

    python ablation_table.py

Reads `results/ablation/`, writes `figures/ablation_table.tex`, and prints the
per-pair extremes the surrounding text quotes.

One row per (model, ablation): the single-axis flips first — precision (fp32),
the D3 correction (MACE), shadow forces (PET) — then the `production` composite,
the whole gap between the production corner and the grm2025-faithful reference.
Statistics are over the extracted structures: median and worst |dC_v(300 K)| with
the acoustic sum rule enforced, the worst single-mode frequency deviation, and
the median relative Frobenius error of the Hessian itself.
"""

import argparse
from statistics import median

import ablation_load

from sadmof.tbx import model_names_short, model_order, savefile
from sadmof.tbx.tables import num_rounder, tabular, to_num

TEMPERATURE = "300"

ABLATION_NAMES = {
    "precision_fp32": "fp32 (full matmuls)",
    "precision_tf32": "fp32 (TF32 matmuls)",
    "d3": "no D3",
    "shadow": "no adaptive-cutoff forces",
    # A tuple spreads the label over the block's rows, one line each.
    "production": ("fp32", "+ no D3", "+ no adaptive-cutoff forces"),
}
# Single-flip axes first, the production composite last. Out of the tabular:
# `precision` (visually duplicates the TF32 row — fp32-proper is negligible)
# and `precision_bare` (diagnostic); both stay in the report.
ROW_ORDER = ("precision_fp32", "precision_tf32", "d3", "shadow", "production")
REPORT_ORDER = ROW_ORDER + ("precision", "precision_bare")


def main():
    build_parser().parse_args()
    grouped = ablation_load.by_ablation()
    if not grouped:
        raise SystemExit("nothing under results/ablation/ yet")

    table(grouped)
    report(grouped)


# ---------------------------------------------------------------------------
# The tabular
# ---------------------------------------------------------------------------


def fixed2(number):
    """Two significant figures, always fixed-point: 0.019, 4.1, 51, 377."""
    if abs(number) >= 100:
        return to_num(f"{number:.0f}")
    return to_num(f"{number:#.2g}".rstrip("."))


def sci2(number):
    """Two significant figures, always scientific: 3.6e-05, 2.4e-02."""
    return to_num(f"{number:.1e}")


def percent_column(values):
    """One formatter per column *within a block*, so no block mixes notations:
    scientific as soon as any entry needs it, plain decimals otherwise."""
    return sci2 if min(values) < 1e-3 else fixed2


def label_line(name, i):
    lines = (name,) if isinstance(name, str) else name
    return lines[i] if i < len(lines) else ""


def table(grouped, out="figures/ablation_table.tex"):
    blocks = []
    for ablation in ROW_ORDER:
        stats = []
        for model in model_order:
            rows = grouped.get((model, ablation))
            if not rows:
                continue
            # No None-tolerance: a missing dC_v would silently shrink the
            # sample, so a partial campaign fails loudly here (abs(None)).
            dcv = [abs(ablation_load.dcv_pct(r, TEMPERATURE)) for r in rows]
            dnu = [r["dnu_max_cm1_asr"] for r in rows]
            frobs = [r["hessian_error"]["rel_frob"] for r in rows]
            stats.append((model, median(dcv), max(dcv), max(dnu), median(frobs)))
        if not stats:
            continue
        med_fmt = percent_column([s[1] for s in stats])
        max_fmt = percent_column([s[2] for s in stats])
        block = [
            [
                label_line(ABLATION_NAMES[ablation], i),
                model_names_short[model],
                med_fmt(med),
                max_fmt(mx),
                fixed2(dnu),
                num_rounder(1, "e")(frob),
            ]
            for i, (model, med, mx, dnu, frob) in enumerate(stats)
        ]
        blocks.append(block)

    titles = [
        "Ablation",
        "Model",
        r"med.\ $|\delta C_V|$ (\%)",
        r"max $|\delta C_V|$ (\%)",
        r"max $|\Delta\nu|$ (cm$^{-1}$)",
        r"med.\ $\lVert\Delta\hess\rVert_F / \lVert\hess\rVert_F$",
    ]
    savefile(tabular(titles, None, layout="ll rrrr", blocks=blocks), out)


# ---------------------------------------------------------------------------
# Quotable numbers
# ---------------------------------------------------------------------------


def report(grouped):
    """Per-pair extremes with the structure they occur in, for transcription."""
    for (model, ablation), rows in sorted(
        grouped.items(),
        key=lambda kv: (REPORT_ORDER.index(kv[0][1]), model_order.index(kv[0][0])),
    ):
        dcv = [(abs(ablation_load.dcv_pct(r, TEMPERATURE)), r["structure"]) for r in rows]
        dnu = [(r["dnu_max_cm1_asr"], r["structure"]) for r in rows]
        frob = [(r["hessian_error"]["rel_frob"], r["structure"]) for r in rows]
        worst_cv, worst_nu, worst_h = max(dcv), max(dnu), max(frob)
        print(
            f"{model}/{ablation} [{rows[0]['ablated']} vs {rows[0]['reference']}]: "
            f"n={len(rows)}, "
            f"|dCv(300K,asr)| med {median(v for v, _ in dcv):.3g} % "
            f"/ max {worst_cv[0]:.3g} % ({worst_cv[1]}); "
            f"max|dnu(asr)| {worst_nu[0]:.3g} cm^-1 ({worst_nu[1]}); "
            f"rel|dH|_F med {median(v for v, _ in frob):.2g} "
            f"/ max {worst_h[0]:.2g} ({worst_h[1]})"
        )


def build_parser():
    return argparse.ArgumentParser(description=__doc__)


if __name__ == "__main__":
    main()
