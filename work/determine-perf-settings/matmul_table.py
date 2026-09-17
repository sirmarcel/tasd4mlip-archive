"""tab:matmul — what TF32 buys, per system and model.

    python matmul_table.py

Reads `results/matmul.json`, writes `figures/matmul_table.tex`, and prints the
per-model ranges the surrounding text quotes.

One row per (model, system): steady-state per-HVP time with TF32 matmuls (JAX's
fp32 default on Hopper, the production path) against full-fp32 matmuls, and the
overhead of pinning the latter. This is the throughput half of the precision
story whose accuracy half is `work/hessian-ablations`' `precision_tf32` rung.
"""

import argparse
from pathlib import Path

import matmul_load

from sadmof.tbx import model_names_short, model_order, savefile, structure_names
from sadmof.tbx.tables import num_rounder, tabular, to_num

HERE = Path(__file__).resolve().parent


def main():
    build_parser().parse_args()
    grouped = matmul_load.by_model()
    if not grouped:
        raise SystemExit("nothing in results/matmul.json")

    table(grouped)
    report(grouped)


# ---------------------------------------------------------------------------
# The tabular
# ---------------------------------------------------------------------------


def table(grouped, out=HERE / "figures" / "matmul_table.tex"):
    ms = num_rounder(1)
    pct = num_rounder(0)

    blocks = []
    for model in model_order:
        rows = grouped.get(model)
        if not rows:
            continue
        block = [
            [
                model_names_short[model],
                structure_names.get(row["structure"], row["structure"]),
                to_num(row["n_atoms"]),
                ms(row["default"]["ms_per_hvp"]),
                ms(row["highest"]["ms_per_hvp"]),
                pct(matmul_load.overhead_pct(row)),
            ]
            for row in rows
        ]
        # model name on the block's first row only
        block[1:] = [["", *row[1:]] for row in block[1:]]
        blocks.append(block)

    titles = [
        "Model",
        "System",
        "$N$",
        r"TF32 (\si{\milli\second}/HVP)",
        r"fp32 (\si{\milli\second}/HVP)",
        r"overhead (\%)",
    ]
    savefile(tabular(titles, None, layout="ll rrrr", blocks=blocks), out)


# ---------------------------------------------------------------------------
# Quotable numbers
# ---------------------------------------------------------------------------


def report(grouped):
    """Per-model ranges with the system they occur in, for transcription.

    Overheads are percent of the TF32 per-HVP time, at the locked engine
    setting on one H100; MACE rows are at tight padding, PET at production.
    """
    setting = matmul_load.dataset()["setting"]
    device = matmul_load.dataset()["device"]
    print(
        f"locked setting: {setting['mode']}, chunk {setting['chunk_size']}, "
        f"remat {setting['remat']}, {setting['precision']} on {device}"
    )
    for model in model_order:
        rows = grouped.get(model)
        if not rows:
            continue
        warm = [(matmul_load.overhead_pct(r), r["structure"]) for r in rows]
        cold = [(matmul_load.overhead_pct(r, "cold"), r["structure"]) for r in rows]
        lo, hi = min(warm), max(warm)
        print(
            f"{model}: n={len(rows)}, pinning `highest` costs "
            f"{lo[0]:.0f} % ({lo[1]}) to {hi[0]:.0f} % ({hi[1]}) per HVP warm; "
            f"cold {min(cold)[0]:.0f}–{max(cold)[0]:.0f} %"
        )
    peaks = [r["peak_ratio"] for r in matmul_load.pairs() if r["peak_ratio"]]
    print(
        f"peak memory ratio (highest/default) over {len(peaks)} pairs: "
        f"{min(peaks):.4f}–{max(peaks):.4f} — the envelope is unaffected"
    )


def build_parser():
    return argparse.ArgumentParser(description=__doc__)


if __name__ == "__main__":
    main()
