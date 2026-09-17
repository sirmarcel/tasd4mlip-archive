"""The paper-facing numbers of the decay analysis, re-derived from `results/decay/`.

What the decay paragraph quotes and claims: the per-hop drop of the median
block norm, its depth at the truncation rung, the fp-dust bound on blocks
beyond `K`, and the chemistry statement — MOFs and COFs decay alike, zeolites
more slowly. Run as a script to print the numbers as JSON; exits nonzero when
the store is incomplete, decay stops being monotone, a beyond-`K` block rises
above dust, or the chemistry claim is contradicted — so a stale or partial
store cannot quietly produce paper numbers.

    uv run python work/hessians/decay_results.py

Medians are histogram readouts at 0.05 dex resolution; quote to one decimal.
The terminal shell `h = K` sits at the fp noise cliff, so the last per-hop
drop overstates the physical rate — the JSON keeps it, the text should not.
"""

import numpy as np

import argparse
import csv
import json
import sys
from statistics import fmean

import common
import decay_load

# Every structure of the ladder roster, plus RSM1885 for the models that have
# a dense rung (MACE is infeasible on its cell). The roster is part of the
# claim: a pair missing from the store or an unexpected extra is an error.
STRUCTURES = (
    "mof210",
    "mil100",
    "mil101",
    "mof177",
    "RSM1876",
    "RSM0023",
    "RSM0047",
    "RSM1162",
    "VFI",
    "20561N3",
    "RSM0010",
    "RSM0274",
    "RSM1877",
    "NPT",
    "18150N2",
    "AFI",
)
EXTRA_PAIRS = (("RSM1885", "pet-xs"), ("RSM1885", "pet-s"))

TRUNCATED = 3
DUST_CEILING = 1e-12  # eV/A^2; measured beyond-K maxima sit ~6 orders below

# The chemistry claim, in numbers: zeolites' mean per-hop drop over shells
# 1..3 must be slower than both other classes by more than the histogram's
# resolution; MOFs and COFs must agree within this margin times two.
CLASS_MARGIN_DEX = 0.1


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(results(), indent=2))


def results():
    expected = {(s, m) for s in STRUCTURES for m in common.EXACT_HOPS} | set(EXTRA_PAIRS)
    present = set(decay_load.available())
    if present != expected:
        fail(
            f"store holds {len(present)} pairs, expected {len(expected)}; "
            f"missing {sorted(expected - present)}, extra {sorted(present - expected)}"
        )

    classes = structure_classes()
    out = {}
    for model in common.EXACT_HOPS:
        structures = sorted({s for s, m in present if m == model})
        medians, beyond = shell_medians(structures, model)
        drops = -np.diff(np.log10(medians))
        if not (drops > 0).all():
            fail(f"{model}: shell medians are not monotonically decaying")
        if beyond > DUST_CEILING:
            fail(f"{model}: beyond-K block norm {beyond:.3g} above fp dust")

        by_class = {}
        for cls in sorted({classes[s] for s in structures}):
            members = [s for s in structures if classes[s] == cls]
            med, _ = shell_medians(members, model)
            by_class[cls] = {
                "n_structures": len(members),
                "shell_medians": med.tolist(),
                "mean_drop_per_hop": drop_rate(med, common.EXACT_HOPS[model]),
            }

        out[model] = {
            "exact_K": common.EXACT_HOPS[model],
            "n_structures": len(structures),
            "shell_medians": medians.tolist(),
            "decades_per_hop": drops.tolist(),
            "decades_below_onsite_at_h3": float(
                np.log10(medians[0]) - np.log10(medians[TRUNCATED])
            ),
            "beyond_K_max_norm": beyond,
            "classes": by_class,
        }
        check_chemistry(model, by_class)

    return out


def check_chemistry(model, by_class):
    rate = {cls: c["mean_drop_per_hop"] for cls, c in by_class.items()}
    if abs(rate["MOF"] - rate["COF"]) > 2 * CLASS_MARGIN_DEX:
        fail(
            f"{model}: MOF and COF rates differ by "
            f"{abs(rate['MOF'] - rate['COF']):.2f} dex/hop — 'decay alike' contradicted"
        )
    slowest_other = min(rate["MOF"], rate["COF"])
    if rate["zeolite"] > slowest_other - CLASS_MARGIN_DEX:
        fail(
            f"{model}: zeolite rate {rate['zeolite']:.2f} dex/hop is not slower than "
            f"MOF/COF ({slowest_other:.2f}) — 'zeolites decay slower' contradicted"
        )


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def shell_medians(structures, model):
    """Median block norm per shell `h = 0..K`, and the beyond-`K` maximum."""
    hist, lognorm_edges, metas = decay_load.aggregate_shells(structures, model)
    exact_hops = metas[0]["exact_hops"]
    shell_edges = np.arange(-0.5, exact_hops + 1)
    c = decay_load.curves(hist, shell_edges, lognorm_edges)
    medians = np.asarray(c["quantiles"][1])
    if np.ma.getmaskarray(c["quantiles"][1]).any():
        fail(f"{model}: a shell has too few blocks for a median over {structures}")
    return medians, max(m["beyond_max_norm"] for m in metas)


def drop_rate(medians, exact_hops):
    """Mean decades lost per hop, past the on-site block and short of the
    terminal shell's fp-noise cliff: drops `h -> h+1` for `h = 1..min(3, K-2)`,
    so no counted drop lands in shell `K`."""
    log = np.log10(medians)
    stop = min(TRUNCATED, exact_hops - 2)
    return fmean(float(log[h] - log[h + 1]) for h in range(1, stop + 1))


def structure_classes():
    """`{structure: class}`; giants are MOFs. (Also defined in decay_figure.py —
    consolidate into decay_load once that file's pending changes land.)"""
    with common.RANKING.open() as f:
        classes = {row["structure"]: row["class"] for row in csv.DictReader(f)}
    return {**classes, **dict.fromkeys(common.giants(), "MOF")}


def fail(message):
    sys.exit(f"decay_results: {message}")


if __name__ == "__main__":
    main()
