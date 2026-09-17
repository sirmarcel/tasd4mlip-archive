"""The paper-facing numbers of the hop ladder, re-derived from `results/ladder/`.

Every quantity the experiment sections quote — exact-K agreement with dense,
end-to-end speedups and the fraction of outright losses, coloring cost relative
to the HVPs, per-rung C_v and Hessian errors with the discarded/contamination
split and their ratio, the rung at which most structures meet the C_v
threshold and the speedup there, the slowest-converging zeolite per model, and
the origin of the residual C_v errors — computed from the committed results
rather than transcribed from a conversation. Run as a script to print them as
JSON; exits nonzero when the grid is incomplete or a quoted claim stops
holding, so a stale or partial store cannot quietly produce paper numbers.

    uv run python work/hessians/ladder_results.py

Speedups are end-to-end for the sparse side (pattern + coloring + Hessian)
against the dense cold Hessian; C_v errors are at 300 K with the acoustic sum
rule enforced; Hessian errors are Frobenius, relative to the dense norm.
"""

import argparse
import json
import sys
from statistics import median

import common
import ladder_load
from decay_results import structure_classes

# The structures the paper's statements aggregate over: every (structure,
# model) pair holds dense, h1..h4, and the model's exact K. RSM1885 stays out:
# MACE is infeasible on its cell, so it can never complete. A structure
# missing from the store *or* a complete structure missing from this list is
# an error — the roster is part of the claim.
EXPECTED = (
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
TRUNCATED_HOPS = (1, 2, 3, 4)
CONVERGED = 3  # the rung the truncation paragraph centers on

# The truncation paragraph calls a rung converged when |dC_v(300 K)| is within
# this, and names the rung at which at least MAJORITY of the structures are.
THRESHOLD_PCT = 0.1
MAJORITY = 0.75

# The exact pattern must reproduce dense to single precision; a relative
# Frobenius error beyond this means the claim is wrong, not imprecise. The
# same applies to C_v at exact K, quoted as "better than 1e-3 %".
FP32_CEILING = 1e-3
EXACT_CV_CEILING_PCT = 1e-3

# Residual C_v errors at converged truncated rungs are attributed to near-zero
# modes crossing the pipeline's drop threshold. Deviations below the table's
# printed resolution are not attributed to anything.
RESIDUAL_FLOOR_PCT = 0.005


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(results(), indent=2))


def results():
    ladders = ladder_load.ladders()
    structures = ladder_load.complete(ladders, common.EXACT_HOPS, TRUNCATED_HOPS)
    if sorted(structures) != sorted(EXPECTED):
        fail(
            f"complete-ladder set is {sorted(structures)}, "
            f"expected {sorted(EXPECTED)} — extraction stale or roster drifted"
        )

    per_model = {}
    losses_total = rungs_total = 0
    exact_err_max = (0.0, None)
    for model in common.EXACT_HOPS:
        exact_hops = common.EXACT_HOPS[model]
        pairs = [ladders[(s, model)] for s in structures]

        exact = [rung_stats(ladder, exact_hops) for ladder in pairs]
        h3 = [rung_stats(ladder, CONVERGED) for ladder in pairs]
        untimed = [e for e in exact if e["speedup"] is None]
        if untimed:
            print(
                f"note: {model} exact-K excludes {len(untimed)} untimed rung(s)",
                file=sys.stderr,
            )
        exact_timed = [e for e in exact if e["speedup"] is not None]
        losses = [e for e in exact_timed if e["speedup"] < 1.0]
        losses_total += len(losses)
        rungs_total += len(exact)
        worst = max(e["hessian_rel_err"] for e in exact)
        if worst > exact_err_max[0]:
            exact_err_max = (worst, model)
        if worst > FP32_CEILING:
            fail(f"{model}: exact-K Hessian error {worst:.3g} exceeds fp32 expectations")

        rungs = {
            hops: rung_summary(pairs, hops)
            for hops in dict.fromkeys((*TRUNCATED_HOPS, exact_hops))
        }
        if rungs[exact_hops]["abs_dcv_pct"]["max"] > EXACT_CV_CEILING_PCT:
            fail(f"{model}: exact-K C_v deviation exceeds {EXACT_CV_CEILING_PCT} %")

        per_model[model] = {
            "exact_K": exact_hops,
            "speedup_exact": spread([e["speedup"] for e in exact_timed]),
            "losses_exact": len(losses),
            "n_pairs": len(exact),
            "n_untimed_exact": len(untimed),
            "coloring_exceeds_hvps_exact": sum(
                bool(e["coloring_dominates"]) for e in exact
            ),
            "coloring_over_hvp_exact": spread(
                [
                    e["coloring_over_hvp"]
                    for e in exact
                    if e["coloring_over_hvp"] is not None
                ]
            ),
            "majority_converged": majority_converged(rungs, exact_hops),
            "zeolites": zeolite_convergence(ladders, structures, model),
            "speedup_h3": spread([e["speedup"] for e in h3]),
            "further_factor_h3_over_exact": spread(
                [
                    e["sparse_s"] / h["sparse_s"]
                    for e, h in zip(exact, h3)
                    if e["sparse_s"] is not None and h["sparse_s"] is not None
                ]
            ),
            # For MACE h4 *is* the exact rung, so its dict has four entries.
            "rungs": {f"h{hops}": summary for hops, summary in rungs.items()},
        }

    worst_h3 = max(
        (
            ((s, m), abs(err_cv(ladders[(s, m)], CONVERGED)))
            for s in structures
            for m in common.EXACT_HOPS
        ),
        key=lambda item: item[1],
    )

    return {
        "structures": sorted(structures),
        "threshold_pct": THRESHOLD_PCT,
        "exact_hessian_rel_err_max": {
            "value": exact_err_max[0],
            "model": exact_err_max[1],
        },
        "losses_at_exact_K": {"count": losses_total, "of": rungs_total},
        "residual_dcv_from_dropped_modes": residual_dcv_origin(ladders, structures),
        "per_model": per_model,
        "worst_case_h3": {
            "structure": worst_h3[0][0],
            "model": worst_h3[0][1],
            "abs_dcv_pct": worst_h3[1],
        },
    }


# ---------------------------------------------------------------------------
# Paragraph-level claims
# ---------------------------------------------------------------------------


def majority_converged(rungs, exact_hops):
    """The smallest rung at which at least MAJORITY of the structures are within
    THRESHOLD_PCT, with the speedup over dense at that rung."""
    for hops, summary in rungs.items():
        if summary["n_within_threshold"] >= MAJORITY * summary["n_pairs"]:
            return {
                "rung": hops,
                "is_exact": hops == exact_hops,
                "n_within_threshold": summary["n_within_threshold"],
                "n_pairs": summary["n_pairs"],
                "speedup": summary["speedup"],
            }
    return None


def zeolite_convergence(ladders, structures, model):
    """Per zeolite, the first rung within THRESHOLD_PCT and the speedup there;
    `slowest` is the one needing the most hops. A rung that passes and a later
    one that fails (a dropped-mode flip) is reported, not hidden."""
    classes = structure_classes()
    out = {}
    for s in structures:
        if classes[s] != "zeolite":
            continue
        ladder = ladders[(s, model)]
        hops = sorted(k for k in ladder if k != "dense")
        within = [abs(err_cv(ladder, k)) <= THRESHOLD_PCT for k in hops]
        first = hops[within.index(True)]
        out[s] = {
            "first_rung_within_threshold": first,
            "speedup_at_first": rung_stats(ladder, first)["speedup"],
            "within_threshold_from_first_on": all(within[within.index(True) :]),
        }
    slowest = max(out, key=lambda s: out[s]["first_rung_within_threshold"])
    return {"per_structure": out, "slowest": {"structure": slowest, **out[slowest]}}


def residual_dcv_origin(ladders, structures):
    """Every truncated rung whose |dC_v| lies between the printed resolution and
    the threshold must differ from dense in the number of dropped modes — the
    soft-mode-crossing explanation the truncation paragraph gives. Fails otherwise."""
    checked = attributed = 0
    for s in structures:
        for model, exact_hops in common.EXACT_HOPS.items():
            ladder = ladders[(s, model)]
            for hops in TRUNCATED_HOPS:
                if hops >= exact_hops:
                    continue
                dcv = abs(err_cv(ladder, hops))
                if RESIDUAL_FLOOR_PCT < dcv <= THRESHOLD_PCT:
                    checked += 1
                    flipped = (
                        ladder[hops]["n_dropped_asr"] != ladder["dense"]["n_dropped_asr"]
                    )
                    attributed += flipped
                    if not flipped:
                        fail(
                            f"{s}/{model} h{hops}: residual |dC_v| {dcv:.3f} % without a "
                            "dropped-mode change — soft-mode explanation contradicted"
                        )
    return {"checked": checked, "with_dropped_mode_change": attributed}


# ---------------------------------------------------------------------------
# Per-rung derivation
# ---------------------------------------------------------------------------


def rung_stats(ladder, hops):
    dense, row = ladder["dense"], ladder[hops]
    sparse_s = ladder_load.sparse_time_s(row)
    # A rung re-derived from a cached Hessian has no timing; its accuracy
    # stats stand, and every timing consumer must skip the Nones explicitly.
    return {
        "speedup": None if sparse_s is None else dense["hessian_cold_s"] / sparse_s,
        "sparse_s": sparse_s,
        "hessian_rel_err": ladder_load.hessian_rel_err(row),
        "coloring_dominates": (
            None
            if row["hessian_cold_s"] is None
            else (row["coloring_s"] or 0.0) > row["hessian_cold_s"]
        ),
        "coloring_over_hvp": (
            None
            if row["hessian_cold_s"] is None
            else (row["coloring_s"] or 0.0) / row["hessian_cold_s"]
        ),
    }


def rung_summary(pairs, hops):
    rows = [(ladder[hops], ladder["dense"]) for ladder in pairs]
    stats = [rung_stats(ladder, hops) for ladder in pairs]
    e = [ladder_load.hessian_rel_err(r) for r, _ in rows]
    cv = [abs(ladder_load.cv_rel_err_pct(r, d)) for r, d in rows]
    share = [contamination_share(r) for r, _ in rows]
    return {
        "n_pairs": len(rows),
        "n_within_threshold": sum(c <= THRESHOLD_PCT for c in cv),
        "speedup": spread([s["speedup"] for s in stats if s["speedup"] is not None]),
        "hessian_rel_err": spread(e),
        "abs_dcv_pct": spread(cv),
        # Relative C_v error over relative Hessian error, both as fractions:
        # ~1 while the truncation is unconverged, 1e-2..1e-3 once it is.
        "dcv_over_hessian_err": spread([c / 100 / h for c, h in zip(cv, e)]),
        "discarded_share_of_sq_err": spread([1 - s for s in share]),
        "contamination_share_of_sq_err": spread(share),
    }


def contamination_share(row):
    e = row["hessian_error"]
    return e["frob_contamination"] ** 2 / e["frob_total"] ** 2


def err_cv(ladder, hops):
    return ladder_load.cv_rel_err_pct(ladder[hops], ladder["dense"])


def spread(values):
    return {"med": median(values), "min": min(values), "max": max(values)}


def fail(message):
    sys.exit(f"ladder_results: {message}")


if __name__ == "__main__":
    main()
