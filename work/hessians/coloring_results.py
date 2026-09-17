"""The paper-facing numbers of the atom-graph coloring, re-derived from `results/ladder/`.

What the coloring appendix quotes: that every sparse rung of the benchmark
carries the lifted coloring (identical to the coordinate-level one, so the same
HVPs), how much faster coloring the atom graph is than coloring the coordinate
pattern at each model's exact `K`, and the coloring stage's total over the
whole benchmark under either route. Run as a script to print them as JSON;
exits nonzero when the store is incomplete, a rung is not lifted, or a quoted
claim stops holding, so a stale or partial store cannot quietly produce paper
numbers.

    uv run python work/hessians/coloring_results.py

The coordinate-route time includes the `coloring.npz` write, the lifted one
does not (README); the ratio is over the coloring stage alone, pattern
construction excluded, which is the scope the appendix states.
"""

import argparse
import json
import sys
from statistics import median

import common
import ladder_load
from decay_results import EXTRA_PAIRS
from ladder_results import EXPECTED

# The lifted route is recorded only after `recolor.py`'s identity check passed
# (`common.merge_reused_coloring`), so "every rung lifted" is the identity
# claim of the appendix.
ROUTE = "lifted"

# Coloring the atom graph must never be slower than coloring the coordinate
# pattern; the quoted "3 to 22x, 17x in the median" is read off the output.
RATIO_FLOOR = 1.0
TRUNCATED_HOPS = (1, 2, 3, 4)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(results(), indent=2))


def results():
    ladders = ladder_load.ladders()
    expected = {(s, m) for s in EXPECTED for m in common.EXACT_HOPS} | set(EXTRA_PAIRS)
    present = set(ladders)
    if present != expected:
        fail(
            f"store holds {len(present)} pairs, expected {len(expected)}; "
            f"missing {sorted(expected - present)}, extra {sorted(present - expected)}"
        )

    sparse = []
    for (structure, model), ladder in sorted(ladders.items()):
        rungs = {*TRUNCATED_HOPS, common.EXACT_HOPS[model]}
        if not rungs <= set(ladder):
            fail(f"{structure}/{model}: rungs {sorted(rungs - set(ladder))} missing")
        for hops, row in ladder.items():
            if hops == "dense":
                continue
            route = row.get("coloring_route")
            if route != ROUTE:
                fail(f"{structure}/{model} h{hops}: coloring route is {route!r}")
            if row["coloring_s"] is None or row["coloring_coord_s"] is None:
                fail(f"{structure}/{model} h{hops}: a coloring time is missing")
            sparse.append((structure, model, hops, row))

    exact = [(s, m, row) for s, m, hops, row in sparse if hops == common.EXACT_HOPS[m]]
    ratios = [row["coloring_coord_s"] / row["coloring_s"] for _, _, row in exact]
    if min(ratios) < RATIO_FLOOR:
        fail("atom-graph coloring slower than the coordinate route at exact K")

    slowest = min(exact, key=lambda e: e[2]["coloring_coord_s"] / e[2]["coloring_s"])
    fastest = max(exact, key=lambda e: e[2]["coloring_coord_s"] / e[2]["coloring_s"])
    return {
        "n_pairs": len(ladders),
        "n_sparse_rungs": len(sparse),
        "all_rungs_lifted": True,
        "exact_K": {
            "n": len(exact),
            "coord_over_atom_coloring": spread(ratios),
            "least_gain": f"{slowest[0]}/{slowest[1]}",
            "largest_gain": f"{fastest[0]}/{fastest[1]}",
            # For reference only: the ratio with pattern construction included.
            "coord_over_atom_incl_pattern": spread(
                [
                    (row["pattern_coord_s"] + row["coloring_coord_s"])
                    / (row["pattern_s"] + row["coloring_s"])
                    for _, _, row in exact
                ]
            ),
        },
        "coloring_stage_total_h": {
            "coordinate_route": sum(row["coloring_coord_s"] for *_, row in sparse) / 3600,
            "atom_graph_route": sum(row["coloring_s"] for *_, row in sparse) / 3600,
        },
    }


def spread(values):
    return {"med": median(values), "min": min(values), "max": max(values)}


def fail(message):
    sys.exit(f"coloring_results: {message}")


if __name__ == "__main__":
    main()
