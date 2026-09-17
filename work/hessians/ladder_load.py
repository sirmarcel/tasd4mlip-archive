"""`results/ladder/` -> rows, error metrics, and DOS curves. The loader shared
by every ladder consumer (`ladder_table.py`, `ladder_figure.py`).

Rows are the extraction's scalars verbatim; everything derived — rung ordering,
end-to-end sparse cost, relative errors — happens here, in one place.
"""

import numpy as np

import sys
from pathlib import Path

from sadmof.io import read_json

RESULTS = Path(__file__).resolve().parent / "results" / "ladder"


def available():
    """`(structure, model)` pairs present in `results/ladder/`, sorted."""
    out = []
    for path in sorted(RESULTS.glob("*.npz")):
        structure, _, model = path.stem.partition("_")
        out.append((structure, model))
    return out


def load(structure, model):
    """One pair's metadata; DOS arrays stay in the `.npz` until `dos` pulls them."""
    return read_json(RESULTS / f"{structure}_{model}.json")


def ladders():
    """`{(structure, model): {rung: row}}` with `rung` in `"dense", 1..K`.

    Integer keys for sparse rungs, so `ladder[meta["exact_hops"]]` reads the
    exact rung. Each row carries the pair-level scalars (`n_atoms`,
    `frob_dense`, ...) folded in, so consumers index rows alone.
    """
    out = {}
    for structure, model in available():
        meta = load(structure, model)
        pair = {k: v for k, v in meta.items() if k not in ("rungs", "provenance")}
        ladder = {}
        for row in meta["rungs"]:
            rung = "dense" if row["rung"] == "dense" else int(row["rung"][1:])
            ladder[rung] = {**pair, **row}
        out[(structure, model)] = ladder
    return out


def dos(structure, model, rung, asr=True):
    """`(centers_cm1, counts)` for one rung's spectrum, one ASR variant."""
    key = f"dos_{'asr_' if asr else ''}{rung if rung == 'dense' else f'h{rung}'}"
    with np.load(RESULTS / f"{structure}_{model}.npz") as z:
        edges, counts = z["dos_edges_cm1"], z[key]
    return 0.5 * (edges[1:] + edges[:-1]), counts


def sparse_time_s(row):
    """What the sparse rung costs end-to-end that dense does not: pattern,
    coloring, and the Hessian itself. `None` for a re-derived (cached) record."""
    if row["hessian_cold_s"] is None:
        return None
    return (row["pattern_s"] or 0.0) + (row["coloring_s"] or 0.0) + row["hessian_cold_s"]


def cv_rel_err_pct(row, reference, temperature="300", asr=True):
    """Percent C_v error of `row` against `reference`, at one temperature."""
    key = "cv_J_per_gK_asr" if asr else "cv_J_per_gK"
    value = (row.get(key) or {}).get(temperature)
    ref = (reference.get(key) or {}).get(temperature)
    return None if value is None or not ref else 100 * (value - ref) / ref


def hessian_rel_err(row, part="frob_total"):
    """One part of the rung's Hessian error, relative to the dense norm."""
    error = row.get("hessian_error")
    return None if error is None else error[part] / row["frob_dense"]


def complete(ladders_by_pair, models, truncated=(1, 2, 3, 4)):
    """Structures holding dense, every truncated rung, and exact `K` for every
    model, in size order. The rest of the roster was deliberately left partial,
    and an absent rung shown as a gap would read like a result.
    """
    kept, dropped = [], []
    for structure in sort_structures(ladders_by_pair):
        if all(
            _full(ladders_by_pair.get((structure, model)), truncated) for model in models
        ):
            kept.append(structure)
        else:
            dropped.append(structure)
    if dropped:
        print(f"skipping incomplete structures: {', '.join(dropped)}", file=sys.stderr)
    return kept


def _full(ladder, truncated):
    if not ladder:
        return False
    exact = next(iter(ladder.values()))["exact_hops"]
    return {"dense", *truncated, exact} <= set(ladder)


def sort_structures(ladders_by_pair):
    """Structures sorted by supercell size (max `n_atoms` over models), largest
    last, so a cost axis reads left to right as small to big."""
    sizes = {}
    for (structure, _model), ladder in ladders_by_pair.items():
        n = max(row["n_atoms"] for row in ladder.values())
        sizes[structure] = max(sizes.get(structure, 0), n)
    return sorted(sizes, key=sizes.get)
