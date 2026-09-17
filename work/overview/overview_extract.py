"""output/ -> results/: block norms, hop matrix, graph, and the schematic's toy chain.

uv run python overview_extract.py --structure mof177
"""

import numpy as np
from jax import ShapeDtypeStruct

import argparse
from pathlib import Path

import asdex
from scipy import sparse as sp

from sadmof.io import read_json, write_json, write_npz

EXP = Path(__file__).resolve().parent
OUTPUT = EXP / "output"
RESULTS = EXP / "results"

# The schematic's toy: a chain, one coordinate per atom, K = 2 truncated at k = 1.
TOY_ATOMS = 7
TOY_K = 2
TOY_TRUNC = 1
TOY_SEED = 0


def main():
    args = build_parser().parse_args()
    src = OUTPUT / args.structure
    record = read_json(src / "record.json")
    graph = np.load(src / "graph.npz")
    hessian = np.load(src / "hessian.npz")["hessian"].astype(np.float64)
    n = record["system"]["n_atoms"]
    K = record["K"]

    norms = np.linalg.norm(hessian, axis=(1, 3)).astype(np.float32)
    hop = graph["hop"]
    assert norms.shape == hop.shape == (n, n)
    assert not np.any(norms[hop < 0] != 0), "nonzero block beyond K"

    RESULTS.mkdir(exist_ok=True)
    write_npz(
        RESULTS / f"{args.structure}.npz",
        {
            "positions": graph["positions"],
            "cell": graph["cell"],
            "numbers": graph["numbers"],
            "edges": graph["edges"],
            "hop": hop,
            "norms": norms,
        },
        compress=True,
    )
    per_hop = {
        k: {
            "pairs": int((hop == k).sum()),
            "median_norm": float(np.median(norms[hop == k])),
            "max_norm": float(norms[hop == k].max()),
        }
        for k in range(K + 1)
    }
    write_json(
        RESULTS / f"{args.structure}.json",
        {
            "structure": args.structure,
            "model": record["model"],
            "K": K,
            "system": record["system"],
            "rungs": record["rungs"],
            "per_hop": per_hop,
            "provenance": record["provenance"],
        },
    )
    write_json(RESULTS / "toy.json", toy_chain(TOY_ATOMS, TOY_K, TOY_TRUNC, TOY_SEED))
    for k, v in per_hop.items():
        print(f"hop {k}: {v['pairs']:6d} pairs, median |Phi| {v['median_norm']:.2e}")
    print(f"wrote {RESULTS}")


# ---------------------------------------------------------------------------
# The toy chain
# ---------------------------------------------------------------------------


def toy_chain(n, K, k_trunc, seed):
    """Exact and truncated star colorings of a path, with what the schematic draws."""
    edges = [(i, i + 1) for i in range(n - 1)]
    i, j = zip(*edges)
    A = sp.csr_matrix((np.ones(2 * len(edges), bool), (i + j, j + i)), shape=(n, n))
    hop = np.full((n, n), -1)
    for h in range(K + 1):
        hop[_reach(A, h).toarray() & (hop < 0)] = h
    # A symmetric toy Hessian on the exact pattern, one decade per hop.
    rng = np.random.default_rng(seed)
    H = np.where(hop >= 0, 10.0 ** (-hop.clip(0)) * rng.uniform(0.5, 1.0, (n, n)), 0.0)
    H = 0.5 * (H + H.T)
    out = {"n": n, "edges": edges, "K": K, "k": k_trunc, "hop": hop.tolist()}
    for tag, hops in (("exact", K), ("trunc", k_trunc)):
        out[tag] = _colored_case(A, H, n, hops)
    return out


def _colored_case(A, H, n, hops):
    pattern = _reach(A, hops).tocoo()
    sparsity = asdex.SparsityPattern.from_coo(
        rows=pattern.row,
        cols=pattern.col,
        shape=(n, n),
        input_avals=(ShapeDtypeStruct((n,), np.float64),),
    )
    coloring = asdex.hessian_coloring_from_sparsity(sparsity, mode="fwd_over_rev")
    colors = np.asarray(coloring.colors)
    seeds = (colors[None, :] == np.arange(coloring.num_colors)[:, None]).astype(float).T
    compressed = H @ seeds
    # asdex decompresses entry m of the pattern from compressed[element, color].
    gather = np.asarray(coloring._gather_indices)
    rows, cols = coloring.sparsity.rows, coloring.sparsity.cols
    recovered = np.zeros((n, n))
    recovered[rows, cols] = compressed[gather[:, 1], gather[:, 0]]
    contaminated = np.argwhere(pattern.toarray() & ~np.isclose(recovered, H))
    reads = [[int(r), int(c), int(e), int(ci)] for r, c, (ci, e) in zip(rows, cols, gather)]
    return {
        "num_colors": int(coloring.num_colors),
        "colors": colors.tolist(),
        "pattern": pattern.toarray().tolist(),
        "compressed": compressed.tolist(),
        "reads": reads,
        "contaminated": contaminated.tolist(),
    }


def _reach(A, hops):
    """Boolean `(I + A)^hops`."""
    n = A.shape[0]
    base = sp.eye(n, dtype=bool, format="csr") + A
    P = sp.eye(n, dtype=bool, format="csr")
    for _ in range(hops):
        P = P @ base
        P.data[:] = True
    return P


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--structure", default="mof177")
    return p


if __name__ == "__main__":
    main()
