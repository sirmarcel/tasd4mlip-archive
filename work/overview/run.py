"""Exact PET-XS Hessian of one structure on CPU, for the overview figure.

    JAX_PLATFORMS=cpu uv run python run.py --structure mof177

Resumes by `record.json` existence.
"""

import numpy as np
import jax

import argparse
import sys
from pathlib import Path

import asdex
from scipy import sparse as sp

from sadmof.io import write_json, write_npz
from sadmof.provenance import provenance
from sadmof.sparse import get_hessian_fn, sparsity_pattern

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
OUTPUT = EXP / "output"

# Production construction: relaxed geometry, converged cell, inputs, engine settings.
sys.path.insert(0, str(ROOT / "work" / "hessians"))
import common  # noqa: E402

MODEL = "pet-xs"
jax.config.update("jax_default_matmul_precision", common.MATMUL_PRECISION)
assert jax.config.jax_default_matmul_precision == common.MATMUL_PRECISION


def main():
    args = build_parser().parse_args()
    out_dir = OUTPUT / args.structure
    if (out_dir / "record.json").exists():
        print(f"done already: {out_dir / 'record.json'}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    K = common.EXACT_HOPS[MODEL]
    dtype = np.dtype(common.PRECISION)
    atoms = common.load_relaxed(args.structure, MODEL)
    model, params, energy_fn = common.load_model(MODEL, dtype)
    mult = common.converged_multiplier(atoms, MODEL, model)
    if mult != (1, 1, 1):
        raise SystemExit(f"{args.structure} needs a {mult} supercell for {MODEL} at K={K}")
    pos, cell, graph, pattern_graph = common.build_inputs(MODEL, model, atoms, dtype)
    n = len(atoms)
    n_padded = int(pattern_graph["atomic_numbers"].shape[0])

    centers = np.asarray(pattern_graph["centers"])
    others = np.asarray(pattern_graph["others"])
    real = (centers < n) & (others < n)
    edges = np.stack([centers[real], others[real]])

    rungs = {}
    for k in range(1, K + 1):
        pattern = sparsity_pattern(pattern_graph, hops=k)
        coloring = asdex.hessian_coloring_from_sparsity(pattern, mode=common.AD_MODE)
        rungs[k] = {
            "num_colors": int(coloring.num_colors),
            "fill": float(pattern.nnz / (9 * n_padded * n_padded)),
        }
        print(
            f"k={k}: {coloring.num_colors} colors, fill {rungs[k]['fill']:.3f}", flush=True
        )

    hessian_fn = jax.jit(
        get_hessian_fn(
            energy_fn, coloring, chunk_size=common.CHUNK_SIZE, remat=common.REMAT
        )
    )
    H, aux = hessian_fn(params, pos, cell, graph)
    H.block_until_ready()
    # PET's k_sel overflow flag: a truncated neighbour selection would make the
    # Hessian and its pattern consistently wrong, so the beyond-K check below
    # cannot catch it.
    assert not bool(np.asarray(aux)), "k_sel overflow"

    dense = np.zeros((n_padded, 3, n_padded, 3), dtype=dtype)
    idx, val = np.asarray(H.indices), np.asarray(H.data)
    dense[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = val

    write_npz(out_dir / "hessian.npz", {"hessian": dense[:n, :, :n, :]})
    write_npz(
        out_dir / "graph.npz",
        {
            "positions": atoms.positions,
            "cell": np.asarray(atoms.cell),
            "numbers": atoms.numbers,
            "edges": edges,
            "hop": hop_matrix(edges, n, K),
        },
        compress=True,
    )
    write_json(
        out_dir / "record.json",
        {
            "structure": args.structure,
            "model": MODEL,
            "K": K,
            "config": {
                "precision": common.PRECISION,
                "matmul_precision": common.MATMUL_PRECISION,
                "ad_mode": common.AD_MODE,
                "chunk_size": common.CHUNK_SIZE,
                "remat": common.REMAT,
                "padding": common.PADDING,
                "no_shadow": True,
            },
            "system": {"n_atoms": n, "n_padded": n_padded, "dim": 3 * n},
            "rungs": rungs,
            "provenance": provenance(),
        },
    )
    print(f"wrote {out_dir}")


def hop_matrix(edges, n, K):
    """Smallest `k` with `((I + A)^k)_ij != 0`, `-1` beyond `K`."""
    A = sp.csr_matrix((np.ones(edges.shape[1], bool), (edges[0], edges[1])), shape=(n, n))
    base = sp.eye(n, dtype=bool, format="csr") + A
    reach = sp.eye(n, dtype=bool, format="csr")
    hop = np.full((n, n), -1, dtype=np.int8)
    for k in range(K + 1):
        hop[reach.toarray() & (hop < 0)] = k
        reach = reach @ base
        reach.data[:] = True
    return hop


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--structure", default="mof177")
    return p


if __name__ == "__main__":
    main()
