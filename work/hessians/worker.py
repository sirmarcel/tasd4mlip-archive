"""One Hessian condition per process (the worker behind `run.py`), fp32.

A condition is `(structure, model, mode, hops)`: geometry -> converged supercell
-> sparsity pattern -> coloring -> Hessian -> frequencies -> C_v, timed end to
end and per stage. Engine settings come from the perf lock and are recorded in
every record so they can be changed later without orphaning old numbers.

An OOM is a recorded result (`status: "oom"`, host or device), not a failure;
condition parameters land in the record before the attempt.
"""

import numpy as np

import argparse
import os
import re
import time
from pathlib import Path

import common

from sadmof.io import atomic_path, write_json, write_npy, write_npz, write_xyz

T_PROC = time.perf_counter()  # end-to-end clock, before any heavy import

# Every timing key is always present, `None` where the stage did not run — a
# consumer should never have to distinguish "absent" from "not applicable".
TIMING_KEYS = (
    "import_s",
    "load_inputs_s",
    "pattern_s",
    "atom_coloring_s",
    "lift_s",
    "coloring_s",
    "save_coloring_s",
    "hessian_cold_s",
    "hessian_warm_s",
    "extract_s",
    "save_s",
    "observables_s",
    "total_s",
)


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    record_path = out_dir / "record.json"

    timings = dict.fromkeys(TIMING_KEYS)
    t0 = time.perf_counter()
    import jax

    timings["import_s"] = time.perf_counter() - t0

    # Read when arrays are first built, so set before load_model/build_inputs.
    jax.config.update("jax_default_matmul_precision", common.MATMUL_PRECISION)

    from sadmof.provenance import provenance

    float_dtype = np.float32  # production Hessian precision; x64 stays off

    t0 = time.perf_counter()
    atoms = common.load_relaxed(args.structure, args.model)
    model, params, energy_fn = common.load_model(args.model, float_dtype)
    if args.supercell == "minconv":
        mult = common.converged_multiplier(atoms, args.model, model)
    else:
        mult = (1, 1, 1)
    super_atoms = atoms * mult
    pos, cell, graph, pattern_graph = common.build_inputs(
        args.model, model, super_atoms, float_dtype
    )
    timings["load_inputs_s"] = time.perf_counter() - t0

    # A library could have mutated the global during input construction.
    assert jax.config.jax_default_matmul_precision == common.MATMUL_PRECISION

    n_real = len(super_atoms)
    n_padded = int(pos.shape[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_xyz(out_dir / "supercell.xyz", super_atoms)

    record = {
        "structure": args.structure,
        "model": args.model,
        "mode": args.mode,
        "hops": args.hops if args.mode == "sparse" else None,
        "supercell": args.supercell,
        "mult": [int(m) for m in mult],
        "label": out_dir.name,
        "status": "ok",
        "config": {
            "precision": common.PRECISION,
            "matmul_precision": common.MATMUL_PRECISION,
            "ad_mode": common.AD_MODE,
            "chunk_size": common.CHUNK_SIZE,
            "remat": common.REMAT,
            "padding": common.PADDING,
            "extra_neighbors": None if args.model == "mace" else common.EXTRA_NEIGHBORS,
            "no_shadow": None if args.model == "mace" else True,
            "coloring_mode": common.AD_MODE if args.mode == "sparse" else None,
            # Requested, not resolved; `None` means unset (jax defaults to 0.75).
            # With `memory.bytes_limit` it pins the arena/module-space split.
            "mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
            "temperatures_K": list(args.temperatures),
            "enforce_asr_stored": bool(args.asr),
        },
        "system": {
            "n_primitive": len(atoms),
            "n_atoms": n_real,
            "n_padded": n_padded,
            "dim": 3 * n_padded,
            # Two different graphs: the model's padded pair list, which the cost
            # laws are fitted on, and the adjacency the pattern is built from.
            # Identical for MACE; for PET the pattern runs on the selected
            # real-real edges, so one number cannot stand for both.
            "n_pairs_padded": int(np.asarray(graph["centers"]).shape[0]),
            "n_pairs_pattern": int(np.asarray(pattern_graph["centers"]).shape[0]),
            "mass_amu": float(super_atoms.get_masses().sum()),
        },
        "sparsity": None,
        "hessian": None,
        "observables": None,
        "timings": timings,
        "memory": None,
        "aux_flag": None,
        "oom_where": None,
        "requested_bytes": None,
        "error": None,
        # After load_model: provenance names the device only once jax is loaded.
        "provenance": provenance(),
    }

    try:
        run_condition(
            args,
            record,
            energy_fn,
            params,
            pos,
            cell,
            graph,
            pattern_graph,
            super_atoms,
            out_dir,
        )
    except MemoryError as e:
        record |= {"status": "oom", "oom_where": "host", "error": repr(e)[:2000]}
    except Exception as e:  # noqa: BLE001 — OOM is data, everything else re-raises
        msg = repr(e)
        if "RESOURCE_EXHAUSTED" not in msg:
            raise
        record |= {"status": "oom", "oom_where": "device", "error": msg[:2000]}
        m = re.search(r"allocat\w+ (\d+) bytes", msg) or re.search(r"(\d+) bytes", msg)
        if m:
            record["requested_bytes"] = int(m.group(1))

    record["memory"] = memory_stats()
    timings["total_s"] = time.perf_counter() - T_PROC
    write_json(record_path, record)

    brief = {
        "status": record["status"],
        "colors": (record["sparsity"] or {}).get("num_colors"),
        "cold_s": timings["hessian_cold_s"],
        "total_s": round(timings["total_s"], 1),
    }
    print(f"{out_dir.name}: {brief}", flush=True)


# ---------------------------------------------------------------------------
# The condition
# ---------------------------------------------------------------------------


def run_condition(
    args, record, energy_fn, params, pos, cell, graph, pattern_graph, super_atoms, out_dir
):
    """Pattern -> coloring -> Hessian -> observables, filling `record` in place."""
    import jax

    timings = record["timings"]
    n_real = len(super_atoms)

    coloring = None
    if args.mode == "sparse":
        coloring = build_coloring(args, record, pattern_graph, n_real, out_dir)

    block, cached = load_or_compute_hessian(
        args, record, energy_fn, params, pos, cell, graph, coloring, n_real, out_dir, jax
    )

    record["hessian"] = {
        "n_hvps": (
            int(coloring.num_colors) if args.mode == "sparse" else 3 * int(pos.shape[0])
        ),
        "dtype": str(block["data"].dtype),
        # Star coloring extracts each unordered pair once from the hub direction
        # and mirrors it, so a sparse Hessian is exactly symmetric and this reads
        # 0.0 by construction — it carries information on dense rungs only.
        "symmetric_by_construction": args.mode == "sparse",
        "raw_asymmetry_rel": 0.0 if args.mode == "sparse" else asymmetry(block["data"]),
        "cached": cached,
        "artifact": artifact_name(args.mode),
    }
    record["observables"] = derive_observables(
        args, timings, block, n_real, super_atoms, out_dir
    )


def build_coloring(args, record, pattern_graph, n_real, out_dir):
    """Sparsity pattern + star coloring, cached to `coloring.npz` for resume.

    `coloring_s` is atom coloring plus lift. The cache write is `save_coloring_s`.
    """
    import asdex

    from sadmof.sparse import lift_atom_coloring, sparsity_patterns

    timings = record["timings"]

    t0 = time.perf_counter()
    atom_pattern, pattern = sparsity_patterns(pattern_graph, hops=args.hops)
    timings["pattern_s"] = time.perf_counter() - t0

    dim = int(pattern.shape[0])
    nnz = int(len(pattern.rows))
    # Padding atoms carry a forced diagonal; subtract it so `fill` describes the
    # real block, which is what the physics and the paper's numbers refer to.
    n_pad_extra = dim // 3 - n_real
    real_nnz = nnz - 9 * n_pad_extra

    cache = out_dir / "coloring.npz"
    if cache.exists():
        t0 = time.perf_counter()
        coloring = asdex.ColoredPattern.load(cache)
        timings["coloring_s"] = (
            time.perf_counter() - t0
        )  # load time, repriced at extraction
        cached = True
        route = None
    else:
        t0 = time.perf_counter()
        colors, _num_colors, _star_set = asdex.color_symmetric(atom_pattern)
        timings["atom_coloring_s"] = time.perf_counter() - t0  # cold: includes numba JIT
        t0 = time.perf_counter()
        coloring = lift_atom_coloring(colors, atom_pattern, pattern, mode=common.AD_MODE)
        timings["lift_s"] = time.perf_counter() - t0
        timings["coloring_s"] = timings["atom_coloring_s"] + timings["lift_s"]
        t0 = time.perf_counter()
        # asdex saves straight to `path`, so route it through the tmp-rename.
        with atomic_path(cache, ".tmp.npz") as tmp:
            coloring.save(tmp)
        timings["save_coloring_s"] = time.perf_counter() - t0
        cached = False
        route = "lifted"

    record["sparsity"] = {
        "nnz": nnz,
        "fill": real_nnz / (3 * n_real) ** 2,
        "fill_padded": nnz / (dim * dim),
        "num_colors": int(coloring.num_colors),
        "coloring_cached": cached,
        "coloring_route": route,
    }
    return coloring


def load_or_compute_hessian(
    args, record, energy_fn, params, pos, cell, graph, coloring, n_real, out_dir, jax
):
    """The real-block Hessian, from the stage cache if one is present.

    Returns:
        `(block, cached)`. `block` is the real-block Hessian as either
        `{"kind": "dense", "data": (3n, 3n) float32}` or
        `{"kind": "sparse", "rows", "cols", "data"}` — the sparse rung is never
        densified here, so nothing allocates `(3n)**2` in float32.
    """
    timings = record["timings"]
    artifact = out_dir / artifact_name(args.mode)
    if artifact.exists():
        return read_artifact(artifact, args.mode, coloring, n_real), True

    if args.mode == "dense":
        from sadmof.dense import get_dense_hessian_fn

        hfn = get_dense_hessian_fn(
            energy_fn, chunk_size=common.CHUNK_SIZE, remat=common.REMAT
        )
    else:
        from sadmof.sparse import get_hessian_fn

        hfn = get_hessian_fn(
            energy_fn, coloring, chunk_size=common.CHUNK_SIZE, remat=common.REMAT
        )
    hfn = jax.jit(hfn)

    t0 = time.perf_counter()
    h, aux = hfn(params, pos, cell, graph)
    jax.block_until_ready((h.data if args.mode == "sparse" else h, aux))
    timings["hessian_cold_s"] = time.perf_counter() - t0  # XLA compile included
    record["aux_flag"] = aux_flag(args.model, aux)

    if args.warm:
        t0 = time.perf_counter()
        h2, aux2 = hfn(params, pos, cell, graph)
        jax.block_until_ready((h2.data if args.mode == "sparse" else h2, aux2))
        timings["hessian_warm_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    block = extract_real_block(h, args.mode, n_real)
    timings["extract_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    write_artifact(artifact, block, n_real)
    timings["save_s"] = time.perf_counter() - t0
    return block, False


def derive_observables(args, timings, block, n_real, super_atoms, out_dir):
    """fp64 Hessian -> frequencies + C_v, with and without the acoustic sum rule.

    Acoustic zeros lifted past `FREQ_THRESHOLD_CM1` stop being dropped, and
    each adds one k_B to C_v; fp32 rounding and truncation both lift them.
    Storing both sets keeps the leak measurable.
    """
    from sadmof.observables import FREQ_THRESHOLD_CM1, cv_from_hessian

    masses = super_atoms.get_masses()
    temps = list(args.temperatures)

    t0 = time.perf_counter()
    freqs, cv = cv_from_hessian(
        hessian_buffer(block, n_real), masses, temps, overwrite=True
    )
    payload = {
        "frequencies_cm1": np.asarray(freqs, dtype=np.float64),
        "masses_amu": np.asarray(masses, dtype=np.float64),
        "temperatures_K": np.asarray(temps, dtype=np.float64),
        "cv_J_per_gK": np.asarray([cv[t] for t in temps], dtype=np.float64),
    }
    cv_asr = freqs_asr = None
    if args.asr:
        freqs_asr, cv_asr = cv_from_hessian(
            hessian_buffer(block, n_real),
            masses,
            temps,
            enforce_asr=True,
            overwrite=True,
        )
        payload["frequencies_cm1_asr"] = np.asarray(freqs_asr, dtype=np.float64)
        payload["cv_J_per_gK_asr"] = np.asarray(
            [cv_asr[t] for t in temps], dtype=np.float64
        )
    timings["observables_s"] = time.perf_counter() - t0

    write_npz(out_dir / "observables.npz", payload)
    return {
        "n_modes": int(len(freqs)),
        # Sub-threshold, not negative: `hessian_to_frequencies` collapses
        # imaginary modes to 0, so a `< 0` count could never fire.
        "n_dropped": n_dropped(freqs, FREQ_THRESHOLD_CM1),
        "n_dropped_asr": n_dropped(freqs_asr, FREQ_THRESHOLD_CM1),
        "cv_J_per_gK": {f"{t:g}": float(cv[t]) for t in temps},
        "cv_J_per_gK_asr": (
            {f"{t:g}": float(cv_asr[t]) for t in temps} if cv_asr else None
        ),
    }


def n_dropped(freqs, threshold):
    return None if freqs is None else int(np.sum(np.asarray(freqs) < threshold))


# ---------------------------------------------------------------------------
# Hessian artifacts
# ---------------------------------------------------------------------------


def artifact_name(mode):
    return "hessian_sparse.npz" if mode == "sparse" else "hessian_raw.npy"


def extract_real_block(h, mode, n_real):
    """Real-block Hessian: dense array, or `(rows, cols, data)` for sparse."""
    if mode == "dense":
        dense = np.asarray(h)[:n_real, :, :n_real, :].reshape(3 * n_real, 3 * n_real)
        return {"kind": "dense", "data": dense}

    idx = np.asarray(h.indices)
    assert idx.shape[1] == 4, f"expected (nnz, 4) BCOO indices, got {idx.shape}"
    data = np.asarray(h.data)
    keep = (idx[:, 0] < n_real) & (idx[:, 2] < n_real)
    idx, data = idx[keep], data[keep]
    return {
        "kind": "sparse",
        "rows": (3 * idx[:, 0] + idx[:, 1]).astype(np.int32),
        "cols": (3 * idx[:, 2] + idx[:, 3]).astype(np.int32),
        "data": data,
    }


def pattern_indices(coloring, n_real):
    """Real-block `(rows, cols)` in the order `extract_real_block` emits.

    The artifact stores only values, so this has to reproduce the BCOO's own
    ordering — asdex emits it in pattern order, which the same filter preserves.
    """
    rows = np.asarray(coloring.sparsity.rows)
    cols = np.asarray(coloring.sparsity.cols)
    keep = (rows < 3 * n_real) & (cols < 3 * n_real)
    return rows[keep].astype(np.int32), cols[keep].astype(np.int32)


def write_artifact(path, block, n_real):
    if block["kind"] == "dense":
        write_npy(path, block["data"])
    else:
        # Indices omitted: `coloring.npz` already holds them, entry for entry.
        write_npz(path, {"data": block["data"], "n_real": np.int32(n_real)})


def read_artifact(path, mode, coloring, n_real):
    if mode == "dense":
        return {"kind": "dense", "data": np.load(path)}
    z = np.load(path)
    rows, cols = pattern_indices(coloring, n_real)
    data = z["data"]
    assert len(data) == len(rows), (
        f"{path}: {len(data)} values against a {len(rows)}-entry pattern; the "
        "coloring and the Hessian are from different runs"
    )
    return {"kind": "sparse", "rows": rows, "cols": cols, "data": data}


def hessian_buffer(block, n_real):
    """Fresh writable float64 `(3n, 3n)` Hessian; `overwrite=True` consumes it,
    so the ASR and no-ASR variants each need their own."""
    if block["kind"] == "dense":
        return block["data"].astype(np.float64)
    w = np.zeros((3 * n_real, 3 * n_real), dtype=np.float64)
    w[block["rows"], block["cols"]] = block["data"]
    return w


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structure", required=True)
    p.add_argument("--model", default="mace", choices=list(common.EXACT_HOPS))
    p.add_argument("--mode", default="sparse", choices=["dense", "sparse"])
    p.add_argument("--hops", type=int, default=None, help="sparse only")
    p.add_argument("--supercell", default="minconv", choices=["minconv", "unit"])
    p.add_argument("--asr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--warm", action="store_true", help="also time a warm call")
    p.add_argument(
        "--temperatures", type=float, nargs="+", default=list(common.TEMPERATURES)
    )
    p.add_argument("--out-dir", required=True)
    return p


def aux_flag(model, aux):
    """PET: the `k_sel` overflow bool, which invalidates the Hessian if set.
    MACE: the per-atom energy sum, a NaN canary."""
    if model == "mace":
        return {"kind": "energy_sum", "value": float(np.sum(np.asarray(aux)))}
    return {"kind": "k_sel_overflow", "value": bool(np.asarray(aux))}


def asymmetry(raw):
    scale = float(np.abs(raw).max())
    return float(np.abs(raw - raw.T).max()) / scale if scale > 0 else 0.0


def memory_stats():
    # Which device this was measured on is recorded by provenance().
    import jax

    stats = jax.local_devices()[0].memory_stats() or {}
    return {
        "peak_bytes": stats.get("peak_bytes_in_use"),
        "bytes_limit": stats.get("bytes_limit"),
    }


if __name__ == "__main__":
    main()
