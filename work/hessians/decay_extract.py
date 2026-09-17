"""Dense Hessians -> block-norm decay against interatomic distance and hop shell.

For every pair the exact pattern connects: the Frobenius norm `||Phi_ij||`, the
coupling distance `d_ij`, and the hop shell `h_ij` (the smallest `k` with
`(I + A)^k` connecting them). `results/` holds a 2D histogram over
`(d_ij, log10 ||Phi_ij||)` per shell, plus the scalars whose exactness matters
(max norm, max distance at a nonzero block, the reach radii `R_h`).

Distances come from `supercell.reach`'s offset-resolved `(src, dst, S)` triples,
not from a minimum-image convention: an alias-free supercell guarantees exactly
one *coupled* image per pair, not that it is the *nearest* one. The two agree at
shallow `K` and part company as the reach outgrows the cell.

Rows are the home cell only. Every other row of the supercell Hessian is its
translate, so `n_prim x N` blocks carry the whole result and the beyond-`K`
check over them is exact.

Runs where the Hessians are; `results/` comes back:

    python decay_extract.py --structures giants          # every dense rung
    python decay_extract.py --structures mil101 --models mace
    python decay_extract.py --structures giants --dry-run

One condition per `<structure>_<model>.npz` + `.json`; existing outputs are
skipped unless `--overwrite`.
"""

import numpy as np

import argparse
import gc
import time
from pathlib import Path

import common

from sadmof.io import read_json, write_json, write_npz
from sadmof.provenance import provenance

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
RESULTS = HERE / "results" / "decay"

# Histogram grid. 0.1 A resolves the first coordination shells. Rounding an
# O(10) eV/A^2 diagonal block puts the fp32 noise floor near 1e-6, so the log
# range spans four decades below it up to the block itself; anything outside is
# clipped into the edge bin and counted in `n_clipped_low` / `n_clipped_high`.
DIST_BIN = 0.1
LOGNORM_LO = -10.0
LOGNORM_HI = 4.0
LOGNORM_BIN = 0.05

# Shell id for pairs the exact pattern does not connect.
BEYOND = 255

# Home-cell rows per pass, bounding the resident slice of the Hessian.
CHUNK = 256


def main():
    args = build_parser().parse_args()
    conditions = [(s, m) for s in common.roster(args.structures) for m in args.models]

    todo = []
    for structure, model in conditions:
        out = RESULTS / f"{structure}_{model}.npz"
        if out.exists() and not args.overwrite:
            print(f"skip {out.name} (exists)")
            continue
        todo.append((structure, model, out))

    print(f"{len(conditions)} conditions, {len(todo)} to extract")
    if args.dry_run:
        for structure, model, out in todo:
            print(f"  {out.stem}")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)

    failed = []
    for k, (structure, model, out) in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {out.stem}", flush=True)
        try:
            extract(structure, model, out, chunk=args.chunk)
        except Exception as e:  # noqa: BLE001 — one bad condition must not sink the rest
            print(f"  FAILED: {e!r}", flush=True)
            failed.append(out.stem)
        gc.collect()
    if failed:
        print(f"{len(failed)} failed: {failed}")


# ---------------------------------------------------------------------------
# One condition
# ---------------------------------------------------------------------------


def extract(structure, model_name, out_path, chunk=CHUNK):
    """Dense rung -> `<out_path>` (arrays) + `<out_path>.json` (scalars)."""
    t0 = time.perf_counter()
    label = f"{structure}_{model_name}_dense"
    record = read_json(OUTPUT / structure / label / "record.json")
    if record["status"] != "ok":
        raise RuntimeError(f"{label}: record status {record['status']!r}")

    system = rebuild_system(structure, model_name, record)

    t = time.perf_counter()
    shell, dist = coupled_maps(system)
    t_pairs = time.perf_counter() - t

    t = time.perf_counter()
    hessian = np.load(OUTPUT / structure / label / "hessian_raw.npy", mmap_mode="r")
    dim = 3 * system["n_real"]
    if hessian.shape != (dim, dim):
        raise RuntimeError(
            f"{label}: Hessian is {hessian.shape}, geometry says {(dim,) * 2}"
        )
    hist = histogram(hessian, shell, dist, system["exact_hops"], chunk=chunk)
    t_hist = time.perf_counter() - t

    meta = {
        "structure": structure,
        "model": model_name,
        "source_label": label,
        "n_atoms": system["n_real"],
        "n_primitive": system["n_prim"],
        "mult": record["mult"],
        "exact_hops": system["exact_hops"],
        # R_h on the primitive cell, per hop h = 1..K: the physical reach of the
        # h-hop pattern, independent of the supercell.
        "reach_radii_A": system["reach_radii"],
        "cell_A": np.asarray(system["cell"]).tolist(),
        "hessian_dtype": record["hessian"]["dtype"],
        "dist_bin_A": DIST_BIN,
        "lognorm_bin": LOGNORM_BIN,
        # Per-shell arrays below run over h = 0..exact_hops, h = 0 being the
        # on-site block; pairs past the exact pattern are the `beyond_*` scalars.
        "pair_count": hist["pair_count"],
        "nonzero_count": hist["nonzero_count"],
        "norm_max": hist["norm_max"],
        "dist_max_nonzero_A": hist["dist_max_nonzero"],
        # The exact pattern's promise, measured: blocks past K hops must be zero.
        "beyond_pairs": hist["beyond_pairs"],
        "beyond_nonzero": hist["beyond_nonzero"],
        "beyond_max_norm": hist["beyond_max_norm"],
        "n_clipped_low": hist["n_clipped_low"],
        "n_clipped_high": hist["n_clipped_high"],
        "timings_s": {
            "coupled_maps": round(t_pairs, 2),
            "histogram": round(t_hist, 2),
            "total": round(time.perf_counter() - t0, 2),
        },
        "provenance": provenance(),
    }

    # The `.npz` is the resume gate in `main`, so it is written last: a kill
    # between the two writes then re-runs the condition instead of leaving a
    # "done" condition without its metadata.
    write_json(out_path.with_suffix(".json"), meta)
    write_npz(
        out_path,
        {
            "dist_edges": hist["dist_edges"],
            "lognorm_edges": hist["lognorm_edges"],
            "hist": counts_as_int32(hist["hist"]),
            "hist_zero": counts_as_int32(hist["hist_zero"]),
        },
        compress=True,
    )

    print(
        f"  N={system['n_real']} rows={system['n_prim']} "
        f"max|Phi|={max(hist['norm_max']):.3g} eV/A^2 "
        f"beyond K: {hist['beyond_nonzero']} nonzero of {hist['beyond_pairs']} "
        f"clipped: {hist['n_clipped_low']} low / {hist['n_clipped_high']} high "
        f"({meta['timings_s']['total']:.0f} s)",
        flush=True,
    )


def rebuild_system(structure, model_name, record):
    """Geometry, supercell and offset-resolved reach as `worker.py` built them.

    Rebuilt rather than stored, since the multiplier is a function of the
    relaxed geometry and only the Hessian was kept. The checks against the
    record catch a Hessian and an analysis describing different systems.
    """
    from sadmof import supercell as sc

    atoms = common.load_relaxed(structure, model_name)
    # PET's adjacency is model-dependent; MACE's is the cutoff, so a MACE-only
    # extraction never touches a checkpoint (nor, therefore, jax).
    model = None
    if model_name != "mace":
        model, _params, _energy_fn = common.load_model(model_name, np.float32)
    hops = common.EXACT_HOPS[model_name]

    reach = common.reach_for(atoms, model_name, model)
    mult = sc.exact_min_cell(sc.differences(reach[-1], len(atoms)))
    if list(mult) != list(record["mult"]):
        raise RuntimeError(
            f"supercell drift: rebuilt {tuple(mult)}, record has {tuple(record['mult'])}"
        )

    super_atoms = atoms * mult
    if len(super_atoms) != record["system"]["n_atoms"]:
        raise RuntimeError(
            f"atom count drift: rebuilt {len(super_atoms)}, "
            f"record has {record['system']['n_atoms']}"
        )

    return {
        "pos_prim": atoms.get_positions(),
        "cell_prim": np.asarray(atoms.cell),
        "pos_super": super_atoms.get_positions(),
        "cell": np.asarray(super_atoms.get_cell()),
        "mult": np.asarray(mult),
        "n_prim": len(atoms),
        "n_real": len(super_atoms),
        "exact_hops": hops,
        "reach": reach,
        "reach_radii": [float(sc.radius(r, atoms)) for r in reach],
    }


# ---------------------------------------------------------------------------
# Home-cell pairs: shell and coupling distance
# ---------------------------------------------------------------------------


def coupled_maps(system):
    """`(shell, distance)`, both `(n_prim, N)`, over home-cell rows.

    `shell` is the smallest `h` whose reach set contains the pair, `BEYOND`
    where none does; `distance` is `|r_dst + S L - r_src|` for that pair's own
    offset `S`, which is the image the Hessian block actually couples through.
    """
    n_prim, n_real = system["n_prim"], system["n_real"]
    pos, cell = system["pos_prim"], system["cell_prim"]

    shell = np.full((n_prim, n_real), BEYOND, dtype=np.uint8)
    dist = np.zeros((n_prim, n_real), dtype=np.float32)

    for h, (src, dst, offsets) in enumerate(system["reach"], 1):
        col = _fold(src, dst, offsets, system, check=(h == len(system["reach"])))
        d = np.linalg.norm(pos[dst] + offsets @ cell - pos[src], axis=1)
        fresh = shell[src, col] == BEYOND
        shell[src[fresh], col[fresh]] = h
        dist[src[fresh], col[fresh]] = d[fresh]

    # Self-coupling is its own shell: on-site force constants at zero distance,
    # which `reach` folds into h = 1 via the identity term.
    diagonal = np.arange(n_prim)
    shell[diagonal, diagonal] = 0
    dist[diagonal, diagonal] = 0.0
    return shell, dist


def _fold(src, dst, offsets, system, check):
    """Supercell column index for primitive atom `dst` at lattice offset `S`.

    Mirrors ase's `Atoms.__mul__` ordering — the `(0,0,0)` image first, then
    `(m0, m1, m2)` row-major over the multiplier — and verifies it against the
    tiled positions rather than trusting it.
    """
    mult, n_prim = system["mult"], system["n_prim"]
    folded = np.mod(offsets, mult)
    image = (folded[:, 0] * mult[1] + folded[:, 1]) * mult[2] + folded[:, 2]
    col = image * n_prim + dst

    if not check:
        return col
    expected = system["pos_prim"][dst] + folded @ system["cell_prim"]
    if not np.allclose(system["pos_super"][col], expected, atol=1e-8):
        raise RuntimeError("supercell atom order does not match ase's tiling")
    # Alias-free means one coupled image per pair; a collision here would make
    # the Hessian block a sum over images and the distance meaningless.
    key = src.astype(np.int64) * system["n_real"] + col
    if len(np.unique(key)) != len(key):
        raise RuntimeError("supercell is not alias-free at the exact hop count")
    return col


# ---------------------------------------------------------------------------
# The histogram
# ---------------------------------------------------------------------------


def histogram(hessian, shell, dist, exact_hops, chunk=CHUNK):
    """`(shell, distance, log10 norm)` counts, plus the exact scalars.

    The Hessian is read a row-block at a time off the memory map, so the
    `(3N, 3N)` array — GB-scale at the giants — is never resident.
    """
    n_prim, n_real = shell.shape
    n_shells = exact_hops + 1  # h = 0..exact_hops; BEYOND is counted, not binned

    coupled = shell != BEYOND
    dist_edges = np.arange(0.0, np.ceil(float(dist[coupled].max())) + DIST_BIN, DIST_BIN)
    n_dist = len(dist_edges) - 1
    lognorm_edges = np.arange(LOGNORM_LO, LOGNORM_HI + LOGNORM_BIN, LOGNORM_BIN)
    n_log = len(lognorm_edges) - 1

    hist = np.zeros((n_shells, n_dist, n_log), dtype=np.int64)
    hist_zero = np.zeros((n_shells, n_dist), dtype=np.int64)
    norm_max = np.zeros(n_shells)
    dist_max_nonzero = np.zeros(n_shells)
    beyond_pairs = beyond_nonzero = 0
    beyond_max_norm = 0.0
    n_clipped_low = n_clipped_high = 0

    for start in range(0, n_prim, chunk):
        stop = min(start + chunk, n_prim)
        block = np.asarray(hessian[3 * start : 3 * stop]).reshape(-1, 3, n_real, 3)
        # Structural zero is tested on the entries, not on the norm: the squares
        # are formed in fp32 and an all-tiny block would underflow to a zero norm
        # while being nonzero. fp64 accumulation for the norm itself.
        filled = np.any(block, axis=(1, 3))
        norms = np.sqrt((block**2).sum(axis=(1, 3), dtype=np.float64))
        del block

        s = shell[start:stop]
        d = dist[start:stop]
        here = s != BEYOND
        past = ~here

        beyond_pairs += int(past.sum())
        if past.any():
            beyond_nonzero += int(np.count_nonzero(filled[past]))
            beyond_max_norm = max(beyond_max_norm, float(norms[past].max()))

        si = s[here].astype(np.int64)
        dh = d[here]
        di = np.minimum((dh / DIST_BIN).astype(np.int64), n_dist - 1)
        v = norms[here]
        nonzero = filled[here]

        log = np.log10(v, where=v > 0.0, out=np.full_like(v, LOGNORM_LO - 1.0))
        raw = np.floor((log - LOGNORM_LO) / LOGNORM_BIN).astype(np.int64)
        n_clipped_low += int(np.count_nonzero(nonzero & (raw < 0)))
        n_clipped_high += int(np.count_nonzero(nonzero & (raw >= n_log)))
        li = np.clip(raw, 0, n_log - 1)

        flat = (si * n_dist + di) * n_log + li
        hist += np.bincount(flat[nonzero], minlength=n_shells * n_dist * n_log).reshape(
            n_shells, n_dist, n_log
        )
        hist_zero += np.bincount(
            (si * n_dist + di)[~nonzero], minlength=n_shells * n_dist
        ).reshape(n_shells, n_dist)

        for index in range(n_shells):
            picked = nonzero & (si == index)
            if picked.any():
                norm_max[index] = max(norm_max[index], float(v[picked].max()))
                dist_max_nonzero[index] = max(
                    dist_max_nonzero[index], float(dh[picked].max())
                )
        del norms, filled, s, d, dh, si, di, v, nonzero, log, raw, li, flat

    return {
        "dist_edges": dist_edges,
        "lognorm_edges": lognorm_edges,
        "hist": hist,
        "hist_zero": hist_zero,
        "pair_count": (hist.sum(axis=(1, 2)) + hist_zero.sum(axis=1)).tolist(),
        "nonzero_count": hist.sum(axis=(1, 2)).tolist(),
        "norm_max": norm_max.tolist(),
        "dist_max_nonzero": dist_max_nonzero.tolist(),
        "beyond_pairs": beyond_pairs,
        "beyond_nonzero": beyond_nonzero,
        "beyond_max_norm": beyond_max_norm,
        "n_clipped_low": n_clipped_low,
        "n_clipped_high": n_clipped_high,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--structures",
        nargs="+",
        default=["giants"],
        help="roster spec, as run.py: goenn:<k> | giants | <identifier>",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(common.EXACT_HOPS),
        choices=list(common.EXACT_HOPS),
    )
    p.add_argument("--chunk", type=int, default=CHUNK, help="home-cell rows per pass")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def counts_as_int32(counts):
    """Narrow a bin-count array for storage; bins hold at most `n_prim * N`."""
    if counts.max(initial=0) > np.iinfo(np.int32).max:
        raise OverflowError("bin counts exceed int32; widen the stored dtype")
    return counts.astype(np.int32)


if __name__ == "__main__":
    main()
