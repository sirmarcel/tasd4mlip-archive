"""Records + stored Hessians + spectra -> the hop-ladder dataset (tab:cost,
fig:truncation).

One condition per (structure, model): every rung's record scalars — timings,
coloring, sparsity, C_v with and without the acoustic sum rule — joined with
the Hessian error of each sparse rung against the pair's dense reference, and a
fixed-grid DOS histogram per rung. `tab:cost` and `fig:truncation` are slices
of this one dataset, so it is extracted once.

The Hessian error separates into the two parts identified in the methods
section: *discarded* couplings — dense entries outside the
rung's pattern — and *contamination* of retained ones — the deviation of the
recovered values from dense on the pattern. Both are reported as Frobenius
norms plus the largest single entry, in fp64, over the full real-block matrix.
Entries are compared through `worker.pattern_indices`, the exact filter and
order `hessian_sparse.npz` was written in.

Runs where the Hessians are; `results/` comes back:

    python ladder_extract.py --structures goenn:10 giants
    python ladder_extract.py --structures mil101 --models mace
    python ladder_extract.py --structures giants --dry-run

One condition per `<structure>_<model>.npz` + `.json`; existing outputs are
skipped unless `--overwrite`.
"""

import numpy as np

import argparse
import gc
import time
from pathlib import Path

import common
import worker

from sadmof.io import read_json, write_json, write_npz
from sadmof.provenance import provenance

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
RESULTS = HERE / "results" / "ladder"

# DOS grid. 1 cm^-1 resolves the low-frequency region truncation distorts;
# frequencies are non-negative (imaginary modes collapse to 0 upstream).
DOS_BIN_CM1 = 1.0

# Scalar rows of the dense Hessian per pass, bounding the resident slice.
CHUNK_ROWS = 768

# Record fields copied into each rung's row verbatim, as in the records.
FIELDS = ("mode", "hops", "label", "status")
SECTIONS = {
    "config": ("matmul_precision",),
    "system": ("n_pairs_pattern",),
    "sparsity": (
        "nnz",
        "fill",
        "num_colors",
        "coloring_cached",
        "coloring_reused",
        "coloring_route",
    ),
    "hessian": ("n_hvps", "dtype", "raw_asymmetry_rel", "cached"),
    "observables": (
        "n_modes",
        "n_dropped",
        "n_dropped_asr",
        "cv_J_per_gK",
        "cv_J_per_gK_asr",
    ),
    "timings": (
        "import_s",
        "load_inputs_s",
        "pattern_s",
        "atom_coloring_s",
        "lift_s",
        "coloring_s",
        "pattern_coord_s",
        "coloring_coord_s",
        "hessian_cold_s",
        "extract_s",
        "save_s",
        "observables_s",
        "total_s",
    ),
    "memory": ("peak_bytes",),
}


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
            extract(structure, model, out, chunk_rows=args.chunk_rows)
        except Exception as e:  # noqa: BLE001 — one bad condition must not sink the rest
            print(f"  FAILED: {e!r}", flush=True)
            failed.append(out.stem)
        gc.collect()
    if failed:
        print(f"{len(failed)} failed: {failed}")


# ---------------------------------------------------------------------------
# One condition
# ---------------------------------------------------------------------------


def extract(structure, model, out_path, chunk_rows=CHUNK_ROWS):
    """One (structure, model) ladder -> `<out_path>` (DOS) + `.json` (scalars)."""
    t0 = time.perf_counter()
    ladder = load_ladder(structure, model)
    dense = ladder["dense"]
    dense_dir = OUTPUT / structure / dense["label"]
    n_real = dense["system"]["n_atoms"]

    rungs = []
    frob_dense = None
    t_errors = 0.0
    for rung, record in ladder.items():
        row = flatten(rung, record)
        if rung != "dense":
            t = time.perf_counter()
            error = hessian_error(
                dense_dir / "hessian_raw.npy",
                OUTPUT / structure / record["label"],
                n_real,
                chunk_rows,
            )
            t_errors += time.perf_counter() - t
            frob_here = error.pop("frob_dense")
            if frob_dense is None:
                frob_dense = frob_here
            elif abs(frob_here - frob_dense) > 1e-9 * frob_dense:
                raise RuntimeError(
                    f"{record['label']}: dense norm drifted across rungs "
                    f"({frob_here} vs {frob_dense})"
                )
            row["hessian_error"] = error
        rungs.append(row)

    t = time.perf_counter()
    dos, dos_edges = dos_histograms(structure, ladder)
    t_dos = time.perf_counter() - t

    meta = {
        "structure": structure,
        "model": model,
        "exact_hops": common.EXACT_HOPS[model],
        "n_primitive": dense["system"]["n_primitive"],
        "n_atoms": n_real,
        "mult": dense["mult"],
        "mass_amu": dense["system"]["mass_amu"],
        "hessian_dtype": dense["hessian"]["dtype"],
        # Frobenius norm of the dense reference, the denominator every relative
        # error in `ladder_load` divides by.
        "frob_dense": frob_dense,
        "dos_bin_cm1": DOS_BIN_CM1,
        "rungs": rungs,
        "timings_s": {
            "hessian_errors": round(t_errors, 2),
            "dos": round(t_dos, 2),
            "total": round(time.perf_counter() - t0, 2),
        },
        "provenance": provenance(),
    }

    # The `.npz` is the resume gate in `main`, so it is written last: a kill
    # between the two writes then re-runs the condition instead of leaving a
    # "done" condition without its metadata.
    write_json(out_path.with_suffix(".json"), meta)
    write_npz(out_path, {"dos_edges_cm1": dos_edges, **dos}, compress=True)

    worst = max(
        (r["hessian_error"]["frob_total"] / frob_dense, r["rung"])
        for r in rungs
        if r["rung"] != "dense"
    )
    print(
        f"  N={n_real} rungs={[r['rung'] for r in rungs]} "
        f"|H|={frob_dense:.6g} worst rel err {worst[0]:.3g} ({worst[1]}) "
        f"({meta['timings_s']['total']:.0f} s)",
        flush=True,
    )


def load_ladder(structure, model):
    """`{rung: record}` for every rung present, keyed `dense, h1..hK`.

    A missing dense reference is a hard error — every sparse metric is measured
    against it. Sparse rungs are whatever the campaign ran; the deliberately
    partial pairs come out partial here too.
    """
    ladder = {}
    candidates = ["dense"] + [f"h{k}" for k in range(1, common.EXACT_HOPS[model] + 1)]
    for rung in candidates:
        path = OUTPUT / structure / f"{structure}_{model}_{rung}" / "record.json"
        if not path.exists():
            continue
        record = read_json(path)
        if record["status"] != "ok":
            raise RuntimeError(f"{path.parent.name}: record status {record['status']!r}")
        merge = common.merge_reused_coloring(record, path.parent)
        if merge not in ("merged", "unneeded"):
            raise RuntimeError(
                f"{path.parent.name}: coloring sidecar {merge}; "
                "recover the record that actually computed the coloring"
            )
        ladder[rung] = record
    if "dense" not in ladder:
        raise RuntimeError(f"{structure}_{model}: no dense reference rung")
    return ladder


def flatten(rung, record):
    row = {"rung": rung, **{key: record.get(key) for key in FIELDS}}
    for section, keys in SECTIONS.items():
        content = record.get(section) or {}
        for key in keys:
            row[key] = content.get(key)
    return row


# ---------------------------------------------------------------------------
# Hessian error against the dense reference
# ---------------------------------------------------------------------------


def hessian_error(dense_path, rung_dir, n_real, chunk_rows=CHUNK_ROWS):
    """Discarded and contamination parts of one sparse rung's error, in fp64.

    The dense reference is read a row-block at a time off the memory map; the
    rung's values are matched entry for entry through the stored coloring, so
    nothing here densifies the sparse Hessian.
    """
    import asdex

    dim = 3 * n_real
    dense = np.load(dense_path, mmap_mode="r")
    if dense.shape != (dim, dim):
        raise RuntimeError(f"{dense_path}: shape {dense.shape}, geometry says {(dim,) * 2}")

    coloring = asdex.ColoredPattern.load(rung_dir / "coloring.npz")
    rows, cols = worker.pattern_indices(coloring, n_real)
    with np.load(rung_dir / "hessian_sparse.npz") as z:
        if int(z["n_real"]) != n_real:
            raise RuntimeError(f"{rung_dir.name}: n_real {int(z['n_real'])} vs {n_real}")
        data = z["data"]
    if len(data) != len(rows):
        raise RuntimeError(
            f"{rung_dir.name}: {len(data)} values against a {len(rows)}-entry "
            "pattern; the coloring and the Hessian are from different runs"
        )

    order = np.argsort(rows, kind="stable")
    rows, cols = rows[order], cols[order]
    values = data[order].astype(np.float64)

    sq_on = sq_contam = sq_disc = 0.0
    max_contam = max_disc = 0.0
    for start in range(0, dim, chunk_rows):
        stop = min(start + chunk_rows, dim)
        chunk = np.asarray(dense[start:stop]).astype(np.float64)
        lo, hi = np.searchsorted(rows, (start, stop))
        r, c, v = rows[lo:hi] - start, cols[lo:hi], values[lo:hi]

        gathered = chunk[r, c]
        sq_on += float(np.sum(gathered**2))
        diff = v - gathered
        sq_contam += float(np.sum(diff**2))
        max_contam = max(max_contam, float(np.abs(diff).max(initial=0.0)))

        chunk[r, c] = 0.0
        sq_disc += float(np.sum(chunk**2))
        max_disc = max(max_disc, float(np.abs(chunk).max(initial=0.0)))
        del chunk, gathered, diff, r, c, v

    return {
        "n_entries_pattern": int(len(rows)),
        "frob_dense": float(np.sqrt(sq_on + sq_disc)),
        "frob_dense_on_pattern": float(np.sqrt(sq_on)),
        "frob_discarded": float(np.sqrt(sq_disc)),
        "frob_contamination": float(np.sqrt(sq_contam)),
        "frob_total": float(np.sqrt(sq_disc + sq_contam)),
        "max_abs_discarded": max_disc,
        "max_abs_contamination": max_contam,
    }


# ---------------------------------------------------------------------------
# DOS histograms
# ---------------------------------------------------------------------------


def dos_histograms(structure, ladder):
    """Per-rung frequency histograms, both ASR variants, on one shared grid."""
    spectra = {}
    for rung, record in ladder.items():
        with np.load(OUTPUT / structure / record["label"] / "observables.npz") as z:
            spectra[rung] = {
                "": np.asarray(z["frequencies_cm1"]),
                "asr_": np.asarray(z["frequencies_cm1_asr"]),
            }

    top = max(s.max() for by in spectra.values() for s in by.values())
    edges = np.arange(0.0, np.ceil(top) + 2 * DOS_BIN_CM1, DOS_BIN_CM1)
    dos = {}
    for rung, by in spectra.items():
        for prefix, freqs in by.items():
            counts, _ = np.histogram(freqs, bins=edges)
            dos[f"dos_{prefix}{rung}"] = counts.astype(np.int32)
    return dos, edges


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
        default=["goenn:10", "giants"],
        help="roster spec, as run.py: goenn:<k> | giants | <identifier>",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(common.EXACT_HOPS),
        choices=list(common.EXACT_HOPS),
    )
    p.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS, help="dense rows per pass")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


if __name__ == "__main__":
    main()
