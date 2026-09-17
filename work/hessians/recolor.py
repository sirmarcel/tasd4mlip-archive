"""Re-time every sparse rung's coloring through the lifted route.

The grid colored the 3N x 3N coordinate pattern with asdex. The lifted route
(`sadmof.sparse.hessian_coloring`) gives the identical colors, so the same HVPs,
at a fraction of the cost. Only the coloring stage is re-measured: per condition,
in a fresh process as the grid timed it, rebuild the inputs, time the pattern
build, atom coloring and lift, check rows, cols and colors against the cached
`coloring.npz`, and write `record.lift.json`, which extraction prices from.

    python recolor.py --dry-run
    python recolor.py --only RSM0023 mil101
    python recolor.py --max-nnz 2e8                # what fits a MIG slice
    python recolor.py --condition output/mil101/mil101_mace_h2   # one, in-process
"""

import numpy as np

import argparse
import resource
import subprocess
import sys
import time
from pathlib import Path

import common

from sadmof.io import read_json, write_json

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
SIDECAR = "record.lift.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only", nargs="+", default=None, help="substring filters on condition labels"
    )
    p.add_argument("--min-nnz", type=float, default=None, help="pattern nnz lower bound")
    p.add_argument("--max-nnz", type=float, default=None, help="pattern nnz upper bound")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--condition", default=None, help="run one condition in-process")
    args = p.parse_args()

    if args.condition:
        recolor(Path(args.condition))
        return

    conds = conditions(args.only, args.min_nnz, args.max_nnz)
    todo = [d for d in conds if not done(d)]
    print(f"{len(conds)} sparse conditions holding a coloring.npz, {len(todo)} to do")
    if args.dry_run:
        for d in todo:
            print(f"  {d.name}")
        return

    failed = []
    for k, d in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {d.name}", flush=True)
        result = subprocess.run([sys.executable, __file__, "--condition", str(d)])
        if result.returncode != 0:
            print(f"  FAILED ({result.returncode}), continuing", flush=True)
            failed.append(d.name)
    if failed:
        print(f"{len(failed)} failed: {failed}")
        sys.exit(1)


def done(condition_dir):
    """A sidecar whose checks passed. A mismatch sidecar is kept, and retried."""
    path = condition_dir / SIDECAR
    if not path.exists():
        return False
    check = read_json(path).get("check") or {}
    return bool(check.get("colors_identical") and check.get("pattern_identical"))


def conditions(only=None, min_nnz=None, max_nnz=None):
    """Sparse `ok` conditions holding a `coloring.npz`, smallest pattern first.

    The nnz bounds split a sweep by host memory, which scales with the pattern.
    """
    found = []
    for record_path in sorted(OUTPUT.glob("*/*/record.json")):
        d = record_path.parent
        # Archived trees under `output/.archive-*`, as `collect.py` skips them.
        if any(part.startswith(".") for part in d.parts):
            continue
        if only and not any(o in d.name for o in only):
            continue
        if not (d / "coloring.npz").exists():
            continue
        record = read_json(record_path)
        if record["mode"] != "sparse" or record["status"] != "ok":
            continue
        nnz = record["sparsity"]["nnz"]
        if (min_nnz is not None and nnz < min_nnz) or (
            max_nnz is not None and nnz >= max_nnz
        ):
            continue
        found.append((nnz, d))
    return [d for _, d in sorted(found, key=lambda t: (t[0], t[1].name))]


def recolor(condition_dir):
    t_proc = time.perf_counter()
    record = read_json(condition_dir / "record.json")
    import jax

    jax.config.update("jax_default_matmul_precision", common.MATMUL_PRECISION)

    import asdex

    from sadmof.provenance import provenance
    from sadmof.sparse import lift_atom_coloring, sparsity_patterns

    t0 = time.perf_counter()
    atoms = common.load_relaxed(record["structure"], record["model"])
    model, _params, _energy_fn = common.load_model(record["model"], np.float32)
    super_atoms = atoms * tuple(record["mult"])
    _pos, _cell, _graph, pattern_graph = common.build_inputs(
        record["model"], model, super_atoms, np.float32
    )
    load_inputs_s = time.perf_counter() - t0
    assert jax.config.jax_default_matmul_precision == common.MATMUL_PRECISION

    t0 = time.perf_counter()
    atom_pattern, pattern = sparsity_patterns(pattern_graph, hops=record["hops"])
    pattern_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    colors, _num_colors, _star_set = asdex.color_symmetric(atom_pattern)
    atom_coloring_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    coloring = lift_atom_coloring(colors, atom_pattern, pattern, mode=common.AD_MODE)
    lift_s = time.perf_counter() - t0

    # Raw arrays: `ColoredPattern.load` would rebuild the star set.
    with np.load(condition_dir / "coloring.npz") as z:
        pattern_identical = bool(
            np.array_equal(z["rows"], np.asarray(pattern.rows))
            and np.array_equal(z["cols"], np.asarray(pattern.cols))
        )
        colors_identical = bool(
            np.array_equal(z["colors"], np.asarray(coloring.colors))
            and int(z["num_colors"]) == int(coloring.num_colors)
        )
        cached_colors = int(z["num_colors"])

    sidecar = {
        "label": condition_dir.name,
        "structure": record["structure"],
        "model": record["model"],
        "hops": record["hops"],
        "coloring_route": "lifted",
        "sparsity": {
            "nnz": int(pattern.nnz),
            "num_colors": int(coloring.num_colors),
            "num_colors_cached": cached_colors,
        },
        "check": {
            "pattern_identical": pattern_identical,
            "colors_identical": colors_identical,
        },
        "timings": {
            "load_inputs_s": load_inputs_s,
            "pattern_s": pattern_s,
            "atom_coloring_s": atom_coloring_s,
            "lift_s": lift_s,
            "coloring_s": atom_coloring_s + lift_s,
            "total_s": time.perf_counter() - t_proc,
        },
        "host_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "provenance": provenance(),
    }
    write_json(condition_dir / SIDECAR, sidecar)
    verdict = "identical" if pattern_identical and colors_identical else "MISMATCH"
    print(
        f"{condition_dir.name}: {coloring.num_colors} colors, {verdict}. "
        f"pattern {pattern_s:.1f} s, coloring {atom_coloring_s + lift_s:.1f} s "
        f"({atom_coloring_s:.1f} atom + {lift_s:.1f} lift), "
        f"coordinate route {coord_coloring_s(record, condition_dir)}",
        flush=True,
    )
    if verdict != "identical":
        sys.exit(2)


def coord_coloring_s(record, condition_dir):
    """The coordinate-route time for the log line: a cached record holds only the
    load time, the measurement is in the TF32 sidecar."""
    if (record.get("sparsity") or {}).get("coloring_cached"):
        tf32_path = condition_dir / "record.tf32.json"
        if not tf32_path.exists():
            return "unknown (cached, no tf32 sidecar)"
        return f"{read_json(tf32_path)['timings']['coloring_s']:.1f} s (tf32 sidecar)"
    return f"{record['timings']['coloring_s']:.1f} s"


if __name__ == "__main__":
    main()
