"""Selection measurements: one flat JSON record per structure, resumable.

Measures exactly what roster selection and its checks consume (selection.py,
check.py): geometry, class, and the minimal alias-free cell at the model's
exact interaction depth. Positional arguments are .xyz files (single- or
multi-frame; identifiers from `info["identifier"]`).

    uv run python measure.py --out output/records \
        ../../sources/processed/goennheimer+moosavi/structures.xyz
"""

import numpy as np

import argparse
import json
import time
from pathlib import Path

from ase.io import read

from sadmof import supercell as sc
from sadmof.provenance import provenance

# The production MACE adjacency (models.mace.atoms_to_inputs, vesin) and its
# measured exact interaction depth (work/hops).
RC = 6.0
HOPS = 4


def measure_structure(atoms):
    atoms = atoms.copy()
    atoms.wrap()
    n = len(atoms)
    cell = np.asarray(atoms.cell)
    widths = sc.perp_widths(cell)
    volume = abs(float(np.linalg.det(cell)))

    t0 = time.perf_counter()
    edges = sc.mace_edges(atoms, RC)
    r = sc.reach(atoms, edges, HOPS)[-1]  # only exact-K reach matters here
    diffs = sc.differences(r, n)
    m_exact = sc.exact_min_cell(diffs)
    _, fill, aliased = sc.fold_stats(r, n, m_exact)
    assert aliased == 0, "exact minimum not alias-free (bug)"

    return {
        "structure_type": atoms.info.get("structure_type", "unknown"),
        "elements": sorted(set(atoms.get_chemical_symbols())),
        "n_atoms": n,
        "cell": cell.tolist(),
        "widths": widths.tolist(),
        "volume": volume,
        "number_density": n / volume,
        "e_max": edges[3],
        "K": HOPS,
        "R_K": sc.radius(r, atoms),
        "reach_per_atom": len(r[0]) / n,
        "minconv": list(m_exact),
        "minconv_n_atoms": n * int(np.prod(m_exact)),
        "minconv_fill": fill,
        "elapsed_s": time.perf_counter() - t0,
    }


def iter_structures(paths):
    for p in map(Path, paths):
        frames = read(p, index=":")
        for idx, atoms in enumerate(frames):
            fallback = p.stem if len(frames) == 1 else f"{p.stem}-{idx:04d}"
            yield atoms.info.get("identifier", fallback), str(p), atoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("structures", nargs="+", help=".xyz files (multi-frame ok)")
    ap.add_argument("--out", required=True, help="output directory for records")
    ap.add_argument("--force", action="store_true", help="recompute existing records")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = {
        "model": "mace",
        "adjacency": {"rc": RC, "hops": HOPS},
        "provenance": provenance(),
    }

    n_done = n_skip = n_fail = 0
    for name, source, atoms in iter_structures(args.structures):
        record_path = out / f"{name}.json"
        if record_path.exists() and not args.force:
            n_skip += 1
            continue
        try:
            record = measure_structure(atoms)
        except Exception as exc:  # keep the sweep going; report at the end
            print(f"FAIL {name}: {exc}")
            n_fail += 1
            continue
        record = {"structure": name, "geometry": source, **stamp, **record}
        record_path.write_text(json.dumps(record, indent=1))
        n_done += 1
        print(
            f"{name:24s} N={record['n_atoms']:5d} R_K={record['R_K']:5.1f}A "
            f"minconv={tuple(record['minconv'])} ({record['minconv_n_atoms']} atoms) "
            f"{record['elapsed_s']:6.2f}s",
            flush=True,
        )
    print(f"\ndone: {n_done} computed, {n_skip} skipped, {n_fail} failed")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
