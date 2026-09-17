"""Sum a cached MACE Hessian with the cached D3 Hessian -> `mace_<fp>+d3`.

Host-only, seconds; requires both source conditions done and ok. Writes
`record.json` + `observables.npz` to `output/<structure>/<structure>_mace_<fp>+d3/`;
the summed Hessian is not stored — it is one `np.load` + add away from its
sources, and `ablation_extract.py` rebuilds it the same way.

`run.py` calls this automatically for every structure whose sources are done.

    python combine_d3.py --structure RSM0010 --fp fp64
"""

import numpy as np

import argparse
import time
from pathlib import Path

import common
import worker

from sadmof.io import read_json, write_json

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structure", required=True)
    p.add_argument("--fp", required=True, choices=list(common.COMBINE_VARIANTS))
    p.add_argument(
        "--temperatures", type=float, nargs="+", default=list(common.TEMPERATURES)
    )
    args = p.parse_args()

    t_proc = time.perf_counter()
    base = OUTPUT / args.structure
    out_dir = base / f"{args.structure}_mace_{args.fp}+d3"
    if (out_dir / "record.json").exists():
        print(f"SKIP {out_dir.name}: record.json exists")
        return

    sources = {}
    for role, label in (
        ("mlip", f"{args.structure}_mace_{args.fp}"),
        ("d3", f"{args.structure}_d3"),
    ):
        cond = base / label
        record = read_json(cond / "record.json")  # missing -> a loud error
        if record["status"] != "ok":
            raise SystemExit(f"{role} condition {label}: status {record['status']!r}")
        sources[role] = (cond, record)

    # The worker's full key set (all `None` — no Hessian ran here) plus this
    # step's own stage, so consumers see one timing schema across all records.
    timings = dict.fromkeys(worker.TIMING_KEYS) | {"load_s": None}
    t0 = time.perf_counter()
    h = sources["mlip"][0] / "hessian_raw.npy"
    raw = np.load(h).astype(np.float64)
    raw += np.load(sources["d3"][0] / "hessian_raw.npy").astype(np.float64)
    timings["load_s"] = time.perf_counter() - t0

    from ase.io import read

    atoms = read(str(sources["mlip"][0] / "geometry.xyz"))
    d3_atoms = read(str(sources["d3"][0] / "geometry.xyz"))
    # Guaranteed today by RELAX_TAGS mapping d3 to the MACE relaxation; pinned
    # here because a same-shape geometry mismatch would go unnoticed by numpy.
    assert np.allclose(atoms.positions, d3_atoms.positions) and np.allclose(
        np.asarray(atoms.cell), np.asarray(d3_atoms.cell)
    ), f"{args.structure}: MACE and D3 Hessians are from different geometries"
    out_dir.mkdir(parents=True, exist_ok=True)
    observables = worker.derive_observables(
        raw, atoms.get_masses(), list(args.temperatures), out_dir, timings
    )

    from sadmof.provenance import provenance

    mlip = sources["mlip"][1]
    timings["total_s"] = time.perf_counter() - t_proc
    record = {
        "structure": args.structure,
        "model": "mace",
        "dtype": mlip["dtype"],
        "shadow": None,
        "label": out_dir.name,
        "status": "ok",
        "config": mlip["config"] | {"temperatures_K": list(args.temperatures)},
        "system": mlip["system"],
        "hessian": {
            # The sum lives in fp64 regardless of the MLIP part's precision.
            "n_hvps": None,
            "dtype": "float64",
            "combined": True,
            "raw_asymmetry_rel": worker.asymmetry(raw),
            "cached": None,
            "artifact": None,
        },
        "observables": observables,
        "timings": timings,
        "memory": None,
        "aux_flag": None,
        "oom_where": None,
        "requested_bytes": None,
        "error": None,
        # Labels, not paths: records travel between the cluster and local trees.
        "sources": {role: cond.name for role, (cond, _) in sources.items()},
        "provenance": provenance(),
    }
    write_json(out_dir / "record.json", record)

    cv300 = observables["cv_J_per_gK"].get("300")
    print(f"{out_dir.name}: {{'status': 'ok', 'cv300': {cv300}}}", flush=True)


if __name__ == "__main__":
    main()
