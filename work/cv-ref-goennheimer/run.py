"""Run the MACE+D3 -> C_v observables pipeline over the relaxed reference set.

Loops over the canonical relaxed structures (committed in place under
`work/relax-goennheimer/output/mace-mp0+d3_bfgs_float64/`), smallest first —
fast feedback — and shells out to `cv.py` once per structure: process
isolation means one crash/OOM doesn't kill the batch, and JAX recompiles per
structure size anyway. Output goes to `output/<label>/<identifier>.npz`
(label defaults to `dense_<dtype>`).

    python run.py                       # all 233 structures, fp64
    python run.py RSM0004 RSM0010      # subset by identifier
    python run.py --dry-run            # print the worklist with atom counts

Resume: re-run the same command; structures whose npz already exists are
skipped. Delete the npz to force a re-run. `collect.py` compares the results
against the Goennheimer references.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ase.io import read

HERE = Path(__file__).resolve().parent
RELAXED = (
    HERE.parents[1] / "work" / "relax-goennheimer" / "output" / "mace-mp0+d3_bfgs_float64"
)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--remat", action="store_true", help="rematerialise (see cv.py)")
    p.add_argument("--label", default=None, help="output subdir (default: dense_<dtype>)")
    p.add_argument("identifiers", nargs="*", help="default: all relaxed structures")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    label = args.label or f"dense_{args.dtype}"
    out_dir = HERE / "output" / label

    by_id = {
        path.parent.name: read(str(path)) for path in sorted(RELAXED.glob("*/relaxed.xyz"))
    }
    if args.identifiers:
        missing = [i for i in args.identifiers if i not in by_id]
        if missing:
            sys.exit(f"unknown identifiers: {missing}")
        worklist = args.identifiers
    else:
        worklist = list(by_id)
    worklist = sorted(worklist, key=lambda i: len(by_id[i]))  # smallest first

    todo = [i for i in worklist if not (out_dir / f"{i}.npz").exists()]
    print(
        f"{len(worklist)} structures, {len(worklist) - len(todo)} done, {len(todo)} to go"
    )

    if args.dry_run:
        for ident in todo:
            print(f"  {ident}  N={len(by_id[ident])}")
        return

    for k, ident in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {ident} (N={len(by_id[ident])})", flush=True)
        result = subprocess.run(
            [
                sys.executable,
                str(HERE / "cv.py"),
                ident,
                "--out",
                str(out_dir / f"{ident}.npz"),
                "--dtype",
                args.dtype,
                "--chunk-size",
                str(args.chunk_size),
                *(["--remat"] if args.remat else []),
            ],
        )
        if result.returncode != 0:
            print(f"  FAILED ({result.returncode}), continuing", flush=True)


if __name__ == "__main__":
    main()
