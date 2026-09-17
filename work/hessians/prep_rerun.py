"""Prepare condition directories for the full-fp32 re-run.

Renames the TF32-era products aside so `run.py` re-runs the Hessian stage while
the spliced `coloring.npz` (precision-independent) stays in place:

    hessian_raw.npy     -> hessian_raw.tf32.npy
    hessian_sparse.npz  -> hessian_sparse.tf32.npz
    observables.npz     -> observables.tf32.npz
    record.json         -> record.tf32.json

Renames only — nothing is deleted. Idempotent: a condition whose record names a
`matmul_precision` is post-re-run and is never touched, and a pair whose source
is already gone counts as moved. `record.json` moves last, so an interrupted
prep leaves the condition visibly unprepped rather than half-invisible.

    python prep_rerun.py                        # dry-run: print the plan
    python prep_rerun.py --apply
    python prep_rerun.py --only mil101 --apply  # canary slice
"""

import argparse
from pathlib import Path

from sadmof.io import read_json

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

# `record.json` last; see module docstring.
RENAMES = (
    ("hessian_raw.npy", "hessian_raw.tf32.npy"),
    ("hessian_sparse.npz", "hessian_sparse.tf32.npz"),
    ("observables.npz", "observables.tf32.npz"),
    ("record.json", "record.tf32.json"),
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", default=None, help="substring filter on condition labels")
    p.add_argument("--apply", action="store_true", help="execute (default: dry-run)")
    args = p.parse_args()

    planned, skipped, blocked = [], [], []
    for cond in conditions(args.only):
        record = read_json(cond / "record.json")
        if (record.get("config") or {}).get("matmul_precision"):
            skipped.append(cond.name)
            continue
        moves, clashes = [], []
        for src, dst in RENAMES:
            if (cond / src).exists() and (cond / dst).exists():
                clashes.append(src)
            elif (cond / src).exists():
                moves.append((src, dst))
        if clashes:
            blocked.append((cond.name, clashes))
        else:
            planned.append((cond, moves))

    verb = "prepping" if args.apply else "would prep (dry-run, see --apply)"
    print(f"{verb} {len(planned)} conditions, {len(skipped)} already re-run")
    for cond, moves in planned:
        print(f"  {cond.name}: {', '.join(src for src, _ in moves)}")
        if args.apply:
            for src, dst in moves:
                (cond / src).rename(cond / dst)
    if blocked:
        for name, clashes in blocked:
            print(f"  BLOCKED {name}: source and target both exist for {clashes}")
        raise SystemExit(f"{len(blocked)} conditions blocked, nothing renamed there")


def conditions(only):
    for record_path in sorted(OUTPUT.glob("*/*/record.json")):
        # Dot-prefixed = archived; Path.glob matches hidden entries.
        if any(part.startswith(".") for part in record_path.parent.parts):
            continue
        if only and only not in record_path.parent.name:
            continue
        yield record_path.parent


if __name__ == "__main__":
    main()
