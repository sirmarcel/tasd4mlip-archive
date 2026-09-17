"""Enumerate the Hessian grid and run it, one worker process at a time, resumably.

Sequential: a cluster job holds one GPU. A worklist runner, not a scheduler — it
enumerates the per-model hop ladder with collision-proof labels and resumes
after a wall-clock timeout.

    python run.py --roster goenn:10 giants          # the full grid
    python run.py --roster giants --models mace     # one model
    python run.py --roster goenn:2 --hops 1 2       # a cheap slice
    python run.py --roster giants --modes dense     # dense only
    python run.py --roster goenn:10 --dry-run       # print the worklist
    python run.py --roster goenn:10 --only RSM1885  # substring-filter labels

A condition is `(structure, model, mode, hops)`. Conditions whose `record.json`
exists are skipped; delete it to re-derive observables from the cached Hessian,
delete the condition directory to recompute from scratch. A worker that dies
without a record gets a `status: "crashed"` stub, so resubmits do not retry it
forever — delete the stub to retry.

Engine and input settings are constants in `common.py`, recorded in every
record; there are no knobs for them here.

Small runs are allocated by naming what to run (`--roster` / `--models` /
`--hops` / `--modes` / `--only`), which stays legible in the sbatch line.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import common

from sadmof.io import write_json

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"


def conditions(args):
    """The worklist: structure x model x (dense + sparse hop ladder)."""
    conds = []
    for structure in common.roster(args.roster):
        for model in args.models:
            for mode in args.modes:
                hop_list = (
                    [None] if mode == "dense" else common.hop_ladder(model, args.hops)
                )
                for hops in hop_list:
                    conds.append(condition(structure, model, mode, hops, args))
    return conds


def condition(structure, model, mode, hops, args):
    rung = "dense" if mode == "dense" else f"h{hops}"
    # `--supercell` is the one flag that changes what is computed, so it has to
    # be in the label or a `unit` run overwrites the converged one.
    override = "" if args.supercell == "minconv" else f"_{args.supercell}"
    label = f"{structure}_{model}_{rung}{override}"
    out_dir = OUTPUT / structure / label

    argv = [
        "--structure",
        structure,
        "--model",
        model,
        "--mode",
        mode,
        "--supercell",
        args.supercell,
        "--asr" if args.asr else "--no-asr",
        "--temperatures",
        *[str(t) for t in args.temperatures],
    ]
    if mode == "sparse":
        argv += ["--hops", str(hops)]
    if args.warm:
        argv += ["--warm"]

    return {
        "out": out_dir / "record.json",
        "dir": out_dir,
        "args": argv,
        # Duplicates what argv encodes, deliberately: it is what identifies a
        # worker that died before writing its own record.
        "meta": {
            "structure": structure,
            "model": model,
            "mode": mode,
            "hops": hops,
            "supercell": args.supercell,
            "label": label,
        },
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--roster",
        nargs="+",
        required=True,
        help="goenn:<k> | giants | <identifier> (repeatable)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(common.EXACT_HOPS),
        choices=list(common.EXACT_HOPS),
    )
    p.add_argument(
        "--modes", nargs="+", default=["sparse", "dense"], choices=["sparse", "dense"]
    )
    p.add_argument(
        "--hops",
        type=int,
        nargs="+",
        default=None,
        help="override the default ladder {1,2,3,4} + K",
    )
    p.add_argument(
        "--supercell",
        default="minconv",
        choices=["minconv", "unit"],
        help="minconv self-corrects to 1x1x1 where the cell already holds the reach",
    )
    p.add_argument("--only", default=None, help="substring filter on condition labels")
    p.add_argument(
        "--skip-crashed",
        action="store_true",
        help="do not retry conditions with a crashed.json from an earlier run",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--asr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--warm", action="store_true")
    p.add_argument(
        "--temperatures", type=float, nargs="+", default=list(common.TEMPERATURES)
    )
    args = p.parse_args()

    conds = conditions(args)
    if args.only:
        conds = [c for c in conds if args.only in c["dir"].name]
    todo = [c for c in conds if not c["out"].exists()]
    print(f"{len(conds)} conditions, {len(conds) - len(todo)} done")

    crashed = [c for c in todo if (c["dir"] / "crashed.json").exists()]
    if crashed:
        # Retried by default; a real OOM lands as a `status: "oom"` record and
        # so counts as done, not as a crash.
        verb = "skipping" if args.skip_crashed else "retrying"
        print(f"  {len(crashed)} crashed previously, {verb} (see --skip-crashed)")
        if args.skip_crashed:
            todo = [c for c in todo if not (c["dir"] / "crashed.json").exists()]

    if args.dry_run:
        for c in todo:
            print(f"  {c['dir'].name}")
        return

    failed = []
    for k, c in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {c['dir'].name}", flush=True)
        result = subprocess.run(
            [
                sys.executable,
                str(HERE / "worker.py"),
                *c["args"],
                "--out-dir",
                str(c["dir"]),
            ],
        )
        if result.returncode != 0:
            print(f"  FAILED ({result.returncode}), continuing", flush=True)
            failed.append(c["dir"].name)
            # Not `record.json`: a crash must not count as done, or a fixable
            # failure (host OOM-kill, upstream relaxation missing) is skipped
            # forever instead of retried once it is fixed.
            if not c["out"].exists():
                c["dir"].mkdir(parents=True, exist_ok=True)
                stub = c["meta"] | {"status": "crashed", "returncode": result.returncode}
                write_json(c["dir"] / "crashed.json", stub)
    if failed:
        print(f"{len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
