"""Enumerate the ablation grid and run it, one worker process at a time, resumably.

Sequential: a cluster job holds one GPU. After the worker loop, the host-side D3
combines run automatically for every structure whose sources are done — they
are idempotent and cost seconds, so there is no flag to forget.

    python run.py --roster goenn:10                  # the full grid
    python run.py --roster goenn:10 --models mace d3 # one family (d3 feeds the combines)
    python run.py --roster RSM0010 --dry-run         # print the worklist
    python run.py --roster goenn:10 --only fp64      # substring-filter labels

A condition is `(structure, model, dtype, shadow)`. Conditions whose
`record.json` exists are skipped; delete it to re-derive observables from the
cached Hessian, delete the condition directory to recompute. A worker that dies
without a record gets a `status: "crashed"` stub, so resubmits do not retry it
forever — delete the stub to retry.

Engine settings are constants in `common.py`, recorded in every record; there
are no knobs for them here. Copied from `work/hessians/run.py`.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import common

from sadmof.io import read_json, write_json

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"


def conditions(args):
    """The worklist: structure x the (model, dtype, shadow, matmul) grid."""
    conds = []
    for structure in common.roster(args.roster):
        for model, dtype, shadow, matmul in common.CONDITIONS:
            if model not in args.models:
                continue
            conds.append(condition(structure, model, dtype, shadow, matmul, args))
    return conds


def condition(structure, model, dtype, shadow, matmul, args):
    label = f"{structure}_{common.variant(model, dtype, shadow, matmul)}"
    out_dir = OUTPUT / structure / label

    argv = [
        "--structure",
        structure,
        "--model",
        model,
        "--dtype",
        dtype,
        "--matmul-precision",
        matmul,
        "--temperatures",
        *[str(t) for t in args.temperatures],
    ]
    if shadow is not None:
        argv += ["--shadow" if shadow else "--no-shadow"]
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
            "dtype": dtype,
            "shadow": shadow,
            "matmul": matmul,
            "label": label,
        },
    }


def combines(args):
    """The `mace_<fp>+d3` combines whose sources are done *ok* and target absent.

    Gated on status, not existence: an OOM source is a recorded result that will
    never combine, and retrying it every submit would keep the failure summary
    noisy forever. Respects `--models` (both sources must be in scope) and
    `--only`, like the worker loop.
    """
    if not {"mace", "d3"} <= set(args.models):
        return []
    todo = []
    for structure in common.roster(args.roster):
        if not _ok(OUTPUT / structure / f"{structure}_d3" / "record.json"):
            continue
        for fp in common.COMBINE_VARIANTS:
            label = f"{structure}_mace_{fp}+d3"
            if args.only and args.only not in label:
                continue
            target = OUTPUT / structure / label / "record.json"
            source = OUTPUT / structure / f"{structure}_mace_{fp}" / "record.json"
            if _ok(source) and not target.exists():
                todo.append((structure, fp))
    return todo


def _ok(record_path):
    return record_path.exists() and read_json(record_path).get("status") == "ok"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--roster",
        nargs="+",
        required=True,
        help="goenn:<k> | <identifier> (repeatable)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(common.MODELS),
        choices=list(common.MODELS),
    )
    p.add_argument("--only", default=None, help="substring filter on condition labels")
    p.add_argument(
        "--skip-crashed",
        action="store_true",
        help="do not retry conditions with a crashed.json from an earlier run",
    )
    p.add_argument("--dry-run", action="store_true")
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
        pending = combines(args)
        if pending:
            print(f"{len(pending)} combines ready now (more once sources land):")
            for structure, fp in pending:
                print(f"  {structure}_mace_{fp}+d3")
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

    pending = combines(args)
    for k, (structure, fp) in enumerate(pending, 1):
        print(f"[combine {k}/{len(pending)}] {structure}_mace_{fp}+d3", flush=True)
        result = subprocess.run(
            [
                sys.executable,
                str(HERE / "combine_d3.py"),
                "--structure",
                structure,
                "--fp",
                fp,
                "--temperatures",
                *[str(t) for t in args.temperatures],
            ],
        )
        if result.returncode != 0:
            print(f"  FAILED ({result.returncode}), continuing", flush=True)
            failed.append(f"{structure}_mace_{fp}+d3")

    if failed:
        print(f"{len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
