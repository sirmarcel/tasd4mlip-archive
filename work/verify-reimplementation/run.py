"""Enumerate the parity grid and run it, one worker process at a time, resumably.

    uv run python run.py --roster goenn:1              # prototype slice
    uv run python run.py --roster goenn:10             # a real roster
    uv run python run.py --roster all                  # the full grm2025 set
    uv run python run.py --roster goenn:10 --models mace --dry-run
    uv run python run.py --roster goenn:50 --stacks jax --devices gpu \
        --matmuls default high highest         # the accelerator rungs

A condition is `(stack, model, dtype, device, matmul)` and gets its own
subprocess: torch needs packages the project env deliberately does not have,
and JAX's x64 and matmul flags have to be set before anything is built. Torch
conditions go through a `uv run --with` overlay (`common.OVERLAY`).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import common

HERE = Path(__file__).resolve().parent


def conditions(args):
    """The worklist: stack x model x dtype x (device, matmul), reference first.

    The reference stack is upstream torch in double precision, which is device
    independent to well below the deviations being measured, so torch is only
    ever enumerated on CPU: an accelerator run needs the JAX side alone.
    """
    conds = []
    for stack in args.stacks:
        for model in args.models:
            for dtype in args.dtypes:
                for device, matmul in precisions(args, stack, dtype):
                    conds.append(
                        {
                            "label": f"{stack}/{model}/{common.variant(dtype, device, matmul)}",
                            "stack": stack,
                            "model": model,
                            "dtype": dtype,
                            "device": device,
                            "matmul": matmul,
                            "dir": common.condition_dir(
                                stack, model, dtype, device, matmul
                            ),
                        }
                    )
    return conds


def precisions(args, stack, dtype):
    """`[(device, matmul)]` for one (stack, dtype); `"none"` leaves JAX's own.

    Matmul mode is an fp32 concept, a JAX one, and a no-op on CPU, so torch,
    fp64 and CPU conditions all collapse to `None` rather than accumulating a
    second name for identical arithmetic. On an accelerator the opposite: an
    unset flag *is* `default`, so `none` is pinned to it and the condition
    records which mode it ran under instead of an absence.
    """
    if stack == "torch":
        return [("cpu", None)]
    if dtype == "float64":
        return [(device, None) for device in args.devices]
    pairs = []
    for device in args.devices:
        for m in args.matmuls:
            if device == "cpu":
                pairs.append((device, None))
            else:
                pairs.append((device, "default" if m == "none" else m))
    return list(dict.fromkeys(pairs))


def pending(cond, structures):
    """Structures of the roster this condition has no record for."""
    return [s for s in structures if not (cond["dir"] / f"{s}.npz").exists()]


def command(cond, structures):
    """The invocation for one condition, overlay included where needed."""
    if cond["stack"] == "torch":
        overlay = [x for pkg in common.OVERLAY[cond["model"]] for x in ("--with", pkg)]
        head = ["uv", "run", *overlay, "python", str(HERE / "torch_worker.py")]
        extra = []
    else:
        head = [sys.executable, str(HERE / "jax_worker.py")]
        extra = ["--device", cond["device"]]
        if cond["matmul"] is not None:
            extra += ["--matmul", cond["matmul"]]
    return [
        *head,
        "--model",
        cond["model"],
        "--dtype",
        cond["dtype"],
        *extra,
        "--structures",
        *structures,
    ]


def main():
    args = build_parser().parse_args()
    structures = common.roster(args.roster)
    conds = conditions(args)

    todo = [(c, pending(c, structures)) for c in conds]
    todo = [(c, missing) for c, missing in todo if missing]
    print(f"{len(structures)} structures, {len(conds)} conditions, {len(todo)} incomplete")

    if args.dry_run:
        for c, missing in todo:
            print(f"  {c['label']}: {len(missing)} to compute")
        return

    failed = []
    for k, (cond, missing) in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {cond['label']} ({len(missing)} structures)", flush=True)
        # Per condition, not global, so CPU and GPU rungs share one worklist.
        env = os.environ | {"JAX_PLATFORMS": common.PLATFORM[cond["device"]]}
        result = subprocess.run(command(cond, missing), cwd=common.ROOT, env=env)
        if result.returncode != 0:
            print(f"  FAILED ({result.returncode}), continuing", flush=True)
            failed.append(cond["label"])
    if failed:
        print(f"{len(failed)} failed: {failed}")
        sys.exit(1)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--roster",
        nargs="+",
        default=["goenn:1"],
        help="`goenn:<k>`, `all`, or explicit identifiers (composable)",
    )
    p.add_argument(
        "--models", nargs="+", default=list(common.MODELS), choices=common.MODELS
    )
    p.add_argument(
        "--dtypes", nargs="+", default=list(common.DTYPES), choices=common.DTYPES
    )
    p.add_argument(
        "--stacks", nargs="+", default=list(common.STACKS), choices=common.STACKS
    )
    p.add_argument(
        "--devices",
        nargs="+",
        default=["cpu"],
        choices=common.DEVICES,
        help="JAX side only; the torch reference is always CPU",
    )
    # "none" rather than a bare default: leaving JAX's own setting alone is one
    # of the rungs, and it has to be nameable on the command line like the rest.
    p.add_argument(
        "--matmuls",
        nargs="+",
        default=["none"],
        choices=["none", *common.MATMULS],
        help="fp32 matmul accumulation modes to sweep; `none` leaves JAX's own",
    )
    p.add_argument("--dry-run", action="store_true", help="print the worklist and stop")
    return p


if __name__ == "__main__":
    main()
