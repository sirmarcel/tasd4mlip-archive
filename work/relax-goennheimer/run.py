"""Relax the Gönnheimer+Moosavi reference set (233 structures) via `sadmof-relax`.

Flag-driven: pick a calculator / optimizer / precision on the command line.
The output dir defaults to a tag derived from that combination
(`output/<calc>_<opt>_<dtype>/<identifier>/`), overridable with --label. Loops
over frames of the merged reference XYZ and shells out to `sadmof.scripts.relax`
once per structure — process isolation (each MOF is a different N, so JAX
recompiles per structure anyway; one crash/OOM doesn't kill the batch), and
`--skip-existing` gives free resume.

Canonical Gönnheimer reproduction (frozen protocol — matches their
`opt_MACE_d3_sum_BFGS.py`: MACE-MP-0 + D3(BJ, PBE), BFGS, fmax=5e-3, fp64):

    python run.py --calculator mace-mp0+d3 --optimizer bfgs --dtype float64

Resume: re-run the same command; structures whose relaxed.xyz already exists
are skipped. Delete it to force a re-run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ase.io import read

from sadmof.relax import OPTIMIZERS

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]  # work/relax-goennheimer -> work -> repo root

STRUCTURES = ROOT / "sources/processed/goennheimer+moosavi/structures.xyz"

# experiment calculator name -> (sadmof.models calculator, checkpoint dir).
# The experiment name is the --calculator choice and output-label prefix; it can
# be finer-grained than the library calculator — pet (XS) and pet-s (S) both use
# the generic "pet" calculator, differing only by checkpoint (and thus by the
# trained num_neighbors_adaptive: XS=8, S=16).
#
# pet-mad-s's model.msgpack is gitignored (100 MB); it reaches the cluster via
# rsync, not git. See sources/processed/pet-mad-s/README.md.
CHECKPOINTS = {
    "mace-mp0": ("mace-mp0", ROOT / "sources/processed/mace-mp-0-medium"),
    "mace-mp0+d3": ("mace-mp0+d3", ROOT / "sources/processed/mace-mp-0-medium"),
    "pet": ("pet", ROOT / "sources/processed/pet-mad-xs"),
    "pet-s": ("pet", ROOT / "sources/processed/pet-mad-s"),
}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--calculator", required=True, choices=sorted(CHECKPOINTS))
    p.add_argument("--optimizer", default="lbfgs-ls", choices=sorted(OPTIMIZERS))
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--fmax", type=float, default=0.005)
    p.add_argument("--max-steps", type=int, default=25000)
    p.add_argument(
        "--num-neighbors",
        type=int,
        default=None,
        help="PET only: override adaptive-cutoff target neighbour count "
        "(default: checkpoint value, 8 for PET-MAD-XS). Appended to the label as _nn<N>.",
    )
    p.add_argument(
        "--label",
        default=None,
        help="output subdir name (default: <calc>_<opt>_<dtype>)",
    )
    p.add_argument("--trajectory", action="store_true")
    p.add_argument(
        "identifiers",
        nargs="*",
        help="structures to relax (default: all frames in the set)",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    nn_suffix = f"_nn{args.num_neighbors}" if args.num_neighbors is not None else ""
    label = args.label or f"{args.calculator}_{args.optimizer}_{args.dtype}{nn_suffix}"
    calc_name, checkpoint = CHECKPOINTS[args.calculator]

    frames = read(str(STRUCTURES), index=":")
    by_id = {a.info["identifier"]: i for i, a in enumerate(frames)}

    if args.identifiers:
        unknown = [x for x in args.identifiers if x not in by_id]
        if unknown:
            sys.exit(f"unknown identifier(s): {unknown}")
        work = [(by_id[x], x, len(frames[by_id[x]])) for x in args.identifiers]
    else:
        work = [(i, a.info["identifier"], len(a)) for i, a in enumerate(frames)]

    out_root = EXP / "output" / label
    print(
        f"{label}: {len(work)} structures  "
        f"(calc={args.calculator}, opt={args.optimizer}, "
        f"dtype={args.dtype}, fmax={args.fmax})"
    )
    print(f"checkpoint: {checkpoint}")
    if args.dry_run:
        for i, ident, n in work:
            print(f"  [{i:3d}] {ident:18s} N={n}")
        return

    for i, ident, n in work:
        out = out_root / ident
        if (out / "relaxed.xyz").exists():
            print(f"=== {ident} (frame {i}, N={n}) — already relaxed, skip")
            continue
        print(f"\n=== {ident} (frame {i}, N={n})", flush=True)
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "sadmof.scripts.relax",
            str(STRUCTURES),
            "--index",
            str(i),
            "--calculator",
            calc_name,
            "--optimizer",
            args.optimizer,
            "--dtype",
            args.dtype,
            "--fmax",
            str(args.fmax),
            "--max-steps",
            str(args.max_steps),
            "--checkpoint",
            str(checkpoint),
            "--skip-existing",
            "-o",
            str(out),
        ]
        if args.trajectory:
            cmd.append("--trajectory")
        if args.num_neighbors is not None:
            cmd += ["--num-neighbors", str(args.num_neighbors)]
        subprocess.run(cmd, cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
