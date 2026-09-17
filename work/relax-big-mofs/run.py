"""Relax the big benchmark MOFs (MOF-177, MOF-210, MIL-100, MIL-101) via `sadmof-relax`.

Same machinery as `relax-goennheimer/run.py` — a frame loop that shells out to
`sadmof.scripts.relax` once per structure (process isolation; `--skip-existing`
gives free resume) — pointed at `sources/processed/big-mofs/structures.xyz`
instead of the Gönnheimer set.

The one strategy change vs the frozen Gönnheimer protocol: **do not use BFGS.**
BFGS is O(N^3)/step (scipy.linalg.eigh on the (3N+9)^2 inverse Hessian) and at
MIL-101's 10821 DOFs that is ~30 s/step — 19626 s total in the prototype run.
`lbfgs-ls` reaches the same minimum in ~1368 s (1.76 s/step), a 14x speedup, so
it is the default here. The cache/verlet settings (skin 1.0, d3_skin 5.0) are
the previously validated values for these exact two MOFs and the package defaults;
they are exposed as flags so the verlet tuning can be re-run if needed.

Canonical big-MOF relaxation (MACE-MP-0 + D3(BJ, PBE), lbfgs-ls, fp64):

    python run.py --calculator mace-mp0+d3

Resume: re-run the same command; structures whose relaxed.xyz already exists are
skipped. Delete it to force a re-run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ase.io import read

from sadmof.relax import OPTIMIZERS

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]  # work/relax-big-mofs -> work -> repo root

STRUCTURES = ROOT / "sources/processed/big-mofs/structures.xyz"

# Same calculator -> (sadmof.models calculator, checkpoint dir) map as
# relax-goennheimer: the experiment name is the --calculator choice and label
# prefix; pet / pet-s share the generic "pet" calculator, differing only by
# checkpoint (and thus trained num_neighbors_adaptive: XS=8, S=16).
CHECKPOINTS = {
    "mace-mp0": ("mace-mp0", ROOT / "sources/processed/mace-mp-0-medium"),
    "mace-mp0+d3": ("mace-mp0+d3", ROOT / "sources/processed/mace-mp-0-medium"),
    "pet": ("pet", ROOT / "sources/processed/pet-mad-xs"),
    "pet-s": ("pet", ROOT / "sources/processed/pet-mad-s"),
}

# Previously validated defaults for these two MOFs (also the package defaults).
DEFAULT_SKIN = 1.0
DEFAULT_D3_SKIN = 5.0


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--calculator", required=True, choices=sorted(CHECKPOINTS))
    # lbfgs-ls, not bfgs: BFGS is O(N^3)/step and unusable at MIL-101 scale.
    p.add_argument("--optimizer", default="lbfgs-ls", choices=sorted(OPTIMIZERS))
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--fmax", type=float, default=0.005)
    p.add_argument("--max-steps", type=int, default=25000)
    p.add_argument(
        "--skin",
        type=float,
        default=DEFAULT_SKIN,
        help=f"base-calculator (MACE/PET) Verlet skin in Å (default: {DEFAULT_SKIN}). "
        "Bigger = fewer NL rebuilds, larger pair lists. Appended to the label as "
        "_skin<s>-<d3> when skin or d3-skin is non-default.",
    )
    p.add_argument(
        "--d3-skin",
        type=float,
        default=DEFAULT_D3_SKIN,
        help=f"D3 Verlet skin in Å for the +d3 composite (default: {DEFAULT_D3_SKIN}).",
    )
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
        help="structures to relax (default: all frames: mof177, mof210, mil100, mil101)",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    nn_suffix = f"_nn{args.num_neighbors}" if args.num_neighbors is not None else ""
    skin_suffix = ""
    if args.skin != DEFAULT_SKIN or args.d3_skin != DEFAULT_D3_SKIN:
        skin_suffix = f"_skin{args.skin:g}-{args.d3_skin:g}"
    label = (
        args.label
        or f"{args.calculator}_{args.optimizer}_{args.dtype}{nn_suffix}{skin_suffix}"
    )
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
        f"(calc={args.calculator}, opt={args.optimizer}, dtype={args.dtype}, "
        f"fmax={args.fmax}, skin={args.skin}, d3_skin={args.d3_skin})"
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
            "--skin",
            str(args.skin),
            "--d3-skin",
            str(args.d3_skin),
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
