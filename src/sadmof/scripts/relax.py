"""Relax one structure with a named calculator (`sadmof-relax` entrypoint).

Reads a single frame from a structure file (generic ASE frame indexing via
``--index``), relaxes cell + positions, and writes ``relaxed.xyz`` plus a
small ``relax_summary.json`` to the output dir. The CLI knows nothing about
multi-structure sets, identifiers, or `sources/` layout — selecting which
frame/structure to run, and the output tree across many of them, is the
caller's concern.

    sadmof-relax structure.xyz --checkpoint path/to/ckpt -o out
    sadmof-relax structures.xyz --index 7 --calculator mace-mp0 \\
        --optimizer bfgs --fmax 0.005 --checkpoint path/to/ckpt -o out
"""

import argparse
import json
import sys
from pathlib import Path

from ase.io import read, write

from ..models import CALCULATORS, get_calculator
from ..provenance import provenance
from ..relax import OPTIMIZERS, relax


def main(argv=None) -> int:
    args = _parse_args(argv)

    out = Path(args.output_dir)
    if args.skip_existing and (out / "relaxed.xyz").exists():
        print(f"Already relaxed: {out / 'relaxed.xyz'}")
        return 0

    structure = Path(args.structure)
    if not structure.exists():
        print(f"ERROR: {structure} not found", file=sys.stderr)
        return 1

    atoms = read(str(structure), index=args.index)
    if isinstance(atoms, list):
        print(
            f"ERROR: --index {args.index!r} selected {len(atoms)} frames; "
            "the CLI relaxes a single frame",
            file=sys.stderr,
        )
        return 1

    print(f"Structure: {structure.name}  ({len(atoms)} atoms)")
    print(f"Calculator: {args.calculator} ({args.dtype})")
    print(f"Optimizer: {args.optimizer}, fmax={args.fmax}, max_steps={args.max_steps}")

    calc_kwargs = {"skin": args.skin, "d3_skin": args.d3_skin}
    if args.num_neighbors is not None:
        # PET-only: override the adaptive-cutoff target neighbour count.
        calc_kwargs["num_neighbors_adaptive"] = args.num_neighbors
    calc = get_calculator(
        args.calculator,
        checkpoint=args.checkpoint,
        dtype=args.dtype,
        **calc_kwargs,
    )

    out.mkdir(parents=True, exist_ok=True)
    traj = str(out / "optimization.traj") if args.trajectory else None
    result = relax(
        atoms,
        calc,
        fmax=args.fmax,
        max_steps=args.max_steps,
        optimizer=args.optimizer,
        trajectory_path=traj,
    )

    write(str(out / "relaxed.xyz"), atoms)
    summary = {
        "structure": str(structure),
        "index": args.index,
        "calculator": args.calculator,
        "checkpoint": str(args.checkpoint),
        "dtype": args.dtype,
        "optimizer": args.optimizer,
        "fmax_target": args.fmax,
        "max_steps": args.max_steps,
        "num_neighbors": args.num_neighbors,
        "n_atoms": len(atoms),
        **result,
        "provenance": provenance(),
    }
    (out / "relax_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"  {result['n_steps']} steps, {result['wall_s']}s, "
        f"converged={result['converged']}, fmax={result['fmax_final']} eV/Å"
    )
    print(f"Saved: {out / 'relaxed.xyz'}")
    return 0


def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="Relax a single structure with a pluggable ASE calculator",
    )
    p.add_argument("structure", help="Input structure file (.xyz, .cif, …)")
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Folder with the converted MACE model.yaml / model.msgpack pair",
    )
    p.add_argument(
        "--calculator",
        default="mace-mp0+d3",
        choices=CALCULATORS,
        help="Calculator name (default: mace-mp0+d3)",
    )
    p.add_argument(
        "--optimizer",
        default="lbfgs-ls",
        choices=sorted(OPTIMIZERS),
        help="Quasi-Newton optimizer (default: lbfgs-ls). bfgs is O(N^3)/step "
        "and unusable above ~1000 DOFs; lbfgs is fastest; lbfgs-ls is the "
        "safe default for any starting geometry.",
    )
    p.add_argument(
        "--fmax",
        type=float,
        default=0.005,
        help="Force convergence threshold in eV/Å (default: 0.005)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=25000,
        help="Maximum optimizer steps (default: 25000)",
    )
    p.add_argument(
        "--dtype",
        default="float64",
        choices=["float32", "float64"],
        help="Precision (default: float64)",
    )
    p.add_argument(
        "--index",
        default="-1",
        help="Frame index for multi-frame files (default: -1, last frame)",
    )
    p.add_argument("-o", "--output-dir", default="output", help="Output directory")
    p.add_argument("--trajectory", action="store_true", help="Write optimization.traj")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Exit 0 if relaxed.xyz already exists in the output dir",
    )
    p.add_argument(
        "--skin",
        type=float,
        default=1.0,
        help="Verlet skin (Å) for the base calculator (MACE or PET)",
    )
    p.add_argument("--d3-skin", type=float, default=5.0, help="D3 Verlet skin (Å)")
    p.add_argument(
        "--num-neighbors",
        type=int,
        default=None,
        help="PET only: override the adaptive-cutoff target neighbour count "
        "(default: the checkpoint's trained value, e.g. 8 for PET-MAD-XS)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
