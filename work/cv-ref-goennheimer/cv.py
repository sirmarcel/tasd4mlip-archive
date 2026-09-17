"""Compute MACE+D3 phonon frequencies + C_v for one relaxed structure -> npz.

The worker behind `run.py` (one structure per process — JAX recompiles per
structure size anyway, and a crash/OOM doesn't kill the batch). Dense AD
Hessians for both terms: these are primitive cells (13-240 atoms), fully
coupled at the MACE+D3 interaction ranges, so there is no sparsity to exploit
— `sadmof.dense.get_dense_hessian_fn` is the exact reference path.

The chain matches `tests/test_e2e_cv.py` layer 2: dense MACE Hessian + dense
D3 Hessian on the real-atom block -> `cv_from_hessian` (raw Hessian,
imaginary modes -> 0, |nu| < 1e-3 cm^-1 dropped, gravimetric J/(g.K), no
frequency scaling) on the Goennheimer temperature grid 250-400 K.
"""

import numpy as np

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# The canonical Goennheimer relaxation, committed in place.
RELAXED = ROOT / "work" / "relax-goennheimer" / "output" / "mace-mp0+d3_bfgs_float64"
CHECKPOINT = ROOT / "sources" / "processed" / "mace-mp-0-medium"

TEMPERATURES = np.arange(250.0, 401.0, 10.0)

# MACE-MP-0 medium graph cutoff (Angstrom) — matches the converted checkpoint.
MACE_CUTOFF = 6.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("identifier")
    p.add_argument("--out", required=True, help="output npz path")
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--chunk-size", type=int, default=32)
    # Without remat the linearization residuals are held in full and grow with
    # system size — run 3610386 OOM'd every cell above ~110 atoms on 24 GB
    # (up to 81 GiB requested). Identical results either way (test_hessian.py);
    # remat costs ~+21% compute (2026-06-05 benchmark).
    p.add_argument("--remat", action="store_true")
    args = p.parse_args()

    import jax

    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    float_dtype = np.dtype(args.dtype)

    from ase.io import read
    from marathon.io import from_dict, read_msgpack, read_yaml

    from sadmof.dense import get_dense_hessian_fn
    from sadmof.models.d3 import D3
    from sadmof.models.d3.inputs import atoms_to_inputs as d3_atoms_to_inputs
    from sadmof.models.mace import atoms_to_inputs as mace_atoms_to_inputs
    from sadmof.observables import cv_from_hessian
    from sadmof.provenance import provenance

    atoms = read(str(RELAXED / args.identifier / "relaxed.xyz"))
    assert atoms.info["identifier"] == args.identifier
    n = len(atoms)

    t0 = time.time()
    model = from_dict(read_yaml(str(CHECKPOINT / "model.yaml")))
    params = read_msgpack(str(CHECKPOINT / "model.msgpack"))
    pos, cell, graph = mace_atoms_to_inputs(
        atoms, cutoff=MACE_CUTOFF, float_dtype=float_dtype
    )
    h_mace = np.asarray(
        jax.jit(
            get_dense_hessian_fn(model.energy, chunk_size=args.chunk_size, remat=args.remat)
        )(params, pos, cell, graph)[0]
    )
    wall_mace = time.time() - t0

    t0 = time.time()
    d3 = D3()
    d3_pos, d3_cell, d3_graph = d3_atoms_to_inputs(
        atoms, cutoff=d3.cutoff, cnthr=d3.cnthr, float_dtype=float_dtype
    )
    h_d3 = np.asarray(
        jax.jit(
            get_dense_hessian_fn(d3.energy, chunk_size=args.chunk_size, remat=args.remat)
        )(D3.load_params(), d3_pos, d3_cell, d3_graph)[0]
    )
    wall_d3 = time.time() - t0

    # Slice both to the real-atom block (padding differs between the two graphs).
    hessian = (h_mace[:n, :, :n, :] + h_d3[:n, :, :n, :]).reshape(3 * n, 3 * n)
    hessian = hessian.astype(np.float64)
    freqs, cv = cv_from_hessian(hessian, atoms.get_masses(), TEMPERATURES)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        identifier=args.identifier,
        n_atoms=np.int32(n),
        frequencies_cm1=np.asarray(freqs, dtype=np.float64),
        temperatures=TEMPERATURES,
        cv_gravimetric=np.array([cv[T] for T in TEMPERATURES], dtype=np.float64),
        dtype=args.dtype,
        chunk_size=np.int32(args.chunk_size),
        remat=args.remat,
        n_pad_mace=np.int32(pos.shape[0]),
        n_pad_d3=np.int32(d3_pos.shape[0]),
        wall_hessian_mace_s=np.float64(wall_mace),
        wall_hessian_d3_s=np.float64(wall_d3),
        provenance=json.dumps(provenance()),
    )
    print(
        f"{args.identifier}: N={n:3d}  mace={wall_mace:6.1f}s  d3={wall_d3:5.1f}s  "
        f"cv300={cv[300.0]:.6f}  -> {out}"
    )


if __name__ == "__main__":
    main()
