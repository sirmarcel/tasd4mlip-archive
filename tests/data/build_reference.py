"""Generate reference energy / forces / stress for the JAX MACE + D3 ports.

References come from the **upstream PyTorch** implementations — `mace-torch`
(MACE-MP-0 medium) and `torch-dftd` (DFT-D3 BJ) — so they are independent of
the JAX code under test in `tests/test_mace_reference.py` /
`tests/test_d3_reference.py`.

Structures are read by identifier from the tracked dataset
`sources/processed/goennheimer+moosavi/structures.xyz`; their geometry is
copied into each `.npz` so the test fixtures are self-contained.

All computation is float64.

Provenance / settings:
- MACE: `mace.calculators.MACECalculator(model_paths=<raw .model>,
  device="cpu", default_dtype="float64")`. The raw checkpoint
  `sources/raw/mace-mp-0-medium/2023-12-03-mace-128-L1_epoch-199.model`
  (gitignored; see its `download.sh`) is the same one converted to the JAX
  msgpack by `sources/processed/mace-mp-0-medium/build.py`.
- D3: `torch_dftd.torch_dftd3_calculator.TorchDFTD3Calculator(device="cpu",
  xc="pbe", damping="bj", dtype=torch.float64)`. Defaults `cutoff=95·Bohr`,
  `cnthr=40·Bohr`, `abc=False`, `bidirectional=True` — matching the JAX
  `models.d3.D3` (cutoff 50.2718 Å, cnthr 21.1671 Å, two-body BJ).

Run (from the repo root):

    uv run --with torch --with mace-torch --with torch-dftd \
        python tests/data/build_reference.py
"""

import numpy as np

from pathlib import Path

import torch
from ase.io import read

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
STRUCTURES_XYZ = (
    REPO_ROOT / "sources" / "processed" / "goennheimer+moosavi" / "structures.xyz"
)
RAW_MACE = (
    REPO_ROOT
    / "sources"
    / "raw"
    / "mace-mp-0-medium"
    / "2023-12-03-mace-128-L1_epoch-199.model"
)
OUT_DIR = HERE / "reference"

# Stable identifiers — a spread of size and metal chemistry.
IDENTIFIERS = ["RSM0004", "RSM0010", "RSM0020", "RSM0304"]


def select_structures():
    frames = read(str(STRUCTURES_XYZ), index=":")
    by_id = {f.info.get("identifier"): f for f in frames}
    out = []
    for ident in IDENTIFIERS:
        if ident not in by_id:
            raise KeyError(f"{ident} not in {STRUCTURES_XYZ}")
        out.append((ident, by_id[ident]))
    return out


def mace_calculator():
    from mace.calculators.mace import MACECalculator

    return MACECalculator(model_paths=str(RAW_MACE), device="cpu", default_dtype="float64")


def d3_calculator():
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

    return TorchDFTD3Calculator(device="cpu", xc="pbe", damping="bj", dtype=torch.float64)


def reference(atoms, calc):
    atoms = atoms.copy()
    atoms.calc = calc
    return {
        "energy": np.float64(atoms.get_potential_energy()),
        "forces": np.asarray(atoms.get_forces(), dtype=np.float64),
        "stress": np.asarray(atoms.get_stress(voigt=False), dtype=np.float64),  # 3×3
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mace = mace_calculator()
    d3 = d3_calculator()

    for ident, atoms in select_structures():
        mace_ref = reference(atoms, mace)
        d3_ref = reference(atoms, d3)
        out = {
            "identifier": ident,
            "n_atoms": np.int32(len(atoms)),
            "positions": np.asarray(atoms.get_positions(), dtype=np.float64),
            "cell": np.asarray(atoms.get_cell()[:], dtype=np.float64),
            "atomic_numbers": np.asarray(atoms.get_atomic_numbers(), dtype=np.int32),
            "pbc": np.asarray(atoms.get_pbc(), dtype=bool),
            "mace_energy": mace_ref["energy"],
            "mace_forces": mace_ref["forces"],
            "mace_stress": mace_ref["stress"],
            "d3_energy": d3_ref["energy"],
            "d3_forces": d3_ref["forces"],
            "d3_stress": d3_ref["stress"],
        }
        path = OUT_DIR / f"{ident}.npz"
        np.savez(path, **out)
        print(
            f"{ident}: N={len(atoms):3d}  "
            f"E_mace={float(mace_ref['energy']):.6f} eV  "
            f"E_d3={float(d3_ref['energy']):.6f} eV  -> {path.name}"
        )


if __name__ == "__main__":
    main()
