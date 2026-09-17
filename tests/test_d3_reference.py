"""Regression test for the JAX DFT-D3(BJ) port against PyTorch values.

Reference energy / forces / stress come from the upstream `torch-dftd`
`TorchDFTD3Calculator` (xc=pbe, damping=bj, defaults cutoff=95·Bohr /
cnthr=40·Bohr, float64), computed by `tests/data/build_reference.py` and
stored under `tests/data/reference/`. Each `.npz` is self-contained
(geometry + references).

Validates the full `D3Calculator` path in fp64: energy, forces, and
ASE-convention stress. D3 is a deterministic pair sum, so the JAX port
reproduces torch_dftd to ~1e-10; thresholds are set conservatively.

Thresholds in float64:
- Energy: <1e-8 eV/atom
- Forces: <1e-7 eV/Å (max component)
- Stress: <1e-7 eV/Å³ (max Voigt component)
"""

import numpy as np
import jax

from pathlib import Path

import pytest
from ase import Atoms
from ase.stress import full_3x3_to_voigt_6_stress

jax.config.update("jax_enable_x64", True)

REF_DIR = Path(__file__).resolve().parent / "data" / "reference"
REF_FILES = sorted(REF_DIR.glob("*.npz"))

pytestmark = pytest.mark.skipif(not REF_FILES, reason=f"no references in {REF_DIR}")


@pytest.fixture(scope="module")
def calc():
    from sadmof.models.d3 import D3Calculator

    return D3Calculator(default_dtype="float64", skin=0.5)


def _load(path):
    d = dict(np.load(path))
    atoms = Atoms(
        numbers=d["atomic_numbers"],
        positions=d["positions"],
        cell=d["cell"],
        pbc=d["pbc"],
    )
    return atoms, d


@pytest.mark.parametrize("path", REF_FILES, ids=[p.stem for p in REF_FILES])
def test_energy(path, calc):
    atoms, ref = _load(path)
    atoms.calc = calc
    e = atoms.get_potential_energy()

    diff_per_atom = abs(e - float(ref["d3_energy"])) / len(atoms)
    assert diff_per_atom < 1e-8, (
        f"{path.stem}: energy diff/atom = {diff_per_atom:.2e} eV "
        f"(JAX={e:.10f}, ref={float(ref['d3_energy']):.10f})"
    )


@pytest.mark.parametrize("path", REF_FILES, ids=[p.stem for p in REF_FILES])
def test_forces(path, calc):
    atoms, ref = _load(path)
    atoms.calc = calc
    f = atoms.get_forces()

    max_force_diff = np.max(np.abs(f - ref["d3_forces"]))
    assert max_force_diff < 1e-7, f"{path.stem}: max force diff = {max_force_diff:.2e} eV/Å"


@pytest.mark.parametrize("path", REF_FILES, ids=[p.stem for p in REF_FILES])
def test_stress(path, calc):
    atoms, ref = _load(path)
    atoms.calc = calc
    s = atoms.get_stress()  # Voigt 6, ASE convention (virial / volume)

    ref_stress = full_3x3_to_voigt_6_stress(ref["d3_stress"])
    max_stress_diff = np.max(np.abs(s - ref_stress))
    assert max_stress_diff < 1e-7, (
        f"{path.stem}: max stress diff = {max_stress_diff:.2e} eV/Å³"
    )
