"""Regression test for the JAX MACE-MP-0 (medium) port against PyTorch values.

Reference energy / forces / stress come from the upstream torch `mace-torch`
MACE-MP-0 medium model (float64), computed by `tests/data/build_reference.py`
and stored under `tests/data/reference/`. Each `.npz` is self-contained
(geometry + references); geometries originate from the tracked dataset
`sources/processed/goennheimer+moosavi/structures.xyz`.

Validates the full `MACECalculator` path in fp64: total energy (with the
per-element baseline), forces, and ASE-convention stress.

Thresholds in float64:
- Energy: <1e-6 eV/atom
- Forces: <1e-5 eV/Å (max component)
- Stress: <1e-5 eV/Å³ (max Voigt component)
"""

import numpy as np

from pathlib import Path

import pytest
from ase import Atoms
from ase.stress import full_3x3_to_voigt_6_stress

from sadmof.paths import checkpoint

REF_DIR = Path(__file__).resolve().parent / "data" / "reference"
CHECKPOINT_DIR = checkpoint("mace-mp-0-medium")
REF_FILES = sorted(REF_DIR.glob("*.npz"))

pytestmark = pytest.mark.skipif(not REF_FILES, reason=f"no references in {REF_DIR}")


@pytest.fixture(scope="module")
def calc():
    if not (CHECKPOINT_DIR / "model.msgpack").exists():
        pytest.skip(f"checkpoint not built at {CHECKPOINT_DIR}")
    from sadmof.models.mace import MACECalculator

    return MACECalculator.from_checkpoint(CHECKPOINT_DIR, default_dtype="float64", skin=0.5)


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

    diff_per_atom = abs(e - float(ref["mace_energy"])) / len(atoms)
    assert diff_per_atom < 1e-6, (
        f"{path.stem}: energy diff/atom = {diff_per_atom:.2e} eV "
        f"(JAX={e:.8f}, ref={float(ref['mace_energy']):.8f})"
    )


@pytest.mark.parametrize("path", REF_FILES, ids=[p.stem for p in REF_FILES])
def test_forces(path, calc):
    atoms, ref = _load(path)
    atoms.calc = calc
    f = atoms.get_forces()

    max_force_diff = np.max(np.abs(f - ref["mace_forces"]))
    assert max_force_diff < 1e-5, f"{path.stem}: max force diff = {max_force_diff:.2e} eV/Å"


@pytest.mark.parametrize("path", REF_FILES, ids=[p.stem for p in REF_FILES])
def test_stress(path, calc):
    atoms, ref = _load(path)
    atoms.calc = calc
    s = atoms.get_stress()  # Voigt 6, ASE convention (virial / volume)

    ref_stress = full_3x3_to_voigt_6_stress(ref["mace_stress"])
    max_stress_diff = np.max(np.abs(s - ref_stress))
    assert max_stress_diff < 1e-5, (
        f"{path.stem}: max stress diff = {max_stress_diff:.2e} eV/Å³"
    )
