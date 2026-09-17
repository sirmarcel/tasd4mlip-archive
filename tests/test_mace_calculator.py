"""Smoke tests for `sadmof.models.mace.MACECalculator`.

Loads the converted checkpoint, attaches the calculator to a small MOF
frame, and checks that:
- `get_potential_energy` / `get_forces` / `get_stress` return finite
  arrays of the expected shapes,
- the per-element baseline lives outside JIT (energy with the baseline
  is the predict-only energy plus Σ atomic_energies[Z]),
- a small displacement reuses the NL cache,
- a large displacement triggers a rebuild.
"""

import numpy as np

import pytest
from ase.io import read

from sadmof.paths import SOURCES, checkpoint

CHECKPOINT_DIR = checkpoint("mace-mp-0-medium")
STRUCTURES_XYZ = SOURCES / "goennheimer+moosavi" / "structures.xyz"


@pytest.fixture(scope="module")
def calc():
    if not (CHECKPOINT_DIR / "model.msgpack").exists():
        pytest.skip(f"checkpoint not built at {CHECKPOINT_DIR}")
    from sadmof.models.mace import MACECalculator

    return MACECalculator.from_checkpoint(CHECKPOINT_DIR, default_dtype="float64", skin=0.5)


@pytest.fixture(scope="module")
def atoms():
    if not STRUCTURES_XYZ.exists():
        pytest.skip(f"structures not built at {STRUCTURES_XYZ}")
    return read(str(STRUCTURES_XYZ), index=0)


def test_energy_forces_stress(calc, atoms):
    atoms = atoms.copy()
    atoms.calc = calc

    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    s = atoms.get_stress()

    assert np.isfinite(e)
    assert f.shape == (len(atoms), 3)
    assert np.all(np.isfinite(f))
    assert s.shape == (6,)
    assert np.all(np.isfinite(s))


def test_baseline_is_applied_fp64(calc, atoms):
    """Calculator energy should equal predict-energy + Σ atomic_energies[Z]."""
    from sadmof.models.mace import atoms_to_inputs

    atoms = atoms.copy()
    atoms.calc = calc
    total = atoms.get_potential_energy()

    pos, cell, graph = atoms_to_inputs(atoms, calc._cutoff)
    interaction = float(calc._predict_fn(calc._params, pos, cell, graph)["energy"])

    baseline = float(calc._atomic_energies[atoms.get_atomic_numbers()].sum())

    assert abs((total - interaction) - baseline) < 1e-6


def test_nl_cache_reuse_on_small_displacement(calc, atoms):
    atoms = atoms.copy()
    atoms.calc = calc

    atoms.get_potential_energy()  # populate NL
    assert calc._nl_ref is not None
    graph_before = calc._graph
    centers_before = np.asarray(graph_before["centers"]).copy()

    # Tiny rattle — should stay within 0.5 Å skin
    atoms.rattle(stdev=0.01, seed=0)
    atoms.get_potential_energy()

    assert calc._graph is graph_before, "NL was rebuilt despite a tiny displacement"
    np.testing.assert_array_equal(centers_before, np.asarray(calc._graph["centers"]))


def test_nl_cache_rebuild_on_large_displacement(calc, atoms):
    atoms = atoms.copy()
    atoms.calc = calc

    atoms.get_potential_energy()
    graph_before = calc._graph

    # Push atom 0 by 1 Å — well beyond skin
    new_positions = atoms.get_positions().copy()
    new_positions[0] += np.array([1.0, 0.0, 0.0])
    atoms.set_positions(new_positions)
    atoms.get_potential_energy()

    assert calc._graph is not graph_before, "NL was not rebuilt for a 1 Å displacement"
