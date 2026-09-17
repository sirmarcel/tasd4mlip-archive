"""Smoke tests for `sadmof.models.d3.D3Calculator` and MACE+D3 composition.

Checks that the calculator returns finite energy/forces/stress of the right
shapes, that its energy matches the model called directly, that the Verlet NL
cache is reused on small displacements and rebuilt on large ones, and that
`SumCalculator([mace, d3])` sums the two contributions.
"""

import numpy as np

from pathlib import Path

import pytest
from ase import Atoms

from sadmof.paths import checkpoint

REF_DIR = Path(__file__).resolve().parent / "data" / "reference"
CHECKPOINT_DIR = checkpoint("mace-mp-0-medium")


@pytest.fixture(scope="module")
def atoms():
    s = dict(np.load(REF_DIR / "RSM0304.npz"))  # 58-atom Co MOF
    return Atoms(
        numbers=s["atomic_numbers"],
        positions=s["positions"],
        cell=s["cell"],
        pbc=s["pbc"],
    )


@pytest.fixture(scope="module")
def calc():
    from sadmof.models.d3 import D3Calculator

    return D3Calculator(default_dtype="float64", skin=0.5)


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


def test_calculator_matches_model(calc, atoms):
    from sadmof.models.d3 import D3, atoms_to_inputs

    atoms = atoms.copy()
    atoms.calc = calc
    e_calc = atoms.get_potential_energy()

    model = D3(xc="pbe")
    pos, cell, graph = atoms_to_inputs(atoms, model.cutoff, model.cnthr)
    e_model = float(model.predict(calc._params, pos, cell, graph)["energy"])

    assert abs(e_calc - e_model) < 1e-6


def test_nl_cache_reuse_on_small_displacement(calc, atoms):
    atoms = atoms.copy()
    atoms.calc = calc

    atoms.get_potential_energy()
    graph_before = calc._graph
    centers_before = np.asarray(graph_before["centers"]).copy()

    atoms.rattle(stdev=0.01, seed=0)
    atoms.get_potential_energy()

    assert calc._graph is graph_before, "NL was rebuilt despite a tiny displacement"
    np.testing.assert_array_equal(centers_before, np.asarray(calc._graph["centers"]))


def test_nl_cache_rebuild_on_large_displacement(calc, atoms):
    atoms = atoms.copy()
    atoms.calc = calc

    atoms.get_potential_energy()
    graph_before = calc._graph

    new_positions = atoms.get_positions().copy()
    new_positions[0] += np.array([1.0, 0.0, 0.0])
    atoms.set_positions(new_positions)
    atoms.get_potential_energy()

    assert calc._graph is not graph_before, "NL was not rebuilt for a 1 Å displacement"


def test_sum_calculator_mace_plus_d3(atoms):
    """MACE + D3 via ASE's SumCalculator sums energy/forces/stress."""
    if not (CHECKPOINT_DIR / "model.msgpack").exists():
        pytest.skip(f"MACE checkpoint not built at {CHECKPOINT_DIR}")
    from ase.calculators.mixing import SumCalculator

    from sadmof.models.d3 import D3Calculator
    from sadmof.models.mace import MACECalculator

    mace = MACECalculator.from_checkpoint(CHECKPOINT_DIR, default_dtype="float64", skin=0.5)
    d3 = D3Calculator(default_dtype="float64", skin=0.5)

    a = atoms.copy()
    e_mace = mace.get_potential_energy(a)
    f_mace = mace.get_forces(a)
    e_d3 = d3.get_potential_energy(a)
    f_d3 = d3.get_forces(a)

    a.calc = SumCalculator([mace, d3])
    e_sum = a.get_potential_energy()
    f_sum = a.get_forces()

    assert abs(e_sum - (e_mace + e_d3)) < 1e-6
    np.testing.assert_allclose(f_sum, f_mace + f_d3, atol=1e-8)
