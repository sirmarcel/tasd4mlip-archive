"""`UMACalculator` against the stock `FAIRChemCalculator`.

Energies, forces and stresses are checked against
`data/uma_calculator_reference.json`, pinned from fairchem's own float64
inference path by `data/build_uma_calculator_reference.py` — torch is not in the
test environment. Where `test_uma.py` feeds both sides the same external vesin
graph to isolate the model port, these numbers come from the stock calculator
end to end: its own internal neighbour list, element references and normaliser.
Geometries ride along in the fixture, so ASE's `bulk`/`rattle` cannot move a
structure out from under a pinned number.

The rest is calculator machinery: the Verlet skin (UMA's envelope is exactly
zero beyond the cutoff, so the extra pairs must change nothing), the neighbour
cache under displacement, the strain derivative against finite differences, the
lazy MOLE re-merge when the composition changes, and one short relaxation.
"""

import numpy as np
import jax

import json
from pathlib import Path

import pytest
from ase import Atoms

from sadmof.paths import checkpoint

jax.config.update("jax_enable_x64", True)

CKPT = checkpoint("uma-s-1p2")
REFERENCE_PATH = Path(__file__).parent / "data" / "uma_calculator_reference.json"

pytestmark = pytest.mark.skipif(
    not (CKPT / "model.npz").exists(),
    reason="uma-s-1p2 checkpoint missing (see sources/processed/uma-s-1p2/README.md)",
)


@pytest.fixture(scope="module")
def reference():
    return json.loads(REFERENCE_PATH.read_text())


def _atoms(reference, name) -> Atoms:
    structure = reference["structures"][name]
    return Atoms(
        numbers=structure["numbers"],
        positions=structure["positions"],
        cell=structure["cell"],
        pbc=structure["pbc"],
    )


def _calculator(task="omat", **kwargs):
    from sadmof.models.uma import UMACalculator

    return UMACalculator(CKPT, task=task, default_dtype="float64", **kwargs)


@pytest.mark.parametrize("task", ["omat", "odac"])
@pytest.mark.parametrize("name", ["si8_strained", "RSM0010"])
def test_matches_stock_calculator(name, task, reference):
    expected = reference["tasks"][task][name]
    atoms = _atoms(reference, name)
    atoms.calc = _calculator(task=task, skin=1.0)

    assert abs(atoms.get_potential_energy() - expected["energy"]) < 1e-5
    np.testing.assert_allclose(atoms.get_forces(), expected["forces"], atol=1e-5)
    np.testing.assert_allclose(atoms.get_stress(), expected["stress"], atol=1e-5)


def test_skin_does_not_change_the_energy(reference):
    """The polynomial envelope is exactly zero at and beyond the cutoff, so the
    pairs a Verlet skin captures may not contribute — nor may the padding the
    bucket strategy adds on top of them."""
    atoms = _atoms(reference, "si8_strained")

    tight = atoms.copy()
    tight.calc = _calculator(skin=0.0, bucket_strategy=None)
    wide = atoms.copy()
    wide.calc = _calculator(skin=1.5)

    assert abs(wide.get_potential_energy() - tight.get_potential_energy()) < 1e-12
    np.testing.assert_allclose(wide.get_forces(), tight.get_forces(), atol=1e-12)
    # The wider list really is wider — otherwise the check above is vacuous.
    assert wide.calc._graph["pair_mask"].sum() > tight.calc._graph["pair_mask"].sum()


def test_verlet_cache_matches_a_fresh_calculator(reference):
    """Displace inside the skin so the cached graph is reused, and compare
    against a calculator that has just rebuilt its neighbour list."""
    atoms = _atoms(reference, "si8_strained")
    cached = atoms.copy()
    cached.calc = _calculator(skin=1.0)
    cached.get_potential_energy()

    rng = np.random.default_rng(0)
    displacement = rng.normal(scale=0.05, size=(len(atoms), 3))
    cached.positions += displacement

    fresh = atoms.copy()
    fresh.positions += displacement
    fresh.calc = _calculator(skin=1.0)

    assert not cached.calc._nl_ref.needs_update(cached)
    assert abs(cached.get_potential_energy() - fresh.get_potential_energy()) < 1e-9
    np.testing.assert_allclose(cached.get_forces(), fresh.get_forces(), atol=1e-9)
    np.testing.assert_allclose(cached.get_stress(), fresh.get_stress(), atol=1e-9)


def test_stress_matches_finite_differences(reference):
    """Central differences of the energy under a symmetric strain, i.e. the
    definition the strain-derivative implementation is supposed to satisfy."""
    atoms = _atoms(reference, "si8_strained")
    atoms.calc = _calculator(skin=1.0)
    stress = atoms.get_stress(voigt=False)
    volume = atoms.get_volume()
    h = 1e-4

    def energy_at(strain):
        deformed = atoms.copy()
        deformed.set_cell(np.array(atoms.cell[:]) @ (np.eye(3) + strain), scale_atoms=True)
        deformed.calc = atoms.calc
        return deformed.get_potential_energy()

    for i in range(3):
        for j in range(i, 3):
            strain = np.zeros((3, 3))
            strain[i, j] = strain[j, i] = h
            # A symmetric perturbation moves both off-diagonal entries at once.
            scale = 2 * h if i == j else 4 * h
            finite = (energy_at(strain) - energy_at(-strain)) / (scale * volume)
            assert abs(finite - stress[i, j]) < 1e-6


def test_composition_change_triggers_a_remerge(reference):
    """The MOLE mixing is composition-specific, so one calculator handed two
    different structures has to re-merge rather than reuse the first mixture."""
    calculator = _calculator(skin=1.0)

    for name in ("si8_strained", "RSM0010"):
        atoms = _atoms(reference, name)
        atoms.calc = calculator
        expected = reference["tasks"]["omat"][name]["energy"]
        assert abs(atoms.get_potential_energy() - expected) < 1e-5


def test_relaxation_converges(reference):
    from sadmof.relax import relax

    atoms = _atoms(reference, "si8_strained")
    result = relax(
        atoms,
        _calculator(skin=1.0),
        fmax=0.05,
        max_steps=50,
        optimizer="lbfgs-ls",
        logfile=None,
    )

    assert result["converged"]
    assert result["fmax_final"] < 0.05
