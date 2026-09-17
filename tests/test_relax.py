"""Smoke tests for the relaxation driver.

Uses ASE's EMT (no MLIP checkpoint dependency) on a rattled Cu supercell —
nonzero atomic forces so convergence is a real signal, not symmetry-trivial.
"""

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from sadmof.relax import OPTIMIZERS, relax


@pytest.fixture
def rattled_cu():
    atoms = bulk("Cu", "fcc", a=3.6).repeat((2, 2, 2))
    atoms.rattle(0.1, seed=42)
    return atoms


@pytest.mark.parametrize("optimizer", sorted(OPTIMIZERS))
def test_relax_converges(rattled_cu, optimizer):
    result = relax(
        rattled_cu,
        EMT(),
        fmax=0.01,
        max_steps=500,
        optimizer=optimizer,
        logfile=None,
    )
    assert result["converged"]
    assert result["fmax_final"] <= 0.01
    assert result["n_steps"] > 0


def test_unknown_optimizer(rattled_cu):
    with pytest.raises(ValueError, match="unknown optimizer"):
        relax(rattled_cu, EMT(), optimizer="nope", logfile=None)
