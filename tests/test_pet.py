"""PET integration against the pet-mad-xs checkpoint.

Covers the two things that matter for the sparse-Hessian path:

1. The energy fn reproduces `UPETCalculator` forces/energy (shadow on) — i.e.
   our `(pos, cell, graph)` plumbing matches pet-jax's own forward.
2. The sparse no_shadow Hessian (selected-NL pattern + asdex colouring)
   equals the dense `jax.hessian` reference on the real DOFs.
"""

import numpy as np
import jax

import pytest
from ase.build import bulk

from sadmof.paths import checkpoint

CKPT = checkpoint("pet-mad-xs")

pytestmark = pytest.mark.skipif(
    not (CKPT / "model.msgpack").exists(),
    reason="pet-mad-xs checkpoint missing (see sources/processed/pet-mad-xs/README.md)",
)


def _atoms():
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)  # 8 atoms, periodic
    atoms.rattle(0.1, seed=1)  # break symmetry → nonzero forces
    return atoms


def test_get_calculator_pet():
    from sadmof.models import CALCULATORS, get_calculator

    # Generic `pet` name; the checkpoint selects the variant (XS/S/…).
    assert "pet" in CALCULATORS
    assert "pet+d3" not in CALCULATORS  # PET is standalone, no dispersion

    calc = get_calculator("pet", checkpoint=CKPT, dtype="float64", stress=False)
    atoms = _atoms()
    atoms.calc = calc
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()


def test_energy_fn_matches_calculator():
    from sadmof.models.pet import (
        UPETCalculator,
        atoms_to_inputs,
        get_energy_fn,
        load_pet,
    )

    model, params, metadata = load_pet(CKPT, dtype="float64")
    atoms = _atoms()

    calc = UPETCalculator.from_checkpoint(
        str(CKPT), default_dtype="float64", no_shadow=False, stress=False
    )
    atoms.calc = calc
    f_ref = atoms.get_forces()
    e_ref = atoms.get_potential_energy()

    pos, cell, graph = atoms_to_inputs(atoms, model)
    energy_fn = get_energy_fn(model, no_shadow=False)  # shadow on, matches calc

    e, _ = energy_fn(params, pos, cell, graph)
    grad = jax.grad(lambda p: energy_fn(params, p, cell, graph)[0])(pos)
    n = len(atoms)
    forces = -np.asarray(grad)[:n]

    np.testing.assert_allclose(forces, f_ref, atol=1e-6)
    # Our energy drops the (constant) composition shifts; add them back.
    shift = sum(metadata["shifts"][int(z)] for z in atoms.get_atomic_numbers())
    np.testing.assert_allclose(float(e) + shift, e_ref, rtol=1e-7)


def test_sparse_hessian_matches_dense():
    from sadmof.models.pet import atoms_to_inputs, get_energy_fn, load_pet
    from sadmof.sparse import get_hessian_fn, hessian_coloring

    model, params, _ = load_pet(CKPT, dtype="float64")
    atoms = _atoms()
    pos, cell, graph = atoms_to_inputs(atoms, model)
    n = len(atoms)
    N = pos.shape[0]
    d = 3 * n

    energy_fn = get_energy_fn(model, no_shadow=True)  # sparse path

    # Dense reference.
    H_dense = jax.hessian(lambda p: energy_fn(params, p, cell, graph)[0])(pos)
    H_dense = np.asarray(H_dense).reshape(3 * N, 3 * N)

    # Sparse via selected-NL pattern. hops = 2L + 1; on this small dense cell the
    # pattern is complete, so sparse must equal dense — which means this checks
    # the plumbing, not the hop count (it would pass for any hops >= 1). The hop
    # count is pinned in `test_hop_count`, on a graph deep enough to show it.
    hops = 2 * model.num_gnn_layers + 1
    coloring = hessian_coloring(
        {
            "centers": graph["sel_centers"],
            "others": graph["sel_others"],
            "atomic_numbers": graph["atomic_numbers"],
        },
        hops,
    )
    hessian = get_hessian_fn(energy_fn, coloring)
    H_sparse, overflow = jax.jit(hessian)(params, pos, cell, graph)
    assert not bool(overflow)  # PET aux: k_sel overflow flag
    H_sparse = np.asarray(H_sparse.todense()).reshape(3 * N, 3 * N)

    np.testing.assert_allclose(H_sparse[:d, :d], H_dense[:d, :d], atol=1e-8)
    assert np.abs(H_dense).max() > 1e-3  # sanity: the Hessian isn't trivially zero
