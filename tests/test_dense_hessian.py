"""Tests for the dense positions-Hessian (sadmof.dense.hessian)."""

import numpy as np
import jax

import pytest

from sadmof.dense import get_dense_hessian_fn
from sadmof.models.mace import atoms_to_inputs
from sadmof.paths import checkpoint
from sadmof.sparse import get_hessian_fn, hessian_coloring

# Hessians/phonons are done in fp64; pin x64 here regardless of test order.
jax.config.update("jax_enable_x64", True)

CHECKPOINT = checkpoint("mace-mp-0-medium")

# MACE-MP-0 medium has 2 interaction layers; the Hessian couples atoms up to
# 2L = 4 graph hops apart, where the reachability pattern is exact.
HOPS = 4


@pytest.fixture(scope="module")
def loaded():
    """Load the MACE model + params once from the processed checkpoint."""
    from marathon.io import from_dict, read_msgpack, read_yaml

    model = from_dict(read_yaml(str(CHECKPOINT / "model.yaml")))
    params = read_msgpack(str(CHECKPOINT / "model.msgpack"))
    return model, params


@pytest.fixture(scope="module")
def si():
    """Bulk silicon (cubic diamond, 8 atoms) → (pos, cell, graph)."""
    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    return atoms_to_inputs(atoms, cutoff=6.0, float_dtype=np.float64)


def _energy_fn(model, params, cell, graph):
    """Scalar energy as a function of positions."""

    def energy(pos):
        return model.energy(params, pos, cell, graph)[0]

    return energy


def test_shape_and_symmetry(loaded, si):
    model, params = loaded
    pos, cell, graph = si
    n = pos.shape[0]
    hessian = get_dense_hessian_fn(model.energy)

    H_raw, aux = jax.jit(hessian)(params, pos, cell, graph)
    H = np.asarray(H_raw)

    assert H.shape == (n, 3, n, 3)
    assert aux.shape == (n,)  # MACE aux: per-atom energies
    H2 = H.reshape(3 * n, 3 * n)
    assert np.allclose(H2, H2.T, atol=1e-5)


def test_matches_jax_hessian(loaded, si):
    """The dense Hessian equals JAX's own (fwd-over-rev) reference."""
    model, params = loaded
    pos, cell, graph = si
    hessian = get_dense_hessian_fn(model.energy)

    H = np.asarray(jax.jit(hessian)(params, pos, cell, graph)[0])
    H_ref = np.asarray(jax.hessian(_energy_fn(model, params, cell, graph))(pos))

    assert np.allclose(H, H_ref, atol=1e-6)


def test_matches_sparse(loaded, si):
    """Dense Hessian equals the sparse (BCOO) Hessian densified — cross-checks
    the two machines."""
    model, params = loaded
    pos, cell, graph = si
    dense = get_dense_hessian_fn(model.energy)
    coloring = hessian_coloring(graph, hops=HOPS)
    sparse = get_hessian_fn(model.energy, coloring)

    H_dense = np.asarray(jax.jit(dense)(params, pos, cell, graph)[0])
    H_sparse = np.asarray(jax.jit(sparse)(params, pos, cell, graph)[0].todense())

    assert np.allclose(H_dense, H_sparse, atol=1e-6)


def test_chunk_size_equivalence(loaded, si):
    """chunk_size only bounds memory; the result must be identical."""
    model, params = loaded
    pos, cell, graph = si
    h_full = get_dense_hessian_fn(model.energy, chunk_size=None)
    h_chunk = get_dense_hessian_fn(model.energy, chunk_size=5)  # 24 dims → 5 chunks

    H_full = np.asarray(jax.jit(h_full)(params, pos, cell, graph)[0])
    H_chunk = np.asarray(jax.jit(h_chunk)(params, pos, cell, graph)[0])

    assert np.allclose(H_full, H_chunk, atol=1e-6)


def test_remat_equivalence(loaded, si):
    """remat only trades compute for memory; the result must be identical."""
    model, params = loaded
    pos, cell, graph = si
    h_remat = get_dense_hessian_fn(model.energy, remat=True)
    h_plain = get_dense_hessian_fn(model.energy, remat=False)

    H_remat = np.asarray(jax.jit(h_remat)(params, pos, cell, graph)[0])
    H_plain = np.asarray(jax.jit(h_plain)(params, pos, cell, graph)[0])

    assert np.allclose(H_remat, H_plain, atol=1e-6)


# --- PET shadow-on: the production path the dense machine exists for ---------

PET_CKPT = checkpoint("pet-mad-xs")


@pytest.mark.skipif(
    not (PET_CKPT / "model.msgpack").exists(),
    reason="pet-mad-xs checkpoint missing (see sources/processed/pet-mad-xs/README.md)",
)
def test_pet_shadow_on_matches_jax_hessian():
    """Shadow-on (`no_shadow=False`) is dense; must match `jax.hessian`."""
    from sadmof.models.pet import atoms_to_inputs as pet_inputs
    from sadmof.models.pet import get_energy_fn, load_pet

    model, params, _ = load_pet(PET_CKPT, dtype="float64")
    atoms = bulk_rattled()
    pos, cell, graph = pet_inputs(atoms, model)
    energy_fn = get_energy_fn(model, no_shadow=False)  # shadow on → dense

    hessian = get_dense_hessian_fn(energy_fn)
    H, overflow = jax.jit(hessian)(params, pos, cell, graph)
    assert not bool(overflow)  # PET aux: k_sel overflow flag
    H = np.asarray(H)
    H_ref = np.asarray(jax.hessian(lambda p: energy_fn(params, p, cell, graph)[0])(pos))

    assert np.allclose(H, H_ref, atol=1e-8)
    assert np.abs(H_ref).max() > 1e-3  # sanity: the Hessian isn't trivially zero


def bulk_rattled():
    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.rattle(0.1, seed=1)  # break symmetry → nonzero forces / shadow coupling
    return atoms
