"""Tests for the sparse positions-Hessian (sadmof.sparse.hessian)."""

import numpy as np
import jax

import asdex
import pytest

from sadmof.models.mace import atoms_to_inputs
from sadmof.paths import checkpoint
from sadmof.sparse import get_hessian_fn, hessian_coloring, sparsity_pattern

# Hessians/phonons are done in fp64; the rest of the suite enables x64 globally
# at import, so pin it here too to keep this module consistent regardless of
# test order.
jax.config.update("jax_enable_x64", True)

CHECKPOINT = checkpoint("mace-mp-0-medium")
pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "model.msgpack").exists(),
    reason="mace-mp-0-medium checkpoint missing (see sources/processed/mace-mp-0-medium/README.md)",
)

# MACE-MP-0 medium has 2 interaction layers, so the Hessian couples atoms up to
# 2L = 4 graph hops apart. At this range the reachability pattern is exact.
#
# This module tests the sparse *machinery* (pattern -> colouring -> decompression),
# not that number: the Si8 fixture below is fully connected at one hop, so the
# pattern is all-ones and these tests pass for any hop count. `test_hop_count`
# pins the hop count itself, on a fixture whose graph is deep enough to show it.
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


def _coloring(graph):
    """A fresh colored pattern, the production route (atom-level coloring,
    lifted). Built per test; reuse is not supported!"""
    return hessian_coloring(graph, hops=HOPS)


def _asdex_coloring(graph):
    """asdex's own coordinate-level coloring, for comparison."""
    return asdex.hessian_coloring_from_sparsity(
        sparsity_pattern(graph, hops=HOPS), mode="fwd_over_rev"
    )


def _energy_fn(model, params, cell, graph):
    """Scalar energy as a function of positions."""

    def energy(pos):
        return model.energy(params, pos, cell, graph)[0]

    return energy


def test_shape_and_symmetry(loaded, si):
    model, params = loaded
    pos, cell, graph = si
    coloring = _coloring(graph)
    hessian = get_hessian_fn(model.energy, coloring)
    n = 3 * pos.shape[0]

    H_bcoo, aux = jax.jit(hessian)(params, pos, cell, graph)
    H = np.asarray(H_bcoo.todense()).reshape(n, n)

    assert H.shape == (n, n)
    assert aux.shape == (pos.shape[0],)  # MACE aux: per-atom energies
    assert coloring.num_colors >= 1
    assert np.allclose(H, H.T, atol=1e-5)


def test_matches_dense(loaded, si):
    """Sparse Hessian equals the dense JAX Hessian (also proves the pattern is
    conservative: any true coupling off the pattern would show up as a mismatch)."""
    model, params = loaded
    pos, cell, graph = si
    hessian = get_hessian_fn(model.energy, _coloring(graph))

    H = np.asarray(jax.jit(hessian)(params, pos, cell, graph)[0].todense())
    H_dense = np.asarray(jax.hessian(_energy_fn(model, params, cell, graph))(pos))

    assert np.allclose(H, H_dense, atol=1e-6)


def test_lifted_matches_asdex_coloring(si):
    _pos, _cell, graph = si
    ours, ref = _coloring(graph), _asdex_coloring(graph)
    assert ours.num_colors == ref.num_colors
    assert np.array_equal(np.asarray(ours.colors), np.asarray(ref.colors))


def test_lifted_hessian_matches_asdex(loaded, si):
    """The lifted coloring and asdex's give the same Hessian through the real
    model. Si8 is complete at one hop, so this checks the plumbing (avals, BCOO,
    aux) on a hand-built `ColoredPattern`; the star structure is covered at
    pattern level in `test_coloring.py`."""
    model, params = loaded
    pos, cell, graph = si
    h_ours = get_hessian_fn(model.energy, _coloring(graph))
    h_ref = get_hessian_fn(model.energy, _asdex_coloring(graph))

    H_ours = np.asarray(jax.jit(h_ours)(params, pos, cell, graph)[0].todense())
    H_ref = np.asarray(jax.jit(h_ref)(params, pos, cell, graph)[0].todense())
    H_dense = np.asarray(jax.hessian(_energy_fn(model, params, cell, graph))(pos))

    assert np.allclose(H_ours, H_ref, atol=1e-6)
    assert np.allclose(H_ours, H_dense, atol=1e-6)


def test_check_hessian_correctness(loaded, si):
    """asdex's own verifier on the pattern + coloring (independent of our
    wrapper). `method="dense"` because the `matvec` path assumes a 2-D `(n, n)`
    Hessian and chokes on our native `(N, 3, N, 3)` output."""
    model, params = loaded
    pos, cell, graph = si
    energy = _energy_fn(model, params, cell, graph)

    asdex.check_hessian_correctness(energy, pos, _coloring(graph), method="dense")


def test_chunk_size_equivalence(loaded, si):
    """chunk_size only bounds memory; the result must be identical."""
    model, params = loaded
    pos, cell, graph = si

    h_full = get_hessian_fn(model.energy, _coloring(graph), chunk_size=None)
    h_chunk = get_hessian_fn(model.energy, _coloring(graph), chunk_size=1)

    H_full = np.asarray(jax.jit(h_full)(params, pos, cell, graph)[0].todense())
    H_chunk = np.asarray(jax.jit(h_chunk)(params, pos, cell, graph)[0].todense())

    assert np.allclose(H_full, H_chunk, atol=1e-6)


def test_remat_equivalence(loaded, si):
    """remat only trades compute for memory; the result must be identical."""
    model, params = loaded
    pos, cell, graph = si

    h_plain = get_hessian_fn(model.energy, _coloring(graph), remat=False)
    h_remat = get_hessian_fn(model.energy, _coloring(graph), remat=True)

    H_plain = np.asarray(jax.jit(h_plain)(params, pos, cell, graph)[0].todense())
    H_remat = np.asarray(jax.jit(h_remat)(params, pos, cell, graph)[0].todense())

    assert np.allclose(H_plain, H_remat, atol=1e-6)
