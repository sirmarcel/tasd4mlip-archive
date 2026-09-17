"""Tests for third-order slices (sparse.fc3, dense.fc3) and the Γ-only
linewidth kernel (observables.linewidths).

The slice tests run MACE on bulk silicon against nested-AD ground truth. The
kernel is pinned mode-by-mode against a stored phono3py reference
(tests/data/build_linewidths.py), certifying the Togo Eq. 10/11 conventions
without phono3py in the test env.
"""

import numpy as np
import jax

from pathlib import Path

import pytest

from sadmof.dense import get_dense_fc3_slice_fn
from sadmof.models.mace import atoms_to_inputs
from sadmof.observables import (
    gamma_only_linewidths,
    mode_basis,
    mode_direction,
    slice_gamma,
)
from sadmof.paths import checkpoint
from sadmof.sparse import get_fc3_slice_fn, hessian_coloring

jax.config.update("jax_enable_x64", True)

CHECKPOINT = checkpoint("mace-mp-0-medium")
pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "model.msgpack").exists(),
    reason="mace-mp-0-medium checkpoint missing (see sources/processed/mace-mp-0-medium/README.md)",
)
HOPS = 4  # MACE-MP-0 medium: 2 interaction layers, Hessian reach 2L hops
FIXTURE = Path(__file__).parent / "data" / "linewidths_ar_reference.npz"


@pytest.fixture(scope="module")
def loaded():
    from marathon.io import from_dict, read_msgpack, read_yaml

    model = from_dict(read_yaml(str(CHECKPOINT / "model.yaml")))
    params = read_msgpack(str(CHECKPOINT / "model.msgpack"))
    return model, params


@pytest.fixture(scope="module")
def si():
    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    return atoms_to_inputs(atoms, cutoff=6.0, float_dtype=np.float64)


@pytest.fixture(scope="module")
def direction(si):
    pos, _, _ = si
    rng = np.random.default_rng(7)
    return jax.numpy.asarray(rng.normal(size=pos.shape))


# --- slices ---------------------------------------------------------------


def test_dense_slice_matches_nested_ad(loaded, si, direction):
    """dense.fc3 equals the contraction of jacfwd-of-hessian with d."""
    model, params = loaded
    pos, cell, graph = si

    def scalar(p):
        return model.energy(params, p, cell, graph)[0]

    _, ref = jax.jvp(jax.hessian(scalar), (pos,), (direction,))
    fc3_slice = get_dense_fc3_slice_fn(model.energy)
    M, aux = jax.jit(fc3_slice)(params, pos, cell, graph, direction)

    assert np.allclose(np.asarray(M), np.asarray(ref), atol=1e-8)
    assert aux.shape == (pos.shape[0],)


def test_sparse_slice_matches_dense(loaded, si, direction):
    """The Hessian coloring recovers the slice exactly (pattern containment)."""
    model, params = loaded
    pos, cell, graph = si

    coloring = hessian_coloring(graph, HOPS)

    sparse_slice = get_fc3_slice_fn(model.energy, coloring)
    M_sparse, _ = jax.jit(sparse_slice)(params, pos, cell, graph, direction)

    dense_slice = get_dense_fc3_slice_fn(model.energy)
    M_dense, _ = jax.jit(dense_slice)(params, pos, cell, graph, direction)

    assert np.allclose(M_sparse.todense(), np.asarray(M_dense), atol=1e-8)


# --- linewidth kernel -----------------------------------------------------


@pytest.fixture(scope="module")
def reference():
    return np.load(FIXTURE)


def _degenerate_means(nu, gamma, tol=1e-6):
    """Average gamma over degenerate groups: the only per-mode quantity that is
    eigenvector-gauge invariant (phono3py averages over degenerate bands too)."""
    edges = np.flatnonzero(np.diff(nu) > tol)
    groups = np.split(np.arange(len(nu)), edges + 1)
    return np.stack([gamma[..., g].mean(axis=-1) for g in groups], axis=-1)


def test_kernel_matches_phono3py(reference):
    """Full-tensor kernel reproduces phono3py's gammas on the same tensors."""
    r = reference
    nu, gamma = gamma_only_linewidths(
        r["fc2"], r["fc3"], r["masses_amu"], r["temperatures"], float(r["sigma_thz"])
    )
    assert np.allclose(nu, np.clip(r["freqs_thz"], 0.0, None), atol=1e-5)
    assert np.allclose(
        _degenerate_means(nu, gamma),
        _degenerate_means(nu, r["gammas_thz"]),
        rtol=1e-4,
        atol=1e-9,
    )


def test_slice_path_matches_full(reference):
    """Per-mode slice_gamma agrees with the full-tensor kernel."""
    r = reference
    fc2, fc3, masses = r["fc2"], r["fc3"], r["masses_amu"]
    nu, gamma_full = gamma_only_linewidths(
        fc2, fc3, masses, r["temperatures"], float(r["sigma_thz"])
    )
    evecs = mode_basis(fc2, masses)[1]

    n = len(masses)
    F = fc3.transpose(0, 3, 1, 4, 2, 5).reshape(3 * n, 3 * n, 3 * n)
    for lam in [4, 3 * n // 2, 3 * n - 1]:
        d = mode_direction(evecs, masses, lam).reshape(3 * n)
        m_slice = np.einsum("abc,a->bc", F, d).reshape(n, 3, n, 3)
        gamma = slice_gamma(
            m_slice, lam, nu, evecs, masses, r["temperatures"], float(r["sigma_thz"])
        )
        assert np.allclose(gamma, gamma_full[:, lam], rtol=1e-10)
