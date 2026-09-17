"""MLIP `graph` → asdex Hessian sparsity pattern.

Two atoms couple in the positions-Hessian iff a chain of neighbour edges
connects them within the model's reach, so the pattern is `(I + A)^hops`,
with `A` the 1-hop adjacency from the `(centers, others)` pair list. The
exact model-dependent `hops` is derived in the preprint and
pinned by `tests/test_hop_count.py`; in practice we often truncate below it.

Padding needs no masking or bookkeeping: both input builders point every
padding pair at a dummy atom slot, so padding edges are self-loops there and
never reach a real atom (guarded by
`tests/test_sparsity.py::test_padding_isolated`).

The pattern is 3x3-block expanded to the padded `3N x 3N` coordinate space;
the `N x N` atom pattern is colored by `sparse.coloring`.
"""

import numpy as np
from numpy.typing import NDArray
from jax import ShapeDtypeStruct

import asdex
from scipy import sparse as sp

from jaxtyping import Array

__all__ = ["atom_sparsity_pattern", "sparsity_pattern", "sparsity_patterns"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sparsity_patterns(
    graph: dict[str, Array], hops: int
) -> tuple[asdex.SparsityPattern, asdex.SparsityPattern]:
    """Atom-level and coordinate-level Hessian sparsity patterns of a padded `graph`.

    Reads `centers`, `others` (the pair list) and `atomic_numbers` (for the
    padded atom count `N`) from `graph`; see `mace.inputs.atoms_to_inputs` for
    the schema.

    Returns:
        `(atom, coord)`: the `(I + A)^hops` reachability pattern over `N` atoms,
        and its 3x3-block expansion into `3N x 3N` coordinate space with the
        positions `(N, 3)` aval attached so asdex differentiates native `(N, 3)`
        arrays. `coord` has exactly `9 * atom.nnz` entries.
    """
    centers, others, n = _unpack(graph)
    pairs = _reachable_pairs(centers, others, n, hops)
    return _atom_pairs_to_sparsity(pairs, n), _pairs_to_sparsity(pairs, n)


def sparsity_pattern(graph: dict[str, Array], hops: int) -> asdex.SparsityPattern:
    """The coordinate-level `3N x 3N` pattern alone; see `sparsity_patterns`."""
    centers, others, n = _unpack(graph)
    return _sparsity_pattern(centers, others, n, hops)


def atom_sparsity_pattern(graph: dict[str, Array], hops: int) -> asdex.SparsityPattern:
    """The atom-level `N x N` pattern alone; see `sparsity_patterns`."""
    centers, others, n = _unpack(graph)
    return _atom_pairs_to_sparsity(_reachable_pairs(centers, others, n, hops), n)


# ---------------------------------------------------------------------------
# Reachability and block expansion
# ---------------------------------------------------------------------------


def _unpack(graph):
    centers = np.asarray(graph["centers"])
    others = np.asarray(graph["others"])
    n = int(graph["atomic_numbers"].shape[0])
    return centers, others, n


def _sparsity_pattern(centers, others, n, hops):
    """Core coordinate-level pattern builder."""
    return _pairs_to_sparsity(_reachable_pairs(centers, others, n, hops), n)


def _reachable_pairs(centers, others, n, hops) -> NDArray[np.int64]:
    """Atom pairs `(i, j)` within `hops` of each other, diagonal included.

    Returns a unique `(n_pairs, 2)` array sorted by `(i, j)`. The diagonal is
    always present so every atom has a self-coupling entry at any `hops >= 0`.
    """
    if hops < 0:
        raise ValueError(f"hops must be non-negative, got {hops}")

    data = np.ones(len(centers), dtype=bool)
    A = sp.csr_matrix((data, (centers, others)), shape=(n, n))

    base = sp.eye(n, dtype=bool, format="csr") + A
    pattern = sp.eye(n, dtype=bool, format="csr")
    for _ in range(hops):
        pattern = pattern @ base
        pattern.data[:] = True  # boolean reachability; drop accumulated counts

    coo = pattern.tocoo()
    diag = np.arange(n)
    pairs = np.column_stack(
        [np.concatenate([coo.row, diag]), np.concatenate([coo.col, diag])]
    )
    return np.unique(pairs.astype(np.int64), axis=0)


def _atom_pairs_to_sparsity(pairs, n):
    """Wrap atom-level pairs as an `(n, n)` `asdex.SparsityPattern`.

    No input aval: this pattern is only ever colored, never differentiated on."""
    return asdex.SparsityPattern(
        rows=pairs[:, 0].astype(np.int32),
        cols=pairs[:, 1].astype(np.int32),
        shape=(n, n),
    )


def _pairs_to_sparsity(pairs, n):
    """Expand atom-level `(i, j)` pairs to 3x3 Cartesian blocks and wrap as an
    `asdex.SparsityPattern` of shape `(3n, 3n)`. The positions `(n, 3)` aval
    is attached (`3*atom + cart` is the row-major flatten of `(n, 3)`) so asdex
    differentiates native `(n, 3)` positions and returns an `(n, 3, n, 3)`
    Hessian. Entries are emitted block by block, not lexsorted; stored sparse
    Hessians are aligned to this order, so it is part of the on-disk format."""
    i, j = pairs[:, 0], pairs[:, 1]
    n_blocks = len(i)
    row_off = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    col_off = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    rows = np.repeat(i * 3, 9) + np.tile(row_off, n_blocks)
    cols = np.repeat(j * 3, 9) + np.tile(col_off, n_blocks)

    return asdex.SparsityPattern.from_coo(
        rows=rows,
        cols=cols,
        shape=(3 * n, 3 * n),
        # Only the (n, 3) shape is load-bearing; asdex validates it and returns
        # an (n, 3, n, 3) Hessian. The aval dtype is nominal — the real dtype
        # follows the positions passed at call time.
        input_avals=(ShapeDtypeStruct((n, 3), np.float64),),
    )
