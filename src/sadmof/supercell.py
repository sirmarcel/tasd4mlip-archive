"""Alias-free supercells from offset-resolved reachability.

The pipeline: build the model's 1-hop edges through its production input
pipeline (mace_edges / pet_edges), offset-resolved — each edge (i, j, S)
records which periodic image of j it points to, not just the pair (i, j) —
then propagate them h hops by sparse boolean matmuls on a finite box of
unit-cell copies, provably large enough to contain all interactions (reach).

One home-cell atom i may then reach the same atom j in several periodic
images S, each with its own separation vector and hence its own force
constant. Our goal is to find a supercell where all of these appear
explicitly, because this allows the computation of "unaliased" force
constants (no pairs are folded together). The criterion: a supercell
multiplier m identifies images S and S' iff their difference D = S - S' is
componentwise divisible by m, so a cell is alias-free iff no within-pair
difference (differences) passes that divisibility test (alias_free).

Our final output is a per-direction supercell multiplier B + 1, where B is
the per-direction maximum of |D| over all differences (extents): no
difference component can then be a nonzero multiple of B + 1, so nothing
folds (closed_cell). (B itself would alias: the extremal difference has a
component equal to B and zeros elsewhere, and zero is divisible by anything.) B + 1 also bounds the exhaustive search for the
minimum-volume alias-free multiplier (exact_min_cell). Any candidate cell is
graded by folding the reach onto it (fold_stats).

Notation:

    S            integer cell offset, meaning "the image of j in cell S"
                 (ASE row convention: Cartesian vector S @ cell)
    diffs        the difference set Delta_h: within-pair offset differences
    extents (B)  per-direction maxima over |Delta_h|
    m, mult      diagonal supercell multiplier; m* = B + 1 is the closed form
    e_max        measured maximum 1-hop edge length
    perp_widths  perpendicular cell widths w_i
    hops (h)     graph hops, up to the model's Hessian range K
"""

import numpy as np

from scipy import sparse as sp

__all__ = [
    "mace_edges",
    "pet_edges",
    "reach",
    "differences",
    "extents",
    "closed_cell",
    "exact_min_cell",
    "alias_free",
    "fold_stats",
    "perp_widths",
    "heur_cell",
    "radius",
    "d_max",
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mace_edges(atoms, rc):
    """1-hop edges (i, j, S, e_max) on the production MACE graph (vesin)."""
    from sadmof.models.mace import atoms_to_inputs

    pos, cell, graph = atoms_to_inputs(atoms, cutoff=rc)
    mask = np.asarray(graph["pair_mask"])
    i = np.asarray(graph["centers"])[mask].astype(np.int32)
    j = np.asarray(graph["others"])[mask].astype(np.int32)
    S = np.asarray(graph["cell_shifts"])[mask].astype(np.int32)

    _check_symmetric(i, j, S, "MACE")
    e_max = _edge_max(np.asarray(pos)[: len(atoms)], np.asarray(cell), i, j, S)
    # Shift-sign guard: recomputed edge lengths stay <= rc only under the
    # "image of j in cell S" convention assumed throughout; a flipped sign
    # adds 2 S @ cell to every cross-cell edge, blowing past the cutoff.
    if e_max > rc + 1e-9:
        raise ValueError(f"edge length {e_max} exceeds cutoff {rc} (shift sign?)")
    return i, j, S, e_max


def pet_edges(atoms, model):
    """1-hop edges (i, j, S, e_max) on PET's adaptive *selected* adjacency."""
    from sadmof.models.pet import atoms_to_inputs

    pos, cell, graph = atoms_to_inputs(atoms, model)
    i = np.asarray(graph["sel_centers"]).astype(np.int32)
    j = np.asarray(graph["sel_others"]).astype(np.int32)
    S = np.asarray(graph["sel_cell_shifts"]).astype(np.int32)

    _check_symmetric(i, j, S, "PET selected")
    e_max = _edge_max(np.asarray(pos)[: len(atoms)], np.asarray(cell), i, j, S)
    # Same shift-sign guard as mace_edges, with headroom: selected edges come
    # from the raw vesin ball at cutoff + skin (0.5 A), so they may exceed the
    # nominal cutoff slightly — but never by whole lattice vectors.
    ceiling = model.cutoff + 1.0
    if e_max > ceiling:
        raise ValueError(f"selected edge {e_max:.2f} A exceeds {ceiling} (shift sign?)")
    return i, j, S, e_max


def reach(atoms, edges, hops):
    """Offset-resolved reachability from the home cell: list over h = 1..hops
    of (src, dst, S) arrays. Requires wrapped positions (cluster-box bound)."""
    n = len(atoms)
    ei, ej, eS, e_max = edges
    shape, home_cell = _cluster_box(atoms.cell, hops, e_max)
    base = _cluster_adjacency(shape, n, ei, ej, eS)

    # R selects the home-cell atoms; each (I + A) matmul extends reach by one
    # hop. Flat cluster index = cell * n + atom, undone by divmod below.
    home = np.ravel_multi_index(home_cell, shape) * n + np.arange(n)
    nc = int(np.prod(shape)) * n
    R = sp.csr_matrix((np.ones(n, bool), (np.arange(n), home)), shape=(n, nc))
    out = []
    for _ in range(hops):
        R = R @ base
        R.data[:] = True
        coo = R.tocoo()
        cell_idx, dst = np.divmod(coo.col, n)
        S = np.stack(np.unravel_index(cell_idx, shape), axis=1) - home_cell
        out.append((coo.row.astype(np.int32), dst.astype(np.int32), S.astype(np.int32)))
    return out


def differences(r, n):
    """The difference set Delta: nonzero offset differences within (i, j) groups."""
    src, dst, S = r
    key = src.astype(np.int64) * n + dst
    order = np.argsort(key, kind="stable")
    key, S = key[order], S[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])  # one per (i, j) group
    sizes = np.diff(np.r_[starts, len(key)])

    if not np.any(sizes > 1):
        return np.zeros((0, 3), np.int32)

    # Differences lie in [-span, span], a small box, so dedup is a flag grid
    # indexed by shifted difference — no row-wise unique (whose structured
    # sort dominates everything else) anywhere.
    span = S.max(axis=0) - S.min(axis=0)
    grid = 2 * span + 1
    flags = np.zeros(int(np.prod(grid)), bool)

    # Equal-size groups stack into one (n_groups, s, 3) slab, so all their
    # pairwise differences are a single broadcast; chunked to bound memory.
    for s in np.unique(sizes[sizes > 1]):
        group_starts = starts[sizes == s]
        step = max(1, _BATCH // int(s * s))
        for a in range(0, len(group_starts), step):
            slab = S[group_starts[a : a + step, None] + np.arange(s)]
            d = (slab[:, :, None, :] - slab[:, None, :, :]).reshape(-1, 3)
            flags[np.ravel_multi_index((d + span).T, grid)] = True
    # Flat-index order is lexicographic in the rows, matching np.unique(axis=0).
    d = np.stack(np.unravel_index(np.flatnonzero(flags), grid), axis=1) - span
    return d[np.any(d != 0, axis=1)].astype(np.int32)


def extents(diffs):
    """Per-direction extents B of the difference set (0 where empty)."""
    return np.abs(diffs).max(axis=0, initial=0).astype(int)


def closed_cell(diffs):
    """The closed-form alias-free multiplier m* = B + 1."""
    return tuple(int(x) for x in extents(diffs) + 1)


def exact_min_cell(diffs):
    """Minimum-volume alias-free multiplier (exhaustive over the B+1 box)."""
    if len(diffs) == 0:
        return (1, 1, 1)
    cap = extents(diffs) + 1
    m = np.stack(
        np.meshgrid(*(np.arange(1, c + 1) for c in cap), indexing="ij"), -1
    ).reshape(-1, 3)
    # Candidates in volume order, so the first alias-free hit is the minimum.
    m = m[np.argsort(m.prod(axis=1), kind="stable")]
    step = max(1, _BATCH // len(diffs))
    for a in range(0, len(m), step):
        chunk = m[a : a + step]
        ok = ~np.any(np.all(diffs[None, :, :] % chunk[:, None, :] == 0, axis=2), axis=1)
        if ok.any():
            return tuple(int(x) for x in chunk[np.argmax(ok)])
    raise AssertionError("unreachable: B + 1 lies in the box and is alias-free")


def alias_free(diffs, mult):
    """True iff no difference is componentwise divisible by the multiplier."""
    if len(diffs) == 0:
        return True
    return not np.any(np.all(diffs % np.asarray(mult) == 0, axis=1))


def fold_stats(r, n, mult):
    """(nnz, fill, aliased pairs) of the pattern folded onto `mult`, atom-level."""
    src, dst, S = r
    m = np.asarray(mult)
    F = S % m
    # One integer key per folded triple (i, j, S mod m); uniques count the
    # home-cell pattern, which the supercell replicates M times.
    key = np.ravel_multi_index((src, dst, F[:, 0], F[:, 1], F[:, 2]), (n, n, *m))
    folded = len(np.unique(key))
    M = int(m.prod())
    return folded * M, folded / (n * n * M), len(src) - folded


def perp_widths(cell):
    """Perpendicular widths of a cell (rows = lattice vectors): 1 / |dual_i|."""
    return 1.0 / np.linalg.norm(np.linalg.inv(np.asarray(cell, dtype=float)), axis=0)


def heur_cell(cell, L):
    """Smallest diagonal multiplier with supercell widths >= L everywhere."""
    return tuple(int(x) for x in np.ceil(L / perp_widths(cell)))


def radius(r, atoms):
    """Measured hop radius R_h: max Cartesian reach-vector length."""
    src, dst, S = r
    pos = atoms.get_positions()
    d = pos[dst] + S @ np.asarray(atoms.cell) - pos[src]
    return float(np.linalg.norm(d, axis=1).max())


def d_max(diffs, cell):
    """The difference diameter: max Cartesian length over the difference set."""
    if len(diffs) == 0:
        return 0.0
    return float(np.linalg.norm(diffs @ np.asarray(cell), axis=1).max())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Cap on elements per broadcasted chunk (differences, exact_min_cell)
_BATCH = 2**24


def _check_symmetric(i, j, S, label):
    # (I + A)^h covers the Hessian pattern only for symmetric adjacency, and
    # sparse.pattern does not symmetrise.
    forward = np.unique(np.column_stack([i, j, S]), axis=0)
    backward = np.unique(np.column_stack([j, i, -S]), axis=0)
    if forward.shape != backward.shape or not np.array_equal(forward, backward):
        raise ValueError(f"{label} adjacency is not symmetric")


def _edge_max(positions, cell, i, j, S):
    d = positions[j] + S @ np.asarray(cell) - positions[i]
    return float(np.linalg.norm(d, axis=1).max())


def _cluster_box(cell, hops, e_max):
    """Finite box of cells covering the h-hop reach ("cluster box" corollary):
    |S_i| <= ceil(h e_max / w_i), plus one spare cell so edges dropped at the
    boundary can never touch reach from the home cell. Returns (box shape,
    home-cell index), the home cell at the centre."""
    b = np.ceil(hops * e_max / perp_widths(cell)).astype(int) + 1
    return 2 * b + 1, b


def _cluster_adjacency(shape, n, ei, ej, eS):
    """Boolean (I + A) on the cluster: each home-cell edge (i, j, S)
    instantiated in every box cell whose shifted target stays in the box.
    Flat index = cell * n + atom."""
    cells = np.stack(
        np.meshgrid(*(np.arange(s) for s in shape), indexing="ij"), -1
    ).reshape(-1, 3)
    rows, cols = [], []
    for S, mask in _by_shift(eS):
        ok = np.all((cells + S >= 0) & (cells + S < shape), axis=1)
        src_cell = np.ravel_multi_index(cells[ok].T, shape)
        dst_cell = np.ravel_multi_index((cells[ok] + S).T, shape)
        rows.append((src_cell[:, None] * n + ei[mask][None, :]).ravel())
        cols.append((dst_cell[:, None] * n + ej[mask][None, :]).ravel())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    nc = int(np.prod(shape)) * n
    A = sp.csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(nc, nc))
    return (A + sp.eye(nc, dtype=bool, format="csr")).tocsr()


def _by_shift(eS):
    uniq, inv = np.unique(eS, axis=0, return_inverse=True)
    for k, S in enumerate(uniq):
        yield S, inv == k
