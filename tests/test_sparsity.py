"""Tests for the graph reachability sparsity pattern (`sadmof.sparse`).

The pattern is `(I + A)^hops` over the `(centers, others)` pair list, 3x3-block
expanded into `3N x 3N` coordinate space. These check the reachability logic at
the atom level and the invariant that sadmof's padding (every padding pair a
self-loop on a single dummy slot) never couples to real atoms.
"""

import numpy as np

import pytest

from sadmof.sparse import atom_sparsity_pattern, sparsity_pattern, sparsity_patterns
from sadmof.sparse.pattern import _sparsity_pattern


def atom_pairs(pattern):
    """Atom-level `(i, j)` pairs present in a 3x3-block coordinate pattern."""
    rows, cols = np.asarray(pattern.rows), np.asarray(pattern.cols)
    return set(zip((rows // 3).tolist(), (cols // 3).tolist()))


def undirected_edges(centers, others):
    """Symmetrise a pair list so reachability is over the undirected graph."""
    c = np.concatenate([centers, others])
    o = np.concatenate([others, centers])
    return c, o


# --- reachability ---------------------------------------------------------


def test_diagonal_only_at_zero_hops():
    # 0 -- 1 -- 2, but hops=0 sees no edges
    centers, others = undirected_edges(np.array([0, 1]), np.array([1, 2]))
    pairs = atom_pairs(_sparsity_pattern(centers, others, n=3, hops=0))
    assert pairs == {(0, 0), (1, 1), (2, 2)}


def test_linear_chain_one_hop():
    # 0 -- 1 -- 2 -- 3 ; one hop couples direct neighbours only
    centers, others = undirected_edges(np.array([0, 1, 2]), np.array([1, 2, 3]))
    pairs = atom_pairs(_sparsity_pattern(centers, others, n=4, hops=1))
    expected = {(i, i) for i in range(4)}
    expected |= {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)}
    assert pairs == expected
    assert (0, 2) not in pairs  # two apart, not yet reachable


def test_linear_chain_reaches_full_density():
    # 0 -- 1 -- 2 -- 3 ; diameter 3, so hops=3 couples everything
    centers, others = undirected_edges(np.array([0, 1, 2]), np.array([1, 2, 3]))
    pairs = atom_pairs(_sparsity_pattern(centers, others, n=4, hops=3))
    assert pairs == {(i, j) for i in range(4) for j in range(4)}


def test_disconnected_clusters_never_couple():
    # {0,1} and {2,3} are separate; no hop count bridges them
    centers, others = undirected_edges(np.array([0, 2]), np.array([1, 3]))
    pairs = atom_pairs(_sparsity_pattern(centers, others, n=4, hops=5))
    assert (0, 2) not in pairs and (1, 3) not in pairs
    assert (0, 1) in pairs and (2, 3) in pairs


def test_star_graph_couples_leaves_at_two_hops():
    # hub 0 with leaves 1,2,3 ; leaves couple to each other only via 2 hops
    centers, others = undirected_edges(np.array([0, 0, 0]), np.array([1, 2, 3]))
    one = atom_pairs(_sparsity_pattern(centers, others, n=4, hops=1))
    two = atom_pairs(_sparsity_pattern(centers, others, n=4, hops=2))
    assert (1, 2) not in one
    assert (1, 2) in two and (2, 3) in two and (1, 3) in two


def test_symmetric_for_undirected_graph():
    centers, others = undirected_edges(np.array([0, 1, 2]), np.array([1, 2, 3]))
    pairs = atom_pairs(_sparsity_pattern(centers, others, n=4, hops=2))
    assert all((j, i) in pairs for (i, j) in pairs)


def test_monotonic_in_hops():
    centers, others = undirected_edges(np.array([0, 1, 2]), np.array([1, 2, 3]))
    prev = atom_pairs(_sparsity_pattern(centers, others, n=4, hops=0))
    for hops in range(1, 5):
        cur = atom_pairs(_sparsity_pattern(centers, others, n=4, hops=hops))
        assert prev <= cur
        prev = cur


def test_negative_hops_rejected():
    with pytest.raises(ValueError):
        _sparsity_pattern(np.array([0]), np.array([1]), n=2, hops=-1)


# --- padding invariant ----------------------------------------------------


def test_padding_isolated():
    """sadmof pads pairs as self-loops on dummy slot n_real; padding atoms must
    couple only to themselves, at any hop count, with no spurious real coupling."""
    n_real, n = 3, 6  # atoms 3,4,5 are padding; dummy slot is 3
    real_c, real_o = undirected_edges(np.array([0, 1]), np.array([1, 2]))
    pad_c = np.full(4, n_real)  # 4 padding pairs, all (3, 3)
    pad_o = np.full(4, n_real)
    centers = np.concatenate([real_c, pad_c])
    others = np.concatenate([real_o, pad_o])

    pairs = atom_pairs(_sparsity_pattern(centers, others, n=n, hops=4))
    for p in (3, 4, 5):
        coupled = {j for (i, j) in pairs if i == p} | {i for (i, j) in pairs if j == p}
        assert coupled == {p}, f"padding atom {p} coupled to {coupled}"


# --- dict entry point -----------------------------------------------------


def test_graph_entry_point():
    graph = {
        "centers": np.array([0, 1, 1, 2]),
        "others": np.array([1, 0, 2, 1]),
        "atomic_numbers": np.zeros(4, dtype=int),
    }
    pattern = sparsity_pattern(graph, hops=1)
    assert pattern.m == pattern.n == 12  # 3 * 4 padded atoms
    assert atom_pairs(pattern) == atom_pairs(
        _sparsity_pattern(graph["centers"], graph["others"], n=4, hops=1)
    )


# --- atom level -------------------------------------------------------------


def _padded_chain():
    """0 -- 1 -- 2 real, atoms 3,4,5 padding as self-loops on dummy slot 3."""
    real_c, real_o = undirected_edges(np.array([0, 1]), np.array([1, 2]))
    centers = np.concatenate([real_c, np.full(4, 3)])
    others = np.concatenate([real_o, np.full(4, 3)])
    return {"centers": centers, "others": others, "atomic_numbers": np.zeros(6, int)}


def test_atom_pattern_is_the_block_structure():
    graph = _padded_chain()
    for hops in range(3):
        atom = atom_sparsity_pattern(graph, hops)
        coord = sparsity_pattern(graph, hops)
        assert atom.m == atom.n == 6
        assert coord.nnz == 9 * atom.nnz
        assert atom_pairs(coord) == set(zip(atom.rows.tolist(), atom.cols.tolist()))


def test_sparsity_patterns_agrees_with_singles():
    graph = _padded_chain()
    atom, coord = sparsity_patterns(graph, hops=2)
    single_atom = atom_sparsity_pattern(graph, hops=2)
    single_coord = sparsity_pattern(graph, hops=2)
    assert np.array_equal(atom.rows, single_atom.rows)
    assert np.array_equal(atom.cols, single_atom.cols)
    assert np.array_equal(coord.rows, single_coord.rows)
    assert np.array_equal(coord.cols, single_coord.cols)


def test_atom_pattern_padding_isolated():
    atom = atom_sparsity_pattern(_padded_chain(), hops=4)
    pairs = set(zip(atom.rows.tolist(), atom.cols.tolist()))
    for p in (3, 4, 5):
        coupled = {j for (i, j) in pairs if i == p} | {i for (i, j) in pairs if j == p}
        assert coupled == {p}


def test_atom_pattern_negative_hops_rejected():
    with pytest.raises(ValueError):
        atom_sparsity_pattern(_padded_chain(), hops=-1)
