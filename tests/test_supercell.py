"""Tests for alias-free supercell determination (`sadmof.supercell`).

The core invariant, checked model-free via ase.neighborlist graphs: the pattern
predicted by folding the offset-resolved reach onto a multiplier agrees exactly
(nnz) with brute-force reachability on the explicitly tiled supercell — for
cubic and non-cubic multipliers, at every hop. On top: analytic cases where the
alias-free cell is known by hand, and independent brute-force minimality.
"""

import numpy as np

import pytest
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy import sparse as sp

from sadmof.paths import checkpoint
from sadmof.supercell import (
    alias_free,
    closed_cell,
    d_max,
    differences,
    exact_min_cell,
    extents,
    fold_stats,
    heur_cell,
    perp_widths,
    radius,
    reach,
)

MULTS = [(1, 1, 1), (2, 2, 2), (2, 1, 3)]


def nl_edges(atoms, rc):
    """Symmetric offset-resolved edges via ase.neighborlist (model-free)."""
    i, j, S = neighbor_list("ijS", atoms, rc)
    d = atoms.positions[j] + S @ np.asarray(atoms.cell) - atoms.positions[i]
    return (
        i.astype(np.int32),
        j.astype(np.int32),
        S.astype(np.int32),
        float(np.linalg.norm(d, axis=1).max()),
    )


def brute_nnz(atoms, rc, mult, hops):
    """Boolean (I + A)^h pair counts on the explicitly tiled supercell."""
    sup = atoms * mult
    i, j, _, _ = nl_edges(sup, rc)
    n = len(sup)
    A = sp.csr_matrix((np.ones(len(i), bool), (i, j)), shape=(n, n))
    base = (A + sp.eye(n, dtype=bool, format="csr")).tocsr()
    R, out = base.copy(), []
    for _ in range(hops):
        out.append(R.nnz)
        R = (R @ base) > 0
    return out


def brute_min_cell(diffs, cap):
    """Independent exhaustive minimum over the cap box (plain loops)."""
    best, best_vol = None, None
    for m1 in range(1, cap[0] + 1):
        for m2 in range(1, cap[1] + 1):
            for m3 in range(1, cap[2] + 1):
                vol = m1 * m2 * m3
                if (best_vol is None or vol < best_vol) and alias_free(diffs, (m1, m2, m3)):
                    best, best_vol = (m1, m2, m3), vol
    return best, best_vol


# --- geometry --------------------------------------------------------------


def test_perp_widths_orthorhombic():
    assert np.allclose(perp_widths(np.diag([3.0, 4.0, 5.0])), [3.0, 4.0, 5.0])


def test_perp_widths_hexagonal():
    a, c = 4.0, 7.0
    cell = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    assert np.allclose(perp_widths(cell), [a * np.sqrt(3) / 2, a * np.sqrt(3) / 2, c])


def test_heur_cell():
    assert heur_cell(np.diag([3.0, 4.0, 12.0]), 10.0) == (4, 3, 1)


# --- analytic chain: single atom, axis neighbors only -----------------------


@pytest.mark.parametrize("hops", [1, 2, 3])
def test_cubic_chain_analytic(hops):
    # Single atom, cubic a=4, rc between a and a*sqrt(2): axis steps only.
    # h-hop offsets are exactly {S : |S|_1 <= h}, so B = 2h and m* = 2h + 1.
    atoms = Atoms("Cu", cell=np.eye(3) * 4.0, pbc=True)
    rs = reach(atoms, nl_edges(atoms, 4.5), hops)
    _, _, S = rs[-1]
    assert set(map(tuple, S)) == {
        (a, b, c)
        for a in range(-hops, hops + 1)
        for b in range(-hops, hops + 1)
        for c in range(-hops, hops + 1)
        if abs(a) + abs(b) + abs(c) <= hops
    }
    diffs = differences(rs[-1], 1)
    assert tuple(extents(diffs)) == (2 * hops,) * 3
    assert closed_cell(diffs) == (2 * hops + 1,) * 3
    assert alias_free(diffs, closed_cell(diffs))
    assert d_max(diffs, atoms.cell) == pytest.approx(2 * hops * 4.0)
    assert radius(rs[-1], atoms) == pytest.approx(hops * 4.0)


def test_anisotropic_chain():
    # rc reaches multiple images along the short axis only: 1-D chain in z.
    atoms = Atoms("Cu", cell=np.diag([12.0, 12.0, 3.0]), pbc=True)
    rs = reach(atoms, nl_edges(atoms, 3.5), 2)
    diffs = differences(rs[-1], 1)
    assert closed_cell(diffs) == (1, 1, 5)
    assert exact_min_cell(diffs) == (1, 1, 5)


# --- fold prediction vs brute force on tiled supercells ---------------------


def _fixture_structures():
    rng = np.random.default_rng(0)
    cubic = Atoms(
        "Cu4",
        positions=rng.uniform(0, 5, (4, 3)),
        cell=np.eye(3) * 5.0,
        pbc=True,
    )
    triclinic = Atoms(
        "Cu3",
        positions=rng.uniform(0, 4, (3, 3)),
        cell=np.array([[5.0, 0, 0], [2.0, 4.5, 0], [1.0, -1.5, 6.0]]),
        pbc=True,
    )
    return {"cubic": cubic, "triclinic": triclinic}


@pytest.mark.parametrize("name", ["cubic", "triclinic"])
@pytest.mark.parametrize("mult", MULTS)
def test_fold_matches_brute_force(name, mult):
    atoms = _fixture_structures()[name]
    atoms.wrap()
    rc, hops = 4.0, 3
    rs = reach(atoms, nl_edges(atoms, rc), hops)
    nnz_brute = brute_nnz(atoms, rc, mult, hops)
    for h, r in enumerate(rs, 1):
        nnz_fold, _, _ = fold_stats(r, len(atoms), mult)
        assert nnz_fold == nnz_brute[h - 1], f"{name} {mult} h={h}"


@pytest.mark.parametrize("name", ["cubic", "triclinic"])
def test_exact_cell_is_minimal_and_alias_free(name):
    atoms = _fixture_structures()[name]
    atoms.wrap()
    rs = reach(atoms, nl_edges(atoms, 4.0), 3)
    for r in rs:
        diffs = differences(r, len(atoms))
        m_star, m_exact = closed_cell(diffs), exact_min_cell(diffs)
        assert alias_free(diffs, m_star)
        assert alias_free(diffs, m_exact)
        ref, ref_vol = brute_min_cell(diffs, m_star)
        assert int(np.prod(m_exact)) == ref_vol, (m_exact, ref)


def test_alias_free_multiplier_has_zero_aliased_pairs():
    atoms = _fixture_structures()["triclinic"]
    atoms.wrap()
    r = reach(atoms, nl_edges(atoms, 4.0), 2)[-1]
    diffs = differences(r, len(atoms))
    _, _, aliased = fold_stats(r, len(atoms), exact_min_cell(diffs))
    assert aliased == 0
    _, _, aliased_1 = fold_stats(r, len(atoms), (1, 1, 1))
    assert aliased_1 > 0  # sanity: the unit cell does alias here


# --- model adjacencies -------------------------------------------------------


def test_mace_edges_production_graph():
    from ase.build import bulk

    from sadmof.supercell import mace_edges

    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.rattle(0.1, seed=1)
    i, j, S, e_max = mace_edges(atoms, 4.0)
    ref = nl_edges(atoms, 4.0)
    assert set(zip(i, j, map(tuple, S))) == set(zip(ref[0], ref[1], map(tuple, ref[2])))
    assert e_max == pytest.approx(ref[3])


PET_CKPT = checkpoint("pet-mad-xs")


@pytest.mark.skipif(
    not (PET_CKPT / "model.msgpack").exists(),
    reason="pet-mad-xs checkpoint missing (see sources/processed/pet-mad-xs/README.md)",
)
def test_pet_edges_fold_matches_brute_force():
    # Folding the selected adjacency must match brute force on the tiled cell;
    # this is also the translation-invariance check of PET's adaptive selection.
    from ase.build import bulk

    from sadmof.models.pet import load_pet
    from sadmof.supercell import pet_edges

    model, _, _ = load_pet(PET_CKPT, dtype="float64")
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.rattle(0.1, seed=1)
    atoms.wrap()
    n, hops, mult = len(atoms), 2, (2, 1, 3)

    rs = reach(atoms, pet_edges(atoms, model), hops)

    sup = atoms * mult
    i, j, _, _ = pet_edges(sup, model)
    A = sp.csr_matrix((np.ones(len(i), bool), (i, j)), shape=(len(sup), len(sup)))
    base = (A + sp.eye(len(sup), dtype=bool, format="csr")).tocsr()
    R = base.copy()
    for h, r in enumerate(rs, 1):
        nnz_fold, _, _ = fold_stats(r, n, mult)
        assert nnz_fold == R.nnz, f"h={h}"
        R = (R @ base) > 0
