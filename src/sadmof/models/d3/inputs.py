"""ASE Atoms → D3 model inputs `(pos, cell, graph)`.

Like `models.mace.inputs`, this builds a single padded sample (one structure
at a time — the whole sadmof inference / Hessian path). D3 differs in needing
two *half* neighbour lists (each pair once): a large one at `cutoff` for the
dispersion sum and a small one at `cnthr` for coordination numbers. Both land
in the same `graph`, the CN list under `cn_*` keys.

Returns `(pos, cell, graph)`:
    pos             (N, 3)   float  — atomic positions; padding rows zero
    cell            (3, 3)   float  — unit cell (rows are lattice vectors)
    graph:
        atomic_numbers  (N,)     int    — Z per atom; padding rows zero
        atom_mask       (N,)     bool   — True on real atoms
        centers         (Pe,)    int    — receiver index i (energy list)
        others          (Pe,)    int    — sender index j (energy list)
        cell_shifts     (Pe, 3)  float  — integer cell shift per energy pair
        pair_mask       (Pe,)    bool   — True on real energy pairs
        cn_centers      (Pc,)    int    — receiver index i (CN list)
        cn_others       (Pc,)    int    — sender index j (CN list)
        cn_cell_shifts  (Pc, 3)  float  — integer cell shift per CN pair
        cn_pair_mask    (Pc,)    bool   — True on real CN pairs
"""

import numpy as np
import jax.numpy as jnp

import ase
from ase.build import bulk
from marathon.utils import next_size

from jaxtyping import Array

from .model import D3

__all__ = ["atoms_to_inputs", "dummy_inputs"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def atoms_to_inputs(
    atoms: ase.Atoms,
    cutoff: float,
    cnthr: float,
    float_dtype: np.dtype | type = np.float64,
    int_dtype: np.dtype | type = np.int64,
) -> tuple[Array, Array, dict[str, Array]]:
    """Build padded D3 model inputs `(pos, cell, graph)` from an `ase.Atoms`.

    Two half neighbour lists (vesin) at `cutoff` (energy) and `cnthr` (CN),
    with bucketed padding from `marathon.utils.next_size("multiples")`. Pass
    `cutoff` / `cnthr` already including any Verlet skin.
    """
    n_real = len(atoms)
    n_atoms_padded = next_size(n_real + 1, strategy="multiples")
    pad_atom_idx = n_real  # masked dummy slot for padding pairs to point at

    positions = np.zeros((n_atoms_padded, 3), dtype=float_dtype)
    atomic_numbers = np.zeros(n_atoms_padded, dtype=int_dtype)
    atom_mask = np.zeros(n_atoms_padded, dtype=bool)

    positions[:n_real] = atoms.get_positions()
    atomic_numbers[:n_real] = atoms.get_atomic_numbers()
    atom_mask[:n_real] = True

    cell = np.asarray(atoms.get_cell()[:], dtype=float_dtype)

    e_i, e_j, e_S = _half_neighbor_list(atoms, cutoff, float_dtype, int_dtype)
    n_e = max(next_size(len(e_i) + 1, strategy="multiples"), 1)
    e_c, e_o, e_s, e_m = _pad_pairs(
        e_i, e_j, e_S, n_e, pad_atom_idx, float_dtype, int_dtype
    )

    cn_i, cn_j, cn_S = _half_neighbor_list(atoms, cnthr, float_dtype, int_dtype)
    n_cn = max(next_size(len(cn_i) + 1, strategy="multiples"), 1)
    cn_c, cn_o, cn_s, cn_m = _pad_pairs(
        cn_i, cn_j, cn_S, n_cn, pad_atom_idx, float_dtype, int_dtype
    )

    graph = {
        "atomic_numbers": jnp.asarray(atomic_numbers),
        "atom_mask": jnp.asarray(atom_mask),
        "centers": jnp.asarray(e_c),
        "others": jnp.asarray(e_o),
        "cell_shifts": jnp.asarray(e_s),
        "pair_mask": jnp.asarray(e_m),
        "cn_centers": jnp.asarray(cn_c),
        "cn_others": jnp.asarray(cn_o),
        "cn_cell_shifts": jnp.asarray(cn_s),
        "cn_pair_mask": jnp.asarray(cn_m),
    }
    return jnp.asarray(positions), jnp.asarray(cell), graph


def dummy_inputs(model: D3) -> tuple[Array, Array, dict[str, Array]]:
    """`(pos, cell, graph)` for a small bulk-Si supercell — for warming the
    JIT cache or `model.init`-style initialisation."""
    atoms = bulk("Si", "diamond", a=5.43) * [2, 2, 2]
    return atoms_to_inputs(atoms, model.cutoff, model.cnthr)


# ---------------------------------------------------------------------------
# Neighbour-list / padding helpers
# ---------------------------------------------------------------------------


def _half_neighbor_list(atoms, cutoff, float_dtype, int_dtype):
    """Half neighbour list via vesin (each pair once)."""
    from vesin import NeighborList

    nl = NeighborList(cutoff=cutoff, full_list=False, sorted=True)
    if atoms.pbc.any():
        i, j, S = nl.compute(
            points=atoms.positions,
            box=atoms.cell[:],
            periodic=atoms.pbc,
            quantities="ijS",
        )
    else:
        i, j = nl.compute(
            points=atoms.positions,
            box=atoms.cell[:],
            periodic=atoms.pbc,
            quantities="ij",
        )
        S = np.zeros((len(i), 3), dtype=float_dtype)
    return i.astype(int_dtype), j.astype(int_dtype), S.astype(float_dtype)


def _pad_pairs(i, j, S, n_pairs, pad_atom_idx, float_dtype, int_dtype):
    """Pad pair arrays to fixed size with a boolean mask; padding pairs point
    at `pad_atom_idx` (a masked dummy atom slot)."""
    n_real = len(i)
    centers = np.full(n_pairs, pad_atom_idx, dtype=int_dtype)
    others = np.full(n_pairs, pad_atom_idx, dtype=int_dtype)
    cell_shifts = np.zeros((n_pairs, 3), dtype=float_dtype)
    pair_mask = np.zeros(n_pairs, dtype=bool)

    centers[:n_real] = i
    others[:n_real] = j
    cell_shifts[:n_real] = S
    pair_mask[:n_real] = True
    return centers, others, cell_shifts, pair_mask
