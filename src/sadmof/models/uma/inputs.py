"""ASE Atoms → UMA model inputs `(pos, cell, graph)`.

Mirrors `mace.inputs.atoms_to_inputs`, with two differences that come from UMA
being a directed message-passing model:

- The neighbour list is a **full** list (vesin `full_list=True`): every ordered
  pair appears once in each direction, because a message flows from `others` to
  `centers` and both directions carry different weights.
- Padding is **off** by default. UMA compiles per structure anyway (the MOLE
  merge is composition-specific), so buckets buy no jit-cache reuse, and a
  padding pair is a zero-length self-loop whose direction is undefined. The
  model handles those safely (`model._norm_and_direction`, and `pair_mask`
  zeroing the envelope), so `bucket_strategy` still works if wanted.

Returns `(pos, cell, graph)`:
    pos             (N, 3)   float  — atomic positions; padding rows zero
    cell            (3, 3)   float  — unit cell (rows are lattice vectors)
    graph:
        cell_shifts     (P, 3)   float  — integer cell shift per pair
        atomic_numbers  (N,)     int    — Z per atom; padding rows zero
        centers         (P,)     int    — receiver atom index per pair (i)
        others          (P,)     int    — sender atom index per pair (j)
        atom_mask       (N,)     bool   — True on real atoms
        pair_mask       (P,)     bool   — True on real pairs

The edge vector the model builds is `pos[others] - pos[centers] + cell_shifts @
cell`. In fairchem's own terms that is `edge_index = (others, centers)` with
`cell_offsets = cell_shifts`, i.e. exactly `ase.neighborlist.neighbor_list("ijS")`
read as `centers = i`, `others = j`, `cell_shifts = S`.
"""

import numpy as np
import jax.numpy as jnp

import warnings

import ase
from ase.build import bulk

from jaxtyping import Array

from .model import UMA

__all__ = ["atoms_to_inputs", "dummy_inputs"]


def atoms_to_inputs(
    atoms: ase.Atoms,
    model: UMA,
    *,
    skin: float = 0.0,
    float_dtype: np.dtype | type = np.float64,
    int_dtype: np.dtype | type = np.int64,
    bucket_strategy: str | None = None,
) -> tuple[Array, Array, dict[str, Array]]:
    """Build UMA model inputs `(pos, cell, graph)` from an `ase.Atoms`.

    Args:
        atoms: The structure.
        model: A `UMA` instance; carries the trained `cutoff` and the
            `max_neighbors` the density warning is against.
        skin: Verlet-skin radius (Å) added to the cutoff for the vesin list.
        float_dtype, int_dtype: Precision of the emitted arrays.
        bucket_strategy: `marathon.utils.next_size` strategy for padding the
            pair axis, or None (default) for exact shapes. The atom axis is
            never bucketed: with padding on it is `n_real + 1`, one dummy slot
            for the padding pairs to point at, as in `pet.inputs`. The atom
            count is fixed for the life of a relaxation, so bucketing it would
            only inflate every per-atom array.
    """
    cutoff = model.cutoff + skin
    centers, others, shifts = _full_neighbor_list(atoms, cutoff, float_dtype, int_dtype)
    _warn_on_dense_graph(centers, len(atoms), model.max_neighbors)

    n_real = len(atoms)
    n_pairs_real = len(centers)
    if bucket_strategy is None:
        n_atoms_padded, n_pairs_padded = n_real, max(n_pairs_real, 1)
    else:
        from marathon.utils import next_size

        n_atoms_padded = n_real + 1
        n_pairs_padded = max(next_size(n_pairs_real + 1, strategy=bucket_strategy), 1)

    # Padding pairs are self-loops on the last atom slot: the dummy atom when
    # padding is on, and the last real atom otherwise — which only happens for a
    # structure with no pairs at all, whose single padding pair is masked out.
    pad_atom_idx = n_atoms_padded - 1

    positions = np.zeros((n_atoms_padded, 3), dtype=float_dtype)
    atomic_numbers = np.zeros(n_atoms_padded, dtype=int_dtype)
    atom_mask = np.zeros(n_atoms_padded, dtype=bool)
    positions[:n_real] = atoms.positions
    atomic_numbers[:n_real] = atoms.get_atomic_numbers()
    atom_mask[:n_real] = True

    padded_centers = np.full(n_pairs_padded, pad_atom_idx, dtype=int_dtype)
    padded_others = np.full(n_pairs_padded, pad_atom_idx, dtype=int_dtype)
    padded_shifts = np.zeros((n_pairs_padded, 3), dtype=float_dtype)
    pair_mask = np.zeros(n_pairs_padded, dtype=bool)
    padded_centers[:n_pairs_real] = centers
    padded_others[:n_pairs_real] = others
    padded_shifts[:n_pairs_real] = shifts
    pair_mask[:n_pairs_real] = True

    graph = {
        "cell_shifts": jnp.asarray(padded_shifts),
        "atomic_numbers": jnp.asarray(atomic_numbers),
        "centers": jnp.asarray(padded_centers),
        "others": jnp.asarray(padded_others),
        "atom_mask": jnp.asarray(atom_mask),
        "pair_mask": jnp.asarray(pair_mask),
    }
    cell = np.asarray(atoms.cell[:], dtype=float_dtype)
    return jnp.asarray(positions), jnp.asarray(cell), graph


def dummy_inputs(model: UMA) -> tuple[Array, Array, dict[str, Array]]:
    """`(pos, cell, graph)` for a small bulk-Si cell — for warming the JIT cache."""
    return atoms_to_inputs(bulk("Si", "diamond", a=5.43, cubic=True), model)


# ---------------------------------------------------------------------------
# Neighbour list
# ---------------------------------------------------------------------------


def _full_neighbor_list(atoms, cutoff, float_dtype, int_dtype):
    """Full neighbour list via vesin (each ordered pair once)."""
    from vesin import NeighborList

    nl = NeighborList(cutoff=cutoff, full_list=True, sorted=True)
    i, j, shifts = nl.compute(
        points=atoms.positions,
        box=atoms.cell[:],
        periodic=atoms.pbc,
        quantities="ijS",
    )
    return i.astype(int_dtype), j.astype(int_dtype), shifts.astype(float_dtype)


def _warn_on_dense_graph(centers: np.ndarray, n_atoms: int, max_neighbors: int) -> None:
    """UMA truncates its *internal* graph to `max_neighbors` per atom. We never
    generate the graph internally, so an external list simply overrides that —
    but a denser list is then a different model from stock UMA."""
    if len(centers) == 0:
        return
    degree = int(np.bincount(centers, minlength=n_atoms).max())
    if degree > max_neighbors:
        warnings.warn(
            f"max degree {degree} exceeds UMA's max_neighbors={max_neighbors}; "
            "stock UMA would truncate to the nearest neighbours, this port does not",
            stacklevel=2,
        )
