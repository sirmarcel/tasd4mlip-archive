"""ASE Atoms → MACE model inputs `(pos, cell, graph)`.

`atoms_to_inputs` produces a single padded sample (not a multi-structure
`marathon.data.Batch`). We don't need batching — one structure at a time
is the whole sadmof inference / Hessian path. Padding still happens
(JAX needs static shapes), but stays inside one structure.

The model contract splits the continuous geometry from the static topology:
`pos` and `cell` are what `predict` (forces / stress) and the Hessian path
differentiate; `graph` is the connectivity the neighbour list hands us,
constant across small displacements (and exactly what the calculator's
Verlet cache holds).

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
"""

import numpy as np
import jax.numpy as jnp

import ase
from ase.build import bulk
from marathon.data import to_sample
from marathon.utils import next_size

from jaxtyping import Array

from .model import MACE

__all__ = ["atoms_to_inputs", "dummy_inputs"]


def atoms_to_inputs(
    atoms: ase.Atoms,
    cutoff: float,
    float_dtype: np.dtype | type = np.float64,
    int_dtype: np.dtype | type = np.int64,
    bucket_strategy: str = "multiples",
) -> tuple[Array, Array, dict[str, Array]]:
    """Build padded MACE model inputs `(pos, cell, graph)` from an `ase.Atoms`.

    Uses `marathon.data.to_sample` for the neighbour list (vesin) and
    `marathon.utils.next_size(bucket_strategy)` padding on both the atom and
    pair axes. The default coarse bucketing serves jit-cache reuse across
    structures; single-structure pipelines that pay compile per structure
    anyway should pass a tight strategy (e.g. `"multiples_of_16"`) — above
    65k pairs the default buckets to powers of two, up to 2x memory.
    `float_dtype` / `int_dtype` flow through `to_sample` so the padded
    arrays come out at the requested precision.
    """
    sample = to_sample(
        atoms,
        cutoff,
        energy=False,
        forces=False,
        stress=False,
        float_dtype=float_dtype,
        int_dtype=int_dtype,
    )
    s = sample.structure
    n_real = s["atomic_numbers"].shape[0]
    n_pairs_real = s["centers"].shape[0]

    n_atoms_padded = next_size(n_real + 1, strategy=bucket_strategy)
    n_pairs_padded = max(next_size(n_pairs_real + 1, strategy=bucket_strategy), 1)
    pad_atom_idx = n_real  # dummy slot for padding pairs to point at

    positions = np.zeros((n_atoms_padded, 3), dtype=float_dtype)
    atomic_numbers = np.zeros(n_atoms_padded, dtype=int_dtype)
    atom_mask = np.zeros(n_atoms_padded, dtype=bool)

    positions[:n_real] = s["positions"]
    atomic_numbers[:n_real] = s["atomic_numbers"]
    atom_mask[:n_real] = True

    centers = np.full(n_pairs_padded, pad_atom_idx, dtype=int_dtype)
    others = np.full(n_pairs_padded, pad_atom_idx, dtype=int_dtype)
    cell_shifts = np.zeros((n_pairs_padded, 3), dtype=float_dtype)
    pair_mask = np.zeros(n_pairs_padded, dtype=bool)

    centers[:n_pairs_real] = s["centers"]
    others[:n_pairs_real] = s["others"]
    cell_shifts[:n_pairs_real] = s["cell_shifts"]
    pair_mask[:n_pairs_real] = True

    cell = np.asarray(s["cell"], dtype=float_dtype)

    graph = {
        "cell_shifts": jnp.asarray(cell_shifts),
        "atomic_numbers": jnp.asarray(atomic_numbers),
        "centers": jnp.asarray(centers),
        "others": jnp.asarray(others),
        "atom_mask": jnp.asarray(atom_mask),
        "pair_mask": jnp.asarray(pair_mask),
    }
    return jnp.asarray(positions), jnp.asarray(cell), graph


def dummy_inputs(model: MACE) -> tuple[Array, Array, dict[str, Array]]:
    """`(pos, cell, graph)` for a small bulk-Si supercell — suitable for
    `model.init` style initialisation or for warming the JIT cache."""
    atoms = bulk("Si", "diamond", a=5.43) * [2, 2, 2]
    return atoms_to_inputs(atoms, model.cutoff)
