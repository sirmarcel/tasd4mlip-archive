"""ASE Atoms → PET model inputs `(pos, cell, graph)`.

PET's adaptive cutoff makes the graph carry two neighbour lists, built from
one selection pass that also sizes `k_sel` (PET's per-atom neighbour budget):

- The **raw** NL (vesin ball at `cutoff + skin`), standard keys plus
  `k_sel_sizer`. `pet.energy.get_energy_fn` re-runs the adaptive selection
  over it every call, keeping the cutoff inside the autograd graph (shadow
  forces); `no_shadow` stop-gradients the cutoff but never changes which
  pairs survive.
- The **selected** adjacency (`sel_centers`, `sel_others`, `sel_cell_shifts`):
  the ~`num_neighbors_adaptive`-neighbour graph the model actually attends
  over — the tight, offset-resolved 1-hop adjacency for
  `sparse.sparsity_pattern` (the raw ball would only inflate the colouring).

Returns `(pos, cell, graph)`, mirroring `mace.inputs.atoms_to_inputs`.
"""

import numpy as np
import jax
import jax.numpy as jnp

import ase
from petjax import to_structure
from petjax.select import _select_edges
from petjax.structure import _bucket_or
from petjax.utils import edge_displacements

from jaxtyping import Array

__all__ = ["atoms_to_inputs"]


def atoms_to_inputs(
    atoms: ase.Atoms,
    model,
    *,
    skin: float = 0.5,
    extra_neighbors: int = 4,
    bucket_strategy: str = "multiples",
    float_dtype: np.dtype | type = np.float64,
    int_dtype: np.dtype | type = np.int64,
) -> tuple[Array, Array, dict[str, Array]]:
    """Build PET model inputs `(pos, cell, graph)` from an `ase.Atoms`.

    Args:
        atoms: The structure.
        model: A `petjax.UPET` instance; carries the adaptive-selection
            hypers (`cutoff`, `cutoff_width_adaptive`,
            `adaptive_cutoff_method`, `num_neighbors_adaptive`).
        skin: Verlet-skin radius (Å) added to the trained cutoff for the raw
            vesin NL.
        extra_neighbors: Slack added to the measured per-atom neighbour count
            when sizing `k_sel`, so the rectangular layout has headroom.
        bucket_strategy: `marathon.utils.next_size` strategy for padding the
            pair axis and rounding `k_sel` (the atom axis is fixed at
            `n_real + 1`).
        float_dtype, int_dtype: Precision of the emitted arrays.
    """
    structure = to_structure(
        atoms,
        model.cutoff,
        skin=skin,
        bucket_strategy=bucket_strategy,
        float_dtype=float_dtype,
        int_dtype=int_dtype,
    )

    # One selection pass: sizes k_sel and yields the sparsity adjacency.
    N_padded = structure["positions"].shape[0]
    R_ij = edge_displacements(
        structure["positions"],
        structure["centers"],
        structure["others"],
        structure["cell_shifts"],
        structure["cell"],
    )
    _, selected = _select_edges(
        R_ij,
        structure["centers"],
        structure["others"],
        structure["pair_mask"],
        N_padded,
        model.num_neighbors_adaptive,
        model.cutoff,
        model.cutoff_width_adaptive,
        method=model.adaptive_cutoff_method,
    )

    counts = jax.ops.segment_sum(
        selected.astype(int),
        structure["centers"],
        num_segments=N_padded,
        indices_are_sorted=True,
    )
    k_sel_actual = int(counts.max())
    # T = k_sel edge tokens + 1 central token; bucket T, then k_sel = T - 1.
    # Mirrors UPETCalculator._build_structure.
    k_sel = _bucket_or(k_sel_actual + 1 + extra_neighbors, bucket_strategy) - 1

    # Selected real–real adjacency for the sparsity pattern. Padding pairs
    # (dummy-atom self-loops) and pairs touching padding are dropped.
    n_real = int(structure["atom_mask"].sum())
    sel = np.asarray(selected).astype(bool)  # `selected` is a jax array → host
    centers = structure["centers"]  # already numpy (to_structure)
    others = structure["others"]
    real_sel = sel & (centers < n_real) & (others < n_real)

    graph = {
        "atomic_numbers": jnp.asarray(structure["atomic_numbers"]),
        "atom_mask": jnp.asarray(structure["atom_mask"]),
        "centers": jnp.asarray(structure["centers"]),
        "others": jnp.asarray(structure["others"]),
        "cell_shifts": jnp.asarray(structure["cell_shifts"]),
        "reverse": jnp.asarray(structure["reverse"]),
        "pair_mask": jnp.asarray(structure["pair_mask"]),
        "k_sel_sizer": jnp.zeros(k_sel, dtype=bool),
        "sel_centers": jnp.asarray(centers[real_sel]),
        "sel_others": jnp.asarray(others[real_sel]),
        "sel_cell_shifts": jnp.asarray(
            np.asarray(structure["cell_shifts"])[real_sel].astype(int_dtype)
        ),
    }
    return jnp.asarray(structure["positions"]), jnp.asarray(structure["cell"]), graph
