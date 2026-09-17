"""Sparse positions-Hessian of an MLIP energy via asdex.

The *decompression* stage of detection → coloring → decompression. This module
turns a colored pattern plus an MLIP energy into a jittable sparse Hessian:

    coloring = hessian_coloring(graph, hops)
    hessian = get_hessian_fn(model.energy, coloring)
    H, aux = jax.jit(hessian)(params, pos, cell, graph)   # BCOO (N, 3, N, 3)

The function handed to asdex differentiates a single `(N, 3)` positions array;
`params`, `cell`, and `graph` are arguments of the returned `hessian`, captured
as tracers under `jax.jit` — never baked into the executable as constants. The
colored pattern carries the `(N, 3)` aval, so positions flow through natively.
"""

import jax

from collections.abc import Callable

import asdex

from jaxtyping import Array

__all__ = ["get_hessian_fn"]


def get_hessian_fn(
    energy_fn: Callable,
    coloring: asdex.ColoredPattern,
    *,
    chunk_size: int | None = None,
    remat: bool = False,
):
    """Build a jittable sparse positions-Hessian from a colored pattern.

    Returns a jittable `hessian(params, pos, cell, graph) -> (BCOO, aux)`,
    Hessian of shape `(N, 3, N, 3)` plus the energy's aux (per-atom energies
    for MACE, the k_sel overflow flag for PET). In `fwd_over_rev` mode the aux
    rides the linearize forward pass, so surfacing it costs nothing.

    `coloring.num_colors` is the per-call HVP count.

    Args:
        energy_fn: An MLIP energy callable `(params, pos, cell, graph) ->
            (scalar, aux)`, e.g. `MACE.energy`. Only the scalar total is
            differentiated.
        coloring: An `asdex.ColoredPattern` built from
            `sparse.pattern.sparsity_pattern`.
        chunk_size: Colors processed in parallel per HVP batch.
        remat: Wrap the energy in `jax.checkpoint`.
    """

    def hessian(params, pos: Array, cell: Array, graph: dict[str, Array]):
        def raw_energy(p: Array) -> tuple[Array, Array]:
            return energy_fn(params, p, cell, graph)

        energy = jax.checkpoint(raw_energy) if remat else raw_energy

        hess_fn = asdex.hessian_from_coloring(
            energy, coloring, has_aux=True, chunk_size=chunk_size
        )
        return hess_fn(pos)

    return hessian
