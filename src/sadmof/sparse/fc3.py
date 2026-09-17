"""Sparse third-order slices via the Hessian machinery.

The slice M_d = D³E·(d, ·, ·) is the positions-Hessian of the scalar
g(x) = d·∇E(x). A nonzero (j, k) entry requires an atom within the model's
reach of both j and k, so nnz(M_d) is contained in the Hessian pattern and the
Hessian star coloring recovers M_d unchanged.

For a phonon mode λ, pass the Cartesian displacement direction
d = W[:, λ] / sqrt(m) (see `observables.linewidths.mode_direction`); the
mass and ħ/2ω factors of the three-phonon matrix element are applied in the
linewidth kernel, not here.
"""

import jax
import jax.numpy as jnp

from collections.abc import Callable

import asdex

from jaxtyping import Array

__all__ = ["get_fc3_slice_fn"]


def get_fc3_slice_fn(
    energy_fn: Callable,
    coloring: asdex.ColoredPattern,
    *,
    chunk_size: int | None = None,
    remat: bool = False,
):
    """Build a jittable sparse third-order slice D³E·(d, ·, ·).

    Returns a jittable `fc3_slice(params, pos, cell, graph, d) -> (BCOO, aux)`
    of shape `(N, 3, N, 3)`, the third-derivative tensor contracted with the
    Cartesian direction `d` of shape `(N, 3)`. Reuses the Hessian coloring
    verbatim; `coloring.num_colors` third-order probes per call, each roughly
    the cost of a few HVPs.

    Args:
        energy_fn: An MLIP energy callable `(params, pos, cell, graph) ->
            (scalar, aux)`, e.g. `MACE.energy`. Only the scalar total is
            differentiated.
        coloring: The `asdex.ColoredPattern` of the *Hessian* pattern.
        chunk_size: Colors processed in parallel per probe batch.
        remat: Wrap the energy in `jax.checkpoint`.
    """

    def fc3_slice(params, pos: Array, cell: Array, graph: dict[str, Array], d: Array):
        def raw_slice_energy(p: Array) -> tuple[Array, Array]:
            grad, aux = jax.grad(lambda q: energy_fn(params, q, cell, graph), has_aux=True)(
                p
            )
            return jnp.vdot(d, grad), aux

        slice_energy = jax.checkpoint(raw_slice_energy) if remat else raw_slice_energy

        slice_fn = asdex.hessian_from_coloring(
            slice_energy, coloring, has_aux=True, chunk_size=chunk_size
        )
        return slice_fn(pos)

    return fc3_slice
