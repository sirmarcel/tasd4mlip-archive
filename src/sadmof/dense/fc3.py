"""Dense third-order slices: a JVP through the dense positions-Hessian.

The sibling of `sparse.fc3` with no pattern and no coloring, and the
ground-truth reference for it: D³E·(d, ·, ·) is the directional derivative of
the Hessian, computed here by differentiating `dense.get_dense_hessian_fn`'s
chunked HVP sweep — peak memory stays bounded by `chunk_size` exactly as for
the dense Hessian itself.

    fc3_slice = get_dense_fc3_slice_fn(model.energy, chunk_size=...)
    M, aux = jax.jit(fc3_slice)(params, pos, cell, graph, d)   # (N, 3, N, 3)

A full dense tensor, affordable only for small systems, is the stack of slices
along the 3N Cartesian basis directions.
"""

import jax

from collections.abc import Callable

from jaxtyping import Array

from .hessian import get_dense_hessian_fn

__all__ = ["get_dense_fc3_slice_fn"]


def get_dense_fc3_slice_fn(
    energy_fn: Callable,
    *,
    chunk_size: int | None = None,
    remat: bool = False,
):
    """Build a jittable dense third-order slice D³E·(d, ·, ·).

    Returns a jittable `fc3_slice(params, pos, cell, graph, d) -> (M, aux)`
    with `M` of shape `(N, 3, N, 3)`: the derivative of the dense Hessian in
    the Cartesian direction `d` of shape `(N, 3)`. The per-call cost is one
    JVP through the full `3N`-HVP sweep, roughly two dense Hessians.

    Args:
        energy_fn: An MLIP energy callable `(params, pos, cell, graph) ->
            (scalar, aux)`. Only the scalar total is differentiated. The aux
            rides through `jax.jvp`, which float aux (MACE) survives; PET's
            integer overflow-flag leaf is untested here.
        chunk_size: Coordinates (HVPs) processed in parallel per batch.
        remat: Wrap the energy in `jax.checkpoint`.
    """
    hessian = get_dense_hessian_fn(energy_fn, chunk_size=chunk_size, remat=remat)

    def fc3_slice(params, pos: Array, cell: Array, graph: dict[str, Array], d: Array):
        def hessian_at(p: Array):
            return hessian(params, p, cell, graph)

        (_, aux), (dH, _) = jax.jvp(hessian_at, (pos,), (d,))
        return dH, aux

    return fc3_slice
