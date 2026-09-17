"""Dense positions-Hessian of an MLIP energy via chunked forward-over-reverse HVPs.

The sibling of `sparse.hessian` for the dense Hessian case.

There is no pattern and no coloring: the dense Hessian is `3N` Hessian-vector
products against the identity basis, so the "coloring" is trivially the identity
(one color per coordinate). The basis is swept in chunks of `chunk_size` via
`jax.lax.map(..., batch_size=...)` over the HVP, bounding peak AD memory. Seeds
are basis *indices*, with the one-hot tangent built inside the HVP, so the
`(3N, 3N)` identity is never materialised.

    hessian = get_dense_hessian_fn(model.energy, chunk_size=...)
    H, aux = jax.jit(hessian)(params, pos, cell, graph)   # (N, 3, N, 3)

Like `sparse.get_hessian_fn`, the function handed to JAX differentiates a single
`(N, 3)` positions array; `params`, `cell`, and `graph` are arguments of the
returned `hessian`, captured by the inner closure as tracers under `jax.jit` —
never baked into the compiled executable as constants.
"""

import jax
import jax.numpy as jnp

from collections.abc import Callable

from jaxtyping import Array

__all__ = ["get_dense_hessian_fn"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_dense_hessian_fn(
    energy_fn: Callable,
    *,
    chunk_size: int | None = None,
    remat: bool = False,
):
    """Build a jittable dense positions-Hessian.

    Returns a jittable `hessian(params, pos, cell, graph) -> (H, aux)`, with
    `H` of shape `(N, 3, N, 3)` plus the energy's aux (per-atom energies for
    MACE and D3, the k_sel overflow flag for PET). The aux rides the linearize
    forward pass, so surfacing it costs nothing.

    The per-call HVP count is `3N` (one per coordinate): the dense Hessian has
    no exploitable structure, so there is no coloring to compress it.

    Args:
        energy_fn: An MLIP energy callable `(params, pos, cell, graph) ->
            (scalar, aux)`, e.g. `MACE.energy` or a `pet.get_energy_fn(...)`
            closure. Only the scalar total is differentiated.
        chunk_size: Coordinates (HVPs) processed in parallel per batch.
        remat: Wrap the energy in `jax.checkpoint`.
    """

    def hessian(params, pos: Array, cell: Array, graph: dict[str, Array]):
        n = pos.shape[0]
        dim = 3 * n

        def raw_energy(p: Array) -> tuple[Array, Array]:
            return energy_fn(params, p, cell, graph)

        energy = jax.checkpoint(raw_energy) if remat else raw_energy

        # fwd-over-rev: linearise the gradient once, then apply its JVP to each
        # identity basis vector. The residuals of `linearize(grad(energy))` are
        # computed once and shared across the whole sweep; `remat` controls
        # whether they are held live or recomputed per HVP.
        _, hvp_fn, aux = jax.linearize(jax.grad(energy, has_aux=True), pos, has_aux=True)

        def single_hvp(idx: Array) -> Array:
            tangent = jax.nn.one_hot(idx, dim, dtype=pos.dtype).reshape(n, 3)
            return hvp_fn(tangent).reshape(dim)

        indices = jnp.arange(dim)
        if chunk_size is None or chunk_size >= dim:
            compressed = jax.vmap(single_hvp)(indices)  # (dim, dim)
        else:
            compressed = jax.lax.map(single_hvp, indices, batch_size=chunk_size)
        # compressed[k, i] = H[i, k]; transpose so the seed axis is the column,
        # giving the standard (N, 3, N, 3) layout H[a, alpha, b, beta].
        return compressed.T.reshape(n, 3, n, 3), aux

    return hessian
