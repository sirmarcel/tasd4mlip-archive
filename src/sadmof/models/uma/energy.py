"""UMA energy function for the HVP / sparse-Hessian pipeline.

`get_energy_fn` returns `energy_fn(params, pos, cell, graph) -> (scalar, aux)`
matching the contract `dense.get_dense_hessian_fn` and `sparse.get_hessian_fn`
differentiate, with `aux` the per-atom energies.

Two per-system constants are deliberately outside the returned energy: the task
normaliser's `mean` and the element references, both left in `load_uma`'s
metadata. Neither depends on the coordinates, so they contribute nothing to a
force or a Hessian and only cost precision in the sum — a calculator adds them
back in fp64 numpy. What is left, `rmsd * sum_i e_i`, is exactly the sum of the
per-atom decomposition `aux` carries.
"""

import jax.numpy as jnp

from collections.abc import Callable

from .model import UMA

__all__ = ["get_energy_fn"]


def get_energy_fn(model: UMA) -> Callable:
    """Build `energy_fn(params, pos, cell, graph) -> (energy, per_atom)`.

    Args:
        model: A `UMA` instance, already carrying the task normaliser's scale
            (see `load_uma`).

    Returns:
        A closure over the static model config. `energy` is the total energy in
        eV without the normaliser's `mean` or the element references; `per_atom`
        is its per-atom decomposition, zero on padding atoms.
    """

    def energy_fn(params, pos, cell, graph):
        per_atom = model(params, pos, cell, graph)
        return jnp.sum(per_atom), per_atom

    return energy_fn
