"""PET energy function for the HVP / sparse-Hessian pipeline.

`get_energy_fn` returns `energy_fn(params, pos, cell, graph) -> (scalar, aux)`
matching the contract `sparse.hessian.get_hessian_fn` differentiates. It runs
PET's adaptive selection (`truncate`) over the raw NL in `graph` every call —
so the cutoff procedure stays inside the autograd graph — then sums the
per-atom model output.

The only shadow/no_shadow difference is the `no_shadow` kwarg, threaded into
`truncate`: it stop-gradients the adaptive cutoff. Inputs are identical for
both modes (see `pet.inputs`). The sparse path uses `no_shadow=True` (coupling
confined to the selected NL → tight pattern); the shadow path
(`no_shadow=False`) extends coupling past the selected NL and is dense only.

Composition shifts are linear in atom count, so their Hessian is zero — we
sum only the scaled model energy (`energy_scale` lives in the param tree and
is applied inside `UPET.__call__`) and never add the shifts.
"""

import jax.numpy as jnp

from collections.abc import Callable

from petjax import truncate

__all__ = ["get_energy_fn"]


def get_energy_fn(model, *, no_shadow: bool = False) -> Callable:
    """Build `energy_fn(params, pos, cell, graph) -> (energy, overflow)`.

    Args:
        model: A `petjax.UPET` instance; carries the adaptive-selection
            hypers (`cutoff`, `cutoff_width_adaptive`,
            `adaptive_cutoff_method`, `num_neighbors_adaptive`).
        no_shadow: If True, `jax.lax.stop_gradient` the adaptive cutoff
            (cuts shadow-force contributions). The sparse path wants this;
            the shadow path leaves it False (dense Hessians only).

    Returns:
        A closure over the static model config. `energy` is the scaled total
        energy (no composition shifts); `overflow` is PET's k_sel-overflow
        flag, returned as the aux (the Hessian factories pass it through).
    """

    def energy_fn(params, pos, cell, graph):
        structure = {
            "positions": pos,
            "cell": cell,
            "atomic_numbers": graph["atomic_numbers"],
            "atom_mask": graph["atom_mask"],
            "centers": graph["centers"],
            "others": graph["others"],
            "cell_shifts": graph["cell_shifts"],
            "reverse": graph["reverse"],
            "pair_mask": graph["pair_mask"],
            "k_sel_sizer": graph["k_sel_sizer"],
        }
        truncated, overflow = truncate(
            structure,
            model.num_neighbors_adaptive,
            model.cutoff,
            model.cutoff_width_adaptive,
            method=model.adaptive_cutoff_method,
            no_shadow=no_shadow,
        )
        return jnp.sum(model.apply(params, **truncated)), overflow

    return energy_fn
