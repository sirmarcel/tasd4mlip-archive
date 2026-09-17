"""PET calculator + checkpoint loader.

`UPETCalculator` is re-exported verbatim from pet-jax (a complete ASE
calculator: adaptive cutoff, Verlet-skin NL cache, stress). `load_pet` is the
loader the Hessian path uses: it returns `(model, params, metadata)` —
`pet.inputs` / `pet.energy` consume only the model (the hypers live on it),
`params` feeds the resulting `energy_fn` — with params cast to the requested
precision (PET-MAD ships fp32; fp64 promotes the cached params and enables
JAX x64).
"""

import jax
import jax.numpy as jnp

from pathlib import Path

from petjax import UPETCalculator

from ...utils import cast_floats

__all__ = ["UPETCalculator", "load_pet"]


def load_pet(checkpoint: str | Path, *, dtype: str = "float64"):
    """Load a PET checkpoint into `(model, params, metadata)`.

    Args:
        checkpoint: A pet-jax checkpoint directory (`model.msgpack` +
            `metadata.yaml`).
        dtype: `"float64"` (default, for Hessian work — also enables JAX
            x64) or `"float32"`.

    Returns:
        `(model, params, metadata)` where `model` is a `petjax.UPET`
        (carrying the adaptive-selection hypers as attributes), `params` is
        its (cast) parameter pytree — `energy_scale` included — and
        `metadata` carries `config` / `shifts`.
    """
    from petjax import UPET, load_checkpoint

    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)

    params, metadata = load_checkpoint(str(checkpoint))
    model = UPET(**metadata["config"])
    params = cast_floats(params, jnp.float64 if dtype == "float64" else jnp.float32)
    return model, params, metadata
