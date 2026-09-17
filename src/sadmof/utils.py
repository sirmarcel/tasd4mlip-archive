"""Generic utilities shared across sadmof models."""

import jax
import jax.numpy as jnp

__all__ = ["cast_floats"]


def cast_floats(tree, dtype):
    """Cast every floating-point leaf of `tree` to `dtype`; pass others through.

    Used to align checkpoint params (saved as fp64) with the calculator's
    runtime dtype, and elsewhere we need to flip a JAX pytree's precision
    without touching the integer index / boolean mask leaves.
    """

    def _cast(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return jnp.asarray(x, dtype=dtype)
        return x

    return jax.tree.map(_cast, tree)
