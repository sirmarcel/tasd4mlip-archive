"""Our JAX predictions for one (model, dtype, device, matmul), on CPU or GPU.

    JAX_PLATFORMS=cpu uv run python jax_worker.py \
        --model mace --dtype float64 --structures RSM0274

One process per condition: `jax_enable_x64` and `jax_default_matmul_precision`
are global flags set at construction, so an fp32 condition sharing a process
with an fp64 one would not be the fp32 stack.

`--matmul` is pinned here rather than left to the calculators, which disagree:
`UPETCalculator` defaults to `high` and mutates the global config, our
`MACECalculator` inherits JAX's default (TF32 on GPU).
"""

import argparse

import common


def main():
    args = build_parser().parse_args()
    out_dir = common.condition_dir("jax", args.model, args.dtype, args.device, args.matmul)

    print(f"jax/{args.model}/{out_dir.name}: {len(args.structures)} structures")
    calc, device = calculator(args.model, args.dtype, args.matmul)
    if device != args.device:
        raise RuntimeError(
            f"--device {args.device} but JAX landed on {device}; refusing to write "
            f"{device} numbers into {out_dir.name}"
        )
    print(f"  device: {device}")
    common.run_condition(
        calc,
        args.structures,
        out_dir,
        meta={
            "stack": "jax",
            "model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            # Explicit `null` where it does not apply (fp64), never absent: an
            # absent key would have to be read as "whatever the default was".
            "matmul_precision": args.matmul,
            "jax_device": device,
            "checkpoint": common.CHECKPOINTS[args.model],
            "skin": common.SKIN,
            "no_shadow": False,
        },
    )


def calculator(model, dtype, matmul):
    """The configured calculator plus the JAX device it actually landed on."""
    import jax

    # Global flags are read when arrays are first built, so the sadmof imports
    # below must stay inside this function, after the update.
    if matmul is not None:
        jax.config.update("jax_default_matmul_precision", matmul)

    from sadmof.paths import checkpoint

    ckpt = checkpoint(common.CHECKPOINTS[model])
    if model == "mace":
        from sadmof.models.mace import MACECalculator

        calc = MACECalculator.from_checkpoint(
            ckpt, default_dtype=dtype, skin=common.SKIN, stress=True
        )
    else:
        from sadmof.models.pet import UPETCalculator

        calc = UPETCalculator.from_checkpoint(
            str(ckpt),
            default_dtype=dtype,
            skin=common.SKIN,
            stress=True,
            matmul_precision=None,
        )
    return calc, jax.local_devices()[0].platform


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=common.MODELS)
    p.add_argument("--dtype", required=True, choices=common.DTYPES)
    p.add_argument("--device", default="cpu", choices=common.DEVICES)
    p.add_argument(
        "--matmul",
        default=None,
        choices=common.MATMULS,
        help="fp32 matmul accumulation mode; unset leaves JAX's global default",
    )
    p.add_argument("--structures", nargs="+", required=True)
    return p


if __name__ == "__main__":
    main()
