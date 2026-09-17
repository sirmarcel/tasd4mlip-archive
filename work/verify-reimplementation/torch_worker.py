"""Upstream torch predictions for one (model, dtype), on CPU.

Runs under a `uv run --with` overlay (`common.OVERLAY`), which is what keeps
`mace-torch` / `metatrain` out of the project environment.

    uv run --with mace-torch python torch_worker.py \
        --model mace --dtype float64 --structures RSM0274
"""

import argparse

import common


def main():
    args = build_parser().parse_args()
    out_dir = common.condition_dir("torch", args.model, args.dtype)

    print(f"torch/{args.model}/{args.dtype}: {len(args.structures)} structures")
    calc = calculator(args.model, args.dtype)
    common.run_condition(
        calc,
        args.structures,
        out_dir,
        meta={
            "stack": "torch",
            "model": args.model,
            "dtype": args.dtype,
            "checkpoint": str(common.RAW[args.model].relative_to(common.ROOT)),
            "device": "cpu",
        },
    )


def calculator(model, dtype):
    if model == "mace":
        return _mace(dtype)
    return _pet(model, dtype)


def _mace(dtype):
    from mace.calculators.mace import MACECalculator

    return MACECalculator(
        model_paths=str(common.RAW["mace"]), device="cpu", default_dtype=dtype
    )


def _pet(model, dtype):
    import torch
    from metatomic.torch.ase_calculator import MetatomicCalculator
    from metatrain.utils.io import load_model

    wrapped = load_model(str(common.RAW[model]))
    # The LLPR wrapper holds only uncertainty machinery; pet-jax's conversion
    # reads the inner module too, so both sides start from the same weights.
    # Cast before `export()`, which takes the dtype off the parameters.
    pet = wrapped.model.to(torch.float64 if dtype == "float64" else torch.float32)
    exported = pet.export()
    assert exported.capabilities().dtype == dtype
    return MetatomicCalculator(exported, device="cpu")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=common.MODELS)
    p.add_argument("--dtype", required=True, choices=common.DTYPES)
    p.add_argument("--structures", nargs="+", required=True)
    return p


if __name__ == "__main__":
    main()
