"""Convert the upstream MACE-MP-0 medium PyTorch checkpoint to a
marathon-loadable (yaml + msgpack) pair for `sadmof.models.mace.MACE`.

Run from this directory:

    uv run --with torch --with mace-torch --with pyyaml python build.py

Inputs:
    ../../raw/mace-mp-0-medium/2023-12-03-mace-128-L1_epoch-199.model

Outputs (next to this script):
    model.yaml      — spec dict with handle sadmof.models.mace.model.MACE
    model.msgpack   — {"params": ..., "constants": ...} flax tree

The conversion logic was originally written for an earlier project; see
the README in this directory for what it does at a high level.
"""

from __future__ import annotations

import json
from pathlib import Path

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent
CHECKPOINT = HERE / "../../raw/mace-mp-0-medium/2023-12-03-mace-128-L1_epoch-199.model"
OUTPUT_DIR = HERE
HANDLE = "sadmof.models.mace.model.MACE"


def expand_to_z(arr: np.ndarray, atomic_numbers: np.ndarray, max_z: int) -> np.ndarray:
    """Expand `(num_elements, ...)` to `(max_z+1, ...)` indexed by Z."""
    expanded = np.zeros((max_z + 1, *arr.shape[1:]), dtype=arr.dtype)
    for idx, z in enumerate(atomic_numbers):
        expanded[z] = arr[idx]
    return expanded


def load_torch_model():
    import torch

    torch.set_default_dtype(torch.float64)
    model = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    model.eval()
    return model


def convert_e3nn_linear_weight(
    torch_weight_flat, irreps_in: e3nn.Irreps, irreps_out: e3nn.Irreps
) -> np.ndarray:
    """PyTorch e3nn flat weight → e3nn_jax FunctionalLinear flat weight.

    Both libraries use the same instruction ordering (irrep pairs with
    matching l/p), so the only thing we do is dtype + shape check.
    """
    linear = e3nn.FunctionalLinear(irreps_in, irreps_out)
    w_np = torch_weight_flat.detach().cpu().numpy().astype(np.float64)
    assert w_np.size == linear.num_weights, (
        f"weight size mismatch: torch has {w_np.size}, "
        f"e3nn_jax expects {linear.num_weights} for {irreps_in} -> {irreps_out}"
    )
    return w_np


def convert_radial_mlp(sd, prefix: str) -> dict:
    """Extract radial MLP weights — 4 Dense layers, no bias."""
    params = {}
    for idx in range(4):
        w = (
            sd[f"{prefix}.layer{idx}.weight"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        # PyTorch e3nn FullyConnectedNet stores weight as [in, out] (not transposed!)
        params[f"Dense_{idx}"] = {"kernel": w}
    return params


def convert_skip_tp(
    torch_weight_flat,
    irreps_in: e3nn.Irreps,
    irreps_out: e3nn.Irreps,
    num_elements_orig: int,
    atomic_numbers: np.ndarray,
    max_z: int,
) -> dict:
    """Convert FullyConnectedTensorProduct skip weights.

    Only 0e×0e→0e paths survive. Weights are rescaled by
    `sqrt((max_z+1) / num_elements)` to compensate for the path_weight
    normalisation change when expanding the element axis.
    """
    params = {}
    w_np = torch_weight_flat.detach().cpu().numpy().astype(np.float64)
    offset = 0
    scale = np.sqrt((max_z + 1) / num_elements_orig)

    for mul_in, ir_in in irreps_in:
        for mul_out, ir_out in irreps_out:
            if ir_out == ir_in:
                size = mul_in * num_elements_orig * mul_out
                block = w_np[offset : offset + size].reshape(
                    mul_in, num_elements_orig, mul_out
                )
                # SpeciesSkipConnection expects (num_elements, mul_out, mul_in)
                block = block.transpose(1, 2, 0) * scale
                block = expand_to_z(block, atomic_numbers, max_z)
                params[f"w_{ir_in}"] = block
                offset += size

    return params


def extract_symmetric_contraction(
    torch_model, layer_idx: int, atomic_numbers: np.ndarray, max_z: int
) -> tuple[dict, dict]:
    """U-matrices (constants) + per-element weights for one symmetric
    contraction module."""
    sc = torch_model.products[layer_idx].symmetric_contractions
    params, constants = {}, {}

    for c_idx, contraction in enumerate(sc.contractions):
        ir_out = list(sc.irreps_out)[c_idx].ir
        ir_out_str = str(ir_out)

        for nu in range(1, contraction.correlation + 1):
            U = contraction.U_tensors(nu).detach().cpu().numpy().astype(np.float64)
            constants[f"U_{ir_out_str}_{nu}"] = U

        w_max = contraction.weights_max.detach().cpu().numpy().astype(np.float64)
        params[f"w_{ir_out_str}_max"] = expand_to_z(w_max, atomic_numbers, max_z)

        # The lower-order weights are stored in reverse: weights[0] = nu=correlation-1, ...
        for w_idx, w in enumerate(contraction.weights):
            w_np = w.detach().cpu().numpy().astype(np.float64)
            nu = contraction.correlation - w_idx - 1
            params[f"w_{ir_out_str}_{nu}"] = expand_to_z(w_np, atomic_numbers, max_z)

    return params, constants


def main() -> None:
    print(f"loading {CHECKPOINT}")
    model = load_torch_model()
    sd = model.state_dict()

    atomic_numbers_np = sd["atomic_numbers"].cpu().numpy().astype(np.int32)
    num_elements_orig = len(atomic_numbers_np)
    max_z = int(atomic_numbers_np.max())
    num_elements = max_z + 1

    config = {
        "cutoff": float(sd["r_max"].item()),
        "num_bessel": int(sd["radial_embedding.bessel_fn.bessel_weights"].shape[0]),
        "num_polynomial_cutoff": int(sd["radial_embedding.cutoff_fn.p"].item()),
        "max_ell": 3,
        "num_interactions": int(sd["num_interactions"].item()),
        "correlation": 3,
        "num_elements": num_elements,
        "avg_num_neighbors": 61.964672446250916,
        "hidden_irreps_str": "128x0e+128x1o",
        "atomic_inter_scale": float(sd["scale_shift.scale"].item()),
        "atomic_inter_shift": float(sd["scale_shift.shift"].item()),
    }
    print(f"config: {json.dumps(config, indent=2)}")
    print(
        f"element mapping: {num_elements_orig} elements → "
        f"{num_elements} Z slots (max Z={max_z})"
    )

    params: dict = {}
    constants: dict = {}

    ae = sd["atomic_energies_fn.atomic_energies"].cpu().numpy().astype(np.float64)
    constants["atomic_energies"] = expand_to_z(ae, atomic_numbers_np, max_z)
    constants["bessel_weights"] = (
        sd["radial_embedding.bessel_fn.bessel_weights"].cpu().numpy().astype(np.float64)
    )

    # Node embedding (Embed table with path_weight baked in)
    irreps_in_embed = e3nn.Irreps(f"{num_elements_orig}x0e")
    irreps_out_embed = e3nn.Irreps("128x0e")
    w_flat = convert_e3nn_linear_weight(
        sd["node_embedding.linear.weight"], irreps_in_embed, irreps_out_embed
    )
    linear_embed = e3nn.FunctionalLinear(irreps_in_embed, irreps_out_embed)
    path_weight = float(linear_embed.instructions[0].path_weight)
    W_embed = w_flat.reshape(num_elements_orig, 128) * path_weight
    params["node_embedding"] = {
        "embedding": expand_to_z(W_embed, atomic_numbers_np, max_z),
    }

    hidden_irreps = e3nn.Irreps("128x0e+128x1o")
    sh_irreps = e3nn.Irreps("1x0e+1x1o+1x2e+1x3o")
    interaction_irreps = e3nn.Irreps("128x0e+128x1o+128x2e+128x3o")
    mul = 128

    for layer_idx in range(config["num_interactions"]):
        prefix = f"interactions.{layer_idx}"
        is_first = layer_idx == 0

        if is_first:
            in_irreps = e3nn.Irreps(f"{mul}x0e")
            skip_out_irreps = hidden_irreps
        else:
            in_irreps = hidden_irreps
            skip_out_irreps = e3nn.Irreps(f"{mul}x0e")

        params[f"linear_up_{layer_idx}"] = {
            "weight": convert_e3nn_linear_weight(
                sd[f"{prefix}.linear_up.weight"], in_irreps, in_irreps
            )
        }
        params[f"skip_tp_{layer_idx}"] = convert_skip_tp(
            sd[f"{prefix}.skip_tp.weight"],
            in_irreps,
            skip_out_irreps,
            num_elements_orig,
            atomic_numbers_np,
            max_z,
        )
        params[f"radial_mlp_{layer_idx}"] = convert_radial_mlp(
            sd, f"{prefix}.conv_tp_weights"
        )

        if is_first:
            conv_out_irreps = interaction_irreps
        else:
            target_ir_set = {ir for _, ir in interaction_irreps}
            tp_irreps = []
            for mul_in, ir_in in hidden_irreps:
                for _, ir_sh in sh_irreps:
                    for ir_out in ir_in * ir_sh:
                        if ir_out in target_ir_set:
                            tp_irreps.append((mul_in, ir_out))
            conv_out_irreps = e3nn.Irreps(tp_irreps)

        params[f"linear_{layer_idx}"] = {
            "weight": convert_e3nn_linear_weight(
                sd[f"{prefix}.linear.weight"], conv_out_irreps, interaction_irreps
            )
        }

        sc_params, sc_constants = extract_symmetric_contraction(
            model, layer_idx, atomic_numbers_np, max_z
        )
        params[f"symmetric_contraction_{layer_idx}"] = sc_params
        constants.setdefault(f"symmetric_contraction_{layer_idx}", {}).update(
            sc_constants
        )

        product_prefix = f"products.{layer_idx}"
        params[f"product_linear_{layer_idx}"] = {
            "weight": convert_e3nn_linear_weight(
                sd[f"{product_prefix}.linear.weight"],
                skip_out_irreps,
                skip_out_irreps,
            )
        }

        if is_first:
            params[f"readout_{layer_idx}"] = {
                "weight": convert_e3nn_linear_weight(
                    sd[f"readouts.{layer_idx}.linear.weight"],
                    hidden_irreps,
                    e3nn.Irreps("1x0e"),
                )
            }
        else:
            params[f"readout_{layer_idx}_linear1"] = {
                "weight": convert_e3nn_linear_weight(
                    sd[f"readouts.{layer_idx}.linear_1.weight"],
                    e3nn.Irreps("128x0e"),
                    e3nn.Irreps("16x0e"),
                )
            }
            params[f"readout_{layer_idx}_linear2"] = {
                "weight": convert_e3nn_linear_weight(
                    sd[f"readouts.{layer_idx}.linear_2.weight"],
                    e3nn.Irreps("16x0e"),
                    e3nn.Irreps("1x0e"),
                )
            }

    yaml_path = OUTPUT_DIR / "model.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({HANDLE: config}, f, default_flow_style=False)
    print(f"wrote {yaml_path}")

    full_params = {"params": params, "constants": constants}
    full_params_jax = jax.tree.map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, full_params
    )

    from marathon.io import write_msgpack

    msgpack_path = OUTPUT_DIR / "model.msgpack"
    write_msgpack(str(msgpack_path), full_params_jax)
    print(f"wrote {msgpack_path}")

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(full_params_jax["params"]))
    n_constants = sum(
        x.size for x in jax.tree_util.tree_leaves(full_params_jax["constants"])
    )
    print(f"learnable parameters: {n_params:,}")
    print(f"constants:            {n_constants:,}")


if __name__ == "__main__":
    main()
