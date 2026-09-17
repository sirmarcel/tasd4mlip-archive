# /// script
# requires-python = ">=3.13, <3.14"
# dependencies = [
#     "fairchem-core",
#     "torch",
#     "torchtnt",
#     "ase",
#     "importlib_metadata",
# ]
#
# [tool.uv]
# override-dependencies = ["numpy>=2.0"]
# [tool.uv.sources]
# fairchem-core = { git = "https://github.com/n-gao/fairchem.git", branch = "ng/sybmolic_shapes", subdirectory = "packages/fairchem-core" }
# torchtnt = { git = "https://github.com/meta-pytorch/tnt" }
# ///
"""Convert the UMA-S-1.2 (fairchem) checkpoint to a torch-free `.npz` + `.json` pair
for `sadmof.models.uma`.

Run from this directory:

    uv run build.py

Inputs:
    ../../raw/uma/checkpoints/uma-s-1p2.pt

Outputs (next to this script):
    config.json   — hyperparameters, normalizers, element references, dataset list
    model.npz     — every weight and buffer, flat "/"-joined keys, fp32 as stored

MOLE is deliberately *not* merged here. The 64 experts of every SO(2) linear are
kept raw, together with the routing MLP and the composition / charge / spin /
dataset embeddings, so `sadmof.models.uma.load_uma` can mix them in numpy for one
concrete (composition, charge, spin, dataset). The mixing weights depend on the
composition, so a merged checkpoint would only be valid for one structure.

The output is ~1.2 GB (289 M expert parameters at fp32) and gitignored; see the
README for how to re-fetch the raw checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
CHECKPOINT = HERE / "../../raw/uma/checkpoints/uma-s-1p2.pt"
# `GaussianSmearing(0, cutoff, n, basis_width_scalar)`, hardcoded at the
# eSCN-MD backbone's construction site rather than stored in the checkpoint.
# `check_gaussian_basis_width` pins it to the module that was just loaded.
GAUSSIAN_BASIS_WIDTH = 2.0


def load_backbone_and_head():
    """Load the predictor on CPU with MOLE left unmerged."""
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    predictor = load_predict_unit(
        str(CHECKPOINT.resolve()),
        device="cpu",
        inference_settings=InferenceSettings(
            tf32=False,
            activation_checkpointing=False,
            merge_mole=False,
            compile=False,
            external_graph_gen=True,
            internal_graph_gen_version=2,
        ),
    )
    module = predictor.model.module
    return predictor, module.backbone, module.output_heads["energyandforcehead"]


def numpy_of(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def radial_mlp(prefix: str, module) -> dict[str, np.ndarray]:
    """`RadialMLP` = Linear, LayerNorm, SiLU, Linear, LayerNorm, SiLU, Linear."""
    net = module.net
    out = {}
    for idx in (0, 3, 6):
        out[f"{prefix}/lin{idx}/w"] = numpy_of(net[idx].weight)
        out[f"{prefix}/lin{idx}/b"] = numpy_of(net[idx].bias)
    for idx in (1, 4):
        out[f"{prefix}/ln{idx}/w"] = numpy_of(net[idx].weight)
        out[f"{prefix}/ln{idx}/b"] = numpy_of(net[idx].bias)
    return out


def so2_conv(prefix: str, conv, mmax: int) -> dict[str, np.ndarray]:
    """One `SO2_Convolution`. `fc_m0` and every `so2_m_conv[*].fc` are MOLE
    (shape `(num_experts, out, in)`); the radial MLP, when present, is not."""
    out = {f"{prefix}/fc_m0/w": numpy_of(conv.fc_m0.weights)}
    out[f"{prefix}/fc_m0/b"] = numpy_of(conv.fc_m0.bias)
    for m in range(1, mmax + 1):
        out[f"{prefix}/m{m}/w"] = numpy_of(conv.so2_m_conv[m - 1].fc.weights)
    if conv.rad_func is not None:
        out.update(radial_mlp(f"{prefix}/rad", conv.rad_func))
    return out


def norm_layer(prefix: str, norm) -> dict[str, np.ndarray]:
    return {
        f"{prefix}/w": numpy_of(norm.affine_weight),
        f"{prefix}/b": numpy_of(norm.affine_bias),
    }


def check_gaussian_basis_width(backbone) -> None:
    """`GaussianSmearing.coeff` is the only trace the width leaves in the loaded
    module, so invert it and compare, at the tolerance of the offset buffer's
    own (default-dtype) rounding."""
    smearing = backbone.distance_expansion
    spacing = float(smearing.offset[1] - smearing.offset[0])
    expected = -0.5 / (GAUSSIAN_BASIS_WIDTH * spacing) ** 2
    assert np.isclose(smearing.coeff, expected, rtol=1e-6), (
        f"basis width {GAUSSIAN_BASIS_WIDTH} implies coeff {expected}, "
        f"checkpoint has {smearing.coeff}"
    )


def to_m_permutation(backbone) -> np.ndarray:
    """`mappingReduced.to_m` as a permutation: row `k` of the m'-ordered vector
    is entry `perm[k]` of the L-ordered one. `sadmof.models.uma.model` re-derives
    this from `lmax`/`mmax` and checks itself against the stored copy."""
    to_m = numpy_of(backbone.mappingReduced.to_m)
    assert np.array_equal(np.sort(to_m, axis=1)[:, -1], np.ones(to_m.shape[0]))
    return np.argmax(to_m, axis=1).astype(np.int64)


def main() -> None:
    predictor, backbone, head = load_backbone_and_head()
    check_gaussian_basis_width(backbone)

    lmax, mmax = backbone.lmax, backbone.mmax
    arrays: dict[str, np.ndarray] = {}

    # -- system-level embeddings feeding both the routing MLP and every node --
    arrays["sphere_embedding"] = numpy_of(backbone.sphere_embedding.weight)
    arrays["source_embedding"] = numpy_of(backbone.source_embedding.weight)
    arrays["target_embedding"] = numpy_of(backbone.target_embedding.weight)
    arrays["composition_embedding"] = numpy_of(backbone.composition_embedding.weight)
    arrays["charge_embedding"] = numpy_of(backbone.charge_embedding.rand_emb.weight)
    arrays["spin_embedding"] = numpy_of(backbone.spin_embedding.rand_emb.weight)
    dataset_names = sorted(backbone.dataset_embedding.dataset_emb_dict.keys())
    for name in dataset_names:
        arrays[f"dataset_embedding/{name}"] = numpy_of(
            backbone.dataset_embedding.dataset_emb_dict[name].weight
        )
    arrays["mix_csd/w"] = numpy_of(backbone.mix_csd.weight)
    arrays["mix_csd/b"] = numpy_of(backbone.mix_csd.bias)

    for idx in (0, 2, 4):
        arrays[f"routing_mlp/lin{idx}/w"] = numpy_of(backbone.routing_mlp[idx].weight)
        arrays[f"routing_mlp/lin{idx}/b"] = numpy_of(backbone.routing_mlp[idx].bias)

    # -- edge degree embedding, blocks, final norm --
    arrays.update(radial_mlp("edge_degree/rad", backbone.edge_degree_embedding.rad_func))

    for i, block in enumerate(backbone.blocks):
        arrays.update(norm_layer(f"blocks/{i}/norm_1", block.norm_1))
        arrays.update(norm_layer(f"blocks/{i}/norm_2", block.norm_2))
        arrays.update(so2_conv(f"blocks/{i}/so2_conv_1", block.edge_wise.so2_conv_1, mmax))
        arrays.update(so2_conv(f"blocks/{i}/so2_conv_2", block.edge_wise.so2_conv_2, mmax))
        aw = block.atom_wise
        arrays[f"blocks/{i}/atom_wise/scalar_mlp/w"] = numpy_of(aw.scalar_mlp[0].weight)
        arrays[f"blocks/{i}/atom_wise/scalar_mlp/b"] = numpy_of(aw.scalar_mlp[0].bias)
        for n in (1, 2):
            lin = getattr(aw, f"so3_linear_{n}")
            arrays[f"blocks/{i}/atom_wise/so3_linear_{n}/w"] = numpy_of(lin.weight)
            arrays[f"blocks/{i}/atom_wise/so3_linear_{n}/b"] = numpy_of(lin.bias)

    arrays.update(norm_layer("norm", backbone.norm))

    # -- head: one MOLE expert per dataset, selected one-hot --
    for idx in (0, 2, 4):
        arrays[f"head/lin{idx}/w"] = numpy_of(head.head.energy_block[idx].weights)
        arrays[f"head/lin{idx}/b"] = numpy_of(head.head.energy_block[idx].bias)

    # -- rotation constants --
    for l in range(lmax + 1):
        arrays[f"Jd_{l}"] = numpy_of(getattr(backbone, f"Jd_{l}"))

    # -- the m-ordering permutation --
    arrays["to_m_perm"] = to_m_permutation(backbone)

    # -- per-task normalizers and element references --
    element_references = {}
    normalizers = {}
    for task_name, task in predictor.model.module.tasks.items():
        if task.property != "energy":
            continue
        dataset = task.datasets[0]
        normalizers[dataset] = {
            "mean": float(task.normalizer.mean),
            "rmsd": float(task.normalizer.rmsd),
        }
        if task.element_references is not None:
            element_references[dataset] = numpy_of(
                task.element_references.element_references
            ).astype(np.float64)
    for dataset, refs in element_references.items():
        arrays[f"element_references/{dataset}"] = refs

    config = {
        "model_id": "UMA-S-1.2",
        "cutoff": float(backbone.cutoff),
        "lmax": int(lmax),
        "mmax": int(mmax),
        "num_layers": int(backbone.num_layers),
        "sphere_channels": int(backbone.sphere_channels),
        "hidden_channels": int(backbone.hidden_channels),
        "edge_channels": int(backbone.edge_channels),
        "num_distance_basis": int(backbone.num_distance_basis),
        "max_num_elements": int(backbone.max_num_elements),
        "max_neighbors": int(backbone.max_neighbors),
        "num_experts": int(backbone.num_experts),
        "envelope_exponent": int(backbone.envelope.p),
        "edge_degree_rescale_factor": float(
            backbone.edge_degree_embedding.rescale_factor
        ),
        # `GaussianSmearing`'s basis is a non-persistent buffer built at torch's
        # *default* dtype, which fairchem sets to the inference precision — so a
        # checkpoint has no one true copy of it, only a rounding of
        # `linspace(0, cutoff, n)`. Record the recipe and let the port evaluate
        # it in whatever precision it runs at, matching fairchem's f64 path.
        "gaussian_basis_width": GAUSSIAN_BASIS_WIDTH,
        "norm_eps": float(backbone.norm.eps),
        "charge_channel_start": int(backbone.charge_channel_start),
        "charge_channel_end": int(backbone.charge_channel_end),
        "spin_channel_start": int(backbone.spin_channel_start),
        "spin_channel_end": int(backbone.spin_channel_end),
        "composition_include_self": bool(np.isclose(backbone.model_version, 1.0)),
        "backbone_datasets": dataset_names,
        "head_datasets": list(head.dataset_names),
        "head_dataset_to_expert": dict(head.dataset_name_to_exp),
        "normalizers": normalizers,
    }

    (HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {HERE / 'config.json'}")

    npz_path = HERE / "model.npz"
    np.savez(npz_path, **arrays)
    total = sum(a.size for a in arrays.values())
    print(f"wrote {npz_path} — {len(arrays)} arrays, {total:,} values")


if __name__ == "__main__":
    main()
