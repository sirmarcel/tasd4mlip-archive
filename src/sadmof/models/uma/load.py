"""Load a converted UMA checkpoint and mix its MOLE experts down for one structure.

UMA's SO(2) linears are *mixtures of linear experts*: 64 weight sets per layer,
combined by coefficients a routing MLP predicts from the system's composition,
charge, spin and dataset. The mixture depends only on those — never on
coordinates — so for a fixed structure it collapses to one ordinary weight per
layer, which is what `sadmof`'s Hessian path differentiates. `load_uma` does that
collapse in numpy at load time; `_merge_params` is the part that needs redoing if
the composition changes.

The mixing runs in float64 whatever `dtype` is asked for, matching fairchem's own
inference path (`predictor.model.double()` before the merge), and the result is
cast afterwards.

`sources/processed/uma-s-1p2/model.npz` is ~1.2 GB because it holds every expert;
`np.load` reads it lazily, so peak memory is a few hundred MB, not the file size.
"""

import numpy as np
from numpy.typing import ArrayLike
import jax
import jax.numpy as jnp

import json
from pathlib import Path

from ...utils import cast_floats
from .model import UMA

__all__ = ["load_uma"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_uma(
    checkpoint: str | Path,
    atomic_numbers: ArrayLike,
    *,
    task: str = "omat",
    charge: int = 0,
    spin: int = 0,
    dtype: str = "float64",
    balance_channels: bool = True,
) -> tuple[UMA, dict, dict]:
    """Build `(model, params, metadata)` for one composition.

    Args:
        checkpoint: Directory holding `config.json` + `model.npz`, i.e. the
            output of `sources/processed/uma-s-1p2/build.py`.
        atomic_numbers: The structure's Z (e.g. `atoms.numbers`). Only the
            multiset matters — the routing MLP sees the mean composition
            embedding, not the geometry — so any structure with the same
            formula reuses the same `params`.
        task: Which of UMA's training datasets to predict for. The two
            MOF-relevant ones are `omat` (PBE, pairs with a separate D3 term as
            MACE-MP-0 does) and `odac` (PBE+D3); `config.json` lists the rest.
            The choice is not a swap of the last layer — the dataset embedding
            feeds the MOLE routing MLP, so the whole mixed backbone changes,
            along with the head expert, the energy normaliser and the element
            references. Both are covered by `tests/test_uma.py`.
        charge, spin: Total system charge and spin multiplicity index. `charge`
            is also the target `balance_channels` drives the `l = 0` channel
            sums to, and rides along in `params` so it cannot desync from the
            mixture it was merged with.
        dtype: `"float64"` (also enables JAX x64, as `load_pet` does) or
            `"float32"` for the returned parameters.
        balance_channels: Passed to `UMA`; see its docstring.

    Returns:
        `(model, params, metadata)`. `metadata` carries `element_references`
        and the energy `normalizer` (per-system constants the energy function
        deliberately omits, since they add nothing to any derivative), and the
        mixing `expert_coefficients` for inspection.
    """
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text())

    numbers = np.asarray(atomic_numbers, dtype=int)
    if task not in config["head_dataset_to_expert"]:
        raise KeyError(
            f"unknown task {task!r}; known: {sorted(config['head_dataset_to_expert'])}"
        )

    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)

    with np.load(checkpoint / "model.npz") as raw:
        params, coefficients = _merge_params(
            raw, config, numbers, task=task, charge=charge, spin=spin
        )
        jd = tuple(np.asarray(raw[f"Jd_{l}"]) for l in range(config["lmax"] + 1))
        element_references = np.asarray(raw[f"element_references/{task}"])
        stored_perm = np.asarray(raw["to_m_perm"])

    float_dtype = jnp.float64 if dtype == "float64" else jnp.float32
    params = cast_floats(params, float_dtype)

    model = UMA(
        cutoff=config["cutoff"],
        max_neighbors=config["max_neighbors"],
        lmax=config["lmax"],
        mmax=config["mmax"],
        num_layers=config["num_layers"],
        sphere_channels=config["sphere_channels"],
        hidden_channels=config["hidden_channels"],
        num_distance_basis=config["num_distance_basis"],
        envelope_exponent=config["envelope_exponent"],
        edge_degree_rescale_factor=config["edge_degree_rescale_factor"],
        norm_eps=config["norm_eps"],
        gaussian_coeff=_gaussian_coeff(config),
        charge_channel_start=config["charge_channel_start"],
        charge_channel_end=config["charge_channel_end"],
        balance_channels=balance_channels,
        energy_scale=config["normalizers"][task]["rmsd"],
        jd=tuple(jnp.asarray(j, dtype=float_dtype) for j in jd),
    )
    if not np.array_equal(model.to_m_perm, stored_perm):
        raise RuntimeError("derived to_m permutation disagrees with the checkpoint")

    metadata = {
        "task": task,
        "charge": charge,
        "spin": spin,
        "element_references": element_references,
        "normalizer": config["normalizers"][task],
        "expert_coefficients": coefficients,
        "config": config,
    }
    return model, params, metadata


def _merge_params(
    raw, config: dict, atomic_numbers: np.ndarray, *, task: str, charge: int, spin: int
) -> tuple[dict, np.ndarray]:
    """Mix the MOLE experts down and reshape the flat archive into a param tree.

    Args:
        raw: An open `np.load` archive (or any `key -> array` mapping).
        config: The parsed `config.json`.
        atomic_numbers: Z of every atom in the structure.
        task: The dataset whose embedding feeds the routing MLP, and whose head
            expert is selected.
        charge, spin: Total system charge and spin multiplicity index; both feed
            the `csd` embedding, and `charge` is emitted as `params["charge"]`,
            the target `UMA._balance` drives the `l = 0` channel sums to.

    Returns:
        `(params, expert_coefficients)`, both float64.
    """
    if config["lmax"] != config["mmax"]:
        raise NotImplementedError(
            "this port assumes lmax == mmax; a truncated mmax needs fairchem's "
            "coefficient_index selection in prepare_wigner"
        )
    if config["spin_channel_end"] > config["spin_channel_start"]:
        raise NotImplementedError(
            "this checkpoint balances spin channels; the port implements the "
            "charge channels only (fairchem's balance_channels, target_offset=1.0)"
        )
    get = lambda key: np.asarray(raw[key], dtype=np.float64)

    csd = _csd_embedding(get, task=task, charge=charge, spin=spin)
    coefficients = _expert_coefficients(get, config, atomic_numbers, csd)
    mix = lambda key: np.einsum("eoi,e->oi", get(key), coefficients)

    mmax = config["mmax"]
    head_expert = config["head_dataset_to_expert"][task]

    blocks = []
    for i in range(config["num_layers"]):
        block = {
            "norm_1": _named(get, f"blocks/{i}/norm_1", ("w", "b")),
            "norm_2": _named(get, f"blocks/{i}/norm_2", ("w", "b")),
            "atom_wise": {
                "scalar_mlp": _named(get, f"blocks/{i}/atom_wise/scalar_mlp", ("w", "b")),
                "so3_linear_1": _named(
                    get, f"blocks/{i}/atom_wise/so3_linear_1", ("w", "b")
                ),
                "so3_linear_2": _named(
                    get, f"blocks/{i}/atom_wise/so3_linear_2", ("w", "b")
                ),
            },
        }
        for conv in (1, 2):
            prefix = f"blocks/{i}/so2_conv_{conv}"
            entry = {
                "fc_m0": {"w": mix(f"{prefix}/fc_m0/w"), "b": get(f"{prefix}/fc_m0/b")}
            }
            for m in range(1, mmax + 1):
                entry[f"m{m}"] = {"w": mix(f"{prefix}/m{m}/w")}
            if conv == 1:
                entry["rad"] = _radial_mlp(get, f"{prefix}/rad")
            block[f"so2_conv_{conv}"] = entry
        blocks.append(block)

    params = {
        "gaussian_offset": np.linspace(
            0.0, config["cutoff"], config["num_distance_basis"], dtype=np.float64
        ),
        "sphere_embedding": get("sphere_embedding"),
        "source_embedding": get("source_embedding"),
        "target_embedding": get("target_embedding"),
        "csd": csd,
        "charge": np.asarray(float(charge)),
        "edge_degree": {"rad": _radial_mlp(get, "edge_degree/rad")},
        "blocks": blocks,
        "norm": _named(get, "norm", ("w", "b")),
        "head": {
            f"lin{idx}": {
                "w": get(f"head/lin{idx}/w")[head_expert],
                "b": get(f"head/lin{idx}/b"),
            }
            for idx in (0, 2, 4)
        },
    }
    return params, coefficients


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _gaussian_coeff(config: dict) -> float:
    """`GaussianSmearing`'s exponent prefactor, from the basis spacing.

    fairchem derives this from the offset buffer it just built, so its value
    depends on the dtype the buffer was created at — see `build.py`. Evaluated
    here in float64, which is what fairchem's own float64 path produces.
    """
    offset = np.linspace(0.0, config["cutoff"], config["num_distance_basis"])
    spacing = config["gaussian_basis_width"] * (offset[1] - offset[0])
    return float(-0.5 / spacing**2)


def _csd_embedding(get, *, task: str, charge: int, spin: int) -> np.ndarray:
    """The charge/spin/dataset vector added to every node's `l = 0` features.

    fairchem's `ChgSpinEmbedding` tables are index-offset lookups: charge runs
    -100..100 (offset by half the table) and spin 0..100. Out of range, a bare
    index would wrap onto some other system's embedding.
    """
    charge_table = get("charge_embedding")
    spin_table = get("spin_embedding")
    charge_offset = charge_table.shape[0] // 2
    if not -charge_offset <= charge <= charge_offset:
        raise ValueError(f"charge {charge} outside [-{charge_offset}, {charge_offset}]")
    if not 0 <= spin < spin_table.shape[0]:
        raise ValueError(f"spin {spin} outside [0, {spin_table.shape[0] - 1}]")

    parts = [
        charge_table[charge + charge_offset],
        spin_table[spin],
        get(f"dataset_embedding/{task}")[0],
    ]
    stacked = np.concatenate(parts)
    return _silu(get("mix_csd/w") @ stacked + get("mix_csd/b"))


def _expert_coefficients(get, config: dict, atomic_numbers, csd) -> np.ndarray:
    """Softmax mixing weights over the 64 experts.

    The composition term is a *mean* over per-element embeddings taken with
    torch's `index_reduce_(..., include_self=True)`, so the zero-initialised
    accumulator counts as one more entry — the denominator is `N + 1`, not `N`.
    """
    composition_by_atom = get("composition_embedding")[np.asarray(atomic_numbers)]
    denominator = len(atomic_numbers) + (1 if config["composition_include_self"] else 0)
    composition = composition_by_atom.sum(axis=0) / denominator

    hidden = np.concatenate([composition, csd])
    for idx in (0, 2, 4):
        hidden = _silu(
            get(f"routing_mlp/lin{idx}/w") @ hidden + get(f"routing_mlp/lin{idx}/b")
        )
    shifted = hidden - hidden.max()
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum() + 0.005


# ---------------------------------------------------------------------------
# Archive reshaping
# ---------------------------------------------------------------------------


def _named(get, prefix: str, keys) -> dict:
    return {key: get(f"{prefix}/{key}") for key in keys}


def _radial_mlp(get, prefix: str) -> dict:
    params = {f"lin{i}": _named(get, f"{prefix}/lin{i}", ("w", "b")) for i in (0, 3, 6)}
    params.update({f"ln{i}": _named(get, f"{prefix}/ln{i}", ("w", "b")) for i in (1, 4)})
    return params
