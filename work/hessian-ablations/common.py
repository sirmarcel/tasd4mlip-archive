"""Shared construction for the ablation workers: roster, geometries, models, grid.

Every condition is a dense unit-cell Hessian; the ablation axes are precision
(fp64/fp32), the D3 correction (MACE, via a host-side combine), and shadow
forces (PET). Engine settings are pinned as constants, as in `work/hessians`:
changing one is a code change, visible in git and in every record's provenance.

Copied and trimmed from `work/hessians/common.py` (no supercell or hop
machinery — unit cells, dense only), with the D3 branch from an
earlier exploratory version.
"""

import numpy as np

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

GOENN_RELAXED = ROOT / "work" / "relax-goennheimer" / "output"
RANKING = ROOT / "work" / "select-goennheimer" / "output" / "ranking.csv"

# The production MACE adjacency cutoff (models.mace.atoms_to_inputs, vesin).
MACE_CUTOFF = 6.0
CHECKPOINTS = {"mace": "mace-mp-0-medium", "pet-xs": "pet-mad-xs", "pet-s": "pet-mad-s"}

# Per-model relaxation label. Each model reads its own relaxation, so the
# Hessian is taken at the minimum of the PES being probed; D3 rides on the
# MACE+D3 PES, so it shares MACE's geometry.
RELAX_TAGS = {
    "mace": "mace-mp0+d3_bfgs_float64",
    "d3": "mace-mp0+d3_bfgs_float64",
    "pet-xs": "pet_bfgs_float64",
    "pet-s": "pet-s_bfgs_float64",
}
MODELS = tuple(RELAX_TAGS)

# Production settings, as pinned in `work/hessians` — except precision, which
# is this experiment's first axis and therefore a flag. PET inputs use the
# tight padding; D3's input builder pads with its own fixed "multiples"
# strategy (no knob), recorded as such per record. AD_MODE is declarative:
# `sadmof.dense.get_dense_hessian_fn` is fwd-over-rev by construction and takes
# no mode argument — the constant only feeds the record.
AD_MODE = "fwd_over_rev"
CHUNK_SIZE = 1
REMAT = True
PADDING = "multiples_of_4"
EXTRA_NEIGHBORS = 0

# Gönnheimer's grid, as used by work/cv-ref-goennheimer.
TEMPERATURES = (250.0, 300.0, 350.0, 400.0)

# The GPU grid: (model, dtype, shadow, matmul), one flipped axis per step;
# shadow is None where the model has no such term. D3 is fp64 only — it is a
# fixed additive correction, computed once per structure and combined host-side
# (`combine_d3.py`) into the `mace_*+d3` siblings. The precision chain has two
# rungs: "highest" pins full-fp32 matmuls (`*_mmhigh`), "default" permits TF32
# on Hopper — the production setting, whose error the mmhigh rung isolates
# (TF32 dominates the apparent fp32 error; see README).
CONDITIONS = (
    ("mace", "float64", None, "default"),
    ("mace", "float32", None, "highest"),
    ("mace", "float32", None, "default"),
    ("d3", "float64", None, "default"),
    ("pet-xs", "float64", True, "default"),
    ("pet-xs", "float64", False, "default"),
    ("pet-xs", "float32", False, "highest"),
    ("pet-xs", "float32", False, "default"),
    ("pet-s", "float64", True, "default"),
    ("pet-s", "float64", False, "default"),
    ("pet-s", "float32", False, "highest"),
    ("pet-s", "float32", False, "default"),
)
COMBINE_VARIANTS = ("fp64", "fp32_mmhigh", "fp32")

# Reference corner per model family: the grm2025-faithful configuration. The
# same corners anchor each family's chain in `ablation_extract.PAIRS` — keep
# them in sync.
REFERENCE_VARIANTS = {
    "mace": "mace_fp64+d3",
    "pet-xs": "pet-xs_fp64_shadow",
    "pet-s": "pet-s_fp64_shadow",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def roster(spec):
    """Ordered structure identifiers for a roster spec.

    Args:
        spec: `goenn:<k>` (first k of the committed ranking) or an explicit
            identifier, composed.

    Returns:
        List of identifiers, deduplicated, in spec order.
    """
    out = []
    for item in spec:
        if item.startswith("goenn:"):
            k = int(item.split(":", 1)[1])
            with RANKING.open() as f:
                out += [row["structure"] for row in csv.DictReader(f)][:k]
        else:
            out.append(item)
    return list(dict.fromkeys(out))


def variant(model, dtype, shadow, matmul="default"):
    """Canonical condition name from the axes, without the structure prefix."""
    if model == "d3":
        return "d3"
    fp = {"float64": "fp64", "float32": "fp32"}[dtype]
    mm = "_mmhigh" if matmul == "highest" else ""
    if model.startswith("pet"):
        return f"{model}_{fp}_{'shadow' if shadow else 'noshadow'}{mm}"
    return f"{model}_{fp}{mm}"


def variants():
    """Every condition name, GPU grid plus combines, in grid order."""
    out = [variant(*c) for c in CONDITIONS]
    idx = out.index("d3")
    combined = [f"mace_{fp}+d3" for fp in COMBINE_VARIANTS]
    return out[:idx] + combined + out[idx:]


def load_relaxed(structure, model_name):
    """Relaxed unit cell for `(structure, model)`, wrapped.

    Raises:
        FileNotFoundError: if no relaxation exists for this pair — an
            unrelaxed model is a missing upstream run, never a silent
            fallback to another model's geometry.
    """
    from ase.io import read

    path = GOENN_RELAXED / RELAX_TAGS[model_name] / structure / "relaxed.xyz"
    if not path.exists():
        raise FileNotFoundError(f"no {model_name} relaxation for {structure!r}: {path}")
    atoms = read(str(path))
    assert atoms.info["identifier"] == structure
    atoms.wrap()
    return atoms


def load_model(model_name, float_dtype, shadow=False):
    """Checkpoint -> (model, params, energy_fn).

    `shadow` keeps PET's adaptive-cutoff force contributions in the gradient
    path (the grm2025-faithful reference); dense Hessians tolerate the
    densified coupling that `work/hessians` had to exclude.
    """
    from sadmof.paths import checkpoint

    if model_name == "mace":
        from marathon.io import from_dict, read_msgpack, read_yaml

        from sadmof.utils import cast_floats

        ckpt = checkpoint(CHECKPOINTS["mace"])
        model = from_dict(read_yaml(str(ckpt / "model.yaml")))
        params = cast_floats(read_msgpack(str(ckpt / "model.msgpack")), float_dtype)
        return model, params, model.energy

    if model_name == "d3":
        from sadmof.models.d3 import D3

        d3 = D3()
        return d3, D3.load_params(), d3.energy

    from sadmof.models.pet import get_energy_fn, load_pet

    dtype = "float64" if np.dtype(float_dtype) == np.float64 else "float32"
    model, params, _ = load_pet(checkpoint(CHECKPOINTS[model_name]), dtype=dtype)
    return model, params, get_energy_fn(model, no_shadow=not shadow)


def build_inputs(model_name, model, atoms, float_dtype):
    """Padded model inputs `(pos, cell, graph)` for one unit cell."""
    if model_name == "mace":
        from sadmof.models.mace import atoms_to_inputs

        return atoms_to_inputs(
            atoms,
            cutoff=MACE_CUTOFF,
            float_dtype=float_dtype,
            bucket_strategy=PADDING,
        )

    if model_name == "d3":
        from sadmof.models.d3 import atoms_to_inputs

        return atoms_to_inputs(
            atoms, cutoff=model.cutoff, cnthr=model.cnthr, float_dtype=float_dtype
        )

    from sadmof.models.pet import atoms_to_inputs

    return atoms_to_inputs(
        atoms,
        model,
        float_dtype=float_dtype,
        bucket_strategy=PADDING,
        extra_neighbors=EXTRA_NEIGHBORS,
    )
