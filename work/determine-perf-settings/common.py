"""Shared construction for the perf-settings workers: geometries, supercells, models.

Geometries are the committed relaxed structures (`work/relax-goennheimer`,
`work/relax-big-mofs`). The converged supercell is recomputed here on the relaxed
geometry at the model's exact hop count — the operative production cell, not the
raw-geometry number from `work/select-goennheimer` (relaxation shifts multipliers).
"""

import numpy as np

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "output"

GOENN_RELAXED = ROOT / "work" / "relax-goennheimer" / "output" / "mace-mp0+d3_bfgs_float64"
BIGMOF_RELAXED = (
    ROOT / "work" / "relax-big-mofs" / "output" / "mace-mp0+d3_lbfgs-ls_float64"
)

# The production MACE adjacency (models.mace.atoms_to_inputs, vesin) and the
# measured exact interaction depths (work/hops).
MACE_CUTOFF = 6.0

# PET `k_sel` slack, matching `work/hessians`. The library default (4) inflates
# the selected-pair tensor; k_sel is small, so slack and bucket both bite there.
PET_EXTRA_NEIGHBORS = 0

# `--padding` choices. `production` is what `work/hessians` runs; on MACE it is
# within 12 elements per axis of `tight`.
PADDING_STRATEGIES = {
    "bucketed": "multiples",
    "tight": "multiples_of_16",
    "production": "multiples_of_4",
}
EXACT_HOPS = {"mace": 4, "pet-xs": 5, "pet-s": 7}
CHECKPOINTS = {"mace": "mace-mp-0-medium", "pet-xs": "pet-mad-xs", "pet-s": "pet-mad-s"}


def load_relaxed(structure):
    """Relaxed geometry by identifier, wrapped (reach needs wrapped positions)."""
    from ase.io import read

    for tree in (GOENN_RELAXED, BIGMOF_RELAXED):
        path = tree / structure / "relaxed.xyz"
        if path.exists():
            atoms = read(str(path))
            assert atoms.info["identifier"] == structure
            atoms.wrap()
            return atoms
    raise FileNotFoundError(f"no relaxed geometry for {structure!r}")


def converged_multiplier(atoms, model_name, model=None):
    """Minimum-volume alias-free multiplier at the model's exact hop count."""
    from sadmof import supercell as sc

    if model_name == "mace":
        edges = sc.mace_edges(atoms, MACE_CUTOFF)
    else:
        edges = sc.pet_edges(atoms, model)
    r = sc.reach(atoms, edges, EXACT_HOPS[model_name])[-1]
    return sc.exact_min_cell(sc.differences(r, len(atoms)))


def load_model(model_name, float_dtype):
    """Checkpoint -> (model, params, energy_fn); PET gets the no_shadow sparse path."""
    from sadmof.paths import checkpoint

    if model_name == "mace":
        from marathon.io import from_dict, read_msgpack, read_yaml

        ckpt = checkpoint(CHECKPOINTS["mace"])
        model = from_dict(read_yaml(str(ckpt / "model.yaml")))
        params = read_msgpack(str(ckpt / "model.msgpack"))
        return model, params, model.energy

    from sadmof.models.pet import get_energy_fn, load_pet

    dtype = "float64" if np.dtype(float_dtype) == np.float64 else "float32"
    model, params, _ = load_pet(checkpoint(CHECKPOINTS[model_name]), dtype=dtype)
    return model, params, get_energy_fn(model, no_shadow=True)


def build_inputs(model_name, model, atoms, float_dtype, bucket_strategy="multiples"):
    """Model inputs plus the graph `sparse.sparsity_pattern` consumes.

    For PET the pattern lives on the selected adjacency (`sel_*` keys), matching
    the no_shadow energy path; MACE patterns use the raw pair list directly.
    """
    if model_name == "mace":
        from sadmof.models.mace import atoms_to_inputs

        pos, cell, graph = atoms_to_inputs(
            atoms,
            cutoff=MACE_CUTOFF,
            float_dtype=float_dtype,
            bucket_strategy=bucket_strategy,
        )
        return pos, cell, graph, graph

    from sadmof.models.pet import atoms_to_inputs

    pos, cell, graph = atoms_to_inputs(
        atoms,
        model,
        float_dtype=float_dtype,
        bucket_strategy=bucket_strategy,
        extra_neighbors=PET_EXTRA_NEIGHBORS,
    )
    pattern_graph = {
        "centers": graph["sel_centers"],
        "others": graph["sel_others"],
        "atomic_numbers": graph["atomic_numbers"],
    }
    return pos, cell, graph, pattern_graph
