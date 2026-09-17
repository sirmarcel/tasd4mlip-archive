"""Shared construction for the Hessian workers: rosters, geometries, supercells, models.

Each model reads its own relaxation, so the Hessian is taken at the minimum of
the PES being probed and the converged supercell is recomputed at run time on
that geometry — relaxation shifts multipliers. Padding is pinned; see `PADDING`.
"""

import numpy as np

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "output"

GOENN_RELAXED = ROOT / "work" / "relax-goennheimer" / "output"
BIGMOF_RELAXED = ROOT / "work" / "relax-big-mofs" / "output"
RANKING = ROOT / "work" / "select-goennheimer" / "output" / "ranking.csv"

# The production MACE adjacency (models.mace.atoms_to_inputs, vesin) and the
# measured exact interaction depths (work/hops).
MACE_CUTOFF = 6.0
EXACT_HOPS = {"mace": 4, "pet-xs": 5, "pet-s": 7}
CHECKPOINTS = {"mace": "mace-mp-0-medium", "pet-xs": "pet-mad-xs", "pet-s": "pet-mad-s"}

# Per-model relaxation label, per upstream experiment.
RELAX_TAGS = {
    GOENN_RELAXED: {
        "mace": "mace-mp0+d3_bfgs_float64",
        "pet-xs": "pet_bfgs_float64",
        "pet-s": "pet-s_bfgs_float64",
    },
    BIGMOF_RELAXED: {
        "mace": "mace-mp0+d3_lbfgs-ls_float64",
        "pet-xs": "pet_lbfgs-ls_float64",
        "pet-s": "pet-s_lbfgs-ls_float64",
    },
}

# Production settings. The engine values are the lock certified by
# `work/determine-perf-settings`; padding and `k_sel` slack are tighter than
# anything it probed. Constants, not flags: changing one is a code change, so
# it shows up in git and in every record's provenance.
AD_MODE = "fwd_over_rev"
CHUNK_SIZE = 1
REMAT = True
PRECISION = "float32"
# IEEE fp32 matmuls; records without the key predate the pin (TF32 default).
MATMUL_PRECISION = "highest"
PADDING = "multiples_of_4"
# PET `k_sel` slack. Zero: `atoms_to_inputs` sizes the budget with the same
# selection the forward pass runs, so `k_sel >= k_sel_actual` by construction;
# the overflow flag is recorded regardless.
EXTRA_NEIGHBORS = 0

# Gönnheimer's grid, as used by work/cv-ref-goennheimer.
TEMPERATURES = (250.0, 300.0, 350.0, 400.0)

# Truncation axis; each model's exact K is added per model. Runs to 4 because
# PET-XS needs 4 of its 5 hops; MACE's K is 4 already.
LADDER = (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def roster(spec):
    """Ordered structure identifiers for a roster spec.

    Args:
        spec: `goenn:<k>` (first k of the committed ranking), `giants`, or an
            explicit identifier.

    Returns:
        List of identifiers, deduplicated, in spec order.
    """
    out = []
    for item in spec:
        if item.startswith("goenn:"):
            k = int(item.split(":", 1)[1])
            with RANKING.open() as f:
                out += [row["structure"] for row in csv.DictReader(f)][:k]
        elif item == "giants":
            out += giants()
        else:
            out.append(item)
    return list(dict.fromkeys(out))


def giants():
    """Big-MOF identifiers relaxed for any model.

    Membership is identity, not per-model availability; `load_relaxed` raises
    for a model that has none.
    """
    found = set()
    for tag in RELAX_TAGS[BIGMOF_RELAXED].values():
        found |= {p.parent.name for p in (BIGMOF_RELAXED / tag).glob("*/relaxed.xyz")}
    return sorted(found)


def hop_ladder(model, hops=None):
    """Sparse hop counts for a model: the truncation axis plus its exact `K`."""
    if hops:
        return sorted(set(hops))
    return sorted({*LADDER, EXACT_HOPS[model]})


def merge_reused_coloring(record, condition_dir):
    """Price a coloring from the sidecar that measured it.

    `record.lift.json` (`recolor.py`, atom-level coloring lifted, checked against
    the cached `coloring.npz`) wins over `record.tf32.json` (`prep_rerun.py`, the
    TF32-era run whose coordinate-level coloring a cached load reused). The
    coordinate-route times survive as `pattern_coord_s` / `coloring_coord_s`.
    Mutates `record` only on success.

    Returns:
        `"merged"`, `"unneeded"` (fresh, no sidecar), `"missing"` (cached, no
        sidecar), `"invalid"` (TF32 sidecar itself cached), `"mismatch"` (lift
        sidecar's check failed).
    """
    from sadmof.io import read_json

    sparsity = record.get("sparsity") or {}
    timings = record.get("timings") or {}
    cached = bool(sparsity.get("coloring_cached"))
    condition_dir = Path(condition_dir)

    lift = None
    lift_path = condition_dir / "record.lift.json"
    if lift_path.exists():
        lift = read_json(lift_path)
        check = lift.get("check") or {}
        if not (check.get("colors_identical") and check.get("pattern_identical")):
            return "mismatch"

    coord_pattern_s, coord_coloring_s = timings.get("pattern_s"), timings.get("coloring_s")
    tf32 = None
    if cached:
        tf32_path = condition_dir / "record.tf32.json"
        if tf32_path.exists():
            tf32 = read_json(tf32_path)
            coord_coloring_s = (tf32.get("timings") or {}).get("coloring_s")
            if (tf32.get("sparsity") or {}).get(
                "coloring_cached"
            ) or coord_coloring_s is None:
                return "invalid"
            coord_pattern_s = (tf32.get("timings") or {}).get("pattern_s", coord_pattern_s)
        elif lift is None:
            return "missing"
        else:
            coord_pattern_s = coord_coloring_s = None

    if lift is None and not cached:
        return "unneeded"

    record["sparsity"] = sparsity
    record["timings"] = timings
    sparsity["coloring_reused"] = True
    if lift is None:
        timings["coloring_s"] = coord_coloring_s
        sparsity["coloring_route"] = "coord"
        return "merged"
    timings["pattern_coord_s"] = coord_pattern_s
    timings["coloring_coord_s"] = coord_coloring_s
    for key in ("pattern_s", "atom_coloring_s", "lift_s", "coloring_s"):
        timings[key] = lift["timings"][key]
    sparsity["coloring_route"] = "lifted"
    return "merged"


def load_relaxed(structure, model_name):
    """Relaxed geometry for `(structure, model)`, wrapped (reach needs wrapped
    positions).

    Raises:
        FileNotFoundError: if no relaxation exists for this pair, naming the
            paths tried — an unrelaxed model is a missing upstream run, never a
            silent fallback to another model's geometry.
    """
    from ase.io import read

    tried = []
    for tree, tags in RELAX_TAGS.items():
        path = tree / tags[model_name] / structure / "relaxed.xyz"
        tried.append(path)
        if path.exists():
            atoms = read(str(path))
            assert atoms.info["identifier"] == structure
            atoms.wrap()
            return atoms
    raise FileNotFoundError(
        f"no {model_name} relaxation for {structure!r}; tried: "
        + ", ".join(str(p) for p in tried)
    )


def reach_for(atoms, model_name, model=None):
    """Offset-resolved reach over `h = 1..K` on the model's own adjacency.

    `model` is needed for PET's adaptive selection only; MACE goes off the
    cutoff. The single place the adjacency is chosen, so the workers and the
    offline analyses cannot drift apart on it.
    """
    from sadmof import supercell as sc

    if model_name == "mace":
        edges = sc.mace_edges(atoms, MACE_CUTOFF)
    else:
        edges = sc.pet_edges(atoms, model)
    return sc.reach(atoms, edges, EXACT_HOPS[model_name])


def converged_multiplier(atoms, model_name, model=None):
    """Minimum-volume alias-free multiplier at the model's exact hop count.

    Returns `(1, 1, 1)` where the cell already contains the reach, so it is
    always correct and `unit` is an override rather than a default. Not the
    `2*L*r_c` width heuristic, which is a lower bound and can under-call.
    """
    from sadmof import supercell as sc

    r = reach_for(atoms, model_name, model)[-1]
    return sc.exact_min_cell(sc.differences(r, len(atoms)))


def load_model(model_name, float_dtype):
    """Checkpoint -> (model, params, energy_fn); PET gets the no_shadow path.

    PET shadow gradients couple past the selected neighbour list, which densifies
    the pattern; the sparse path is no-shadow by construction and the dense path
    matches it so the two are comparable.
    """
    from sadmof.paths import checkpoint

    if model_name == "mace":
        from marathon.io import from_dict, read_msgpack, read_yaml

        from sadmof.utils import cast_floats

        ckpt = checkpoint(CHECKPOINTS["mace"])
        model = from_dict(read_yaml(str(ckpt / "model.yaml")))
        params = cast_floats(read_msgpack(str(ckpt / "model.msgpack")), float_dtype)
        return model, params, model.energy

    from sadmof.models.pet import get_energy_fn, load_pet

    dtype = "float64" if np.dtype(float_dtype) == np.float64 else "float32"
    model, params, _ = load_pet(checkpoint(CHECKPOINTS[model_name]), dtype=dtype)
    return model, params, get_energy_fn(model, no_shadow=True)


def build_inputs(model_name, model, atoms, float_dtype):
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
            bucket_strategy=PADDING,
        )
        return pos, cell, graph, graph

    from sadmof.models.pet import atoms_to_inputs

    pos, cell, graph = atoms_to_inputs(
        atoms,
        model,
        float_dtype=float_dtype,
        bucket_strategy=PADDING,
        extra_neighbors=EXTRA_NEIGHBORS,
    )
    pattern_graph = {
        "centers": graph["sel_centers"],
        "others": graph["sel_others"],
        "atomic_numbers": graph["atomic_numbers"],
    }
    return pos, cell, graph, pattern_graph
