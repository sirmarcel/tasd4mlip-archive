"""Measure the Hessian hop count `K` on graphs we control exactly.

`K = 2L` (node readout) and `K = 2L + 1` (edge readout) are derived in the
preprint. This script checks them, over an `L` sweep so the result is a law
rather than three coincidences.

Two passes. Freshly initialised models at each depth first: random weights make
every structurally-allowed coupling generically nonzero, so the measurement
isolates the architecture from the trained magnitudes. Then the production
checkpoints on the same graphs, which rule out accidental cancellation and
record the per-hop decay.

Both models are swept over the same depths. Note that the random pass has very
little dynamic range at the outer hops — a freshly initialised MACE couples at
~1e-42 of the diagonal at `L=2`, losing ~28 orders per added layer — so the
sweep is bounded by fp64 rather than by anything structural. Underflow would
show up as a hop-count mismatch, not as a wrong answer quietly accepted.

    JAX_PLATFORMS=cpu uv run python work/hops/law.py
"""

import numpy as np
import jax
import jax.numpy as jnp

import argparse
import json
from pathlib import Path

from scipy import sparse as sp
from scipy.sparse.csgraph import shortest_path

from sadmof.paths import checkpoint

jax.config.update("jax_enable_x64", True)

EXP = Path(__file__).resolve().parent  # output dir only; repo paths via sadmof.paths

MACE_CKPT = checkpoint("mace-mp-0-medium")
PET_CKPTS = ["pet-mad-xs", "pet-mad-s"]

N_ATOMS = 24
SEEDS = (1, 2, 3)
DEPTHS = (1, 2, 3, 4)  # message-passing layers, swept for both models
BOND_FRACTION = 0.45  # bond length as a fraction of the model cutoff


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # `nargs="+"`, not `"*"`: an empty list would measure nothing, pass both
    # assertions vacuously, exit 0, and overwrite the output with empty results.
    p.add_argument("--families", nargs="+", choices=sorted(FAMILIES), default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("-o", "--output", type=Path, default=EXP / "output" / "law.json")
    args = p.parse_args()
    families = args.families or sorted(FAMILIES)
    seeds = args.seeds or list(SEEDS)

    results = {
        "random_weights": random_weight_law(families, seeds),
        "checkpoints": checkpoint_law(families),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.output}")


def random_weight_law(families, seeds):
    """Measured vs predicted `K` over an `L` sweep, with random weights."""
    print("\n## Random weights — the architecture law")
    print(f"{'model':6} {'L':>2} {'family':12} {'diam':>4} {'K':>10} {'predicted':>10}")

    rows = []
    for family in families:
        for L in DEPTHS:
            rows.append(_measure_mace(family, L, seeds, weights="random"))
        for L in DEPTHS:
            rows.append(_measure_pet(family, L, seeds, weights="random"))

    for r in rows:
        _print_row(r)
    _assert_law(rows)
    return rows


def checkpoint_law(families):
    """The same measurement with trained weights, plus the per-hop decay.

    Trained couplings are small at the outer hops but strictly nonzero, then
    exactly zero past `K` — this is what rules out the measured boundary being
    an artefact of the random weights.
    """
    print("\n## Trained checkpoints — same graphs, real weights")
    print(
        f"{'model':12} {'L':>2} {'family':12} {'diam':>4} {'K':>4} {'pred':>5}  "
        f"decay ||H_d|| / ||H_0||"
    )

    rows = []
    for family in families:
        rows.append(_measure_mace(family, None, (1,), weights="checkpoint"))
        for name in PET_CKPTS:
            rows.append(_measure_pet(family, None, (1,), weights="checkpoint", name=name))

    for r in rows:
        decay = " ".join(f"{v:.0e}" for v in r["decay"])
        print(
            f"{r['model']:12} {r['L']:>2} {r['family']:12} {r['diameter']:>4} "
            f"{r['hops'][0]:>4} {r['predicted']:>5}  {decay}"
        )
    _assert_law(rows)
    return rows


# ---------------------------------------------------------------------------
# Graph families — (undirected edge list, positions)
# ---------------------------------------------------------------------------


def path(n: int, bond: float, seed: int):
    """Open chain `0-1-...-(n-1)`. Maximal graph distance per atom: the
    extremal probe for the hop count."""
    return [(k - 1, k) for k in range(1, n)], _wiggled_chain(n, bond, seed)


def ring(n: int, bond: float, seed: int):
    """Closed chain. A synthetic stand-in for periodicity: a walk can leave an
    atom in one direction and come back around."""
    edges = [(k, (k + 1) % n) for k in range(n)]
    rng = np.random.RandomState(seed)
    radius = bond / (2.0 * np.sin(np.pi / n))
    angles = 2.0 * np.pi * np.arange(n) / n
    pos = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(n)])
    pos += 0.05 * bond * rng.randn(n, 3)  # break the symmetry, keep bonds ~= bond
    return edges, jnp.asarray(pos)


def multishell(n: int, bond: float, seed: int):
    """Chain with 1st *and* 2nd neighbours bonded (degree 4). Same geometry as
    `path`, twice the node degree — the control that the answer is a property
    of hops, not of degree."""
    edges = [(k - 1, k) for k in range(1, n)] + [(k - 2, k) for k in range(2, n)]
    return edges, _wiggled_chain(n, bond, seed)


def caterpillar(n: int, bond: float, seed: int):
    """Chain spine with one pendant leaf per spine atom.

    The closest synthetic analogue of a framework: terminal H atoms are graph
    leaves, and the extremal pair in a real structure is typically leaf-to-leaf.
    Here the extremal pair sits two hops further apart than the spine diameter.
    """
    n_spine = n // 2
    spine = _wiggled_chain(n_spine, bond, seed)
    rng = np.random.RandomState(seed + 100)
    pos = np.zeros((2 * n_spine, 3))
    pos[:n_spine] = spine
    edges = [(k - 1, k) for k in range(1, n_spine)]
    axis = np.array([1.0, 0.0, 0.0])
    for k in range(n_spine):
        offset = rng.randn(3)
        offset -= (offset @ axis) * axis  # hang the leaf off the chain axis
        pos[n_spine + k] = spine[k] + bond * offset / np.linalg.norm(offset)
        edges.append((k, n_spine + k))
    return edges, jnp.asarray(pos)


FAMILIES = {
    "path": path,
    "ring": ring,
    "multishell": multishell,
    "caterpillar": caterpillar,
}


def _wiggled_chain(n: int, bond: float, seed: int, wiggle: float = 0.35):
    """Open chain with every bond exactly `bond` and no collinearity or
    symmetry that could zero a coupling by accident."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((n, 3))
    for k in range(1, n):
        step = np.array([1.0, 0.0, 0.0]) + wiggle * rng.randn(3)
        pos[k] = pos[k - 1] + bond * step / np.linalg.norm(step)
    return jnp.asarray(pos)


# ---------------------------------------------------------------------------
# Model harnesses — hand-built graphs, no neighbour-list machinery
# ---------------------------------------------------------------------------


def mace_graph(edges, n: int, Z: int = 6):
    """Symmetrised flat pair list. MACE aggregates with `segment_sum`, so no
    rectangular layout is needed."""
    centers = np.array([a for e in edges for a in e], dtype=np.int64)
    others = np.array([b for e in edges for b in reversed(e)], dtype=np.int64)
    return {
        "centers": jnp.asarray(centers),
        "others": jnp.asarray(others),
        "cell_shifts": jnp.zeros((len(centers), 3)),
        "atomic_numbers": jnp.full(n, Z, dtype=jnp.int64),
        "pair_mask": jnp.ones(len(centers), dtype=bool),
        "atom_mask": jnp.ones(n, dtype=bool),
    }


def pet_layout(edges, n: int, species_idx: int = 0):
    """Rectangular `[n, k_sel]` layout for `UPET.__call__`.

    `k_sel` is the maximum degree; unused slots are self-loop padding.
    `reverse[p]` is the flat index of the opposite directed edge, mirroring
    petjax's `_get_corresponding_edges` — the map the final message-passing
    step reads through, and the reason PET's hop count is odd.
    """
    neighbours = [[] for _ in range(n)]
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)
    neighbours = [sorted(set(x)) for x in neighbours]

    k = max(len(x) for x in neighbours)
    P = n * k
    centers = np.repeat(np.arange(n, dtype=np.int64), k)
    others = np.repeat(np.arange(n, dtype=np.int64), k)  # self-loop padding
    pair_mask = np.zeros(P, dtype=bool)
    slot = {}
    for i in range(n):
        for s, j in enumerate(neighbours[i]):
            p = i * k + s
            others[p] = j
            pair_mask[p] = True
            slot[(i, j)] = p

    reverse = np.arange(P, dtype=np.int64)
    for (i, j), p in slot.items():
        reverse[p] = slot[(j, i)]  # undirected edge list => always present

    return {
        "centers": jnp.asarray(centers),
        "neighbors": jnp.asarray(others),
        "reverse": jnp.asarray(reverse),
        "pair_mask": jnp.asarray(pair_mask),
        "atom_mask": jnp.ones(n, dtype=bool),
        "species": jnp.full(n, species_idx, dtype=jnp.int64),
    }


def mace_total_fn(model, params, graph):
    """Total energy as a function of positions."""

    def total(pos):
        return model.energy(params, pos, jnp.eye(3), graph)[0]

    return total


def pet_total_fn(model, params, layout):
    """Total energy as a function of positions.

    `pair_cutoffs=None` selects the static trained cutoff, bypassing the
    adaptive selection — which is the point: this script measures the
    architecture on a graph it controls exactly, and the production
    `get_energy_fn` would re-derive the graph by adaptive selection. That path
    is structurally equivalent (`no_shadow=True` stop-gradients the cutoff, so
    it is constant w.r.t. positions), and is exercised against real structures
    in `tests/test_hop_count.py` rather than assumed here.
    """
    centers, neighbors = layout["centers"], layout["neighbors"]

    def total(pos):
        R_ij = pos[neighbors] - pos[centers]
        per_atom = model.apply(
            params,
            R_ij,
            centers,
            neighbors,
            layout["species"],
            layout["reverse"],
            layout["pair_mask"],
            layout["atom_mask"],
            pair_cutoffs=None,
        )
        return jnp.sum(per_atom)

    return total


def mace_init(model, graph, pos, seed: int):
    """Freshly initialised MACE parameters for this graph."""
    return model.init(
        jax.random.PRNGKey(seed),
        pos[graph["others"]] - pos[graph["centers"]],
        graph["centers"],
        graph["others"],
        graph["atomic_numbers"],
        graph["pair_mask"],
        graph["atom_mask"],
    )


def pet_init(model, layout, pos, seed: int):
    """Freshly initialised PET parameters for this layout."""
    return model.init(
        jax.random.PRNGKey(seed),
        pos[layout["neighbors"]] - pos[layout["centers"]],
        layout["centers"],
        layout["neighbors"],
        layout["species"],
        layout["reverse"],
        layout["pair_mask"],
        layout["atom_mask"],
        pair_cutoffs=None,
    )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def graph_distances(edges, n: int) -> np.ndarray:
    """All-pairs hop distance on the undirected graph."""
    a = np.array([x[0] for x in edges])
    b = np.array([x[1] for x in edges])
    A = sp.csr_matrix((np.ones(len(a), bool), (a, b)), shape=(n, n))
    return shortest_path(A, method="D", unweighted=True, directed=False)


def hop_curve(total_fn, pos, distances):
    """`(measured K, max block norm per hop distance)`.

    `||H[a, :, b, :]||_F` binned by `d(a, b)`. Beyond the model's hop count the
    entries are structurally absent from the autodiff graph, so the bin is a
    bit-exact `0.0` and the boundary needs no tolerance.
    """
    H = np.asarray(jax.hessian(total_fn)(pos))  # (n, 3, n, 3)
    blocks = np.linalg.norm(H, axis=(1, 3))  # (n, n)
    d_max = int(distances[np.isfinite(distances)].max())
    curve = [float(blocks[distances == d].max(initial=0.0)) for d in range(d_max + 1)]
    hops = max((d for d, v in enumerate(curve) if v > 0.0), default=0)
    return hops, curve


def _measure_mace(family, L, seeds, *, weights):
    from marathon.io import from_dict, read_msgpack, read_yaml

    from sadmof.models.mace.model import MACE

    if weights == "checkpoint":
        model = from_dict(read_yaml(str(MACE_CKPT / "model.yaml")))
        trained = read_msgpack(str(MACE_CKPT / "model.msgpack"))
        L = model.num_interactions

        def build(edges, pos, seed):
            return mace_total_fn(model, trained, mace_graph(edges, len(pos)))
    else:
        # `MACE()` defaults are the MP-0 medium configuration, with
        # `use_agnesi` / `use_zbl` off — which is what that checkpoint uses, and
        # the only configuration `init` can build (the agnesi / ZBL constants
        # would initialise to zeros and NaN the forward pass).
        model = MACE(num_interactions=L)

        def build(edges, pos, seed):
            graph = mace_graph(edges, len(pos))
            return mace_total_fn(model, mace_init(model, graph, pos, seed), graph)

    return _measure(
        model_name="MACE-MP-0" if weights == "checkpoint" else "MACE",
        family=family,
        L=L,
        predicted=2 * L,
        cutoff=float(model.cutoff),
        seeds=seeds,
        build=build,
    )


def _measure_pet(family, L, seeds, *, weights, name=None):
    from petjax import UPET

    from sadmof.models.pet import load_pet

    if weights == "checkpoint":
        model, trained, metadata = load_pet(str(checkpoint(name)), dtype="float64")
        L = model.num_gnn_layers
        species_idx = metadata["species_to_index"][6]

        def build(edges, pos, seed):
            return pet_total_fn(model, trained, pet_layout(edges, len(pos), species_idx))
    else:
        # `UPET()` defaults are the PET-MAD-XS configuration apart from depth,
        # which is the sweep variable. Unlike MACE, no layout constraint caps it.
        model = UPET(num_gnn_layers=L)

        def build(edges, pos, seed):
            layout = pet_layout(edges, len(pos))
            return pet_total_fn(model, pet_init(model, layout, pos, seed), layout)

    return _measure(
        model_name=name or "PET",
        family=family,
        L=L,
        predicted=2 * L + 1,
        cutoff=float(model.cutoff),
        seeds=seeds,
        build=build,
    )


def _measure(*, model_name, family, L, predicted, cutoff, seeds, build):
    """Run one (model, family) over seeds and collect hop counts + decay."""
    bond = BOND_FRACTION * cutoff
    hops, curves, diameter = [], [], None
    for seed in seeds:
        edges, pos = FAMILIES[family](N_ATOMS, bond, seed)
        distances = graph_distances(edges, len(pos))
        diameter = int(distances[np.isfinite(distances)].max())
        if diameter <= predicted:
            raise RuntimeError(
                f"{model_name} L={L} on {family}: graph diameter {diameter} does not "
                f"exceed the predicted hop count {predicted} — the measurement would "
                f"saturate and cannot distinguish K from the graph's own diameter"
            )
        k, curve = hop_curve(build(edges, pos, seed), pos, distances)
        hops.append(k)
        curves.append(curve)

    curve = curves[0]
    return {
        "model": model_name,
        "L": L,
        "family": family,
        "diameter": diameter,
        "hops": hops,
        "predicted": predicted,
        "decay": [v / curve[0] for v in curve[: predicted + 2]],
    }


def _print_row(r):
    ok = "OK" if all(h == r["predicted"] for h in r["hops"]) else "MISMATCH"
    print(
        f"{r['model']:6} {r['L']:>2} {r['family']:12} {r['diameter']:>4} "
        f"{str(r['hops']):>10} {r['predicted']:>10} {ok}"
    )


def _assert_law(rows):
    """Fail loudly: the law is the deliverable, not a printed observation."""
    bad = [r for r in rows if any(h != r["predicted"] for h in r["hops"])]
    if bad:
        raise AssertionError(
            "measured hop count differs from the predicted law for: "
            + ", ".join(f"{r['model']} L={r['L']} on {r['family']}" for r in bad)
        )
    # Not a tolerance check: nothing is computed past K, so anything other than
    # a bit-exact zero means the predicted reach is wrong.
    leaked = [r for r in rows if r["decay"][r["predicted"] + 1] != 0.0]
    if leaked:
        raise AssertionError(
            "coupling past hop K, which should be structurally absent: "
            + ", ".join(f"{r['model']} on {r['family']}" for r in leaked)
        )


if __name__ == "__main__":
    main()
