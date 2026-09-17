"""Pin the Hessian hop count `K` against the production input pipelines:
`K = 2L` for MACE (node readout), `K = 2L + 1` for PET (edge readout).
Derived in the preprint, measured in `work/hops`.

The fixture is the whole difficulty: on a compact cell (like the cubic Si8
used elsewhere, fully connected at *one* hop) the pattern saturates to
all-ones and any hop count "matches". An elongated cell with a
nearest-neighbour-only list buys a graph diameter of 8 from 32 atoms —
changing the graph, not the architecture, whose property `K` is.

The check is exact, not a tolerance: beyond `K` hops the mixed partial is
absent from the autodiff graph, so fp64 returns bit-exact `0.0`; a coupling at
exactly `K` hops proves `K` cannot be reduced. Cost is a few HVPs (3 per
probed row block `H[a]`), not a dense Hessian.
"""

import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase.build import bulk
from scipy import sparse as sp
from scipy.sparse.csgraph import shortest_path

from sadmof.paths import checkpoint
from sadmof.sparse.pattern import _sparsity_pattern

# Structural zeros are only bit-exact in fp64; the assertions below rely on it.
jax.config.update("jax_enable_x64", True)

MACE_CKPT = checkpoint("mace-mp-0-medium")
PET_CKPT = checkpoint("pet-mad-xs")

NL_CUTOFF = 2.6  # nearest neighbours only, well inside either model's cutoff
PET_NEIGHBORS = 4  # PET's adaptive budget, matched to the same graph
PROBES = (0, 5)


@pytest.fixture(scope="module")
def atoms():
    """Elongated Si supercell: 32 atoms, graph diameter 8."""
    a = bulk("Si", "diamond", a=5.43, cubic=True) * (4, 1, 1)
    a.rattle(0.05, seed=1)  # no symmetry that could zero a coupling by accident
    return a


# --- the check ------------------------------------------------------------


def assert_hop_count(energy, pos, centers, others, n, hops):
    """Assert `hops` is exactly the Hessian's reach on this graph.

    Args:
        energy: Scalar energy as a function of positions.
        pos: Positions, padded shape `(N, 3)`.
        centers, others: The pair list `sparse.sparsity_pattern` would consume.
        n: Number of real atoms.
        hops: The predicted `K`.
    """
    adjacency = _adjacency(centers, others, n)
    distances = shortest_path(
        adjacency, method="D", unweighted=True, directed=False, indices=PROBES
    )

    assert distances.max() > hops, (
        f"fixture saturated: the probes reach only {distances.max():.0f} hops, so a "
        f"pattern at {hops} hops cannot be distinguished from any larger one"
    )
    assert _pattern_fill(centers, others, pos.shape[0], n, hops) < 1.0, (
        "the K-hop pattern is all-ones on this fixture; it would match any Hessian"
    )

    blocks = _row_block_norms(energy, pos, PROBES)[:, :n]

    beyond = blocks[distances > hops]
    assert (beyond == 0.0).all(), (
        f"coupling beyond {hops} hops (max {beyond.max():.3e}): the pattern is "
        "missing nonzeros and the sparse Hessian would be wrong"
    )
    # A coupling at exactly K hops lies outside the (K-1)-hop pattern, so this
    # is the statement that K cannot be reduced.
    assert (blocks[distances == hops] > 0.0).any(), (
        f"no coupling at {hops} hops; a smaller pattern would suffice"
    )


# --- models ---------------------------------------------------------------


def test_mace_hop_count(atoms):
    """MACE reads energy off node features: `K = 2L`."""
    from marathon.io import from_dict, read_msgpack, read_yaml

    from sadmof.models.mace import atoms_to_inputs

    model = from_dict(read_yaml(str(MACE_CKPT / "model.yaml")))
    params = read_msgpack(str(MACE_CKPT / "model.msgpack"))
    pos, cell, graph = atoms_to_inputs(atoms, cutoff=NL_CUTOFF, float_dtype=np.float64)

    assert_hop_count(
        lambda p: model.energy(params, p, cell, graph)[0],
        pos,
        np.asarray(graph["centers"]),
        np.asarray(graph["others"]),
        len(atoms),
        hops=2 * model.num_interactions,
    )


@pytest.mark.skipif(
    not (PET_CKPT / "model.msgpack").exists(),
    reason="pet-mad-xs checkpoint missing (see sources/processed/pet-mad-xs/README.md)",
)
def test_pet_hop_count(atoms):
    """PET reads energy off edge messages, one hop wider: `K = 2L + 1`.

    Runs the no-shadow path, which is the one the sparse Hessian uses: the
    adaptive cutoff is stop-gradiented, so coupling stays inside the *selected*
    adjacency rather than the raw neighbour ball. The neighbour budget is
    lowered to match the fixture's nearest-neighbour graph -- at the trained
    budget PET would select 16 neighbours here and saturate.
    """
    from petjax import UPET

    from sadmof.models.pet import atoms_to_inputs, get_energy_fn, load_pet

    _, params, metadata = load_pet(str(PET_CKPT), dtype="float64")
    metadata["config"]["num_neighbors_adaptive"] = PET_NEIGHBORS
    model = UPET(**metadata["config"])

    pos, cell, graph = atoms_to_inputs(atoms, model, float_dtype=np.float64)
    energy_fn = get_energy_fn(model, no_shadow=True)
    assert not bool(energy_fn(params, pos, cell, graph)[1]), "k_sel overflow"

    assert_hop_count(
        lambda p: energy_fn(params, p, cell, graph)[0],
        pos,
        np.asarray(graph["sel_centers"]),
        np.asarray(graph["sel_others"]),
        len(atoms),
        hops=2 * model.num_gnn_layers + 1,
    )


# --- helpers --------------------------------------------------------------


def _adjacency(centers, others, n):
    """Real-atom adjacency; padding pairs point at a dummy slot and drop out."""
    real = (centers < n) & (others < n)
    a = sp.csr_matrix(
        (np.ones(int(real.sum()), bool), (centers[real], others[real])), shape=(n, n)
    )
    a.setdiag(False)
    a.eliminate_zeros()
    assert (a != a.T).nnz == 0, "adjacency is not symmetric; (I + A)^k would under-cover"
    return a


def _pattern_fill(centers, others, n_padded, n_real, hops):
    """Real-atom fill of the production `(I + A)^hops` pattern.

    Built at the padded size, exactly as production does, then restricted to
    real atoms so padding slots (which carry only their forced diagonal) do not
    dilute the number the saturation check reads.
    """
    pattern = _sparsity_pattern(centers, others, n_padded, hops)
    mask = np.zeros((n_padded, n_padded), dtype=bool)
    mask[np.asarray(pattern.rows) // 3, np.asarray(pattern.cols) // 3] = True
    return mask[:n_real, :n_real].mean()


def _row_block_norms(energy, pos, probes):
    """`||H[a, :, b, :]||_F` for each probe `a`, via 3 HVPs per probe."""
    _, hvp = jax.linearize(jax.grad(energy), pos)
    rows = np.stack(
        [np.stack([np.asarray(hvp(_seed(pos, a, k))) for k in range(3)]) for a in probes]
    )  # (probes, 3, N, 3)
    return np.linalg.norm(rows, axis=(1, 3))


def _seed(pos, atom, cart):
    return jnp.zeros(pos.shape, dtype=pos.dtype).at[atom, cart].set(1.0)
