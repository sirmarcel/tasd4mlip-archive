"""UMA integration against the uma-s-1p2 checkpoint.

Energies and forces are checked against `data/uma_reference.json`, pinned from
fairchem's own float64 inference path by `data/build_uma_reference.py` — torch
is not in the test environment. Both MOF-relevant heads are covered: `omat`
(PBE) and `odac` (PBE+D3). They are not a thin output layer apart — the dataset
embedding feeds the MOLE routing MLP, so the whole mixed backbone differs. The
remaining tests are the Hessian pipeline: symmetry through
`dense.get_dense_hessian_fn`, and sparse-equals-dense through `sadmof.sparse`.
"""

import numpy as np
import jax

import json
import subprocess
import sys
from pathlib import Path

import pytest
from ase.build import bulk
from ase.io import read

from typing import Any, NamedTuple

from sadmof.paths import checkpoint

jax.config.update("jax_enable_x64", True)

CKPT = checkpoint("uma-s-1p2")
REFERENCE_PATH = Path(__file__).parent / "data" / "uma_reference.json"

pytestmark = pytest.mark.skipif(
    not (CKPT / "model.npz").exists(),
    reason="uma-s-1p2 checkpoint missing (see sources/processed/uma-s-1p2/README.md)",
)


@pytest.fixture(scope="module")
def reference():
    return json.loads(REFERENCE_PATH.read_text())


def _atoms(name):
    if name == "si8":
        atoms = bulk("Si", "diamond", a=5.43, cubic=True)
        atoms.rattle(0.1, seed=1)
        return atoms
    if name == "si8_222":
        atoms = bulk("Si", "diamond", a=5.43, cubic=True) * [2, 2, 2]
        atoms.rattle(0.1, seed=1)
        return atoms
    return read(Path(__file__).parent / "data" / "relaxed" / f"{name}.xyz")


class Setup(NamedTuple):
    atoms: Any
    model: Any
    params: dict
    metadata: dict
    energy_fn: Any
    inputs: tuple


def _setup(name, reference, task="omat", **kwargs) -> Setup:
    from sadmof.models.uma import atoms_to_inputs, get_energy_fn, load_uma

    atoms = _atoms(name)
    model, params, metadata = load_uma(
        CKPT,
        atoms.numbers,
        task=task,
        charge=reference["charge"],
        spin=reference["spin"],
        dtype="float64",
        **kwargs,
    )
    return Setup(
        atoms, model, params, metadata, get_energy_fn(model), atoms_to_inputs(atoms, model)
    )


def _total_energy(setup: Setup):
    """`(total, energy, per_atom)` — the port's energy plus the two per-system
    constants it leaves to the caller, as `UMACalculator` adds them."""
    energy, per_atom = setup.energy_fn(setup.params, *setup.inputs)
    references = setup.metadata["element_references"][setup.atoms.numbers].sum()
    shift = setup.metadata["normalizer"]["mean"]
    return float(energy) + float(references) + float(shift), energy, per_atom


@pytest.mark.parametrize("task", ["omat", "odac"])
@pytest.mark.parametrize("name", ["si8", "RSM0010"])
def test_energy_and_forces_match_torch(name, task, reference):
    expected = reference["tasks"][task][name]
    setup = _setup(name, reference, task=task)
    params, (pos, cell, graph) = setup.params, setup.inputs

    assert graph["centers"].shape[0] == expected["n_edges"]

    total, energy, per_atom = _total_energy(setup)
    forces = -np.asarray(
        jax.grad(lambda p: setup.energy_fn(params, p, cell, graph)[0])(pos)
    )

    assert abs(total - expected["energy"]) < 1e-5
    np.testing.assert_allclose(forces, np.array(expected["forces"]), atol=1e-5)
    # The per-atom decomposition must add up to the energy it is derived from.
    assert abs(float(np.sum(per_atom)) - float(energy)) < 1e-9


def test_heads_disagree(reference):
    """omat and odac are different levels of theory, so a shared fixture value
    would mean the task argument is not reaching the model."""
    omat = reference["tasks"]["omat"]["RSM0010"]["energy"]
    odac = reference["tasks"]["odac"]["RSM0010"]["energy"]
    assert abs(omat - odac) > 1.0


def test_energy_matches_torch_on_64_atoms(reference):
    """Parity on a 64-atom cell, the 2x2x2 supercell of the rattled Si8."""
    expected = reference["tasks"]["omat"]["si8_222"]["energy"]
    total, _, _ = _total_energy(_setup("si8_222", reference))

    assert abs(total - expected) < 1e-5


def test_float64_params_without_x64_preset():
    """`dtype="float64"` has to enable JAX x64 itself: with the flag off, JAX
    truncates a float64 array to float32 and only warns. Run in a subprocess
    because this module turns x64 on at import."""
    script = (
        "import jax, numpy as np\n"
        "from sadmof.models.uma import load_uma\n"
        "assert not jax.config.jax_enable_x64\n"
        f"_, params, _ = load_uma({str(CKPT)!r}, [14, 14], dtype='float64')\n"
        "dtypes = {leaf.dtype for leaf in jax.tree.leaves(params)}\n"
        "assert dtypes == {np.dtype('float64')}, dtypes\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_dense_hessian_is_symmetric(reference):
    from sadmof.dense import get_dense_hessian_fn

    setup = _setup("si8", reference)
    pos, cell, graph = setup.inputs
    hessian_fn = get_dense_hessian_fn(setup.energy_fn)
    hessian, _ = jax.jit(hessian_fn)(setup.params, pos, cell, graph)

    n = pos.shape[0]
    hessian = np.asarray(hessian).reshape(3 * n, 3 * n)
    assert np.abs(hessian).max() > 1e-3
    assert np.abs(hessian - hessian.T).max() < 1e-8


def test_sparse_hessian_matches_dense(reference):
    from sadmof.sparse import get_hessian_fn, hessian_coloring

    setup = _setup("si8", reference)
    params, energy_fn = setup.params, setup.energy_fn
    pos, cell, graph = setup.inputs
    n = pos.shape[0]

    dense = jax.hessian(lambda p: energy_fn(params, p, cell, graph)[0])(pos)
    dense = np.asarray(dense).reshape(3 * n, 3 * n)

    # 10 hops: the edge-degree embedding plus four blocks, doubled for the
    # Hessian. On this small cell the pattern is complete at any hop count, so
    # this checks the plumbing, not the reach.
    coloring = hessian_coloring(graph, 10)
    sparse, _ = jax.jit(get_hessian_fn(energy_fn, coloring))(params, pos, cell, graph)
    sparse = np.asarray(sparse.todense()).reshape(3 * n, 3 * n)

    np.testing.assert_allclose(sparse, dense, atol=1e-8)
    assert np.abs(dense).max() > 1e-3


def test_balance_channels_flag_changes_the_energy(reference):
    """The charge-balancing shift is the model's only non-local term; the flag
    that disables it has to actually reach the forward pass."""
    setup = _setup("si8", reference)
    unbalanced = _setup("si8", reference, balance_channels=False).energy_fn

    balanced_energy = float(setup.energy_fn(setup.params, *setup.inputs)[0])
    unbalanced_energy = float(unbalanced(setup.params, *setup.inputs)[0])
    assert abs(balanced_energy - unbalanced_energy) > 1e-6
