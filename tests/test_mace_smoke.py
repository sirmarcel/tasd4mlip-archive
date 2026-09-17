"""Smoke tests for the ported MACE model.

These check that the model:
- loads from the marathon-style yaml + msgpack pair we produce at
  `sources/processed/mace-mp-0-medium/`,
- runs forward end-to-end on a small real MOF structure from
  `sources/processed/goennheimer+moosavi/structures.xyz`,
- returns finite, sensibly-shaped energies / forces / stress,
- is differentiable through `energy` and jit-able through `predict`.

PyTorch-parity reference tests (against pre-computed energies / forces /
stress) live in `test_mace_reference.py`.
"""

import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase.io import read

from sadmof.paths import SOURCES, checkpoint

jax.config.update("jax_enable_x64", True)

CHECKPOINT_DIR = checkpoint("mace-mp-0-medium")
STRUCTURES_XYZ = SOURCES / "goennheimer+moosavi" / "structures.xyz"


@pytest.fixture(scope="module")
def model_and_params():
    if not (CHECKPOINT_DIR / "model.msgpack").exists():
        pytest.skip(
            f"checkpoint not built at {CHECKPOINT_DIR}; "
            "run sources/processed/mace-mp-0-medium/build.py first"
        )
    from marathon.io import from_dict, read_msgpack, read_yaml

    model = from_dict(read_yaml(str(CHECKPOINT_DIR / "model.yaml")))
    params = read_msgpack(str(CHECKPOINT_DIR / "model.msgpack"))
    return model, params


@pytest.fixture(scope="module")
def inputs(model_and_params):
    from sadmof.models.mace import atoms_to_inputs

    model, _ = model_and_params
    if not STRUCTURES_XYZ.exists():
        pytest.skip(f"structures not built at {STRUCTURES_XYZ}")
    atoms = read(str(STRUCTURES_XYZ), index=0)
    pos, cell, graph = atoms_to_inputs(atoms, model.cutoff)
    return pos, cell, graph, int(len(atoms))


def test_imports():
    from sadmof.models.mace import MACE, atoms_to_inputs, dummy_inputs

    assert MACE().cutoff == 6.0
    assert callable(atoms_to_inputs)
    assert callable(dummy_inputs)


def test_model_loads_with_expected_handle(model_and_params):
    from sadmof.models.mace import MACE

    model, params = model_and_params
    assert isinstance(model, MACE)
    assert model.cutoff == 6.0
    assert model.num_elements == 95
    assert "params" in params
    assert "constants" in params


def test_inputs_schema(inputs):
    pos, cell, graph, n_real = inputs
    expected = {
        "cell_shifts",
        "atomic_numbers",
        "centers",
        "others",
        "atom_mask",
        "pair_mask",
    }
    assert set(graph) == expected
    assert pos.shape[1] == 3
    assert cell.shape == (3, 3)
    assert graph["atom_mask"][:n_real].all()
    assert not graph["atom_mask"][n_real:].any()


def test_energy_finite_and_differentiable(model_and_params, inputs):
    model, params = model_and_params
    pos, cell, graph, n_real = inputs

    total, per_atom = model.energy(params, pos, cell, graph)

    assert jnp.isfinite(total)
    assert per_atom.shape[0] == graph["atomic_numbers"].shape[0]
    assert jnp.all(jnp.isfinite(per_atom))
    # Padding atoms are masked
    assert jnp.all(per_atom[n_real:] == 0)

    # Differentiable w.r.t. params (asdex / training path).
    def loss(p):
        e, _ = model.energy(p, pos, cell, graph)
        return e

    grads = jax.grad(loss, allow_int=True)(params)
    grad_leaves = jax.tree_util.tree_leaves(grads["params"])
    assert any(jnp.any(g != 0) for g in grad_leaves), "all param gradients are zero"

    # Differentiable w.r.t. positions (Hessian path).
    def energy_of_positions(positions):
        return model.energy(params, positions, cell, graph)[0]

    grad_positions = jax.grad(energy_of_positions)(pos)
    assert grad_positions.shape == pos.shape
    assert jnp.any(grad_positions[:n_real] != 0), "no force on any real atom"


def test_predict_shapes_and_finiteness(model_and_params, inputs):
    model, params = model_and_params
    pos, cell, graph, n_real = inputs

    preds = model.predict(params, pos, cell, graph)

    assert set(preds) == {"energy", "forces", "stress"}

    n_atoms_padded = graph["atomic_numbers"].shape[0]
    assert preds["energy"].shape == ()
    assert preds["forces"].shape == (n_atoms_padded, 3)
    assert preds["stress"].shape == (3, 3)

    assert jnp.isfinite(preds["energy"])
    assert jnp.all(jnp.isfinite(preds["forces"]))
    assert jnp.all(jnp.isfinite(preds["stress"]))

    # Padding atoms have zero forces
    np.testing.assert_array_equal(np.array(preds["forces"][n_real:]), 0.0)


def test_predict_jits(model_and_params, inputs):
    model, params = model_and_params
    pos, cell, graph, _ = inputs

    jitted = jax.jit(model.predict)
    preds = jitted(params, pos, cell, graph)

    assert jnp.isfinite(preds["energy"])
    assert jnp.all(jnp.isfinite(preds["forces"]))
    assert jnp.all(jnp.isfinite(preds["stress"]))
