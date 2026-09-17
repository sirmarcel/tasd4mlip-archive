"""ASE calculator for the JAX UMA model.

Same Verlet-skin neighbour-list cache as `mace.calculator`: vesin runs at
``cutoff + skin`` and the resulting `graph` is reused across small-displacement
steps. UMA's polynomial envelope is exactly zero at and beyond the cutoff — in
value *and* in derivative, since `jnp.where` gates both — so the extra pairs a
skin captures contribute nothing.

What MACE does not need is the lazy **MOLE merge**. `load_uma` mixes the 64
experts down for one composition, charge and spin, so the calculator holds the
checkpoint rather than a parameter tree and merges on the first `calculate()`,
re-merging only when the composition changes.

On top of `get_energy_fn` the calculator adds the two coordinate-independent
terms the energy function leaves out — the task normaliser's `mean` and the
per-element references — in fp64 numpy outside JIT.
"""

import numpy as np
import jax
import jax.numpy as jnp

from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import BaseCalculator
from ase.stress import full_3x3_to_voigt_6_stress

from ..neighbor_cache import NLReference
from .energy import get_energy_fn
from .inputs import atoms_to_inputs
from .load import load_uma

__all__ = ["UMACalculator"]


class UMACalculator(BaseCalculator):
    """ASE calculator wrapping the JAX UMA port with a Verlet-cached NL.

    Args:
        checkpoint: Directory holding `config.json` + `model.npz`, i.e. the
            output of `sources/processed/uma-s-1p2/build.py`.
        task: Which training dataset to predict for — `omat` (PBE, pairs with a
            separate D3 term) or `odac` (PBE+D3 already). Not a swapped output
            layer: the dataset embedding feeds the MOLE routing MLP, so the
            whole mixed backbone changes.
        skin: Verlet-skin radius (Å). Pairs are captured at ``cutoff + skin``
            and the cache is rebuilt only when the worst-case displacement could
            pull a new pair inside the true cutoff.
        stress: If True, compute and report stress.
        default_dtype: ``"float64"`` (default) or ``"float32"``. Parameters and
            inputs are built at this dtype; the normaliser shift and the element
            references are always added in fp64 numpy.
        charge, spin: Total system charge and spin multiplicity index, folded
            into the expert mixing.
        bucket_strategy: `marathon.utils.next_size` strategy for padding the
            pair axis, so a Verlet rebuild does not recompile the model for
            every new pair count. The atom axis stays at `n_real + 1`.
        balance_channels: Passed to `UMA`; see its docstring.
    """

    name = "uma-jax"
    parameters: dict = {}
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        checkpoint: str | Path,
        task: str = "omat",
        skin: float = 0.5,
        stress: bool = True,
        default_dtype: str = "float64",
        charge: int = 0,
        spin: int = 0,
        bucket_strategy: str = "multiples",
        balance_channels: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._checkpoint = Path(checkpoint)
        self._task = str(task)
        self._skin = float(skin)
        self._stress = bool(stress)
        self._charge = int(charge)
        self._spin = int(spin)
        self._bucket_strategy = bucket_strategy
        self._balance_channels = bool(balance_channels)

        if default_dtype == "float64":
            jax.config.update("jax_enable_x64", True)
        self._default_dtype = default_dtype
        self._dtype = jnp.float64 if default_dtype == "float64" else jnp.float32

        # Filled by the first `calculate()`, once a composition is known.
        self._model = None
        self._params: dict | None = None
        self._energy_shift: float = 0.0
        self._element_references: np.ndarray | None = None
        self._predict_fn = None
        self._composition: np.ndarray | None = None

        self._graph: dict | None = None
        self._nl_ref: NLReference | None = None

        if not self._stress:
            self.implemented_properties = ["energy", "forces"]

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, **kwargs) -> "UMACalculator":
        """Alias of the constructor, for symmetry with the other calculators."""
        return cls(checkpoint, **kwargs)

    # -- ASE entry point --

    def calculate(self, atoms, properties=None, system_changes=None):
        self._ensure_params(atoms)
        if self._graph is None or self._nl_ref is None or self._nl_ref.needs_update(atoms):
            self._build_inputs(atoms)
        pos, cell = self._refresh_geometry(atoms)

        predict_fn, element_references = self._predict_fn, self._element_references
        assert predict_fn is not None and element_references is not None
        results = predict_fn(self._params, pos, cell, self._graph)

        n_real = len(atoms)
        references = float(element_references[atoms.get_atomic_numbers()].sum())

        energy = float(results["energy"]) + self._energy_shift + references
        forces = np.array(results["forces"][:n_real], dtype=np.float64)
        self.results = {"energy": energy, "forces": forces}

        if self._stress and atoms.pbc.any():
            virial = np.array(results["stress"], dtype=np.float64)
            volume = atoms.get_volume()
            self.results["stress"] = full_3x3_to_voigt_6_stress(virial / volume)

        return self.results

    # -- internals --

    def _ensure_params(self, atoms: Atoms) -> None:
        """Mix the MOLE experts for this composition, if not already done.

        Only the multiset of `Z` reaches the routing MLP (through the mean
        composition embedding), so a reordering or a displacement never needs a
        re-merge — the neighbour-list cache handles those. A re-merge does re-read
        the 1.2 GB archive and recompile, so a loop over a roster of compositions
        is better served by one calculator per composition.
        """
        composition = np.sort(atoms.get_atomic_numbers())
        merged = self._composition
        if merged is not None and np.array_equal(composition, merged):
            return

        model, params, metadata = load_uma(
            self._checkpoint,
            atoms.get_atomic_numbers(),
            task=self._task,
            charge=self._charge,
            spin=self._spin,
            dtype=self._default_dtype,
            balance_channels=self._balance_channels,
        )
        self._model = model
        self._params = params
        self._energy_shift = float(metadata["normalizer"]["mean"])
        self._element_references = np.asarray(
            metadata["element_references"], dtype=np.float64
        )
        self._predict_fn = jax.jit(_get_predict_fn(model))
        self._composition = composition
        self._graph = None
        self._nl_ref = None

    def _build_inputs(self, atoms: Atoms) -> None:
        """Run vesin (via `atoms_to_inputs`) at ``cutoff + skin``; cache the
        static `graph` and stamp the NL ref."""
        assert self._model is not None
        np_dtype = np.dtype(self._dtype)
        _, _, graph = atoms_to_inputs(
            atoms,
            self._model,
            skin=self._skin,
            float_dtype=np_dtype,
            bucket_strategy=self._bucket_strategy,
        )
        self._graph = graph

        max_shift = int(np.max(np.abs(np.asarray(graph["cell_shifts"]))))
        self._nl_ref = NLReference(self._skin, max_shift)
        self._nl_ref.save_reference(atoms)

    def _refresh_geometry(self, atoms: Atoms) -> tuple:
        """Build `(pos, cell)` from the current atoms; the cached `self._graph`
        is reused directly by the caller."""
        assert self._graph is not None
        np_dtype = np.dtype(self._dtype)
        n_real = len(atoms)

        n_atoms_padded = int(self._graph["atom_mask"].shape[0])
        positions_padded = np.zeros((n_atoms_padded, 3), dtype=np_dtype)
        positions_padded[:n_real] = atoms.get_positions()
        cell = np.array(atoms.get_cell()[:], dtype=np_dtype)

        return jnp.asarray(positions_padded), jnp.asarray(cell)


def _get_predict_fn(model):
    """`predict(params, pos, cell, graph) -> {energy, forces, stress}`.

    One `value_and_grad` over positions and the strain tensor `eps`: `pos` and
    `cell` go through `F = I + eps`, so at `eps -> 0` the gradient is the virial
    `dE/deps` (eV, no volume normalisation). Symmetrised as fairchem's
    `compute_forces_and_stress` does — the two halves agree analytically, by
    rotational invariance, and only differ at round-off.
    """
    energy_fn = get_energy_fn(model)

    def predict(params, pos, cell, graph):
        def energy_at(pos, eps):
            deformation = jnp.eye(3, dtype=pos.dtype) + eps
            return energy_fn(params, pos @ deformation, cell @ deformation, graph)

        (energy, _), (neg_forces, virial) = jax.value_and_grad(
            energy_at, argnums=(0, 1), has_aux=True
        )(pos, jnp.zeros((3, 3), dtype=pos.dtype))

        forces = -neg_forces * graph["atom_mask"][:, None]
        return {
            "energy": energy,
            "forces": forces,
            "stress": (virial + virial.T) / 2,
        }

    return predict
