"""ASE calculator for the JAX MACE model.

Verlet-skin neighbor list cache: vesin runs at ``cutoff + skin`` and the
resulting `graph` (connectivity: `centers`, `others`, `cell_shifts`, …) is
reused across small-displacement steps — `_refresh_geometry` rebuilds only
`pos` and `cell` from the atoms each step and pairs them with the cached graph.

The per-element baseline (``params["constants"]["atomic_energies"]``)
lives outside JIT and is added in fp64 numpy to avoid numerical issues.
"""

import numpy as np
import jax
import jax.numpy as jnp

from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import BaseCalculator
from ase.stress import full_3x3_to_voigt_6_stress

from ...utils import cast_floats
from ..neighbor_cache import NLReference
from .inputs import atoms_to_inputs
from .model import MACE

__all__ = ["MACECalculator"]


class MACECalculator(BaseCalculator):
    """ASE calculator wrapping `MACE.predict` with a Verlet-cached NL.

    Args:
        model: A `MACE` instance.
        params: Its parameter pytree (e.g. from `marathon.io.read_msgpack`).
        skin: Verlet-skin radius (Å). Pairs are captured at ``cutoff + skin``
            and the cache is rebuilt only when the worst-case displacement could
            pull a new pair inside the true cutoff.
        stress: If True, compute and report stress.
        default_dtype: ``"float32"`` (default) or ``"float64"``. Floating-point
            params and inputs are cast to this dtype; baseline addition is
            always fp64 numpy.
    """

    name = "mace-jax"
    parameters: dict = {}
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: MACE,
        params: dict,
        skin: float = 0.5,
        stress: bool = True,
        default_dtype: str = "float32",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model = model
        self._cutoff = float(model.cutoff)
        self._skin = float(skin)
        self._stress = bool(stress)

        if default_dtype == "float64":
            jax.config.update("jax_enable_x64", True)
        self._dtype = jnp.float64 if default_dtype == "float64" else jnp.float32

        # Pull per-element baseline out before casting — it lives in fp64
        # numpy and is added post-JIT.
        self._atomic_energies = np.array(
            params["constants"]["atomic_energies"], dtype=np.float64
        )
        self._params = cast_floats(params, self._dtype)

        self._predict_fn = jax.jit(self._model.predict)

        self._graph: dict | None = None
        self._nl_ref: NLReference | None = None

        if not self._stress:
            self.implemented_properties = ["energy", "forces"]

    @classmethod
    def from_checkpoint(cls, folder: str | Path, **kwargs) -> "MACECalculator":
        """Build from a yaml + msgpack pair produced by
        `sources/processed/mace-mp-0-medium/build.py`."""
        from marathon.io import from_dict, read_msgpack, read_yaml

        folder = Path(folder)
        model = from_dict(read_yaml(str(folder / "model.yaml")))
        params = read_msgpack(str(folder / "model.msgpack"))
        return cls(model, params, **kwargs)

    # -- ASE entry point --

    def calculate(self, atoms, properties=None, system_changes=None):
        if self._graph is None or self._nl_ref is None or self._nl_ref.needs_update(atoms):
            self._build_inputs(atoms)
        pos, cell = self._refresh_geometry(atoms)

        results = self._predict_fn(self._params, pos, cell, self._graph)

        n_real = len(atoms)
        Zs = atoms.get_atomic_numbers()
        baseline = float(self._atomic_energies[Zs].sum())

        energy = float(results["energy"]) + baseline
        forces = np.array(results["forces"][:n_real], dtype=np.float64)
        self.results = {"energy": energy, "forces": forces}

        if self._stress and atoms.pbc.any():
            virial = np.array(results["stress"], dtype=np.float64)
            volume = atoms.get_volume()
            self.results["stress"] = full_3x3_to_voigt_6_stress(virial / volume)

        return self.results

    # -- internals --

    def _build_inputs(self, atoms: Atoms) -> None:
        """Run vesin (via `atoms_to_inputs`) at ``cutoff + skin``; cache the
        static `graph` and stamp the NL ref."""
        np_dtype = np.dtype(self._dtype)
        _, _, graph = atoms_to_inputs(
            atoms, cutoff=self._cutoff + self._skin, float_dtype=np_dtype
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
