"""ASE calculator for the JAX DFT-D3(BJ) model.

Same Verlet-skin neighbour-list caching as `models.mace.MACECalculator`:
vesin runs at ``cutoff + skin`` / ``cnthr + skin`` and the resulting `graph`
(both half pair lists) is reused across small-displacement steps;
`_refresh_geometry` rebuilds only `pos` and `cell` each step.

D3 has no per-element baseline, so — unlike MACE — there is nothing to add
post-JIT. To get MACE + D3 (energy, forces, stress all summed), compose this
with `MACECalculator` via ASE's `ase.calculators.mixing.SumCalculator`:

    from ase.calculators.mixing import SumCalculator
    atoms.calc = SumCalculator([mace_calc, d3_calc])

Use the same `default_dtype` for both so the global JAX x64 flag is consistent.
"""

import numpy as np
import jax
import jax.numpy as jnp

from ase.calculators.calculator import BaseCalculator
from ase.stress import full_3x3_to_voigt_6_stress

from ...utils import cast_floats
from ..neighbor_cache import NLReference
from .inputs import atoms_to_inputs
from .model import D3

__all__ = ["D3Calculator"]


class D3Calculator(BaseCalculator):
    """ASE calculator wrapping `D3.predict` with a Verlet-cached NL.

    Args:
        model: A `D3` instance. Defaults to `D3(xc=xc)`.
        params: Its parameter pytree. Defaults to `D3.load_params()` (the
            tables shipped with the package).
        xc: Functional name for the BJ parameters, used only when `model`
            is not given.
        skin: Verlet-skin radius (Å), added to both pair-list cutoffs.
        stress: If True, compute and report stress.
        default_dtype: ``"float32"`` (default) or ``"float64"``. The D3
            reference values are reproduced in fp64; fp32 is adequate for
            relaxation / MD but loses precision in the long-range sum.
    """

    name = "d3-jax"
    parameters: dict = {}
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: D3 | None = None,
        params: dict | None = None,
        xc: str = "pbe",
        skin: float = 0.5,
        stress: bool = True,
        default_dtype: str = "float32",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model = model if model is not None else D3(xc=xc)
        self._cutoff = float(self._model.cutoff)
        self._cnthr = float(self._model.cnthr)
        self._skin = float(skin)
        self._stress = bool(stress)

        if default_dtype == "float64":
            jax.config.update("jax_enable_x64", True)
        self._dtype = jnp.float64 if default_dtype == "float64" else jnp.float32

        if params is None:
            params = D3.load_params()
        self._params = cast_floats(params, self._dtype)

        self._predict_fn = jax.jit(self._model.predict)

        self._graph: dict | None = None
        self._nl_ref: NLReference | None = None

        if not self._stress:
            self.implemented_properties = ["energy", "forces"]

    # -- ASE entry point --

    def calculate(self, atoms, properties=None, system_changes=None):
        if self._graph is None or self._nl_ref is None or self._nl_ref.needs_update(atoms):
            self._build_inputs(atoms)
        pos, cell = self._refresh_geometry(atoms)

        results = self._predict_fn(self._params, pos, cell, self._graph)

        n_real = len(atoms)
        self.results = {
            "energy": float(results["energy"]),
            "forces": np.array(results["forces"][:n_real], dtype=np.float64),
        }

        if self._stress and atoms.pbc.any():
            virial = np.array(results["stress"], dtype=np.float64)
            volume = atoms.get_volume()
            self.results["stress"] = full_3x3_to_voigt_6_stress(virial / volume)

        return self.results

    # -- internals --

    def _build_inputs(self, atoms) -> None:
        """Run vesin (via `atoms_to_inputs`) at ``cutoff + skin`` and
        ``cnthr + skin``; cache the static `graph` and stamp the NL ref."""
        np_dtype = np.dtype(self._dtype)
        _, _, graph = atoms_to_inputs(
            atoms,
            cutoff=self._cutoff + self._skin,
            cnthr=self._cnthr + self._skin,
            float_dtype=np_dtype,
        )
        self._graph = graph

        max_shift = int(np.max(np.abs(np.asarray(graph["cell_shifts"]))))
        self._nl_ref = NLReference(self._skin, max_shift)
        self._nl_ref.save_reference(atoms)

    def _refresh_geometry(self, atoms) -> tuple:
        """Build `(pos, cell)` from the current atoms; `self._graph` is reused
        directly by the caller."""
        assert self._graph is not None
        np_dtype = np.dtype(self._dtype)
        n_real = len(atoms)

        n_atoms_padded = int(self._graph["atom_mask"].shape[0])
        positions_padded = np.zeros((n_atoms_padded, 3), dtype=np_dtype)
        positions_padded[:n_real] = atoms.get_positions()
        cell = np.array(atoms.get_cell()[:], dtype=np_dtype)

        return jnp.asarray(positions_padded), jnp.asarray(cell)
