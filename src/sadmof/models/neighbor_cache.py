"""Verlet-skin neighbor-list cache shared by the JAX ASE calculators.

A calculator builds its neighbour list once at ``cutoff + skin`` and reuses
the resulting static `graph` across small-displacement steps. `NLReference`
remembers the geometry at the last rebuild and decides when the cache is
stale; the calculator rebuilds only when it says so.
"""

import numpy as np

from ase import Atoms

__all__ = ["NLReference"]


class NLReference:
    """Verlet-skin tracker: remembers the geometry at last NL rebuild and
    decides when the cache is stale.

    The standard skin test for fixed cells is ``2 · max_disp > skin``.
    With a non-trivial unit cell we additionally bound the per-pair
    contribution from cell deformation by ``max_cell_shift · |Δa_i|`` —
    overly conservative but cheap.
    """

    def __init__(self, skin: float, max_cell_shift: int):
        self.skin = skin
        self._ref_positions: np.ndarray | None = None
        self._ref_cell: np.ndarray | None = None
        self._ref_numbers: np.ndarray | None = None
        self._ref_pbc: np.ndarray | None = None
        self._max_cell_shift = max_cell_shift

    def needs_update(self, atoms: Atoms) -> bool:
        if self._ref_positions is None:
            return True
        if len(atoms) != len(self._ref_positions):
            return True
        if (atoms.get_atomic_numbers() != self._ref_numbers).any():
            return True
        if (atoms.get_pbc() != self._ref_pbc).any():
            return True

        disp = atoms.get_positions() - self._ref_positions
        max_disp = float(np.linalg.norm(disp, axis=1).max(initial=0.0))

        if self._max_cell_shift > 0:
            cell_change = np.array(atoms.get_cell()[:]) - self._ref_cell
            cell_contrib = float(
                self._max_cell_shift * np.linalg.norm(cell_change, axis=1).sum()
            )
        else:
            cell_contrib = 0.0

        return bool(2 * max_disp + cell_contrib > self.skin)

    def save_reference(self, atoms: Atoms) -> None:
        self._ref_positions = atoms.get_positions().copy()
        self._ref_cell = np.array(atoms.get_cell()[:]).copy()
        self._ref_numbers = atoms.get_atomic_numbers().copy()
        self._ref_pbc = atoms.get_pbc().copy()
