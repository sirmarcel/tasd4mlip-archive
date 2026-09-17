"""DFT-D3(BJ) dispersion correction — JAX / flax model.

Cleanroom port of torch_dftd's DFT-D3(BJ), for use as an additive correction
to MACE or another MLIP. Only BJ damping is implemented (the MACE-MP default).

Follows the same `(params, pos, cell, graph)` contract as `models.mace.MACE`:
`pos` and `cell` are the differentiable geometry; `graph` carries the static
topology. D3 needs two half neighbour lists — a large one (`cutoff`) for the
dispersion sum and a small one (`cnthr`, `cn_*` keys) for coordination numbers
— so the graph carries both.

Reference: Grimme et al., JCC 32, 1456 (2011); Grimme et al., JCP 132, 154104 (2010).
"""

import numpy as np
import jax
import jax.numpy as jnp

from pathlib import Path

import flax.linen as nn

from jaxtyping import Array, Float

from ..mace.model import safe_norm

__all__ = ["D3", "D3_BJ_PARAMS"]


# ---------------------------------------------------------------------------
# Constants (matching torch_dftd exactly)
# ---------------------------------------------------------------------------

D3_AUTOANG = 0.52917726  # bohr -> angstrom
D3_AUTOEV = 27.21138505  # hartree -> eV
D3_K1 = 16.0
D3_K3 = -4.0

# XC-dependent BJ-damping parameters (BJ only — the MACE-MP default).
D3_BJ_PARAMS = {
    "pbe": {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407},
    "pbe0": {"s6": 1.0, "s8": 1.2177, "a1": 0.4145, "a2": 4.8593},
    "b3-lyp": {"s6": 1.0, "s8": 1.9889, "a1": 0.3981, "a2": 4.4211},
    "tpss": {"s6": 1.0, "s8": 1.9435, "a1": 0.4535, "a2": 4.4752},
    "hse06": {"s6": 1.0, "s8": 2.310, "a1": 0.383, "a2": 5.685},
    "revpbe": {"s6": 1.0, "s8": 2.3550, "a1": 0.5238, "a2": 3.5016},
    "r2scan": {"s6": 1.0, "s8": 0.7898, "a1": 0.4948, "a2": 5.7308},
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class D3(nn.Module):
    """DFT-D3(BJ) dispersion correction — flax module + marathon-style
    energy/predict, mirroring `models.mace.MACE`.

    `__call__` returns per-atom energies (the D3 pair sum spread evenly over
    real atoms, since dispersion has no natural per-atom decomposition).
    `energy` / `predict` take `(params, pos, cell, graph)`; `predict` returns
    `{energy, forces, stress}` for a single structure.

    Static parameter tables (`c6ab`, `rcov`, `r2r4`) live in the ``constants``
    collection; load them with `D3.load_params()`.
    """

    cutoff: float = 50.2718  # angstrom (≈ 95 bohr, torch_dftd default)
    cnthr: float = 21.1671  # angstrom (≈ 40 bohr, CN cutoff)
    xc: str = "pbe"

    @nn.compact
    def __call__(
        self,
        pos: Float[Array, "atoms 3"],
        cell: Float[Array, "3 3"],
        graph: dict[str, Array],
    ) -> Float[Array, " atoms"]:
        """Per-atom D3(BJ) energies (eV), masked to real atoms."""
        Z = graph["atomic_numbers"]
        atom_mask = graph["atom_mask"]

        c6ab = self.variable(
            "constants", "c6ab", lambda: jnp.zeros((95, 95, 5, 5, 3))
        ).value
        rcov = self.variable("constants", "rcov", lambda: jnp.zeros((95,))).value
        r2r4 = self.variable("constants", "r2r4", lambda: jnp.zeros((95,))).value

        xc_params = D3_BJ_PARAMS[self.xc]

        # --- CN pair list (small cutoff) ---
        cn_R_ij = (
            pos[graph["cn_others"]]
            - pos[graph["cn_centers"]]
            + graph["cn_cell_shifts"] @ cell
        )
        cn_r_ang = safe_norm(cn_R_ij, axis=-1)
        cn_r_ang = jnp.where(graph["cn_pair_mask"], cn_r_ang, 1.0)
        # Hard cutoff: drop skin pairs beyond the true CN cutoff.
        cn_mask = graph["cn_pair_mask"] & (cn_r_ang < self.cnthr)
        cn_r_bohr = cn_r_ang / D3_AUTOANG

        nc = _ncoord(Z, cn_r_bohr, graph["cn_centers"], graph["cn_others"], rcov, cn_mask)

        # --- Energy pair list (large cutoff) ---
        e_R_ij = pos[graph["others"]] - pos[graph["centers"]] + graph["cell_shifts"] @ cell
        e_r_ang = safe_norm(e_R_ij, axis=-1)
        e_r_ang = jnp.where(graph["pair_mask"], e_r_ang, 1.0)
        # Hard cutoff: drop skin pairs beyond the true energy cutoff.
        e_mask = graph["pair_mask"] & (e_r_ang < self.cutoff)
        e_r_bohr = e_r_ang / D3_AUTOANG

        e_hartree = _edisp_bj(
            Z,
            e_r_bohr,
            graph["centers"],
            graph["others"],
            nc,
            c6ab,
            r2r4,
            e_mask,
            s6=xc_params["s6"],
            s8=xc_params["s8"],
            a1=xc_params["a1"],
            a2=xc_params["a2"],
        )
        e_ev = e_hartree * D3_AUTOEV

        # Distribute evenly over real atoms.
        n_real = jnp.sum(atom_mask)
        per_atom = jnp.where(atom_mask, e_ev / jnp.maximum(n_real, 1.0), 0.0)
        return per_atom

    def energy(
        self,
        params: dict,
        pos: Float[Array, "atoms 3"],
        cell: Float[Array, "3 3"],
        graph: dict[str, Array],
    ) -> tuple[Float[Array, ""], Float[Array, " atoms"]]:
        """Total energy + per-atom energies, both masked.

        Returns `(scalar_total, per_atom)` — the `(scalar, aux)` shape
        `jax.value_and_grad(has_aux=True)` expects.
        """
        per_atom = jnp.asarray(self.apply(params, pos, cell, graph))
        return jnp.sum(per_atom), per_atom

    def predict(
        self,
        params: dict,
        pos: Float[Array, "atoms 3"],
        cell: Float[Array, "3 3"],
        graph: dict[str, Array],
    ) -> dict[str, Array]:
        """Scalar energy + per-atom forces + 3×3 stress for a single structure.

        Same strain-derivative scheme as `MACE.predict`: `pos` and `cell` are
        mapped through `F = I + ε`, and at `ε → 0` the gradient w.r.t. `ε` is
        the virial (eV, no volume normalisation — the calculator divides by
        volume for ASE's convention).
        """

        def energy_at(pos, eps):
            F = jnp.eye(3) + eps
            return self.energy(params, pos @ F, cell @ F, graph)

        (total, _), (neg_forces, virial) = jax.value_and_grad(
            energy_at, argnums=(0, 1), has_aux=True
        )(pos, jnp.zeros((3, 3)))

        forces = -neg_forces * graph["atom_mask"][..., None]
        return {"energy": total, "forces": forces, "stress": virial}

    @staticmethod
    def load_params(params_path: str | Path | None = None) -> dict:
        """Load the static D3 tables (`c6ab`, `rcov`, `r2r4`) into a params dict.

        Defaults to the `dftd3_params.npz` shipped alongside this module.
        """
        if params_path is None:
            params_path = Path(__file__).parent / "dftd3_params.npz"
        d3_data = np.load(params_path)
        return {
            "constants": {
                "c6ab": jnp.asarray(d3_data["c6ab"]),
                "rcov": jnp.asarray(d3_data["rcov"]),
                "r2r4": jnp.asarray(d3_data["r2r4"]),
            },
        }


# ---------------------------------------------------------------------------
# Core D3 functions (JAX, no dynamic control flow)
# ---------------------------------------------------------------------------


def _ncoord(Z, r, idx_i, idx_j, rcov, pair_mask):
    """Fractional coordination numbers via inverse damping.

    Uses a half neighbour list (each pair once), so accumulates on both i and j.
    """
    Zi = Z[idx_i]
    Zj = Z[idx_j]
    rco = rcov[Zi] + rcov[Zj]
    rr = rco / r
    damp = 1.0 / (1.0 + jnp.exp(-D3_K1 * (rr - 1.0)))
    damp = damp * pair_mask

    n_atoms = Z.shape[0]
    nc = jax.ops.segment_sum(damp, idx_i, num_segments=n_atoms)
    nc = nc + jax.ops.segment_sum(damp, idx_j, num_segments=n_atoms)
    return nc


def _getc6(Zi, Zj, nci, ncj, c6ab):
    """Interpolate C6 coefficients from the reference table."""
    c6ab_flat = c6ab.reshape(-1, 25, 3)
    index = Zi * 95 + Zj

    cn0 = c6ab_flat[index, :, 0]
    cn1 = c6ab_flat[index, :, 1]
    cn2 = c6ab_flat[index, :, 2]

    r = (cn1 - nci[:, None]) ** 2 + (cn2 - ncj[:, None]) ** 2

    k3_rnc = jnp.where(cn0 > 0.0, D3_K3 * r, -1.0e20)
    weights = jax.nn.softmax(k3_rnc, axis=1)
    c6 = jnp.sum(weights * cn0, axis=1)
    return c6


def _edisp_bj(Z, r, idx_i, idx_j, nc, c6ab, r2r4, pair_mask, s6, s8, a1, a2):
    """D3(BJ) dispersion energy in Hartree.

    All distances in bohr. Uses a half neighbour list (×2 at the end).
    CN is pre-computed on the separate CN pair list.
    """
    r2 = r**2
    r6 = r2**3
    r8 = r6 * r2

    Zi = Z[idx_i]
    Zj = Z[idx_j]
    nci = nc[idx_i]
    ncj = nc[idx_j]

    c6 = _getc6(Zi, Zj, nci, ncj, c6ab)
    c8 = 3.0 * c6 * r2r4[Zi] * r2r4[Zj]

    # BJ damping — double-where to avoid NaN gradients on padding pairs.
    safe_c6 = jnp.where(pair_mask, c6, 1.0)
    safe_c8 = jnp.where(pair_mask, c8, 1.0)
    safe_r6 = jnp.where(pair_mask, r6, 1.0)
    safe_r8 = jnp.where(pair_mask, r8, 1.0)

    tmp = a1 * jnp.sqrt(safe_c8 / safe_c6) + a2
    tmp2 = tmp**2
    tmp6 = tmp2**3
    tmp8 = tmp6 * tmp2
    e6 = 1.0 / (safe_r6 + tmp6)
    e8 = 1.0 / (safe_r8 + tmp8)

    e6 = -0.5 * s6 * safe_c6 * e6
    e8 = -0.5 * s8 * safe_c8 * e8
    e68 = jnp.where(pair_mask, e6 + e8, 0.0)

    # Half list: multiply by 2.
    return jnp.sum(e68) * 2.0
