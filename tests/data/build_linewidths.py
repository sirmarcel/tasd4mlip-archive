"""Build the phono3py reference fixture for the Γ-only linewidth kernel.

Small periodic Ar LJ crystal (real eV/Å/amu units), fc2/fc3 by dense nested
AD, mode linewidths from phono3py 4.x with the tensors injected via the bare
`fc2`/`fc3` setters. The kernel test pins `sadmof.observables.linewidths`
against the stored gammas, so the Togo Eq. 10/11 conventions (including the
1/3! in the matrix element) are certified without phono3py in the test env.

Needs an env with phono3py, jax, and ase — NOT the project env, which stays
phono3py-free.
"""

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from ase.build import bulk
from ase.neighborlist import primitive_neighbor_list
from phono3py import Phono3py
from phonopy.structure.atoms import PhonopyAtoms

EPS = 0.0104  # eV
SIGMA_LJ = 3.4  # A
CUTOFF = 7.0  # A
A_LATT = 5.315  # A, near the LJ minimum for this cutoff
TEMPERATURES = [30.0, 60.0]
SIGMA_THZ = 0.25


def main():
    atoms = bulk("Ar", "fcc", a=A_LATT, cubic=True) * (2, 1, 1)
    n = len(atoms)
    i, j, S = primitive_neighbor_list("ijS", atoms.pbc, atoms.cell, atoms.positions, CUTOFF)
    shifts = jnp.asarray(S, dtype=jnp.float64) @ jnp.asarray(atoms.cell[:])
    ii, jj = jnp.asarray(i), jnp.asarray(j)

    def energy(pos):
        r = jnp.linalg.norm(pos[jj] + shifts - pos[ii], axis=-1)
        sr6 = (SIGMA_LJ / r) ** 6
        return 0.5 * jnp.sum(4.0 * EPS * (sr6**2 - sr6))

    pos = jnp.asarray(atoms.positions)
    fc2 = np.asarray(jax.hessian(energy)(pos)).transpose(0, 2, 1, 3)
    fc3 = np.asarray(jax.jacfwd(jax.hessian(energy))(pos)).transpose(0, 2, 4, 1, 3, 5)

    ph3 = Phono3py(
        PhonopyAtoms(symbols=["Ar"] * n, cell=atoms.cell[:], positions=atoms.positions),
        supercell_matrix=np.eye(3, dtype=int),
        primitive_matrix=np.eye(3),
    )
    masses = np.asarray(ph3.primitive.masses)
    ph3.fc2 = np.ascontiguousarray(fc2)
    ph3.fc3 = np.ascontiguousarray(fc3)
    ph3.mesh_numbers = [1, 1, 1]
    ph3.sigmas = [SIGMA_THZ]
    ph3.init_phph_interaction()
    ise = ph3.run_imag_self_energy(
        [0], temperatures=TEMPERATURES, frequency_points_at_bands=True
    )
    gammas = np.asarray(ise.gammas)[0, :, 0, :]
    ph3.run_phonon_solver()
    freqs = np.sort(np.asarray(ph3.get_phonon_data()[0])[0])
    assert gammas.max() > 0, "fixture without any scattering is useless"

    np.savez(
        "linewidths_ar_reference.npz",
        fc2=fc2,
        fc3=fc3,
        masses_amu=masses,
        temperatures=np.asarray(TEMPERATURES),
        sigma_thz=SIGMA_THZ,
        gammas_thz=gammas,
        freqs_thz=freqs,
        phono3py_version=np.array("see README"),
    )
    print(
        f"{n} atoms, freqs {freqs.min():.3f}..{freqs.max():.3f} THz, "
        f"gamma_max {gammas.max():.4e} THz -> linewidths_ar_reference.npz"
    )


if __name__ == "__main__":
    main()
