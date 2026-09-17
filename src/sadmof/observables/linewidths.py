"""Γ-only three-phonon linewidths from third-order force constants.

Implements Togo, Chaput, Tanaka, PRB 91, 094306 (2015), Eqs. 10/11/13 in the
Γ-only case (cell = unit cell, mesh 1×1×1): momentum conservation is trivial,
all Bloch phases are 1, eigenvectors are real, and −λ = λ. The matrix element
carries a 1/3!, paired with Eq. 11's 18π/ħ² prefactor — without it the result
is off by exactly (3!)². The kernel is pinned mode-by-mode against phono3py
(`tests/test_fc3.py`).

Two entry points share one golden-rule core:

- `gamma_only_linewidths(fc2, fc3, ...)` — all modes from a materialized
  tensor; affordable for small cells only.
- `slice_gamma(m_slice, lam, ...)` — one mode's linewidth from the slice
  D³E·(d_λ, ·, ·) with d_λ = `mode_direction(evecs, masses, lam)`; the large-N
  path, one sparse slice per interrogated mode.

Inputs in eV/Å² (fc2), eV/Å³ (fc3, slices), amu; output in THz without 2π
(phono3py's gamma convention; τ_λ = 1/(4π γ·10¹²) s).
"""

import numpy as np

from scipy.constants import Boltzmann as KB
from scipy.constants import electron_volt as EV
from scipy.constants import hbar as HBAR
from scipy.constants import physical_constants

__all__ = [
    "gamma_only_linewidths",
    "mode_basis",
    "mode_direction",
    "slice_gamma",
]

AMU = physical_constants["atomic mass constant"][0]
ANG = 1e-10
THZ = 1e12

# Modes below this frequency are excluded from targets and scattering partners
# (phono3py's cutoff_frequency): the acoustic zeros at Γ, where occupations
# diverge. THz without 2pi.
FREQ_CUTOFF_THZ = 1e-4

_PREFACTOR = 18.0 * np.pi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mode_basis(fc2, masses_amu):
    """Diagonalize the Γ-point dynamical matrix.

    Args:
        fc2: `(n, n, 3, 3)` force constants, eV/Å².
        masses_amu: `(n,)` atomic masses, amu.

    Returns:
        `(nu_thz, evecs)`: frequencies in THz (without 2π, ascending; imaginary
        modes clipped to 0) and the `(3n, 3n)` eigenvector matrix of the
        mass-weighted dynamical matrix, one mode per column.
    """
    n = len(masses_amu)
    m3 = np.repeat(np.asarray(masses_amu) * AMU, 3)
    D = fc2.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n) * (EV / ANG**2)
    D = D / np.sqrt(np.outer(m3, m3))
    w2, evecs = np.linalg.eigh(0.5 * (D + D.T))
    nu_thz = np.sqrt(np.clip(w2, 0.0, None)) / (2.0 * np.pi * THZ)
    return nu_thz, evecs


def mode_direction(evecs, masses_amu, lam):
    """Cartesian contraction direction for mode `lam`.

    The `(n, 3)` vector `W[:, lam] / sqrt(m_amu)` to hand to a slice function;
    the remaining mass normalization (amu → kg) and the √(ħ/2ω) amplitudes are
    applied inside `slice_gamma`.
    """
    n = len(masses_amu)
    d = evecs[:, lam] / np.sqrt(np.repeat(masses_amu, 3))
    return d.reshape(n, 3)


def slice_gamma(
    m_slice,
    lam,
    nu_thz,
    evecs,
    masses_amu,
    temperatures,
    sigma_thz,
):
    """Linewidth Γ_λ(ω_λ) of one mode from its third-order slice.

    Args:
        m_slice: `(n, 3, n, 3)` slice D³E·(d_λ, ·, ·) in eV/Å³, computed with
            `d_λ = mode_direction(evecs, masses_amu, lam)`.
        lam: Target mode index into the `mode_basis` output.
        nu_thz: `(3n,)` mode frequencies, THz.
        evecs: `(3n, 3n)` eigenvectors from `mode_basis`.
        masses_amu: `(n,)` atomic masses, amu.
        temperatures: `(nT,)` in K.
        sigma_thz: Gaussian smearing width for the energy deltas, THz.

    Returns:
        `(nT,)` Γ_λ(ω_λ) in THz without 2π; 0 for an excluded (near-zero) mode.
    """
    nu = np.asarray(nu_thz)
    live = nu > FREQ_CUTOFF_THZ
    if not live[lam]:
        return np.zeros(len(temperatures))

    n = len(masses_amu)
    m3 = np.repeat(np.asarray(masses_amu) * AMU, 3)
    U = evecs / np.sqrt(m3)[:, None]

    # rotate the two open indices; the λ index was contracted in amu units,
    # so one amu→kg factor is still outstanding
    M = m_slice.reshape(3 * n, 3 * n) * (EV / ANG**3) / np.sqrt(AMU)
    V = U.T @ M @ U

    omega = 2.0 * np.pi * THZ * nu
    amp = np.where(live, np.sqrt(HBAR / (2.0 * np.maximum(omega, 1.0))), 0.0)
    V = V * amp[lam] * amp[:, None] * amp[None, :] / 6.0
    return _golden_rule(np.abs(V) ** 2, lam, nu, omega, live, temperatures, sigma_thz)


def gamma_only_linewidths(fc2, fc3, masses_amu, temperatures, sigma_thz):
    """Linewidths of all modes from materialized force constants.

    Args:
        fc2: `(n, n, 3, 3)` eV/Å².
        fc3: `(n, n, n, 3, 3, 3)` eV/Å³, phono3py index order.
        masses_amu: `(n,)` amu.
        temperatures: `(nT,)` K.
        sigma_thz: Gaussian smearing width, THz.

    Returns:
        `(nu_thz, gamma_thz)`: frequencies `(3n,)` and Γ_λ(ω_λ) of shape
        `(nT, 3n)`, both THz without 2π; excluded modes carry 0.
    """
    nu, evecs = mode_basis(fc2, masses_amu)
    n = len(masses_amu)
    live = nu > FREQ_CUTOFF_THZ
    m3 = np.repeat(np.asarray(masses_amu) * AMU, 3)
    U = evecs / np.sqrt(m3)[:, None]

    F = fc3.transpose(0, 3, 1, 4, 2, 5).reshape(3 * n, 3 * n, 3 * n) * (EV / ANG**3)
    P = np.einsum("abc,al->lbc", F, U, optimize=True)
    P = np.einsum("lbc,bm->lmc", P, U, optimize=True)
    P = np.einsum("lmc,cn->lmn", P, U, optimize=True)

    omega = 2.0 * np.pi * THZ * nu
    amp = np.where(live, np.sqrt(HBAR / (2.0 * np.maximum(omega, 1.0))), 0.0)
    P = P * (amp[:, None, None] * amp[None, :, None] * amp[None, None, :]) / 6.0
    P2 = np.abs(P) ** 2

    gamma = np.zeros((len(temperatures), 3 * n))
    for lam in np.nonzero(live)[0]:
        gamma[:, lam] = _golden_rule(P2[lam], lam, nu, omega, live, temperatures, sigma_thz)
    return nu, gamma


# ---------------------------------------------------------------------------
# Golden-rule core
# ---------------------------------------------------------------------------


def _golden_rule(p2, lam, nu, omega, live, temperatures, sigma_thz):
    """Eq. 11 for one target mode; `p2` is the `(3n, 3n)` matrix |Φ_λλ'λ''|²."""

    def g(x_thz):
        return np.exp(-0.5 * (x_thz / sigma_thz) ** 2) / (sigma_thz * np.sqrt(2.0 * np.pi))

    d_sum = g(nu[lam] - nu[:, None] - nu[None, :])
    d_diff = g(nu[lam] + nu[:, None] - nu[None, :])

    # delta in angular frequency = gaussian in nu / (2π·THZ); the output gamma
    # is itself in nu convention, so (2π·THZ)² lands in the denominator
    pref = _PREFACTOR / HBAR**2 / (2.0 * np.pi * THZ) ** 2

    gamma = np.zeros(len(temperatures))
    for it, T in enumerate(temperatures):
        occ = np.zeros_like(nu)
        occ[live] = 1.0 / np.expm1(HBAR * omega[live] / (KB * T))
        ch1 = (occ[:, None] + occ[None, :] + 1.0) * d_sum
        ch2 = (occ[:, None] - occ[None, :]) * (d_diff - d_diff.T)
        gamma[it] = pref * np.sum(p2 * (ch1 + ch2))
    return gamma
