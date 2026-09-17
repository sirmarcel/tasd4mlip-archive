"""Hessian → phonon frequencies → heat capacity.

Pure-numpy observables layer: turn a Cartesian positions-Hessian (eV/Å²) into
phonon frequencies (cm⁻¹) by mass-weighted eigendecomposition, then into the
harmonic heat capacity C_v(T). No jax and no I/O — callers densify the sparse
Hessian (`np.asarray(H.todense()).reshape(3N, 3N)`), slice it to the real
atoms, and pass it in alongside the matching `(N,)` masses.

The frequency step is ASE's mass-weighted eigendecomposition, lifted from
`ase.vibrations.VibrationsData` (ase/vibrations/data.py). The C_v pipeline
wrapped around it — raw Hessian by default, imaginary modes taken to 0,
harmonic per-mode C_v, gravimetric reporting — follows Gönnheimer, Reuter &
Margraf, JCTC 21, 4742 (2025), whose code copied the same ASE block. The
standard force-constant acoustic sum rule is available opt-in (`enforce_asr`).
"""

import numpy as np

from ase import units
from scipy.constants import N_A, c, h, k

__all__ = [
    "FREQ_THRESHOLD_CM1",
    "cv_from_hessian",
    "hessian_to_frequencies",
    "cv_curve",
    "heat_capacity",
]

# Modes below this frequency (cm⁻¹) — acoustic and imaginary — are dropped from
# C_v: the QHO term's soft-mode branch (→ k as ν → 0) makes the three acoustic
# modes and any small negative eigenvalues contribute spuriously.
FREQ_THRESHOLD_CM1 = 1e-3


# ----
# Public API
# ----


def cv_from_hessian(
    hessian,
    masses,
    temperatures,
    *,
    enforce_asr=False,
    freq_scale=1.0,
    threshold=FREQ_THRESHOLD_CM1,
    overwrite=False,
):
    """Hessian → gravimetric C_v(T) curve in one call.

    Args:
        hessian: (3N, 3N) Cartesian Hessian in eV/Å² (real atoms only).
        masses: (N,) atomic masses in amu.
        temperatures: Iterable of temperatures in K.
        enforce_asr: Impose the acoustic sum rule before diagonalising (see
            `hessian_to_frequencies`). Default False for Gönnheimer parity.
        freq_scale: Multiplicative correction applied to all frequencies before
            C_v. Gönnheimer et al. use 1.181 to undo MACE-MP-0 softening.
        threshold: Minimum frequency (cm⁻¹) contributing to C_v.
        overwrite: Permission to destroy `hessian`; see `hessian_to_frequencies`.

    Returns:
        (frequencies_cm1, cv): the `(3N,)` frequencies and a dict mapping
        T → C_v in J/(g·K).
    """
    freqs = hessian_to_frequencies(
        hessian, masses, enforce_asr=enforce_asr, overwrite=overwrite
    )
    cv = cv_curve(
        freqs,
        temperatures,
        float(np.sum(masses)),
        freq_scale=freq_scale,
        threshold=threshold,
    )
    return freqs, cv


def hessian_to_frequencies(hessian, masses, *, enforce_asr=False, overwrite=False):
    """Mass-weighted eigendecomposition → phonon frequencies in cm⁻¹.

    Symmetrises the Hessian, mass-weights it to the dynamical matrix
    W_ij = H_ij / √(mᵢ mⱼ), diagonalises, and converts the eV/Å²/amu
    eigenvalues ω² to cm⁻¹. The mass-weighting and unit conversion are
    `ase.vibrations.VibrationsData._energies_and_modes` (ase/vibrations/data.py)
    verbatim; ASE returns the energies complex, and taking the real part below
    is the downstream choice (see Returns).

    Args:
        hessian: (3N, 3N) Cartesian Hessian in eV/Å².
        masses: (N,) atomic masses in amu.
        enforce_asr: If True, impose the force-constant acoustic sum rule on the
            symmetrised Hessian first (`_acoustic_sum_rule`). Default False to
            match Gönnheimer et al., who diagonalise the raw Hessian.
        overwrite: Permission to destroy `hessian`, so the steps run in place
            rather than allocating a `(3N, 3N)` copy each. Needs a C-contiguous
            float64 array; anything else is copied once, as if False. Results
            are identical either way.

    Returns:
        (3N,) frequencies in cm⁻¹. Imaginary modes (negative ω²) collapse to 0,
        the Gönnheimer convention (`np.real` of a purely imaginary energy) — not
        ASE's: `VibrationsData` keeps them complex, its `Phonons` class
        (phonons.py) reports them as negative.
    """
    h_arr = np.asarray(hessian, dtype=float)
    # Ours to destroy if the caller said so, or if asarray already copied.
    mutable = (overwrite or h_arr is not hessian) and h_arr.flags.c_contiguous
    if mutable:
        w = h_arr
        _symmetrise_inplace(w)
        if enforce_asr:
            _acoustic_sum_rule_inplace(w, len(masses))
    else:
        w = (h_arr + h_arr.T) / 2
        if enforce_asr:
            w = _acoustic_sum_rule(w, len(masses))
    # √m repeated per Cartesian triple → [m₀, m₀, m₀, m₁, …]: the x₁y₁z₁x₂…
    # ordering of the (N, 3) row-major flatten the sparsity pattern also uses.
    inv_sqrt_m = np.repeat(np.asarray(masses, dtype=float) ** -0.5, 3)
    w *= inv_sqrt_m[:, None]
    w *= inv_sqrt_m[None, :]
    omega2 = np.linalg.eigvalsh(w)
    # ASE's eV/Å²/amu → eV factor. The complex sqrt sends negative ω² to a
    # purely imaginary energy whose real part is 0, so imaginary modes read as
    # 0 (and are then dropped by the C_v threshold).
    conv = units._hbar * units.m / np.sqrt(units._e * units._amu)
    energies = conv * omega2.astype(complex) ** 0.5
    return np.real(energies) / units.invcm


def cv_curve(
    frequencies_cm1,
    temperatures,
    total_mass_amu,
    *,
    freq_scale=1.0,
    threshold=FREQ_THRESHOLD_CM1,
):
    """Gravimetric C_v(T) curve from phonon frequencies.

    Args:
        frequencies_cm1: (3N,) phonon frequencies in cm⁻¹.
        temperatures: Iterable of temperatures in K.
        total_mass_amu: Total system mass in amu (Σ masses); divides the molar
            C_v to gravimetric units.
        freq_scale: Multiplicative frequency correction (see `cv_from_hessian`).
        threshold: Minimum frequency (cm⁻¹) to include.

    Returns:
        dict mapping T → C_v in J/(g·K).
    """
    scaled = np.asarray(frequencies_cm1, dtype=float) * freq_scale
    return {
        T: heat_capacity(scaled, T, threshold=threshold)[0] / total_mass_amu
        for T in temperatures
    }


def heat_capacity(frequencies_cm1, T, *, threshold=FREQ_THRESHOLD_CM1):
    """Harmonic (independent-oscillator) C_v at temperature T.

    Each mode above `threshold` contributes one quantum-harmonic-oscillator
    term with x = hν/(k T); modes below it (acoustic / imaginary) are skipped.

    Args:
        frequencies_cm1: (3N,) phonon frequencies in cm⁻¹.
        T: Temperature in K.
        threshold: Minimum frequency (cm⁻¹) to include.

    Returns:
        (C_v, n_skipped): C_v in J/(mol·K) and the count of skipped modes.
    """
    nu = np.asarray(frequencies_cm1, dtype=float)
    keep = nu >= threshold
    # ν[cm⁻¹] · 100 · c[m/s] = ν in Hz (the 100 converts cm⁻¹ → m⁻¹).
    x = (h * c * 100.0 / (k * T)) * nu[keep]
    # QHO C_v per mode, written as k·(½x / sinh(½x))² — analytically identical
    # to k·x²eˣ/(eˣ−1)² but stable as modes freeze out (x → ∞ ⇒ term → 0,
    # whereas the eˣ/(eˣ−1)² form overflows to NaN).
    cv = k * (0.5 * x / np.sinh(0.5 * x)) ** 2
    return float(cv.sum() * N_A), int((~keep).sum())


# ----
# Acoustic sum rule (force-constant form)
# ----


def _symmetrise_inplace(h, block=4096):
    """`h = (h + h.T) / 2` without a second `(3N, 3N)` buffer.

    Blockwise so the temporaries are `block**2`, not `n**2`; the per-element
    arithmetic is the same, so the result is bit-identical.
    """
    n = h.shape[0]
    for a in range(0, n, block):
        for b in range(a, n, block):
            upper, lower = h[a : a + block, b : b + block], h[b : b + block, a : a + block]
            s = (upper + lower.T) / 2
            upper[...], lower[...] = s, s.T


def _acoustic_sum_rule(hessian, n_atoms):
    """Impose the force-constant acoustic sum rule on a (3N, 3N) Hessian.

    A uniform translation must cost no energy, so the 3×3 force-constant blocks
    in each row must sum to zero: Σ_j Φ_ij,ab = 0. This subtracts the per-row
    mean over the partner-atom axis to enforce that, making a uniform
    translation an exact zero mode. It is phonopy's "type 1"
    `set_translational_invariance` (phonopy/harmonic/force_constants.py) —
    a uniform subtraction of the row-sum error — applied here to the assembled
    Hessian on both atom axes (the partner-atom sum, the one that matters, is
    exact as it is applied last).

    Args:
        hessian: (3N, 3N) symmetric Cartesian Hessian.
        n_atoms: Number of atoms N.

    Returns:
        (3N, 3N) Hessian with the translational (row-sum) violation removed.
    """
    out = hessian.copy()
    _acoustic_sum_rule_inplace(out, n_atoms)
    return out


def _acoustic_sum_rule_inplace(hessian, n_atoms):
    """`_acoustic_sum_rule` writing into `hessian`, which must be C-contiguous
    (the reshape has to alias it). The means are `(3, 3N)`-sized, so cheap."""
    assert hessian.flags.c_contiguous, "reshape would copy, losing the writes"
    fc = hessian.reshape(n_atoms, 3, n_atoms, 3)  # [i, a, j, b]
    fc -= fc.mean(axis=0, keepdims=True)  # Σ over first atom i → 0
    fc -= fc.mean(axis=2, keepdims=True)  # Σ over partner atom j → 0 (exact)
