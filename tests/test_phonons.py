"""Tests for the phonon / heat-capacity observables (sadmof.observables.phonons).

These pin the physics that matters for C_v: the QHO limits, the cm⁻¹
conversion, and the force-constant sum rule.
"""

import numpy as np

import pytest
from ase import units
from scipy.constants import N_A, c, h, k

from sadmof.observables import (
    cv_curve,
    cv_from_hessian,
    heat_capacity,
    hessian_to_frequencies,
)
from sadmof.observables.phonons import _acoustic_sum_rule

R = N_A * k  # molar gas constant, J/(mol·K)

# ASE's eV/Å²/amu eigenvalue → cm⁻¹ conversion (mirrors hessian_to_frequencies).
CONV = units._hbar * units.m / np.sqrt(units._e * units._amu)


# --- heat_capacity --------------------------------------------------------


def test_heat_capacity_classical_limit():
    # T ≫ hν/k: every mode → Dulong–Petit (k per mode), so M modes → M·R.
    cv, n_skipped = heat_capacity(np.full(5, 50.0), T=1e7)
    assert n_skipped == 0
    assert cv == pytest.approx(5 * R, rel=1e-4)


def test_heat_capacity_freezes_out_at_low_T():
    # T ≪ hν/k: mode frozen, C_v → 0 (and no overflow NaN, unlike eˣ/(eˣ−1)²).
    cv, _ = heat_capacity(np.array([3000.0]), T=30.0)
    assert cv == pytest.approx(0.0, abs=1e-9)


def test_heat_capacity_skips_subthreshold_modes():
    freqs = np.array([-10.0, 0.0, 1e-6, 100.0])
    cv, n_skipped = heat_capacity(freqs, T=300.0)
    assert n_skipped == 3
    assert cv == pytest.approx(heat_capacity(np.array([100.0]), 300.0)[0])


def test_heat_capacity_matches_closed_form():
    nu, T = 100.0, 300.0
    x = h * c * 100.0 * nu / (k * T)
    expected = R * x**2 * np.exp(x) / (np.exp(x) - 1) ** 2
    cv, _ = heat_capacity(np.array([nu]), T)
    assert cv == pytest.approx(expected, rel=1e-10)


# --- hessian_to_frequencies ----------------------------------------------


def test_frequencies_scalar_hessian():
    # H = α·I with unit masses ⇒ every ω² = α ⇒ all frequencies equal & > 0.
    alpha = 3.0
    freqs = hessian_to_frequencies(alpha * np.eye(12), np.ones(4))
    assert np.allclose(freqs, CONV * np.sqrt(alpha) / units.invcm)


def test_negative_eigenvalues_collapse_to_zero():
    freqs = hessian_to_frequencies(-2.0 * np.eye(3), np.ones(1))
    assert np.allclose(freqs, 0.0)


# --- acoustic sum rule (force-constant form) ------------------------------


def test_acoustic_sum_rule_zeroes_partner_atom_block_sum():
    # After enforcement, each atom's 3×3 force-constant blocks summed over the
    # partner atom must vanish (Σ_j Φ_ij = 0) — uniform translation costs no
    # force. This is the condition applied last, so it is exact.
    rng = np.random.default_rng(0)
    n = 5
    a = rng.standard_normal((3 * n, 3 * n))
    h_c = _acoustic_sum_rule((a + a.T) / 2, n)
    block_row_sum = h_c.reshape(n, 3, n, 3).sum(axis=2)
    assert np.allclose(block_row_sum, 0.0, atol=1e-12)


def test_asr_suppresses_spurious_acoustic_modes():
    # Ring of n atoms with nearest-neighbour springs: every 3×3 block-row sums
    # to zero, so translations are exact zero modes. A symmetric diagonal bump
    # breaks that and lifts the three acoustic modes; the ASR pushes them back.
    n = 6
    lap = 2 * np.eye(n) - np.roll(np.eye(n), 1, 0) - np.roll(np.eye(n), -1, 0)
    h = np.kron(lap, np.eye(3))
    h[0, 0] += 0.3  # diagonal ⇒ stays symmetric, but breaks the row sum
    masses = np.ones(n)
    acoustic_raw = np.sort(np.abs(hessian_to_frequencies(h, masses)))[:3]
    acoustic_asr = np.sort(np.abs(hessian_to_frequencies(h, masses, enforce_asr=True)))[:3]
    assert acoustic_raw.max() > 1.0  # contamination is real (cm⁻¹)
    assert acoustic_asr.max() < 0.1 * acoustic_raw.max()


# --- cv_curve / cv_from_hessian ------------------------------------------


def test_cv_curve_is_gravimetric():
    freqs = np.array([100.0, 200.0, 300.0])
    light = cv_curve(freqs, [300.0], total_mass_amu=10.0)
    heavy = cv_curve(freqs, [300.0], total_mass_amu=20.0)
    assert light[300.0] == pytest.approx(2 * heavy[300.0])


def test_cv_from_hessian_roundtrip():
    H = 5.0 * np.eye(12)
    masses = np.full(4, 12.0)
    freqs, cv = cv_from_hessian(H, masses, [200.0, 300.0])
    assert len(freqs) == 12
    assert set(cv) == {200.0, 300.0}
    assert all(v > 0 for v in cv.values())
    assert cv[300.0] > cv[200.0]  # C_v rises with T below saturation


def test_overwrite_matches_copying_path_bitwise():
    rng = np.random.default_rng(0)
    n_atoms = 40
    masses = rng.uniform(1.0, 60.0, n_atoms)
    h = rng.normal(size=(3 * n_atoms, 3 * n_atoms))
    h = h + h.T * 0.3  # deliberately asymmetric: symmetrisation must matter

    for enforce_asr in (False, True):
        reference = hessian_to_frequencies(h.copy(), masses, enforce_asr=enforce_asr)
        scratch = h.copy()
        lean = hessian_to_frequencies(
            scratch, masses, enforce_asr=enforce_asr, overwrite=True
        )
        assert np.array_equal(reference, lean)


def test_overwrite_false_leaves_input_untouched():
    rng = np.random.default_rng(1)
    masses = rng.uniform(1.0, 60.0, 12)
    h = rng.normal(size=(36, 36))
    before = h.copy()
    hessian_to_frequencies(h, masses)
    assert np.array_equal(h, before)


def test_symmetrise_inplace_matches_expression():
    from sadmof.observables.phonons import _symmetrise_inplace

    rng = np.random.default_rng(2)
    for n, block in ((37, 8), (64, 64), (13, 4)):
        h = rng.normal(size=(n, n))
        expected = (h + h.T) / 2
        got = h.copy()
        _symmetrise_inplace(got, block=block)
        assert np.array_equal(got, expected)
