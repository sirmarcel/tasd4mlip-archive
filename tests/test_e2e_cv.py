"""End-to-end MACE+D3 → C_v pipeline, grounded against Gönnheimer et al. (JCTC 2025).

Three layers of grounding against the tracked reference set
(`sources/processed/goennheimer+moosavi/`):

1. Observables only: `cv_curve` on Gönnheimer's published phonon spectra must
   reproduce their published C_v values. This pins our conventions (raw
   Hessian, imaginary→0, 1e-3 cm⁻¹ threshold, gravimetric units, no frequency
   scaling in the CSV values) to machine precision — a full sweep over all 233
   structures × 4 temperatures agreed to <1e-14 J/(g·K) (2026-06-11).

2. Hessian → C_v on pre-relaxed fixtures (`tests/data/relaxed/`, built by
   `build_relaxed.py` with the canonical Gönnheimer relaxation protocol):
   dense AD MACE Hessian + dense AD D3 Hessian → frequencies → C_v, compared
   to Gönnheimer's published spectra and C_v per structure. These cells are
   small (13–58 atoms), so the Hessian is effectively fully coupled and the
   dense path is the right tool; the sparse path has its own equivalence
   tests (`test_hessian.py`, `test_dense_hessian.py`).

3. Full pipeline on RSM0004 (13 atoms): the canonical relaxation (MACE-MP-0
   medium + D3(BJ, PBE), BFGS over FrechetCellFilter, fmax 5×10⁻³ eV/Å, fp64)
   from the unrelaxed source frame, then layer 2's Hessian → C_v.

Gönnheimer's Hessians are AD like ours (torch-autograd `get_hessian` on the
upstream MACE implementation) except for the D3 term, which they
central-difference over the dftd3-binary forces (ASE `DFTD3`, δ=1e-4) where we
differentiate the JAX D3 port. Those implementation differences plus slightly
different relaxed geometries make layers 2–3 loose-tolerance. Observed
deviations across the four fixtures (2026-06-11): max |Δν| 0.33–1.88 cm⁻¹,
max |ΔC_v| 8e-6–5.1e-5 J/(g·K); thresholds are set ~2.5–4× above the worst case.
"""

import numpy as np
import jax

from pathlib import Path

import pytest
from ase.io import read

from sadmof.paths import SOURCES, checkpoint

# Hessians/phonons are done in fp64 (see test_hessian.py).
jax.config.update("jax_enable_x64", True)

DATASET = SOURCES / "goennheimer+moosavi"
CHECKPOINT = checkpoint("mace-mp-0-medium")
RELAXED_DIR = Path(__file__).resolve().parent / "data" / "relaxed"

# MACE-MP-0 medium graph cutoff (Å) — matches the converted checkpoint.
MACE_CUTOFF = 6.0

TEMPERATURES = [250.0, 300.0, 350.0, 400.0]

# The same reference-set structures as build_reference.py / build_relaxed.py.
IDENTIFIERS = ["RSM0004", "RSM0010", "RSM0020", "RSM0304"]

pytestmark = pytest.mark.skipif(
    not (DATASET / "structures.xyz").exists(),
    reason=f"dataset not built at {DATASET}",
)


@pytest.fixture(scope="module")
def goenn_frequencies():
    return np.load(DATASET / "frequencies_uc.npz")


@pytest.fixture(scope="module")
def mace():
    """The raw MACE model + params (for Hessians; the calculator is separate)."""
    if not (CHECKPOINT / "model.msgpack").exists():
        pytest.skip(f"checkpoint not built at {CHECKPOINT}")
    from marathon.io import from_dict, read_msgpack, read_yaml

    model = from_dict(read_yaml(str(CHECKPOINT / "model.yaml")))
    params = read_msgpack(str(CHECKPOINT / "model.msgpack"))
    return model, params


def goenn_cv(atoms, T):
    return float(atoms.info[f"goenn_uc_cv_{int(T)}"])


def mace_d3_frequencies_cv(atoms, mace_model, mace_params):
    """Dense AD MACE + D3 Hessians on `atoms` → (frequencies, C_v curve)."""
    from sadmof.dense import get_dense_hessian_fn
    from sadmof.models.d3 import D3
    from sadmof.models.d3.inputs import atoms_to_inputs as d3_atoms_to_inputs
    from sadmof.models.mace import atoms_to_inputs as mace_atoms_to_inputs
    from sadmof.observables import cv_from_hessian

    n = len(atoms)

    pos, cell, graph = mace_atoms_to_inputs(
        atoms, cutoff=MACE_CUTOFF, float_dtype=np.float64
    )
    h_mace = np.asarray(
        jax.jit(get_dense_hessian_fn(mace_model.energy, chunk_size=32))(
            mace_params, pos, cell, graph
        )[0]
    )

    d3 = D3()
    d3_pos, d3_cell, d3_graph = d3_atoms_to_inputs(atoms, cutoff=d3.cutoff, cnthr=d3.cnthr)
    h_d3 = np.asarray(
        jax.jit(get_dense_hessian_fn(d3.energy, chunk_size=32))(
            D3.load_params(), d3_pos, d3_cell, d3_graph
        )[0]
    )

    # Slice both to the real-atom block (padding differs between the two graphs).
    hessian = (h_mace[:n, :, :n, :] + h_d3[:n, :, :n, :]).reshape(3 * n, 3 * n)
    return cv_from_hessian(hessian, atoms.get_masses(), TEMPERATURES)


def assert_grounded(atoms, freqs, cv, ref_freqs):
    """Frequencies + C_v against Gönnheimer's published values."""
    ours = np.sort(freqs)
    theirs = np.sort(ref_freqs)
    assert ours.shape == theirs.shape
    # 5 cm⁻¹ on a ~0–3000 cm⁻¹ spectrum is ~2e-3 relative.
    assert np.abs(ours - theirs).max() < 5.0
    for T in TEMPERATURES:
        # 2e-4 J/(g·K) is ~0.04% relative.
        assert cv[T] == pytest.approx(goenn_cv(atoms, T), abs=2e-4)


# --- layer 1: observables vs Gönnheimer's published spectra + C_v -----------


@pytest.mark.parametrize("identifier", IDENTIFIERS)
def test_cv_curve_reproduces_goennheimer(goenn_frequencies, identifier):
    from sadmof.observables import cv_curve

    by_id = {
        f.info["identifier"]: f for f in read(str(DATASET / "structures.xyz"), index=":")
    }
    atoms = by_id[identifier]
    cv = cv_curve(
        goenn_frequencies[identifier], TEMPERATURES, float(atoms.get_masses().sum())
    )
    for T in TEMPERATURES:
        assert cv[T] == pytest.approx(goenn_cv(atoms, T), abs=1e-12)


# --- layer 2: Hessian → C_v on pre-relaxed fixtures --------------------------


@pytest.mark.parametrize("identifier", IDENTIFIERS)
def test_hessian_cv_grounds_against_goennheimer(goenn_frequencies, mace, identifier):
    path = RELAXED_DIR / f"{identifier}.xyz"
    if not path.exists():
        pytest.skip(f"relaxed fixture not built at {path}")
    atoms = read(str(path))

    model, params = mace
    freqs, cv = mace_d3_frequencies_cv(atoms, model, params)
    assert_grounded(atoms, freqs, cv, goenn_frequencies[identifier])


# --- layer 3: full pipeline (relax from scratch) on RSM0004 ------------------


def test_e2e_relax_hessian_cv(goenn_frequencies, mace):
    from sadmof.models import get_calculator
    from sadmof.relax import relax

    by_id = {
        f.info["identifier"]: f for f in read(str(DATASET / "structures.xyz"), index=":")
    }
    atoms = by_id["RSM0004"].copy()

    calc = get_calculator("mace-mp0+d3", checkpoint=CHECKPOINT, dtype="float64")
    result = relax(atoms, calc, fmax=0.005, max_steps=25000, optimizer="bfgs", logfile=None)
    assert result["converged"]
    assert result["fmax_final"] <= 0.005

    model, params = mace
    freqs, cv = mace_d3_frequencies_cv(atoms, model, params)
    assert_grounded(atoms, freqs, cv, goenn_frequencies["RSM0004"])
