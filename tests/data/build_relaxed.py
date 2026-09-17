"""Generate relaxed test fixtures for the end-to-end C_v test.

Relaxes the reference structures with the canonical Gönnheimer protocol
(MACE-MP-0 medium + DFT-D3(BJ, PBE), BFGS over FrechetCellFilter,
fmax = 5×10⁻³ eV/Å, fp64 — see `work/relax-goennheimer/README.md`) and writes
one extended-XYZ per structure to `tests/data/relaxed/`, `info` (identifier +
Gönnheimer/DFT C_v references) carried over from the source frames.

Unlike `build_reference.py`, this uses the sadmof stack itself — the fixtures
feed grounding tests whose reference values (Gönnheimer's published spectra
and C_v) are independent of our code, so the relaxation is not circular.

Structures are read by identifier from the tracked dataset
`sources/processed/goennheimer+moosavi/structures.xyz`; the converted MACE
checkpoint `sources/processed/mace-mp-0-medium/` must be built.

Run (from the repo root):

    uv run python tests/data/build_relaxed.py
"""

from pathlib import Path

from ase.io import read, write

from sadmof.models import get_calculator
from sadmof.relax import relax

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
STRUCTURES_XYZ = (
    REPO_ROOT / "sources" / "processed" / "goennheimer+moosavi" / "structures.xyz"
)
CHECKPOINT = REPO_ROOT / "sources" / "processed" / "mace-mp-0-medium"
OUT_DIR = HERE / "relaxed"

# The same spread of size and metal chemistry as build_reference.py.
IDENTIFIERS = ["RSM0004", "RSM0010", "RSM0020", "RSM0304"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_id = {f.info.get("identifier"): f for f in read(str(STRUCTURES_XYZ), index=":")}
    calc = get_calculator("mace-mp0+d3", checkpoint=CHECKPOINT, dtype="float64")

    for ident in IDENTIFIERS:
        atoms = by_id[ident].copy()
        result = relax(
            atoms, calc, fmax=0.005, max_steps=25000, optimizer="bfgs", logfile=None
        )
        assert result["converged"], f"{ident} did not converge: {result}"
        atoms.calc = None
        path = OUT_DIR / f"{ident}.xyz"
        write(str(path), atoms)
        print(
            f"{ident}: N={len(atoms):3d}  steps={result['n_steps']:4d}  "
            f"fmax={result['fmax_final']:.6f}  ({result['wall_s']:.0f} s)  -> {path.name}"
        )


if __name__ == "__main__":
    main()
