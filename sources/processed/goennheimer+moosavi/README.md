# goennheimer+moosavi

The 233-structure porous-material reference set used in Gönnheimer, Reuter & Margraf (JCTC 2025), merged with the matching Moosavi (Nat. Mater. 2022) CIFs and DFT $C_V$ values into a single extended-XYZ file plus the primitive-cell phonon spectra.

## Inputs

`../../raw/` is not included in this archive. Both inputs are public and must be fetched once before rebuilding.

- `../../raw/moosavi-2022/`: the Materials Cloud `p1-2y` archive, [DOI:10.24435/materialscloud:p1-2y](https://doi.org/10.24435/materialscloud:p1-2y), record `7p3z3-t3q91`, CC-BY-4.0. Only `DFT_calculations.zip` and `database_heat_capacity.zip` are used here, unzipped in place (see Build).
- `../../raw/goennheimer-2025/`: a clone of [Nilsgoe/AD_heat_capacity](https://github.com/Nilsgoe/AD_heat_capacity) at commit `e96845e` (2025-01-22).

## Outputs

- `structures.xyz`: 233 extended-XYZ frames, sorted by atom count.
- `frequencies_uc.npz`: primitive-cell phonon spectra (1D float64, cm⁻¹), keyed by identifier. 233 entries.

Per-frame `info`:

| Key | Source | Description |
|---|---|---|
| `identifier` | CIF filename | e.g. `RSM0004` |
| `structure_type` | Moosavi DFT CSV | `MOF`, `COF` or `zeolite` (Moosavi's `zeo` becomes `zeolite`), `unknown` if absent upstream |
| `n_atoms` | CIF | primitive-cell atom count |
| `elements` | CIF | sorted, dash-joined chemical symbols |
| `dft_cv_{T}` | Moosavi DFT CSV | T in {250, 275, 300, 325, 350, 375, 400} K, gravimetric J/(g·K) |
| `goenn_uc_cv_{T}` | Gönnheimer UC CSV | T in {250, 260, …, 400} K, same units |
| `goenn_uc_converged` | Gönnheimer UC CSV (`Opt`) | bool |
| `goenn_sc_cv_{T}` | Gönnheimer 2×2×2 CSV | same temperatures, NaN if upstream is empty |
| `goenn_sc_converged` | Gönnheimer 2×2×2 CSV (`Opt`) | bool |

Gönnheimer pipeline, for reference: BFGS and FrechetCellFilter relaxation to fmax = 5×10⁻³ eV/Å, AD MACE Hessian plus numerical D3 Hessian, mass-weighted eigendecomposition, modes with |ν| < 10⁻³ cm⁻¹ discarded, Einstein-model $C_V$.

## Known upstream gaps

- `RSM1072`: no row in `DFT_cp_allstructures.csv`, so `structure_type=unknown` and the `dft_cv_*` keys are omitted.
- `RSM1854`: the 2×2×2 Hessian ran out of memory upstream, so `goenn_sc_cv_*` are NaN and `goenn_sc_converged=False`.

## Build

Run from this directory. The two Moosavi zips must be unzipped in place once:

```bash
( cd ../../raw/moosavi-2022 && unzip -q DFT_calculations.zip && unzip -q database_heat_capacity.zip )

uv run --with ase --with pandas --with numpy python build_structures.py
uv run --with ase --with pandas --with numpy python build_frequencies.py
```
