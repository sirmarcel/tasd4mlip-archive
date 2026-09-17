# moosavi-exp

The 8 experimentally-measured MOFs assembled by Moosavi et al. (Nat. Mater. 2022) to validate their ML heat-capacity predictor against calorimetry, with crystal structures and full experimental Cp(T) curves. This is the primary source of experimental Cp reference data for the MOF benchmarks in this work.

**Paper.** Moosavi et al., *A data-science approach to predict the heat capacity of nanoporous materials*, Nature Materials 21, 1419–1425 (2022). DOI [10.1038/s41563-022-01374-3](https://doi.org/10.1038/s41563-022-01374-3).

## Inputs

`../../raw/` is not included in this archive. The input is public and must be fetched once before rebuilding.

- `../../raw/moosavi-2022/Figures/Fig4/Fig_exp/cv_exp/`: from `Figures.zip` of the Materials Cloud `p1-2y` archive, [DOI:10.24435/materialscloud:p1-2y](https://doi.org/10.24435/materialscloud:p1-2y), CC-BY-4.0 (see Build): 8 experimental CIFs in `cifs/`, the `cv_names.csv` index (id, filename, method, density, Cp@300K), and `experimental_data.csv` with the full Cp(T) curves.

The CIFs are *not* DFT-optimized. They are the original crystallographic structures from the primary literature (e.g. Park et al. PNAS 2006 for ZIF-8), redistributed unchanged by Moosavi et al. Each CIF filename embeds its own publication lineage and is preserved in `source_cif`.

## Outputs

- `structures.xyz`: 8 extended-XYZ frames, sorted by atom count.
- `experimental_curves.npz`: full experimental Cp(T) curves, one `(2, N)` float64 array per MOF keyed by identifier. Row 0 is temperature in **Kelvin**, row 1 is gravimetric Cp in J/(g·K).

Per-frame `info`:

| Key | Source | Description |
|---|---|---|
| `identifier` | cv_names.csv | `mof1`..`mof8`, the nickname used throughout the Moosavi share |
| `common_name` | Moosavi paper | clean chemical name, e.g. `ZIF-8`, `Zn2(D-cam)2(bpy)` |
| `source_cif` | cv_names.csv | original CIF filename, which carries the crystallographic-paper provenance |
| `exp_method` | cv_names.csv | `DSC` (Moosavi's own measurement) or `Lit` (curve compiled from primary literature by Moosavi et al.) |
| `density_g_cm3` | cv_names.csv | crystallographic density in g/cm³ |
| `exp_cp_300` | cv_names.csv | scalar gravimetric Cp at 300 K in J/(g·K), the value behind Moosavi's Fig. 4 parity plots |
| `n_atoms` | CIF | primitive-cell atom count |
| `elements` | CIF | sorted, dash-joined chemical symbols |

## Composition

```
mof4 Co2(S-mal)2(bpy)        N=96   DSC
mof3 Co2(L-asp)2(bpe)        N=108  DSC
mof6 Cu(INA)2                N=108  DSC
mof5 Co3(btc)(DMF)3(HCOO)3   N=138  Lit
mof1 Ca(ndc)(DMF)            N=140  Lit
mof2 Zn2(D-cam)2(bpy)        N=156  DSC
mof8 ZIF-8                   N=276  DSC
mof7 Mn3(ndc)3(DMF)4         N=468  Lit
```

`DSC` values are original Moosavi et al. measurements (DSC, 20–200 °C, sapphire standard, argon atmosphere). `Lit` curves were compiled by Moosavi et al. from primary-literature calorimetry. The per-entry primary references are in the Moosavi SI and are not transcribed here.

Upstream `cv_names.csv` lists 11 experimental MOFs, but only `mof1` to `mof8` have both a CIF and a full curve in the deposit. The other three (MOF-74-Co, MOF-74-Zn, MOF-5) lack a crystal structure upstream and are not included here.

## Temperature-unit gotcha

The two halves of the upstream `experimental_data.csv` store temperature in different units, undocumented upstream: DSC entries in **degrees Celsius** (20–200 °C, 10,800 points each), Lit entries in **Kelvin** (about 198–550 K, 45–71 points). The build script converts DSC curves to Kelvin using the `exp_method` column, so `experimental_curves.npz` is uniformly Kelvin. Going back to the raw CSV means converting again.

Cross-checking `exp_cp_300` against the curves at 300 K agrees to about 0.5% for every entry, with `exp_cp_300` systematically 0.5 to 1% low. That is consistent with Moosavi's parity plot using a slightly different reference temperature or a windowed average.

## Build

Run from this directory. The `cv_exp` subtree must be extracted from `Figures.zip` once (the rest of that 393 MiB zip is not needed):

```bash
( cd ../../raw/moosavi-2022 && unzip -q Figures.zip 'Figures/Fig4/Fig_exp/cv_exp/*' )

uv run --with ase --with pandas --with numpy python build_structures.py
```
