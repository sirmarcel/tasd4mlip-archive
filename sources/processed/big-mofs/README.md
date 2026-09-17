# big-mofs

Four large benchmark MOFs that are **not** in the Moosavi experimental set (`../moosavi-exp/`): one extended-XYZ frame per MOF, plus full experimental Cp(T) curves transcribed from the primary calorimetry literature where such measurements exist (MOF-177, MIL-101). MOF-210 and MIL-100 are structure-only, no calorimetry data was located, and they serve as scale benchmarks.

| id | MOF | Primitive N | Cp method | Reference |
|---|---|---:|---|---|
| `mof177` | MOF-177 (Zn₄O(BTB)₂) | 808 | Calvet calorimeter | Kloutse et al. 2015 |
| `mof210` | MOF-210 (Zn₄O(BTE)₄⁄₃(BPDC)) | 1,854 | none | none |
| `mil100` | MIL-100(Cr) (Cr₃F(H₂O)₂O(BTC)₂) | 2,788 | none | none |
| `mil101` | MIL-101(Cr) (Cr₃F(H₂O)₂O(BDC)₃) | 3,604 | Temperature-modulated DSC | Liu et al. 2017 |

Unlike `../moosavi-exp/`, where the reference values are machine-readable from the Moosavi share, the reference values here are **our own transcription from the primary papers**. There is no machine-readable upstream. The numbers are hardcoded in `build_structures.py` with citations inline and were verified against the source PDFs.

## Inputs

`../../raw/` is not included in this archive. The four CIFs come from the RASPA2 CIF library, upstream path `structures/mofs/cif/` in [numat/RASPA2](https://github.com/numat/RASPA2) at commit `e9683341af9ecc65f6812bbbcfb25ee51ee0a177`, and must be downloaded into `../../raw/raspa2/` before rebuilding. They ship upstream as-is as primitive cells, none is DFT-optimized. Expected sha256:

| File | sha256 |
|---|---|
| `MOF-177.cif` | `c4a696f2bde8d29dd5fc3d286a86c5fda44e2e601f8d3700b9528e1dfb5e3cc5` |
| `MOF-210-primitive.cif` | `4d082830d8d3f1c447f61e83b1c383deed8de67e7f78358ee6a245d8af28e008` |
| `MIL-100-primitive.cif` | `00debae760c2c5d3b81a30fe8769038f0fd731904cf18a72d52f8e5d4d4bb80a` |
| `MIL-101-primitive.cif` | `c899b6ded7a48d3fccfaaff6f99505a5f43f72a24e08f36b6c6eeff5b8cf9186` |

## Outputs

- `structures.xyz`: 4 extended-XYZ frames, sorted by atom count.
- `experimental_curves.npz`: one `(2, N)` float64 array per Cp-bearing MOF (`mof177`, `mil101`) keyed by identifier. Row 0 is temperature in **Kelvin**, row 1 is gravimetric Cp in J/(g·K).

Per-frame `info`:

| Key | Description |
|---|---|
| `identifier` | `mof177`, `mof210`, `mil100`, or `mil101` |
| `common_name` | e.g. `MIL-101(Cr)` |
| `chemical_formula` | formula unit, used for molar-mass conversion where Cp data exists |
| `source_cif` / `source_cif_note` | upstream CIF filename and provenance string |
| `exp_method` | `TMDSC` (MIL-101) or `Calvet` (MOF-177). The `exp_*` keys are absent on the structure-only entries |
| `exp_paper_bibkey` / `exp_paper_citation` | `lxzz17` / `kzcc15` plus compact citation |
| `exp_sample` | activation and handling protocol |
| `exp_cp_300` | gravimetric Cp at 300 K in J/(g·K) |
| `exp_cp_298_15` | MIL-101 only, the paper's in-text scalar at 298.15 K |
| `exp_cp_uncertainty_pct` | authors' stated uncertainty, percent of Cp |
| `exp_curve_range_K` / `exp_curve_source` | range and reconstruction of the stored curve |
| `n_atoms`, `elements` | primitive-cell atom count, sorted dash-joined symbols |

## Reference-value provenance

**MOF-177, Kloutse et al. 2015.** Kloutse, Zacharia, Cossement, Chahine, *Microporous Mesoporous Mater.* 217 (2015) 1–5, DOI [10.1016/j.micromeso.2015.05.047](https://doi.org/10.1016/j.micromeso.2015.05.047). Calvet calorimeter (SETARAM BT 2.15), 80–320 K, NIST sapphire and Cu calibration, in-house synthesis, argon glovebox handling, about 1% propagated uncertainty. The paper tabulates no raw data, only a 5th-order polynomial fit (their Eq. 5, gravimetric). The stored curve is that polynomial sampled on a 5 K grid over its valid 80–320 K range, giving **Cp(300 K) = 0.7505 J/(g·K)**. Use above 320 K is extrapolation. This fit has elsewhere been attributed to "Song et al. 2017", it is original to Kloutse et al.

**MIL-101(Cr), Liu et al. 2017.** Liu, Xu, Liu, Zhou, Zhao, *J. Therm. Anal. Calorim.* 129 (2017) 509–514, DOI [10.1007/s10973-017-6168-9](https://doi.org/10.1007/s10973-017-6168-9). TMDSC (TA Q1000), 253–413 K, sapphire calibration (±2.2%), sample activated at 220 °C in vacuum and kept dehydrated, so the data correspond to the **dehydrated framework** Cr₃F(H₂O)₂O(BDC)₃ (M = 719.39 g/mol, the two Cr-coordinated waters retained). The curve is the paper's Table 2 transcribed verbatim (34 molar Cp points, converted to gravimetric). Two scalars are exposed to avoid silent off-by-2-K comparisons:

- `exp_cp_298_15` = 0.5226 J/(g·K), the paper's in-text Cp(298.15 K) = 375.92 J/mol/K.
- `exp_cp_300` = 0.5297 J/(g·K), linear interpolation between the 298.15 and 303 K entries. Use this against model predictions at 300 K.

## Build

Run from this directory, with the four CIFs in place under `../../raw/raspa2/`:

```bash
uv run --with ase --with numpy python build_structures.py
```
