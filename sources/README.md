# sources/

External inputs the code depends on: datasets, reference structures, model checkpoints. Everything under here originates outside this project.

## Layout

```
sources/
├── raw/         # Original artefacts as obtained (downloaded datasets, upstream files).
│                # NOT included in this archive. Every entry is re-obtainable from a
│                # pinned public source, listed below and in the per-folder READMEs.
└── processed/   # Derived material built from raw/: parsed datasets, merged reference
                 # data, converted checkpoints. Included, except the model weights.
```

Where both exist, the same name appears under `raw/` and `processed/`.

## Datasets

- `moosavi-2022/` *(raw only)*: supplementary archive for **Moosavi et al., "A data-science approach to predict the heat capacity of nanoporous materials", Nat. Mater. 21 (2022)**. Available in full from [Materials Cloud, DOI:10.24435/materialscloud:p1-2y](https://doi.org/10.24435/materialscloud:p1-2y), record `7p3z3-t3q91`. Six files (about 1.65 GiB): `README.txt`, `files_description.md`, `database_heat_capacity.zip` (8.6 MiB), `DFT_calculations.zip` (137 MiB), `Figures.zip` (393 MiB), `ML.zip` (1.1 GiB). CC-BY-4.0. Consumed by `processed/goennheimer+moosavi/` (DFT CIFs and reference $C_V$) and `processed/moosavi-exp/` (experimental CIFs and Cp curves).

- `goennheimer-2025/` *(raw only)*: clone of **[Nilsgoe/AD_heat_capacity](https://github.com/Nilsgoe/AD_heat_capacity)**, the code and data accompanying Gönnheimer et al. 2025 on AD-based heat capacity for MOFs and COFs. Snapshot at commit `e96845e` (2025-01-22), about 44 MiB. Consumed by `processed/goennheimer+moosavi/`.

- `raspa2/` *(raw only)*: four MOF CIFs from the CIF library shipped with [numat/RASPA2](https://github.com/numat/RASPA2), pinned to commit `e9683341af9ecc65f6812bbbcfb25ee51ee0a177` and sha256-verified. Consumed by `processed/big-mofs/`.

- `goennheimer+moosavi/` *(processed)*: the 233-structure Gönnheimer benchmark set merged with the matching Moosavi CIFs and DFT $C_V$ values into one extended-XYZ file plus the primitive-cell phonon spectra.

- `moosavi-exp/` *(processed)*: the 8 experimentally-measured MOFs Moosavi et al. used to validate against calorimetry: original crystallographic structures and full experimental Cp(T) curves, machine-readable from the Moosavi share.

- `big-mofs/` *(processed)*: MOF-177, MOF-210, MIL-100(Cr) and MIL-101(Cr), large benchmark MOFs outside the Moosavi set: RASPA2 structures plus experimental Cp(T) reference values transcribed by us from the primary calorimetry papers (Kloutse et al. 2015, Liu et al. 2017).

## Model checkpoints

**The three converted checkpoints are not included in this archive.** Each `model.msgpack` is a large binary derived from upstream weights that are publicly available, so each folder ships only its README, its config file and, for MACE, its converter. The README pins the upstream weights, gives the one-command conversion, and records the sha256 of the converted file the experiments ran with.

- `mace-mp-0-medium/`: MACE-MP-0 "medium" (L1) foundation model. Upstream PyTorch checkpoint from the `ACEsuit/mace-mp` release assets, marathon-style Flax port produced by `build.py` in the folder.
- `pet-mad-s/`, `pet-mad-xs/`: PET-MAD v1.5.0 checkpoints, "s" and "xs" variants. Upstream metatrain `.ckpt` from Hugging Face (`lab-cosmo/upet`), pet-jax (Flax) conversions produced by `petjax-convert`.
- `uma-s-1p2/`: UMA-S-1.2 (Meta FAIR), a pure-JAX port not used by the preprint. Gated upstream weights on Hugging Face (`facebook/UMA`), converted by `build.py` in the folder.

The test suite and every experiment under `work/` load these checkpoints, so rebuilding them is the first step after `uv sync`.
