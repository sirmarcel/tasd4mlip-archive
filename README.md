# tasd4mlip: code and data archive

Code and data for the preprint *Truncated automatic sparse differentiation for machine learning interatomic potentials* by Marcel F. Langer, Adrian Hill, and Michele Ceriotti (arXiv, 2026). The archive contains the `sadmof` Python package, every experiment behind a figure, table, or number in the preprint (run scripts, extracted results, figures, and raw run records), and the processed external inputs the experiments read. (Note: `sadmof` was the previous codename for this project, we keep it to avoid a bulk rename at this point. We will rename it later.)

Published on Zenodo at [doi:10.5281/zenodo.22813524](https://doi.org/10.5281/zenodo.22813524) and mirrored at [github.com/sirmarcel/tasd4mlip-archive](https://github.com/sirmarcel/tasd4mlip-archive). The GitHub mirror keeps the binary data in Git LFS, so a full clone downloads about 180 MB of LFS objects. `GIT_LFS_SKIP_SMUDGE=1 git clone ...` fetches the code, structures, and JSON records only.

**This is intended as an archive, and not provided as "production implementation" of the method. We plan to ship truncated ASD in `pet-jax` as soon as possible.**

## Layout

```
├── src/sadmof/         # The sadmof package: models (MACE, PET, D3), sparse and dense
│                       # Hessian engines, observables, relaxation, I/O, plot style.
├── tests/              # pytest suite (physics pins, hop counts, end-to-end C_v chain).
├── work/               # The experiments. One directory per experiment, see the map below.
├── sources/            # External inputs (data sets, model checkpoints), see sources/README.md.
├── pyproject.toml      # Environment definition (uv), dependencies pinned to exact revisions.
├── ASSEMBLY.json       # Which revision of the internal repository this archive was built from.
└── LICENSE             # MIT for code, CC BY 4.0 for data.
```

## Environment

Managed with [`uv`](https://docs.astral.sh/uv/). `pyproject.toml` pins `asdex` and `marathon-train` to the PyPI releases the experiments ran with, and `pet-jax`, which has no PyPI release, to the exact upstream revision.

```
uv sync                 # base environment (no plotting stack)
uv sync --extra plots   # + matplotlib, needed to rebuild figures
```

GPU runs need a CUDA-enabled jaxlib matching the pinned `jax<0.11`. (We found errors with `jax==0.11` so exclude it for now.)

The model checkpoints are not included. Each folder under `sources/processed/` pins the upstream weights and converts them with one command, and records the sha256 of the converted file the experiments ran with. Rebuild all three before running anything:

```
cd sources/processed/mace-mp-0-medium && ...   # see README.md there
cd sources/processed/pet-mad-s        && ...
cd sources/processed/pet-mad-xs       && ...
```

Then the tests run on CPU:

```
uvx --with-editable . pytest tests/
```

The package also carries a JAX port of UMA-S-1.2 and third-order derivative machinery. Neither is used by the preprint (yet!), and the UMA checkpoint is not part of this archive. The tests that need it skip.

## From experiment to figure

Every figure and table in the preprint is produced inside the experiment that measured it, in two committed stages:

```
work/<exp>/<topic>_extract.py  ->  work/<exp>/results/   # structured, small
work/<exp>/<topic>_load.py                               # results/ -> arrays, shared loader
work/<exp>/<topic>_figure.py   ->  work/<exp>/figures/   # PDF + PNG
work/<exp>/<topic>_table.py    ->  work/<exp>/figures/   # .tex fragments
work/<exp>/<topic>_results.py                            # the numbers quoted in the text, as JSON
```

`results/` is small and self-contained: every figure rebuilds from this archive alone, without re-running the GPU campaigns. `output/` holds the raw per-run records (one JSON record per condition, plus stored spectra and, where feasible, Hessians) that the extract scripts read. The dense Hessians of the large frameworks, up to hundreds of GB, are not included, and everything derived from them is. We only include runs that are actually used in the preprint.

To rebuild, for example, the truncation figure:

```
cd work/hessians && uv run --extra plots python ladder_figure.py   # writes figures/truncation.pdf
```

## Experiment map

| directory | question it answers | in the preprint |
|---|---|---|
| `work/hops` | how many graph hops does an MLIP Hessian span (the exact `K`)? | the `K` values throughout |
| `work/select-goennheimer` | which structures form the benchmark rosters? | rosters of every table |
| `work/relax-goennheimer` | relaxed geometries for the 233-structure reference set | input to `hessians`, `hessian-ablations`, `cv-ref-goennheimer` |
| `work/relax-big-mofs` | relaxed geometries for the four large MOFs | input to `hessians`, `overview` |
| `work/cv-ref-goennheimer` | does the dense fp64 pipeline reproduce the published reference `C_V`? | cross-check of the ablation reference corner |
| `work/verify-reimplementation` | do the JAX models match the upstream PyTorch models? | parity table |
| `work/determine-perf-settings` | which HVP engine setting runs production, and what does TF32 cost? | TF32 throughput numbers |
| `work/hessian-ablations` | what do fp32, no-D3, and no-shadow-forces cost in accuracy? | ablation table and numbers |
| `work/hessians` | what does a Hessian cost, and how accurate is it, versus pattern truncation? | truncation, decay, error-split, and calorimetry figures, giant-MOF and ladder tables |
| `work/overview` | how does truncated sparse AD look on one framework? | overview figure |

## External data

`sources/README.md` documents the provenance of every external input: the Gönnheimer and Moosavi reference sets, experimental calorimetry curves, upstream CIF libraries, and the MACE-MP-0 and PET-MAD model checkpoints, with pinned URLs and checksums for everything not included directly.

## Citing

Please cite the preprint. The Zenodo record has its own DOI for citing the archive itself.

## License

Code under the MIT License, data under CC BY 4.0. See `LICENSE`.
