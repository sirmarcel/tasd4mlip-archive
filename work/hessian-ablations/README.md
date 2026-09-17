# hessian-ablations

How much accuracy do the production pipeline's departures from a reference Hessian calculation cost? Computes dense unit-cell Hessians over the `goenn:10` roster and flips one axis at a time away from the reference corner: numerical precision (fp64 to fp32, with the TF32 matmul mode as its own rung), the D3 dispersion correction for MACE, and PET's shadow forces. Source of the preprint's ablation table and of the macros that bind the ablation paragraph's numbers to the data.

## Scope and method

We ablate on unit cells only and with dense Hessians only. The ablation axes are local physics, so the unit cell shows them, and dense unit-cell Hessians are cheap where converged supercells are not. Sparsity is `work/hessians`' axis and is ablated there. Engine settings are pinned in `common.py` as `fwd_over_rev`, chunk size 1, remat on, `multiples_of_4` padding and no PET `k_sel` slack, and are recorded in every record. Precision is an axis here rather than a constant, and D3's input builder pads with its own fixed strategy, recorded as such. Per structure the grid is twelve GPU Hessians plus three host-side combines.

| condition | role |
|---|---|
| `mace_fp64`, `mace_fp32_mmhigh`, `mace_fp32` | bare MACE precision chain, `fp32_mmhigh` is the production corner |
| `d3` | D3 alone, fp64, computed once and combined below |
| `mace_fp64+d3` | MACE reference |
| `mace_fp32_mmhigh+d3`, `mace_fp32+d3` | precision rungs at fixed D3 |
| `pet-{xs,s}_fp64_shadow` | PET reference |
| `pet-{xs,s}_fp64_noshadow` | shadow flip |
| `pet-{xs,s}_fp32_noshadow_mmhigh`, `pet-{xs,s}_fp32_noshadow` | precision rungs, `fp32_noshadow_mmhigh` is the production corner |

D3 is a fixed additive fp64 correction, so it is computed once per structure and summed with either MACE Hessian by `combine_d3.py`, which `run.py` runs by itself once the sources are done. PET's training labels already include D3, so there is no PET+D3 condition. `*_mmhigh` pins matmul precision to `highest`, so the fp32 rungs separate the number format from the TF32 matmul mode. The preprint explains the chain. We compare the acoustic-sum-rule variants, because one leaked acoustic mode adds a `k_B` to the heat capacity and would swamp the effect being measured.

## Inputs

- `work/relax-goennheimer/output/`, the relaxed geometries, one tag per model.
- `work/select-goennheimer/output/ranking.csv`, which `--roster goenn:<k>` indexes into.
- `sources/processed/mace-mp-0-medium/`, `sources/processed/pet-mad-xs/`, `sources/processed/pet-mad-s/`, the converted checkpoints.
- `work/cv-ref-goennheimer/output/`, optional, read by `collect.py` for a per-structure cross-check of `mace_fp64+d3`.

## Usage

```bash
uv run python run.py --roster goenn:10 --dry-run
uv run python run.py --roster goenn:10
uv run python run.py --roster RSM0010 --models mace d3
uv run python combine_d3.py --structure RSM0010 --fp fp32_mmhigh
uv run python collect.py
uv run python ablation_extract.py --structures goenn:10
uv run --extra plots python ablation_table.py
uv run --extra plots python ablation_numbers.py
```

Deleting a record re-derives observables from the cached Hessian, deleting the directory recomputes it. An out-of-memory run is a recorded result. The campaign and the extraction ran on kuma.

## Output

`output/<structure>/<structure>_<condition>/` holds `record.json` (config, system, Hessian, observables, timings, memory, provenance), `geometry.xyz` (the exact unit cell used), `hessian_raw.npy` (the dense real-block Hessian in the working precision) and `observables.npz` (frequencies, masses, temperature grid, heat capacities with and without the acoustic sum rule). Unit cells are small enough that we ship the Hessians. The combined `…+d3` conditions hold `record.json` and `observables.npz` only, since the summed Hessian is one load and add away from its sources and `ablation_extract.py` rebuilds it the same way.

`results/ablation/<structure>.{json,npz}` holds every condition's heat-capacity and mode-count scalars, all spectra, and per ablation pair the heat-capacity deviation at all temperatures and both acoustic-sum-rule variants, the elementwise spectrum deviation as max and RMS, and the Hessian-level error as relative Frobenius and max entry, computed in fp64.

`figures/ablation_table.tex` is the LaTeX fragment the preprint includes. `figures/ablation_numbers.tex` is one `\newcommand` per number the ablation paragraph quotes, so the prose is bound to the data.

## Related

- `work/hessians`, the cost and truncation experiment whose pinned fp32, no-D3 and no-shadow settings this one measures the accuracy cost of.
- `work/determine-perf-settings`, the throughput gain of the TF32 default.
- `work/cv-ref-goennheimer`, the fp64 MACE+D3 primitive-cell reference the cross-check compares against.
- `work/relax-goennheimer`, the geometries.
