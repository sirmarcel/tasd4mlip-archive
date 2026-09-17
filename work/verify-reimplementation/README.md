# verify-reimplementation

Do our JAX models predict what the upstream PyTorch models predict? Computes energy, forces and stress for the same structures with both stacks in both precisions, and reports every prediction path as a deviation from the upstream fp64 model. Source of the preprint's parity table in the model appendix.

## Scope and method

MACE-MP-0 medium is our own port in `sadmof.models.mace`, converted by `sources/processed/mace-mp-0-medium/build.py`. PET-MAD XS and S go through `pet-jax` (which we also test, as it is also our code). Both sides of each model trace to one upstream file, the raw checkpoint the conversion read, so a deviation is a statement about the implementation and not about which weights were loaded.

We take `torch/float64`, the upstream implementation in double precision, as the reference and report every other path against it. `torch/float32` is the scale bar: an fp32 deviation from our side only means something next to the fp32 deviation upstream shows on the same structures. `jax/float64` holds precision fixed and isolates the implementation. `jax/float32` is the stack as inference actually runs it. `jax/float32_gpu_mmdefault` is the same in single precision with JAX's default matmul mode on an accelerator, which is TF32.

We use the dataset's own geometries, unrelaxed, since we do not test the relaxation protocol here. PET keeps its adaptive-cutoff force terms, which are part of the model. D3 is not part of this experiment, `tests/test_d3_reference.py` pins it against `torch-dftd` directly.

One structure, RSM1877, deviates at 5.6×10⁻⁴ eV/Å in the PET-XS `jax` fp32 row, an order of magnitude above the roster median. The cause is a backend kernel choice. PET's bump cutoff is `0.5·(1 + tanh(1/tan(πx)))` in both stacks, character for character, and its AD gradient `1 - tanh²(u)` vanishes once fp32 `tanh(u)` rounds to exactly 1. That happens at `u = 8.00` under XLA against `u = 8.665` under PyTorch, and RSM1877 has four N–H pairs at `u = 8.17`. Neither the neighbour selection nor the adaptive-cutoff gradient is involved. Second derivatives amplify the effect more than forces do, so the force column understates it for Hessian work.

The torch references need `mace-torch` for MACE or `metatrain` plus `metatomic-torch` for PET, none of which belong in the project environment. `run.py` launches those workers through a `uv run --with` overlay defined in `common.OVERLAY` and the JAX workers in the project environment, so a full grid is still one command. For the PET reference we take the inner PET module out of the LLPR wrapper the PET-MAD checkpoints ship in and cast it before `export()`, so the fp64 reference is a promotion of the shipped fp32 weights rather than a different model.

## Inputs

- `sources/processed/goennheimer+moosavi/structures.xyz`, the unrelaxed reference set.
- `work/select-goennheimer/output/ranking.csv`, which `--roster goenn:<k>` indexes into.
- `sources/processed/mace-mp-0-medium/`, `sources/processed/pet-mad-xs/`, `sources/processed/pet-mad-s/`, the converted checkpoints, for the JAX side.
- `sources/raw/mace-mp-0-medium/2023-12-03-mace-128-L1_epoch-199.model`, `sources/raw/pet-mad-xs/pet-mad-xs-v1.5.0.ckpt` and `sources/raw/pet-mad-s/pet-mad-s-v1.5.0.ckpt`, the upstream checkpoints, for the torch side. The checkpoint rebuild leaves them at these paths.

## Usage

```bash
uv run python run.py --roster goenn:50
uv run python run.py --roster goenn:10 --dry-run
uv run python run.py --roster goenn:10 --models mace --stacks jax
uv run python run.py --roster goenn:50 --stacks jax --devices gpu \
    --matmuls default high highest
uv run python parity_extract.py --roster goenn:50
uv run --extra plots python parity_table.py
```

A condition is `(stack, model, dtype, device, matmul)`, twelve of them in the CPU grid, and one worker loops the roster inside it, skipping structures whose npz is already there. Delete an npz to recompute it. The CPU grid runs locally. The accelerator rungs ran on kuma on a full H100, since TF32 is a tensor-core behaviour and the answer is architecture specific. The torch reference never leaves CPU.

## Output

`output/<stack>/<model>/<variant>/<identifier>.npz` holds energy, forces, stress, atom count and wall time, widened to fp64 on write so an fp32 result is stored exactly. `meta.json` next to it records the checkpoint, device, matmul mode, timing and provenance. Variants on disk are `float32` and `float64` for both stacks, plus `float64_gpu`, `float32_gpu_mmdefault`, `float32_gpu_mmhigh` and `float32_gpu_mmhighest` on the JAX side, 50 structures each.

`results/parity.json` holds the per-structure deviations, unaggregated, with the reference scales (`e_ref_per_atom`, `f_ref_max`, `stress_ref_max`) they sit against. `figures/parity_table.tex` is the LaTeX fragment the preprint includes, cells being median with worst in parentheses over the roster. `parity_table.py` also prints the per-metric extremes and the worst-structure identities to stdout.

## Related

- `work/select-goennheimer`, the ranking the rosters index into.
- `work/hessian-ablations`, which measures what the sparse pipeline's `no_shadow` path costs.
- `work/determine-perf-settings`, the throughput axis these single cold calls say nothing about.
