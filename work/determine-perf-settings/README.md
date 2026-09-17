# determine-perf-settings

Which single Hessian-vector-product engine setting, over AD mode by chunk size by remat, maximises Hessian throughput on an H100 while fitting every system the preprint needs? Every production Hessian downstream runs at the setting chosen here, so per-structure timings stay comparable and nothing runs out of memory mid-campaign. The `matmul` phase additionally measures the cost of pinning `jax_default_matmul_precision=highest` against the JAX default and supplies the throughput numbers quoted in the preprint's precision paragraph. `work/hessian-ablations` covers the accuracy side of the same question.

## Scope and method

We probe in fp32 with MACE-MP-0 medium, whose Hessians need the most memory of the three models, on converged supercells at exact `K = 4`, on five systems that span the size and density range of the roster and the giant MOFs: RSM0023, RSM1831, RSM1885, RSM0254 and mil101. The PET models are checked at the chosen setting afterwards. The probe kernel mirrors the per-mode engines of `asdex` and sweeps a bounded set of seeds instead of the full color set. `collect.py` corrects the timings for fixed per-call cost and adds the memory a real sparse run holds on top of the probe, as documented in its docstring.

The phases are `smoke`, GPU-path validation, `patterns`, pattern and coloring statistics, `grid` and `extremes`, the mode by chunk by remat sweeps, `certify`, the real production path end to end at the chosen setting, `pet`, PET-XS and PET-S at production input sizing, and `matmul`, the chosen setting against `default` and `highest` matmul precision.

The setting we chose, written to `output/settings.json` by `collect.py --write`, is `fwd_over_rev`, chunk size 1, remat on, fp32, tight padding. We derived it on an NVIDIA H100, and it does not transfer to other accelerators. Matmul precision is not part of it. The `matmul` records live in a separate tree that never feeds the derivation, which `check_decision_rule.py` pins.

## Inputs

- `work/relax-goennheimer/output/` and `work/relax-big-mofs/output/`, the relaxed geometries.
- `work/select-goennheimer/output/`, the committed raw-geometry survey the five probe systems were picked from.
- `sources/processed/mace-mp-0-medium/`, `sources/processed/pet-mad-xs/`, `sources/processed/pet-mad-s/`, the converted checkpoints.

## Usage

```bash
uv run python run.py patterns --dry-run
uv run python run.py patterns
uv run python run.py grid --shard 0/4
uv run python collect.py --write
uv run python run.py certify
uv run python matmul_extract.py
uv run --extra plots python matmul_table.py
uv run python check_decision_rule.py
```

The phases run in the order above, with `collect.py --write` between `extremes` and `certify` because the later phases read `settings.json`. `--shard I/N` splits a phase into non-overlapping stripes for concurrent jobs, and `--only <substr>` cherry-picks conditions. The campaign ran on kuma on a full H100.

## Output

`output/{patterns,probe,certify,matmul}/<label>.json`, one file per condition, with the `highest` rung of a `matmul` pair suffixed `_mmhigh`, plus `output/settings.json`, the chosen setting that downstream experiments read.

`results/matmul.json` pairs the two rungs of each condition and holds the warm, cold and peak-memory ratios of `highest` over `default`. `figures/matmul_table.tex` is the tabular of those ratios, and `matmul_table.py` also prints the per-model ranges the preprint's precision paragraph quotes.

## Related

- `work/hessians` and `work/hessian-ablations`, the production campaigns that run at this setting.
- `work/hessian-ablations`, the accuracy cost of the TF32 default.
