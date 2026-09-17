# hessians

What does an MLIP Hessian cost, and how accurate is it, as a function of how much of the sparsity pattern is kept? Computes Hessians and their vibrational observables in three modes, `dense`, the exact sparse pattern at each model's interaction depth `K`, and the truncated patterns `k < K`, with end-to-end and per-stage timings, over a roster of porous frameworks. Source of the preprint's cost table, giants table, truncation figure, decay figure, error-split figure, calorimetry figure, and the atom-graph coloring numbers of the appendix.

## Scope and method

We run everything in fp32 with matmul precision pinned to `highest`, in `fwd_over_rev` mode, at chunk size 1, with remat on, the settings we chose in `work/determine-perf-settings`. Inputs use `multiples_of_4` padding and no PET `k_sel` slack, both tighter than anything that experiment probed. All of it lives as constants in `common.py` and is recorded in every record. We deliberately expose no flags for them: changing one is a code change, visible in git and in each record's provenance.

`--roster` accepts `goenn:<k>`, `giants`, and explicit identifiers, composed. `goenn:<k>` takes the first `k` rows of the committed ranking in `work/select-goennheimer`, and prefixes nest, so growing a roster never changes what is already in it. `giants` is MOF-177, MOF-210, MIL-100 and MIL-101. Each model reads its own relaxation, listed as `RELAX_TAGS` in `common.py`, so the Hessian is taken at the minimum of the potential energy surface being probed. We treat a missing relaxation as a hard error rather than falling back to another model's geometry.

The cell is the minimum-volume alias-free multiplier at the model's exact `K`, recomputed at run time on the relaxed geometry. It comes out `(1,1,1)` where the unit cell already holds the reach, so `--supercell minconv` is always correct and `--supercell unit` is an override. The hop ladder per model is `{1, 2, 3, 4}` plus its exact `K`, which is 4 for MACE, 5 for PET-XS and 7 for PET-S. At `k = K` the sparse result equals dense up to floating-point associativity. The two error components of a truncated rung, discarded couplings and contamination of retained ones, are separated offline from the stored Hessians by `ladder_extract.py`, as described in the preprint.

`hessian_cold_s` is the cold one-shot including XLA compile, the cost paid per structure, and `--warm` adds a warm call. `total_s` is the end-to-end process wall clock, and `collect.py` reports it minus the summed stages, so an unmeasured stage surfaces as a residual. Observables are stored with and without the acoustic sum rule, and `n_dropped` counts the modes below the `10⁻³ cm⁻¹` acoustic threshold in each set. We evaluate heat capacities at 250, 300, 350 and 400 K. `recolor.py` re-times the coloring stage of every sparse rung through the lifted route, `sadmof.sparse.hessian_coloring`, which produces the identical colors as the coordinate-level coloring the grid ran, and writes a `record.lift.json` sidecar that extraction takes the coloring cost from.

## Inputs

- `work/relax-goennheimer/output/` and `work/relax-big-mofs/output/`, the relaxed geometries, one tag per model.
- `work/select-goennheimer/output/ranking.csv`, which `--roster goenn:<k>` indexes into.
- `work/hops`, where `K = 4 / 5 / 7` comes from, and `work/determine-perf-settings`, where the engine settings come from.
- `sources/processed/mace-mp-0-medium/`, `sources/processed/pet-mad-xs/`, `sources/processed/pet-mad-s/`, the converted checkpoints.
- `sources/processed/big-mofs/experimental_curves.npz`, the digitised calorimetry the calorimetry figure, table and results script compare against.

## Usage

```bash
uv run python run.py --roster goenn:10 giants --dry-run
uv run python run.py --roster giants --models mace
uv run python run.py --roster goenn:2 --hops 1 2 --modes dense --only RSM0023
uv run python collect.py
uv run python recolor.py
uv run python decay_extract.py --structures giants
uv run python ladder_extract.py --structures goenn:10 giants
uv run --extra plots python decay_figure.py
uv run --extra plots python ladder_figure.py
uv run --extra plots python errsplit_figure.py
uv run --extra plots python calorimetry_figure.py
uv run --extra plots python ladder_table.py
uv run --extra plots python giants_table.py
uv run --extra plots python calorimetry_table.py
```

`ladder_results.py`, `decay_results.py`, `calorimetry_results.py` and `coloring_results.py` take no arguments, print the numbers the preprint quotes as JSON, and exit nonzero when a quoted claim no longer holds.

Deleting a record re-derives observables from the cached Hessian, deleting the directory recomputes it. An out-of-memory run is a recorded result. The campaign and both extractions ran on kuma on a full H100. We include the campaign scripts for provenance. The 285 shipped conditions sum to about 180 H100-hours, so we expect figures to be rebuilt from `results/`, not from a re-run.

## Output

`output/<structure>/<label>/` holds `record.json`, the config, system, sparsity, Hessian, observables, timings, memory and provenance of one condition, and `observables.npz`, the frequencies, masses, temperature grid and heat capacities with and without the acoustic sum rule. We ship those two. The Hessians themselves (`hessian_raw.npy`, `hessian_sparse.npz`), the cached colorings (`coloring.npz`), the tiled geometries (`supercell.xyz`) and the `record.lift.json` coloring sidecars run to hundreds of gigabytes across the grid and stay on the cluster, so extraction runs there.

`results/decay/<structure>_<model>.{json,npz}` holds a 2D histogram over interatomic distance and `log10 ‖Φ_ij‖` per hop shell, binned at `0.1 Å` and `0.05` dex, plus the scalars whose exactness matters, `norm_max`, `dist_max_nonzero_A`, and the per-hop reach radii `R_h`. Quantiles read off the histogram carry its bin resolution, while the scalars in the `.json` are exact. `results/ladder/<structure>_<model>.{json,npz}` holds every rung's record scalars joined with the Hessian error of each sparse rung against the pair's dense reference, and a fixed-grid density-of-states histogram per rung.

`figures/` holds `decay.{pdf,png}`, `decay_shell.{pdf,png}`, `decay_chemistry.{pdf,png}`, `truncation.{pdf,png}`, `errsplit.{pdf,png}`, `calorimetry.{pdf,png}`, `ladder_table.tex`, `giants_table.tex` and `calorimetry_table.tex`. The preprint uses `decay_chemistry`, `truncation`, `errsplit`, `calorimetry`, `ladder_table` and `giants_table`.

## Related

- `work/determine-perf-settings`, the engine settings and the memory envelope.
- `work/hops`, where the exact `K` per model comes from.
- `work/hessian-ablations`, the accuracy cost of the pinned fp32, no-D3 and no-shadow settings.
- `work/select-goennheimer`, the ranking the rosters index into, and `work/relax-goennheimer` and `work/relax-big-mofs`, the geometries.
- `work/cv-ref-goennheimer`, the primitive-cell fp64 reference heat capacities.
- `work/overview`, the single-structure Hessian behind the overview figure, which imports this experiment's construction.
