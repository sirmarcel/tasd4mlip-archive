# select-goennheimer

Turn the 233-structure Gönnheimer+Moosavi set into one frozen ranking, so that every other experiment can name a roster by size. `work/hessians`, `work/hessian-ablations`, `work/verify-reimplementation` and `work/determine-perf-settings` all resolve `goenn:<k>` to the first k rows of `output/ranking.csv`. The prefixes nest, so 10 ⊂ 20 ⊂ 50 and growing a roster never changes the structures already in it.

## Scope and method

Three scripts. `measure.py` writes one flat record per structure, covering geometry, structure class, and the smallest supercell whose Hessian contains every force constant exactly once. `selection.py` turns the records into the ranking. `check.py` verifies that the committed ranking still matches a recomputation from the records.

We measure the structures as they come from the source dataset, without relaxing them first, because the ranking characterises the dataset and should not depend on a relaxation protocol. Adjacency is the production MACE setting, vesin with `r_c = 6.0` Å, at the exact interaction depth `K = 4` measured in `work/hops`.

`selection.py` implements the per-class farthest-point sampling and the 8:1:1 interleaving described in the preprint's data appendix. The merge is deterministic, and the one structure without a class label counts as a MOF.

We left the other measured quantities out of the ranking, since they order the population the same way as the three features above, which `check.py` verifies. A benchmark set that must satisfy a constraint, such as a maximum atom count for a Hessian that has to fit in memory, skips the violating rows at the point of use. The ranking itself is never reordered.

This experiment selects structures and grades nothing else. The converged supercell each Hessian actually runs in is recomputed at run time on the relaxed geometry by `work/hessians`, since relaxation shifts the multipliers.

## Inputs

- `sources/processed/goennheimer+moosavi/structures.xyz`, the merged reference set, unrelaxed.

## Usage

```bash
uv run python measure.py --out output/records \
    ../../sources/processed/goennheimer+moosavi/structures.xyz
uv run python selection.py output/records --out output/ranking.csv
uv run python check.py output/records output/ranking.csv
```

`measure.py` is resumable, an existing record is skipped unless `--force` is passed. `check.py` fails loudly if `ranking.csv` no longer matches a recomputation from the records, if the 8:1:1 class pattern is broken, or if the density and reach equivalence stops holding. It also prints the class and element makeup of the first 10, 20 and 50 structures and how much of the population's range they span. Everything here runs on CPU in seconds to minutes, so no cluster submission is involved.

## Output

- `output/ranking.csv`, the frozen ranking: `rank, structure, class, log_n_atoms, number_density, minconv_fill`, one row per structure. Every roster spec indexes into this file.
- `output/records/<identifier>.json`, 233 flat records: identity (`structure_type`, `elements`), geometry (`n_atoms`, cell, widths, volume, `number_density`), the scale of the interaction pattern (`R_K`, `reach_per_atom`, `e_max`), the minimal supercell (`minconv`, `minconv_n_atoms`, `minconv_fill`), and a provenance stamp.

## Related

- `work/hops`, where the exact interaction depth `K = 4` comes from.
- `work/relax-goennheimer`, the relaxations of the same set.
- `work/hessians`, `work/hessian-ablations`, `work/verify-reimplementation`, `work/determine-perf-settings`, the consumers of the ranking.
