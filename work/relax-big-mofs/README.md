# relax-big-mofs

Relaxed geometries for the four large benchmark MOFs: MOF-177 (808 atoms), MOF-210 (1854), MIL-100(Cr) (2788) and MIL-101(Cr) (3604). These are the `giants` roster of `work/hessians`, which reads a geometry from here for each model.

## Scope and method

Same frame loop as `relax-goennheimer`, pointed at the big-MOF set. The one protocol change is the optimizer. BFGS is O(N³) per step, a `scipy.linalg.eigh` on the (3N+9)² inverse Hessian, which is prohibitive at MIL-101's 10821 degrees of freedom. `LBFGSLineSearch` is limited memory and reaches the same minimum at a fraction of the cost, so we use `lbfgs-ls` here.

Everything else matches the siblings: fmax = 5×10⁻³ eV/Å, 25000 steps, fp64, primitive cell only. Verlet cache settings are the package defaults, base calculator `--skin 1.0` and `--d3-skin 5.0`, exposed as flags so the tuning can be re-run. A non-default skin is recorded in the label as `_skin<s>-<d3>`.

The available calculators are `mace-mp0`, `mace-mp0+d3`, `pet` (PET-MAD-XS) and `pet-s` (PET-MAD-S), as in `relax-goennheimer`.

## Inputs

- `sources/processed/big-mofs/structures.xyz`, the four RASPA2 library structures. None is DFT-optimized, so the first steps carry large forces.
- `sources/processed/mace-mp-0-medium/`, `sources/processed/pet-mad-xs/`, `sources/processed/pet-mad-s/`, the converted checkpoints.

## Usage

```bash
uv run python run.py --calculator mace-mp0+d3
uv run python run.py --calculator mace-mp0+d3 mof177
uv run python run.py --calculator pet-s mil101 --dry-run
uv run python run.py --calculator mace-mp0+d3 --skin 2.0 --d3-skin 10.0
uv run python collect.py
```

With no identifiers all four MOFs are relaxed. Re-running resumes, since a structure with `relaxed.xyz` present is skipped. If `lbfgs-ls` stalls on a Wolfe-condition failure, retry with `--optimizer lbfgs`. The campaign ran on kuma on a full H100, because MIL-101's fp64 D3 pair graph on a 62.8 Å cell does not fit a MIG slice.

## Output

`output/<label>/<identifier>/relaxed.xyz` plus a sibling `relax_summary.json` with step count, convergence, final fmax, wall time and provenance.

We ship three labels, all four MOFs each: `mace-mp0+d3_lbfgs-ls_float64`, `pet_lbfgs-ls_float64`, `pet-s_lbfgs-ls_float64`.

`collect.py` writes `output/summary.csv`, one row per relaxation.

## Related

- `work/relax-goennheimer`, the same machinery on the reference set.
- `work/hessians`, which uses these geometries as its `giants` roster.
