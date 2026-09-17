# relax-goennheimer

Relaxed geometries for the 233-structure Gönnheimer+Moosavi reference set, one relaxation per production model. `work/hessians` and `work/hessian-ablations` read their unit-cell geometries from here, and `work/cv-ref-goennheimer` runs its reference C_v pipeline over the MACE+D3 label.

## Scope and method

With the MACE+D3 label we reproduce the protocol of Gönnheimer et al. (JCTC 2025): MACE-MP-0 medium plus DFT-D3(BJ, PBE), BFGS over `FrechetCellFilter`, fmax = 5×10⁻³ eV/Å, 25000 steps, fp64. We pass every protocol parameter on the command line, so the reproduction does not rest on a default that could later change. The PET labels use the same optimizer, tolerance and precision on the same structures. We relax in the primitive cell only.

`run.py` loops the frames of the reference XYZ and runs `sadmof.scripts.relax` in a subprocess per structure, so one crash or OOM does not end the batch. Structures whose `relaxed.xyz` already exists are skipped, so a re-run continues where the last one stopped.

| `--calculator` | model |
|---|---|
| `mace-mp0` | MACE-MP-0 medium |
| `mace-mp0+d3` | MACE-MP-0 medium + DFT-D3(BJ, PBE) |
| `pet` | PET-MAD-XS, no D3, 8 neighbours by default |
| `pet-s` | PET-MAD-S, no D3, 16 neighbours by default |

`--num-neighbors N` overrides the PET adaptive-cutoff target and appends `_nn<N>` to the output label.

## Inputs

- `sources/processed/goennheimer+moosavi/structures.xyz`, the merged reference set.
- `sources/processed/mace-mp-0-medium/`, `sources/processed/pet-mad-xs/`, `sources/processed/pet-mad-s/`, the converted checkpoints.

## Usage

```bash
uv run python run.py --calculator mace-mp0+d3 --optimizer bfgs --dtype float64 \
    --fmax 0.005 --max-steps 25000
uv run python run.py --calculator pet --optimizer bfgs --dtype float64 RSM0004 RSM0010
uv run python run.py --calculator mace-mp0+d3 --optimizer bfgs --dtype float64 --dry-run
uv run python collect.py
```

With no identifiers all 233 frames are relaxed. The output directory defaults to `<calc>_<opt>_<dtype>` and `--label` overrides it. The campaign ran on kuma, one MIG slice per job, since the set stays at or below 240 atoms.

## Output

`output/<label>/<identifier>/relaxed.xyz` holds the relaxed structure with the source dataset keys carried in `info`. The sibling `relax_summary.json` records step count, convergence, final fmax, wall time and provenance.

We ship three labels, each complete at 233/233 converged: `mace-mp0+d3_bfgs_float64`, `pet_bfgs_float64`, `pet-s_bfgs_float64`.

`collect.py` writes `output/summary.csv` and prints a per-label convergence and timing table. It rewrites the file rather than appending, so passing labels leaves only those rows.

## Related

- `work/relax-big-mofs`, the same machinery on the four large MOFs.
- `work/select-goennheimer`, the ranking that turns this set into the nested `goenn:<k>` rosters.
- `work/hessians`, `work/hessian-ablations` and `work/cv-ref-goennheimer`, the consumers.
