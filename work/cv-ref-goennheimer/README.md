# cv-ref-goennheimer

Run the full observables chain, MACE-MP-0 medium + D3(BJ, PBE) dense AD Hessian to phonon frequencies to harmonic C_v, over the 233 canonically relaxed primitive cells, and check it against the spectra and C_v values published by Gönnheimer et al. (JCTC 2025). The resulting `output/dense_float64/` is the fp64 reference the `mace_fp64+d3` cross-check of `work/hessian-ablations` compares against, and the accuracy anchor `work/hessians` points to for the primitive-cell regime.

## Scope and method

Same chain and same conventions as the four-structure end-to-end test in `tests/test_e2e_cv.py`: raw Hessian, imaginary modes set to zero, modes with |ν| < 10⁻³ cm⁻¹ dropped, gravimetric J/(g·K), no frequency scaling. Temperature grid 250 to 400 K in 10 K steps.

Dense Hessians via `sadmof.dense.get_dense_hessian_fn` on primitive cells of 13 to 240 atoms.

Gönnheimer's own pipeline is AD except for the D3 Hessian, which they take by finite differences over the dftd3 binary's forces, and our relaxed geometries differ slightly from theirs, so the agreement is at implementation noise for most structures, with a tail from X–H stretch hypersensitivity without C_v effect, saddle points on the reference side, and basin flips. `collect.py` prints the deviation statistics, and reports structures whose reference carries `goenn_uc_converged=False` separately.

## Inputs

- `work/relax-goennheimer/output/mace-mp0+d3_bfgs_float64/`, the canonical Gönnheimer reproduction at 233/233 converged. Each frame's `info` carries the source dataset keys, including `goenn_uc_cv_{T}` and `dft_cv_{T}`.
- `sources/processed/mace-mp-0-medium/`, the converted checkpoint.
- `sources/processed/goennheimer+moosavi/frequencies_uc.npz`, the published primitive-cell spectra, read by `collect.py`.

## Usage

```bash
uv run python run.py
uv run python run.py RSM0004 RSM0010
uv run python run.py --dry-run
uv run python run.py --remat
uv run python collect.py
```

`run.py` shells out to `cv.py` once per structure, smallest first, so one crash or OOM does not end the batch. Re-running resumes, since a structure whose npz exists is skipped. `--remat` rematerialises the linearization residuals, which is needed on smaller GPU memory above roughly 110 atoms and costs about 21 % compute for identical results. `collect.py` takes an optional label and defaults to `dense_float64`. The campaign ran on kuma, mostly on a 24 GB MIG slice, with one dense-pair-graph structure finished on a full H100.

## Output

`output/dense_float64/<identifier>.npz`, 233 files, each holding our frequencies, the C_v curve over the temperature grid, stage timings and provenance.

`collect.py` writes `output/dense_float64/summary.csv` and prints the aggregate deviation statistics against the published reference.

## Related

- `work/relax-goennheimer`, the geometries this runs on.
- `work/hessian-ablations`, which cross-checks its `mace_fp64+d3` corner against this reference.
- `work/hessians`, the sparse and truncated counterpart at supercell size.
