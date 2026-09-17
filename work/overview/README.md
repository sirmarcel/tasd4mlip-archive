# overview

The preprint's overview figure: how truncated sparse AD works, on a toy chain, and what the Hessian of a real framework looks like both around one atom and as a whole. One structure, one model, computed on a laptop CPU.

## Scope and method

MOF-177 with PET-XS at the exact hop count `K = 5`. Its 808-atom unit cell already holds the model's reach, so no supercell is needed, and the exact pattern takes 750 HVPs, a few minutes on CPU. We import the geometry, the converged-cell check, the model inputs and the engine settings from `work/hessians/common.py`, so this is the production Hessian construction, recomputed where a figure script can reach it.

The toy in panel (a) is a 7-atom chain with one coordinate per atom and a model with `K = 2`, truncated at `k = 1`. Its star colorings, compressed products, decompression reads and contaminated entries come from `asdex` applied to a synthetic Hessian with one decade of decay per hop, so the schematic is a computed result rather than a drawing.

The pipeline has four steps.

```
run.py               structure -> output/<structure>/   exact Hessian, graph, hop matrix, record
overview_extract.py  output/   -> results/              block norms, hop matrix, graph, toy chain, summary
overview_load.py     results/  -> arrays
overview_figure.py   results/  -> figures/overview.pdf and .png
```

We chose the marked atom, a Zn near the cell centre, so the shells run node, linker, node. `overview_figure.py` derives everything else data-bound from it: the projection (the principal plane of that atom's `K`-hop cluster, unwrapped around it), the bonds (1.2 times the sum of covalent radii), the atom order of the matrix image (reverse Cuthill-McKee on the one-hop graph, so nested patterns read as nested bands), and the colour mapping of block norms. It writes those decisions to `output/build/` and calls `typst compile` on `overview_figure.typ`, which decides everything stylistic. Layout iterations therefore recompile in under a second without touching the data. `typst` has to be on the path for that last step.

## Inputs

- `work/relax-big-mofs/output/`, the relaxed MOF-177 geometry, reached through `work/hessians/common.py`, which `run.py` and `overview_extract.py` import directly.
- `sources/processed/pet-mad-xs/`, the converted checkpoint.
- `work/hops`, where `K = 5` for PET-XS comes from.

## Usage

```bash
JAX_PLATFORMS=cpu uv run python run.py --structure mof177
JAX_PLATFORMS=cpu uv run python overview_extract.py --structure mof177
uv run --extra plots python overview_figure.py
uv run --extra plots python overview_figure.py --centre ATOM
```

Everything here runs on CPU on a laptop, with no cluster involved. `run.py` resumes by `record.json` existence, so it is a no-op once the Hessian is there. Only the last step needs matplotlib and Typst, so the figure rebuilds from the committed `results/` without rerunning the Hessian.

## Output

`output/mof177/` holds `record.json` (engine config, system size, the number of colors and the fill per rung, provenance), `hessian.npz` (the exact sparse Hessian) and `graph.npz` (positions, cell, atomic numbers, one-hop edges, hop matrix). One unit cell is small enough that we ship the Hessian. `output/build/` holds the data-bound decisions handed to Typst, `structure.json`, `matrix.png`, `toy.json` and `style.json`.

`results/` is committed and small. `mof177.npz` holds positions, cell, atomic numbers, one-hop edges, the hop matrix and the block norms of the exact Hessian. `mof177.json` holds the number of colors and the fill per rung `k = 1 … 5`, the per-hop block-norm statistics, and provenance. `toy.json` holds the toy chain's colorings and compressed products.

`figures/` holds `overview.pdf` and `overview.png`.

## Related

- `work/hessians`, the production Hessians and the construction this experiment imports.
- `work/hops`, where `K = 5` for PET-XS comes from.
- `work/relax-big-mofs`, the geometry.
