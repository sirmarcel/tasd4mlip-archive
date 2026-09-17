# hops

How many graph hops does the positions-Hessian of a message-passing MLIP span? That hop count `K` makes `sadmof.sparse.sparsity_pattern(graph, hops=K)` exact rather than approximate, and it caps the truncation hierarchy `k = 1 … K`. This experiment confirms the hop-count law derived in the preprint and supplies the measured `K` per model that every other experiment in this archive uses. Trust, but verify...

## Scope and method

The law is `K = 2L` for a node readout and `K = 2L + 1` for an edge readout, with `L` the number of message-passing layers. `law.py` checks it over a sweep of `L` in `{1, 2, 3, 4}` on four hand-built graph families, `path`, `ring`, `multishell` and `caterpillar`, of 24 atoms each, with every bond at 0.45 of the model cutoff. We build the graphs directly rather than through the neighbour-list machinery, so the measurement is about the architecture and nothing else.

There are two passes. Freshly initialised models at each depth first, over seeds `{1, 2, 3}`, since random weights make every structurally allowed coupling generically nonzero. Then the production checkpoints on the same graphs, which rule out accidental cancellation and record the per-hop decay. We run everything in fp64. The random pass has little dynamic range at the outer hops, since the couplings of a freshly initialised model shrink by tens of orders of magnitude per added layer, so the sweep is bounded by fp64 rather than by anything structural. Underflow would show up as a hop-count mismatch rather than as a wrong answer quietly accepted.

The measured hop counts:

| checkpoint | readout | `L` | `K` |
|---|---|---|---|
| MACE-MP-0 medium | node | 2 | 4 |
| PET-MAD-XS | edge | 2 | 5 |
| PET-MAD-S | edge | 3 | 7 |

CPU only, no structures and no cluster, but not quick: the full default sweep is 108 Hessians and runs for close to an hour. `--families path --seeds 1` is the fast smoke test.

## Inputs

- `sources/processed/mace-mp-0-medium/`, `sources/processed/pet-mad-xs/`, `sources/processed/pet-mad-s/`, the converted checkpoints.

No other experiment feeds this one.

## Usage

```bash
JAX_PLATFORMS=cpu uv run python law.py
JAX_PLATFORMS=cpu uv run python law.py --families path ring --seeds 1 2
JAX_PLATFORMS=cpu uv run python law.py -o output/law.json
uv run python results.py
```

`law.py` writes `output/law.json` and prints the table. It exits nonzero if a measured count contradicts the law, if hop `K+1` is not a bit-exact zero, or if a graph's diameter fails to exceed `K`. The last check is there because a saturated pattern matches any hop count, which would make the measurement vacuous. A partial sweep overwrites the output, so keep `--families` and `--seeds` for smoke tests.

`results.py` is the document-facing extraction and uses the standard library only. It re-derives the confirmation from `output/law.json`, checking the law, the sharp boundary at `K+1`, that no graph saturated, and that the sweep was the full one rather than a smoke run, then prints the checkable facts as JSON including the table above. The preprint's claims are checked against `results.results()` rather than against `law.json` directly.

## Output

`output/law.json`, the full sweep, with the measured and predicted `K` per model, depth, family and seed for both passes. There is no `results/` or `figures/` directory. This experiment produces numbers rather than a figure, and `results.py` recomputes them on demand.

## Related

- `tests/test_hop_count.py` pins the same numbers through the production input pipelines, including PET's adaptive cutoff without shadow forces, periodicity, padding and `k_sel`. Those are library invariants, so the test suite enforces them rather than this experiment measuring them.
- `work/hessians`, where the exact pattern is checked to stay sparse at scale and the truncation hierarchy below `K` is measured.
- `work/overview`, which uses `K = 5` for PET-XS.
- We discuss the caveats that travel with `K` in the preprint: PET's number is the no-shadow one, D3 is excluded, and "exactly zero" is an fp64 statement.
