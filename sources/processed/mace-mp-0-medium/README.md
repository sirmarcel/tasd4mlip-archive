# mace-mp-0-medium

Marathon-style Flax port of MACE-MP-0 "medium" (L1), produced from the upstream PyTorch checkpoint by `build.py`.

`model.yaml` is the model config, included in this archive. Its spec-dict handle is `sadmof.models.mace.model.MACE`, consumable by `marathon.io.from_dict`.

**`model.msgpack` (the Flax parameter tree, about 46 MB, loadable with `marathon.io.read_msgpack`) is not included in this archive.** Rebuild it as below before running the tests or any experiment.

## Conversion

The PyTorch state dict and the Flax tree differ in several details that `build.py` reconciles.

- **Element indexing.** The PyTorch model lists 89 elements in `atomic_numbers` and indexes per-element tensors by *position* in that list. The Flax tree indexes by Z directly, so per-element arrays are expanded from `(89, …)` to `(95, …)`, that is `max_z + 1`, with rows for uncovered Z left at zero. This affects node embedding, atomic energies, skip connections, and symmetric-contraction weights.
- **`path_weight` bake-in.** e3nn's `Linear` applies a per-instruction `sqrt(path_weight)` factor at apply time. The Flax `EquivariantLinear` does not, so the factor is folded into the saved weight instead. The node embedding's `path_weight` is also baked into the `Embed` table.
- **Skip TP rescaling.** The effective normalisation of the PyTorch `FullyConnectedTensorProduct` scales with `num_elements`. Expanding from 89 to 95 element slots requires rescaling by `sqrt(95 / 89)` to keep the numeric behaviour invariant.
- **U matrices** of the symmetric contractions are stored as Flax `variable("constants", ...)` rather than buffers.

## Rebuilding

Run from this directory. First stage the upstream PyTorch checkpoint, which is not included in this archive:

```bash
mkdir -p ../../raw/mace-mp-0-medium
curl -fL -o ../../raw/mace-mp-0-medium/2023-12-03-mace-128-L1_epoch-199.model \
    https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model
# expected sha256:
#   01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845
```

Then run the conversion. PyTorch and mace-torch are not project dependencies, uv pulls them in transiently. The build takes under a minute on a laptop.

```bash
uv run --with torch --with mace-torch --with pyyaml python build.py
```

## Checksum

sha256 of `model.msgpack` as used by every experiment under `work/`: `c52001034169b25dcea5c9c33c145708a5abaa65d735c83559954b20a1d319bf`. A rebuilt checkpoint should match. If it does not, the upstream weights or the converter changed.
