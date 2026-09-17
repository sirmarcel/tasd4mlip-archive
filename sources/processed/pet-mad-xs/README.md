# pet-mad-xs

pet-jax (Flax) checkpoint for PET-MAD "xs" (v1.5.0), converted from the upstream metatrain `.ckpt`.

`metadata.yaml` is the config, included in this archive: architecture hypers (`num_gnn_layers=2`, `cutoff=7.5`, `num_neighbors_adaptive=8`, and the rest) and per-element composition `shifts` keyed by atomic number. `energy_scale` lives in the parameter tree, and embeddings are indexed by atomic number directly, with no `species_to_index`.

**`model.msgpack` (the Flax parameter tree, about 17 MiB, loadable with `petjax.load_checkpoint` or `UPETCalculator.from_checkpoint`) is not included in this archive.** It is fp32, since PET-MAD ships fp32, and is promoted to fp64 for Hessian work. Rebuild it as below before running the tests or any experiment that uses this model.

## Rebuilding

`petjax-convert` downloads the `.ckpt` from Hugging Face (`lab-cosmo/upet`) into `--cache` and converts directly, with no TorchScript intermediate. The conversion dependencies (`torch`, `metatomic-torch`, `metatrain`) are not project dependencies, uv pulls them in transiently. Run from this directory:

```bash
uv run --with metatrain --with metatomic-torch \
    petjax-convert pet-mad-xs --out . --cache ../../raw/pet-mad-xs
```

The conversion is idempotent: an existing `model.msgpack` short-circuits it.

## Checksum

sha256 of `model.msgpack` as used by every experiment under `work/`: `b0e6b8b572756d4d4614896e21bb739cec0de97ec323521b0a9f7c79c232a9a3`. A rebuilt checkpoint should match. If it does not, the upstream weights or the converter changed.
