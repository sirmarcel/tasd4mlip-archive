# pet-mad-s

pet-jax (Flax) checkpoint for PET-MAD "s" (v1.5.0), converted from the upstream metatrain `.ckpt`. This is the larger of the two PET variants used here (`num_gnn_layers=3`, `cutoff=8.0`, `num_neighbors_adaptive=16`), PET-MAD-XS aims for only 8 neighbours.

`metadata.yaml` is the config, included in this archive: architecture hypers and per-element composition `shifts` keyed by atomic number. `energy_scale` lives in the parameter tree, and embeddings are indexed by atomic number directly.

**`model.msgpack` (the Flax parameter tree, about 100 MB, loadable with `petjax.load_checkpoint` or `UPETCalculator.from_checkpoint`) is not included in this archive.** Rebuild it as below before running any experiment that uses this model.

## Rebuilding

`petjax-convert` downloads the `.ckpt` from Hugging Face (`lab-cosmo/upet`) into `--cache` and converts directly to `model.msgpack` and `metadata.yaml`, with no TorchScript intermediate. The conversion dependencies (`torch`, `metatomic-torch`, `metatrain`) are not project dependencies, uv pulls them in transiently. Run from this directory:

```bash
uv run --with metatrain --with metatomic-torch \
    petjax-convert pet-mad-s --out . --cache ../../raw/pet-mad-s
```

The conversion is idempotent: an existing `model.msgpack` short-circuits it.

## Checksum

sha256 of `model.msgpack` as used by every experiment under `work/`: `339c42b7efc5c6f96bf44a441a0ce1c96baf3a77e5c78a932b1a7bc8ee44a29a`. A rebuilt checkpoint should match. If it does not, the upstream weights or the converter changed.
