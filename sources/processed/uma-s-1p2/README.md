# uma-s-1p2

Torch-free conversion of [UMA-S-1.2](https://huggingface.co/facebook/UMA) (Meta FAIR, [fairchem](https://github.com/facebookresearch/fairchem)) for `sadmof.models.uma`, a pure-JAX port of the eSCN-MD MoE backbone and its MLP energy head. The preprint does not use this model. The port ships with the package, and the two tests that need the checkpoint skip without it.

The model is gated: accept the FAIR Chemistry License v1 on Hugging Face to download it, and acknowledge the models in any publication that uses them.

`config.json` is included: hyperparameters, per-task energy normalisers, dataset lists and the dataset to head-expert map. **`model.npz`, every weight and buffer as flat `/`-joined keys in fp32, about 1.2 GB, is not included in this archive.**

## Conversion

Three things in `build.py` are not a plain copy of the upstream state dict. The MOLE experts are left unmerged, since merging is composition-specific and happens at load time in `sadmof.models.uma.load`. The Gaussian basis is stored as a recipe (`gaussian_basis_width`) rather than an array, because upstream rebuilds that buffer at the inference precision. The energy normaliser and element references are pulled out of the per-task objects, and the port applies the scale inside the model and leaves the mean and references to the calculator.

## Rebuilding

Stage the upstream checkpoint (Hugging Face `facebook/UMA`, file `uma-s-1p2.pt`, about 2.3 GB) at `../../raw/uma/checkpoints/uma-s-1p2.pt`, then run the conversion from this directory. It is a `uv run` script with inline metadata, so fairchem, torch and their dependencies are pulled in transiently. About two minutes on a laptop.

```bash
uv run build.py
```

fairchem is pinned in `build.py` to the `n-gao/fairchem@ng/sybmolic_shapes` fork. Nothing in the conversion depends on that fork's symbolic-shape machinery, so upstream fairchem would likely work too, untested.
