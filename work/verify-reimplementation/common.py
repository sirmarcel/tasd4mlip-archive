"""Shared setup for the parity workers: roster, checkpoints, condition layout.

The two stacks never share a process — the torch references run under a `uv run
--with` overlay, the JAX predictions in the project env — so everything both
sides must agree on (which structures, which checkpoint, where a record goes)
lives here rather than in either worker.
"""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "output"

STRUCTURES = ROOT / "sources" / "processed" / "goennheimer+moosavi" / "structures.xyz"
RANKING = ROOT / "work" / "select-goennheimer" / "output" / "ranking.csv"

MODELS = ("mace", "pet-xs", "pet-s")
DTYPES = ("float64", "float32")
STACKS = ("torch", "jax")
DEVICES = ("cpu", "gpu")

# What `JAX_PLATFORMS` has to be set to for each. The GPU entry keeps `cpu`
# available behind `cuda`: pet-jax pins its k_sel sizing kernel to a CPU device
# (`select.determine_k_sel`), so a CUDA-only backend list makes every PET
# condition die on `Unknown backend cpu`. First entry is the default device, so
# the forward still runs on the GPU, and `jax_worker` reads the device back to
# catch a silent fall-through to CPU.
PLATFORM = {"cpu": "cpu", "gpu": "cuda,cpu"}

# JAX fp32 matmul accumulation modes: a no-op on CPU, the measured axis on an
# accelerator, where `default` is TF32. Pinned per condition rather than
# inherited, because the model families disagree -- `UPETCalculator` defaults to
# `high` and mutates global JAX config, our MACE calculator takes JAX's default.
MATMULS = ("default", "high", "highest")

# The reference every comparison is taken against: the upstream implementation
# at the precision that defines the model, independent of anything we wrote.
REFERENCE = ("torch", "float64")

# JAX side: the converted checkpoints, by `sadmof.paths` name.
CHECKPOINTS = {
    "mace": "mace-mp-0-medium",
    "pet-xs": "pet-mad-xs",
    "pet-s": "pet-mad-s",
}

# Torch side: the upstream artifacts those conversions were built from.
RAW = {
    "mace": ROOT / "sources/raw/mace-mp-0-medium/2023-12-03-mace-128-L1_epoch-199.model",
    "pet-xs": ROOT / "sources/raw/pet-mad-xs/pet-mad-xs-v1.5.0.ckpt",
    "pet-s": ROOT / "sources/raw/pet-mad-s/pet-mad-s-v1.5.0.ckpt",
}

# `uv run --with` overlay each torch reference needs on top of the project env.
# mace-torch and metatrain are kept in separate overlays: they are separate
# resolutions, and nothing needs them at once.
OVERLAY = {
    "mace": ("mace-torch",),
    "pet-xs": ("metatrain", "metatomic-torch"),
    "pet-s": ("metatrain", "metatomic-torch"),
}

# Verlet skin for both JAX calculators. Geometries are static here, so it only
# sizes the cached neighbour list and never changes a prediction.
SKIN = 0.5

# Distributions whose version pins a parity number, on top of `provenance`'s
# default sadmof + JAX stack. Absent ones are skipped, so one list covers both
# workers.
PACKAGES = ("torch", "mace-torch", "metatrain", "metatomic-torch")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def roster(spec):
    """Ordered structure identifiers for a roster spec.

    Args:
        spec: Iterable of `goenn:<k>` (first k rows of the committed
            `work/select-goennheimer` ranking), `all` (every structure in the
            dataset, dataset order), or an explicit identifier.

    Returns:
        List of identifiers, deduplicated, in spec order.
    """
    out = []
    for item in spec:
        if item.startswith("goenn:"):
            k = int(item.split(":", 1)[1])
            with RANKING.open() as f:
                out += [row["structure"] for row in csv.DictReader(f)][:k]
        elif item == "all":
            out += list(frames())
        else:
            out.append(item)
    return list(dict.fromkeys(out))


def frames():
    """`{identifier: Atoms}` for the whole dataset, in file order, unrelaxed."""
    from ase.io import read

    return {
        f.info["identifier"]: f for f in read(str(STRUCTURES), index=":", format="extxyz")
    }


def select(structures):
    """`[(identifier, Atoms)]` for `structures`, in the order given.

    Raises:
        KeyError: If an identifier is not in the dataset, naming it.
    """
    available = frames()
    missing = [s for s in structures if s not in available]
    if missing:
        raise KeyError(f"not in {STRUCTURES.name}: {missing}")
    return [(s, available[s]) for s in structures]


def run_condition(calc, structures, out_dir, *, meta):
    """Predict energy / forces / stress for each structure into `out_dir`.

    A structure whose `.npz` exists is skipped; `meta.json` is rewritten each
    call, so it describes the process that last touched the condition.
    """
    import numpy as np

    import time

    from sadmof.io import write_json, write_npz
    from sadmof.provenance import DEFAULT_PACKAGES, provenance

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    done = 0
    for identifier, atoms in select(structures):
        path = out_dir / f"{identifier}.npz"
        if path.exists():
            continue
        atoms = atoms.copy()
        atoms.calc = calc
        t0 = time.time()
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress(voigt=False)
        seconds = time.time() - t0
        # Widened to fp64 on write: an fp32 prediction is stored exactly, and no
        # later comparison has to round one side to reach the other.
        write_npz(
            path,
            {
                "identifier": identifier,
                "n_atoms": np.int32(len(atoms)),
                "energy": np.float64(energy),
                "forces": np.asarray(forces, dtype=np.float64),
                "stress": np.asarray(stress, dtype=np.float64),
                "seconds": np.float64(seconds),
            },
        )
        done += 1
        print(
            f"  {identifier}: N={len(atoms):4d} E={energy:.8f} eV ({seconds:.2f}s)",
            flush=True,
        )

    write_json(
        out_dir / "meta.json",
        meta
        | {
            "structures": list(structures),
            "computed": done,
            "seconds": time.time() - started,
            "provenance": provenance(packages=(*DEFAULT_PACKAGES, *PACKAGES)),
        },
    )
    print(f"  {done} computed, {len(structures) - done} already present")


def variant(dtype, device="cpu", matmul=None):
    """The directory name for one precision configuration.

    Only what departs from the CPU baseline is spelled out, so the original
    CPU conditions keep the names their records already live under: `float32`,
    then `float32_gpu_mmhigh`, `float64_gpu`. Matmul mode is part of the name
    because on an accelerator it changes the arithmetic.
    """
    parts = [dtype]
    if device != "cpu":
        parts.append(device)
    if matmul is not None:
        parts.append(f"mm{matmul}")
    return "_".join(parts)


def condition_dir(stack, model, dtype, device="cpu", matmul=None):
    return OUTPUT / stack / model / variant(dtype, device, matmul)


def variants_present(model):
    """`[(stack, variant)]` on disk for `model`, so the extraction picks up new
    precision configurations without being told about them."""
    out = []
    for stack in STACKS:
        root = OUTPUT / stack / model
        if root.is_dir():
            out += [(stack, p.name) for p in sorted(root.iterdir()) if p.is_dir()]
    return out


def read_record(stack, model, variant_name, structure):
    """One prediction as `{energy, forces, stress, n_atoms}`, all fp64.

    Addressed by the variant's directory name (`float32`, `float32_gpu_mmhigh`,
    ...), so a consumer can read whatever `variants_present` reports without
    reconstructing how it was configured. Names must not contain `/`:
    `parity_extract` splits its keys on it.
    """
    import numpy as np

    with np.load(OUTPUT / stack / model / variant_name / f"{structure}.npz") as z:
        return {
            "energy": float(z["energy"]),
            "forces": np.asarray(z["forces"], dtype=np.float64),
            "stress": np.asarray(z["stress"], dtype=np.float64),
            "n_atoms": int(z["n_atoms"]),
        }
