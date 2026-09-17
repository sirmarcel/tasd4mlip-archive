"""`results/parity.json` -> rows and reduced deviations, for every parity
consumer."""

import numpy as np

from pathlib import Path

from sadmof.io import read_json

RESULTS = Path(__file__).resolve().parent / "results"

# The paper's four rows, in table order: the upstream precision scale bar
# first, so the two that follow are read against it rather than against zero.
COMPARISONS = (
    "torch/float32",
    "jax/float64",
    "jax/float32",
    "jax/float32_gpu_mmdefault",
)

COMPARISON_NAMES = {
    "torch/float32": (r"\torch{}", "fp32"),
    "jax/float64": (r"\jax{}", "fp64"),
    "jax/float32": (r"\jax{}", "fp32"),
    "jax/float32_gpu_mmdefault": (r"\jax{}", "fp32, TF32"),
}


def all_comparisons(data):
    """Every comparison in the extraction, the paper's three first.

    What follows them is whatever precision configurations were run, so a
    device or matmul rung shows up in the report without being declared here.
    """
    found = list(data.get("comparisons", []))
    known = [key for key in COMPARISONS if key in found]
    return known + [key for key in found if key not in known]


# The deviations a consumer reduces, and the unit each carries.
METRICS = {
    "de_per_atom": "eV/atom",
    "df_max": "eV/Å",
    "df_rms": "eV/Å",
    "dstress_max": "eV/Å³",
}


def parity():
    """The whole extraction: `{"reference", "roster", "models", ...}`."""
    return read_json(RESULTS / "parity.json")


def rows(data=None):
    """`{(model, comparison): {metric: array}}`, deviations as absolute values.

    Signs are dropped here: every consumer asks how far apart the two stacks
    are, and a signed energy shift that cancels across a roster would make a
    mean look like agreement.
    """
    data = data if data is not None else parity()
    out = {}
    for model, comparisons in data["models"].items():
        for comparison, row in comparisons.items():
            arrays = {
                key: np.abs(np.asarray(row[key], dtype=np.float64)) for key in METRICS
            }
            arrays["structures"] = list(row["structures"])
            arrays["n_atoms"] = np.asarray(row["n_atoms"], dtype=int)
            out[(model, comparison)] = arrays
    return out


def reduce(arrays, how="max"):
    """One number per metric: `max` (the worst structure) or `median`.

    An empty roster gives `None`, which `tables.rounder` renders as a dash.
    """
    fn = {"max": np.max, "median": np.median}[how]
    return {
        metric: (float(fn(arrays[metric])) if len(arrays[metric]) else None)
        for metric in METRICS
    }


def worst(arrays, metric):
    """`(structure, value)` of the structure that deviates most in `metric`."""
    if not len(arrays[metric]):
        return None, None
    i = int(np.argmax(arrays[metric]))
    return arrays["structures"][i], float(arrays[metric][i])
