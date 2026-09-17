"""`results/ablation/` -> condition rows, pair metrics, and spectra. The loader
shared by every ablation consumer (`ablation_table.py`, future figures).

Rows are the extraction's scalars verbatim; grouping and any derived statistics
happen in the consumers — the pair metrics are already relative quantities.
"""

import numpy as np

from pathlib import Path

from sadmof.io import read_json

RESULTS = Path(__file__).resolve().parent / "results" / "ablation"


def available():
    """Structure identifiers present in `results/ablation/`, sorted."""
    return sorted(p.stem for p in RESULTS.glob("*.json"))


def load(structure):
    """One structure's metadata; spectra stay in the `.npz` until `spectra`."""
    return read_json(RESULTS / f"{structure}.json")


def pairs():
    """Every extracted pair row, with the structure-level scalars folded in."""
    out = []
    for structure in available():
        meta = load(structure)
        for pair in meta["pairs"]:
            out.append({"structure": structure, "n_atoms": meta["n_atoms"], **pair})
    return out


def by_ablation():
    """`{(family, ablation): [row, ...]}`, rows in `available()` order."""
    grouped = {}
    for row in pairs():
        grouped.setdefault((row["family"], row["ablation"]), []).append(row)
    return grouped


def dcv_pct(row, temperature="300", asr=True):
    """Signed percent C_v deviation of the ablated condition, one temperature."""
    return row["dcv_pct"]["asr" if asr else "raw"].get(temperature)


def spectra(structure, variant, asr=True):
    """One condition's frequencies (cm^-1), one ASR variant."""
    key = f"freq_{'asr_' if asr else ''}{variant}"
    with np.load(RESULTS / f"{structure}.npz") as z:
        return np.asarray(z[key])
