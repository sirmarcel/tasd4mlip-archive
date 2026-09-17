"""`results/matmul.json` -> pair rows and the setting they were measured at.

The loader shared by every matmul consumer (`matmul_table.py`, future figures).
Rows are the extraction's fields verbatim; grouping and rounding happen in the
consumers.
"""

from pathlib import Path

from sadmof.io import read_json

RESULTS = Path(__file__).resolve().parent / "results" / "matmul.json"


def dataset():
    """The whole extracted dataset, including `setting` and `provenance`."""
    if not RESULTS.exists():
        raise SystemExit(f"no {RESULTS} yet — run matmul_extract.py")
    return read_json(RESULTS)


def pairs():
    """Every condition row, both rungs folded in."""
    return dataset()["pairs"]


def by_model(rows=None):
    """`{model: [row, ...]}`, rows sorted by system size."""
    grouped = {}
    for row in pairs() if rows is None else rows:
        grouped.setdefault(row["model"], []).append(row)
    return {m: sorted(rows, key=lambda r: r["n_atoms"]) for m, rows in grouped.items()}


def overhead_pct(row, regime="warm"):
    """What pinning `highest` costs, in percent of the TF32 time.

    Positive means TF32 is the faster of the two, i.e. what TF32 buys.
    """
    value = row.get(f"{regime}_ratio")
    return None if value is None else (value - 1.0) * 100.0
