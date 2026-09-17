"""Aggregate relax_summary.json across output/<label>/<id>/ into one table.

    python collect.py            # every label under output/
    python collect.py LABEL ...  # only these labels

Writes output/summary.csv (one row per relaxation) and prints a per-label
convergence + timing aggregate.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
OUT = EXP / "output"

FIELDS = [
    "label",
    "identifier",
    "calculator",
    "optimizer",
    "dtype",
    "n_atoms",
    "n_steps",
    "converged",
    "fmax_final",
    "wall_s",
]


def main():
    labels = sys.argv[1:] or sorted(d.name for d in OUT.iterdir() if d.is_dir())

    rows = []
    for label in labels:
        for summary in sorted((OUT / label).glob("*/relax_summary.json")):
            d = json.loads(summary.read_text())
            row = {k: d.get(k) for k in FIELDS}
            row["label"] = label
            row["identifier"] = summary.parent.name
            rows.append(row)

    if not rows:
        sys.exit(f"no relax_summary.json found under {OUT}")

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    by_label: dict[str, list] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    print(
        f"{'label':36s} {'N':>4s} {'converged':>10s} {'med steps':>10s} {'wall sum/s':>11s}"
    )
    for label, rs in by_label.items():
        conv = sum(1 for r in rs if r["converged"])
        steps = sorted(r["n_steps"] for r in rs if r["n_steps"] is not None)
        med = steps[len(steps) // 2] if steps else 0
        wall = sum((r["wall_s"] or 0) for r in rs)
        print(f"{label:36s} {len(rs):4d} {conv:>6d}/{len(rs):<3d} {med:10d} {wall:11.0f}")

    print(f"\nwrote {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
