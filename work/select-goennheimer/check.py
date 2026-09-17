"""Post-selection checks on the frozen ranking.

Hard-fails (AssertionError) if the committed ranking.csv does not match a
recomputation from the records, if the class quotas are violated, or if the
feature-redundancy fact behind the README's dropped-axes argument no longer
holds. Everything else is a report: class/element makeup and population
coverage of roster prefixes, on the selection features and the dropped ones.

    uv run python check.py output/records output/ranking.csv
"""

import numpy as np

import argparse
import csv
from pathlib import Path

from selection import FEATURES, SHARES, load_records, ranking

PREFIXES = (10, 20, 50)
DROPPED = ["w_min", "anisotropy", "reach_per_atom"]


def dropped_features(record):
    w = np.array(record["widths"])
    return [w.min(), w.max() / w.min(), record["reach_per_atom"]]


def _require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", help="measurement record directory")
    ap.add_argument("ranking", help="frozen ranking.csv")
    args = ap.parse_args()

    records = load_records(args.records)
    by_name = {r["structure"]: r for r in records}
    rows = ranking(records)

    with Path(args.ranking).open() as fh:
        frozen = [row["structure"] for row in csv.DictReader(fh)]

    # The freeze: the committed ranking must be exactly what the records give.
    _require(len(frozen) == len(rows), "ranking.csv row count != record count")
    for i, (name, _, _) in enumerate(rows):
        _require(
            frozen[i] == name,
            f"ranking.csv diverges at rank {i + 1}: {frozen[i]} != {name}",
        )

    # Class quotas per block of ten, while every class has members left.
    classes = [cls for _, cls, _ in rows]
    n_class = {c: classes.count(c) for c in SHARES}
    for k in PREFIXES:
        counts = {c: classes[:k].count(c) for c in SHARES}
        for c, share in SHARES.items():
            expected = min(share * k // 10, n_class[c])
            _require(
                counts[c] == expected,
                f"prefix {k}: {counts[c]} {c} rows, quota says {expected}",
            )

    # The README drops reach/atom as an axis because it is rank-degenerate with
    # density on this population; keep that claim tied to the data.
    density = np.array([by_name[n]["number_density"] for n, _, _ in rows])
    reach = np.array([by_name[n]["reach_per_atom"] for n, _, _ in rows])
    rho = spearman(density, reach)
    _require(rho > 0.99, f"density/reach rank correlation {rho:.3f} <= 0.99")

    # Report: makeup and coverage.
    X = np.array([x for _, _, x in rows])
    D = np.array([dropped_features(by_name[n]) for n, _, _ in rows])
    all_names, all_vals = FEATURES + DROPPED, np.hstack([X, D])
    pop_rank = np.argsort(np.argsort(all_vals, axis=0), axis=0) / (len(rows) - 1)

    print(f"{len(rows)} structures; classes {n_class}; "
          f"spearman(density, reach/atom) = {rho:.3f}")
    for k in PREFIXES:
        counts = {c: classes[:k].count(c) for c in SHARES}
        elements = sorted({e for n, _, _ in rows[:k] for e in by_name[n]["elements"]})
        print(f"\nprefix {k}: {counts}, {len(elements)} elements")
        for j, name in enumerate(all_names):
            lo, hi = pop_rank[:k, j].min(), pop_rank[:k, j].max()
            tag = "" if j < len(FEATURES) else "  (not a selection axis)"
            print(f"  {name:>16s}: rank-span {lo:.2f}-{hi:.2f}"
                  f"  [{all_vals[:k, j].min():.4g}, {all_vals[:k, j].max():.4g}]{tag}")
    print("\nOK")


if __name__ == "__main__":
    main()
