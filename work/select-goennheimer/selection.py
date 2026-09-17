"""Nested benchmark subset selection: one frozen class-stratified FPS ranking.

Within each structure class, ranks structures by greedy maximin (Kennard-Stone)
distance in a rank-transformed feature space; the per-class rankings are
interleaved by a fixed 8:1:1 (MOF:COF:zeolite) quota, so every roster of size k
is the first k rows and minority classes stay represented at any size. Method
details and feature rationale: README.md.

    uv run python selection.py output/records --out output/ranking.csv
"""

import numpy as np

import argparse
import csv
import json
from pathlib import Path

FEATURES = [
    "log_n_atoms",
    "number_density",
    "minconv_fill",
]

# Per block of ten; the single "unknown"-class structure rides in the MOF
# stratum. Interleave picks the class with the largest quota deficit,
# tie-broken by this dict's order.
SHARES = {"MOF": 8, "COF": 1, "zeolite": 1}


def features(record):
    return [
        np.log(record["n_atoms"]),
        record["number_density"],
        record["minconv_fill"],
    ]


def fps_order(ranks, subset):
    """Greedy maximin (L2) over `subset` rows, seeded at the subset medoid."""
    R = ranks[subset]
    order = [int(np.argmin(np.linalg.norm(R - np.median(R, axis=0), axis=1)))]
    d = np.linalg.norm(R - R[order[0]], axis=1)
    for _ in range(len(subset) - 1):
        nxt = int(np.argmax(d))  # argmax is stable: first index wins ties
        order.append(nxt)
        d = np.minimum(d, np.linalg.norm(R - R[nxt], axis=1))
    return [subset[i] for i in order]


def interleave(per_class):
    """Merge per-class rankings by largest deficit against the SHARES quota.

    A class that runs out simply stops competing; the fixed target fractions
    make the sequence deterministic and prefix-stable.
    """
    total_share = sum(SHARES.values())
    counts = {c: 0 for c in per_class}
    merged = []
    while any(counts[c] < len(per_class[c]) for c in per_class):
        avail = [c for c in per_class if counts[c] < len(per_class[c])]
        deficit = lambda c: SHARES[c] / total_share * (len(merged) + 1) - counts[c]
        c = max(avail, key=lambda c: (deficit(c), -list(SHARES).index(c)))
        merged.append(per_class[c][counts[c]])
        counts[c] += 1
    return merged


def ranking(records):
    """The frozen order: (name, class, feature values) rows, best rank first."""
    names = [r["structure"] for r in records]
    classes = [
        c if (c := r.get("structure_type", "unknown")) in SHARES else "MOF"
        for r in records
    ]
    X = np.array([features(r) for r in records])
    ranks = np.argsort(np.argsort(X, axis=0), axis=0) / (len(X) - 1)

    per_class = {
        c: fps_order(ranks, [i for i, ci in enumerate(classes) if ci == c])
        for c in SHARES
    }
    return [(names[i], classes[i], X[i]) for i in interleave(per_class)]


def load_records(record_dir):
    return [json.loads(p.read_text()) for p in sorted(Path(record_dir).glob("*.json"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", help="measurement record directory")
    ap.add_argument("--out", default="output/ranking.csv")
    args = ap.parse_args()

    rows = ranking(load_records(args.records))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "structure", "class", *FEATURES])
        for rank, (name, cls, x) in enumerate(rows, 1):
            writer.writerow([rank, name, cls, *[f"{v:.6g}" for v in x]])
    print(f"{len(rows)} structures -> {out}")
    print("first 10:", [f"{name} ({cls})" for name, cls, _ in rows[:10]])


if __name__ == "__main__":
    main()
