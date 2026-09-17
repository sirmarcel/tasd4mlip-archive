"""`output/matmul/` -> the matmul-precision dataset (`results/matmul.json`).

Pairs the two rungs of each `matmul` condition — TF32 (JAX's fp32 default on
Hopper, what production runs) against `highest` (full-fp32 matmuls) — and stores
what each cost, per system and model.

    python matmul_extract.py
    python matmul_extract.py --dry-run

`warm_ratio` is the quantity the phase exists for: steady-state per-HVP time,
the regime a production Hessian's ~10^3 HVPs live in, with compile and XLA
autotuning cancelled by the probe's two-sweep-length difference quotient.
`cold_ratio` (compile + one full sweep) is kept alongside because the cold
regime is what `work/hessian-ablations` measured, and the two disagree.

Ratios are stored unrounded; rounding is the consumers' business.
"""

import argparse
from pathlib import Path

from sadmof.io import read_json, write_json
from sadmof.provenance import provenance

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output" / "matmul"
RESULTS = HERE / "results"

RUNGS = ("default", "highest")
# What makes two records the same condition seen at two matmul precisions.
PAIR_KEYS = ("structure", "model", "padding", "mode", "chunk_size", "remat")
# Copied per rung, verbatim from the record.
RUNG_KEYS = (
    "status",
    "ms_per_hvp",
    "ms_per_hvp_raw",
    "fixed_s",
    "grad_s",
    "sweep_s",
    "compile_s",
    "peak_bytes",
    "bytes_limit",
    "n_probe",
)
# Condition-level facts that must agree across the pair.
SHARED_KEYS = (
    "supercell",
    "mult",
    "n_primitive",
    "n_atoms",
    "n_padded",
    "dim",
    "n_pairs_padded",
)


def main():
    args = build_parser().parse_args()
    records = [read_json(p) for p in sorted(OUTPUT.glob("*.json"))]
    if not records:
        raise SystemExit(f"nothing under {OUTPUT}")

    pairs = build_pairs(records)
    incomplete = [p for p in pairs if any(p[r] is None for r in RUNGS)]
    for pair in incomplete:
        missing = [r for r in RUNGS if pair[r] is None]
        print(f"skipping {pair['structure']}/{pair['model']}: no {', '.join(missing)} rung")
    pairs = [p for p in pairs if p not in incomplete]

    dataset = {
        "setting": read_json(HERE / "output" / "settings.json"),
        "device": single_device(records),
        "n_pairs": len(pairs),
        "pairs": pairs,
        "provenance": provenance(),
    }
    print(f"{len(records)} records -> {len(pairs)} pairs")
    for pair in pairs:
        print(
            f"  {pair['structure']:9s} {pair['model']:7s} {pair['padding']:11s} "
            f"warm {pair['warm_ratio']:.3f}  cold {pair['cold_ratio']:.3f}"
        )
    if args.dry_run:
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "matmul.json", dataset)
    print(f"wrote {RESULTS / 'matmul.json'}")


def build_pairs(records):
    """Group records by condition, one row per condition with both rungs."""
    grouped = {}
    for record in records:
        key = tuple(record.get(k) for k in PAIR_KEYS)
        rung = record.get("matmul_precision", "default")
        if rung in grouped.setdefault(key, {}):
            raise SystemExit(f"two `{rung}` records for {key}")
        grouped[key][rung] = record

    pairs = []
    for key, rungs in sorted(grouped.items(), key=lambda kv: [str(x) for x in kv[0]]):
        row = dict(zip(PAIR_KEYS, key))
        row |= shared(rungs.values())
        for name in RUNGS:
            record = rungs.get(name)
            row[name] = {k: record.get(k) for k in RUNG_KEYS} if record else None
        row["warm_ratio"] = ratio(row, "ms_per_hvp")
        row["cold_ratio"] = ratio(row, cold_s)
        row["peak_ratio"] = ratio(row, "peak_bytes")
        pairs.append(row)
    return pairs


def shared(records):
    """Condition-level facts, asserted equal across the rungs that carry them."""
    out = {}
    for key in SHARED_KEYS:
        values = {str(r.get(key)) for r in records if r.get(key) is not None}
        if len(values) > 1:
            raise SystemExit(f"rungs disagree on {key}: {sorted(values)}")
        out[key] = next((r.get(key) for r in records if r.get(key) is not None), None)
    return out


def cold_s(rung):
    """One full sweep including its compile — the probe times the two apart."""
    if rung.get("compile_s") is None or rung.get("sweep_s") is None:
        return None
    return rung["compile_s"] + rung["sweep_s"]


def ratio(row, quantity):
    """`highest` over `default`; None unless both rungs ran ok and reported it."""
    values = []
    for name in RUNGS:
        rung = row.get(name)
        if rung is None or rung.get("status") != "ok":
            return None
        value = quantity(rung) if callable(quantity) else rung.get(quantity)
        if not value:
            return None
        values.append(value)
    return values[1] / values[0]


def single_device(records):
    """The one accelerator every record came from; bail on a mixture, as the
    ratios are only meaningful within one card."""
    devices = {
        kind
        for r in records
        if (kind := ((r.get("provenance") or {}).get("devices") or {}).get("kind"))
    }
    if len(devices) > 1:
        raise SystemExit(f"records mix accelerators: {sorted(devices)}")
    return next(iter(devices), None)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="print, write nothing")
    return p


if __name__ == "__main__":
    main()
