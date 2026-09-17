"""Aggregate the perf-settings results and derive the locked setting (stdlib-only).

    python collect.py            # tables + the derived setting
    python collect.py --write    # also write output/settings.json

Implements the decision rule and memory-floor model specified in the README.
`fits` is three-valued (yes / no / unknown); unknown is never feasible.
"""

import argparse
import json
import sys
from pathlib import Path

from run import CERT_EXTREMES, MEDIAN

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

# 1.0: the floor addition is already conservative and probes run on the
# production-identical relaxed cells; RSM1885 fits by 0.3% and certify is the
# proof. (Was 0.85 pre-lock; decided 2026-08-18: a working single setting
# beats margin.)
HEADROOM = 1.0
PRODUCTION_HOPS = {"mace": 2, "pet-xs": 3, "pet-s": 3}  # floor joined per model

GIB = 1024**3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true")
    args = p.parse_args()

    patterns = load(OUTPUT / "patterns")
    probes = load(OUTPUT / "probe")
    certify = load(OUTPUT / "certify")
    # The matmul rider lives in its own tree and never reaches `derive`: it is a
    # measurement about the lock, not an input to it (README, Phases).
    matmul = load(OUTPUT / "matmul")
    device = single_device(patterns, probes, certify, matmul)

    for p_ in patterns:
        p_["floor_bytes"] = floor_bytes(p_)
        p_["floor_gib"] = gib(p_.get("floor_bytes"))
        if p_.get("coloring_s") is not None:
            p_["coloring_s"] = round(p_["coloring_s"], 1)
    for r in probes + matmul:
        annotate(r, patterns)
    for c in certify:
        # measured end to end, so no floor model — the peak is the whole story
        c["peak_gib"] = gib(c.get("peak_bytes"))
        c["limit_gib"] = gib(c.get("bytes_limit"))

    show(
        "probe",
        probes,
        [
            "structure",
            "model",
            "mode",
            "chunk_size",
            "remat",
            "padding",
            "status",
            "ms_per_hvp",
            "peak_gib",
            "adjusted_gib",
            "fits",
        ],
    )
    show(
        "patterns",
        patterns,
        [
            "structure",
            "model",
            "hops",
            "padding",
            "n_atoms",
            "nnz",
            "num_colors",
            "coloring_s",
            "floor_gib",
        ],
    )
    show(
        "certify",
        certify,
        [
            "structure",
            "model",
            "hops",
            "chunk_size",
            "status",
            "num_colors",
            "hessian_cold_s",
            "peak_gib",
            "limit_gib",
            "frobenius",
        ],
    )

    show(
        "matmul rider (TF32 default vs full-fp32 `highest`, at the locked setting)",
        matmul_pairs(matmul),
        [
            "structure",
            "model",
            "padding",
            "status",
            "ms_default",
            "ms_highest",
            "warm_ratio",
            "cold_s_default",
            "cold_s_highest",
            "cold_ratio",
            "peak_default",
            "peak_highest",
        ],
    )

    # feasibility is judged within one padding regime; tight wins ties for the lock
    setting = {}
    for pad in sorted({r.get("padding", "bucketed") for r in probes}):
        s = derive([r for r in probes if r.get("padding", "bucketed") == pad])
        print(f"\nderived setting [{pad}]: {json.dumps(s, indent=1)}")
        if "chunk_size" in s or not setting:
            setting = s | {"padding": pad}
    if "chunk_size" in setting:
        # The lock is a property of one accelerator, so it travels with one.
        setting["device"] = device
    if args.write and "chunk_size" in setting:
        (OUTPUT / "settings.json").write_text(json.dumps(setting, indent=1))
        print("wrote output/settings.json")


def single_device(*groups):
    """The one accelerator every record came from; bail on a mixture.

    Timings and memory peaks are not comparable across cards, and the filenames
    carry no platform — so two clusters' outputs fetched into one tree
    interleave silently. Refuse rather than filter: selecting a subset here is
    the cherry-picking this exists to prevent.
    """
    seen = {}
    for rows in groups:
        for r in rows:
            seen.setdefault(device_of(r), []).append(r.get("structure"))
    known = {d: v for d, v in seen.items() if d is not None}
    if len(known) > 1:
        detail = "\n".join(
            f"  {d}: {len(v)} records ({', '.join(sorted(set(v))[:4])}…)"
            for d, v in sorted(known.items())
        )
        sys.exit(
            f"output/ mixes records from {len(known)} accelerators:\n{detail}\n"
            "Separate them into one tree per machine and collect each alone."
        )
    if None in seen:
        print(f"warning: {len(seen[None])} records do not name their accelerator")
    return next(iter(known), None)


def device_of(row):
    """Accelerator that produced a record, from provenance, falling back to
    worker.py's own `device`."""
    devices = (row.get("provenance") or {}).get("devices") or {}
    return devices.get("kind") or row.get("device")


def derive(probes):
    """The largest chunk that fits both extremes, at its median-system speed;
    fwd_over_rev first, then the rev_over_fwd fallback. Only `fits is True`
    counts — missing extreme rows and unresolved floors are not feasible."""
    if not any(r.get("model") == "mace" for r in probes):
        # e.g. the `production` regime, which carries only the PET sanity rows:
        # nothing to derive from, and not a failure to report as one.
        return {"error": "no MACE rows in this padding regime; nothing to derive"}
    for mode in ("fwd_over_rev", "rev_over_fwd"):
        rows = [
            r
            for r in probes
            if r.get("model") == "mace" and r.get("mode") == mode and r.get("remat")
        ]
        feasible = set()
        for chunk in sorted({r["chunk_size"] for r in rows}):
            at_extremes = [
                r
                for r in rows
                if r["chunk_size"] == chunk and r.get("structure") in CERT_EXTREMES
            ]
            covered = {r["structure"] for r in at_extremes}
            if covered == set(CERT_EXTREMES) and all(
                r["fits"] is True for r in at_extremes
            ):
                feasible.add(chunk)
        if not feasible:
            continue
        ranked = sorted(
            (
                r
                for r in rows
                if r.get("structure") == MEDIAN
                and r["chunk_size"] in feasible
                and r.get("status") == "ok"
            ),
            key=lambda r: r["ms_per_hvp"],
        )
        if not ranked:
            return {
                "error": f"{mode}: no median-system rows for feasible chunks "
                f"{sorted(feasible)}"
            }
        best = ranked[0]
        return {
            "mode": mode,
            "chunk_size": best["chunk_size"],
            "remat": True,
            "precision": "float32",
            "headroom": HEADROOM,
            "feasible_chunks": sorted(feasible),
            "ms_per_hvp_at_median": round(best["ms_per_hvp"], 2),
        }
    return {"error": "no chunk fits both extremes in either mode; a human looks now"}


def annotate(r, patterns):
    """Join the device-floor onto a probe row and grade feasibility (3-valued)."""
    r["peak_gib"] = gib(r.get("peak_bytes"))
    key = (
        r.get("structure"),
        r.get("model"),
        PRODUCTION_HOPS.get(r.get("model")),
        r.get("padding", "bucketed"),
    )
    pat = next(
        (
            p
            for p in patterns
            if (
                p.get("structure"),
                p.get("model"),
                p.get("hops"),
                p.get("padding", "bucketed"),
            )
            == key
            and p.get("floor_bytes")
        ),
        None,
    )
    if r.get("status") in ("oom", "crashed"):
        r["adjusted_gib"] = None
        r["fits"] = False
    elif r.get("peak_bytes") and r.get("bytes_limit") and pat:
        floor_delta = pat["floor_bytes"] - r["n_probe"] * r["dim"] * 4
        r["adjusted_bytes"] = r["peak_bytes"] + max(floor_delta, 0)
        r["adjusted_gib"] = gib(r["adjusted_bytes"])
        r["fits"] = r["adjusted_bytes"] <= HEADROOM * r["bytes_limit"]
    else:
        # ran ok but the floor is unresolved (patterns record missing/failed)
        r["adjusted_gib"] = None
        r["fits"] = None


def matmul_pairs(rows):
    """Join the two matmul rungs of each condition into one row.

    `warm_ratio` is the answer the phase exists for — steady-state per-HVP, the
    regime a production Hessian's ~10^3 HVPs live in. `cold_ratio` (compile +
    one full sweep) is the regime a single unit-cell Hessian sees, and is the
    number the hessian-ablations diagnostic measured.
    """
    keys = ("structure", "model", "padding", "mode", "chunk_size", "remat")
    pairs = {}
    for r in rows:
        key = tuple(r.get(k) for k in keys)
        pairs.setdefault(key, {})[r.get("matmul_precision", "default")] = r

    out = []
    for key, rungs in sorted(pairs.items(), key=lambda kv: [str(x) for x in kv[0]]):
        row = dict(zip(keys, key))
        row["status"] = "/".join(
            rungs[n].get("status", "-") if n in rungs else "missing"
            for n in ("default", "highest")
        )
        for name in ("default", "highest"):
            r = rungs.get(name)
            row[f"ms_{name}"] = r.get("ms_per_hvp") if r else None
            row[f"cold_s_{name}"] = cold_s(r)
            row[f"peak_{name}"] = r.get("peak_gib") if r else None
        row["warm_ratio"] = ratio(row["ms_highest"], row["ms_default"])
        row["cold_ratio"] = ratio(row["cold_s_highest"], row["cold_s_default"])
        out.append(row)
    return out


def cold_s(row):
    """One full sweep including its compile — the probe times both separately."""
    if not row or row.get("compile_s") is None or row.get("sweep_s") is None:
        return None
    return round(row["compile_s"] + row["sweep_s"], 2)


def ratio(numerator, denominator):
    return round(numerator / denominator, 2) if numerator and denominator else None


def floor_bytes(pat):
    """Recomputed from the record's fields so stale stored values cannot leak in."""
    if all(pat.get(k) for k in ("num_colors", "dim", "nnz")):
        return 2 * pat["num_colors"] * pat["dim"] * 4 + 36 * pat["nnz"]
    return pat.get("floor_bytes")


def load(directory):
    rows = [json.loads(f.read_text()) for f in sorted(directory.glob("*.json"))]
    return rows


def gib(nbytes):
    return round(nbytes / GIB, 1) if nbytes else None


def show(title, rows, cols):
    if not rows:
        print(f"\n== {title}: nothing yet")
        return
    print(f"\n== {title}")
    table = [[fmt(r.get(c)) for c in cols] for r in rows]
    widths = [max(len(c), *(len(t[i]) for t in table)) for i, c in enumerate(cols)]
    print("  " + "  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for t in table:
        print("  " + "  ".join(v.ljust(w) for v, w in zip(t, widths)))


def fmt(v):
    if isinstance(v, float):
        return f"{v:.2f}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return "-" if v is None else str(v)


if __name__ == "__main__":
    main()
