"""Aggregate the Hessian records into the paper-facing tables.

Reads `output/<structure>/<label>/record.json` — the lightweight product; the
Hessians and colorings stay on the cluster. Prints three tables: cost (the
`tab:cost` input), the C_v hop ladder (the `fig:truncation` input), and failures,
so a partial tree is legible rather than silently short.

Those tables see only conditions that produced a file. `--grid` answers the other
question — what the roster *should* contain and what is absent — by enumerating
the same grid `run.py` does and diffing it against the tree.

    python collect.py
    python collect.py --models mace --csv summary.csv
    python collect.py --grid              # every structure in the tree
    python collect.py --grid goenn:10     # a roster spec, as run.py
"""

import argparse
import csv
import sys
from pathlib import Path

import common

from sadmof.io import read_json

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

# Timing keys that partition the wall clock; `total_s` is the end-to-end number
# and `hessian_warm_s` is an opt-in extra call, so neither belongs in the sum.
STAGE_KEYS = (
    "import_s",
    "load_inputs_s",
    "pattern_s",
    "coloring_s",
    "hessian_cold_s",
    "extract_s",
    "save_s",
    "observables_s",
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", type=Path, default=OUTPUT)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--csv", type=Path, default=None, help="write the cost table")
    p.add_argument(
        "--grid",
        nargs="*",
        default=None,
        metavar="SPEC",
        help="print which conditions exist, then exit; bare --grid covers every "
        "structure in the tree, a SPEC is a run.py --roster spec",
    )
    args = p.parse_args()

    rows = load(args.input)
    if not rows:
        sys.exit(f"no records under {args.input}")
    if args.models:
        rows = [r for r in rows if r.get("model") in args.models]

    if args.grid is not None:
        show_grid(rows, args.grid, args.models or list(common.EXACT_HOPS))
        return

    device = single_device(rows, args.input)
    matmul = single_matmul(rows, args.input)
    print(f"{len(rows)} records, device: {device or 'unrecorded'}, matmul: {matmul}")

    annotate(rows)
    show_cost(rows)
    show_ladder(rows)
    show_failures(rows)
    if args.csv:
        write_csv(args.csv, rows)
        print(f"\nwrote {args.csv}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def show_cost(rows):
    """Per-condition cost: HVPs, colors, end-to-end and per-stage wall clock."""
    ok = [r for r in rows if r.get("status") == "ok"]
    show(
        "cost",
        sorted(ok, key=lambda r: (r["structure"], r["model"], r["_rung_order"])),
        [
            "structure",
            "model",
            "rung",
            "n_atoms",
            "n_hvps",
            "num_colors",
            "fill",
            "hessian_cold_s",
            "coloring_s",
            "total_s",
            "unaccounted_s",
            "peak_gib",
            "speedup",
            "cached",
        ],
    )


def show_ladder(rows):
    """C_v at 300 K against the same model's exact-K rung, per structure."""
    ok = [r for r in rows if r.get("status") == "ok" and r.get("cv300") is not None]
    show(
        "C_v(300 K) vs the model's exact-K rung",
        sorted(ok, key=lambda r: (r["structure"], r["model"], r["_rung_order"])),
        [
            "structure",
            "model",
            "rung",
            "cv300",
            "cv300_asr",
            "rel_err_pct",
            "rel_err_pct_asr",
            "n_dropped",
            "n_dropped_asr",
            "raw_asymmetry_rel",
        ],
    )


def show_failures(rows):
    """Records that exist but did not complete.

    Silent about conditions that were never run — they leave no file. Use
    `--grid` for that.
    """
    bad = [r for r in rows if r.get("status") != "ok"]
    if not bad:
        print("\n== failures: none among recorded conditions")
        return
    show(
        "failures: recorded but not ok",
        sorted(bad, key=lambda r: (r["structure"], r["model"], r["_rung_order"])),
        ["structure", "model", "rung", "status", "oom_where", "requested_gib", "flag"],
    )


def show_grid(rows, spec, models):
    """`run.py`'s expected grid against what is on disk.

    Says nothing about *why* a condition is absent; that is the run log's job.
    """
    structures = (
        common.roster(spec)
        if spec
        else sorted({r["structure"] for r in rows if r.get("structure")})
    )
    status = {}
    for r in rows:
        rung = "D" if r.get("mode") == "dense" else r.get("hops")
        status[(r.get("structure"), r.get("model"), rung)] = r.get("status")

    slots = {m: [*common.hop_ladder(m), "D"] for m in models}
    pad = max(len(x) for x in [*structures, "structure"])
    cell = {m: 2 * len(slots[m]) - 1 for m in models}

    scope = " ".join(spec) if spec else "everything recorded"
    print(f"\n== grid: {scope}   (rung = ok, x = failed, . = absent)")
    print(
        "  " + "structure".ljust(pad) + "  " + "   ".join(m.ljust(cell[m]) for m in models)
    )
    ruler = "   ".join(" ".join(str(x) for x in slots[m]).ljust(cell[m]) for m in models)
    print("  " + " " * pad + "  " + ruler)

    ok = failed = absent = 0
    for structure in structures:
        line = []
        for m in models:
            marks = []
            for slot in slots[m]:
                st = status.get((structure, m, slot))
                if st == "ok":
                    ok += 1
                    marks.append(str(slot))
                elif st is None:
                    absent += 1
                    marks.append(".")
                else:
                    failed += 1
                    marks.append("x")
            line.append(" ".join(marks))
        print("  " + structure.ljust(pad) + "  " + "   ".join(line))

    print(
        f"\n  {ok} ok, {failed} failed, {absent} never run, of "
        f"{ok + failed + absent} ({len(structures)} structures x {len(models)} models)"
    )


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def annotate(rows):
    """Flatten each record into the scalar columns the tables print."""
    for r in rows:
        cfg, sys_, sp = (
            r.get("config") or {},
            r.get("system") or {},
            r.get("sparsity") or {},
        )
        hess, obs = r.get("hessian") or {}, r.get("observables") or {}
        t, mem = r.get("timings") or {}, r.get("memory") or {}

        r["rung"] = "dense" if r.get("mode") == "dense" else f"h{r.get('hops')}"
        r["_rung_order"] = -1 if r.get("mode") == "dense" else (r.get("hops") or 0)
        r["n_atoms"] = sys_.get("n_atoms")
        r["n_hvps"] = hess.get("n_hvps")
        r["num_colors"] = sp.get("num_colors")
        r["fill"] = sp.get("fill")
        r["raw_asymmetry_rel"] = hess.get("raw_asymmetry_rel")
        r["cached"] = hess.get("cached")
        r["n_dropped"] = obs.get("n_dropped")
        r["n_dropped_asr"] = obs.get("n_dropped_asr")
        r["chunk_size"] = cfg.get("chunk_size")
        r["flag"] = aux_flag(r.get("aux_flag"))

        for k in ("hessian_cold_s", "coloring_s", "total_s"):
            r[k] = t.get(k)
        # Visible drift check: the timed stages should account for the wall
        # clock. Stages that legitimately did not run (pattern/coloring on a
        # dense rung) count as zero; a stage that stopped being *measured*
        # therefore inflates the residual rather than blanking it. A merged
        # coloring_s exceeds this run's wall clock by design.
        if t.get("total_s") is not None and not sp.get("coloring_reused"):
            r["unaccounted_s"] = t["total_s"] - sum(t.get(k) or 0.0 for k in STAGE_KEYS)
        else:
            r["unaccounted_s"] = None

        r["peak_gib"] = gib(mem.get("peak_bytes"))
        r["requested_gib"] = gib(r.get("requested_bytes"))
        r["cv300"] = (obs.get("cv_J_per_gK") or {}).get("300")
        r["cv300_asr"] = (obs.get("cv_J_per_gK_asr") or {}).get("300")

    _relative(rows)


def _relative(rows):
    """`speedup` vs the same (structure, model) dense rung; `rel_err_pct` vs its
    exact-K rung. All stay `None` when the reference is missing or failed — a
    roster member that OOMs at dense must not silently drop its whole ladder.

    `speedup` is on `hessian_cold_s`, not `total_s`: the eigendecomposition is
    O(n^3) and identical on both rungs, so end-to-end dilutes the ratio. It also
    makes a re-derived record (cached Hessian, no cold timing) read `-`.

    The C_v error is reported both ways. Without the ASR the metric carries a
    whole k_B whenever a rung drops one more sub-threshold mode than the
    reference, which on small cells swamps the difference being measured.
    """
    by_key = {}
    for r in rows:
        by_key.setdefault((r.get("structure"), r.get("model")), []).append(r)

    for group in by_key.values():
        dense = next(
            (r for r in group if r.get("mode") == "dense" and r.get("status") == "ok"),
            None,
        )
        exact = next(
            (
                r
                for r in group
                if r.get("mode") == "sparse"
                and r.get("status") == "ok"
                and r.get("hops") == common.EXACT_HOPS.get(r.get("model"))
            ),
            None,
        )
        for r in group:
            r["speedup"] = None
            if dense and r.get("hessian_cold_s") and dense.get("hessian_cold_s"):
                r["speedup"] = dense["hessian_cold_s"] / r["hessian_cold_s"]
            r["rel_err_pct"] = _rel_err(r, exact, "cv300")
            r["rel_err_pct_asr"] = _rel_err(r, exact, "cv300_asr")


def _rel_err(row, reference, key):
    """Percent error in `key` against the exact-K rung, `None` without one."""
    value = row.get(key)
    ref = reference.get(key) if reference else None
    return None if value is None or not ref else 100 * (value - ref) / ref


def aux_flag(flag):
    """`k_sel_overflow` / `energy_sum` rendered as a short marker, `-` if clean."""
    if not flag:
        return None
    value = flag.get("value")
    if flag.get("kind") == "k_sel_overflow":
        return "OVERFLOW" if value else None
    return "NaN" if value != value else None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load(directory):
    """Records, plus `crashed.json` stubs so a crashed condition stays visible.

    Dot-prefixed directories (archived conditions) are skipped explicitly:
    `pathlib.Path.glob`, unlike the `glob` module, matches hidden entries.
    """
    root = Path(directory)
    rows, unmerged = [], []
    for f in sorted(root.glob("*/*/record.json")):
        if not _live(f):
            continue
        r = read_json(f)
        merge = common.merge_reused_coloring(r, f.parent)
        if merge in ("missing", "invalid", "mismatch"):
            unmerged.append(f"{f.parent.name} ({merge})")
        rows.append(r)
    if unmerged:
        # Diagnostic only — ladder_extract refuses these outright.
        print(
            f"warning: {len(unmerged)} records have no usable coloring sidecar and "
            f"keep their own timings: {', '.join(unmerged[:4])}…"
        )
    done = {(r.get("structure"), r.get("label")) for r in rows}
    for f in sorted(root.glob("*/*/crashed.json")):
        if not _live(f):
            continue
        stub = read_json(f)
        if (stub.get("structure"), stub.get("label")) not in done:
            rows.append(stub)
    return rows


def _live(path):
    return not any(part.startswith(".") for part in path.parent.parts)


def single_device(rows, where):
    """The one accelerator every record came from; bail on a mixture.

    Timings are this experiment's deliverable and are not comparable across
    cards, and the paths carry no platform — so two clusters' outputs fetched
    into one tree interleave silently. Refuse rather than filter: selecting a
    subset here is the cherry-picking this exists to prevent.
    """
    seen = {}
    for r in rows:
        seen.setdefault(device_of(r), []).append(r.get("structure"))
    known = {d: v for d, v in seen.items() if d is not None}
    if len(known) > 1:
        detail = "\n".join(
            f"  {d}: {len(v)} records ({', '.join(sorted(set(v))[:4])}…)"
            for d, v in sorted(known.items())
        )
        sys.exit(
            f"{where} mixes records from {len(known)} accelerators:\n{detail}\n"
            "Separate them into one tree per machine and collect each alone."
        )
    if None in seen:
        print(f"warning: {len(seen[None])} records do not name their accelerator")
    return next(iter(known), None)


def device_of(r):
    devices = (r.get("provenance") or {}).get("devices") or {}
    return devices.get("kind")


def single_matmul(rows, where):
    """The one matmul mode every record ran at; bail on a mixture.

    Absent key = TF32-era; records without a config (crashed stubs) are ignored.
    """
    seen = {}
    for r in rows:
        cfg = r.get("config")
        mode = cfg.get("matmul_precision", "default (TF32)") if cfg else None
        seen.setdefault(mode, []).append(r.get("structure"))
    known = {m: v for m, v in seen.items() if m is not None}
    if len(known) > 1:
        detail = "\n".join(
            f"  {m}: {len(v)} records ({', '.join(sorted(set(v))[:4])}…)"
            for m, v in sorted(known.items())
        )
        sys.exit(
            f"{where} mixes records from {len(known)} matmul modes:\n{detail}\n"
            "Finish the re-run (or collect a prepped subtree) before comparing."
        )
    return next(iter(known), "unrecorded")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_csv(path, rows):
    cols = [
        "structure",
        "model",
        "rung",
        "status",
        "n_atoms",
        "n_hvps",
        "num_colors",
        "fill",
        "chunk_size",
        "hessian_cold_s",
        "coloring_s",
        "total_s",
        "peak_gib",
        "speedup",
        "cached",
        "cv300",
        "cv300_asr",
        "rel_err_pct",
        "rel_err_pct_asr",
        "n_dropped",
        "n_dropped_asr",
        "flag",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


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
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4g}"
    return "-" if v is None else str(v)


def gib(v):
    return None if v is None else v / 2**30


if __name__ == "__main__":
    main()
