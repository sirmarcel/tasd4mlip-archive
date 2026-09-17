"""Aggregate the ablation records into quick-look tables.

Reads `output/<structure>/<label>/record.json` — the lightweight product. Prints
the C_v table against each family's reference corner, failures, the grid view,
and the cross-check of `mace_fp64+d3` against `work/cv-ref-goennheimer` (same
PES, same pipeline family — expected at fp64/implementation noise).

The paper-facing numbers come from `ablation_extract.py` -> `results/`; this is
the campaign-steering view. Trimmed from `work/hessians/collect.py`.

    python collect.py
    python collect.py --grid goenn:10
"""

import numpy as np

import argparse
import sys
from pathlib import Path

import common

from sadmof.io import read_json

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
CV_REF = common.ROOT / "work" / "cv-ref-goennheimer" / "output" / "dense_float64"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", type=Path, default=OUTPUT)
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

    if args.grid is not None:
        show_grid(rows, args.grid)
        return

    print(f"{len(rows)} records")
    annotate(rows)
    show_cv(rows)
    show_failures(rows)
    show_crosscheck(rows, args.input)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def show_cv(rows):
    """C_v at 300 K against the family's reference corner (fp64+d3 / fp64_shadow).

    The D3-alone condition is a correction term, not a material — its own C_v
    is meaningless, so it appears in the grid and failures views only.
    """
    ok = [
        r
        for r in rows
        if r.get("status") == "ok" and r.get("model") != "d3" and r.get("cv300") is not None
    ]
    show(
        "C_v(300 K) vs the family's reference corner",
        sorted(ok, key=lambda r: (r["structure"], r["_order"])),
        [
            "structure",
            "variant",
            "n_atoms",
            "cv300",
            "cv300_asr",
            "rel_err_pct",
            "rel_err_pct_asr",
            "n_dropped",
            "n_dropped_asr",
            "raw_asymmetry_rel",
            "hessian_cold_s",
            "peak_gib",
        ],
    )


def show_failures(rows):
    """Records that exist but did not complete; `--grid` covers the never-run."""
    bad = [r for r in rows if r.get("status") != "ok"]
    if not bad:
        print("\n== failures: none among recorded conditions")
        return
    show(
        "failures: recorded but not ok",
        sorted(bad, key=lambda r: (r["structure"], r.get("variant") or "")),
        ["structure", "variant", "status", "oom_where", "requested_gib", "flag"],
    )


def show_grid(rows, spec):
    """`run.py`'s expected grid (plus combines) against what is on disk."""
    structures = (
        common.roster(spec)
        if spec
        else sorted({r["structure"] for r in rows if r.get("structure")})
    )
    slots = common.variants()
    status = {}
    for r in rows:
        status[(r.get("structure"), variant_of(r))] = r.get("status")

    pad = max(len(x) for x in [*structures, "structure"])
    print("\n== grid   (o = ok, x = failed, . = absent)")
    for k, v in enumerate(slots, 1):
        print(f"  {k:>2}  {v}")
    print(
        "  "
        + "structure".ljust(pad)
        + "  "
        + " ".join(f"{k:>2}" for k in range(1, len(slots) + 1))
    )

    ok = failed = absent = 0
    for structure in structures:
        marks = []
        for slot in slots:
            st = status.get((structure, slot))
            if st == "ok":
                ok += 1
                marks.append("o")
            elif st is None:
                absent += 1
                marks.append(".")
            else:
                failed += 1
                marks.append("x")
        print("  " + structure.ljust(pad) + "  " + " ".join(f"{m:>2}" for m in marks))

    print(
        f"\n  {ok} ok, {failed} failed, {absent} never run, of "
        f"{ok + failed + absent} ({len(structures)} structures x {len(slots)} conditions)"
    )


def show_crosscheck(rows, output_root):
    """`mace_fp64+d3` against the committed cv-ref numbers, per structure.

    Two runs of the same sadmof chain (same relaxation, same D3 port; only
    padding, chunking, and remat differ), so the expectation is float
    associativity noise — max|dnu| ~2e-5 cm^-1 on the campaign records, ΔC_v
    orders below cv-ref's ~1e-5 J/(g·K) agreement with Gönnheimer, which is a
    cross-pipeline figure and NOT the tolerance here. Printed, not asserted —
    a deviation wants investigation, not a crashed collect.
    """
    print("\n== cross-check: mace_fp64+d3 vs work/cv-ref-goennheimer (no ASR)")
    refs = [r for r in rows if variant_of(r) == "mace_fp64+d3" and r.get("status") == "ok"]
    if not refs:
        print("  nothing to check yet")
        return
    for r in sorted(refs, key=lambda r: r["structure"]):
        path = CV_REF / f"{r['structure']}.npz"
        if not path.exists():
            print(f"  {r['structure']}: no cv-ref result ({path})")
            continue
        with np.load(path) as z:
            temps = np.asarray(z["temperatures"], dtype=float)
            cv_ref = float(np.asarray(z["cv_gravimetric"])[temps == 300.0][0])
            freqs_ref = np.asarray(z["frequencies_cm1"])
        cv_here = (r.get("observables") or {}).get("cv_J_per_gK", {}).get("300")
        line = f"  {r['structure']}: dCv(300K) {cv_here - cv_ref:+.2e} J/(g.K)"
        obs_path = Path(output_root) / r["structure"] / r["label"] / "observables.npz"
        if obs_path.exists():
            with np.load(obs_path) as z:
                freqs = np.asarray(z["frequencies_cm1"])
            if freqs.shape == freqs_ref.shape:
                line += f", max|dnu| {np.abs(freqs - freqs_ref).max():.3g} cm^-1"
            else:
                line += f", mode count differs ({freqs.size} vs {freqs_ref.size})"
        print(line)


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def annotate(rows):
    """Flatten each record into the scalar columns the tables print."""
    order = {v: k for k, v in enumerate(common.variants())}
    for r in rows:
        obs = r.get("observables") or {}
        t, mem, hess = r.get("timings") or {}, r.get("memory") or {}, r.get("hessian") or {}

        r["variant"] = variant_of(r)
        r["_order"] = order.get(r["variant"], len(order))
        r["n_atoms"] = (r.get("system") or {}).get("n_atoms")
        r["raw_asymmetry_rel"] = hess.get("raw_asymmetry_rel")
        r["n_dropped"] = obs.get("n_dropped")
        r["n_dropped_asr"] = obs.get("n_dropped_asr")
        r["flag"] = aux_flag(r.get("aux_flag"))
        r["hessian_cold_s"] = t.get("hessian_cold_s")
        r["peak_gib"] = gib(mem.get("peak_bytes"))
        r["requested_gib"] = gib(r.get("requested_bytes"))
        r["cv300"] = (obs.get("cv_J_per_gK") or {}).get("300")
        r["cv300_asr"] = (obs.get("cv_J_per_gK_asr") or {}).get("300")

    _relative(rows)


def _relative(rows):
    """C_v errors against the family's reference corner; `None` without one."""
    by_label = {(r.get("structure"), r.get("variant")): r for r in rows}
    for r in rows:
        r["rel_err_pct"] = r["rel_err_pct_asr"] = None
        family = r.get("model")
        if family not in common.REFERENCE_VARIANTS:
            continue
        ref = by_label.get((r["structure"], common.REFERENCE_VARIANTS[family]))
        if ref is None or ref.get("status") != "ok":
            continue
        r["rel_err_pct"] = _rel_err(r, ref, "cv300")
        r["rel_err_pct_asr"] = _rel_err(r, ref, "cv300_asr")


def _rel_err(row, reference, key):
    value, ref = row.get(key), reference.get(key)
    return None if value is None or not ref else 100 * (value - ref) / ref


def variant_of(record):
    label, structure = record.get("label") or "", record.get("structure") or ""
    return label.removeprefix(f"{structure}_")


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
    rows = [read_json(f) for f in sorted(root.glob("*/*/record.json")) if _live(f)]
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


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


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
