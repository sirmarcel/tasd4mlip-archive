"""Feed collect.py synthetic records to pin the decision rule.

Covers the cases real output rarely contains all at once: OOM rows, crashed
stubs, and ok rows with no matching patterns record (unresolved floor). Asserts
collect survives them, falls back to rev_over_fwd when no fwd chunk fits, and
ranks feasible chunks by median-system speed over the certification envelope.
Also pins the matmul rider's isolation: its rows are shaped to win the mode if
they ever reached `derive`, and must not.

    uv run python work/determine-perf-settings/check_decision_rule.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EXP = Path(__file__).resolve().parent
TMP = Path(tempfile.mkdtemp(prefix="perf-decision-rule-"))

shutil.rmtree(TMP, ignore_errors=True)
(TMP / "output" / "probe").mkdir(parents=True)
(TMP / "output" / "patterns").mkdir(parents=True)
(TMP / "output" / "certify").mkdir(parents=True)
(TMP / "output" / "matmul").mkdir(parents=True)
for f in ("collect.py", "run.py"):
    shutil.copy(EXP / f, TMP / f)

LIMIT = 94_982_821_105
DIM = 49_152


def pattern(structure, floor):
    return {
        "task": "patterns",
        "structure": structure,
        "model": "mace",
        "hops": 2,
        "nnz": 10_000_000,
        "num_colors": 2000,
        "coloring_s": 60.0,
        "floor_bytes": floor,
        "status": "ok",
    }


def probe(structure, mode, chunk, status, peak=None, remat=True):
    r = {
        "task": "probe",
        "structure": structure,
        "model": "mace",
        "mode": mode,
        "chunk_size": chunk,
        "remat": remat,
        "n_probe": max(2 * chunk, 32),
        "dim": DIM,
        "status": status,
        "bytes_limit": LIMIT,
    }
    if status == "ok":
        r |= {"peak_bytes": peak, "ms_per_hvp": 100.0 / chunk, "sweep_s": 1.0}
    return r


rows = [
    # fwd_over_rev: OOM on both extremes at every chunk -> no fwd feasible
    probe("RSM1885", "fwd_over_rev", 1, "oom"),
    probe("RSM0254", "fwd_over_rev", 1, "oom"),
    # crashed stub (host OOM) — minimal keys only
    {
        "task": "probe",
        "structure": "RSM1885",
        "model": "mace",
        "mode": "fwd_over_rev",
        "chunk_size": 4,
        "remat": True,
        "status": "crashed",
        "returncode": 137,
    },
    # rev_over_fwd: fits both extremes at chunk 1; chunk 4 misses RSM0254
    probe("RSM1885", "rev_over_fwd", 1, "ok", peak=50 * 2**30),
    probe("RSM0254", "rev_over_fwd", 1, "ok", peak=55 * 2**30),
    probe("RSM1885", "rev_over_fwd", 4, "ok", peak=60 * 2**30),
    # median rows for ranking
    probe("RSM1831", "rev_over_fwd", 1, "ok", peak=30 * 2**30),
    probe("RSM1831", "rev_over_fwd", 4, "ok", peak=35 * 2**30),
    # unknown floor: ok row with no patterns record for its structure
    probe("RSM0023", "fwd_over_rev", 2, "ok", peak=80 * 2**30),
]
for k, r in enumerate(rows):
    (TMP / "output" / "probe" / f"row{k}.json").write_text(json.dumps(r))

# Rider rows: fwd_over_rev fitting the envelope with a fast median row, i.e.
# exactly what would flip the derived mode away from the rev_over_fwd fallback
# if output/matmul/ were ever folded into the probe set.
matmul_rows = [
    probe(structure, "fwd_over_rev", 1, "ok", peak=peak)
    | {"matmul_precision": mm, "ms_per_hvp": ms, "compile_s": 5.0}
    for structure, peak in (("RSM1885", 50 * 2**30), ("RSM1831", 30 * 2**30))
    for mm, ms in (("default", 10.0), ("highest", 30.0))
]
for k, r in enumerate(matmul_rows):
    (TMP / "output" / "matmul" / f"row{k}.json").write_text(json.dumps(r))
for s in ("RSM1885", "RSM0254", "RSM1831"):
    (TMP / "output" / "patterns" / f"{s}_mace_h2.json").write_text(
        json.dumps(pattern(s, 500_000_000))
    )

out = subprocess.run(
    [sys.executable, str(TMP / "collect.py")], capture_output=True, text=True, cwd=TMP
)
print(out.stdout[-1200:])
print(out.stderr[-500:] if out.returncode else "", file=sys.stderr)
setting = out.stdout.split("derived setting [bucketed]:")[1]
assert out.returncode == 0, "collect crashed"
assert '"mode": "rev_over_fwd"' in setting, "fallback not taken"
# Envelope is CERT_EXTREMES = [RSM1885] since 2026-08-18, so chunk 4 (which has
# an RSM1885 row that fits, and is faster at the median) is correctly feasible.
assert '"chunk_size": 4' in setting, "envelope/ranking regression"
rider = out.stdout.split("== matmul rider")[1].split("\nderived setting")[0]
assert "3.00" in rider, "rider warm ratio not reported"
print(
    "PASS: no crash, rev_over_fwd fallback, envelope + ranking as intended, "
    "matmul rider reported and out of the rule"
)
