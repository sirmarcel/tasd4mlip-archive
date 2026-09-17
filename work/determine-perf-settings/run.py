"""Orchestrate the perf-settings phases: one worker process per condition, resumable.

    python run.py patterns            # pattern/coloring stats per (system, hops)
    python run.py grid                # HVP-engine grid at the representatives
    python run.py extremes            # feasibility ladder at the certification extremes
    python run.py certify             # full production path at the locked settings
    python run.py pet                 # PET-XS/S sanity checks at the locked settings
    python run.py matmul              # TF32 vs full-fp32 matmuls at the locked settings
    python run.py <phase> --dry-run   # print the condition list
    python run.py <phase> --shard 0/3 # 1st of 3 non-overlapping stripes (parallel jobs)
    python run.py <phase> --only RSM1885   # substring-filter the condition list

Conditions whose output json exists are skipped; delete the json to force a
re-run (a worker that dies without a record gets a `status: "crashed"` stub).
`certify`, `pet` and `matmul` read the locked settings from `output/settings.json`
(written by `collect.py --write`) unless overridden via flags. Systems, phase
scoping, and the decision rule: README.

Padding: tight (multiples_of_16) everywhere since the 08-18 cutover; the
bucketed pre-cutover campaign lives in `output/archive-bucketed/` (padding is
recorded per json, and `collect.py` derives per regime).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

# chunk caps scoped by the smoke measurement (README, Phases)
GRID_CHUNKS = [1, 2, 4, 8, 16]
LADDER_CHUNKS = [1, 2, 4, 16, 64]
ROF_CHUNKS = [1, 4, 16, 64]

# PET sanity runs at production input sizing, not the lock's: k_sel is small,
# so bucket and slack both bite there.
PET_PADDING = "production"

MATMUL_PRECISIONS = ["default", "highest"]

SMALL, MEDIAN, GIANT = "RSM0023", "RSM1831", "mil101"
EXTREMES = ["RSM1885", "RSM0254"]
# The certification envelope: RSM0254 (density extreme) is excluded from the
# feasibility gate — infeasible at any probed setting pending the intra-block
# chunking lever — but stays in the probe roster.
CERT_EXTREMES = ["RSM1885"]
PATTERN_SYSTEMS = [SMALL, MEDIAN, GIANT, *EXTREMES]


def conditions(phase, settings):
    if phase == "smoke":
        # GPU-path validation on the cheapest real conditions; the full phases
        # skip these as done.
        return [patterns_cond(SMALL, "mace", 2), probe_cond(SMALL, chunk=4)]

    if phase == "patterns":
        # h4 only below the extremes (host-memory cost; README)
        return [patterns_cond(s, "mace", 2) for s in PATTERN_SYSTEMS] + [
            patterns_cond(s, "mace", 4) for s in (SMALL, MEDIAN, GIANT)
        ]

    if phase == "grid":
        conds = []
        for s in (SMALL, MEDIAN):
            conds += [probe_cond(s, chunk=c) for c in GRID_CHUNKS]
        conds += [probe_cond(SMALL, chunk=c, remat=False) for c in (1, 4)]
        conds += [probe_cond(MEDIAN, chunk=1, remat=False)]
        conds += [probe_cond(SMALL, mode="rev_over_fwd", chunk=c) for c in (4, 16)]
        conds += [probe_cond(MEDIAN, mode="rev_over_fwd", chunk=c) for c in ROF_CHUNKS]
        conds += [probe_cond(GIANT, chunk=c) for c in (1, 4, 16)]
        return conds

    if phase == "extremes":
        size, density = EXTREMES
        return [
            *[probe_cond(size, chunk=c) for c in LADDER_CHUNKS],
            probe_cond(size, chunk=1, remat=False),
            *[probe_cond(size, mode="rev_over_fwd", chunk=c) for c in ROF_CHUNKS],
            *[probe_cond(density, chunk=c) for c in (1, 2, 4, 16)],
            *[probe_cond(density, mode="rev_over_fwd", chunk=c) for c in (1, 4, 16)],
        ]

    if phase == "certify":
        return [certify_cond(s, "mace", settings) for s in (SMALL, GIANT, *CERT_EXTREMES)]

    if phase == "pet":
        conds = []
        for model, hops in (("pet-xs", 3), ("pet-s", 3)):
            for s in (MEDIAN, EXTREMES[0]):
                conds.append(patterns_cond(s, model, hops, padding=PET_PADDING))
                conds.append(probe_cond(s, model=model, padding=PET_PADDING, **settings))
                # one chunk step above the locked setting: headroom signal
                conds.append(
                    probe_cond(
                        s,
                        model=model,
                        padding=PET_PADDING,
                        **settings | {"chunk": 2 * settings["chunk"]},
                    )
                )
        return conds

    if phase == "matmul":
        # Is TF32 buying throughput at production sizes? The probe's difference
        # quotient cancels compile and autotuning, which is exactly what the
        # cold unit-cell ratio could not. A rider on the
        # lock, not an input to it — hence its own output directory.
        conds = []
        for s in (SMALL, MEDIAN, GIANT, *CERT_EXTREMES):
            for mm in MATMUL_PRECISIONS:
                conds.append(probe_cond(s, matmul=mm, subdir="matmul", **settings))
        for model in ("pet-xs", "pet-s"):
            for mm in MATMUL_PRECISIONS:
                conds.append(
                    probe_cond(
                        MEDIAN,
                        model=model,
                        padding=PET_PADDING,
                        matmul=mm,
                        subdir="matmul",
                        **settings,
                    )
                )
        return conds

    sys.exit(f"unknown phase {phase!r}")


def patterns_cond(structure, model, hops, padding="tight"):
    label = f"{structure}_{model}_h{hops}"
    return {
        "out": OUTPUT / "patterns" / f"{label}.json",
        "args": [
            "patterns",
            *base_args(structure, model),
            "--hops",
            str(hops),
            "--padding",
            padding,
        ],
        "meta": {
            "task": "patterns",
            "structure": structure,
            "model": model,
            "hops": hops,
            "padding": padding,
        },
    }


def probe_cond(
    structure,
    model="mace",
    mode="fwd_over_rev",
    chunk=4,
    remat=True,
    padding="tight",
    matmul="default",
    subdir="probe",
):
    label = f"{structure}_{model}_{mode}_c{chunk}_{'remat' if remat else 'noremat'}"
    if matmul == "highest":
        label += "_mmhigh"
    return {
        "out": OUTPUT / subdir / f"{label}.json",
        "args": [
            "probe",
            *base_args(structure, model),
            "--mode",
            mode,
            "--chunk-size",
            str(chunk),
            "--remat" if remat else "--no-remat",
            "--padding",
            padding,
            "--matmul-precision",
            matmul,
        ],
        "meta": {
            "task": "probe",
            "structure": structure,
            "model": model,
            "mode": mode,
            "chunk_size": chunk,
            "remat": remat,
            "padding": padding,
            "matmul_precision": matmul,
        },
    }


def certify_cond(structure, model, settings, hops=2, padding="tight"):
    chunk, mode, remat = settings["chunk"], settings["mode"], settings["remat"]
    label = (
        f"{structure}_{model}_h{hops}_{mode}_c{chunk}_{'remat' if remat else 'noremat'}"
        + ("_tight" if padding == "tight" else "")
    )
    return {
        "out": OUTPUT / "certify" / f"{label}.json",
        "args": [
            "certify",
            *base_args(structure, model),
            "--hops",
            str(hops),
            "--mode",
            mode,
            "--chunk-size",
            str(chunk),
            "--remat" if remat else "--no-remat",
            "--padding",
            padding,
        ],
        "meta": {
            "task": "certify",
            "structure": structure,
            "model": model,
            "hops": hops,
            "mode": mode,
            "chunk_size": chunk,
            "remat": remat,
            "padding": padding,
        },
    }


def base_args(structure, model):
    supercell = "unit" if structure == GIANT else "minconv"
    return ["--structure", structure, "--model", model, "--supercell", supercell]


def load_settings(args):
    """Locked settings for certify/pet/matmul: flags win, else output/settings.json."""
    if args.chunk_size is not None:
        return {"chunk": args.chunk_size, "mode": args.mode, "remat": args.remat}
    path = OUTPUT / "settings.json"
    if not path.exists():
        sys.exit("no output/settings.json (run collect.py --write) and no --chunk-size")
    s = json.loads(path.read_text())
    return {"chunk": s["chunk_size"], "mode": s["mode"], "remat": s["remat"]}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "phase",
        choices=["smoke", "patterns", "grid", "extremes", "certify", "pet", "matmul"],
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--chunk-size", type=int, default=None, help="certify/pet/matmul override"
    )
    p.add_argument("--mode", default="fwd_over_rev")
    p.add_argument("--remat", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--shard", default=None, help="I/N: run the I-th of N stripes")
    p.add_argument("--only", default=None, help="substring filter on condition labels")
    args = p.parse_args()

    settings = load_settings(args) if args.phase in ("certify", "pet", "matmul") else None
    conds = conditions(args.phase, settings)
    # Striping is over the full stable condition list, so parallel shards of the
    # same phase never overlap, resubmits included.
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        conds = conds[i::n]
    if args.only:
        conds = [c for c in conds if args.only in c["out"].name]
    todo = [c for c in conds if not c["out"].exists()]
    print(f"{args.phase}: {len(conds)} conditions, {len(conds) - len(todo)} done")

    if args.dry_run:
        for c in todo:
            print(f"  {c['out'].name}")
        return

    failed = []
    for k, c in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {c['out'].name}", flush=True)
        result = subprocess.run(
            [sys.executable, str(HERE / "worker.py"), *c["args"], "--out", str(c["out"])],
        )
        if result.returncode != 0:
            print(f"  FAILED ({result.returncode}), continuing", flush=True)
            failed.append(c["out"].name)
            # stub: keeps killed workers identified and stops eternal retries
            if not c["out"].exists():
                c["out"].parent.mkdir(parents=True, exist_ok=True)
                stub = c["meta"] | {"status": "crashed", "returncode": result.returncode}
                c["out"].write_text(json.dumps(stub, indent=1))
    if failed:
        print(f"{len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
