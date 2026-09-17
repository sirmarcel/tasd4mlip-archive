"""One ablation condition per process (the worker behind `run.py`).

A condition is `(structure, model, dtype, shadow)`: relaxed unit cell -> dense
Hessian -> frequencies -> C_v, timed end to end and per stage. One process per
condition keeps the JAX x64 state per-dtype and makes an OOM cost one Hessian.

An OOM is a recorded result (`status: "oom"`, host or device), not a failure;
condition parameters land in the record before the attempt.

Adapted from `work/hessians/worker.py`, dense-only.
"""

import numpy as np

import argparse
import os
import re
import time
from pathlib import Path

import common

from sadmof.io import write_json, write_npy, write_npz, write_xyz

T_PROC = time.perf_counter()  # end-to-end clock, before any heavy import

# Every timing key is always present, `None` where the stage did not run — a
# consumer should never have to distinguish "absent" from "not applicable".
TIMING_KEYS = (
    "import_s",
    "load_inputs_s",
    "hessian_cold_s",
    "hessian_warm_s",
    "extract_s",
    "save_s",
    "observables_s",
    "total_s",
)


def main():
    args = build_parser().parse_args()
    if args.model == "d3" and args.dtype != "float64":
        raise SystemExit("d3 is the fixed fp64 correction; run it with --dtype float64")
    if args.shadow and not args.model.startswith("pet"):
        raise SystemExit("--shadow is a PET axis")

    out_dir = Path(args.out_dir)
    record_path = out_dir / "record.json"

    timings = dict.fromkeys(TIMING_KEYS)
    t0 = time.perf_counter()
    import jax

    # x64 must be decided before the first array op; fp32 processes never
    # enable it, so no precision contamination across conditions.
    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    # "default" permits TF32 matmuls on Hopper — the production setting.
    # "highest" is the diagnostic that separates fp32-proper from TF32.
    if args.matmul_precision == "highest":
        jax.config.update("jax_default_matmul_precision", "highest")
    timings["import_s"] = time.perf_counter() - t0

    from sadmof.provenance import provenance

    float_dtype = np.float64 if args.dtype == "float64" else np.float32

    t0 = time.perf_counter()
    atoms = common.load_relaxed(args.structure, args.model)
    model, params, energy_fn = common.load_model(args.model, float_dtype, args.shadow)
    pos, cell, graph = common.build_inputs(args.model, model, atoms, float_dtype)
    timings["load_inputs_s"] = time.perf_counter() - t0

    n_real = len(atoms)
    n_padded = int(pos.shape[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_xyz(out_dir / "geometry.xyz", atoms)

    record = {
        "structure": args.structure,
        "model": args.model,
        "dtype": args.dtype,
        "shadow": args.shadow if args.model.startswith("pet") else None,
        "label": out_dir.name,
        "status": "ok",
        "config": {
            "ad_mode": common.AD_MODE,
            "chunk_size": common.CHUNK_SIZE,
            "remat": common.REMAT,
            # D3's input builder pads with its own fixed strategy (no knob).
            "padding": "multiples" if args.model == "d3" else common.PADDING,
            "matmul_precision": args.matmul_precision,
            "extra_neighbors": (
                common.EXTRA_NEIGHBORS if args.model.startswith("pet") else None
            ),
            # Requested, not resolved; `None` means unset (jax defaults to 0.75).
            "mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
            "temperatures_K": list(args.temperatures),
            "enforce_asr_stored": True,
        },
        "system": {
            "n_atoms": n_real,
            "n_padded": n_padded,
            "dim": 3 * n_padded,
            "n_pairs_padded": int(np.asarray(graph["centers"]).shape[0]),
            "mass_amu": float(atoms.get_masses().sum()),
        },
        "hessian": None,
        "observables": None,
        "timings": timings,
        "memory": None,
        "aux_flag": None,
        "oom_where": None,
        "requested_bytes": None,
        "error": None,
        # After load_model: provenance names the device only once jax is loaded.
        "provenance": provenance(),
    }

    try:
        run_condition(args, record, energy_fn, params, pos, cell, graph, atoms, out_dir)
    except MemoryError as e:
        record |= {"status": "oom", "oom_where": "host", "error": repr(e)[:2000]}
    except Exception as e:  # noqa: BLE001 — OOM is data, everything else re-raises
        msg = repr(e)
        if "RESOURCE_EXHAUSTED" not in msg:
            raise
        record |= {"status": "oom", "oom_where": "device", "error": msg[:2000]}
        m = re.search(r"allocat\w+ (\d+) bytes", msg) or re.search(r"(\d+) bytes", msg)
        if m:
            record["requested_bytes"] = int(m.group(1))

    record["memory"] = memory_stats()
    timings["total_s"] = time.perf_counter() - T_PROC
    write_json(record_path, record)

    brief = {
        "status": record["status"],
        "cold_s": timings["hessian_cold_s"],
        "total_s": round(timings["total_s"], 1),
    }
    print(f"{out_dir.name}: {brief}", flush=True)


# ---------------------------------------------------------------------------
# The condition
# ---------------------------------------------------------------------------


def run_condition(args, record, energy_fn, params, pos, cell, graph, atoms, out_dir):
    """Hessian -> observables, filling `record` in place."""
    import jax

    timings = record["timings"]
    n_real = len(atoms)

    raw, cached = load_or_compute_hessian(
        args, record, energy_fn, params, pos, cell, graph, n_real, out_dir, jax
    )

    record["hessian"] = {
        "n_hvps": 3 * int(pos.shape[0]),
        "dtype": str(raw.dtype),
        "combined": False,
        "raw_asymmetry_rel": asymmetry(raw),
        "cached": cached,
        "artifact": "hessian_raw.npy",
    }
    record["observables"] = derive_observables(
        raw, atoms.get_masses(), list(args.temperatures), out_dir, timings
    )


def load_or_compute_hessian(
    args, record, energy_fn, params, pos, cell, graph, n_real, out_dir, jax
):
    """The real-block dense Hessian, from the stage cache if one is present."""
    timings = record["timings"]
    artifact = out_dir / "hessian_raw.npy"
    if artifact.exists():
        return np.load(artifact), True

    from sadmof.dense import get_dense_hessian_fn

    hfn = jax.jit(
        get_dense_hessian_fn(energy_fn, chunk_size=common.CHUNK_SIZE, remat=common.REMAT)
    )

    t0 = time.perf_counter()
    h, aux = hfn(params, pos, cell, graph)
    jax.block_until_ready((h, aux))
    timings["hessian_cold_s"] = time.perf_counter() - t0  # XLA compile included
    record["aux_flag"] = aux_flag(args.model, aux)

    if args.warm:
        t0 = time.perf_counter()
        h2, aux2 = hfn(params, pos, cell, graph)
        jax.block_until_ready((h2, aux2))
        timings["hessian_warm_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    raw = np.asarray(h)[:n_real, :, :n_real, :].reshape(3 * n_real, 3 * n_real)
    timings["extract_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    write_npy(artifact, raw)
    timings["save_s"] = time.perf_counter() - t0
    return raw, False


def derive_observables(raw, masses, temperatures, out_dir, timings):
    """fp64 Hessian -> frequencies + C_v, with and without the acoustic sum rule.

    Shared with `combine_d3.py`, which is why it takes the raw real-block
    array rather than the worker's argparse namespace. `astype(float64)`
    copies, so the ASR and no-ASR variants each get their own buffer for
    `overwrite=True` to consume. Both variants are always stored — every
    comparison needs the ASR one, the cv-ref cross-check the other.
    """
    from sadmof.observables import FREQ_THRESHOLD_CM1, cv_from_hessian

    t0 = time.perf_counter()
    freqs, cv = cv_from_hessian(
        raw.astype(np.float64), masses, temperatures, overwrite=True
    )
    freqs_asr, cv_asr = cv_from_hessian(
        raw.astype(np.float64), masses, temperatures, enforce_asr=True, overwrite=True
    )
    timings["observables_s"] = time.perf_counter() - t0

    write_npz(
        out_dir / "observables.npz",
        {
            "frequencies_cm1": np.asarray(freqs, dtype=np.float64),
            "frequencies_cm1_asr": np.asarray(freqs_asr, dtype=np.float64),
            "masses_amu": np.asarray(masses, dtype=np.float64),
            "temperatures_K": np.asarray(temperatures, dtype=np.float64),
            "cv_J_per_gK": np.asarray([cv[t] for t in temperatures], dtype=np.float64),
            "cv_J_per_gK_asr": np.asarray(
                [cv_asr[t] for t in temperatures], dtype=np.float64
            ),
        },
    )
    return {
        "n_modes": int(len(freqs)),
        # Sub-threshold, not negative: `hessian_to_frequencies` collapses
        # imaginary modes to 0, so a `< 0` count could never fire.
        "n_dropped": n_dropped(freqs, FREQ_THRESHOLD_CM1),
        "n_dropped_asr": n_dropped(freqs_asr, FREQ_THRESHOLD_CM1),
        "cv_J_per_gK": {f"{t:g}": float(cv[t]) for t in temperatures},
        "cv_J_per_gK_asr": {f"{t:g}": float(cv_asr[t]) for t in temperatures},
    }


def n_dropped(freqs, threshold):
    return int(np.sum(np.asarray(freqs) < threshold))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structure", required=True)
    p.add_argument("--model", default="mace", choices=list(common.MODELS))
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument(
        "--shadow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="PET: keep the adaptive-cutoff force contributions",
    )
    p.add_argument(
        "--matmul-precision",
        default="default",
        choices=["default", "highest"],
        help="highest: disable TF32 matmuls (diagnostic; default is production)",
    )
    p.add_argument("--warm", action="store_true", help="also time a warm call")
    p.add_argument(
        "--temperatures", type=float, nargs="+", default=list(common.TEMPERATURES)
    )
    p.add_argument("--out-dir", required=True)
    return p


def aux_flag(model, aux):
    """PET: the `k_sel` overflow bool, which invalidates the Hessian if set.
    MACE and D3: the per-atom energy sum, a NaN canary."""
    if model.startswith("pet"):
        return {"kind": "k_sel_overflow", "value": bool(np.asarray(aux))}
    return {"kind": "energy_sum", "value": float(np.sum(np.asarray(aux)))}


def asymmetry(raw):
    scale = float(np.abs(raw).max())
    return float(np.abs(raw - raw.T).max()) / scale if scale > 0 else 0.0


def memory_stats():
    # Which device this was measured on is recorded by provenance().
    import jax

    stats = jax.local_devices()[0].memory_stats() or {}
    return {
        "peak_bytes": stats.get("peak_bytes_in_use"),
        "bytes_limit": stats.get("bytes_limit"),
    }


if __name__ == "__main__":
    main()
