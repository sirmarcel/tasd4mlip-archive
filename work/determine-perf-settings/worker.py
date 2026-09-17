"""One perf-settings condition per process (the worker behind `run.py`), fp32.

Tasks: `patterns` (sparsity pattern + coloring stats and the analytic device
floor of a real sparse run), `probe` (steady-state ms/HVP + peak memory of the
bare HVP engine, mirroring asdex's per-mode engines), `certify` (one full
production-path Hessian at given settings). Method and floor model: README.

`--matmul-precision highest` pins full-fp32 matmuls; the default permits TF32
on Hopper and is what production runs. It is the `matmul` phase's axis.

An OOM is a recorded result (`status: "oom"`, host or device), not a failure;
condition parameters land in the record before the attempt.
"""

import numpy as np

import argparse
import json
import re
import time
from pathlib import Path

import common


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["patterns", "probe", "certify"])
    p.add_argument("--structure", required=True)
    p.add_argument("--model", default="mace", choices=list(common.EXACT_HOPS))
    p.add_argument("--supercell", default="minconv", choices=["minconv", "unit"])
    p.add_argument("--hops", type=int, default=2, help="pattern hops (patterns/certify)")
    p.add_argument(
        "--mode", default="fwd_over_rev", choices=["fwd_over_rev", "rev_over_fwd"]
    )
    p.add_argument("--chunk-size", type=int, default=4)
    p.add_argument("--remat", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--padding", default="tight", choices=["bucketed", "tight", "production"]
    )
    p.add_argument(
        "--matmul-precision",
        default="default",
        choices=["default", "highest"],
        help="highest: disable TF32 matmuls (default is the production setting)",
    )
    p.add_argument("--probe-hvps", type=int, default=None, help="full sweep length")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--warm", action="store_true", help="certify: also time a warm call")
    p.add_argument("--out", required=True, help="output json path")
    args = p.parse_args()

    # Before the first array op. fp32 processes never enable x64, so precision
    # here is the matmul mode alone: "default" permits TF32 on Hopper.
    if args.matmul_precision == "highest":
        import jax

        jax.config.update("jax_default_matmul_precision", "highest")

    from sadmof.provenance import provenance

    float_dtype = np.float32  # production Hessian precision; x64 stays off

    t0 = time.perf_counter()
    atoms = common.load_relaxed(args.structure)
    model, params, energy_fn = common.load_model(args.model, float_dtype)
    if args.supercell == "minconv":
        mult = common.converged_multiplier(atoms, args.model, model)
    else:
        mult = (1, 1, 1)
    super_atoms = atoms * mult
    # tight = pay compile per structure (we do anyway), skip the power-of-two
    # pair buckets that cost up to 2x memory at the extremes
    strategy = common.PADDING_STRATEGIES[args.padding]
    pos, cell, graph, pattern_graph = common.build_inputs(
        args.model, model, super_atoms, float_dtype, bucket_strategy=strategy
    )
    setup_s = time.perf_counter() - t0

    result = {
        "task": args.task,
        "structure": args.structure,
        "model": args.model,
        "supercell": args.supercell,
        "mult": list(mult),
        "n_primitive": len(atoms),
        "n_atoms": len(super_atoms),
        "n_padded": int(pos.shape[0]),
        "dim": 3 * int(pos.shape[0]),
        "n_pairs_padded": int(np.asarray(pattern_graph["centers"]).shape[0]),
        "padding": args.padding,
        "matmul_precision": args.matmul_precision,
        "pet_extra_neighbors": common.PET_EXTRA_NEIGHBORS,
        "setup_s": setup_s,
        "status": "ok",
        # After load_model: provenance names the device only once jax is loaded.
        "provenance": provenance(),
    }
    # Condition parameters go in before the attempt so OOM rows stay identified.
    if args.task in ("probe", "certify"):
        result |= {"mode": args.mode, "chunk_size": args.chunk_size, "remat": args.remat}
    if args.task in ("patterns", "certify"):
        result["hops"] = args.hops
    if args.task == "probe":
        result["n_probe"] = probe_length(args)

    try:
        if args.task == "patterns":
            result |= run_patterns(args, pattern_graph)
        elif args.task == "probe":
            result |= run_probe(args, energy_fn, params, pos, cell, graph)
        else:
            result |= run_certify(args, energy_fn, params, pos, cell, graph, pattern_graph)
    except MemoryError as e:
        result |= {"status": "oom", "oom_where": "host", "error": repr(e)[:2000]}
    except Exception as e:  # noqa: BLE001 — OOM is data, everything else re-raises
        msg = repr(e)
        if "RESOURCE_EXHAUSTED" not in msg:
            raise
        result |= {"status": "oom", "oom_where": "device", "error": msg[:2000]}
        m = re.search(r"allocat\w+ (\d+) bytes", msg) or re.search(r"(\d+) bytes", msg)
        if m:
            result["requested_bytes"] = int(m.group(1))

    result |= memory_stats()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic: a worker killed mid-write must not leave a truncated file that
    # resume-by-existence would treat as done.
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(out)
    brief = {k: result.get(k) for k in ("status", "ms_per_hvp", "peak_bytes", "num_colors")}
    print(f"{out.name}: {brief}", flush=True)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def run_patterns(args, pattern_graph):
    import asdex

    from sadmof.sparse import sparsity_pattern

    t0 = time.perf_counter()
    pattern = sparsity_pattern(pattern_graph, hops=args.hops)
    pattern_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    coloring = asdex.hessian_coloring_from_sparsity(pattern, mode="fwd_over_rev")
    coloring_s = time.perf_counter() - t0

    dim = pattern.shape[0]
    return {
        "hops": args.hops,
        "nnz": int(len(pattern.rows)),
        "fill": len(pattern.rows) / (dim * dim),
        "num_colors": int(coloring.num_colors),
        "pattern_s": pattern_s,
        "coloring_s": coloring_s,
        # device floor of a real sparse run (README, Method)
        "floor_bytes": 2 * int(coloring.num_colors) * dim * 4 + 36 * int(len(pattern.rows)),
    }


def run_probe(args, energy_fn, params, pos, cell, graph):
    import jax
    import jax.numpy as jnp

    n_full = probe_length(args)
    n_half = n_full // 2
    # Arrays are jit arguments (never baked-in constants) and live on device up
    # front, so warm calls don't re-pay the host->device transfer.
    params, pos, cell, graph = jax.device_put((params, pos, cell, graph))
    sweep_jit = jax.jit(get_sweep_fn(energy_fn, args.mode, args.chunk_size, args.remat))

    # gradient-only reference: cost context + sanity scale for fixed_s
    grad_jit = jax.jit(
        lambda params, p, cell, graph: jax.grad(
            lambda q: energy_fn(params, q, cell, graph)[0]
        )(p)
    )
    jax.block_until_ready(grad_jit(params, pos, cell, graph))
    t_grad = min(timed(lambda: grad_jit(params, pos, cell, graph), args.repeats))

    walls = {}
    compile_s = {}
    for n in (n_half, n_full):
        idx = jnp.arange(n)  # basis seeds; seed content does not affect cost
        f = lambda: sweep_jit(params, pos, cell, graph, idx)  # noqa: E731
        t0 = time.perf_counter()
        jax.block_until_ready(f())
        cold = time.perf_counter() - t0
        walls[n] = min(timed(f, args.repeats))
        compile_s[n] = cold - walls[n]

    # difference estimate: cancels the fixed per-call cost shared by both lengths
    ms_per_hvp = (walls[n_full] - walls[n_half]) / (n_full - n_half) * 1e3
    fixed_s = walls[n_half] - n_half * ms_per_hvp / 1e3
    return {
        "ms_per_hvp": ms_per_hvp,
        "ms_per_hvp_raw": walls[n_full] / n_full * 1e3,
        "fixed_s": fixed_s,
        "grad_s": t_grad,
        "sweep_s": walls[n_full],
        "compile_s": compile_s[n_full],
    }


def run_certify(args, energy_fn, params, pos, cell, graph, pattern_graph):
    import jax

    import asdex

    from sadmof.sparse import get_hessian_fn, sparsity_pattern

    t0 = time.perf_counter()
    pattern = sparsity_pattern(pattern_graph, hops=args.hops)
    coloring = asdex.hessian_coloring_from_sparsity(pattern, mode=args.mode)
    coloring_s = time.perf_counter() - t0

    hessian = jax.jit(
        get_hessian_fn(energy_fn, coloring, chunk_size=args.chunk_size, remat=args.remat)
    )
    t0 = time.perf_counter()
    h, aux = hessian(params, pos, cell, graph)
    jax.block_until_ready((h.data, aux))
    cold_s = time.perf_counter() - t0

    result = {
        "num_colors": int(coloring.num_colors),
        "coloring_s": coloring_s,
        "hessian_cold_s": cold_s,
        "frobenius": float(jax.numpy.linalg.norm(h.data)),
        # summary only: MACE's aux is 3N per-atom energies; the sum flags NaNs
        # and PET's overflow bool survives as 0/1
        "aux_sum": float(np.sum(np.asarray(aux))),
        "aux_shape": list(np.shape(aux)),
    }
    if args.warm:
        result["hessian_warm_s"] = min(
            timed(
                lambda: jax.block_until_ready(hessian(params, pos, cell, graph)[0].data),
                args.repeats,
            )
        )
    return result


# ---------------------------------------------------------------------------
# HVP engine probe kernel
# ---------------------------------------------------------------------------


def get_sweep_fn(energy_fn, mode, chunk_size, remat):
    """A bounded HVP sweep mirroring asdex's per-mode engines primitive for
    primitive (`asdex._differentiation`); rev_over_rev is excluded (README)."""
    import jax

    def sweep(params, pos, cell, graph, seed_idx):
        n = pos.shape[0]
        dim = 3 * n

        def raw_energy(p):
            return energy_fn(params, p, cell, graph)

        energy = jax.checkpoint(raw_energy) if remat else raw_energy

        def seed(idx):
            return jax.nn.one_hot(idx, dim, dtype=pos.dtype).reshape(n, 3)

        if mode == "fwd_over_rev":
            _, hvp_fn, _aux = jax.linearize(
                jax.grad(energy, has_aux=True), pos, has_aux=True
            )

            def single(idx):
                return hvp_fn(seed(idx)).reshape(dim)

        else:  # rev_over_fwd

            def scalar(p):
                return energy(p)[0]

            def single(idx):
                def inner(p):
                    primal, tangent = jax.jvp(scalar, (p,), (seed(idx),))
                    return tangent, primal

                grad, _primal = jax.grad(inner, has_aux=True)(pos)
                return grad.reshape(dim)

        return jax.lax.map(single, seed_idx, batch_size=chunk_size)

    return sweep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def probe_length(args):
    return args.probe_hvps or max(2 * args.chunk_size, 32)


def timed(f, repeats):
    import jax

    walls = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(f())
        walls.append(time.perf_counter() - t0)
    return walls


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
