"""Extract the paper-facing results of the hops experiment from `output/law.json`.

The paper cites this experiment for one confirmation: the measured Hessian hop
count matches the predicted law — `K = 2L` for a node readout, `K = 2L + 1` for
an edge readout — with bit-exact zeros past `K`. `results()` re-derives that
confirmation from the results on disk instead of trusting that `law.py` exited
zero, rejects anything short of the full default sweep (a smoke run is not
paper-grade evidence), and returns the checkable facts: the verified law, the
sweep supporting it, and `K` for each production checkpoint. Run as a script to
print them as JSON; exits nonzero on any contradiction. Stdlib only.

    uv run python work/hops/results.py
"""

import argparse
import json
from pathlib import Path

EXP = Path(__file__).resolve().parent

# The full default sweep of `law.py`; `results()` rejects anything less.
FAMILIES = ("caterpillar", "multishell", "path", "ring")
DEPTHS = (1, 2, 3, 4)
N_SEEDS = 3
RANDOM_MODELS = {"MACE": "node", "PET": "edge"}  # model -> readout
CHECKPOINTS = ("MACE-MP-0", "pet-mad-xs", "pet-mad-s")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-i", "--input", type=Path, default=EXP / "output" / "law.json")
    args = p.parse_args()
    print(json.dumps(results(args.input), indent=2))


def results(path=EXP / "output" / "law.json"):
    """The paper-facing facts, re-derived from the measurement on disk.

    Raises:
        AssertionError: if the data contradicts the law, couples past `K`,
            saturates a graph diameter, or covers less than the full sweep.
    """
    data = json.loads(Path(path).read_text())
    random_rows, ckpt_rows = data["random_weights"], data["checkpoints"]

    for row in random_rows + ckpt_rows:
        _check_row(row)
    _check_random_sweep(random_rows)
    checkpoints = _collect_checkpoints(ckpt_rows)

    return {
        "law": {
            "MACE": {"readout": "node", "K": "2L"},
            "PET": {"readout": "edge", "K": "2L + 1"},
        },
        "sweep": {
            "depths": list(DEPTHS),
            "families": list(FAMILIES),
            "seeds": N_SEEDS,
            "n_hessians": sum(len(r["hops"]) for r in random_rows + ckpt_rows),
        },
        # `_check_row` raised on any coupling past K, so reaching this return
        # *is* the confirmation.
        "past_K_bit_exact_zero": True,
        "checkpoints": checkpoints,
    }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_row(row):
    """One measurement: law holds, boundary is sharp, graph does not saturate."""
    tag = f"{row['model']} L={row['L']} on {row['family']}"
    K = row["predicted"]
    _require(all(h == K for h in row["hops"]), f"{tag}: measured {row['hops']}, not {K}")
    _require(row["decay"][K] > 0.0, f"{tag}: no coupling at hop {K}")
    _require(row["decay"][K + 1] == 0.0, f"{tag}: coupling past hop {K}")
    _require(row["diameter"] > K, f"{tag}: diameter {row['diameter']} saturates K={K}")


def _check_random_sweep(rows):
    """Every (model, family, L) cell of the full sweep, all seeds, right law."""
    index = {(r["model"], r["family"], r["L"]): r for r in rows}
    _require(len(index) == len(rows), "duplicate (model, family, L) rows")
    expected = {
        (model, family, L) for model in RANDOM_MODELS for family in FAMILIES for L in DEPTHS
    }
    _require(
        set(index) == expected,
        "random-weights results are not the full default sweep: "
        f"missing {sorted(expected - set(index))}, extra {sorted(set(index) - expected)}",
    )
    for (model, family, L), row in index.items():
        tag = f"{model} L={L} on {family}"
        _require(len(row["hops"]) == N_SEEDS, f"{tag}: fewer than {N_SEEDS} seeds")
        K = _predicted(RANDOM_MODELS[model], L)
        _require(
            row["predicted"] == K, f"{tag}: predicted {row['predicted']}, law says {K}"
        )


def _collect_checkpoints(rows):
    """Per-checkpoint `(L, K, readout)`, required identical across families."""
    _require(
        len(rows) == len(CHECKPOINTS) * len(FAMILIES),
        f"expected {len(CHECKPOINTS) * len(FAMILIES)} checkpoint rows, got {len(rows)}",
    )
    out = {}
    for name in CHECKPOINTS:
        mine = [r for r in rows if r["model"] == name]
        _require(
            sorted(r["family"] for r in mine) == sorted(FAMILIES),
            f"{name}: not measured once on each of {FAMILIES}",
        )
        depths, counts = {r["L"] for r in mine}, {r["predicted"] for r in mine}
        _require(
            len(depths) == 1 and len(counts) == 1,
            f"{name}: inconsistent L={depths} or K={counts} across families",
        )
        (L,), (K,) = depths, counts
        readout = next((ro for ro in ("node", "edge") if _predicted(ro, L) == K), None)
        _require(readout is not None, f"{name}: K={K} fits neither law for L={L}")
        out[name] = {"L": L, "K": K, "readout": readout}
    return out


def _predicted(readout, L):
    return 2 * L if readout == "node" else 2 * L + 1


def _require(condition, message):
    # not `assert`: these checks are the deliverable, `python -O` must not strip them
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
