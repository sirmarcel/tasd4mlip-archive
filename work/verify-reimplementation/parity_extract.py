"""`output/` -> `results/parity.json`: every prediction as a deviation from the
upstream fp64 reference.

    uv run python parity_extract.py --roster goenn:10

Deviations are stored per structure, unaggregated; `parity_load.py` reduces.
A structure missing a record on either side of a comparison is dropped from
that comparison and listed on stdout, and the drop is recorded alongside the
kept structures.
"""

import numpy as np

import argparse
from pathlib import Path

import common

from sadmof.io import write_json
from sadmof.provenance import provenance

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main():
    args = build_parser().parse_args()
    structures = common.roster(args.roster)
    ref_stack, ref_dtype = common.REFERENCE

    out = {
        "reference": f"{ref_stack}/{ref_dtype}",
        "roster": args.roster,
        "models": {},
    }
    seen = []
    for model in args.models:
        found = {
            key: compare(model, *key.split("/", 1), structures)
            for key in comparisons(model)
        }
        # A condition killed during checkpoint load leaves the directory it
        # created and no records. Dropping it here keeps an empty comparison out
        # of the reductions, which have no number to report for one.
        for key in [key for key, row in found.items() if not row["structures"]]:
            print(f"  {model} {key}: no paired structures, skipped")
            del found[key]
        seen += [key for key in found if key not in seen]
        out["models"][model] = found
    out["comparisons"] = seen

    out["provenance"] = provenance()
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS / "parity.json", out)
    print(f"wrote {RESULTS / 'parity.json'}")


def comparisons(model):
    """`stack/variant` keys on disk for `model`, reference excluded.

    Discovered rather than enumerated, so a run that adds a precision
    configuration (a GPU rung, a matmul mode) is picked up without touching
    this script.
    """
    reference = "/".join(common.REFERENCE)
    return [
        key
        for key in (f"{stack}/{v}" for stack, v in common.variants_present(model))
        if key != reference
    ]


def compare(model, stack, variant_name, structures):
    """Per-structure deviations of one (model, stack, variant) from the reference.

    Energies per atom; forces and stress as the largest single deviating
    component.
    """
    ref_stack, ref_dtype = common.REFERENCE
    rows = {key: [] for key in FIELDS}
    kept, dropped = [], []

    for structure in structures:
        try:
            ref = common.read_record(ref_stack, model, ref_dtype, structure)
            test = common.read_record(stack, model, variant_name, structure)
        except FileNotFoundError:
            dropped.append(structure)
            continue

        n = ref["n_atoms"]
        df = test["forces"] - ref["forces"]
        ds = test["stress"] - ref["stress"]
        kept.append(structure)
        rows["n_atoms"].append(n)
        rows["e_ref_per_atom"].append(ref["energy"] / n)
        rows["de_per_atom"].append((test["energy"] - ref["energy"]) / n)
        rows["f_ref_max"].append(float(np.abs(ref["forces"]).max()))
        rows["df_max"].append(float(np.abs(df).max()))
        rows["df_rms"].append(float(np.sqrt(np.mean(df**2))))
        rows["stress_ref_max"].append(float(np.abs(ref["stress"]).max()))
        rows["dstress_max"].append(float(np.abs(ds).max()))

    if dropped:
        print(
            f"  {model} {stack}/{variant_name}: {len(dropped)} without a pair, "
            f"dropped: {dropped}"
        )
    return {"structures": kept, "dropped": dropped, **rows}


# Per-structure columns, in the order `compare` fills them. `*_ref_*` are the
# scales the deviations sit against, kept so a relative statement never has to
# go back to `output/`.
FIELDS = (
    "n_atoms",
    "e_ref_per_atom",
    "de_per_atom",
    "f_ref_max",
    "df_max",
    "df_rms",
    "stress_ref_max",
    "dstress_max",
)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roster", nargs="+", default=["goenn:1"])
    p.add_argument(
        "--models", nargs="+", default=list(common.MODELS), choices=common.MODELS
    )
    return p


if __name__ == "__main__":
    main()
