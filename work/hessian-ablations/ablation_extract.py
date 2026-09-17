"""Records + stored Hessians + spectra -> the ablation dataset (results/ablation).

One output per structure: every condition's C_v and mode-count scalars, the
spectra, and — per ablation pair — the frequency and Hessian-level deviation of
the ablated condition from its reference-ward partner. The pairs flip one axis
each; `production` is the composite gap between the production corner and the
grm2025-faithful reference.

Combined (`+d3`) Hessians are not stored on disk; they are rebuilt here as
`mace + d3` in fp64, exactly as `combine_d3.py` built the one that produced the
stored observables.

Runs where the Hessians are; `results/` comes back:

    python ablation_extract.py --structures goenn:10
    python ablation_extract.py --structures RSM0010 --dry-run

One structure per `<structure>.npz` (spectra) + `.json` (scalars, the resume
gate — written last); existing outputs are skipped unless `--overwrite`.

Families whose conditions are incomplete are skipped with a note, so a partial
campaign extracts what it can.
"""

import numpy as np

import argparse
import time
from pathlib import Path

import common

from sadmof.io import read_json, write_json, write_npz
from sadmof.provenance import provenance

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
RESULTS = HERE / "results" / "ablation"

FAMILIES = {
    "mace": (
        "mace_fp64+d3",
        "mace_fp32_mmhigh+d3",
        "mace_fp32+d3",
        "mace_fp64",
        "mace_fp32_mmhigh",
        "mace_fp32",
        "d3",
    ),
    "pet-xs": (
        "pet-xs_fp64_shadow",
        "pet-xs_fp64_noshadow",
        "pet-xs_fp32_noshadow_mmhigh",
        "pet-xs_fp32_noshadow",
    ),
    "pet-s": (
        "pet-s_fp64_shadow",
        "pet-s_fp64_noshadow",
        "pet-s_fp32_noshadow_mmhigh",
        "pet-s_fp32_noshadow",
    ),
}

# (name, ablated, reference-ward): each pair flips one axis, except `precision`
# (both precision rungs at once) and the `production` composites, which span
# the whole chain. The precision chain is fp64 -> fp32 at matmul "highest"
# (`precision_fp32`, number format alone, the production matmul
# setting) -> fp32 at the TF32 default (`precision_tf32`, disabled
# in production). The reference-ward side of each family's chain is
# `common.REFERENCE_VARIANTS` — keep them in sync.
PAIRS = {
    "mace": (
        ("precision_fp32", "mace_fp32_mmhigh+d3", "mace_fp64+d3"),
        ("precision_tf32", "mace_fp32+d3", "mace_fp32_mmhigh+d3"),
        ("precision", "mace_fp32+d3", "mace_fp64+d3"),
        ("precision_bare", "mace_fp32", "mace_fp64"),
        ("d3", "mace_fp64", "mace_fp64+d3"),
        ("production", "mace_fp32_mmhigh", "mace_fp64+d3"),
    ),
    "pet-xs": (
        ("shadow", "pet-xs_fp64_noshadow", "pet-xs_fp64_shadow"),
        ("precision_fp32", "pet-xs_fp32_noshadow_mmhigh", "pet-xs_fp64_noshadow"),
        ("precision_tf32", "pet-xs_fp32_noshadow", "pet-xs_fp32_noshadow_mmhigh"),
        ("precision", "pet-xs_fp32_noshadow", "pet-xs_fp64_noshadow"),
        ("production", "pet-xs_fp32_noshadow_mmhigh", "pet-xs_fp64_shadow"),
    ),
    "pet-s": (
        ("shadow", "pet-s_fp64_noshadow", "pet-s_fp64_shadow"),
        ("precision_fp32", "pet-s_fp32_noshadow_mmhigh", "pet-s_fp64_noshadow"),
        ("precision_tf32", "pet-s_fp32_noshadow", "pet-s_fp32_noshadow_mmhigh"),
        ("precision", "pet-s_fp32_noshadow", "pet-s_fp64_noshadow"),
        ("production", "pet-s_fp32_noshadow_mmhigh", "pet-s_fp64_shadow"),
    ),
}

# Record fields copied into each condition's row verbatim. `aux_flag` rides
# along because a PET `k_sel` overflow invalidates the Hessian, and that has to
# be visible in the committed dataset, not only in the cluster-side collect.
FIELDS = ("model", "dtype", "shadow", "label", "status", "aux_flag")
SECTIONS = {
    "system": ("n_atoms", "mass_amu"),
    "hessian": ("dtype", "combined", "raw_asymmetry_rel", "cached"),
    "observables": (
        "n_modes",
        "n_dropped",
        "n_dropped_asr",
        "cv_J_per_gK",
        "cv_J_per_gK_asr",
    ),
    "timings": ("hessian_cold_s", "total_s"),
}


def main():
    args = build_parser().parse_args()
    structures = common.roster(args.structures)

    todo = []
    for structure in structures:
        out = RESULTS / f"{structure}.json"
        # The gate re-extracts when a newly complete family appears; it cannot
        # see new conditions or pairs inside an existing family, so any change
        # to FAMILIES/PAIRS needs one --overwrite pass (as job 4132715 did).
        if out.exists() and not args.overwrite:
            have = set(read_json(out)["families"])
            fresh = set(complete_families(load_records(structure)))
            if fresh <= have:
                print(f"skip {out.name} (exists)")
                continue
            print(f"re-extract {structure}: newly complete {sorted(fresh - have)}")
        todo.append((structure, out))

    print(f"{len(structures)} structures, {len(todo)} to extract")
    if args.dry_run:
        for structure, _out in todo:
            print(f"  {structure}")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)

    failed = []
    for k, (structure, out) in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {structure}", flush=True)
        try:
            extract(structure, out)
        except Exception as e:  # noqa: BLE001 — one bad structure must not sink the rest
            print(f"  FAILED: {e!r}", flush=True)
            failed.append(structure)
    if failed:
        print(f"{len(failed)} failed: {failed}")


# ---------------------------------------------------------------------------
# One structure
# ---------------------------------------------------------------------------


def extract(structure, out_path):
    t0 = time.perf_counter()
    records = load_records(structure)

    families = complete_families(records)
    for family in FAMILIES.keys() - set(families):
        missing = [v for v in FAMILIES[family] if v not in records]
        print(f"  {family}: incomplete, skipping ({', '.join(missing)})")
    if not families:
        raise RuntimeError("no complete family")

    conditions = {v: flatten(records[v]) for family in families for v in FAMILIES[family]}

    spectra = {}
    for v in conditions:
        with np.load(OUTPUT / structure / records[v]["label"] / "observables.npz") as z:
            spectra[v] = {
                "": np.asarray(z["frequencies_cm1"]),
                "asr_": np.asarray(z["frequencies_cm1_asr"]),
            }

    pairs = []
    for family in families:
        for name, ablated, reference in PAIRS[family]:
            pairs.append(
                {
                    "family": family,
                    "ablation": name,
                    "ablated": ablated,
                    "reference": reference,
                    "dcv_pct": cv_errors(conditions[ablated], conditions[reference]),
                    **frequency_errors(spectra[ablated], spectra[reference]),
                    "hessian_error": hessian_error(structure, ablated, reference),
                }
            )

    n_atoms = next(iter(conditions.values()))["n_atoms"]
    meta = {
        "structure": structure,
        "n_atoms": n_atoms,
        "families": sorted(families),
        "conditions": conditions,
        "pairs": pairs,
        "timings_s": {"total": round(time.perf_counter() - t0, 2)},
        "provenance": provenance(),
    }

    write_npz(
        out_path.with_suffix(".npz"),
        {
            f"freq_{prefix}{v}": freqs
            for v, by in spectra.items()
            for prefix, freqs in by.items()
        },
    )
    # The `.json` is the resume gate in `main`, so it is written last.
    write_json(out_path, meta)

    worst = max(pairs, key=lambda p: abs(p["dcv_pct"]["asr"].get("300") or 0.0))
    print(
        f"  N={n_atoms} families={sorted(families)} worst |dCv(300K)| "
        f"{abs(worst['dcv_pct']['asr'].get('300') or 0.0):.3g} % "
        f"({worst['family']}/{worst['ablation']})",
        flush=True,
    )


def load_records(structure):
    """`{variant: record}` for every condition present and ok.

    A non-ok record (an OOM is a recorded result) counts as absent here: it
    makes its own family incomplete without sinking the structure's other
    families. Its existence is `collect.py`'s business.
    """
    records = {}
    for variant in common.variants():
        path = OUTPUT / structure / f"{structure}_{variant}" / "record.json"
        if not path.exists():
            continue
        record = read_json(path)
        if record["status"] != "ok":
            print(f"  {path.parent.name}: status {record['status']!r}, treating as absent")
            continue
        records[variant] = record
    return records


def complete_families(records):
    """Families whose every condition has an ok record, in `FAMILIES` order."""
    return [f for f, variants in FAMILIES.items() if all(v in records for v in variants)]


def flatten(record):
    row = {key: record.get(key) for key in FIELDS}
    # Absent in records predating the mmhigh rung: nothing set it, so the JAX
    # default (TF32 on Hopper) was in effect. Combines inherit their MLIP
    # source's config, so this is faithful for them too.
    row["matmul"] = (record.get("config") or {}).get("matmul_precision", "default")
    row["git_sha"] = ((record.get("provenance") or {}).get("git") or {}).get("sha")
    for section, keys in SECTIONS.items():
        content = record.get(section) or {}
        for key in keys:
            # Section-prefix on collision: `hessian.dtype` (the stored array's
            # precision) must not shadow the condition's model dtype.
            out = f"{section}_{key}" if key in row else key
            row[out] = content.get(key)
    return row


# ---------------------------------------------------------------------------
# Pair metrics
# ---------------------------------------------------------------------------


def cv_errors(ablated, reference):
    """Percent C_v deviation per temperature, both ASR variants."""
    out = {}
    for tag, key in (("asr", "cv_J_per_gK_asr"), ("raw", "cv_J_per_gK")):
        a, b = ablated.get(key) or {}, reference.get(key) or {}
        out[tag] = {t: 100 * (a[t] - ref) / ref for t, ref in b.items() if t in a and ref}
    return out


def frequency_errors(ablated, reference):
    """Elementwise spectrum deviation (modes sorted ascending on both sides)."""
    out = {}
    for prefix, tag in (("asr_", "asr"), ("", "raw")):
        diff = ablated[prefix] - reference[prefix]
        out[f"dnu_max_cm1_{tag}"] = float(np.abs(diff).max())
        out[f"dnu_rms_cm1_{tag}"] = float(np.sqrt(np.mean(diff**2)))
    return out


def hessian_error(structure, ablated, reference):
    """Frobenius and max-entry deviation between the pair's fp64 Hessians."""
    a = load_hessian(structure, ablated)
    b = load_hessian(structure, reference)
    if a.shape != b.shape:
        raise RuntimeError(f"{ablated} vs {reference}: shapes {a.shape} / {b.shape}")
    diff = a - b
    frob_ref = float(np.linalg.norm(b))
    frob_diff = float(np.linalg.norm(diff))
    return {
        "frob_ref": frob_ref,
        "frob_diff": frob_diff,
        "rel_frob": frob_diff / frob_ref,
        "max_abs_diff": float(np.abs(diff).max()),
        "max_abs_ref": float(np.abs(b).max()),
    }


def load_hessian(structure, variant):
    """One condition's real-block Hessian in fp64.

    `+d3` variants are rebuilt from their stored sources, exactly as
    `combine_d3.py` built them.
    """
    if variant.endswith("+d3"):
        mlip = load_hessian(structure, variant.removesuffix("+d3"))
        return mlip + load_hessian(structure, "d3")
    path = OUTPUT / structure / f"{structure}_{variant}" / "hessian_raw.npy"
    return np.load(path).astype(np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--structures",
        nargs="+",
        default=["goenn:10"],
        help="roster spec, as run.py: goenn:<k> | <identifier>",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


if __name__ == "__main__":
    main()
