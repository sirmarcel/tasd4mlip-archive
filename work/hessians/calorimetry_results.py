"""The calorimetry comparison for the giant MOFs, re-derived from the stores.

What the preprint quotes: predicted C_v(300 K) of the two giants — dense rung,
acoustic sum rule enforced — against published heat-capacity measurements
(MOF-177: Kloutse et al. 2015; MIL-101: Liu et al. 2017), and the relative
deviation in percent. Experimental values are the digitised curves in
`sources/processed/big-mofs/experimental_curves.npz`, interpolated linearly to
300 K. Run as a script to print the numbers as JSON; exits nonzero when a pair
is missing, a curve does not bracket 300 K, an experimental value drifts from
its pinned reference, or a deviation leaves the known regime (overestimation,
below 100 %) — so a wrong or stale store cannot quietly produce paper numbers.

    uv run python work/hessians/calorimetry_results.py

Predicted C_v is harmonic and per gram; the measured quantity is C_p, but for
these rigid frameworks at 300 K the difference is far below the deviations
being quoted.
"""

import numpy as np

import argparse
import json
import sys

import common
import ladder_load

CURVES = common.ROOT / "sources" / "processed" / "big-mofs" / "experimental_curves.npz"
TEMPERATURE = 300.0

SOURCES = {
    "mof177": "Kloutse et al. 2015",
    "mil101": "Liu et al. 2017",
}

# Pinned references for the interpolated experimental values, so a swapped or
# re-digitised curve trips the check instead of shifting the quoted deviations.
# Both are interpolations at TEMPERATURE, matching the predictions' grid; Liu's
# tabulated room-temperature point (298.15 K, 0.5226) sits below the mil101 pin.
EXPECTED_EXP = {"mof177": 0.7505, "mil101": 0.5297}
EXP_RTOL = 0.01
# The known regime: every model overestimates, and nothing is off by 2x.
MAX_DEVIATION_PCT = 100.0


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(results(), indent=2))


def results():
    ladders = ladder_load.ladders()

    with np.load(CURVES) as curves:
        if set(curves.files) != set(SOURCES):
            fail(f"curve store holds {sorted(curves.files)}, expected {sorted(SOURCES)}")
        experiment = {s: experimental_cp(curves[s], s) for s in SOURCES}

    out = {}
    for structure, exp in experiment.items():
        models = {}
        for model in common.EXACT_HOPS:
            cv = predicted_cv(ladders, structure, model)
            deviation = 100 * (cv - exp) / exp
            if cv <= exp:
                fail(
                    f"{structure}/{model}: predicted {cv:.4f} does not exceed "
                    f"experiment {exp:.4f} — known overestimation contradicted"
                )
            if deviation > MAX_DEVIATION_PCT:
                fail(f"{structure}/{model}: deviation {deviation:.1f} % above 100 %")
            models[model] = {"cv_300K": cv, "rel_dev_pct": deviation}
        out[structure] = {
            "exp_Cp_300K": exp,
            "source": SOURCES[structure],
            "models": models,
        }
    return out


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def experimental_cp(curve, structure):
    """The measured C_p at `TEMPERATURE`, linearly interpolated on the curve."""
    temperatures, cp = curve
    if not (np.diff(temperatures) > 0).all():
        fail(f"{structure}: curve temperatures are not strictly increasing")
    if not temperatures[0] <= TEMPERATURE <= temperatures[-1]:
        fail(
            f"{structure}: {TEMPERATURE} K outside the measured range "
            f"{temperatures[0]:g}-{temperatures[-1]:g} K"
        )
    value = float(np.interp(TEMPERATURE, temperatures, cp))
    if abs(value / EXPECTED_EXP[structure] - 1) > EXP_RTOL:
        fail(
            f"{structure}: interpolated C_p {value:.4f} J/(g K) is not within "
            f"{100 * EXP_RTOL:g} % of the pinned {EXPECTED_EXP[structure]}"
        )
    return value


def predicted_cv(ladders, structure, model):
    """The dense rung's gravimetric C_v at `TEMPERATURE`, ASR enforced."""
    dense = ladders.get((structure, model), {}).get("dense")
    if dense is None:
        fail(f"no dense rung for ({structure}, {model}) in results/ladder/")
    value = (dense.get("cv_J_per_gK_asr") or {}).get(f"{TEMPERATURE:g}")
    if value is None:
        fail(f"({structure}, {model}): dense rung has no ASR C_v at {TEMPERATURE:g} K")
    return value


def fail(message):
    sys.exit(f"calorimetry_results: {message}")


if __name__ == "__main__":
    main()
