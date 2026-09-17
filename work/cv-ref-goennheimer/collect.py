"""Compare computed frequencies + C_v against the Goennheimer references.

Merges every npz under `output/<label>/` with the references riding in the
canonical relaxed structures' `info` (`goenn_uc_cv_{T}`) and the published
spectra (`sources/processed/goennheimer+moosavi/frequencies_uc.npz`), writes
`output/<label>/summary.csv`, and prints aggregate deviation statistics.

    python collect.py                  # default label dense_float64
    python collect.py dense_float32
"""

import numpy as np

import csv
import json
import sys
from pathlib import Path

from ase.io import read

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RELAXED = ROOT / "work" / "relax-goennheimer" / "output" / "mace-mp0+d3_bfgs_float64"
GOENN_FREQS = ROOT / "sources" / "processed" / "goennheimer+moosavi" / "frequencies_uc.npz"


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "dense_float64"
    out_dir = HERE / "output" / label

    by_id = {
        path.parent.name: read(str(path)) for path in sorted(RELAXED.glob("*/relaxed.xyz"))
    }
    ref_freqs = np.load(GOENN_FREQS)

    rows = []
    for path in sorted(out_dir.glob("*.npz")):
        data = np.load(path)
        ident = str(data["identifier"])
        atoms = by_id[ident]
        temps = data["temperatures"]
        cv = data["cv_gravimetric"]

        ref_cv = np.array([atoms.info[f"goenn_uc_cv_{int(T)}"] for T in temps])
        d_cv = np.abs(cv - ref_cv)
        cv300 = cv[list(temps).index(300.0)]
        ref300 = ref_cv[list(temps).index(300.0)]

        dnu = np.abs(np.sort(data["frequencies_cm1"]) - np.sort(ref_freqs[ident]))

        rows.append(
            {
                "identifier": ident,
                "n_atoms": int(data["n_atoms"]),
                "goenn_uc_converged": bool(atoms.info["goenn_uc_converged"]),
                "max_dnu_cm1": dnu.max(),
                "rms_dnu_cm1": np.sqrt((dnu**2).mean()),
                "cv300": cv300,
                "goenn_cv300": ref300,
                "max_dcv": d_cv.max(),
                "rel_dcv300": abs(cv300 - ref300) / ref300,
                "wall_mace_s": float(data["wall_hessian_mace_s"]),
                "wall_d3_s": float(data["wall_hessian_d3_s"]),
                "git_sha": json.loads(str(data["provenance"]))["git"]["sha"][:7],
            }
        )

    if not rows:
        sys.exit(f"no results under {out_dir}")

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def stats(key, subset):
        vals = np.array([r[key] for r in subset])
        return f"median {np.median(vals):.3g}  p95 {np.percentile(vals, 95):.3g}  max {vals.max():.3g}"

    n_total = len(by_id)
    print(f"{len(rows)}/{n_total} structures collected -> {csv_path}")
    for converged in (True, False):
        subset = [r for r in rows if r["goenn_uc_converged"] is converged]
        if not subset:
            continue
        print(f"\ngoenn_uc_converged={converged} ({len(subset)}):")
        print(f"  max|dnu| (cm^-1): {stats('max_dnu_cm1', subset)}")
        print(f"  rms|dnu| (cm^-1): {stats('rms_dnu_cm1', subset)}")
        print(f"  max|dCv| (J/g/K): {stats('max_dcv', subset)}")
        print(f"  rel|dCv| at 300K: {stats('rel_dcv300', subset)}")

    worst = sorted(rows, key=lambda r: -r["max_dcv"])[:10]
    print("\nworst 10 by max|dCv|:")
    for r in worst:
        print(
            f"  {r['identifier']}  N={r['n_atoms']:3d}  conv={r['goenn_uc_converged']}  "
            f"max|dnu|={r['max_dnu_cm1']:8.3f}  max|dCv|={r['max_dcv']:.2e}"
        )


if __name__ == "__main__":
    main()
