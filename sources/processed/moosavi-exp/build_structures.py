"""
Build structures.xyz and experimental_curves.npz: the 8 experimentally-measured
MOFs that Moosavi et al. (Nat. Mater. 2022) used to validate their ML heat-
capacity predictor against calorimetry, one extended-XYZ frame per MOF plus a
companion NPZ with the full experimental Cp(T) curves.

Run from this directory. Expects ../../raw/moosavi-2022/Figures/Fig4/Fig_exp/
cv_exp/ to have been unzipped in place from Figures.zip.

Scope. The upstream cv_names.csv lists 11 experimental MOFs (mof1..mof11), but
only mof1..mof8 have CIFs and full experimental curves in the deposit. The
remaining three (MOF-74-Co, MOF-74-Zn, MOF-5) have no crystal structure in the
deposit and are not included here; see the README for what reference data they
do have.
"""

import numpy as np

from collections import Counter
from pathlib import Path

import pandas as pd
from ase.io import read, write

HERE = Path(__file__).parent
CV_EXP = HERE / "../../raw/moosavi-2022/Figures/Fig4/Fig_exp/cv_exp"
CIF_DIR = CV_EXP / "cifs"
CV_NAMES_CSV = CV_EXP / "cv_names.csv"
EXPERIMENTAL_CSV = CV_EXP / "experimental_data.csv"

OUTPUT_XYZ = HERE / "structures.xyz"
OUTPUT_CURVES = HERE / "experimental_curves.npz"

INCLUDED_IDS = ("mof1", "mof2", "mof3", "mof4", "mof5", "mof6", "mof7", "mof8")

# Common chemical names for the 8 included MOFs. Upstream cv_names.csv only
# gives CIF filenames; these clean names come from the Moosavi paper (Table S1
# / Fig. 4) and the original crystallographic publications referenced by the
# CIF filenames.
COMMON_NAMES = {
    "mof1": "Ca(ndc)(DMF)",
    "mof2": "Zn2(D-cam)2(bpy)",
    "mof3": "Co2(L-asp)2(bpe)",
    "mof4": "Co2(S-mal)2(bpy)",
    "mof5": "Co3(btc)(DMF)3(HCOO)3",
    "mof6": "Cu(INA)2",
    "mof7": "Mn3(ndc)3(DMF)4",
    "mof8": "ZIF-8",
}

CELSIUS_TO_KELVIN = 273.15


def load_experimental_curves(methods: dict[str, str]) -> dict[str, np.ndarray]:
    """Return a dict identifier -> (2, N) float64 array with rows [T_K, Cp].

    The upstream CSV stores each MOF as a (T_mofN, Cp_mofN) column pair, with
    MOFs interleaved in an arbitrary order. Curves are re-sorted by
    temperature and trimmed to the rows where both T and Cp are finite.

    Temperature-unit gotcha. The two halves of experimental_data.csv use
    different units: the DSC-measured MOFs (5 of 8) store T in degrees
    Celsius (the 20-200 C DSC range reported in Moosavi et al. 2022), while
    the Lit-sourced MOFs (3 of 8) store T in Kelvin. Both are normalised to
    Kelvin here so the downstream consumer sees one consistent axis. The unit
    is selected per identifier from the exp_method column of cv_names.csv.
    """
    df = pd.read_csv(EXPERIMENTAL_CSV)
    curves: dict[str, np.ndarray] = {}
    for identifier in INCLUDED_IDS:
        t_col = f"T_{identifier}"
        cp_col = f"Cp_{identifier}"
        if t_col not in df.columns or cp_col not in df.columns:
            raise KeyError(f"Missing {t_col}/{cp_col} in {EXPERIMENTAL_CSV}")
        t = df[t_col].to_numpy(dtype=np.float64)
        cp = df[cp_col].to_numpy(dtype=np.float64)
        finite = np.isfinite(t) & np.isfinite(cp)
        t, cp = t[finite], cp[finite]
        method = methods[identifier]
        if method == "DSC":
            t = t + CELSIUS_TO_KELVIN
        elif method != "Lit":
            raise ValueError(
                f"Unknown exp_method {method!r} for {identifier}; "
                f"cannot determine temperature unit"
            )
        order = np.argsort(t)
        curves[identifier] = np.stack([t[order], cp[order]], axis=0)
    return curves


def build_frames() -> list:
    names = pd.read_csv(CV_NAMES_CSV).set_index("id")

    frames = []
    for identifier in INCLUDED_IDS:
        row = names.loc[identifier]
        cif_filename = str(row["name"])
        cif_path = CIF_DIR / cif_filename
        if not cif_path.exists():
            raise SystemExit(f"Missing CIF for {identifier}: {cif_path}")
        atoms = read(cif_path)
        symbols = set(atoms.get_chemical_symbols())

        atoms.info.clear()
        atoms.info["identifier"] = identifier
        atoms.info["common_name"] = COMMON_NAMES[identifier]
        atoms.info["source_cif"] = cif_filename
        atoms.info["exp_method"] = str(row["exp_method"])
        atoms.info["density_g_cm3"] = float(row["density"])
        atoms.info["exp_cp_300"] = float(row["Cp@300K"])
        atoms.info["n_atoms"] = len(atoms)
        atoms.info["elements"] = "-".join(sorted(symbols))
        frames.append(atoms)

    frames.sort(key=lambda a: a.info["n_atoms"])
    return frames


def summarize(frames: list, curves: dict[str, np.ndarray]) -> None:
    print(f"Built {len(frames)} frames")
    methods = Counter(a.info["exp_method"] for a in frames)
    print(f"  exp_method: {dict(methods)}")
    for a in frames:
        identifier = a.info["identifier"]
        curve = curves[identifier]
        t_min, t_max = curve[0, 0], curve[0, -1]
        print(
            f"  {identifier:5s}  {a.info['common_name']:<28s}  "
            f"N={a.info['n_atoms']:4d}  "
            f"method={a.info['exp_method']:<3s}  "
            f"Cp(300K)={a.info['exp_cp_300']:.4f}  "
            f"curve: {curve.shape[1]} pts, T=[{t_min:.1f}, {t_max:.1f}] K"
        )


def main() -> None:
    frames = build_frames()
    methods = {a.info["identifier"]: a.info["exp_method"] for a in frames}
    curves = load_experimental_curves(methods)

    write(OUTPUT_XYZ, frames)
    print(f"Wrote {OUTPUT_XYZ}")

    np.savez(OUTPUT_CURVES, **curves)
    print(f"Wrote {OUTPUT_CURVES}")

    summarize(frames, curves)


if __name__ == "__main__":
    main()
