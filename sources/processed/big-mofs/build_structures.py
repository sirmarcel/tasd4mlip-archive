"""
Build structures.xyz and experimental_curves.npz: the big benchmark MOFs that
are not in the Moosavi et al. 2022 experimental set. MOF-177 and MIL-101(Cr)
are paired with experimental Cp(T) reference curves transcribed from the
primary calorimetry literature; MOF-210 and MIL-100(Cr) are structure-only
(no calorimetry data located; they serve as scale benchmarks).

Run from this directory. Expects ../../raw/raspa2/ to have been populated via
its download.sh. The experimental reference values are hardcoded below, with
citations inline; they were verified against the source PDFs
(../../raw/literature/{lxzz17,kzcc15}.pdf).
"""

import numpy as np

from collections import Counter
from pathlib import Path

from ase.io import read, write

HERE = Path(__file__).parent
RASPA2 = HERE / "../../raw/raspa2"

OUTPUT_XYZ = HERE / "structures.xyz"
OUTPUT_CURVES = HERE / "experimental_curves.npz"


# -----------------------------------------------------------------------------
# MIL-101(Cr) — Liu et al., J. Therm. Anal. Calorim. 129 (2017) 509-514
# DOI 10.1007/s10973-017-6168-9. lxzz17.
#
# Temperature-modulated DSC on a TA Q1000, sapphire (alpha-Al2O3) calibration,
# 10 K/min ramp, argon. Sample activated at 220 C in vacuum overnight and
# handled under glovebox / IR-lamp to prevent rehydration. Three replicates,
# calibration agreement +/- 2.2%, polynomial fit residual +/- 0.4%.
#
# Paper reports molar Cp (J/mol/K) for the dehydrated framework
# Cr3F(H2O)2O(BDC)3 with M = 719.39 g/mol (two coordinated waters retained,
# lattice water removed). Values below are the raw Table 2 points (253-413 K,
# every 5 K, plus the paper's own 298.15 K entry), as (T [K], Cp_molar
# [J/mol/K]) pairs. The scalar at 298.15 K (375.92 J/mol/K) is the value the
# paper explicitly quotes in-text; the 300 K value is a linear interpolation
# across the 298.15 / 303 K entries.
# -----------------------------------------------------------------------------

MIL101_MOLAR_MASS_G_PER_MOL = 719.39

# fmt: off
MIL101_LIU2017_TABLE2_MOLAR = np.array(
    [
        (253.00, 241.90), (258.00, 258.17), (263.00, 273.91), (268.00, 289.20),
        (273.00, 304.13), (278.00, 318.77), (283.00, 333.18), (288.00, 347.41),
        (293.00, 361.50), (298.00, 375.49), (298.15, 375.92), (303.00, 389.42),
        (308.00, 403.31), (313.00, 417.16), (318.00, 431.01), (323.00, 444.85),
        (328.00, 458.70), (333.00, 472.55), (338.00, 486.41), (343.00, 500.27),
        (348.00, 514.14), (353.00, 528.00), (358.00, 541.87), (363.00, 555.73),
        (368.00, 569.58), (373.00, 583.44), (378.00, 597.30), (383.00, 611.17),
        (388.00, 625.06), (393.00, 639.00), (398.00, 653.01), (403.00, 667.11),
        (408.00, 681.35), (413.00, 695.75),
    ],
    dtype=np.float64,
)
# fmt: on


def mil101_curve() -> np.ndarray:
    """Return MIL-101(Cr) gravimetric Cp curve as (2, N) array [T_K, Cp].

    Converts Liu et al. 2017 Table 2 molar Cp entries to gravimetric by
    dividing by the dehydrated framework molar mass (719.39 g/mol).
    """
    t = MIL101_LIU2017_TABLE2_MOLAR[:, 0]
    cp_molar = MIL101_LIU2017_TABLE2_MOLAR[:, 1]
    cp_grav = cp_molar / MIL101_MOLAR_MASS_G_PER_MOL
    return np.stack([t, cp_grav], axis=0)


# Scalar reference value. Liu et al. quote Cp(298.15 K) = 375.92 J/mol/K
# explicitly in the text; we expose both the as-quoted 298.15 K value and a
# linearly interpolated 300 K value to avoid any silent off-by-2-K confusion
# in downstream comparisons.
MIL101_CP_298_15 = 375.92 / MIL101_MOLAR_MASS_G_PER_MOL  # 0.5226 J/g/K
MIL101_CP_300 = float(
    np.interp(300.0, mil101_curve()[0], mil101_curve()[1])
)  # 0.5297 J/g/K


# -----------------------------------------------------------------------------
# MOF-177 — Kloutse et al., Microporous Mesoporous Mater. 217 (2015) 1-5
# DOI 10.1016/j.micromeso.2015.05.047. kzcc15.
#
# Calvet calorimeter (SETARAM BT 2.15), 80-320 K range, 1 K/min ramp, NIST
# sapphire + Cu calibration. Sample synthesized in-house per Tranchemontagne
# et al. Tetrahedron 64 (2008) 8553; outgassed at 125 C before measurement,
# handled in argon glovebox (< 0.1 ppm O2/H2O). Propagated error ~1% of Cp.
#
# Kloutse fits a 5th-order polynomial (their Eq. 5, gravimetric, T in K) over
# the measured 80-320 K range:
#   Cp(T) = a0 + a1 T + a2 T^2 + a3 T^3 + a4 T^4       [J/g/K]
# with the coefficients below. The 300 K value 0.7505 J/g/K is obtained by
# evaluating this polynomial — the paper itself does not tabulate raw data.
# The curve is sampled on a 5 K grid over the polynomial's valid range.
# -----------------------------------------------------------------------------

MOF177_KLOUTSE2015_EQ5_COEFFS = np.array(
    [0.13525, -3.043e-3, 4.907e-5, -1.784e-7, 2.381e-10],
    dtype=np.float64,
)

MOF177_KLOUTSE2015_VALID_RANGE_K = (80.0, 320.0)


def mof177_curve() -> np.ndarray:
    """Return MOF-177 gravimetric Cp curve as (2, N) [T_K, Cp] evaluated on a
    5 K grid across Kloutse et al.'s fit range (80-320 K).
    """
    t_lo, t_hi = MOF177_KLOUTSE2015_VALID_RANGE_K
    t = np.arange(t_lo, t_hi + 1e-6, 5.0, dtype=np.float64)
    cp = np.polynomial.polynomial.polyval(t, MOF177_KLOUTSE2015_EQ5_COEFFS)
    return np.stack([t, cp], axis=0)


MOF177_CP_300 = float(
    np.polynomial.polynomial.polyval(300.0, MOF177_KLOUTSE2015_EQ5_COEFFS)
)  # 0.7505 J/g/K


# -----------------------------------------------------------------------------
# Per-entry spec
# -----------------------------------------------------------------------------

ENTRIES = [
    {
        "identifier": "mil101",
        "common_name": "MIL-101(Cr)",
        "chemical_formula": "Cr3F(H2O)2O(BDC)3",
        "cif": RASPA2 / "MIL-101-primitive.cif",
        "source_cif_note": (
            "github.com/numat/RASPA2 structures/mofs/cif/MIL-101-primitive.cif, "
            "shipped upstream as the rhombohedral primitive cell"
        ),
        "exp_method": "TMDSC",
        "exp_paper_bibkey": "lxzz17",
        "exp_paper_citation": (
            "Liu, Xu, Liu, Zhou, Zhao, J. Therm. Anal. Calorim. 129 (2017) "
            "509-514, DOI 10.1007/s10973-017-6168-9"
        ),
        "exp_sample": (
            "dehydrated framework, activated at 220 C in vacuum overnight, "
            "handled in glovebox"
        ),
        "exp_cp_300": MIL101_CP_300,
        "exp_cp_298_15": MIL101_CP_298_15,
        "exp_cp_uncertainty_pct": 2.2,
        "curve_fn": mil101_curve,
        "curve_range_K": (253.0, 413.0),
        "curve_source": "Liu et al. 2017, Table 2 (34 points, every 5 K)",
    },
    {
        "identifier": "mof177",
        "common_name": "MOF-177",
        "chemical_formula": "Zn4O(BTB)2",
        "cif": RASPA2 / "MOF-177.cif",
        "source_cif_note": (
            "github.com/numat/RASPA2 structures/mofs/cif/MOF-177.cif (primitive cell)"
        ),
        "exp_method": "Calvet",
        "exp_paper_bibkey": "kzcc15",
        "exp_paper_citation": (
            "Kloutse, Zacharia, Cossement, Chahine, Microporous Mesoporous "
            "Mater. 217 (2015) 1-5, DOI 10.1016/j.micromeso.2015.05.047"
        ),
        "exp_sample": (
            "in-house synthesis (Tranchemontagne et al. 2008), outgassed at "
            "125 C, argon glovebox handling"
        ),
        "exp_cp_300": MOF177_CP_300,
        "exp_cp_uncertainty_pct": 1.0,
        "curve_fn": mof177_curve,
        "curve_range_K": MOF177_KLOUTSE2015_VALID_RANGE_K,
        "curve_source": (
            "Kloutse et al. 2015 Eq. 5 (5th-order polynomial, valid 80-320 K)"
        ),
    },
    # Structure-only entries: no experimental Cp reference located in the
    # calorimetry literature; included as scale benchmarks.
    {
        "identifier": "mof210",
        "common_name": "MOF-210",
        "chemical_formula": "Zn4O(BTE)4/3(BPDC)",
        "cif": RASPA2 / "MOF-210-primitive.cif",
        "source_cif_note": (
            "github.com/numat/RASPA2 structures/mofs/cif/MOF-210-primitive.cif, "
            "shipped upstream as the rhombohedral primitive cell"
        ),
    },
    {
        "identifier": "mil100",
        "common_name": "MIL-100(Cr)",
        "chemical_formula": "Cr3F(H2O)2O(BTC)2",
        "cif": RASPA2 / "MIL-100-primitive.cif",
        "source_cif_note": (
            "github.com/numat/RASPA2 structures/mofs/cif/MIL-100-primitive.cif, "
            "shipped upstream as the rhombohedral primitive cell"
        ),
    },
]


def build_frames() -> tuple[list, dict[str, np.ndarray]]:
    frames = []
    curves: dict[str, np.ndarray] = {}
    for entry in ENTRIES:
        cif_path = entry["cif"]
        if not cif_path.exists():
            raise SystemExit(f"Missing CIF: {cif_path} — run ../../raw/raspa2/download.sh")
        atoms = read(cif_path)
        symbols = set(atoms.get_chemical_symbols())

        atoms.info.clear()
        atoms.info["identifier"] = entry["identifier"]
        atoms.info["common_name"] = entry["common_name"]
        atoms.info["chemical_formula"] = entry["chemical_formula"]
        atoms.info["source_cif"] = str(cif_path.name)
        atoms.info["source_cif_note"] = entry["source_cif_note"]
        if "curve_fn" in entry:
            atoms.info["exp_method"] = entry["exp_method"]
            atoms.info["exp_paper_bibkey"] = entry["exp_paper_bibkey"]
            atoms.info["exp_paper_citation"] = entry["exp_paper_citation"]
            atoms.info["exp_sample"] = entry["exp_sample"]
            atoms.info["exp_cp_300"] = float(entry["exp_cp_300"])
            if "exp_cp_298_15" in entry:
                atoms.info["exp_cp_298_15"] = float(entry["exp_cp_298_15"])
            atoms.info["exp_cp_uncertainty_pct"] = float(
                entry["exp_cp_uncertainty_pct"]
            )
            atoms.info["exp_curve_range_K"] = (
                f"{entry['curve_range_K'][0]:.1f}-{entry['curve_range_K'][1]:.1f}"
            )
            atoms.info["exp_curve_source"] = entry["curve_source"]
            curves[entry["identifier"]] = entry["curve_fn"]()
        atoms.info["n_atoms"] = len(atoms)
        atoms.info["elements"] = "-".join(sorted(symbols))

        frames.append(atoms)

    frames.sort(key=lambda a: a.info["n_atoms"])
    return frames, curves


def summarize(frames: list, curves: dict[str, np.ndarray]) -> None:
    print(f"Built {len(frames)} frames")
    methods = Counter(a.info.get("exp_method", "none") for a in frames)
    print(f"  exp_method: {dict(methods)}")
    for a in frames:
        identifier = a.info["identifier"]
        line = (
            f"  {identifier:7s}  {a.info['common_name']:<14s}  "
            f"N={a.info['n_atoms']:5d}  "
        )
        if identifier in curves:
            curve = curves[identifier]
            t_min, t_max = curve[0, 0], curve[0, -1]
            cp300 = float(np.interp(300.0, curve[0], curve[1]))
            line += (
                f"method={a.info['exp_method']:<6s}  "
                f"Cp(300K)={a.info['exp_cp_300']:.4f}  "
                f"curve[300K]={cp300:.4f}  "
                f"curve: {curve.shape[1]} pts, T=[{t_min:.1f}, {t_max:.1f}] K"
            )
        else:
            line += "structure-only (no experimental Cp)"
        print(line)


def main() -> None:
    frames, curves = build_frames()

    write(OUTPUT_XYZ, frames)
    print(f"Wrote {OUTPUT_XYZ}")

    np.savez(OUTPUT_CURVES, **curves)
    print(f"Wrote {OUTPUT_CURVES}")

    summarize(frames, curves)


if __name__ == "__main__":
    main()
