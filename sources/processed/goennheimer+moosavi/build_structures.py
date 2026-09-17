"""
Build structures.xyz: merge the Moosavi DFT CIFs / Cv reference with the
Gönnheimer MACE-MP-0+D3(BJ) Cv reference into one extended-XYZ file (one frame
per structure, sorted by atom count).

Run from this directory. Expects ../../raw/moosavi-2022/{DFT_calculations,
database_heat_capacity}/ to have been unzipped in place, and
../../raw/goennheimer-2025/ to be a checkout of Nilsgoe/AD_heat_capacity.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from ase.io import read, write


HERE = Path(__file__).parent
MOOSAVI = HERE / "../../raw/moosavi-2022"
GOENN = HERE / "../../raw/goennheimer-2025"

MOOSAVI_CIF_DIR = MOOSAVI / "DFT_calculations/DFT_structures"
MOOSAVI_DFT_CSV = MOOSAVI / "database_heat_capacity/DFT/DFT_cp_allstructures.csv"
GOENN_UC_CSV = GOENN / "dft_comparison/C_v_screening_opt_MACE_for_CVs_BFGS_used_opt_sum_d3.csv"
GOENN_SC_CSV = GOENN / "s222_comparison/combined_C_v_screening_opt_MACE_for_CVs_BFGS_s222_used_opt_sum_d3.csv"

OUTPUT = HERE / "structures.xyz"

DFT_TEMPERATURES = (250, 275, 300, 325, 350, 375, 400)
GOENN_TEMPERATURES = tuple(range(250, 401, 10))


def load_moosavi_dft() -> dict[str, dict]:
    df = pd.read_csv(MOOSAVI_DFT_CSV, index_col=0)
    out: dict[str, dict] = {}
    for identifier, row in df.iterrows():
        raw_type = str(row["structure_type"])
        entry = {"structure_type": "zeolite" if raw_type == "zeo" else raw_type}
        for T in DFT_TEMPERATURES:
            value = row[f"Cv_gravimetric_{T:.2f}"]
            if not pd.isna(value):
                entry[f"dft_cv_{T}"] = float(value)
        out[str(identifier)] = entry
    return out


def _load_goenn_csv(csv_path: Path, key_prefix: str) -> dict[str, dict]:
    df = pd.read_csv(csv_path)
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        identifier = str(row["File_Name"]).removesuffix(".traj")
        converged_raw = row.get("Opt")
        if isinstance(converged_raw, str):
            converged = converged_raw.strip().lower() == "true"
        else:
            converged = bool(converged_raw) and not pd.isna(converged_raw)
        entry = {f"{key_prefix}_converged": converged}
        for T in GOENN_TEMPERATURES:
            value = row[f"Cv_gravimetric_{T}"]
            entry[f"{key_prefix}_cv_{T}"] = (
                float("nan") if pd.isna(value) else float(value)
            )
        out[identifier] = entry
    return out


def main() -> None:
    dft = load_moosavi_dft()
    goenn_uc = _load_goenn_csv(GOENN_UC_CSV, "goenn_uc")
    goenn_sc = _load_goenn_csv(GOENN_SC_CSV, "goenn_sc")

    frames: list = []
    missing_dft: list[str] = []
    missing_cif: list[str] = []
    for identifier in sorted(goenn_uc.keys()):
        cif_path = MOOSAVI_CIF_DIR / f"{identifier}.cif"
        if not cif_path.exists():
            missing_cif.append(identifier)
            continue
        atoms = read(cif_path)
        symbols = set(atoms.get_chemical_symbols())

        dft_entry = dft.get(identifier)
        if dft_entry is None:
            missing_dft.append(identifier)
            structure_type = "unknown"
            dft_fields: dict = {}
        else:
            structure_type = dft_entry["structure_type"]
            dft_fields = {k: v for k, v in dft_entry.items() if k != "structure_type"}

        atoms.info.clear()
        atoms.info["identifier"] = identifier
        atoms.info["structure_type"] = structure_type
        atoms.info["n_atoms"] = len(atoms)
        atoms.info["elements"] = "-".join(sorted(symbols))
        atoms.info.update(dft_fields)
        atoms.info.update(goenn_uc[identifier])
        atoms.info.update(goenn_sc[identifier])
        frames.append(atoms)

    frames.sort(key=lambda a: a.info["n_atoms"])
    if missing_cif:
        raise SystemExit(f"Missing CIFs in Moosavi share for: {missing_cif}")

    write(OUTPUT, frames)
    print(f"Wrote {OUTPUT} ({len(frames)} frames)")
    print(f"  structure_type: {dict(Counter(a.info['structure_type'] for a in frames))}")
    n = [a.info["n_atoms"] for a in frames]
    print(f"  n_atoms: min={min(n)} max={max(n)} median={sorted(n)[len(n)//2]}")
    if missing_dft:
        print(f"  no DFT Cv reference: {missing_dft}")


if __name__ == "__main__":
    main()
