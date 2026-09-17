"""
Extract MACE-MP-0+D3(BJ) phonon frequencies (1D float64, cm^-1, one array per
structure, keyed by identifier) from the Gönnheimer `extended_*.csv` `freq`
columns into frequencies_uc.npz and frequencies_sc.npz.

The 2x2x2 CSV has an empty `freq` cell for RSM1854 (upstream CUDA OOM); that
identifier is omitted from frequencies_sc.npz.

Run from this directory. Expects ../../raw/goennheimer-2025/ to be a checkout
of Nilsgoe/AD_heat_capacity.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).parent
GOENN = HERE / "../../raw/goennheimer-2025"

GOENN_UC_CSV = GOENN / "dft_comparison/extended_C_v_screening_opt_MACE_for_CVs_BFGS_used_opt_sum_d3.csv"
GOENN_SC_CSV = GOENN / "s222_comparison/extended_combined_C_v_screening_opt_MACE_for_CVs_BFGS_s222_used_opt_sum_d3.csv"

OUTPUT_UC = HERE / "frequencies_uc.npz"
OUTPUT_SC = HERE / "frequencies_sc.npz"


def parse_freq_column(csv_path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    df = pd.read_csv(csv_path)
    out: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    for _, row in df.iterrows():
        identifier = str(row["File_Name"]).removesuffix(".traj")
        raw = row["freq"]
        if not isinstance(raw, str):
            skipped.append(identifier)
            continue
        out[identifier] = np.asarray(ast.literal_eval(raw), dtype=np.float64)
    return out, skipped


def main() -> None:
    uc, uc_skipped = parse_freq_column(GOENN_UC_CSV)
    np.savez(OUTPUT_UC, **uc)
    print(f"Wrote {OUTPUT_UC} ({len(uc)} structures, skipped {uc_skipped})")

    sc, sc_skipped = parse_freq_column(GOENN_SC_CSV)
    np.savez(OUTPUT_SC, **sc)
    print(f"Wrote {OUTPUT_SC} ({len(sc)} structures, skipped {sc_skipped})")


if __name__ == "__main__":
    main()
