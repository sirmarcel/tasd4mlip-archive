# /// script
# requires-python = ">=3.13, <3.14"
# dependencies = [
#     "fairchem-core",
#     "torch",
#     "torchtnt",
#     "ase",
#     "importlib_metadata",
#     "vesin",
# ]
#
# [tool.uv]
# override-dependencies = ["numpy>=2.0"]
# [tool.uv.sources]
# fairchem-core = { git = "https://github.com/n-gao/fairchem.git", branch = "ng/sybmolic_shapes", subdirectory = "packages/fairchem-core" }
# torchtnt = { git = "https://github.com/meta-pytorch/tnt" }
# ///
"""Pin torch UMA-S-1.2 energies and forces as a JSON fixture for `test_uma.py`.

Run from this directory:

    uv run build_uma_reference.py

Inputs:
    ../../sources/raw/uma/checkpoints/uma-s-1p2.pt   (gitignored)
    relaxed/RSM0010.xyz

Output:
    uma_reference.json

fairchem (and therefore torch) is not in the sadmof environment, so the numbers
are generated once here and committed. Everything runs at float64:
`base_precision_dtype` defaults to float32 and silently undoes a
`model.double()`, which is worth ~4e-7 eV on an 8-atom cell.

The graph is external (vesin, full list) and matches
`sadmof.models.uma.inputs.atoms_to_inputs` exactly, so the fixture isolates the
model port rather than the neighbour list. Stock rotation settings are used —
the port's Euler path with the roll pinned to zero is checked *against* them.

Both MOF-relevant heads are pinned. They share the backbone up to the final
norm and diverge only in the dataset embedding (which enters the MOLE routing,
so the *whole* mixed backbone differs), the head expert, and the element
references — enough that checking one would not cover the other.
"""

import numpy as np

import json
from pathlib import Path

import torch

HERE = Path(__file__).parent
CHECKPOINT = HERE / "../../sources/raw/uma/checkpoints/uma-s-1p2.pt"
CUTOFF = 6.0
TASKS = ("omat", "odac")


def structures() -> dict:
    from ase.build import bulk
    from ase.io import read

    si8 = bulk("Si", "diamond", a=5.43, cubic=True)
    si8.rattle(0.1, seed=1)
    si8_supercell = bulk("Si", "diamond", a=5.43, cubic=True) * [2, 2, 2]
    si8_supercell.rattle(0.1, seed=1)
    return {
        "si8": si8,
        "si8_222": si8_supercell,
        "RSM0010": read(HERE / "relaxed" / "RSM0010.xyz"),
    }


def neighbor_list(atoms):
    from vesin import NeighborList

    nl = NeighborList(cutoff=CUTOFF, full_list=True, sorted=True)
    i, j, shifts = nl.compute(
        points=atoms.positions, box=atoms.cell[:], periodic=atoms.pbc, quantities="ijS"
    )
    return i.astype(np.int64), j.astype(np.int64), shifts.astype(np.float64)


def load_predictor():
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    return load_predict_unit(
        str(CHECKPOINT.resolve()),
        device="cpu",
        inference_settings=InferenceSettings(
            tf32=False,
            activation_checkpointing=False,
            merge_mole=False,
            compile=False,
            external_graph_gen=True,
            internal_graph_gen_version=2,
            base_precision_dtype="float64",
        ),
    )


def predict(predictor, atoms, task) -> dict:
    from fairchem.core.datasets.atomic_data import AtomicData

    i, j, shifts = neighbor_list(atoms)
    data = AtomicData(
        pos=torch.tensor(atoms.positions, dtype=torch.float64, requires_grad=True),
        atomic_numbers=torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long),
        cell=torch.tensor(np.array(atoms.cell[:]), dtype=torch.float64).unsqueeze(0),
        pbc=torch.tensor(np.array(atoms.pbc)).unsqueeze(0),
        natoms=torch.tensor([len(atoms)]),
        edge_index=torch.tensor(np.stack([j, i])),
        cell_offsets=torch.tensor(shifts),
        nedges=torch.tensor([len(i)]),
        charge=torch.tensor([0]),
        spin=torch.tensor([0]),
        fixed=torch.zeros(len(atoms), dtype=torch.long),
        tags=torch.zeros(len(atoms), dtype=torch.long),
        batch=torch.zeros(len(atoms), dtype=torch.long),
        dataset=task,
    )
    out = predictor.predict(data)
    return {
        "n_atoms": len(atoms),
        "n_edges": int(len(i)),
        "energy": float(out["energy"].detach().reshape(-1)[0]),
        "forces": out["forces"].detach().numpy().tolist(),
    }


def main() -> None:
    # One predictor serves every dataset: with merge_mole=False the head's
    # one-hot expert coefficients are rebuilt from `data.dataset` each forward.
    predictor = load_predictor()
    records = {"cutoff": CUTOFF, "charge": 0, "spin": 0, "tasks": {}}
    for task in TASKS:
        records["tasks"][task] = {}
        for name, atoms in structures().items():
            record = predict(predictor, atoms, task)
            records["tasks"][task][name] = record
            print(
                f"{task:5s} {name}: {record['n_atoms']} atoms, "
                f"{record['n_edges']} edges, E = {record['energy']!r} eV"
            )

    path = HERE / "uma_reference.json"
    path.write_text(json.dumps(records, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
