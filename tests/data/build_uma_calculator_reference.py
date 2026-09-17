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
"""Pin stock `FAIRChemCalculator` energies, forces and stresses for `test_uma_calculator.py`.

Run from this directory:

    uv run build_uma_calculator_reference.py

Inputs:
    ../../sources/raw/uma/checkpoints/uma-s-1p2.pt   (gitignored)
    relaxed/RSM0010.xyz

Output:
    uma_calculator_reference.json

Where `build_uma_reference.py` feeds fairchem an *external* vesin graph to
isolate the model port, this one runs the stock calculator end to end — its own
internal neighbour list, its own element references and normaliser — because
that is what `UMACalculator` has to reproduce.

Geometries are written into the fixture rather than rebuilt from `ase.build` in
the test, so a change in ASE's `bulk`/`rattle` can never silently move the
structure out from under a pinned number. Si8 carries a small symmetric strain
so its stress has non-zero off-diagonal components.

Everything runs at float64: `base_precision_dtype` defaults to float32 and
silently undoes a `model.double()`.
"""

import numpy as np

import json
from pathlib import Path

HERE = Path(__file__).parent
CHECKPOINT = HERE / "../../sources/raw/uma/checkpoints/uma-s-1p2.pt"
TASKS = ("omat", "odac")

# Small symmetric strain, applied to Si8 so the stress is not near-hydrostatic.
STRAIN = np.array(
    [
        [0.010, 0.002, -0.003],
        [0.002, -0.005, 0.001],
        [-0.003, 0.001, 0.008],
    ]
)


def structures() -> dict:
    from ase.build import bulk
    from ase.io import read

    si8 = bulk("Si", "diamond", a=5.43, cubic=True)
    si8.rattle(0.1, seed=1)
    si8.set_cell(np.array(si8.cell[:]) @ (np.eye(3) + STRAIN), scale_atoms=True)
    return {
        "si8_strained": si8,
        "RSM0010": read(HERE / "relaxed" / "RSM0010.xyz"),
    }


def load_predictor():
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    # `merge_mole=False` keeps the 64 experts around and re-mixes them per
    # forward, so one predictor serves every (task, composition) pair here.
    return load_predict_unit(
        str(CHECKPOINT.resolve()),
        device="cpu",
        inference_settings=InferenceSettings(
            tf32=False,
            activation_checkpointing=False,
            merge_mole=False,
            compile=False,
            base_precision_dtype="float64",
        ),
    )


def report_tasks(predictor) -> None:
    """The stress task's normaliser has to be the energy task's for the port's
    single `energy_scale` to cover both, so print what the checkpoint carries."""
    for name, task in predictor.model.module.tasks.items():
        refs = task.element_references is not None
        print(
            f"  task {name:16s} property={task.property:8s} level={task.level:6s} "
            f"mean={float(task.normalizer.mean)!r} rmsd={float(task.normalizer.rmsd)!r} "
            f"element_references={refs}"
        )


def predict(predictor, atoms, task) -> dict:
    from fairchem.core import FAIRChemCalculator

    calc = FAIRChemCalculator(predictor, task_name=task)
    atoms = atoms.copy()
    atoms.calc = calc
    return {
        "energy": float(atoms.get_potential_energy()),
        "forces": atoms.get_forces().tolist(),
        "stress": atoms.get_stress(voigt=True).tolist(),
    }


def main() -> None:
    predictor = load_predictor()
    print("checkpoint tasks:")
    report_tasks(predictor)

    records = {"charge": 0, "spin": 0, "structures": {}, "tasks": {}}
    for name, atoms in structures().items():
        records["structures"][name] = {
            "numbers": atoms.get_atomic_numbers().tolist(),
            "positions": atoms.get_positions().tolist(),
            "cell": np.array(atoms.cell[:]).tolist(),
            "pbc": np.array(atoms.pbc).tolist(),
        }

    for task in TASKS:
        records["tasks"][task] = {}
        for name, atoms in structures().items():
            record = predict(predictor, atoms, task)
            records["tasks"][task][name] = record
            print(
                f"{task:5s} {name}: {len(atoms)} atoms, E = {record['energy']!r} eV, "
                f"max|F| = {np.abs(record['forces']).max()!r} eV/A, "
                f"max|sigma| = {np.abs(record['stress']).max()!r} eV/A^3"
            )

    path = HERE / "uma_calculator_reference.json"
    path.write_text(json.dumps(records, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
