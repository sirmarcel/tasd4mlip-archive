"""results/ -> arrays, for every consumer of the overview data."""

import numpy as np

from pathlib import Path

from sadmof.io import read_json

RESULTS = Path(__file__).resolve().parent / "results"
STRUCTURE = "mof177"


def load(structure=STRUCTURE):
    arrays = dict(np.load(RESULTS / f"{structure}.npz"))
    summary = read_json(RESULTS / f"{structure}.json")
    toy = read_json(RESULTS / "toy.json")
    return arrays, summary, toy
