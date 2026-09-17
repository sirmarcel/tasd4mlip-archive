"""Atomic writes for experiment artifacts.

Experiments resume by output existence, so a process killed mid-write must not
leave a truncated file that the next run reads as complete. Every writer here
goes to a sibling temporary and renames onto the target, which is atomic within
a filesystem.
"""

import numpy as np

import json
from contextlib import contextmanager
from pathlib import Path

__all__ = [
    "read_json",
    "write_json",
    "write_npy",
    "write_npz",
    "write_xyz",
    "atomic_path",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(obj, indent=1))


def write_npy(path, array):
    with atomic_path(path, ".tmp.npy") as tmp:
        np.save(tmp, array)


def write_npz(path, arrays, compress=False):
    """Uncompressed by default: raw float payloads do not compress and the
    deflate pass dominates the write. Pass `compress=True` for count arrays."""
    save = np.savez_compressed if compress else np.savez
    with atomic_path(path, ".tmp.npz") as tmp:
        save(tmp, **arrays)


def write_xyz(path, atoms):
    with atomic_path(path, ".tmp.xyz") as tmp:
        atoms.write(str(tmp))


@contextmanager
def atomic_path(path, suffix=".tmp"):
    """Yield a temporary sibling of `path`, renamed onto it on clean exit.

    For writers that insist on owning the file themselves (`asdex`'s `save`).
    The suffix has to keep the extension numpy and friends dispatch on, hence
    the `.tmp.npz` / `.tmp.xyz` forms rather than a bare `.tmp`.
    """
    path = Path(path)
    tmp = path.with_name(path.name + suffix)
    yield tmp
    tmp.replace(path)
