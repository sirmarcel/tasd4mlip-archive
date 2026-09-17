"""Repo layout: where `sources/` lives, and which checkpoint is which.

Everything here is derived from where `sadmof` was imported from, never written
down: `ROOT` is the repo checkout containing `src/sadmof/`, so no absolute
machine-specific path appears anywhere, and a checkout that moves or is cloned
elsewhere keeps working.

This only holds for a source/editable install, which is how the project is
consumed (`uv sync` installs it editable, and the cluster gets the whole
checkout by rsync). If `sadmof` is ever installed as a plain wheel into
site-packages there is no repo to point at, and `ROOT` raises rather than
handing back a wrong directory. `$SADMOF_ROOT` overrides the derivation.

Callers that need a checkpoint should pass the resolved path on explicitly —
`models.get_calculator` still takes `checkpoint=` and does not consult this
module, so a wrong name fails here, at the lookup, instead of silently
defaulting somewhere downstream.
"""

import os
from pathlib import Path

__all__ = ["ROOT", "SOURCES", "CHECKPOINTS", "checkpoint"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """The checkout containing this package: `<root>/src/sadmof/paths.py`."""
    override = os.environ.get("SADMOF_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        # Same sentinel as the derived path: a stale or mistyped override that
        # happens to name a directory would otherwise produce a plausible but
        # wrong SOURCES, which is exactly what this module exists to prevent.
        if not (root / "pyproject.toml").exists():
            raise RuntimeError(
                f"SADMOF_ROOT={override!r} does not look like a sadmof checkout "
                "(no pyproject.toml)"
            )
        return root

    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").exists():
        raise RuntimeError(
            f"cannot locate the sadmof checkout from {__file__} (looked at {root}). "
            "This module assumes a source/editable install; set SADMOF_ROOT to the "
            "checkout if sadmof is installed some other way."
        )
    return root


ROOT = _repo_root()
SOURCES = ROOT / "sources" / "processed"

# Keyed by directory name, which is the one spelling that cannot drift from
# what is on disk. Experiments keep their own CLI-facing aliases (`mace`,
# `pet-xs`, `mace-mp0+d3`, ...) and map them onto these.
CHECKPOINTS = {
    "mace-mp-0-medium": SOURCES / "mace-mp-0-medium",
    "pet-mad-xs": SOURCES / "pet-mad-xs",
    "pet-mad-s": SOURCES / "pet-mad-s",
    "uma-s-1p2": SOURCES / "uma-s-1p2",
}


def checkpoint(name: str) -> Path:
    """Directory of a processed checkpoint, by name.

    Does not check that the contents are present: the converted weights are
    obtained separately (see the README in each checkpoint directory), so a
    missing file is reported by whatever tries to read it, not a bad name.

    Raises:
        KeyError: If `name` is not a known checkpoint.
    """
    try:
        return CHECKPOINTS[name]
    except KeyError:
        raise KeyError(
            f"unknown checkpoint {name!r}; known: {sorted(CHECKPOINTS)}"
        ) from None
