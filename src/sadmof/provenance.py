"""Run provenance: which code, which dependencies, and which machine produced
an output.

Stdlib plus git; every part degrades to ``None``/skipped when unavailable, so
this is always safe to call.
"""

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

__all__ = [
    "provenance",
    "git_info",
    "package_versions",
    "dep_repos",
    "host_info",
    "device_info",
]

# Distributions whose versions pin a sadmof result: sadmof itself plus the
# compute stack (jax + the model / AD / I/O deps). Names not installed are
# skipped, so this list can stay broad.
DEFAULT_PACKAGES = (
    "sadmof",
    "jax",
    "jaxlib",
    "ase",
    "e3nn-jax",
    "flax",
    "asdex",
    "pet-jax",
    "marathon-train",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def provenance(packages=DEFAULT_PACKAGES, *, repo=None) -> dict:
    """Collect run provenance: git state, package versions, host and devices.

    `devices` is filled in only if the caller already imported jax; importing it
    here could preallocate a GPU in a process that never wanted one.

    Args:
        packages: distribution names to record versions for; any not installed
            are skipped. Defaults to sadmof + its compute stack.
        repo: a path inside the git repo to describe. Defaults to sadmof's own
            source tree (so an editable install reports the checkout it runs
            from).

    Returns:
        ``{"git": {"sha", "dirty"} | None, "versions": {name: version},
        "deps": {name: {"sha", "dirty"}}, "host": {...},
        "devices": {...} | None}``.
    """
    return {
        "git": git_info(repo),
        "versions": package_versions(packages),
        "deps": dep_repos(packages),
        "host": host_info(),
        "devices": device_info(),
    }


def host_info() -> dict:
    """Hostname, CPU architecture, and SLURM placement.

    The SLURM keys, absent outside a job, are what name the cluster — hostnames
    are opaque node ids.
    """
    info = {"node": platform.node(), "machine": platform.machine()}
    slurm = {
        key.lower().removeprefix("slurm_"): os.environ[key]
        for key in ("SLURM_CLUSTER_NAME", "SLURM_JOB_ID", "SLURM_JOB_PARTITION")
        if key in os.environ
    }
    if slurm:
        info["slurm"] = slurm
    return info


def device_info() -> dict | None:
    """The accelerators jax is using, or ``None`` if jax is not imported."""
    jax = sys.modules.get("jax")
    if jax is None:
        return None
    try:
        devices = jax.local_devices()
        return {
            "platform": devices[0].platform,
            "kind": devices[0].device_kind,
            "count": len(devices),
            # Observed at call time, not requested; unset is jax's DEFAULT.
            "matmul_precision": jax.config.jax_default_matmul_precision or "default",
        }
    except Exception:  # noqa: BLE001 — provenance must never break a run
        return None


def git_info(repo=None) -> dict | None:
    """Describe the git state of ``repo`` (default: sadmof's source tree).

    Returns ``{"sha": <full HEAD sha>, "dirty": <bool>}``, or ``None`` if the
    path is not inside a git repo or git is unavailable (e.g. a wheel install).
    """
    root = Path(repo) if repo is not None else Path(__file__).resolve().parent
    sha = _git(root, "rev-parse", "HEAD")
    if sha is None:
        return None
    # -uno: "dirty" means tracked files differ from HEAD. Untracked files
    # (the cluster's env.sh, output dirs, …) don't change the code that ran.
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return {"sha": sha, "dirty": bool(status)}


def package_versions(names) -> dict:
    """Map distribution name -> installed version, skipping any not installed."""
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def dep_repos(names) -> dict:
    """Map distribution name -> git state, for those installed from a checkout.

    An editable dependency is pinned by its commit, not its version: the clone
    is consumed wherever it sits, and its version string moves only on an
    upstream release. Non-editable, non-git, and uninstalled names are skipped.
    """
    repos = {}
    for name in names:
        # sadmof's own checkout is already the top-level `git` entry.
        if name == "sadmof":
            continue
        path = _editable_path(name)
        if path is None:
            continue
        # A checkout copied without its .git (rsync to a cluster) makes git walk
        # up to the enclosing repo and report that one's sha as the dep's.
        toplevel = _git(path, "rev-parse", "--show-toplevel")
        if toplevel is None or Path(toplevel).resolve() != path.resolve():
            continue
        info = git_info(path)
        if info is not None:
            repos[name] = info
    return repos


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _editable_path(name) -> Path | None:
    """Source directory of an editable install, via PEP 610 `direct_url.json`.

    `None` for anything else — a wheel from an index has no source tree to
    describe.
    """
    try:
        dist = importlib.metadata.distribution(name)
        raw = dist.read_text("direct_url.json")
    except (importlib.metadata.PackageNotFoundError, OSError):
        return None
    if raw is None:
        return None
    try:
        direct_url = json.loads(raw)
    except ValueError:
        return None
    if not direct_url.get("dir_info", {}).get("editable"):
        return None
    url = direct_url.get("url", "")
    if not url.startswith("file://"):
        return None
    return Path(unquote(urlparse(url).path))


def _git(root, *args) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
