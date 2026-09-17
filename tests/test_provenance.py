import sys

from sadmof.provenance import (
    dep_repos,
    device_info,
    git_info,
    host_info,
    package_versions,
    provenance,
)


def test_package_versions_reports_known_skips_unknown():
    versions = package_versions(["sadmof", "this-package-does-not-exist-xyz"])
    assert "sadmof" in versions
    assert isinstance(versions["sadmof"], str)
    assert "this-package-does-not-exist-xyz" not in versions


def test_git_info_shape():
    # Run from sadmof's own tree, which is a git checkout in dev/editable
    # installs; None is the valid answer for a wheel install with no repo.
    info = git_info()
    if info is not None:
        assert set(info) == {"sha", "dirty"}
        assert isinstance(info["sha"], str) and info["sha"]
        assert isinstance(info["dirty"], bool)


def test_provenance_structure():
    prov = provenance()
    assert set(prov) == {"git", "versions", "deps", "host", "devices"}
    assert isinstance(prov["versions"], dict)
    assert prov["git"] is None or set(prov["git"]) == {"sha", "dirty"}


def test_dep_repos_shape_and_skips():
    # Only editable installs that are git checkouts appear; a name that is not
    # installed at all must not raise.
    repos = dep_repos(["asdex", "jax", "this-package-does-not-exist-xyz"])
    assert "this-package-does-not-exist-xyz" not in repos
    for info in repos.values():
        assert set(info) == {"sha", "dirty"}
        assert isinstance(info["sha"], str) and info["sha"]
        assert isinstance(info["dirty"], bool)


def test_dep_repos_excludes_sadmof_itself():
    # sadmof's own checkout is the top-level `git` entry.
    assert "sadmof" not in dep_repos(["sadmof"])


def test_host_info_identifies_machine():
    info = host_info()
    assert isinstance(info["node"], str)
    assert isinstance(info["machine"], str) and info["machine"]
    # SLURM keys only exist inside a job; when present they must be strings.
    assert set(info) <= {"node", "machine", "slurm"}
    assert all(isinstance(v, str) for v in info.get("slurm", {}).values())


def test_device_info_does_not_initialise_jax(monkeypatch):
    # The contract that keeps provenance cheap: no jax in sys.modules -> no
    # backend init, just None. Guards against someone importing jax at module
    # scope here later.
    monkeypatch.delitem(sys.modules, "jax", raising=False)
    assert device_info() is None


def test_device_info_shape_when_jax_present():
    import jax  # noqa: F401 — importing is the point: it populates sys.modules

    info = device_info()
    assert info is None or set(info) == {"platform", "kind", "count", "matmul_precision"}
