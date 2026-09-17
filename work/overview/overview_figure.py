"""results/ -> figures/overview.pdf|png, the preprint's overview figure.

uv run python overview_figure.py [--centre ATOM]
"""

import numpy as np

import argparse
import subprocess
from pathlib import Path

from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list
from matplotlib import colormaps, colors
from matplotlib.image import imsave
from overview_load import load
from scipy import sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee

from sadmof.io import write_json

EXP = Path(__file__).resolve().parent
BUILD = EXP / "output" / "build"
FIGURES = EXP / "figures"
TYP = EXP / "overview_figure.typ"

# Magnitude scale of the block norms, eV/Å², shared by (c) and (d).
LOG_MIN, LOG_MAX = -6.0, 2.0
# Hop shells, index = hop count; 0 is drawn as the marked atom, not a colour.
HOP_COLORS = ["#000000", "#E9A17C", "#E07A67", "#CC5A6A", "#A64B75", "#6E3F73"]
UPPER_GROUND = "#F7F1EE"  # ground of the hop-count triangle in (d)
BOND_FACTOR = 1.2  # times the sum of covalent radii
VIEW_MARGIN = 1.5  # Å around the marked atom's K-hop cluster


def main():
    args = build_parser().parse_args()
    arrays, summary, toy = load()
    BUILD.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    K = summary["K"]

    write_json(BUILD / "structure.json", local_view(arrays, K, args.centre))
    write_json(BUILD / "toy.json", toy)
    write_json(
        BUILD / "style.json", {"hopcolors": HOP_COLORS, "upper_ground": UPPER_GROUND}
    )
    imsave(BUILD / "matrix.png", matrix_image(arrays, K))

    for fmt, extra in (("pdf", []), ("png", ["--ppi", "250"])):
        target = FIGURES / f"overview.{fmt}"
        subprocess.run(
            ["typst", "compile", "--format", fmt, *extra, str(TYP), str(target)],
            check=True,
            cwd=EXP,
        )
        print(f"wrote {target}")


# ---------------------------------------------------------------------------
# (b), (c): the structure around one atom
# ---------------------------------------------------------------------------


def local_view(arrays, K, centre):
    """Atoms projected onto the principal plane of the marked atom's K-hop cluster."""
    pos, cell, Z = arrays["positions"], arrays["cell"], arrays["numbers"]
    hop, norms, edges = arrays["hop"], arrays["norms"], arrays["edges"]
    n = len(Z)
    atoms = Atoms(numbers=Z, positions=pos, cell=cell, pbc=True)
    if centre is None:
        # A node atom near the cell centre, so the shells walk node -> linker -> node.
        zn = np.where(Z == 30)[0]
        centre = int(zn[np.argmin(np.linalg.norm(pos[zn] - cell.sum(0) / 2, axis=1))])

    # Unwrapped around the centre, so the neighbourhood is contiguous.
    vec = atoms.get_distances(centre, np.arange(n), mic=True, vector=True)
    shell = hop[centre].astype(int)
    cluster = shell >= 0
    _, _, axes = np.linalg.svd(vec[cluster] - vec[cluster].mean(0), full_matrices=False)
    xy = vec @ axes[:2].T
    depth = vec @ axes[2]

    cmap = colormaps["viridis"]

    def hexcol(v):
        t = (np.log10(max(v, 10**LOG_MIN)) - LOG_MIN) / (LOG_MAX - LOG_MIN)
        return colors.to_hex(cmap(t))

    magnitude = norms[centre]
    atom_rows = [
        {
            "i": int(i),
            "x": float(xy[i, 0]),
            "y": float(xy[i, 1]),
            "r": float(covalent_radii[Z[i]]),
            "hop": int(shell[i]),
            "magcol": hexcol(magnitude[i]) if magnitude[i] > 0 else None,
        }
        for i in np.argsort(depth)  # painter's order, back to front
    ]

    cutoffs = {
        (a, b): BOND_FACTOR * (covalent_radii[a] + covalent_radii[b])
        for a in set(Z)
        for b in set(Z)
    }
    bi, bj, bD = neighbor_list("ijD", atoms, cutoffs)
    bonds = [
        {"i": int(i), "j": int(j), "incluster": bool(cluster[i] and cluster[j])}
        for i, j, D in zip(bi, bj, bD)
        # One direction per bond, and only the image the unwrapped picture shows.
        if i < j and np.linalg.norm((vec[j] - vec[i]) - D) < 1e-6
    ]

    onehop = [int(j) for j in edges[1][edges[0] == centre]]
    return {
        "centre": centre,
        "K": K,
        "atoms": atom_rows,
        "bonds": bonds,
        "onehop": onehop,
        # The farthest neighbour PET's adaptive selection keeps for this atom.
        "cutoff_r": float(np.linalg.norm(vec[onehop], axis=1).max()),
        "view": float(np.abs(xy[cluster]).max() + VIEW_MARGIN),
        "logmin": LOG_MIN,
        "logmax": LOG_MAX,
    }


# ---------------------------------------------------------------------------
# (d): the whole Hessian at block resolution
# ---------------------------------------------------------------------------


def matrix_image(arrays, K):
    """Hop count above the diagonal, block norms below, structural zeros transparent."""
    hop, norms, edges = arrays["hop"], arrays["norms"], arrays["edges"]
    n = len(norms)
    A = sp.csr_matrix((np.ones(edges.shape[1], bool), (edges[0], edges[1])), shape=(n, n))
    # Bandwidth-reducing order, so the nested patterns read as nested bands.
    perm = reverse_cuthill_mckee(A, symmetric_mode=True)
    M = norms[perm][:, perm].astype(np.float64)
    hp = hop[perm][:, perm]
    cmap = colormaps["viridis"]

    img = np.zeros((n, n, 4))
    nz = M > 0
    img[nz] = cmap(
        (np.log10(np.maximum(M[nz], 10**LOG_MIN)) - LOG_MIN) / (LOG_MAX - LOG_MIN)
    )
    upper = np.triu(np.ones((n, n), bool), 1)
    for k in range(K + 1):
        img[upper & (hp == k)] = colors.to_rgba(HOP_COLORS[k])
    img[upper & (hp < 0)] = (0, 0, 0, 0)
    return img


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--centre", type=int, default=None, help="atom to mark (default: a central Zn)"
    )
    return p


if __name__ == "__main__":
    main()
