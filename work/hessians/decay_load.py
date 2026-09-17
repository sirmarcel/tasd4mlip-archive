"""`results/decay/` -> plottable curves. The loader shared by every decay consumer.

`decay_extract.py` stores counts, not curves: a `(shell, distance, log10 norm)`
histogram per condition, over the pairs the exact pattern connects. This turns
those into a median-and-band per distance.

Quantiles are read off the cumulative histogram and carry the log-bin resolution
(0.05 dex). The exact scalars (`norm_max`, `dist_max_nonzero_A`,
`reach_radii_A`) come out of the `.json` untouched.
"""

import numpy as np

from pathlib import Path

from sadmof.io import read_json

RESULTS = Path(__file__).resolve().parent / "results" / "decay"


def available():
    """`(structure, model)` pairs present in `results/decay/`, sorted."""
    out = []
    for path in sorted(RESULTS.glob("*.npz")):
        structure, _, model = path.stem.partition("_")
        out.append((structure, model))
    return out


def load(structure, model):
    """One condition: the histogram arrays plus its `.json` metadata."""
    stem = RESULTS / f"{structure}_{model}"
    with np.load(str(stem) + ".npz") as z:
        data = {key: z[key] for key in z.files}
    data["meta"] = read_json(stem.with_suffix(".json"))
    return data


def aggregate(structures, model):
    """Sum conditions of one model into a single histogram.

    Every condition bins distance from 0 in the same width, so alignment is by
    index and the shorter histograms are zero-padded up to the longest. The
    shell axis is summed out.

    Returns:
        `(hist, hist_zero, dist_edges, lognorm_edges, metas)` — `hist` is
        `(n_dist, n_log)` counts of nonzero blocks, `hist_zero` is `(n_dist,)`
        counts of structurally zero ones.
    """
    loaded = [load(structure, model) for structure in structures]
    lognorm_edges = loaded[0]["lognorm_edges"]
    dist_edges = max((item["dist_edges"] for item in loaded), key=len)
    # Both grids are pinned by `decay_extract` constants, so a mismatch means
    # results from two different versions of it: index alignment would then be
    # silent nonsense rather than an error.
    for item in loaded:
        if not np.array_equal(item["lognorm_edges"], lognorm_edges):
            raise ValueError("conditions were binned on different log-norm grids")
        prefix = dist_edges[: len(item["dist_edges"])]
        if not np.allclose(item["dist_edges"], prefix):
            raise ValueError("conditions were binned on different distance grids")

    n_dist = max(len(item["dist_edges"]) - 1 for item in loaded)
    hist = np.zeros((n_dist, len(lognorm_edges) - 1), dtype=np.int64)
    hist_zero = np.zeros(n_dist, dtype=np.int64)
    for item in loaded:
        h = item["hist"].sum(axis=0)
        hist[: h.shape[0]] += h
        z = item["hist_zero"].sum(axis=0)
        hist_zero[: z.shape[0]] += z

    return hist, hist_zero, dist_edges, lognorm_edges, [i["meta"] for i in loaded]


def aggregate_shells(structures, model):
    """Sum conditions of one model into a per-shell histogram.

    The distance axis is summed out; shells run `h = 0..K` and agree across
    structures because `K` is the model's.

    Returns:
        `(hist, lognorm_edges, metas)` — `hist` is `(K + 1, n_log)` counts of
        nonzero blocks.
    """
    loaded = [load(structure, model) for structure in structures]
    lognorm_edges = loaded[0]["lognorm_edges"]
    for item in loaded:
        if not np.array_equal(item["lognorm_edges"], lognorm_edges):
            raise ValueError("conditions were binned on different log-norm grids")
    hist = np.sum([item["hist"].sum(axis=1) for item in loaded], axis=0)
    return hist, lognorm_edges, [i["meta"] for i in loaded]


def curves(hist, dist_edges, lognorm_edges, quantiles=(0.1, 0.5, 0.9), min_count=32):
    """Per-distance-bin quantiles and maximum of `||Phi_ij||`, over nonzero blocks.

    Bins holding fewer than `min_count` nonzero blocks are masked: the tail of
    the distance axis is sparsely populated.

    Every returned norm is the center of its log bin, maximum included, so the
    curves share one convention and sit half a bin apart at most.

    Returns:
        `dict` with `distance` (bin centers), `quantiles` `(len(quantiles),
        n_dist)`, `maximum` `(n_dist,)`, and `count` `(n_dist,)`. All norm
        arrays are masked where `count < min_count`.
    """
    counts = hist.sum(axis=1)
    enough = counts >= min_count

    centers = bin_centers(lognorm_edges)
    cumulative = np.cumsum(hist, axis=1)
    totals = cumulative[:, -1:]

    out = np.empty((len(quantiles), hist.shape[0]))
    for k, q in enumerate(quantiles):
        # First bin whose cumulative count reaches the quantile; empty rows land
        # on the last index and are masked out below.
        index = (cumulative < q * totals).sum(axis=1)
        out[k] = 10.0 ** centers[np.minimum(index, len(centers) - 1)]

    occupied = np.where(hist > 0, np.arange(hist.shape[1]), -1).max(axis=1)
    maximum = 10.0 ** centers[np.maximum(occupied, 0)]

    mask = ~enough
    return {
        "distance": bin_centers(dist_edges),
        "quantiles": np.ma.masked_array(out, np.broadcast_to(mask, out.shape)),
        "maximum": np.ma.masked_array(maximum, mask),
        "count": counts,
    }


def bin_centers(edges):
    return 0.5 * (edges[1:] + edges[:-1])
