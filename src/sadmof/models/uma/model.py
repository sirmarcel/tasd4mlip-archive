"""Pure-JAX port of UMA-S (fairchem's eSCN-MD MoE backbone + MLP energy head).

`UMA` is a plain frozen dataclass, not a flax module: the parameter tree comes
from `sources/processed/uma-s-1p2/build.py` and is mixed down from 64 MOLE
experts to one plain set of weights per structure (see `load._merge_params`), so
there is nothing to `init` and no submodule tree to name. `__call__` returns
per-atom energies in eV; `energy.py` sums them.

Architecture, all of it at `lmax = mmax = 2`:

- Every edge carries a Wigner-D rotation built from Euler angles of its
  direction vector (`beta` from the y-component, `alpha` from x/z). The third
  angle is a random roll per call in fairchem; it is a rotation about the edge
  axis, and at `mmax == lmax` the energy is invariant to it (the SO(2) blocks
  are complex-linear and commute with the z-phase it induces), so this port
  pins it to zero.
- Node features live in two orderings. **L-order** is e3nn's
  `(l=0; l=1,m=-1,0,1; l=2,m=-2..2)`; **m'-order** groups by `|m|` as
  `(m=0: l0,l1,l2 | m=1: Re l1,l2, Im l1,l2 | m=2: Re l2, Im l2)`. The Wigner
  matrices are pre-composed with that permutation, so the edgewise path works in
  m'-order and the atomwise path in L-order.
- The SO(2) convolutions are complex-linear maps on each `|m| > 0` block: with
  the stored weight split as `[W1; W2]`, `out = (W1 + i W2)(x_r + i x_i)`.
- `balance_channels` shifts the `l = 0` channels 0..2 so their per-system sum
  equals the total charge (`params["charge"]`). It couples every atom to every
  other atom, which is the one non-local term in the model — loading with
  `balance_channels=False` drops it.
"""

import numpy as np
import jax
import jax.numpy as jnp

import functools
from dataclasses import dataclass, field

from jaxtyping import Array, Float

__all__ = ["UMA"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class UMA:
    """UMA-S backbone + energy head, as a callable over a merged parameter tree.

    Defaults are the `uma-s-1p2` checkpoint's, except `jd`, which is required:
    it holds the e3nn `J` matrices (one per degree) that the Euler-angle Wigner
    construction needs. They come from the checkpoint and are the only
    array-valued field, small enough to ride along as a jit constant.

    Args:
        max_neighbors: Stock UMA's per-atom neighbour cap. This port consumes an
            external graph and never truncates, so the value is only what
            `inputs.atoms_to_inputs` warns against.
        balance_channels: Apply the charge-balancing shift after every layer,
            towards the total charge carried by `params["charge"]`. Off, the
            model is strictly semi-local in positions.
    """

    cutoff: float = 6.0
    max_neighbors: int = 300
    lmax: int = 2
    mmax: int = 2
    num_layers: int = 4
    sphere_channels: int = 128
    hidden_channels: int = 128
    num_distance_basis: int = 32
    envelope_exponent: int = 5
    edge_degree_rescale_factor: float = 5.0
    gaussian_coeff: float = -3.3368057544446743
    norm_eps: float = 1e-5
    charge_channel_start: int = 0
    charge_channel_end: int = 3
    balance_channels: bool = True
    energy_scale: float = 1.0
    jd: tuple = field(default=(), repr=False)

    def __post_init__(self):
        if len(self.jd) != self.lmax + 1:
            raise ValueError(
                f"jd must hold one e3nn J matrix per degree ({self.lmax + 1}); "
                "load it from the checkpoint with `load_uma`"
            )

    def __call__(
        self,
        params: dict,
        pos: Float[Array, "atoms 3"],
        cell: Float[Array, "3 3"],
        graph: dict[str, Array],
    ) -> Float[Array, " atoms"]:
        """Per-atom energies in eV, scaled by the task normaliser's `rmsd`.

        The normaliser's `mean` and the per-element references are per-system
        and per-atom constants; both are left to the calculator (through
        `load_uma`'s metadata) since neither reaches a derivative.

        `graph` carries `centers` (receiver i), `others` (sender j),
        `cell_shifts`, `atomic_numbers`, `atom_mask` and `pair_mask`; the edge
        vector is `pos[j] - pos[i] + shift @ cell`, fairchem's own convention
        once its `edge_index` is read as `(others, centers)`.
        """
        node = self.node_embeddings(params, pos, cell, graph)
        energies = _mlp(params["head"], node[:, 0, :], (0, 2, 4))[:, 0]
        return jnp.where(graph["atom_mask"], energies * self.energy_scale, 0.0)

    def node_embeddings(
        self,
        params: dict,
        pos: Float[Array, "atoms 3"],
        cell: Float[Array, "3 3"],
        graph: dict[str, Array],
    ) -> Float[Array, "atoms coeffs channels"]:
        """The backbone's `(N, (lmax+1)^2, C)` node features, in L-order."""
        z = graph["atomic_numbers"]
        centers, others = graph["centers"], graph["others"]
        n_atoms = z.shape[0]

        edge_vec = pos[others] - pos[centers] + graph["cell_shifts"] @ cell
        distance, direction = _norm_and_direction(edge_vec)

        wigner, wigner_inv = self.edge_wigner(direction)
        envelope = _polynomial_envelope(distance / self.cutoff, self.envelope_exponent)
        envelope = jnp.where(graph["pair_mask"], envelope, 0.0)
        wigner_inv_env = wigner_inv * envelope[:, None, None]

        x_edge = jnp.concatenate(
            [
                _gaussian_smearing(
                    distance, params["gaussian_offset"], self.gaussian_coeff
                ),
                params["source_embedding"][z[others]],
                params["target_embedding"][z[centers]],
            ],
            axis=1,
        )

        # Node init: element embedding plus the charge/spin/dataset vector.
        x = jnp.zeros(
            (n_atoms, self.sph_feature_size, self.sphere_channels), dtype=pos.dtype
        )
        x = x.at[:, 0, :].set(params["sphere_embedding"][z] + params["csd"])

        radial = _radial_mlp(params["edge_degree"]["rad"], x_edge)
        radial = radial.reshape(-1, self.m_size[0], self.sphere_channels)
        contribution = wigner_inv_env[:, :, : self.m_size[0]] @ radial
        x = x.at[centers].add(contribution / self.edge_degree_rescale_factor)

        for layer in range(self.num_layers):
            x = self._block(
                params["blocks"][layer],
                params["csd"],
                x,
                x_edge,
                wigner,
                wigner_inv_env,
                graph,
            )
            if self.balance_channels:
                x = self._balance(x, graph["atom_mask"], params["charge"])

        return _equivariant_rms_norm(params["norm"], x, self.lmax, self.norm_eps)

    def edge_wigner(
        self, direction: Float[Array, "edges 3"]
    ) -> tuple[Float[Array, "edges coeffs coeffs"], Float[Array, "edges coeffs coeffs"]]:
        """`(wigner, wigner_inv)`, already permuted into m'-order on the side
        that the edgewise convolution consumes."""
        beta = _safe_acos(direction[:, 1])
        alpha = _safe_atan2(direction[:, 0], direction[:, 2])
        # fairchem's intrinsic-to-extrinsic swap, with its random roll set to 0.
        wigner = _eulers_to_wigner(jnp.zeros_like(alpha), -beta, -alpha, self.jd)
        wigner_inv = jnp.swapaxes(wigner, 1, 2)
        perm = self.to_m_perm
        return wigner[:, perm, :], wigner_inv[:, :, perm]

    # -- derived index constants; cheap enough to rebuild on access --

    @property
    def sph_feature_size(self) -> int:
        return (self.lmax + 1) ** 2

    @property
    def to_m_perm(self) -> np.ndarray:
        return _to_m_permutation(self.lmax, self.mmax)

    @property
    def m_size(self) -> tuple[int, ...]:
        return _m_sizes(self.lmax, self.mmax)

    # -- internals --

    def _block(self, params, csd, x, x_edge, wigner, wigner_inv_env, graph):
        residual = x
        x = _equivariant_rms_norm(params["norm_1"], x, self.lmax, self.norm_eps)
        x = x.at[:, 0, :].add(csd)
        x = self._edgewise(params, x, x_edge, wigner, wigner_inv_env, graph) + residual

        residual = x
        x = _equivariant_rms_norm(params["norm_2"], x, self.lmax, self.norm_eps)
        return self._atomwise(params["atom_wise"], x) + residual

    def _edgewise(self, params, x, x_edge, wigner, wigner_inv_env, graph):
        message = jnp.concatenate([x[graph["others"]], x[graph["centers"]]], axis=2)
        message = wigner @ message

        message, gating = _so2_convolution(
            params["so2_conv_1"],
            message,
            self.lmax,
            self.mmax,
            self.hidden_channels,
            x_edge=x_edge,
            extra_m0=self.lmax * self.hidden_channels,
        )
        message = _gate_activation(
            gating, message, self.lmax, self.mmax, self.hidden_channels, m_prime=True
        )
        message, _ = _so2_convolution(
            params["so2_conv_2"], message, self.lmax, self.mmax, self.sphere_channels
        )

        rotated = wigner_inv_env @ message
        out = jnp.zeros_like(x)
        return out.at[graph["centers"]].add(rotated)

    def _atomwise(self, params, x):
        gating = jax.nn.silu(_linear(params["scalar_mlp"], x[:, 0:1, :]))
        x = _so3_linear(params["so3_linear_1"], x, self.lmax)
        x = _gate_activation(
            gating, x, self.lmax, self.lmax, self.hidden_channels, m_prime=False
        )
        return _so3_linear(params["so3_linear_2"], x, self.lmax)

    def _balance(self, x, atom_mask, charge):
        start, end = self.charge_channel_start, self.charge_channel_end
        channels = x[:, 0, start:end]
        totals = jnp.sum(jnp.where(atom_mask[:, None], channels, 0.0), axis=0)
        n_real = jnp.sum(atom_mask.astype(x.dtype))
        return x.at[:, 0, start:end].add(-(totals - charge) / n_real)


# ---------------------------------------------------------------------------
# Rotations: Euler angles with clamped-denominator derivatives
# ---------------------------------------------------------------------------

# fairchem clamps the backward denominators of acos/atan2 (its `Safeacos` /
# `Safeatan2`) so an edge along the pole, or a zero-length one, cannot blow up a
# force. That is not a cosmetic guard: any cell whose lattice vector is shorter
# than the cutoff has self-image edges exactly on the pole, where the Euler
# parametrisation is singular even though the energy is not.
#
# Each rule returns the *wrapped* function as its primal, so differentiating the
# rule again re-enters the clamped version rather than the bare `jnp` primitive —
# without that, second-order AD walks straight back into the singularity and
# every Hessian comes out NaN. The clamps saturate there, so the second
# derivative of the denominator is zero, which is also what torch's double
# backward through `Safeacos` gives.
_ANGLE_EPS = 1e-7


@jax.custom_jvp
def _safe_acos(x: Array) -> Array:
    return jnp.arccos(x)


@_safe_acos.defjvp
def _safe_acos_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    clamped = jnp.clip(x, -1.0 + _ANGLE_EPS, 1.0 - _ANGLE_EPS)
    denom = jnp.maximum(jnp.sqrt(1.0 - clamped**2), _ANGLE_EPS)
    return _safe_acos(x), -dx / denom


@jax.custom_jvp
def _safe_atan2(y: Array, x: Array) -> Array:
    return jnp.arctan2(y, x)


@_safe_atan2.defjvp
def _safe_atan2_jvp(primals, tangents):
    (y, x), (dy, dx) = primals, tangents
    denom = jnp.maximum(x**2 + y**2, _ANGLE_EPS)
    return _safe_atan2(y, x), (x * dy - y * dx) / denom


def _norm_and_direction(vec: Array) -> tuple[Array, Array]:
    """`(|vec|, vec/|vec|)`, both zero (and with zero derivative) at `vec = 0`.

    Padding pairs are self-loops with no shift, so their vector *is* zero; the
    double-`where` keeps the NaN that `d/dr sqrt(r.r)` would produce out of the
    tangent as well as out of the value.
    """
    squared = jnp.sum(vec**2, axis=-1)
    nonzero = squared > 0.0
    distance = jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, squared, 1.0)), 0.0)
    direction = vec / jnp.where(nonzero, distance, 1.0)[:, None]
    return distance, jnp.clip(direction, -1.0, 1.0)


def _z_rot_mat(angle: Array, degree: int) -> Array:
    """Rotation about the e3nn m=0 axis, in the real spherical-harmonic basis."""
    size = 2 * degree + 1
    inds = np.arange(size)
    freqs = np.arange(degree, -degree - 1, -1)
    matrix = jnp.zeros((angle.shape[0], size, size), dtype=angle.dtype)
    phases = freqs[None, :] * angle[:, None]
    matrix = matrix.at[:, inds, size - 1 - inds].set(jnp.sin(phases))
    return matrix.at[:, inds, inds].set(jnp.cos(phases))


def _wigner_d(degree: int, alpha: Array, beta: Array, gamma: Array, jd: tuple) -> Array:
    j = jnp.asarray(jd[degree], dtype=alpha.dtype)
    return (
        _z_rot_mat(alpha, degree)
        @ j
        @ _z_rot_mat(beta, degree)
        @ j
        @ _z_rot_mat(gamma, degree)
    )


def _eulers_to_wigner(alpha: Array, beta: Array, gamma: Array, jd: tuple) -> Array:
    """Block-diagonal Wigner-D over degrees `0..len(jd)-1`, in L-order."""
    blocks = [_wigner_d(l, alpha, beta, gamma, jd) for l in range(len(jd))]
    size = sum(block.shape[1] for block in blocks)
    wigner = jnp.zeros((alpha.shape[0], size, size), dtype=alpha.dtype)
    start = 0
    for block in blocks:
        end = start + block.shape[1]
        wigner = wigner.at[:, start:end, start:end].set(block)
        start = end
    return wigner


# ---------------------------------------------------------------------------
# Radial basis, envelope, plain MLPs
# ---------------------------------------------------------------------------


def _gaussian_smearing(distance: Array, offset: Array, coeff: float) -> Array:
    return jnp.exp(coeff * (distance[:, None] - offset[None, :]) ** 2)


def _polynomial_envelope(scaled: Array, exponent: int) -> Array:
    """Exactly zero at and beyond the cutoff, with `exponent` vanishing derivatives."""
    p = float(exponent)
    a = -(p + 1) * (p + 2) / 2
    b = p * (p + 2)
    c = -p * (p + 1) / 2
    value = 1 + (scaled**p) * (a + scaled * (b + c * scaled))
    return jnp.where(scaled < 1, value, 0.0)


def _linear(params: dict, x: Array) -> Array:
    out = x @ params["w"].T
    return out + params["b"] if "b" in params else out


def _layer_norm(params: dict, x: Array, eps: float = 1e-5) -> Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + eps) * params["w"] + params["b"]


def _mlp(params: dict, x: Array, layers: tuple[int, ...]) -> Array:
    """Linear/SiLU stack; `layers` are the torch `nn.Sequential` indices."""
    for idx in layers[:-1]:
        x = jax.nn.silu(_linear(params[f"lin{idx}"], x))
    return _linear(params[f"lin{layers[-1]}"], x)


def _radial_mlp(params: dict, x: Array) -> Array:
    """fairchem's `RadialMLP`: Linear, LayerNorm, SiLU, twice, then Linear."""
    for lin, ln in ((0, 1), (3, 4)):
        x = jax.nn.silu(_layer_norm(params[f"ln{ln}"], _linear(params[f"lin{lin}"], x)))
    return _linear(params["lin6"], x)


# ---------------------------------------------------------------------------
# Equivariant layers
# ---------------------------------------------------------------------------


def _equivariant_rms_norm(params: dict, x: Array, lmax: int, eps: float) -> Array:
    """`EquivariantRMSNormArraySphericalHarmonicsV2`: centre `l = 0` across
    channels, then scale by one RMS taken over degree-balanced components."""
    centred = x.at[:, 0:1, :].add(-jnp.mean(x[:, 0:1, :], axis=2, keepdims=True))
    weight = jnp.asarray(_balance_degree_weight(lmax), dtype=x.dtype)
    norm = jnp.sum(centred**2 * weight[None, :, None], axis=1, keepdims=True)
    norm = jnp.mean(norm, axis=2, keepdims=True)
    norm = (norm + eps) ** -0.5
    scale = params["w"][_l_expand_index(lmax)][None]
    out = centred * (norm * scale)
    return out.at[:, 0:1, :].add(params["b"])


def _so3_linear(params: dict, x: Array, lmax: int) -> Array:
    """Per-degree channel mixing; the bias lands on `l = 0` only."""
    weight = params["w"][_l_expand_index(lmax)]  # (coeffs, out, in)
    out = jnp.einsum("bmi,moi->bmo", x, weight)
    return out.at[:, 0:1, :].add(params["b"])


def _gate_activation(
    gating: Array, x: Array, lmax: int, mmax: int, num_channels: int, *, m_prime: bool
) -> Array:
    """SiLU on the scalar, sigmoid gates broadcast over each degree's components."""
    gates = jax.nn.sigmoid(gating).reshape(gating.shape[0], lmax, num_channels)
    gates = gates[:, _gate_expand_index(lmax, mmax, m_prime), :]
    return jnp.concatenate([jax.nn.silu(x[:, 0:1, :]), x[:, 1:, :] * gates], axis=1)


def _so2_convolution(
    params: dict,
    x: Array,
    lmax: int,
    mmax: int,
    out_channels: int,
    *,
    x_edge: Array | None = None,
    extra_m0: int = 0,
) -> tuple[Array, Array]:
    """SO(2) convolution over m'-ordered features.

    With a radial MLP present (`so2_conv_1`), `x_edge` is expanded into
    per-`|m|` multiplicative weights first, and the leading `extra_m0` outputs
    of the `m = 0` linear are split off as the gate activation's scalars — the
    second return value, empty when `extra_m0` is zero.
    """
    n_edges = x.shape[0]
    sizes = _m_sizes(lmax, mmax)
    splits = np.cumsum([sizes[0]] + [2 * s for s in sizes[1:]])[:-1].tolist()
    blocks = jnp.split(x, splits, axis=1)

    edge_blocks = None
    if x_edge is not None:
        radial = _radial_mlp(params["rad"], x_edge)
        # Per-|m| radial weights, sized by each block's own input width.
        widths = [params["fc_m0"]["w"].shape[1]]
        widths += [params[f"m{m}"]["w"].shape[1] for m in range(1, mmax + 1)]
        edge_blocks = jnp.split(radial, np.cumsum(widths)[:-1].tolist(), axis=1)

    x_0 = blocks[0].reshape(n_edges, -1)
    if edge_blocks is not None:
        x_0 = x_0 * edge_blocks[0]
    x_0 = _linear(params["fc_m0"], x_0)
    gating, x_0 = x_0[:, :extra_m0], x_0[:, extra_m0:]
    out = [x_0.reshape(n_edges, -1, out_channels)]

    for m in range(1, mmax + 1):
        x_m = blocks[m].reshape(n_edges, 2, -1)
        if edge_blocks is not None:
            x_m = x_m * edge_blocks[m][:, None, :]
        real, imag = _so2_m_conv(params[f"m{m}"]["w"], x_m, out_channels)
        out.extend((real, imag))

    return jnp.concatenate(out, axis=1), gating


def _so2_m_conv(weight: Array, x_m: Array, out_channels: int) -> tuple[Array, Array]:
    """Complex-linear map `(W1 + i W2)(x_r + i x_i)` for one `|m| > 0` block."""
    half = weight.shape[0] // 2
    w1, w2 = weight[:half], weight[half:]
    real, imag = x_m[:, 0], x_m[:, 1]
    out_real = real @ w1.T - imag @ w2.T
    out_imag = real @ w2.T + imag @ w1.T
    return (
        out_real.reshape(x_m.shape[0], -1, out_channels),
        out_imag.reshape(x_m.shape[0], -1, out_channels),
    )


# ---------------------------------------------------------------------------
# Index constants
# ---------------------------------------------------------------------------


@functools.cache
def _lm_order(lmax: int, mmax: int) -> tuple[np.ndarray, np.ndarray]:
    """`(l, m)` of every coefficient in L-order, truncated at `mmax`."""
    degrees, orders = [], []
    for l in range(lmax + 1):
        m = np.arange(-min(mmax, l), min(mmax, l) + 1)
        orders.append(m)
        degrees.append(np.full(len(m), l))
    return np.concatenate(degrees), np.concatenate(orders)


@functools.cache
def _to_m_permutation(lmax: int, mmax: int) -> np.ndarray:
    """L-order indices, listed in m'-order: `x_m = x[perm]`."""
    _, orders = _lm_order(lmax, mmax)
    perm = [np.flatnonzero(orders == 0)]
    for m in range(1, mmax + 1):
        perm.append(np.flatnonzero(orders == m))
        perm.append(np.flatnonzero(orders == -m))
    return np.concatenate(perm)


@functools.cache
def _m_sizes(lmax: int, mmax: int) -> tuple[int, ...]:
    """Coefficients per `|m|` (the real half only, for `m > 0`)."""
    _, orders = _lm_order(lmax, mmax)
    return tuple(int((orders == m).sum()) for m in range(mmax + 1))


@functools.cache
def _l_expand_index(lmax: int) -> np.ndarray:
    """Degree of each L-ordered coefficient — expands per-degree weights."""
    return np.concatenate([np.full(2 * l + 1, l) for l in range(lmax + 1)])


@functools.cache
def _gate_expand_index(lmax: int, mmax: int, m_prime: bool) -> np.ndarray:
    """Gate index of each non-scalar coefficient, in m'-order or L-order."""
    if not m_prime:
        return np.concatenate(
            [np.full(min(2 * l + 1, 2 * mmax + 1), l - 1) for l in range(1, lmax + 1)]
        )
    index = [np.arange(lmax)]
    for m in range(1, mmax + 1):
        index += [np.arange(m - 1, lmax), np.arange(m - 1, lmax)]
    return np.concatenate(index)


@functools.cache
def _balance_degree_weight(lmax: int) -> np.ndarray:
    """RMS-norm weights that give every degree the same say: `1/((2l+1)(lmax+1))`."""
    return np.concatenate(
        [np.full(2 * l + 1, 1.0 / (2 * l + 1) / (lmax + 1)) for l in range(lmax + 1)]
    )
