"""MACE ScaleShiftMACE — flax module + marathon-style energy/predict.

Public surface: `MACE` with `__call__` (per-atom energies),
`energy(params, batch)`, and `predict(params, batch)`. Loaded from a
yaml + msgpack pair produced at `sources/processed/mace-mp-0-medium/`.
MOF0 head: `use_agnesi=True`, `use_zbl=True`, `residual_first_layer=False`.
"""

import jax
import jax.numpy as jnp

import functools

import e3nn_jax as e3nn
import flax.linen as nn

from jaxtyping import Array, Bool, Float, Int

__all__ = ["MACE"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class MACE(nn.Module):
    """MACE ScaleShiftMACE — covers both MACE-MP-0 and MACE-MP-MOF0.

    Defaults reflect the MP-0 medium (L1) checkpoint. The MOF0 head sets
    `use_agnesi=True`, `use_zbl=True`, `residual_first_layer=False`, and
    a different `num_elements` / `mlp_hidden_dim`.

    Any `num_interactions >= 1` is valid; as in PyTorch MACE, every layer
    keeps `hidden_irreps` with a linear readout, except the last, which
    collapses to scalars with an MLP readout. At `num_interactions=1` the
    single layer takes the *last*-layer role, so its parameters do not
    coincide with layer 0 of a deeper model.

    `init()` from scratch yields a working model for the default (MP-0)
    configuration. With `use_agnesi` / `use_zbl` the physical constants
    (covalent radii, ZBL coefficients) init to zeros, which produces NaNs
    in the forward pass or its gradients — fill them from a checkpoint or
    from `ase.data.covalent_radii` plus the canonical ZBL coefficients.
    """

    cutoff: float = 6.0
    num_bessel: int = 10
    num_polynomial_cutoff: int = 5
    max_ell: int = 3
    num_interactions: int = 2
    correlation: int = 3
    num_elements: int = 95
    avg_num_neighbors: float = 61.964672446250916
    hidden_irreps_str: str = "128x0e+128x1o"
    atomic_inter_scale: float = 0.8041538754478097
    atomic_inter_shift: float = 0.16409696359187365
    mlp_hidden_dim: int = 16
    residual_first_layer: bool = True
    use_agnesi: bool = False
    use_zbl: bool = False

    def setup(self) -> None:
        self.hidden_irreps = e3nn.Irreps(self.hidden_irreps_str)
        self.sh_irreps = e3nn.Irreps([(1, (l, (-1) ** l)) for l in range(self.max_ell + 1)])
        # Layer-0 output: hidden mul × sh irreps (e.g. 128x0e+128x1o+128x2e+128x3o)
        self.interaction_irreps_0 = e3nn.Irreps(
            [(self.hidden_irreps[0].mul, ir) for _, ir in self.sh_irreps]
        )
        # TP output for layers > 0: hidden ⊗ sh, filtered to the interaction target set
        target_ir_set = {ir for _, ir in self.interaction_irreps_0}
        tp_irreps_later = []
        for mul_in, ir_in in self.hidden_irreps:
            for _, ir_sh in self.sh_irreps:
                for ir_out in ir_in * ir_sh:
                    if ir_out in target_ir_set:
                        tp_irreps_later.append((mul_in, ir_out))
        self.tp_irreps_later = e3nn.Irreps(tp_irreps_later)  # ty: ignore[invalid-argument-type]

    @nn.compact
    def __call__(
        self,
        R_ij: Float[Array, "pairs 3"],
        i: Int[Array, " pairs"],
        j: Int[Array, " pairs"],
        Z_i: Int[Array, " atoms"],
        pair_mask: Bool[Array, " pairs"],
        atom_mask: Bool[Array, " atoms"],
    ) -> Float[Array, " atoms"]:
        """Per-atom energies (no scatter, no force/stress)."""
        n_atoms = Z_i.shape[0]
        hidden_irreps = self.hidden_irreps
        mul = hidden_irreps[0].mul

        # Per-element baseline (`constants["atomic_energies"]`) is NOT applied
        # here. It is up to a calculator/wrapper to add it post-JIT in fp64
        # — large baselines (≈ -3500 eV for Pb) would swamp the ≈ 0.1 eV
        # interaction term in fp32, and JAX runs fp32 by default.

        # --- Edge features ---
        r_hat, r_ij = normalize_and_return_norm(R_ij, axis=-1)
        # bessel_basis divides by r — feed 1.0 for padded pairs, then mask out
        r_ij_safe = jnp.where(pair_mask, r_ij, 1.0)

        cutoff_vals = (
            polynomial_cutoff(r_ij_safe, self.cutoff, p=self.num_polynomial_cutoff)
            * pair_mask
        )

        if self.use_agnesi:
            agnesi_a = self.variable("constants", "agnesi_a", lambda: jnp.ones(())).value
            agnesi_p = self.variable("constants", "agnesi_p", lambda: jnp.ones(())).value
            agnesi_q = self.variable("constants", "agnesi_q", lambda: jnp.ones(())).value
            agnesi_cov_radii = self.variable(
                "constants", "agnesi_covalent_radii", lambda: jnp.zeros(119)
            ).value
            Z_sender = Z_i[j]
            Z_receiver = Z_i[i]
            r_bessel = agnesi_transform(
                r_ij_safe,
                Z_sender,
                Z_receiver,
                agnesi_a,
                agnesi_p,
                agnesi_q,
                agnesi_cov_radii,
            )
        else:
            r_bessel = r_ij_safe

        bessel_weights = self.variable(
            "constants",
            "bessel_weights",
            lambda: jnp.pi / self.cutoff * jnp.arange(1, self.num_bessel + 1),
        ).value
        bessel_prefactor = jnp.sqrt(2.0 / self.cutoff)
        edge_feats = bessel_basis(r_bessel, bessel_weights, bessel_prefactor)
        edge_feats = edge_feats * pair_mask[..., None] * cutoff_vals[..., None]

        if self.use_zbl:
            zbl_a_exp = self.variable("constants", "zbl_a_exp", lambda: jnp.zeros(())).value
            zbl_a_prefactor = self.variable(
                "constants", "zbl_a_prefactor", lambda: jnp.zeros(())
            ).value
            zbl_c = self.variable("constants", "zbl_c", lambda: jnp.zeros(4)).value
            zbl_p = self.variable("constants", "zbl_p", lambda: jnp.array(5.0)).value
            zbl_cov_radii = self.variable(
                "constants", "zbl_covalent_radii", lambda: jnp.zeros(119)
            ).value
            node_zbl = zbl_pair_repulsion(
                r_ij_safe,
                Z_i[j],
                Z_i[i],
                i,
                n_atoms,
                zbl_a_exp,
                zbl_a_prefactor,
                zbl_c,
                zbl_p,
                zbl_cov_radii,
                pair_mask,
            )
        else:
            node_zbl = jnp.zeros(n_atoms, dtype=R_ij.dtype)

        edge_sh = e3nn.spherical_harmonics(
            list(range(self.max_ell + 1)),
            e3nn.IrrepsArray("1o", r_hat),
            normalize=False,
            normalization="component",
        )

        # --- Node embedding ---
        node_feats_array = nn.Embed(
            num_embeddings=self.num_elements,
            features=mul,
            name="node_embedding",
        )(Z_i)

        # --- Interaction layers ---
        interaction_irreps = self.interaction_irreps_0
        node_es_list = []
        for layer_idx in range(self.num_interactions):
            is_first = layer_idx == 0
            is_last = layer_idx == self.num_interactions - 1
            use_residual = self.residual_first_layer or not is_first

            if is_first:
                in_irreps = e3nn.Irreps(f"{mul}x0e")
                tp_out_irreps = self.interaction_irreps_0
            else:
                in_irreps = hidden_irreps
                tp_out_irreps = self.tp_irreps_later

            num_tp_weights = sum(mul_o for mul_o, _ in tp_out_irreps)

            # Layer output: full hidden irreps, except the last layer, which
            # collapses to scalars — only the invariant readout follows.
            out_irreps = e3nn.Irreps(f"{mul}x0e") if is_last else hidden_irreps

            # Skip connection (species-dependent)
            if use_residual:
                sc = SpeciesSkipConnection(
                    irreps_in=in_irreps,
                    irreps_out=out_irreps,
                    num_elements=self.num_elements,
                    name=f"skip_tp_{layer_idx}",
                )(node_feats_array, Z_i)
            else:
                sc = None

            node_feats_up = EquivariantLinear(
                irreps_in=in_irreps,
                irreps_out=in_irreps,
                name=f"linear_up_{layer_idx}",
            )(e3nn.IrrepsArray(in_irreps, node_feats_array)).array

            tp_weights = RadialMLP(
                hidden_sizes=(64, 64, 64),
                output_size=num_tp_weights,
                name=f"radial_mlp_{layer_idx}",
            )(edge_feats)

            sender_feats = node_feats_up[j]
            message = weighted_tensor_product(
                e3nn.IrrepsArray(in_irreps, sender_feats),
                edge_sh,
                tp_weights,
                in_irreps,
                self.sh_irreps,
                tp_out_irreps,
            )

            msg_array = message.array * pair_mask[..., None]
            node_msg = jax.ops.segment_sum(msg_array, i, num_segments=n_atoms)

            msg_irreps = message.irreps
            node_msg = EquivariantLinear(
                irreps_in=msg_irreps,
                irreps_out=interaction_irreps,
                name=f"linear_{layer_idx}",
            )(e3nn.IrrepsArray(msg_irreps, node_msg)).array
            node_msg = node_msg / self.avg_num_neighbors

            if not use_residual:
                # Non-residual variant: skip_tp acts on the aggregated message
                node_msg = SpeciesSkipConnection(
                    irreps_in=interaction_irreps,
                    irreps_out=interaction_irreps,
                    num_elements=self.num_elements,
                    name=f"skip_tp_{layer_idx}",
                )(node_msg, Z_i)

            contracted = SymmetricContraction(
                irreps_in=interaction_irreps,
                irreps_out=out_irreps,
                correlation=self.correlation,
                num_elements=self.num_elements,
                name=f"symmetric_contraction_{layer_idx}",
            )(node_msg, Z_i)

            contracted = EquivariantLinear(
                irreps_in=out_irreps,
                irreps_out=out_irreps,
                name=f"product_linear_{layer_idx}",
            )(e3nn.IrrepsArray(out_irreps, contracted)).array

            node_feats_array = contracted + sc if sc is not None else contracted

            if not is_last:
                node_es = EquivariantLinear(
                    irreps_in=hidden_irreps,
                    irreps_out=e3nn.Irreps("1x0e"),
                    name=f"readout_{layer_idx}",
                )(e3nn.IrrepsArray(hidden_irreps, node_feats_array)).array[..., 0]
            else:
                scalars_irreps = e3nn.Irreps(f"{mul}x0e")
                mlp_irreps = e3nn.Irreps(f"{self.mlp_hidden_dim}x0e")
                r = EquivariantLinear(
                    irreps_in=scalars_irreps,
                    irreps_out=mlp_irreps,
                    name=f"readout_{layer_idx}_linear1",
                )(e3nn.IrrepsArray(scalars_irreps, node_feats_array[..., :mul])).array
                r = jax.nn.silu(r) * SILU_NORMALIZE2MOM
                node_es = EquivariantLinear(
                    irreps_in=mlp_irreps,
                    irreps_out=e3nn.Irreps("1x0e"),
                    name=f"readout_{layer_idx}_linear2",
                )(e3nn.IrrepsArray(mlp_irreps, r)).array[..., 0]

            node_es_list.append(node_es)

        node_inter_es = sum(node_es_list)
        node_inter_es = self.atomic_inter_scale * node_inter_es + self.atomic_inter_shift

        return (node_inter_es + node_zbl) * atom_mask

    def energy(
        self,
        params: dict,
        pos: Float[Array, "atoms 3"],
        cell: Float[Array, "3 3"],
        graph: dict[str, Array],
    ) -> tuple[Float[Array, ""], Float[Array, " atoms"]]:
        """Total energy + per-atom energies, both masked.

        `pos` (positions) and `cell` are the differentiable geometry; `graph`
        carries the static topology (`centers`, `others`, `cell_shifts`,
        `atomic_numbers`, `atom_mask`, `pair_mask`) — see
        `inputs.atoms_to_inputs`. Edge displacements are computed here, so
        `jax.grad` w.r.t. `pos` lands a position gradient (forces / Hessian)
        and w.r.t. `cell` the virial.

        Returns `(scalar_total, per_atom)`. The `(scalar, aux)` shape is
        what `jax.value_and_grad(has_aux=True)` expects.

        Does NOT include the per-element baseline; that's added post-JIT
        in fp64 by `MACECalculator`.
        """
        R_ij = pos[graph["others"]] - pos[graph["centers"]] + graph["cell_shifts"] @ cell
        per_atom = jnp.asarray(
            self.apply(
                params,
                R_ij,
                graph["centers"],
                graph["others"],
                graph["atomic_numbers"],
                graph["pair_mask"],
                graph["atom_mask"],
            )
        )
        per_atom = per_atom * graph["atom_mask"]
        return jnp.sum(per_atom), per_atom

    def predict(
        self,
        params: dict,
        pos: Float[Array, "atoms 3"],
        cell: Float[Array, "3 3"],
        graph: dict[str, Array],
    ) -> dict[str, Array]:
        """Scalar energy + per-atom forces + 3×3 stress for a single structure.

        Forces come from `-∂E/∂pos`; stress is the virial `dU/dε` at zero strain
        (eV, no volume normalisation — the calculator divides by volume to get
        ASE's convention). Both come from one `value_and_grad` over the strain
        tensor `ε`: `pos` and `cell` are mapped through the deformation gradient
        `F = I + ε`, and at `ε → 0` the gradient w.r.t. `ε` is the virial tensor.
        """

        def energy_at(pos, eps):
            F = jnp.eye(3) + eps
            return self.energy(params, pos @ F, cell @ F, graph)

        (total, _), (neg_forces, virial) = jax.value_and_grad(
            energy_at, argnums=(0, 1), has_aux=True
        )(pos, jnp.zeros((3, 3)))

        forces = -neg_forces * graph["atom_mask"][..., None]
        return {"energy": total, "forces": forces, "stress": virial}


# ---------------------------------------------------------------------------
# Flax modules (parts)
# ---------------------------------------------------------------------------


class RadialMLP(nn.Module):
    """MLP matching e3nn `FullyConnectedNet` with SiLU activations.

    Each layer normalises the weight by `1 / sqrt(in_dim)` (so weights are
    stored raw), and intermediate activations are scaled by
    `SILU_NORMALIZE2MOM` to match e3nn's `normalize2mom(silu)`.
    """

    hidden_sizes: tuple[int, ...]
    output_size: int

    @nn.compact
    def __call__(self, x: Float[Array, "... in_dim"]) -> Float[Array, "... out_dim"]:
        for size in self.hidden_sizes:
            h_in = x.shape[-1]
            x = nn.Dense(size, use_bias=False)(x) / jnp.sqrt(h_in)
            x = jax.nn.silu(x) * SILU_NORMALIZE2MOM
        h_in = x.shape[-1]
        x = nn.Dense(self.output_size, use_bias=False)(x) / jnp.sqrt(h_in)
        return x


class EquivariantLinear(nn.Module):
    """Equivariant linear map. Wraps `e3nn.FunctionalLinear` and adds
    batch-dimension support via `jax.vmap`.

    Weights are a single flat array (`linear.num_weights`) split into
    per-instruction blocks inside the call.
    """

    irreps_in: e3nn.Irreps
    irreps_out: e3nn.Irreps

    @nn.compact
    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        linear = e3nn.FunctionalLinear(self.irreps_in, self.irreps_out)
        w = self.param(
            "weight",
            nn.initializers.normal(stddev=0.01),
            (linear.num_weights,),
        )
        weights = linear.split_weights(w)

        irreps_in = e3nn.Irreps(self.irreps_in)
        irreps_out = e3nn.Irreps(self.irreps_out)

        leading_shape = x.shape[:-1]
        x_flat = x.array.reshape(-1, x.shape[-1])

        def apply_one(arr):
            return linear(weights, e3nn.IrrepsArray(irreps_in, arr)).array

        out_flat = jax.vmap(apply_one)(x_flat)
        out = out_flat.reshape(*leading_shape, irreps_out.dim)
        return e3nn.IrrepsArray(irreps_out, out)


class SpeciesSkipConnection(nn.Module):
    """Species-dependent linear map: `out[u] = Σ_v W[u, z, v] · feat[v]`.

    Matches PyTorch MACE's `skip_tp` (`FullyConnectedTensorProduct` in
    'uvw' mode, acting on `(node_feats, node_attrs_onehot)`). Only
    `0e × 0e → 0e` paths survive because `node_attrs` is scalar; for
    `l > 0` paths the irrep equality `ir_in == ir_out` still gates the
    coupling, and the CG for `l × 0e → l` contributes a `1/sqrt(ir.dim)`
    that cancels the `sqrt(ir.dim)` baked into the path weight.
    """

    irreps_in: e3nn.Irreps
    irreps_out: e3nn.Irreps
    num_elements: int

    @nn.compact
    def __call__(
        self,
        node_feats: Float[Array, "n irreps_in_dim"],
        element_indices: Int[Array, " n"],
    ) -> Float[Array, "n irreps_out_dim"]:
        n_nodes = node_feats.shape[0]
        out_chunks = []
        for mul_out, ir_out in self.irreps_out:
            matched = False
            in_offset = 0
            for mul_in, ir_in in self.irreps_in:
                if ir_in == ir_out:
                    W = self.param(
                        f"w_{ir_in}",
                        nn.initializers.normal(
                            stddev=1.0 / jnp.sqrt(mul_in * self.num_elements)
                        ),
                        (self.num_elements, mul_out, mul_in),
                    )
                    W_selected = W[element_indices]

                    chunk_in = node_feats[
                        ..., in_offset : in_offset + mul_in * ir_in.dim
                    ].reshape(n_nodes, mul_in, ir_in.dim)
                    chunk_out = jnp.einsum("nuv,nvm->num", W_selected, chunk_in)

                    path_weight = 1.0 / jnp.sqrt(mul_in * self.num_elements)
                    chunk_out = chunk_out * path_weight
                    out_chunks.append(chunk_out.reshape(n_nodes, mul_out * ir_out.dim))
                    matched = True
                    break
                in_offset += mul_in * ir_in.dim

            if not matched:
                out_chunks.append(
                    jnp.zeros((n_nodes, mul_out * ir_out.dim), dtype=node_feats.dtype)
                )

        return jnp.concatenate(out_chunks, axis=-1)


class SymmetricContraction(nn.Module):
    r"""MACE Eq. 10-11 symmetric contraction (recursive U-matrix form).

    For `correlation=3` and a 0e output:

        out = U3·W3·x·y  → [b, c, d, d]
        c   = U2·W2·y + out;  out = c·x → [b, c, d]
        c   = U1·W1·y + out;  out = c·x → [b, c]   (final)

    Per-element weights are gathered by `element_indices` rather than
    expressed as a one-hot ⊗ dense contraction.
    """

    irreps_in: e3nn.Irreps
    irreps_out: e3nn.Irreps
    correlation: int
    num_elements: int

    @nn.compact
    def __call__(
        self,
        x: Float[Array, "n irreps_in_dim"],
        element_indices: Int[Array, " n"],
    ) -> Float[Array, "n irreps_out_dim"]:
        mul = self.irreps_in[0].mul
        x_r = _reshape_irreps_to_channels(x, self.irreps_in)
        per_channel_irreps = e3nn.Irreps([(1, ir) for _, ir in self.irreps_in])

        out_chunks = []
        for mul_out, ir_out in self.irreps_out:
            assert mul_out == mul
            num_eq = ir_out.dim

            Us = []
            for nu in range(1, self.correlation + 1):
                U = self.variable(
                    "constants",
                    f"U_{ir_out}_{nu}",
                    functools.partial(_symmetric_basis, per_channel_irreps, ir_out, nu),
                ).value
                Us.append(U)

            w_max = self.param(
                f"w_{ir_out}_max",
                nn.initializers.normal(0.01),
                (self.num_elements, Us[-1].shape[-1], mul),
            )
            w_lower = []
            for step in range(self.correlation - 1):
                nu = self.correlation - step - 1
                w = self.param(
                    f"w_{ir_out}_{nu}",
                    nn.initializers.normal(0.01),
                    (self.num_elements, Us[nu - 1].shape[-1], mul),
                )
                w_lower.append(w)

            out = _symmetric_contraction_recursive(
                Us, w_max, w_lower, x_r, element_indices, num_eq, self.correlation
            )

            if num_eq == 1:
                out_chunks.append(out)
            else:
                out_chunks.append(out.reshape(out.shape[0], mul * num_eq))

        return jnp.concatenate(out_chunks, axis=-1)


# ---------------------------------------------------------------------------
# Tensor product / symmetric contraction helpers
# ---------------------------------------------------------------------------


def weighted_tensor_product(
    node_feats_sender: e3nn.IrrepsArray,
    edge_sh: e3nn.IrrepsArray,
    radial_weights: Float[Array, "edges num_tp_weights"],
    input_irreps: e3nn.Irreps,
    sh_irreps: e3nn.Irreps,
    target_irreps: e3nn.Irreps,
) -> e3nn.IrrepsArray:
    """uvu-mode weighted tensor product for MACE message passing.

    For each Clebsch-Gordan path `(l_in, l_sh) → l_out` with `l_out` in
    `target_irreps`, the radial MLP provides per-channel weights and the
    contraction is `tp = einsum("ijo,...ci,...j->...co", CG, in, sh) * w`.

    Instruction ordering matches PyTorch MACE's
    `tp_out_irreps_with_instructions` (sort by output irrep), which is
    what the saved weights expect.
    """
    target_ir_set = {ir for _, ir in target_irreps}

    in_chunks: dict = {}
    offset = 0
    for mul, ir in input_irreps:
        in_chunks[ir] = node_feats_sender.array[
            ..., offset : offset + mul * ir.dim
        ].reshape(*node_feats_sender.array.shape[:-1], mul, ir.dim)
        offset += mul * ir.dim

    sh_chunks: dict = {}
    offset = 0
    for mul, ir in sh_irreps:
        assert mul == 1
        sh_chunks[ir] = edge_sh.array[..., offset : offset + ir.dim]
        offset += ir.dim

    instructions = []
    for ir_in in in_chunks:
        for ir_sh in sh_chunks:
            for ir_out in ir_in * ir_sh:
                if ir_out in target_ir_set:
                    instructions.append((ir_in, ir_sh, ir_out))

    instructions.sort(key=lambda instr: (instr[2].l, -instr[2].p))

    out_chunks = []
    out_irreps_list = []
    weight_offset = 0

    for ir_in, ir_sh, ir_out in instructions:
        chunk_in = in_chunks[ir_in]
        chunk_sh = sh_chunks[ir_sh]
        mul = chunk_in.shape[-2]

        cg = e3nn.clebsch_gordan(ir_in.l, ir_sh.l, ir_out.l) * jnp.sqrt(ir_out.dim)
        tp = jnp.einsum("ijo,...ci,...j->...co", cg, chunk_in, chunk_sh)

        w = radial_weights[..., weight_offset : weight_offset + mul]
        tp = tp * w[..., :, None]
        weight_offset += mul

        out_chunks.append(tp.reshape(*tp.shape[:-2], mul * ir_out.dim))
        out_irreps_list.append((mul, ir_out))

    out_irreps = e3nn.Irreps(out_irreps_list)  # ty: ignore[invalid-argument-type]
    out_array = jnp.concatenate(out_chunks, axis=-1)
    return e3nn.IrrepsArray(out_irreps, out_array)


def _symmetric_basis(
    irreps_in: e3nn.Irreps, ir_out: e3nn.Irrep, nu: int
) -> Float[Array, "..."]:
    """Symmetric tensor-product basis in PyTorch-MACE axis layout:
    `(ir_out.dim?, d, ..., d, num_paths)` with `d = irreps_in.dim` repeated
    `nu` times and the m-axis leading for l > 0.

    Default initializer for the `U_*` constants. Checkpoints override these
    with the torch-extracted U matrices, which use a redundant path basis
    (more, linearly dependent paths) spanning the same symmetric subspace —
    weight shapes follow `U.shape[-1]` either way, so both are
    self-consistent but not interchangeable.
    """
    basis = e3nn.reduced_symmetric_tensor_product_basis(
        irreps_in, nu, keep_ir=e3nn.Irreps([(1, ir_out)])
    )
    arr = jnp.asarray(basis.array)
    num_paths = arr.shape[-1] // ir_out.dim
    arr = arr.reshape(arr.shape[:-1] + (num_paths, ir_out.dim))
    if ir_out.dim == 1:
        return arr[..., 0]
    return jnp.moveaxis(arr, -1, 0)


def _symmetric_contraction_recursive(
    Us, w_max, w_lower, x, element_indices, num_eq: int, correlation: int
):
    U_max = Us[correlation - 1]
    w_sel = w_max[element_indices]

    if num_eq == 1:
        out = jnp.einsum("wxik,bkc,bci->bcwx", U_max, w_sel, x)
    else:
        out = jnp.einsum("vwxik,bkc,bci->bcvwx", U_max, w_sel, x)

    for step, w in enumerate(w_lower):
        nu = correlation - step - 1
        U = Us[nu - 1]
        w_sel = w[element_indices]
        if num_eq == 1:
            if nu == 2:
                c = jnp.einsum("wxk,bkc->bcwx", U, w_sel)
                out = jnp.einsum("bcwi,bci->bcw", c + out, x)
            elif nu == 1:
                c = jnp.einsum("wk,bkc->bcw", U, w_sel)
                out = jnp.einsum("bci,bci->bc", c + out, x)
        else:
            if nu == 2:
                c = jnp.einsum("vwxk,bkc->bcvwx", U, w_sel)
                out = jnp.einsum("bcvwi,bci->bcvw", c + out, x)
            elif nu == 1:
                c = jnp.einsum("vwk,bkc->bcvw", U, w_sel)
                out = jnp.einsum("bcvi,bci->bcv", c + out, x)
    return out


def _reshape_irreps_to_channels(
    x: Float[Array, "n flat_dim"], irreps: e3nn.Irreps
) -> Float[Array, "n mul coupling_dim"]:
    """Flat irreps layout → `(n, mul, Σ ir.dim)` (PyTorch `reshape_irreps`).

    Input: `[128 scalars | 384 vectors | 640 matrices | 896 tensors]`.
    Output: `[n, 128, 16]` with each channel carrying `[0e, 1o, 2e, 3o]`.
    """
    batch = x.shape[0]
    mul = irreps[0].mul
    chunks = []
    offset = 0
    for m, ir in irreps:
        assert m == mul
        chunks.append(x[:, offset : offset + mul * ir.dim].reshape(batch, mul, ir.dim))
        offset += mul * ir.dim
    return jnp.concatenate(chunks, axis=-1)


# ---------------------------------------------------------------------------
# Radial / pair-energy pieces
# ---------------------------------------------------------------------------


def bessel_basis(
    r: Float[Array, " edges"],
    bessel_weights: Float[Array, " num_basis"],
    prefactor: Float[Array, ""],
) -> Float[Array, "edges num_basis"]:
    """MACE Eq. 7 Bessel radial basis. `bessel_weights = π/r_max · [1..K]`."""
    return prefactor * jnp.sin(bessel_weights * r[..., None]) / r[..., None]


def polynomial_cutoff(
    r: Float[Array, " edges"],
    r_max: float | Float[Array, "*"],
    p: float | Float[Array, "*"] = 5,
) -> Float[Array, " edges"]:
    """Polynomial envelope going smoothly from 1 to 0 over `[0, r_max]`."""
    r_over_r_max = r / r_max
    envelope = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * r_over_r_max**p
        + p * (p + 2.0) * r_over_r_max ** (p + 1)
        - (p * (p + 1.0) / 2.0) * r_over_r_max ** (p + 2)
    )
    return jnp.where(r < r_max, envelope, 0.0)


def agnesi_transform(
    r: Float[Array, " edges"],
    Z_i: Int[Array, " edges"],
    Z_j: Int[Array, " edges"],
    a: Float[Array, ""],
    p: Float[Array, ""],
    q: Float[Array, ""],
    covalent_radii: Float[Array, " 119"],
) -> Float[Array, " edges"]:
    """Learnable soft-core distance scaling used by MACE-MP-MOF0.

    Maps `r → 1 / (1 + a·(r/r₀)^q / (1 + (r/r₀)^(q-p)))` where `r₀` is the
    sum of element-pair covalent radii.
    """
    r_0 = 0.5 * (covalent_radii[Z_i] + covalent_radii[Z_j])
    r_over_r0 = r / r_0
    return 1.0 / (1.0 + a * r_over_r0**q / (1.0 + r_over_r0 ** (q - p)))


def zbl_pair_repulsion(
    r: Float[Array, " edges"],
    Z_i_edge: Int[Array, " edges"],
    Z_j_edge: Int[Array, " edges"],
    i: Int[Array, " edges"],
    n_atoms: int,
    a_exp: Float[Array, ""],
    a_prefactor: Float[Array, ""],
    c: Float[Array, " 4"],
    p: Float[Array, ""],
    covalent_radii: Float[Array, " 119"],
    pair_mask: Bool[Array, " edges"],
) -> Float[Array, " n_atoms"]:
    """Screened nuclear repulsion (ZBL) summed per receiver atom."""
    Z_s = Z_i_edge.astype(r.dtype)
    Z_r = Z_j_edge.astype(r.dtype)
    a = a_prefactor * 0.529 / (Z_s**a_exp + Z_r**a_exp)
    r_over_a = r / a
    phi = (
        c[0] * jnp.exp(-3.2 * r_over_a)
        + c[1] * jnp.exp(-0.9423 * r_over_a)
        + c[2] * jnp.exp(-0.4028 * r_over_a)
        + c[3] * jnp.exp(-0.2016 * r_over_a)
    )
    r_safe = jnp.where(pair_mask, r, 1.0)
    v_edges = (14.3996 * Z_s * Z_r) / r_safe * phi
    r_cut = covalent_radii[Z_i_edge] + covalent_radii[Z_j_edge]
    envelope = polynomial_cutoff(r, r_cut, p=p)
    v_edges = 0.5 * v_edges * envelope * pair_mask
    return jax.ops.segment_sum(v_edges, i, num_segments=n_atoms)


# ---------------------------------------------------------------------------
# Low-level numerics
# ---------------------------------------------------------------------------
#
# `safe_norm` and `normalize_and_return_norm` are vendored from e3x
# (https://github.com/google-research/e3x), authors Oliver T. Unke and
# Hartmut Maennel. Apache 2.0 license.


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def safe_norm(x, axis=None, keepdims=False):
    """L2-norm with numerically stable forward and backward pass.

    Uses max-scaling to avoid overflow and a custom JVP that returns zero
    tangent when the norm is zero (instead of NaN).

    Diverges from e3x: the rescaling max is taken per-`axis` slice, not over
    the whole array. A global max structurally couples every atom to every
    other, and only the custom JVP keeps that coupling out of the Hessian;
    per-slice scaling makes the locality of the sparsity pattern a property
    of the primal itself.
    """
    a = jnp.maximum(jnp.max(jnp.abs(x), axis=axis, keepdims=True), jnp.finfo(x.dtype).tiny)
    b = x / a
    n = a * jnp.sqrt(jnp.sum(b * b, axis=axis, keepdims=True))
    if not keepdims:
        n = jnp.squeeze(n, axis=axis)
    return n


@safe_norm.defjvp
def _safe_norm_jvp(axis, keepdims, primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    n = safe_norm(x, axis=axis, keepdims=keepdims)
    safe_n = jnp.where(n > 0, n, 1)
    if not keepdims and axis is not None:
        safe_n = jnp.expand_dims(safe_n, axis=axis)
    tangent_out = jnp.sum(x_dot * x / safe_n, axis=axis, keepdims=keepdims)
    return n, tangent_out


def normalize_and_return_norm(x, axis=-1):
    """Return `(x / ||x||, ||x||)` with safe handling of zero-length vectors."""
    n = safe_norm(x, axis=axis, keepdims=True)
    safe = n * n > jnp.finfo(x.dtype).tiny
    x_normed = x / jnp.where(safe, n, 1.0)
    return x_normed, jnp.squeeze(n, axis=axis)


# e3nn `normalize2mom` constant for SiLU.
SILU_NORMALIZE2MOM = 1.6791768074035645
