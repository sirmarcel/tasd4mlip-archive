"""Calculator factory: a short name -> a configured ASE calculator.

Maps relaxation-oriented names to the package's JAX calculators: MACE-MP-0,
PET, and UMA-S (optionally + D3(BJ) via ASE's `SumCalculator`). The `pet` name
is generic over any pet-jax checkpoint (PET-MAD XS/S and future variants); the
checkpoint argument selects the variant. PET models used standalone — no D3.
UMA is named per head: ``uma-omat`` is PBE without dispersion and takes a D3
term exactly as MACE-MP-0 does, while ``uma-odac`` is PBE+D3 already, so no
``uma-odac+d3`` exists — it would double count.
The checkpoint is passed in explicitly by the caller, no defaults, but at
least `sadmof.paths` resolves the repo's names to directories.
"""

from pathlib import Path

__all__ = ["get_calculator", "CALCULATORS"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


CALCULATORS = [
    "mace-mp0",
    "mace-mp0+d3",
    "pet",
    "uma-omat",
    "uma-omat+d3",
    "uma-odac",
]


def get_calculator(
    name: str,
    *,
    checkpoint: str | Path,
    dtype: str = "float64",
    stress: bool = True,
    d3_skin: float = 5.0,
    **kwargs,
):
    """Build an ASE calculator by name.

    Args:
        name: One of ``CALCULATORS`` — ``mace-mp0`` (MACE-MP-0 alone),
            ``mace-mp0+d3`` (MACE-MP-0 + DFT-D3(BJ, PBE) via `SumCalculator`),
            ``pet`` (a PET model alone; no D3, see module docstring), or
            ``uma-omat`` / ``uma-omat+d3`` / ``uma-odac`` (UMA-S at the named
            head). The ``pet`` variant (XS / S / …) is set by ``checkpoint``.
        checkpoint: For MACE, a folder with the converted ``model.yaml`` /
            ``model.msgpack`` pair (see
            `sources/processed/mace-mp-0-medium/build.py`); for PET, a pet-jax
            checkpoint dir (``model.msgpack`` / ``metadata.yaml``, e.g.
            `sources/processed/pet-mad-xs/`); for UMA, a converted checkpoint
            dir (``config.json`` / ``model.npz``, e.g.
            `sources/processed/uma-s-1p2/`).
        dtype: ``"float64"`` (default, for relaxation) or ``"float32"``. Shared
            by every sub-calculator so the global JAX x64 flag stays consistent.
        stress: Report stress — required for cell relaxation.
        d3_skin: D3 Verlet-skin radius (Å) for the ``+d3`` composite's second
            calculator; ignored otherwise (it can't ride `**kwargs`, which go
            to the base calculator — the `SumCalculator` has two).
        **kwargs: Forwarded to the base calculator (MACE, PET or UMA) — e.g.
            ``skin``, for PET also ``no_shadow`` / ``bucket_strategy`` /
            ``extra_neighbors`` / ``cutoff_override``, and for UMA also
            ``charge`` / ``spin`` / ``bucket_strategy``. Unset → the
            calculator's own defaults.

    Returns:
        An ASE calculator (a `MACECalculator`, a `UPETCalculator`, a
        `UMACalculator`, or a `SumCalculator` of one of those plus D3).
    """
    if name not in CALCULATORS:
        raise ValueError(f"unknown calculator {name!r}; choose from {CALCULATORS}")

    base, _, dispersion = name.partition("+")

    if base == "pet":
        from .pet import UPETCalculator

        calculator = UPETCalculator.from_checkpoint(
            str(checkpoint), default_dtype=dtype, stress=stress, **kwargs
        )
    elif base.startswith("uma-"):
        from .uma import UMACalculator

        calculator = UMACalculator.from_checkpoint(
            checkpoint,
            task=base.removeprefix("uma-"),
            default_dtype=dtype,
            stress=stress,
            **kwargs,
        )
    else:
        from .mace import MACECalculator

        calculator = MACECalculator.from_checkpoint(
            checkpoint, default_dtype=dtype, stress=stress, **kwargs
        )

    if not dispersion:
        return calculator

    from ase.calculators.mixing import SumCalculator

    from .d3 import D3Calculator

    d3 = D3Calculator(default_dtype=dtype, skin=d3_skin, stress=stress)
    return SumCalculator([calculator, d3])
