"""Backend-agnostic structure relaxation.

`relax` runs an ASE optimizer over a `FrechetCellFilter` (cell + positions
together) and returns convergence/timing metadata. It writes nothing and
mutates the passed `atoms` in place — file output is the caller's concern.
"""

import importlib
import time
import warnings

# FrechetCellFilter calls scipy.linalg.logm on a near-identity deformation
# gradient every force/stress eval; its error estimator fires spurious
# RuntimeWarnings on large cells, flooding logs.
warnings.filterwarnings("ignore", message="logm result may be inaccurate")

__all__ = ["relax", "OPTIMIZERS"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


OPTIMIZERS = {
    # name -> (module_path, class_name, default_kwargs)
    # bfgs: dense quasi-Newton, most robust but O(N^3)/step — small systems only.
    "bfgs": ("ase.optimize", "BFGS", {}),
    # lbfgs: cheap per step, but can overshoot when initial forces are large.
    "lbfgs": ("ase.optimize", "LBFGS", {}),
    # lbfgs-ls: lbfgs + Wolfe line search; more force evals but never
    # overshoots. The safe default.
    "lbfgs-ls": ("ase.optimize", "LBFGSLineSearch", {}),
}


def relax(
    atoms,
    calc,
    *,
    fmax: float = 0.005,
    max_steps: int = 25000,
    optimizer: str = "lbfgs-ls",
    trajectory_path: str | None = None,
    logfile: str | None = "-",
) -> dict:
    """Relax cell + positions with a quasi-Newton optimizer.

    Args:
        atoms: ASE Atoms to relax (modified in place).
        calc: ASE calculator providing energy/forces/stress.
        fmax: Force convergence threshold (eV/Å).
        max_steps: Maximum optimizer steps.
        optimizer: One of ``OPTIMIZERS``; ``lbfgs-ls`` is the safe default.
        trajectory_path: If given, write the optimization trajectory there.
        logfile: ASE optimizer logfile (``"-"`` for stdout, None to silence).

    Returns:
        dict with ``n_steps``, ``converged``, ``wall_s``, and ``fmax_final``
        (eV/Å, measured on the filter gradient).
    """
    from ase.filters import FrechetCellFilter

    atoms.calc = calc
    ecf = FrechetCellFilter(atoms)
    opt = _make_optimizer(optimizer, ecf, logfile=logfile, trajectory=trajectory_path)

    t0 = time.time()
    opt.run(fmax=fmax, steps=max_steps)
    wall = time.time() - t0

    # Read convergence and fmax from the wrapped FrechetCellFilter so both
    # refer to the filter gradient (atomic + cell rows) the optimizer actually
    # converges on — not bare `atoms.get_forces()`.
    n_steps = opt.get_number_of_steps()
    converged = bool(opt.converged())
    fmax_final = float(opt.optimizable.gradient_norm(opt.optimizable.get_gradient()))

    return {
        "n_steps": n_steps,
        "converged": converged,
        "wall_s": round(wall, 1),
        "fmax_final": round(fmax_final, 6),
    }


# ---------------------------------------------------------------------------
# Optimizer registry
# ---------------------------------------------------------------------------


def _make_optimizer(name, ecf, logfile, trajectory):
    if name not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {name!r}; choose from {sorted(OPTIMIZERS)}")
    module_path, cls_name, defaults = OPTIMIZERS[name]
    cls = getattr(importlib.import_module(module_path), cls_name)
    kwargs = dict(defaults)
    kwargs.update({"logfile": logfile, "trajectory": trajectory})
    return cls(ecf, **kwargs)
