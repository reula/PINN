"""Broyden's method as a *root finder*, from ``jaxopt``.

This is the only Broyden in the JAX ecosystem.  ``jax.scipy.optimize`` offers a
single method, ``"BFGS"``, and no Broyden; ``jaxopt.Broyden`` is a limited-memory
rank-1 quasi-Newton method for ``f(x) = 0``, and its own docstring is careful
about why that is a different problem from minimisation: the function is not a
gradient, so its Jacobian is not symmetric, symmetry cannot enter the secant
conditions, and each update is rank-1 rather than rank-2.

That difference is the reason this is worth running here rather than a curiosity.
The ansatz is hard-constrained, so the exact solution is (nearly) representable and
the residual can be driven to (nearly) zero -- ``r(theta) = 0`` is a meaningful
target, not a proxy for one.  A root finder attacks that directly, where the three
minimisers minimise ``||r||^2`` and are free to leave the residual spread out.

Two structural requirements, both about the method rather than the code:

* **A square system.**  Broyden needs as many equations as unknowns.  The default
  configuration already satisfies this exactly -- ``n_coll`` defaults to the
  parameter count, so the residual is square at 2201.  When the soft hand-over adds
  its two penalty rows per edge point the system is over-determined, and the phase
  selects ``n`` rows at an even stride.  Which ``n`` equations you impose is part of
  a root-finding formulation, so the choice is reported.
* **No redrawing inside the phase.**  The secant history *is* the approximate
  Jacobian of the residual it was built from.  Redrawing the collocation points
  mid-run does not merely fail to help: it silently invalidates the model.  Use
  ``resample_rounds``, which restarts the solver on a fresh system and is the only
  place redrawing belongs.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from ..config import Config
from ..losses import Objective

INSTALL_HINT = ("jaxopt is not installed in this interpreter.  It is the only "
                "JAX Broyden and is pure Python:\n"
                "    pip install jaxopt==0.8.5\n"
                "It needs only jax>=0.2.18, so it works against the hub's 0.11.1.")


def square_rows(r, n: int):
    """Row selection making an over-determined residual square, at an even stride.

    Deterministic, so a run is reproducible, and spread over the whole vector so
    neither the PDE rows nor the hand-over penalty rows are dropped wholesale.
    """
    m = int(r.shape[0])
    if m == n:
        return r
    if m < n:
        raise ValueError(
            f"Broyden is a root finder and needs a square (or thicker) system: "
            f"{m} equations for {n} unknowns.  Raise n_coll.")
    idx = jnp.linspace(0, m - 1, n)
    return r[jnp.round(idx).astype(jnp.int32)]


def jaxopt_broyden_phase(objective: Objective, flat, cfg: Config,
                         verbose: bool = True,
                         callback: Optional[Callable[[int, float, object], None]] = None,
                         step_offset: int = 0,
                         resample: Optional[Callable[[], Objective]] = None,
                         ) -> Tuple[jax.Array, List[Dict], Dict]:
    """Solve ``r(theta) = 0`` with ``jaxopt.Broyden``.

    ``resample`` is ignored on purpose -- see the module docstring.
    """
    try:
        import jaxopt
    except ImportError as exc:                                   # pragma: no cover
        raise ImportError(INSTALL_HINT) from exc

    t0 = time.time()
    theta = jnp.asarray(flat)
    n = int(theta.shape[0])
    n_rows = int(objective.residual(theta).shape[0])

    def fun(x):
        return square_rows(objective.residual(x), n)

    maxiter = int(getattr(cfg, "broyden_steps", 500))
    tol = float(getattr(cfg, "broyden_tol", 1e-12))
    hist_size = int(getattr(cfg, "broyden_history", 0)) or None

    solver = jaxopt.Broyden(
        fun=fun, maxiter=maxiter, tol=tol, history_size=hist_size, jit=True,
        stepsize=float(getattr(cfg, "broyden_stepsize", 0.0)),
        linesearch=str(getattr(cfg, "broyden_linesearch", "backtracking")),
        maxls=int(getattr(cfg, "broyden_maxls", 15)),
        decrease_factor=float(getattr(cfg, "broyden_decrease", 0.8)))

    loss_fn = objective._loss          # already jitted; .loss() converts to float
    theta_0 = theta
    loss0 = float(loss_fn(theta))
    state = solver.init_state(theta)

    history: List[Dict] = [{"step": step_offset + 0, "loss": loss0}]
    if callback:
        callback(step_offset, loss0, theta)

    stopped = "budget"
    it = 0
    for it in range(1, maxiter + 1):
        theta, state = solver.update(theta, state)
        loss = float(loss_fn(theta))
        history.append({"step": step_offset + it, "loss": loss})
        if callback:
            callback(step_offset + it, loss, theta)
        if verbose and (it % max(int(getattr(cfg, "log_every", 25)), 1) == 0):
            print(f"[broyden] step {it:6d}  loss {loss:.6e}  "
                  f"|r| {float(state.error):.3e}  ls {int(state.num_linesearch_iter)}  "
                  f"({time.time() - t0:.1f}s)", flush=True)
        if not bool(jnp.isfinite(state.error)):
            stopped = "diverged"
            break
        if float(state.error) < tol:
            stopped = "|r| below tolerance"
            break

    loss1 = float(loss_fn(theta))
    info = {
        "optimizer": "jaxopt_broyden",
        "iterations": it,
        "wall": time.time() - t0,
        "stopped": stopped,
        "loss_final": loss1,
        "loss_initial": loss0,
        "residual_norm_final": float(state.error),
        "n_parameters": n,
        "n_residual_rows": n_rows,
        "n_equations_used": min(n, n_rows),
        "squared_by": "none" if n_rows == n else "even-stride row selection",
        "history_size": hist_size if hist_size else "full",
        "resamples": 0,
        "first_step_norm": float(jnp.linalg.norm(theta_0 - theta))
        if not jnp.allclose(theta, theta_0) else float(jnp.linalg.norm(
            objective.residual(theta_0))),
    }
    if verbose:
        print(f"[broyden] Broyden: loss {loss0:.6e} -> {loss1:.6e} in {it} iterations "
              f"({info['wall']:.1f}s); {stopped}; |r| {info['residual_norm_final']:.3e}, "
              f"equations {info['n_equations_used']}/{n_rows}", flush=True)
    return theta, history, info
