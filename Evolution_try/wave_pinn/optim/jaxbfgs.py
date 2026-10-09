"""BFGS from ``jax.scipy.optimize`` -- the JAX-native minimiser.

``jax.scipy.optimize.minimize`` offers exactly one method, ``"BFGS"``, and no
Broyden of any kind.  The only Broyden in the JAX ecosystem is ``jaxopt.Broyden``,
and that one is a *root finder* for ``r(x) = 0`` rather than a minimiser -- which
is a different (and for this problem interesting) thing; see ``jaxopt_phases``.
Being here it needs nothing beyond jax itself, which matters on a shared hub
where installing a package means touching somebody else's virtualenv.

Two properties of this phase are worth knowing before comparing its numbers with
SSBroyden's or DSGNAR's:

* It is a ``lax.while_loop``, so there is **no per-iteration callback**.  The
  history this returns has the starting and the final loss and nothing in
  between.  Plots over iterations cannot show its trajectory -- that is a
  property of the routine, not of the run.
* It is unconstrained BFGS with a backtracking-free line search, on a
  least-squares objective.  It will converge quickly from a good starting point
  and can stall on a badly scaled one; both optimisers it is compared against
  exploit the least-squares structure, which this one does not.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp

from ..config import Config
from ..losses import Objective


def jaxbfgs_phase(objective: Objective, flat, cfg: Config,
                  verbose: bool = True,
                  callback: Optional[Callable[[int, float, object], None]] = None,
                  step_offset: int = 0,
                  resample: Optional[Callable[[], Objective]] = None,
                  ) -> Tuple[jax.Array, List[Dict], Dict]:
    """Minimise ``objective`` with ``jax.scipy.optimize.minimize(method="BFGS")``.

    ``resample`` is accepted for interface compatibility and ignored: this routine
    hands the whole optimisation to JAX as one ``while_loop``, with no place to
    interrupt it.  Use ``resample_rounds`` if redraws are wanted -- the rounds
    mechanism sits outside the phase and works with every optimiser.
    """
    import jax.scipy.optimize as jso

    t0 = time.time()
    flat = jnp.asarray(flat)
    loss_fn = objective.loss

    loss0 = float(loss_fn(flat))
    maxiter = int(getattr(cfg, "jaxbfgs_steps", 2000))
    gtol = float(getattr(cfg, "jaxbfgs_tol", 1e-16))

    result = jso.minimize(loss_fn, flat, method="BFGS", tol=gtol,
                          options={"maxiter": maxiter, "gtol": gtol})

    flat_out = jnp.asarray(result.x)
    loss1 = float(result.fun)
    nit = int(result.n_iter)
    status = str(result.status)

    history = [{"step": step_offset + 1, "loss": loss0},
               {"step": step_offset + max(nit, 1), "loss": loss1}]
    if callback:
        callback(step_offset + 1, loss0, flat)
        callback(step_offset + max(nit, 1), loss1, flat_out)

    info = {
        "optimizer": "jaxbfgs",
        "iterations": nit,
        "wall": time.time() - t0,
        "stopped": "converged" if bool(result.success) else status,
        "loss_final": loss1,
        "loss_initial": loss0,
        "resamples": 0,
        "n_parameters": int(flat.shape[0]),
        "no_trajectory": True,       # lax.while_loop: nothing per iteration to record
    }
    if verbose:
        print(f"[jaxbfgs] BFGS: loss {loss0:.6e} -> {loss1:.6e} in {nit} iterations "
              f"({info['wall']:.1f}s); {info['stopped']}", flush=True)
        print("[jaxbfgs] note: jax.scipy runs a lax.while_loop, so there is no "
              "per-iteration loss history to plot", flush=True)
    return flat_out, history, info
