"""The Wolfe-condition quasi-Newton family, as a measured negative control.

Tried because the Broyden's failure looked like a line-search problem, and it was
one.  ``jaxopt.Broyden`` uses ``linesearch="backtracking"`` -- its own docstring says
*"for now only backtracking"* -- and backtracking can only shrink a step by
``decrease_factor`` a bounded number of times (``0.8^15 = 0.035`` by default).  A
Broyden whose first step is ``-r`` with ``|r| = 48`` against ``|theta| = 10`` needs
to shrink by far more than that, which is why every configuration tried in
``jaxopt_broyden.py`` either exploded or stood still.

The methods here use a proper **strong-Wolfe zoom** line search, which brackets and
bisects and has no such floor.  ``jaxopt.BFGS``, ``jaxopt.LBFGS`` and
``jaxopt.NonlinearCG`` all default to ``linesearch="zoom"``; ``optax.lbfgs`` takes
``optax.scale_by_zoom_linesearch``.

Measured on `T = 2`, `n_coll = 2201`, float64, from a random initialisation, loss
`1.068e+00` (and from 1000 Adam steps, loss `2.408e-03`):

    method (60 iterations)          from init    from Adam
    jaxopt.BFGS   (zoom/Wolfe)      1.685e-03    7.233e-04
    jaxopt.LBFGS  (zoom/Wolfe)      4.696e-03    1.121e-03
    jaxopt.NonlinearCG (zoom)       3.283e-02    1.990e-03
    optax.lbfgs + zoom (Wolfe)      1.027e-02    1.193e-03
    jaxopt.Broyden (backtracking)   6.759e+05    2.115e+09   <- explodes

So the line search is the whole of the Broyden's problem, and the Wolfe family does
make smooth progress.  It is still not competitive, and to convergence the gap is
not small (300 iterations, from a random initialisation):

    jaxopt.BFGS   (zoom/Wolfe)   300 it   loss 3.328e-04   rel L2 4.232e-01    71 s
    optax.lbfgs + zoom (Wolfe)   300 it   loss 1.043e-03   rel L2 6.532e-01   241 s
    DSGNAR, s = 733              186 it   loss 2.243e-14   rel L2 6.756e-06

Ten orders of magnitude in the loss and sixty thousand in the error, in more
iterations.  And note the shape of the failure rather than just its size: a loss of
`3.3e-04` with a relative error of `0.42` is the sample-fitting regime, the same
"loss stops predicting the error" effect this project keeps meeting -- with
`n_coll = n_params` a gradient method can drive the sampled residual down while the
solution stays wrong between the points.

Which is the useful conclusion.  Explicit curvature -- what DSGNAR builds from
sketched Jacobian-vector products -- is doing the work here; a quasi-Newton method
that only ever sees gradients is not merely slower, it stops at a different place.
These are kept as an optimiser choice because that is a claim worth being able to
re-run, not because it is a contender.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from ..config import Config
from ..losses import Objective


def _record(callback, history, step_offset, step, loss, flat):
    history.append({"step": step_offset + step, "loss": loss})
    if callback:
        callback(step_offset + step, loss, flat)


def wolfe_phase(objective: Objective, flat, cfg: Config,
                verbose: bool = True,
                callback: Optional[Callable[[int, float, object], None]] = None,
                step_offset: int = 0,
                resample: Optional[Callable[[], Objective]] = None,
                ) -> Tuple[jax.Array, List[Dict], Dict]:
    """``jaxopt``'s gradient-based quasi-Newton solvers, all on a zoom line search.

    ``cfg.wolfe_method`` selects BFGS, LBFGS or NonlinearCG.  ``resample`` is
    accepted and ignored: a quasi-Newton method's curvature estimate is tied to the
    function it was built from, so redrawing mid-phase invalidates it.  Use
    ``resample_rounds``, which restarts the solver on a fresh system.
    """
    try:
        import jaxopt
    except ImportError as exc:
        # Every other optional dependency in this package degrades with a reason
        # rather than a bare traceback (`load_minimize` returns (None, why)).  This
        # one is reachable by a config choice, so it has to explain itself: the hub
        # this project runs on does NOT have jaxopt, and a user who sets one of the
        # jaxopt_* optimisers there deserves to be told that rather than shown
        # "No module named 'jaxopt'".
        raise ImportError(
            "the Wolfe family needs `jaxopt`, which is not installed here.  It is an "
            "optional dependency (`pip install jaxopt`); SSBroyden, DSGNAR and the "
            "trust-region phase do not need it."
        ) from exc

    t0 = time.time()
    flat = jnp.asarray(flat)
    method = str(getattr(cfg, "wolfe_method", "bfgs")).lower()
    cls = {"bfgs": jaxopt.BFGS, "lbfgs": jaxopt.LBFGS, "ncg": jaxopt.NonlinearCG,
           "nonlinear_cg": jaxopt.NonlinearCG}.get(method)
    if cls is None:
        raise ValueError(f"unknown wolfe_method {method!r}; use bfgs, lbfgs or ncg")

    maxiter = int(getattr(cfg, "wolfe_steps", 300))
    kw = dict(fun=objective._loss, maxiter=maxiter,
              tol=float(getattr(cfg, "wolfe_tol", 1e-16)))
    if cls is jaxopt.LBFGS:
        kw["history_size"] = int(getattr(cfg, "wolfe_memory", 10))
    solver = cls(**kw)

    history: List[Dict] = []
    loss_fn = objective._loss
    loss0 = float(loss_fn(flat))
    _record(callback, history, step_offset, 0, loss0, flat)

    state = solver.init_state(flat)
    step = 0
    for step in range(1, maxiter + 1):
        flat, state = solver.update(flat, state)
        loss = float(loss_fn(flat))
        _record(callback, history, step_offset, step, loss, flat)
        if verbose and step % max(int(getattr(cfg, "log_every", 25) or 25), 1) == 0:
            print(f"[wolfe] {method} step {step:5d}  loss {loss:.6e}  "
                  f"|grad| {float(state.error):.3e}  ({time.time() - t0:.1f}s)", flush=True)
        if not jnp.isfinite(state.error):
            break
        if float(state.error) <= kw["tol"]:
            break

    info = {
        "optimizer": f"jaxopt_{method}",
        "iterations": step,
        "wall": time.time() - t0,
        "stopped": "gradient below tol" if float(state.error) <= kw["tol"] else "budget",
        "loss_final": float(loss_fn(flat)),
        "loss_initial": loss0,
        "grad_norm_final": float(state.error),
        "line_search": "zoom (strong Wolfe)",
        "resamples": 0,
    }
    if verbose:
        print(f"[wolfe] {method}: loss {loss0:.6e} -> {info['loss_final']:.6e} in "
              f"{step} iterations ({info['wall']:.1f}s); {info['stopped']}", flush=True)
    return flat, history, info


def optax_lbfgs_phase(objective: Objective, flat, cfg: Config,
                      verbose: bool = True,
                      callback: Optional[Callable[[int, float, object], None]] = None,
                      step_offset: int = 0,
                      resample: Optional[Callable[[], Objective]] = None,
                      ) -> Tuple[jax.Array, List[Dict], Dict]:
    """``optax.lbfgs`` with ``scale_by_zoom_linesearch`` -- Optax's Wolfe search.

    Written as an explicit loop because Optax is a gradient transform, not a solver,
    and its ``lbfgs`` wants the value and the gradient handed to ``update``.
    """
    import optax

    t0 = time.time()
    flat = jnp.asarray(flat)
    maxiter = int(getattr(cfg, "wolfe_steps", 300))
    memory = int(getattr(cfg, "wolfe_memory", 10))
    solver = optax.lbfgs(
        memory_size=memory,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=int(getattr(cfg, "wolfe_maxls", 30))))
    state = solver.init(flat)
    value_and_grad = jax.jit(jax.value_and_grad(objective._loss))

    history: List[Dict] = []
    loss0 = float(objective._loss(flat))
    _record(callback, history, step_offset, 0, loss0, flat)

    step = 0
    loss = loss0
    for step in range(1, maxiter + 1):
        value, grad = value_and_grad(flat)
        updates, state = solver.update(grad, state, flat, value=value, grad=grad,
                                       value_fn=objective._loss)
        flat = optax.apply_updates(flat, updates)
        loss = float(value)
        _record(callback, history, step_offset, step, loss, flat)
        if verbose and step % max(int(getattr(cfg, "log_every", 25) or 25), 1) == 0:
            print(f"[wolfe] optax_lbfgs step {step:5d}  loss {loss:.6e}  "
                  f"({time.time() - t0:.1f}s)", flush=True)
        if not jnp.isfinite(loss):
            break

    info = {
        "optimizer": "optax_lbfgs",
        "iterations": step,
        "wall": time.time() - t0,
        "stopped": "budget",
        "loss_final": float(objective._loss(flat)),
        "loss_initial": loss0,
        "line_search": "zoom (strong Wolfe)",
        "history_size": memory,
        "resamples": 0,
    }
    if verbose:
        print(f"[wolfe] optax_lbfgs: loss {loss0:.6e} -> {info['loss_final']:.6e} in "
              f"{step} iterations ({info['wall']:.1f}s)", flush=True)
    return flat, history, info
