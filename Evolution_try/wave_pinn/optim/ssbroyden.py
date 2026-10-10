"""SSBroyden, the self-scaling Broyden quasi-Newton method.

The algorithm is Optim.jl's ``SSBroyden``, and the implementation used here is
Crunch's JAX port of it (``ssbroyden2`` in ``Crunch/Optimizers/bfgs.py``), the
same code the ``Jax/`` and ``Stationary/`` examples in this repository drive.
A copy is vendored under ``Evolution_try/vendor/Crunch`` so this project runs
standalone; set ``CRUNCH_ROOT`` to point at another checkout.

What the phase does
-------------------
Crunch's ``minimize`` has no per-iteration callback, so -- exactly as the
repository's production driver does -- the phase runs in *blocks* of
``cfg.qn_block`` iterations, records the loss at every block boundary, and
carries the dense inverse Hessian ``H`` (and the parameter vector) from one
block to the next.  Two details matter and are easy to get wrong:

* the callable handed to Crunch must be **jitted** (Crunch calls
  ``jax.value_and_grad`` on it with no jit of its own, and an unjitted objective
  means one compiled primitive per operation);
* ``initial_scale=True`` engages SSBroyden's ``tau_k^A`` first-step rescaling.
  With ``H = I`` the first trial step is ``-grad``, which a Wolfe line search
  often cannot bracket, and the block then returns having taken zero iterations
  (status 3).  It is needed on the first block, on any block after ``H`` was
  reset, and after a failed block; not otherwise.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from ..config import Config, resample_schedule
from ..losses import Objective


def crunch_root() -> Optional[str]:
    """First importable Crunch root: ``$CRUNCH_ROOT``, the vendored copy, ``<repo>/Jax``."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = os.environ.get("CRUNCH_ROOT")
    if env:
        candidates = [env]
    else:
        candidates = [
            os.path.join(here, "vendor"),
            os.path.join(os.path.dirname(here), "Jax"),
            here,
        ]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "Crunch", "Optimizers")):
            return cand
    return None


def load_minimize():
    """Import Crunch's backtracking ``minimize``; ``(None, reason)`` when unavailable."""
    root = crunch_root()
    if root is None:
        return None, "no Crunch checkout found (set CRUNCH_ROOT)"
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from Crunch.Optimizers.minimize_backtracking import minimize  # noqa: WPS433
        return minimize, root
    except Exception as exc:                                    # pragma: no cover
        for mod in [m for m in list(sys.modules)
                    if m == "Crunch" or m.startswith("Crunch.")]:
            sys.modules.pop(mod, None)
        return None, f"{root}: {exc}"


def ssbroyden_phase(objective: Objective, flat0, cfg: Config,
                    verbose: bool = True,
                    callback: Optional[Callable[[int, float, object], None]] = None,
                    step_offset: int = 0,
                    resample: Optional[Callable[[], Objective]] = None,
                    ) -> Tuple[jax.Array, List[Dict], Dict]:
    """Run SSBroyden in blocks.  Returns ``(flat, history, info)``.

    ``history`` is a list of ``{"step", "loss", "nit", "status", "wall"}`` dicts,
    one per block, with a final entry for the last accepted point.
    ``info`` records whether Crunch was found and why the phase stopped.
    """
    minimize, where = load_minimize()
    if minimize is None:
        raise RuntimeError(f"SSBroyden unavailable: {where}")

    n = objective.n
    dtype = objective.dtype
    gb = n * n * jnp.dtype(dtype).itemsize / 2 ** 30
    if gb > cfg.qn_max_H_gb:
        raise RuntimeError(
            f"a dense {n}x{n} inverse Hessian needs {gb:.2f} GB > qn_max_H_gb={cfg.qn_max_H_gb}; "
            "shrink the network or raise the cap")

    block = int(cfg.qn_block) if cfg.qn_block and cfg.qn_block > 0 else int(cfg.qn_steps)
    block = max(1, min(block, cfg.qn_steps))
    n_blocks = max(1, int(round(cfg.qn_steps / block)))

    flat = jnp.asarray(flat0)
    H = jnp.eye(n, dtype=dtype)
    loss = objective.loss(flat)
    history: List[Dict] = [{"step": step_offset, "loss": loss, "nit": 0, "status": -1,
                            "wall": 0.0}]
    if callback:
        callback(step_offset, loss, flat)

    stale = 0
    resample_count = 0
    total_nit = 0
    t0 = time.time()
    stopped = "budget"
    # REDRAWING THE COLLOCATION SAMPLE.  The objective function is unchanged --
    # only the points estimating it are -- so the carried inverse Hessian remains
    # a valid approximation and is deliberately kept.  What must be reset is the
    # plateau counter: the loss before and after a redraw are measured on
    # different samples and are not comparable.
    first_redraw, growth = (resample_schedule(cfg, n_blocks * block) if resample is not None
                            else (0.0, 1.0))
    next_resample = first_redraw if first_redraw > 0 else None
    for b in range(n_blocks):
        if next_resample is not None and total_nit >= next_resample:
            objective = resample()
            loss = objective.loss(flat)
            next_resample = next_resample * growth
            resample_count += 1
            stale = 0
            prev = loss
            if verbose:
                print(f"[qn] redrew the collocation sample at step {total_nit}: "
                      f"loss on the new sample {loss:.6e}", flush=True)
        else:
            prev = loss
        loss_fn = objective._loss                   # already jitted
        first = (b == 0)
        res = minimize(loss_fn, flat, args=(), method="BFGS",
                       options={"maxiter": block,
                                "gtol": cfg.qn_gtol,
                                "initial_H": H,
                                "update_method": "ssbroyden2",
                                "initial_scale": bool(cfg.qn_initial_scale and (first or stale > 0))})
        new_loss = float(res.fun)
        nit = int(res.nit)
        status = int(res.status)
        if not (new_loss == new_loss) or new_loss == float("inf"):      # NaN / inf
            stopped = "diverged"
            if verbose:
                print(f"[qn] block {b+1}: non-finite loss {new_loss}; stopping", flush=True)
            break
        improved = new_loss < loss
        flat = res.x
        loss = new_loss
        total_nit += nit
        # A failed block's inverse Hessian is not information: Crunch hands back the
        # state unchanged, and accepting that H makes every later block fail the same way.
        if status == 0 and res.hess_inv is not None:
            H = res.hess_inv
        if improved:
            stale = 0
        else:
            stale += 1
        history.append({"step": step_offset + total_nit, "loss": loss, "nit": nit, "status": status,
                        "wall": time.time() - t0})
        if callback:
            callback(step_offset + total_nit, loss, flat)
        if verbose:
            print(f"[qn] block {b+1}/{n_blocks}: step {step_offset + total_nit:6d}  loss {prev:.6e} -> "
                  f"{loss:.6e}  nit {nit:4d}  status {status}  ({time.time()-t0:.1f}s)",
                  flush=True)
        if status == 0:
            stopped = "converged"
            break
        if nit == 0 and not improved:
            stopped = "no progress"
            break
        if cfg.plateau_tol > 0.0:
            gain = (prev - loss) / max(abs(prev), 1e-300)
            if gain <= cfg.plateau_tol:
                if stale >= max(1, cfg.plateau_patience):
                    stopped = f"plateau (relative gain <= {cfg.plateau_tol:g} for {stale} blocks)"
                    break
            else:
                stale = 0

    info = {"optimizer": "ssbroyden", "crunch_root": where, "n_parameters": n,
            "H_gb": gb, "iterations": total_nit, "stopped": stopped,
            "wall": time.time() - t0, "blocks": len(history) - 1,
            "resamples": resample_count,
            # every phase reports its final loss under the same key, so the per-round
            # table has something to show; without it SSBroyden's rounds read `nan`
            "loss_final": float(loss)}
    if verbose:
        print(f"[qn] SSBroyden: loss {history[0]['loss']:.6e} -> {loss:.6e} in "
              f"{total_nit} iterations ({info['wall']:.1f}s); {stopped}", flush=True)
    return flat, history, info
