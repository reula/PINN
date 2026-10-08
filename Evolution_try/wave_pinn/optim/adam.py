"""Adam, used either on its own (``optimizer="adam"``) or as a warm-up phase.

Adam is only ever a warm-up here: the whole point of the project is the
second-order phases, and on these problems Adam plateaus several orders of
magnitude above what SSBroyden reaches.  It is kept because PINN practice
usually starts with it and because an Adam phase changes what the quasi-Newton
phase sees (and therefore whether ``initial_scale`` is needed).
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from ..config import Config, resample_schedule
from ..losses import Objective


def _make_optimizer(cfg: Config):
    import optax

    if cfg.scheduler == "none":
        sched = cfg.lr
    elif cfg.scheduler == "cosine":
        sched = optax.cosine_decay_schedule(cfg.lr, decay_steps=max(1, cfg.steps))
    else:
        raise ValueError(f"unknown scheduler {cfg.scheduler!r}")
    return optax.adam(sched)


def adam_phase(objective: Objective, flat0, cfg: Config,
               steps: Optional[int] = None,
               verbose: bool = True,
               callback: Optional[Callable[[int, float, object], None]] = None,
               step_offset: int = 0,
               resample: Optional[Callable[[], Objective]] = None,
               ) -> Tuple[jax.Array, List[Dict], Dict]:
    """Run Adam for ``steps`` iterations (default ``cfg.steps``)."""
    opt = _make_optimizer(cfg)
    steps = int(cfg.steps if steps is None else steps)
    flat = jnp.asarray(flat0)
    opt_state = opt.init(flat)
    vg = objective.value_and_grad

    resample_count = 0
    history: List[Dict] = []
    t0 = time.time()
    loss = objective.loss(flat)
    history.append({"step": step_offset, "loss": loss, "wall": 0.0})
    if callback:
        callback(step_offset, loss, flat)

    first_redraw, growth = (resample_schedule(cfg, steps) if resample is not None else (0.0, 1.0))
    next_redraw = first_redraw if first_redraw > 0 else None
    for i in range(1, steps + 1):
        if next_redraw is not None and i >= next_redraw:
            objective = resample()
            vg = objective.value_and_grad
            next_redraw = next_redraw * growth
            resample_count += 1
        loss, grad = vg(flat)
        updates, opt_state = opt.update(grad, opt_state, flat)
        flat = optax_apply(updates, flat)
        if i % max(1, cfg.log_every) == 0 or i == steps:
            history.append({"step": step_offset + i, "loss": float(loss),
                            "wall": time.time() - t0})
            if callback:
                callback(step_offset + i, float(loss), flat)
            if verbose:
                print(f"[adam] step {step_offset + i:6d}  loss {float(loss):.6e}  "
                      f"({time.time()-t0:.1f}s)", flush=True)

    info = {"optimizer": "adam", "iterations": steps, "wall": time.time() - t0,
            "stopped": "budget", "resamples": resample_count}
    return flat, history, info


def optax_apply(updates, flat):
    import optax
    return optax.apply_updates(flat, updates)
