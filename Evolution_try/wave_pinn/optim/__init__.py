"""Optimiser dispatch.

``run_optimizer`` turns ``cfg.optimizer`` into a sequence of phases:

===============  ==========================================================
``adam``         Adam only.
``ssbroyden``    SSBroyden from the initial parameters.
``adam+ssbroyden`` Adam warm-up, then SSBroyden (the usual PINN recipe).
``dsgnar``       DSGNAR (doubly-sketched Gauss-Newton) from the initial parameters.
``adam+dsgnar``  Adam warm-up, then DSGNAR.
===============  ==========================================================

Both quasi-Newton paths keep the *same* objective object, so the comparison
between them is a comparison of optimisers and nothing else.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import jax

from ..config import Config
from ..losses import Objective
from . import adam as _adam


def _phases(name: str) -> List[str]:
    if name == "adam":
        return ["adam"]
    if name == "ssbroyden":
        return ["ssbroyden"]
    if name == "adam+ssbroyden":
        return ["adam", "ssbroyden"]
    if name == "dsgnar":
        return ["dsgnar"]
    if name == "adam+dsgnar":
        return ["adam", "dsgnar"]
    raise ValueError(f"unknown optimizer {name!r}")


def run_optimizer(objective: Objective, flat0, cfg: Config,
                  verbose: bool = True,
                  callback: Optional[Callable[[int, float, object], None]] = None,
                  resample: Optional[Callable[[], Objective]] = None,
                  ) -> Tuple[jax.Array, List[Dict], List[Dict], List[List[Dict]]]:
    """Run ``cfg.optimizer``.

    Returns ``(flat, history, phase_infos, phase_histories)``.  ``history`` is a
    single step-indexed list spanning every phase (what the plots want);
    ``phase_histories`` keeps the phases apart, and ``phase_infos`` describes
    each one separately.
    """
    flat = jax.numpy.asarray(flat0)

    history: List[Dict] = []

    def _cb(step: int, loss: float, flat_now) -> None:
        history.append({"step": int(step), "loss": float(loss)})
        if callback:
            callback(step, loss, flat_now)

    infos: List[Dict] = []
    phase_histories: List[List[Dict]] = []
    step = 0
    for phase in _phases(cfg.optimizer):
        if phase == "adam":
            n = int(cfg.steps)
            flat, hist, info = _adam.adam_phase(objective, flat, cfg, steps=n,
                                                verbose=verbose, callback=_cb,
                                                step_offset=step, resample=resample)
            step += n
        elif phase == "ssbroyden":
            from .ssbroyden import ssbroyden_phase
            flat, hist, info = ssbroyden_phase(objective, flat, cfg,
                                               verbose=verbose, callback=_cb,
                                               step_offset=step, resample=resample)
            step += int(info.get("iterations", 0))
        elif phase == "dsgnar":
            from .dsgnar import dsgnar_phase
            flat, hist, info = dsgnar_phase(objective, flat, cfg,
                                            verbose=verbose, callback=_cb,
                                            step_offset=step, resample=resample)
            step += int(info.get("iterations", 0))
        else:                                                   # pragma: no cover
            raise ValueError(f"unknown phase {phase!r}")
        infos.append(info)
        phase_histories.append(hist)

    return flat, history, infos, phase_histories
