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
import jax.numpy as jnp

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
    if name == "trustregion":
        return ["trustregion"]
    if name == "adam+trustregion":
        return ["adam", "trustregion"]
    if name == "jaxopt_broyden":
        return ["jaxopt_broyden"]
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

    def _dispatch(phase: str, objective: Objective, flat, step: int,
                  resample):
        if phase == "adam":
            n = int(cfg.steps)
            return (*_adam.adam_phase(objective, flat, cfg, steps=n,
                                      verbose=verbose, callback=_cb,
                                      step_offset=step, resample=resample), n)
        if phase == "ssbroyden":
            from .ssbroyden import ssbroyden_phase
            out = ssbroyden_phase(objective, flat, cfg, verbose=verbose, callback=_cb,
                                  step_offset=step, resample=resample)
        elif phase == "trustregion":
            from .trustregion import trust_region_phase
            out = trust_region_phase(objective, flat, cfg, verbose=verbose, callback=_cb,
                                     step_offset=step, resample=resample)
        elif phase == "dsgnar":
            from .dsgnar import dsgnar_phase
            out = dsgnar_phase(objective, flat, cfg, verbose=verbose, callback=_cb,
                               step_offset=step, resample=resample)
        elif phase == "jaxopt_broyden":
            from .jaxopt_broyden import jaxopt_broyden_phase
            out = jaxopt_broyden_phase(objective, flat, cfg, verbose=verbose,
                                       callback=_cb, step_offset=step, resample=resample)
        else:                                                   # pragma: no cover
            raise ValueError(f"unknown phase {phase!r}")
        return (*out, int(out[2].get("iterations", 0)))

    infos: List[Dict] = []
    phase_histories: List[List[Dict]] = []
    step = 0
    phases = _phases(cfg.optimizer)
    rounds = max(1, int(getattr(cfg, "resample_rounds", 1) or 1))

    for i, phase in enumerate(phases):
        # Rounds apply to the phase that does the actual converging, and only when
        # somebody asked for redraws.  Each round runs the optimiser to ITS OWN
        # stopping criterion on a fixed sample; the redraw happens between rounds.
        # That is the opposite of redrawing mid-run, where the optimiser chases a
        # target that moves every few iterations and never converges to any of them.
        n_rounds = rounds if (i == len(phases) - 1 and resample is not None) else 1
        round_infos: List[Dict] = []
        hist_all: List[Dict] = []
        total_iters = 0
        res_total = 0
        for r in range(n_rounds):
            before = flat
            flat, hist, info, n_it = _dispatch(
                phase, objective, flat, step,
                resample=(None if n_rounds > 1 else resample))
            # How far this round had to move the solution in order to fit the
            # sample it was just given.  This is the stability number: for a
            # converged, sample-independent solution it goes to zero, and if it
            # does not, the solution is fitting the collocation points.
            nb = max(float(jnp.linalg.norm(before)), 1e-300)
            rel_change = float(jnp.linalg.norm(flat - before)) / nb
            step += n_it
            total_iters += n_it
            res_total += int(info.get("resamples") or 0)
            hist_all.extend(hist)
            loss_key = "loss_final" if "loss_final" in info else "final_loss"
            rec = {"round": r, "iterations": n_it,
                   "final_loss": float(info.get(loss_key, float("nan"))),
                   "param_rel_change": rel_change}
            if r < n_rounds - 1:
                objective = resample()
                # the loss of the round's solution on the sample it has NOT seen:
                # the jump is how much of that solution was sample-specific
                rec["loss_on_next_sample"] = float(objective.loss(flat))
            round_infos.append(rec)
            if n_rounds > 1 and verbose:
                extra = ("  redraw -> %.3e" % rec["loss_on_next_sample"]
                         if "loss_on_next_sample" in rec else "")
                chg = ("" if rel_change != rel_change
                       else "   |dtheta|/|theta| %.2e" % rel_change)
                print(f"[{phase}] round {r+1}/{n_rounds}: {n_it} iterations, "
                      f"loss {rec['final_loss']:.3e}{extra}{chg}", flush=True)
        info = dict(info)
        info["iterations"] = total_iters
        info["rounds"] = round_infos
        # every round after the first was preceded by one redraw
        info["resamples"] = res_total + max(0, n_rounds - 1)
        if verbose and n_rounds > 1:
            print(f"[{phase}] {n_rounds} rounds, {total_iters} iterations total", flush=True)
        infos.append(info)
        phase_histories.append(hist_all)

    return flat, history, infos, phase_histories
